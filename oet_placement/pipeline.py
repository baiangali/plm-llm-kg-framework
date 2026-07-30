"""
End-to-end orchestration of the three-stage framework (Section 3.1).

Data flow, following Figure 2 and Table 1:

    mention + context
        -> Stage 1  : ranked pool of candidate concepts or edges
        -> Stage 2  : enriched candidate set E_i, truncated to the top k
        -> Stage 3  : final ranking of insertion edges
        -> curator

No stage modifies the ontology. Insertion is performed only after expert
confirmation, which is what positions the framework as decision support rather
than automated ontology construction.

The candidate set is built once per k and then reused by every selector, so that
Table 5 compares ranking quality over a *matched candidate pool* and the effect
of the selection method is separable from the effect of candidate-set size.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .config import PipelineConfig
from .data import (
    Edge,
    Mention,
    Ontology,
    aggregate_pairs,
    dataset_paths,
    dataset_statistics,
    load_edge_pairs,
    load_mentions,
)
from .logical_validation import GraphBackend, summarise, validate_top_ranked
from .metrics import PlacementScores, evaluate_rankings
from .stage2_enrichment import (
    CandidateSet,
    build_candidate_set,
    dot_product_scorer,
    enrichment_statistics,
    mean_endpoint_cosine_scorer,
)

# --------------------------------------------------------------------------- #
# Stage 1 dispatch
# --------------------------------------------------------------------------- #


@dataclass
class RetrievalOutput:
    """What Stage 1 hands to Stage 2 for one mention."""

    mention_id: str
    concepts: List[str] = field(default_factory=list)
    seed_edges: Optional[List[Edge]] = None
    concept_similarity: Dict[str, float] = field(default_factory=dict)
    edge_scores: Dict[Edge, float] = field(default_factory=dict)


def run_stage1(
    config: PipelineConfig,
    ontology: Ontology,
    mentions: Sequence[Mention],
    retriever,
) -> Dict[str, RetrievalOutput]:
    """Retrieve a candidate pool for every mention (Section 3.3)."""
    outputs: Dict[str, RetrievalOutput] = {}

    if config.retriever == "bi_encoder":
        # The fine-tuned bi-encoder aligns mentions with serialised edges, so
        # Eq. (4) is bypassed and enrichment starts from the retrieved edges.
        retrieved = retriever.retrieve_edges(mentions, top_n=config.n_retrieved_concepts)
        for m, edges in zip(mentions, retrieved):
            outputs[m.mention_id] = RetrievalOutput(
                mention_id=m.mention_id,
                concepts=_endpoints_in_rank_order(edges),
                seed_edges=[e for e, _ in edges],
                edge_scores={e: s for e, s in edges},
            )
        return outputs

    if config.retriever == "fixed_embeddings":
        retrieved = retriever.retrieve_concepts_batch(
            mentions, top_n=config.n_retrieved_concepts
        )
    else:  # inverted_index
        retrieved = [
            retriever.retrieve_concepts(m, top_n=config.n_retrieved_concepts)
            for m in mentions
        ]

    for m, concepts in zip(mentions, retrieved):
        outputs[m.mention_id] = RetrievalOutput(
            mention_id=m.mention_id,
            concepts=[c for c, _ in concepts],
            concept_similarity={c: s for c, s in concepts},
        )
    return outputs


def _endpoints_in_rank_order(edges: Sequence[Tuple[Edge, float]]) -> List[str]:
    """Concepts appearing in the retrieved edges, ordered by best edge rank.

    Used only to decide whether the leaf-priority rule of Section 3.4 fires,
    which is defined in terms of "the top-ranked retrieved concept".
    """
    seen: Dict[str, None] = {}
    for edge, _ in edges:
        for endpoint in (edge.parent, edge.child):
            if endpoint not in seen:
                seen[endpoint] = None
    return list(seen)


# --------------------------------------------------------------------------- #
# Stage 2 dispatch
# --------------------------------------------------------------------------- #


def run_stage2(
    config: PipelineConfig,
    ontology: Ontology,
    mentions: Sequence[Mention],
    stage1: Mapping[str, RetrievalOutput],
    k: int,
    retriever=None,
) -> Dict[str, CandidateSet]:
    """Construct, enrich, rank and truncate the candidate set (Section 3.4)."""
    candidate_sets: Dict[str, CandidateSet] = {}

    for m in mentions:
        out = stage1.get(m.mention_id)
        if out is None:
            continue

        if config.retriever == "bi_encoder":
            # Enriched edges may be absent from the precomputed index, so they are
            # scored on demand with Eq. (10).
            seeds = out.seed_edges or []
            from .stage2_enrichment import enrich_edges

            pool = enrich_edges(ontology, seeds) if config.use_enrichment else set(seeds)
            edge_scores = (
                retriever.score_edges(ontology, m, sorted(pool))
                if retriever is not None
                else out.edge_scores
            )
            scorer = dot_product_scorer(edge_scores)
            candidate_sets[m.mention_id] = build_candidate_set(
                ontology,
                m.mention_id,
                retrieved_concepts=out.concepts,
                scorer=scorer,
                k=k,
                use_enrichment=config.use_enrichment,
                use_leaf_priority=config.use_leaf_priority,
                seed_edges=seeds,
            )
        else:
            scorer = mean_endpoint_cosine_scorer(ontology, out.concept_similarity)
            candidate_sets[m.mention_id] = build_candidate_set(
                ontology,
                m.mention_id,
                retrieved_concepts=out.concepts,
                scorer=scorer,
                k=k,
                use_enrichment=config.use_enrichment,
                use_leaf_priority=config.use_leaf_priority,
            )

    return candidate_sets


# --------------------------------------------------------------------------- #
# Stage 3 dispatch
# --------------------------------------------------------------------------- #


def run_stage3(
    config: PipelineConfig,
    ontology: Ontology,
    mentions: Sequence[Mention],
    candidate_sets: Mapping[str, CandidateSet],
    selector=None,
) -> Dict[str, List[Edge]]:
    """Final ranking of insertion edges (Sections 3.5-3.6).

    With no selector the Stage 2 ordering is returned unchanged, which is the
    "Edge-Bi-encoder" row of Table 5 -- retrieval-order baseline over the same
    matched pool.
    """
    baseline = {mid: list(cs.edges) for mid, cs in candidate_sets.items()}
    if selector is None or config.selector is None:
        return baseline

    if config.selector == "cross_encoder":
        return selector.rerank(ontology, mentions, baseline)

    if config.selector in ("llm_zero_shot", "llm_tuned"):
        predictions = selector.select(ontology, mentions, baseline)
        return {
            mid: (predictions[mid].ranking if mid in predictions else edges)
            for mid, edges in baseline.items()
        }

    raise ValueError(f"unknown selector {config.selector!r}")


# --------------------------------------------------------------------------- #
# Full run
# --------------------------------------------------------------------------- #


@dataclass
class RunResult:
    config: Dict
    k: int
    scores: PlacementScores
    enrichment: Dict[str, float]
    timing: Dict[str, float]
    logical_validity: Dict[str, Optional[float]]
    rankings: Dict[str, List[Edge]] = field(default_factory=dict)

    def to_json(self) -> Dict:
        return {
            "config": self.config,
            "k": self.k,
            "scores": {
                "n_mentions": self.scores.n_mentions,
                "InR_any": {str(k): v for k, v in self.scores.inr_any.items()},
                "InR_all": {str(k): v for k, v in self.scores.inr_all.items()},
                "InR_any_lf": {str(k): v for k, v in self.scores.inr_any_leaf.items()},
                "InR_all_lf": {str(k): v for k, v in self.scores.inr_all_leaf.items()},
                "InR_any_nlf": {str(k): v for k, v in self.scores.inr_any_non_leaf.items()},
                "InR_all_nlf": {str(k): v for k, v in self.scores.inr_all_non_leaf.items()},
                "MRR": self.scores.mrr,
                "MAP": self.scores.map,
            },
            "enrichment": self.enrichment,
            "timing": self.timing,
            "logical_validity": self.logical_validity,
        }


def run(
    config: PipelineConfig,
    ontology: Ontology,
    mentions: Sequence[Mention],
    retriever,
    selector=None,
    k: Optional[int] = None,
    eval_k_values: Sequence[int] = (1, 5, 10, 50),
    validate_logic: bool = True,
) -> RunResult:
    """Run all three stages over one mention set and score the result."""
    k = k or max(config.k_values)

    t0 = time.perf_counter()
    stage1 = run_stage1(config, ontology, mentions, retriever)
    t1 = time.perf_counter()
    candidate_sets = run_stage2(config, ontology, mentions, stage1, k, retriever=retriever)
    t2 = time.perf_counter()
    rankings = run_stage3(config, ontology, mentions, candidate_sets, selector)
    t3 = time.perf_counter()

    n = max(1, len(mentions))
    timing = {
        "stage1_ms_per_mention": 1000.0 * (t1 - t0) / n,
        "stage2_ms_per_mention": 1000.0 * (t2 - t1) / n,
        "stage3_ms_per_mention": 1000.0 * (t3 - t2) / n,
        "total_ms_per_mention": 1000.0 * (t3 - t0) / n,
    }

    scores = evaluate_rankings(
        mentions, rankings, k_values=sorted({*eval_k_values, k})
    )
    validity = (
        summarise(validate_top_ranked(ontology, mentions, rankings, GraphBackend(ontology)))
        if validate_logic
        else {}
    )

    return RunResult(
        config=asdict(config),
        k=k,
        scores=scores,
        enrichment=enrichment_statistics(list(candidate_sets.values())),
        timing=timing,
        logical_validity=validity,
        rankings=rankings,
    )


# --------------------------------------------------------------------------- #
# Ablation -- Section 4.5, Table 6
# --------------------------------------------------------------------------- #


def run_ablation(
    config: PipelineConfig,
    ontology: Ontology,
    mentions: Sequence[Mention],
    retriever,
    cross_encoder=None,
    llm=None,
    k: int = 50,
) -> Dict[str, RunResult]:
    """Isolate the contribution of each framework component.

    All configurations are evaluated over the same mentions under the same split
    indices, which is what makes the rows comparable.

      retrieval only          Stage 1, no enrichment, no selection
      + enrichment            Stage 1 + Stage 2
      + cross-encoder         Stage 1 + Stage 2 + fine-tuned selection
      + LLM (tuned)           Stage 1 + Stage 2 + instruction-tuned selection
      - leaf priority         the full configuration with the rule of Section 3.4 removed
    """
    from copy import deepcopy

    results: Dict[str, RunResult] = {}

    retrieval_only = deepcopy(config)
    retrieval_only.use_enrichment = False
    retrieval_only.selector = None
    results["retrieval only"] = run(retrieval_only, ontology, mentions, retriever, None, k=k)

    with_enrichment = deepcopy(config)
    with_enrichment.use_enrichment = True
    with_enrichment.selector = None
    results["+ enrichment"] = run(with_enrichment, ontology, mentions, retriever, None, k=k)

    if cross_encoder is not None:
        cfg = deepcopy(config)
        cfg.use_enrichment = True
        cfg.selector = "cross_encoder"
        results["+ cross-encoder"] = run(cfg, ontology, mentions, retriever, cross_encoder, k=k)

    if llm is not None:
        cfg = deepcopy(config)
        cfg.use_enrichment = True
        cfg.selector = "llm_tuned"
        results["+ LLM (tuned)"] = run(cfg, ontology, mentions, retriever, llm, k=k)

    no_leaf_rule = deepcopy(config)
    no_leaf_rule.use_enrichment = True
    no_leaf_rule.use_leaf_priority = False
    no_leaf_rule.selector = config.selector if cross_encoder is not None else None
    results["- leaf priority"] = run(
        no_leaf_rule, ontology, mentions, retriever, cross_encoder, k=k
    )

    return results


def format_ablation(results: Mapping[str, RunResult], k: int = 10) -> str:
    """Render the ablation in the layout of Table 6."""
    lines = ["Configuration\tInR_any@%d\tInR_all@%d\tMRR" % (k, k)]
    for name, result in results.items():
        lines.append(
            f"{name}\t{100 * result.scores.inr_any.get(k, 0.0):.1f}"
            f"\t{100 * result.scores.inr_all.get(k, 0.0):.1f}"
            f"\t{result.scores.mrr:.3f}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Dataset preparation
# --------------------------------------------------------------------------- #


def load_dataset(
    config: PipelineConfig,
    split_mode: str = "native",
) -> Tuple[Ontology, Dict[str, List[Mention]]]:
    """Load the ontology and the mention splits for one MM-S14 part.

    ``split_mode``
        ``"native"``
            Use the splits shipped with the dataset. Training and validation come
            from the in-KB mention files; evaluation uses ``test-NIL``, whose
            mentions denote concepts introduced in the later SNOMED CT release
            and are therefore out-of-knowledge-base by construction (Section 1.1).
        ``"resplit"``
            Pool the out-of-KB mentions and re-split them 70/15/15 at the mention
            level under seed 42, as described in Section 3.7.
    """
    paths = dataset_paths(config.dataset_root, config.dataset)
    ontology = Ontology.load(paths["root"], edge_catalogue=config.edge_catalogue)

    if split_mode == "native":
        splits = {
            "train": load_mentions(paths["mention_train"]),
            "valid": load_mentions(paths["mention_valid"]),
            "valid_nil": load_mentions(paths["mention_valid_nil"]),
            "test_nil": load_mentions(paths["mention_test_nil"]),
        }
        return ontology, splits

    if split_mode == "resplit":
        from .data import make_splits

        pooled = load_mentions(paths["mention_test_nil_all"], out_of_kb_only=True)
        train, valid, test = make_splits(pooled, seed=config.seed)
        return ontology, {"train": train, "valid": valid, "test_nil": test}

    raise ValueError(f"unknown split_mode {split_mode!r}")


def report_statistics(ontology: Ontology, splits: Mapping[str, Sequence[Mention]]) -> str:
    """Print the Table 3 statistics actually observed in the files on disk."""
    lines = ["Statistic\t" + "\t".join(splits.keys())]
    keys = [
        "mentions",
        "gold_edges_total",
        "mean_gold_edges_per_mention",
        "median_gold_edges_per_mention",
        "single_gold_edge_pct",
        "leaf_pct",
        "non_leaf_pct",
        "mean_depth",
        "mean_degree",
    ]
    stats = {name: dataset_statistics(ms, ontology) for name, ms in splits.items()}
    for key in keys:
        row = [key]
        for name in splits:
            value = stats[name].get(key)
            row.append("-" if value is None else f"{value:.2f}")
        lines.append("\t".join(row))
    lines.append(f"concepts (|V|)\t{len(ontology)}")
    lines.append(f"atomic edges\t{len(ontology.atomic_edges)}")
    lines.append(f"edge catalogue (|E|)\t{len(ontology.edge_catalogue)}")
    return "\n".join(lines)


def save_results(results: Mapping[str, RunResult], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {name: result.to_json() for name, result in results.items()}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def save_predictions(
    ontology: Ontology,
    mentions: Sequence[Mention],
    rankings: Mapping[str, Sequence[Edge]],
    path: str,
    top_k: int = 10,
) -> None:
    """Write the curator-facing output: a ranked list of insertion edges per mention."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for m in mentions:
            ranked = list(rankings.get(m.mention_id, ()))[:top_k]
            fh.write(
                json.dumps(
                    {
                        "mention_id": m.mention_id,
                        "mention": m.mention,
                        "predictions": [
                            {
                                "rank": i + 1,
                                "parent_concept": e.parent,
                                "child_concept": e.child,
                                "parent": ontology.title(e.parent),
                                "child": ontology.title(e.child),
                                "is_gold": e in m.gold_edges,
                            }
                            for i, e in enumerate(ranked)
                        ],
                        "gold_edges": [
                            {"parent_concept": e.parent, "child_concept": e.child}
                            for e in sorted(m.gold_edges)
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
