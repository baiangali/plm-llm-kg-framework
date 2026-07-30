"""
Stage 2 -- Edge generation and structural enrichment (Section 3.4).

This is the stage the paper identifies as its principal methodological
contribution: it sits between retrieval and selection and expands the candidate
set by traversing the local structure of the ontology *at inference time*, not
only during training.

    Eq. (4)   S = union_i {P_i -> A} union union_j {A -> C_j}
                    union union_ij {P_i -> C_j},  plus the leaf edge A -> NULL
    Eq. (5)   S1(P, C) = {p -> C  : p in Par(P)}
    Eq. (6)   S2(P, C) = {P -> c  : c in Ch(C)}
    Eq. (7)   S3(P, C) = {p -> c  : p in Par(P), c in Ch(C)}
    Eq. (8)   E(P, C) = {P -> C} union S1 union S2 union S3
    Eq. (9)   score(m, P, C) = 1/2 [cos(m, P) + cos(m, C)]
    Eq. (10)  s(m, e) = v_m . v_e

This module is pure Python: the traversal and the ranking are independent of any
particular encoder, which is what allows Section 4.5 to ablate enrichment while
holding retrieval fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .data import NULL_ID, Edge, Ontology, leaf_edge

# --------------------------------------------------------------------------- #
# Edge construction from retrieved concepts -- Eq. (4)
# --------------------------------------------------------------------------- #


def construct_edges_from_concept(
    ontology: Ontology,
    concept_idx: str,
    include_leaf: bool = True,
) -> Set[Edge]:
    """Build candidate edges around one retrieved concept ``A``, Eq. (4).

    The ontology is traversed one hop up and one hop down from ``A``. The result
    comprises every one-hop edge incident to ``A`` together with the two-hop
    edges passing through it, so that ``A`` itself may be either the parent, the
    child, or the concept displaced by the insertion.

    The leaf edge ``A -> NULL`` is appended so that placements at the bottom of
    the hierarchy remain reachable.
    """
    parents = ontology.parents_of(concept_idx)
    children = ontology.children_of(concept_idx)

    edges: Set[Edge] = set()
    # P_i -> A : the new concept is inserted above A, under one of its parents.
    edges.update(Edge(p, concept_idx) for p in parents)
    # A -> C_j : the new concept is inserted below A, above one of its children.
    edges.update(Edge(concept_idx, c) for c in children)
    # P_i -> C_j : the new concept replaces A's position between them.
    edges.update(Edge(p, c) for p in parents for c in children)

    if include_leaf:
        edges.add(leaf_edge(concept_idx))

    return edges


def construct_edges(
    ontology: Ontology,
    retrieved_concepts: Sequence[str],
    include_leaf: bool = True,
) -> Set[Edge]:
    """Apply Eq. (4) to every retrieved concept and take the union."""
    seeds: Set[Edge] = set()
    for idx in retrieved_concepts:
        seeds |= construct_edges_from_concept(ontology, idx, include_leaf=include_leaf)
    return seeds


# --------------------------------------------------------------------------- #
# Enrichment of a seed edge -- Eq. (5)-(8)
# --------------------------------------------------------------------------- #


def enrich_edge(ontology: Ontology, edge: Edge) -> Set[Edge]:
    """Expand one seed edge over its local neighbourhood, Eq. (8).

    The parent is lifted up the hierarchy (Eq. 5), the child is descended down it
    (Eq. 6), and both moves are combined (Eq. 7). The seed itself is retained.

    For a leaf edge ``P -> NULL`` the set ``Ch(NULL)`` is empty, so Eq. (6) and
    Eq. (7) contribute nothing and the expansion reduces to lifting the parent:
    ``{p -> NULL : p in Par(P)}``.
    """
    parent, child = edge.parent, edge.child
    parents_of_parent = ontology.parents_of(parent)
    children_of_child = ontology.children_of(child)

    enriched: Set[Edge] = {edge}
    # Eq. (5)
    enriched.update(Edge(p, child) for p in parents_of_parent)
    # Eq. (6)
    enriched.update(Edge(parent, c) for c in children_of_child)
    # Eq. (7)
    enriched.update(Edge(p, c) for p in parents_of_parent for c in children_of_child)
    return enriched


def enrich_edges(
    ontology: Ontology,
    seed_edges: Iterable[Edge],
    enrich_leaf_edges: bool = True,
) -> Set[Edge]:
    """Enrich every seed edge and merge, deduplicating the overlapping expansions.

    Section 3.4 states that leaf edges ``P -> NULL`` are enriched where their
    parent ``P`` occurs in a predicted non-leaf edge, so that coverage of
    low-level placements is maintained. That condition is applied here: a leaf
    seed is expanded only if its parent also appears as an endpoint of some
    non-leaf seed.
    """
    seeds = list(seed_edges)
    non_leaf_endpoints: Set[str] = set()
    for e in seeds:
        if not e.is_leaf:
            non_leaf_endpoints.add(e.parent)
            non_leaf_endpoints.add(e.child)

    enriched: Set[Edge] = set()
    for e in seeds:
        if e.is_leaf:
            enriched.add(e)
            if enrich_leaf_edges and e.parent in non_leaf_endpoints:
                enriched |= enrich_edge(ontology, e)
        else:
            enriched |= enrich_edge(ontology, e)
    return enriched


# --------------------------------------------------------------------------- #
# Ranking -- Eq. (9) and Eq. (10)
# --------------------------------------------------------------------------- #

#: A scorer maps an edge to a scalar compatibility with the current mention.
EdgeScorer = Callable[[Edge], float]


def mean_endpoint_cosine_scorer(
    ontology: Ontology,
    concept_similarity: Dict[str, float],
    null_similarity: float = 0.0,
) -> EdgeScorer:
    """Eq. (9): the mean cosine similarity of the mention to the two endpoints.

    ``concept_similarity`` holds ``cos(m, x)`` for every concept encountered.
    Concepts absent from the map -- those outside the retrieved neighbourhood --
    score 0, which is the neutral value of cosine similarity for the normalised
    embeddings used in Section 3.3.

    For a leaf edge the child is ``NULL`` and carries ``null_similarity``, whose
    default of 0 makes ``score(m, P, NULL) = cos(m, P) / 2``. The leaf-edge
    prioritisation rule below is what compensates for that halving; it is the
    rule ablated in Section 4.5.
    """

    def score(edge: Edge) -> float:
        parent_sim = concept_similarity.get(edge.parent, 0.0)
        child_sim = (
            null_similarity if edge.child == NULL_ID
            else concept_similarity.get(edge.child, 0.0)
        )
        return 0.5 * (parent_sim + child_sim)

    return score


def dot_product_scorer(edge_scores: Dict[Edge, float]) -> EdgeScorer:
    """Eq. (10): rank by the dot product of the mention and edge vectors.

    Used when Stage 1 is the fine-tuned bi-encoder, which embeds the serialised
    edge as a whole rather than its endpoints separately.
    """

    def score(edge: Edge) -> float:
        return edge_scores.get(edge, float("-inf"))

    return score


def apply_leaf_priority(
    ranked: Sequence[Edge],
    top_concept_is_leaf: bool,
    scores: Optional[Dict[Edge, float]] = None,
) -> List[Edge]:
    """Leaf-edge prioritisation rule of Section 3.4.

    "If the top-ranked retrieved concept was classified as a leaf, leaf edges are
    prioritized among those with the highest score."

    The rule is implemented as a stable partition: leaf edges are moved ahead of
    non-leaf edges while the relative order induced by the score is preserved
    within each group. It is a no-op when the top-ranked concept is not a leaf,
    and is disabled entirely by the ``use_leaf_priority`` ablation switch.
    """
    if not top_concept_is_leaf:
        return list(ranked)
    leaves = [e for e in ranked if e.is_leaf]
    others = [e for e in ranked if not e.is_leaf]
    return leaves + others


def rank_edges(
    candidates: Iterable[Edge],
    scorer: EdgeScorer,
    k: Optional[int] = None,
    top_concept_is_leaf: bool = False,
    use_leaf_priority: bool = True,
) -> List[Edge]:
    """Score, sort, optionally apply the leaf rule, and truncate to the top k.

    Candidates outside the top k are discarded at the stage boundary, which is
    what makes k the operating point traded off in Section 4.9: raising it lifts
    recall but transfers review effort back to the curator.
    """
    scored: List[Tuple[float, Edge]] = [(scorer(e), e) for e in candidates]
    # Sort by descending score; ties broken deterministically on the identifiers
    # so that a run is reproducible under the fixed seed of Section 3.7.
    scored.sort(key=lambda item: (-item[0], item[1].parent, item[1].child))
    ranked = [e for _, e in scored]

    if use_leaf_priority:
        ranked = apply_leaf_priority(ranked, top_concept_is_leaf)

    return ranked[:k] if k is not None else ranked


# --------------------------------------------------------------------------- #
# The stage as a whole
# --------------------------------------------------------------------------- #


@dataclass
class CandidateSet:
    """The artefact passed across the Stage 2 / Stage 3 boundary (Table 1)."""

    mention_id: str
    edges: List[Edge]
    scores: Dict[Edge, float]
    n_before_enrichment: int
    n_after_enrichment: int

    @property
    def enrichment_factor(self) -> float:
        if not self.n_before_enrichment:
            return 0.0
        return self.n_after_enrichment / self.n_before_enrichment


def build_candidate_set(
    ontology: Ontology,
    mention_id: str,
    retrieved_concepts: Sequence[str],
    scorer: EdgeScorer,
    k: int = 50,
    use_enrichment: bool = True,
    use_leaf_priority: bool = True,
    seed_edges: Optional[Iterable[Edge]] = None,
) -> CandidateSet:
    """Run Stage 2 end to end for one mention.

    Parameters
    ----------
    retrieved_concepts
        Concepts ranked by Stage 1, highest first. Used both to construct seed
        edges via Eq. (4) and to decide whether the leaf-priority rule fires.
    scorer
        Eq. (9) for the fixed-embedding configuration, Eq. (10) for the
        fine-tuned bi-encoder.
    seed_edges
        Supplied directly when Stage 1 retrieves edges rather than concepts, in
        which case Eq. (4) is bypassed and enrichment starts from the retrieved
        edges themselves.
    use_enrichment
        The ablation switch of Section 4.5. When false the candidate set is the
        seed set, which is the "retrieval alone" row of Table 6.
    """
    if seed_edges is None:
        seeds = construct_edges(ontology, retrieved_concepts)
    else:
        seeds = set(seed_edges)

    n_before = len(seeds)
    candidates = enrich_edges(ontology, seeds) if use_enrichment else seeds
    n_after = len(candidates)

    top_concept_is_leaf = bool(retrieved_concepts) and ontology.is_leaf_concept(
        retrieved_concepts[0]
    )

    ranked = rank_edges(
        candidates,
        scorer,
        k=k,
        top_concept_is_leaf=top_concept_is_leaf,
        use_leaf_priority=use_leaf_priority,
    )

    return CandidateSet(
        mention_id=mention_id,
        edges=ranked,
        scores={e: scorer(e) for e in ranked},
        n_before_enrichment=n_before,
        n_after_enrichment=n_after,
    )


def enrichment_statistics(candidate_sets: Sequence[CandidateSet]) -> Dict[str, float]:
    """Mean candidates before / after enrichment, the "Enrichment effect" of Table 8."""
    if not candidate_sets:
        return {}
    before = sum(c.n_before_enrichment for c in candidate_sets) / len(candidate_sets)
    after = sum(c.n_after_enrichment for c in candidate_sets) / len(candidate_sets)
    return {
        "mean_candidates_before": before,
        "mean_candidates_after": after,
        "ranking_cost_increase": (after / before) if before else 0.0,
    }
