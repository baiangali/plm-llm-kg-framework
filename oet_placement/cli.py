"""
Command-line entry points.

    python -m oet_placement stats        --dataset MM-S14-Disease
    python -m oet_placement retrieve     --retriever inverted_index --k 50
    python -m oet_placement experiment   --retriever bi_encoder --selector cross_encoder
    python -m oet_placement ablate       --k 50

``stats``, and ``retrieve`` with the ``inverted_index`` retriever, need nothing
beyond the standard library. The dense retrievers and both selectors require
torch/transformers, and the instruction-tuned selector additionally requires peft.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

from .config import PipelineConfig
from .data import Mention, Ontology
from .metrics import evaluate_rankings, format_table
from .pipeline import (
    RunResult,
    format_ablation,
    load_dataset,
    report_statistics,
    run,
    run_ablation,
    run_stage1,
    run_stage2,
    save_predictions,
    save_results,
)


def _build_config(args) -> PipelineConfig:
    config = PipelineConfig(
        dataset_root=args.root,
        dataset=args.dataset,
        retriever=getattr(args, "retriever", "inverted_index"),
        selector=getattr(args, "selector", None),
        n_retrieved_concepts=getattr(args, "n_retrieved", 50),
        use_enrichment=not getattr(args, "no_enrichment", False),
        use_leaf_priority=not getattr(args, "no_leaf_priority", False),
        edge_catalogue=getattr(args, "edge_catalogue", "all"),
        output_dir=getattr(args, "output", "runs"),
        device=getattr(args, "device", "cuda"),
    )
    if getattr(args, "k", None):
        config.k_values = [args.k]
    return config


def _build_retriever(config: PipelineConfig, ontology: Ontology, checkpoint: Optional[str]):
    if config.retriever == "inverted_index":
        from .stage1_retrieval import InvertedIndexRetriever

        print("[stage 1] building inverted index ...")
        return InvertedIndexRetriever().fit(ontology)

    if config.retriever == "fixed_embeddings":
        from .stage1_retrieval import FixedEmbeddingRetriever

        print("[stage 1] encoding the concept catalogue with frozen SapBERT ...")
        return FixedEmbeddingRetriever(
            base_model=config.bi_encoder.base_model, device=config.device
        ).fit(ontology)

    if config.retriever == "bi_encoder":
        from .stage1_retrieval import EdgeBiEncoder

        model = EdgeBiEncoder(config.bi_encoder, device=config.device)
        if checkpoint and os.path.exists(checkpoint):
            print(f"[stage 1] loading bi-encoder checkpoint from {checkpoint}")
            model.load(checkpoint)
        print("[stage 1] indexing the edge catalogue ...")
        model.index_edges(ontology)
        return model

    raise ValueError(f"unknown retriever {config.retriever!r}")


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def cmd_stats(args) -> int:
    config = _build_config(args)
    ontology, splits = load_dataset(config, split_mode=args.split_mode)
    print(f"# {config.dataset}  ({args.split_mode} splits)")
    print(report_statistics(ontology, splits))
    return 0


def cmd_retrieve(args) -> int:
    config = _build_config(args)
    ontology, splits = load_dataset(config, split_mode=args.split_mode)
    mentions = splits[args.split]
    retriever = _build_retriever(config, ontology, args.checkpoint)

    result = run(config, ontology, mentions, retriever, selector=None, k=args.k)
    print(result.scores)
    print(json.dumps(result.enrichment, indent=2))
    print(json.dumps(result.timing, indent=2))

    if args.output:
        save_results({f"{config.retriever}@{args.k}": result}, os.path.join(args.output, "retrieval.json"))
        save_predictions(
            ontology, mentions, result.rankings, os.path.join(args.output, "predictions.jsonl")
        )
    return 0


def cmd_experiment(args) -> int:
    """Train, then evaluate the full pipeline over a matched candidate pool."""
    config = _build_config(args)
    ontology, splits = load_dataset(config, split_mode=args.split_mode)
    train, valid = splits["train"], splits.get("valid", [])
    test = splits[args.split]

    if args.max_train:
        train = train[: args.max_train]
    if args.max_valid:
        valid = valid[: args.max_valid]

    # ---- Stage 1 -------------------------------------------------------- #
    if config.retriever == "bi_encoder" and args.train_retriever:
        from .stage1_retrieval import EdgeBiEncoder

        model = EdgeBiEncoder(config.bi_encoder, device=config.device)

        def evaluate_bi(m):
            cfg = PipelineConfig(**{**config.__dict__, "selector": None})
            res = run(cfg, ontology, valid, m, None, k=10, validate_logic=False)
            return res.scores.inr_any.get(10, 0.0)

        print("[stage 1] fine-tuning the Edge-Bi-encoder ...")
        model.train(
            ontology,
            train,
            valid,
            output_dir=os.path.join(config.output_dir, "bi_encoder"),
            evaluate_fn=evaluate_bi if valid else None,
        )
        retriever = model
    else:
        retriever = _build_retriever(config, ontology, args.checkpoint)

    # ---- Stages 1+2 over a matched pool --------------------------------- #
    k = args.k
    print(f"[stage 2] building candidate sets at k={k} ...")
    pools: Dict[str, Dict[str, List]] = {}
    for name, subset in (("train", train), ("valid", valid), ("test", test)):
        if not subset:
            continue
        stage1 = run_stage1(config, ontology, subset, retriever)
        sets = run_stage2(config, ontology, subset, stage1, k, retriever=retriever)
        pools[name] = {mid: cs.edges for mid, cs in sets.items()}

    # ---- Stage 3 -------------------------------------------------------- #
    selector = None
    if config.selector == "cross_encoder":
        from .stage3_cross_encoder import EdgeCrossEncoder, build_examples

        selector = EdgeCrossEncoder(config.cross_encoder, device=config.device)
        if args.selector_checkpoint and os.path.exists(args.selector_checkpoint):
            selector.load(args.selector_checkpoint)
        else:
            examples = build_examples(train, pools["train"])
            print(f"[stage 3] training the Edge-Cross-encoder on {len(examples)} pairs ...")

            def evaluate_cross(model):
                reranked = model.rerank(ontology, valid, pools["valid"])
                return evaluate_rankings(valid, reranked, k_values=[5]).inr_any[5]

            selector.train(
                ontology,
                examples,
                output_dir=os.path.join(config.output_dir, "cross_encoder"),
                evaluate_fn=evaluate_cross if valid else None,
            )

    elif config.selector in ("llm_zero_shot", "llm_tuned"):
        from .stage3_llm import LLMEdgeSelector

        selector = LLMEdgeSelector(
            config.llm, device=config.device, adapter_path=args.selector_checkpoint
        )
        if config.selector == "llm_tuned" and not args.selector_checkpoint:
            print("[stage 3] instruction-tuning with LoRA ...")
            selector.train(
                ontology,
                train,
                pools["train"],
                output_dir=os.path.join(config.output_dir, "llm_tuned"),
            )

    # ---- Evaluation ----------------------------------------------------- #
    results: Dict[str, RunResult] = {}
    for split_name, subset in (("valid", valid), ("test", test)):
        if not subset:
            continue
        results[split_name] = run(
            config, ontology, subset, retriever, selector, k=k
        )
        print(f"\n== {split_name} ==")
        print(results[split_name].scores)

    if config.output_dir:
        save_results(results, os.path.join(config.output_dir, "experiment.json"))
        if "test" in results:
            save_predictions(
                ontology,
                test,
                results["test"].rankings,
                os.path.join(config.output_dir, "test_predictions.jsonl"),
            )
    return 0


def cmd_ablate(args) -> int:
    config = _build_config(args)
    ontology, splits = load_dataset(config, split_mode=args.split_mode)
    mentions = splits[args.split]
    retriever = _build_retriever(config, ontology, args.checkpoint)

    cross_encoder = None
    if args.selector_checkpoint:
        from .stage3_cross_encoder import EdgeCrossEncoder

        cross_encoder = EdgeCrossEncoder(config.cross_encoder, device=config.device)
        cross_encoder.load(args.selector_checkpoint)

    results = run_ablation(
        config, ontology, mentions, retriever, cross_encoder=cross_encoder, k=args.k
    )
    print(format_ablation(results, k=min(10, args.k)))
    if args.output:
        save_results(results, os.path.join(args.output, "ablation.json"))
    return 0


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oet_placement",
        description="Three-stage language-model framework for concept placement in ontologies.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument(
            "--root",
            default="data",
            help="directory containing the MM-S14-* parts (default: data)",
        )
        p.add_argument(
            "--dataset", default="MM-S14-Disease", choices=["MM-S14-Disease", "MM-S14-CPP"]
        )
        p.add_argument(
            "--split-mode", default="native", choices=["native", "resplit"], dest="split_mode"
        )
        p.add_argument("--edge-catalogue", default="all", choices=["all", "atomic"])
        p.add_argument("--output", default="runs")
        p.add_argument("--device", default="cuda")

    p_stats = sub.add_parser("stats", help="dataset statistics (Table 3)")
    common(p_stats)
    p_stats.set_defaults(func=cmd_stats)

    p_ret = sub.add_parser("retrieve", help="Stages 1+2 only (Table 4)")
    common(p_ret)
    p_ret.add_argument(
        "--retriever", default="inverted_index",
        choices=["inverted_index", "fixed_embeddings", "bi_encoder"],
    )
    p_ret.add_argument("--split", default="test_nil")
    p_ret.add_argument("--k", type=int, default=50)
    p_ret.add_argument("--n-retrieved", type=int, default=50, dest="n_retrieved")
    p_ret.add_argument("--checkpoint", default=None)
    p_ret.add_argument("--no-enrichment", action="store_true", dest="no_enrichment")
    p_ret.add_argument("--no-leaf-priority", action="store_true", dest="no_leaf_priority")
    p_ret.set_defaults(func=cmd_retrieve)

    p_exp = sub.add_parser("experiment", help="train and evaluate the full pipeline (Table 5)")
    common(p_exp)
    p_exp.add_argument(
        "--retriever", default="bi_encoder",
        choices=["inverted_index", "fixed_embeddings", "bi_encoder"],
    )
    p_exp.add_argument(
        "--selector", default="cross_encoder",
        choices=["cross_encoder", "llm_zero_shot", "llm_tuned", "none"],
    )
    p_exp.add_argument("--split", default="test_nil")
    p_exp.add_argument("--k", type=int, default=50)
    p_exp.add_argument("--n-retrieved", type=int, default=50, dest="n_retrieved")
    p_exp.add_argument("--train-retriever", action="store_true", dest="train_retriever")
    p_exp.add_argument("--checkpoint", default=None)
    p_exp.add_argument("--selector-checkpoint", default=None, dest="selector_checkpoint")
    p_exp.add_argument("--max-train", type=int, default=None, dest="max_train")
    p_exp.add_argument("--max-valid", type=int, default=None, dest="max_valid")
    p_exp.add_argument("--no-enrichment", action="store_true", dest="no_enrichment")
    p_exp.add_argument("--no-leaf-priority", action="store_true", dest="no_leaf_priority")
    p_exp.set_defaults(func=cmd_experiment)

    p_abl = sub.add_parser("ablate", help="component ablation (Table 6)")
    common(p_abl)
    p_abl.add_argument(
        "--retriever", default="bi_encoder",
        choices=["inverted_index", "fixed_embeddings", "bi_encoder"],
    )
    p_abl.add_argument("--split", default="test_nil")
    p_abl.add_argument("--k", type=int, default=50)
    p_abl.add_argument("--n-retrieved", type=int, default=50, dest="n_retrieved")
    p_abl.add_argument("--checkpoint", default=None)
    p_abl.add_argument("--selector-checkpoint", default=None, dest="selector_checkpoint")
    p_abl.set_defaults(func=cmd_ablate)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "selector", None) == "none":
        args.selector = None
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
