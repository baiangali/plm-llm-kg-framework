"""
Language-Model-Based Architecture for Automatic Concept Placement in Ontologies.

Reference implementation of the three-stage framework described in:

    Sadirmekova, Zh.; Sambetbayeva, M.; Abdygalym, B.; Taberkhan, R.; Sultangaziyeva, A.
    "Language-Model-Based Architecture for Automatic Concept Placement in Ontologies."

Stage 1 -- Edge retrieval          (Section 3.3, Eq. 1-3)
Stage 2 -- Edge generation and
           structural enrichment   (Section 3.4, Eq. 4-10)
Stage 3 -- Edge selection          (Sections 3.5-3.6, Eq. 11-13)
Evaluation                         (Section 4.2, Eq. 14-18)
Logical validation                 (Section 4.6)

Heavy dependencies (torch / transformers / peft) are imported lazily inside the
modules that need them, so that the purely structural parts of the framework --
the ontology model, Stage 2, the metrics, and the logical validation -- run on a
stdlib-only installation.
"""

__version__ = "1.0.0"

from .config import (
    BiEncoderConfig,
    CrossEncoderConfig,
    LlamaTunedConfig,
    PipelineConfig,
)
from .data import (
    NULL_ID,
    NULL_TITLE,
    Edge,
    Mention,
    Ontology,
    load_mentions,
    load_edge_pairs,
    make_splits,
    iter_jsonl,
    resolve_jsonl_sources,
)
from .metrics import evaluate_rankings, PlacementScores
from .stage2_enrichment import (
    construct_edges_from_concept,
    enrich_edge,
    build_candidate_set,
)

__all__ = [
    "__version__",
    "BiEncoderConfig",
    "CrossEncoderConfig",
    "LlamaTunedConfig",
    "PipelineConfig",
    "NULL_ID",
    "NULL_TITLE",
    "Edge",
    "Mention",
    "Ontology",
    "load_mentions",
    "load_edge_pairs",
    "make_splits",
    "iter_jsonl",
    "resolve_jsonl_sources",
    "evaluate_rankings",
    "PlacementScores",
    "construct_edges_from_concept",
    "enrich_edge",
    "build_candidate_set",
]
