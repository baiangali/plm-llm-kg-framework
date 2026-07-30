"""
Training configuration -- a direct transcription of Table 2 of the paper.

Every default in this module is the value reported in the paper. Nothing here is
tuned; deviations should be made explicit at the call site so that a run can
always be described as "Table 2 defaults, except ...".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Global reproducibility settings (Section 3.7)
# --------------------------------------------------------------------------- #

RANDOM_SEED = 42
SPLIT_RATIOS: Tuple[float, float, float] = (0.70, 0.15, 0.15)  # train / val / test


@dataclass
class BiEncoderConfig:
    """Stage 1 Edge-Bi-encoder -- Table 2, column 1."""

    base_model: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    fine_tuning: str = "full"
    learning_rate: float = 2e-5
    lr_schedule: str = "linear"
    warmup_ratio: float = 0.10
    batch_size: int = 32
    epochs: int = 10
    optimizer: str = "adamw"
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.01
    max_len_mention: int = 128
    max_len_edge: int = 64
    #: Margin alpha of the max-margin triplet loss, Eq. (3).
    margin: float = 0.2
    #: Section 3.3, "Negative sampling": one in-batch negative + four hard negatives.
    n_in_batch_negatives: int = 1
    n_hard_negatives: int = 4
    #: Early stopping on InR_any@10 measured on the validation split, patience 3.
    early_stopping_metric: str = "InR_any@10"
    early_stopping_patience: int = 3
    fp16: bool = True
    seed: int = RANDOM_SEED
    #: Pool from which hard negatives are mined once, before training, and then frozen.
    hard_negative_pool: int = 100


@dataclass
class CrossEncoderConfig:
    """Stage 3 multi-label Edge-Cross-encoder -- Table 2, column 2."""

    base_model: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    fine_tuning: str = "full"
    learning_rate: float = 3e-5
    lr_schedule: str = "linear"
    warmup_ratio: float = 0.10
    batch_size: int = 16
    epochs: int = 5
    optimizer: str = "adamw"
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.01
    max_seq_length: int = 256
    #: Early stopping on InR_any@5 measured on the validation split, patience 2.
    early_stopping_metric: str = "InR_any@5"
    early_stopping_patience: int = 2
    fp16: bool = True
    seed: int = RANDOM_SEED


@dataclass
class LlamaTunedConfig:
    """Stage 3 explainable instruction tuning -- Table 2, column 3 (Section 3.6)."""

    base_model: str = "meta-llama/Llama-2-7b-hf"
    fine_tuning: str = "lora"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Sequence[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    learning_rate: float = 2e-4
    lr_schedule: str = "cosine"
    warmup_ratio: float = 0.03
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    epochs: int = 3
    optimizer: str = "adamw"
    adam_betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    max_seq_length: int = 4096
    bf16: bool = True
    seed: int = RANDOM_SEED
    #: Section 3.6: the 4,096-token window admits candidate lists of roughly 50 edges.
    max_candidates_in_prompt: int = 50
    #: Whether to train the model to emit an "### Explanation" block before "### Response".
    explainable: bool = True


@dataclass
class PipelineConfig:
    """End-to-end run configuration."""

    #: Directory holding the MM-S14-* parts. Relative to the repository root,
    #: the distribution lives under ``data/``, so ``data/MM-S14-Disease`` and
    #: ``data/MM-S14-CPP`` are the two datasets addressed by ``dataset`` below.
    dataset_root: str = "data"
    dataset: str = "MM-S14-Disease"  # or "MM-S14-CPP"
    #: Candidate-list lengths evaluated throughout the paper.
    k_values: Sequence[int] = field(default_factory=lambda: [10, 50])
    #: Stage 1 retriever: "inverted_index" | "fixed_embeddings" | "bi_encoder".
    retriever: str = "bi_encoder"
    #: Stage 3 selector: None | "cross_encoder" | "llm_zero_shot" | "llm_tuned".
    selector: Optional[str] = "cross_encoder"
    #: Number of concepts retrieved in Stage 1 before edge construction.
    n_retrieved_concepts: int = 50
    #: Ablation switches (Section 4.5).
    use_enrichment: bool = True
    use_leaf_priority: bool = True
    #: Use the atomic subsumption graph only, or include complex (verbalised) parents.
    edge_catalogue: str = "all"  # "all" | "atomic"
    #: Where checkpoints, candidate caches and result tables are written.
    output_dir: str = "runs"
    seed: int = RANDOM_SEED
    device: str = "cuda"

    bi_encoder: BiEncoderConfig = field(default_factory=BiEncoderConfig)
    cross_encoder: CrossEncoderConfig = field(default_factory=CrossEncoderConfig)
    llm: LlamaTunedConfig = field(default_factory=LlamaTunedConfig)
