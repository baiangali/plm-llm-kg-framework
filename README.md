# Language-Model-Based Architecture for Automatic Concept Placement in Ontologies

Reference implementation of the three-stage framework described in the manuscript **“Language-Model-Based Architecture for Automatic Concept Placement in Ontologies”**.

The task is **out-of-knowledge-base concept placement**: given a biomedical mention whose concept is absent from the current ontology, the system predicts one or more insertion edges `P → C` in the existing subsumption hierarchy. For leaf placement, `C = NULL`.

The framework supports ontology curators; it does not automatically modify the ontology.

## Framework

1. **Stage 1 — Edge Retrieval**  
   Inverted-index baseline, fixed SapBERT embeddings, or a fine-tuned Edge-Bi-encoder.

2. **Stage 2 — Edge Generation and Enrichment**  
   Candidate edges are constructed and expanded using direct parents, direct children, parent–child combinations, and leaf edges.

3. **Stage 3 — Edge Selection**  
   Final ranking is produced by a multi-label Edge-Cross-encoder or by Llama-2 using zero-shot prompting or explainable instruction tuning.

## Repository Structure

```text
data/
├── MM-S14-Disease/
└── MM-S14-CPP/

oet_placement/
├── config.py
├── data.py
├── pipeline.py
├── stage1_retrieval.py
├── stage2_enrichment.py
├── stage3_cross_encoder.py
├── stage3_llm.py
├── metrics.py
└── logical_validation.py

tests/test_core.py
IMPLEMENTATION.md
requirements-oet.txt
```

The manuscript implementation is contained in `oet_placement/`. Root-level legacy files are not part of this framework.

## Installation

Python 3.9+ is recommended.

```bash
python -m venv .venv
```

Windows CMD:

```bat
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements-oet.txt
```

## Dataset

The experiments use:

- `MM-S14-Disease`
- `MM-S14-CPP`

Source dataset:

- Zenodo: https://zenodo.org/records/10432003
- DOI: https://doi.org/10.5281/zenodo.10432003

The datasets are stored under `data/`.

The original file

```text
data/MM-S14-CPP/mention-edge-pair-level/train.jsonl
```

is stored as two line-preserving parts because it exceeds the GitHub Free Git LFS per-file limit:

```text
train_part_001.jsonl
train_part_002.jsonl
```

The loader streams these files automatically as one logical JSONL input.

For Git LFS:

```bash
git lfs install
git lfs pull
```

## Quick Start

Dataset statistics:

```bat
python -m oet_placement stats --root data --dataset MM-S14-Disease
```

Lexical retrieval with structural enrichment:

```bat
python -m oet_placement retrieve --root data --dataset MM-S14-Disease --retriever inverted_index --k 50
```

Fixed SapBERT retrieval:

```bat
python -m oet_placement retrieve --root data --dataset MM-S14-Disease --retriever fixed_embeddings --k 50
```

Full Edge-Bi-encoder + Edge-Cross-encoder experiment:

```bat
python -m oet_placement experiment --root data --dataset MM-S14-Disease --retriever bi_encoder --train-retriever --selector cross_encoder --k 50
```

## Configuration

### Global settings

| Parameter | Value |
|---|---:|
| Random seed | 42 |
| Train / validation / test | 70% / 15% / 15% |
| Candidate-list sizes | 10, 50 |
| Default dataset | MM-S14-Disease |
| Candidate catalogue | all |
| Structural enrichment | enabled |
| Leaf-priority rule | enabled |

### Edge-Bi-encoder

| Parameter | Value |
|---|---|
| Base model | SapBERT (PubMedBERT-base) |
| Fine-tuning | Full |
| Learning rate | 2e-5 |
| LR schedule / warmup | Linear / 10% |
| Batch size | 32 |
| Epochs | 10 |
| Optimizer | AdamW |
| Adam betas | (0.9, 0.999) |
| Weight decay | 0.01 |
| Max length | 128 mention / 64 edge |
| Margin | 0.2 |
| Negatives | 1 in-batch + 4 hard |
| Early stopping | InR_any@10, patience 3 |
| Precision | fp16 |

### Edge-Cross-encoder

| Parameter | Value |
|---|---|
| Base model | PubMedBERT-base |
| Fine-tuning | Full |
| Learning rate | 3e-5 |
| LR schedule / warmup | Linear / 10% |
| Batch size | 16 |
| Epochs | 5 |
| Optimizer | AdamW |
| Adam betas | (0.9, 0.999) |
| Weight decay | 0.01 |
| Max sequence length | 256 |
| Early stopping | InR_any@5, patience 2 |
| Precision | fp16 |

### Llama-2-7B with LoRA

| Parameter | Value |
|---|---|
| Base model | Llama-2-7B |
| Fine-tuning | LoRA |
| LoRA | r=16, alpha=32, dropout=0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Learning rate | 2e-4 |
| LR schedule / warmup | Cosine / 3% |
| Batch size | 4 |
| Gradient accumulation | 8 |
| Epochs | 3 |
| Optimizer | AdamW |
| Adam betas | (0.9, 0.95) |
| Weight decay | 0.0 |
| Max sequence length | 4096 |
| Precision | bf16 |
| Maximum candidates in prompt | 50 |

All defaults are defined in [`oet_placement/config.py`](oet_placement/config.py).

## Evaluation

The implementation reports:

- `InR_any@k`: at least one gold insertion edge appears in the top-`k`;
- `InR_all@k`: the complete gold edge set appears in the top-`k`;
- Mean Reciprocal Rank;
- Mean Average Precision.

## Testing

```bash
python -m compileall oet_placement tests
python -m unittest discover -s tests -v
python -m oet_placement --help
```

## Notes

- Trained checkpoints are not included.
- Dense-model experiments require GPU resources.
- Llama-2 access may be gated.
- Full ELK validation requires DeepOnto and a working JVM.
- Predicted insertions require expert confirmation.

## Citation

```bibtex
@unpublished{sadirmekova2025conceptplacement,
  author = {Sadirmekova, Zhanna and Sambetbayeva, Madina and Abdygalym, Bayangali and Taberkhan, Roman and Sultangaziyeva, Anar},
  title = {Language-Model-Based Architecture for Automatic Concept Placement in Ontologies},
  year = {2025},
  note = {Manuscript under review}
}
```

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for the detailed equation-to-code mapping.
