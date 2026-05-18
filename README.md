# PLM-LLM Framework for Semantic Processing and Knowledge Graph Construction

Source code for the paper:

> **A Distributed PLM-LLM Framework for Semantic Processing and Knowledge Graph
> Construction in Mobile and Edge Computing Environments**
> Sadirmekova Zh., Sambetbayeva M., Abdygalym B., Taberkhan R., Karbozova I.
> *The 23rd International Conference on Mobile Systems and Pervasive Computing (MobiSPC 2026)*, Athens, Greece.

## Overview

The framework implements a three-stage distributed pipeline for multilingual
(Kazakh / Russian / English) semantic processing:

1. **Edge layer (PLM)** — candidate extraction using XLM-RoBERTa: tokenization,
   contextual encoding, named-entity recognition, candidate filtering.
2. **Cloud layer (LLM)** — semantic interpretation using GPT-4 with
   instruction-based prompting: concept validation, ontology class assignment,
   relation generation.
3. **Knowledge graph layer** — ontology-guided ranking, conflict detection, and
   integration into the knowledge graph.

## Repository structure

```
plm-llm-kg-framework/
├── README.md
├── requirements.txt
├── pipeline.py              # main three-stage pipeline
├── benchmark.py             # system-level evaluation (latency, bandwidth, memory)
├── ontology.py              # ontology operations and ranking
├── contrastive.py           # contrastive learning utilities
├── prompts/
│   ├── prompt_en.txt        # instruction prompt for English
│   ├── prompt_ru.txt        # instruction prompt for Russian
│   └── prompt_kz.txt        # instruction prompt for Kazakh
└── configs/
    ├── ranking_weights.json # alpha, beta, gamma, delta weights
    └── hyperparameters.json # PLM training hyperparameters
```

## Installation

```bash
git clone https://github.com/baiangali/plm-llm-kg-framework.git
cd plm-llm-kg-framework
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

## Dataset

The replication dataset is available at:
https://github.com/baiangali/plm-llm-kg-dataset

Download `train.jsonl`, `val.jsonl`, `test.jsonl`, `ontology.json` and place
them in a local `data/` folder.

## Running the pipeline

```bash
export OPENAI_API_KEY=sk-...          # optional; falls back to simulation
python pipeline.py --input data/test.jsonl --ontology data/ontology.json --out results/
```

Output: `results/predictions.jsonl` with extracted concepts, assigned ontology
classes, predicted relations, and confidence scores.

## System-level evaluation

To reproduce Table 7 of the paper (latency, throughput, bandwidth, edge memory):

```bash
python benchmark.py --texts data/test.jsonl --n 100
```

See `benchmark.py` docstring for the measurement protocol.

## Configuration

Ranking weights (Section 3.3 of the paper):

| Weight | Value | Description |
|---|---|---|
| α | 0.35 | semantic similarity |
| β | 0.30 | LLM confidence |
| γ | 0.25 | structural compatibility |
| δ | 0.10 | conflict penalty |

Auto-insertion threshold τ = 0.72; review queue range [0.55, 0.72).

These values are stored in `configs/ranking_weights.json` and can be re-tuned
on validation data.

## Hyperparameters

| Parameter | Value |
|---|---|
| PLM model | xlm-roberta-base |
| Max sequence length | 512 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Optimizer | AdamW |
| Epochs | 5 |
| Contrastive temperature | 0.07 |
| Embedding size | 768 |
| Top-K candidates | 10 |

## Citation

```bibtex
@inproceedings{sadirmekova2026plmllm,
  title     = {A Distributed PLM-LLM Framework for Semantic Processing and
               Knowledge Graph Construction in Mobile and Edge Computing
               Environments},
  author    = {Sadirmekova, Zhanna and Sambetbayeva, Madina and
               Abdygalym, Bayangali and Taberkhan, Roman and
               Karbozova, Indira},
  booktitle = {Proceedings of the 23rd International Conference on Mobile
               Systems and Pervasive Computing (MobiSPC)},
  year      = {2026},
  publisher = {Elsevier}
}
```

## Acknowledgements

This research has been funded by the Committee of Science of the Ministry of
Science and Higher Education of the Republic of Kazakhstan, Grant AP26195165.

## License

MIT License — see `LICENSE` file.
