# Implementation of the three-stage concept-placement framework

Reference implementation of the algorithm described in *"Language-Model-Based
Architecture for Automatic Concept Placement in Ontologies"* (Sadirmekova,
Sambetbayeva, Abdygalym, Taberkhan, Sultangaziyeva), written against the
MM-S14-Disease and MM-S14-CPP data shipped under `data/` in this repository.

Every module carries the paper's equation numbers in its docstrings, so a section
of the paper maps to a specific block of code.

---

## Layout

| File | Paper section | Contents |
|---|---|---|
| [config.py](oet_placement/config.py) | Table 2, §3.7 | Every hyperparameter, transcribed. Nothing tuned. |
| [serialization.py](oet_placement/serialization.py) | §3.3 notation, §3.2 | Eq. (1), (2), (11); special tokens; verbalisation of complex class expressions. |
| [data.py](oet_placement/data.py) | §1.1, §3.2, Table 3 | `Ontology` (`Par(x)`, `Ch(x)`, descendant queries), `Mention`, gold-edge parsing, mention-level splits. |
| [stage1_retrieval.py](oet_placement/stage1_retrieval.py) | §3.3, Table 4 | Inverted index (BM25), fixed SapBERT embeddings, `EdgeBiEncoder` with the Eq. (3) triplet loss and frozen hard negatives. |
| [stage2_enrichment.py](oet_placement/stage2_enrichment.py) | §3.4 | Eq. (4)–(10): edge construction, neighbourhood enrichment, ranking, leaf-priority rule. |
| [stage3_cross_encoder.py](oet_placement/stage3_cross_encoder.py) | §3.5 | Eq. (11)–(13): joint encoding, `s_cross = v_cross · w`, multi-label BCE. |
| [stage3_llm.py](oet_placement/stage3_llm.py) | §3.6 | Zero-shot prompting; five-step explanation template; LoRA tuning; leakage guard. |
| [metrics.py](oet_placement/metrics.py) | §4.2 | Eq. (14)–(18) in ANY and ALL modes, MRR, MAP, leaf/non-leaf breakdown, paired bootstrap. |
| [logical_validation.py](oet_placement/logical_validation.py) | §4.6, Table 7 | Cycle/consistency/satisfiability checks; graph backend and ELK backend. |
| [pipeline.py](oet_placement/pipeline.py) | §3.1, §4.5 | Stage dispatch, matched candidate pools, ablation runner, cost timing. |
| [cli.py](oet_placement/cli.py) | — | `stats`, `retrieve`, `experiment`, `ablate`. |
| [tests/test_core.py](tests/test_core.py) | — | 65 tests over the dependency-free core. |

---

## Design decisions worth knowing

**Stage 2 is pure Python and encoder-agnostic.** Construction and enrichment
(Eq. 4–8) are set operations over `Par(·)`/`Ch(·)`; only the *scorer* differs
between retrievers. This is what allows §4.5 to ablate enrichment while holding
retrieval fixed, and it means the paper's central contribution is testable
without a GPU.

**The candidate pool is built once per `k` and reused by every selector.** Table 5
compares ranking quality over a matched pool, so the effect of the selection
method is separable from the effect of candidate-set size. `run_stage3` therefore
takes the Stage 2 output as given and only reorders it — a reranker cannot
recover an edge that enrichment failed to generate, which is why `InR@k` is
invariant under reranking when `k` equals the pool size.

**The bi-encoder scores enriched edges on demand.** Eq. (5)–(7) produce two-hop
edges that need not exist in the precomputed catalogue index, so
`EdgeBiEncoder.score_edges` encodes the candidate set directly rather than
looking it up. Looking them up would silently drop exactly the edges enrichment
was introduced to add.

**Leaf enrichment is conditional.** §3.4 states that `P → NULL` is enriched where
`P` occurs in a predicted non-leaf edge. `enrich_edges` enforces that guard
rather than expanding every leaf seed, which would flood the pool.

**One large file ships split, and the loader hides that.**
`data/MM-S14-CPP/mention-edge-pair-level/train.jsonl` is 2.3 GB, over the GitHub
Free Git LFS quota, so it ships as `train_part_001.jsonl` and
`train_part_002.jsonl`. `data.resolve_jsonl_sources` uses the original file when
it exists and otherwise falls back to `<stem>_part_*.jsonl` in lexicographic
order; `data.iter_jsonl` streams the parts in sequence as one logical file and
raises `FileNotFoundError` when neither form is present. The split is
line-by-line, so no part begins or ends mid-record and nothing is buffered beyond
the current line — the 2.3 GB file is never held in memory. Callers name
`train.jsonl` and are unaffected.

**Several shipped files carry a UTF-8 BOM** — `MM-S14-Disease/mention-edge-pair-level/train.jsonl`
and `MM-S14-CPP/.../valid.jsonl` among them — which `json.loads` rejects. The
reader opens files as `utf-8-sig`, which strips a leading BOM when present and is
otherwise identical to `utf-8`.

**The instruction-tuning template is supervision-only.** `build_explanation_target`
is instantiated with gold parents, children and option indices; `build_prompt` is
not. `assert_no_leakage` checks mechanically that no gold label or answer marker
appears in an inference-time prompt outside the candidate list, so §3.6's claim
that "there is no path by which reference information could reach the model" is
verified per call rather than asserted.

**Unchecked conditions are reported as `None`, not as a pass.** The default
`GraphBackend` decides the cycle condition exactly — inserting `c` at `P → C`
creates a cycle iff `P` is already a descendant of `C` — but consistency and
unsatisfiability depend on disjointness and existential axioms that the taxonomy
alone does not carry. It returns `None` for those two. Reproducing Table 7 in
full needs `DeepOntoELKBackend`, which runs ELK over the shipped `.owl` files and
requires `deeponto` plus a JVM.

**ROC-AUC is deliberately absent** — §4.2 rules it out, and implementing it would
invite a comparison the paper argues against.

---

## Running it

All commands are run from the repository root. `--root` defaults to `data`, which
is where the two MM-S14 parts live, so it can be omitted; it is spelled out below
for clarity.

```powershell
pip install -r requirements.txt          # torch/transformers/peft; see notes below
python -m unittest discover -s tests -v  # no dependencies needed

# Dataset statistics as they actually are on disk (stdlib only)
python -m oet_placement stats --root data --dataset MM-S14-Disease

# Stages 1+2 with the lexical baseline (stdlib only) -- the Table 4 bottom row
python -m oet_placement retrieve --root data --retriever inverted_index --k 50

# Full pipeline: fine-tune the bi-encoder, then the cross-encoder, then evaluate
python -m oet_placement experiment `
    --root data --dataset MM-S14-Disease `
    --retriever bi_encoder --train-retriever `
    --selector cross_encoder --k 50

# Component ablation (Table 6)
python -m oet_placement ablate --root data --k 50 `
    --checkpoint runs/bi_encoder/best `
    --selector-checkpoint runs/cross_encoder/best
```

`--split-mode native` (default) uses the splits shipped with the dataset and
evaluates on `test-NIL`, whose mentions denote concepts introduced in the later
SNOMED CT release and are therefore out-of-knowledge-base by construction.
`--split-mode resplit` pools the out-of-KB mentions and re-splits them 70/15/15
at the mention level under seed 42, as §3.7 describes.

---

## Status and caveats

**Test status.** `python -m unittest discover -s tests` runs 65 tests on
CPython 3.11 with no third-party packages installed; all pass. Five of them —
the `TestRealDataset` class — are skipped unless `data/MM-S14-Disease` is
present, and exercise the real ontology rather than toy fixtures. Every module
imports with `torch`, `transformers`, `numpy` and `peft` blocked, so the
structural core and `--help` work on a bare interpreter.

**Model training and inference are unverified.** Nothing in `stage1_retrieval`,
`stage3_cross_encoder` or `stage3_llm` beyond prompt construction and parsing has
been executed: that needs a GPU and the gated Llama-2 weights. The tests cover
the dependency-free core only, so no accuracy figure in the paper is reproduced
or checked here.

**Three points where the on-disk data does not match Table 3.** Flagging these
because they affect how a reproduction should be set up, not to dispute the
paper — the released artefact may differ from what was used for the reported runs.

| Table 3 | On disk (Disease / CPP) |
|---|---|
| Mentions train/val/test: 1,842 / 395 / 395 (Disease), 2,106 / 451 / 452 (CPP) | Out-of-KB mention pool is 605 (Disease) and 1,000 (CPP) in `test-NIL-all.jsonl`; a 70/15/15 split of those gives roughly 424/91/91 and 700/150/150. The in-KB `train.jsonl` holds 11,812 / 34,704. |
| Total candidate space \|E\|: 64,318 / 71,940 | Edge catalogues hold 237,826 / 625,994 edges (`-edges-all`) and 232,829 / 606,593 (`-edges-atomic`). Entity catalogues hold 64,900 / 175,895 concepts. The Disease figure is near its *concept* count; neither figure matches an edge count. |
| Mean candidates before/after enrichment: 412 / 3,847 | Reproducible from a run — `enrichment_statistics()` reports it. Worth checking against your own numbers. |

**Hyperparameters are Table 2 values, unmodified**, including ones the paper marks
as untuned (the in-batch/hard negative ratio was "fixed in advance rather than
tuned on the validation split"). Changing them is a deviation to state explicitly.

**Base model identifiers** in `config.py` are the public HuggingFace checkpoints
matching the paper's descriptions (SapBERT-from-PubMedBERT-fulltext,
BiomedBERT-base, Llama-2-7b-hf). Llama-2 is gated and needs an accepted licence.

**Single-seed runs prove little.** §4.4 says differences of one to two percentage
points should not be treated as established. `metrics.paired_bootstrap` is
provided so that claim can be checked rather than assumed.
