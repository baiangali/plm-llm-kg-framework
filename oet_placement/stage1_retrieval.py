"""
Stage 1 -- Edge retrieval (Section 3.3).

Three retrieval strategies are provided, matching the three rows of Table 4:

  * :class:`InvertedIndexRetriever`  -- lexical baseline (BM25 over labels and
    synonyms), dependency-free.
  * :class:`FixedEmbeddingRetriever` -- concept retrieval with frozen SapBERT
    embeddings, ranked by cosine similarity.
  * :class:`EdgeBiEncoder`           -- the fine-tuned bi-encoder, which aligns
    mentions with *serialised edges* rather than with concepts.

The bi-encoder is trained with the max-margin triplet loss of Eq. (3),

    L = max(0, alpha + s(m, e-) - s(m, e+)),   alpha = 0.2

with one in-batch negative and four hard negatives per positive pair. The hard
negatives are mined once, before training, from the highest-ranked incorrect
candidate edges, and the set is then frozen (Section 3.3, "Negative sampling").

``torch`` and ``transformers`` are imported lazily so that the lexical baseline
and every downstream structural component remain usable without them.
"""

from __future__ import annotations

import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .config import BiEncoderConfig
from .data import NULL_ID, Edge, Mention, Ontology
from .serialization import (
    add_special_tokens,
    serialize_edge,
    serialize_mention,
    strip_semantic_tag,
    tokenise_for_lexical_match,
)


# --------------------------------------------------------------------------- #
# Lexical baseline -- inverted index
# --------------------------------------------------------------------------- #


class InvertedIndexRetriever:
    """BM25 over concept titles and synonyms.

    Section 4.10 motivates this baseline: it is reproducible, widely used in
    semantic retrieval, and establishes a lower bound against which the
    contribution of the context-dependent and structural components is isolated.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self._doc_ids: List[str] = []
        self._doc_len: List[int] = []
        self._avg_len: float = 0.0
        self._idf: Dict[str, float] = {}

    def fit(self, ontology: Ontology, with_synonyms: bool = True) -> "InvertedIndexRetriever":
        doc_freq: Counter = Counter()
        for concept in ontology.concepts.values():
            doc_id = len(self._doc_ids)
            self._doc_ids.append(concept.idx)

            surface = [strip_semantic_tag(concept.title)]
            if with_synonyms:
                surface.extend(concept.synonyms)
            tokens: List[str] = []
            for text in surface:
                tokens.extend(tokenise_for_lexical_match(text))

            self._doc_len.append(len(tokens))
            counts = Counter(tokens)
            for term, tf in counts.items():
                self._postings[term].append((doc_id, tf))
            doc_freq.update(counts.keys())

        n_docs = max(1, len(self._doc_ids))
        self._avg_len = sum(self._doc_len) / n_docs
        for term, df in doc_freq.items():
            # Robertson/Sparck-Jones idf with the usual +0.5 smoothing.
            self._idf[term] = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        return self

    def retrieve_concepts(self, mention: Mention, top_n: int = 50) -> List[Tuple[str, float]]:
        query = tokenise_for_lexical_match(mention.mention)
        scores: Dict[int, float] = defaultdict(float)
        for term in query:
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf.get(term, 0.0)
            for doc_id, tf in postings:
                norm = 1.0 - self.b + self.b * (self._doc_len[doc_id] / max(1e-9, self._avg_len))
                scores[doc_id] += idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm)

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
        return [(self._doc_ids[doc_id], score) for doc_id, score in ranked]


# --------------------------------------------------------------------------- #
# Dense encoders (torch)
# --------------------------------------------------------------------------- #


def _require_torch():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The dense retrievers require torch and transformers. "
            "Install them with `pip install -r requirements.txt`."
        ) from exc
    import torch
    import transformers

    return torch, transformers


def _cls_pool(hidden_state, attention_mask=None):
    """SapBERT and Sentence-BERT style pooling: the final-layer [CLS] vector."""
    return hidden_state[:, 0, :]


class _TextEncoder:
    """Batched text -> vector encoder over a HuggingFace model."""

    def __init__(self, model, tokenizer, max_length: int, device: str, normalize: bool):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.normalize = normalize

    def encode(self, texts: Sequence[str], batch_size: int = 256, show_progress: bool = False):
        torch, _ = _require_torch()
        import numpy as np

        self.model.eval()
        out: List["np.ndarray"] = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, desc="encoding", unit="batch")
            except ImportError:
                pass

        with torch.no_grad():
            for start in iterator:
                batch = list(texts[start : start + batch_size])
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**enc).last_hidden_state
                vecs = _cls_pool(hidden, enc.get("attention_mask"))
                if self.normalize:
                    vecs = torch.nn.functional.normalize(vecs, p=2, dim=-1)
                out.append(vecs.detach().cpu().float().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, 768), dtype="float32")


class FixedEmbeddingRetriever:
    """Concept retrieval with frozen SapBERT embeddings, ranked by cosine similarity.

    Section 3.3: "Retrieval is performed with a pre-trained language model whose
    parameters are not updated." Complex concepts are verbalised first, which
    :meth:`Ontology.title` handles.
    """

    def __init__(
        self,
        base_model: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        device: str = "cuda",
        max_length: int = 64,
        with_synonyms: bool = True,
    ) -> None:
        torch, transformers = _require_torch()
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(base_model)
        model = transformers.AutoModel.from_pretrained(base_model).to(self.device)
        self.encoder = _TextEncoder(model, self.tokenizer, max_length, self.device, normalize=True)
        self.with_synonyms = with_synonyms
        self.concept_ids: List[str] = []
        self.concept_matrix = None  # np.ndarray [n_concepts, dim], L2-normalised

    def fit(self, ontology: Ontology, batch_size: int = 256) -> "FixedEmbeddingRetriever":
        self.concept_ids = list(ontology.concepts.keys())
        texts = [
            ontology.concept_text(idx, with_synonyms=self.with_synonyms)
            for idx in self.concept_ids
        ]
        self.concept_matrix = self.encoder.encode(texts, batch_size=batch_size, show_progress=True)
        return self

    def encode_mentions(self, mentions: Sequence[Mention], batch_size: int = 128):
        texts = [
            serialize_mention(m.context_left, m.mention, m.context_right) for m in mentions
        ]
        return self.encoder.encode(texts, batch_size=batch_size)

    def similarity_map(self, mention_vector, concept_ids: Iterable[str]) -> Dict[str, float]:
        """``cos(m, x)`` for the concepts named, as required by Eq. (9)."""
        import numpy as np

        index = {cid: i for i, cid in enumerate(self.concept_ids)}
        result: Dict[str, float] = {}
        for cid in concept_ids:
            row = index.get(cid)
            if row is not None:
                result[cid] = float(np.dot(self.concept_matrix[row], mention_vector))
        return result

    def retrieve_concepts(self, mention: Mention, top_n: int = 50) -> List[Tuple[str, float]]:
        import numpy as np

        vec = self.encode_mentions([mention])[0]
        sims = self.concept_matrix @ vec
        top = np.argpartition(-sims, min(top_n, len(sims) - 1))[:top_n]
        top = top[np.argsort(-sims[top])]
        return [(self.concept_ids[i], float(sims[i])) for i in top]

    def retrieve_concepts_batch(
        self, mentions: Sequence[Mention], top_n: int = 50, batch_size: int = 128
    ) -> List[List[Tuple[str, float]]]:
        import numpy as np

        vectors = self.encode_mentions(mentions, batch_size=batch_size)
        results: List[List[Tuple[str, float]]] = []
        for vec in vectors:
            sims = self.concept_matrix @ vec
            n = min(top_n, len(sims))
            top = np.argpartition(-sims, n - 1)[:n]
            top = top[np.argsort(-sims[top])]
            results.append([(self.concept_ids[i], float(sims[i])) for i in top])
        return results


# --------------------------------------------------------------------------- #
# Edge-Bi-encoder -- Eq. (2), (3), (10)
# --------------------------------------------------------------------------- #


def _build_bi_encoder_module(config: BiEncoderConfig, device: str):
    """Two separate encoders, one for mentions and one for serialised edges.

    Section 3.3 motivates the separation by "the differing statistical properties
    of natural-language context and ontological serialisations".
    """
    torch, transformers = _require_torch()

    class _BiEncoderModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(config.base_model)
            self.mention_encoder = transformers.AutoModel.from_pretrained(config.base_model)
            self.edge_encoder = transformers.AutoModel.from_pretrained(config.base_model)
            add_special_tokens(self.tokenizer, self.mention_encoder)
            self.edge_encoder.resize_token_embeddings(len(self.tokenizer))

        def encode_mention(self, **inputs):
            return self.mention_encoder(**inputs).last_hidden_state[:, 0, :]

        def encode_edge(self, **inputs):
            return self.edge_encoder(**inputs).last_hidden_state[:, 0, :]

        def score(self, mention_vec, edge_vec):
            """Eq. (10): ``s(m, e) = v_m . v_e``."""
            return (mention_vec * edge_vec).sum(dim=-1)

    return _BiEncoderModule().to(device)


class EdgeBiEncoder:
    """The fine-tuned Stage 1 retriever, operating over serialised edges."""

    def __init__(self, config: Optional[BiEncoderConfig] = None, device: str = "cuda") -> None:
        torch, _ = _require_torch()
        self.config = config or BiEncoderConfig()
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.module = _build_bi_encoder_module(self.config, self.device)
        self.edge_index: List[Edge] = []
        self.edge_matrix = None  # np.ndarray [n_edges, dim]

    # -- encoding -------------------------------------------------------- #

    def _encode(self, texts: Sequence[str], which: str, batch_size: int = 256):
        torch, _ = _require_torch()
        import numpy as np

        max_len = (
            self.config.max_len_mention if which == "mention" else self.config.max_len_edge
        )
        encode_fn = (
            self.module.encode_mention if which == "mention" else self.module.encode_edge
        )
        self.module.eval()
        chunks: List["np.ndarray"] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start : start + batch_size])
                enc = self.module.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                ).to(self.device)
                chunks.append(encode_fn(**enc).detach().cpu().float().numpy())
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 768), dtype="float32")

    def index_edges(
        self,
        ontology: Ontology,
        edges: Optional[Sequence[Edge]] = None,
        batch_size: int = 256,
    ) -> "EdgeBiEncoder":
        """Embed the edge catalogue so that retrieval is a single matrix product."""
        self.edge_index = list(edges if edges is not None else ontology.edge_catalogue)
        texts = [serialize_edge(*ontology.edge_text(e)) for e in self.edge_index]
        self.edge_matrix = self._encode(texts, "edge", batch_size=batch_size)
        return self

    def retrieve_edges(
        self,
        mentions: Sequence[Mention],
        top_n: int = 50,
        batch_size: int = 128,
    ) -> List[List[Tuple[Edge, float]]]:
        """Rank the edge catalogue against each mention by Eq. (10)."""
        import numpy as np

        texts = [
            serialize_mention(m.context_left, m.mention, m.context_right) for m in mentions
        ]
        vectors = self._encode(texts, "mention", batch_size=batch_size)

        results: List[List[Tuple[Edge, float]]] = []
        for vec in vectors:
            scores = self.edge_matrix @ vec
            n = min(top_n, len(scores))
            top = np.argpartition(-scores, n - 1)[:n]
            top = top[np.argsort(-scores[top])]
            results.append([(self.edge_index[i], float(scores[i])) for i in top])
        return results

    def score_edges(
        self,
        ontology: Ontology,
        mention: Mention,
        edges: Sequence[Edge],
        batch_size: int = 256,
    ) -> Dict[Edge, float]:
        """Eq. (10) for an arbitrary edge set, including edges built by Stage 2.

        Enrichment (Eq. 5-7) can produce two-hop edges that are absent from the
        precomputed catalogue index, so the candidate set is encoded on demand
        rather than looked up.
        """
        import numpy as np

        edges = list(edges)
        if not edges:
            return {}
        edge_texts = [serialize_edge(*ontology.edge_text(e)) for e in edges]
        edge_vecs = self._encode(edge_texts, "edge", batch_size=batch_size)
        mention_vec = self._encode(
            [serialize_mention(mention.context_left, mention.mention, mention.context_right)],
            "mention",
        )[0]
        scores = edge_vecs @ mention_vec
        return {e: float(s) for e, s in zip(edges, scores)}

    def concept_similarity(
        self,
        ontology: Ontology,
        mention: Mention,
        concept_ids: Sequence[str],
        batch_size: int = 256,
    ) -> Dict[str, float]:
        """``cos(m, x)`` over named concepts, for the Eq. (9) fallback ranking."""
        import numpy as np

        concept_ids = list(concept_ids)
        if not concept_ids:
            return {}
        texts = [ontology.concept_text(idx) for idx in concept_ids]
        vecs = self._encode(texts, "edge", batch_size=batch_size)
        mention_vec = self._encode(
            [serialize_mention(mention.context_left, mention.mention, mention.context_right)],
            "mention",
        )[0]
        vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
        mention_vec = mention_vec / (np.linalg.norm(mention_vec) + 1e-12)
        return {idx: float(s) for idx, s in zip(concept_ids, vecs @ mention_vec)}

    # -- hard negative mining -------------------------------------------- #

    def mine_hard_negatives(
        self,
        ontology: Ontology,
        mentions: Sequence[Mention],
        n_hard: int = 4,
        pool: int = 100,
        batch_size: int = 128,
    ) -> Dict[str, List[Edge]]:
        """Highest-ranked *incorrect* candidate edges, mined once before training.

        Section 3.3: the hard-negative set remains fixed throughout training, and
        structural negatives drawn from the ontological neighbourhood of the gold
        parent are deliberately not used.
        """
        if self.edge_matrix is None:
            self.index_edges(ontology, batch_size=batch_size)

        retrieved = self.retrieve_edges(mentions, top_n=pool, batch_size=batch_size)
        negatives: Dict[str, List[Edge]] = {}
        for m, candidates in zip(mentions, retrieved):
            wrong = [e for e, _ in candidates if e not in m.gold_edges]
            negatives[m.mention_id] = wrong[:n_hard]
        return negatives

    # -- training -------------------------------------------------------- #

    def train(
        self,
        ontology: Ontology,
        train_mentions: Sequence[Mention],
        valid_mentions: Optional[Sequence[Mention]] = None,
        output_dir: str = "runs/bi_encoder",
        evaluate_fn=None,
    ) -> "EdgeBiEncoder":
        """Fine-tune with the max-margin triplet loss of Eq. (3).

        ``evaluate_fn(model) -> float`` is called after each epoch and should
        return the early-stopping metric named in Table 2 (InR_any@10 on the
        validation split). Training stops after ``early_stopping_patience``
        epochs without improvement, and the best checkpoint is restored.
        """
        torch, transformers = _require_torch()
        cfg = self.config
        os.makedirs(output_dir, exist_ok=True)

        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)

        hard_negatives = self.mine_hard_negatives(
            ontology, train_mentions, n_hard=cfg.n_hard_negatives, pool=cfg.hard_negative_pool
        )

        # Flatten to one training example per (mention, gold edge) pair.
        examples: List[Tuple[Mention, Edge]] = [
            (m, e) for m in train_mentions for e in m.gold_edges
        ]

        optimizer = torch.optim.AdamW(
            self.module.parameters(),
            lr=cfg.learning_rate,
            betas=cfg.adam_betas,
            weight_decay=cfg.weight_decay,
        )
        steps_per_epoch = math.ceil(len(examples) / cfg.batch_size)
        total_steps = steps_per_epoch * cfg.epochs
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(cfg.warmup_ratio * total_steps),
            num_training_steps=total_steps,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16 and self.device == "cuda")

        best_metric = float("-inf")
        best_epoch = -1
        epochs_without_improvement = 0
        rng = random.Random(cfg.seed)

        for epoch in range(cfg.epochs):
            self.module.train()
            order = list(range(len(examples)))
            rng.shuffle(order)
            running_loss = 0.0

            for step in range(steps_per_epoch):
                batch_idx = order[step * cfg.batch_size : (step + 1) * cfg.batch_size]
                if not batch_idx:
                    continue
                batch = [examples[i] for i in batch_idx]

                mention_texts = [
                    serialize_mention(m.context_left, m.mention, m.context_right)
                    for m, _ in batch
                ]
                positive_texts = [
                    serialize_edge(*ontology.edge_text(e)) for _, e in batch
                ]

                # Negatives: cfg.n_in_batch_negatives in-batch + cfg.n_hard_negatives hard.
                negative_texts: List[List[str]] = []
                for i, (m, gold) in enumerate(batch):
                    negs: List[Edge] = []
                    for _ in range(cfg.n_in_batch_negatives):
                        if len(batch) > 1:
                            j = rng.randrange(len(batch))
                            attempts = 0
                            while (batch[j][1] in m.gold_edges) and attempts < 5:
                                j = rng.randrange(len(batch))
                                attempts += 1
                            if batch[j][1] not in m.gold_edges:
                                negs.append(batch[j][1])
                    negs.extend(hard_negatives.get(m.mention_id, [])[: cfg.n_hard_negatives])
                    negative_texts.append(
                        [serialize_edge(*ontology.edge_text(e)) for e in negs]
                    )

                n_neg = max(1, max(len(n) for n in negative_texts))
                # Pad ragged negative lists by repeating the last entry; the loss
                # of a duplicated negative is identical, so this does not change
                # the objective, only the tensor shape.
                flat_negatives: List[str] = []
                mask: List[List[float]] = []
                for negs in negative_texts:
                    row = list(negs) if negs else [""]
                    valid = len(row)
                    while len(row) < n_neg:
                        row.append(row[-1])
                    flat_negatives.extend(row)
                    mask.append([1.0] * valid + [0.0] * (n_neg - valid))

                enc_m = self.module.tokenizer(
                    mention_texts, padding=True, truncation=True,
                    max_length=cfg.max_len_mention, return_tensors="pt",
                ).to(self.device)
                enc_p = self.module.tokenizer(
                    positive_texts, padding=True, truncation=True,
                    max_length=cfg.max_len_edge, return_tensors="pt",
                ).to(self.device)
                enc_n = self.module.tokenizer(
                    flat_negatives, padding=True, truncation=True,
                    max_length=cfg.max_len_edge, return_tensors="pt",
                ).to(self.device)

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=cfg.fp16 and self.device == "cuda"):
                    v_m = self.module.encode_mention(**enc_m)            # [B, d]
                    v_p = self.module.encode_edge(**enc_p)               # [B, d]
                    v_n = self.module.encode_edge(**enc_n)               # [B * n_neg, d]
                    v_n = v_n.view(len(batch), n_neg, -1)

                    s_pos = self.module.score(v_m, v_p).unsqueeze(1)     # [B, 1]
                    s_neg = (v_m.unsqueeze(1) * v_n).sum(dim=-1)         # [B, n_neg]

                    # Eq. (3): L = max(0, alpha + s(m, e-) - s(m, e+))
                    losses = torch.clamp(cfg.margin + s_neg - s_pos, min=0.0)
                    mask_t = torch.tensor(mask, device=self.device, dtype=losses.dtype)
                    loss = (losses * mask_t).sum() / mask_t.sum().clamp(min=1.0)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                running_loss += float(loss.detach())

            # Re-index the edge catalogue so that validation reflects the updated encoders.
            if evaluate_fn is not None and valid_mentions:
                self.index_edges(ontology)
                metric = evaluate_fn(self)
                improved = metric > best_metric
                print(
                    f"[bi-encoder] epoch {epoch + 1}/{cfg.epochs} "
                    f"loss={running_loss / max(1, steps_per_epoch):.4f} "
                    f"{cfg.early_stopping_metric}={metric:.4f}"
                    + ("  *" if improved else "")
                )
                if improved:
                    best_metric, best_epoch = metric, epoch
                    epochs_without_improvement = 0
                    self.save(os.path.join(output_dir, "best"))
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= cfg.early_stopping_patience:
                        print(f"[bi-encoder] early stop at epoch {epoch + 1}")
                        break
            else:
                print(
                    f"[bi-encoder] epoch {epoch + 1}/{cfg.epochs} "
                    f"loss={running_loss / max(1, steps_per_epoch):.4f}"
                )

        if best_epoch >= 0:
            self.load(os.path.join(output_dir, "best"))
            print(f"[bi-encoder] restored best checkpoint from epoch {best_epoch + 1}")
        self.index_edges(ontology)
        return self

    # -- persistence ----------------------------------------------------- #

    def save(self, path: str) -> None:
        torch, _ = _require_torch()
        os.makedirs(path, exist_ok=True)
        torch.save(self.module.state_dict(), os.path.join(path, "pytorch_model.bin"))
        self.module.tokenizer.save_pretrained(path)
        with open(os.path.join(path, "bi_encoder_config.json"), "w", encoding="utf-8") as fh:
            json.dump(self.config.__dict__, fh, indent=2, default=str)

    def load(self, path: str) -> "EdgeBiEncoder":
        torch, _ = _require_torch()
        state = torch.load(os.path.join(path, "pytorch_model.bin"), map_location=self.device)
        self.module.load_state_dict(state)
        return self
