"""
Stage 3 -- Edge selection with the multi-label Edge-Cross-encoder (Section 3.5).

The mention and the candidate edge are encoded jointly, which permits explicit
modelling of interactions between the subtokens of the mention and those of the
edge:

    Eq. (11)  [CLS] ctx_l [Ms] mention [Me] ctx_r [SEP]
              parent [P-TAG] child [C-TAG] [SEP]
    Eq. (12)  s_cross(m, e) = v_cross . w
    Eq. (13)  L = - sum_(m,e) [ y log sigma(s) + (1 - y) log(1 - sigma(s)) ]

Because a mention may have several valid edges, selection is framed as
multi-label classification rather than as a softmax over candidates: all k
candidates of a mention are scored by a shared encoder, so the scores are
comparable across candidates and the top-k ordering is well defined.

Reranking only reorders the pool it is given. When k equals the candidate-pool
size, InR@k is therefore invariant under reranking -- the property used as a
consistency check in Section 4.4.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .config import CrossEncoderConfig
from .data import Edge, Mention, Ontology
from .serialization import add_special_tokens, serialize_cross


def _require_torch():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The cross-encoder requires torch and transformers. "
            "Install them with `pip install -r requirements.txt`."
        ) from exc
    import torch
    import transformers

    return torch, transformers


@dataclass
class CrossExample:
    mention: Mention
    edge: Edge
    label: int


def build_examples(
    mentions: Sequence[Mention],
    candidates: Mapping[str, Sequence[Edge]],
) -> List[CrossExample]:
    """Label each candidate edge as a valid or invalid placement for its mention.

    ``y(m, e) = 1`` exactly when ``e`` is in the gold set ``Y(m)``. Candidates
    are the top-k output of Stage 2, so the classifier is trained on the same
    distribution it is applied to at inference.
    """
    examples: List[CrossExample] = []
    for m in mentions:
        for edge in candidates.get(m.mention_id, ()):
            examples.append(CrossExample(m, edge, int(edge in m.gold_edges)))
    return examples


def _build_module(config: CrossEncoderConfig, device: str):
    torch, transformers = _require_torch()

    class _CrossEncoderModule(torch.nn.Module):
        """Encoder plus the linear projection ``w`` of Eq. (12)."""

        def __init__(self) -> None:
            super().__init__()
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(config.base_model)
            self.encoder = transformers.AutoModel.from_pretrained(config.base_model)
            add_special_tokens(self.tokenizer, self.encoder)
            hidden = self.encoder.config.hidden_size
            self.dropout = torch.nn.Dropout(0.1)
            #: ``w`` in Eq. (12); a single output unit, not a two-way softmax,
            #: because the task is multi-label rather than multi-class.
            self.classifier = torch.nn.Linear(hidden, 1)

        def forward(self, **inputs):
            hidden = self.encoder(**inputs).last_hidden_state
            v_cross = hidden[:, 0, :]  # final-layer [CLS] representation
            return self.classifier(self.dropout(v_cross)).squeeze(-1)

    return _CrossEncoderModule().to(device)


class EdgeCrossEncoder:
    """Fine-tuned multi-label reranker over the enriched candidate set."""

    def __init__(self, config: Optional[CrossEncoderConfig] = None, device: str = "cuda") -> None:
        torch, _ = _require_torch()
        self.config = config or CrossEncoderConfig()
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.module = _build_module(self.config, self.device)

    # -- tensorisation ---------------------------------------------------- #

    def _encode_batch(self, ontology: Ontology, batch: Sequence[CrossExample]):
        segments_a: List[str] = []
        segments_b: List[str] = []
        for ex in batch:
            parent_text, child_text = ontology.edge_text(ex.edge)
            a, b = serialize_cross(
                ex.mention.context_left,
                ex.mention.mention,
                ex.mention.context_right,
                parent_text,
                child_text,
            )
            segments_a.append(a)
            segments_b.append(b)
        return self.module.tokenizer(
            segments_a,
            segments_b,
            padding=True,
            truncation="longest_first",
            max_length=self.config.max_seq_length,
            return_tensors="pt",
        ).to(self.device)

    # -- training --------------------------------------------------------- #

    def train(
        self,
        ontology: Ontology,
        train_examples: Sequence[CrossExample],
        output_dir: str = "runs/cross_encoder",
        evaluate_fn=None,
    ) -> "EdgeCrossEncoder":
        """Optimise the binary cross-entropy of Eq. (13).

        ``evaluate_fn(model) -> float`` should return InR_any@5 on the validation
        split (Table 2); training stops after two epochs without improvement.
        """
        torch, transformers = _require_torch()
        cfg = self.config
        os.makedirs(output_dir, exist_ok=True)
        torch.manual_seed(cfg.seed)
        rng = random.Random(cfg.seed)

        optimizer = torch.optim.AdamW(
            self.module.parameters(),
            lr=cfg.learning_rate,
            betas=cfg.adam_betas,
            weight_decay=cfg.weight_decay,
        )
        steps_per_epoch = math.ceil(len(train_examples) / cfg.batch_size)
        total_steps = steps_per_epoch * cfg.epochs
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(cfg.warmup_ratio * total_steps),
            num_training_steps=total_steps,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16 and self.device == "cuda")
        loss_fn = torch.nn.BCEWithLogitsLoss()

        best_metric = float("-inf")
        best_epoch = -1
        stale = 0

        for epoch in range(cfg.epochs):
            self.module.train()
            order = list(range(len(train_examples)))
            rng.shuffle(order)
            running = 0.0

            for step in range(steps_per_epoch):
                idx = order[step * cfg.batch_size : (step + 1) * cfg.batch_size]
                if not idx:
                    continue
                batch = [train_examples[i] for i in idx]
                enc = self._encode_batch(ontology, batch)
                labels = torch.tensor(
                    [float(ex.label) for ex in batch], device=self.device
                )

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=cfg.fp16 and self.device == "cuda"):
                    logits = self.module(**enc)          # s_cross(m, e), Eq. (12)
                    loss = loss_fn(logits.float(), labels)  # Eq. (13)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                running += float(loss.detach())

            if evaluate_fn is not None:
                metric = evaluate_fn(self)
                improved = metric > best_metric
                print(
                    f"[cross-encoder] epoch {epoch + 1}/{cfg.epochs} "
                    f"loss={running / max(1, steps_per_epoch):.4f} "
                    f"{cfg.early_stopping_metric}={metric:.4f}" + ("  *" if improved else "")
                )
                if improved:
                    best_metric, best_epoch, stale = metric, epoch, 0
                    self.save(os.path.join(output_dir, "best"))
                else:
                    stale += 1
                    if stale >= cfg.early_stopping_patience:
                        print(f"[cross-encoder] early stop at epoch {epoch + 1}")
                        break
            else:
                print(
                    f"[cross-encoder] epoch {epoch + 1}/{cfg.epochs} "
                    f"loss={running / max(1, steps_per_epoch):.4f}"
                )

        if best_epoch >= 0:
            self.load(os.path.join(output_dir, "best"))
            print(f"[cross-encoder] restored best checkpoint from epoch {best_epoch + 1}")
        return self

    # -- inference -------------------------------------------------------- #

    def rerank(
        self,
        ontology: Ontology,
        mentions: Sequence[Mention],
        candidates: Mapping[str, Sequence[Edge]],
        batch_size: int = 64,
    ) -> Dict[str, List[Edge]]:
        """Reorder each mention's candidate list by ``s_cross``, Eq. (12).

        The membership of the list is unchanged: this stage selects among the
        candidates Stage 2 produced and cannot recover an edge that enrichment
        failed to generate.
        """
        torch, _ = _require_torch()
        self.module.eval()

        flat: List[CrossExample] = [
            CrossExample(m, e, 0) for m in mentions for e in candidates.get(m.mention_id, ())
        ]
        scores: List[float] = []
        with torch.no_grad():
            for start in range(0, len(flat), batch_size):
                batch = flat[start : start + batch_size]
                enc = self._encode_batch(ontology, batch)
                logits = self.module(**enc)
                scores.extend(logits.detach().cpu().float().tolist())

        reranked: Dict[str, List[Edge]] = {}
        cursor = 0
        for m in mentions:
            edges = list(candidates.get(m.mention_id, ()))
            edge_scores = scores[cursor : cursor + len(edges)]
            cursor += len(edges)
            order = sorted(
                range(len(edges)),
                key=lambda i: (-edge_scores[i], edges[i].parent, edges[i].child),
            )
            reranked[m.mention_id] = [edges[i] for i in order]
        return reranked

    def score_edges(
        self,
        ontology: Ontology,
        mention: Mention,
        edges: Sequence[Edge],
        batch_size: int = 64,
    ) -> List[Tuple[Edge, float]]:
        """Per-candidate probabilities ``sigma(s_cross)``, for curator-facing output."""
        torch, _ = _require_torch()
        self.module.eval()
        out: List[Tuple[Edge, float]] = []
        with torch.no_grad():
            for start in range(0, len(edges), batch_size):
                chunk = list(edges[start : start + batch_size])
                enc = self._encode_batch(
                    ontology, [CrossExample(mention, e, 0) for e in chunk]
                )
                probs = torch.sigmoid(self.module(**enc)).detach().cpu().float().tolist()
                out.extend(zip(chunk, probs))
        return sorted(out, key=lambda kv: -kv[1])

    # -- persistence ------------------------------------------------------ #

    def save(self, path: str) -> None:
        torch, _ = _require_torch()
        os.makedirs(path, exist_ok=True)
        torch.save(self.module.state_dict(), os.path.join(path, "pytorch_model.bin"))
        self.module.tokenizer.save_pretrained(path)
        with open(os.path.join(path, "cross_encoder_config.json"), "w", encoding="utf-8") as fh:
            json.dump(self.config.__dict__, fh, indent=2, default=str)

    def load(self, path: str) -> "EdgeCrossEncoder":
        torch, _ = _require_torch()
        state = torch.load(os.path.join(path, "pytorch_model.bin"), map_location=self.device)
        self.module.load_state_dict(state)
        return self
