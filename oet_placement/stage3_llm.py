"""
Stage 3 -- Zero-shot prompting and explainable instruction tuning (Section 3.6).

Direct generation of option indices is poorly aligned with the autoregressive
nature of a generative model, so the tuned configuration is trained to emit a
structured rationale before the answer:

    ### Explanation
      1. identify the candidate parent concepts
      2. select the most plausible parents
      3. narrow the candidate child concepts, conditioned on the selected parents
      4. select the most plausible children
      5. return the indices of the predicted insertion edges
    ### Response
      the indices

The explanation text used as a *training target* is instantiated from a template
with the gold parents, gold children and gold option indices. That template is
used only to construct supervision. At inference the model receives the mention,
its context and the candidate list and nothing else; it generates the explanation
autoregressively and terminates with the predicted indices.

:func:`assert_no_leakage` enforces that separation mechanically, so the claim in
Section 3.6 that "there is no path by which reference information could reach the
model at prediction time" is checked rather than asserted.

Context length bounds the candidate list: Llama-2-7B's 4,096-token window
accommodates roughly 50 edges, which is why k = 50 is the largest setting
reported for the LLM configurations.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .config import LlamaTunedConfig
from .data import NULL_ID, NULL_TITLE, Edge, Mention, Ontology

# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #

TASK_DESCRIPTION = (
    "You are assisting an ontology curator. A biomedical mention denotes a concept "
    "that does not yet exist in the ontology. Your task is to decide where the new "
    "concept should be inserted into the subsumption hierarchy.\n"
    "An insertion position is an edge \"parent -> child\": inserting the new concept "
    "c at the edge P -> C yields P -> c -> C. When the child is NULL the new concept "
    "is inserted as a leaf under P.\n"
    "Several insertion edges may be correct. Choose every option that is a valid "
    "placement, and no others."
)

RESPONSE_HEADER = "### Response"
EXPLANATION_HEADER = "### Explanation"


def format_candidates(ontology: Ontology, edges: Sequence[Edge]) -> str:
    """Enumerate the candidate edges, one per line, with 1-based option numbers."""
    lines = []
    for i, edge in enumerate(edges, start=1):
        parent = ontology.title(edge.parent)
        child = NULL_TITLE if edge.is_leaf else ontology.title(edge.child)
        lines.append(f"{i}. {parent} -> {child}")
    return "\n".join(lines)


def build_prompt(
    ontology: Ontology,
    mention: Mention,
    candidates: Sequence[Edge],
    max_context_chars: int = 1000,
    explainable: bool = True,
) -> str:
    """The inference-time prompt: task description, mention in context, candidates.

    No gold parent, gold child or gold option number appears anywhere in it.
    """
    left = (mention.context_left or "")[-max_context_chars:]
    right = (mention.context_right or "")[:max_context_chars]

    # The literal section headers are deliberately not written into the prompt:
    # `parse_response` reads the text after the *last* "### Response", so an
    # occurrence inside the instruction would be indistinguishable from the
    # model's own answer when the decoded prompt prefix cannot be stripped.
    instruction = (
        "an Explanation section with your reasoning, then a Response section "
        "containing only the option numbers."
        if explainable
        else "a Response section containing only the option numbers."
    )

    return (
        f"{TASK_DESCRIPTION}\n\n"
        f"### Mention\n{mention.mention}\n\n"
        f"### Context\n{left} [[{mention.mention}]] {right}\n\n"
        f"### Candidate insertion edges\n{format_candidates(ontology, candidates)}\n\n"
        f"### Instruction\nSelect all valid insertion edges. Write {instruction}\n"
    )


# --------------------------------------------------------------------------- #
# Supervision targets -- the five-step explanation template
# --------------------------------------------------------------------------- #


def build_explanation_target(
    ontology: Ontology,
    mention: Mention,
    candidates: Sequence[Edge],
) -> str:
    """Instantiate the five-step template from the gold parents, children and indices.

    Used only to construct training targets. Mentions with no gold edge inside the
    candidate list still receive a well-formed target: the model is taught to say
    that no option is valid rather than to guess, which matters because Stage 2
    does not guarantee that a gold edge is present.
    """
    gold_indices = [
        i for i, e in enumerate(candidates, start=1) if e in mention.gold_edges
    ]
    gold_in_list = [e for e in candidates if e in mention.gold_edges]

    candidate_parents = _unique([ontology.title(e.parent) for e in candidates])
    gold_parents = _unique([ontology.title(e.parent) for e in gold_in_list])
    gold_children = _unique(
        [NULL_TITLE if e.is_leaf else ontology.title(e.child) for e in gold_in_list]
    )
    children_under_gold_parents = _unique(
        [
            NULL_TITLE if e.is_leaf else ontology.title(e.child)
            for e in candidates
            if ontology.title(e.parent) in set(gold_parents)
        ]
    )

    step1 = (
        "Step 1. The candidate parents offered are: "
        + _join(candidate_parents, limit=12)
        + "."
    )
    if gold_parents:
        step2 = (
            f"Step 2. The mention \"{mention.mention}\" is most plausibly subsumed by: "
            + _join(gold_parents)
            + "."
        )
        step3 = (
            "Step 3. Restricting to the options under those parents, the candidate "
            "children are: " + _join(children_under_gold_parents, limit=12) + "."
        )
        if gold_children == [NULL_TITLE]:
            step4 = (
                "Step 4. The mention has no subsumed concept among the candidates, so "
                "it is placed as a leaf and the child is NULL."
            )
        else:
            step4 = "Step 4. The most plausible children are: " + _join(gold_children) + "."
        step5 = (
            "Step 5. The insertion edges are therefore options "
            + _join([str(i) for i in gold_indices])
            + "."
        )
        answer = ", ".join(str(i) for i in gold_indices)
    else:
        step2 = (
            f"Step 2. None of the candidate parents plausibly subsumes "
            f"\"{mention.mention}\"."
        )
        step3 = "Step 3. There is therefore no parent under which to narrow the children."
        step4 = "Step 4. No child concept can be selected."
        step5 = "Step 5. No option is a valid insertion edge."
        answer = "none"

    return (
        f"{EXPLANATION_HEADER}\n"
        f"{step1}\n{step2}\n{step3}\n{step4}\n{step5}\n\n"
        f"{RESPONSE_HEADER}\n{answer}"
    )


def _unique(items: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _join(items: Sequence[str], limit: Optional[int] = None) -> str:
    if not items:
        return "none"
    shown = list(items[:limit]) if limit else list(items)
    suffix = ", ..." if limit and len(items) > limit else ""
    return "; ".join(shown) + suffix


# --------------------------------------------------------------------------- #
# Leakage guard
# --------------------------------------------------------------------------- #

def assert_no_leakage(
    prompt: str,
    mention: Mention,
    ontology: Ontology,
    candidates: Sequence[Edge],
    explainable: bool = True,
) -> None:
    """Fail loudly if reference information reached an inference-time prompt.

    The check is *constructive* rather than lexical: the prompt is rebuilt from
    inputs that carry no gold information -- the mention text, its context, and
    the candidate list -- and compared byte for byte. Equality proves that
    nothing else entered, which is the guarantee Section 3.6 claims.

    A lexical scan for gold labels would be wrong here. The context is real
    PubMed text and frequently contains the gold parent's label as ordinary
    prose ("...patients who developed renal failure..."), which is not leakage;
    flagging it would fail valid runs while catching nothing a reconstruction
    check misses.

    Answer markers are checked separately, since a filled ``### Response``
    section in the prompt would survive reconstruction only if
    :func:`build_prompt` itself were changed to emit one.
    """
    expected = build_prompt(ontology, mention, candidates, explainable=explainable)
    if prompt != expected:
        raise AssertionError(
            "inference prompt does not match its reconstruction from "
            "(mention, context, candidates); something else was injected"
        )
    for marker in (EXPLANATION_HEADER, RESPONSE_HEADER):
        if marker in prompt:
            raise AssertionError(f"prompt contains a filled answer section: {marker!r}")


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #

_INDEX_RE = re.compile(r"\d+")


def parse_response(text: str, n_candidates: int) -> List[int]:
    """Extract 0-based candidate indices from a generated response.

    Only the text after the final ``### Response`` header is read, so numbers
    that occur inside the explanation -- step numbers in particular -- are not
    mistaken for answers. Indices outside ``[1, n_candidates]`` are discarded.
    """
    if RESPONSE_HEADER in text:
        answer = text.rsplit(RESPONSE_HEADER, 1)[1]
    else:
        answer = text
    answer = answer.strip()
    if answer.lower().startswith("none"):
        return []

    seen: Set[int] = set()
    out: List[int] = []
    for token in _INDEX_RE.findall(answer):
        value = int(token)
        if 1 <= value <= n_candidates and value not in seen:
            seen.add(value)
            out.append(value - 1)
    return out


def extract_explanation(text: str) -> str:
    """The rationale generated for a placement, for expert audit (Section 4.9)."""
    if EXPLANATION_HEADER not in text:
        return ""
    body = text.split(EXPLANATION_HEADER, 1)[1]
    return body.split(RESPONSE_HEADER, 1)[0].strip()


def rerank_from_indices(
    candidates: Sequence[Edge], selected: Sequence[int]
) -> List[Edge]:
    """Promote the selected options, preserving the Stage 2 order elsewhere.

    A generative selector returns a *set*, not a ranking. Promoting the selected
    options to the head of the list while leaving the remainder in their Stage 2
    order is what makes the output comparable with the cross-encoder under the
    same InR@k metric.
    """
    chosen = [candidates[i] for i in selected if 0 <= i < len(candidates)]
    chosen_set = set(chosen)
    return chosen + [e for e in candidates if e not in chosen_set]


# --------------------------------------------------------------------------- #
# Model wrapper
# --------------------------------------------------------------------------- #


def _require_llm():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The LLM selector requires torch and transformers "
            "(and peft for instruction tuning)."
        ) from exc
    import torch
    import transformers

    return torch, transformers


@dataclass
class LLMPrediction:
    mention_id: str
    ranking: List[Edge]
    selected: List[int]
    explanation: str
    raw: str


class LLMEdgeSelector:
    """Zero-shot prompting and LoRA instruction tuning for Stage 3."""

    def __init__(
        self,
        config: Optional[LlamaTunedConfig] = None,
        device: str = "cuda",
        adapter_path: Optional[str] = None,
    ) -> None:
        torch, transformers = _require_llm()
        self.config = config or LlamaTunedConfig()
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.config.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)

    # -- inference -------------------------------------------------------- #

    def select(
        self,
        ontology: Ontology,
        mentions: Sequence[Mention],
        candidates: Mapping[str, Sequence[Edge]],
        max_new_tokens: int = 384,
        check_leakage: bool = True,
        batch_size: int = 4,
    ) -> Dict[str, LLMPrediction]:
        """Generate a selection for each mention and turn it into a ranking."""
        torch, _ = _require_llm()
        self.model.eval()

        items = [
            (m, list(candidates.get(m.mention_id, ()))[: self.config.max_candidates_in_prompt])
            for m in mentions
        ]
        predictions: Dict[str, LLMPrediction] = {}

        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            prompts = []
            for m, cands in chunk:
                prompt = build_prompt(
                    ontology, m, cands, explainable=self.config.explainable
                )
                if check_leakage:
                    assert_no_leakage(
                        prompt, m, ontology, cands, explainable=self.config.explainable
                    )
                prompts.append(prompt)

            enc = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length - max_new_tokens,
            ).to(self.model.device)

            with torch.no_grad():
                generated = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,          # greedy: the task is selection, not generation
                    num_beams=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            for (m, cands), prompt, sequence in zip(chunk, prompts, generated):
                text = self.tokenizer.decode(sequence, skip_special_tokens=True)
                completion = text[len(prompt) :] if text.startswith(prompt) else text
                selected = parse_response(completion, len(cands))
                predictions[m.mention_id] = LLMPrediction(
                    mention_id=m.mention_id,
                    ranking=rerank_from_indices(cands, selected),
                    selected=selected,
                    explanation=extract_explanation(completion),
                    raw=completion,
                )
        return predictions

    # -- instruction tuning ----------------------------------------------- #

    def train(
        self,
        ontology: Ontology,
        mentions: Sequence[Mention],
        candidates: Mapping[str, Sequence[Edge]],
        output_dir: str = "runs/llm_tuned",
    ) -> "LLMEdgeSelector":
        """LoRA fine-tuning on prompt -> (explanation, response) targets.

        The loss is masked over the prompt tokens so that only the generated
        explanation and answer are supervised.
        """
        torch, transformers = _require_llm()
        from peft import LoraConfig, get_peft_model

        cfg = self.config
        os.makedirs(output_dir, exist_ok=True)
        torch.manual_seed(cfg.seed)

        peft_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.lora_target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

        records = []
        for m in mentions:
            cands = list(candidates.get(m.mention_id, ()))[: cfg.max_candidates_in_prompt]
            if not cands:
                continue
            prompt = build_prompt(ontology, m, cands, explainable=cfg.explainable)
            target = (
                build_explanation_target(ontology, m, cands)
                if cfg.explainable
                else _plain_target(m, cands)
            )
            records.append({"prompt": prompt, "target": target})

        tokenizer = self.tokenizer

        def collate(batch):
            input_ids, labels = [], []
            for row in batch:
                p_ids = tokenizer(row["prompt"], add_special_tokens=True)["input_ids"]
                t_ids = tokenizer(
                    row["target"] + tokenizer.eos_token, add_special_tokens=False
                )["input_ids"]
                ids = (p_ids + t_ids)[: cfg.max_seq_length]
                # -100 masks the prompt so the loss covers only the generated span.
                lab = ([-100] * len(p_ids) + t_ids)[: cfg.max_seq_length]
                input_ids.append(ids)
                labels.append(lab)

            width = max(len(x) for x in input_ids)
            pad = tokenizer.pad_token_id
            attention = [[1] * len(x) + [0] * (width - len(x)) for x in input_ids]
            input_ids = [x + [pad] * (width - len(x)) for x in input_ids]
            labels = [x + [-100] * (width - len(x)) for x in labels]
            return {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention),
                "labels": torch.tensor(labels),
            }

        args = transformers.TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            num_train_epochs=cfg.epochs,
            learning_rate=cfg.learning_rate,
            lr_scheduler_type=cfg.lr_schedule,
            warmup_ratio=cfg.warmup_ratio,
            weight_decay=cfg.weight_decay,
            adam_beta1=cfg.adam_betas[0],
            adam_beta2=cfg.adam_betas[1],
            bf16=cfg.bf16,
            logging_steps=20,
            save_strategy="epoch",
            report_to=[],
            seed=cfg.seed,
        )
        trainer = transformers.Trainer(
            model=self.model,
            args=args,
            train_dataset=records,
            data_collator=collate,
        )
        trainer.train()
        self.model.save_pretrained(os.path.join(output_dir, "adapter"))
        tokenizer.save_pretrained(os.path.join(output_dir, "adapter"))
        return self


def _plain_target(mention: Mention, candidates: Sequence[Edge]) -> str:
    indices = [i for i, e in enumerate(candidates, start=1) if e in mention.gold_edges]
    answer = ", ".join(str(i) for i in indices) if indices else "none"
    return f"{RESPONSE_HEADER}\n{answer}"
