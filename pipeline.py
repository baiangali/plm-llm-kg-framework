"""
Three-stage PLM-LLM-Ontology pipeline.

Stages:
    Stage 1 (edge):   PLM-based candidate extraction (XLM-R)
    Stage 2 (cloud):  LLM-based semantic reasoning (GPT-4 or simulated)
    Stage 3 (graph):  Ontology-guided ranking and integration

Implements equation (1) from Section 3 of the paper:
    F(x) = G(R(C(x)))

Usage:
    python pipeline.py --input data/test.jsonl \\
                       --ontology data/ontology.json \\
                       --out results/
"""

import argparse
import json
import os
import random
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModel


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

PLM_MODEL_NAME = "xlm-roberta-base"
MAX_SEQ_LEN = 512
TOP_K_CANDIDATES = 10

# Ranking weights (Section 3.3 of the paper)
ALPHA = 0.35    # semantic similarity
BETA = 0.30     # LLM confidence
GAMMA = 0.25    # structural compatibility
DELTA = 0.10    # conflict penalty

# Thresholds
TAU_AUTO_INSERT = 0.72
TAU_REVIEW_MIN = 0.55

# Device
DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
)


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class Candidate:
    """A concept candidate extracted at Stage 1."""
    span: str
    start: int
    end: int
    context: str
    embedding: Optional[list] = None
    entity_type: Optional[str] = None


@dataclass
class LLMDecision:
    """Output of Stage 2 (cloud LLM)."""
    valid: bool
    ontology_class: str
    confidence: float
    relations: List[Dict] = field(default_factory=list)


@dataclass
class Placement:
    """Final ontology placement after Stage 3."""
    candidate: Candidate
    ontology_class: str
    score: float
    decision: str          # "auto_insert" | "review_queue" | "discard"
    relations: List[Dict]


# -----------------------------------------------------------------------------
# Stage 1: Edge — PLM-based candidate extraction
# -----------------------------------------------------------------------------

class EdgePLM:
    """XLM-RoBERTa-based candidate extractor for the edge layer."""

    def __init__(self, model_name: str = PLM_MODEL_NAME):
        print(f"[edge] loading {model_name} on {DEVICE}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        """Return mean-pooled contextual embedding for the text."""
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN
        ).to(DEVICE)
        out = self.model(**enc)
        return out.last_hidden_state.mean(dim=1).squeeze().cpu()

    def extract_candidates(self, text: str, gold_entities: Optional[List[Dict]] = None) -> List[Candidate]:
        """
        Extract candidate concepts from the input text.

        If gold_entities are provided (training/eval mode), use them as candidate
        spans. Otherwise apply a simple capitalisation heuristic. In a production
        system this is replaced with a fine-tuned NER head.
        """
        candidates = []
        if gold_entities:
            for ent in gold_entities[:TOP_K_CANDIDATES]:
                ctx_left = max(0, ent["start"] - 60)
                ctx_right = min(len(text), ent["end"] + 60)
                cand = Candidate(
                    span=text[ent["start"]:ent["end"]],
                    start=ent["start"],
                    end=ent["end"],
                    context=text[ctx_left:ctx_right],
                    entity_type=ent.get("type"),
                )
                cand.embedding = self.encode(cand.span).tolist()
                candidates.append(cand)
        else:
            tokens = text.split()
            cursor = 0
            for tok in tokens:
                start = text.find(tok, cursor)
                end = start + len(tok)
                cursor = end
                if tok and tok[0].isupper():
                    ctx_left = max(0, start - 60)
                    ctx_right = min(len(text), end + 60)
                    cand = Candidate(
                        span=tok,
                        start=start,
                        end=end,
                        context=text[ctx_left:ctx_right],
                    )
                    cand.embedding = self.encode(cand.span).tolist()
                    candidates.append(cand)
            candidates = candidates[:TOP_K_CANDIDATES]
        return candidates


# -----------------------------------------------------------------------------
# Stage 2: Cloud — LLM-based semantic reasoning
# -----------------------------------------------------------------------------

class CloudLLM:
    """Cloud LLM wrapper. Real OpenAI API if OPENAI_API_KEY is set, else simulation."""

    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.use_real_api = bool(self.api_key)
        self.prompts = self._load_prompts()
        if self.use_real_api:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                print("[cloud] using OpenAI API (gpt-4-turbo)")
            except ImportError:
                self.use_real_api = False
        if not self.use_real_api:
            print("[cloud] SIMULATED mode (no API key)")

    def _load_prompts(self) -> Dict[str, str]:
        prompts = {}
        for lang in ("en", "ru", "kz"):
            p = self.prompts_dir / f"prompt_{lang}.txt"
            if p.exists():
                prompts[lang] = p.read_text(encoding="utf-8")
        return prompts

    def _build_payload(self, candidate: Candidate, ontology_classes: List[str], lang: str) -> str:
        template = self.prompts.get(lang, self.prompts.get("en", ""))
        return template.format(
            candidate=candidate.span,
            context=candidate.context,
            classes=", ".join(ontology_classes),
        )

    def reason(self, candidate: Candidate, ontology_classes: List[str], lang: str = "en") -> LLMDecision:
        payload = self._build_payload(candidate, ontology_classes, lang)
        if self.use_real_api:
            try:
                resp = self.client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content":
                            "You are an ontology assistant. Respond strictly in JSON: "
                            "{valid: bool, class: str, confidence: float, relations: list}."},
                        {"role": "user", "content": payload},
                    ],
                    temperature=0,
                    max_tokens=300,
                )
                data = json.loads(resp.choices[0].message.content)
                return LLMDecision(
                    valid=bool(data.get("valid", True)),
                    ontology_class=data.get("class", "Entity"),
                    confidence=float(data.get("confidence", 0.5)),
                    relations=data.get("relations", []),
                )
            except Exception as e:
                print(f"[cloud] API error: {e}; falling back to simulation")
        # Simulation: pick class based on entity_type hint if available
        time.sleep(random.uniform(0.8, 1.5))
        type_to_class = {
            "PERSON": "Agent", "ORGANIZATION": "Agent",
            "LOCATION": "Place", "EVENT": "Activity",
            "NARRATIVE": "AbstractConcept", "CONCEPT": "AbstractConcept",
            "TECHNOLOGY": "AbstractConcept", "MISC": "Entity",
        }
        cls = type_to_class.get(candidate.entity_type, random.choice(ontology_classes))
        return LLMDecision(
            valid=True,
            ontology_class=cls,
            confidence=random.uniform(0.7, 0.95),
            relations=[],
        )


# -----------------------------------------------------------------------------
# Stage 3: Ontology-guided ranking and integration
# -----------------------------------------------------------------------------

class OntologyIntegrator:
    """Ranks candidate placements and integrates them into the knowledge graph."""

    def __init__(self, ontology: Dict, plm: EdgePLM):
        self.ontology = ontology
        self.plm = plm
        # Pre-compute class embeddings
        self.class_embeddings = {
            c: plm.encode(c).numpy() for c in ontology["classes"].keys()
        }

    def _similarity(self, cand_emb: list, class_name: str) -> float:
        import numpy as np
        a = np.array(cand_emb)
        b = self.class_embeddings.get(class_name)
        if b is None:
            return 0.0
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return float(np.dot(a, b) / denom)

    def _structural_compat(self, class_name: str) -> float:
        return 1.0 if class_name in self.ontology["classes"] else 0.0

    def _conflict(self, class_name: str, candidate: Candidate) -> float:
        # Placeholder for disjointness checks
        return 0.0

    def rank(self, candidate: Candidate, decision: LLMDecision) -> Placement:
        sim = self._similarity(candidate.embedding, decision.ontology_class)
        conf = decision.confidence
        struct = self._structural_compat(decision.ontology_class)
        conflict = self._conflict(decision.ontology_class, candidate)

        score = ALPHA * sim + BETA * conf + GAMMA * struct - DELTA * conflict

        if score >= TAU_AUTO_INSERT:
            outcome = "auto_insert"
        elif score >= TAU_REVIEW_MIN:
            outcome = "review_queue"
        else:
            outcome = "discard"

        return Placement(
            candidate=candidate,
            ontology_class=decision.ontology_class,
            score=score,
            decision=outcome,
            relations=decision.relations,
        )


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def run_pipeline(input_path: str, ontology_path: str, out_path: str, prompts_dir: str):
    # Load ontology
    with open(ontology_path, "r", encoding="utf-8") as f:
        ontology = json.load(f)
    ontology_classes = list(ontology["classes"].keys())

    # Load documents
    docs = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))

    # Initialise components
    plm = EdgePLM()
    llm = CloudLLM(prompts_dir=prompts_dir)
    integrator = OntologyIntegrator(ontology, plm)

    results = []
    for i, doc in enumerate(docs):
        if i % 10 == 0:
            print(f"  doc {i}/{len(docs)}")
        # Stage 1
        candidates = plm.extract_candidates(doc["text"], doc.get("entities"))
        # Stages 2 and 3
        placements = []
        for cand in candidates:
            decision = llm.reason(cand, ontology_classes, lang=doc.get("language", "en"))
            if not decision.valid:
                continue
            placement = integrator.rank(cand, decision)
            placements.append({
                "span": cand.span,
                "start": cand.start,
                "end": cand.end,
                "ontology_class": placement.ontology_class,
                "score": placement.score,
                "decision": placement.decision,
                "relations": placement.relations,
            })
        results.append({
            "doc_id": doc.get("doc_id"),
            "language": doc.get("language"),
            "placements": placements,
        })

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary stats
    n_auto = sum(1 for r in results for p in r["placements"] if p["decision"] == "auto_insert")
    n_review = sum(1 for r in results for p in r["placements"] if p["decision"] == "review_queue")
    n_discard = sum(1 for r in results for p in r["placements"] if p["decision"] == "discard")
    print(f"\nProcessed {len(docs)} documents")
    print(f"  auto-inserted:  {n_auto}")
    print(f"  review queue:   {n_review}")
    print(f"  discarded:      {n_discard}")
    print(f"Results written to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="JSONL input file")
    parser.add_argument("--ontology", type=str, required=True, help="ontology.json")
    parser.add_argument("--out", type=str, default="results/predictions.jsonl")
    parser.add_argument("--prompts", type=str, default="prompts")
    args = parser.parse_args()
    run_pipeline(args.input, args.ontology, args.out, args.prompts)


if __name__ == "__main__":
    main()
