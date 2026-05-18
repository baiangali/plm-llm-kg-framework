"""
System-level evaluation for the PLM-LLM edge-cloud framework.

Measures four metrics across three configurations:
  - cloud_only      : full document is sent to the LLM
  - edge_only       : PLM (XLM-R) runs locally, no LLM call
  - edge_cloud      : PLM filters candidates locally, only candidates go to LLM

Outputs:
  - results/raw_runs.json     : per-document raw measurements
  - results/summary.csv       : aggregated table for the paper
  - results/table7.tex        : LaTeX snippet ready to paste

Usage:
    python benchmark.py --texts data/sample_texts.txt --n 100
"""

import argparse
import gc
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional

import psutil
import torch
from transformers import AutoTokenizer, AutoModel

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

PLM_MODEL_NAME = "xlm-roberta-base"   # change to xlm-roberta-large if you have VRAM
CONTEXT_WINDOW_TOKENS = 20            # candidate context window for edge_cloud
MAX_SEQ_LEN = 256
WARMUP_RUNS = 3                       # discarded before measurement
SIMULATED_LLM_LATENCY_MS = (800, 1500)  # used only when no real API key

# -----------------------------------------------------------------------------
# Device selection
# -----------------------------------------------------------------------------

def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = pick_device()

# -----------------------------------------------------------------------------
# Memory measurement
# -----------------------------------------------------------------------------

def reset_peak_memory():
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()

def get_peak_memory_mb() -> float:
    if DEVICE == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    # CPU / MPS: report RSS (resident set size) of the process
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

# -----------------------------------------------------------------------------
# PLM wrapper (edge component)
# -----------------------------------------------------------------------------

class EdgePLM:
    def __init__(self, model_name: str = PLM_MODEL_NAME):
        print(f"[edge] loading {model_name} on {DEVICE} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def encode(self, text: str):
        """Run PLM forward pass and return token embeddings + offsets."""
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        out = self.model(**enc)
        return out.last_hidden_state[0].cpu(), offsets

    def extract_candidates(self, text: str) -> List[Dict]:
        """
        Simple heuristic candidate extractor: capitalised tokens / multi-token spans.
        Replace with your real NER head if you have one. This is sufficient for
        bandwidth measurement because what matters is the *size* of what is sent.
        """
        _, offsets = self.encode(text)
        candidates = []
        tokens = text.split()
        cursor = 0
        for tok in tokens:
            start = text.find(tok, cursor)
            end = start + len(tok)
            cursor = end
            if tok and (tok[0].isupper() or tok.isdigit()):
                left = max(0, start - 60)
                right = min(len(text), end + 60)
                candidates.append({
                    "span": tok,
                    "start": start,
                    "end": end,
                    "context": text[left:right],
                })
        return candidates

# -----------------------------------------------------------------------------
# LLM wrapper (cloud component): real OpenAI or simulated
# -----------------------------------------------------------------------------

class CloudLLM:
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.use_real_api = bool(self.api_key)
        if self.use_real_api:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                print("[cloud] using real OpenAI API")
            except ImportError:
                print("[cloud] openai package not installed; falling back to simulation")
                self.use_real_api = False
        if not self.use_real_api:
            print("[cloud] SIMULATED MODE (no API key) — latency drawn from "
                  f"{SIMULATED_LLM_LATENCY_MS} ms range")

    def call(self, payload: str) -> Dict:
        """Send payload to LLM; return dict with latency_ms and response."""
        start = time.perf_counter()
        if self.use_real_api:
            try:
                resp = self.client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content":
                            "You are an ontology assistant. Respond strictly in JSON: "
                            "{valid, class, confidence, relations}."},
                        {"role": "user", "content": payload},
                    ],
                    temperature=0,
                    max_tokens=200,
                )
                text = resp.choices[0].message.content
            except Exception as e:
                text = f"[error] {e}"
        else:
            # Simulated latency: uniform draw inside the configured range
            import random
            sim_ms = random.uniform(*SIMULATED_LLM_LATENCY_MS)
            time.sleep(sim_ms / 1000.0)
            text = '{"valid": true, "class": "Concept", "confidence": 0.8, "relations": []}'
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {"latency_ms": latency_ms, "response": text, "bytes": len(payload.encode("utf-8"))}

# -----------------------------------------------------------------------------
# Three configurations
# -----------------------------------------------------------------------------

@dataclass
class RunResult:
    config: str
    doc_id: int
    total_latency_ms: float
    edge_latency_ms: float
    cloud_latency_ms: float
    bytes_sent: int
    peak_mem_mb: float
    n_candidates: int

def run_cloud_only(text: str, llm: CloudLLM, doc_id: int) -> RunResult:
    """Send the entire document to the LLM."""
    reset_peak_memory()
    t0 = time.perf_counter()
    res = llm.call(text)
    total = (time.perf_counter() - t0) * 1000.0
    return RunResult(
        config="cloud_only",
        doc_id=doc_id,
        total_latency_ms=total,
        edge_latency_ms=0.0,
        cloud_latency_ms=res["latency_ms"],
        bytes_sent=res["bytes"],
        peak_mem_mb=get_peak_memory_mb(),
        n_candidates=0,
    )

def run_edge_only(text: str, plm: EdgePLM, doc_id: int) -> RunResult:
    """Only PLM runs locally; no LLM call. No semantic reasoning."""
    reset_peak_memory()
    t0 = time.perf_counter()
    cands = plm.extract_candidates(text)
    total = (time.perf_counter() - t0) * 1000.0
    return RunResult(
        config="edge_only",
        doc_id=doc_id,
        total_latency_ms=total,
        edge_latency_ms=total,
        cloud_latency_ms=0.0,
        bytes_sent=0,
        peak_mem_mb=get_peak_memory_mb(),
        n_candidates=len(cands),
    )

def run_edge_cloud(text: str, plm: EdgePLM, llm: CloudLLM, doc_id: int) -> RunResult:
    """PLM filters; only candidates+context go to LLM."""
    reset_peak_memory()
    t0 = time.perf_counter()
    cands = plm.extract_candidates(text)
    t_edge = (time.perf_counter() - t0) * 1000.0

    # Build a compact payload: candidate + local context only
    payload_parts = []
    for c in cands[:10]:  # top-K candidates
        payload_parts.append(f"CANDIDATE: {c['span']}\nCONTEXT: {c['context']}")
    payload = "\n---\n".join(payload_parts) if payload_parts else "NO_CANDIDATES"

    res = llm.call(payload)
    total = (time.perf_counter() - t0) * 1000.0

    return RunResult(
        config="edge_cloud",
        doc_id=doc_id,
        total_latency_ms=total,
        edge_latency_ms=t_edge,
        cloud_latency_ms=res["latency_ms"],
        bytes_sent=res["bytes"],
        peak_mem_mb=get_peak_memory_mb(),
        n_candidates=len(cands),
    )

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def load_texts(path: Optional[str], n: int) -> List[str]:
    if path and Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        return texts[:n]
    # Fallback: synthetic multilingual texts so the script runs out of the box
    print(f"[data] {path} not found, using built-in synthetic sample")
    samples = [
        "The Ministry of Digital Development launched a new program in Astana to support startups in the technology sector.",
        "Министерство цифрового развития запустило новую программу в Астане для поддержки технологических стартапов.",
        "Цифрлық даму министрлігі Астанада технологиялық стартаптарды қолдау үшін жаңа бағдарлама іске қосты.",
        "Operation Sandstorm was reportedly coordinated through several Telegram channels and amplified via X platform.",
        "Президент подписал указ о создании национального центра искусственного интеллекта.",
    ]
    # Replicate to reach n
    out = []
    while len(out) < n:
        out.extend(samples)
    return out[:n]

def summarise(runs: List[RunResult]) -> Dict[str, Dict[str, float]]:
    by_config: Dict[str, List[RunResult]] = {}
    for r in runs:
        by_config.setdefault(r.config, []).append(r)
    summary = {}
    for cfg, items in by_config.items():
        lat = [r.total_latency_ms for r in items]
        edge = [r.edge_latency_ms for r in items]
        cloud = [r.cloud_latency_ms for r in items]
        bw = [r.bytes_sent for r in items]
        mem = [r.peak_mem_mb for r in items]
        total_time_s = sum(lat) / 1000.0
        summary[cfg] = {
            "n": len(items),
            "latency_ms_mean": statistics.mean(lat),
            "latency_ms_median": statistics.median(lat),
            "latency_ms_p95": sorted(lat)[int(0.95 * len(lat)) - 1] if len(lat) > 1 else lat[0],
            "edge_ms_mean": statistics.mean(edge),
            "cloud_ms_mean": statistics.mean(cloud),
            "throughput_docs_per_s": len(items) / total_time_s if total_time_s > 0 else 0.0,
            "bandwidth_kb_per_doc": statistics.mean(bw) / 1024.0,
            "peak_memory_mb": max(mem),
        }
    return summary

def write_latex_table(summary: Dict[str, Dict[str, float]], path: Path):
    order = ["cloud_only", "edge_only", "edge_cloud"]
    labels = {
        "cloud_only": "Cloud-only (raw text $\\to$ LLM)",
        "edge_only": "Edge-only (PLM, no reasoning)",
        "edge_cloud": "Proposed edge--cloud",
    }
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{System-level evaluation: latency, throughput, bandwidth and edge memory.}",
        "\\label{tab:system}",
        "\\begin{tabular}{lcccc}",
        "\\hline",
        "Configuration & Latency (ms) & Throughput (docs/s) & Bandwidth (KB/doc) & Edge memory (MB) \\\\",
        "\\hline",
    ]
    for cfg in order:
        if cfg not in summary:
            continue
        s = summary[cfg]
        lines.append(
            f"{labels[cfg]} & {s['latency_ms_mean']:.0f} & "
            f"{s['throughput_docs_per_s']:.2f} & "
            f"{s['bandwidth_kb_per_doc']:.2f} & "
            f"{s['peak_memory_mb']:.0f} \\\\"
        )
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(lines), encoding="utf-8")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--texts", type=str, default="data/sample_texts.txt",
                        help="file with one document per line")
    parser.add_argument("--n", type=int, default=100, help="number of documents")
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = load_texts(args.texts, args.n)
    print(f"[data] {len(texts)} documents loaded")

    plm = EdgePLM()
    llm = CloudLLM()

    # Warmup (discarded)
    print("[warmup] running warmup passes ...")
    for t in texts[:WARMUP_RUNS]:
        plm.extract_candidates(t)

    runs: List[RunResult] = []
    for i, text in enumerate(texts):
        if i % 10 == 0:
            print(f"  doc {i}/{len(texts)}")
        runs.append(run_cloud_only(text, llm, i))
        runs.append(run_edge_only(text, plm, i))
        runs.append(run_edge_cloud(text, plm, llm, i))

    # Save raw
    with open(out_dir / "raw_runs.json", "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in runs], f, indent=2, ensure_ascii=False)

    summary = summarise(runs)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # CSV
    import csv
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "n", "latency_ms_mean", "latency_ms_median",
                    "latency_ms_p95", "throughput_docs_per_s",
                    "bandwidth_kb_per_doc", "peak_memory_mb"])
        for cfg, s in summary.items():
            w.writerow([cfg, s["n"], f"{s['latency_ms_mean']:.1f}",
                        f"{s['latency_ms_median']:.1f}",
                        f"{s['latency_ms_p95']:.1f}",
                        f"{s['throughput_docs_per_s']:.2f}",
                        f"{s['bandwidth_kb_per_doc']:.2f}",
                        f"{s['peak_memory_mb']:.1f}"])

    write_latex_table(summary, out_dir / "table7.tex")

    print("\n=== Summary ===")
    for cfg, s in summary.items():
        print(f"{cfg:14s} | latency {s['latency_ms_mean']:7.1f} ms | "
              f"throughput {s['throughput_docs_per_s']:5.2f} docs/s | "
              f"bandwidth {s['bandwidth_kb_per_doc']:6.2f} KB | "
              f"mem {s['peak_memory_mb']:6.1f} MB")

    # Derived improvements
    if "cloud_only" in summary and "edge_cloud" in summary:
        co = summary["cloud_only"]
        ec = summary["edge_cloud"]
        bw_red = (1 - ec["bandwidth_kb_per_doc"] / max(co["bandwidth_kb_per_doc"], 1e-9)) * 100
        lat_red = (1 - ec["latency_ms_mean"] / max(co["latency_ms_mean"], 1e-9)) * 100
        print(f"\nBandwidth reduction (edge_cloud vs cloud_only): {bw_red:.1f}%")
        print(f"Latency reduction   (edge_cloud vs cloud_only): {lat_red:.1f}%")

    print(f"\nResults written to {out_dir.resolve()}")

if __name__ == "__main__":
    main()
