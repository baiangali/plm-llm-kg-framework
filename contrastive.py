"""
Contrastive learning utilities for the PLM-LLM framework.

Implements the contrastive training scheme described in Section 3.3 of the paper:
    - Positive pairs (c, o+): gold concept-class assignments
    - Hard negatives (c, o-): sibling classes in the ontology hierarchy
    - Random negatives: unrelated branches of the ontology
    - Ratio: 1 positive : 3 hard negatives : 1 random negative
    - Loss: InfoNCE with temperature 0.07
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


TEMPERATURE = 0.07
RATIO_POS = 1
RATIO_HARD_NEG = 3
RATIO_RAND_NEG = 1


def build_sibling_map(ontology: Dict) -> Dict[str, List[str]]:
    """For each class, return its siblings (classes sharing the same parent)."""
    parents = {c: info.get("parent") for c, info in ontology["classes"].items()}
    siblings = defaultdict(list)
    by_parent = defaultdict(list)
    for c, p in parents.items():
        if p is not None:
            by_parent[p].append(c)
    for p, children in by_parent.items():
        for c in children:
            siblings[c] = [x for x in children if x != c]
    return dict(siblings)


def build_contrastive_pairs(
    gold_pairs: List[Tuple[str, str]],
    ontology: Dict,
    seed: int = 42,
) -> List[Dict]:
    """
    Construct contrastive training examples.

    Args:
        gold_pairs: list of (concept, ontology_class) gold assignments
        ontology: ontology dict with "classes" key

    Returns:
        list of training examples, each with keys: anchor, positive, hard_negatives, random_negatives
    """
    random.seed(seed)
    siblings = build_sibling_map(ontology)
    all_classes = list(ontology["classes"].keys())
    examples = []
    for concept, gold_class in gold_pairs:
        hard_negs = siblings.get(gold_class, [])
        random.shuffle(hard_negs)
        hard_negs = hard_negs[:RATIO_HARD_NEG]
        # If not enough hard negatives, pad with siblings of siblings
        if len(hard_negs) < RATIO_HARD_NEG:
            pool = [c for c in all_classes if c != gold_class and c not in hard_negs]
            random.shuffle(pool)
            hard_negs += pool[:RATIO_HARD_NEG - len(hard_negs)]
        # Random negatives
        excluded = {gold_class, *hard_negs}
        rand_pool = [c for c in all_classes if c not in excluded]
        random.shuffle(rand_pool)
        rand_negs = rand_pool[:RATIO_RAND_NEG]

        examples.append({
            "anchor": concept,
            "positive": gold_class,
            "hard_negatives": hard_negs,
            "random_negatives": rand_negs,
        })
    return examples


def info_nce_loss(
    anchor_emb: torch.Tensor,
    positive_emb: torch.Tensor,
    negative_embs: torch.Tensor,
    temperature: float = TEMPERATURE,
) -> torch.Tensor:
    """
    InfoNCE loss.

    Args:
        anchor_emb:   (D,) or (B, D)
        positive_emb: same shape as anchor_emb
        negative_embs: (N, D) or (B, N, D)
    """
    if anchor_emb.dim() == 1:
        anchor_emb = anchor_emb.unsqueeze(0)
        positive_emb = positive_emb.unsqueeze(0)
        negative_embs = negative_embs.unsqueeze(0)

    anchor_emb = F.normalize(anchor_emb, dim=-1)
    positive_emb = F.normalize(positive_emb, dim=-1)
    negative_embs = F.normalize(negative_embs, dim=-1)

    pos_score = (anchor_emb * positive_emb).sum(dim=-1, keepdim=True)            # (B, 1)
    neg_scores = torch.bmm(negative_embs, anchor_emb.unsqueeze(-1)).squeeze(-1)  # (B, N)
    logits = torch.cat([pos_score, neg_scores], dim=-1) / temperature
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ontology", type=str, required=True)
    parser.add_argument("--train", type=str, required=True, help="train.jsonl")
    parser.add_argument("--out", type=str, default="contrastive_pairs.jsonl")
    args = parser.parse_args()

    with open(args.ontology, "r", encoding="utf-8") as f:
        ontology = json.load(f)

    type_to_class = ontology.get("entity_type_mapping", {})
    gold_pairs = []
    with open(args.train, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            for ent in doc.get("entities", []):
                cls = type_to_class.get(ent["type"], "Entity")
                gold_pairs.append((ent["text"], cls))

    examples = build_contrastive_pairs(gold_pairs, ontology)
    with open(args.out, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"wrote {len(examples)} contrastive examples to {args.out}")
