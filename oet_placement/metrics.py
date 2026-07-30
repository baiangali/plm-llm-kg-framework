"""
Evaluation metrics -- Section 4.2.

Link Neighbourhood Recall (LNR), reported in the tables as ``InR``, is set-level
recall at k under insertion semantics. Two modes are separated explicitly, which
is the paper's contribution at the level of the protocol rather than the measure:

    Eq. (16)  ANY:  E_gold(M) intersect E_pred^k(M) != empty
    Eq. (17)  ALL:  E_gold(M) subset of E_pred^k(M)
    Eq. (18)  LNR_k^mode = (1/|M|) sum_i I(m_i satisfies mode)

ANY reflects the ability to localise the concept in a semantically correct region
of the hierarchy; ALL reflects the ability to structure it fully. Reporting only
one conflates two different claims about automation.

MRR and MAP are computed from the same ranked lists to permit comparison with the
ranking and link-prediction literature. ROC-AUC is deliberately not implemented:
Section 4.2 states that the per-mention candidate space is large, imbalanced and
variable in size, so the measure is dominated by trivially rejected negatives.

This module has no third-party dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .data import Edge, Mention

# --------------------------------------------------------------------------- #
# Per-mention primitives
# --------------------------------------------------------------------------- #


def inr_any(gold: Iterable[Edge], predicted: Sequence[Edge], k: int) -> bool:
    """Eq. (16): at least one gold edge appears in the top-k ranking."""
    top_k = set(predicted[:k])
    return bool(set(gold) & top_k)


def inr_all(gold: Iterable[Edge], predicted: Sequence[Edge], k: int) -> bool:
    """Eq. (17): the complete gold edge set appears in the top-k ranking."""
    gold_set = set(gold)
    if not gold_set:
        return False
    return gold_set <= set(predicted[:k])


def reciprocal_rank(gold: Iterable[Edge], predicted: Sequence[Edge]) -> float:
    """Reciprocal rank of the highest-ranked gold edge; 0 if none is retrieved."""
    gold_set = set(gold)
    for rank, edge in enumerate(predicted, start=1):
        if edge in gold_set:
            return 1.0 / rank
    return 0.0


def average_precision(gold: Iterable[Edge], predicted: Sequence[Edge]) -> float:
    """Average precision over the gold edge set for one ranked list.

    Gold edges that never appear in the ranking contribute zero precision, so the
    denominator is ``|E_gold|`` rather than the number of gold edges retrieved.
    """
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, edge in enumerate(predicted, start=1):
        if edge in gold_set:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(gold_set)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


@dataclass
class PlacementScores:
    """Aggregated results for one configuration on one split."""

    n_mentions: int = 0
    inr_any: Dict[int, float] = field(default_factory=dict)
    inr_all: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    map: float = 0.0
    #: Breakdown over mentions whose gold set contains a leaf / a non-leaf edge,
    #: reported as the ``lf`` and ``nlf`` columns of Table 4.
    inr_any_leaf: Dict[int, float] = field(default_factory=dict)
    inr_all_leaf: Dict[int, float] = field(default_factory=dict)
    inr_any_non_leaf: Dict[int, float] = field(default_factory=dict)
    inr_all_non_leaf: Dict[int, float] = field(default_factory=dict)
    n_leaf_mentions: int = 0
    n_non_leaf_mentions: int = 0

    def as_row(self, k: int) -> Dict[str, float]:
        """Flat record in the layout of Tables 4 and 5, percentages except MRR/MAP."""
        return {
            "k": k,
            "InR_any": 100.0 * self.inr_any.get(k, 0.0),
            "InR_all": 100.0 * self.inr_all.get(k, 0.0),
            "InR_any_lf": 100.0 * self.inr_any_leaf.get(k, 0.0),
            "InR_all_lf": 100.0 * self.inr_all_leaf.get(k, 0.0),
            "InR_any_nlf": 100.0 * self.inr_any_non_leaf.get(k, 0.0),
            "InR_all_nlf": 100.0 * self.inr_all_non_leaf.get(k, 0.0),
            "MRR": self.mrr,
            "MAP": self.map,
        }

    def __repr__(self) -> str:  # pragma: no cover - display only
        ks = sorted(self.inr_any)
        parts = [
            f"InR_any@{k}={100 * self.inr_any[k]:.1f} InR_all@{k}={100 * self.inr_all[k]:.1f}"
            for k in ks
        ]
        return (
            f"PlacementScores(n={self.n_mentions}, "
            + ", ".join(parts)
            + f", MRR={self.mrr:.3f}, MAP={self.map:.3f})"
        )


def evaluate_rankings(
    mentions: Sequence[Mention],
    rankings: Mapping[str, Sequence[Edge]],
    k_values: Sequence[int] = (1, 5, 10, 50),
) -> PlacementScores:
    """Compute Eq. (18) in both modes, plus MRR and MAP.

    Parameters
    ----------
    mentions
        Evaluation mentions carrying their gold edge sets ``Y(m)``.
    rankings
        Mapping from ``mention_id`` to the ranked list of predicted edges. A
        mention with no entry is scored as a miss rather than skipped, so that
        the denominator ``|M|`` of Eq. (18) is the full evaluation set.
    k_values
        Cut-offs at which LNR is reported.
    """
    scores = PlacementScores(n_mentions=len(mentions))
    if not mentions:
        return scores

    k_values = sorted(set(k_values))
    any_hits = {k: 0 for k in k_values}
    all_hits = {k: 0 for k in k_values}
    any_hits_lf = {k: 0 for k in k_values}
    all_hits_lf = {k: 0 for k in k_values}
    any_hits_nlf = {k: 0 for k in k_values}
    all_hits_nlf = {k: 0 for k in k_values}
    n_lf = n_nlf = 0
    rr_sum = ap_sum = 0.0

    for m in mentions:
        predicted = list(rankings.get(m.mention_id, ()))
        gold = m.gold_edges

        is_lf = m.has_leaf_gold
        is_nlf = m.has_non_leaf_gold
        n_lf += int(is_lf)
        n_nlf += int(is_nlf)

        for k in k_values:
            hit_any = inr_any(gold, predicted, k)
            hit_all = inr_all(gold, predicted, k)
            any_hits[k] += int(hit_any)
            all_hits[k] += int(hit_all)
            if is_lf:
                # Restricted to the leaf subset of the gold edges, so that the
                # ``lf`` columns measure leaf placement rather than the mention.
                leaf_gold = {e for e in gold if e.is_leaf}
                any_hits_lf[k] += int(inr_any(leaf_gold, predicted, k))
                all_hits_lf[k] += int(inr_all(leaf_gold, predicted, k))
            if is_nlf:
                non_leaf_gold = {e for e in gold if not e.is_leaf}
                any_hits_nlf[k] += int(inr_any(non_leaf_gold, predicted, k))
                all_hits_nlf[k] += int(inr_all(non_leaf_gold, predicted, k))

        rr_sum += reciprocal_rank(gold, predicted)
        ap_sum += average_precision(gold, predicted)

    n = len(mentions)
    scores.inr_any = {k: any_hits[k] / n for k in k_values}
    scores.inr_all = {k: all_hits[k] / n for k in k_values}
    scores.inr_any_leaf = {k: (any_hits_lf[k] / n_lf if n_lf else 0.0) for k in k_values}
    scores.inr_all_leaf = {k: (all_hits_lf[k] / n_lf if n_lf else 0.0) for k in k_values}
    scores.inr_any_non_leaf = {
        k: (any_hits_nlf[k] / n_nlf if n_nlf else 0.0) for k in k_values
    }
    scores.inr_all_non_leaf = {
        k: (all_hits_nlf[k] / n_nlf if n_nlf else 0.0) for k in k_values
    }
    scores.n_leaf_mentions = n_lf
    scores.n_non_leaf_mentions = n_nlf
    scores.mrr = rr_sum / n
    scores.map = ap_sum / n
    return scores


# --------------------------------------------------------------------------- #
# Significance testing -- Section 4.4
# --------------------------------------------------------------------------- #


def paired_bootstrap(
    mentions: Sequence[Mention],
    rankings_a: Mapping[str, Sequence[Edge]],
    rankings_b: Mapping[str, Sequence[Edge]],
    k: int = 10,
    mode: str = "any",
    n_resamples: int = 10_000,
    seed: int = 42,
) -> Dict[str, float]:
    """Paired bootstrap over mention-level predictions.

    Section 4.4 states that the reported figures come from a single run per
    configuration, so differences of one to two percentage points should not be
    treated as established, and that establishing which are stable requires this
    test. It is provided here so that claim can be checked rather than assumed.

    Returns the observed difference ``b - a``, a two-sided p-value, and the 95%
    percentile interval of the difference.
    """
    import random as _random

    metric = inr_any if mode == "any" else inr_all
    per_mention = [
        (
            int(metric(m.gold_edges, list(rankings_a.get(m.mention_id, ())), k)),
            int(metric(m.gold_edges, list(rankings_b.get(m.mention_id, ())), k)),
        )
        for m in mentions
    ]
    n = len(per_mention)
    if n == 0:
        return {"observed_diff": 0.0, "p_value": 1.0, "ci_low": 0.0, "ci_high": 0.0}

    observed = sum(b - a for a, b in per_mention) / n

    rng = _random.Random(seed)
    diffs: List[float] = []
    for _ in range(n_resamples):
        total = 0
        for _ in range(n):
            a, b = per_mention[rng.randrange(n)]
            total += b - a
        diffs.append(total / n)

    centred = [d - observed for d in diffs]
    extreme = sum(1 for d in centred if abs(d) >= abs(observed))
    diffs.sort()

    return {
        "observed_diff": 100.0 * observed,
        "p_value": (extreme + 1) / (n_resamples + 1),
        "ci_low": 100.0 * diffs[int(0.025 * n_resamples)],
        "ci_high": 100.0 * diffs[min(n_resamples - 1, int(0.975 * n_resamples))],
    }


# --------------------------------------------------------------------------- #
# Table rendering
# --------------------------------------------------------------------------- #


def format_table(
    rows: Sequence[Tuple[str, int, PlacementScores, Optional[PlacementScores]]],
    columns: Sequence[str] = ("InR_any", "InR_all", "InR_any_lf", "InR_all_lf", "InR_any_nlf", "InR_all_nlf"),
) -> str:
    """Render results as ``validation / test`` cells, the layout of Table 4.

    ``rows`` is a sequence of ``(model_name, k, validation_scores, test_scores)``.
    Passing ``None`` for the test scores prints the validation figure alone.
    """
    header = ["Model / Set", "k"] + list(columns) + ["MRR", "MAP"]
    lines = ["\t".join(header)]
    for name, k, val, test in rows:
        val_row = val.as_row(k)
        test_row = test.as_row(k) if test is not None else None
        cells = [name, str(k)]
        for col in columns:
            if test_row is None:
                cells.append(f"{val_row[col]:.1f}")
            else:
                cells.append(f"{val_row[col]:.1f} / {test_row[col]:.1f}")
        for col in ("MRR", "MAP"):
            if test_row is None:
                cells.append(f"{val_row[col]:.3f}")
            else:
                cells.append(f"{val_row[col]:.3f} / {test_row[col]:.3f}")
        lines.append("\t".join(cells))
    return "\n".join(lines)
