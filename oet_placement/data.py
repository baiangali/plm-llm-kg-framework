"""
Ontology model and dataset loading for MM-S14-Disease and MM-S14-CPP.

Formalisation follows Section 1.1 of the paper. An ontology is a pair
``O = (V, E)`` with ``E subset of V x (V union {NULL})``. An edge is written
``P -> C``; ``C = NULL`` denotes a leaf edge at which the inserted concept
terminates the hierarchy.

The MM-S14 distribution encodes ``NULL`` as the literal identifier
``SCTID_NULL`` with title ``NULL``, which is the convention adopted here.

This module has no third-party dependencies.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, Iterator, List, NamedTuple, Optional, Sequence, Set, Tuple

from .serialization import is_complex, verbalise

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Identifier used by the MM-S14 files for the absent child of a leaf edge.
NULL_ID = "SCTID_NULL"
#: Human-readable title of the above.
NULL_TITLE = "NULL"
#: ``label_concept`` value marking a mention whose concept is absent from the
#: older ontology version -- the out-of-knowledge-base condition of Section 1.1.
OUT_OF_KB_MARKER = "SCTID-less"


class Edge(NamedTuple):
    """An ordered parent-child pair. ``child == NULL_ID`` marks a leaf edge."""

    parent: str
    child: str

    @property
    def is_leaf(self) -> bool:
        return self.child == NULL_ID

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.parent} -> {self.child}"


def leaf_edge(parent: str) -> Edge:
    """The leaf edge ``P -> NULL``."""
    return Edge(parent, NULL_ID)


# --------------------------------------------------------------------------- #
# Concepts and the ontology
# --------------------------------------------------------------------------- #


@dataclass
class Concept:
    idx: str
    title: str
    text: str = ""
    synonyms: List[str] = field(default_factory=list)
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return not self.children


class Ontology:
    """The older SNOMED CT version used for training and candidate retrieval.

    Section 3.1 gives the ontology three simultaneous roles: it is the candidate
    space, the source of supervision, and -- through :meth:`parents_of` and
    :meth:`children_of`, which Stage 2 calls at inference time -- an active
    structural resource during prediction.
    """

    def __init__(self) -> None:
        self.concepts: Dict[str, Concept] = {}
        #: Titles of every identifier seen, including complex class expressions.
        self._titles: Dict[str, str] = {NULL_ID: NULL_TITLE}
        #: Atomic subsumption edges, i.e. the transitive reduction over named classes.
        self.atomic_edges: List[Edge] = []
        #: Full edge catalogue including complex parents and derived (degree > 0) edges.
        self.edge_catalogue: List[Edge] = []
        self._descendant_cache: Dict[Tuple[str, str], bool] = {}

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    @classmethod
    def load(
        cls,
        dataset_dir: str,
        edge_catalogue: str = "all",
    ) -> "Ontology":
        """Load the entity and edge catalogues of one MM-S14 part.

        ``dataset_dir`` is e.g. ``<root>/MM-S14-Disease``. ``edge_catalogue`` is
        ``"all"`` (atomic + complex, Section 3.2 "in scope" plus verbalised
        complex parents) or ``"atomic"`` (named classes only).
        """
        onto = cls()
        ontology_dir = os.path.join(dataset_dir, "ontology")

        entity_file = _find_file(ontology_dir, "_syn_attr_hyp-all.jsonl")
        onto._load_entities(entity_file)

        atomic_file = _find_file(ontology_dir, "-edges-atomic.jsonl")
        onto.atomic_edges = onto._load_edges(atomic_file, direct_only=True)

        catalogue_suffix = (
            "-edges-all.jsonl" if edge_catalogue == "all" else "-edges-atomic.jsonl"
        )
        catalogue_file = _find_file(ontology_dir, catalogue_suffix)
        onto.edge_catalogue = onto._load_edges(catalogue_file, direct_only=False)

        return onto

    def _load_entities(self, path: str) -> None:
        for row in iter_jsonl(path):
            idx = row["idx"]
            parents = _split_pipe(row.get("parents_idx", ""))
            children = _split_pipe(row.get("children_idx", ""))
            concept = Concept(
                idx=idx,
                title=row.get("title", "") or row.get("entity", ""),
                text=row.get("text", "") or "",
                synonyms=_split_pipe(row.get("synonyms", "")),
                parents=parents,
                children=children,
            )
            self.concepts[idx] = concept
            self._titles[idx] = concept.title

            # The catalogue also carries the titles of the neighbours, which is
            # the only place where the labels of complex parents appear.
            for ids, titles in (
                (parents, _split_pipe(row.get("parents", ""))),
                (children, _split_pipe(row.get("children", ""))),
            ):
                if len(ids) == len(titles):
                    for i, t in zip(ids, titles):
                        self._titles.setdefault(i, t)

    def _load_edges(self, path: str, direct_only: bool) -> List[Edge]:
        edges: List[Edge] = []
        for row in iter_jsonl(path):
            if direct_only and row.get("degree", 0) != 0:
                continue
            parent, child = row["parent_idx"], row["child_idx"]
            self._titles.setdefault(parent, row.get("parent", ""))
            self._titles.setdefault(child, row.get("child", NULL_TITLE))
            edges.append(Edge(parent, child))
        return edges

    # ------------------------------------------------------------------ #
    # Structural accessors -- Par(x) and Ch(x) of Section 3.4
    # ------------------------------------------------------------------ #

    def parents_of(self, idx: str) -> List[str]:
        """``Par(x)``: the direct parents of ``x``. Empty for NULL and unknown ids."""
        if idx == NULL_ID:
            return []
        concept = self.concepts.get(idx)
        return list(concept.parents) if concept else []

    def children_of(self, idx: str) -> List[str]:
        """``Ch(x)``: the direct children of ``x``. Empty for NULL and unknown ids."""
        if idx == NULL_ID:
            return []
        concept = self.concepts.get(idx)
        return list(concept.children) if concept else []

    def is_leaf_concept(self, idx: str) -> bool:
        concept = self.concepts.get(idx)
        return concept is not None and concept.is_leaf

    # ------------------------------------------------------------------ #
    # Labels
    # ------------------------------------------------------------------ #

    def title(self, idx: str) -> str:
        """Label of a named class, or the verbalisation of a complex expression."""
        if idx == NULL_ID or not idx:
            return NULL_TITLE
        known = self._titles.get(idx)
        if known:
            return known
        if is_complex(idx):
            rendered = verbalise(idx, lambda i: self._titles.get(i, i))
            self._titles[idx] = rendered
            return rendered
        return idx

    def concept_text(self, idx: str, with_synonyms: bool = False) -> str:
        """Text used to embed a concept: its title, optionally plus synonyms."""
        concept = self.concepts.get(idx)
        if concept is None:
            return self.title(idx)
        if with_synonyms and concept.synonyms:
            return concept.title + " ; " + " ; ".join(concept.synonyms)
        return concept.title

    def edge_text(self, edge: Edge) -> Tuple[str, str]:
        """``(parent_label, child_label)`` for Eq. (2) and Eq. (11)."""
        return self.title(edge.parent), self.title(edge.child)

    # ------------------------------------------------------------------ #
    # Graph queries used by the logical validation of Section 4.6
    # ------------------------------------------------------------------ #

    def is_descendant(self, candidate: str, ancestor: str, max_nodes: int = 200_000) -> bool:
        """True if ``candidate`` is reachable from ``ancestor`` by following Ch(.)."""
        if candidate == ancestor:
            return True
        if candidate == NULL_ID or ancestor == NULL_ID:
            return False
        key = (candidate, ancestor)
        cached = self._descendant_cache.get(key)
        if cached is not None:
            return cached

        seen: Set[str] = {ancestor}
        stack = list(self.children_of(ancestor))
        found = False
        while stack and len(seen) < max_nodes:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            if node == candidate:
                found = True
                break
            stack.extend(self.children_of(node))

        self._descendant_cache[key] = found
        return found

    def depth(self, idx: str, max_depth: int = 64) -> int:
        """Shortest distance to a root, used for the depth statistics of Table 3."""
        if idx == NULL_ID:
            return -1
        frontier = {idx}
        seen: Set[str] = set()
        for d in range(max_depth):
            nxt: Set[str] = set()
            for node in frontier:
                parents = self.parents_of(node)
                if not parents:
                    return d
                nxt.update(p for p in parents if p not in seen)
            seen |= frontier
            if not nxt:
                return d
            frontier = nxt
        return max_depth

    def __len__(self) -> int:
        return len(self.concepts)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"Ontology(concepts={len(self.concepts)}, "
            f"atomic_edges={len(self.atomic_edges)}, "
            f"catalogue={len(self.edge_catalogue)})"
        )


# --------------------------------------------------------------------------- #
# Mentions
# --------------------------------------------------------------------------- #


@dataclass
class Mention:
    """A textual mention with its context and its gold insertion edges.

    ``gold_edges`` is the set ``Y(m)`` of Section 1.1. It is set-valued: a
    mention may admit several valid insertion positions, which is what separates
    the ANY and ALL evaluation modes of Section 4.2.
    """

    mention: str
    context_left: str = ""
    context_right: str = ""
    gold_edges: FrozenSet[Edge] = frozenset()
    label_concept: str = ""
    label_concept_ori: str = ""
    label_concept_umls: str = ""
    label_title: str = ""
    label_text: str = ""
    mention_id: str = ""

    @property
    def is_out_of_kb(self) -> bool:
        """The condition ``c not in V`` of Section 1.1, enforced by construction."""
        return (
            self.label_concept == OUT_OF_KB_MARKER
            or self.label_title.strip().upper() == NULL_TITLE
        )

    @property
    def has_leaf_gold(self) -> bool:
        return any(e.is_leaf for e in self.gold_edges)

    @property
    def has_non_leaf_gold(self) -> bool:
        return any(not e.is_leaf for e in self.gold_edges)


def load_mentions(
    path: str,
    out_of_kb_only: bool = False,
) -> List[Mention]:
    """Load a ``mention-level-(concept-placement)`` file.

    Each row aggregates all gold edges of one mention, which is the granularity
    the evaluation of Section 4.2 operates at.
    """
    mentions: List[Mention] = []
    for i, row in enumerate(iter_jsonl(path)):
        gold = _parse_gold_edges(row)
        m = Mention(
            mention=row.get("mention", ""),
            context_left=row.get("context_left", ""),
            context_right=row.get("context_right", ""),
            gold_edges=frozenset(gold),
            label_concept=row.get("label_concept", ""),
            label_concept_ori=row.get("label_concept_ori", ""),
            label_concept_umls=row.get("label_concept_UMLS", ""),
            label_title=row.get("label_title", ""),
            label_text=row.get("label", "") or "",
            mention_id=f"{os.path.basename(path)}:{i}",
        )
        if out_of_kb_only and not m.is_out_of_kb:
            continue
        mentions.append(m)
    return mentions


class MentionEdgePair(NamedTuple):
    """One row of a ``mention-edge-pair-level`` file: a positive training example."""

    mention: Mention
    edge: Edge


def load_edge_pairs(path: str, out_of_kb_only: bool = False) -> List[MentionEdgePair]:
    """Load a ``mention-edge-pair-level`` file, used to train Stages 1 and 3."""
    pairs: List[MentionEdgePair] = []
    for i, row in enumerate(iter_jsonl(path)):
        parent = (row.get("parent_concept") or "").strip()
        child = (row.get("child_concept") or "").strip() or NULL_ID
        edge = Edge(parent, child)
        m = Mention(
            mention=row.get("mention", ""),
            context_left=row.get("context_left", ""),
            context_right=row.get("context_right", ""),
            gold_edges=frozenset({edge}),
            label_concept=row.get("label_concept", ""),
            label_concept_ori=row.get("label_concept_ori", ""),
            label_concept_umls=row.get("label_concept_UMLS", ""),
            label_title=row.get("entity_label_title", ""),
            label_text=row.get("entity_label", "") or "",
            mention_id=f"{os.path.basename(path)}:{i}",
        )
        if out_of_kb_only and not m.is_out_of_kb:
            continue
        pairs.append(MentionEdgePair(m, edge))
    return pairs


def aggregate_pairs(pairs: Sequence[MentionEdgePair]) -> List[Mention]:
    """Collapse mention-edge pairs into mention-level examples with set-valued gold.

    Mentions are keyed on ``(context_left, mention, context_right)``, which is
    the identity used for the mention-level split of Section 3.7 -- "no mention
    appearing in more than one split".
    """
    grouped: Dict[Tuple[str, str, str], Mention] = {}
    order: List[Tuple[str, str, str]] = []
    for pair in pairs:
        key = (pair.mention.context_left, pair.mention.mention, pair.mention.context_right)
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = Mention(
                mention=pair.mention.mention,
                context_left=pair.mention.context_left,
                context_right=pair.mention.context_right,
                gold_edges=frozenset({pair.edge}),
                label_concept=pair.mention.label_concept,
                label_concept_ori=pair.mention.label_concept_ori,
                label_concept_umls=pair.mention.label_concept_umls,
                label_title=pair.mention.label_title,
                label_text=pair.mention.label_text,
                mention_id=pair.mention.mention_id,
            )
            order.append(key)
        else:
            grouped[key] = Mention(
                **{
                    **existing.__dict__,
                    "gold_edges": existing.gold_edges | {pair.edge},
                }
            )
    return [grouped[k] for k in order]


# --------------------------------------------------------------------------- #
# Splits -- Section 3.7
# --------------------------------------------------------------------------- #


def make_splits(
    mentions: Sequence[Mention],
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> Tuple[List[Mention], List[Mention], List[Mention]]:
    """Mention-level 70/15/15 split under a fixed seed (Section 3.7).

    Splitting is on the mention identity, so no mention appears in more than one
    split. The same seed and the same input ordering therefore yield the same
    indices for every method and every ablation setting.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1, got {ratios}")

    keys = sorted({(m.context_left, m.mention, m.context_right) for m in mentions})
    rng = random.Random(seed)
    rng.shuffle(keys)

    n = len(keys)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    assignment: Dict[Tuple[str, str, str], int] = {}
    for i, key in enumerate(keys):
        assignment[key] = 0 if i < n_train else (1 if i < n_train + n_val else 2)

    buckets: Tuple[List[Mention], List[Mention], List[Mention]] = ([], [], [])
    for m in mentions:
        buckets[assignment[(m.context_left, m.mention, m.context_right)]].append(m)
    return buckets


def split_indices_report(
    splits: Tuple[Sequence[Mention], Sequence[Mention], Sequence[Mention]]
) -> Dict[str, List[str]]:
    """Serialisable record of which mention went to which split (Section 3.7)."""
    names = ["train", "valid", "test"]
    return {name: [m.mention_id for m in split] for name, split in zip(names, splits)}


# --------------------------------------------------------------------------- #
# Dataset statistics -- Table 3
# --------------------------------------------------------------------------- #


def dataset_statistics(mentions: Sequence[Mention], ontology: Ontology) -> Dict[str, float]:
    """Reproduce the statistics reported in Table 3 for a given mention set."""
    if not mentions:
        return {}
    counts = [len(m.gold_edges) for m in mentions]
    counts_sorted = sorted(counts)
    n = len(counts)
    median = (
        counts_sorted[n // 2]
        if n % 2
        else (counts_sorted[n // 2 - 1] + counts_sorted[n // 2]) / 2
    )
    all_edges = [e for m in mentions for e in m.gold_edges]
    n_leaf = sum(1 for e in all_edges if e.is_leaf)

    depths = [ontology.depth(e.parent) for e in all_edges if not is_complex(e.parent)]
    degrees = [
        len(ontology.children_of(e.parent))
        for e in all_edges
        if e.parent in ontology.concepts
    ]

    return {
        "mentions": n,
        "gold_edges_total": len(all_edges),
        "mean_gold_edges_per_mention": sum(counts) / n,
        "median_gold_edges_per_mention": median,
        "single_gold_edge_pct": 100.0 * sum(1 for c in counts if c == 1) / n,
        "leaf_pct": 100.0 * n_leaf / max(1, len(all_edges)),
        "non_leaf_pct": 100.0 * (len(all_edges) - n_leaf) / max(1, len(all_edges)),
        "mean_depth": (sum(depths) / len(depths)) if depths else float("nan"),
        "mean_degree": (sum(degrees) / len(degrees)) if degrees else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


#: Infix marking a shard of a file that was split for distribution. The largest
#: MM-S14 file, ``MM-S14-CPP/mention-edge-pair-level/train.jsonl``, exceeds the
#: GitHub Free Git LFS quota and ships as ``train_part_001.jsonl`` and
#: ``train_part_002.jsonl``. The split is line-by-line, so every part is itself
#: well-formed JSONL and concatenating them reproduces the original byte stream.
_PART_INFIX = "_part_"


def resolve_jsonl_sources(path) -> List[Path]:
    """Locate the file or files backing one logical JSONL path.

    The original file is used when it exists. Otherwise the split parts matching
    ``<stem>_part_*<suffix>`` in the same directory are returned in lexicographic
    order, which is the order in which they were written and therefore the order
    that reconstructs the original line sequence.

    Raises
    ------
    FileNotFoundError
        If neither the original file nor any split part exists.
    """
    target = Path(path)
    if target.is_file():
        return [target]

    pattern = f"{target.stem}{_PART_INFIX}*{target.suffix}"
    parts = sorted(p for p in target.parent.glob(pattern) if p.is_file())
    if parts:
        return parts

    raise FileNotFoundError(
        f"{target} does not exist and no split parts matching "
        f"{pattern!r} were found in {target.parent}"
    )


def iter_jsonl(path) -> Iterator[dict]:
    """Stream a JSONL file, transparently concatenating any split parts.

    The parts are read one at a time and yielded row by row, so a multi-gigabyte
    file is never held in memory. Line boundaries are preserved by the split
    itself, so no part begins or ends mid-record and each is parsed independently.

    Several files in the MM-S14 distribution begin with a UTF-8 byte-order mark
    (``MM-S14-Disease/mention-edge-pair-level/train.jsonl`` among them), which
    ``json.loads`` rejects. ``utf-8-sig`` strips a leading BOM when one is present
    and is otherwise identical to ``utf-8``.
    """
    for source in resolve_jsonl_sources(path):
        with source.open("r", encoding="utf-8-sig") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _split_pipe(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [part for part in value.split("|") if part]


def _find_file(directory: str, suffix: str) -> str:
    """Locate the single file in ``directory`` ending with ``suffix``."""
    matches = sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.endswith(suffix)
    )
    if not matches:
        raise FileNotFoundError(f"no file ending in {suffix!r} under {directory!r}")
    return matches[0]


def _parse_gold_edges(row: dict) -> Set[Edge]:
    """Recover ``Y(m)`` from a mention-level row.

    The ``parents-children_concept`` field pairs each parent with each child as
    ``"<parent>-<child>"``, pipe-separated. Parent identifiers may be complex
    class expressions, so the split point is located by matching against the
    known parent identifiers rather than by splitting on the first hyphen.
    """
    raw = row.get("parents-children_concept") or ""
    parent_ids = _split_pipe(row.get("parents_concept", ""))
    if not raw:
        # Fall back to the cross product of the parent and child lists.
        child_ids = _split_pipe(row.get("children_concept", "")) or [NULL_ID]
        return {Edge(p, c) for p in parent_ids for c in child_ids}

    known_parents = sorted(parent_ids, key=len, reverse=True)
    edges: Set[Edge] = set()
    for chunk in _split_pipe(raw):
        parent = child = None
        for candidate in known_parents:
            if chunk.startswith(candidate + "-"):
                parent = candidate
                child = chunk[len(candidate) + 1 :]
                break
        if parent is None:
            parent, _, child = chunk.rpartition("-")
        if not parent:
            continue
        edges.add(Edge(parent, child or NULL_ID))
    return edges


def dataset_paths(root: str, dataset: str) -> Dict[str, str]:
    """Canonical file locations within one MM-S14 part."""
    base = os.path.join(root, dataset)
    mention_level = os.path.join(base, "mention-level-(concept-placement)")
    pair_level = os.path.join(base, "mention-edge-pair-level")
    return {
        "root": base,
        "ontology": os.path.join(base, "ontology"),
        "mention_train": os.path.join(mention_level, "train.jsonl"),
        "mention_valid": os.path.join(mention_level, "valid.jsonl"),
        "mention_valid_nil": os.path.join(mention_level, "valid-NIL.jsonl"),
        "mention_test_nil": os.path.join(mention_level, "test-NIL.jsonl"),
        "mention_test_nil_all": os.path.join(mention_level, "test-NIL-all.jsonl"),
        "mention_test_in_kb": os.path.join(mention_level, "test-in-KB.jsonl"),
        "pair_train": os.path.join(pair_level, "train.jsonl"),
        "pair_valid": os.path.join(pair_level, "valid.jsonl"),
        "pair_valid_nil": os.path.join(pair_level, "valid-NIL.jsonl"),
        "pair_test_nil": os.path.join(pair_level, "test-NIL.jsonl"),
        "pair_test_nil_all": os.path.join(pair_level, "test-NIL-all.jsonl"),
    }
