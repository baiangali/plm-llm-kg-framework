"""
Textual serialisation of mentions and edges (Section 3.3, "Notation") and
rule-based verbalisation of complex class expressions (Section 3.2).

The special tokens below are added to the tokenizer vocabulary by
``add_special_tokens`` so that they survive subword tokenisation as single units.

    Eq. (1)   mention:    [CLS] ctx_l [Ms] mention [Me] ctx_r [SEP]
    Eq. (2)   edge:       [CLS] parent [P-TAG] child [C-TAG] [SEP]
    Eq. (11)  cross:      [CLS] ctx_l [Ms] mention [Me] ctx_r [SEP]
                          parent [P-TAG] child [C-TAG] [SEP]

``[CLS]`` and ``[SEP]`` are supplied by the HuggingFace tokenizer, so the
functions here return the *inner* text of each segment. For Eq. (11) the two
segments are returned as a pair and handed to the tokenizer as a sentence pair,
which reproduces the ``[CLS] a [SEP] b [SEP]`` layout exactly.
"""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Special tokens
# --------------------------------------------------------------------------- #

MENTION_START = "[Ms]"
MENTION_END = "[Me]"
PARENT_TAG = "[P-TAG]"
CHILD_TAG = "[C-TAG]"
NULL_TOKEN = "[NULL]"

SPECIAL_TOKENS: List[str] = [
    MENTION_START,
    MENTION_END,
    PARENT_TAG,
    CHILD_TAG,
    NULL_TOKEN,
]


def add_special_tokens(tokenizer, model=None):
    """Register the markers of Section 3.3 and resize embeddings if a model is given."""
    n_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": SPECIAL_TOKENS}
    )
    if model is not None and n_added:
        model.resize_token_embeddings(len(tokenizer))
    return n_added


# --------------------------------------------------------------------------- #
# Eq. (1) -- mention serialisation
# --------------------------------------------------------------------------- #


def serialize_mention(
    context_left: str,
    mention: str,
    context_right: str,
    max_context_words: int = 64,
) -> str:
    """Serialise a mention with its context, Eq. (1).

    Absent contexts are the empty string, as stated in Section 3.3. Contexts are
    truncated by words from the side away from the mention so that the mention
    span itself is never truncated away by the tokenizer's length cap.
    """
    left = (context_left or "").strip()
    right = (context_right or "").strip()

    if max_context_words is not None and max_context_words > 0:
        left_words = left.split()
        right_words = right.split()
        if len(left_words) > max_context_words:
            left = " ".join(left_words[-max_context_words:])
        if len(right_words) > max_context_words:
            right = " ".join(right_words[:max_context_words])

    parts = [left, MENTION_START, (mention or "").strip(), MENTION_END, right]
    return " ".join(p for p in parts if p).strip()


# --------------------------------------------------------------------------- #
# Eq. (2) -- edge serialisation
# --------------------------------------------------------------------------- #


def serialize_edge(parent_text: str, child_text: str | None) -> str:
    """Serialise an ontology edge, Eq. (2).

    A leaf edge ``P -> NULL`` places ``[NULL]`` in the child position, giving

        renal failure [P-TAG] [NULL] [C-TAG]

    and a non-leaf edge

        renal failure [P-TAG] chronic renal failure [C-TAG]
    """
    parent_text = (parent_text or "").strip()
    child = (child_text or "").strip()
    if not child or child.upper() == "NULL":
        child = NULL_TOKEN
    return f"{parent_text} {PARENT_TAG} {child} {CHILD_TAG}".strip()


# --------------------------------------------------------------------------- #
# Eq. (11) -- joint mention/edge serialisation for the cross-encoder
# --------------------------------------------------------------------------- #


def serialize_cross(
    context_left: str,
    mention: str,
    context_right: str,
    parent_text: str,
    child_text: str | None,
    max_context_words: int = 64,
) -> Tuple[str, str]:
    """Return the ``(segment_a, segment_b)`` pair of Eq. (11)."""
    return (
        serialize_mention(context_left, mention, context_right, max_context_words),
        serialize_edge(parent_text, child_text),
    )


# --------------------------------------------------------------------------- #
# Section 3.2 -- verbalisation of complex class expressions
# --------------------------------------------------------------------------- #

# Complex concept identifiers in the MM-S14 catalogues use the DeepOnto rendering
#
#     [EX.](<609096000> [EX.](<42752001> <55985003>))
#
# i.e. an existential restriction  EX.(role, filler)  where role is an atomic
# concept id and filler is either an atomic id or a nested restriction. Role
# grouping (Spackman et al.) appears as the outermost role 609096000.

_ATOMIC_RE = re.compile(r"^<([^<>]+)>$")
_EX_RE = re.compile(r"^\[EX\.\]\((.*)\)$", re.DOTALL)


def is_complex(concept_id: str) -> bool:
    """True if the identifier denotes a complex class expression rather than a named class."""
    return bool(concept_id) and ("[" in concept_id or "<" in concept_id)


def _split_top_level(body: str) -> List[str]:
    """Split an ``[EX.](...)`` body into its two arguments, respecting nesting."""
    parts: List[str] = []
    depth = 0
    current: List[str] = []
    for ch in body:
        if ch in "(<":
            depth += 1
        elif ch in ")>":
            depth -= 1
        if ch == " " and depth == 0:
            if current:
                parts.append("".join(current))
                current = []
            continue
        current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


def verbalise(concept_id: str, title_of) -> str:
    """Render a (possibly complex) class expression as natural language.

    ``title_of`` maps a named-class identifier to its label. Section 3.2 states
    that complex expressions are handled at the level of representation: they are
    verbalised so that a language model can encode them alongside atomic labels,
    but they are not manipulated symbolically.

    Example
    -------
    ``[EX.](<609096000> [EX.](<42752001> <55985003>))`` becomes

        "something that is Role group (attribute) some something that is
         Due to (attribute) some Atopic reaction (disorder)"
    """
    concept_id = (concept_id or "").strip()
    if not concept_id:
        return ""

    m = _ATOMIC_RE.match(concept_id)
    if m:
        return title_of(m.group(1))

    m = _EX_RE.match(concept_id)
    if m:
        args = _split_top_level(m.group(1))
        if len(args) == 2:
            role = verbalise(args[0], title_of)
            filler = verbalise(args[1], title_of)
            return f"something that is {role} some {filler}"
        # Malformed or unexpected arity: fall back to the concatenation of parts.
        return " ".join(verbalise(a, title_of) for a in args)

    # A bare named-class identifier.
    return title_of(concept_id)


def normalise_label(text: str) -> str:
    """Light normalisation of ontology labels used for lexical matching only."""
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def strip_semantic_tag(title: str) -> str:
    """Remove the trailing SNOMED CT semantic tag, e.g. "(disorder)".

    Kept separate from :func:`normalise_label` because the tag is informative for
    the language model and is *not* stripped in the serialisations of Eq. (1),
    (2) and (11); it is stripped only where a bag-of-words match would otherwise
    be dominated by the tag, which every concept in a subtree shares.
    """
    return re.sub(r"\s*\([^()]*\)\s*$", "", title or "").strip()


def tokenise_for_lexical_match(text: str) -> Sequence[str]:
    """Word-level tokenisation used by the inverted-index baseline (Section 4.3)."""
    return re.findall(r"[a-z0-9]+", normalise_label(text))
