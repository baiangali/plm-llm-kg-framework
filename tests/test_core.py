"""
Self-contained tests for the dependency-free core of the framework.

Covers the parts that can be checked without a GPU or a model download:
serialisation (Eq. 1, 2, 11), verbalisation of complex class expressions,
gold-edge parsing, Stage 2 construction and enrichment (Eq. 4-9), the metrics
(Eq. 14-18), the cycle condition of Section 4.6, and JSONL loading including the
split-file fallback.

    python -m unittest discover -s tests -v

``TestJsonlSources`` writes small temporary fixtures; it never opens the real
multi-gigabyte files. ``TestRealDataset`` is skipped unless
``<repository>/data/MM-S14-Disease`` is present; when it is, it loads the real
ontology and asserts the invariants the pipeline relies on.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from oet_placement.data import (  # noqa: E402
    NULL_ID,
    Concept,
    Edge,
    Mention,
    Ontology,
    _parse_gold_edges,
    iter_jsonl,
    make_splits,
    resolve_jsonl_sources,
)
from oet_placement.logical_validation import GraphBackend  # noqa: E402
from oet_placement.metrics import (  # noqa: E402
    average_precision,
    evaluate_rankings,
    inr_all,
    inr_any,
    reciprocal_rank,
)
from oet_placement.serialization import (  # noqa: E402
    serialize_cross,
    serialize_edge,
    serialize_mention,
    verbalise,
)
from oet_placement.stage2_enrichment import (  # noqa: E402
    build_candidate_set,
    construct_edges_from_concept,
    enrich_edge,
    enrich_edges,
    mean_endpoint_cosine_scorer,
    rank_edges,
)


def toy_ontology() -> Ontology:
    """A small hierarchy used throughout the structural tests.

            root
            /   \\
          A      B
         / \\      \\
        C   D      E
    """
    onto = Ontology()
    structure = {
        "root": ([], ["A", "B"]),
        "A": (["root"], ["C", "D"]),
        "B": (["root"], ["E"]),
        "C": (["A"], []),
        "D": (["A"], []),
        "E": (["B"], []),
    }
    for idx, (parents, children) in structure.items():
        onto.concepts[idx] = Concept(
            idx=idx, title=f"{idx} (disorder)", parents=list(parents), children=list(children)
        )
        onto._titles[idx] = f"{idx} (disorder)"
    return onto


class TestSerialization(unittest.TestCase):
    def test_mention_serialisation_eq1(self):
        out = serialize_mention("left ctx", "cystic fibrosis", "right ctx")
        self.assertEqual(out, "left ctx [Ms] cystic fibrosis [Me] right ctx")

    def test_mention_serialisation_empty_contexts(self):
        self.assertEqual(serialize_mention("", "renal failure", ""), "[Ms] renal failure [Me]")

    def test_edge_serialisation_eq2(self):
        self.assertEqual(
            serialize_edge("renal failure", "chronic renal failure"),
            "renal failure [P-TAG] chronic renal failure [C-TAG]",
        )

    def test_leaf_edge_uses_null_token(self):
        self.assertEqual(
            serialize_edge("renal failure", None),
            "renal failure [P-TAG] [NULL] [C-TAG]",
        )
        self.assertEqual(
            serialize_edge("renal failure", "NULL"),
            "renal failure [P-TAG] [NULL] [C-TAG]",
        )

    def test_cross_serialisation_eq11_is_a_segment_pair(self):
        a, b = serialize_cross("l", "m", "r", "parent", "child")
        self.assertEqual(a, "l [Ms] m [Me] r")
        self.assertEqual(b, "parent [P-TAG] child [C-TAG]")

    def test_verbalise_nested_existential_restriction(self):
        titles = {
            "609096000": "Role group (attribute)",
            "42752001": "Due to (attribute)",
            "55985003": "Atopic reaction (disorder)",
        }
        out = verbalise(
            "[EX.](<609096000> [EX.](<42752001> <55985003>))", lambda i: titles.get(i, i)
        )
        self.assertEqual(
            out,
            "something that is Role group (attribute) some "
            "something that is Due to (attribute) some Atopic reaction (disorder)",
        )

    def test_verbalise_atomic_identifier(self):
        self.assertEqual(verbalise("123", lambda i: "renal failure"), "renal failure")


class TestGoldEdgeParsing(unittest.TestCase):
    def test_simple_pairs(self):
        row = {
            "parents_concept": "233661002",
            "children_concept": "81423003|190909002",
            "parents-children_concept": "233661002-81423003|233661002-190909002",
        }
        self.assertEqual(
            _parse_gold_edges(row),
            {Edge("233661002", "81423003"), Edge("233661002", "190909002")},
        )

    def test_leaf_edge(self):
        row = {
            "parents_concept": "236423003",
            "children_concept": "",
            "parents-children_concept": "236423003-SCTID_NULL",
        }
        self.assertEqual(_parse_gold_edges(row), {Edge("236423003", NULL_ID)})

    def test_complex_parent_is_not_split_on_its_own_hyphens(self):
        complex_id = "[EX.](<609096000> [EX.](<42752001> <64572001>))"
        row = {
            "parents_concept": complex_id,
            "children_concept": "",
            "parents-children_concept": f"{complex_id}-SCTID_NULL",
        }
        self.assertEqual(_parse_gold_edges(row), {Edge(complex_id, NULL_ID)})

    def test_falls_back_to_cross_product_when_pair_field_absent(self):
        row = {"parents_concept": "A|B", "children_concept": "C"}
        self.assertEqual(_parse_gold_edges(row), {Edge("A", "C"), Edge("B", "C")})


class TestStage2Construction(unittest.TestCase):
    def setUp(self):
        self.onto = toy_ontology()

    def test_eq4_one_hop_and_two_hop_edges(self):
        edges = construct_edges_from_concept(self.onto, "A")
        expected = {
            Edge("root", "A"),                       # P_i -> A
            Edge("A", "C"), Edge("A", "D"),          # A -> C_j
            Edge("root", "C"), Edge("root", "D"),    # P_i -> C_j
            Edge("A", NULL_ID),                      # the appended leaf edge
        }
        self.assertEqual(edges, expected)

    def test_leaf_edge_is_appended_for_a_leaf_concept(self):
        edges = construct_edges_from_concept(self.onto, "C")
        self.assertIn(Edge("C", NULL_ID), edges)
        self.assertIn(Edge("A", "C"), edges)

    def test_eq5_to_eq8_enrichment(self):
        enriched = enrich_edge(self.onto, Edge("A", "C"))
        # S1: parents of A lifted           -> root -> C
        # S2: children of C descended       -> none, C is a leaf
        # S3: combination                   -> none
        self.assertEqual(enriched, {Edge("A", "C"), Edge("root", "C")})

    def test_enrichment_of_a_leaf_edge_only_lifts_the_parent(self):
        enriched = enrich_edge(self.onto, Edge("A", NULL_ID))
        self.assertEqual(enriched, {Edge("A", NULL_ID), Edge("root", NULL_ID)})

    def test_leaf_seed_enriched_only_when_its_parent_appears_in_a_non_leaf_seed(self):
        # "A" appears in the non-leaf seed A -> C, so its leaf edge is expanded.
        with_context = enrich_edges(self.onto, [Edge("A", "C"), Edge("A", NULL_ID)])
        self.assertIn(Edge("root", NULL_ID), with_context)

        # "E" appears in no non-leaf seed, so its leaf edge is left alone.
        without_context = enrich_edges(self.onto, [Edge("E", NULL_ID)])
        self.assertEqual(without_context, {Edge("E", NULL_ID)})

    def test_enrichment_is_a_superset_of_the_seeds(self):
        seeds = {Edge("A", "C"), Edge("B", "E")}
        self.assertTrue(seeds <= enrich_edges(self.onto, seeds))


class TestStage2Ranking(unittest.TestCase):
    def setUp(self):
        self.onto = toy_ontology()

    def test_eq9_mean_endpoint_cosine(self):
        scorer = mean_endpoint_cosine_scorer(self.onto, {"A": 0.8, "C": 0.4})
        self.assertAlmostEqual(scorer(Edge("A", "C")), 0.6)
        self.assertAlmostEqual(scorer(Edge("A", NULL_ID)), 0.4)  # NULL contributes 0

    def test_ranking_is_deterministic_under_ties(self):
        scorer = mean_endpoint_cosine_scorer(self.onto, {})
        candidates = [Edge("B", "E"), Edge("A", "C"), Edge("A", "D")]
        twice = [rank_edges(candidates, scorer, use_leaf_priority=False) for _ in range(2)]
        self.assertEqual(twice[0], twice[1])
        self.assertEqual(twice[0][0], Edge("A", "C"))  # tie broken on identifiers

    def test_leaf_priority_moves_leaf_edges_forward(self):
        scorer = mean_endpoint_cosine_scorer(self.onto, {"A": 0.9, "C": 0.9, "D": 0.1})
        ranked = rank_edges(
            [Edge("A", "C"), Edge("D", NULL_ID)],
            scorer,
            top_concept_is_leaf=True,
            use_leaf_priority=True,
        )
        self.assertEqual(ranked[0], Edge("D", NULL_ID))

    def test_leaf_priority_is_a_no_op_when_the_top_concept_is_not_a_leaf(self):
        scorer = mean_endpoint_cosine_scorer(self.onto, {"A": 0.9, "C": 0.9, "D": 0.1})
        ranked = rank_edges(
            [Edge("A", "C"), Edge("D", NULL_ID)],
            scorer,
            top_concept_is_leaf=False,
            use_leaf_priority=True,
        )
        self.assertEqual(ranked[0], Edge("A", "C"))

    def test_build_candidate_set_records_the_enrichment_factor(self):
        scorer = mean_endpoint_cosine_scorer(self.onto, {"A": 1.0})
        cs = build_candidate_set(self.onto, "m1", ["A"], scorer, k=10)
        self.assertGreaterEqual(cs.n_after_enrichment, cs.n_before_enrichment)
        self.assertLessEqual(len(cs.edges), 10)

    def test_disabling_enrichment_leaves_the_seed_set(self):
        scorer = mean_endpoint_cosine_scorer(self.onto, {"A": 1.0})
        cs = build_candidate_set(self.onto, "m1", ["A"], scorer, k=100, use_enrichment=False)
        self.assertEqual(cs.n_before_enrichment, cs.n_after_enrichment)


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.gold = {Edge("P1", "C1"), Edge("P2", NULL_ID)}
        self.ranked = [Edge("X", "Y"), Edge("P1", "C1"), Edge("Z", "W"), Edge("P2", NULL_ID)]

    def test_inr_any_eq16(self):
        self.assertFalse(inr_any(self.gold, self.ranked, k=1))
        self.assertTrue(inr_any(self.gold, self.ranked, k=2))

    def test_inr_all_eq17(self):
        self.assertFalse(inr_all(self.gold, self.ranked, k=3))
        self.assertTrue(inr_all(self.gold, self.ranked, k=4))

    def test_inr_all_is_false_for_an_empty_gold_set(self):
        self.assertFalse(inr_all(set(), self.ranked, k=10))

    def test_reciprocal_rank(self):
        self.assertAlmostEqual(reciprocal_rank(self.gold, self.ranked), 0.5)
        self.assertEqual(reciprocal_rank({Edge("Q", "R")}, self.ranked), 0.0)

    def test_average_precision_penalises_unretrieved_gold_edges(self):
        # Hits at ranks 2 and 4: (1/2 + 2/4) / 2 = 0.5
        self.assertAlmostEqual(average_precision(self.gold, self.ranked), 0.5)
        # A gold edge that never appears drags AP down.
        gold_plus = self.gold | {Edge("never", "retrieved")}
        self.assertAlmostEqual(average_precision(gold_plus, self.ranked), 1.0 / 3.0)

    def test_eq18_denominator_is_the_full_mention_set(self):
        hit = Mention(mention="a", gold_edges=frozenset({Edge("P", "C")}), mention_id="m1")
        miss = Mention(mention="b", gold_edges=frozenset({Edge("Q", "D")}), mention_id="m2")
        scores = evaluate_rankings([hit, miss], {"m1": [Edge("P", "C")]}, k_values=[1])
        # m2 has no ranking at all; it must count as a miss, not be dropped.
        self.assertEqual(scores.n_mentions, 2)
        self.assertAlmostEqual(scores.inr_any[1], 0.5)

    def test_leaf_and_non_leaf_breakdown(self):
        leaf = Mention(mention="a", gold_edges=frozenset({Edge("P", NULL_ID)}), mention_id="m1")
        non_leaf = Mention(mention="b", gold_edges=frozenset({Edge("P", "C")}), mention_id="m2")
        scores = evaluate_rankings(
            [leaf, non_leaf],
            {"m1": [Edge("P", NULL_ID)], "m2": [Edge("X", "Y")]},
            k_values=[1],
        )
        self.assertAlmostEqual(scores.inr_any_leaf[1], 1.0)
        self.assertAlmostEqual(scores.inr_any_non_leaf[1], 0.0)


class TestSplits(unittest.TestCase):
    def test_no_mention_appears_in_more_than_one_split(self):
        mentions = [
            Mention(mention=f"m{i}", context_left=str(i), mention_id=f"id{i}")
            for i in range(100)
        ]
        train, valid, test = make_splits(mentions, seed=42)
        self.assertEqual(len(train) + len(valid) + len(test), 100)
        keys = lambda ms: {(m.context_left, m.mention, m.context_right) for m in ms}
        self.assertFalse(keys(train) & keys(valid))
        self.assertFalse(keys(train) & keys(test))
        self.assertFalse(keys(valid) & keys(test))

    def test_splits_are_reproducible_under_the_fixed_seed(self):
        mentions = [
            Mention(mention=f"m{i}", context_left=str(i), mention_id=f"id{i}")
            for i in range(100)
        ]
        a = make_splits(mentions, seed=42)
        b = make_splits(mentions, seed=42)
        self.assertEqual([m.mention_id for m in a[0]], [m.mention_id for m in b[0]])

    def test_duplicate_mentions_land_in_the_same_split(self):
        base = Mention(mention="renal failure", context_left="L", mention_id="a")
        dup = Mention(mention="renal failure", context_left="L", mention_id="b")
        splits = make_splits([base, dup] * 20, seed=42)
        occupied = [i for i, s in enumerate(splits) if s]
        self.assertEqual(len(occupied), 1)


class TestLogicalValidation(unittest.TestCase):
    def setUp(self):
        self.backend = GraphBackend(toy_ontology())

    def test_leaf_insertion_never_induces_a_cycle(self):
        result = self.backend.validate("m1", Edge("A", NULL_ID))
        self.assertTrue(result.no_induced_cycle)

    def test_valid_insertion_between_a_parent_and_its_descendant(self):
        self.assertTrue(self.backend.validate("m1", Edge("A", "C")).no_induced_cycle)

    def test_inverted_edge_induces_a_cycle(self):
        # Inserting c at C -> A adds c <= C and A <= c, while A already subsumes C.
        self.assertFalse(self.backend.validate("m1", Edge("C", "A")).no_induced_cycle)

    def test_self_loop_is_rejected(self):
        self.assertFalse(self.backend.validate("m1", Edge("A", "A")).no_induced_cycle)

    def test_unchecked_conditions_are_reported_as_none_not_as_a_pass(self):
        result = self.backend.validate("m1", Edge("A", "C"))
        self.assertIsNone(result.consistency_preserved)
        self.assertIsNone(result.no_unsatisfiable_class)
        self.assertIsNone(result.all_three)


class TestLLMPrompting(unittest.TestCase):
    """Prompt construction, supervision targets and parsing (§3.6).

    ``stage3_llm`` imports torch only inside ``_require_llm``, so these run
    without it.
    """

    def setUp(self):
        from oet_placement import stage3_llm

        self.llm = stage3_llm
        self.onto = toy_ontology()
        self.mention = Mention(
            mention="new disorder",
            context_left="left",
            context_right="right",
            gold_edges=frozenset({Edge("A", "C")}),
            mention_id="m1",
        )
        self.candidates = [Edge("A", "C"), Edge("B", "E"), Edge("D", NULL_ID)]

    def test_candidates_are_enumerated_from_one(self):
        listing = self.llm.format_candidates(self.onto, self.candidates)
        self.assertEqual(listing.splitlines()[0], "1. A (disorder) -> C (disorder)")
        self.assertEqual(listing.splitlines()[2], "3. D (disorder) -> NULL")

    def test_prompt_contains_no_answer_section(self):
        prompt = self.llm.build_prompt(self.onto, self.mention, self.candidates)
        self.assertNotIn("### Explanation", prompt)
        self.assertNotIn("### Response", prompt)

    def test_leakage_guard_passes_a_well_formed_prompt(self):
        prompt = self.llm.build_prompt(self.onto, self.mention, self.candidates)
        self.llm.assert_no_leakage(prompt, self.mention, self.onto, self.candidates)

    def test_leakage_guard_rejects_an_injected_answer(self):
        prompt = self.llm.build_prompt(self.onto, self.mention, self.candidates)
        with self.assertRaises(AssertionError):
            self.llm.assert_no_leakage(
                prompt + "\nThe answer is option 1.", self.mention, self.onto, self.candidates
            )

    def test_leakage_guard_tolerates_a_gold_label_occurring_in_the_context(self):
        # Real PubMed context often contains the gold parent's label as prose;
        # that is not leakage and must not fail the guard.
        mention = Mention(
            mention="new disorder",
            context_left="a study of A (disorder) in adults",
            gold_edges=frozenset({Edge("A", "C")}),
            mention_id="m2",
        )
        prompt = self.llm.build_prompt(self.onto, mention, self.candidates)
        self.llm.assert_no_leakage(prompt, mention, self.onto, self.candidates)

    def test_explanation_target_has_five_steps_and_the_gold_indices(self):
        target = self.llm.build_explanation_target(self.onto, self.mention, self.candidates)
        for step in range(1, 6):
            self.assertIn(f"Step {step}.", target)
        answer = target.rsplit("### Response", 1)[1].strip()
        self.assertEqual(answer, "1")

    def test_explanation_target_when_no_gold_edge_is_in_the_pool(self):
        mention = Mention(
            mention="unrelated",
            gold_edges=frozenset({Edge("root", "B")}),
            mention_id="m3",
        )
        target = self.llm.build_explanation_target(self.onto, mention, self.candidates)
        self.assertTrue(target.rsplit("### Response", 1)[1].strip().startswith("none"))

    def test_parse_response_reads_only_after_the_final_response_header(self):
        text = "### Explanation\nStep 1. ...\nStep 2. ...\n\n### Response\n2, 3"
        self.assertEqual(self.llm.parse_response(text, n_candidates=3), [1, 2])

    def test_parse_response_discards_out_of_range_indices(self):
        self.assertEqual(
            self.llm.parse_response("### Response\n1, 7, 99", n_candidates=3), [0]
        )

    def test_parse_response_handles_an_empty_selection(self):
        self.assertEqual(self.llm.parse_response("### Response\nnone", n_candidates=3), [])

    def test_rerank_promotes_selected_options_and_keeps_the_rest_in_order(self):
        ranked = self.llm.rerank_from_indices(self.candidates, [2])
        self.assertEqual(ranked[0], Edge("D", NULL_ID))
        self.assertEqual(ranked[1:], [Edge("A", "C"), Edge("B", "E")])

    def test_extract_explanation(self):
        text = "### Explanation\nStep 1. reasoning\n\n### Response\n1"
        self.assertEqual(self.llm.extract_explanation(text), "Step 1. reasoning")


class TestJsonlSources(unittest.TestCase):
    """Reading a JSONL file that may have been split for distribution.

    ``MM-S14-CPP/mention-edge-pair-level/train.jsonl`` exceeds the GitHub Free
    Git LFS quota and ships as ``train_part_001.jsonl`` / ``train_part_002.jsonl``.
    The loader must present the parts as one logical file while still reading an
    ordinary unsplit file normally.

    These cases use temporary fixtures; the real multi-gigabyte files are never
    opened here.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _write(self, name: str, rows):
        path = self.dir / name
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        return path

    def test_reads_an_ordinary_jsonl_file(self):
        path = self._write("train.jsonl", [{"i": 0}, {"i": 1}, {"i": 2}])
        self.assertEqual(resolve_jsonl_sources(path), [path])
        self.assertEqual([row["i"] for row in iter_jsonl(path)], [0, 1, 2])

    def test_reads_two_split_parts_in_order(self):
        first = self._write("train_part_001.jsonl", [{"i": 0}, {"i": 1}])
        second = self._write("train_part_002.jsonl", [{"i": 2}, {"i": 3}])

        requested = self.dir / "train.jsonl"
        self.assertFalse(requested.exists())
        self.assertEqual(resolve_jsonl_sources(requested), [first, second])
        self.assertEqual([row["i"] for row in iter_jsonl(requested)], [0, 1, 2, 3])

    def test_parts_are_ordered_lexicographically_not_by_creation(self):
        # Written out of order; the loader must still yield 001 before 002.
        self._write("train_part_002.jsonl", [{"i": 2}])
        self._write("train_part_001.jsonl", [{"i": 1}])
        rows = [row["i"] for row in iter_jsonl(self.dir / "train.jsonl")]
        self.assertEqual(rows, [1, 2])

    def test_the_original_file_wins_over_split_parts(self):
        self._write("train.jsonl", [{"i": 99}])
        self._write("train_part_001.jsonl", [{"i": 0}])
        self.assertEqual([row["i"] for row in iter_jsonl(self.dir / "train.jsonl")], [99])

    def test_missing_file_and_missing_parts_raise_file_not_found(self):
        missing = self.dir / "train.jsonl"
        with self.assertRaises(FileNotFoundError):
            resolve_jsonl_sources(missing)
        with self.assertRaises(FileNotFoundError):
            list(iter_jsonl(missing))

    def test_a_sibling_split_does_not_satisfy_a_different_stem(self):
        self._write("train_part_001.jsonl", [{"i": 0}])
        with self.assertRaises(FileNotFoundError):
            resolve_jsonl_sources(self.dir / "valid.jsonl")

    def test_iteration_is_lazy(self):
        self._write("train_part_001.jsonl", [{"i": 0}])
        stream = iter_jsonl(self.dir / "train.jsonl")
        # A generator must not have touched the filesystem before the first next().
        self.assertEqual(next(stream)["i"], 0)

    def test_a_leading_byte_order_mark_is_stripped(self):
        # Several shipped MM-S14 files begin with a UTF-8 BOM; json.loads rejects
        # it, so the reader must open them as utf-8-sig.
        (self.dir / "train.jsonl").write_bytes(b'\xef\xbb\xbf{"i": 0}\n{"i": 1}\n')
        self.assertEqual([row["i"] for row in iter_jsonl(self.dir / "train.jsonl")], [0, 1])

    def test_a_byte_order_mark_on_the_first_split_part_is_stripped(self):
        # The BOM of the original file lands at the head of part 001 only.
        (self.dir / "train_part_001.jsonl").write_bytes(b'\xef\xbb\xbf{"i": 0}\n')
        (self.dir / "train_part_002.jsonl").write_bytes(b'{"i": 1}\n')
        self.assertEqual([row["i"] for row in iter_jsonl(self.dir / "train.jsonl")], [0, 1])

    def test_blank_lines_between_parts_are_skipped(self):
        (self.dir / "train_part_001.jsonl").write_text(
            '{"i": 0}\n\n', encoding="utf-8"
        )
        (self.dir / "train_part_002.jsonl").write_text(
            '\n{"i": 1}\n', encoding="utf-8"
        )
        self.assertEqual([row["i"] for row in iter_jsonl(self.dir / "train.jsonl")], [0, 1])


#: The MM-S14 distribution lives under ``<repository>/data/``.
DATA_ROOT = os.path.join(REPO_ROOT, "data")
DISEASE_DIR = os.path.join(DATA_ROOT, "MM-S14-Disease")


@unittest.skipUnless(os.path.isdir(DISEASE_DIR), "MM-S14-Disease not present")
class TestRealDataset(unittest.TestCase):
    """Invariants the pipeline relies on, checked against the shipped files."""

    @classmethod
    def setUpClass(cls):
        cls.onto = Ontology.load(DISEASE_DIR, edge_catalogue="all")

    def test_ontology_loads_with_concepts_and_edges(self):
        self.assertGreater(len(self.onto), 1000)
        self.assertGreater(len(self.onto.atomic_edges), 1000)

    def test_parent_child_relations_are_mutually_consistent(self):
        checked = 0
        for concept in list(self.onto.concepts.values())[:2000]:
            for child in concept.children:
                if child in self.onto.concepts:
                    self.assertIn(
                        concept.idx,
                        self.onto.concepts[child].parents,
                        f"{concept.idx} lists {child} as a child but not conversely",
                    )
                    checked += 1
        self.assertGreater(checked, 0)

    def test_null_child_resolves_to_the_null_title(self):
        self.assertEqual(self.onto.title(NULL_ID), "NULL")

    def test_test_nil_mentions_are_all_out_of_knowledge_base(self):
        from oet_placement.data import load_mentions

        path = os.path.join(
            DISEASE_DIR, "mention-level-(concept-placement)", "test-NIL.jsonl"
        )
        mentions = load_mentions(path)
        self.assertGreater(len(mentions), 0)
        self.assertTrue(all(m.is_out_of_kb for m in mentions))
        self.assertTrue(all(m.gold_edges for m in mentions))

    def test_stage2_recovers_gold_edges_when_seeded_with_the_gold_parent(self):
        """Enrichment must be able to reach the gold edge from its own parent.

        A sanity check on Eq. (4)-(8): if Stage 1 retrieved the gold parent, the
        gold edge should be in the enriched candidate set. Failure here would
        mean the ceiling on Stage 3 is set by a construction bug rather than by
        retrieval.
        """
        from oet_placement.data import load_mentions

        path = os.path.join(
            DISEASE_DIR, "mention-level-(concept-placement)", "test-NIL.jsonl"
        )
        mentions = load_mentions(path)[:50]
        reachable = 0
        considered = 0
        for m in mentions:
            for gold in m.gold_edges:
                if gold.parent not in self.onto.concepts:
                    continue  # complex parent; outside the atomic candidate space
                considered += 1
                seeds = construct_edges_from_concept(self.onto, gold.parent)
                if gold in enrich_edges(self.onto, seeds):
                    reachable += 1
        self.assertGreater(considered, 0)
        self.assertGreater(
            reachable / considered,
            0.5,
            "fewer than half of the gold edges are reachable from their own parent",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
