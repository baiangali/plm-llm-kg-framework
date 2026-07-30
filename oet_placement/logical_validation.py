"""
Logical validation of predicted placements -- Section 4.6.

The ranking metrics of Section 4.2 measure agreement with gold insertion edges.
They do not establish that the resulting ontology remains logically well formed,
which is a separate property and the one that matters for any placement intended
to be applied. Three conditions are reported, reproducing Table 7:

  * consistency is preserved,
  * no cycle is induced in the subsumption hierarchy,
  * no class becomes unsatisfiable.

Two backends are provided.

``GraphBackend`` (default, no dependencies) decides the cycle condition exactly:
inserting ``c`` at ``P -> C`` adds ``c subclass-of P`` and ``C subclass-of c``, so
a cycle arises exactly when ``P`` is already a descendant of ``C``, or when
``P == C``. Consistency and unsatisfiability cannot be decided on the taxonomy
alone -- they depend on the disjointness and existential axioms of the full OWL EL
ontology -- so this backend reports them as ``None`` (not checked) rather than
claiming a pass.

``DeepOntoELKBackend`` runs the ELK reasoner over the ``.owl`` file shipped with
each dataset and decides all three. It is what the reported figures require; the
graph backend alone would understate what is being checked.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .data import NULL_ID, Edge, Mention, Ontology


@dataclass
class ValidationResult:
    """Outcome for one predicted placement."""

    mention_id: str
    edge: Edge
    consistency_preserved: Optional[bool]
    no_induced_cycle: Optional[bool]
    no_unsatisfiable_class: Optional[bool]
    note: str = ""

    @property
    def all_three(self) -> Optional[bool]:
        flags = (
            self.consistency_preserved,
            self.no_induced_cycle,
            self.no_unsatisfiable_class,
        )
        if any(f is None for f in flags):
            return None
        return all(flags)


# --------------------------------------------------------------------------- #
# Graph backend -- exact for the cycle condition
# --------------------------------------------------------------------------- #


class GraphBackend:
    """Structural checks over the subsumption graph, without a reasoner."""

    def __init__(self, ontology: Ontology) -> None:
        self.ontology = ontology

    def validate(self, mention_id: str, edge: Edge) -> ValidationResult:
        cycle_free, note = self._cycle_free(edge)
        return ValidationResult(
            mention_id=mention_id,
            edge=edge,
            consistency_preserved=None,
            no_induced_cycle=cycle_free,
            no_unsatisfiable_class=None,
            note=note,
        )

    def _cycle_free(self, edge: Edge) -> Tuple[bool, str]:
        """Inserting ``c`` at ``P -> C`` adds ``c <= P`` and ``C <= c``.

        The resulting hierarchy contains a cycle iff ``P`` is reachable downward
        from ``C``, because then ``P -> c -> C -> ... -> P``. A leaf insertion
        (``C = NULL``) adds only ``c <= P`` and can never induce a cycle.
        """
        if edge.child == NULL_ID:
            return True, "leaf insertion; no child axiom added"
        if edge.parent == edge.child:
            return False, "parent and child coincide"
        if self.ontology.is_descendant(edge.parent, edge.child):
            return False, "parent is already a descendant of the child"
        return True, ""


# --------------------------------------------------------------------------- #
# ELK backend -- the reasoner used for the reported figures
# --------------------------------------------------------------------------- #


class DeepOntoELKBackend:
    """Classify the ontology with ELK after inserting each predicted concept.

    Requires ``deeponto`` (which bundles the ELK 0.5.0 reasoner and a JVM bridge)
    and the ``.owl`` file of the older SNOMED CT release, both of which the
    MM-S14 distribution provides under ``<dataset>/ontology/``.

    Each placement is checked against a *fresh copy* of the ontology: placements
    are independent hypotheses submitted to a curator, not a cumulative edit, so
    validating them cumulatively would confound errors.
    """

    def __init__(self, owl_path: str, ontology: Ontology, iri_prefix: str = "http://snomed.info/id/") -> None:
        try:
            from deeponto.onto import Ontology as DeepOntology  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "DeepOntoELKBackend requires `pip install deeponto` and a working JVM."
            ) from exc
        if not os.path.exists(owl_path):
            raise FileNotFoundError(owl_path)
        self.owl_path = owl_path
        self.ontology = ontology
        self.iri_prefix = iri_prefix
        self._graph_backend = GraphBackend(ontology)

    def _iri(self, concept_id: str) -> str:
        return f"{self.iri_prefix}{concept_id}"

    def validate(self, mention_id: str, edge: Edge, new_concept_iri: str = "http://example.org/NewConcept") -> ValidationResult:
        from deeponto.onto import Ontology as DeepOntology
        from deeponto.onto import OntologyReasoner

        onto = DeepOntology(self.owl_path)
        owl_api = onto.owl_data_factory
        manager = onto.owl_manager

        new_class = owl_api.getOWLClass(onto.owl_iri(new_concept_iri))
        parent = owl_api.getOWLClass(onto.owl_iri(self._iri(edge.parent)))

        axioms = [owl_api.getOWLSubClassOfAxiom(new_class, parent)]
        if edge.child != NULL_ID:
            child = owl_api.getOWLClass(onto.owl_iri(self._iri(edge.child)))
            axioms.append(owl_api.getOWLSubClassOfAxiom(child, new_class))
        for axiom in axioms:
            manager.addAxiom(onto.owl_onto, axiom)

        reasoner = OntologyReasoner(onto, reasoner_type="elk")
        consistent = bool(reasoner.owl_reasoner.isConsistent())
        unsatisfiable = list(
            reasoner.owl_reasoner.getUnsatisfiableClasses().getEntitiesMinusBottom()
        )
        # ELK classifies rather than reporting cycles directly; a cycle manifests
        # as mutual subsumption between the endpoints, which the graph check
        # decides exactly and far more cheaply.
        cycle_free, note = self._graph_backend._cycle_free(edge)

        return ValidationResult(
            mention_id=mention_id,
            edge=edge,
            consistency_preserved=consistent,
            no_induced_cycle=cycle_free,
            no_unsatisfiable_class=(len(unsatisfiable) == 0),
            note=note,
        )


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def validate_top_ranked(
    ontology: Ontology,
    mentions: Sequence[Mention],
    rankings: Mapping[str, Sequence[Edge]],
    backend=None,
) -> List[ValidationResult]:
    """Validate each mention's top-ranked predicted placement (Section 4.6)."""
    backend = backend or GraphBackend(ontology)
    results: List[ValidationResult] = []
    for m in mentions:
        ranked = list(rankings.get(m.mention_id, ()))
        if not ranked:
            continue
        results.append(backend.validate(m.mention_id, ranked[0]))
    return results


def summarise(results: Sequence[ValidationResult]) -> Dict[str, Optional[float]]:
    """Percentages in the layout of Table 7. ``None`` marks a condition not checked."""
    if not results:
        return {}

    def pct(attr: str) -> Optional[float]:
        values = [getattr(r, attr) for r in results]
        if all(v is None for v in values):
            return None
        checked = [v for v in values if v is not None]
        return 100.0 * sum(checked) / len(checked) if checked else None

    return {
        "n": len(results),
        "consistency_preserved": pct("consistency_preserved"),
        "no_induced_cycle": pct("no_induced_cycle"),
        "no_unsatisfiable_class": pct("no_unsatisfiable_class"),
        "all_three": pct("all_three"),
    }


def format_table7(rows: Sequence[Tuple[str, str, Dict[str, Optional[float]]]]) -> str:
    """Render ``(method, dataset, summary)`` triples in the layout of Table 7."""
    header = [
        "Method", "Dataset", "Consistency preserved",
        "No induced cycle", "No unsatisfiable class", "All three",
    ]
    lines = ["\t".join(header)]
    keys = ["consistency_preserved", "no_induced_cycle", "no_unsatisfiable_class", "all_three"]
    for method, dataset, summary in rows:
        cells = [method, dataset]
        for key in keys:
            value = summary.get(key)
            cells.append("n/a" if value is None else f"{value:.1f}")
        lines.append("\t".join(cells))
    return "\n".join(lines)
