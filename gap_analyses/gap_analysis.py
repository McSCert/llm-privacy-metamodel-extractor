"""
gap_analysis.py — Cross-law gap analysis for PrivacyPolicyMetamodel v4.

What this module does
---------------------
Queries the ModelRepository to answer structured comparison questions
across privacy laws. Each query returns a GapResult — a typed object
that can be rendered as a table, exported as JSON, or fed into a
compliance report generator.

Query catalogue
---------------
Q1  legal_basis_coverage    — which LegalBasisTypes each law allows
Q2  rights_coverage         — which RightTypes each law grants
Q3  retention_coverage      — which laws mandate retention rules
Q4  dpia_coverage           — which laws mandate impact assessments
Q5  transfer_coverage       — which TransferMechanisms each law recognises
Q6  constraint_coverage     — which ConstraintTypes each law imposes
Q7  purpose_coverage        — which PurposeCategories each law governs
Q8  cross_law_delta         — obligations in law A absent from law B
Q9  coverage_matrix         — full concept-by-law presence matrix

These queries are intentionally law-agnostic — they work on whatever
laws are present in the repository. Adding a new law (e.g. India DPDPA)
automatically appears in all comparison results without any code change.

Research use
------------
The coverage_matrix() result is the primary deliverable for a research
paper — it shows at a glance which concepts each law addresses and
where the regulatory gaps are. The cross_law_delta() function produces
the evidence for specific gap claims (e.g. "CCPA does not mandate
data retention periods in any of the 12 extracted statements").
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from .repository import ModelRepository


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class ConceptCoverage:
    """
    Coverage of one concept dimension across all laws.

    concept     : the metamodel concept name (e.g. "LegalBasisType")
    by_law      : dict mapping law → set of values found
                  e.g. {"GDPR": {"Consent","Contract"}, "CCPA": {"LegitimateInterest"}}
    universe    : full set of all values seen across ALL laws (the union)
    """
    concept:  str
    by_law:   dict[str, set[str]]
    universe: set[str]

    def present_in(self, law: str) -> set[str]:
        """Values present for a specific law."""
        return self.by_law.get(law.upper(), set())

    def absent_from(self, law: str) -> set[str]:
        """Values in the universe but absent from this law."""
        return self.universe - self.present_in(law)

    def only_in(self, law: str) -> set[str]:
        """Values unique to this law (not present in any other law)."""
        others = set()
        for other_law, vals in self.by_law.items():
            if other_law != law.upper():
                others.update(vals)
        return self.present_in(law) - others


@dataclass
class FlagCoverage:
    """
    Boolean flag presence across laws.

    flag        : e.g. "has_retention", "has_dpia"
    by_law      : dict mapping law → (statements_with_flag, total_statements)
    """
    flag:   str
    by_law: dict[str, tuple[int, int]]

    def rate(self, law: str) -> float:
        """Fraction of statements with the flag set for a law."""
        flagged, total = self.by_law.get(law.upper(), (0, 0))
        return flagged / total if total > 0 else 0.0

    def mandates(self, law: str, threshold: float = 0.3) -> bool:
        """
        Heuristic: True if > threshold of statements carry this flag.
        A law that mentions DPIA in 50% of its statements treats it as
        a recurring obligation, not a one-off provision.
        """
        return self.rate(law) >= threshold


@dataclass
class CrossLawDelta:
    """
    Obligations present in law A but absent (or weaker) in law B.

    law_a, law_b  : the two laws being compared
    gaps          : list of (concept_dimension, missing_values, note)
    """
    law_a:  str
    law_b:  str
    gaps:   list[tuple[str, set[str], str]] = field(default_factory=list)

    def summary(self) -> str:
        if not self.gaps:
            return f"No gaps found: {self.law_a} and {self.law_b} appear equivalent."
        lines = [f"Obligations in {self.law_a} not present in {self.law_b}:\n"]
        for dimension, missing, note in self.gaps:
            vals = ", ".join(sorted(missing)) if missing else "(structural gap)"
            lines.append(f"  {dimension}: {vals}")
            if note:
                lines.append(f"    → {note}")
        return "\n".join(lines)


@dataclass
class CoverageMatrix:
    """
    Full concept × law presence matrix.

    laws       : ordered list of law names
    dimensions : ordered list of (dimension_label, concept_label) tuples
    cells      : dict mapping (dimension_label, law) → True/False/set
    """
    laws:       list[str]
    dimensions: list[str]
    cells:      dict[tuple[str, str], Any]

    def get(self, dimension: str, law: str) -> Any:
        return self.cells.get((dimension, law.upper()))


# ── GapAnalyser ───────────────────────────────────────────────────────────────

class GapAnalyser:
    """
    Cross-law gap analysis queries over a ModelRepository.

    Usage
    -----
        repo = ModelRepository("data/model_repo.db")
        analyser = GapAnalyser(repo)

        # Individual queries
        lb = analyser.legal_basis_coverage()
        rights = analyser.rights_coverage()
        delta = analyser.cross_law_delta("GDPR", "CCPA")

        # Full matrix
        matrix = analyser.coverage_matrix()
        print(analyser.format_matrix(matrix))

        # Full report
        print(analyser.full_report())
    """

    def __init__(self, repo: ModelRepository):
        self._repo = repo

    @property
    def laws(self) -> list[str]:
        return sorted(self._repo.laws())

    # ── Q1: Legal basis coverage ──────────────────────────────────────────────

    def legal_basis_coverage(self) -> ConceptCoverage:
        """Which LegalBasisTypes does each law allow?"""
        by_law = {
            law: self._repo.distinct_values(law, "legal_basis_types")
            for law in self.laws
        }
        universe = set().union(*by_law.values())
        return ConceptCoverage("LegalBasisType", by_law, universe)

    # ── Q2: Rights coverage ───────────────────────────────────────────────────

    def rights_coverage(self) -> ConceptCoverage:
        """Which RightTypes does each law grant?"""
        by_law = {
            law: self._repo.distinct_values(law, "right_types")
            for law in self.laws
        }
        universe = set().union(*by_law.values())
        return ConceptCoverage("RightType", by_law, universe)

    # ── Q3: Retention coverage ────────────────────────────────────────────────

    def retention_coverage(self) -> FlagCoverage:
        """Does each law mandate retention rules (has_retention flag)?"""
        by_law = {
            law: (
                self._repo.flag_count(law, "has_retention"),
                self._repo.statement_count(law),
            )
            for law in self.laws
        }
        return FlagCoverage("has_retention", by_law)

    # ── Q4: DPIA / risk assessment coverage ───────────────────────────────────

    def dpia_coverage(self) -> FlagCoverage:
        """Does each law mandate risk/impact assessments?"""
        by_law = {
            law: (
                self._repo.flag_count(law, "has_dpia"),
                self._repo.statement_count(law),
            )
            for law in self.laws
        }
        return FlagCoverage("has_dpia", by_law)

    # ── Q5: Transfer mechanism coverage ──────────────────────────────────────

    def transfer_coverage(self) -> ConceptCoverage:
        """Which TransferMechanisms does each law recognise?"""
        by_law = {
            law: self._repo.distinct_values(law, "transfer_mechanisms")
            for law in self.laws
        }
        universe = set().union(*by_law.values())
        return ConceptCoverage("TransferMechanism", by_law, universe)

    # ── Q6: Constraint coverage ───────────────────────────────────────────────

    def constraint_coverage(self) -> ConceptCoverage:
        """Which ConstraintTypes does each law impose?"""
        by_law = {
            law: self._repo.distinct_values(law, "constraint_types")
            for law in self.laws
        }
        universe = set().union(*by_law.values())
        return ConceptCoverage("ConstraintType", by_law, universe)

    # ── Q7: Purpose coverage ──────────────────────────────────────────────────

    def purpose_coverage(self) -> ConceptCoverage:
        """Which PurposeCategories does each law govern?"""
        by_law = {
            law: self._repo.distinct_values(law, "purpose_categories")
            for law in self.laws
        }
        universe = set().union(*by_law.values())
        return ConceptCoverage("PurposeCategory", by_law, universe)

    # ── Q8: Cross-law delta ───────────────────────────────────────────────────

    def cross_law_delta(
        self,
        law_a: str,
        law_b: str,
        flag_threshold: float = 0.3,
    ) -> CrossLawDelta:
        """
        Identify obligations present in law_a but absent or weaker in law_b.

        For enum dimensions: a gap exists when law_a has a value law_b lacks.
        For flag dimensions: a gap exists when law_a mandates (above threshold)
        but law_b does not.

        Parameters
        ----------
        law_a           : the "reference" law (the more demanding one, e.g. GDPR)
        law_b           : the law being compared against it (e.g. CCPA)
        flag_threshold  : fraction of statements that must carry a flag for it
                          to be considered a "mandate" (default 0.3 = 30%)
        """
        law_a = law_a.upper()
        law_b = law_b.upper()
        delta = CrossLawDelta(law_a=law_a, law_b=law_b)

        # Enum dimension gaps
        enum_checks = [
            (self.legal_basis_coverage(), "legal basis types", ""),
            (self.rights_coverage(),      "data subject rights",
             "Missing rights indicate reduced individual control."),
            (self.transfer_coverage(),    "transfer mechanisms",
             "Missing mechanisms restrict lawful international data flows."),
            (self.constraint_coverage(),  "constraint types",
             "Missing constraint types suggest fewer operational restrictions."),
            (self.purpose_coverage(),     "purpose categories", ""),
        ]
        for coverage, label, note in enum_checks:
            missing = coverage.present_in(law_a) - coverage.present_in(law_b)
            if missing:
                delta.gaps.append((label, missing, note))

        # Flag dimension gaps
        ret_a  = FlagCoverage("has_retention", {
            law_a: (self._repo.flag_count(law_a, "has_retention"),
                    self._repo.statement_count(law_a)),
            law_b: (self._repo.flag_count(law_b, "has_retention"),
                    self._repo.statement_count(law_b)),
        })
        dpia_a = FlagCoverage("has_dpia", {
            law_a: (self._repo.flag_count(law_a, "has_dpia"),
                    self._repo.statement_count(law_a)),
            law_b: (self._repo.flag_count(law_b, "has_dpia"),
                    self._repo.statement_count(law_b)),
        })
        cw_a   = FlagCoverage("has_consent_withdrawal", {
            law_a: (self._repo.flag_count(law_a, "has_consent_withdrawal"),
                    self._repo.statement_count(law_a)),
            law_b: (self._repo.flag_count(law_b, "has_consent_withdrawal"),
                    self._repo.statement_count(law_b)),
        })

        flag_checks = [
            (ret_a,  "retention policy mandate",
             f"{law_a} requires storage limitation; {law_b} does not enforce it at scale."),
            (dpia_a, "risk/impact assessment mandate",
             f"{law_a} mandates DPIA for high-risk processing; {law_b} treats it as optional."),
            (cw_a,   "consent withdrawal mandate",
             f"{law_a} requires withdrawal mechanics; {law_b} has weaker or no equivalent."),
        ]
        for flag_cov, label, note in flag_checks:
            a_mandates = flag_cov.mandates(law_a, flag_threshold)
            b_mandates = flag_cov.mandates(law_b, flag_threshold)
            if a_mandates and not b_mandates:
                rate_a = flag_cov.rate(law_a)
                rate_b = flag_cov.rate(law_b)
                delta.gaps.append((
                    label,
                    set(),
                    f"{note} "
                    f"({law_a}: {rate_a:.0%} of statements; "
                    f"{law_b}: {rate_b:.0%} of statements)",
                ))

        return delta

    # ── Q9: Coverage matrix ───────────────────────────────────────────────────

    def coverage_matrix(self) -> CoverageMatrix:
        """
        Build the full concept × law presence matrix.

        Returns a CoverageMatrix where each cell is either:
          - A set of values (for enum dimensions)
          - A float rate (for flag dimensions)
          - True/False (simplified presence)
        """
        laws = self.laws

        # Collect all coverage data
        coverages: dict[str, ConceptCoverage | FlagCoverage] = {
            "Legal basis types":   self.legal_basis_coverage(),
            "Rights granted":      self.rights_coverage(),
            "Constraint types":    self.constraint_coverage(),
            "Purpose categories":  self.purpose_coverage(),
            "Transfer mechanisms": self.transfer_coverage(),
            "Retention rules":     self.retention_coverage(),
            "DPIA/risk assessment": self.dpia_coverage(),
            "Consent withdrawal":   FlagCoverage("has_consent_withdrawal", {
                law: (
                    self._repo.flag_count(law, "has_consent_withdrawal"),
                    self._repo.statement_count(law),
                )
                for law in laws
            }),
        }

        dimensions = list(coverages.keys())
        cells: dict[tuple[str, str], Any] = {}

        for dim, cov in coverages.items():
            for law in laws:
                if isinstance(cov, ConceptCoverage):
                    cells[(dim, law)] = cov.present_in(law)
                else:
                    cells[(dim, law)] = cov.rate(law)

        return CoverageMatrix(laws=laws, dimensions=dimensions, cells=cells)

    # ── Formatting ────────────────────────────────────────────────────────────

    def format_coverage(self, cov: ConceptCoverage, indent: int = 2) -> str:
        pad = " " * indent
        lines = [f"{'─'*60}", f"{cov.concept} coverage", f"{'─'*60}"]
        lines.append(f"Universe (all values seen across all laws):")
        lines.append(f"  {', '.join(sorted(cov.universe)) or '(none)'}")
        lines.append("")
        for law in self.laws:
            present = cov.present_in(law)
            absent  = cov.absent_from(law)
            lines.append(f"  {law}")
            lines.append(f"    present : {', '.join(sorted(present)) or '(none)'}")
            if absent:
                lines.append(f"    absent  : {', '.join(sorted(absent))}")
        return "\n".join(lines)

    def format_flag(self, flag: FlagCoverage, label: str) -> str:
        lines = [f"{'─'*60}", f"{label}", f"{'─'*60}"]
        for law in self.laws:
            flagged, total = flag.by_law.get(law, (0, 0))
            rate = flagged / total if total > 0 else 0.0
            bar  = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
            mandates = "MANDATES" if flag.mandates(law) else "optional"
            lines.append(f"  {law:<8} {bar} {rate:>5.0%}  ({flagged}/{total} stmts)  [{mandates}]")
        return "\n".join(lines)

    def format_matrix(self, matrix: CoverageMatrix) -> str:
        """
        Render the coverage matrix as a compact ASCII table.

        Enum cells show count of values present.
        Flag cells show percentage rate with ✓ (mandates) or · (optional).
        """
        col_w = 10
        dim_w = 24

        # Header row
        header = f"{'Dimension':<{dim_w}}" + "".join(
            f"{law:>{col_w}}" for law in matrix.laws
        )
        sep = "─" * len(header)

        lines = [
            "Coverage Matrix — PrivacyPolicyMetamodel v4",
            "=" * len(header),
            header,
            sep,
        ]

        for dim in matrix.dimensions:
            row = f"{dim:<{dim_w}}"
            for law in matrix.laws:
                cell = matrix.get(dim, law)
                if isinstance(cell, set):
                    val = str(len(cell)) if cell else "—"
                elif isinstance(cell, float):
                    if cell == 0.0:
                        val = "—"
                    elif cell >= 0.3:
                        val = f"✓{cell:.0%}"
                    else:
                        val = f"·{cell:.0%}"
                else:
                    val = str(cell)
                row += f"{val:>{col_w}}"
            lines.append(row)

        lines.append(sep)
        lines.append(
            "Values: integers = distinct enum values found; "
            "✓ = mandated (≥30% of stmts); · = occasional; — = absent"
        )
        return "\n".join(lines)

    def format_delta(self, delta: CrossLawDelta) -> str:
        lines = [
            "─" * 60,
            f"Gap analysis: {delta.law_a} vs {delta.law_b}",
            f"(Obligations in {delta.law_a} not present in {delta.law_b})",
            "─" * 60,
        ]
        if not delta.gaps:
            lines.append("  No significant gaps found.")
        else:
            for dimension, missing, note in delta.gaps:
                if missing:
                    lines.append(f"  {dimension}:")
                    for v in sorted(missing):
                        lines.append(f"    · {v}")
                else:
                    lines.append(f"  {dimension}:")
                if note:
                    lines.append(f"    → {note}")
                lines.append("")
        return "\n".join(lines)

    def full_report(self) -> str:
        """
        Generate a complete human-readable gap analysis report
        covering all 9 query types.
        """
        repo_stats = self._repo.stats()
        laws = self.laws
        matrix = self.coverage_matrix()

        sections: list[str] = []

        # Header
        sections.append("=" * 60)
        sections.append("PRIVACY POLICY GAP ANALYSIS REPORT")
        sections.append(f"PrivacyPolicyMetamodel v4")
        sections.append("=" * 60)
        sections.append(f"\nRepository: {repo_stats['total']} statements across {len(laws)} laws")
        for law, n in repo_stats["by_law"].items():
            sections.append(f"  {law}: {n} statements")

        # Q9 — Coverage matrix (most important — goes first)
        sections.append("\n\n" + self.format_matrix(matrix))

        # Q1 — Legal basis
        lb = self.legal_basis_coverage()
        sections.append("\n\n" + self.format_coverage(lb))

        # Q2 — Rights
        rights = self.rights_coverage()
        sections.append("\n\n" + self.format_coverage(rights))

        # Q3 — Retention
        ret = self.retention_coverage()
        sections.append("\n\n" + self.format_flag(ret, "Retention policy mandate (has_retention)"))

        # Q4 — DPIA
        dpia = self.dpia_coverage()
        sections.append("\n\n" + self.format_flag(dpia, "DPIA / risk assessment mandate (has_dpia)"))

        # Q5 — Transfer
        xfr = self.transfer_coverage()
        sections.append("\n\n" + self.format_coverage(xfr))

        # Q6 — Constraints
        con = self.constraint_coverage()
        sections.append("\n\n" + self.format_coverage(con))

        # Q7 — Purposes
        pur = self.purpose_coverage()
        sections.append("\n\n" + self.format_coverage(pur))

        # Q8 — Pairwise deltas (all pairs)
        if len(laws) >= 2:
            sections.append("\n\n" + "─" * 60)
            sections.append("PAIRWISE GAP ANALYSIS")
            for i, a in enumerate(laws):
                for b in laws[i+1:]:
                    sections.append("\n" + self.format_delta(
                        self.cross_law_delta(a, b)
                    ))
                    sections.append(self.format_delta(
                        self.cross_law_delta(b, a)
                    ))

        return "\n".join(sections)

    def to_json(self) -> str:
        """
        Export all gap analysis results as a JSON object.
        Suitable for downstream processing or integration with
        a compliance management system.
        """
        laws = self.laws
        matrix = self.coverage_matrix()

        def set_to_list(s: Any) -> Any:
            return sorted(s) if isinstance(s, set) else s

        result = {
            "laws": laws,
            "stats": self._repo.stats(),
            "legal_basis_coverage":  {
                law: sorted(self.legal_basis_coverage().present_in(law))
                for law in laws
            },
            "rights_coverage": {
                law: sorted(self.rights_coverage().present_in(law))
                for law in laws
            },
            "constraint_coverage": {
                law: sorted(self.constraint_coverage().present_in(law))
                for law in laws
            },
            "purpose_coverage": {
                law: sorted(self.purpose_coverage().present_in(law))
                for law in laws
            },
            "transfer_coverage": {
                law: sorted(self.transfer_coverage().present_in(law))
                for law in laws
            },
            "retention_rates": {
                law: round(self.retention_coverage().rate(law), 3)
                for law in laws
            },
            "dpia_rates": {
                law: round(self.dpia_coverage().rate(law), 3)
                for law in laws
            },
            "coverage_matrix": {
                dim: {
                    law: set_to_list(matrix.get(dim, law))
                    for law in laws
                }
                for dim in matrix.dimensions
            },
            "pairwise_deltas": [
                {
                    "law_a": a,
                    "law_b": b,
                    "gaps": [
                        {
                            "dimension": dim,
                            "missing_values": sorted(missing),
                            "note": note,
                        }
                        for dim, missing, note in self.cross_law_delta(a, b).gaps
                    ],
                }
                for i, a in enumerate(laws)
                for b in laws[i+1:]
            ],
        }
        return json.dumps(result, indent=2)
