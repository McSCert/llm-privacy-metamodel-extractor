#!/usr/bin/env python3
"""
evaluate.py — Precision / Recall / F1 evaluation of the extraction pipeline.

Compares pipeline output stored in ModelRepository (SQLite) against a
manually-annotated gold standard (.xlsx or .csv) to produce per-concept
P/R/F1 scores.

Usage
-----
    python evaluate.py \
        --gold  evaluation/pipeda_gold_standard.xlsx \
        --repo  data/model_repo.db \
        --law   PIPEDA \
        --out   evaluation/results.json

    # Markdown table output (for paper):
    python evaluate.py --gold evaluation/pipeda_gold_standard.xlsx \
        --repo data/model_repo.db --law PIPEDA --format markdown

Gold standard format (.xlsx sheet or .csv)
------------------------------------------
The header row is located automatically (first row containing "principle"),
so a title row above the header is fine.

Required columns (case-insensitive):
    principle                 — article number only, e.g. "4.1", "4.10"
    LegalBasis_type           — single value | ABSENT | UNCERTAIN
    ProcessingActivity_action — single value | ABSENT | UNCERTAIN
    Actor_role                — single value | ABSENT | UNCERTAIN
    Purpose_category          — SET       | ABSENT | UNCERTAIN
    Right_type                — SET       | ABSENT | UNCERTAIN
    Constraint_type           — SET       | ABSENT | UNCERTAIN
    RetentionPolicy_present   — TRUE | FALSE | UNCERTAIN
    DataTransfer_present      — TRUE | FALSE | UNCERTAIN
    ConsentWithdrawal_present — TRUE | FALSE | UNCERTAIN

Optional columns:
    principle_name            — human-readable name (display only)
    article_ref               — exact pipeline article_ref; overrides matching
    notes                     — free text

SET-valued columns hold one or more enum literals separated by "|", e.g.
    Access | Rectification
Order and surrounding whitespace are irrelevant; duplicates are collapsed.
Purpose, Right and Constraint are [0..*] in the metamodel and are therefore
scored as sets.  LegalBasis, ProcessingActivity.action and Actor.role are
[1] in the metamodel; a set there is unreachable by construction, so those
three are validated to be single-valued and scored as a set of size one.

Special values (whole cell, not a set member):
    ABSENT    — concept genuinely not in the text; pipeline should extract none
    UNCERTAIN — annotator unsure; row excluded from P/R for this concept

Evaluation logic
----------------
Set-valued concepts are scored PER LABEL, not per row:

    TP = |predicted ∩ gold|
    FP = |predicted \\ gold|      (extra / hallucinated labels)
    FN = |gold \\ predicted|      (missed labels)
    TN = 1 when gold and predicted are both empty (reported, not scored)

This is the change from the pre-2026 metric, which asked only whether the
single gold value appeared somewhere in the predicted set.  Under that rule
a model could emit every literal in the enum and score perfect recall at no
cost in precision, so over-extraction was invisible.  Per-label scoring makes
multi-instance extraction measurable in both directions.

For boolean concepts:
    TP = gold=TRUE  AND pipeline extracted the concept (non-empty)
    TN = gold=FALSE AND pipeline did NOT extract it
    FP = gold=FALSE AND pipeline extracted it
    FN = gold=TRUE  AND pipeline did NOT extract it

Article matching
----------------
A gold row matches the pipeline statement whose article_ref begins with the
same article number token: gold "4.1" matches "4.1 Principle 1 — Accountability"
but NOT "4.10 Principle 10 — Challenging" and NOT "6.1 For the purposes of
clause 4.3 ...".  When several statements share a leading token, a Schedule 1
principle heading ("<n> Principle ...") wins; any remaining ambiguity is
reported as an error rather than resolved silently.  Set an `article_ref`
column in the gold file to pin a row explicitly.

Metrics
-------
    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    F1        = 2 * P * R / (P + R)

    macro_*   = mean of per-concept precision / recall; macro_f1 is their
                harmonic mean (comparable with earlier reports)
    mean_f1   = mean of per-concept F1 (reported alongside; the two differ)
    micro_*   = computed from summed TP / FP / FN across concepts

Output
------
    Console : match report + per-concept table + averages
    --out   : JSON file with full results
    --format markdown : GitHub/LaTeX-friendly table
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# =============================================================================
# CONSTANTS
# =============================================================================

# Set-valued concepts ([0..*] in the metamodel) and their repository columns
_SET_CONCEPTS: dict[str, Optional[str]] = {
    "Purpose_category":         "purpose_categories",
    "Right_type":               "right_types",
    "Constraint_type":          "constraint_types",
}

# Single-valued concepts ([1] in the metamodel); scored as singleton sets
_SINGLE_CONCEPTS: dict[str, Optional[str]] = {
    "LegalBasis_type":          "legal_basis_types",
    "ProcessingActivity_action": "processing_actions",
    "Actor_role":               None,   # not denormalised — read from JSON
}

# Evaluation order for the report
_ENUM_CONCEPTS: dict[str, Optional[str]] = {
    "LegalBasis_type":           _SINGLE_CONCEPTS["LegalBasis_type"],
    "ProcessingActivity_action": _SINGLE_CONCEPTS["ProcessingActivity_action"],
    "Actor_role":                _SINGLE_CONCEPTS["Actor_role"],
    "Purpose_category":          _SET_CONCEPTS["Purpose_category"],
    "Right_type":                _SET_CONCEPTS["Right_type"],
    "Constraint_type":           _SET_CONCEPTS["Constraint_type"],
}

_BOOL_CONCEPTS: dict[str, str] = {
    "RetentionPolicy_present":    "has_retention",
    "DataTransfer_present":       "has_transfer",
    "ConsentWithdrawal_present":  "has_consent_withdrawal",
}

_ABSENT      = "ABSENT"
_UNCERTAIN   = "UNCERTAIN"
_SET_SEP     = "|"

# Allowed literals per concept, mirroring privacy_schema/enums.py.
# Used only to warn about typos in the gold file; not enforced.
_ALLOWED: dict[str, set[str]] = {
    "LegalBasis_type": {
        "Consent", "Contract", "LegalObligation", "LegitimateInterest",
        "VitalInterest", "PublicTask"},
    "ProcessingActivity_action": {
        "Collect", "Store", "Use", "Share", "Transfer", "Delete"},
    "Actor_role": {
        "DataSubject", "DataController", "DataProcessor", "ThirdParty"},
    "Purpose_category": {
        "ServiceProvision", "Security", "LegalCompliance", "Marketing",
        "Analytics", "Research"},
    "Right_type": {
        "Access", "Rectification", "Erasure", "Restriction", "Portability",
        "Objection", "AutomatedDecisionOptOut"},
    "Constraint_type": {
        "Temporal", "Geographic", "Usage", "Security", "Retention",
        "PurposeLimitation", "Accuracy", "Transparency"},
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ConceptResult:
    concept:   str
    tp:        int = 0
    fp:        int = 0
    fn:        int = 0
    tn:        int = 0   # rows where gold and prediction are both empty
    skipped:   int = 0   # UNCERTAIN rows excluded

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def support(self) -> int:
        """Number of gold-standard positive labels (TP + FN)."""
        return self.tp + self.fn


@dataclass
class EvaluationReport:
    law:            str
    gold_path:      str
    repo_path:      str
    n_principles:   int
    results:        list[ConceptResult] = field(default_factory=list)
    matches:        list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def macro_precision(self) -> float:
        vals = [r.precision for r in self.results if r.support > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def macro_recall(self) -> float:
        vals = [r.recall for r in self.results if r.support > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def macro_f1(self) -> float:
        p, r = self.macro_precision, self.macro_recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def mean_f1(self) -> float:
        vals = [r.f1 for r in self.results if r.support > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def micro_precision(self) -> float:
        tp = sum(r.tp for r in self.results)
        fp = sum(r.fp for r in self.results)
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    @property
    def micro_recall(self) -> float:
        tp = sum(r.tp for r in self.results)
        fn = sum(r.fn for r in self.results)
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    @property
    def micro_f1(self) -> float:
        p, r = self.micro_precision, self.micro_recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def n_unmatched(self) -> int:
        return sum(1 for _, _, ref in self.matches if ref == "NO MATCH")


# =============================================================================
# GOLD STANDARD LOADER
# =============================================================================

def _rows_from_xlsx(path: Path, sheet: Optional[str]) -> list[list]:
    try:
        import openpyxl
    except ImportError:
        print("ERROR: reading .xlsx gold standards requires openpyxl "
              "(pip install openpyxl), or export the sheet to .csv.")
        sys.exit(1)
    wb = openpyxl.load_workbook(path, data_only=True)
    if sheet:
        if sheet not in wb.sheetnames:
            print(f"ERROR: sheet {sheet!r} not in {path} "
                  f"(sheets: {wb.sheetnames})")
            sys.exit(1)
        ws = wb[sheet]
    else:
        ws = wb.worksheets[0]
    return [list(r) for r in ws.iter_rows(values_only=True)]


def _rows_from_csv(path: Path) -> list[list]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [list(r) for r in csv.reader(f)]


def load_gold_standard(path: Path, sheet: Optional[str] = None) -> list[dict]:
    """
    Load and normalise the gold standard from .xlsx or .csv.

    The header row is the first row containing a cell equal to "principle"
    (case-insensitive), so a title row above the header is tolerated.
    Returns a list of dicts keyed by the header names as written.
    """
    raw = (_rows_from_xlsx(path, sheet) if path.suffix.lower() in (".xlsx", ".xlsm")
           else _rows_from_csv(path))

    header_idx = None
    for i, row in enumerate(raw):
        cells = [str(c).strip().lower() if c is not None else "" for c in row]
        if "principle" in cells:
            header_idx = i
            break
    if header_idx is None:
        print(f"ERROR: no header row containing 'principle' found in {path}")
        sys.exit(1)

    header = [str(c).strip() if c is not None else "" for c in raw[header_idx]]
    rows: list[dict] = []
    for row in raw[header_idx + 1:]:
        vals = list(row) + [None] * (len(header) - len(row))
        rec = {h: ("" if v is None else str(v).strip())
               for h, v in zip(header, vals) if h}
        if not rec.get("principle"):
            continue          # blank or trailing row
        rows.append(rec)

    if not rows:
        print(f"ERROR: gold standard has no data rows: {path}")
        sys.exit(1)

    required = ({"principle"} | set(_ENUM_CONCEPTS.keys())
                | set(_BOOL_CONCEPTS.keys()))
    missing = required - set(rows[0].keys())
    if missing:
        print(f"ERROR: gold standard missing columns: {sorted(missing)}\n"
              f"       Found: {sorted(rows[0].keys())}")
        sys.exit(1)

    _validate_gold(rows, path)
    return rows


def parse_gold_cell(value: str) -> Optional[set[str]]:
    """
    Parse one gold cell into a set of labels.

    Returns None for UNCERTAIN (row excluded), an empty set for ABSENT or a
    blank cell, otherwise the set of "|"-separated literals.
    """
    v = (value or "").strip()
    if v.upper() == _UNCERTAIN:
        return None
    if v == "" or v.upper() == _ABSENT:
        return set()
    return {p.strip() for p in v.split(_SET_SEP) if p.strip()}


def _validate_gold(rows: list[dict], path: Path) -> None:
    """Warn about duplicate principles, unknown literals and illegal sets."""
    warnings: list[str] = []

    seen: dict[str, int] = {}
    for i, row in enumerate(rows, 1):
        p = row["principle"]
        if p in seen:
            warnings.append(
                f"duplicate principle {p!r} (rows {seen[p]} and {i}) — "
                f"both rows will match the same statement")
        seen[p] = i

    for i, row in enumerate(rows, 1):
        for concept in _ENUM_CONCEPTS:
            labels = parse_gold_cell(row.get(concept, ""))
            if labels is None:
                continue
            unknown = labels - _ALLOWED.get(concept, labels)
            if unknown:
                warnings.append(
                    f"row {i} ({row['principle']}) {concept}: "
                    f"unknown literal(s) {sorted(unknown)}")
            if concept in _SINGLE_CONCEPTS and len(labels) > 1:
                warnings.append(
                    f"row {i} ({row['principle']}) {concept}: {len(labels)} "
                    f"values, but the metamodel allows one — only a single "
                    f"value is reachable")
        for concept in _BOOL_CONCEPTS:
            v = row.get(concept, "").strip().upper()
            if v not in {"TRUE", "FALSE", _UNCERTAIN, ""}:
                warnings.append(
                    f"row {i} ({row['principle']}) {concept}: expected "
                    f"TRUE/FALSE/UNCERTAIN, found {v!r}")

    if warnings:
        print(f"GOLD STANDARD WARNINGS ({path}):")
        for w in warnings:
            print(f"  ! {w}")
        print()


# =============================================================================
# PIPELINE OUTPUT LOADER
# =============================================================================

def load_pipeline_output(repo_path: Path, law: str) -> dict[str, dict]:
    """
    Load all stored statements for a law from ModelRepository.

    Returns {article_ref: {column_name: value}} including both
    denormalised columns and the parsed statement_json.
    """
    if not repo_path.exists():
        print(f"ERROR: Repository not found: {repo_path}")
        sys.exit(1)

    conn = sqlite3.connect(str(repo_path))
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        "SELECT * FROM statements WHERE law=? ORDER BY article_ref",
        (law.upper(),),
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print(f"ERROR: No statements found for law={law} in {repo_path}.\n"
              f"       Run the pipeline first: "
              f"python run_pipeline.py --input {law}=laws/{law.lower()}.pdf")
        sys.exit(1)

    result: dict[str, dict] = {}
    for row in rows:
        entry = dict(row)
        for col in ["legal_basis_types", "right_types", "constraint_types",
                    "purpose_categories", "processing_actions",
                    "sensitivity_levels", "transfer_mechanisms"]:
            try:
                entry[col] = set(json.loads(entry.get(col, "[]")))
            except (json.JSONDecodeError, TypeError):
                entry[col] = set()
        try:
            entry["_statement"] = json.loads(entry.get("statement_json", "{}"))
        except (json.JSONDecodeError, TypeError):
            entry["_statement"] = {}
        result[entry["article_ref"]] = entry

    return result


# =============================================================================
# ARTICLE REF MATCHING
# =============================================================================

def _leading_token(article_ref: str) -> str:
    """First whitespace-delimited token of an article_ref, e.g. '4.10'."""
    return article_ref.strip().split()[0] if article_ref.strip() else ""


def _is_principle_heading(article_ref: str) -> bool:
    """True for Schedule 1 headings like '4.6 Principle 6 — Accuracy'."""
    parts = article_ref.strip().split()
    return len(parts) > 1 and parts[1].lower().startswith("principle")


def match_article(
    principle: str,
    pipeline: dict[str, dict],
    explicit_ref: str = "",
) -> tuple[Optional[dict], str]:
    """
    Find the pipeline statement for a gold principle.

    Matching is anchored on the leading article-number token, so "4.1" never
    matches "4.10" or "6.1 For the purposes of clause 4.3 ...".  When several
    statements share the token, a Schedule 1 principle heading wins.  Genuine
    ambiguity returns a diagnostic instead of a silent guess.

    Returns (statement_or_None, matched_ref_or_status).
    """
    if explicit_ref:
        if explicit_ref in pipeline:
            return pipeline[explicit_ref], explicit_ref
        return None, f"NO MATCH (article_ref {explicit_ref!r} not in repo)"

    token = principle.strip()
    candidates = [(ref, stmt) for ref, stmt in pipeline.items()
                  if _leading_token(ref) == token]

    if not candidates:
        return None, "NO MATCH"
    if len(candidates) == 1:
        return candidates[0][1], candidates[0][0]

    headings = [c for c in candidates if _is_principle_heading(c[0])]
    if len(headings) == 1:
        return headings[0][1], headings[0][0]

    pool = headings or candidates
    refs = ", ".join(sorted(r for r, _ in pool))
    return None, f"AMBIGUOUS ({refs})"


# =============================================================================
# EVALUATION ENGINE
# =============================================================================

def _get_actor_role(statement: dict) -> set[str]:
    """Extract actor.role from the full statement JSON — not denormalised."""
    actor = statement.get("_statement", {}).get("actor", {}) or {}
    role = actor.get("role", "")
    return {role} if role and role != "_Unset" else set()


def predicted_labels(concept: str, col: Optional[str],
                     stmt: Optional[dict]) -> set[str]:
    """Labels the pipeline produced for one concept of one statement."""
    if stmt is None:
        return set()
    if concept == "Actor_role":
        return _get_actor_role(stmt)
    values = stmt.get(col, set()) if col else set()
    return {v for v in values if v and v != "_Unset"}


def evaluate_enum_concept(
    concept:   str,
    col:       Optional[str],
    gold_rows: list[dict],
    pipeline:  dict[str, dict],
) -> ConceptResult:
    """
    Evaluate one enum-valued concept across all gold rows, scoring per label.

    Every predicted label that is not in the gold set costs precision, so
    over-extraction is penalised; every gold label the pipeline missed costs
    recall.  A row where gold and prediction are both empty counts as a true
    negative and contributes to neither.
    """
    result = ConceptResult(concept=concept)

    for row in gold_rows:
        gold = parse_gold_cell(row.get(concept, ""))
        if gold is None:                       # UNCERTAIN
            result.skipped += 1
            continue

        stmt, _ = match_article(row["principle"], pipeline,
                                row.get("article_ref", ""))
        predicted = predicted_labels(concept, col, stmt)

        if not gold and not predicted:
            result.tn += 1
            continue

        result.tp += len(predicted & gold)
        result.fp += len(predicted - gold)
        result.fn += len(gold - predicted)

    return result


def evaluate_bool_concept(
    concept:   str,
    col:       str,
    gold_rows: list[dict],
    pipeline:  dict[str, dict],
) -> ConceptResult:
    """Evaluate one boolean concept (present/absent) across all gold rows."""
    result = ConceptResult(concept=concept)

    for row in gold_rows:
        gold_value = row.get(concept, "").strip().upper()
        if gold_value == _UNCERTAIN:
            result.skipped += 1
            continue

        stmt, _ = match_article(row["principle"], pipeline,
                                row.get("article_ref", ""))
        predicted_present = bool(stmt.get(col, 0)) if stmt else False
        gold_present = gold_value == "TRUE"

        if gold_present and predicted_present:
            result.tp += 1
        elif gold_present:
            result.fn += 1
        elif predicted_present:
            result.fp += 1
        else:
            result.tn += 1

    return result


def majority_label(concept: str, gold_rows: list[dict]) -> frozenset[str]:
    """
    The most frequent gold label-set for one enum concept.

    This is the prediction a trivial model would make if it ignored the text
    entirely and always answered with the commonest annotation.
    """
    counts: dict[frozenset[str], int] = {}
    for row in gold_rows:
        labels = parse_gold_cell(row.get(concept, ""))
        if labels is None:
            continue
        key = frozenset(labels)
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return frozenset()
    return max(counts.items(), key=lambda kv: (kv[1], len(kv[0])))[0]


def majority_bool(concept: str, gold_rows: list[dict]) -> bool:
    """TRUE/FALSE majority for one boolean concept."""
    n_true = n_false = 0
    for row in gold_rows:
        v = row.get(concept, "").strip().upper()
        if v == "TRUE":
            n_true += 1
        elif v == "FALSE":
            n_false += 1
    return n_true > n_false


def run_majority_baseline(gold_rows: list[dict], law: str,
                          gold_path: Path) -> EvaluationReport:
    """
    Score the majority-class baseline against the same gold standard.

    Several gold columns are heavily skewed (for PIPEDA, Actor_role is
    DataController in every row), so a constant answer can score highly.
    Reporting this baseline shows how much of a pipeline's score reflects
    extraction rather than the annotator's default conventions.
    """
    report = EvaluationReport(
        law          = law,
        gold_path    = str(gold_path),
        repo_path    = "(majority-class baseline)",
        n_principles = len(gold_rows),
    )

    for concept in _ENUM_CONCEPTS:
        predicted = set(majority_label(concept, gold_rows))
        result = ConceptResult(concept=concept)
        for row in gold_rows:
            gold = parse_gold_cell(row.get(concept, ""))
            if gold is None:
                result.skipped += 1
                continue
            if not gold and not predicted:
                result.tn += 1
                continue
            result.tp += len(predicted & gold)
            result.fp += len(predicted - gold)
            result.fn += len(gold - predicted)
        report.results.append(result)

    for concept in _BOOL_CONCEPTS:
        predicted_present = majority_bool(concept, gold_rows)
        result = ConceptResult(concept=concept)
        for row in gold_rows:
            v = row.get(concept, "").strip().upper()
            if v == _UNCERTAIN:
                result.skipped += 1
                continue
            gold_present = v == "TRUE"
            if gold_present and predicted_present:
                result.tp += 1
            elif gold_present:
                result.fn += 1
            elif predicted_present:
                result.fp += 1
            else:
                result.tn += 1
        report.results.append(result)

    return report


def print_baseline_report(report: EvaluationReport,
                          gold_rows: list[dict]) -> None:
    """Print the majority-class baseline table and the constant it predicts."""
    sep = "=" * 78
    print(sep)
    print("MAJORITY-CLASS BASELINE (ignores the source text entirely)")
    print(sep)
    for concept in _ENUM_CONCEPTS:
        labels = majority_label(concept, gold_rows)
        print(f"  {concept:<28} always predicts: "
              f"{' | '.join(sorted(labels)) if labels else 'ABSENT'}")
    for concept in _BOOL_CONCEPTS:
        print(f"  {concept:<28} always predicts: "
              f"{majority_bool(concept, gold_rows)}")
    print("-" * 78)
    print(f"{'Concept':<28} {'P':>7} {'R':>7} {'F1':>7} "
          f"{'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} {'Support':>8}")
    print("-" * 78)
    for r in report.results:
        print(f"{r.concept:<28} "
              f"{_fmt_pct(r.precision):>7} {_fmt_pct(r.recall):>7} "
              f"{_fmt_pct(r.f1):>7} "
              f"{r.tp:>4} {r.fp:>4} {r.fn:>4} {r.tn:>4} {r.support:>8}")
    print("-" * 78)
    print(f"{'MACRO AVERAGE':<28} "
          f"{_fmt_pct(report.macro_precision):>7} "
          f"{_fmt_pct(report.macro_recall):>7} "
          f"{_fmt_pct(report.macro_f1):>7}")
    print(f"{'MICRO AVERAGE':<28} "
          f"{_fmt_pct(report.micro_precision):>7} "
          f"{_fmt_pct(report.micro_recall):>7} "
          f"{_fmt_pct(report.micro_f1):>7}")
    print(sep)


def run_evaluation(
    gold_path: Path,
    repo_path: Path,
    law:       str,
    sheet:     Optional[str] = None,
) -> EvaluationReport:
    """Run the full evaluation and return an EvaluationReport."""
    gold_rows = load_gold_standard(gold_path, sheet)
    pipeline = load_pipeline_output(repo_path, law)

    report = EvaluationReport(
        law          = law,
        gold_path    = str(gold_path),
        repo_path    = str(repo_path),
        n_principles = len(gold_rows),
    )

    for row in gold_rows:
        _, ref = match_article(row["principle"], pipeline,
                               row.get("article_ref", ""))
        report.matches.append(
            (row["principle"], row.get("principle_name", ""), ref))

    for concept, col in _ENUM_CONCEPTS.items():
        report.results.append(
            evaluate_enum_concept(concept, col, gold_rows, pipeline))

    for concept, col in _BOOL_CONCEPTS.items():
        report.results.append(
            evaluate_bool_concept(concept, col, gold_rows, pipeline))

    return report


# =============================================================================
# OUTPUT FORMATTERS
# =============================================================================

def _fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def print_match_report(report: EvaluationReport) -> None:
    """Print which pipeline statement each gold row was scored against."""
    print("ARTICLE MATCHING")
    print("-" * 78)
    for principle, name, ref in report.matches:
        flag = "  " if ref not in ("NO MATCH",) and not ref.startswith(
            ("NO MATCH", "AMBIGUOUS")) else "!!"
        print(f"{flag} {principle:>6}  {name[:26]:<28} -> {ref}")
    if report.n_unmatched:
        print(f"\n{report.n_unmatched} gold row(s) unmatched — these score as "
              f"'pipeline extracted nothing'.")
    print()


def print_console_report(report: EvaluationReport) -> None:
    """Print a formatted console report."""
    sep = "=" * 78
    print(sep)
    print(f"EVALUATION REPORT — {report.law}")
    print(f"Gold standard : {report.gold_path}")
    print(f"Repository    : {report.repo_path}")
    print(f"Principles    : {report.n_principles}")
    print(f"Scoring       : per-label (set) for enum concepts")
    print(sep)
    print_match_report(report)
    print(f"{'Concept':<28} {'P':>7} {'R':>7} {'F1':>7} "
          f"{'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} {'Support':>8} {'Skip':>5}")
    print("-" * 78)

    for r in report.results:
        print(f"{r.concept:<28} "
              f"{_fmt_pct(r.precision):>7} "
              f"{_fmt_pct(r.recall):>7} "
              f"{_fmt_pct(r.f1):>7} "
              f"{r.tp:>4} {r.fp:>4} {r.fn:>4} {r.tn:>4} "
              f"{r.support:>8} {r.skipped:>5}")

    print("-" * 78)
    print(f"{'MACRO AVERAGE':<28} "
          f"{_fmt_pct(report.macro_precision):>7} "
          f"{_fmt_pct(report.macro_recall):>7} "
          f"{_fmt_pct(report.macro_f1):>7}")
    print(f"{'MICRO AVERAGE':<28} "
          f"{_fmt_pct(report.micro_precision):>7} "
          f"{_fmt_pct(report.micro_recall):>7} "
          f"{_fmt_pct(report.micro_f1):>7}")
    print(f"{'MEAN PER-CONCEPT F1':<28} {'':>7} {'':>7} "
          f"{_fmt_pct(report.mean_f1):>7}")
    print(sep)


def print_markdown_table(report: EvaluationReport) -> None:
    """Print a GitHub/LaTeX-friendly Markdown table."""
    print(f"## Evaluation Results — {report.law}\n")
    print(f"Gold standard: `{report.gold_path}` | "
          f"Principles: {report.n_principles} | Scoring: per-label\n")
    print("| Concept | Precision | Recall | F1 | TP | FP | FN | Support |")
    print("|---|---|---|---|---|---|---|---|")
    for r in report.results:
        print(f"| {r.concept} "
              f"| {_fmt_pct(r.precision)} "
              f"| {_fmt_pct(r.recall)} "
              f"| {_fmt_pct(r.f1)} "
              f"| {r.tp} | {r.fp} | {r.fn} | {r.support} |")
    print(f"| **Macro Average** "
          f"| **{_fmt_pct(report.macro_precision)}** "
          f"| **{_fmt_pct(report.macro_recall)}** "
          f"| **{_fmt_pct(report.macro_f1)}** | | | | |")
    print(f"| **Micro Average** "
          f"| **{_fmt_pct(report.micro_precision)}** "
          f"| **{_fmt_pct(report.micro_recall)}** "
          f"| **{_fmt_pct(report.micro_f1)}** | | | | |")
    if report.n_unmatched:
        print(f"\n> {report.n_unmatched} gold row(s) had no matching "
              f"pipeline statement.")


def save_json_report(report: EvaluationReport, out_path: Path) -> None:
    """Save full results as JSON for downstream processing."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "law":             report.law,
        "gold_path":       report.gold_path,
        "repo_path":       report.repo_path,
        "n_principles":    report.n_principles,
        "scoring":         "per-label",
        "macro_precision": report.macro_precision,
        "macro_recall":    report.macro_recall,
        "macro_f1":        report.macro_f1,
        "mean_f1":         report.mean_f1,
        "micro_precision": report.micro_precision,
        "micro_recall":    report.micro_recall,
        "micro_f1":        report.micro_f1,
        "matches": [
            {"principle": p, "principle_name": n, "matched_article_ref": r}
            for p, n, r in report.matches
        ],
        "concepts": [
            {
                "concept":   r.concept,
                "precision": r.precision,
                "recall":    r.recall,
                "f1":        r.f1,
                "tp":        r.tp,
                "fp":        r.fp,
                "fn":        r.fn,
                "tn":        r.tn,
                "support":   r.support,
                "skipped":   r.skipped,
            }
            for r in report.results
        ],
    }
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\nJSON results written to {out_path}")


# =============================================================================
# GOLD STANDARD TEMPLATE GENERATOR
# =============================================================================

def generate_template(out_path: Path) -> None:
    """
    Write an empty gold standard CSV template with the correct headers
    and the 10 PIPEDA principle rows pre-filled.
    """
    principles = [
        ("4.1",  "Accountability"),
        ("4.2",  "Identifying Purposes"),
        ("4.3",  "Consent"),
        ("4.4",  "Limiting Collection"),
        ("4.5",  "Limiting Use, Disclosure and Retention"),
        ("4.6",  "Accuracy"),
        ("4.7",  "Safeguards"),
        ("4.8",  "Openness"),
        ("4.9",  "Individual Access"),
        ("4.10", "Challenging Compliance"),
    ]

    headers = (["principle", "principle_name"]
               + list(_ENUM_CONCEPTS.keys())
               + list(_BOOL_CONCEPTS.keys())
               + ["notes"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for ref, name in principles:
            writer.writerow([ref, name] + [""] * (len(headers) - 2))

    print(f"Template written to {out_path}")
    print("Fill in each cell using: ABSENT | UNCERTAIN | <literal> "
          "[| <literal> ...]")
    print("Set-valued (may hold several literals separated by '|'): "
          "Purpose_category, Right_type, Constraint_type")
    print("Single-valued (one literal only): LegalBasis_type, "
          "ProcessingActivity_action, Actor_role")
    print("Allowed values per concept:")
    for concept in _ENUM_CONCEPTS:
        allowed = " | ".join(sorted(_ALLOWED[concept]))
        print(f"  {concept:<26}: {allowed}")
    for concept in _BOOL_CONCEPTS:
        print(f"  {concept:<26}: TRUE | FALSE | UNCERTAIN")


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evaluate.py",
        description="Precision/Recall/F1 evaluation of privacy extraction pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Generate empty template:\n"
            "  python evaluate.py --template evaluation/pipeda_gold_standard.csv\n\n"
            "Run evaluation (console output):\n"
            "  python evaluate.py \\\n"
            "    --gold evaluation/pipeda_gold_standard.xlsx \\\n"
            "    --repo data/model_repo.db \\\n"
            "    --law  PIPEDA\n\n"
            "Run evaluation (Markdown table for paper):\n"
            "  python evaluate.py \\\n"
            "    --gold evaluation/pipeda_gold_standard.xlsx \\\n"
            "    --repo data/model_repo.db \\\n"
            "    --law  PIPEDA \\\n"
            "    --format markdown\n\n"
            "Save JSON results:\n"
            "  python evaluate.py \\\n"
            "    --gold evaluation/pipeda_gold_standard.xlsx \\\n"
            "    --repo data/model_repo.db \\\n"
            "    --law  PIPEDA \\\n"
            "    --out  evaluation/results.json\n"
        ),
    )
    p.add_argument("--gold", metavar="PATH",
                   help="Path to gold standard .xlsx or .csv file.")
    p.add_argument("--sheet", default=None, metavar="NAME",
                   help="Worksheet name for .xlsx gold standards "
                        "(default: first sheet).")
    p.add_argument("--repo", default="data/model_repo.db", metavar="PATH",
                   help="Path to ModelRepository SQLite file "
                        "(default: data/model_repo.db).")
    p.add_argument("--law", default="PIPEDA", metavar="LAW",
                   help="Law name to evaluate (default: PIPEDA).")
    p.add_argument("--out", default=None, metavar="PATH",
                   help="Save full results as JSON to this path.")
    p.add_argument("--format", choices=["console", "markdown"],
                   default="console",
                   help="Output format (default: console).")
    p.add_argument("--baseline", action="store_true",
                   help="Also report the majority-class baseline computed "
                        "from the gold standard (a constant answer per "
                        "concept, ignoring the source text).")
    p.add_argument("--strict-match", action="store_true",
                   help="Exit with an error if any gold row cannot be matched "
                        "to exactly one pipeline statement.")
    p.add_argument("--template", metavar="PATH",
                   help="Generate an empty gold standard CSV template "
                        "at this path and exit.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.template:
        generate_template(Path(args.template))
        return

    if not args.gold:
        print("ERROR: --gold is required. Use --template to generate an "
              "empty template.")
        sys.exit(1)

    gold_path = Path(args.gold)
    repo_path = Path(args.repo)

    if not gold_path.exists():
        print(f"ERROR: Gold standard file not found: {gold_path}")
        print(f"       Generate a template: "
              f"python evaluate.py --template {gold_path}")
        sys.exit(1)

    report = run_evaluation(gold_path, repo_path, args.law, args.sheet)

    if args.strict_match and report.n_unmatched:
        print_match_report(report)
        print(f"ERROR: {report.n_unmatched} gold row(s) unmatched "
              f"(--strict-match). Add an 'article_ref' column to pin them.")
        sys.exit(1)

    if args.format == "markdown":
        print_markdown_table(report)
    else:
        print_console_report(report)

    if args.baseline:
        gold_rows = load_gold_standard(gold_path, args.sheet)
        baseline = run_majority_baseline(gold_rows, args.law, gold_path)
        print()
        print_baseline_report(baseline, gold_rows)
        print(f"Pipeline macro F1 {_fmt_pct(report.macro_f1)}  vs  "
              f"baseline macro F1 {_fmt_pct(baseline.macro_f1)}")

    if args.out:
        save_json_report(report, Path(args.out))


if __name__ == "__main__":
    main()
