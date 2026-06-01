#!/usr/bin/env python3
"""
evaluate.py — Precision / Recall / F1 evaluation of the extraction pipeline.

Compares pipeline output stored in ModelRepository (SQLite) against a
manually-annotated gold standard CSV to produce per-concept P/R/F1 scores.

Usage
-----
    python evaluate.py \
        --gold  evaluation/pipeda_gold_standard.csv \
        --repo  data/model_repo.db \
        --law   PIPEDA \
        --out   evaluation/results.json

    # Markdown table output (for paper):
    python evaluate.py \
        --gold  evaluation/pipeda_gold_standard.csv \
        --repo  data/model_repo.db \
        --law   PIPEDA \
        --format markdown

Gold Standard CSV format
------------------------
Required columns (case-insensitive header):
    principle               — article_ref substring, e.g. "4.1" or "4.1 Principle 1"
    principle_name          — human-readable name (optional, for display)
    LegalBasis_type         — e.g. "Consent" | "LegalObligation" | "ABSENT" | "UNCERTAIN"
    ProcessingActivity_action — e.g. "Collect" | "ABSENT" | "UNCERTAIN"
    Actor_role              — e.g. "DataController" | "ABSENT" | "UNCERTAIN"
    Purpose_category        — e.g. "ServiceProvision" | "ABSENT" | "UNCERTAIN"
    Right_type              — e.g. "Access" | "ABSENT" | "UNCERTAIN"
    Constraint_type         — e.g. "PurposeLimitation" | "ABSENT" | "UNCERTAIN"
    RetentionPolicy_present — TRUE | FALSE | UNCERTAIN
    DataTransfer_present    — TRUE | FALSE | UNCERTAIN
    ConsentWithdrawal_present — TRUE | FALSE | UNCERTAIN

Special values:
    ABSENT    — concept is genuinely not in the text; pipeline should NOT extract it
    UNCERTAIN — annotator was unsure; row excluded from P/R calculation for this concept

Evaluation logic
----------------
For enum-valued concepts (LegalBasis_type, etc.):
    TP = gold has value V AND pipeline extracted V (exact enum match)
    FP = gold is ABSENT AND pipeline extracted something
       + gold has value V AND pipeline extracted W ≠ V
    FN = gold has value V AND pipeline extracted nothing / wrong value

For boolean concepts (RetentionPolicy_present, etc.):
    TP = gold=TRUE  AND pipeline extracted the concept (non-empty)
    TN = gold=FALSE AND pipeline did NOT extract the concept
    FP = gold=FALSE AND pipeline extracted something
    FN = gold=TRUE  AND pipeline did NOT extract the concept

Metrics
-------
    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    F1        = 2 * P * R / (P + R)

Output
------
    Console : formatted table per concept + macro-averaged P/R/F1
    --out   : JSON file with full results for the paper
    --format markdown : GitHub/LaTeX-friendly table
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# =============================================================================
# CONSTANTS
# =============================================================================

# Concepts evaluated and how they map to repository columns
_ENUM_CONCEPTS: dict[str, str] = {
    "LegalBasis_type":          "legal_basis_types",
    "ProcessingActivity_action":"processing_actions",
    "Actor_role":               None,   # not denormalised — read from JSON
    "Purpose_category":         "purpose_categories",
    "Right_type":               "right_types",
    "Constraint_type":          "constraint_types",
}

_BOOL_CONCEPTS: dict[str, str] = {
    "RetentionPolicy_present":    "has_retention",
    "DataTransfer_present":       "has_transfer",
    "ConsentWithdrawal_present":  "has_consent_withdrawal",
}

_SPECIAL_VALUES = {"ABSENT", "UNCERTAIN", ""}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ConceptResult:
    concept:   str
    tp:        int = 0
    fp:        int = 0
    fn:        int = 0
    tn:        int = 0
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
        """Number of gold-standard positives (TP + FN)."""
        return self.tp + self.fn


@dataclass
class EvaluationReport:
    law:            str
    gold_path:      str
    repo_path:      str
    n_principles:   int
    results:        list[ConceptResult] = field(default_factory=list)

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


# =============================================================================
# GOLD STANDARD LOADER
# =============================================================================

def load_gold_standard(csv_path: Path) -> list[dict]:
    """
    Load and normalise the gold standard CSV.
    Returns a list of dicts with lowercase-stripped keys.
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Normalise header names: strip whitespace, preserve case for values
        for raw_row in reader:
            row = {k.strip(): v.strip() for k, v in raw_row.items()}
            rows.append(row)

    if not rows:
        print(f"ERROR: Gold standard CSV is empty: {csv_path}")
        sys.exit(1)

    # Validate required columns
    required = {"principle"} | set(_ENUM_CONCEPTS.keys()) | set(_BOOL_CONCEPTS.keys())
    missing  = required - set(rows[0].keys())
    if missing:
        print(
            f"ERROR: Gold standard CSV missing columns: {sorted(missing)}\n"
            f"       Found: {sorted(rows[0].keys())}"
        )
        sys.exit(1)

    return rows


# =============================================================================
# PIPELINE OUTPUT LOADER
# =============================================================================

def load_pipeline_output(repo_path: Path, law: str) -> dict[str, dict]:
    """
    Load all stored statements for a law from ModelRepository.

    Returns {article_ref: {column_name: value}} including both
    denormalised columns and the full statement_json.
    """
    if not repo_path.exists():
        print(f"ERROR: Repository not found: {repo_path}")
        sys.exit(1)

    conn = sqlite3.connect(str(repo_path))
    conn.row_factory = sqlite3.Row
    cur  = conn.execute(
        "SELECT * FROM statements WHERE law=? ORDER BY article_ref",
        (law.upper(),),
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print(
            f"ERROR: No statements found for law={law} in {repo_path}.\n"
            f"       Run the pipeline first: "
            f"python run_pipeline.py --input {law}=laws/{law.lower()}.pdf"
        )
        sys.exit(1)

    result: dict[str, dict] = {}
    for row in rows:
        article_ref = row["article_ref"]
        entry = dict(row)
        # Parse JSON columns
        for col in [
            "legal_basis_types", "right_types", "constraint_types",
            "purpose_categories", "processing_actions",
            "sensitivity_levels", "transfer_mechanisms",
        ]:
            try:
                entry[col] = set(json.loads(entry.get(col, "[]")))
            except (json.JSONDecodeError, TypeError):
                entry[col] = set()
        # Parse full statement JSON
        try:
            entry["_statement"] = json.loads(entry.get("statement_json", "{}"))
        except (json.JSONDecodeError, TypeError):
            entry["_statement"] = {}
        result[article_ref] = entry

    return result


# =============================================================================
# ARTICLE REF MATCHING
# =============================================================================

def _match_article(
    principle: str,
    pipeline_statements: dict[str, dict],
) -> Optional[dict]:
    """
    Find the pipeline statement that best matches a gold standard principle ref.

    Matching strategy (in order):
    1. Exact match
    2. Pipeline article_ref contains the principle substring
    3. Principle contains the pipeline article_ref

    Returns the matched statement dict or None.
    """
    principle = principle.strip()

    # Exact match
    if principle in pipeline_statements:
        return pipeline_statements[principle]

    # Substring match — find the best (longest) matching article_ref
    candidates = [
        (ref, stmt) for ref, stmt in pipeline_statements.items()
        if principle in ref or ref.split()[0] in principle
    ]
    if candidates:
        # Pick the most specific match (longest article_ref)
        best_ref = max(candidates, key=lambda x: len(x[0]))
        return best_ref[1]

    return None


# =============================================================================
# EVALUATION ENGINE
# =============================================================================

def _get_actor_role(statement: dict) -> set[str]:
    """Extract actor.role from the full statement JSON — not denormalised."""
    actor = statement.get("_statement", {}).get("actor", {})
    if not actor:
        actor = statement.get("_statement", {}).get("actor", {})
    role = actor.get("role", "")
    return {role} if role else set()


def evaluate_enum_concept(
    concept:    str,
    col:        Optional[str],
    gold_rows:  list[dict],
    pipeline:   dict[str, dict],
) -> ConceptResult:
    """
    Evaluate one enum-valued concept across all gold standard rows.
    """
    result = ConceptResult(concept=concept)

    for row in gold_rows:
        gold_value = row.get(concept, "").strip()

        # Skip UNCERTAIN rows
        if gold_value.upper() == "UNCERTAIN":
            result.skipped += 1
            continue

        # Find matching pipeline statement
        stmt = _match_article(row["principle"], pipeline)

        # Get predicted values from pipeline
        if stmt is None:
            predicted: set[str] = set()
        elif concept == "Actor_role":
            predicted = _get_actor_role(stmt)
        else:
            predicted = stmt.get(col, set()) if col else set()

        gold_absent = gold_value.upper() == "ABSENT" or gold_value == ""

        if gold_absent:
            # Gold: concept should NOT be present
            if predicted:
                result.fp += 1   # pipeline hallucinated something
            else:
                result.tn += 1   # correctly predicted absent
        else:
            # Gold: concept should be present with value gold_value
            if gold_value in predicted:
                result.tp += 1   # correct enum value extracted
            elif predicted:
                result.fp += 1   # wrong value extracted
                result.fn += 1   # correct value missed
            else:
                result.fn += 1   # nothing extracted

    return result


def evaluate_bool_concept(
    concept:   str,
    col:       str,
    gold_rows: list[dict],
    pipeline:  dict[str, dict],
) -> ConceptResult:
    """
    Evaluate one boolean concept (present/absent) across all gold standard rows.
    """
    result = ConceptResult(concept=concept)

    for row in gold_rows:
        gold_value = row.get(concept, "").strip().upper()

        if gold_value == "UNCERTAIN":
            result.skipped += 1
            continue

        stmt = _match_article(row["principle"], pipeline)

        # Pipeline boolean: 1 = present, 0 = absent
        if stmt is None:
            predicted_present = False
        else:
            predicted_present = bool(stmt.get(col, 0))

        gold_present = gold_value == "TRUE"

        if gold_present and predicted_present:
            result.tp += 1
        elif gold_present and not predicted_present:
            result.fn += 1
        elif not gold_present and predicted_present:
            result.fp += 1
        else:
            result.tn += 1

    return result


def run_evaluation(
    gold_path: Path,
    repo_path: Path,
    law:       str,
) -> EvaluationReport:
    """
    Run the full evaluation and return an EvaluationReport.
    """
    gold_rows = load_gold_standard(gold_path)
    pipeline  = load_pipeline_output(repo_path, law)

    report = EvaluationReport(
        law          = law,
        gold_path    = str(gold_path),
        repo_path    = str(repo_path),
        n_principles = len(gold_rows),
    )

    # Evaluate enum concepts
    for concept, col in _ENUM_CONCEPTS.items():
        result = evaluate_enum_concept(concept, col, gold_rows, pipeline)
        report.results.append(result)

    # Evaluate boolean concepts
    for concept, col in _BOOL_CONCEPTS.items():
        result = evaluate_bool_concept(concept, col, gold_rows, pipeline)
        report.results.append(result)

    return report


# =============================================================================
# OUTPUT FORMATTERS
# =============================================================================

def _fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def print_console_report(report: EvaluationReport) -> None:
    """Print a formatted console report."""
    sep = "=" * 70
    print(sep)
    print(f"EVALUATION REPORT — {report.law}")
    print(f"Gold standard : {report.gold_path}")
    print(f"Repository    : {report.repo_path}")
    print(f"Principles    : {report.n_principles}")
    print(sep)
    print(
        f"{'Concept':<32} {'P':>7} {'R':>7} {'F1':>7} "
        f"{'TP':>4} {'FP':>4} {'FN':>4} {'Support':>8} {'Skipped':>8}"
    )
    print("-" * 70)

    for r in report.results:
        print(
            f"{r.concept:<32} "
            f"{_fmt_pct(r.precision):>7} "
            f"{_fmt_pct(r.recall):>7} "
            f"{_fmt_pct(r.f1):>7} "
            f"{r.tp:>4} {r.fp:>4} {r.fn:>4} "
            f"{r.support:>8} {r.skipped:>8}"
        )

    print("-" * 70)
    print(
        f"{'MACRO AVERAGE':<32} "
        f"{_fmt_pct(report.macro_precision):>7} "
        f"{_fmt_pct(report.macro_recall):>7} "
        f"{_fmt_pct(report.macro_f1):>7}"
    )
    print(sep)


def print_markdown_table(report: EvaluationReport) -> None:
    """Print a GitHub/LaTeX-friendly Markdown table."""
    print(f"## Evaluation Results — {report.law}\n")
    print(f"Gold standard: `{report.gold_path}` | Principles: {report.n_principles}\n")
    print(
        "| Concept | Precision | Recall | F1 | TP | FP | FN | Support |"
    )
    print(
        "|---|---|---|---|---|---|---|---|"
    )
    for r in report.results:
        print(
            f"| {r.concept} "
            f"| {_fmt_pct(r.precision)} "
            f"| {_fmt_pct(r.recall)} "
            f"| {_fmt_pct(r.f1)} "
            f"| {r.tp} | {r.fp} | {r.fn} | {r.support} |"
        )
    print(
        f"| **Macro Average** "
        f"| **{_fmt_pct(report.macro_precision)}** "
        f"| **{_fmt_pct(report.macro_recall)}** "
        f"| **{_fmt_pct(report.macro_f1)}** "
        f"| | | | |"
    )


def save_json_report(report: EvaluationReport, out_path: Path) -> None:
    """Save full results as JSON for downstream processing."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "law":            report.law,
        "gold_path":      report.gold_path,
        "repo_path":      report.repo_path,
        "n_principles":   report.n_principles,
        "macro_precision": report.macro_precision,
        "macro_recall":    report.macro_recall,
        "macro_f1":        report.macro_f1,
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
        ("4.1", "Accountability"),
        ("4.2", "Identifying Purposes"),
        ("4.3", "Consent"),
        ("4.4", "Limiting Collection"),
        ("4.5", "Limiting Use, Disclosure and Retention"),
        ("4.6", "Accuracy"),
        ("4.7", "Safeguards"),
        ("4.8", "Openness"),
        ("4.9", "Individual Access"),
        ("4.10", "Challenging Compliance"),
    ]

    headers = (
        ["principle", "principle_name"]
        + list(_ENUM_CONCEPTS.keys())
        + list(_BOOL_CONCEPTS.keys())
        + ["notes"]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for ref, name in principles:
            row = [ref, name] + [""] * (len(headers) - 3) + [""]
            writer.writerow(row)

    print(f"Template written to {out_path}")
    print(f"Fill in the empty cells using: ABSENT | UNCERTAIN | <enum value>")
    print(f"Allowed values per concept:")
    print(f"  LegalBasis_type          : Consent | Contract | LegalObligation | LegitimateInterest")
    print(f"  ProcessingActivity_action: Collect | Use | Disclose | Store | Transfer | Share | Delete")
    print(f"  Actor_role               : DataController | DataProcessor | DataSubject | ThirdParty")
    print(f"  Purpose_category         : ServiceProvision | LegalCompliance | Marketing | Research | Security | Analytics")
    print(f"  Right_type               : Access | Rectification | Erasure | Portability | Objection | OptOut")
    print(f"  Constraint_type          : PurposeLimitation | Storage | Security | Accuracy | Transparency")
    print(f"  RetentionPolicy_present  : TRUE | FALSE | UNCERTAIN")
    print(f"  DataTransfer_present     : TRUE | FALSE | UNCERTAIN")
    print(f"  ConsentWithdrawal_present: TRUE | FALSE | UNCERTAIN")


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
            "    --gold evaluation/pipeda_gold_standard.csv \\\n"
            "    --repo data/model_repo.db \\\n"
            "    --law  PIPEDA\n\n"
            "Run evaluation (Markdown table for paper):\n"
            "  python evaluate.py \\\n"
            "    --gold evaluation/pipeda_gold_standard.csv \\\n"
            "    --repo data/model_repo.db \\\n"
            "    --law  PIPEDA \\\n"
            "    --format markdown\n\n"
            "Save JSON results:\n"
            "  python evaluate.py \\\n"
            "    --gold evaluation/pipeda_gold_standard.csv \\\n"
            "    --repo data/model_repo.db \\\n"
            "    --law  PIPEDA \\\n"
            "    --out  evaluation/results.json\n"
        ),
    )
    p.add_argument(
        "--gold", metavar="CSV",
        help="Path to gold standard CSV file.",
    )
    p.add_argument(
        "--repo", default="data/model_repo.db", metavar="PATH",
        help="Path to ModelRepository SQLite file (default: data/model_repo.db).",
    )
    p.add_argument(
        "--law", default="PIPEDA", metavar="LAW",
        help="Law name to evaluate (default: PIPEDA).",
    )
    p.add_argument(
        "--out", default=None, metavar="PATH",
        help="Save full results as JSON to this path.",
    )
    p.add_argument(
        "--format", choices=["console", "markdown"], default="console",
        help="Output format (default: console).",
    )
    p.add_argument(
        "--template", metavar="PATH",
        help="Generate an empty gold standard CSV template at this path and exit.",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    # Template generation mode
    if args.template:
        generate_template(Path(args.template))
        return

    if not args.gold:
        print("ERROR: --gold is required. Use --template to generate an empty template.")
        sys.exit(1)

    gold_path = Path(args.gold)
    repo_path = Path(args.repo)

    if not gold_path.exists():
        print(f"ERROR: Gold standard file not found: {gold_path}")
        print(f"       Generate a template: python evaluate.py --template {gold_path}")
        sys.exit(1)

    report = run_evaluation(gold_path, repo_path, args.law)

    if args.format == "markdown":
        print_markdown_table(report)
    else:
        print_console_report(report)

    if args.out:
        save_json_report(report, Path(args.out))


if __name__ == "__main__":
    main()