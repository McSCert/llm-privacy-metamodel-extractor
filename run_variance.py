#!/usr/bin/env python3
"""
run_variance.py — measure the noise floor of the extraction pipeline.

Runs the SAME configuration n times into separate repositories, scores each
with evaluate.py, and reports:

  1. per-concept F1 mean / min / max / range  (the noise floor)
  2. headline macro / micro / mean-F1 spread
  3. article-level agreement: how many articles produced an identical
     statement in every run, by statement_id fingerprint
  4. per-concept label stability: which articles changed their extracted
     labels between runs, and what they changed to

Why: temperature=0.0 with a fixed seed does NOT make this pipeline
reproducible (verified: 11+ of 27 articles differed between two identical
runs).  Any claimed improvement must therefore exceed the range measured
here before it counts as a result.

Usage
-----
    # 3 runs of the current configuration, then the report
    python run_variance.py --n 3 --tag a2 \
        --gold evaluation/pipeda_gold_standard.xlsx \
        --sheet "PIPEDA Gold Standard"

    # score DBs that already exist, without re-running the pipeline
    python run_variance.py --tag a2 --report-only \
        --gold evaluation/pipeda_gold_standard.xlsx \
        --sheet "PIPEDA Gold Standard"

    # a different backend / model
    python run_variance.py --n 5 --tag local \
        --backend local --local-model mistral-nemo:latest \
        --gold evaluation/pipeda_gold_standard.xlsx \
        --sheet "PIPEDA Gold Standard"

Artifacts are written to data/model_repo.<tag>_run<i>.db and
evaluation/variance_<tag>_run<i>.json, so a later --report-only re-reads
them without spending tokens.  The chunk store is never rebuilt: retrieval
is held fixed so that any difference is attributable to the model.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import subprocess
import sys
from pathlib import Path


# ── running ───────────────────────────────────────────────────────────────────

def run_pipeline(law: str, pdf: str, repo: Path, backend: str,
                 model: str, local_model: str) -> bool:
    """One pipeline run into its own repository. Returns True on success."""
    cmd = [sys.executable, "run_pipeline.py",
           "--input", f"{law}={pdf}",
           "--backend", backend,
           "--repo", str(repo)]
    if backend == "openai":
        cmd += ["--model", model]
    else:
        cmd += ["--local-model", local_model]

    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  !! pipeline failed (exit {result.returncode})")
        print("  " + "\n  ".join(result.stderr.strip().splitlines()[-8:]))
        return False
    return True


def score(repo: Path, gold: str, sheet: str, law: str, out: Path) -> dict:
    """Score one repository with evaluate.py and return its JSON results."""
    cmd = [sys.executable, "evaluate.py",
           "--gold", gold, "--repo", str(repo), "--law", law,
           "--out", str(out)]
    if sheet:
        cmd += ["--sheet", sheet]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not out.exists():
        print(f"  !! evaluate.py failed for {repo}")
        print("  " + "\n  ".join((result.stderr or result.stdout)
                                 .strip().splitlines()[-8:]))
        return {}
    return json.loads(out.read_text())


# ── repository inspection ─────────────────────────────────────────────────────

_LABEL_COLUMNS = [
    ("Purpose", "purpose_categories"),
    ("Right", "right_types"),
    ("Constraint", "constraint_types"),
    ("LegalBasis", "legal_basis_types"),
    ("Action", "processing_actions"),
]


def load_statements(repo: Path, law: str) -> dict[str, dict]:
    """{article_ref: {statement_id, labels per concept, booleans}}."""
    if not repo.exists():
        return {}
    conn = sqlite3.connect(str(repo))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM statements WHERE law=?", (law.upper(),)).fetchall()
    conn.close()

    out: dict[str, dict] = {}
    for row in rows:
        entry = {"statement_id": row["statement_id"]}
        for name, col in _LABEL_COLUMNS:
            try:
                entry[name] = tuple(sorted(json.loads(row[col] or "[]")))
            except (json.JSONDecodeError, TypeError):
                entry[name] = ()
        for name, col in (("Retention", "has_retention"),
                          ("Transfer", "has_transfer"),
                          ("Withdrawal", "has_consent_withdrawal")):
            entry[name] = bool(row[col])
        out[row["article_ref"]] = entry
    return out


def _short(ref: str) -> str:
    return ref.split()[0] if ref.strip() else ref


# ── reporting ─────────────────────────────────────────────────────────────────

def _fmt(v: float) -> str:
    return f"{v * 100:.1f}"


def report_scores(results: list[dict]) -> None:
    """Per-concept and headline F1 mean / min / max / range across runs."""
    n = len(results)
    print("=" * 84)
    print(f"SCORE VARIANCE ACROSS {n} IDENTICAL RUNS")
    print("=" * 84)
    print(f"{'Concept':<28} {'mean':>7} {'min':>7} {'max':>7} "
          f"{'RANGE':>7} {'support':>8}")
    print("-" * 84)

    concepts = [c["concept"] for c in results[0]["concepts"]]
    ranges: dict[str, float] = {}
    for concept in concepts:
        f1s = [next(c["f1"] for c in r["concepts"] if c["concept"] == concept)
               for r in results]
        support = next(c["support"] for c in results[0]["concepts"]
                       if c["concept"] == concept)
        spread = max(f1s) - min(f1s)
        ranges[concept] = spread
        print(f"{concept:<28} {_fmt(statistics.mean(f1s)):>7} "
              f"{_fmt(min(f1s)):>7} {_fmt(max(f1s)):>7} "
              f"{_fmt(spread):>7} {support:>8}")

    print("-" * 84)
    for label, key in (("MACRO F1", "macro_f1"),
                       ("MICRO F1", "micro_f1"),
                       ("MEAN PER-CONCEPT F1", "mean_f1")):
        vals = [r[key] for r in results]
        print(f"{label:<28} {_fmt(statistics.mean(vals)):>7} "
              f"{_fmt(min(vals)):>7} {_fmt(max(vals)):>7} "
              f"{_fmt(max(vals) - min(vals)):>7}")
    print("=" * 84)

    worst = max(ranges.items(), key=lambda kv: kv[1])
    print(f"\nNOISE FLOOR: the widest per-concept F1 range is "
          f"{_fmt(worst[1])} points ({worst[0]}).")
    print("An improvement smaller than this is not distinguishable from "
          "run-to-run variation.\n")


def report_agreement(runs: list[dict[str, dict]]) -> None:
    """Article-level and per-concept stability across runs."""
    if not all(runs):
        print("(agreement skipped — a repository was missing)")
        return

    common = set(runs[0])
    for r in runs[1:]:
        common &= set(r)

    identical = [a for a in common
                 if len({r[a]["statement_id"] for r in runs}) == 1]
    print("=" * 84)
    print("ARTICLE-LEVEL AGREEMENT")
    print("=" * 84)
    print(f"Articles present in every run : {len(common)}")
    print(f"Byte-identical statements     : {len(identical)} "
          f"({len(identical) / len(common) * 100:.0f}%)")
    print("Note: statement_id hashes only the first 64 chars of the "
          "statement JSON,\n      so this is an UPPER bound on agreement.\n")

    print(f"{'Concept':<14} {'stable':>8} {'unstable':>9}   unstable articles")
    print("-" * 84)
    for name, _ in _LABEL_COLUMNS:
        unstable = sorted(a for a in common
                          if len({r[a][name] for r in runs}) > 1)
        stable = len(common) - len(unstable)
        shown = ", ".join(_short(a) for a in unstable[:8])
        if len(unstable) > 8:
            shown += f", +{len(unstable) - 8} more"
        print(f"{name:<14} {stable:>8} {len(unstable):>9}   {shown}")
    for name in ("Retention", "Transfer", "Withdrawal"):
        unstable = sorted(a for a in common
                          if len({r[a][name] for r in runs}) > 1)
        shown = ", ".join(_short(a) for a in unstable[:8])
        print(f"{name:<14} {len(common) - len(unstable):>8} "
              f"{len(unstable):>9}   {shown}")
    print("=" * 84)


def report_flips(runs: list[dict[str, dict]], principles_only: bool) -> None:
    """Show what each unstable article actually produced, run by run."""
    if not all(runs):
        return
    common = set(runs[0])
    for r in runs[1:]:
        common &= set(r)
    if principles_only:
        common = {a for a in common if "Principle" in a}

    names = [n for n, _ in _LABEL_COLUMNS] + ["Retention", "Transfer",
                                              "Withdrawal"]
    flips = [(a, n) for a in sorted(common) for n in names
             if len({r[a][n] for r in runs}) > 1]
    if not flips:
        print("No label flips among the reported articles.\n")
        return

    print("LABEL FLIPS" + (" (gold principles only)" if principles_only else ""))
    print("-" * 84)
    for article, concept in flips:
        values = " | ".join(
            ("+".join(r[article][concept])
             if isinstance(r[article][concept], tuple)
             else str(r[article][concept])) or "(none)"
            for r in runs)
        print(f"  {_short(article):>6}  {concept:<12} {values}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        prog="run_variance.py",
        description="Measure the pipeline's run-to-run noise floor.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=3,
                   help="Number of identical runs (default 3).")
    p.add_argument("--tag", default="var",
                   help="Name for this batch; used in DB and JSON filenames.")
    p.add_argument("--law", default="PIPEDA")
    p.add_argument("--pdf", default="laws/pipeda.pdf")
    p.add_argument("--backend", default="openai", choices=["openai", "local"])
    p.add_argument("--model", default="gpt-4o",
                   help="OpenAI model. Prefer a dated ID (gpt-4o-2024-11-20) "
                        "so the served weights cannot change between runs.")
    p.add_argument("--local-model", default="mistral-nemo:latest")
    p.add_argument("--gold", required=True)
    p.add_argument("--sheet", default=None,
                   help="Worksheet name for an .xlsx gold standard.")
    p.add_argument("--report-only", action="store_true",
                   help="Skip the pipeline; score existing DBs for this tag.")
    p.add_argument("--all-articles", action="store_true",
                   help="Show label flips for every article, not just the "
                        "gold principles.")
    args = p.parse_args()

    if not Path("run_pipeline.py").exists():
        print("Run this from the repo root.")
        return 1

    Path("data").mkdir(exist_ok=True)
    Path("evaluation").mkdir(exist_ok=True)

    repos = [Path(f"data/model_repo.{args.tag}_run{i}.db")
             for i in range(1, args.n + 1)]
    jsons = [Path(f"evaluation/variance_{args.tag}_run{i}.json")
             for i in range(1, args.n + 1)]

    if args.report_only:
        repos = [r for r in repos if r.exists()]
        jsons = jsons[:len(repos)]
        if len(repos) < 2:
            print(f"Need at least 2 existing DBs for tag {args.tag!r}; "
                  f"found {len(repos)}.")
            return 1
        print(f"Reporting on {len(repos)} existing repositories.\n")
    else:
        print(f"Running {args.n}x {args.backend} "
              f"({args.model if args.backend == 'openai' else args.local_model}) "
              f"on {args.law}.")
        print("The chunk store is NOT rebuilt, so retrieval is identical "
              "across runs.\n")
        for i, repo in enumerate(repos, 1):
            if repo.exists():
                repo.unlink()          # a stale DB would silently merge runs
            print(f"[run {i}/{args.n}]")
            if not run_pipeline(args.law, args.pdf, repo, args.backend,
                                args.model, args.local_model):
                return 1
        print()

    results = []
    for repo, out in zip(repos, jsons):
        data = score(repo, args.gold, args.sheet, args.law, out)
        if not data:
            return 1
        results.append(data)

    print()
    report_scores(results)
    runs = [load_statements(r, args.law) for r in repos]
    report_agreement(runs)
    report_flips(runs, principles_only=not args.all_articles)

    print(f"Kept: {', '.join(str(r) for r in repos)}")
    print(f"      {', '.join(str(j) for j in jsons)}")
    print(f"Re-report without spending tokens: "
          f"python run_variance.py --tag {args.tag} --report-only "
          f"--gold {args.gold}"
          + (f' --sheet "{args.sheet}"' if args.sheet else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
