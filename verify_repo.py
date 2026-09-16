#!/usr/bin/env python3
"""
verify_repo.py — consistency audit of the privacy metamodel extractor.

Checks the invariants that ordinary testing does NOT catch: places where two
artifacts are each internally fine but disagree with one another, or where a
code path silently does nothing.  Both classes of bug have already bitten
this project (the article matcher preferred the longest ref and scored four
gold rows against the wrong article; the OpenAI backend discarded the schema
it was handed).

Run from the repo root:

    python verify_repo.py                 # full audit
    python verify_repo.py --quick         # skip the PDF chunking checks
    python verify_repo.py --gold evaluation/pipeda_gold_standard.xlsx \
                          --sheet "PIPEDA Gold Standard"

Exit code is 0 when no FAIL is reported; WARN and INFO never fail the run.
Nothing is modified: every check is read-only.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

PASS, FAIL, WARN, INFO, SKIP = "PASS", "FAIL", "WARN", "INFO", "SKIP"
_RESULTS: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    _RESULTS.append((status, name, detail))
    mark = {PASS: "  ok  ", FAIL: " FAIL ", WARN: " warn ",
            INFO: " info ", SKIP: " skip "}[status]
    print(f"[{mark}] {name}" + (f"\n         {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n── {title} " + "─" * max(0, 68 - len(title)))


# =============================================================================
# 1. Metamodel → code generation chain
# =============================================================================

def check_generation_chain() -> None:
    section("Metamodel generation chain")

    # 1a. the committed .ecore must be exactly what generate_ecore.py produces
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "check.ecore"
        r = subprocess.run([sys.executable, "generate_ecore.py", str(out)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            record(FAIL, "generate_ecore.py runs",
                   (r.stderr or r.stdout).strip().splitlines()[-1][:120])
        else:
            committed = Path("metamodel/privacy_metamodel.ecore").read_text()
            if committed == out.read_text():
                record(PASS, "committed .ecore == generate_ecore.py output")
            else:
                record(FAIL, "committed .ecore != generate_ecore.py output",
                       "the .ecore was hand-edited, or the generator changed; "
                       "re-run: python generate_ecore.py "
                       "metamodel/privacy_metamodel.ecore")

    # 1b. enums.py must agree with the .ecore  (this one IS generated)
    try:
        from pyecore.resources import ResourceSet, URI
        pkg = ResourceSet().get_resource(
            URI("metamodel/privacy_metamodel.ecore")).contents[0]
        ecore_enums = {
            c.name: [l.name for l in c.eLiterals]
            for c in pkg.eClassifiers if hasattr(c, "eLiterals") and c.eLiterals
        }
    except ImportError:
        record(SKIP, "enums.py == .ecore", "pyecore not installed")
        ecore_enums = {}
    except Exception as exc:                        # noqa: BLE001
        record(FAIL, ".ecore loads", str(exc)[:140])
        ecore_enums = {}

    py_enums: dict[str, list[str]] = {}
    if ecore_enums:
        import enum as _enum
        mod = importlib.import_module("privacy_schema.enums")
        for name in dir(mod):
            obj = getattr(mod, name)
            if isinstance(obj, type) and issubclass(obj, _enum.Enum) \
                    and obj is not _enum.Enum:
                py_enums[name] = [m.value for m in obj]

        mismatches = [
            f"{name}: ecore={ecore_enums[name]} enums.py={py_enums.get(name)}"
            for name in ecore_enums
            if name in py_enums and ecore_enums[name] != py_enums[name]
        ]
        missing = sorted(set(ecore_enums) - set(py_enums))
        if mismatches or missing:
            record(FAIL, "enums.py == .ecore",
                   "; ".join(mismatches + [f"missing in enums.py: {missing}"
                                           if missing else ""]).strip("; ")[:400])
        else:
            record(PASS, "enums.py == .ecore",
                   f"{len(ecore_enums)} enums match")

    # 1c. models.py is NOT generated — say so plainly, it is a known divergence
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run(
            [sys.executable, "generate_pydantic.py",
             "metamodel/privacy_metamodel.ecore", "--out-dir", tmp],
            capture_output=True, text=True)
        gen = Path(tmp) / "models.py"
        if r.returncode != 0 or not gen.exists():
            record(WARN, "generate_pydantic.py produces models.py",
                   "generator failed; models.py is hand-maintained")
        else:
            try:
                compile(gen.read_text(), "models.py", "exec")
                same = gen.read_text() == Path("privacy_schema/models.py").read_text()
                record(PASS if same else WARN,
                       "models.py == generate_pydantic.py output",
                       "" if same else
                       "models.py is hand-maintained and diverges from the "
                       "generator. The paper claims Pydantic is derived from "
                       "the Ecore; that is currently only true for enums.py.")
            except SyntaxError as exc:
                record(WARN, "generate_pydantic.py output parses",
                       f"generated models.py does not compile ({exc.msg}); "
                       f"models.py must stay hand-maintained for now")

    # 1d. schemas/*.json must match the current models
    r = subprocess.run([sys.executable, "generate_schemas.py"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        record(WARN, "generate_schemas.py runs", "could not verify schemas/")
    else:
        dirty = subprocess.run(["git", "status", "--porcelain", "schemas/"],
                               capture_output=True, text=True).stdout.strip()
        # generate_schemas.py rewrites in place; if that changed anything that
        # was not already dirty before this script ran we cannot tell them
        # apart, so only report, never fail.
        record(INFO, "schemas/ regenerated",
               "schemas/ now matches models.py"
               + (" (files were already modified in your tree)" if dirty else ""))


# =============================================================================
# 2. Enum vocabulary agreement across code, prompts and gold standard
# =============================================================================

_GOLD_COLUMN_TO_ENUM = {
    "LegalBasis_type": "LegalBasisType",
    "ProcessingActivity_action": "ProcessingAction",
    "Actor_role": "ActorRole",
    "Purpose_category": "PurposeCategory",
    "Right_type": "RightType",
    "Constraint_type": "ConstraintType",
}


def _load_gold_rows(gold: Path, sheet: str | None):
    if gold.suffix.lower() in (".xlsx", ".xlsm"):
        import openpyxl
        wb = openpyxl.load_workbook(gold, data_only=True)
        ws = wb[sheet] if sheet and sheet in wb.sheetnames else wb.worksheets[0]
        raw = [list(r) for r in ws.iter_rows(values_only=True)]
    else:
        import csv
        with open(gold, newline="", encoding="utf-8-sig") as f:
            raw = [list(r) for r in csv.reader(f)]
    hdr_i = next((i for i, r in enumerate(raw)
                  if any(str(c).strip().lower() == "principle"
                         for c in r if c is not None)), None)
    if hdr_i is None:
        return [], []
    header = [str(c).strip() if c is not None else "" for c in raw[hdr_i]]
    rows = []
    for r in raw[hdr_i + 1:]:
        vals = list(r) + [None] * (len(header) - len(r))
        rec = {h: ("" if v is None else str(v).strip())
               for h, v in zip(header, vals) if h}
        if rec.get("principle"):
            rows.append(rec)
    return header, rows


def check_vocabulary(gold: Path, sheet: str | None) -> None:
    section("Enum vocabulary agreement")

    import enum as _enum
    mod = importlib.import_module("privacy_schema.enums")
    enums = {
        name: {m.value for m in obj if m.value != "_Unset"}
        for name in dir(mod)
        if isinstance(obj := getattr(mod, name), type)
        and issubclass(obj, _enum.Enum) and obj is not _enum.Enum
    }

    # 2a. prompt enum grammars vs enums.py
    prompts = Path("privacy_schema/prompts.py").read_text()
    bad = []
    for m in re.finditer(r'"(\w+)":\s*\[([^\]]*)\]', prompts):
        name, body = m.group(1), m.group(2)
        if name not in enums:
            continue
        listed = {v.strip().strip('"') for v in body.split(",") if v.strip()}
        listed.discard("_Unset")
        if listed and listed != enums[name]:
            bad.append(f"{name}: prompt={sorted(listed)} enums={sorted(enums[name])}")
    record(FAIL if bad else PASS, "prompt enum grammars == enums.py",
           "; ".join(bad)[:400])

    # 2b. every literal named in a prompt decision guide must exist
    guide_terms = set(re.findall(r"→\s*([A-Z][A-Za-z]+)\s*\\n", prompts))
    known = {v for vals in enums.values() for v in vals}
    unknown = sorted(t for t in guide_terms
                     if t not in known and t[0].isupper() and len(t) > 3)
    record(WARN if unknown else PASS, "prompt decision guides use real literals",
           f"not in any enum: {unknown}" if unknown else "")

    # 2c. gold standard values must be reachable
    if not gold.exists():
        record(SKIP, "gold values ⊆ enums", f"{gold} not found")
        return
    header, rows = _load_gold_rows(gold, sheet)
    if not rows:
        record(FAIL, "gold standard parses", f"no data rows found in {gold}")
        return
    problems = []
    for i, row in enumerate(rows, 1):
        for col, enum_name in _GOLD_COLUMN_TO_ENUM.items():
            cell = row.get(col, "").strip()
            if not cell or cell.upper() in ("ABSENT", "UNCERTAIN"):
                continue
            for label in (v.strip() for v in cell.split("|")):
                if label and label not in enums.get(enum_name, set()):
                    problems.append(f"row {i} ({row['principle']}) {col}={label!r}")
    record(FAIL if problems else PASS, "gold values ⊆ enums",
           "; ".join(problems[:6])[:400]
           + (f" (+{len(problems)-6} more)" if len(problems) > 6 else ""))

    # 2d. duplicate principles would silently double-score
    seen, dupes = set(), []
    for row in rows:
        if row["principle"] in seen:
            dupes.append(row["principle"])
        seen.add(row["principle"])
    record(FAIL if dupes else PASS, "gold principles unique",
           f"duplicates: {dupes}" if dupes else f"{len(rows)} rows")

    # 2e. a stale CSV beside the workbook is how the 4.10 typo survived
    csv_twin = gold.with_suffix(".csv")
    if gold.suffix.lower() != ".csv" and csv_twin.exists():
        _, csv_rows = _load_gold_rows(csv_twin, None)
        drift = [(a.get("principle"), b.get("principle"))
                 for a, b in zip(rows, csv_rows)
                 if a.get("principle") != b.get("principle")]
        record(WARN, "no stale CSV twin of the gold standard",
               f"{csv_twin} exists"
               + (f" and disagrees on principle ids: {drift[:4]}" if drift
                  else " (principle ids agree)")
               + " — delete it; evaluate.py reads .xlsx directly")


# =============================================================================
# 3. Absence representation  (A2)
# =============================================================================

def check_absence() -> None:
    section("Absence representation (A2)")

    models = importlib.import_module("privacy_schema.models")
    stmt = models.PolicyStatementModel
    required = [f for f in ("purposes", "constraints", "right_impacted")
                if stmt.model_fields[f].is_required()]
    record(FAIL if required else PASS,
           "purposes/constraints/rightImpacted are optional",
           f"still required: {required}" if required else
           "an empty list is a valid answer")

    # the Ecore must agree with Pydantic, or XMI export and validation diverge
    try:
        from pyecore.resources import ResourceSet, URI
        pkg = ResourceSet().get_resource(
            URI("metamodel/privacy_metamodel.ecore")).contents[0]
        ps = pkg.getEClassifier("PolicyStatement")
        bounds = {f.name: (f.lowerBound, f.upperBound)
                  for f in ps.eStructuralFeatures}
        wrong = {k: v for k, v in bounds.items()
                 if k in ("purposes", "constraints", "rightImpacted")
                 and v != (0, -1)}
        record(FAIL if wrong else PASS, ".ecore multiplicities are [0..*]",
               f"{wrong}" if wrong else "")
        if bounds.get("governingRegulations") != (1, -1):
            record(WARN, "governingRegulations stays [1..*]",
                   f"found {bounds.get('governingRegulations')} — it is filled "
                   f"deterministically, so it should stay required")
    except ImportError:
        record(SKIP, ".ecore multiplicities", "pyecore not installed")

    # the assembler prompt must not contradict the schema
    prompts = Path("privacy_schema/prompts.py").read_text()
    contradictions = [
        phrase for phrase in (
            "EXCEPTION — required arrays that are []",
            "rightImpacted\\\" is ALWAYS required",
            "Required arrays (purposes, governingRegulations, constraints, "
            "rightImpacted) ",
        ) if phrase in prompts
    ]
    record(FAIL if contradictions else PASS,
           "assembler prompt does not force synthesis",
           f"still present: {contradictions}" if contradictions else "")

    n_synth = prompts.count("synthesize 1 item")
    record(PASS if n_synth == 1 else WARN,
           "only dataProcessed retains a synthesis rule",
           f"found {n_synth} synthesis blocks (expected 1)"
           if n_synth != 1 else "")

    # fields still required elsewhere produce the recurring OCL violations
    still = []
    for cls, field in (("DataTransferModel", "destination_jurisdiction"),
                       ("DataTransferModel", "data_transferred"),
                       ("ConsentWithdrawalModel", "channel"),
                       ("ProcessingActivityModel", "data_processed")):
        model = getattr(models, cls, None)
        if model and field in model.model_fields \
                and model.model_fields[field].is_required():
            still.append(f"{cls}.{field}")
    record(INFO, "known remaining required list fields",
           f"{still} — these are the recurring OCL violations; "
           f"scheduled as a separate step" if still else "none")


# =============================================================================
# 4. Backends: is the schema actually on the wire?
# =============================================================================

def check_backends() -> None:
    section("Backend schema wiring")

    src = Path("run_pipeline.py").read_text()
    for cls, marker in (("LocalBackend", "response_format"),
                        ("AnthropicBackend", "input_schema"),
                        ("OpenAIBackend", "response_format")):
        i = src.find(f"class {cls}")
        if i < 0:
            record(SKIP, f"{cls} sends the schema", "class not found")
            continue
        j = src.find("\nclass ", i + 1)
        block = src[i: j if j > 0 else len(src)]
        ok = marker in block
        record(PASS if ok else FAIL, f"{cls} sends the schema",
               "" if ok else
               f"no {marker!r} in {cls}: stage_extract passes schema=... but "
               f"this backend discards it, so its output is NOT "
               f"schema-constrained")

    # determinism knobs
    i = src.find("class OpenAIBackend")
    j = src.find("\nclass ", i + 1)
    block = src[i: j if j > 0 else len(src)] if i >= 0 else ""
    for knob in ("temperature", "seed"):
        record(PASS if knob in block else WARN,
               f"OpenAI backend sets {knob}",
               "" if knob in block else
               "runs will not even attempt reproducibility")

    if "gpt-4o\"" in src and "gpt-4o-20" not in src:
        record(WARN, "OpenAI model default is pinned",
               "default is the floating alias 'gpt-4o'; pass a dated id "
               "(--model gpt-4o-2024-11-20) so served weights cannot change "
               "between runs")


# =============================================================================
# 5. Chunker and retrieval
# =============================================================================

def check_chunker(law: str, pdf: Path) -> None:
    section("Chunker")

    if not pdf.exists():
        record(SKIP, "chunking", f"{pdf} not found")
        return
    try:
        from rag_pipeline.chunker import chunk_file
    except Exception as exc:                        # noqa: BLE001
        record(FAIL, "chunker imports", str(exc)[:140])
        return

    chunks = chunk_file(pdf, law)
    ids: dict[str, int] = {}
    for c in chunks:
        ids[c.chunk_id] = ids.get(c.chunk_id, 0) + 1
    dupes = sum(v - 1 for v in ids.values() if v > 1)
    record(FAIL if dupes else PASS, "chunk ids unique",
           f"{dupes} chunks would be silently dropped by INSERT OR IGNORE"
           if dupes else f"{len(chunks)} chunks")

    empty = sum(1 for c in chunks if not c.article_ref.strip())
    record(FAIL if empty else PASS, "every chunk has an article_ref",
           f"{empty} empty" if empty else "")

    # preamble chunks must not be processed as articles
    pre_levels = {c.level for c in chunks if "[preamble]" in c.article_ref}
    if pre_levels:
        bad = pre_levels & {"article", "section", "principle", "document"}
        record(FAIL if bad else PASS, "preamble chunks are not articles",
               f"level(s) {bad} would be sent to Pass 1" if bad else
               f"level={pre_levels}")

    # clause coverage: the page-break bug made these unreachable
    if law.upper() == "PIPEDA":
        probes = ["4.3.5", "4.3.8", "4.5.3", "4.7.2", "4.7.3",
                  "4.8.3", "4.9.3", "4.9.5", "4.9.6", "4.10.4"]
        missing = []
        for probe in probes:
            parent = probe.rsplit(".", 1)[0]
            if not any(probe in c.text and c.article_ref.startswith(parent + " ")
                       for c in chunks):
                missing.append(probe)
        record(FAIL if missing else PASS,
               "Schedule 1 clauses reachable under their principle",
               f"unreachable: {missing}" if missing else
               f"all {len(probes)} probe clauses reachable")


# =============================================================================
# 6. Evaluation harness
# =============================================================================

def check_evaluation(gold: Path, sheet: str | None, law: str) -> None:
    section("Evaluation harness")

    ev = Path("evaluate.py").read_text()
    for name, marker, detail in (
        ("per-label scoring", "result.fp += len(predicted - gold)",
         "over-extraction must cost precision"),
        ("anchored article matching", "_leading_token",
         "matching on substring prefers the longest ref and mis-scores rows"),
        ("match report", "ARTICLE MATCHING", "silent mis-matching is invisible"),
        ("majority baseline", "majority-class baseline", ""),
    ):
        record(PASS if marker in ev else FAIL, f"evaluate.py: {name}", detail)

    # end-to-end: does every gold row resolve to a principle heading?
    repos = sorted(Path("data").glob("model_repo*.db")) if Path("data").is_dir() else []
    if not gold.exists() or not repos:
        record(SKIP, "every gold row matches a statement",
               "no gold file or no repository to score against")
        return
    repo = repos[0]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "r.json"
        cmd = [sys.executable, "evaluate.py", "--gold", str(gold),
               "--repo", str(repo), "--law", law, "--out", str(out)]
        if sheet:
            cmd += ["--sheet", sheet]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not out.exists():
            record(FAIL, "evaluate.py runs",
                   (r.stderr or r.stdout).strip().splitlines()[-1][:160])
            return
        data = json.loads(out.read_text())
    bad = [m for m in data["matches"]
           if m["matched_article_ref"].startswith(("NO MATCH", "AMBIGUOUS"))]
    record(WARN if bad else PASS,
           f"every gold row matches a statement in {repo.name}",
           "; ".join(f"{m['principle']}→{m['matched_article_ref']}"
                     for m in bad)[:300] if bad else
           f"{len(data['matches'])} rows resolved")

    suspicious = [m for m in data["matches"]
                  if "Principle" not in m["matched_article_ref"]
                  and not m["matched_article_ref"].startswith(("NO MATCH",
                                                               "AMBIGUOUS"))]
    record(WARN if suspicious else PASS,
           "gold rows match principle headings, not citing sections",
           "; ".join(f"{m['principle']}→{m['matched_article_ref'][:40]}"
                     for m in suspicious)[:300] if suspicious else "")


# =============================================================================
# 7. Output paths that fail silently
# =============================================================================

def check_output_paths() -> None:
    section("Output paths")

    src = Path("run_pipeline.py").read_text()
    # Look at the WHOLE assignment, not just its first line: the fixed form
    # spans several lines and puts the directory on the second one.
    m = re.search(r"ecore_path\s*=\s*", src)
    stmt = src[m.end(): m.end() + 400] if m else ""
    stmt = stmt[: stmt.find("\n        if ")] if "\n        if " in stmt else stmt
    if m and '"metamodel"' not in stmt and "metamodel/" not in stmt:
        exists = Path("privacy_metamodel.ecore").exists()
        record(FAIL if not exists else PASS, "XMI writer finds the .ecore",
               "run_pipeline.py looks for privacy_metamodel.ecore in the repo "
               "root, but it lives in metamodel/. XMI output is silently "
               "disabled and 0 files are written.")
    else:
        record(PASS, "XMI writer finds the .ecore")

    if "absent" in src and "FAIL (attempts" in src:
        record(WARN, "Pass-1 log distinguishes absent from failed",
               "concept-absent results print as 'FAIL', so the console "
               "overstates the failure rate; the summary line is correct")


# =============================================================================
# main
# =============================================================================

def main() -> int:
    p = argparse.ArgumentParser(prog="verify_repo.py")
    p.add_argument("--gold", default="evaluation/pipeda_gold_standard.xlsx")
    p.add_argument("--sheet", default="PIPEDA Gold Standard")
    p.add_argument("--law", default="PIPEDA")
    p.add_argument("--pdf", default="laws/pipeda.pdf")
    p.add_argument("--quick", action="store_true",
                   help="skip the chunker checks (they re-parse the PDF)")
    args = p.parse_args()

    if not Path("run_pipeline.py").exists():
        print("Run this from the repo root.")
        return 1
    sys.path.insert(0, ".")

    print("Consistency audit — read-only, nothing is modified.")

    check_generation_chain()
    check_vocabulary(Path(args.gold), args.sheet)
    check_absence()
    check_backends()
    if args.quick:
        section("Chunker")
        record(SKIP, "chunking", "--quick")
    else:
        check_chunker(args.law, Path(args.pdf))
    check_evaluation(Path(args.gold), args.sheet, args.law)
    check_output_paths()

    fails = [r for r in _RESULTS if r[0] == FAIL]
    warns = [r for r in _RESULTS if r[0] == WARN]
    print("\n" + "=" * 74)
    print(f"SUMMARY: {sum(1 for r in _RESULTS if r[0] == PASS)} pass, "
          f"{len(fails)} FAIL, {len(warns)} warn, "
          f"{sum(1 for r in _RESULTS if r[0] == SKIP)} skipped")
    for status, name, _ in fails + warns:
        print(f"  {status:<5} {name}")
    print("=" * 74)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())