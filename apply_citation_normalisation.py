#!/usr/bin/env python3
"""
apply_citation_normalisation.py — wires citations.py into run_pipeline.py.

Adds, as a post-processing pass on *validated* statements only:
  1. canonical source_clause form  "<LAW> <clause>"
  2. validation of every citation against the article_refs actually in the
     corpus, which catches the hallucinated-law class ('PIPADE')
  3. a _cited_clauses roll-up on the statement root
  4. a citation-quality block in the pipeline summary

Design constraint: the hook sits AFTER PolicyStatementModel.model_validate,
so it cannot influence any extraction or assembly decision. Scored evaluation
output must be byte-identical after this change. If it is not, that is a bug
in this patch, not a result.

Idempotent. Run from the repo root:

    python3 apply_citation_normalisation.py
    git diff --stat
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

TARGET = Path("run_pipeline.py")


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def patch_xmi_writer() -> None:
    """
    Declare _cited_clauses pipeline-only.

    The writer already skips features it cannot find in the Ecore, so this is
    not load-bearing — but an unlisted field relies on that silent skip, and
    `source_clause` is listed for exactly this reason.
    """
    p = Path("tranform_format/pydantic_to_xmi.py")
    if not p.exists():
        print(f"  ! {p} not found — skipped")
        return
    txt = p.read_text()
    if "_cited_clauses" in txt:
        print(f"  = {p} already declares _cited_clauses")
        return
    anchor = '    "source_clause",\n'
    if anchor not in txt:
        print(f"  ! could not anchor in {p} — skipped")
        return
    p.write_text(txt.replace(anchor, anchor + '    "_cited_clauses",\n', 1))
    print(f"✓ Patched {p} (_cited_clauses declared pipeline-only)")


def patch_verify_repo() -> None:
    """
    Add the drift check for citations.KNOWN_LAWS.

    citations.py keeps its own copy of the law names so it stays import-free.
    A second copy of a vocabulary is exactly the pattern that has already
    caused three failures in this repo (prompts.py:23, prompts.py:612,
    evaluate.py:169), so it gets a check the moment it is created rather than
    after it drifts.
    """
    p = Path("verify_repo.py")
    if not p.exists():
        print(f"  ! {p} not found — skipped")
        return
    txt = p.read_text()
    if "citations.KNOWN_LAWS" in txt:
        print(f"  = {p} already checks KNOWN_LAWS")
        return

    anchor = "def check_absence() -> None:"
    if anchor not in txt:
        print(f"  ! could not anchor in {p} — skipped")
        return

    block = '''def check_citations() -> None:
    section("Citation normalisation")

    try:
        import citations
        import run_pipeline
    except Exception as exc:
        record(FAIL, "citations module imports", str(exc)[:200])
        return

    # 3a. KNOWN_LAWS must not drift from JURISDICTION_MAP (second copy of a
    #     vocabulary — the failure mode that bit prompts.py and evaluate.py).
    declared = set(citations.KNOWN_LAWS)
    actual   = set(run_pipeline.JURISDICTION_MAP)
    record(
        FAIL if declared != actual else PASS,
        "citations.KNOWN_LAWS == JURISDICTION_MAP",
        f"citations={sorted(declared)} pipeline={sorted(actual)}"
        if declared != actual else "",
    )

    # 3b. Normalisation must be idempotent — it runs on stored statements and
    #     a re-run must not keep rewriting them.
    known = citations.known_clauses(
        ["4.3 Principle 3 — Consent", "4.3.1", "4.3.8"], "PIPEDA"
    )
    once, _, _  = citations.normalise_citation("PIPEDA Art.4.3", "PIPEDA", known)
    twice, _, _ = citations.normalise_citation(once, "PIPEDA", known)
    record(FAIL if once != twice else PASS, "citation normalisation idempotent",
           f"{once!r} -> {twice!r}" if once != twice else "")

    # 3c. The hallucinated-law detector must fire.
    _, _, flag = citations.normalise_citation(
        "PIPADE Art.4.3", "PIPEDA", known
    )
    record(FAIL if flag != citations.LAW_MISMATCH else PASS,
           "hallucinated law name detected",
           f"got {flag!r}, expected law_mismatch" if flag != citations.LAW_MISMATCH else "")

    # 3d. The index must hold identifiers, not ancestor-derived bare digits.
    #     Bare top-level digits make almost any citation validate.
    bare = {c for c in known if c.isdigit() and len(c) <= 2}
    record(WARN if bare else PASS, "citation index free of bare digits",
           f"bare paths in index: {sorted(bare)}" if bare else "")


'''
    txt = txt.replace(anchor, block + anchor, 1)

    # Call it from main(), right after the vocabulary check.
    call_anchor = re.search(r"^(\s*)check_vocabulary\(([^\n]*)\)\n", txt, re.M)
    if call_anchor:
        indent = call_anchor.group(1)
        txt = txt[:call_anchor.end()] + f"{indent}check_citations()\n" + txt[call_anchor.end():]
    else:
        print("  ! could not find the check_vocabulary call site — "
              "add check_citations() to main() by hand")

    p.write_text(txt)
    print(f"✓ Patched {p} (check_citations: 4 checks)")


def main() -> None:
    if not TARGET.exists():
        die(f"{TARGET} not found — run this from the repo root.")
    if not Path("citations.py").exists():
        die("citations.py not found — add it before running this patch.")

    src = TARGET.read_text()
    original = src

    if "_KNOWN_REFS" in src:
        print("Already applied (found _KNOWN_REFS). Nothing to do.")
        return

    # ── 1. import ────────────────────────────────────────────────────────────
    anchor = "from privacy_schema.prompts import ("
    if anchor not in src:
        # Fall back to any privacy_schema import.
        m = re.search(r"^from privacy_schema[^\n]*\n", src, re.M)
        if not m:
            die("could not find a privacy_schema import to anchor on.")
        insert_at = m.end()
    else:
        insert_at = src.index(anchor)

    src = (
        src[:insert_at]
        + "from citations import known_clauses, normalise_statement, PROBLEM_FLAGS\n"
        + src[insert_at:]
    )

    # ── 2. stats counters ────────────────────────────────────────────────────
    stats_anchor = "    tokens_out:                  int = 0\n"
    if stats_anchor not in src:
        die("could not find PipelineStats.tokens_out to anchor counters on.")
    src = src.replace(
        stats_anchor,
        stats_anchor
        + "    citation_flags:              dict = field(default_factory=dict)\n"
        + "    citation_problems:           list = field(default_factory=list)\n",
        1,
    )

    # ── 3. summary block ─────────────────────────────────────────────────────
    summary_anchor = (
        '        if self.tokens_in or self.tokens_out:\n'
        '            log.info(f"  Tokens  in / out     : '
        '{self.tokens_in} / {self.tokens_out}")\n'
    )
    if summary_anchor not in src:
        die("could not find the tokens line in log_summary to anchor on.")
    summary_block = summary_anchor + '''
        if self.citation_flags:
            total = sum(self.citation_flags.values())
            bad   = sum(
                n for f, n in self.citation_flags.items() if f in PROBLEM_FLAGS
            )
            log.info(
                f"  Citations normalised : {total}  "
                f"(valid={total - bad}  problems={bad}  "
                f"{100 * (total - bad) / total:.1f}% clean)"
            )
            for flag, n in sorted(self.citation_flags.items()):
                marker = "  <-- " if flag in PROBLEM_FLAGS else "      "
                log.info(f"      {flag:<16}{n:>4}{marker}")
            for p in self.citation_problems[:20]:
                log.warning(f"  CITATION  {p}")
            if len(self.citation_problems) > 20:
                log.warning(
                    f"  CITATION  ... and {len(self.citation_problems) - 20} more"
                )
'''
    src = src.replace(summary_anchor, summary_block, 1)

    # ── 4. known-refs index ──────────────────────────────────────────────────
    helper_anchor = "def _build_article_filter("
    if helper_anchor not in src:
        die("could not find _build_article_filter to anchor the index on.")
    index_code = '''# Canonical clause paths that actually exist in the corpus, per law. Populated
# from chunks.db at the start of stage_extract and read during assembly to
# validate citations. Empty when assembly runs without a preceding extract, in
# which case validation degrades to format-only (law mismatches still caught).
_KNOWN_REFS: dict[str, frozenset[str]] = {}


def _load_known_refs(db_path: Path) -> None:
    """Index DISTINCT article_ref per law so citations can be validated."""
    import sqlite3
    try:
        con = sqlite3.connect(str(db_path))
        rows = con.execute(
            "SELECT law, article_ref FROM chunks"
        ).fetchall()
        con.close()
    except Exception as exc:
        log.warning(f"  Could not index article_refs for citation checks: {exc}")
        return

    per_law: dict[str, list[str]] = {}
    for law, ref in rows:
        per_law.setdefault(law, []).append(ref)

    for law, refs in per_law.items():
        _KNOWN_REFS[law] = known_clauses(refs, law)
        log.info(
            f"  Citation index       : {law} -> "
            f"{len(_KNOWN_REFS[law])} valid clause paths"
        )


'''
    src = src.replace(helper_anchor, index_code + helper_anchor, 1)

    # ── 5. call the loader from stage_extract ────────────────────────────────
    extract_anchor = '    log.info(f"  Concept-tag filter : ' \
                     "{'ON' if use_concept_tags else 'OFF'}\")\n"
    if extract_anchor not in src:
        die("could not find the concept-tag log line in stage_extract.")
    src = src.replace(
        extract_anchor,
        extract_anchor + "    _load_known_refs(db_path)\n",
        1,
    )

    # ── 6. the hook, after validation ────────────────────────────────────────
    hook_anchor = (
        "            stats.pass2_success += 1\n"
        "            return validated.model_dump(by_alias=True)\n"
    )
    if hook_anchor not in src:
        die("could not find the pass2_success return to anchor the hook on.")
    hook = '''            stats.pass2_success += 1
            dumped = validated.model_dump(by_alias=True)

            # ── Citation normalisation ────────────────────────────────────
            # Runs AFTER validation, so it cannot influence extraction or
            # assembly. Scored output must be unchanged by this block.
            #
            # Guarded separately: this enclosing try/except treats any
            # exception as a validation failure and retries the assembly, so
            # an unguarded bug here would silently turn into ASSEMBLY FAILED.
            # Provenance polish must never be able to lose a valid statement.
            try:
                dumped, flags, problems = normalise_statement(
                    dumped, law, _KNOWN_REFS.get(law),
                )
                for f, n in flags.items():
                    stats.citation_flags[f] = stats.citation_flags.get(f, 0) + n
                stats.citation_problems.extend(
                    f"{law}/{article_ref}  {p}" for p in problems
                )
            except Exception as norm_exc:          # pragma: no cover
                log.warning(
                    f"  Citation normalisation failed @ {law}/{article_ref}: "
                    f"{norm_exc} — storing statement un-normalised"
                )

            return dumped
'''
    src = src.replace(hook_anchor, hook, 1)

    if src == original:
        die("nothing changed — anchors matched but no edit applied.")

    TARGET.write_text(src)
    patch_xmi_writer()
    patch_verify_repo()
    print(f"✓ Patched {TARGET}")
    print("  - import citations")
    print("  - PipelineStats.citation_flags / citation_problems")
    print("  - citation-quality block in log_summary")
    print("  - _KNOWN_REFS index + _load_known_refs(db_path)")
    print("  - normalise_statement hook after Pass-2 validation")
    print()
    print("Next:  python3 tests/test_citations.py      (no pytest needed)")
    print("       python3 verify_repo.py")


if __name__ == "__main__":
    main()
