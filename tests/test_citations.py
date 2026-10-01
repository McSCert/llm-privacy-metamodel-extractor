"""
test_citations.py — regression tests for source_clause normalisation.

No test framework required — this repo has none in requirements.txt, and
test_embedder.py is a plain script. Run it directly:

    python3 tests/test_citations.py

It also works under pytest if you happen to have it (every check is a
test_* function using plain asserts), but pytest is not needed.

The observed-PIPEDA-4.3 case is the exact set of values found in the stored
statement: seven formats at once, including the 'PIPADE' hallucination. It is
the reason this module exists, so it is the first test.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from citations import (                                    # noqa: E402
    EMPTY,
    LAW_MISMATCH,
    LAW_ONLY,
    NO_CLAUSE,
    OK,
    UNKNOWN_CLAUSE,
    ancestors,
    known_clauses,
    law_name_verdict,
    normalise_citation,
    normalise_statement,
)

# PIPEDA as the chunker indexes it: Schedule 1's ten principles, the
# sub-clauses of Principle 3, and some of the Act body — the Act sections
# matter, because the model cites them (7.3, 7.4, 6.1) and a Schedule-1-only
# fixture would make those look like hallucinations.
PIPEDA_CORPUS = (
    [f"4.{i} Principle {i} — Title" for i in range(1, 11)]
    + [f"4.3.{i}" for i in range(1, 9)]
    + ["6.1 For the purposes of clause ...",
       "7.3 (1) For the purpose of clause ...",
       "7.4 (1) Despite clause 4.5 of Schedule 1 ...",
       "PIPEDA"]
)
GDPR_CORPUS = ["Art.6", "Art.6(1)(a)", "Art.17", "Art.7(3)", "Art.77"]

PIPEDA = known_clauses(PIPEDA_CORPUS, "PIPEDA")
GDPR   = known_clauses(GDPR_CORPUS, "GDPR")


# ── The seven formats actually observed in one stored statement ─────────────
def test_observed_formats_collapse():
    for raw, expected in [
        ("PIPEDA Art.4.3",               "PIPEDA 4.3"),
        ("4.3",                          "PIPEDA 4.3"),
        ("4.3.3",                        "PIPEDA 4.3.3"),
        ("4.3 Principle 3 - Consent",    "PIPEDA 4.3"),
        ("PIPEDA Principle 3 - Consent", "PIPEDA 4.3"),
        ("4.3.1",                        "PIPEDA 4.3.1"),
        ("PIPEDA 4.3.8",                 "PIPEDA 4.3.8"),
        ("PIPEDA Principle 3",           "PIPEDA 4.3"),
    ]:
        canon, _, flag = normalise_citation(raw, "PIPEDA", PIPEDA)
        assert canon == expected, f"{raw!r} -> {canon!r}, wanted {expected!r}"
        assert flag == OK, f"{raw!r} flagged {flag}"


# ── The law-name detector ───────────────────────────────────────────────────
def test_detector_is_generic_not_a_misspelling_list():
    """
    Nothing in citations.py contains the string 'PIPADE'. The rule is
    structural — near-miss edit distance to a known law name, plus a caps
    truncation rule — so it catches any wrong law name, in any casing.
    A caps-only rule missed 'Pipade' and 'PIP'.
    """
    for raw in [
        "PIPADE Art.4.3",     # the observed one: transposition, caps
        "Pipade Art.4.3",     # same misspelling, title case
        "pipade 4.3",         # same misspelling, lower case
        "PIPEDAA 4.3",        # doubled letter
        "PIDEPA 4.3",         # transposition
        "PIP 4.3",            # truncation
        "PRIVACYACT 4.3",     # not a near-miss of anything, but caps
    ]:
        canon, _, flag = normalise_citation(raw, "PIPEDA", PIPEDA)
        assert flag == LAW_MISMATCH, f"{raw!r} not flagged (got {flag})"
        assert canon == "PIPEDA 4.3", f"{raw!r} -> {canon!r}"


def test_prose_and_headings_are_not_law_names():
    """
    The near-miss rule must not fire on ordinary words. 'copy' is
    edit-distance 2 from 'CCPA' and flagged a real PIPEDA ref until the rule
    was tightened; PART and DIVISION are the only two caps words in the whole
    corpus that collided, and are handled as structural noise.
    """
    for word in [
        "copy",          # edit-distance 2 from CCPA — a real false positive
        "PART",          # caps structural heading, 28x in the corpus
        "DIVISION",      # caps structural heading, 49x in the corpus
        "Consent", "Accountability", "Safeguards", "Disclosure", "Openness",
        "clause", "Schedule", "Principle",
    ]:
        assert law_name_verdict(word, "PIPEDA") is None, \
            f"{word!r} wrongly treated as a law name"


def test_lowercase_known_law_still_counts():
    """Casing must not let a genuine cross-law citation through unflagged."""
    canon, _, flag = normalise_citation("gdpr art.17", "PIPEDA", PIPEDA)
    assert (canon, flag) == ("GDPR 17", LAW_MISMATCH)


def test_hallucinated_law_is_caught_and_corrected():
    """'PIPADE' is a misspelling of the law in scope: flag it, correct it."""
    canon, _, flag = normalise_citation(
        "PIPADE Art.4.3 Principle 3 - Consent", "PIPEDA", PIPEDA
    )
    assert flag == LAW_MISMATCH
    assert canon == "PIPEDA 4.3"


def test_cross_law_citation_is_preserved_not_relabelled():
    """A different *recognised* law is evidence; rewriting it destroys that."""
    canon, _, flag = normalise_citation("GDPR Art.17", "PIPEDA", PIPEDA)
    assert flag == LAW_MISMATCH
    assert canon == "GDPR 17"


def test_structural_headings_do_not_break_a_citation():
    canon, _, flag = normalise_citation(
        "PART 1 DIVISION 2 clause 4.3", "PIPEDA", PIPEDA
    )
    assert (canon, flag) == ("PIPEDA 4.3", OK)


# ── Clause parsing ──────────────────────────────────────────────────────────
def test_most_specific_clause_wins():
    """'Schedule 1' must not beat 'Clause 4.7', and must not leak as '1'."""
    canon, allc, flag = normalise_citation(
        "Schedule 1, Clause 4.7", "PIPEDA", PIPEDA
    )
    assert canon == "PIPEDA 4.7"
    assert allc == ["PIPEDA 4.7"]
    assert flag == OK


def test_unknown_clause_is_flagged_but_kept():
    """A clause with no known ancestor is flagged — and still kept, because a
    wrong-but-parsable citation is evidence an auditor needs to see."""
    canon, _, flag = normalise_citation("99.7", "PIPEDA", PIPEDA)
    assert canon == "PIPEDA 99.7"
    assert flag == UNKNOWN_CLAUSE


def test_sub_paragraph_of_a_known_clause_validates():
    """
    The chunker indexes refs at the level it segments the law, so '4.2(b)' and
    '4.9.5' never appear in the index even though they are real citations.
    Validating against ancestors is what stops the checker reporting correct
    sub-paragraph citations as hallucinations — 13 of them, on real data.
    """
    for raw, expected in [
        ("PIPEDA 4.2(b)",    "PIPEDA 4.2(b)"),
        ("4.9.5",            "PIPEDA 4.9.5"),
        ("PIPEDA 7.4(1)(b)", "PIPEDA 7.4(1)(b)"),
    ]:
        canon, _, flag = normalise_citation(raw, "PIPEDA", PIPEDA)
        assert canon == expected, raw
        assert flag == OK, f"{raw} should validate via its ancestors"


def test_known_validation_depth_is_a_stated_limitation():
    """
    Ancestor acceptance is deliberately permissive: any path under a real
    clause passes, so a fabricated deep sub-clause is NOT caught. Clause-level
    hallucination is only detectable to the depth the corpus is indexed.
    Law-level hallucination is caught independently and reliably.

    This test documents the limitation so it cannot be mistaken for a bug.
    """
    _, _, flag = normalise_citation("4.3.99", "PIPEDA", PIPEDA)
    assert flag == OK, "known limitation: not detectable at this index depth"

    _, _, flag = normalise_citation("PIPADE 4.3.99", "PIPEDA", PIPEDA)
    assert flag == LAW_MISMATCH, "law-level check must still fire"


def test_echoed_prompt_label_does_not_destroy_the_citation():
    """
    The model sometimes echoes a prompt label:
        'PIPEDA ARTICLE/SECTION: 4.1 (3)'
    Splitting at the colon would discard the only digits in the string.
    """
    canon, _, flag = normalise_citation(
        "PIPEDA ARTICLE/SECTION: 4.1 (3)", "PIPEDA", PIPEDA
    )
    assert canon == "PIPEDA 4.1(3)"
    assert flag == OK


def test_empty_citation_is_not_invented():
    """An object with no citation must not acquire one."""
    assert normalise_citation("", "PIPEDA", None) == ("", [], EMPTY)


def test_prose_only_citation():
    canon, _, flag = normalise_citation(
        "the organization's policy", "PIPEDA", PIPEDA
    )
    assert flag == NO_CLAUSE
    assert canon == "PIPEDA"


def test_law_only_is_legitimate():
    assert normalise_citation("PIPEDA", "PIPEDA", PIPEDA)[0::2] == \
        ("PIPEDA", LAW_ONLY)


def test_two_digit_principle():
    canon, _, _ = normalise_citation("Principle 10", "PIPEDA", PIPEDA)
    assert canon == "PIPEDA 4.10"


def test_ancestors_walks_both_separators():
    assert ancestors("7(3)(a)") == ["7(3)(a)", "7(3)", "7"]
    assert ancestors("4.3.8")   == ["4.3.8", "4.3", "4"]
    assert ancestors("4.1(3)")  == ["4.1(3)", "4.1", "4"]
    assert ancestors("4")       == ["4"]


# ── Law-specific behaviour must not leak across laws ───────────────────────
def test_principle_mapping_is_pipeda_only():
    """GDPR has no 'Principle N -> 4.N' structure; the mapping must not fire."""
    canon, _, _ = normalise_citation("GDPR Principle 3", "GDPR", GDPR)
    assert canon == "GDPR 3"


def test_gdpr_subparagraph_forms():
    for raw, expected in [
        ("GDPR Art.6(1)(a)", "GDPR 6(1)(a)"),
        ("Article 17",       "GDPR 17"),
        ("Art.7(3)",         "GDPR 7(3)"),
    ]:
        canon, _, flag = normalise_citation(raw, "GDPR", GDPR)
        assert canon == expected, raw
        assert flag == OK, raw


def test_nonexistent_gdpr_article_flagged():
    _, _, flag = normalise_citation("GDPR Art.99", "GDPR", GDPR)
    assert flag == UNKNOWN_CLAUSE


def test_ccpa_section_symbol():
    canon, _, _ = normalise_citation("§1798.100(a)", "CCPA", None)
    assert canon == "CCPA 1798.100(a)"


# ── Statement-level roll-up ────────────────────────────────────────────────
def test_statement_rollup_on_the_real_43_statement():
    stmt = {
        "source_clause": "PIPEDA Art.4.3",
        "purposes": [{"source_clause": "4.3"}, {"source_clause": "4.3.3"}],
        "constraints": [
            {"source_clause": "4.3.1"},
            {"source_clause": "4.3.2"},
            {"source_clause": "4.3.3"},
        ],
        "rightImpacted": [{"source_clause": "PIPEDA 4.3.8"}],
        "legalBasis": {
            "source_clause": "PIPEDA Principle 3 - Consent",
            "jurisdiction": [{"source_clause": "PIPEDA Principle 3 - Consent"}],
        },
        "governingRegulations": [{
            "source_clause": "4.3 Principle 3 - Consent",
            "jurisdiction": [{"source_clause": "PIPEDA"}],
        }],
        "consentWithdrawal": [
            {"source_clause": "PIPADE Art.4.3 Principle 3 - Consent"}
        ],
    }
    out, counts, problems = normalise_statement(stmt, "PIPEDA", PIPEDA)

    assert out["_cited_clauses"] == [
        "PIPEDA 4.3", "PIPEDA 4.3.1", "PIPEDA 4.3.2",
        "PIPEDA 4.3.3", "PIPEDA 4.3.8",
    ]
    assert out["constraints"][0]["source_clause"] == "PIPEDA 4.3.1"
    assert out["legalBasis"]["jurisdiction"][0]["source_clause"] == "PIPEDA 4.3"
    assert counts[LAW_MISMATCH] == 1
    assert len(problems) == 1
    assert "PIPADE" in problems[0]


def test_normalisation_is_idempotent():
    """Canonical input must survive a second pass unchanged."""
    stmt = {"source_clause": "PIPEDA 4.3",
            "constraints": [{"source_clause": "4.3.1"}]}
    once, _, _ = normalise_statement(dict(stmt), "PIPEDA", PIPEDA)
    twice, _, problems = normalise_statement(dict(once), "PIPEDA", PIPEDA)
    once_cmp = {k: v for k, v in once.items() if k != "_cited_clauses"}
    twice_cmp = {k: v for k, v in twice.items() if k != "_cited_clauses"}
    assert once_cmp == twice_cmp
    assert problems == []


def test_statement_untouched_apart_from_citations():
    """Nothing but source_clause and the roll-up may change."""
    stmt = {
        "statementId": "S1",
        "source_clause": "4.3",
        "constraints": [{"type": "Security", "source_clause": "4.3.1"}],
    }
    out, _, _ = normalise_statement(stmt, "PIPEDA", PIPEDA)
    assert out["statementId"] == "S1"
    assert out["constraints"][0]["type"] == "Security"
    assert set(out) == {"statementId", "source_clause",
                        "constraints", "_cited_clauses"}


# ── Runner (no framework needed) ───────────────────────────────────────────
def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception:
            failed.append(name)
            print(f"  FAIL  {name}")
            print("        " + traceback.format_exc().replace("\n", "\n        "))

    print(f"\n{len(tests) - len(failed)} passed, {len(failed)} failed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
