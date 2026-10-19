"""
citations.py — Canonical form, normalisation and validation for `source_clause`.

WHY THIS EXISTS
---------------
Every extracted object carries a `source_clause` string that is supposed to
record which clause of the law justified the extraction. In practice the LLM
writes it in whatever shape it likes. One real PIPEDA 4.3 statement contained
seven different formats at once:

    'PIPEDA Art.4.3'   'PIPEDA 4.3.8'   '4.3.1'   '4.3 Principle 3 - Consent'
    'PIPEDA Principle 3 - Consent'      'PIPEDA'   'PIPADE Art.4.3 ...'

Two separate defects:
  1. No canonical form, so the field cannot be used for automated lookup,
     cross-model matching, or clause-level conformance claims.
  2. No validation, which is how 'PIPADE' — a hallucinated law name — reached
     the stored model unnoticed.

CANONICAL FORM
--------------
    "<LAW> <clause>"       e.g. "PIPEDA 4.3.8", "GDPR 6(1)(a)"
    "<LAW>"                when the citation names the law only

SCOPE
-----
This module is pure: no I/O, no pipeline imports. It is applied as a
post-processing pass on a *validated* statement, so it cannot influence any
extraction or assembly decision. See `normalise_statement`.

NOTE ON THE METAMODEL
---------------------
`source_clause` is deliberately NOT an Ecore feature — pydantic_to_xmi.py
drops it via _PIPELINE_FIELDS, and generate_ecore.py documents it as
"pipeline-only". So this work hardens the *audit trail* in the repository DB,
not the exported model. Promoting provenance into the metamodel is a separate
design decision.
"""

from __future__ import annotations

import re
from typing import Any, Iterable


# ── Laws the pipeline knows about ────────────────────────────────────────────
# Mirrors JURISDICTION_MAP in run_pipeline.py. Kept as a plain frozenset here
# so this module stays import-free; verify_repo.py should assert they agree.
KNOWN_LAWS: frozenset[str] = frozenset({
    "GDPR", "LGPD", "CCPA", "CPRA", "PIPEDA",
})

# Structural noise the model sprinkles into citations. Stripped before the
# clause path is read. Order matters: longest first.
_NOISE = re.compile(
    r"\b(?:Articles?|Art\.?|Principles?|Schedules?|Sections?|Sec\.?|"
    r"Subsections?|Paragraphs?|Paras?\.?|Clauses?|"
    # Structural headings. PART and DIVISION are the only two words in the
    # whole PIPEDA corpus (of 1982 distinct words) that the law-name detector
    # would otherwise mistake for a law acronym, since they are set in caps.
    # Recital/Chapter/Annex are the GDPR equivalents.
    r"Parts?|Divisions?|Chapters?|Annexe?s?|Recitals?|Items?|Titles?"
    r")\b|§",
    re.IGNORECASE,
)

# A clause path: 4, 4.3, 4.3.8, 6(1)(a), 1798.100(a)
_CLAUSE = re.compile(r"\d+(?:\.\d+)*(?:\s*\([0-9A-Za-z]+\))*")

# An alphabetic run long enough to be a law name attempt.
_WORD = re.compile(r"[A-Za-z][A-Za-z]{2,}")

# PIPEDA Schedule 1: "Principle N" IS clause 4.N. The model frequently cites
# the principle number alone. This mapping is correct and law-specific.
_PRINCIPLE_NUMBER = re.compile(r"\bPrinciple\s+(\d{1,2})\b", re.IGNORECASE)


# ── Flags ────────────────────────────────────────────────────────────────────
OK             = "ok"              # canonical, clause known to the corpus
LAW_ONLY       = "law_only"        # names the law, no clause (often legitimate)
EMPTY          = "empty"           # nothing to normalise
LAW_MISMATCH   = "law_mismatch"    # named a law that is not the law in scope
UNKNOWN_CLAUSE = "unknown_clause"  # clause not among the retrieved refs
NO_CLAUSE      = "no_clause"       # prose only, no parsable clause path

PROBLEM_FLAGS = frozenset({LAW_MISMATCH, UNKNOWN_CLAUSE, NO_CLAUSE})


def _edit_distance(a: str, b: str, cap: int = 3) -> int:
    """Levenshtein distance, short-circuited at `cap`. Small inputs only."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def law_name_verdict(token: str, law: str) -> str | None:
    """
    Decide whether an alphabetic token in a citation is a law-name problem.

    Returns None when the token is not a law-name attempt at all (prose, or
    the law in scope), otherwise the law it should be attributed to.

    Three things must be true at once for this to be useful: it must catch a
    misspelling in ANY casing ('PIPADE', 'Pipade'), it must catch a truncation
    ('PIP'), and it must not fire on the ordinary words that appear inside real
    article refs ('Consent', 'Accountability', 'Safeguards', 'Disclosure').
    A caps-only rule fails the first two; a match-any-word rule fails the
    third. Near-miss distance to a known law name is what separates them.
    """
    t = token.upper()
    if t == law.upper():
        return None
    if t in KNOWN_LAWS:
        return t                        # genuine cross-law citation
    if _NOISE.fullmatch(token):
        return None

    # Casing decides how much benefit of the doubt a token gets. Prose inside a
    # real article_ref is lower/title case, and short words collide with the
    # short law acronyms by accident: 'copy' is edit-distance 2 from 'CCPA',
    # which produced a false positive on a real PIPEDA ref.
    if token.isupper():
        # A truncation, written in caps: 'PIP' for PIPEDA.
        if 3 <= len(t) < len(law) and law.upper().startswith(t):
            return law.upper()
        if len(t) >= 4:
            # Near-miss of the law in scope, or of some other known law.
            if _edit_distance(t, law.upper()) <= 2:
                return law.upper()
            for other in KNOWN_LAWS:
                if _edit_distance(t, other) <= 2:
                    return other
            # Caps and not a near-miss of anything: still suspect, because
            # prose inside a ref is not written in caps.
            return law.upper()
        return None

    # Lower/mixed case: only a near-miss of the law IN SCOPE counts, and only
    # at length 5+, which is what keeps four-letter English words from
    # colliding with CCPA / CPRA / GDPR / LGPD.
    if len(t) >= 5 and _edit_distance(t, law.upper()) <= 2:
        return law.upper()
    return None


def ancestors(clause: str) -> list[str]:
    """
    Every enclosing clause of a path, most specific first, including itself.

        "7(3)(a)"  → ["7(3)(a)", "7(3)", "7"]
        "4.3.8"    → ["4.3.8", "4.3", "4"]
        "4.1(3)"   → ["4.1(3)", "4.1", "4"]

    Needed because the chunker indexes refs at the level it segments the law,
    so a paragraph the model legitimately cites ("4.2(b)") will not appear in
    the index even though its parent clause does. Validating against ancestors
    rather than exact membership is what keeps a correct sub-paragraph citation
    from being reported as a hallucination.
    """
    out = [clause]
    cur = clause
    while True:
        if cur.endswith(")"):
            cut = cur.rfind("(")
            if cut > 0:
                cur = cur[:cut]
                out.append(cur)
                continue
        if "." in cur:
            cur = cur.rsplit(".", 1)[0]
            out.append(cur)
            continue
        break
    return out


def _is_known(clause: str, known: frozenset[str] | set[str]) -> bool:
    """A clause validates when it, or any clause enclosing it, is in the corpus."""
    return any(a in known for a in ancestors(clause))


def canonical_clause(raw: str, law: str) -> tuple[str, list[str], str, str]:
    """
    Reduce one raw citation to (cited_law, clauses, primary, flag).

    `cited_law`  the law the citation actually names — the law in scope when it
                 names none, or when the token is an unrecognised near-miss of
                 it (the PIPADE case, treated as a typo and corrected).
                 A *different recognised* law is preserved, because silently
                 relabelling it would destroy the evidence of a cross-law
                 citation.
    `clauses`    every clause path found, most specific first.
    `primary`    clauses[0], or "" when none parsed.
    `flag`       OK / LAW_ONLY / EMPTY / NO_CLAUSE / LAW_MISMATCH.

    >>> canonical_clause("PIPEDA Art.4.3", "PIPEDA")[2:]
    ('4.3', 'ok')
    >>> canonical_clause("PIPADE Art.4.3 Principle 3 - Consent", "PIPEDA")[0]
    'PIPEDA'
    >>> canonical_clause("GDPR Art.17", "PIPEDA")[0]
    'GDPR'
    >>> canonical_clause("Schedule 1, Clause 4.7", "PIPEDA")[2]
    '4.7'
    """
    text = (raw or "").strip()
    if not text:
        return law, [], "", EMPTY

    flag      = OK
    cited_law = law

    # ── Law token check ──────────────────────────────────────────────────────
    words = _WORD.findall(text)
    named_in_scope = any(w.upper() == law.upper() for w in words)
    for w in words:
        verdict = law_name_verdict(w, law)
        if verdict is not None:
            flag = LAW_MISMATCH
            # A different recognised law is a genuine cross-law citation and is
            # preserved; a misspelling or truncation of the law in scope is
            # corrected to it.
            cited_law = verdict
            break

    # ── PIPEDA principle → clause, before noise stripping eats "Principle" ──
    # Schedule 1 clause 4.N *is* Principle N. Law-specific and only applied
    # when the citation is understood to be about PIPEDA.
    principle_clause = ""
    if cited_law.upper() == "PIPEDA":
        m = _PRINCIPLE_NUMBER.search(text)
        if m:
            principle_clause = f"4.{int(m.group(1))}"

    # ── Clause paths ─────────────────────────────────────────────────────────
    stripped = _NOISE.sub(" ", text)
    # Drop a trailing prose title: "4.3 3 - Consent" → "4.3 3".
    # But only when the leading segment actually holds a citation. The model
    # sometimes echoes a prompt label, e.g.
    #   "PIPEDA ARTICLE/SECTION: 4.1 (3)"
    # where the digits live *after* the separator and splitting would discard
    # the whole citation.
    head = re.split(r"[-—–:]", stripped, maxsplit=1)[0]
    if re.search(r"\d", head):
        stripped = head

    found = [re.sub(r"\s+", "", m.group(0)) for m in _CLAUSE.finditer(stripped)]
    if principle_clause:
        # "Schedule 1 ... Principle 3" must not resolve to "1". The mapped
        # form is authoritative; a bare principle number is dropped.
        found = [c for c in found if c != principle_clause.split(".")[-1]]
        found.insert(0, principle_clause)

    # Most specific first — "Schedule 1, Clause 4.7" must prefer 4.7 over 1.
    def specificity(c: str) -> tuple[int, int]:
        return (c.count(".") + c.count("("), len(c))

    clauses: list[str] = sorted(dict.fromkeys(found), key=specificity, reverse=True)

    if not clauses:
        if named_in_scope and len(words) == 1:
            return cited_law, [], "", (LAW_ONLY if flag == OK else flag)
        return cited_law, [], "", (NO_CLAUSE if flag == OK else flag)

    return cited_law, clauses, clauses[0], flag


def normalise_citation(
    raw:         str,
    law:         str,
    known:       frozenset[str] | set[str] | None = None,
) -> tuple[str, list[str], str]:
    """
    Normalise one citation to canonical "<LAW> <clause>" form.

    Returns (canonical, all_canonical, flag). `all_canonical` holds every
    clause the string cited, so a citation naming two clauses contributes both
    to the statement roll-up instead of being silently truncated.

    `known` is the set of canonical clause paths that actually exist in the
    corpus for this law (see `known_clauses`). When supplied, a clause outside
    it is flagged UNKNOWN_CLAUSE — the citation is still normalised, because a
    wrong-but-parsable ref is more useful to an auditor than a discarded one.
    An empty citation stays empty: provenance is never invented.
    """
    cited_law, clauses, primary, flag = canonical_clause(raw, law)

    if flag == EMPTY:
        return "", [], EMPTY

    if not primary:
        return cited_law, [], flag

    # Only check the corpus when the citation is about the law we retrieved for.
    in_scope = cited_law.upper() == law.upper()
    if known is not None and flag == OK and in_scope and not _is_known(primary, known):
        flag = UNKNOWN_CLAUSE

    # The primary is kept whatever it is — a flagged citation is evidence and
    # must not be discarded. Secondary clauses must validate, so that stray
    # numbers ("Schedule 1, Clause 4.7" → "1") never enter the roll-up.
    kept = [clauses[0]]
    if known is not None and in_scope:
        kept += [c for c in clauses[1:] if _is_known(c, known)]
    else:
        kept += clauses[1:]

    allc = [f"{cited_law} {c}" for c in kept]
    return allc[0], allc, flag


# An article_ref's own identifier: the citation anchored at its start.
# The chunker stores refs as identifier + the opening prose, e.g.
#   "7.4 (1) Despite clause 4.5 of Schedule 1 ..."
# Only "7.4(1)" identifies that chunk — "4.5" and "1" are prose. Scanning the
# whole ref would seed the index with clause paths the law does not have.
_LEADING_REF = re.compile(
    r"^\s*(?:§\s*)?(\d+(?:\.\d+)*(?:\s*\([0-9A-Za-z]+\))*)"
)


def known_clauses(article_refs: Iterable[str], law: str) -> frozenset[str]:
    """
    Build the set of valid clause paths for a law from the corpus' own
    article_ref values (i.e. `SELECT DISTINCT article_ref FROM chunks WHERE
    law = ?`), canonicalised so the comparison is canonical-to-canonical.

    Only each ref's *leading* identifier is indexed — see _LEADING_REF. Refs
    with no leading number (PIPEDA "Principle 3 — Consent") fall back to the
    full normaliser so the principle mapping still applies.

    The index holds identifiers ONLY — ancestors are deliberately NOT added.
    Walking ancestors into the index seeds it with bare top-level digits
    (1, 2, … 7, derived from refs like "10.1(1)" and "7.4(1)"), and once those
    are present almost any citation validates through one of them. Measured on
    the real PIPEDA corpus that inflated the index from 69 identifiers to 97
    paths and took the citation checker to a meaningless 100% pass rate.
    Ancestor tolerance belongs on the citation side, in `_is_known`, where it
    accepts a sub-paragraph of a real clause without widening the index.
    """
    out: set[str] = set()
    for ref in article_refs:
        m = _LEADING_REF.match(ref or "")
        if m:
            out.add(re.sub(r"\s+", "", m.group(1)))
        else:
            _, clauses, _, _ = canonical_clause(ref, law)
            out.update(clauses)
    return frozenset(out)


def normalise_statement(
    stmt:  dict,
    law:   str,
    known: frozenset[str] | set[str] | None = None,
) -> tuple[dict, dict[str, int], list[str]]:
    """
    Walk a validated statement dict, normalise every `source_clause` in place,
    and roll the distinct clause set up onto the root.

    Returns (stmt, flag_counts, problems) where `problems` lists
    "<path>: <raw> → <canonical> [<flag>]" for anything in PROBLEM_FLAGS.

    The roll-up is written to `_cited_clauses`, a new pipeline-only field.
    It is NOT written into `source_clause`, because that would change a scalar
    field into a string-encoded list; and underscore-prefixed pipeline fields
    are already the established convention here (`_extraction_confidence`,
    `_warnings`), and are dropped on XMI export.
    """
    counts: dict[str, int] = {}
    problems: list[str] = []
    seen: set[str] = set()

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, val in node.items():
                if key == "source_clause" and isinstance(val, str):
                    canon, allc, flag = normalise_citation(val, law, known)
                    counts[flag] = counts.get(flag, 0) + 1
                    if flag in PROBLEM_FLAGS:
                        problems.append(
                            f"{path or 'root'}: {val!r} → {canon!r} [{flag}]"
                        )
                    node[key] = canon
                    seen.update(c for c in allc if " " in c)
                else:
                    walk(val, f"{path}.{key}" if path else key)
        elif isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, f"{path}[{i}]")

    walk(stmt, "")

    def sort_key(ref: str) -> tuple:
        cited_law, _, clause = ref.partition(" ")
        # In-scope law first, then numerically by clause path.
        return (
            cited_law.upper() != law.upper(),
            cited_law,
            tuple(int(p) if p.isdigit() else 0
                  for p in re.split(r"[.()]", clause) if p),
            clause,
        )

    stmt["_cited_clauses"] = sorted(seen, key=sort_key)
    return stmt, counts, problems
