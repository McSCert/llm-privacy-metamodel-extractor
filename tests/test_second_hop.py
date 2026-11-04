"""
test_second_hop.py — regression tests for obligation-bearing cross-references.

No test framework required (same convention as test_citations.py):

    python3 tests/test_second_hop.py

The corpus test pins the full set of references the detector follows in
PIPEDA Schedule 1. If it changes, the experiment changed — review the
reference table (diagnose_second_hop.py --second-hop) before re-running.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_pipeline.second_hop import (                      # noqa: E402
    LABEL,
    append_references,
    find_references,
    obligation_references,
    opening_rule,
    own_clause,
)

# ── Detection ────────────────────────────────────────────────────────────────

def test_obligation_phrase_is_followed():
    t = ("4.4.1 Organizations shall specify the type of information collected "
         "as part of their information-handling policies and practices, in "
         "accordance with the Openness prin-\nciple (Clause 4.8).")
    assert obligation_references(t) == ["4.8"]


def test_coordinated_references_share_the_trigger():
    t = ("4.2.1 The organization shall document the purposes ... in order to "
         "comply with the Openness principle (Clause 4.8) and the Individual "
         "Access principle (Clause 4.9). 4.2.2 Identifying the purposes ...")
    assert obligation_references(t) == ["4.8", "4.9"]


def test_pointer_references_are_ignored():
    for t in (
        "This principle is linked closely to the Consent principle (Clause 4.3).",
        "For an elaboration on consent, please refer to the Consent principle (Clause 4.3).",
        "shall document this purpose (see Clause 4.2.1).",
        "The concept of sensitivity is discussed in Clause 4.3.4.",
    ):
        assert obligation_references(t) == [], t


def test_trigger_does_not_cross_a_sentence_end():
    t = ("must comply with the policy. 4.4.3 This principle is linked closely "
         "to the Identifying Purposes principle (Clause 4.2).")
    assert obligation_references(t) == []


def test_sentence_final_clause_number_is_not_glued_to_the_next():
    t = "is discussed in Clause 4.3.4.\n4.7.3\nThe methods of protection"
    assert [r for r, _, _ in find_references(t)] == ["4.3.4"]


def test_self_reference_is_excluded():
    t = "in accordance with Clause 4.4.2 and Clause 4.8"
    assert obligation_references(t, own="4.4") == ["4.8"]


def test_own_clause():
    assert own_clause("4.4 Principle 4 — Limiting") == "4.4"
    assert own_clause("4.10 Principle 10 — Challenging") == "4.10"
    assert own_clause("4.5.3") == "4.5.3"


# ── Fetching ─────────────────────────────────────────────────────────────────

def test_opening_rule_stops_at_first_subclause():
    t = ("4.8 Principle 8 — Openness\nAn organization shall make readily "
         "available ...\n4.8.1\nOrganizations shall be open ...")
    assert opening_rule("4.8", t).endswith("readily available ...")


def test_opening_rule_stops_at_note():
    t = ("4.9 Principle 9 — Individual\nAccess\nUpon request ... amended as "
         "appropriate.\nNote: In certain situations ...\n4.9.1\n...")
    assert opening_rule("4.9", t).endswith("amended as appropriate.")


def test_opening_rule_does_not_stop_at_4_10():
    t = "4.1 Principle 1 — Accountability\nAn organization is responsible ...\n4.1.1\n..."
    assert "4.1.1" not in opening_rule("4.1", t)


class _FakeStore:
    def __init__(self, rows):
        self.rows = rows

    def get_chunks(self, law, article_ref=None, concept_tags=None):
        return [r for r in self.rows if article_ref in (None, r["article_ref"])]


def test_append_adds_labelled_block_and_keeps_primary_text_first():
    store = _FakeStore([{"article_ref": "4.8 Principle 8 — Openness",
                         "text": "4.8 Principle 8 — Openness\nRULE\n4.8.1\nMORE"}])
    primary = "in accordance with the Openness principle (Clause 4.8)."
    out = append_references(store, "PIPEDA", "4.4 Principle 4 — Limiting", primary)
    assert out.startswith(primary)
    assert LABEL.format(ref="4.8") in out
    assert "RULE" in out and "MORE" not in out


def test_append_is_identity_without_references():
    store = _FakeStore([])
    t = "linked closely to the Consent principle (Clause 4.3)."
    assert append_references(store, "PIPEDA", "4.4 Principle 4", t) is t


def test_missing_reference_is_skipped_not_invented():
    store = _FakeStore([])
    t = "in accordance with Clause 9.9."
    assert append_references(store, "PIPEDA", "4.4 Principle 4", t) == t


# ── Corpus ───────────────────────────────────────────────────────────────────

def test_pipeda_schedule1_follows_exactly_three_references():
    db = ROOT / "data" / "chunks.db"
    if not db.exists():
        print("  skip  (no data/chunks.db)")
        return
    from rag_pipeline.store import ChunkStore
    found = {}
    with ChunkStore(db) as store:
        for r in store.get_chunks(law="PIPEDA"):
            if " Principle " not in r["article_ref"]:
                continue
            own = own_clause(r["article_ref"])
            refs = obligation_references(r["text"], own)
            if refs:
                found[own] = refs
    assert found == {"4.2": ["4.8", "4.9"], "4.4": ["4.8"]}, found


# ── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
