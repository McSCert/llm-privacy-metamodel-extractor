"""
rag_pipeline/second_hop.py — follow obligation-bearing cross-references.

WHY THIS EXISTS
---------------
Retrieval is scoped to the article under extraction, so when a clause applies
an obligation stated in another clause, the model never sees that obligation.
PIPEDA 4.4.1: "Organizations shall specify the type of information collected
... in accordance with the Openness principle (Clause 4.8)". The Openness rule
lives in 4.8; extracting 4.4 alone misses the Transparency constraint that
gold v2.1 records at 4.4 (annotation rule of 2026-10-05: a cross-reference
incorporates the referenced obligation into the citing clause).

WHAT IT DOES
------------
1. Detect only OBLIGATION-BEARING references: a clause reference that follows
   an obligation phrase ("in accordance with", "comply with", ...) in the same
   sentence. Pointer references ("linked closely to", "please refer to",
   "see Clause", "is discussed in") are ignored. In PIPEDA Schedule 1 that
   leaves exactly 4.2.1 -> 4.8, 4.9 and 4.4.1 -> 4.8.
2. Fetch the referenced text:
     - a whole principle ("Clause 4.8")  -> its OPENING RULE only: the
       principle heading and statement, up to its first numbered sub-clause
       or "Note:" paragraph;
     - a specific clause ("Clause 4.5.3") -> that clause's own chunk.
3. Append each one under a fixed label, after the primary text:
       --- REFERENCED CONTEXT (Clause 4.8, cited by this clause) ---
   No instruction is added to the prompt; the label is the only new wording.
4. One hop only: references inside fetched text are not followed.
5. After Pass 1, drop any extracted list value whose `source_clause` points
   outside the principle being extracted (`filter_out_of_principle`). With
   referenced context in the prompt, the model sometimes lifts a value *out of*
   the referenced clause (PIPEDA 4.2: an `Access` right cited "Clause 4.9")
   instead of incorporating an obligation into the citing clause (4.4:
   `Transparency` cited "4.4.1"). The gold rule (2026-10-05) incorporates
   obligations, not values stated or granted elsewhere; the citation tells the
   two apart. Measured on PIPEDA: removes exactly that one value.

SCOPE AND LIMITS
----------------
- The reference pattern ("Clause N.N") and the principle lookup
  ("<N.N> Principle ...") are PIPEDA Schedule 1 conventions. Other laws
  (GDPR "Article 6(1)") need their own patterns; nothing fires for them today.
- Deferring references without a clause number ("the purposes identified by
  the organization") are deliberately NOT handled here. That is experiment 2.
- Off unless Retriever(second_hop=True) / run_pipeline --second-hop. With it
  off, retrieval output is byte-identical to before this module existed.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Iterable, Optional

log = logging.getLogger(__name__)

LABEL = "--- REFERENCED CONTEXT (Clause {ref}, cited by this clause) ---"

# Phrases that make the following reference part of the citing clause's
# obligation. Kept short and explicit so the detector is reviewable.
OBLIGATION_PHRASES: tuple[str, ...] = (
    "in accordance with",
    "in compliance with",
    "comply with",
    "pursuant to",
    "as required by",
)

_TRIGGER = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in OBLIGATION_PHRASES) + r")\b",
    re.IGNORECASE,
)
_CLAUSE_REF = re.compile(r"\bClauses?\s+(\d+(?:\.\d+)+)", re.IGNORECASE)
# A sentence ends at a full stop followed by whitespace. "4.8" has no
# whitespace after its dot, so clause numbers never end a sentence.
_SENTENCE_END = re.compile(r"\.\s")
# A trigger governs references at most this far into its sentence.
_MAX_SCOPE = 250


def _flatten(text: str) -> str:
    """Undo PDF line wrapping, for detection only (fetched text stays raw)."""
    t = re.sub(r"-\s*\n\s*", "", text)          # "prin-\nciple" -> "principle"
    t = re.sub(r"\s+", " ", t)
    # Clause numbers are never split across lines in the PIPEDA corpus
    # (checked 2026-11-02). Do not try to re-join "4. 8": that also glues a
    # sentence-final "4.3.4." to the next clause number "4.7.3".
    return t


def own_clause(article_ref: str) -> str:
    """'4.4 Principle 4 — Limiting' -> '4.4'; '4.5.3' -> '4.5.3'."""
    m = re.match(r"\s*(\d+(?:\.\d+)*)", article_ref or "")
    return m.group(1) if m else ""


def _is_self(ref: str, own: str) -> bool:
    return bool(own) and (ref == own or ref.startswith(own + "."))


def find_references(text: str) -> list[tuple[str, bool, str]]:
    """
    Every clause reference in *text*, in order, as
    (clause, obligation_bearing, context_snippet).
    """
    flat = _flatten(text)
    # Character spans governed by an obligation phrase.
    scopes: list[tuple[int, int]] = []
    for m in _TRIGGER.finditer(flat):
        end = _SENTENCE_END.search(flat, m.end())
        stop = min(end.start() if end else len(flat), m.end() + _MAX_SCOPE)
        scopes.append((m.start(), stop))

    out = []
    for m in _CLAUSE_REF.finditer(flat):
        governed = any(a <= m.start() < b for a, b in scopes)
        snippet = flat[max(0, m.start() - 70): m.end() + 1]
        out.append((m.group(1), governed, snippet))
    return out


def obligation_references(text: str, own: str = "") -> list[str]:
    """Obligation-bearing references in *text*, de-duplicated, self excluded."""
    seen: list[str] = []
    for ref, governed, _ in find_references(text):
        if governed and not _is_self(ref, own) and ref not in seen:
            seen.append(ref)
    return seen


def opening_rule(principle_ref: str, text: str) -> str:
    """
    The heading and statement of a principle chunk, up to its first numbered
    sub-clause ("4.8.1") or a "Note:" paragraph.
    """
    stop = re.compile(
        rf"^\s*(?:{re.escape(principle_ref)}\.1\b|Note:)", re.MULTILINE)
    m = stop.search(text)
    return (text[: m.start()] if m else text).strip()


def fetch_referenced_text(store, law: str, ref: str) -> Optional[str]:
    """Opening rule for a principle reference, else the clause's own chunk."""
    rows = store.get_chunks(law=law)
    if ref.count(".") == 1:
        for r in rows:
            if r["article_ref"].startswith(f"{ref} Principle"):
                return opening_rule(ref, r["text"])
    for r in rows:
        if r["article_ref"] == ref:
            return r["text"].strip()
    return None


def append_references(store, law: str, article_ref: str, rag_text: str) -> str:
    """
    Return *rag_text* followed by one labelled block per obligation-bearing
    reference it contains. Returns *rag_text* unchanged when there are none.
    """
    if not rag_text:
        return rag_text
    blocks = []
    for ref in obligation_references(rag_text, own_clause(article_ref)):
        body = fetch_referenced_text(store, law, ref)
        if not body:
            log.warning(f"[second-hop] {law} {article_ref}: Clause {ref} "
                        f"cited but not found in the corpus — skipped")
            continue
        blocks.append(LABEL.format(ref=ref) + "\n" + body)
        log.debug(f"[second-hop] {law} {article_ref}: + Clause {ref} "
                  f"({len(body)} chars)")
    if not blocks:
        return rag_text
    return rag_text + "\n\n" + "\n\n".join(blocks)


def reference_table(store, law: str,
                    article_refs: Optional[Iterable[str]] = None) -> str:
    """
    Human-readable table of every clause reference in the principle-level
    chunks, marked FOLLOW (obligation-bearing) or ignore (pointer), with the
    size of what would be fetched. For review before any LLM run.
    """
    rows = [r for r in store.get_chunks(law=law) if " Principle " in r["article_ref"]]
    if article_refs is not None:
        wanted = set(article_refs)
        rows = [r for r in rows if r["article_ref"] in wanted]
    lines = [f"{'from':6} {'to':7} {'action':7} {'fetch':>6}  context"]
    for r in rows:
        own = own_clause(r["article_ref"])
        for ref, governed, snippet in find_references(r["text"]):
            follow = governed and not _is_self(ref, own)
            body = fetch_referenced_text(store, law, ref) if follow else None
            size = f"{len(body)}c" if body else ("MISSING" if follow else "")
            lines.append(f"{own:6} {ref:7} {'FOLLOW' if follow else 'ignore':7} "
                         f"{size:>6}  …{snippet}")
    return "\n".join(lines)


def _cites_outside(source_clause: str, law: str, own: str) -> Optional[str]:
    """
    The canonical clause a value cites, if it lies outside *own* (not equal to
    it and not a descendant of it); otherwise None. Values with no parseable
    clause are kept.
    """
    from citations import canonical_clause      # pure module, repo root
    primary = canonical_clause(source_clause or "", law)[2]
    if primary and not _is_self(primary, own):
        return primary
    return None


def filter_out_of_principle(
    concepts: dict[str, str], law: str, article_ref: str,
) -> tuple[dict[str, str], list[dict]]:
    """
    Drop Pass-1 list values whose `source_clause` points outside the article.

    *concepts* maps concept name -> Pass-1 JSON string, as stored in the
    Pass-1 record. Only JSON *lists* are filtered (Right, Purpose, Constraint
    and other multi-instance concepts); single-object concepts are untouched.
    A concept's string is rewritten only when something is removed, so an
    unaffected concept stays byte-identical.

    Returns (new_concepts, removed), where each removed entry is
    {"concept", "type", "cites", "source_clause"}.
    """
    own = own_clause(article_ref)
    out = dict(concepts)
    removed: list[dict] = []
    if not own:
        return out, removed
    for concept, raw in concepts.items():
        try:
            obj = json.loads(raw) if raw else None
        except (TypeError, ValueError):
            continue
        if not isinstance(obj, list):
            continue
        keep = []
        for v in obj:
            sc = v.get("source_clause", "") if isinstance(v, dict) else ""
            cited = _cites_outside(sc, law, own)
            if cited:
                removed.append({"concept": concept,
                                "type": v.get("type"),
                                "cites": cited,
                                "source_clause": sc})
                continue
            keep.append(v)
        if len(keep) != len(obj):
            out[concept] = json.dumps(keep)
    return out, removed
