#!/usr/bin/env python3
"""
apply_a1_multi_instance.py — A1: let Pass 1 return MORE THAN ONE instance
of Purpose, Right and Constraint.

The problem
-----------
Every run so far has produced exactly one label per list-valued concept per
article — 90 out of 90 across three configurations and two models.  The
cause is not prompt weakness and not the downstream schema: the Pass-1
prompts ask for "a Purpose instance" and specify a SINGLE JSON object, so
the model structurally cannot return two rights for PIPEDA 4.9 even though
4.9 grants both access and correction.  A2's [0..*] relaxation made ZERO
expressible; it did not make TWO expressible.

Observed consequence: once the chunker fix made clause 4.9.5 reachable, the
model swapped Access for Rectification instead of returning both, so a
better-informed extraction scored WORSE.

What changes
------------
  privacy_schema/models.py    + PurposeListModel / RightListModel /
                                ConstraintListModel  (wrapper objects; the
                                API requires an object, not a bare array,
                                as the top-level schema)
  privacy_schema/prompts.py   the three prompts now ask for ALL instances
                                and emit {"purposes": [ ... ]}; absence is
                                an empty array, replacing the
                                {"_no_X_stated": true} sentinel
  run_pipeline.py             validators map -> the wrapper models
                              _is_concept_absent  understands an empty
                                wrapper array
                              _unwrap_list_concept  turns {"purposes": [...]}
                                into a bare JSON array for the assembler
                              _override_constraint_type  walks a list

_wrap_for_assembler already passes arrays through untouched, and
EMPTY_FALLBACKS already uses "[]" for these three, so the assembler side
needs no change.

Run from the repo root:
    python apply_a1_multi_instance.py            # check only
    python apply_a1_multi_instance.py --write    # apply
"""
import subprocess
import sys
from pathlib import Path

# ── models.py: wrapper models appended at end of file ─────────────────────────
MODELS_ANCHOR = "class PolicyStatementModel(_Base):"
MODELS_ADDITION = '''# ---------------------------------------------------------------------------
# Pass-1 list wrappers (A1: multi-instance extraction)
# ---------------------------------------------------------------------------
# The OpenAI / Ollama json_schema response format requires an OBJECT at the
# top level, so a bare List[...] cannot be used as the decoding schema.  Each
# wrapper holds exactly one array field; run_pipeline unwraps it immediately
# after validation, so nothing downstream sees the wrapper.


class PurposeListModel(_Base):
    """All Purpose instances stated in one article. Empty list = none stated."""
    purposes: List[PurposeModel] = Field(default_factory=list)


class RightListModel(_Base):
    """All Right instances stated in one article. Empty list = none stated."""
    rights: List[RightModel] = Field(default_factory=list)


class ConstraintListModel(_Base):
    """All Constraint instances stated in one article. Empty list = none."""
    constraints: List[ConstraintModel] = Field(default_factory=list)


'''

# ── prompts.py ────────────────────────────────────────────────────────────────
PURPOSE_OLD = '''        "## Task: Extract a Purpose instance\\n\\n"
        "### What you are extracting\\n"
        "If NO purpose is described, return exactly:\\n"
        "{\\"_no_purpose_stated\\": true}\\n\\n"
        "The reason or objective for which personal data is processed.\\n"
        "Maps to: GDPR Art.5(1)(b) | LGPD Art.6 | CCPA business-purpose | PIPEDA Principle 2\\n\\n"
        "### Output schema\\n"
        "{\\n"
        '  "purposeId": "",\\n'
        '  "description": "<specific purpose as stated in the legal text>",\\n'
        '  "category": "<PurposeCategory>",\\n'
        '  "source_clause": "<article reference>"\\n'
        "}\\n\\n"'''
PURPOSE_NEW = '''        "## Task: Extract ALL Purpose instances\\n\\n"
        "### What you are extracting\\n"
        "Every distinct reason or objective for which personal data is "
        "processed, as stated in this article.\\n"
        "Maps to: GDPR Art.5(1)(b) | LGPD Art.6 | CCPA business-purpose | PIPEDA Principle 2\\n\\n"
        "If NO purpose is described, return exactly: {\\"purposes\\": []}\\n"
        "An empty array is a valid, meaningful answer — it records that the "
        "article states no purpose. Do not invent one.\\n\\n"
        "### Output schema\\n"
        "{\\n"
        '  "purposes": [\\n'
        "    {\\n"
        '      "purposeId": "",\\n'
        '      "description": "<specific purpose as stated in the legal text>",\\n'
        '      "category": "<PurposeCategory>",\\n'
        '      "source_clause": "<article or clause reference>"\\n'
        "    }\\n"
        "  ]\\n"
        "}\\n\\n"'''

PURPOSE_RULES_ANCHOR = '''        "research / scientific / academic            Research\\n\\n"'''
PURPOSE_RULES_NEW = '''        "research / scientific / academic            Research\\n\\n"
        "### Key rules\\n"
        "- Return one array entry per DISTINCT purpose. An article that names\\n"
        "  three purposes yields three entries, not one merged entry.\\n"
        "- Sub-clauses count: a purpose stated in 4.3.5 is a separate entry\\n"
        "  from one stated in the principle heading.\\n"
        "- Two entries that differ only in wording are ONE purpose — merge them.\\n"
        "- Cite the most specific clause you can in source_clause.\\n\\n"'''

RIGHT_OLD = '''        "## Task: Extract a Right instance\\n\\n"
        "### What you are extracting\\n"
        "A data-subject right affected by this processing statement.\\n"
        "Maps to: GDPR Art.15-22 | LGPD Art.17-22 | CCPA §1798.100-145 | PIPEDA Principle 9\\n\\n"
        "If NO right is described, return exactly:\\n"
        '{"_no_right_stated": true}\\n\\n'
        "### Output schema\\n"
        "{\\n"
        '  "rightId": "",\\n'
        '  "type": "<RightType>",\\n'
        '  "triggerCondition": "<condition under which the right may be exercised>",\\n'
        '  "fulfillmentProcess": "<how the controller must respond>",\\n'
        '  "source_clause": "<article reference>"\\n'
        "}\\n\\n"'''
RIGHT_NEW = '''        "## Task: Extract ALL Right instances\\n\\n"
        "### What you are extracting\\n"
        "Every data-subject right affected by this processing statement.\\n"
        "Maps to: GDPR Art.15-22 | LGPD Art.17-22 | CCPA §1798.100-145 | PIPEDA Principle 9\\n\\n"
        "If NO right is described, return exactly: {\\"rights\\": []}\\n"
        "An empty array is a valid, meaningful answer — it records that the "
        "article grants no right. Do not invent one.\\n\\n"
        "### Output schema\\n"
        "{\\n"
        '  "rights": [\\n'
        "    {\\n"
        '      "rightId": "",\\n'
        '      "type": "<RightType>",\\n'
        '      "triggerCondition": "<condition under which the right may be exercised>",\\n'
        '      "fulfillmentProcess": "<how the controller must respond>",\\n'
        '      "source_clause": "<article or clause reference>"\\n'
        "    }\\n"
        "  ]\\n"
        "}\\n\\n"'''

RIGHT_RULES_OLD = '''        "- Extract ONE Right per call — the primary right described in this article.\\n"'''
RIGHT_RULES_NEW = '''        "- Return one array entry per DISTINCT right. An article that grants\\n"
        "  both access and correction yields TWO entries, not one.\\n"
        "  (PIPEDA Principle 9 is exactly this case: 4.9 grants access and\\n"
        "  4.9.5 grants correction — both belong in the array.)\\n"
        "- Sub-clauses count: a right stated in a numbered sub-clause is a\\n"
        "  separate entry from one stated in the principle heading.\\n"
        "- Two entries with the same `type` are ONE right — merge them.\\n"'''

CONSTRAINT_OLD = '''        "## Task: Extract a Constraint instance\\n\\n"
        "### What you are extracting\\n"
        "If NO explicit constraint or restriction is stated in this article, "
        "return exactly: {\\"_no_constraint_stated\\": true}\\n\\n"
        "Not the legal basis (why), not the purpose (what for),\\n"
        "but a specific operational rule that limits or shapes processing.\\n"
        "Examples: retention limits, geographic restrictions, encryption requirements,\\n"
        "purpose-limitation rules, usage restrictions.\\n\\n"
        "### Output schema\\n"
        "{\\n"
        '  "constraintId": "",\\n'
        '  "type": "<ConstraintType>",\\n'
        '  "expression": "<natural-language statement of the constraint>",\\n'
        '  "enforcementLevel": "<Mandatory | Recommended | BestEffort>",\\n'
        '  "source_clause": "<article reference>"\\n'
        "}\\n\\n"'''
CONSTRAINT_NEW = '''        "## Task: Extract ALL Constraint instances\\n\\n"
        "### What you are extracting\\n"
        "Every specific operational rule that limits or shapes processing.\\n"
        "Not the legal basis (why), not the purpose (what for).\\n"
        "Examples: retention limits, geographic restrictions, encryption requirements,\\n"
        "purpose-limitation rules, usage restrictions.\\n\\n"
        "If NO explicit constraint is stated, return exactly: "
        "{\\"constraints\\": []}\\n"
        "An empty array is a valid, meaningful answer — it records that the "
        "article states no constraint. Do not invent one.\\n\\n"
        "### Output schema\\n"
        "{\\n"
        '  "constraints": [\\n'
        "    {\\n"
        '      "constraintId": "",\\n'
        '      "type": "<ConstraintType>",\\n'
        '      "expression": "<natural-language statement of the constraint>",\\n'
        '      "enforcementLevel": "<Mandatory | Recommended | BestEffort>",\\n'
        '      "source_clause": "<article or clause reference>"\\n'
        "    }\\n"
        "  ]\\n"
        "}\\n\\n"'''

CONSTRAINT_RULES_OLD = '''        "- Extract the PRIMARY constraint for this call.\\n\\n"'''
CONSTRAINT_RULES_NEW = '''        "- Return one array entry per DISTINCT constraint. An article that\\n"
        "  both limits retention AND requires accuracy yields TWO entries.\\n"
        "- Sub-clauses count: a constraint stated in a numbered sub-clause is\\n"
        "  a separate entry from one stated in the principle heading.\\n"
        "- Two entries with the same `type` are ONE constraint — merge them.\\n\\n"'''

# ── run_pipeline.py ───────────────────────────────────────────────────────────
VALIDATORS_OLD = '''        "Purpose":            PurposeModel,
        "Right":              RightModel,
        "Constraint":         ConstraintModel,'''
VALIDATORS_NEW = '''        "Purpose":            PurposeListModel,
        "Right":              RightListModel,
        "Constraint":         ConstraintListModel,'''

IMPORT_OLD = """    ConstraintModel,
    ActorModel,
    PolicyStatementModel,
)"""
IMPORT_NEW = """    ConstraintModel,
    ActorModel,
    PolicyStatementModel,
    PurposeListModel,
    RightListModel,
    ConstraintListModel,
)"""

ABSENT_OLD = '''    # ── Path 1: explicit absent sentinel ──────────────────────────────────────
    sentinel_keys = ['''
ABSENT_NEW = '''    # ── Path 0: empty list from a multi-instance concept (A1) ─────────────────
    # Purpose/Right/Constraint return {"purposes": []} etc. An empty array is
    # the model reporting absence, not a failure.
    wrapper_key = _LIST_WRAPPER_KEY.get(concept)
    if wrapper_key is not None and wrapper_key in parsed:
        return not parsed[wrapper_key]

    # ── Path 1: explicit absent sentinel ──────────────────────────────────────
    sentinel_keys = ['''

HELPERS_ANCHOR = "def _is_concept_absent(concept: str, parsed: dict) -> bool:"
HELPERS_ADDITION = '''# A1: Pass-1 returns {"<key>": [ ... ]} for these concepts. The wrapper exists
# only because the json_schema response format needs an object at top level.
_LIST_WRAPPER_KEY: dict[str, str] = {
    "Purpose":    "purposes",
    "Right":      "rights",
    "Constraint": "constraints",
}


def _unwrap_list_concept(concept: str, json_str: str) -> str:
    """
    Turn {"purposes": [...]} into a bare JSON array.

    Everything downstream (_wrap_for_assembler, the assembler prompt, the
    repository) already expects an array for these concepts, so the wrapper
    is stripped as soon as validation has passed.
    """
    key = _LIST_WRAPPER_KEY.get(concept)
    if key is None:
        return json_str
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        return json_str
    if isinstance(parsed, dict) and key in parsed:
        return json.dumps(parsed[key])
    if isinstance(parsed, dict):
        return json.dumps([parsed])     # tolerate a bare object
    return json_str


'''

OVERRIDE_OLD = '''    if parsed.get("type") != "Security":
        return extracted_json  # model picked something else — trust it

    rag_lower = rag_text.lower()
'''
OVERRIDE_NEW = '''    # A1: Constraint extraction is now a list; correct each entry in turn.
    if isinstance(parsed, dict) and "constraints" in parsed:
        items = parsed["constraints"]
        changed = False
        for item in items:
            fixed = json.loads(_override_constraint_type(json.dumps(item),
                                                         rag_text))
            if fixed != item:
                item.clear()
                item.update(fixed)
                changed = True
        return json.dumps(parsed) if changed else extracted_json

    if parsed.get("type") != "Security":
        return extracted_json  # model picked something else — trust it

    rag_lower = rag_text.lower()
'''

UNWRAP_OLD = '''        # ── Post-process constraint type override ──────────────────────────
        if concept == "Constraint":
            raw = _override_constraint_type(raw, rag_text)
'''
UNWRAP_NEW = '''        # ── Post-process constraint type override ──────────────────────────
        if concept == "Constraint":
            raw = _override_constraint_type(raw, rag_text)

        # ── A1: strip the list wrapper before anything downstream sees it ──
        raw = _unwrap_list_concept(concept, raw)
        n_items = len(json.loads(raw)) if concept in _LIST_WRAPPER_KEY else 1
        if n_items > 1:
            log.debug(f"    {concept}@{article_ref}: {n_items} instances")
'''

EDITS = [
    ("privacy_schema/models.py", MODELS_ANCHOR, MODELS_ADDITION + MODELS_ANCHOR),
    ("privacy_schema/prompts.py", PURPOSE_OLD, PURPOSE_NEW),
    ("privacy_schema/prompts.py", PURPOSE_RULES_ANCHOR, PURPOSE_RULES_NEW),
    ("privacy_schema/prompts.py", RIGHT_OLD, RIGHT_NEW),
    ("privacy_schema/prompts.py", RIGHT_RULES_OLD, RIGHT_RULES_NEW),
    ("privacy_schema/prompts.py", CONSTRAINT_OLD, CONSTRAINT_NEW),
    ("privacy_schema/prompts.py", CONSTRAINT_RULES_OLD, CONSTRAINT_RULES_NEW),
    ("run_pipeline.py", IMPORT_OLD, IMPORT_NEW),
    ("run_pipeline.py", VALIDATORS_OLD, VALIDATORS_NEW),
    ("run_pipeline.py", HELPERS_ANCHOR, HELPERS_ADDITION + HELPERS_ANCHOR),
    ("run_pipeline.py", ABSENT_OLD, ABSENT_NEW),
    ("run_pipeline.py", OVERRIDE_OLD, OVERRIDE_NEW),
    ("run_pipeline.py", UNWRAP_OLD, UNWRAP_NEW),
]


def main() -> int:
    write = "--write" in sys.argv
    if not Path("run_pipeline.py").exists():
        print("Run this from the repo root.")
        return 1

    texts: dict[str, str] = {}
    bad = 0
    for path, old, new in EDITS:
        s = texts.setdefault(path, Path(path).read_text(encoding="utf-8"))
        label = old.strip().splitlines()[0][:52]
        if new in s:
            status = "ALREADY"
        elif s.count(old) == 1:
            status = "OK"
            texts[path] = s.replace(old, new)
        else:
            status = f"NOT FOUND (matches={s.count(old)})"
            bad += 1
        print(f"  {status:22s} {path}: {label!r}")

    if bad:
        print(f"\n{bad} edit(s) not found — nothing written. Paste this output.")
        return 1
    if not write:
        print("\nCheck passed. Re-run with --write to apply.")
        return 0

    for path, s in texts.items():
        Path(path).write_text(s, encoding="utf-8")

    subprocess.run([sys.executable, "generate_schemas.py"], check=False,
                   stdout=subprocess.DEVNULL)

    print("\nVerifying ...")
    check = subprocess.run(
        [sys.executable, "-c", '''
import json, sys
sys.path.insert(0, ".")
from privacy_schema.models import (PurposeListModel, RightListModel,
                                   ConstraintListModel)
from run_pipeline import (_unwrap_list_concept, _is_concept_absent,
                          _inline_refs, _LIST_WRAPPER_KEY)

two = {"rights": [
    {"rightId": "", "type": "Access", "triggerCondition": "on request",
     "fulfillmentProcess": "respond in 30 days", "source_clause": "4.9"},
    {"rightId": "", "type": "Rectification", "triggerCondition": "on challenge",
     "fulfillmentProcess": "amend the record", "source_clause": "4.9.5"}]}
m = RightListModel.model_validate(two)
print("  two rights validate        :", len(m.rights) == 2)
print("  empty list validates       :", RightListModel.model_validate(
    {"rights": []}).rights == [])
print("  unwrap -> bare array       :",
      json.loads(_unwrap_list_concept("Right", json.dumps(two)))[1]["type"]
      == "Rectification")
print("  empty array reads absent   :",
      _is_concept_absent("Right", {"rights": []}) is True)
print("  non-empty reads present    :",
      _is_concept_absent("Right", two) is False)
for name, model in (("Purpose", PurposeListModel), ("Right", RightListModel),
                    ("Constraint", ConstraintListModel)):
    js = _inline_refs(model.model_json_schema())
    ok = js.get("type") == "object" and "$ref" not in json.dumps(js)
    print(f"  {name:<10} schema is object :", ok)
'''], capture_output=True, text=True)
    print((check.stdout or check.stderr).rstrip())
    print("\nDone. Next: run_variance.py --n 3 --tag a1 (noise floor is 0.0, "
          "so any change is real).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
