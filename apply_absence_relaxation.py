#!/usr/bin/env python3
"""
apply_absence_relaxation.py — A2 step: let the pipeline say "not stated".

Relaxes purposes / constraints / rightImpacted from [1..*] to [0..*] and
removes the assembler's forced-synthesis instructions, so an empty array
becomes a valid answer instead of something the model must invent a value
to avoid.

Touches four files:
    generate_ecore.py        lower=1 -> lower=0 on the three references
    privacy_schema/models.py drop min_length=1, default to []
    privacy_schema/prompts.py  remove the EXCEPTION clause, the two
                             "required array" FIELD RULES lines, and the
                             rightImpacted / purposes / constraints
                             SYNTHESIS RULES blocks
    run_pipeline.py          reframe the per-call synthesis block

Deliberately NOT changed: governingRegulations (filled deterministically,
never by the LLM), dataProcessed, jurisdiction, channel, and DataTransfer's
required fields.  Those are a separate step; changing them together would
make the run uninterpretable.

Run from the repo root:
    python apply_absence_relaxation.py            # check only, writes nothing
    python apply_absence_relaxation.py --write    # apply + regenerate

All-or-nothing: if any edit is NOT FOUND, nothing is written.
Idempotent: edits already present are reported as ALREADY and skipped.
"""
import subprocess
import sys
from pathlib import Path

EDITS: list[tuple[str, str, str]] = []

# ── generate_ecore.py — the metamodel source of truth ─────────────────────────
for _decl in (
    'EReference("purposes",           Purpose,           lower=%s, upper=-1, containment=True)',
    'EReference("constraints",        Constraint,        lower=%s, upper=-1, containment=True)',
    'EReference("rightImpacted",      Right,             lower=%s, upper=-1, containment=True)',
):
    EDITS.append(("generate_ecore.py", _decl % "1", _decl % "0"))

# ── privacy_schema/models.py — hand-maintained, NOT regenerated ───────────────
EDITS += [
    ("privacy_schema/models.py",
     "    purposes: List[PurposeModel] = Field(min_length=1)",
     "    purposes: List[PurposeModel] = Field(default_factory=list)"),
    ("privacy_schema/models.py",
     "    constraints: List[ConstraintModel] = Field(min_length=1)",
     "    constraints: List[ConstraintModel] = Field(default_factory=list)"),
    ("privacy_schema/models.py",
     '    right_impacted: List[RightModel] = Field(alias="rightImpacted", min_length=1)',
     '    right_impacted: List[RightModel] = Field(alias="rightImpacted", default_factory=list)'),
]

# ── privacy_schema/prompts.py — remove the contradictions ─────────────────────
EDITS += [
    # (a) the EXCEPTION clause
    ("privacy_schema/prompts.py",
     '        "EXCEPTION — required arrays that are []: you MUST synthesize at least one item "\n'
     '        "(see SYNTHESIS RULES below) rather than copying the empty array.\\n\\n"\n\n',
     ''),
    # (b) the two FIELD RULES lines
    ("privacy_schema/prompts.py",
     '        "- Required arrays (purposes, governingRegulations, constraints, rightImpacted) "\n'
     '        "must contain at least one item — never [] and NEVER OMITTED from the output.\\n"\n'
     '        "- \\"rightImpacted\\" is ALWAYS required. Even if rights input is [], synthesize "\n'
     '        "one item using SYNTHESIS RULES. Omitting this key is a hard validation failure.\\n"\n',
     '        "- \\"governingRegulations\\" must contain at least one item.\\n"\n'
     '        "- purposes, constraints and rightImpacted may be [] when the article does "\n'
     '        "not state one. An empty array is a valid, meaningful answer — it records "\n'
     '        "that the law is silent, and is preferred over a guess.\\n"\n'),
    # (c) the section title
    ("privacy_schema/prompts.py",
     '        "SYNTHESIS RULES — apply when a required array is [] in the Pass 1 input:\\n"',
     '        "SYNTHESIS RULES — apply only to dataProcessed:\\n"'),
    # (d) the rightImpacted / purposes / constraints blocks
    ("privacy_schema/prompts.py",
     '        "  rightImpacted []:  synthesize 1 item — ALL fields required:\\n"\n'
     '        "    { \\"rightId\\": \\"\\", "\n'
     '        "\\"type\\": \\"<RightType — infer: accountability/request-handling→Access, "\n'
     '        "limitation/opt-out→Restriction, correction→Rectification>\\", "\n'
     '        "\\"triggerCondition\\": \\"<REQUIRED non-empty — e.g. \'Upon written request by data subject\'>\\", "\n'
     '        "\\"fulfillmentProcess\\": \\"<REQUIRED non-empty — e.g. \'Organization must respond within 30 days\'>\\", "\n'
     '        "\\"source_clause\\": \\"\\" }\\n"\n'
     '        "    triggerCondition and fulfillmentProcess are REQUIRED — "\n'
     '        "never empty string, never omitted.\\n\\n"\n\n'
     '        "  purposes []:       synthesize 1 item — infer category from legalBasis.type and article subject:\\n"\n'
     '        "                     LegalObligation/compliance article → LegalCompliance\\n"\n'
     '        "                     service/product delivery → ServiceProvision\\n"\n'
     '        "                     fraud/data protection → Security\\n"\n'
     '        "                     description must paraphrase what the article governs.\\n\\n"\n\n'
     '        "  constraints []:    synthesize 1 item — infer type from the dominant obligation:\\n"\n'
     '        "                     compliance/purpose-scoping → PurposeLimitation\\n"\n'
     '        "                     security/protection requirement → Security\\n"\n'
     '        "                     time-based rule → Temporal\\n"\n'
     '        "                     expression MUST be a non-empty natural-language rule\\n"\n'
     '        "                     derived from the article — e.g.:\\n"\n'
     '        "                     PurposeLimitation → \'Personal data must only be used\\n"\n'
     '        "                       for the purpose identified at time of collection.\'\\n"\n'
     '        "                     Security → \'Organization must protect personal data\\n"\n'
     '        "                       against loss, theft, and unauthorized access.\'\\n"\n'
     '        "                     enforcementLevel MUST be \'Mandatory\' unless article\\n"\n'
     '        "                     uses \'should\' or \'may\' — never empty string.\\n\\n"\n\n',
     ''),
]

# ── run_pipeline.py — reframe the per-call block ──────────────────────────────
EDITS.append((
    "run_pipeline.py",
    '            + "\\n\\nApply the SYNTHESIS RULES.\\n\\n"',
    '            + "\\n\\nAn empty array is the correct answer when the article does "\n'
    '              "not state one. Do not invent a value to fill it.\\n\\n"',
))


def main() -> int:
    write = "--write" in sys.argv
    if not Path("generate_ecore.py").exists():
        print("Run this from the repo root.")
        return 1

    texts: dict[str, str] = {}
    bad = 0
    for path, old, new in EDITS:
        s = texts.setdefault(path, Path(path).read_text(encoding="utf-8"))
        label = old.strip().splitlines()[0][:58]
        if (new and new in s) or (not new and old not in s):
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

    py = sys.executable
    subprocess.run([py, "generate_ecore.py", "metamodel/privacy_metamodel.ecore"],
                   check=True)
    subprocess.run([py, "generate_schemas.py"], check=True,
                   stdout=subprocess.DEVNULL)
    # models.py is hand-maintained: generate_pydantic.py's models.py output does
    # not even parse, so it is never regenerated here.  enums.py is untouched by
    # this change, so no regeneration is needed for it either.

    print("\nVerifying the three fields are now optional ...")
    check = subprocess.run(
        [py, "-c",
         "from privacy_schema.models import PolicyStatementModel as P;"
         "print({f: P.model_fields[f].is_required() "
         "for f in ('purposes','constraints','right_impacted')})"],
        capture_output=True, text=True)
    print("  " + (check.stdout or check.stderr).strip())
    print("\nDone. Now run: git diff --stat   (expect 8 files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())