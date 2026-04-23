"""
extraction_demo.py — Concrete end-to-end example of the LegalBasis extraction flow.

This script shows exactly what happens at each stage of the pipeline for
a single concept (LegalBasis) extracted from GDPR Art.6(1)(a).

Run:
    python extraction_demo.py

The script simulates what the LLM would return (mock_llm_response) so you
can run the full pipeline without an API key.  Replace mock_llm_call() with
your actual Anthropic / OpenAI client when integrating.
"""

import json
import sys
import warnings

sys.path.insert(0, "/home/claude")

from .prompts import build_concept_prompt, build_assembler_prompt, SYSTEM_PROMPT

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: RAG retrieves this chunk for the query "lawful basis for processing"
# ─────────────────────────────────────────────────────────────────────────────

RETRIEVED_CHUNK = """
Article 6 — Lawfulness of processing

1. Processing shall only be lawful if and to the extent that at least one of
the following applies:

(a) the data subject has given consent to the processing of his or her personal
data for one or more specific purposes;

(b) processing is necessary for the performance of a contract to which the data
subject is party or in order to take steps at the request of the data subject
prior to entering into a contract;

(c) processing is necessary for compliance with a legal obligation to which the
controller is subject;

(d) processing is necessary in order to protect the vital interests of the data
subject or of another natural person;

(e) processing is necessary for the performance of a task carried out in the
public interest or in the exercise of official authority vested in the controller;

(f) processing is necessary for the purposes of the legitimate interests pursued
by the controller or by a third party, except where such interests are overridden
by the interests or fundamental rights and freedoms of the data subject which
require protection of personal data, in particular where the data subject is a
child.
""".strip()

LAW_NAME    = "GDPR"
ARTICLE_REF = "Art.6(1)(a)"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Build the prompts
# ─────────────────────────────────────────────────────────────────────────────

system_prompt, user_prompt = build_concept_prompt(
    concept="LegalBasis",
    law_name=LAW_NAME,
    article_ref=ARTICLE_REF,
    legal_text=RETRIEVED_CHUNK,
)

print("=" * 70)
print("STEP 2 — SYSTEM PROMPT (sent on every API call)")
print("=" * 70)
print(system_prompt)

print()
print("=" * 70)
print("STEP 2 — USER PROMPT (concept-specific, with RAG text injected)")
print("=" * 70)
print(user_prompt)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: LLM call (mocked — replace with real client)
# ─────────────────────────────────────────────────────────────────────────────

def mock_llm_call(system: str, user: str) -> str:
    """
    Simulate what a well-behaved LLM returns for the LegalBasis extraction.
    In production, replace this with:

        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text
    """
    return json.dumps({
        "basisId": "",
        "type": "Consent",
        "evidence": (
            "the data subject has given consent to the processing of his or her "
            "personal data for one or more specific purposes"
        ),
        "jurisdiction": [
            {
                "jurisdictionId": "EU",
                "name": "European Union",
                "description": "EU GDPR jurisdiction — Regulation (EU) 2016/679",
                "source_clause": "GDPR Art.6(1)(a)",
            }
        ],
        "source_clause": "GDPR Art.6(1)(a)",
        "_extraction_confidence": "high",
    }, indent=2)


raw_output = mock_llm_call(system_prompt, user_prompt)

print()
print("=" * 70)
print("STEP 3 — RAW LLM OUTPUT")
print("=" * 70)
print(raw_output)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Parse and validate against the Pydantic schema
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 70)
print("STEP 4 — PYDANTIC VALIDATION")
print("=" * 70)

# Strip the pipeline-internal _extraction_confidence field before validation
parsed_dict = json.loads(raw_output)
parsed_dict.pop("_extraction_confidence", None)

# Demonstrate what happens with a GOOD extraction
try:
    # Import without pydantic — demonstrate the validation logic manually
    # since pydantic is not installed in this sandbox.
    # In production this is just:
    #   from privacy_schema import LegalBasisModel
    #   instance = LegalBasisModel.model_validate(parsed_dict)

    # Manual structural validation (mirrors what Pydantic does)
    LEGAL_BASIS_TYPES = {
        "Consent", "Contract", "LegalObligation",
        "LegitimateInterest", "VitalInterest", "PublicTask",
    }
    required = ["type", "evidence", "jurisdiction"]
    errors = []

    for field in required:
        if not parsed_dict.get(field):
            errors.append(f"Missing required field: {field}")

    if parsed_dict.get("type") not in LEGAL_BASIS_TYPES:
        errors.append(f"Invalid LegalBasisType: {parsed_dict.get('type')}")

    if not parsed_dict.get("evidence", "").strip():
        errors.append("evidence must not be empty (OCL: evidence->notEmpty())")

    jurisdictions = parsed_dict.get("jurisdiction", [])
    if not jurisdictions:
        errors.append("jurisdiction must have at least 1 item (multiplicity 1..*)")

    for j in jurisdictions:
        if not j.get("jurisdictionId") and not j.get("name"):
            errors.append("Jurisdiction missing both jurisdictionId and name")

    if errors:
        print(f"  ✗  VALIDATION FAILED:")
        for e in errors:
            print(f"      - {e}")
    else:
        print(f"  ✓  Validation passed")
        print(f"  ✓  type              = {parsed_dict['type']}")
        print(f"  ✓  evidence          = \"{parsed_dict['evidence'][:60]}...\"")
        print(f"  ✓  jurisdiction[0]   = {parsed_dict['jurisdiction'][0]['jurisdictionId']} "
              f"({parsed_dict['jurisdiction'][0]['name']})")
        print(f"  ✓  source_clause     = {parsed_dict['source_clause']}")

except Exception as e:
    print(f"  ✗  Unexpected error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Demonstrate a BAD extraction — missing evidence, wrong enum value
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 70)
print("STEP 5 — VALIDATION OF A BAD EXTRACTION (hallucinated enum)")
print("=" * 70)

bad_output = {
    "basisId": "",
    "type": "UserAgreement",          # ← not in LegalBasisType enum
    "evidence": "",                   # ← empty — OCL violation
    "jurisdiction": [],               # ← empty — multiplicity 1..* violation
    "source_clause": "GDPR Art.6",
}

errors = []
if bad_output.get("type") not in LEGAL_BASIS_TYPES:
    errors.append(
        f"Invalid LegalBasisType: '{bad_output['type']}'. "
        f"Must be one of: {sorted(LEGAL_BASIS_TYPES)}"
    )
if not bad_output.get("evidence", "").strip():
    errors.append("evidence is empty — every LegalBasis requires evidence text")
if not bad_output.get("jurisdiction"):
    errors.append("jurisdiction is empty — multiplicity is 1..*, at least 1 required")

print(f"  ✗  VALIDATION FAILED ({len(errors)} errors):")
for e in errors:
    print(f"      - {e}")
print()
print("  → This extraction is REJECTED. The pipeline moves it to the")
print("    rejection queue for human review or LLM retry with a corrective")
print("    prompt that includes the error messages above.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Show the two-pass prompt structure
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 70)
print("STEP 6 — PASS 2: PolicyStatement assembler prompt (header only)")
print("=" * 70)
print("In the full pipeline, after all concepts for an article are extracted,")
print("build_assembler_prompt() composes them into a PolicyStatement:\n")

assembler_system, assembler_user = build_assembler_prompt(
    actor_json         = '{"actorId":"","name":"Data Controller","role":"DataController","source_clause":"GDPR Art.4(7)"}',
    purposes_json      = '[{"purposeId":"","description":"Processing for specific consented purposes","category":"ServiceProvision","source_clause":"GDPR Art.6(1)(a)"}]',
    processing_activity_json = '{"activityId":"","description":"Consent-based processing","action":"Use","riskAssessmentReference":null,"dataProcessed":[{"dataId":"","description":"personal data","source":"user provided","category":"Identifier","sensitivity":"Low","identifiability":"Identified","source_clause":"GDPR Art.6(1)"}],"source_clause":"GDPR Art.6(1)"}',
    legal_basis_json   = json.dumps(parsed_dict, indent=2),
    regulations_json   = '[{"regulationId":"","name":"GDPR","version":"2016/679","description":"EU General Data Protection Regulation","jurisdiction":[{"jurisdictionId":"EU","name":"European Union","description":"","source_clause":"GDPR"}],"source_clause":"GDPR"}]',
    constraints_json   = '[{"constraintId":"","type":"PurposeLimitation","expression":"Data processed only for consented purposes","enforcementLevel":"Mandatory","source_clause":"GDPR Art.5(1)(b)"}]',
    rights_json        = '[{"rightId":"","type":"Erasure","triggerCondition":"Consent withdrawn","fulfillmentProcess":"Delete within 30 days","source_clause":"GDPR Art.17"}]',
    source_clause      = "GDPR Art.6(1)(a)",
)

# Print just the first 30 lines of the assembler prompt
lines = assembler_user.splitlines()
for line in lines[:30]:
    print(line)
print(f"  ... [{len(lines) - 30} more lines with Pass 1 JSON objects] ...")

print()
print("=" * 70)
print("Pipeline flow summary:")
print("  RAG retrieves chunks per concept")
print("  → build_concept_prompt()  generates (system, user) for each concept")
print("  → LLM returns raw JSON")
print("  → LegalBasisModel.model_validate() checks schema + OCL constraints")
print("  → build_assembler_prompt() composes validated objects")
print("  → LLM returns assembled PolicyStatement JSON")
print("  → PolicyStatementModel.model_validate() runs final OCL checks")
print("  → Validated instance written to model repository")
print("=" * 70)
