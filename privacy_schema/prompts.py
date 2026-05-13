"""
prompts.py — LLM extraction prompt templates for PrivacyPolicyMetamodel.

Two-pass extraction strategy
  Pass 1 — one API call per metamodel concept, targeting RAG chunks for that concept.
  Pass 2 — assembler composes Pass 1 objects into a full PolicyStatement.

Usage:
    from privacy_schema.prompts import build_concept_prompt, build_assembler_prompt
    system, user = build_concept_prompt("LegalBasis", "GDPR", "Art.6(1)(a)", rag_chunk)
    raw = llm_call(system, user)
    instance = LegalBasisModel.model_validate(json.loads(raw))
"""
from __future__ import annotations
import textwrap

_ENUM_GRAMMARS: dict[str, list[str]] = {
    "ActorRole":            ["DataSubject","DataController","DataProcessor","ThirdParty"],
    "ProcessingAction":     ["Collect","Store","Use","Share","Transfer","Delete"],
    "LegalBasisType":       ["Consent","Contract","LegalObligation","LegitimateInterest","VitalInterest","PublicTask"],
    "PurposeCategory":      ["ServiceProvision","Security","LegalCompliance","Marketing","Analytics","Research"],
    "ConstraintType":       ["Temporal","Geographic","Usage","Security","Retention","PurposeLimitation"],
    "RightType":            ["Access","Rectification","Erasure","Restriction","Portability","Objection","AutomatedDecisionOptOut"],
    "RetentionUnit":        ["Days","Months","Years","Indefinite"],
    "RetentionTrigger":     ["CollectionDate","ContractEnd","LastActivity","LegalObligationExpiry","ConsentWithdrawal","AccountDeletion"],
    "TransferMechanism":    ["AdequacyDecision","StandardContractualClauses","BindingCorporateRules","Consent","ContractNecessity","LegitimateInterest","Other"],
    "WithdrawalChannel":    ["OnlineForm","Email","WrittenRequest","InAppToggle","PhoneRequest","InPerson"],
    "PersonalDataCategory": ["Identifier","ContactInformation","LocationData","FinancialData","HealthData","BiometricData","BehavioralData","TechnicalData","ContentData"],
    "SensitivityLevel":     ["Low","Medium","High","SpecialCategory"],
    "Identifiability":      ["Identified","Pseudonymous","Anonymous"],
}

def _enum_block(names: list[str]) -> str:
    return "\n".join(f"  {n}: {' | '.join(_ENUM_GRAMMARS.get(n,[]))}" for n in names)

SYSTEM_PROMPT = (
    "You are a legal-text information-extraction engine for an MBSE privacy-compliance pipeline.\n\n"
    "Return ONLY a single valid JSON object — no markdown fences, no explanation.\n\n"
    "RULES:\n"
    "1. ENUM VALUES: use only the exact values listed in the prompt (case-sensitive).\n"
    "2. MISSING DATA: required strings → \"\"; required lists (1..*) → extract at least one item\n"
    "   and set \"_extraction_confidence\": \"low\" if uncertain; optional → null.\n"
    "3. TRACEABILITY: every object must populate source_clause with the article reference.\n"
    "4. NO HALLUCINATION: do not infer or invent. For ambiguous text set\n"
    "   \"_extraction_confidence\": \"medium\".\n"
    "5. IDs: leave all *Id fields as empty string \"\"."
)


# =============================================================================
# PASS 1 — CONCEPT PROMPT BUILDERS
# =============================================================================

def _legal_basis_prompt(law: str, ref: str, text: str) -> str:
    eg = _enum_block(["LegalBasisType"])
    return (
        "## Task: Extract a LegalBasis instance\n\n"
        "### What you are extracting\n"
        "The legal justification making a processing activity lawful.\n"
        "Maps to: GDPR Art.6(1)(a-f) | LGPD Art.7 | CCPA business-purpose | PIPEDA Sch.1\n\n"
        "### Output schema\n"
        "{\n"
        '  "basisId": "",\n'
        '  "type": "<LegalBasisType>",\n'
        '  "evidence": "<verbatim or near-verbatim quote from the legal text>",\n'
        '  "jurisdiction": [\n'
        '    {\n'
        '      "jurisdictionId": "<short code: EU | BR | CA-US | CA | UK | SG | IN | AU>",\n'
        '      "name": "<full jurisdiction name>",\n'
        '      "description": "",\n'
        '      "source_clause": "<article reference>"\n'
        '    }\n'
        '  ],\n'
        '  "source_clause": "<article reference>"\n'
        "}\n\n"
        "### Enum grammar\n" + eg + "\n\n"
        "### Decision guide for `type`\n"
        "Text contains...                           type\n"
        "--------------------------------------------------\n"
        "consent / opts in / agrees to              Consent\n"
        "contract / agreement / service terms       Contract\n"
        "legal obligation / required by law         LegalObligation\n"
        "legitimate interest / business purpose     LegitimateInterest\n"
        "vital interest / protect life              VitalInterest\n"
        "public task / official authority           PublicTask\n\n"
        "### OCL constraint hints\n"
        "- evidence is mandatory — must quote or paraphrase the legal text.\n"
        "- jurisdiction is 1..* — include at least one object.\n\n"
        "### Anti-hallucination rules\n"
        "- Extract ONE basis per call (the primary one for this article).\n"
        "- Never set type=Consent unless the text uses a word from the guide above.\n\n"
        "### Few-shot example\n"
        "INPUT:\n"
        "Article 6(1)(a) — Processing shall be lawful where the data subject\n"
        "has given consent to the processing of his or her personal data for\n"
        "one or more specific purposes.\n\n"
        "CORRECT OUTPUT:\n"
        "{\n"
        '  "basisId": "",\n'
        '  "type": "Consent",\n'
        '  "evidence": "the data subject has given consent to the processing of his or her personal data for one or more specific purposes",\n'
        '  "jurisdiction": [{"jurisdictionId": "EU", "name": "European Union", "description": "EU GDPR jurisdiction", "source_clause": "GDPR Art.6(1)(a)"}],\n'
        '  "source_clause": "GDPR Art.6(1)(a)"\n'
        "}\n\n"
        f"### Now extract from the following text:\n"
        f"LAW: {law}\n"
        f"ARTICLE/SECTION: {ref}\n"
        "---\n"
        f"{text}\n"
        "---\n"
    )


def _processing_activity_prompt(law: str, ref: str, text: str) -> str:
    eg = _enum_block(["ProcessingAction","PersonalDataCategory","SensitivityLevel","Identifiability"])
    return (
        "## Task: Extract a ProcessingActivity instance\n\n"
        "### What you are extracting\n"
        "A single processing operation on personal data (collect/store/use/share/transfer/delete).\n"
        "If multiple operations are described, extract only the DOMINANT one.\n\n"
        "### Output schema\n"
        "{\n"
        '  "activityId": "",\n'
        '  "description": "<plain-language description>",\n'
        '  "action": "<ProcessingAction>",\n'
        '  "riskAssessmentReference": "<DPIA/RIPD/PIA citation or null>",\n'
        '  "dataProcessed": [\n'
        '    {\n'
        '      "dataId": "",\n'
        '      "description": "<data type description — e.g. email address, location data>",\n'
        '      "source": "<how obtained — e.g. directly from data subject, third party>",\n'
        '      "category": "<PersonalDataCategory>",\n'
        '      "sensitivity": "<SensitivityLevel>",\n'
        '      "identifiability": "<Identifiability>",\n'
        '      "specialCategory": false,\n'
        '      "source_clause": "<article reference>"\n'
        '    }\n'
        '  ],\n'
        '  "source_clause": "<article reference>"\n'
        "}\n\n"
        "### Enum grammar\n" + eg + "\n\n"
        "### Key rules\n"
        "- dataProcessed is 1..* — always include at least one item.\n"
        "- description and source in each dataProcessed item are REQUIRED non-empty strings.\n"
        "- If the article does not name a specific data type, use:\n"
        '  description="personal data of data subjects", source="DataSubject", category="Identifier"\n'
        f"### Now extract from the following text:\n"
        f"LAW: {law}\n"
        f"ARTICLE/SECTION: {ref}\n"
        "---\n"
        f"{text}\n"
        "---\n"
    )


def _actor_prompt(law: str, ref: str, text: str) -> str:
    eg = _enum_block(["ActorRole"])
    return (
        "## Task: Extract an Actor instance\n\n"
        "### What you are extracting\n"
        "The primary actor whose obligations or rights are described.\n"
        "Maps to: GDPR Art.4(7-8) | LGPD Art.5 | CCPA 'business' | PIPEDA 'organization'\n\n"
        "### Output schema\n"
        "{\n"
        '  "actorId": "",\n'
        '  "name": "<actor name — e.g. Data Controller, Organization, Business>",\n'
        '  "role": "<ActorRole>",\n'
        '  "source_clause": "<article reference>"\n'
        "}\n\n"
        "### Enum grammar\n" + eg + "\n\n"
        "### Decision guide for `role`\n"
        "Text refers to...                           role\n"
        "--------------------------------------------------\n"
        "controller / organization / business        DataController\n"
        "processor / service provider                DataProcessor\n"
        "data subject / individual / consumer        DataSubject\n"
        "third party / recipient                     ThirdParty\n\n"
        f"### Now extract from the following text:\n"
        f"LAW: {law}\n"
        f"ARTICLE/SECTION: {ref}\n"
        "---\n"
        f"{text}\n"
        "---\n"
    )


def _purpose_prompt(law: str, ref: str, text: str) -> str:
    eg = _enum_block(["PurposeCategory"])
    return (
        "## Task: Extract a Purpose instance\n\n"
        "### What you are extracting\n"
        "The reason or objective for which personal data is processed.\n"
        "Maps to: GDPR Art.5(1)(b) | LGPD Art.6 | CCPA business-purpose | PIPEDA Principle 2\n\n"
        "### Output schema\n"
        "{\n"
        '  "purposeId": "",\n'
        '  "description": "<specific purpose as stated in the legal text>",\n'
        '  "category": "<PurposeCategory>",\n'
        '  "source_clause": "<article reference>"\n'
        "}\n\n"
        "### Enum grammar\n" + eg + "\n\n"
        "### Decision guide for `category`\n"
        "Text contains...                            category\n"
        "--------------------------------------------------\n"
        "provide service / fulfil contract           ServiceProvision\n"
        "security / fraud prevention / protection    Security\n"
        "legal obligation / comply with law          LegalCompliance\n"
        "marketing / advertising / promotion         Marketing\n"
        "analytics / statistics / research           Analytics\n"
        "research / scientific / academic            Research\n\n"
        f"### Now extract from the following text:\n"
        f"LAW: {law}\n"
        f"ARTICLE/SECTION: {ref}\n"
        "---\n"
        f"{text}\n"
        "---\n"
    )


def _right_prompt(law: str, ref: str, text: str) -> str:
    eg = _enum_block(["RightType"])
    return (
        "## Task: Extract a Right instance\n\n"
        "### What you are extracting\n"
        "A data-subject right affected by this processing statement.\n"
        "Maps to: GDPR Art.15-22 | LGPD Art.17-22 | CCPA §1798.100-145 | PIPEDA Principle 9\n\n"
        "If NO right is described, return exactly:\n"
        '{"_no_right_stated": true}\n\n'
        "### Output schema\n"
        "{\n"
        '  "rightId": "",\n'
        '  "type": "<RightType>",\n'
        '  "triggerCondition": "<condition under which the right may be exercised>",\n'
        '  "fulfillmentProcess": "<how the controller must respond>",\n'
        '  "source_clause": "<article reference>"\n'
        "}\n\n"
        "### Enum grammar\n" + eg + "\n\n"
        "### Decision guide for `type`\n"
        "Text contains...                                    type\n"
        "------------------------------------------------------------\n"
        "right of access / right to obtain copy             Access\n"
        "rectification / correct inaccurate data            Rectification\n"
        "erasure / right to be forgotten / deletion         Erasure\n"
        "restriction of processing / limit processing       Restriction\n"
        "data portability / receive in machine-readable     Portability\n"
        "object to processing / opt out of processing       Objection\n"
        "automated decision / profiling opt-out             AutomatedDecisionOptOut\n\n"
        "### Key rules\n"
        "- Extract ONE Right per call — the primary right described in this article.\n"
        "- triggerCondition: the circumstance that activates the right\n"
        "  (e.g. 'data subject makes a written request', 'processing is based on consent').\n"
        "- fulfillmentProcess: what the controller must do and within what timeframe\n"
        "  (e.g. 'controller must respond within 30 days', 'delete data without undue delay').\n"
        "- Both triggerCondition and fulfillmentProcess are REQUIRED non-empty strings.\n\n"
        f"### Now extract from the following text:\n"
        f"LAW: {law}\n"
        f"ARTICLE/SECTION: {ref}\n"
        "---\n"
        f"{text}\n"
        "---\n"
    )


def _constraint_prompt(law: str, ref: str, text: str) -> str:
    eg = _enum_block(["ConstraintType"])
    return (
        "## Task: Extract a Constraint instance\n\n"
        "### What you are extracting\n"
        "Not the legal basis (why), not the purpose (what for),\n"
        "but a specific operational rule that limits or shapes processing.\n"
        "Examples: retention limits, geographic restrictions, encryption requirements,\n"
        "purpose-limitation rules, usage restrictions.\n\n"
        "### Output schema\n"
        "{\n"
        '  "constraintId": "",\n'
        '  "type": "<ConstraintType>",\n'
        '  "expression": "<natural-language statement of the constraint>",\n'
        '  "enforcementLevel": "<Mandatory | Recommended | BestEffort>",\n'
        '  "source_clause": "<article reference>"\n'
        "}\n\n"
        "### Enum grammar\n" + eg + "\n\n"
        "### Decision guide for `type`\n"
        "Text contains...                                    type\n"
        "------------------------------------------------------------\n"
        "time limit / retention period / no longer than      Retention\n"
        "only in [country] / within the EEA / geographic     Geographic\n"
        "only for [purpose] / not used for / purpose limit   PurposeLimitation\n"
        "encrypt / pseudonymise / anonymise / secure         Security\n"
        "must not share / limited access / restrict use      Usage\n"
        "date / deadline / within N days / time-based        Temporal\n\n"
        "### Key rules\n"
        "- expression must be a self-contained natural-language rule that a\n"
        "  compliance engineer could evaluate — not just a paraphrase of the heading.\n"
        "- enforcementLevel: 'Mandatory' for SHALL/MUST, 'Recommended' for SHOULD,\n"
        "  'BestEffort' for MAY/CAN.\n"
        "- Extract the PRIMARY constraint for this call.\n\n"
        f"### Now extract from the following text:\n"
        f"LAW: {law}\n"
        f"ARTICLE/SECTION: {ref}\n"
        "---\n"
        f"{text}\n"
        "---\n"
    )


def _retention_policy_prompt(law: str, ref: str, text: str) -> str:
    eg = _enum_block(["RetentionUnit","RetentionTrigger"])
    return (
        "## Task: Extract a RetentionPolicy instance\n\n"
        "### What you are extracting\n"
        "A rule specifying how long personal data must or may be retained.\n"
        "Maps to: GDPR Art.5(1)(e) | LGPD Art.15 | CCPA retention | PIPEDA Principle 5\n\n"
        "If NO retention rule is stated, return exactly:\n"
        '{"_no_retention_stated": true}\n\n'
        "### Output schema\n"
        "{\n"
        '  "policyId": "",\n'
        '  "duration": <integer or null>,\n'
        '  "unit": "<RetentionUnit or null>",\n'
        '  "trigger": "<RetentionTrigger>",\n'
        '  "description": "<plain-language retention rule>",\n'
        '  "source_clause": "<article reference>"\n'
        "}\n\n"
        "### Enum grammar\n" + eg + "\n\n"
        f"### Now extract from the following text:\n"
        f"LAW: {law}\n"
        f"ARTICLE/SECTION: {ref}\n"
        "---\n"
        f"{text}\n"
        "---\n"
    )


def _data_transfer_prompt(law: str, ref: str, text: str) -> str:
    eg = _enum_block(["TransferMechanism"])
    return (
        "## Task: Extract a DataTransfer instance\n\n"
        "### What you are extracting\n"
        "A cross-border transfer of personal data, including the legal mechanism.\n"
        "Maps to: GDPR Art.44-49 | LGPD Art.33-36 | CCPA §1798.140(ad) | PIPEDA Principle 1\n\n"
        "If NO cross-border transfer is described, return exactly:\n"
        '{"_no_transfer_stated": true}\n\n'
        "### Output schema\n"
        "{\n"
        '  "transferId": "",\n'
        '  "destinationCountry": "<country or region name>",\n'
        '  "mechanism": "<TransferMechanism>",\n'
        '  "adequacyReference": "<adequacy decision name or null>",\n'
        '  "source_clause": "<article reference>"\n'
        "}\n\n"
        "### Enum grammar\n" + eg + "\n\n"
        "### Decision guide for `mechanism`\n"
        "Text contains...                                    mechanism\n"
        "------------------------------------------------------------\n"
        "adequacy decision / adequate protection             AdequacyDecision\n"
        "standard contractual clauses / SCCs                StandardContractualClauses\n"
        "binding corporate rules / BCRs                     BindingCorporateRules\n"
        "consent of data subject                            Consent\n"
        "necessary for contract performance                 ContractNecessity\n"
        "legitimate interest                                LegitimateInterest\n"
        "other safeguard / derogation                       Other\n\n"
        f"### Now extract from the following text:\n"
        f"LAW: {law}\n"
        f"ARTICLE/SECTION: {ref}\n"
        "---\n"
        f"{text}\n"
        "---\n"
    )


def _consent_withdrawal_prompt(law: str, ref: str, text: str) -> str:
    eg = _enum_block(["WithdrawalChannel"])
    return (
        "## Task: Extract a ConsentWithdrawal instance\n\n"
        "### What you are extracting\n"
        "The mechanism by which a data subject may withdraw consent.\n"
        "Maps to: GDPR Art.7(3) | LGPD Art.8§5 | CCPA §1798.120 | PIPEDA Principle 3\n\n"
        "If NO withdrawal mechanism is described, return exactly:\n"
        '{"_no_withdrawal_stated": true}\n\n'
        "### Output schema\n"
        "{\n"
        '  "withdrawalId": "",\n'
        '  "channel": ["<WithdrawalChannel>"],\n'
        '  "deadline": "<time the controller has to act — e.g. without undue delay>",\n'
        '  "effectOnPriorProcessing": "<whether withdrawal affects prior processing>",\n'
        '  "source_clause": "<article reference>"\n'
        "}\n\n"
        "### Enum grammar\n" + eg + "\n\n"
        "### Key rules\n"
        "- channel is a LIST — always use [...] even for a single channel.\n"
        "- deadline and effectOnPriorProcessing are REQUIRED non-empty strings.\n\n"
        "### Few-shot example\n"
        "INPUT:\n"
        "Article 7(3) — The data subject shall have the right to withdraw his or her\n"
        "consent at any time. The withdrawal of consent shall not affect the lawfulness\n"
        "of processing based on consent before its withdrawal.\n\n"
        "CORRECT OUTPUT:\n"
        "{\n"
        '  "withdrawalId":"","channel":["InAppToggle","WrittenRequest"],\n'
        '  "deadline":"without undue delay",\n'
        '  "effectOnPriorProcessing":"Does not affect lawfulness of processing prior to withdrawal (GDPR Art.7(3))",\n'
        f'  "source_clause":"{ref}"\n'
        "}\n\n"
        f"### Now extract from the following text:\n"
        f"LAW: {law}\n"
        f"ARTICLE/SECTION: {ref}\n"
        "---\n"
        f"{text}\n"
        "---\n"
    )


def _personal_data_prompt(law: str, ref: str, text: str) -> str:
    eg = _enum_block(["PersonalDataCategory","SensitivityLevel","Identifiability"])
    return (
        "## Task: Extract a PersonalData instance\n\n"
        "### What you are extracting\n"
        "A category of personal data mentioned in the article.\n\n"
        "### Output schema\n"
        "{\n"
        '  "dataId": "",\n'
        '  "category": "<PersonalDataCategory>",\n'
        '  "description": "<specific data type — e.g. email address, GPS coordinates>",\n'
        '  "source": "<how obtained — e.g. directly from data subject>",\n'
        '  "sensitivity": "<SensitivityLevel>",\n'
        '  "identifiability": "<Identifiability>",\n'
        '  "specialCategory": false,\n'
        '  "source_clause": "<article reference>"\n'
        "}\n\n"
        "### Enum grammar\n" + eg + "\n\n"
        "### Key rules\n"
        "- description and source are REQUIRED non-empty strings.\n"
        "- If the article does not name a specific data type, use:\n"
        '  description="personal data of data subjects", source="directly from data subject"\n\n'
        f"### Now extract from the following text:\n"
        f"LAW: {law}\n"
        f"ARTICLE/SECTION: {ref}\n"
        "---\n"
        f"{text}\n"
        "---\n"
    )


_CONCEPT_BUILDERS = {
    "LegalBasis":         _legal_basis_prompt,
    "ProcessingActivity": _processing_activity_prompt,
    "RetentionPolicy":    _retention_policy_prompt,
    "ConsentWithdrawal":  _consent_withdrawal_prompt,
    "Right":              _right_prompt,
    "Purpose":            _purpose_prompt,
    "DataTransfer":       _data_transfer_prompt,
    "Constraint":         _constraint_prompt,
    "Actor":              _actor_prompt,
    "PersonalData":       _personal_data_prompt,
}

_CONCEPT_PROMPTS = {
    name: fn("GDPR", "Art.X", "<legal text>")
    for name, fn in _CONCEPT_BUILDERS.items()
}


def build_concept_prompt(
    concept: str, law_name: str, article_ref: str, legal_text: str
) -> tuple[str, str]:
    builder = _CONCEPT_BUILDERS.get(concept)
    if builder is None:
        raise ValueError(
            f"No prompt builder for '{concept}'. "
            f"Available: {list(_CONCEPT_BUILDERS)}"
        )
    return SYSTEM_PROMPT, builder(law_name, article_ref, legal_text)


# =============================================================================
# PASS 2 — ASSEMBLER PROMPT
# =============================================================================

def build_assembler_prompt(
    actor_json: str,
    purposes_json: str,
    processing_activity_json: str,
    legal_basis_json: str,
    regulations_json: str,
    constraints_json: str,
    rights_json: str,
    source_clause: str,
    retention_json: str = "[]",
    transfers_json: str = "[]",
    withdrawal_json: str = "[]",
) -> tuple[str, str]:
    user = (
        "## Task: Assemble a PolicyStatement from Pass 1 extractions\n\n"
        "Compose the Pass 1 objects below into a single PolicyStatement JSON.\n"
        "For fields that have content: copy values exactly as given — do not paraphrase or expand.\n"
        "EXCEPTION — required arrays that are []: you MUST synthesize at least one item "
        "(see SYNTHESIS RULES below) rather than copying the empty array.\n\n"

        # ── Output schema ─────────────────────────────────────────────────────
        "## Output schema — you MUST return exactly this structure\n\n"
        "```json\n"
        "{\n"
        '  "statementId": "",\n'
        '  "description": "<one-sentence summary of what this statement covers>",\n'
        '  "source_clause": "<top-level article reference>",\n\n'
        '  "actor": { <copy ACTOR object exactly> },\n\n'
        '  "purposes": [ <copy PURPOSES array exactly> ],\n\n'
        '  "processingActivity": { <copy PROCESSING ACTIVITY object exactly> },\n\n'
        '  "legalBasis": {\n'
        '    "basisId": "",\n'
        '    "type": "<LegalBasisType>",\n'
        '    "evidence": "<string>",\n'
        '    "jurisdiction": [\n'
        '      { "jurisdictionId": "EU", "name": "European Union", '
        '"description": "", "source_clause": "" }\n'
        '    ],\n'
        '    "source_clause": "<string>"\n'
        '  },\n\n'
        '  "governingRegulations": [\n'
        '    {\n'
        '      "regulationId": "",\n'
        '      "name": "<law name e.g. GDPR>",\n'
        '      "jurisdiction": [\n'
        '        { "jurisdictionId": "EU", "name": "European Union", '
        '"description": "", "source_clause": "" }\n'
        '      ],\n'
        '      "source_clause": "<article reference>"\n'
        '    }\n'
        '  ],\n\n'
        '  "constraints": [ <copy CONSTRAINTS array exactly> ],\n\n'
        '  "rightImpacted": [ <copy RIGHTS IMPACTED array exactly — REQUIRED, never omit> ],\n\n'
        '  "retentionPolicies": [ <copy RETENTION POLICIES array, or [] if empty> ],\n\n'
        '  "dataTransfers": [ <copy DATA TRANSFERS array, or [] if empty> ],\n\n'
        '  "consentWithdrawal": [ <copy CONSENT WITHDRAWAL array — see schema below> ]\n'
        "}\n"
        "```\n\n"

        # ── Field rules ───────────────────────────────────────────────────────
        "FIELD RULES:\n"
        "- \"statementId\" must always be exactly \"\" (empty string — the pipeline fills it).\n"
        "- Required arrays (purposes, governingRegulations, constraints, rightImpacted) "
        "must contain at least one item — never [] and NEVER OMITTED from the output.\n"
        "- \"rightImpacted\" is ALWAYS required. Even if rights input is [], synthesize "
        "one item using SYNTHESIS RULES. Omitting this key is a hard validation failure.\n"
        "- Optional arrays (retentionPolicies, dataTransfers, consentWithdrawal) "
        "may be [] if the Pass 1 input is empty.\n"
        "- processingActivity.dataProcessed must contain at least one item — never [].\n"
        "- Use camelCase field names EXACTLY as shown. "
        "Do NOT use snake_case (e.g. 'legal_basis' is WRONG, 'legalBasis' is correct).\n\n"

        "ENUM ENFORCEMENT — use ONLY these exact values (case-sensitive):\n"
        "  constraints[].type        : Temporal | Geographic | Usage | Security | Retention | PurposeLimitation\n"
        "  purposes[].category       : ServiceProvision | Security | LegalCompliance | Marketing | Analytics | Research\n"
        "  legalBasis.type           : Consent | Contract | LegalObligation | LegitimateInterest | VitalInterest | PublicTask\n"
        "  rightImpacted[].type      : Access | Rectification | Erasure | Restriction | Portability | Objection | AutomatedDecisionOptOut\n"
        "  processingActivity.action : Collect | Store | Use | Share | Transfer | Delete\n"
        "Any value not in the list above is INVALID and will cause a hard failure. "
        "If unsure, pick the closest match — never invent a new value.\n\n"

        # ── Synthesis rules ───────────────────────────────────────────────────
        "SYNTHESIS RULES — apply when a required array is [] in the Pass 1 input:\n"

        "  rightImpacted []:  synthesize 1 item — ALL fields required:\n"
        "    { \"rightId\": \"\", "
        "\"type\": \"<RightType — infer: accountability/request-handling→Access, "
        "limitation/opt-out→Restriction, correction→Rectification>\", "
        "\"triggerCondition\": \"<REQUIRED non-empty — e.g. 'Upon written request by data subject'>\", "
        "\"fulfillmentProcess\": \"<REQUIRED non-empty — e.g. 'Organization must respond within 30 days'>\", "
        "\"source_clause\": \"\" }\n"
        "    triggerCondition and fulfillmentProcess are REQUIRED — "
        "never empty string, never omitted.\n\n"

        "  purposes []:       synthesize 1 item — infer category from legalBasis.type and article subject:\n"
        "                     LegalObligation/compliance article → LegalCompliance\n"
        "                     service/product delivery → ServiceProvision\n"
        "                     fraud/data protection → Security\n"
        "                     description must paraphrase what the article governs.\n\n"

        "  constraints []:    synthesize 1 item — infer type from the dominant obligation:\n"
        "                     compliance/purpose-scoping → PurposeLimitation\n"
        "                     security/protection requirement → Security\n"
        "                     time-based rule → Temporal\n"
        "                     expression MUST be a non-empty natural-language rule\n"
        "                     derived from the article — e.g.:\n"
        "                     PurposeLimitation → 'Personal data must only be used\n"
        "                       for the purpose identified at time of collection.'\n"
        "                     Security → 'Organization must protect personal data\n"
        "                       against loss, theft, and unauthorized access.'\n"
        "                     enforcementLevel MUST be 'Mandatory' unless article\n"
        "                     uses 'should' or 'may' — never empty string.\n\n"

        "  dataProcessed []:  synthesize 1 item — ALL fields required:\n"
        "    { \"dataId\": \"\", "
        "\"category\": \"Identifier\", "
        "\"description\": \"<REQUIRED non-empty — e.g. 'personal data of data subjects'>\", "
        "\"source\": \"<REQUIRED non-empty — e.g. 'directly from data subject'>\", "
        "\"sensitivity\": \"Low\", "
        "\"identifiability\": \"Identified\", "
        "\"specialCategory\": false, "
        "\"source_clause\": \"\" }\n"
        "    description and source are REQUIRED — never leave them empty.\n\n"

        # ── ConsentWithdrawal schema ──────────────────────────────────────────
        "## ConsentWithdrawal object schema (required when consentWithdrawal is not [])\n\n"
        "Each item in the consentWithdrawal array MUST have exactly these fields:\n"
        "```json\n"
        "{\n"
        "  \"withdrawalId\": \"\",\n"
        "  \"channel\": [\"<one or more of: OnlineForm | Email | WrittenRequest "
        "| InAppToggle | PhoneRequest | InPerson>\"],\n"
        "  \"deadline\": \"<string — time the controller has to act, "
        "e.g. 'without undue delay', '30 days', 'immediately'>\",\n"
        "  \"effectOnPriorProcessing\": \"<string — whether withdrawal affects "
        "lawfulness of prior processing, e.g. 'Does not affect prior processing'>\",\n"
        "  \"source_clause\": \"<article reference>\"\n"
        "}\n"
        "```\n"
        "CONSENT WITHDRAWAL RULES:\n"
        "- \"channel\" is a LIST — always use [\"...\"] even for a single channel.\n"
        "- \"channel\" values must be exact enum strings from the list above — "
        "no freeform text.\n"
        "- \"deadline\" and \"effectOnPriorProcessing\" are required strings — "
        "never null or omitted.\n"
        "- \"withdrawalId\" must always be \"\" (pipeline fills it).\n\n"

        # ── Consistency checks ────────────────────────────────────────────────
        "## Consistency checks — add \"_warnings\":[...] to the root object if any fire\n"
        "1. legalBasis.type==\"Consent\" and consentWithdrawal==[]\n"
        "   → \"constraint_4: Consent basis present but no withdrawal mechanics extracted\"\n"
        "2. processingActivity.action in [Transfer,Share] and any "
        "dataProcessed.sensitivity in [High,SpecialCategory] "
        "and riskAssessmentReference==null\n"
        "   → \"constraint_3: High-risk transfer without risk assessment reference\"\n"
        "3. Any governingRegulation has empty jurisdiction list\n"
        "   → \"constraint_2: Regulation missing jurisdiction\"\n\n"

        # ── Pass 1 inputs ─────────────────────────────────────────────────────
        "## Pass 1 inputs — copy these into the output schema above\n\n"
        f"ACTOR:\n{actor_json}\n\n"
        f"PURPOSES:\n{purposes_json}\n\n"
        f"PROCESSING ACTIVITY:\n{processing_activity_json}\n\n"
        f"LEGAL BASIS:\n{legal_basis_json}\n\n"
        f"GOVERNING REGULATIONS:\n{regulations_json}\n\n"
        f"CONSTRAINTS:\n{constraints_json}\n\n"
        f"RIGHTS IMPACTED:  ⚠ rightImpacted is a REQUIRED top-level key in your output\n"
        f"{rights_json}\n\n"
        f"RETENTION POLICIES:\n{retention_json}\n\n"
        f"DATA TRANSFERS:\n{transfers_json}\n\n"
        f"CONSENT WITHDRAWAL:\n{withdrawal_json}\n\n"
        f"SOURCE CLAUSE: {source_clause}\n\n"
        "Return ONLY the JSON object. No markdown fences, no explanation.\n"
    )
    return SYSTEM_PROMPT, user