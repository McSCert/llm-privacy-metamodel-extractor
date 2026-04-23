"""
prompts.py — LLM extraction prompt templates for PrivacyPolicyMetamodel v4.

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
        '      "description": "<data category e.g. email address>",\n'
        '      "source": "<how obtained e.g. provided by user>",\n'
        '      "category": "<PersonalDataCategory>",\n'
        '      "sensitivity": "<SensitivityLevel>",\n'
        '      "identifiability": "<Identifiability>",\n'
        '      "source_clause": "<article reference>"\n'
        '    }\n'
        '  ],\n'
        '  "source_clause": "<article reference>"\n'
        "}\n\n"
        "### Enum grammar\n" + eg + "\n\n"
        "### OCL constraint_3 — riskAssessmentReference\n"
        "If action is Transfer or Share AND any data has sensitivity High or SpecialCategory:\n"
        "search the text for DPIA/RIPD/risk assessment/Art.35 mention.\n"
        "Populate if found; set null if absent (validator will emit a warning).\n\n"
        "### Sensitivity guide\n"
        "Name, email, phone          ContactInformation  Low\n"
        "IP address, device ID       TechnicalData       Low\n"
        "Precise location (GPS)      LocationData        Medium\n"
        "Financial / card data       FinancialData       High\n"
        "Health / medical records    HealthData          SpecialCategory\n"
        "Biometric data              BiometricData       SpecialCategory\n"
        "Racial / ethnic origin      Identifier          SpecialCategory\n"
        "Behavioural / profiling     BehavioralData      Medium\n\n"
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
        "Maps to: GDPR Art.5(1)(e) | LGPD Art.15 | CCPA §1798.100(e) | PIPEDA Principle 5\n\n"
        "If NO retention period is stated, return exactly:\n"
        '{"_no_retention_stated": true}\n\n'
        "### Output schema\n"
        "{\n"
        '  "retentionId": "",\n'
        '  "duration": <integer — -1 for indefinite>,\n'
        '  "unit": "<RetentionUnit>",\n'
        '  "trigger": "<RetentionTrigger>",\n'
        '  "basisArticle": "<legal article citation or null>",\n'
        '  "source_clause": "<article reference>"\n'
        "}\n\n"
        "### Enum grammar\n" + eg + "\n\n"
        "### Parsing guide\n"
        '"no longer than necessary"           -1  Indefinite  LastActivity\n'
        '"deleted within 30 days of request"  30  Days        AccountDeletion\n'
        '"3 years after contract end"          3  Years       ContractEnd\n'
        '"6 months from collection"            6  Months      CollectionDate\n'
        '"until consent withdrawn"            -1  Indefinite  ConsentWithdrawal\n\n'
        "### Few-shot example\n"
        'INPUT: "Personal data shall be kept no longer than necessary (Art.5(1)(e) GDPR)."\n\n'
        "CORRECT OUTPUT:\n"
        "{\n"
        '  "retentionId":"","duration":-1,"unit":"Indefinite","trigger":"LastActivity",\n'
        '  "basisArticle":"GDPR Art.5(1)(e)","source_clause":"GDPR Art.5(1)(e)"\n'
        "}\n\n"
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
        "Maps to: GDPR Art.7(3) | LGPD Art.8§5 | CCPA §1798.120 | PIPEDA Principle 3\n\n"
        "If NO withdrawal mechanics are described, return exactly:\n"
        '{"_no_withdrawal_stated": true}\n\n'
        "### Output schema\n"
        "{\n"
        '  "withdrawalId": "",\n'
        '  "channel": ["<WithdrawalChannel>", ...],\n'
        '  "deadline": "<e.g. without undue delay | 30 days | immediately>",\n'
        '  "effectOnPriorProcessing": "<law position on prior processing>",\n'
        '  "source_clause": "<article reference>"\n'
        "}\n\n"
        "### Enum grammar\n" + eg + "\n\n"
        "### Key rules\n"
        "- channel is a LIST — include ALL channels mentioned.\n"
        "- effectOnPriorProcessing: always state the law position explicitly.\n"
        "  GDPR Art.7(3): withdrawal does NOT affect prior processing lawfulness.\n"
        "  LGPD Art.8§5: requires confirmation of cessation impossibility.\n\n"
        "### Few-shot example\n"
        'INPUT: "Consent may be withdrawn via account settings or written request.\n'
        'Withdrawal does not affect lawfulness of prior processing."\n\n'
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
        "  (e.g. 'provide copy within 30 days', 'erase without undue delay').\n\n"
        "### Few-shot example\n"
        "INPUT:\n"
        "Article 17 — The data subject shall have the right to obtain erasure of\n"
        "personal data without undue delay where the data are no longer necessary\n"
        "for the purposes for which they were collected.\n\n"
        "CORRECT OUTPUT:\n"
        "{\n"
        '  "rightId": "",\n'
        '  "type": "Erasure",\n'
        '  "triggerCondition": "Personal data are no longer necessary for the purposes for which they were collected",\n'
        '  "fulfillmentProcess": "Controller must erase personal data without undue delay",\n'
        f'  "source_clause": "{ref}"\n'
        "}\n\n"
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
        "The processing purpose — WHY personal data is being processed.\n"
        "Maps to: GDPR Art.5(1)(b) purpose limitation | LGPD Art.6 | CCPA disclosed purpose\n"
        "         | PIPEDA Principle 2\n\n"
        "### Output schema\n"
        "{\n"
        '  "purposeId": "",\n'
        '  "description": "<specific purpose as stated in the legal text>",\n'
        '  "category": "<PurposeCategory>",\n'
        '  "source_clause": "<article reference>"\n'
        "}\n\n"
        "### Enum grammar\n" + eg + "\n\n"
        "### Decision guide for `category`\n"
        "Text contains...                                    category\n"
        "------------------------------------------------------------\n"
        "provide service / fulfil contract / deliver         ServiceProvision\n"
        "fraud prevention / security / protect system        Security\n"
        "comply with law / legal obligation / tax / audit    LegalCompliance\n"
        "marketing / advertising / promote / direct mail     Marketing\n"
        "analytics / statistics / improve service / measure  Analytics\n"
        "research / scientific / academic / study            Research\n\n"
        "### Key rules\n"
        "- description must quote or closely paraphrase the text — not a generic label.\n"
        "- If multiple purposes are stated, extract the PRIMARY one for this call.\n"
        "- 'Legitimate interest' is a legal BASIS, not a purpose category — do not\n"
        "  map it to any PurposeCategory; instead describe the actual underlying purpose.\n\n"
        "### Few-shot example\n"
        "INPUT:\n"
        "We process your contact information to send you promotional emails about\n"
        "our products and services that may interest you.\n\n"
        "CORRECT OUTPUT:\n"
        "{\n"
        '  "purposeId": "",\n'
        '  "description": "Send promotional emails about products and services",\n'
        '  "category": "Marketing",\n'
        f'  "source_clause": "{ref}"\n'
        "}\n\n"
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
        "A cross-border transfer of personal data — structurally distinct from\n"
        "domestic processing. Only extract when the text explicitly governs an\n"
        "international or cross-border transfer.\n"
        "Maps to: GDPR Art.44-49 | LGPD Art.33 | CCPA data sale/share | PIPEDA Principle 1\n\n"
        "If NO cross-border transfer is described, return exactly:\n"
        '{"_no_transfer_stated": true}\n\n'
        "### Output schema\n"
        "{\n"
        '  "transferId": "",\n'
        '  "mechanism": "<TransferMechanism>",\n'
        '  "adequacyDecisionRef": "<EC adequacy decision citation or null>",\n'
        '  "destinationJurisdiction": [\n'
        '    {\n'
        '      "jurisdictionId": "<short code: JP | US | UK | BR | IN | AU | CA | SG>",\n'
        '      "name": "<full country/region name>",\n'
        '      "description": "",\n'
        '      "source_clause": "<article reference>"\n'
        '    }\n'
        '  ],\n'
        '  "dataTransferred": [\n'
        '    {\n'
        '      "dataId": "",\n'
        '      "description": "<data category transferred>",\n'
        '      "source": "<origin of the data>",\n'
        '      "category": "<PersonalDataCategory>",\n'
        '      "sensitivity": "<SensitivityLevel>",\n'
        '      "identifiability": "<Identifiability>",\n'
        '      "source_clause": "<article reference>"\n'
        '    }\n'
        '  ],\n'
        '  "source_clause": "<article reference>"\n'
        "}\n\n"
        "### Enum grammar\n" + eg + "\n\n"
        "### Decision guide for `mechanism`\n"
        "Text contains...                                      mechanism\n"
        "----------------------------------------------------------------\n"
        "adequacy decision / adequate level of protection      AdequacyDecision\n"
        "standard contractual clauses / SCCs / model clauses   StandardContractualClauses\n"
        "binding corporate rules / BCR / intra-group           BindingCorporateRules\n"
        "explicit consent / data subject consents to transfer  Consent\n"
        "necessary for contract / contract performance         ContractNecessity\n"
        "legitimate interest / compelling legitimate grounds   LegitimateInterest\n"
        "none of the above / other safeguard                   Other\n\n"
        "### OCL constraint hint — adequacyDecisionRef (constraint_dt1)\n"
        "If mechanism=AdequacyDecision, you MUST populate adequacyDecisionRef\n"
        "with the EC decision citation (e.g. 'EC Decision 2019/419 for Japan').\n"
        "If the text mentions adequacy but does not name the decision, use the\n"
        "country name as the reference (e.g. 'Adequacy — Japan').\n\n"
        "### Few-shot example\n"
        "INPUT:\n"
        "Article 46(2)(c) — Standard data protection clauses adopted by the\n"
        "Commission may be used to transfer personal data to recipients in the\n"
        "United States.\n\n"
        "CORRECT OUTPUT:\n"
        "{\n"
        '  "transferId": "",\n'
        '  "mechanism": "StandardContractualClauses",\n'
        '  "adequacyDecisionRef": null,\n'
        '  "destinationJurisdiction": [{"jurisdictionId": "US", "name": "United States", "description": "", "source_clause": "GDPR Art.46(2)(c)"}],\n'
        '  "dataTransferred": [{"dataId": "", "description": "personal data", "source": "controller", "category": "Identifier", "sensitivity": "Low", "identifiability": "Identified", "source_clause": "GDPR Art.46(2)(c)"}],\n'
        f'  "source_clause": "{ref}"\n'
        "}\n\n"
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
        "A data-handling constraint — a restriction or obligation on HOW personal\n"
        "data may be processed. Not the legal basis (why), not the purpose (what for),\n"
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
        "- Extract the PRIMARY constraint for this call. If the article contains\n"
        "  multiple constraints of different types, prefer the most restrictive.\n\n"
        "### Few-shot example\n"
        "INPUT:\n"
        "Personal data shall be processed in a manner that ensures appropriate\n"
        "security, including protection against unauthorised or unlawful processing\n"
        "and against accidental loss using appropriate technical measures (Art.5(1)(f)).\n\n"
        "CORRECT OUTPUT:\n"
        "{\n"
        '  "constraintId": "",\n'
        '  "type": "Security",\n'
        '  "expression": "Personal data must be protected against unauthorised processing and accidental loss using appropriate technical and organisational measures",\n'
        '  "enforcementLevel": "Mandatory",\n'
        f'  "source_clause": "{ref}"\n'
        "}\n\n"
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
        "The entity that is the subject of this processing statement — who is\n"
        "collecting, processing, or subject to the data activity.\n"
        "Maps to: GDPR Art.4(7-8) | LGPD Art.5(V-VII) | CCPA §1798.140 | PIPEDA Sch.1\n\n"
        "### Output schema\n"
        "{\n"
        '  "actorId": "",\n'
        '  "name": "<name of the entity, e.g. \'Data Controller\', \'Third-Party Processor\'>",\n'
        '  "role": "<ActorRole>",\n'
        '  "source_clause": "<article reference>"\n'
        "}\n\n"
        "### Enum grammar\n" + eg + "\n\n"
        "### Decision guide for `role`\n"
        "Text contains...                                    role\n"
        "------------------------------------------------------------\n"
        "data subject / individual / consumer / user         DataSubject\n"
        "controller / organization / company / business      DataController\n"
        "processor / service provider / vendor / agent       DataProcessor\n"
        "third party / recipient / partner / affiliate       ThirdParty\n\n"
        "### Cross-law terminology map\n"
        "GDPR 'controller'          → DataController\n"
        "LGPD 'controlador'         → DataController\n"
        "CCPA 'business'            → DataController\n"
        "GDPR 'processor'           → DataProcessor\n"
        "LGPD 'operador'            → DataProcessor\n"
        "CCPA 'service provider'    → DataProcessor\n"
        "GDPR/LGPD 'data subject'   → DataSubject\n"
        "CCPA 'consumer'            → DataSubject\n"
        "PIPEDA 'individual'        → DataSubject\n\n"
        "### Key rules\n"
        "- name should be the generic role title from the text, not a company name,\n"
        "  unless the article explicitly names a specific organisation.\n"
        "- If the article describes an obligation on the controller about the data\n"
        "  subject, the Actor is the controller (the entity bearing the obligation).\n\n"
        "### Few-shot example\n"
        "INPUT:\n"
        "Article 4(7) — 'Controller' means the natural or legal person, public\n"
        "authority, agency or other body which, alone or jointly with others,\n"
        "determines the purposes and means of the processing of personal data.\n\n"
        "CORRECT OUTPUT:\n"
        "{\n"
        '  "actorId": "",\n'
        '  "name": "Data Controller",\n'
        '  "role": "DataController",\n'
        f'  "source_clause": "{ref}"\n'
        "}\n\n"
        f"### Now extract from the following text:\n"
        f"LAW: {law}\n"
        f"ARTICLE/SECTION: {ref}\n"
        "---\n"
        f"{text}\n"
        "---\n"
    )


def _personal_data_prompt(law: str, ref: str, text: str) -> str:
    eg = _enum_block(["PersonalDataCategory", "SensitivityLevel", "Identifiability"])
    return (
        "## Task: Extract a PersonalData instance\n\n"
        "### What you are extracting\n"
        "A category of personal data named or implied in the article.\n"
        "Maps to: GDPR Art.4(1) + Art.9 | LGPD Art.5(I) + Art.11 | CCPA §1798.140(v)\n\n"
        "Extract the MOST SPECIFIC data category described. If multiple categories\n"
        "are named, extract the most sensitive one for this call.\n\n"
        "### Output schema\n"
        "{\n"
        '  "dataId": "",\n'
        '  "description": "<specific data description, e.g. \'email address\', \'GPS location\'>",\n'
        '  "source": "<how the data is obtained, e.g. \'provided by user\', \'automatically collected\'>",\n'
        '  "category": "<PersonalDataCategory>",\n'
        '  "sensitivity": "<SensitivityLevel>",\n'
        '  "identifiability": "<Identifiability>",\n'
        '  "source_clause": "<article reference>"\n'
        "}\n\n"
        "### Enum grammar\n" + eg + "\n\n"
        "### Classification guide\n"
        "description                      category              sensitivity      identifiability\n"
        "----------------------------------------------------------------------------------------\n"
        "Name, email, phone number        ContactInformation    Low              Identified\n"
        "National ID, passport number     Identifier            High             Identified\n"
        "IP address, cookie ID            TechnicalData         Low              Pseudonymous\n"
        "GPS / precise location           LocationData          Medium           Identified\n"
        "General area / city              LocationData          Low              Pseudonymous\n"
        "Bank account, credit card        FinancialData         High             Identified\n"
        "Medical records, diagnoses       HealthData            SpecialCategory  Identified\n"
        "Fingerprint, face scan           BiometricData         SpecialCategory  Identified\n"
        "Racial / ethnic origin           Identifier            SpecialCategory  Identified\n"
        "Browsing history / clicks        BehavioralData        Medium           Pseudonymous\n"
        "User-generated content / posts   ContentData           Low              Identified\n\n"
        "### Key rules\n"
        "- description must be specific — 'personal data' alone is not acceptable.\n"
        "- source: 'provided by user', 'automatically collected', 'obtained from third party',\n"
        "  'inferred from behaviour', or 'generated by system'.\n"
        "- sensitivity=SpecialCategory only for GDPR Art.9 / LGPD Art.11 categories:\n"
        "  health, biometric, racial/ethnic, religious, political, sexual orientation, genetic.\n\n"
        "### Few-shot example\n"
        "INPUT:\n"
        "We collect your full name and email address when you create an account.\n\n"
        "CORRECT OUTPUT:\n"
        "{\n"
        '  "dataId": "",\n'
        '  "description": "Email address",\n'
        '  "source": "Provided by user at account creation",\n'
        '  "category": "ContactInformation",\n'
        '  "sensitivity": "Low",\n'
        '  "identifiability": "Identified",\n'
        f'  "source_clause": "{ref}"\n'
        "}\n\n"
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


def build_concept_prompt(concept: str, law_name: str, article_ref: str, legal_text: str) -> tuple[str, str]:
    builder = _CONCEPT_BUILDERS.get(concept)
    if builder is None:
        raise ValueError(f"No prompt builder for '{concept}'. Available: {list(_CONCEPT_BUILDERS)}")
    return SYSTEM_PROMPT, builder(law_name, article_ref, legal_text)


def build_assembler_prompt(
    actor_json: str, purposes_json: str, processing_activity_json: str,
    legal_basis_json: str, regulations_json: str, constraints_json: str,
    rights_json: str, source_clause: str,
    retention_json: str = "[]", transfers_json: str = "[]", withdrawal_json: str = "[]",
) -> tuple[str, str]:
    user = (
        "## Task: Assemble a PolicyStatement from Pass 1 extractions\n\n"
        "Compose the objects below into a single PolicyStatement JSON.\n"
        "Do NOT re-read or reinterpret the legal text. Only flag consistency issues.\n\n"
        "### Consistency checks — add \"_warnings\":[...] if any fire\n"
        "1. legalBasis.type==\"Consent\" and consentWithdrawal==[]\n"
        "   → \"constraint_4: Consent basis present but no withdrawal mechanics extracted\"\n"
        "2. processingActivity.action in [Transfer,Share] and any dataProcessed.sensitivity\n"
        "   in [High,SpecialCategory] and riskAssessmentReference==null\n"
        "   → \"constraint_3: High-risk transfer without risk assessment reference\"\n"
        "3. Any governingRegulation has empty jurisdiction list\n"
        "   → \"constraint_2: Regulation missing jurisdiction\"\n\n"
        f"ACTOR:\n{actor_json}\n\n"
        f"PURPOSES:\n{purposes_json}\n\n"
        f"PROCESSING ACTIVITY:\n{processing_activity_json}\n\n"
        f"LEGAL BASIS:\n{legal_basis_json}\n\n"
        f"GOVERNING REGULATIONS:\n{regulations_json}\n\n"
        f"CONSTRAINTS:\n{constraints_json}\n\n"
        f"RIGHTS IMPACTED:\n{rights_json}\n\n"
        f"RETENTION POLICIES:\n{retention_json}\n\n"
        f"DATA TRANSFERS:\n{transfers_json}\n\n"
        f"CONSENT WITHDRAWAL:\n{withdrawal_json}\n\n"
        f"SOURCE CLAUSE: {source_clause}\n"
    )
    return SYSTEM_PROMPT, user
