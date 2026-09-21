"""
generate_ecore.py — Generates privacy_metamodel.ecore from the JSON metamodel
                    and enums defined in enums.py.

Design decisions (per researcher's choices):
  - Single flat EPackage (no sub-packages) for clean Eclipse EMF loading
  - Shared entities (Actor, PersonalData, Regulation, Jurisdiction) are owned
    ONCE by PrivacyPolicy in catalogues; statements reference them
    non-containment so one instance can be shared and matched by identity
  - Other associations remain containment=True (inline per statement)
  - Absence convention: empty collection / unset = concept ABSENT from the
    source text; enum literal _Unset = present but value undetermined
  - Back-references (owner ← owned) are DROPPED (derivable from containment tree)
  - source_clause is DROPPED (pipeline-only, not part of the model)
  - channel on ConsentWithdrawal is multi-valued EAttribute (upper=-1)

Output: privacy_metamodel.ecore

Usage:
    python generate_ecore.py
    # → writes privacy_metamodel.ecore in current directory
"""

from pyecore.ecore import (
    EPackage, EClass, EAttribute, EReference, EEnum, EEnumLiteral,
    EString, EInt, ELong,
)
from pyecore.resources import ResourceSet, URI

NS_URI    = "http://www.example.org/privacypolicy"
NS_PREFIX = "pp"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Root package
# ─────────────────────────────────────────────────────────────────────────────

pp = EPackage(name="privacyPolicy", nsURI=NS_URI, nsPrefix=NS_PREFIX)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Enums  (sourced from enums.py — not present in the JSON metamodel)
# ─────────────────────────────────────────────────────────────────────────────

def _make_enum(name, literals):
    e = EEnum(name, literals=literals)
    pp.eClassifiers.append(e)
    return e

ActorRole = _make_enum("ActorRole", [
    "_Unset",
    "DataSubject", "DataController", "DataProcessor", "ThirdParty",
])

ProcessingAction = _make_enum("ProcessingAction", [
    "_Unset",
    "Collect", "Store", "Use", "Share", "Transfer", "Delete",
])

LegalBasisType = _make_enum("LegalBasisType", [
    "_Unset",
    "Consent", "Contract", "LegalObligation",
    "LegitimateInterest", "VitalInterest", "PublicTask",
])

PurposeCategory = _make_enum("PurposeCategory", [
    "_Unset",
    "ServiceProvision", "Security", "LegalCompliance",
    "Marketing", "Analytics", "Research",
])

ConstraintType = _make_enum("ConstraintType", [
    "_Unset",
    "Temporal", "Geographic", "Usage", "Security",
    "Retention", "PurposeLimitation",
    "Accuracy", "Transparency",
])

EnforcementLevel = _make_enum("EnforcementLevel", [
    "_Unset",
    "Mandatory", "Conditional", "Recommended", "Prohibited",
])

RightType = _make_enum("RightType", [
    "_Unset",
    "Access", "Rectification", "Erasure", "Restriction",
    "Portability", "Objection", "AutomatedDecisionOptOut",
])

RetentionUnit = _make_enum("RetentionUnit", [
    "_Unset",
    "Days", "Months", "Years", "Indefinite",
])

RetentionTrigger = _make_enum("RetentionTrigger", [
    "_Unset",
    "CollectionDate", "ContractEnd", "LastActivity",
    "LegalObligationExpiry", "ConsentWithdrawal", "AccountDeletion",
])

WithdrawalChannel = _make_enum("WithdrawalChannel", [
    "_Unset",
    "OnlineForm", "Email", "WrittenRequest",
    "InAppToggle", "PhoneRequest", "InPerson",
])

TransferMechanism = _make_enum("TransferMechanism", [
    "_Unset",
    "AdequacyDecision", "StandardContractualClauses",
    "BindingCorporateRules", "Consent", "ContractNecessity",
    "LegitimateInterest", "Other",
])

PersonalDataCategory = _make_enum("PersonalDataCategory", [
    "_Unset",
    "Identifier", "ContactInformation", "LocationData", "FinancialData",
    "HealthData", "BiometricData", "BehavioralData", "TechnicalData",
    "ContentData",
])

SensitivityLevel = _make_enum("SensitivityLevel", [
    "_Unset",
    "Low", "Medium", "High", "SpecialCategory",
])

Identifiability = _make_enum("Identifiability", [
    "_Unset",
    "Identified", "Pseudonymous", "Anonymous",
])


# ─────────────────────────────────────────────────────────────────────────────
# 3. EClasses — created FIRST (all of them), features added SECOND
#    to avoid forward-reference errors inside pyecore
# ─────────────────────────────────────────────────────────────────────────────

PrivacyPolicy       = EClass("PrivacyPolicy")
PolicyStatement     = EClass("PolicyStatement")
Actor               = EClass("Actor")
LegalBasis          = EClass("LegalBasis")
ProcessingActivity  = EClass("ProcessingActivity")
DataTransfer        = EClass("DataTransfer")
Purpose             = EClass("Purpose")
PersonalData        = EClass("PersonalData")
Constraint          = EClass("Constraint")
Right               = EClass("Right")
RetentionPolicy     = EClass("RetentionPolicy")
ConsentWithdrawal   = EClass("ConsentWithdrawal")
Regulation          = EClass("Regulation")
Jurisdiction        = EClass("Jurisdiction")

_all_classes = [
    PrivacyPolicy, PolicyStatement, Actor, LegalBasis,
    ProcessingActivity, DataTransfer, Purpose, PersonalData,
    Constraint, Right, RetentionPolicy, ConsentWithdrawal,
    Regulation, Jurisdiction,
]
for c in _all_classes:
    pp.eClassifiers.append(c)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Attributes and containment references per class
#    Convention:
#      EAttribute(name, type, lower=1)          → required (1)
#      EAttribute(name, type, lower=0, upper=1) → optional (0..1)
#      EAttribute(name, type, lower=1, upper=-1)→ 1..* (multi-valued)
#      EReference(..., containment=True, lower, upper) → containment
# ─────────────────────────────────────────────────────────────────────────────

# ── PrivacyPolicy ─────────────────────────────────────────────────────────────
PrivacyPolicy.eStructuralFeatures.extend([
    EAttribute("policyId",   EString, lower=1, iD=True),
    EAttribute("version",    EString, lower=1),
    EAttribute("validFrom",  ELong,   lower=0, upper=1),
    EAttribute("validTo",    ELong,   lower=0, upper=1),
    # Containment: one policy owns all its statements
    EReference("statements", PolicyStatement,
               lower=1, upper=-1, containment=True),
    # Catalogues: the policy owns ONE instance of each shared entity.
    # Statements point at these non-containment, so "Canada" or "PIPEDA"
    # exists once per policy instead of being duplicated per statement.
    EReference("actorCatalogue",        Actor,        lower=0, upper=-1, containment=True),
    EReference("dataCatalogue",         PersonalData, lower=0, upper=-1, containment=True),
    EReference("regulationCatalogue",   Regulation,   lower=0, upper=-1, containment=True),
    EReference("jurisdictionCatalogue", Jurisdiction, lower=0, upper=-1, containment=True),
])

# ── PolicyStatement ───────────────────────────────────────────────────────────
PolicyStatement.eStructuralFeatures.extend([
    EAttribute("statementId",  EString, lower=1, iD=True),
    EAttribute("description",  EString, lower=1),
    # Contained children — owned by this statement, absent when not stated
    EReference("purposes",           Purpose,           lower=0, upper=-1, containment=True),
    EReference("processingActivity", ProcessingActivity,lower=0, upper=1,  containment=True),
    EReference("legalBasis",         LegalBasis,        lower=0, upper=1,  containment=True),
    EReference("constraints",        Constraint,        lower=0, upper=-1, containment=True),
    EReference("rightImpacted",      Right,             lower=0, upper=-1, containment=True),
    # References into the policy catalogues (non-containment)
    EReference("actor",                Actor,      lower=0, upper=1,  containment=False),
    EReference("governingRegulations", Regulation, lower=1, upper=-1, containment=False),
    # Optional contained children
    EReference("retentionPolicies",  RetentionPolicy,   lower=0, upper=-1, containment=True),
    EReference("dataTransfers",      DataTransfer,      lower=0, upper=-1, containment=True),
    EReference("consentWithdrawal",  ConsentWithdrawal, lower=0, upper=-1, containment=True),
])

# ── Actor ─────────────────────────────────────────────────────────────────────
Actor.eStructuralFeatures.extend([
    EAttribute("actorId", EString,    lower=1, iD=True),
    EAttribute("name",    EString,    lower=1),
    EAttribute("role",    ActorRole,  lower=1),
])

# ── LegalBasis ────────────────────────────────────────────────────────────────
LegalBasis.eStructuralFeatures.extend([
    EAttribute("basisId",  EString,       lower=1, iD=True),
    EAttribute("type",     LegalBasisType,lower=1),
    EAttribute("evidence", EString,       lower=0, upper=1),
    EReference("jurisdiction", Jurisdiction, lower=0, upper=-1, containment=False),
])

# ── ProcessingActivity ────────────────────────────────────────────────────────
ProcessingActivity.eStructuralFeatures.extend([
    EAttribute("activityId",            EString,         lower=1, iD=True),
    EAttribute("description",           EString,         lower=1),
    EAttribute("action",                ProcessingAction,lower=1),
    EAttribute("riskAssessmentReference", EString,       lower=0, upper=1),
    EReference("dataProcessed", PersonalData, lower=0, upper=-1, containment=False),
])

# ── DataTransfer ──────────────────────────────────────────────────────────────
DataTransfer.eStructuralFeatures.extend([
    EAttribute("transferId",          EString,          lower=1, iD=True),
    EAttribute("mechanism",           TransferMechanism,lower=1),
    EAttribute("adequacyDecisionRef", EString,          lower=0, upper=1),
    EReference("destinationJurisdiction", Jurisdiction,
               lower=0, upper=-1, containment=False),
    EReference("dataTransferred", PersonalData,
               lower=0, upper=-1, containment=False),
])

# ── Purpose ───────────────────────────────────────────────────────────────────
Purpose.eStructuralFeatures.extend([
    EAttribute("purposeId",   EString,         lower=1, iD=True),
    EAttribute("description", EString,         lower=1),
    EAttribute("category",    PurposeCategory, lower=1),
])

# ── PersonalData ──────────────────────────────────────────────────────────────
PersonalData.eStructuralFeatures.extend([
    EAttribute("dataId",          EString,             lower=1, iD=True),
    EAttribute("description",     EString,             lower=1),
    EAttribute("source",          EString,             lower=0, upper=1),
    EAttribute("category",        PersonalDataCategory,lower=1),
    EAttribute("sensitivity",     SensitivityLevel,    lower=1),
    EAttribute("identifiability", Identifiability,     lower=1),
])

# ── Constraint ────────────────────────────────────────────────────────────────
Constraint.eStructuralFeatures.extend([
    EAttribute("constraintId",    EString,         lower=1, iD=True),
    EAttribute("type",            ConstraintType,  lower=1),
    EAttribute("expression",      EString,         lower=1),
    EAttribute("enforcementLevel",EnforcementLevel,lower=1),
])

# ── Right ─────────────────────────────────────────────────────────────────────
Right.eStructuralFeatures.extend([
    EAttribute("rightId",           EString,   lower=1, iD=True),
    EAttribute("type",              RightType, lower=1),
    EAttribute("triggerCondition",  EString,   lower=0, upper=1),
    EAttribute("fulfillmentProcess",EString,   lower=0, upper=1),
])

# ── RetentionPolicy ───────────────────────────────────────────────────────────
RetentionPolicy.eStructuralFeatures.extend([
    EAttribute("retentionId",   EString,         lower=1, iD=True),
    EAttribute("duration",      EInt,            lower=0, upper=1),
    EAttribute("unit",          RetentionUnit,   lower=1),
    EAttribute("trigger",       RetentionTrigger,lower=1),
    EAttribute("basisArticle",  EString,         lower=0, upper=1),
])

# ── ConsentWithdrawal ─────────────────────────────────────────────────────────
# channel is 0..* (multi-valued EAttribute of EEnum type)
ConsentWithdrawal.eStructuralFeatures.extend([
    EAttribute("withdrawalId",          EString,          lower=1, iD=True),
    EAttribute("channel",               WithdrawalChannel,lower=0, upper=-1),
    EAttribute("deadline",              EString,          lower=0, upper=1),
    EAttribute("effectOnPriorProcessing", EString,        lower=0, upper=1),
])

# ── Regulation ────────────────────────────────────────────────────────────────
Regulation.eStructuralFeatures.extend([
    EAttribute("regulationId", EString, lower=1, iD=True),
    EAttribute("name",         EString, lower=1),
    EAttribute("version",      EString, lower=0, upper=1),
    EAttribute("description",  EString, lower=0, upper=1),
    EReference("jurisdiction", Jurisdiction,
               lower=0, upper=-1, containment=False),
])

# ── Jurisdiction ──────────────────────────────────────────────────────────────
Jurisdiction.eStructuralFeatures.extend([
    EAttribute("jurisdictionId", EString, lower=1, iD=True),
    EAttribute("name",           EString, lower=1),
    EAttribute("description",    EString, lower=0, upper=1),
])


# ─────────────────────────────────────────────────────────────────────────────
# 5. Serialize to .ecore
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    out_path = sys.argv[1] if len(sys.argv) > 1 else "privacy_metamodel.ecore"

    rset = ResourceSet()
    resource = rset.create_resource(URI(out_path))
    resource.append(pp)
    resource.save()

    print(f"✓  Written: {out_path}")
    print(f"   nsURI   : {NS_URI}")
    print(f"   nsPrefix: {NS_PREFIX}")
    print(f"   Classes : {len(_all_classes)}")
    print(f"   Enums   : {sum(1 for c in pp.eClassifiers if isinstance(c, EEnum))}")
