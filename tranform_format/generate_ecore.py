"""
generate_ecore.py — Generates privacy_metamodel.ecore from the JSON metamodel
                    and enums defined in enums.py.

Design decisions (per researcher's choices):
  - Single flat EPackage (no sub-packages) for clean Eclipse EMF loading
  - ALL associations → containment=True (inline serialization per policy)
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
    "DataSubject", "DataController", "DataProcessor", "ThirdParty",
])

ProcessingAction = _make_enum("ProcessingAction", [
    "Collect", "Store", "Use", "Share", "Transfer", "Delete",
])

LegalBasisType = _make_enum("LegalBasisType", [
    "Consent", "Contract", "LegalObligation",
    "LegitimateInterest", "VitalInterest", "PublicTask",
])

PurposeCategory = _make_enum("PurposeCategory", [
    "ServiceProvision", "Security", "LegalCompliance",
    "Marketing", "Analytics", "Research",
])

ConstraintType = _make_enum("ConstraintType", [
    "Temporal", "Geographic", "Usage", "Security",
    "Retention", "PurposeLimitation",
])

RightType = _make_enum("RightType", [
    "Access", "Rectification", "Erasure", "Restriction",
    "Portability", "Objection", "AutomatedDecisionOptOut",
])

RetentionUnit = _make_enum("RetentionUnit", [
    "Days", "Months", "Years", "Indefinite",
])

RetentionTrigger = _make_enum("RetentionTrigger", [
    "CollectionDate", "ContractEnd", "LastActivity",
    "LegalObligationExpiry", "ConsentWithdrawal", "AccountDeletion",
])

WithdrawalChannel = _make_enum("WithdrawalChannel", [
    "OnlineForm", "Email", "WrittenRequest",
    "InAppToggle", "PhoneRequest", "InPerson",
])

TransferMechanism = _make_enum("TransferMechanism", [
    "AdequacyDecision", "StandardContractualClauses",
    "BindingCorporateRules", "Consent", "ContractNecessity",
    "LegitimateInterest", "Other",
])

PersonalDataCategory = _make_enum("PersonalDataCategory", [
    "Identifier", "ContactInformation", "LocationData", "FinancialData",
    "HealthData", "BiometricData", "BehavioralData", "TechnicalData",
    "ContentData",
])

SensitivityLevel = _make_enum("SensitivityLevel", [
    "Low", "Medium", "High", "SpecialCategory",
])

Identifiability = _make_enum("Identifiability", [
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
    EAttribute("policyId",   EString, lower=1),
    EAttribute("version",    EString, lower=1),
    EAttribute("validFrom",  ELong,   lower=1),
    EAttribute("validTo",    ELong,   lower=1),
    # Containment: one policy owns all its statements
    EReference("statements", PolicyStatement,
               lower=1, upper=-1, containment=True),
])

# ── PolicyStatement ───────────────────────────────────────────────────────────
PolicyStatement.eStructuralFeatures.extend([
    EAttribute("statementId",  EString, lower=1),
    EAttribute("description",  EString, lower=1),
    # Required contained children
    EReference("actor",              Actor,             lower=1, upper=1,  containment=True),
    EReference("purposes",           Purpose,           lower=1, upper=-1, containment=True),
    EReference("processingActivity", ProcessingActivity,lower=1, upper=1,  containment=True),
    EReference("legalBasis",         LegalBasis,        lower=1, upper=1,  containment=True),
    EReference("governingRegulations", Regulation,      lower=1, upper=-1, containment=True),
    EReference("constraints",        Constraint,        lower=1, upper=-1, containment=True),
    EReference("rightImpacted",      Right,             lower=1, upper=-1, containment=True),
    # Optional contained children
    EReference("retentionPolicies",  RetentionPolicy,   lower=0, upper=-1, containment=True),
    EReference("dataTransfers",      DataTransfer,      lower=0, upper=-1, containment=True),
    EReference("consentWithdrawal",  ConsentWithdrawal, lower=0, upper=-1, containment=True),
])

# ── Actor ─────────────────────────────────────────────────────────────────────
Actor.eStructuralFeatures.extend([
    EAttribute("actorId", EString,    lower=1),
    EAttribute("name",    EString,    lower=1),
    EAttribute("role",    ActorRole,  lower=1),
])

# ── LegalBasis ────────────────────────────────────────────────────────────────
LegalBasis.eStructuralFeatures.extend([
    EAttribute("basisId",  EString,       lower=1),
    EAttribute("type",     LegalBasisType,lower=1),
    EAttribute("evidence", EString,       lower=1),
    EReference("jurisdiction", Jurisdiction, lower=1, upper=-1, containment=True),
])

# ── ProcessingActivity ────────────────────────────────────────────────────────
ProcessingActivity.eStructuralFeatures.extend([
    EAttribute("activityId",            EString,         lower=1),
    EAttribute("description",           EString,         lower=1),
    EAttribute("action",                ProcessingAction,lower=1),
    EAttribute("riskAssessmentReference", EString,       lower=0, upper=1),
    EReference("dataProcessed", PersonalData, lower=1, upper=-1, containment=True),
])

# ── DataTransfer ──────────────────────────────────────────────────────────────
DataTransfer.eStructuralFeatures.extend([
    EAttribute("transferId",          EString,          lower=1),
    EAttribute("mechanism",           TransferMechanism,lower=1),
    EAttribute("adequacyDecisionRef", EString,          lower=0, upper=1),
    EReference("destinationJurisdiction", Jurisdiction,
               lower=1, upper=-1, containment=True),
    EReference("dataTransferred", PersonalData,
               lower=1, upper=-1, containment=True),
])

# ── Purpose ───────────────────────────────────────────────────────────────────
Purpose.eStructuralFeatures.extend([
    EAttribute("purposeId",   EString,         lower=1),
    EAttribute("description", EString,         lower=1),
    EAttribute("category",    PurposeCategory, lower=1),
])

# ── PersonalData ──────────────────────────────────────────────────────────────
PersonalData.eStructuralFeatures.extend([
    EAttribute("dataId",          EString,             lower=1),
    EAttribute("description",     EString,             lower=1),
    EAttribute("source",          EString,             lower=1),
    EAttribute("category",        PersonalDataCategory,lower=1),
    EAttribute("sensitivity",     SensitivityLevel,    lower=1),
    EAttribute("identifiability", Identifiability,     lower=1),
])

# ── Constraint ────────────────────────────────────────────────────────────────
Constraint.eStructuralFeatures.extend([
    EAttribute("constraintId",    EString,       lower=1),
    EAttribute("type",            ConstraintType,lower=1),
    EAttribute("expression",      EString,       lower=1),
    EAttribute("enforcementLevel",EString,       lower=1),
])

# ── Right ─────────────────────────────────────────────────────────────────────
Right.eStructuralFeatures.extend([
    EAttribute("rightId",           EString,   lower=1),
    EAttribute("type",              RightType, lower=1),
    EAttribute("triggerCondition",  EString,   lower=1),
    EAttribute("fulfillmentProcess",EString,   lower=1),
])

# ── RetentionPolicy ───────────────────────────────────────────────────────────
RetentionPolicy.eStructuralFeatures.extend([
    EAttribute("retentionId",   EString,         lower=1),
    EAttribute("duration",      EInt,            lower=1),
    EAttribute("unit",          RetentionUnit,   lower=1),
    EAttribute("trigger",       RetentionTrigger,lower=1),
    EAttribute("basisArticle",  EString,         lower=0, upper=1),
])

# ── ConsentWithdrawal ─────────────────────────────────────────────────────────
# channel is 1..* (multi-valued EAttribute of EEnum type)
ConsentWithdrawal.eStructuralFeatures.extend([
    EAttribute("withdrawalId",          EString,          lower=1),
    EAttribute("channel",               WithdrawalChannel,lower=1, upper=-1),
    EAttribute("deadline",              EString,          lower=1),
    EAttribute("effectOnPriorProcessing", EString,        lower=1),
])

# ── Regulation ────────────────────────────────────────────────────────────────
Regulation.eStructuralFeatures.extend([
    EAttribute("regulationId", EString, lower=1),
    EAttribute("name",         EString, lower=1),
    EAttribute("version",      EString, lower=1),
    EAttribute("description",  EString, lower=1),
    EReference("jurisdiction", Jurisdiction,
               lower=1, upper=-1, containment=True),
])

# ── Jurisdiction ──────────────────────────────────────────────────────────────
Jurisdiction.eStructuralFeatures.extend([
    EAttribute("jurisdictionId", EString, lower=1),
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
