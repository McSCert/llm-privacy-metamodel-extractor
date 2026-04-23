"""
Controlled vocabularies — generated from PrivacyPolicyMetamodel v4,
package: ControlledVocabularies.

13 enums, one per controlled vocabulary in the metamodel.

Every enum here is the exclusive set of values the LLM extractor may
output for the corresponding metamodel attribute.  Adding a new value
requires a metamodel change first, then regenerating this file.
"""

from enum import Enum


# ── Actors ────────────────────────────────────────────────────────────────────

class ActorRole(str, Enum):
    DataSubject    = "DataSubject"
    DataController = "DataController"
    DataProcessor  = "DataProcessor"
    ThirdParty     = "ThirdParty"


# ── Processing ────────────────────────────────────────────────────────────────

class ProcessingAction(str, Enum):
    Collect  = "Collect"
    Store    = "Store"
    Use      = "Use"
    Share    = "Share"
    Transfer = "Transfer"
    Delete   = "Delete"


# ── Legal ─────────────────────────────────────────────────────────────────────

class LegalBasisType(str, Enum):
    Consent             = "Consent"
    Contract            = "Contract"
    LegalObligation     = "LegalObligation"
    LegitimateInterest  = "LegitimateInterest"
    VitalInterest       = "VitalInterest"
    PublicTask          = "PublicTask"


# ── Purposes ──────────────────────────────────────────────────────────────────

class PurposeCategory(str, Enum):
    ServiceProvision = "ServiceProvision"
    Security         = "Security"
    LegalCompliance  = "LegalCompliance"
    Marketing        = "Marketing"
    Analytics        = "Analytics"
    Research         = "Research"


# ── PolicyRules ───────────────────────────────────────────────────────────────

class ConstraintType(str, Enum):
    Temporal         = "Temporal"
    Geographic       = "Geographic"
    Usage            = "Usage"
    Security         = "Security"
    Retention        = "Retention"
    PurposeLimitation = "PurposeLimitation"


class RightType(str, Enum):
    Access                  = "Access"
    Rectification           = "Rectification"
    Erasure                 = "Erasure"
    Restriction             = "Restriction"
    Portability             = "Portability"
    Objection               = "Objection"
    AutomatedDecisionOptOut = "AutomatedDecisionOptOut"


class RetentionUnit(str, Enum):
    Days       = "Days"
    Months     = "Months"
    Years      = "Years"
    Indefinite = "Indefinite"


class RetentionTrigger(str, Enum):
    CollectionDate         = "CollectionDate"
    ContractEnd            = "ContractEnd"
    LastActivity           = "LastActivity"
    LegalObligationExpiry  = "LegalObligationExpiry"
    ConsentWithdrawal      = "ConsentWithdrawal"
    AccountDeletion        = "AccountDeletion"


class WithdrawalChannel(str, Enum):
    OnlineForm      = "OnlineForm"
    Email           = "Email"
    WrittenRequest  = "WrittenRequest"
    InAppToggle     = "InAppToggle"
    PhoneRequest    = "PhoneRequest"
    InPerson        = "InPerson"


# ── Transfer ──────────────────────────────────────────────────────────────────

class TransferMechanism(str, Enum):
    AdequacyDecision          = "AdequacyDecision"
    StandardContractualClauses = "StandardContractualClauses"
    BindingCorporateRules     = "BindingCorporateRules"
    Consent                   = "Consent"
    ContractNecessity         = "ContractNecessity"
    LegitimateInterest        = "LegitimateInterest"
    Other                     = "Other"


# ── PersonalData ──────────────────────────────────────────────────────────────

class PersonalDataCategory(str, Enum):
    Identifier          = "Identifier"
    ContactInformation  = "ContactInformation"
    LocationData        = "LocationData"
    FinancialData       = "FinancialData"
    HealthData          = "HealthData"
    BiometricData       = "BiometricData"
    BehavioralData      = "BehavioralData"
    TechnicalData       = "TechnicalData"
    ContentData         = "ContentData"


class SensitivityLevel(str, Enum):
    Low             = "Low"
    Medium          = "Medium"
    High            = "High"
    SpecialCategory = "SpecialCategory"


class Identifiability(str, Enum):
    Identified   = "Identified"
    Pseudonymous = "Pseudonymous"
    Anonymous    = "Anonymous"
