"""
privacy_schema — Pydantic extraction schema for PrivacyPolicyMetamodel v4.

Public API
----------
Models (import these in your extraction pipeline):
    PrivacyPolicyModel
    PolicyStatementModel
    ActorModel
    LegalBasisModel
    ProcessingActivityModel
    DataTransferModel
    PurposeModel
    PersonalDataModel
    ConstraintModel
    RightModel
    RetentionPolicyModel
    ConsentWithdrawalModel
    RegulationModel
    JurisdictionModel

Enums (use as the LLM's output grammar — no free-text values):
    ActorRole, ProcessingAction, LegalBasisType, PurposeCategory,
    ConstraintType, RightType, RetentionUnit, RetentionTrigger,
    TransferMechanism, WithdrawalChannel,
    PersonalDataCategory, SensitivityLevel, Identifiability
"""

from .enums import (
    ActorRole,
    ConstraintType,
    Identifiability,
    LegalBasisType,
    PersonalDataCategory,
    ProcessingAction,
    PurposeCategory,
    RetentionTrigger,
    RetentionUnit,
    RightType,
    SensitivityLevel,
    TransferMechanism,
    WithdrawalChannel,
)

from .models import (
    ActorModel,
    ConstraintModel,
    ConsentWithdrawalModel,
    DataTransferModel,
    JurisdictionModel,
    LegalBasisModel,
    PersonalDataModel,
    PolicyStatementModel,
    PrivacyPolicyModel,
    ProcessingActivityModel,
    PurposeModel,
    RegulationModel,
    RetentionPolicyModel,
    RightModel,
)

__all__ = [
    # enums
    "ActorRole", "ConstraintType", "Identifiability", "LegalBasisType",
    "PersonalDataCategory", "ProcessingAction", "PurposeCategory",
    "RetentionTrigger", "RetentionUnit", "RightType", "SensitivityLevel",
    "TransferMechanism", "WithdrawalChannel",
    # models
    "ActorModel", "ConstraintModel", "ConsentWithdrawalModel",
    "DataTransferModel", "JurisdictionModel", "LegalBasisModel",
    "PersonalDataModel", "PolicyStatementModel", "PrivacyPolicyModel",
    "ProcessingActivityModel", "PurposeModel", "RegulationModel",
    "RetentionPolicyModel", "RightModel",
]
