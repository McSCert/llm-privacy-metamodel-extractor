"""
Pydantic extraction schema — generated from PrivacyPolicyMetamodel v4.

Design rules (metamodel → Python):
  multiplicity "1"    → required field, no default
  multiplicity "0..1" → Optional[T] = None
  multiplicity "1..*" → List[T],          min_length=1   (Pydantic v2)
  multiplicity "0..*" → List[T] = []
  type "long"         → int
  non-navigable assoc → omitted (back-references are not extraction targets)

Each model carries a `source_clause` field (not in the metamodel) for
RAG traceability — the LLM must populate it with the retrieved chunk
citation (e.g. "GDPR Art.6(1)(a)") that justified the extraction.

OCL constraints are implemented as @model_validator methods.
  severity "error"   → raises ValueError  → LLM output rejected
  severity "warning" → appends to ValidationWarnings, instance accepted

Usage:
    from privacy_schema import PolicyStatementModel
    instance = PolicyStatementModel.model_validate(llm_json_output)
"""

from __future__ import annotations

import uuid
import warnings
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from .enums import (
    ActorRole, ConstraintType, Identifiability, LegalBasisType,
    PersonalDataCategory, ProcessingAction, PurposeCategory,
    RetentionTrigger, RetentionUnit, RightType, SensitivityLevel,
    TransferMechanism, WithdrawalChannel,
)


# ── Utility ───────────────────────────────────────────────────────────────────

def _new_id(prefix: str) -> str:
    """Default ID factory — used when the LLM omits an ID field."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class _Base(BaseModel):
    """Common config for all schema models."""
    model_config = {"populate_by_name": True, "str_strip_whitespace": True}


# ── RegulatoryContext ─────────────────────────────────────────────────────────

class JurisdictionModel(_Base):
    """
    Metamodel: RegulatoryContext.Jurisdiction
    Canonical jurisdiction registry entry.  The LLM must resolve
    jurisdiction names to these canonical instances rather than
    free-texting strings like 'EU', 'European Union', or 'Europe'.
    """
    jurisdiction_id: str = Field(
        default_factory=lambda: _new_id("jur"),
        alias="jurisdictionId",
        description="Stable identifier, e.g. 'EU', 'CA-US', 'BR'.",
    )
    name: str = Field(
        description="Full canonical name, e.g. 'European Union'."
    )
    description: str = Field(
        default="",
        description="Short description of the jurisdiction's privacy regime."
    )
    source_clause: str = Field(
        default="",
        description="RAG chunk citation that identified this jurisdiction."
    )


class RegulationModel(_Base):
    """
    Metamodel: RegulatoryContext.Regulation
    The governing law for a PolicyStatement.
    """
    regulation_id: str = Field(
        default_factory=lambda: _new_id("reg"),
        alias="regulationId",
    )
    name: str = Field(
        description="Short name of the regulation, e.g. 'GDPR', 'CCPA', 'LGPD'."
    )
    version: str = Field(
        default="",
        description="Version or amendment identifier."
    )
    description: str = Field(
        default="",
        description="Brief description of the regulation's scope."
    )
    jurisdiction: List[JurisdictionModel] = Field(
        min_length=1,
        description="[OCL constraint_2] Every regulation must have at least one jurisdiction."
    )
    source_clause: str = Field(
        default="",
        description="RAG chunk citation for this regulation reference."
    )


# ── Actors ────────────────────────────────────────────────────────────────────

class ActorModel(_Base):
    """
    Metamodel: Actors.Actor
    The entity performing or subject to processing.
    """
    actor_id: str = Field(
        default_factory=lambda: _new_id("act"),
        alias="actorId",
    )
    name: str = Field(
        description="Name of the actor, e.g. 'Acme Corp', 'User', 'Payment Processor'."
    )
    role: ActorRole = Field(
        description="Role from ActorRole enum."
    )
    source_clause: str = Field(
        default="",
        description="RAG chunk citation."
    )


# ── PersonalData ──────────────────────────────────────────────────────────────

class PersonalDataModel(_Base):
    """
    Metamodel: PersonalDataModel.PersonalData
    A category of personal data involved in a processing activity.
    """
    data_id: str = Field(
        default_factory=lambda: _new_id("dat"),
        alias="dataId",
    )
    description: str = Field(
        description="Human-readable description of the data, e.g. 'email address'."
    )
    source: str = Field(
        description="How the data is obtained, e.g. 'provided by user', 'inferred'."
    )
    category: PersonalDataCategory
    sensitivity: SensitivityLevel
    identifiability: Identifiability
    source_clause: str = Field(
        default="",
        description="RAG chunk citation."
    )


# ── Processing ────────────────────────────────────────────────────────────────

class ProcessingActivityModel(_Base):
    """
    Metamodel: Processing.ProcessingActivity
    A single processing operation on personal data.

    OCL constraint_3 (warning):
        Transfer or Share of High/SpecialCategory data implies
        riskAssessmentReference must be present.
    """
    activity_id: str = Field(
        default_factory=lambda: _new_id("prc"),
        alias="activityId",
    )
    description: str = Field(
        description="Free-text description of what the processing does."
    )
    action: ProcessingAction
    risk_assessment_reference: Optional[str] = Field(
        default=None,
        alias="riskAssessmentReference",
        description=(
            "Citation of a DPIA (GDPR Art.35), RIPD (LGPD Art.38), risk assessment "
            "(CPRA), or voluntary PIA (PIPEDA). Required when action is Transfer or "
            "Share on High/SpecialCategory data; a compliance gap signal if absent."
        ),
    )
    data_processed: List[PersonalDataModel] = Field(
        min_length=1,
        alias="dataProcessed",
        description="Personal data categories involved. At least one required."
    )
    source_clause: str = Field(
        default="",
        description="RAG chunk citation."
    )

    @model_validator(mode="after")
    def ocl_constraint_3_warning(self) -> "ProcessingActivityModel":
        """
        OCL constraint_3 (warning / GDPR Art.35 / LGPD Art.38 / CPRA):
        High-risk Transfer or Share without a risk assessment reference.
        """
        high_risk_actions = {ProcessingAction.Transfer, ProcessingAction.Share}
        high_sensitivity  = {SensitivityLevel.High, SensitivityLevel.SpecialCategory}

        if self.action in high_risk_actions:
            has_sensitive = any(
                d.sensitivity in high_sensitivity for d in self.data_processed
            )
            if has_sensitive and not self.risk_assessment_reference:
                warnings.warn(
                    f"[constraint_3 / warning] ProcessingActivity '{self.activity_id}': "
                    f"action={self.action.value} on High/SpecialCategory data but "
                    f"riskAssessmentReference is empty. Hard violation under GDPR Art.35 "
                    f"and LGPD Art.38; regulatory obligation under CPRA; best-practice under PIPEDA.",
                    stacklevel=2,
                )
        return self


class DataTransferModel(_Base):
    """
    Metamodel: Processing.DataTransfer
    A cross-border data transfer — structurally distinct from ProcessingActivity
    so GDPR Ch.V, LGPD Art.33, and CCPA data-sale provisions each get their
    own typed extraction target.

    OCL constraint_dt1 (error):
        mechanism = AdequacyDecision implies adequacyDecisionRef must be set.
    """
    transfer_id: str = Field(
        default_factory=lambda: _new_id("xfr"),
        alias="transferId",
    )
    mechanism: TransferMechanism = Field(
        description="Legal instrument authorising the cross-border transfer."
    )
    adequacy_decision_ref: Optional[str] = Field(
        default=None,
        alias="adequacyDecisionRef",
        description=(
            "Reference to the adequacy decision document. "
            "Required when mechanism=AdequacyDecision."
        ),
    )
    destination_jurisdiction: List[JurisdictionModel] = Field(
        min_length=1,
        alias="destinationJurisdiction",
        description="Destination jurisdiction(s) for the transfer."
    )
    data_transferred: List[PersonalDataModel] = Field(
        min_length=1,
        alias="dataTransferred",
        description="Personal data categories being transferred."
    )
    source_clause: str = Field(
        default="",
        description="RAG chunk citation, e.g. 'GDPR Art.46(2)(c)'."
    )

    @model_validator(mode="after")
    def ocl_constraint_dt1(self) -> "DataTransferModel":
        """
        OCL constraint_dt1 (error):
        AdequacyDecision mechanism requires adequacyDecisionRef.
        """
        if (
            self.mechanism == TransferMechanism.AdequacyDecision
            and not self.adequacy_decision_ref
        ):
            raise ValueError(
                f"[constraint_dt1 / error] DataTransfer '{self.transfer_id}': "
                f"mechanism=AdequacyDecision but adequacyDecisionRef is empty. "
                f"Cite the EC adequacy decision document (e.g. 'EC Decision 2019/419 for Japan')."
            )
        return self


# ── Purposes ──────────────────────────────────────────────────────────────────

class PurposeModel(_Base):
    """
    Metamodel: Purposes.Purpose
    A processing purpose. Every PolicyStatement requires at least one.
    """
    purpose_id: str = Field(
        default_factory=lambda: _new_id("pur"),
        alias="purposeId",
    )
    description: str = Field(
        description="Specific purpose as stated in the legal text."
    )
    category: PurposeCategory
    source_clause: str = Field(
        default="",
        description="RAG chunk citation."
    )


# ── Legal ─────────────────────────────────────────────────────────────────────

class LegalBasisModel(_Base):
    """
    Metamodel: Legal.LegalBasis
    The legal justification for a processing activity.
    The evidence field must quote or closely paraphrase the legal text.
    """
    basis_id: str = Field(
        default_factory=lambda: _new_id("lb"),
        alias="basisId",
    )
    type: LegalBasisType
    evidence: str = Field(
        description=(
            "Verbatim or near-verbatim legal text that establishes this basis, "
            "e.g. 'Art.6(1)(a) GDPR — the data subject has given consent'."
        )
    )
    jurisdiction: List[JurisdictionModel] = Field(
        min_length=1,
        description="Jurisdiction(s) in which this legal basis applies."
    )
    source_clause: str = Field(
        default="",
        description="RAG chunk citation."
    )


# ── PolicyRules ───────────────────────────────────────────────────────────────

class ConstraintModel(_Base):
    """
    Metamodel: PolicyRules.Constraint
    A data-handling constraint (temporal, geographic, usage, etc.).
    """
    constraint_id: str = Field(
        default_factory=lambda: _new_id("con"),
        alias="constraintId",
    )
    type: ConstraintType
    expression: str = Field(
        description="Natural-language or formal expression of the constraint."
    )
    enforcement_level: str = Field(
        alias="enforcementLevel",
        description="e.g. 'Mandatory', 'Recommended', 'BestEffort'."
    )
    source_clause: str = Field(
        default="",
        description="RAG chunk citation."
    )


class RightModel(_Base):
    """
    Metamodel: PolicyRules.Right
    A data-subject right impacted by this statement.
    """
    right_id: str = Field(
        default_factory=lambda: _new_id("rig"),
        alias="rightId",
    )
    type: RightType
    trigger_condition: str = Field(
        alias="triggerCondition",
        description="Condition under which the right may be exercised."
    )
    fulfillment_process: str = Field(
        alias="fulfillmentProcess",
        description="How the controller must respond when the right is invoked."
    )
    source_clause: str = Field(
        default="",
        description="RAG chunk citation."
    )


class RetentionPolicyModel(_Base):
    """
    Metamodel: PolicyRules.RetentionPolicy
    Storage-limitation rule — maps to GDPR Art.5(1)(e),
    LGPD Art.15, CCPA §1798.100(e), and equivalents.
    Use duration=-1 with unit=Indefinite when the law permits indefinite retention.
    """
    retention_id: str = Field(
        default_factory=lambda: _new_id("ret"),
        alias="retentionId",
    )
    duration: int = Field(
        description="Numeric duration value. Use -1 for indefinite."
    )
    unit: RetentionUnit
    trigger: RetentionTrigger = Field(
        description="The event that starts the retention clock."
    )
    basis_article: Optional[str] = Field(
        default=None,
        alias="basisArticle",
        description="Legal article mandating/permitting this period, e.g. 'GDPR Art.5(1)(e)'."
    )
    source_clause: str = Field(
        default="",
        description="RAG chunk citation."
    )


class ConsentWithdrawalModel(_Base):
    """
    Metamodel: PolicyRules.ConsentWithdrawal
    Mechanics for withdrawing consent — GDPR Art.7(3), LGPD Art.8§5,
    CCPA §1798.120. Must be present whenever LegalBasis.type = Consent
    (enforced as constraint_4 warning on PolicyStatement).
    channel is 1..* — laws often require multiple channels.
    """
    withdrawal_id: str = Field(
        default_factory=lambda: _new_id("wdr"),
        alias="withdrawalId",
    )
    channel: List[WithdrawalChannel] = Field(
        min_length=1,
        description="Channel(s) through which withdrawal can be exercised."
    )
    deadline: str = Field(
        description=(
            "Maximum time the controller has to act after withdrawal request, "
            "e.g. '30 days', 'without undue delay', 'immediately'."
        )
    )
    effect_on_prior_processing: str = Field(
        alias="effectOnPriorProcessing",
        description=(
            "Whether withdrawal affects the lawfulness of prior processing. "
            "GDPR Art.7(3): does not affect prior processing. "
            "LGPD Art.8§5: controller must confirm impossibility of immediate cessation."
        )
    )
    source_clause: str = Field(
        default="",
        description="RAG chunk citation."
    )


# ── Core ──────────────────────────────────────────────────────────────────────

class PolicyStatementModel(_Base):
    """
    Metamodel: Core.PolicyStatement
    The central extraction unit — one instance per article / clause
    in a legal text.  All 12 navigable associations from the metamodel
    are represented as typed fields below.

    OCL constraints enforced here:
      constraint_2 (error):   every regulation must have non-empty jurisdiction.
      constraint_4 (warning): Consent legal basis implies consentWithdrawal present.

    OCL constraints delegated to nested models:
      constraint_3  (warning) → ProcessingActivityModel
      constraint_dt1 (error)  → DataTransferModel
    """
    statement_id: str = Field(
        default_factory=lambda: _new_id("stmt"),
        alias="statementId",
    )
    description: str = Field(
        description="Summary of what this statement covers."
    )

    # ── required associations (1 or 1..*) ────────────────────────────────────
    actor: ActorModel = Field(
        description="The entity (controller/processor/subject) this statement concerns."
    )
    purposes: List[PurposeModel] = Field(
        min_length=1,
        description="[1..*] At least one processing purpose required."
    )
    processing_activity: ProcessingActivityModel = Field(
        alias="processingActivity",
        description="[1] The processing operation described by this statement."
    )
    legal_basis: LegalBasisModel = Field(
        alias="legalBasis",
        description="[1] Legal justification. Every statement must be legally grounded."
    )
    governing_regulations: List[RegulationModel] = Field(
        min_length=1,
        alias="governingRegulations",
        description="[1..*] Laws governing this statement."
    )
    constraints: List[ConstraintModel] = Field(
        min_length=1,
        description="[1..*] Data-handling constraints attached to this statement."
    )
    rights_impacted: List[RightModel] = Field(
        min_length=1,
        alias="rightImpacted",
        description="[1..*] Data-subject rights affected."
    )

    # ── optional associations (0..*) ─────────────────────────────────────────
    retention_policies: List[RetentionPolicyModel] = Field(
        default=[],
        alias="retentionPolicies",
        description="[0..*] Storage-limitation rules. Leave empty if article is silent."
    )
    data_transfers: List[DataTransferModel] = Field(
        default=[],
        alias="dataTransfers",
        description="[0..*] Cross-border transfers. Leave empty if no transfer described."
    )
    consent_withdrawal: List[ConsentWithdrawalModel] = Field(
        default=[],
        alias="consentWithdrawal",
        description=(
            "[0..*] Consent-withdrawal mechanics. Should be present when "
            "legalBasis.type = Consent (constraint_4 warning if absent)."
        )
    )

    source_clause: str = Field(
        default="",
        description="Top-level RAG chunk citation for this statement, e.g. 'GDPR Art.6'."
    )

    @model_validator(mode="after")
    def ocl_constraint_2(self) -> "PolicyStatementModel":
        """
        OCL constraint_2 (error):
        Every governing regulation must have at least one jurisdiction.
        """
        for reg in self.governing_regulations:
            if not reg.jurisdiction:
                raise ValueError(
                    f"[constraint_2 / error] Regulation '{reg.name}' in statement "
                    f"'{self.statement_id}' has no jurisdiction. Every regulation "
                    f"must reference at least one canonical Jurisdiction instance."
                )
        return self

    @model_validator(mode="after")
    def ocl_constraint_4_warning(self) -> "PolicyStatementModel":
        """
        OCL constraint_4 (warning):
        Consent legal basis implies consentWithdrawal must be modelled.
        GDPR Art.7(3), LGPD Art.8§5, CCPA §1798.120.
        """
        if (
            self.legal_basis.type == LegalBasisType.Consent
            and not self.consent_withdrawal
        ):
            warnings.warn(
                f"[constraint_4 / warning] PolicyStatement '{self.statement_id}': "
                f"legalBasis.type=Consent but consentWithdrawal is empty. "
                f"GDPR Art.7(3), LGPD Art.8§5, and CCPA §1798.120 require withdrawal "
                f"to be as easy as giving consent.",
                stacklevel=2,
            )
        return self


class PrivacyPolicyModel(_Base):
    """
    Metamodel: Core.PrivacyPolicy
    Top-level extraction container.  One instance per ingested legal document
    or per law-specific extraction run.

    OCL constraint01 (error): statements must not be empty.
    """
    policy_id: str = Field(
        default_factory=lambda: _new_id("pol"),
        alias="policyId",
    )
    version: str = Field(
        description="Version of the law or document, e.g. '2018', '2020-amendment'."
    )
    valid_from: int = Field(
        alias="validFrom",
        description="Unix timestamp of the law's effective date."
    )
    valid_to: int = Field(
        alias="validTo",
        description="Unix timestamp of expiry. Use 9999999999 for open-ended."
    )
    statements: List[PolicyStatementModel] = Field(
        min_length=1,
        description="[OCL constraint01] At least one statement required."
    )

    @model_validator(mode="after")
    def ocl_constraint01(self) -> "PrivacyPolicyModel":
        """OCL constraint01 (error): statements->size() > 0"""
        if not self.statements:
            raise ValueError(
                "[constraint01 / error] PrivacyPolicy must contain at least one statement."
            )
        return self
