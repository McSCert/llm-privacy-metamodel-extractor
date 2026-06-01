"""
_ocl_validators.py — Hand-maintained OCL semantic constraints.

This file is the ONLY file in privacy_schema/ that is NOT generated.
It contains the Pydantic @model_validator methods that implement the
OCL constraints defined in the metamodel but not expressible in Ecore's
type system (cross-field rules, cross-object rules, conditional requirements).

Relationship to the generator:
  generate_pydantic.py reads privacy_metamodel.ecore and generates:
    → privacy_schema/enums.py      (fully generated — do not edit)
    → privacy_schema/models.py     (fully generated — do not edit)

  This file is imported by the generated models.py. The generator emits
  an import line and validator registration calls for every entry in the
  _OCL_REGISTRY dict inside generate_pydantic.py. When you add a new OCL
  constraint, add its implementation here AND register it in the generator.

OCL constraint registry (what each validator enforces):
  constraint_3   (warning) → ProcessingActivityModel
      Transfer or Share of High/SpecialCategory data without DPIA reference.
      Source: GDPR Art.35, LGPD Art.38, CPRA.

  constraint_dt1 (error)   → DataTransferModel
      AdequacyDecision mechanism requires adequacyDecisionRef to be set.
      Source: GDPR Art.45.

  constraint_2   (error)   → PolicyStatementModel
      Every governing regulation must have at least one jurisdiction.
      Source: metamodel structural invariant.

  constraint_4   (warning) → PolicyStatementModel
      Consent legal basis implies consentWithdrawal must be modelled.
      Source: GDPR Art.7(3), LGPD Art.8§5, CCPA §1798.120.

  constraint_01  (error)   → PrivacyPolicyModel
      A PrivacyPolicy must contain at least one statement.
      Source: metamodel structural invariant.

Adding a new constraint:
  1. Implement it here as a plain function: def ocl_<name>(self): ...
  2. Register it in generate_pydantic.py → _OCL_REGISTRY[ClassName].
  3. Re-run: python generate_pydantic.py
  4. The generated models.py will import and attach it automatically.
"""

from __future__ import annotations

import warnings


# =============================================================================
# ProcessingActivityModel validators
# =============================================================================

def ocl_constraint_3_warning(self) -> "ProcessingActivityModel":
    """
    OCL constraint_3 (warning):
    Transfer or Share of High/SpecialCategory data implies
    riskAssessmentReference must be present.

    Sources: GDPR Art.35, LGPD Art.38, CPRA (mandatory),
             PIPEDA (best-practice).
    """
    # Import inside function to avoid circular import at module level.
    from .enums import ProcessingAction, SensitivityLevel

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
                f"and LGPD Art.38; regulatory obligation under CPRA; "
                f"best-practice under PIPEDA.",
                stacklevel=2,
            )
    return self


# =============================================================================
# DataTransferModel validators
# =============================================================================

def ocl_constraint_dt1(self) -> "DataTransferModel":
    """
    OCL constraint_dt1 (error):
    mechanism = AdequacyDecision implies adequacyDecisionRef must be set.

    Source: GDPR Art.45 — transfers based on adequacy require the EC decision
    to be cited explicitly for audit traceability.
    """
    from .enums import TransferMechanism

    if (
        self.mechanism == TransferMechanism.AdequacyDecision
        and not self.adequacy_decision_ref
    ):
        raise ValueError(
            f"[constraint_dt1 / error] DataTransfer '{self.transfer_id}': "
            f"mechanism=AdequacyDecision but adequacyDecisionRef is empty. "
            f"Cite the EC adequacy decision document "
            f"(e.g. 'EC Decision 2019/419 for Japan')."
        )
    return self


# =============================================================================
# PolicyStatementModel validators
# =============================================================================

def ocl_constraint_2(self) -> "PolicyStatementModel":
    """
    OCL constraint_2 (error):
    Every governing regulation must have at least one jurisdiction.

    Source: metamodel structural invariant — a regulation with no jurisdiction
    is unresolvable during gap analysis and XMI serialisation.
    """
    for reg in self.governing_regulations:
        if not reg.jurisdiction:
            raise ValueError(
                f"[constraint_2 / error] Regulation '{reg.name}' in statement "
                f"'{self.statement_id}' has no jurisdiction. Every regulation "
                f"must reference at least one canonical Jurisdiction instance."
            )
    return self


def ocl_constraint_4_warning(self) -> "PolicyStatementModel":
    """
    OCL constraint_4 (warning):
    Consent legal basis implies consentWithdrawal must be modelled.

    Sources: GDPR Art.7(3), LGPD Art.8§5, CCPA §1798.120.
    Withdrawal must be as easy to invoke as giving consent.
    """
    from .enums import LegalBasisType

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


# =============================================================================
# PrivacyPolicyModel validators
# =============================================================================

def ocl_constraint_01(self) -> "PrivacyPolicyModel":
    """
    OCL constraint_01 (error):
    statements->size() > 0

    Source: metamodel structural invariant — a PrivacyPolicy with no statements
    carries no extractable compliance information.
    """
    if not self.statements:
        raise ValueError(
            "[constraint_01 / error] PrivacyPolicy must contain "
            "at least one PolicyStatement."
        )
    return self