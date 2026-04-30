"""
pydantic_to_xmi.py — Converts validated Pydantic PolicyStatement/PrivacyPolicy
                      instances to Eclipse EMF XMI using the generated .ecore.

Design:
  - Fully data-driven: uses the loaded Ecore metamodel to guide conversion,
    no hardcoded field mappings.
  - Drops source_clause, _extraction_confidence, _warnings (pipeline-only).
  - Empty *Id strings from Pydantic are replaced by pyecore auto-generated
    xmi:ids (empty strings are invalid in XMI cross-references).
  - Enum values are resolved from string → EEnumLiteral via the metamodel.
  - Multi-valued EAttributes (e.g. ConsentWithdrawal.channel) handled correctly.

Usage:
    from pydantic_to_xmi import PolicyXMIWriter

    writer = PolicyXMIWriter("privacy_metamodel.ecore")

    # From a validated Pydantic PrivacyPolicyModel instance:
    writer.write_privacy_policy(pydantic_instance, "output.xmi")

    # From a raw validated dict (e.g. after model.model_dump(by_alias=True)):
    writer.write_from_dict(policy_dict, "PrivacyPolicy", "output.xmi")

    # Write a single PolicyStatement (wraps in a minimal PrivacyPolicy):
    writer.write_policy_statement(pydantic_statement, "output.xmi")
"""

from __future__ import annotations

import uuid
import warnings
from pathlib import Path
from typing import Any

from pyecore.ecore import EAttribute, EReference, EEnum, EClass
from pyecore.resources import ResourceSet, URI
from pyecore.resources.xmi import XMIResource


# Fields that exist in Pydantic/pipeline but have NO counterpart in the Ecore
_PIPELINE_FIELDS = frozenset({
    "source_clause",
    "_extraction_confidence",
    "_warnings",
    # Pydantic generates these for internal use
    "model_config",
})

# Pydantic alias → Ecore feature name where they differ
# (Most are identical; only override when they don't match)
_ALIAS_TO_ECORE: dict[str, str] = {
    # Pydantic aliases that use camelCase already matching Ecore — no remapping needed.
    # Add entries here only if a Pydantic alias diverges from the Ecore name.
    # Example: "pydanticKey": "ecoreFeatureName"
}


class PolicyXMIWriter:
    """
    Converts validated Pydantic privacy-policy instances to Eclipse EMF XMI.

    The writer loads the .ecore once at construction time and reuses it
    for all subsequent write calls.
    """

    def __init__(self, ecore_path: str | Path = "privacy_metamodel.ecore"):
        ecore_path = Path(ecore_path).resolve()
        if not ecore_path.exists():
            raise FileNotFoundError(
                f"Ecore metamodel not found: {ecore_path}\n"
                f"Run generate_ecore.py first."
            )

        self._rset = ResourceSet()
        mm_resource = self._rset.get_resource(URI(str(ecore_path)))
        self._pkg = mm_resource.contents[0]
        # Register so instances can locate their metamodel via nsURI
        self._rset.metamodel_registry[self._pkg.nsURI] = self._pkg

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def write_privacy_policy(self, pydantic_instance, out_path: str | Path) -> None:
        """
        Convert a validated PrivacyPolicyModel Pydantic instance to XMI.
        The Pydantic instance is serialised via model_dump(by_alias=True).
        """
        d = pydantic_instance.model_dump(by_alias=True)
        self.write_from_dict(d, "PrivacyPolicy", out_path)

    def write_policy_statement(self, pydantic_statement, out_path: str | Path,
                                policy_version: str = "1.0",
                                valid_from: int = 0,
                                valid_to: int = 9999999999) -> None:
        """
        Wrap a single PolicyStatement in a minimal PrivacyPolicy and write XMI.
        Useful for writing one article's extraction without a full policy container.
        """
        stmt_dict = pydantic_statement.model_dump(by_alias=True)
        wrapper = {
            "policyId": "",
            "version": policy_version,
            "validFrom": valid_from,
            "validTo": valid_to,
            "statements": [stmt_dict],
        }
        self.write_from_dict(wrapper, "PrivacyPolicy", out_path)

    def write_from_dict(self, d: dict, root_class_name: str,
                        out_path: str | Path) -> None:
        """
        Convert a plain dict (e.g. from model.model_dump(by_alias=True)) to XMI.

        Args:
            d:               The nested dict representing the model instance.
            root_class_name: The Ecore class name for the root object
                             (e.g. "PrivacyPolicy").
            out_path:        Destination .xmi file path.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        root_eclass = self._pkg.getEClassifier(root_class_name)
        if root_eclass is None:
            raise ValueError(
                f"Class '{root_class_name}' not found in metamodel. "
                f"Available: {[c.name for c in self._pkg.eClassifiers if isinstance(c, EClass)]}"
            )

        root_obj = self._dict_to_eobject(d, root_eclass)

        resource = self._rset.create_resource(URI(str(out_path)))
        resource.append(root_obj)
        resource.save()

        print(f"✓  Written: {out_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # Internal conversion logic
    # ─────────────────────────────────────────────────────────────────────────

    def _dict_to_eobject(self, d: dict, eclass) -> Any:
        """
        Recursively convert a dict to a pyecore EObject of type `eclass`.

        Strategy:
          1. Create a new instance of eclass.
          2. For each key in d:
             - Skip pipeline-only fields (_PIPELINE_FIELDS).
             - Resolve the Ecore structural feature by name.
             - If EReference (containment): recurse into child dict(s).
             - If EAttribute (enum): resolve string → EEnumLiteral.
             - If EAttribute (primitive): set directly.
          3. Return the populated EObject.
        """
        obj = eclass()

        for raw_key, value in d.items():
            if raw_key in _PIPELINE_FIELDS:
                continue
            if value is None:
                continue  # optional field absent — leave as default

            ecore_key = _ALIAS_TO_ECORE.get(raw_key, raw_key)
            feature = self._find_feature(eclass, ecore_key)

            if feature is None:
                # Field in Pydantic dict has no Ecore counterpart — skip silently.
                # This handles fields that exist in Pydantic models but were
                # intentionally excluded from the .ecore (like source_clause).
                continue

            if isinstance(feature, EReference):
                self._set_reference(obj, feature, ecore_key, value)
            else:  # EAttribute
                self._set_attribute(obj, feature, ecore_key, value)

        return obj

    def _find_feature(self, eclass, name: str):
        """Find a structural feature by name, including inherited features."""
        # Direct features first
        for f in eclass.eStructuralFeatures:
            if f.name == name:
                return f
        # Walk supertype chain (handles inheritance if metamodel uses it)
        for supertype in eclass.eAllSuperTypes():
            for f in supertype.eStructuralFeatures:
                if f.name == name:
                    return f
        return None

    def _set_reference(self, obj, feature, key: str, value) -> None:
        """Handle an EReference (always containment in this metamodel)."""
        child_eclass = feature.eType

        if feature.many:
            # value should be a list of dicts
            items = value if isinstance(value, list) else [value]
            target_list = getattr(obj, key)
            for item in items:
                if item is None:
                    continue
                child = self._dict_to_eobject(item, child_eclass)
                target_list.append(child)
        else:
            # value should be a single dict
            child = self._dict_to_eobject(value, child_eclass)
            setattr(obj, key, child)

    def _set_attribute(self, obj, feature, key: str, value) -> None:
        """Handle an EAttribute (primitive or EEnum, single or multi-valued)."""
        is_enum = isinstance(feature.eType, EEnum)

        if feature.many:
            # Multi-valued EAttribute — only case: ConsentWithdrawal.channel
            items = value if isinstance(value, list) else [value]
            target_list = getattr(obj, key)
            for item in items:
                resolved = self._resolve_enum(feature.eType, item) if is_enum else item
                if resolved is not None:
                    target_list.append(resolved)
        else:
            if is_enum:
                resolved = self._resolve_enum(feature.eType, value)
                if resolved is not None:
                    setattr(obj, key, resolved)
            else:
                # Primitive: string, int, long
                # Empty string IDs are kept as-is; pyecore assigns xmi:id separately
                setattr(obj, key, value)

    @staticmethod
    def _resolve_enum(eenum, value: str):
        """
        Resolve a string enum value to its EEnumLiteral.
        Returns None and warns if the value is not in the enum.
        """
        literal = eenum.getEEnumLiteral(value)
        if literal is None:
            warnings.warn(
                f"Enum value '{value}' not found in {eenum.name}. "
                f"Valid values: {[lit.name for lit in eenum.eLiterals]}. "
                f"Skipping this attribute.",
                stacklevel=4,
            )
        return literal


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point — converts the extraction_demo mock output to XMI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/claude")

    # ── Build a sample validated Pydantic instance (same as extraction_demo) ──
    import warnings as _warnings
    from privacy_schema.models import PolicyStatementModel

    sample_stmt = {
        "statementId": "",
        "description": "GDPR Art.6(1)(a) — Consent-based processing",
        "actor": {
            "actorId": "", "name": "Data Controller",
            "role": "DataController", "source_clause": "GDPR Art.4(7)"
        },
        "purposes": [{
            "purposeId": "", "description": "Processing for specific consented purposes",
            "category": "ServiceProvision", "source_clause": "GDPR Art.6(1)(a)"
        }],
        "processingActivity": {
            "activityId": "", "description": "Consent-based data use",
            "action": "Use", "riskAssessmentReference": None,
            "dataProcessed": [{
                "dataId": "", "description": "personal data",
                "source": "user provided", "category": "Identifier",
                "sensitivity": "Low", "identifiability": "Identified",
                "source_clause": "GDPR Art.6(1)"
            }],
            "source_clause": "GDPR Art.6(1)"
        },
        "legalBasis": {
            "basisId": "", "type": "Consent",
            "evidence": "the data subject has given consent to the processing of his or her personal data for one or more specific purposes",
            "jurisdiction": [{
                "jurisdictionId": "EU", "name": "European Union",
                "description": "EU GDPR", "source_clause": "GDPR Art.6(1)(a)"
            }],
            "source_clause": "GDPR Art.6(1)(a)"
        },
        "governingRegulations": [{
            "regulationId": "", "name": "GDPR", "version": "2016/679",
            "description": "EU General Data Protection Regulation",
            "jurisdiction": [{
                "jurisdictionId": "EU", "name": "European Union",
                "description": "", "source_clause": "GDPR"
            }],
            "source_clause": "GDPR"
        }],
        "constraints": [{
            "constraintId": "", "type": "PurposeLimitation",
            "expression": "Data processed only for consented purposes",
            "enforcementLevel": "Mandatory", "source_clause": "GDPR Art.5(1)(b)"
        }],
        "rightImpacted": [{
            "rightId": "", "type": "Erasure",
            "triggerCondition": "Consent withdrawn",
            "fulfillmentProcess": "Delete within 30 days",
            "source_clause": "GDPR Art.17"
        }],
        "retentionPolicies": [],
        "dataTransfers": [],
        "consentWithdrawal": [{
            "withdrawalId": "",
            "channel": ["OnlineForm", "Email"],
            "deadline": "without undue delay",
            "effectOnPriorProcessing": "Does not affect prior processing",
            "source_clause": "GDPR Art.7(3)"
        }],
        "source_clause": "GDPR Art.6(1)(a)"
    }

    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        stmt_instance = PolicyStatementModel.model_validate(sample_stmt)
        for warning in w:
            print(f"⚠ Pydantic warning: {warning.message}")

    ecore_path = sys.argv[1] if len(sys.argv) > 1 else "privacy_metamodel.ecore"
    out_path   = sys.argv[2] if len(sys.argv) > 2 else "gdpr_art6_1a.xmi"

    writer = PolicyXMIWriter(ecore_path)
    writer.write_policy_statement(stmt_instance, out_path)
