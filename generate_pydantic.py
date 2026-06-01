#!/usr/bin/env python3
"""
generate_pydantic.py — Derive privacy_schema/enums.py and privacy_schema/models.py
                        from privacy_metamodel.ecore.

This script is the SINGLE SOURCE OF TRUTH gateway.
Edit the metamodel in generate_ecore.py, regenerate the .ecore, then run this
script to propagate every structural change into the Python extraction schema.

Workflow
--------
  1. Edit class/attribute/enum in  generate_ecore.py
  2. python generate_ecore.py           →  privacy_metamodel.ecore  (updated)
  3. python generate_pydantic.py        →  privacy_schema/enums.py  (regenerated)
                                        →  privacy_schema/models.py (regenerated)
  4. git diff privacy_schema/           →  review what changed
  5. Run tests / pipeline dry-run       →  confirm nothing broke

What is generated (do NOT hand-edit these files):
  privacy_schema/enums.py   — one Python Enum class per EEnum in the .ecore
  privacy_schema/models.py  — one Pydantic BaseModel per EClass in the .ecore,
                              with correct field types, multiplicities, and aliases

What is NOT generated (hand-maintained, must be kept in sync manually):
  privacy_schema/_ocl_validators.py — cross-field OCL constraint implementations

How validators are attached:
  Each model class that has OCL constraints is listed in _OCL_REGISTRY below.
  The generator emits an import + model_validator decorator call for each entry.
  To add a new constraint:
    1. Implement the function in _ocl_validators.py
    2. Add its name to _OCL_REGISTRY[ClassName] below
    3. Re-run this script

Usage:
    python generate_pydantic.py [path/to/privacy_metamodel.ecore]

    Default ecore path: privacy_metamodel.ecore (project root)
    Default output dir: privacy_schema/

    Override output:
    python generate_pydantic.py privacy_metamodel.ecore --out-dir privacy_schema
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path

# ── pyecore imports ───────────────────────────────────────────────────────────
try:
    from pyecore.ecore import EEnum, EClass, EAttribute, EReference
    from pyecore.resources import ResourceSet, URI
except ImportError:
    print("ERROR: pyecore not installed.  pip install pyecore")
    sys.exit(1)


# =============================================================================
# CONFIGURATION — edit these when metamodel structure changes
# =============================================================================

# Short prefix used in _new_id() for each class.
# Add a new entry whenever a new EClass is added to the metamodel.
_ID_PREFIXES: dict[str, str] = {
    "PrivacyPolicy":      "pol",
    "PolicyStatement":    "stmt",
    "Actor":              "act",
    "LegalBasis":         "lb",
    "ProcessingActivity": "prc",
    "DataTransfer":       "xfr",
    "Purpose":            "pur",
    "PersonalData":       "dat",
    "Constraint":         "con",
    "Right":              "rig",
    "RetentionPolicy":    "ret",
    "ConsentWithdrawal":  "cwd",
    "Regulation":         "reg",
    "Jurisdiction":       "jur",
}

# OCL validators to attach to each class.
# Key   = EClass name (matches EClass.name in the .ecore)
# Value = list of (function_name_in__ocl_validators, validator_mode)
#         mode is always "after" for these constraints.
# Add a new entry whenever a new OCL constraint is implemented.
_OCL_REGISTRY: dict[str, list[tuple[str, str]]] = {
    "ProcessingActivity": [
        ("ocl_constraint_3_warning", "after"),
    ],
    "DataTransfer": [
        ("ocl_constraint_dt1", "after"),
    ],
    "PolicyStatement": [
        ("ocl_constraint_2",         "after"),
        ("ocl_constraint_4_warning", "after"),
    ],
    "PrivacyPolicy": [
        ("ocl_constraint_01", "after"),
    ],
}

# EClass names to EXCLUDE from model generation.
# These are metamodel implementation details with no Pydantic counterpart.
_EXCLUDED_CLASSES: frozenset[str] = frozenset()

# Ecore primitive type name → Python type annotation string
_PRIMITIVE_TYPE_MAP: dict[str, str] = {
    "EString":  "str",
    "EInt":     "int",
    "ELong":    "int",
    "EBoolean": "bool",
    "EFloat":   "float",
    "EDouble":  "float",
}

# Topological generation order for EClasses.
# Classes that reference other classes must come after their dependencies.
# Add new classes in dependency order.
_CLASS_ORDER: list[str] = [
    "Jurisdiction",
    "Regulation",
    "Actor",
    "PersonalData",
    "ProcessingActivity",
    "DataTransfer",
    "Purpose",
    "LegalBasis",
    "Constraint",
    "Right",
    "RetentionPolicy",
    "ConsentWithdrawal",
    "PolicyStatement",
    "PrivacyPolicy",
]

# Fields that exist only in the pipeline (not in Ecore) but are injected
# into every generated model for RAG traceability.
# Format: (field_name, alias, description, default_value_code)
_PIPELINE_FIELDS: list[tuple[str, str, str, str]] = [
    (
        "source_clause",
        "source_clause",
        "RAG chunk citation that justified this extraction, e.g. 'GDPR Art.6(1)(a)'.",
        '""',
    ),
]


# =============================================================================
# UTILITIES
# =============================================================================

def _camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case. 'basisId' → 'basis_id'."""
    s1 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    return re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s1).lower()


def _model_name(eclass_name: str) -> str:
    """EClass name → Pydantic model class name. 'LegalBasis' → 'LegalBasisModel'."""
    return f"{eclass_name}Model"


def _python_type(feature, enum_names: set[str]) -> str:
    """
    Resolve an EStructuralFeature's type to a Python type annotation string.

    EAttribute with EString/EInt/ELong → primitive type
    EAttribute with EEnum              → enum class name
    EReference                         → model class name (with 'Model' suffix)
    """
    etype = feature.eType
    if etype is None:
        return "str"  # safe default

    type_name = etype.name

    if type_name in _PRIMITIVE_TYPE_MAP:
        return _PRIMITIVE_TYPE_MAP[type_name]

    if type_name in enum_names:
        return type_name  # enum class name (defined in enums.py)

    # EReference → another EClass → Model
    return _model_name(type_name)


def _is_id_feature(feature_name: str) -> bool:
    """Return True if this feature is an ID field (ends with 'Id')."""
    return feature_name.endswith("Id")


def _field_declaration(
    feature,
    enum_names: set[str],
    class_name: str,
) -> str:
    """
    Emit a single Pydantic Field declaration for one EStructuralFeature.

    Returns a string like:
        basis_id: str = Field(default_factory=lambda: _new_id("lb"), alias="basisId")
    """
    fname       = feature.name                 # camelCase from Ecore
    snake       = _camel_to_snake(fname)       # snake_case for Python field name
    lower       = feature.lowerBound           # 0 or 1
    upper       = feature.upperBound           # 1 or -1 (many)
    py_type     = _python_type(feature, enum_names)
    is_ref      = isinstance(feature, EReference)
    is_id       = _is_id_feature(fname)
    has_alias   = fname != snake               # emit alias only when names differ
    alias_arg   = f'alias="{fname}", ' if has_alias else ""

    # ── Determine annotation and Field() arguments ────────────────────────────

    if upper == -1:
        # Multi-valued (list)
        if lower >= 1:
            # 1..* — required, non-empty
            annotation = f"List[{py_type}]"
            field_args = f"{alias_arg}min_length=1"
        else:
            # 0..* — optional list, defaults to []
            annotation = f"List[{py_type}]"
            field_args = f"default=[], {alias_arg}".rstrip(", ")
            if alias_arg:
                field_args = f"default=[], {alias_arg}".rstrip(", ")
            else:
                field_args = "default=[]"
    else:
        # Single-valued
        if is_id:
            # ID fields get a default_factory so the LLM can omit them
            prefix = _ID_PREFIXES.get(class_name, class_name[:3].lower())
            annotation = py_type
            field_args = (
                f'default_factory=lambda: _new_id("{prefix}"), {alias_arg}'
            ).rstrip(", ")
        elif lower == 0:
            # Optional (0..1)
            annotation = f"Optional[{py_type}]"
            field_args = f"default=None, {alias_arg}".rstrip(", ")
        else:
            # Required (1)
            annotation = py_type
            field_args = alias_arg.rstrip(", ") if alias_arg else ""

    # Remove trailing comma/space from field_args
    field_args = field_args.strip().rstrip(",").strip()

    if field_args:
        return f"    {snake}: {annotation} = Field({field_args})"
    else:
        return f"    {snake}: {annotation}"


# =============================================================================
# ECORE LOADING
# =============================================================================

def load_ecore(ecore_path: Path):
    """Load the .ecore file and return the root EPackage."""
    rset = ResourceSet()
    resource = rset.get_resource(URI(str(ecore_path)))
    pkg = resource.contents[0]
    return pkg


# =============================================================================
# ENUMS GENERATION
# =============================================================================

def generate_enums(pkg, out_path: Path) -> list[str]:
    """
    Walk all EEnums in the package and emit enums.py.
    Returns a list of generated enum class names.
    """
    enum_names: list[str] = []
    blocks: list[str] = []

    for classifier in pkg.eClassifiers:
        if not isinstance(classifier, EEnum):
            continue

        name     = classifier.name
        literals = [lit.name for lit in classifier.eLiterals]
        enum_names.append(name)

        lines = [f"class {name}(str, Enum):"]
        for lit in literals:
            lines.append(f'    {lit} = "{lit}"')
        blocks.append("\n".join(lines))

    header = textwrap.dedent("""\
        \"\"\"
        enums.py — Controlled vocabularies.

        AUTO-GENERATED by generate_pydantic.py from privacy_metamodel.ecore.
        DO NOT EDIT THIS FILE MANUALLY.
        To change an enum: edit generate_ecore.py → run generate_ecore.py
        → run generate_pydantic.py.

        13 enums, one per EEnum in the metamodel.
        Every enum here is the exclusive set of values the LLM extractor may
        output for the corresponding metamodel attribute.
        \"\"\"

        from enum import Enum

    """)

    content = header + "\n\n".join(blocks) + "\n"
    out_path.write_text(content, encoding="utf-8")
    print(f"  ✓  enums.py      ({len(enum_names)} enums)")
    return enum_names


# =============================================================================
# MODELS GENERATION
# =============================================================================

def _collect_classes(pkg) -> dict[str, object]:
    """Return {class_name: EClass} for all EClasses in the package."""
    return {
        c.name: c
        for c in pkg.eClassifiers
        if isinstance(c, EClass) and c.name not in _EXCLUDED_CLASSES
    }


def _emit_class_block(
    eclass,
    enum_names: set[str],
    ocl_fn_names: list[str],
) -> str:
    """
    Emit the full Pydantic class definition for one EClass.

    Parameters
    ----------
    eclass       : pyecore EClass object
    enum_names   : set of all EEnum names (for type resolution)
    ocl_fn_names : list of OCL function names to attach as model_validators
    """
    class_name  = eclass.name
    model_class = _model_name(class_name)
    features    = list(eclass.eStructuralFeatures)

    lines: list[str] = []
    lines.append(f'class {model_class}(_Base):')
    lines.append(f'    """')
    lines.append(f'    Metamodel class: {class_name}')
    lines.append(f'    Auto-generated from privacy_metamodel.ecore.')

    # Note OCL validators if any
    if ocl_fn_names:
        lines.append(f'')
        lines.append(f'    OCL validators attached from _ocl_validators.py:')
        for fn in ocl_fn_names:
            lines.append(f'      {fn}')

    lines.append(f'    """')

    # ── Field declarations ────────────────────────────────────────────────────
    for feature in features:
        lines.append(_field_declaration(feature, enum_names, class_name))

    # ── Pipeline-only fields (not in Ecore) ───────────────────────────────────
    for field_name, alias, description, default in _PIPELINE_FIELDS:
        alias_arg = f'alias="{alias}", ' if alias != field_name else ""
        lines.append(
            f'    {field_name}: str = Field('
            f'default={default}, '
            f'{alias_arg}'
            f'description="{description}")'
        )

    # ── OCL validator attachments ─────────────────────────────────────────────
    if ocl_fn_names:
        lines.append("")
        for fn_name in ocl_fn_names:
            lines.append(f"    @model_validator(mode='after')")
            lines.append(f"    def _{fn_name}(self):")
            lines.append(f"        return {fn_name}(self)")
            lines.append("")

    return "\n".join(lines)


def generate_models(pkg, enum_names: list[str], out_path: Path) -> None:
    """
    Walk all EClasses in _CLASS_ORDER and emit models.py.
    """
    enum_set     = set(enum_names)
    all_classes  = _collect_classes(pkg)

    # ── Collect all OCL function names for import ─────────────────────────────
    all_ocl_fns: list[str] = []
    for fns in _OCL_REGISTRY.values():
        for fn_name, _ in fns:
            if fn_name not in all_ocl_fns:
                all_ocl_fns.append(fn_name)

    # ── Build enum import list ─────────────────────────────────────────────────
    enum_import = (
        "from .enums import (\n"
        + "".join(f"    {e},\n" for e in sorted(enum_set))
        + ")"
    )

    # ── Build OCL import block ─────────────────────────────────────────────────
    if all_ocl_fns:
        ocl_import = (
            "from ._ocl_validators import (\n"
            + "".join(f"    {fn},\n" for fn in all_ocl_fns)
            + ")"
        )
    else:
        ocl_import = "# No OCL validators registered."

    # ── Header ────────────────────────────────────────────────────────────────
    header = textwrap.dedent(f"""\
        \"\"\"
        models.py — Pydantic extraction schema.

        AUTO-GENERATED by generate_pydantic.py from privacy_metamodel.ecore.
        DO NOT EDIT THIS FILE MANUALLY.
        To change a field: edit generate_ecore.py → run generate_ecore.py
        → run generate_pydantic.py.

        Design rules (Ecore multiplicity → Python):
          lower=1, upper=1   → required field, no default
          lower=0, upper=1   → Optional[T] = None
          lower=1, upper=-1  → List[T], min_length=1
          lower=0, upper=-1  → List[T] = []
          EString/ELong      → str / int
          EEnum              → enum class from enums.py
          EReference         → nested model class (containment)

        ID fields (ending in 'Id') get a _new_id() default_factory so the
        LLM can omit them safely.

        source_clause is injected on every model for RAG traceability.
        It has no counterpart in the Ecore metamodel.

        OCL constraints are imported from _ocl_validators.py and attached
        as @model_validator methods. Edit that file to change constraint logic.
        \"\"\"

        from __future__ import annotations

        import uuid
        import warnings
        from typing import List, Optional

        from pydantic import BaseModel, Field, model_validator

        {enum_import}

        {ocl_import}


        # ── Utility ───────────────────────────────────────────────────────────────────

        def _new_id(prefix: str) -> str:
            \"\"\"Default ID factory — used when the LLM omits an ID field.\"\"\"
            return f"{{prefix}}_{{uuid.uuid4().hex[:8]}}"


        class _Base(BaseModel):
            \"\"\"Common config for all schema models.\"\"\"
            model_config = {{"populate_by_name": True, "str_strip_whitespace": True}}

    """)

    # ── Class blocks ──────────────────────────────────────────────────────────
    class_blocks: list[str] = []
    generated: list[str] = []

    for class_name in _CLASS_ORDER:
        if class_name not in all_classes:
            print(f"  ⚠  WARNING: {class_name} in _CLASS_ORDER not found in .ecore — skipping")
            continue

        eclass    = all_classes[class_name]
        ocl_fns   = [fn for fn, _ in _OCL_REGISTRY.get(class_name, [])]
        block     = _emit_class_block(eclass, enum_set, ocl_fns)
        class_blocks.append(block)
        generated.append(class_name)

    # Catch any EClasses in the .ecore not listed in _CLASS_ORDER
    unlisted = set(all_classes) - set(_CLASS_ORDER) - _EXCLUDED_CLASSES
    if unlisted:
        print(
            f"  ⚠  WARNING: These EClasses are in the .ecore but not in "
            f"_CLASS_ORDER — not generated: {sorted(unlisted)}\n"
            f"     Add them to _CLASS_ORDER in generate_pydantic.py."
        )

    content = header + "\n\n\n".join(class_blocks) + "\n"
    out_path.write_text(content, encoding="utf-8")
    print(f"  ✓  models.py     ({len(generated)} models)")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="generate_pydantic.py",
        description=(
            "Derive privacy_schema/enums.py and privacy_schema/models.py "
            "from privacy_metamodel.ecore."
        ),
    )
    parser.add_argument(
        "ecore",
        nargs="?",
        default="privacy_metamodel.ecore",
        metavar="PATH_TO_ECORE",
        help="Path to the .ecore file (default: privacy_metamodel.ecore)",
    )
    parser.add_argument(
        "--out-dir",
        default="privacy_schema",
        metavar="DIR",
        help="Output directory for enums.py and models.py (default: privacy_schema/)",
    )
    args = parser.parse_args()

    ecore_path = Path(args.ecore).resolve()
    out_dir    = Path(args.out_dir).resolve()

    if not ecore_path.exists():
        print(
            f"ERROR: .ecore file not found: {ecore_path}\n"
            f"Run generate_ecore.py first."
        )
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading : {ecore_path}")
    print(f"Output  : {out_dir}/")
    print()

    pkg = load_ecore(ecore_path)

    # 1. Generate enums.py
    enum_names = generate_enums(pkg, out_dir / "enums.py")

    # 2. Generate models.py
    generate_models(pkg, enum_names, out_dir / "models.py")

    print()
    print("Done. Run the pipeline dry-run to confirm:")
    print("  python run_pipeline.py --input PIPEDA=laws/pipeda.pdf --dry-run")


if __name__ == "__main__":
    main()