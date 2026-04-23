"""
gap_analysis — Cross-law compliance gap analysis for PrivacyPolicyMetamodel v4.

Usage
-----
    from gap_analysis.repository import ModelRepository
    from gap_analysis.gap_analysis import GapAnalyser

    # Store validated statements
    repo = ModelRepository("data/model_repo.db")
    repo.store("GDPR",  "Art.6",  gdpr_art6_statement)
    repo.store("CCPA",  "§1798.100", ccpa_statement)
    repo.store("LGPD",  "Art.7",  lgpd_statement)

    # Run analysis
    analyser = GapAnalyser(repo)

    # Individual queries
    lb_coverage = analyser.legal_basis_coverage()
    rights      = analyser.rights_coverage()
    delta       = analyser.cross_law_delta("GDPR", "CCPA")

    # Full formatted report
    print(analyser.full_report())

    # Export as JSON for downstream processing
    json_output = analyser.to_json()
"""

from .repository  import ModelRepository
from .gap_analysis import (
    GapAnalyser,
    ConceptCoverage,
    FlagCoverage,
    CrossLawDelta,
    CoverageMatrix,
)

__all__ = [
    "ModelRepository",
    "GapAnalyser",
    "ConceptCoverage",
    "FlagCoverage",
    "CrossLawDelta",
    "CoverageMatrix",
]
