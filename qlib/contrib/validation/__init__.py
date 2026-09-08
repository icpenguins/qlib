# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Training-run promotion gates.

See ``qlib.contrib.validation.score_quality`` for the Alpha158/LightGBM
production-promotion gate (``validate_score_quality``). Design rationale and
threshold provenance are documented in
``.team-code/validate_score_quality.md``.
"""

from .score_quality import (
    ScoreQualityGateError,
    ScoreQualityReport,
    validate_score_quality,
    extract_portfolio_metrics,
    extract_ic_metrics,
)

__all__ = [
    "ScoreQualityGateError",
    "ScoreQualityReport",
    "validate_score_quality",
    "extract_portfolio_metrics",
    "extract_ic_metrics",
]
