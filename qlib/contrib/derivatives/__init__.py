# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Quantitative Derivatives & Dealer Gamma Exposure (GEX) Module
==============================================================
Standalone quantitative package implementing options microstructure analytics:
- Dealer Gamma Exposure (GEX) per strike and aggregated
- Volatility Trigger / Gamma Flip Point S*
- Major Call Gamma Wall, Put Gamma Wall, and Max Pain
- 25-Delta Risk Reversal Skew & Variance Risk Premium (VRP)
- Options data loading, caching, and calibrated synthetic surface generation
"""

from .gex import BlackScholesGreeks, DealerGammaEngine, compute_dealer_gex_summary
from .vol_surface import VolatilitySurfaceFeatures
from .options_data import OptionsDataLoader, SyntheticOptionSurfaceGenerator
from .forced_dealer_hedging import calculate_forced_dealer_hedging_demand
from .liquidity_impact_ratio import calculate_liquidity_impact_ratio
from .post_earnings_volatility import (
    calibrate_post_earnings_volatility,
    calibrate_post_earnings_volatility_surface,
)
from .positive_gamma_squeeze import compute_positive_gamma_squeeze_index
from .negative_gamma_squeeze import compute_negative_gamma_squeeze_index
from .squeeze_probability_calibration import (
    generate_dual_squeeze_label,
    fit_platt_calibrator,
    calibrate_squeeze_probability,
)
from .conformal_prediction_bounds import calculate_conformal_bounds
from .data_provenance_guard import DataProvenance, DataProvenanceGuard, validate_data_provenance
from .factor_orthogonalization import orthogonalize_gsi_factors
from .earnings_gamma_squeeze_engine import evaluate_earnings_gamma_squeeze

__all__ = [
    "BlackScholesGreeks",
    "DealerGammaEngine",
    "compute_dealer_gex_summary",
    "VolatilitySurfaceFeatures",
    "OptionsDataLoader",
    "SyntheticOptionSurfaceGenerator",
    "calculate_forced_dealer_hedging_demand",
    "calculate_liquidity_impact_ratio",
    "calculate_historical_iv_crush",
    "calibrate_post_earnings_volatility",
    "calibrate_post_earnings_volatility_surface",
    "compute_positive_gamma_squeeze_index",
    "compute_negative_gamma_squeeze_index",
    "generate_dual_squeeze_label",
    "fit_platt_calibrator",
    "calibrate_squeeze_probability",
    "calculate_conformal_bounds",
    "DataProvenance",
    "DataProvenanceGuard",
    "validate_data_provenance",
    "orthogonalize_gsi_factors",
    "evaluate_earnings_gamma_squeeze",
]

