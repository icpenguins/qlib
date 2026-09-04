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

__all__ = [
    "BlackScholesGreeks",
    "DealerGammaEngine",
    "compute_dealer_gex_summary",
    "VolatilitySurfaceFeatures",
    "OptionsDataLoader",
    "SyntheticOptionSurfaceGenerator",
]

