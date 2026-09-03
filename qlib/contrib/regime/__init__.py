# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Market Regime Classification and Bayesian Online Changepoint Detection (BOCD)
=============================================================================
Institutional-grade non-lagging regime classification combining Bayesian Online
Changepoint Detection (Adams & MacKay 2007) with macro credit risk appetite spreads
(HYG/IEI) and multi-horizon realized volatility surfaces.
"""

from .bocd import BayesianOnlineChangepointDetector, StudentTConjugatePrior, ConstantHazard
from .macro_vol_features import MacroVolFeatureExtractor
from .regime_classifier import MarketRegimeClassifier

__all__ = [
    "BayesianOnlineChangepointDetector",
    "StudentTConjugatePrior",
    "ConstantHazard",
    "MacroVolFeatureExtractor",
    "MarketRegimeClassifier",
]

