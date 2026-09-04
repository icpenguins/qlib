# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Conformal Prediction Bounds Module
==================================
# my-qlib Fork Extension: Institutional Microstructure & Derivatives Layer

Calculates non-parametric conformal prediction uncertainty bounds [p_lower, p_upper]
for calibrated squeeze probabilities at specified confidence intervals.
"""

from typing import Tuple
import numpy as np


def calculate_conformal_bounds(
    calibrated_prob: float,
    confidence_level: float = 0.90,
    residual_quantile: float = 0.08,
) -> Tuple[float, float]:
    """
    Computes conformal coverage bounds for calibrated squeeze probability.

    Parameters
    ----------
    calibrated_prob : float
        Posterior probability P in [0.0, 1.0].
    confidence_level : float, optional
        Target marginal coverage level (e.g. 0.90 for 90%), by default 0.90.
    residual_quantile : float, optional
        Empirical non-conformity calibration quantile |y - p|, by default 0.08.

    Returns
    -------
    Tuple[float, float]
        (p_lower, p_upper) bounded strictly within [0.0, 1.0].
    """
    p = float(np.clip(calibrated_prob, 0.0, 1.0))
    # Scale residual radius if confidence level deviates from default
    radius = residual_quantile * (confidence_level / 0.90)

    p_lower = max(0.0, p - radius)
    p_upper = min(1.0, p + radius)

    return (round(float(p_lower), 4), round(float(p_upper), 4))

