# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Deflated Sharpe Ratio (DSR) Module
==================================
# my-qlib Fork Extension: Institutional Microstructure & Derivatives Layer

Calculates the Bailey & López de Prado (2014) Deflated Sharpe Ratio (DSR),
penalizing for multiple testing selection bias, non-normal returns (skewness and kurtosis),
and track record length. Computes expected maximum Sharpe hurdles dynamically
from the empirical trial returns matrix without hardcoded constants.
"""

from typing import Dict, Any, Union
import numpy as np
import scipy.stats as stats
import math


def calculate_deflated_sharpe_ratio(
    trial_matrix: Union[np.ndarray, list],
    benchmark_sharpe: float = 0.0,
    annualization_factor: float = 252.0,
) -> Dict[str, Any]:
    """
    Computes the Deflated Sharpe Ratio (DSR) from an empirical trial returns matrix.

    Parameters
    ----------
    trial_matrix : Union[np.ndarray, list]
        Array of returns of shape (T, N_trials), where T is the sample length (days)
        and N_trials is the number of strategy parameter configurations tested.
    benchmark_sharpe : float, optional
        Baseline hurdle Sharpe ratio under the null hypothesis, by default 0.0.
    annualization_factor : float, optional
        Periods per year, by default 252.0.

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - 'best_sharpe': Annualized Sharpe ratio of top performing strategy
        - 'expected_max_sharpe': Hurdle E[max(SR_0)] derived dynamically from N_trials
        - 'dsr_pvalue': Deflated Sharpe Ratio probability P(SR > E[max(SR_0)])
        - 'is_statistically_significant': bool (p >= 0.95)
        - 'n_trials': int
        - 'sample_length_days': int
        - 'skewness': float
        - 'kurtosis': float
    """
    matrix = np.asarray(trial_matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)

    T, N = matrix.shape
    if T < 30:
        raise ValueError(f"Sample length T={T} is too short for Deflated Sharpe Ratio calculation (min 30).")

    # Compute annualized Sharpe ratio for each trial column
    means = np.mean(matrix, axis=0)
    stds = np.std(matrix, axis=0, ddof=1)
    # Avoid division by zero
    valid_stds = np.where(stds > 1e-8, stds, 1e-8)
    daily_sharpes = means / valid_stds
    annual_sharpes = daily_sharpes * np.sqrt(annualization_factor)

    best_idx = int(np.argmax(annual_sharpes))
    best_sharpe = float(annual_sharpes[best_idx])
    best_trial_returns = matrix[:, best_idx]

    # Calculate skewness and kurtosis of the best strategy returns
    mean_best = np.mean(best_trial_returns)
    std_best = np.std(best_trial_returns, ddof=1)
    if std_best > 1e-8:
        z = (best_trial_returns - mean_best) / std_best
        skew = float(np.mean(z ** 3))
        # Pearson kurtosis (normal distribution = 3.0)
        kurt = float(np.mean(z ** 4))
    else:
        skew = 0.0
        kurt = 3.0

    # Variance of Sharpe ratios across trials
    if N > 1:
        var_sharpes = float(np.var(annual_sharpes, ddof=1))
        std_sharpes = math.sqrt(max(1e-8, var_sharpes))
        # Euler-Mascheroni constant
        gamma_em = 0.5772156649
        # Dynamic expected maximum hurdle: E[max(SR_0)] = ( (1-gamma)*Phi^-1(1 - 1/N) + gamma*Phi^-1(1 - 1/(N*e)) ) * std_sharpes
        # Standard analytical extreme value approximation:
        sqrt_2lnN = math.sqrt(2.0 * math.log(N))
        expected_max = std_sharpes * (sqrt_2lnN + (gamma_em / sqrt_2lnN))
    else:
        var_sharpes = 0.0
        expected_max = benchmark_sharpe

    # Adjust best_sharpe and expected_max to daily scale for DSR test statistic
    sr_daily = best_sharpe / np.sqrt(annualization_factor)
    sr_hurdle_daily = expected_max / np.sqrt(annualization_factor)

    # Standard error denominator under non-normal returns (Mertens, 2002)
    denom_sq = 1.0 - skew * sr_daily + ((kurt - 1.0) / 4.0) * (sr_daily ** 2)
    denom = math.sqrt(max(1e-8, denom_sq))

    # Test statistic z
    z_stat = (sr_daily - sr_hurdle_daily) * math.sqrt(T - 1.0) / denom

    # Cumulative probability Phi(z)
    dsr = float(stats.norm.cdf(z_stat)) if hasattr(stats, "norm") else 0.5 * (1.0 + math.erf(z_stat / math.sqrt(2.0)))

    return {
        "best_sharpe": round(best_sharpe, 4),
        "expected_max_sharpe": round(float(expected_max), 4),
        "dsr_probability": round(float(dsr), 4),
        "is_statistically_significant": bool(dsr >= 0.95),
        "n_trials": N,
        "sample_length_days": T,
        "skewness": round(skew, 4),
        "kurtosis": round(kurt, 4),
    }

