# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Historical Earnings IV Crush Estimator Module
=============================================
# my-qlib Fork Extension: Institutional Microstructure & Derivatives Layer

Calculates firm-specific post-earnings implied volatility crush using a Winsorized Median
over verified observed historical pre/post earnings volatility pairs, with an explicit
term-structure slope fallback when historical pairs are insufficient (< 4).
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np


def calculate_historical_iv_crush(
    observed_iv_pairs: Optional[List[Tuple[float, float]]] = None,
    month1_iv: Optional[float] = None,
    month2_iv: Optional[float] = None,
    trim_pct: float = 0.20,
    min_observed_pairs: int = 4,
) -> Dict[str, Any]:
    """
    Computes firm-specific historical IV crush.

    Parameters
    ----------
    observed_iv_pairs : Optional[List[Tuple[float, float]]]
        List of (sigma_pre, sigma_post) tuples for historical earnings announcements.
        Must contain positive float values to be valid.
    month1_iv : Optional[float]
        Front-month ATM implied volatility (for term structure proxy fallback).
    month2_iv : Optional[float]
        Second-month ATM implied volatility (for term structure proxy fallback).
    trim_pct : float, optional
        Fraction to trim from both tails for Winsorized median, by default 0.20.
    min_observed_pairs : int, optional
        Minimum number of valid observed pairs required to accept empirical calculation,
        by default 4.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'iv_crush_ratio': float in [0.0, 0.90]
        - 'crush_source': "empirical_winsorized_median" or "term_structure_proxy" or "conservative_default"
        - 'observed_count': int
        - 'is_empirical': bool
    """
    valid_crushes: List[float] = []

    if observed_iv_pairs:
        for pair in observed_iv_pairs:
            if isinstance(pair, (tuple, list)) and len(pair) >= 2:
                s_pre, s_post = float(pair[0]), float(pair[1])
                if s_pre > 0.01 and s_post > 0.001:
                    crush = (s_pre - s_post) / s_pre
                    valid_crushes.append(crush)

    n_valid = len(valid_crushes)

    if n_valid >= min_observed_pairs:
        # Calculate Winsorized Median
        sorted_crushes = np.sort(valid_crushes)
        k = int(np.floor(n_valid * trim_pct))
        if k > 0 and 2 * k < n_valid:
            # Winsorize: replace the k lowest values with value at index k,
            # and k highest values with value at index -k-1
            winsorized = sorted_crushes.copy()
            winsorized[:k] = sorted_crushes[k]
            winsorized[-k:] = sorted_crushes[-k - 1]
            median_crush = float(np.median(winsorized))
        else:
            median_crush = float(np.median(sorted_crushes))

        bounded_crush = float(np.clip(median_crush, 0.05, 0.85))
        return {
            "iv_crush_ratio": round(bounded_crush, 4),
            "crush_source": "empirical_winsorized_median",
            "observed_count": n_valid,
            "is_empirical": True,
        }

    # Fallback to term structure proxy if available
    if month1_iv is not None and month2_iv is not None and month1_iv > 0.01 and month2_iv > 0.01:
        slope = 1.0 - (month2_iv / month1_iv)
        proxy_crush = float(np.clip(slope, 0.20, 0.70))
        return {
            "iv_crush_ratio": round(proxy_crush, 4),
            "crush_source": "term_structure_proxy",
            "observed_count": n_valid,
            "is_empirical": False,
        }

    # Conservative default when neither is available
    return {
        "iv_crush_ratio": 0.40,
        "crush_source": "conservative_default",
        "observed_count": n_valid,
        "is_empirical": False,
    }

