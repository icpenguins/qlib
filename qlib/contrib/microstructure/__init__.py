# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Institutional Microstructure, AVWAP & Volume Profile
=====================================================
Modules for Anchored VWAP, Volume Profile Kernel Density Estimation,
and order flow liquidity features.
"""

from typing import Dict, Any, Tuple
import pandas as pd

from .anchored_vwap import AnchoredVWAPCalculator
from .volume_profile import VolumeProfileKDE


def compute_microstructure_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    High-level convenience function to compute all institutional microstructure features:
    - Multi-Anchor VWAPs (YTD, 52W High, 52W Low) with dispersion bands
    - Continuous Volume Profile (KDE) with POC, VAH, VAL, and Liquidity Voids

    Parameters
    ----------
    df : pd.DataFrame
        Stock history DataFrame.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        (enriched_df, microstructure_summary_dict)
    """
    avwap_calc = AnchoredVWAPCalculator()
    enriched_df, avwap_summary = avwap_calc.compute_all_institutional_anchors(df)

    vp_calc = VolumeProfileKDE(lookback=63)
    vp_summary = vp_calc.compute_profile(df)

    summary = {
        "avwap": avwap_summary,
        "volume_profile": vp_summary,
    }

    return enriched_df, summary


__all__ = [
    "AnchoredVWAPCalculator",
    "VolumeProfileKDE",
    "compute_microstructure_features",
]

