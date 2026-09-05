#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared Technical Indicators Module
==================================
Provides standardized, vector-optimized technical indicator computations:
- Relative Strength Index (RSI) using Wilder smoothing
- Bollinger Bands (Upper, Lower, Middle, %B)
- Rolling Maximum Drawdown
"""

from typing import Tuple
import pandas as pd
import numpy as np


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) using the standard Wilder rolling average.

    Parameters
    ----------
    series : pd.Series
        Price series (e.g., closing prices).
    period : int, optional
        Lookback period, by default 14.

    Returns
    -------
    pd.Series
        RSI values ranging from 0.0 to 100.0.
    """
    if series.empty or len(series) < 2:
        return pd.Series(index=series.index, dtype=float)

    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()

    rs = gain / (loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Compute Bollinger Bands and %B position oscillator.

    Parameters
    ----------
    series : pd.Series
        Price series.
    window : int, optional
        Rolling moving average window, by default 20.
    num_std : float, optional
        Number of standard deviations for outer bands, by default 2.0.

    Returns
    -------
    Tuple[pd.Series, pd.Series, pd.Series, pd.Series]
        (middle_band, upper_band, lower_band, pct_b)
    """
    middle_band = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()

    upper_band = middle_band + num_std * rolling_std
    lower_band = middle_band - num_std * rolling_std
    pct_b = (series - lower_band) / (upper_band - lower_band + 1e-9)

    return middle_band, upper_band, lower_band, pct_b


def compute_rolling_drawdown(series: pd.Series, window: int = 252) -> pd.Series:
    """
    Compute rolling peak-to-trough drawdown from the rolling peak.

    Parameters
    ----------
    series : pd.Series
        Price series.
    window : int, optional
        Lookback window in bars (default 252 bars for ~1 year).

    Returns
    -------
    pd.Series
        Drawdown values (<= 0.0).
    """
    roll_max = series.rolling(window=window, min_periods=min(30, len(series))).max()
    drawdown = (series - roll_max) / (roll_max + 1e-9)
    return drawdown
