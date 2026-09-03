# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Macro Credit Spreads & Multi-Horizon Realized Volatility Surface Features
========================================================================
Extracts institutional credit spread risk indicators (HYG/IEI, HYG/LQD)
and multi-horizon realized volatility surfaces to condition regime detection
without overfitting to raw equity price returns.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class MacroVolFeatureExtractor:
    """
    Extracts multi-horizon volatility surfaces and macro credit risk proxies.

    Parameters
    ----------
    short_vol_window : int, optional
        Short-term volatility window in trading days (default 5).
    med_vol_window : int, optional
        Medium-term volatility window in trading days (default 21).
    long_vol_window : int, optional
        Long-term volatility window in trading days (default 63).
    vov_window : int, optional
        Window for volatility-of-volatility calculation (default 21).
    """

    def __init__(
        self,
        short_vol_window: int = 5,
        med_vol_window: int = 21,
        long_vol_window: int = 63,
        vov_window: int = 21,
    ):
        self.short_vol_window = short_vol_window
        self.med_vol_window = med_vol_window
        self.long_vol_window = long_vol_window
        self.vov_window = vov_window

    def compute_volatility_surface(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        high_col: Optional[str] = "high",
        low_col: Optional[str] = "low",
    ) -> pd.DataFrame:
        """
        Compute multi-horizon annualized realized volatility surface.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing stock price series with a 'date' or datetime index.
        price_col : str
            Closing price column name.
        high_col : Optional[str]
            High price column name (for Garman-Klass / Parkinson vol).
        low_col : Optional[str]
            Low price column name.

        Returns
        -------
        pd.DataFrame
            Enriched DataFrame with:
            - vol_5d, vol_21d, vol_63d (annualized realized volatilities)
            - vol_ratio (vol_5d / vol_21d)
            - vol_term_spread (vol_5d - vol_63d)
            - vov_21d (volatility of volatility)
            - garman_klass_vol (if high/low available)
        """
        res = df.copy()
        prices = res[price_col].astype(float)
        log_ret = np.log(prices / prices.shift(1))

        # Close-to-close annualized rolling realized volatilities
        annual_factor = np.sqrt(252.0)
        res["vol_5d"] = log_ret.rolling(window=self.short_vol_window, min_periods=self.short_vol_window).std() * annual_factor
        res["vol_21d"] = log_ret.rolling(window=self.med_vol_window, min_periods=self.med_vol_window).std() * annual_factor
        res["vol_63d"] = log_ret.rolling(window=self.long_vol_window, min_periods=self.long_vol_window).std() * annual_factor

        # Forward fill initial windows gracefully
        res["vol_5d"] = res["vol_5d"].bfill().fillna(0.15)
        res["vol_21d"] = res["vol_21d"].bfill().fillna(0.18)
        res["vol_63d"] = res["vol_63d"].bfill().fillna(0.20)

        # Volatility term structure ratios
        # vol_ratio > 1.20 indicates acute short-term volatility inversion / stress
        res["vol_ratio"] = res["vol_5d"] / np.maximum(res["vol_21d"], 1e-4)
        res["vol_term_spread"] = res["vol_5d"] - res["vol_63d"]

        # Volatility of Volatility (VoV)
        res["vov_21d"] = (
            res["vol_21d"].rolling(window=self.vov_window, min_periods=self.vov_window).std()
        ).bfill().fillna(0.02)

        # Parkinson / Garman-Klass range-based volatility if high/low exist
        if high_col in res.columns and low_col in res.columns:
            highs = res[high_col].astype(float)
            lows = res[low_col].astype(float)
            # Parkinson volatility: sqrt( (1 / (4 * ln(2))) * sum(ln(H/L)^2) )
            parkinson_daily = np.sqrt((1.0 / (4.0 * np.log(2.0))) * (np.log(highs / lows) ** 2))
            res["vol_parkinson_21d"] = (
                parkinson_daily.rolling(window=self.med_vol_window, min_periods=self.med_vol_window).mean()
                * annual_factor
            ).bfill().fillna(res["vol_21d"])

        return res

    def compute_credit_spread_features(
        self,
        hyg_df: Optional[pd.DataFrame] = None,
        iei_df: Optional[pd.DataFrame] = None,
        dates: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Compute institutional credit risk appetite ratio: HYG / IEI (or HYG / LQD).

        Parameters
        ----------
        hyg_df : Optional[pd.DataFrame]
            High-yield bond ETF DataFrame with 'date' and 'close'.
        iei_df : Optional[pd.DataFrame]
            Treasury ETF DataFrame with 'date' and 'close'.
        dates : Optional[pd.Series]
            Target date index to align against.

        Returns
        -------
        pd.DataFrame
            DataFrame with:
            - credit_ratio (HYG close / IEI close)
            - credit_mom_21d (21-day % change in credit ratio)
            - credit_zscore (rolling 63-day z-score of credit ratio)
            - credit_stress_flag (1 if credit ratio breaks down, 0 otherwise)
        """
        if hyg_df is not None and iei_df is not None and not hyg_df.empty and not iei_df.empty:
            # Merge on date
            h_sub = hyg_df[["date", "close"]].rename(columns={"close": "hyg_close"}).copy()
            i_sub = iei_df[["date", "close"]].rename(columns={"close": "iei_close"}).copy()
            h_sub["date"] = pd.to_datetime(h_sub["date"]).dt.strftime("%Y-%m-%d")
            i_sub["date"] = pd.to_datetime(i_sub["date"]).dt.strftime("%Y-%m-%d")

            merged = pd.merge(h_sub, i_sub, on="date", how="inner").sort_values("date")
            merged["credit_ratio"] = merged["hyg_close"] / np.maximum(merged["iei_close"], 1e-4)
            merged["credit_mom_21d"] = merged["credit_ratio"].pct_change(21).fillna(0.0)

            roll_mean = merged["credit_ratio"].rolling(window=63, min_periods=21).mean()
            roll_std = merged["credit_ratio"].rolling(window=63, min_periods=21).std()
            merged["credit_zscore"] = ((merged["credit_ratio"] - roll_mean) / np.maximum(roll_std, 1e-4)).fillna(0.0)
            merged["credit_stress_flag"] = (merged["credit_zscore"] < -1.25).astype(int)

            if dates is not None:
                # Reindex to match requested dates
                d_df = pd.DataFrame({"date": [str(d)[:10] for d in dates]})
                out = pd.merge(d_df, merged, on="date", how="left").ffill().bfill()
                return out

            return merged

        # Fallback if credit ETFs are not directly provided:
        # Construct synthetic credit risk proxy using dates
        n = len(dates) if dates is not None else 1
        dummy_dates = [str(d)[:10] for d in dates] if dates is not None else ["1970-01-01"]
        return pd.DataFrame(
            {
                "date": dummy_dates,
                "credit_ratio": np.ones(n, dtype=np.float64),
                "credit_mom_21d": np.zeros(n, dtype=np.float64),
                "credit_zscore": np.zeros(n, dtype=np.float64),
                "credit_stress_flag": np.zeros(n, dtype=np.int64),
            }
        )

    def extract_bocd_signal(
        self,
        df: pd.DataFrame,
        credit_df: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Construct a multi-factor macroeconomic stress signal to feed into BOCD.
        Fuses volatility term-structure inversion, credit spread momentum,
        and drawdown velocity rather than fitting raw price returns.

        Parameters
        ----------
        df : pd.DataFrame
            Stock data with computed volatility surface.
        credit_df : Optional[pd.DataFrame]
            Computed credit spread features.

        Returns
        -------
        pd.Series
            Normalized institutional regime stress series Z_t for BOCD analysis.
        """
        # 1. Volatility inversion component: log(vol_5d / vol_21d)
        vol_ratio = df.get("vol_ratio", pd.Series(1.0, index=df.index))
        vol_signal = np.log(np.maximum(vol_ratio, 0.2))

        # 2. Vol term spread normalized
        vol_term_spread = df.get("vol_term_spread", pd.Series(0.0, index=df.index))
        vol_term_norm = vol_term_spread / (df.get("vol_63d", pd.Series(0.2, index=df.index)) + 1e-4)

        # 3. Drawdown velocity
        if "close" in df.columns:
            cummax = df["close"].cummax()
            dd = (df["close"] - cummax) / cummax
            dd_velocity = dd - dd.shift(5).fillna(0.0)
        else:
            dd_velocity = pd.Series(0.0, index=df.index)

        # 4. Credit momentum component (if available)
        credit_mom = pd.Series(0.0, index=df.index)
        if credit_df is not None and "credit_mom_21d" in credit_df.columns:
            if "date" in df.columns and "date" in credit_df.columns:
                m = pd.merge(
                    df[["date"]].astype(str),
                    credit_df[["date", "credit_mom_21d"]].astype(str),
                    on="date",
                    how="left",
                )["credit_mom_21d"].astype(float).fillna(0.0)
                credit_mom = pd.Series(m.values, index=df.index)

        # Synthesize multi-factor stress signal
        # High positive values = high volatility shock, credit widening, rapid drawdown
        stress_signal = (
            1.5 * vol_signal
            + 1.0 * vol_term_norm
            - 2.5 * credit_mom
            - 2.0 * dd_velocity
        )

        # Rolling standard score (z-score)
        roll_m = stress_signal.rolling(window=63, min_periods=10).mean()
        roll_s = stress_signal.rolling(window=63, min_periods=10).std()
        normalized_signal = ((stress_signal - roll_m) / np.maximum(roll_s, 1e-4)).bfill().fillna(0.0)

        return normalized_signal

