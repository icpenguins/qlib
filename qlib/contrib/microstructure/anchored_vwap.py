# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Anchored Volume-Weighted Average Price (AVWAP) Engine
=====================================================
Computes institutional Anchored VWAP trajectories, volume-weighted dispersion
variance, continuous standard deviation bands (+/-1 sigma, +/-2 sigma), and
standardized Z-score deviation spreads across dynamic macro inflection anchors:
- YTD (Year-to-Date)
- QTD (Quarter-to-Date)
- 52-Week High Anchor
- 52-Week Low Anchor
- Custom / Event Anchors (Earnings, Macro Inflections)
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd


class AnchoredVWAPCalculator:
    """
    Computes Anchored VWAP and dispersion bands for institutional order flow analysis.

    Parameters
    ----------
    price_col : str, optional
        Column name for closing price (default 'close').
    volume_col : str, optional
        Column name for volume (default 'volume').
    high_col : Optional[str], optional
        Column name for high price (default 'high').
    low_col : Optional[str], optional
        Column name for low price (default 'low').
    date_col : str, optional
        Column name for date (default 'date').
    """

    def __init__(
        self,
        price_col: str = "close",
        volume_col: str = "volume",
        high_col: Optional[str] = "high",
        low_col: Optional[str] = "low",
        date_col: str = "date",
    ):
        self.price_col = price_col
        self.volume_col = volume_col
        self.high_col = high_col
        self.low_col = low_col
        self.date_col = date_col

    def _get_typical_price(self, df: pd.DataFrame) -> pd.Series:
        """Compute typical price (H+L+C)/3 if high and low are available, else close."""
        if (
            self.high_col in df.columns
            and self.low_col in df.columns
            and self.price_col in df.columns
        ):
            h = df[self.high_col].astype(float)
            l = df[self.low_col].astype(float)
            c = df[self.price_col].astype(float)
            return (h + l + c) / 3.0
        return df[self.price_col].astype(float)

    def calculate_single_anchor(
        self,
        df: pd.DataFrame,
        anchor_date: Union[str, pd.Timestamp, np.datetime64],
        prefix: str = "avwap",
    ) -> pd.DataFrame:
        """
        Calculate Anchored VWAP and +/- 1 sigma, +/- 2 sigma dispersion bands
        from a specific anchor date forward.

        Parameters
        ----------
        df : pd.DataFrame
            Stock price DataFrame sorted chronologically.
        anchor_date : Union[str, pd.Timestamp, np.datetime64]
            Date from which accumulation begins.
        prefix : str, optional
            Column name prefix for generated outputs.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - <prefix> : Anchored VWAP line
            - <prefix>_std : Volume-weighted standard deviation
            - <prefix>_upper_1s : +1 sigma band
            - <prefix>_lower_1s : -1 sigma band
            - <prefix>_upper_2s : +2 sigma band
            - <prefix>_lower_2s : -2 sigma band
            - <prefix>_zscore : Standardized price distance from AVWAP
        """
        res = df.copy()
        dates = pd.to_datetime(res[self.date_col])
        anchor_dt = pd.to_datetime(anchor_date)

        typical_p = self._get_typical_price(res).values
        vols = res[self.volume_col].astype(float).values
        # Ensure volumes are non-negative
        vols = np.maximum(vols, 1.0)
        pv = typical_p * vols

        n = len(res)
        avwap_vals = np.full(n, np.nan, dtype=np.float64)
        std_vals = np.full(n, np.nan, dtype=np.float64)

        # Find first index on or after anchor_date
        match_indices = np.where(dates >= anchor_dt)[0]
        if len(match_indices) == 0:
            # Anchor date is after entire dataset
            start_idx = n
        else:
            start_idx = match_indices[0]

        if start_idx < n:
            sub_pv = pv[start_idx:]
            sub_v = vols[start_idx:]
            sub_p = typical_p[start_idx:]

            cum_pv = np.cumsum(sub_pv)
            cum_v = np.cumsum(sub_v)
            cum_v_safe = np.maximum(cum_v, 1.0)

            sub_avwap = cum_pv / cum_v_safe
            avwap_vals[start_idx:] = sub_avwap

            # Vectorized volume-weighted variance calculation:
            # E[(P - AVWAP)^2] = (sum(V * P^2) / cum_v) - AVWAP^2
            cum_pv2 = np.cumsum(sub_v * (sub_p ** 2))
            var_vals = (cum_pv2 / cum_v_safe) - (sub_avwap ** 2)
            var_vals = np.maximum(var_vals, 0.0)
            sub_std = np.sqrt(var_vals)
            std_vals[start_idx:] = sub_std

        res[prefix] = avwap_vals
        res[f"{prefix}_std"] = std_vals
        res[f"{prefix}_upper_1s"] = avwap_vals + std_vals
        res[f"{prefix}_lower_1s"] = avwap_vals - std_vals
        res[f"{prefix}_upper_2s"] = avwap_vals + 2.0 * std_vals
        res[f"{prefix}_lower_2s"] = avwap_vals - 2.0 * std_vals

        # Standardized Z-Score deviation
        close_p = res[self.price_col].astype(float).values
        denom = np.where(std_vals > 1e-4, std_vals, 1e-4)
        zscore = np.where(~np.isnan(avwap_vals), (close_p - avwap_vals) / denom, np.nan)
        res[f"{prefix}_zscore"] = zscore

        return res

    def identify_anchor_dates(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Identify standard institutional market anchor dates automatically from price history:
        - YTD: First trading date of the latest year in data.
        - QTD: First trading date of the latest calendar quarter in data.
        - 52W High: Date of the 52-week peak price.
        - 52W Low: Date of the 52-week trough price.
        """
        if df.empty or self.date_col not in df.columns:
            return {}

        dates = pd.to_datetime(df[self.date_col])
        latest_date = dates.max()
        latest_year = latest_date.year

        # 1. YTD Anchor Date
        ytd_dates = dates[dates.dt.year == latest_year]
        ytd_anchor = ytd_dates.min().strftime("%Y-%m-%d")

        # 2. QTD Anchor Date
        latest_quarter = latest_date.quarter
        qtd_dates = dates[(dates.dt.year == latest_year) & (dates.dt.quarter == latest_quarter)]
        qtd_anchor = qtd_dates.min().strftime("%Y-%m-%d") if not qtd_dates.empty else ytd_anchor

        # 3. 52-Week High & Low Anchors (~trailing 252 trading days)
        trailing_window = df.tail(252)
        if not trailing_window.empty:
            high_idx = trailing_window[self.price_col].astype(float).idxmax()
            low_idx = trailing_window[self.price_col].astype(float).idxmin()
            high_52w_anchor = str(df.loc[high_idx, self.date_col])[:10]
            low_52w_anchor = str(df.loc[low_idx, self.date_col])[:10]
        else:
            high_52w_anchor = ytd_anchor
            low_52w_anchor = ytd_anchor

        return {
            "ytd": ytd_anchor,
            "qtd": qtd_anchor,
            "high_52w": high_52w_anchor,
            "low_52w": low_52w_anchor,
        }

    def compute_all_institutional_anchors(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Compute YTD, 52W High, and 52W Low Anchored VWAPs and extract summary diagnosis.

        Parameters
        ----------
        df : pd.DataFrame
            Stock history DataFrame.

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, Any]]
            (enriched_df, summary_dict)
        """
        if df.empty or len(df) < 5:
            return df, {}

        anchors = self.identify_anchor_dates(df)
        res = df.copy()

        # Compute YTD AVWAP
        if "ytd" in anchors:
            res = self.calculate_single_anchor(res, anchors["ytd"], prefix="avwap_ytd")

        # Compute 52W High AVWAP
        if "high_52w" in anchors:
            res = self.calculate_single_anchor(res, anchors["high_52w"], prefix="avwap_52w_high")

        # Compute 52W Low AVWAP
        if "low_52w" in anchors:
            res = self.calculate_single_anchor(res, anchors["low_52w"], prefix="avwap_52w_low")

        latest = res.iloc[-1]
        close_p = float(latest[self.price_col])

        ytd_val = float(latest.get("avwap_ytd", np.nan))
        ytd_std = float(latest.get("avwap_ytd_std", np.nan))
        ytd_z = float(latest.get("avwap_ytd_zscore", np.nan))

        high_val = float(latest.get("avwap_52w_high", np.nan))
        high_z = float(latest.get("avwap_52w_high_zscore", np.nan))

        low_val = float(latest.get("avwap_52w_low", np.nan))
        low_z = float(latest.get("avwap_52w_low_zscore", np.nan))

        # Actionable Diagnosis
        if not np.isnan(ytd_z):
            if ytd_z > 2.0:
                ytd_regime = "Significantly Extended Above YTD AVWAP (+2σ overbought)"
                ytd_action = "Mean-reversion pullback risk elevated; avoid chasing breakouts."
            elif ytd_z > 0.0:
                ytd_regime = "Bullish Institutional Markup (Trading above YTD AVWAP)"
                ytd_action = "Institutional buyers in control; YTD AVWAP acts as primary dynamic support."
            elif ytd_z > -1.5:
                ytd_regime = "Moderate Discount Below YTD AVWAP (-1σ support zone)"
                ytd_action = "Potential institutional accumulation / dip-buy zone."
            else:
                ytd_regime = "Deep Institutional Discount Below YTD AVWAP (<-1.5σ oversold)"
                ytd_action = "Severe liquidation extension; await AVWAP reclaim confirmation."
        else:
            ytd_regime = "N/A"
            ytd_action = "Insufficient anchor history."

        summary = {
            "anchors": anchors,
            "ytd": {
                "date": anchors.get("ytd", ""),
                "value": round(ytd_val, 2) if not np.isnan(ytd_val) else None,
                "upper_1s": round(ytd_val + ytd_std, 2) if not np.isnan(ytd_val) else None,
                "lower_1s": round(ytd_val - ytd_std, 2) if not np.isnan(ytd_val) else None,
                "zscore": round(ytd_z, 2) if not np.isnan(ytd_z) else None,
                "spread_pct": round(((close_p - ytd_val) / ytd_val) * 100.0, 2) if not np.isnan(ytd_val) and ytd_val > 0 else None,
                "regime": ytd_regime,
                "action": ytd_action,
            },
            "high_52w": {
                "date": anchors.get("high_52w", ""),
                "value": round(high_val, 2) if not np.isnan(high_val) else None,
                "zscore": round(high_z, 2) if not np.isnan(high_z) else None,
                "spread_pct": round(((close_p - high_val) / high_val) * 100.0, 2) if not np.isnan(high_val) and high_val > 0 else None,
            },
            "low_52w": {
                "date": anchors.get("low_52w", ""),
                "value": round(low_val, 2) if not np.isnan(low_val) else None,
                "zscore": round(low_z, 2) if not np.isnan(low_z) else None,
                "spread_pct": round(((close_p - low_val) / low_val) * 100.0, 2) if not np.isnan(low_val) and low_val > 0 else None,
            },
        }

        return res, summary

