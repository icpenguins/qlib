# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Market Regime Classifier
========================
Combines Bayesian Online Changepoint Detection (BOCD) with macro credit spreads
and multi-horizon realized volatility surfaces to produce real-time, non-lagging
market regime classifications.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd

from .bocd import BayesianOnlineChangepointDetector
from .macro_vol_features import MacroVolFeatureExtractor


class MarketRegimeClassifier:
    """
    Synthesizes Bayesian Online Changepoint Detection with Macro Credit Spreads
    and Realized Volatility Surfaces into 4 institutional market states:

    - State 0: Low-Vol Trending Bull (Risk-On)
    - State 1: Mean-Reverting Choppy Neutral (Consolidation)
    - State 2: High-Vol Liquidation / Risk-Off (Stress)
    - State 3: Regime Transition / Inflection Alert (High Changepoint Probability)

    Parameters
    ----------
    expected_run_length : float, optional
        Expected duration between regime changepoints (default 63 trading days).
    bocd_threshold : float, optional
        Changepoint probability threshold to trigger Regime Transition Alert (default 0.35).
    """

    REGIME_NAMES = {
        0: "Low-Vol Trending Bull",
        1: "Mean-Reverting Choppy Neutral",
        2: "High-Vol Liquidation / Risk-Off",
        3: "Regime Transition / Inflection Alert",
    }

    REGIME_COLORS = {
        0: "#10b981",  # Emerald Green
        1: "#3b82f6",  # Blue
        2: "#ef4444",  # Red
        3: "#f59e0b",  # Amber / Warning
    }

    def __init__(
        self,
        expected_run_length: float = 63.0,
        bocd_threshold: float = 0.35,
    ):
        self.expected_run_length = expected_run_length
        self.bocd_threshold = bocd_threshold
        self.feature_extractor = MacroVolFeatureExtractor()
        self.detector = BayesianOnlineChangepointDetector(expected_run_length=expected_run_length)

    def analyze(
        self,
        df: pd.DataFrame,
        hyg_df: Optional[pd.DataFrame] = None,
        iei_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Run full regime analysis across a stock price DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Stock price DataFrame with 'date', 'close', and optionally 'high', 'low', 'open', 'volume'.
        hyg_df : Optional[pd.DataFrame]
            High Yield ETF DataFrame.
        iei_df : Optional[pd.DataFrame]
            Treasury ETF DataFrame.

        Returns
        -------
        pd.DataFrame
            Enriched DataFrame with:
            - vol_5d, vol_21d, vol_63d, vol_ratio
            - credit_ratio, credit_mom_21d, credit_zscore
            - changepoint_prob, expected_run_length
            - regime_state (0, 1, 2, 3)
            - regime_name
            - prob_state_0, prob_state_1, prob_state_2, prob_state_3
        """
        if df.empty or len(df) < 10:
            raise ValueError("Insufficient data points for regime analysis (minimum 10 bars required).")

        res = df.copy()
        dates = res["date"] if "date" in res.columns else res.index

        # 1. Compute multi-horizon volatility surface
        res = self.feature_extractor.compute_volatility_surface(res)

        # 2. Extract credit spread features
        credit_df = self.feature_extractor.compute_credit_spread_features(
            hyg_df=hyg_df, iei_df=iei_df, dates=dates
        )

        # 3. Form macro stress signal and run BOCD
        stress_signal = self.feature_extractor.extract_bocd_signal(res, credit_df)
        bocd_results = self.detector.batch_process(stress_signal)

        res["changepoint_prob"] = bocd_results["changepoint_prob"].values
        res["expected_run_length"] = bocd_results["expected_run_length"].values

        # 4. Integrate credit spread columns
        if "credit_ratio" in credit_df.columns:
            res["credit_ratio"] = credit_df["credit_ratio"].values[:len(res)]
            res["credit_mom_21d"] = credit_df["credit_mom_21d"].values[:len(res)]
            res["credit_zscore"] = credit_df["credit_zscore"].values[:len(res)]
        else:
            res["credit_ratio"] = 1.0
            res["credit_mom_21d"] = 0.0
            res["credit_zscore"] = 0.0

        # 5. Classify 4-state regime and posterior probabilities
        regime_states = []
        regime_names = []
        p0_list = []
        p1_list = []
        p2_list = []
        p3_list = []

        sma50 = res["close"].rolling(window=50, min_periods=10).mean()

        for i in range(len(res)):
            cp_p = float(res["changepoint_prob"].iloc[i])
            v_ratio = float(res["vol_ratio"].iloc[i])
            v_21 = float(res["vol_21d"].iloc[i])
            c_mom = float(res["credit_mom_21d"].iloc[i])
            close = float(res["close"].iloc[i])
            ma50 = float(sma50.iloc[i])

            # Raw unnormalized state likelihoods
            # State 3: Transition / Inflection Alert (driven by BOCD changepoint probability)
            w3 = cp_p * 2.5

            # State 2: Liquidation / Risk-Off (driven by vol inversion > 1.15, high vol, credit collapse)
            w2 = (
                max(0.0, v_ratio - 1.0) * 2.0
                + (v_21 / 0.30) * 1.0
                + max(0.0, -c_mom * 20.0)
                + (1.0 if close < ma50 else 0.0) * 0.8
            )

            # State 0: Low-Vol Trending Bull (low vol ratio, price > 50ma, credit expanding)
            w0 = (
                max(0.0, 1.2 - v_ratio) * 1.5
                + (1.0 if close >= ma50 else 0.0) * 1.5
                + max(0.0, c_mom * 15.0)
                + max(0.0, 0.25 - v_21) * 2.0
            )

            # State 1: Mean-Reverting Choppy (low volatility, neutral credit, low trend strength)
            w1 = (
                max(0.0, 0.25 - abs(close - ma50) / ma50) * 2.0
                + max(0.0, 0.22 - v_21) * 1.5
                + 0.5
            )

            # Softmax / Normalize
            weights = np.array([w0, w1, w2, w3], dtype=np.float64)
            weights = np.maximum(weights, 0.01)
            probs = weights / np.sum(weights)

            p0, p1, p2, p3 = probs[0], probs[1], probs[2], probs[3]
            p0_list.append(round(float(p0), 4))
            p1_list.append(round(float(p1), 4))
            p2_list.append(round(float(p2), 4))
            p3_list.append(round(float(p3), 4))

            # Discrete state decision:
            # If changepoint prob exceeds threshold, trigger State 3 (Alert)
            if cp_p >= self.bocd_threshold or p3 > 0.45:
                state = 3
            else:
                state = int(np.argmax([p0, p1, p2]))

            regime_states.append(state)
            regime_names.append(self.REGIME_NAMES[state])

        res["regime_state"] = regime_states
        res["regime_name"] = regime_names
        res["prob_state_0"] = p0_list
        res["prob_state_1"] = p1_list
        res["prob_state_2"] = p2_list
        res["prob_state_3"] = p3_list

        return res

    def get_current_regime_summary(self, df_with_regime: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract the latest real-time regime diagnosis and risk guidance.

        Parameters
        ----------
        df_with_regime : pd.DataFrame
            DataFrame produced by analyze().

        Returns
        -------
        Dict[str, Any]
            Dictionary summarizing current regime, changepoint risk %,
            vol surface, credit spread status, and actionable risk guidance.
        """
        latest = df_with_regime.iloc[-1]
        state = int(latest["regime_state"])
        cp_prob = float(latest["changepoint_prob"])
        exp_rl = float(latest["expected_run_length"])
        vol_5 = float(latest["vol_5d"]) * 100.0
        vol_21 = float(latest["vol_21d"]) * 100.0
        vol_ratio = float(latest["vol_ratio"])
        credit_mom = float(latest.get("credit_mom_21d", 0.0)) * 100.0

        guidance = {
            0: {
                "action": "FAVOR TREND FOLLOWING",
                "risk_multiplier": 1.0,
                "description": "Low-volatility expansion with supportive liquidity. Trend continuation favored.",
                "badge_class": "bg-emerald-500/10 text-emerald-400 border-emerald-500/30",
            },
            1: {
                "action": "FAVOR MEAN-REVERSION",
                "risk_multiplier": 0.8,
                "description": "Volatility compression in range-bound market. Avoid breakout chasing; buy support and trim at resistance.",
                "badge_class": "bg-blue-500/10 text-blue-400 border-blue-500/30",
            },
            2: {
                "action": "RISK-OFF CAPITAL PRESERVATION",
                "risk_multiplier": 0.4,
                "description": "High volatility inversion and liquidity stress. Reduce position size, tighten stops, and wait for vol peaks.",
                "badge_class": "bg-red-500/10 text-red-400 border-red-500/30",
            },
            3: {
                "action": "REGIME SHIFT ALERT / PAUSE ENTRIES",
                "risk_multiplier": 0.5,
                "description": "Active Bayesian changepoint detected. Market structure is undergoing transition; wait for run-length stabilization.",
                "badge_class": "bg-amber-500/10 text-amber-400 border-amber-500/30",
            },
        }

        info = guidance.get(state, guidance[1])

        return {
            "state": state,
            "name": self.REGIME_NAMES[state],
            "color": self.REGIME_COLORS[state],
            "changepoint_prob_pct": round(cp_prob * 100.0, 1),
            "expected_run_length_days": round(exp_rl, 1),
            "vol_5d_pct": round(vol_5, 1),
            "vol_21d_pct": round(vol_21, 1),
            "vol_ratio": round(vol_ratio, 2),
            "credit_mom_pct": round(credit_mom, 2),
            "action": info["action"],
            "risk_multiplier": info["risk_multiplier"],
            "description": info["description"],
            "badge_class": info["badge_class"],
            "probabilities": {
                "bull": float(latest["prob_state_0"]),
                "neutral": float(latest["prob_state_1"]),
                "risk_off": float(latest["prob_state_2"]),
                "transition": float(latest["prob_state_3"]),
            },
        }

