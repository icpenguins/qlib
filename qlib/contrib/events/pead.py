# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Post-Earnings Announcement Drift (PEAD) Engine
==============================================
Quantitative modeling of Standardized Unexpected Earnings (SUE),
announcement Cumulative Abnormal Return (CAR[0, 1]), and multi-week
momentum drift continuation.

Mathematical Foundation:
------------------------
1. Standardized Unexpected Earnings (SUE):
       SUE_q = [Actual_EPS_q - Consensus_EPS_q] / sigma(Forecast_Error)

2. Announcement Immediate Gap (CAR[0, 1]):
       Gap% = (P_open,t - P_close,t-1) / P_close,t-1
       CAR[0, 1] = (P_close,t+1 - P_close,t-1) / P_close,t-1 - R_mkt

3. Exponential Post-Earnings Drift Attenuation:
       PEAD_Momentum(Delta_t) = SUE_q * exp(-Delta_t / tau_drift)
       where tau_drift ~= 21 trading days (1 month half-life).

4. Key Momentum Event Extractor:
       Extracts historical inflection points (Earnings Beats/Misses,
       BOCD Structural Changepoints, FOMC rate pivots) to display
       interactively on the main stock chart.
"""

import math
import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class PEADEngine:
    """
    Evaluates Post-Earnings Announcement Drift (PEAD) alpha dynamics,
    Standardized Unexpected Earnings (SUE), and momentum inflection events.
    """

    def __init__(self, drift_half_life_days: float = 21.0):
        self.tau = max(5.0, float(drift_half_life_days))

    @staticmethod
    def compute_sue(
        actual_eps: float,
        estimate_eps: float,
        surprise_history: Optional[List[float]] = None,
    ) -> float:
        """
        Compute Standardized Unexpected Earnings (SUE).
        """
        diff = actual_eps - estimate_eps
        if surprise_history and len(surprise_history) >= 2:
            std_err = float(np.std(surprise_history, ddof=1))
            if std_err > 1e-4:
                return round(float(diff / std_err), 2)

        # Normalization fallback based on estimate scale
        scale = max(0.05, abs(estimate_eps) * 0.15)
        return round(float(diff / scale), 2)

    def evaluate_recent_pead(
        self,
        df: pd.DataFrame,
        earnings_history: List[Dict[str, Any]],
        current_date: Union[str, datetime.date, pd.Timestamp],
    ) -> Dict[str, Any]:
        """
        Analyze the most recent quarterly earnings report for active PEAD drift momentum.
        """
        if not earnings_history:
            return {
                "has_pead": False,
                "latest_report_date": None,
                "days_since_report": None,
                "sue_score": 0.0,
                "eps_actual": None,
                "eps_estimate": None,
                "surprise_pct": 0.0,
                "announcement_gap_pct": 0.0,
                "post_earnings_drift_pct": 0.0,
                "drift_regime": "None / No Event History",
                "pead_drift_score": 0.0,
                "badge_class": "bg-gray-800 text-gray-400 border-gray-700",
                "description": "No prior earnings history recorded.",
            }

        curr_dt = pd.to_datetime(current_date).tz_localize(None).floor("D")

        # Sort history by date descending
        past_reports = []
        for rep in earnings_history:
            rep_dt = pd.to_datetime(rep.get("date") or rep.get("quarter")).tz_localize(None).floor("D")
            if rep_dt <= curr_dt:
                past_reports.append((rep_dt, rep))

        past_reports.sort(key=lambda x: x[0], reverse=True)
        if not past_reports:
            return self.evaluate_recent_pead(df, [], current_date)

        latest_dt, latest_rep = past_reports[0]
        latest_date_str = latest_dt.strftime("%Y-%m-%d")

        # Trading days since announcement
        trading_days_since = len(pd.bdate_range(start=latest_dt + pd.Timedelta(days=1), end=curr_dt))

        act = float(latest_rep.get("eps_actual", latest_rep.get("actual", 0.0)))
        est = float(latest_rep.get("eps_estimate", latest_rep.get("estimate", 0.0)))
        surp_pct = float(latest_rep.get("surprise_pct", latest_rep.get("surprisePercent", 0.0)))

        # SUE calculation
        hist_diffs = [
            float(r.get("eps_actual", r.get("actual", 0.0))) - float(r.get("eps_estimate", r.get("estimate", 0.0)))
            for _, r in past_reports[:8]
        ]
        sue = self.compute_sue(act, est, hist_diffs)

        # Calculate price reactions from df
        gap_pct = 0.0
        drift_pct = 0.0
        df_sorted = df.copy().sort_values("date").reset_index(drop=True)
        date_matches = df_sorted.index[df_sorted["date"] >= latest_date_str].tolist()

        if date_matches:
            ev_idx = date_matches[0]
            # Gap reaction: close[ev_idx] vs close[ev_idx - 1]
            if ev_idx > 0:
                p_prior = float(df_sorted.loc[ev_idx - 1, "close"])
                p_ev = float(df_sorted.loc[ev_idx, "close"])
                gap_pct = round(((p_ev - p_prior) / p_prior) * 100.0, 2)

            # Drift from day +1 to current close
            curr_idx = len(df_sorted) - 1
            if curr_idx > ev_idx:
                p_post = float(df_sorted.loc[ev_idx, "close"])
                p_now = float(df_sorted.loc[curr_idx, "close"])
                drift_pct = round(((p_now - p_post) / p_post) * 100.0, 2)

        # Exponential drift decay score
        drift_score = round(sue * math.exp(-trading_days_since / self.tau), 2)

        # Drift regime classification
        if trading_days_since <= 45 and (sue >= 1.0 or gap_pct >= 2.5):
            drift_regime = "Active Bullish PEAD Drift"
            badge_class = "bg-emerald-500/15 text-emerald-400 border-emerald-500/30"
            desc = (
                f"Active Post-Earnings Bullish Drift (Beat +{surp_pct:.1f}%, SUE +{sue:.2f}, "
                f"Day-1 Gap +{gap_pct:+.1f}%). Institutional accumulation remains active."
            )
        elif trading_days_since <= 45 and (sue <= -1.0 or gap_pct <= -2.5):
            drift_regime = "Active Bearish PEAD Breakdown"
            badge_class = "bg-rose-500/15 text-rose-400 border-rose-500/30"
            desc = (
                f"Active Post-Earnings Bearish Drift (Miss {surp_pct:.1f}%, SUE {sue:.2f}, "
                f"Day-1 Gap {gap_pct:+.1f}%). De-gross long exposure; institutional selling persists."
            )
        else:
            drift_regime = "Matured / Neutral PEAD"
            badge_class = "bg-gray-800 text-gray-300 border-gray-700"
            desc = f"Latest quarterly earnings reaction matured ({trading_days_since} trading days ago). Normal price action resumes."

        return {
            "has_pead": True,
            "latest_report_date": latest_date_str,
            "days_since_report": trading_days_since,
            "sue_score": sue,
            "eps_actual": round(act, 2),
            "eps_estimate": round(est, 2),
            "surprise_pct": round(surp_pct, 2),
            "announcement_gap_pct": gap_pct,
            "post_earnings_drift_pct": drift_pct,
            "drift_regime": drift_regime,
            "pead_drift_score": drift_score,
            "badge_class": badge_class,
            "description": desc,
        }

    @staticmethod
    def compute_report_reaction(
        df_sorted: pd.DataFrame,
        report_date: str,
        drift_trading_days: int = 21,
    ) -> Dict[str, Optional[float]]:
        """
        Compute the Day-1 announcement gap and a fixed N-trading-day post-earnings
        drift for a single historical earnings report date.

        Returns ``None`` for a field when the price history does not extend far
        enough to compute it (e.g. the drift window runs past the last available
        trading day), rather than fabricating a value.
        """
        date_matches = df_sorted.index[df_sorted["date"] >= report_date].tolist()
        if not date_matches:
            return {"announcement_gap_pct": None, "drift_pct": None}
        ev_idx = date_matches[0]

        gap_pct = None
        if ev_idx > 0:
            p_prior = float(df_sorted.loc[ev_idx - 1, "close"])
            p_ev = float(df_sorted.loc[ev_idx, "close"])
            if p_prior > 0:
                gap_pct = round(((p_ev - p_prior) / p_prior) * 100.0, 2)

        drift_pct = None
        target_idx = ev_idx + drift_trading_days
        if target_idx < len(df_sorted):
            p_post = float(df_sorted.loc[ev_idx, "close"])
            p_target = float(df_sorted.loc[target_idx, "close"])
            if p_post > 0:
                drift_pct = round(((p_target - p_post) / p_post) * 100.0, 2)

        return {"announcement_gap_pct": gap_pct, "drift_pct": drift_pct}

    def evaluate_earnings_history(
        self,
        df: pd.DataFrame,
        earnings_history: List[Dict[str, Any]],
        current_date: Union[str, datetime.date, pd.Timestamp],
        lookback: int = 4,
        drift_trading_days: int = 21,
    ) -> List[Dict[str, Any]]:
        """
        Annotate the most recent ``lookback`` earnings reports with SUE score,
        Day-1 announcement gap, and a fixed ``drift_trading_days``-trading-day
        post-earnings drift -- computed with the same methodology
        :meth:`evaluate_recent_pead` uses for the single most recent report, so
        the per-quarter history table and the "most recent report" summary card
        can never disagree about the same event.

        Prior to this method, the per-quarter history rows fed to the report
        template carried only raw EPS/surprise fields (see
        ``events_data.py``'s synthetic generator) with no SUE/gap/drift computed
        at all -- the template read nonexistent keys for those three fields and
        always rendered "N/A", even though the most-recent-report summary card
        (fed by :meth:`evaluate_recent_pead`) showed real numbers for what could
        be the very same report.
        """
        if not earnings_history:
            return []

        curr_dt = pd.to_datetime(current_date).tz_localize(None).floor("D")
        past_reports = []
        for rep in earnings_history:
            rep_dt = pd.to_datetime(rep.get("date") or rep.get("quarter")).tz_localize(None).floor("D")
            if rep_dt <= curr_dt:
                past_reports.append((rep_dt, rep))
        past_reports.sort(key=lambda x: x[0], reverse=True)

        df_sorted = df.copy().sort_values("date").reset_index(drop=True)
        annotated = []
        for i, (rep_dt, rep) in enumerate(past_reports[:lookback]):
            act = float(rep.get("eps_actual", rep.get("actual", 0.0)))
            est = float(rep.get("eps_estimate", rep.get("estimate", 0.0)))
            surp_pct = float(rep.get("surprise_pct", rep.get("surprisePercent", 0.0)))
            hist_diffs = [
                float(r.get("eps_actual", r.get("actual", 0.0))) - float(r.get("eps_estimate", r.get("estimate", 0.0)))
                for _, r in past_reports[i + 1 : i + 9]
            ]
            sue = self.compute_sue(act, est, hist_diffs)
            reaction = self.compute_report_reaction(df_sorted, rep_dt.strftime("%Y-%m-%d"), drift_trading_days)

            annotated.append({
                "date": rep_dt.strftime("%Y-%m-%d"),
                "eps_actual": round(act, 2),
                "eps_estimate": round(est, 2),
                "surprise_pct": round(surp_pct, 2),
                "sue_score": sue,
                "announcement_gap_pct": reaction["announcement_gap_pct"],
                "drift_pct": reaction["drift_pct"],
            })
        return annotated

    @staticmethod
    def extract_key_momentum_events(
        df: pd.DataFrame,
        earnings_history: Optional[List[Dict[str, Any]]] = None,
        bocd_changepoints: Optional[List[Dict[str, Any]]] = None,
        fomc_dates: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract key inflection events that materially altered price momentum.
        Returns a sorted list of event objects designed for plotting on the main chart.
        """
        df_sorted = df.copy().sort_values("date").reset_index(drop=True)
        if df_sorted.empty:
            return []

        date_to_idx = {d: i for i, d in enumerate(df_sorted["date"])}
        events = []

        # 1. Earnings Surprise Events
        if earnings_history:
            for rep in earnings_history:
                rep_date = rep.get("date") or rep.get("quarter")
                if not rep_date or rep_date not in date_to_idx:
                    # Find closest trading day within 3 calendar days
                    rep_dt = pd.to_datetime(rep_date)
                    matched_idx = None
                    for offset in range(-2, 3):
                        test_str = (rep_dt + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
                        if test_str in date_to_idx:
                            matched_idx = date_to_idx[test_str]
                            rep_date = test_str
                            break
                    if matched_idx is None:
                        continue
                else:
                    matched_idx = date_to_idx[rep_date]

                close_p = float(df_sorted.loc[matched_idx, "close"])
                # Calculate Day-1 price reaction
                prior_idx = max(0, matched_idx - 1)
                p_prior = float(df_sorted.loc[prior_idx, "close"])
                gap_pct = round(((close_p - p_prior) / p_prior) * 100.0, 1)

                # Calculate 30-day forward drift
                drift_idx = min(len(df_sorted) - 1, matched_idx + 21)
                p_drift = float(df_sorted.loc[drift_idx, "close"])
                drift_30d = round(((p_drift - close_p) / close_p) * 100.0, 1)

                act = float(rep.get("eps_actual", rep.get("actual", 0.0)))
                est = float(rep.get("eps_estimate", rep.get("estimate", 0.0)))
                surp_pct = float(rep.get("surprise_pct", rep.get("surprisePercent", 0.0)))

                is_beat = (surp_pct >= 0 or gap_pct >= 0)
                event_type = "EARNINGS_BEAT" if is_beat else "EARNINGS_MISS"
                badge_text = f"E ▲ +{surp_pct:.1f}%" if is_beat else f"E ▼ {surp_pct:.1f}%"
                badge_color = "#10b981" if is_beat else "#f43f5e"

                events.append({
                    "date": rep_date,
                    "index": matched_idx,
                    "price": round(close_p, 2),
                    "type": event_type,
                    "badge": badge_text,
                    "badge_color": badge_color,
                    "category": "Corporate Earnings",
                    "title": f"Quarterly Earnings {'Beat' if is_beat else 'Miss'}",
                    "details": f"Reported: ${act:.2f} vs Est: ${est:.2f} ({surp_pct:+.1f}%)",
                    "gap_pct": gap_pct,
                    "drift_30d_pct": drift_30d,
                    "momentum_impact": f"Day-1: {gap_pct:+.1f}% | 30d Drift: {drift_30d:+.1f}%",
                })

        # 2. BOCD Structural Regime Changepoint Pivots
        if bocd_changepoints:
            for cp in bocd_changepoints:
                cp_date = cp.get("date")
                if not cp_date or cp_date not in date_to_idx:
                    continue
                cp_idx = date_to_idx[cp_date]
                close_p = float(df_sorted.loc[cp_idx, "close"])
                state = cp.get("state", 0)
                state_name = cp.get("name", f"State {state}")

                badge_color = "#8b5cf6" if state == 0 else "#f59e0b" if state == 1 else "#f43f5e"
                events.append({
                    "date": cp_date,
                    "index": cp_idx,
                    "price": round(close_p, 2),
                    "type": "BOCD_SHIFT",
                    "badge": f"⚡ State {state}",
                    "badge_color": badge_color,
                    "category": "Regime Pivot",
                    "title": f"Bayesian Changepoint: {state_name}",
                    "details": cp.get("description", "Structural volatility/trend break identified by BOCD."),
                    "gap_pct": 0.0,
                    "drift_30d_pct": None,
                    "momentum_impact": f"Regime transitioned to {state_name}",
                })

        # 3. Major FOMC Macro Announcements
        if fomc_dates:
            for f_date in fomc_dates:
                if f_date in date_to_idx:
                    f_idx = date_to_idx[f_date]
                    close_p = float(df_sorted.loc[f_idx, "close"])
                    # Check if market had a significant reaction (> 1.5% move)
                    prior_idx = max(0, f_idx - 1)
                    p_prior = float(df_sorted.loc[prior_idx, "close"])
                    reaction_pct = round(((close_p - p_prior) / p_prior) * 100.0, 1)

                    if abs(reaction_pct) >= 1.2:
                        events.append({
                            "date": f_date,
                            "index": f_idx,
                            "price": round(close_p, 2),
                            "type": "FOMC_PIVOT",
                            "badge": "◆ FOMC",
                            "badge_color": "#06b6d4",
                            "category": "Macro Catalyst",
                            "title": "Federal Reserve FOMC Policy Decision",
                            "details": f"Fed interest rate policy announcement. Market reacted {reaction_pct:+.1f}%.",
                            "gap_pct": reaction_pct,
                            "drift_30d_pct": None,
                            "momentum_impact": f"Macro reaction: {reaction_pct:+.1f}%",
                        })

        # Sort all momentum events chronologically
        events.sort(key=lambda x: x["date"])
        return events

