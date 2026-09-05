#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Predictive Buy Timing & Monte Carlo Forecasting Engine
======================================================
Decomposed collaborating services for forward price projection:
- Parameter extraction from Regime (BOCD), GEX, and Events (PEAD)
- Support and resistance dynamic synthesis
- Thread-safe Monte Carlo Geometric Brownian Motion path simulation
- Regime-conditional buy timing recommendation rules
- Catalyst-aware execution window optimization
"""

import sys
import math
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.domain_models import (
    RegimeParams,
    GEXParams,
    PEADParams,
    BuyWindow,
    ForecastSeriesPoint,
    PredictiveForecastResult,
)
from scripts.indicators import compute_rsi

try:
    from qlib.contrib.events import RiskDegrossingEngine
except Exception:
    try:
        from events import RiskDegrossingEngine
    except Exception:
        RiskDegrossingEngine = None

# Model mechanics named constants
GEX_POS_VOL_DAMPENER: float = 0.85
GEX_NEG_VOL_ACCELERATOR: float = 1.25
VOL_MIN_CLAMP: float = 0.005
VOL_MAX_CLAMP: float = 0.045
DRIFT_MEAN_REVERSION_COEFF: float = 0.02
BOCD_JUMP_SCALE_MULT: float = 1.5
EARNINGS_GAP_SCALE_MULT: float = 2.5


class RegimeParameterExtractor:
    """Extracts and computes hazard metrics from BOCD output."""

    @staticmethod
    def extract(regime: Optional[Dict[str, Any]], forecast_days: int) -> RegimeParams:
        if not regime or not isinstance(regime, dict):
            return RegimeParams()

        state = regime.get("state")
        name = regime.get("name")
        cp_hazard_pct = float(regime.get("changepoint_prob_pct", 0.0))
        exp_run_length = float(regime.get("expected_run_length_days", 63.0))
        risk_mult = float(regime.get("risk_multiplier", 1.0))
        vol_21d_pct = regime.get("vol_21d_pct")

        h_daily = 1.0 / max(10.0, exp_run_length)
        forward_cp_prob = (1.0 - (1.0 - h_daily) ** forecast_days) * 100.0

        return RegimeParams(
            state=state,
            name=name,
            changepoint_hazard_pct=cp_hazard_pct,
            forward_changepoint_prob_pct=forward_cp_prob,
            expected_run_length_days=exp_run_length,
            risk_multiplier=risk_mult,
            daily_hazard=h_daily,
            vol_21d_pct=vol_21d_pct,
        )


class GEXParameterExtractor:
    """Extracts strike walls, gamma flip, and volatility scaling from GEX output."""

    @staticmethod
    def extract(derivatives: Optional[Dict[str, Any]]) -> GEXParams:
        if not derivatives or not isinstance(derivatives, dict):
            return GEXParams()

        gex_data = derivatives.get("gex", derivatives)
        if not isinstance(gex_data, dict):
            return GEXParams()

        net_gex_m = float(gex_data.get("net_gex_millions", gex_data.get("net_gex_dollar_per_1pct", 0.0) / 1e6))
        gex_regime_desc = gex_data.get("regime", "")
        call_wall = gex_data.get("call_wall", gex_data.get("call_wall_strike"))
        put_wall = gex_data.get("put_wall", gex_data.get("put_wall_strike"))
        gamma_flip = gex_data.get("gamma_flip_price")
        max_pain = gex_data.get("max_pain", gex_data.get("max_pain_strike"))

        gex_regime_state = 0
        gex_vol_mult = 1.0

        if "+GEX" in gex_regime_desc or net_gex_m > 0:
            gex_regime_state = 1
            gex_vol_mult = GEX_POS_VOL_DAMPENER
        elif "-GEX" in gex_regime_desc or net_gex_m < 0:
            gex_regime_state = -1
            gex_vol_mult = GEX_NEG_VOL_ACCELERATOR
        else:
            gex_regime_state = gex_data.get("regime_state", 0)
            if gex_regime_state > 0:
                gex_vol_mult = GEX_POS_VOL_DAMPENER
            elif gex_regime_state < 0:
                gex_vol_mult = GEX_NEG_VOL_ACCELERATOR

        return GEXParams(
            net_gex_millions=net_gex_m,
            regime_state=gex_regime_state,
            regime_desc=gex_regime_desc,
            call_wall=float(call_wall) if call_wall is not None and not pd.isna(call_wall) else None,
            put_wall=float(put_wall) if put_wall is not None and not pd.isna(put_wall) else None,
            gamma_flip=float(gamma_flip) if gamma_flip is not None and not pd.isna(gamma_flip) else None,
            max_pain=float(max_pain) if max_pain is not None and not pd.isna(max_pain) else None,
            vol_multiplier=gex_vol_mult,
        )


class EventParameterExtractor:
    """Extracts corporate catalyst calendar dates, de-grossing haircuts, and PEAD metrics."""

    @staticmethod
    def extract(events: Optional[Dict[str, Any]]) -> PEADParams:
        if not events or not isinstance(events, dict):
            return PEADParams()

        cat_info = events.get("catalyst_status") or events.get("catalyst") or {}
        next_earnings_date = cat_info.get("next_earnings_date") or events.get("next_earnings_date") or cat_info.get("next_event_date")
        earnings_days_away = cat_info.get("days_to_earnings") if cat_info.get("days_to_earnings") is not None else (
            cat_info.get("earnings_days_away") if cat_info.get("earnings_days_away") is not None else events.get("earnings_days_away")
        )
        earnings_proximity = cat_info.get("status_code") or cat_info.get("earnings_proximity") or events.get("earnings_proximity", "SAFE")

        degross_info = events.get("degrossing", {})
        if "position_haircut" in degross_info:
            event_degross_mult = float(degross_info["position_haircut"])
        else:
            event_degross_mult = float(events.get("degross_multiplier", 1.0))

        pead_info = events.get("pead", {})
        pead_regime = pead_info.get("drift_regime", "")
        sue_score = float(pead_info.get("sue_score", 0.0))
        pead_gap_pct = float(pead_info.get("announcement_gap_pct", 0.0))
        pead_drift_pct = float(pead_info.get("post_earnings_drift_pct", 0.0))
        pead_drift_score = float(pead_info.get("pead_drift_score", 0.0))

        return PEADParams(
            next_earnings_date=next_earnings_date,
            earnings_days_away=earnings_days_away,
            earnings_proximity=earnings_proximity,
            catalyst_status=earnings_proximity,
            event_degross_multiplier=event_degross_mult,
            pead_regime=pead_regime,
            sue_score=sue_score,
            pead_gap_pct=pead_gap_pct,
            pead_drift_pct=pead_drift_pct,
            pead_drift_score=pead_drift_score,
        )


class SupportResistanceSynthesizer:
    """Synthesizes dynamic multi-layer support and resistance boundaries."""

    @staticmethod
    def synthesize(
        current_price: float,
        df: pd.DataFrame,
        sma50: float,
        bb_upper: float,
        bb_lower: float,
        microstructure: Optional[Dict[str, Any]],
        gex_params: GEXParams,
    ) -> Tuple[float, float]:
        recent_low_60d = float(df["close"].tail(60).min())
        key_support = max(recent_low_60d, bb_lower, min(sma50, current_price * 0.96))
        resistance = max(bb_upper, float(df["close"].tail(60).max()), current_price * 1.05)

        if microstructure and isinstance(microstructure, dict):
            avwap_ytd = microstructure.get("avwap", {}).get("ytd", {})
            ytd_lower = avwap_ytd.get("lower_1s")
            ytd_upper = avwap_ytd.get("upper_1s")

            if ytd_lower and not pd.isna(ytd_lower):
                key_support = max(key_support, float(ytd_lower))
            if ytd_upper and not pd.isna(ytd_upper):
                resistance = max(resistance, float(ytd_upper))

            vp = microstructure.get("volume_profile", {})
            val = vp.get("val")
            vah = vp.get("vah")
            if val and not pd.isna(val) and val < current_price:
                key_support = max(key_support, float(val))
            if vah and not pd.isna(vah) and vah > current_price:
                resistance = max(resistance, float(vah))

        # GEX boundaries
        if gex_params.put_wall and gex_params.put_wall < current_price:
            key_support = max(key_support, gex_params.put_wall)
        if gex_params.max_pain and gex_params.max_pain < current_price * 0.98:
            key_support = max(key_support, gex_params.max_pain)
        if gex_params.call_wall and gex_params.call_wall > current_price:
            resistance = max(resistance, gex_params.call_wall)

        return key_support, resistance


class MonteCarloSimulator:
    """Simulates forward price trajectories using Geometric Brownian Motion with jump shocks."""

    @staticmethod
    def simulate(
        current_price: float,
        forecast_days: int,
        simulations: int,
        daily_vol: float,
        drift: float,
        sma50: float,
        regime_params: RegimeParams,
        gex_params: GEXParams,
        pead_params: PEADParams,
        future_dates: List[str],
        seed: Optional[int] = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed=seed)
        daily_vol_clamped = max(VOL_MIN_CLAMP, min(VOL_MAX_CLAMP, daily_vol * gex_params.vol_multiplier))
        adj_drift = max(-0.0015, min(0.0015, drift * regime_params.risk_multiplier))

        price_paths = np.zeros((simulations, forecast_days))
        price_paths[:, 0] = current_price

        t_earn = future_dates.index(pead_params.next_earnings_date) if (pead_params.next_earnings_date and pead_params.next_earnings_date in future_dates) else -1

        for t in range(1, forecast_days):
            z = rng.standard_normal(simulations)
            reversion = DRIFT_MEAN_REVERSION_COEFF * (sma50 - price_paths[:, t - 1]) / price_paths[:, t - 1]
            step_return = adj_drift + reversion + daily_vol_clamped * z

            # Regime changepoint hazard jump
            if regime_params.daily_hazard > 0:
                jump_occurred = rng.random(simulations) < regime_params.daily_hazard
                if np.any(jump_occurred):
                    jump_direction = -0.5 * daily_vol_clamped if regime_params.state == 2 else 0.0
                    jump_shocks = rng.laplace(loc=jump_direction, scale=BOCD_JUMP_SCALE_MULT * daily_vol_clamped, size=simulations)
                    step_return += jump_occurred * jump_shocks

            # Corporate earnings announcement gap jump
            if t == t_earn:
                earn_direction = 0.003 * (pead_params.sue_score if pead_params.sue_score != 0 else 1.0)
                earn_gap_shocks = rng.normal(loc=earn_direction, scale=EARNINGS_GAP_SCALE_MULT * daily_vol_clamped, size=simulations)
                step_return += earn_gap_shocks

            price_paths[:, t] = price_paths[:, t - 1] * (1.0 + step_return)

        p10_bear = np.percentile(price_paths, 10, axis=0)
        p50_median = np.percentile(price_paths, 50, axis=0)
        p90_bull = np.percentile(price_paths, 90, axis=0)

        return p10_bear, p50_median, p90_bull


class RecommendationEngine:
    """Evaluates the 7-branch regime decision tree and generates actionable narratives."""

    @staticmethod
    def evaluate(
        current_price: float,
        current_rsi: float,
        pct_b: float,
        sma20: float,
        sma50: float,
        sma200: float,
        key_support: float,
        regime_params: RegimeParams,
        gex_params: GEXParams,
        pead_params: PEADParams,
        future_dates: List[str],
        min_median_idx: int,
    ) -> Tuple[str, str, float, float, str, str]:
        forecast_days = len(future_dates)
        regime_state = regime_params.state

        if regime_state == 2:
            recommendation = "RISK-OFF / CAPITAL PRESERVATION"
            action_summary = (
                f"Active High-Volatility Liquidation Regime (BOCD State 2, {regime_params.changepoint_hazard_pct:.1f}% hazard). "
                f"Capital preservation is paramount. Delay large allocations until volatility structure normalizes."
            )
            entry_low = round(min(key_support * 0.94, current_price * 0.90), 2)
            entry_high = round(min(key_support * 0.98, current_price * 0.94), 2)
            opt_window_start = future_dates[min(15, forecast_days - 1)]
            opt_window_end = future_dates[min(35, forecast_days - 1)]
        elif regime_state == 3 or (regime_params.changepoint_hazard_pct >= 35.0):
            recommendation = "REGIME SHIFT ALERT / PAUSE ENTRIES"
            action_summary = (
                f"Bayesian changepoint alert active ({regime_params.changepoint_hazard_pct:.1f}% instant hazard). "
                f"Market structure is in an inflection phase; pause new entries until run-length stabilizes."
            )
            entry_low = round(key_support * 0.95, 2)
            entry_high = round(current_price * 0.97, 2)
            opt_window_start = future_dates[min(10, forecast_days - 1)]
            opt_window_end = future_dates[min(25, forecast_days - 1)]
        elif regime_state == 0:
            if current_price > sma20 and current_rsi > 65:
                recommendation = "BULLISH MOMENTUM / DIP ACCUMULATION"
                action_summary = (
                    f"Sustained bullish markup regime (BOCD State 0, run-length {regime_params.expected_run_length_days:.0f}d). "
                    f"Trend momentum is strong; accumulate on minor intraday pullbacks."
                )
                entry_low = round(max(key_support, current_price * 0.96), 2)
                entry_high = round(current_price * 0.985, 2)
                opt_window_start = future_dates[min(3, forecast_days - 1)]
                opt_window_end = future_dates[min(15, forecast_days - 1)]
            elif current_rsi > 70 or pct_b > 0.85:
                recommendation = "BUY ON PULLBACK"
                action_summary = (
                    f"Bull trend active (BOCD State 0), but short-term overbought (RSI {current_rsi:.1f}). "
                    f"Wait for shallow pullback toward support before adding exposure."
                )
                entry_low = round(max(key_support, current_price * 0.96), 2)
                entry_high = round(current_price * 0.985, 2)
                opt_window_start = future_dates[min(3, forecast_days - 1)]
                opt_window_end = future_dates[min(15, forecast_days - 1)]
            else:
                recommendation = "STRONG BUY / TREND ACCUMULATION"
                action_summary = "Low-volatility expansion with supportive macro liquidity (BOCD State 0). Accumulate with high confidence."
                entry_low = round(current_price * 0.98, 2)
                entry_high = round(current_price * 1.01, 2)
                opt_window_start = future_dates[0]
                opt_window_end = future_dates[min(12, forecast_days - 1)]
        elif regime_state == 1:
            recommendation = "RANGE ACCUMULATION / BUY SUPPORT"
            action_summary = "Range-bound consolidation regime (BOCD State 1). Accumulate near support and trim near resistance."
            entry_low = round(key_support * 0.98, 2)
            entry_high = round(key_support * 1.02, 2)
            opt_window_start = future_dates[5]
            opt_window_end = future_dates[min(20, forecast_days - 1)]
        else:
            # Fallback technical rules
            if current_rsi < 35 or pct_b < 0.15:
                recommendation = "STRONG BUY"
                action_summary = "Stock is currently oversold near major technical support. Immediate entry recommended."
                entry_low = current_price * 0.985
                entry_high = current_price * 1.01
                opt_window_start = future_dates[0]
                opt_window_end = future_dates[min(10, forecast_days - 1)]
            elif current_rsi > 70 or pct_b > 0.85:
                recommendation = "BUY ON PULLBACK"
                action_summary = (
                    f"Stock is currently in short-term overbought territory (RSI {current_rsi:.1f}). "
                    f"Wait for a pullback toward the projected support zone before deploying capital."
                )
                entry_low = round(min(key_support, current_price * 0.94), 2)
                entry_high = round(current_price * 0.975, 2)
                dip_center = max(5, min_median_idx)
                opt_window_start = future_dates[max(0, dip_center - 5)]
                opt_window_end = future_dates[min(forecast_days - 1, dip_center + 7)]
            elif current_price > sma50 and sma50 > sma200:
                recommendation = "ACCUMULATE / DIP BUY"
                action_summary = "Healthy uptrend in place above major moving averages. Accumulate on any shallow dip."
                entry_low = round(max(key_support, current_price * 0.96), 2)
                entry_high = round(current_price * 0.995, 2)
                opt_window_start = future_dates[2]
                opt_window_end = future_dates[min(20, forecast_days - 1)]
            else:
                recommendation = "HOLD / CAUTIOUS BUY"
                action_summary = "Consolidation phase. Accumulate cautiously near tested support levels."
                entry_low = round(key_support * 0.98, 2)
                entry_high = round(key_support * 1.02, 2)
                opt_window_start = future_dates[5]
                opt_window_end = future_dates[min(25, forecast_days - 1)]

        # GEX tactical commentary
        if gex_params.regime_state < 0 and gex_params.gamma_flip is not None:
            action_summary += (
                f" [GEX Alert: -GEX regime active ({gex_params.net_gex_millions:+.1f}M/1%). Dealer dynamic hedging accelerates drops below "
                f"Gamma Flip ${gex_params.gamma_flip:.2f}; enforce strict stop-loss rules.]"
            )
        elif gex_params.regime_state > 0 and gex_params.put_wall is not None and gex_params.call_wall is not None:
            action_summary += (
                f" [GEX Note: +GEX regime active (+${gex_params.net_gex_millions:.1f}M/1%). Dealer counter-trading pins price between Put Wall "
                f"${gex_params.put_wall:.2f} and Call Wall ${gex_params.call_wall:.2f}.]"
            )

        # Corporate catalyst / PEAD checks
        if pead_params.earnings_proximity == "CRITICAL_EVENT" or (pead_params.earnings_days_away is not None and pead_params.earnings_days_away <= 1):
            recommendation = "EVENT RISK / PRE-EARNINGS DE-GROSSING"
            action_summary = (
                f"Binary earnings catalyst on {pead_params.next_earnings_date} ({pead_params.earnings_days_away or 1} day away). "
                f"Freeze new entries to avoid overnight gap liquidations; enforce 100% pre-event position de-grossing."
            )
        elif pead_params.earnings_proximity == "IMMINENT_DEGROSS" or (pead_params.earnings_days_away is not None and pead_params.earnings_days_away <= 4):
            if recommendation != "RISK-OFF / CAPITAL PRESERVATION":
                recommendation = "IMMINENT CATALYST / 50% DE-GROSSING"
                action_summary = (
                    f"Corporate earnings scheduled on {pead_params.next_earnings_date} ({pead_params.earnings_days_away} days away). "
                    f"Limit long exposure to 50% capital sizing haircut until event uncertainty passes."
                )
        elif "bullish" in pead_params.pead_regime.lower() and recommendation not in ["RISK-OFF / CAPITAL PRESERVATION", "REGIME SHIFT ALERT / PAUSE ENTRIES"]:
            recommendation = "PEAD POST-EARNINGS DRIFT ACCUMULATION"
            action_summary += (
                f" [PEAD Alert: Active post-earnings bullish drift (SUE {pead_params.sue_score:+.2f}). "
                f"Institutional underreaction provides positive drift momentum.]"
            )

        return recommendation, action_summary, float(entry_low), float(entry_high), opt_window_start, opt_window_end


def predict_future_buy_timing(
    df: pd.DataFrame,
    forecast_days: int = 63,
    simulations: int = 1000,
    regime: Optional[Dict[str, Any]] = None,
    microstructure: Optional[Dict[str, Any]] = None,
    derivatives: Optional[Dict[str, Any]] = None,
    events: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Perform quantitative and machine-learning predictive analysis on forward buy timing,
    conditioned on BOCD regimes, microstructure (AVWAP), Dealer GEX, and PEAD catalysts.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    if len(df) < 50:
        raise ValueError("Insufficient data points for 3-month predictive forecasting (minimum 50 required).")

    latest_date_str = df["date"].iloc[-1]
    latest_dt = pd.to_datetime(latest_date_str)
    current_price = float(df["close"].iloc[-1])

    # 1. Technical Indicators
    sma20 = float(df["close"].rolling(20).mean().iloc[-1])
    sma50 = float(df["close"].rolling(50).mean().iloc[-1])
    sma200 = float(df["close"].rolling(200, min_periods=30).mean().iloc[-1])

    recent_returns = df["close"].pct_change().dropna()
    daily_vol = float(recent_returns.tail(60).std())
    drift = float(recent_returns.tail(60).mean())

    rolling_std20 = float(df["close"].rolling(20).std().iloc[-1])
    bb_upper = sma20 + 2 * rolling_std20
    bb_lower = sma20 - 2 * rolling_std20
    pct_b = (current_price - bb_lower) / (bb_upper - bb_lower + 1e-9)

    current_rsi = float(compute_rsi(df["close"], period=14).iloc[-1])

    # 2. Extract domain parameters
    regime_p = RegimeParameterExtractor.extract(regime, forecast_days)
    if regime_p.vol_21d_pct is not None and regime_p.vol_21d_pct > 0:
        daily_vol = (float(regime_p.vol_21d_pct) / 100.0) / math.sqrt(252.0)
    if regime and isinstance(regime, dict) and float(regime.get("vol_ratio", 1.0)) > 1.15:
        daily_vol *= 1.15

    gex_p = GEXParameterExtractor.extract(derivatives)
    pead_p = EventParameterExtractor.extract(events)

    # 3. Future trading calendar
    future_dates = []
    curr = latest_dt
    while len(future_dates) < forecast_days:
        curr += pd.Timedelta(days=1)
        if curr.weekday() < 5:
            future_dates.append(curr.strftime("%Y-%m-%d"))

    # 4. Monte Carlo Simulation
    p10_bear, p50_median, p90_bull = MonteCarloSimulator.simulate(
        current_price=current_price,
        forecast_days=forecast_days,
        simulations=simulations,
        daily_vol=daily_vol,
        drift=drift,
        sma50=sma50,
        regime_params=regime_p,
        gex_params=gex_p,
        pead_params=pead_p,
        future_dates=future_dates,
        seed=42,
    )

    # 5. Dynamic Support & Resistance
    key_support, resistance = SupportResistanceSynthesizer.synthesize(
        current_price=current_price,
        df=df,
        sma50=sma50,
        bb_upper=bb_upper,
        bb_lower=bb_lower,
        microstructure=microstructure,
        gex_params=gex_p,
    )

    min_median_idx = int(np.argmin(p50_median[:40]))
    rec, action_summary, entry_low, entry_high, opt_start, opt_end = RecommendationEngine.evaluate(
        current_price=current_price,
        current_rsi=current_rsi,
        pct_b=pct_b,
        sma20=sma20,
        sma50=sma50,
        sma200=sma200,
        key_support=key_support,
        regime_params=regime_p,
        gex_params=gex_p,
        pead_params=pead_p,
        future_dates=future_dates,
        min_median_idx=min_median_idx,
    )

    # 6. Event-Risk Buy Window Shifting
    if events and isinstance(events, dict) and pead_p.next_earnings_date and RiskDegrossingEngine is not None:
        opt_start, opt_end, was_delayed = RiskDegrossingEngine.adjust_buy_window(
            opt_start,
            opt_end,
            pead_p.next_earnings_date,
            latest_date_str,
            min_buffer_days=2,
        )
        if was_delayed:
            action_summary += f" [Event Timing Adjustment: Buy window delayed to {opt_start} following {pead_p.next_earnings_date} earnings release.]"

    target_price_3m = round(float(p50_median[-1]), 2)
    expected_gain_pct = round(((target_price_3m - current_price) / current_price) * 100.0, 2)
    stop_loss = round(float(min(key_support * 0.96, entry_low * 0.96)), 2)
    downside_risk = abs((stop_loss - current_price) / current_price)
    upside_reward = max(0.01, (target_price_3m - current_price) / current_price)
    risk_reward = round(upside_reward / (downside_risk + 1e-6), 2)

    forecast_points = [
        ForecastSeriesPoint(
            date=f_date,
            bear_p10=round(float(p10_bear[i]), 2),
            median_p50=round(float(p50_median[i]), 2),
            bull_p90=round(float(p90_bull[i]), 2),
        )
        for i, f_date in enumerate(future_dates)
    ]

    is_capital_preservation = (
        rec in [
            "RISK-OFF / CAPITAL PRESERVATION",
            "REGIME SHIFT ALERT / PAUSE ENTRIES",
            "EVENT RISK / PRE-EARNINGS DE-GROSSING",
        ]
        or "DO NOT BUY" in rec
        or "PRESERVATION" in rec
        or regime_p.state == 2
    )
    is_entry_allowed = not is_capital_preservation
    execution_posture = "ACTIONABLE_BUY" if is_entry_allowed else "ENTRIES_INHIBITED"

    buy_window_dto = BuyWindow(
        start_date=opt_start,
        end_date=opt_end,
        is_active=is_entry_allowed,
        status="ACTIVE" if is_entry_allowed else "SUSPENDED",
        description=f"Between {opt_start} and {opt_end}" if is_entry_allowed else f"Entries suspended due to {rec} regime",
        modeled_window_dates=[opt_start, opt_end],
    )

    result_dto = PredictiveForecastResult(
        current_price=current_price,
        current_date=latest_date_str,
        current_rsi=round(current_rsi, 1),
        sma50=round(sma50, 2),
        sma200=round(sma200, 2),
        key_support=round(key_support, 2),
        key_resistance=round(resistance, 2),
        recommendation=rec,
        action_summary=action_summary,
        is_entry_allowed=is_entry_allowed,
        is_capital_preservation=is_capital_preservation,
        execution_posture=execution_posture,
        entry_corridor_display=f"${entry_low:.2f} - ${entry_high:.2f}" if is_entry_allowed else "ENTRIES INHIBITED",
        optimal_entry_range=(round(entry_low, 2), round(entry_high, 2)),
        optimal_buy_window=buy_window_dto,
        target_price_3m=target_price_3m,
        expected_return_pct=expected_gain_pct,
        stop_loss=stop_loss,
        risk_reward_ratio=risk_reward,
        forecast_days=forecast_days,
        forecast_series=forecast_points,
        bocd_regime_state=regime_p.state,
        bocd_regime_name=regime_p.name,
        bocd_changepoint_hazard_pct=regime_p.changepoint_hazard_pct,
        bocd_forward_changepoint_prob_pct=round(regime_p.forward_changepoint_prob_pct, 1) if regime_p.forward_changepoint_prob_pct is not None else None,
        bocd_expected_run_length_days=round(regime_p.expected_run_length_days, 1) if regime else None,
        dealer_gex_regime=gex_p.regime_desc,
        gex_regime=gex_p.regime_desc,
        dealer_net_gex_m=gex_p.net_gex_millions,
        call_gamma_wall=round(gex_p.call_wall, 2) if gex_p.call_wall is not None else None,
        call_wall_price=round(gex_p.call_wall, 2) if gex_p.call_wall is not None else None,
        put_gamma_wall=round(gex_p.put_wall, 2) if gex_p.put_wall is not None else None,
        put_wall_price=round(gex_p.put_wall, 2) if gex_p.put_wall is not None else None,
        gamma_flip_price=round(gex_p.gamma_flip, 2) if gex_p.gamma_flip is not None else None,
        max_pain_price=round(gex_p.max_pain, 2) if gex_p.max_pain is not None else None,
        gex_vol_multiplier=gex_p.vol_multiplier,
        next_earnings_date=pead_p.next_earnings_date,
        earnings_days_away=pead_p.earnings_days_away,
        earnings_proximity=pead_p.earnings_proximity,
        catalyst_status=pead_p.catalyst_status or "SAFE",
        event_degross_multiplier=pead_p.event_degross_multiplier,
        event_haircut=pead_p.event_degross_multiplier,
        action_recommendation=rec,
        optimal_buy_window_start=opt_start,
        optimal_buy_window_end=opt_end,
        pead_regime=pead_p.pead_regime,
        pead_drift_regime=pead_p.pead_regime,
        pead_sue_score=pead_p.sue_score,
        pead_announcement_gap_pct=pead_p.pead_gap_pct,
        pead_post_earnings_drift_pct=pead_p.pead_drift_pct,
        pead_drift_score=pead_p.pead_drift_score,
    )

    return result_dto.to_dict()
