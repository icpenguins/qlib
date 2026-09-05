#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Typed Domain Models & Data Transfer Objects (DTOs)
==================================================
Defines strongly typed, immutable dataclasses for:
- Market Regime & BOCD Parameters
- Dealer Gamma Exposure (GEX) Parameters
- Corporate Catalyst & PEAD Event Parameters
- Buy Execution Windows & Corridors
- Forecast Series Trajectories
- Predictive Forecast Results
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple


@dataclass(frozen=True)
class RegimeParams:
    """Parameters parsed from Bayesian Online Changepoint Detection (BOCD)."""
    state: Optional[int] = None
    name: Optional[str] = None
    changepoint_hazard_pct: float = 0.0
    forward_changepoint_prob_pct: Optional[float] = None
    expected_run_length_days: float = 63.0
    risk_multiplier: float = 1.0
    daily_hazard: float = 0.0
    vol_21d_pct: Optional[float] = None


@dataclass(frozen=True)
class GEXParams:
    """Parameters parsed from Dealer Gamma Exposure (GEX) analytics."""
    net_gex_millions: float = 0.0
    regime_state: int = 0
    regime_desc: Optional[str] = None
    call_wall: Optional[float] = None
    put_wall: Optional[float] = None
    gamma_flip: Optional[float] = None
    max_pain: Optional[float] = None
    vol_multiplier: float = 1.0


@dataclass(frozen=True)
class PEADParams:
    """Parameters parsed from Corporate Catalyst & Post-Earnings Drift models."""
    next_earnings_date: Optional[str] = None
    earnings_days_away: Optional[int] = None
    earnings_proximity: str = "SAFE"
    catalyst_status: str = "SAFE"
    event_degross_multiplier: float = 1.0
    pead_regime: str = ""
    sue_score: float = 0.0
    pead_gap_pct: float = 0.0
    pead_drift_pct: float = 0.0
    pead_drift_score: float = 0.0


@dataclass(frozen=True)
class BuyWindow:
    """Optimal execution window for capital allocation."""
    start_date: str
    end_date: str
    is_active: bool
    status: str
    description: str
    modeled_window_dates: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "is_active": self.is_active,
            "status": self.status,
            "description": self.description,
            "modeled_window_dates": list(self.modeled_window_dates),
        }


@dataclass(frozen=True)
class ForecastSeriesPoint:
    """A single forward trading day trajectory point."""
    date: str
    bear_p10: float
    median_p50: float
    bull_p90: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "bear_p10": self.bear_p10,
            "median_p50": self.median_p50,
            "bull_p90": self.bull_p90,
        }


@dataclass(frozen=True)
class PredictiveForecastResult:
    """Canonical predictive forecasting and buy timing analysis result."""
    current_price: float
    current_date: str
    current_rsi: float
    sma50: float
    sma200: float
    key_support: float
    key_resistance: float
    recommendation: str
    action_summary: str
    is_entry_allowed: bool
    is_capital_preservation: bool
    execution_posture: str
    entry_corridor_display: str
    optimal_entry_range: Tuple[float, float]
    optimal_buy_window: BuyWindow
    target_price_3m: float
    expected_return_pct: float
    stop_loss: float
    risk_reward_ratio: float
    forecast_days: int
    forecast_series: List[ForecastSeriesPoint]
    bocd_regime_state: Optional[int]
    bocd_regime_name: Optional[str]
    bocd_changepoint_hazard_pct: Optional[float]
    bocd_forward_changepoint_prob_pct: Optional[float]
    bocd_expected_run_length_days: Optional[float]
    dealer_gex_regime: Optional[str]
    gex_regime: Optional[str]
    dealer_net_gex_m: float
    call_gamma_wall: Optional[float]
    call_wall_price: Optional[float]
    put_gamma_wall: Optional[float]
    put_wall_price: Optional[float]
    gamma_flip_price: Optional[float]
    max_pain_price: Optional[float]
    gex_vol_multiplier: float
    next_earnings_date: Optional[str]
    earnings_days_away: Optional[int]
    earnings_proximity: str
    catalyst_status: str
    event_degross_multiplier: float
    event_haircut: float
    action_recommendation: str
    optimal_buy_window_start: str
    optimal_buy_window_end: str
    pead_regime: str
    pead_drift_regime: str
    pead_sue_score: float
    pead_announcement_gap_pct: float
    pead_post_earnings_drift_pct: float
    pead_drift_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to canonical dictionary format for 100% JSON contract backward compatibility."""
        return {
            "current_price": self.current_price,
            "current_date": self.current_date,
            "current_rsi": self.current_rsi,
            "sma50": self.sma50,
            "sma200": self.sma200,
            "key_support": self.key_support,
            "key_resistance": self.key_resistance,
            "recommendation": self.recommendation,
            "action_summary": self.action_summary,
            "is_entry_allowed": self.is_entry_allowed,
            "is_capital_preservation": self.is_capital_preservation,
            "execution_posture": self.execution_posture,
            "entry_corridor_display": self.entry_corridor_display,
            "optimal_entry_range": list(self.optimal_entry_range),
            "optimal_buy_window": self.optimal_buy_window.to_dict(),
            "target_price_3m": self.target_price_3m,
            "expected_return_pct": self.expected_return_pct,
            "stop_loss": self.stop_loss,
            "risk_reward_ratio": self.risk_reward_ratio,
            "forecast_days": self.forecast_days,
            "forecast_series": [p.to_dict() for p in self.forecast_series],
            "bocd_regime_state": self.bocd_regime_state,
            "bocd_regime_name": self.bocd_regime_name,
            "bocd_changepoint_hazard_pct": self.bocd_changepoint_hazard_pct,
            "bocd_forward_changepoint_prob_pct": self.bocd_forward_changepoint_prob_pct,
            "bocd_expected_run_length_days": self.bocd_expected_run_length_days,
            "dealer_gex_regime": self.dealer_gex_regime,
            "gex_regime": self.gex_regime,
            "dealer_net_gex_m": self.dealer_net_gex_m,
            "call_gamma_wall": self.call_gamma_wall,
            "call_wall_price": self.call_wall_price,
            "put_gamma_wall": self.put_gamma_wall,
            "put_wall_price": self.put_wall_price,
            "gamma_flip_price": self.gamma_flip_price,
            "max_pain_price": self.max_pain_price,
            "gex_vol_multiplier": self.gex_vol_multiplier,
            "next_earnings_date": self.next_earnings_date,
            "earnings_days_away": self.earnings_days_away,
            "earnings_proximity": self.earnings_proximity,
            "catalyst_status": self.catalyst_status,
            "event_degross_multiplier": self.event_degross_multiplier,
            "event_haircut": self.event_haircut,
            "action_recommendation": self.action_recommendation,
            "optimal_buy_window_start": self.optimal_buy_window_start,
            "optimal_buy_window_end": self.optimal_buy_window_end,
            "pead_regime": self.pead_regime,
            "pead_drift_regime": self.pead_drift_regime,
            "pead_sue_score": self.pead_sue_score,
            "pead_announcement_gap_pct": self.pead_announcement_gap_pct,
            "pead_post_earnings_drift_pct": self.pead_post_earnings_drift_pct,
            "pead_drift_score": self.pead_drift_score,
        }
