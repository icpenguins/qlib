# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Event Risk, Catalyst Awareness & PEAD Models
============================================
Provides institutional corporate catalyst awareness, pre-event risk de-grossing,
Post-Earnings Announcement Drift (PEAD) alpha modeling, and key momentum event
extraction for charting.
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd

from .event_calendar import (
    EventCalendarEngine,
    EventProximity,
    FOMC_SCHEDULE_2024_2027,
    CPI_SCHEDULE_2024_2027,
)
from .pead import PEADEngine
from .risk_degrossing import RiskDegrossingEngine
from .events_data import EventsDataLoader, SyntheticEventScheduleGenerator
from .empirical_sue import calculate_empirical_sue
from .earnings_event_clock import EarningsEventClock, InvalidEventExecutionError, resolve_earnings_event_execution


def compute_event_risk_features(
    df: pd.DataFrame,
    symbol: str = "STOCK",
    data_dir: Optional[Any] = None,
    current_date: Optional[Union[str, pd.Timestamp]] = None,
    bocd_changepoints: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Comprehensive orchestrator computing corporate catalyst proximity,
    risk de-grossing multipliers, PEAD drift dynamics, and key momentum
    events for main chart plotting.
    """
    if current_date is None:
        if not df.empty and "date" in df.columns:
            curr_str = str(df["date"].iloc[-1])[:10]
        else:
            curr_str = pd.Timestamp.now().strftime("%Y-%m-%d")
    else:
        curr_str = str(current_date)[:10]

    # 1. Load or generate events data
    loader = EventsDataLoader(data_dir=data_dir)
    events_payload = loader.load_or_generate_events(
        symbol=symbol,
        current_date=curr_str,
    )

    earnings_dates = events_payload.get("all_earnings_dates", [])
    earnings_history = events_payload.get("earnings_history", [])

    # 2. Evaluate calendar catalyst proximity
    cal_engine = EventCalendarEngine()
    catalyst_status = cal_engine.evaluate_catalyst_status(
        current_date=curr_str,
        earnings_dates=earnings_dates,
        symbol=symbol,
    )

    # 3. Evaluate PEAD drift on most recent report
    pead_engine = PEADEngine()
    pead_summary = pead_engine.evaluate_recent_pead(
        df=df,
        earnings_history=earnings_history,
        current_date=curr_str,
    )

    # 4. Calculate Risk De-Grossing Multiplier
    degross_mult = RiskDegrossingEngine.calculate_degross_multiplier(
        event_days_away=catalyst_status.get("earnings_days_away"),
        event_type="earnings",
    )
    # Blend with macro degross
    composite_degross = min(degross_mult, catalyst_status.get("degross_multiplier", 1.0))

    degrossing_info = {
        "position_haircut": composite_degross,
        "is_event_imminent": catalyst_status.get("earnings_proximity") in ("IMMINENT_DEGROSS", "CRITICAL_EVENT"),
        "binary_gap_sd": 0.045 if composite_degross < 1.0 else 0.0,
        "risk_advice": "Liquidate/hedge long delta prior to announcement" if composite_degross == 0.0 else (
            "Reduce position risk by 50% ahead of catalyst" if composite_degross < 1.0 else "Normal risk budget"
        ),
    }

    # 5. Extract Key Momentum Events for Main Historical Chart
    momentum_events = PEADEngine.extract_key_momentum_events(
        df=df,
        earnings_history=earnings_history,
        bocd_changepoints=bocd_changepoints,
        fomc_dates=FOMC_SCHEDULE_2024_2027,
    )

    return {
        "symbol": symbol.upper(),
        "current_date": curr_str,
        "source": events_payload.get("source", "unknown"),
        "catalyst": catalyst_status,
        "catalyst_status": catalyst_status,
        "degrossing": degrossing_info,
        "pead": pead_summary,
        "degross_multiplier": composite_degross,
        "next_earnings_date": catalyst_status.get("next_earnings_date"),
        "earnings_days_away": catalyst_status.get("earnings_days_away"),
        "earnings_proximity": catalyst_status.get("earnings_proximity"),
        # Each row carries its own computed sue_score/announcement_gap_pct/drift_pct
        # (same methodology as `pead_summary`'s most-recent-report figures) rather
        # than raw, un-annotated earnings_history records.
        "recent_earnings_history": pead_engine.evaluate_earnings_history(
            df=df, earnings_history=earnings_history, current_date=curr_str,
        ),
        "momentum_events": momentum_events,
        "catalyst_schedule": events_payload.get("catalyst_schedule", []),
    }


__all__ = [
    "EventCalendarEngine",
    "EventProximity",
    "FOMC_SCHEDULE_2024_2027",
    "CPI_SCHEDULE_2024_2027",
    "PEADEngine",
    "RiskDegrossingEngine",
    "EventsDataLoader",
    "SyntheticEventScheduleGenerator",
    "compute_event_risk_features",
    "calculate_empirical_sue",
    "EarningsEventClock",
    "InvalidEventExecutionError",
    "resolve_earnings_event_execution",
]
