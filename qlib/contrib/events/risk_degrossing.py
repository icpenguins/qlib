# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Event Risk De-Grossing & Volatility Shock Engine
================================================
Calculates institutional position de-grossing factors to protect trading capital
against unhedgeable overnight binary gap risk (quarterly earnings announcements
and emergency FOMC rate decisions).

Key Capabilities:
1. Dynamic De-Grossing Multiplier (w_event in [0.0, 1.0]):
       Enforces position reduction within 48 hours of binary events.

2. Predictive Buy Window Adjustment:
       Automatically shifts optimal entry windows past imminent catalyst dates
       so the quantitative strategy buys into confirmed post-announcement drift
       rather than gambling on the binary coin flip.

3. Binary Gap Shock Parameterization:
       Calibrates discrete jump variance for forward Monte Carlo path simulation.
"""

import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class RiskDegrossingEngine:
    """
    Manages pre-event risk reduction and buy timing rescheduling.
    """

    @staticmethod
    def calculate_degross_multiplier(
        event_days_away: Optional[int],
        event_type: str = "earnings",
        regime_state: int = 0,
    ) -> float:
        """
        Calculate recommended risk allocation factor w_event in [0.0, 1.0].
        """
        if event_days_away is None:
            return 1.0

        d = int(event_days_away)

        if event_type.lower() == "earnings":
            if d <= 1:
                # Event within 24-48h: Complete entry freeze / 100% de-grossing
                w = 0.0
            elif d <= 3:
                # 2-3 days away: 50% de-grossing
                w = 0.50
            elif d <= 6:
                # 4-6 days away: 25% de-grossing
                w = 0.75
            else:
                w = 1.0
        else:
            # Macro (FOMC / CPI)
            if d <= 1:
                w = 0.25
            elif d <= 3:
                w = 0.60
            elif d <= 5:
                w = 0.80
            else:
                w = 1.0

        # In State 2 (Risk-Off Liquidation), scale down further
        if regime_state == 2:
            w *= 0.5

        return round(float(w), 2)

    @staticmethod
    def adjust_buy_window(
        window_start_date: str,
        window_end_date: str,
        event_date: Optional[str],
        current_date: str,
        min_buffer_days: int = 2,
    ) -> Tuple[str, str, bool]:
        """
        If a binary event falls right before or during the optimal buy window,
        delay window_start_date until min_buffer_days AFTER the event.

        Returns: (adjusted_start, adjusted_end, was_delayed)
        """
        if not event_date:
            return window_start_date, window_end_date, False

        curr_dt = pd.to_datetime(current_date)
        ev_dt = pd.to_datetime(event_date)
        w_start_dt = pd.to_datetime(window_start_date)
        w_end_dt = pd.to_datetime(window_end_date)

        # If event is already in the past, no adjustment needed
        if ev_dt < curr_dt:
            return window_start_date, window_end_date, False

        # If event occurs within 3 days before window_start, or inside [window_start, window_end]
        if (ev_dt >= w_start_dt - pd.Timedelta(days=3)) and (ev_dt <= w_end_dt):
            # Shift start to event date + buffer business days
            new_start_dt = ev_dt
            added_bdays = 0
            while added_bdays < min_buffer_days:
                new_start_dt += pd.Timedelta(days=1)
                if new_start_dt.weekday() < 5:
                    added_bdays += 1

            # Ensure end date is at least 10 business days after new start
            new_end_dt = max(w_end_dt, new_start_dt + pd.Timedelta(days=14))

            return (
                new_start_dt.strftime("%Y-%m-%d"),
                new_end_dt.strftime("%Y-%m-%d"),
                True,
            )

        return window_start_date, window_end_date, False

    @staticmethod
    def estimate_binary_gap_volatility(
        historical_gaps_pct: Optional[List[float]] = None,
        daily_vol: float = 0.02,
    ) -> float:
        """
        Estimate discrete jump volatility scale for the binary event.
        """
        if historical_gaps_pct and len(historical_gaps_pct) > 0:
            rms_gap = float(np.sqrt(np.mean(np.square(historical_gaps_pct))) / 100.0)
            return max(daily_vol * 1.5, rms_gap)
        return daily_vol * 2.5

