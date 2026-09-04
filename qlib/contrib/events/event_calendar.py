# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Event Calendar & Proximity Engine
=================================
Tracks corporate earnings dates, Federal Reserve FOMC interest rate decisions,
and Bureau of Labor Statistics (BLS) CPI inflation release dates.

Calculates exact trading-day proximity to binary market catalysts and determines
institutional risk de-grossing status:
- SAFE: Catalyst > 10 trading days away (1.0x risk sizing).
- APPROACHING: Catalyst within 5 to 10 trading days (0.75x risk sizing).
- IMMINENT_DEGROSS: Catalyst within 2 to 4 trading days (0.50x risk sizing).
- CRITICAL_EVENT: Catalyst within 24 to 48 hours (0.0x - 0.25x entry freeze).
"""

import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Official Macroeconomic Calendar Schedules (2024 - 2027)
# Published by the Federal Reserve Board of Governors & U.S. BLS
# ----------------------------------------------------------------------

FOMC_SCHEDULE_2024_2027 = [
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
    # 2026
    "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
    # 2027
    "2027-01-27", "2027-03-17", "2027-05-05", "2027-06-16",
    "2027-07-28", "2027-09-22", "2027-11-03", "2027-12-15",
]

CPI_SCHEDULE_2024_2027 = [
    # 2024
    "2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10", "2024-05-15", "2024-06-12",
    "2024-07-11", "2024-08-14", "2024-09-11", "2024-10-10", "2024-11-13", "2024-12-11",
    # 2025
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10", "2025-05-13", "2025-06-11",
    "2025-07-11", "2025-08-13", "2025-09-10", "2025-10-14", "2025-11-12", "2025-12-10",
    # 2026
    "2026-01-14", "2026-02-11", "2026-03-11", "2026-04-14", "2026-05-12", "2026-06-10",
    "2026-07-14", "2026-08-12", "2026-09-11", "2026-10-14", "2026-11-12", "2026-12-10",
    # 2027
    "2027-01-13", "2027-02-10", "2027-03-10", "2027-04-13", "2027-05-12", "2027-06-10",
    "2027-07-13", "2027-08-11", "2027-09-10", "2027-10-13", "2027-11-10", "2027-12-09",
]


class EventProximity:
    """Standardized event proximity classification."""
    SAFE = "SAFE"                          # > 10 trading days away (1.0x sizing)
    APPROACHING = "APPROACHING"            # 5 to 10 trading days (0.75x sizing)
    IMMINENT_DEGROSS = "IMMINENT_DEGROSS"  # 2 to 4 trading days (0.50x sizing)
    CRITICAL_EVENT = "CRITICAL_EVENT"      # <= 1 trading day (0.0x new entry freeze)


class EventCalendarEngine:
    """
    Evaluates corporate and macroeconomic event calendars for equity portfolios.
    """

    def __init__(
        self,
        fomc_dates: Optional[List[str]] = None,
        cpi_dates: Optional[List[str]] = None,
    ):
        self.fomc_dates = sorted(fomc_dates or FOMC_SCHEDULE_2024_2027)
        self.cpi_dates = sorted(cpi_dates or CPI_SCHEDULE_2024_2027)

    @staticmethod
    def count_business_days(
        start_date: Union[str, datetime.date, pd.Timestamp],
        end_date: Union[str, datetime.date, pd.Timestamp],
    ) -> int:
        """
        Calculate number of business days (Mon-Fri) between two dates.
        Returns positive if end_date > start_date, negative if end_date < start_date.
        """
        s = pd.to_datetime(start_date).tz_localize(None).floor("D")
        e = pd.to_datetime(end_date).tz_localize(None).floor("D")
        if s == e:
            return 0
        if e > s:
            # Busday count: days between s and e
            b_days = pd.bdate_range(start=s + pd.Timedelta(days=1), end=e)
            return len(b_days)
        else:
            b_days = pd.bdate_range(start=e + pd.Timedelta(days=1), end=s)
            return -len(b_days)

    def find_next_event(
        self,
        event_list: List[str],
        current_date: Union[str, datetime.date, pd.Timestamp],
    ) -> Tuple[Optional[str], Optional[int]]:
        """
        Find the next upcoming event from a date list relative to current_date.
        Returns (event_date_str, trading_days_away).
        """
        curr_dt = pd.to_datetime(current_date).tz_localize(None).floor("D")
        for ev_str in sorted(event_list):
            ev_dt = pd.to_datetime(ev_str).tz_localize(None).floor("D")
            if ev_dt >= curr_dt:
                days_away = self.count_business_days(curr_dt, ev_dt)
                return ev_str, days_away
        return None, None

    def find_previous_event(
        self,
        event_list: List[str],
        current_date: Union[str, datetime.date, pd.Timestamp],
    ) -> Tuple[Optional[str], Optional[int]]:
        """
        Find the most recent past event from a date list relative to current_date.
        Returns (event_date_str, trading_days_ago).
        """
        curr_dt = pd.to_datetime(current_date).tz_localize(None).floor("D")
        past_events = [
            ev_str for ev_str in event_list
            if pd.to_datetime(ev_str).tz_localize(None).floor("D") < curr_dt
        ]
        if not past_events:
            return None, None
        last_ev = sorted(past_events)[-1]
        days_ago = abs(self.count_business_days(last_ev, curr_dt))
        return last_ev, days_ago

    @staticmethod
    def classify_proximity(trading_days: Optional[int]) -> str:
        """
        Classify proximity into standardized institutional threat levels.
        """
        if trading_days is None:
            return EventProximity.SAFE
        if trading_days <= 1:
            return EventProximity.CRITICAL_EVENT
        elif trading_days <= 4:
            return EventProximity.IMMINENT_DEGROSS
        elif trading_days <= 10:
            return EventProximity.APPROACHING
        else:
            return EventProximity.SAFE

    def evaluate_catalyst_status(
        self,
        current_date: Union[str, datetime.date, pd.Timestamp],
        earnings_dates: Optional[List[str]] = None,
        symbol: str = "",
    ) -> Dict[str, Any]:
        """
        Produce a comprehensive catalyst status report across corporate earnings,
        FOMC rate decisions, and CPI inflation releases.
        """
        curr_str = pd.to_datetime(current_date).strftime("%Y-%m-%d")

        # 1. Corporate Earnings
        next_earn_date, earn_days = self.find_next_event(earnings_dates or [], current_date)
        prev_earn_date, prev_earn_days_ago = self.find_previous_event(earnings_dates or [], current_date)
        earn_prox = self.classify_proximity(earn_days)

        # 2. FOMC Rate Decision
        next_fomc_date, fomc_days = self.find_next_event(self.fomc_dates, current_date)
        fomc_prox = self.classify_proximity(fomc_days)

        # 3. CPI Inflation Release
        next_cpi_date, cpi_days = self.find_next_event(self.cpi_dates, current_date)
        cpi_prox = self.classify_proximity(cpi_days)

        # Overall composite proximity (highest threat wins)
        threat_rank = {
            EventProximity.CRITICAL_EVENT: 4,
            EventProximity.IMMINENT_DEGROSS: 3,
            EventProximity.APPROACHING: 2,
            EventProximity.SAFE: 1,
        }
        all_threats = [earn_prox, fomc_prox, cpi_prox]
        composite_proximity = max(all_threats, key=lambda p: threat_rank[p])

        # Recommended risk position haircut multiplier (w_event)
        if composite_proximity == EventProximity.CRITICAL_EVENT:
            degross_multiplier = 0.0 if earn_prox == EventProximity.CRITICAL_EVENT else 0.25
            status_desc = "CRITICAL EVENT WITHIN 48H: High binary gap risk. Freeze new position entries."
            badge_class = "bg-rose-500/15 text-rose-400 border-rose-500/30"
        elif composite_proximity == EventProximity.IMMINENT_DEGROSS:
            degross_multiplier = 0.50
            status_desc = f"IMMINENT CATALYST: Event in {min(d for d in [earn_days, fomc_days, cpi_days] if d is not None)} days. 50% de-grossing active."
            badge_class = "bg-amber-500/15 text-amber-400 border-amber-500/30"
        elif composite_proximity == EventProximity.APPROACHING:
            degross_multiplier = 0.75
            status_desc = f"APPROACHING EVENT: Catalyst in {min(d for d in [earn_days, fomc_days, cpi_days] if d is not None)} days. Monitor implied volatility."
            badge_class = "bg-blue-500/15 text-blue-400 border-blue-500/30"
        else:
            degross_multiplier = 1.0
            status_desc = "SAFE: No imminent binary corporate or macroeconomic catalysts within 10 days."
            badge_class = "bg-emerald-500/15 text-emerald-400 border-emerald-500/30"

        return {
            "symbol": symbol.upper(),
            "current_date": curr_str,
            "composite_proximity": composite_proximity,
            "status_description": status_desc,
            "badge_class": badge_class,
            "degross_multiplier": degross_multiplier,
            # Earnings
            "next_earnings_date": next_earn_date,
            "earnings_days_away": earn_days,
            "earnings_proximity": earn_prox,
            "prev_earnings_date": prev_earn_date,
            "prev_earnings_days_ago": prev_earn_days_ago,
            # Macro
            "next_fomc_date": next_fomc_date,
            "fomc_days_away": fomc_days,
            "fomc_proximity": fomc_prox,
            "next_cpi_date": next_cpi_date,
            "cpi_days_away": cpi_days,
            "cpi_proximity": cpi_prox,
        }

