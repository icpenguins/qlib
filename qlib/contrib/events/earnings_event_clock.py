# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Earnings Event Clock & Execution Sanitizer
=========================================
# my-qlib Fork Extension: Institutional Microstructure & Derivatives Layer

Enforces discrete point-in-time event timing for corporate earnings releases,
strictly separating After-Market-Close (AMC) and Before-Market-Open (BMO) schedules.
Physically prohibits unachievable fills at T_0 close to eliminate lookahead bias.
"""

from typing import Dict, Any, Optional
from datetime import datetime, time, timedelta


class InvalidEventExecutionError(ValueError):
    """Raised when an execution fill request violates market microstructure or event timing invariants."""
    pass


class EarningsEventClock:
    """
    Point-in-Time Event Clock enforcing institutional earnings execution discipline.
    """

    @staticmethod
    def resolve_event_execution(
        event_date: str,
        reporting_time: str,
        requested_fill_target: str = "T1_OPEN",
    ) -> Dict[str, Any]:
        """
        Validates event phase timing and determines compliant execution fill timestamps.

        Parameters
        ----------
        event_date : str
            Date of earnings release (YYYY-MM-DD).
        reporting_time : str
            "AMC" (After Market Close) or "BMO" (Before Market Open) or "DURING_HOURS".
        requested_fill_target : str, optional
            Requested fill price target, by default "T1_OPEN".
            Options: "T1_OPEN", "T1_VWAP30", "T0_CLOSE".

        Returns
        -------
        Dict[str, Any]
            Execution specification including signal timestamp, announcement timestamp,
            and valid execution timestamp.

        Raises
        ------
        InvalidEventExecutionError
            If requested fill violates the invariant (e.g. attempting T0_CLOSE fill on AMC).
        """
        if not event_date:
            event_date = datetime.today().strftime("%Y-%m-%d")
        try:
            dt_event = datetime.strptime(str(event_date)[:10], "%Y-%m-%d")
        except Exception:
            dt_event = datetime.today()
            event_date = dt_event.strftime("%Y-%m-%d")

        rpt = reporting_time.strip().upper() if reporting_time else "AMC"
        fill_req = requested_fill_target.strip().upper() if requested_fill_target else "T1_OPEN"

        # INVARIANT CHECK: Reject T0_CLOSE fills for AMC announcements
        if rpt == "AMC" and fill_req in ("T0_CLOSE", "T0_MOC", "CLOSE"):
            raise InvalidEventExecutionError(
                f"AMC earnings announced after market close on {event_date}. "
                "Executing fill at T0 close violates causality and introduces lookahead bias! "
                "Compliant execution must fill at T1 open or T1 morning VWAP."
            )

        if rpt == "AMC":
            # Signal formed at T0 MOC (15:55 EST)
            signal_ts = f"{event_date} 15:55:00"
            # Announcement released at T0 16:01 EST
            announcement_ts = f"{event_date} 16:01:00"
            # Execution on next trading day T1 Open (09:30 EST)
            dt_t1 = dt_event + timedelta(days=1)
            # Skip weekend if event was on Friday
            if dt_event.weekday() == 4:  # Friday
                dt_t1 = dt_event + timedelta(days=3)
            fill_date = dt_t1.strftime("%Y-%m-%d")

            if fill_req in ("T1_VWAP30", "VWAP"):
                fill_ts = f"{fill_date} 10:00:00"
                fill_type = "T1_VWAP30"
            else:
                fill_ts = f"{fill_date} 09:30:00"
                fill_type = "T1_OPEN"

        elif rpt == "BMO":
            # Signal formed at T0 MOC (15:55 EST) on prior day
            dt_t0 = dt_event - timedelta(days=1)
            if dt_event.weekday() == 0:  # Monday
                dt_t0 = dt_event - timedelta(days=3)
            signal_ts = f"{dt_t0.strftime('%Y-%m-%d')} 15:55:00"
            announcement_ts = f"{event_date} 07:00:00"
            fill_date = event_date

            if fill_req in ("T1_VWAP30", "VWAP"):
                fill_ts = f"{fill_date} 10:00:00"
                fill_type = "T1_VWAP30"
            else:
                fill_ts = f"{fill_date} 09:30:00"
                fill_type = "T1_OPEN"

        else:
            # DURING_HOURS or Unspecified: Conservative fallback
            signal_ts = f"{event_date} 09:00:00"
            announcement_ts = f"{event_date} 12:00:00"
            dt_t1 = dt_event + timedelta(days=1)
            fill_ts = f"{dt_t1.strftime('%Y-%m-%d')} 09:30:00"
            fill_type = "T1_OPEN"

        return {
            "reporting_time": rpt,
            "signal_timestamp": signal_ts,
            "announcement_timestamp": announcement_ts,
            "execution_timestamp": fill_ts,
            "execution_fill_type": fill_type,
            "is_compliant": True,
        }


def resolve_earnings_event_execution(
    event_date: str,
    reporting_time: str,
    requested_fill_target: str = "T1_OPEN",
) -> Dict[str, Any]:
    """Convenience wrapper for EarningsEventClock.resolve_event_execution."""
    return EarningsEventClock.resolve_event_execution(
        event_date=event_date,
        reporting_time=reporting_time,
        requested_fill_target=requested_fill_target,
    )

