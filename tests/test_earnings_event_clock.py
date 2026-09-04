"""Institutional Test Battery: Earnings Event Clock & AMC/BMO Discipline."""

import unittest
from qlib.contrib.events.earnings_event_clock import (
    EarningsEventClock,
    InvalidEventExecutionError,
    resolve_earnings_event_execution,
)


class TestEarningsEventClock(unittest.TestCase):
    def test_amc_t0_close_fill_raises_invariant_error(self):
        # INVARIANT CHECK: Requesting T0 close on AMC earnings announcement
        # MUST raise an explicit InvalidEventExecutionError to prevent lookahead bias!
        with self.assertRaises(InvalidEventExecutionError):
            resolve_earnings_event_execution(
                event_date="2024-04-25",
                reporting_time="AMC",
                requested_fill_target="T0_CLOSE",
            )

    def test_amc_compliant_t1_open_execution(self):
        # AMC event on Thursday 2024-04-25
        res = resolve_earnings_event_execution(
            event_date="2024-04-25",
            reporting_time="AMC",
            requested_fill_target="T1_OPEN",
        )
        self.assertTrue(res["is_compliant"])
        self.assertEqual(res["signal_timestamp"], "2024-04-25 15:55:00")
        self.assertEqual(res["announcement_timestamp"], "2024-04-25 16:01:00")
        # Execution strictly on next day Friday 2024-04-26 at 09:30:00
        self.assertEqual(res["execution_timestamp"], "2024-04-26 09:30:00")
        self.assertEqual(res["execution_fill_type"], "T1_OPEN")

    def test_bmo_compliant_execution(self):
        # BMO event on Tuesday 2024-04-30
        res = resolve_earnings_event_execution(
            event_date="2024-04-30",
            reporting_time="BMO",
            requested_fill_target="T1_OPEN",
        )
        self.assertTrue(res["is_compliant"])
        # Signal formed at T0 MOC Monday 2024-04-29
        self.assertEqual(res["signal_timestamp"], "2024-04-29 15:55:00")
        # Announcement Tuesday 07:00
        self.assertEqual(res["announcement_timestamp"], "2024-04-30 07:00:00")
        # Execution Tuesday Open 09:30
        self.assertEqual(res["execution_timestamp"], "2024-04-30 09:30:00")


if __name__ == "__main__":
    unittest.main()

