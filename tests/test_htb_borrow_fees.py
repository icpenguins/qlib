"""Institutional Test Battery: Hard-To-Borrow Fees & Locate Capacity."""

import unittest
from qlib.contrib.backtest.borrow_fee_engine import BorrowFeeEngine, ZeroLocateCapacityError


class TestHTBBorrowFees(unittest.TestCase):
    def setUp(self):
        self.engine = BorrowFeeEngine()

    def test_zero_locate_raises_invariant_error(self):
        # INVARIANT CHECK: Zero locate capacity MUST raise ZeroLocateCapacityError
        with self.assertRaises(ZeroLocateCapacityError):
            self.engine.calculate_borrow_cost(
                short_value=2_000_000.0,
                annual_fee_rate=0.25,
                days_held=3,
                locate_available=False,
            )

    def test_htb_cost_accrual(self):
        # $1,000,000 short at 30% borrow rate for 3 days
        # Cost = 1,000,000 * 0.30 * (3 / 360) = $2,500
        res = self.engine.calculate_borrow_cost(
            short_value=1_000_000.0,
            annual_fee_rate=0.30,
            days_held=3,
            locate_available=True,
        )
        self.assertTrue(res["is_hard_to_borrow"])
        self.assertAlmostEqual(res["accrued_cost_dollars"], 2500.0, places=1)
        self.assertTrue(res["locate_granted"])


if __name__ == "__main__":
    unittest.main()

