"""Institutional Test Battery: Purged Walk-Forward Cross-Validation."""

import unittest
import pandas as pd
from qlib.contrib.backtest.purged_walk_forward_cv import PurgedWalkForwardCV


class TestPurgedWalkForwardCV(unittest.TestCase):
    def test_zero_event_overlap_invariant(self):
        # Generate 1500 consecutive business days (~6 years)
        dates = pd.date_range("2018-01-01", periods=1500, freq="B")

        cv = PurgedWalkForwardCV(
            train_window_days=500,
            test_window_days=125,
            embargo_days=10,
            step_days=125,
        )

        fold_count = 0
        for train_idx, test_idx in cv.split(dates):
            fold_count += 1
            train_dates = set(dates[train_idx])
            test_dates = set(dates[test_idx])

            # INVARIANT ASSERTION: Zero overlap between train and test dates
            overlap = train_dates.intersection(test_dates)
            self.assertEqual(len(overlap), 0, f"Detected {len(overlap)} overlapping dates in fold {fold_count}!")

            # INVARIANT ASSERTION: Embargo gap must be at least 10 days
            min_test = min(test_dates)
            max_train = max(train_dates)
            gap_days = (min_test - max_train).days
            self.assertGreaterEqual(gap_days, 10, f"Embargo gap {gap_days} is less than required 10 days in fold {fold_count}!")

        self.assertGreater(fold_count, 2, "Expected at least 3 valid walk-forward folds.")


if __name__ == "__main__":
    unittest.main()

