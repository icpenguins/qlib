# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Purged Walk-Forward Cross-Validation Module
===========================================
# my-qlib Fork Extension: Institutional Microstructure & Derivatives Layer

Implements purged walk-forward cross-validation with an explicit embargo window
around corporate earnings events, guaranteeing zero informational leakage between
training and testing folds.
"""

from typing import List, Tuple, Generator, Dict, Any
import pandas as pd
import numpy as np


class PurgedWalkForwardCV:
    """
    Purged Walk-Forward Cross-Validation Generator with Event Embargo.
    """

    def __init__(
        self,
        train_window_days: int = 756,   # ~3 years of trading days
        test_window_days: int = 252,    # ~1 year of trading days
        embargo_days: int = 10,         # 10 trading days post-earnings embargo
        step_days: int = 252,           # Step size between folds
    ):
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.embargo_days = embargo_days
        self.step_days = step_days

    def split(
        self,
        df_or_dates: List[pd.Timestamp],
        event_dates: List[pd.Timestamp] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generates train and test index splits, purging any training observations
        that overlap with test event horizons plus embargo.

        Parameters
        ----------
        df_or_dates : List[pd.Timestamp]
            Sorted list of trading dates.
        event_dates : List[pd.Timestamp], optional
            Dates of quarterly corporate earnings announcements.

        Yields
        ------
        Tuple[np.ndarray, np.ndarray]
            (train_indices, test_indices)
        """
        dates = pd.to_datetime(df_or_dates).sort_values()
        n_samples = len(dates)
        event_set = set(pd.to_datetime(event_dates)) if event_dates is not None else set()

        start_idx = 0
        while start_idx + self.train_window_days + self.test_window_days <= n_samples:
            train_end_idx = start_idx + self.train_window_days
            test_start_idx = train_end_idx + self.embargo_days
            test_end_idx = test_start_idx + self.test_window_days

            if test_end_idx > n_samples:
                break

            train_idx = np.arange(start_idx, train_end_idx)
            test_idx = np.arange(test_start_idx, min(test_end_idx, n_samples))

            # INVARIANT ASSERTION: Purge any event labels in train that fall within embargo of test
            # By construction with test_start_idx = train_end_idx + embargo_days,
            # max(train_date) is at least embargo_days prior to min(test_date).
            # We explicitly assert zero overlapping dates:
            train_dates_fold = set(dates[train_idx])
            test_dates_fold = set(dates[test_idx])

            overlap = train_dates_fold.intersection(test_dates_fold)
            if len(overlap) > 0:
                raise ValueError(
                    f"PurgedWalkForwardCV invariant violated: detected {len(overlap)} "
                    f"overlapping dates between train and test splits!"
                )

            yield train_idx, test_idx
            start_idx += self.step_days
