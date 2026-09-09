#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit Tests for scripts/ensemble_lib.py
=======================================
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import qlib
from download_us_selected_data import dump_to_qlib_format
from ensemble_lib import find_available_benchmark, run_portfolio_backtest


class TestFindAvailableBenchmark(unittest.TestCase):
    """
    Regression tests for the 2026-09-09 fix: `train_ensemble.py` previously
    hardcoded --benchmark's default (SPY) with no check against the actual
    --data_dir, so a data directory built from a constituents-only download
    (which doesn't include the ETF itself) failed deep inside
    `qlib.backtest.report.PortfolioMetrics.init_bench` -- only after a full
    training run had already completed. `find_available_benchmark` validates
    up front, using the exact expression/function
    (`$close/Ref($close,1)-1` via `qlib.utils.resam.get_higher_eq_freq_feature`)
    that check actually uses, so it can never disagree with the real backtest
    step's own verdict.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        qlib_dir = Path(cls.tmpdir.name)

        dates = pd.bdate_range(start="2024-01-02", periods=60).strftime("%Y-%m-%d").tolist()
        df_present = pd.DataFrame({
            "date": dates,
            "symbol": ["PRESENT"] * len(dates),
            "open": [1.0] * len(dates),
            "high": [1.0] * len(dates),
            "low": [1.0] * len(dates),
            "close": [1.0 + 0.001 * i for i in range(len(dates))],
            "volume": [1000.0] * len(dates),
            "factor": [1.0] * len(dates),
            "change": [0.0] * len(dates),
        })
        # A symbol with exactly ONE row: its Ref($close,1)-shifted return is
        # NaN (nothing to shift from). Confirmed directly against
        # qlib.utils.resam.get_higher_eq_freq_feature: this still returns a
        # length-1 result (NaN value, not an empty frame). Qlib's own
        # `_cal_benchmark` gate (`len(_temp_result) == 0`) does not inspect
        # the *values*, only whether anything came back at all -- so it
        # accepts this and proceeds to `.fillna(0)`, silently producing a
        # zero-return benchmark day rather than raising. This test exists to
        # pin that (surprising, but real and verified) behavior: this
        # function must match qlib's actual leniency here, not add a
        # stricter check that could reject a symbol the real backtest step
        # would have accepted.
        df_single_row = pd.DataFrame({
            "date": [dates[0]],
            "symbol": ["SINGLEROW"],
            "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0],
            "volume": [1000.0], "factor": [1.0], "change": [0.0],
        })

        dump_to_qlib_format(
            {"PRESENT": df_present, "SINGLEROW": df_single_row},
            qlib_dir,
            freq="day",
        )
        qlib.init(provider_uri=str(qlib_dir), region="us")
        cls.window = (dates[0], dates[-1])

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_finds_present_symbol(self):
        start, end = self.window
        self.assertEqual(find_available_benchmark(["PRESENT"], start, end), "PRESENT")

    def test_returns_none_when_nothing_resolves(self):
        start, end = self.window
        self.assertIsNone(find_available_benchmark(["MISSING1", "MISSING2"], start, end))

    def test_falls_back_past_missing_candidates(self):
        start, end = self.window
        self.assertEqual(
            find_available_benchmark(["MISSING1", "MISSING2", "PRESENT"], start, end),
            "PRESENT",
        )

    def test_matches_qlib_own_leniency_for_a_single_row_series(self):
        """
        A single-row series' Ref($close,1)-shifted return is NaN, but qlib's
        own `_cal_benchmark` gate only checks `len(result) == 0`, not whether
        the values are non-NaN -- so it treats this as valid (see class
        docstring / setUpClass comment for the verified evidence). This
        function must accept it too: rejecting it would be *stricter* than
        the real backtest step, which could wrongly force a fallback search
        away from a symbol that would have actually worked.
        """
        start, end = self.window
        self.assertEqual(find_available_benchmark(["SINGLEROW"], start, end), "SINGLEROW")


class TestRunPortfolioBacktestCodes(unittest.TestCase):
    """
    Regression tests for the 2026-09-09 fix: ``run_portfolio_backtest()`` never threaded a
    resolved trading universe into ``exchange_kwargs["codes"]``, so ``qlib.backtest.get_exchange()``
    fell back to its own signature default ``codes="all"``. ``Exchange.get_quote_from_qlib()``
    (``qlib/backtest/exchange.py``) then passes that literal string straight into
    ``D.features("all", ...)``, which qlib resolves as a MARKET NAME
    (``<data_dir>/instruments/all.txt``), not a wildcard -- reproducing the exact reported
    ``ValueError: instrument not exists: .../all.txt`` whenever that file doesn't exist (e.g.
    it was renamed to a differently-named universe file, as happened in practice).

    Builds a tiny real qlib data directory (same ``download_us_selected_data.dump_to_qlib_format``
    helper `TestFindAvailableBenchmark` uses) and then deliberately renames its ``all.txt`` away
    -- mirroring the real-world scenario verbatim -- so these tests exercise the actual qlib
    backtest/exchange code path, not a mock.
    """

    N_SYMBOLS = 6
    N_DAYS = 90

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        qlib_dir = Path(cls.tmpdir.name)

        dates = pd.bdate_range(start="2024-01-02", periods=cls.N_DAYS).strftime("%Y-%m-%d").tolist()
        cls.symbols = [f"SYM{i}" for i in range(cls.N_SYMBOLS)]
        rng = np.random.RandomState(42)

        normalized_dfs = {}
        for i, sym in enumerate(cls.symbols):
            # Each symbol gets a distinct, deterministic drift so TopkDropoutStrategy's
            # top-K selection actually differentiates between names across the window
            # (a flat/identical price series across symbols would make every day's
            # ranking degenerate/tied).
            drift = 0.0005 * (i - cls.N_SYMBOLS / 2.0)
            noise = rng.normal(0, 0.002, size=len(dates))
            closes = 100.0 * np.cumprod(1.0 + drift + noise)
            normalized_dfs[sym] = pd.DataFrame({
                "date": dates,
                "symbol": [sym] * len(dates),
                "open": closes,
                "high": closes * 1.001,
                "low": closes * 0.999,
                "close": closes,
                "volume": [1_000_000.0] * len(dates),
                "factor": [1.0] * len(dates),
                "change": [0.0] * len(dates),
            })

        dump_to_qlib_format(normalized_dfs, qlib_dir, freq="day")

        # Reproduce the real-world scenario verbatim: `all.txt` (written by dump_to_qlib_format,
        # matching the OLD provider layout) gets renamed to a differently-named universe file
        # (matching this session's real D:/trading/qlib2 rename to sp500.txt), so
        # <data_dir>/instruments/all.txt no longer exists.
        all_txt = qlib_dir / "instruments" / "all.txt"
        renamed_txt = qlib_dir / "instruments" / "sp500.txt"
        assert all_txt.exists(), "dump_to_qlib_format should have written instruments/all.txt"
        all_txt.rename(renamed_txt)
        assert not all_txt.exists()

        qlib.init(provider_uri=str(qlib_dir), region="us")

        # A resolved (datetime, instrument)-MultiIndexed prediction series over a window safely
        # inside the dumped calendar, mirroring what train_ensemble.py's real pipeline produces.
        window_dates = pd.to_datetime(dates[10:70])
        cls.benchmark = cls.symbols[0]
        cls.start_time = window_dates[0].strftime("%Y-%m-%d")
        cls.end_time = window_dates[-1].strftime("%Y-%m-%d")

        idx = pd.MultiIndex.from_product([window_dates, cls.symbols], names=["datetime", "instrument"])
        # Deterministic per-(date, symbol) scores correlated with the symbol's own drift rank,
        # so TopkDropoutStrategy has a genuine (not tied/degenerate) top-K to select each day.
        scores = [i + rng.normal(0, 0.01) for _ in window_dates for i in range(cls.N_SYMBOLS)]
        cls.pred_series = pd.Series(scores, index=idx, name="score")

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_missing_codes_raises_clear_error_instead_of_silent_qlib_default(self):
        """Omitting `codes` must fail loudly and immediately, with an actionable message --
        never silently fall through to qlib's own get_exchange() codes="all" default."""
        with self.assertRaises(ValueError) as ctx:
            run_portfolio_backtest(self.pred_series, benchmark=self.benchmark, topk=2, n_drop=1)
        self.assertIn("requires `codes`", str(ctx.exception))

    def test_explicit_codes_all_reproduces_the_exact_reported_bug(self):
        """Pins the root-cause diagnosis: a caller that explicitly (mis)uses qlib's own
        codes="all" default against this data directory hits the EXACT reported error, via the
        real qlib exchange/backtest code path (not a mock) -- confirming the diagnosis, not just
        the fix. `status` must be "failed" (not silently masked as a flat/zero result), and the
        captured error must be the real `ValueError: instrument not exists: .../all.txt`."""
        result = run_portfolio_backtest(
            self.pred_series, benchmark=self.benchmark, codes="all", topk=2, n_drop=1
        )
        self.assertEqual(result["status"], "failed")
        self.assertIn("instrument not exists", result["error"])
        self.assertIn("all.txt", result["error"])

    def test_resolved_codes_list_completes_the_backtest_without_all_txt(self):
        """The actual fix: passing the already-resolved instrument list (what
        `parse_instruments()` returns) lets the real backtest complete even though
        `<data_dir>/instruments/all.txt` does not exist."""
        result = run_portfolio_backtest(
            self.pred_series, benchmark=self.benchmark, codes=self.symbols, topk=2, n_drop=1
        )
        self.assertEqual(result.get("error"), None, msg=f"Backtest failed unexpectedly: {result}")
        self.assertEqual(result["status"], "ok")
        for key in (
            "annualized_return_gross",
            "information_ratio_gross",
            "annualized_return_net",
            "information_ratio_net",
        ):
            self.assertTrue(result[key] == result[key], msg=f"{key} is NaN: {result}")  # not NaN

    def test_resolved_codes_as_market_name_string_also_completes(self):
        """`codes` also accepts a market-name string (the other shape `parse_instruments()`
        returns) provided that named instruments file actually exists -- here "sp500", the
        renamed file, rather than the no-longer-existent "all"."""
        result = run_portfolio_backtest(
            self.pred_series, benchmark=self.benchmark, codes="sp500", topk=2, n_drop=1
        )
        self.assertEqual(result["status"], "ok", msg=f"Backtest failed unexpectedly: {result}")


if __name__ == "__main__":
    unittest.main()
