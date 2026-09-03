#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit and Integration Tests for Targeted US Stock Data Downloader
===============================================================
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Add repo root and scripts directory to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from download_us_selected_data import (
    DEFAULT_US_SYMBOLS,
    parse_symbols,
    load_symbols_from_file,
    normalize_symbol_data,
    dump_to_qlib_format,
    build_parser,
)


class TestDownloadUSSelectedData(unittest.TestCase):
    """Test suite for targeted US data collection and formatting."""

    def test_default_symbols(self):
        """Ensure all 12 requested US symbols are present by default."""
        expected = ["VOO", "FIX", "CRDO", "MSFT", "INTC", "MU", "ANET", "IBM", "TSLA", "NVDA", "SPY", "QQQ"]
        self.assertEqual(DEFAULT_US_SYMBOLS, expected)

    def test_parse_symbols(self):
        """Test symbol parsing across various input formats."""
        # Comma-separated string
        s1 = parse_symbols("VOO, FIX, crdo, msft")
        self.assertEqual(s1, ["VOO", "FIX", "CRDO", "MSFT"])

        # Space-separated string
        s2 = parse_symbols("intc mu anet IBM")
        self.assertEqual(s2, ["INTC", "MU", "ANET", "IBM"])

        # List with duplicates and whitespace
        s3 = parse_symbols(["tsla", "NVDA ", "spy", "TSLA", "qqq"])
        self.assertEqual(s3, ["TSLA", "NVDA", "SPY", "QQQ"])

    def test_normalize_symbol_data(self):
        """Test Qlib 1D price normalization logic."""
        # Create synthetic raw stock data for 3 trading days
        # Day 1: close = 100, adjclose = 100
        # Day 2: 2-for-1 split occurs, close = 55, adjclose = 110 (split adjusted)
        # Day 3: close = 60, adjclose = 120
        raw_df = pd.DataFrame({
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "open": [98.0, 54.0, 58.0],
            "high": [102.0, 56.0, 62.0],
            "low": [97.0, 52.0, 57.0],
            "close": [100.0, 55.0, 60.0],
            "adjclose": [100.0, 110.0, 120.0],
            "volume": [1000.0, 2000.0, 1500.0],
            "symbol": "MSFT",
        })

        norm_df = normalize_symbol_data(raw_df, "MSFT")

        # Required columns
        expected_cols = ["date", "symbol", "open", "high", "low", "close", "volume", "factor", "change"]
        for col in expected_cols:
            self.assertIn(col, norm_df.columns)

        # In Qlib 1D normalization:
        # First valid trading day's close is standardized to 1.0!
        self.assertAlmostEqual(norm_df.loc[0, "close"], 1.0, places=5)

        # Factor = adjclose / close
        self.assertAlmostEqual(norm_df.loc[0, "factor"], 1.0, places=5)
        self.assertAlmostEqual(norm_df.loc[1, "factor"], 2.0, places=5)
        self.assertAlmostEqual(norm_df.loc[2, "factor"], 2.0, places=5)

        # First day change is 0.0
        self.assertAlmostEqual(norm_df.loc[0, "change"], 0.0, places=5)

        # Day 2 split-adjusted return: from 100 to 110 (+10%)
        # Day 2 close before baseline normalization was 55 * 2.0 = 110
        # Normalized close = 110 / 100 = 1.10
        self.assertAlmostEqual(norm_df.loc[1, "close"], 1.10, places=5)
        self.assertAlmostEqual(norm_df.loc[1, "change"], 0.10, places=5)

    def test_dump_to_qlib_format(self):
        """Test dumping normalized data to Qlib directory layout and binary format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
            df_msft = pd.DataFrame({
                "date": dates,
                "symbol": ["MSFT"] * 3,
                "open": [0.98, 1.08, 1.16],
                "high": [1.02, 1.12, 1.24],
                "low": [0.97, 1.04, 1.14],
                "close": [1.0, 1.10, 1.20],
                "volume": [1000.0, 1000.0, 750.0],
                "factor": [1.0, 2.0, 2.0],
                "change": [0.0, 0.10, 0.0909],
            })

            df_nvda = pd.DataFrame({
                "date": ["2024-01-03", "2024-01-04"],
                "symbol": ["NVDA"] * 2,
                "open": [0.95, 1.05],
                "high": [1.05, 1.15],
                "low": [0.90, 1.00],
                "close": [1.00, 1.10],
                "volume": [5000.0, 6000.0],
                "factor": [1.0, 1.0],
                "change": [0.0, 0.10],
            })

            data_map = {"MSFT": df_msft, "NVDA": df_nvda}
            dump_to_qlib_format(data_map, tmp_path, freq="day")

            # Verify calendars/day.txt
            cal_file = tmp_path / "calendars" / "day.txt"
            self.assertTrue(cal_file.exists())
            cal_lines = cal_file.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(cal_lines, dates)

            # Verify instruments/all.txt
            inst_file = tmp_path / "instruments" / "all.txt"
            self.assertTrue(inst_file.exists())
            inst_lines = inst_file.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(inst_lines), 2)
            inst_dict = {line.split("\t")[0]: (line.split("\t")[1], line.split("\t")[2]) for line in inst_lines}
            self.assertEqual(inst_dict["MSFT"], ("2024-01-02", "2024-01-04"))
            self.assertEqual(inst_dict["NVDA"], ("2024-01-03", "2024-01-04"))

            # Verify binary feature files for MSFT
            msft_close_bin = tmp_path / "features" / "MSFT" / "close.day.bin"
            self.assertTrue(msft_close_bin.exists())
            msft_arr = np.fromfile(str(msft_close_bin), dtype="<f")
            # Index 0 is start_index in calendar (0 for MSFT)
            self.assertEqual(msft_arr[0], 0.0)
            np.testing.assert_allclose(msft_arr[1:], [1.0, 1.10, 1.20], rtol=1e-4)

            # Verify binary feature files for NVDA (started on 2024-01-03, index 1)
            nvda_close_bin = tmp_path / "features" / "NVDA" / "close.day.bin"
            self.assertTrue(nvda_close_bin.exists())
            nvda_arr = np.fromfile(str(nvda_close_bin), dtype="<f")
            # Index 0 is start_index in calendar (1 for NVDA)
            self.assertEqual(nvda_arr[0], 1.0)
            np.testing.assert_allclose(nvda_arr[1:], [1.00, 1.10], rtol=1e-4)

    def test_load_symbols_from_txt_file(self):
        """Test loading symbols from a text file with comments and mixed delimiters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "tickers.txt"
            content = """# Targeted tickers
            VOO, FIX
            # Semis
            CRDO
            MSFT, INTC, MU
            ANET   IBM
            TSLA # EV manufacturer
            NVDA, SPY, QQQ
            """
            file_path.write_text(content, encoding="utf-8")

            # Direct load
            symbols = load_symbols_from_file(file_path)
            self.assertEqual(symbols, DEFAULT_US_SYMBOLS)

            # Auto-detection via parse_symbols with string path
            s_parsed = parse_symbols(str(file_path))
            self.assertEqual(s_parsed, DEFAULT_US_SYMBOLS)

            # Auto-detection via parse_symbols with list containing file path
            s_list_parsed = parse_symbols([str(file_path)])
            self.assertEqual(s_list_parsed, DEFAULT_US_SYMBOLS)

    def test_load_symbols_from_csv_file(self):
        """Test loading symbols from a CSV file with a column header."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "watchlist.csv"
            content = "Ticker,Weight\nAAPL,0.3\nMSFT,0.4\nGOOG,0.3\n"
            file_path.write_text(content, encoding="utf-8")

            symbols = load_symbols_from_file(file_path)
            self.assertEqual(symbols, ["AAPL", "MSFT", "GOOG"])

    def test_load_symbols_from_json_file(self):
        """Test loading symbols from a JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "universe.json"
            content = '["VOO", "FIX", "CRDO", "MSFT"]'
            file_path.write_text(content, encoding="utf-8")

            symbols = load_symbols_from_file(file_path)
            self.assertEqual(symbols, ["VOO", "FIX", "CRDO", "MSFT"])

    def test_parser_target_dir(self):
        """Test that CLI parser correctly recognizes --target_dir and its aliases."""
        parser = build_parser()
        args1 = parser.parse_args(["--target_dir", "/tmp/custom_data"])
        self.assertEqual(args1.target_dir, "/tmp/custom_data")

        args2 = parser.parse_args(["-o", "/tmp/out_data"])
        self.assertEqual(args2.target_dir, "/tmp/out_data")

        args3 = parser.parse_args(["--output_dir", "/tmp/out_data_2"])
        self.assertEqual(args3.target_dir, "/tmp/out_data_2")

    def test_target_dir_storage_creation(self):
        """Test that dump_to_qlib_format correctly populates specified target_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "my_custom_store"
            dates = ["2024-01-02", "2024-01-03"]
            df_sample = pd.DataFrame({
                "date": dates,
                "symbol": ["MSFT", "MSFT"],
                "open": [1.0, 1.1],
                "high": [1.05, 1.15],
                "low": [0.95, 1.05],
                "close": [1.0, 1.1],
                "volume": [1000.0, 1200.0],
                "factor": [1.0, 1.0],
                "change": [0.0, 0.1],
            })

            # Dump directly into target_dir / "qlib_data"
            qlib_out = target_path / "qlib_data"
            dump_to_qlib_format({"MSFT": df_sample}, qlib_out)

            self.assertTrue(qlib_out.joinpath("calendars", "day.txt").exists())
            self.assertTrue(qlib_out.joinpath("instruments", "all.txt").exists())
            self.assertTrue(qlib_out.joinpath("features", "MSFT", "close.day.bin").exists())


if __name__ == "__main__":
    unittest.main()

