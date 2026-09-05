#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit & Integration Tests for LightGBM Alpha158 US Equities Pipeline
===================================================================
Tests workflow configuration, universe curation, binary feature integrity (VWAP),
Python fallback operators, model inference, and stock analysis engine integration.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from ruamel.yaml import YAML

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.get_russell1000_symbols import get_curated_russell1000_universe, write_qlib_instrument_file
from scripts.download_us_selected_data import normalize_symbol_data
from scripts.infer_alpha158 import Alpha158Scorer
from scripts.stock_analysis_engine import compute_alpha158_features
from scripts.stock_analysis_data import prepare_analysis_json_payload


class TestLightGBMAlpha158US:
    """Test suite for US LightGBM Alpha158 training & inference ecosystem."""

    def test_workflow_config_schema(self):
        """Verify YAML config exists, parses cleanly, and adheres to US market specification."""
        config_path = REPO_ROOT / "examples" / "benchmarks" / "LightGBM" / "workflow_config_lightgbm_Alpha158_us_russell1000.yaml"
        assert config_path.exists(), f"Config missing at {config_path}"

        yaml = YAML(typ="safe", pure=True)
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.load(f)

        # Qlib Init
        assert config["qlib_init"]["region"] == "us"
        assert "us_data" in config["qlib_init"]["provider_uri"]
        assert config["market"] == "russell1000"
        assert config["benchmark"] == "SPY"

        # Model & Dataset
        task = config["task"]
        assert task["model"]["class"] == "LGBModel"
        assert task["model"]["kwargs"]["loss"] == "mse"
        assert task["dataset"]["kwargs"]["handler"]["class"] == "Alpha158"

        # Segments
        segments = task["dataset"]["kwargs"]["segments"]
        assert "train" in segments and "valid" in segments and "test" in segments

        # Exchange Friction (US Equities have zero halt limit bands)
        exchange_kwargs = config["port_analysis_config"]["backtest"]["exchange_kwargs"]
        assert exchange_kwargs.get("limit_threshold") is None
        assert exchange_kwargs.get("open_cost") <= 0.0005
        assert exchange_kwargs.get("close_cost") <= 0.0005

    def test_russell1000_universe_curation(self, tmp_path):
        """Verify Russell 1000 universe generator produces valid Qlib instrument TSV."""
        seed_symbols = get_curated_russell1000_universe(seed_only=True)
        assert len(seed_symbols) >= 50
        assert "MSFT" in seed_symbols
        assert "AAPL" in seed_symbols
        assert "NVDA" in seed_symbols

        # Verify formatting
        for s in seed_symbols:
            assert s == s.upper()
            assert len(s) <= 6

        # Verify file output
        test_file = tmp_path / "instruments" / "russell1000.txt"
        write_qlib_instrument_file(seed_symbols, test_file, start_date="2020-01-01", end_date="2026-09-05")
        assert test_file.exists()

        lines = test_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == len(seed_symbols)
        first_line_parts = lines[0].split("\t")
        assert len(first_line_parts) == 3
        assert first_line_parts[1] == "2020-01-01"

    def test_vwap_normalization_and_dump_integrity(self):
        """Verify normalize_symbol_data computes VWAP correctly for Alpha158 compliance."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        raw_df = pd.DataFrame({
            "date": dates,
            "open": [100.0 + i for i in range(10)],
            "high": [105.0 + i for i in range(10)],
            "low": [98.0 + i for i in range(10)],
            "close": [102.0 + i for i in range(10)],
            "adjclose": [102.0 + i for i in range(10)],
            "volume": [1000000 + i * 10000 for i in range(10)],
            "symbol": ["TEST"] * 10,
        })

        norm_df = normalize_symbol_data(raw_df, "TEST")
        assert not norm_df.empty
        assert "vwap" in norm_df.columns
        assert "change" in norm_df.columns
        assert "factor" in norm_df.columns

        # Verify VWAP value: (high + low + close) / 3 scaled by first close
        first_close = raw_df["close"].iloc[0]
        expected_raw_vwap = (raw_df["high"].iloc[0] + raw_df["low"].iloc[0] + raw_df["close"].iloc[0]) / 3.0
        expected_scaled_vwap = expected_raw_vwap / first_close
        assert abs(norm_df["vwap"].iloc[0] - expected_scaled_vwap) < 1e-4

    def test_rolling_and_expanding_python_fallbacks(self):
        """Verify Python fallback implementations for rolling and expanding operators."""
        from qlib.data._libs.rolling import rolling_mean, rolling_slope, rolling_rsquare, rolling_resi
        from qlib.data._libs.expanding import expanding_mean, expanding_slope, expanding_rsquare, expanding_resi

        arr = np.array([10.0, 12.0, 11.0, 14.0, 15.0, 17.0, 16.0, 19.0, 20.0, 22.0], dtype=np.float64)

        # Rolling
        r_mean = rolling_mean(arr, 3)
        assert len(r_mean) == len(arr)
        assert not np.isnan(r_mean[-1])

        r_slope = rolling_slope(arr, 3)
        assert len(r_slope) == len(arr)
        assert not np.isnan(r_slope[-1])

        r_r2 = rolling_rsquare(arr, 3)
        assert len(r_r2) == len(arr)
        assert 0.0 <= r_r2[-1] <= 1.0

        r_resi = rolling_resi(arr, 3)
        assert len(r_resi) == len(arr)

        # Expanding
        e_mean = expanding_mean(arr)
        assert len(e_mean) == len(arr)
        assert abs(e_mean[-1] - np.mean(arr)) < 1e-6

        e_slope = expanding_slope(arr)
        assert len(e_slope) == len(arr)
        assert not np.isnan(e_slope[-1])

    def test_alpha158_inference_scorer_api(self):
        """Verify Alpha158Scorer schema, percentile calculation, and conviction badges."""
        scorer = Alpha158Scorer()
        score_res = scorer.get_score("MSFT")

        assert isinstance(score_res, dict)
        assert score_res["symbol"] == "MSFT"
        assert "alpha158_score" in score_res
        assert "percentile" in score_res
        assert "rank" in score_res
        assert "conviction" in score_res
        assert "conviction_badge" in score_res
        assert "top_factors" in score_res
        assert len(score_res["top_factors"]) >= 2
        assert 0.0 <= score_res["percentile"] <= 100.0

    def test_stock_analysis_engine_and_data_integration(self):
        """Verify compute_alpha158_features integrates into stock_analysis_data payload."""
        alpha_info = compute_alpha158_features("MSFT")
        assert alpha_info is not None
        assert alpha_info["symbol"] == "MSFT"

        # Verify integration in canonical data contract
        mock_analysis = {
            "symbol": "MSFT",
            "request_date": "2026-09-04",
            "latest_data_date": "2026-09-04",
            "is_up_to_date": True,
            "forecast_days": 63,
            "historical_data": pd.DataFrame({
                "date": pd.date_range("2024-01-01", periods=40, freq="D"),
                "close": [400.0 + i for i in range(40)],
                "open": [399.0 + i for i in range(40)],
                "high": [405.0 + i for i in range(40)],
                "low": [398.0 + i for i in range(40)],
                "volume": [1000000] * 40,
            }),
            "alpha158": alpha_info,
        }

        canonical_payload = prepare_analysis_json_payload(mock_analysis)
        assert "alpha158" in canonical_payload
        assert canonical_payload["alpha158"]["symbol"] == "MSFT"
        assert "alpha158_score" in canonical_payload["alpha158"]
        assert "percentile" in canonical_payload["alpha158"]
