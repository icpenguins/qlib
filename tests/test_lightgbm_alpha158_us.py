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
from scripts.train_alpha158_lightgbm import (
    audit_universe_and_features,
    audit_dataset_segments,
    calculate_ic_metrics,
    resolve_factor_attribution,
    print_institutional_summary_banner,
)


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

    def test_audit_universe_and_features(self, tmp_path):
        """Verify pre-flight universe storage auditor detects valid, missing, and corrupt data."""
        qlib_dir = tmp_path / "qlib_data"
        inst_dir = qlib_dir / "instruments"
        feat_dir = qlib_dir / "features"
        inst_dir.mkdir(parents=True)
        feat_dir.mkdir(parents=True)

        # Create mock instruments file with 3 tickers: AAPL (valid), MSFT (missing), TSLA (corrupt)
        inst_file = inst_dir / "test_universe.txt"
        inst_file.write_text("AAPL\t2020-01-01\t2026-09-04\nMSFT\t2020-01-01\t2026-09-04\nTSLA\t2020-01-01\t2026-09-04\nOLD\t2015-01-01\t2019-12-31\n", encoding="utf-8")

        # AAPL: valid directory and binaries > 4 bytes
        aapl_dir = feat_dir / "aapl"
        aapl_dir.mkdir(parents=True)
        for req in ["close.day.bin", "open.day.bin", "high.day.bin", "low.day.bin", "volume.day.bin"]:
            (aapl_dir / req).write_bytes(b"\x00" * 32)

        # TSLA: corrupt directory with 0-byte binary
        tsla_dir = feat_dir / "tsla"
        tsla_dir.mkdir(parents=True)
        (tsla_dir / "close.day.bin").write_bytes(b"\x00" * 2)

        audit = audit_universe_and_features(qlib_dir, market="test_universe", start_date="2020-01-01")

        assert audit["targeted_total"] == 4
        assert audit["valid_count"] == 1
        assert "AAPL" in audit["valid_tickers"]
        assert audit["missing_count"] == 1
        assert "MSFT" in audit["missing_tickers"]
        assert audit["corrupt_count"] == 1
        assert "TSLA" in audit["corrupt_tickers"]
        assert audit["delisted_count"] == 1
        assert "OLD" in audit["delisted_tickers"]

    def test_calculate_ic_metrics_annualized(self):
        """Verify IC, Rank IC, and Annualized ICIR calculations."""
        dates = pd.date_range("2025-01-01", periods=10, freq="B")
        instruments = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
        idx = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])

        # Create perfectly correlated predictions and labels
        pred_df = pd.DataFrame({"score": np.linspace(0.1, 1.0, len(idx))}, index=idx)
        label_df = pd.DataFrame({"label": np.linspace(0.1, 1.0, len(idx))}, index=idx)

        metrics = calculate_ic_metrics(pred_df, label_df)

        assert metrics["mean_ic"] > 0.99
        assert metrics["rank_ic"] > 0.99
        assert metrics["icir"] > 0.0
        assert metrics["annualized_icir"] > metrics["icir"]
        assert metrics["daily_observations"] == 10

    def test_resolve_factor_attribution(self):
        """Verify canonical formula and economic ontology resolution for Alpha158 factors."""
        raw_names = ["Column_0", "CORD20", "ROC60", "KMID"]
        importances = np.array([500.0, 1200.0, 800.0, 300.0])

        attributed = resolve_factor_attribution(raw_names, importances)

        assert len(attributed) == 4
        assert attributed[0]["feature"] == "CORD20"
        assert attributed[0]["gain"] == 1200.0
        assert "Price Return - Volume Change Correlation" in attributed[0]["name"]
        assert "institutional accumulation" in attributed[0]["description"].lower()

        # Check Column_0 resolved to KMID
        assert any(f["feature"] == "KMID" for f in attributed)

    def test_print_institutional_summary_banner(self, capsys):
        """Verify institutional banner renders all 6 sections cleanly."""
        universe_audit = {"targeted_total": 100, "valid_count": 95, "missing_count": 5, "corrupt_count": 0, "delisted_count": 0}
        segment_stats = {
            "train": {"rows": 100000, "start_date": "2020-01-01", "end_date": "2023-12-31", "trading_days": 1000, "active_ticker_count": 95, "daily_breadth_min": 90, "daily_breadth_mean": 94.5, "daily_breadth_max": 95},
            "valid": {"rows": 25000, "start_date": "2024-01-01", "end_date": "2024-12-31", "trading_days": 250, "active_ticker_count": 95, "daily_breadth_min": 90, "daily_breadth_mean": 94.5, "daily_breadth_max": 95},
            "test": {"rows": 40000, "start_date": "2025-01-01", "end_date": "2026-09-04", "trading_days": 420, "active_ticker_count": 95, "daily_breadth_min": 90, "daily_breadth_mean": 94.5, "daily_breadth_max": 95},
        }
        model_stats = {"num_trees": 38, "features_count": 158, "max_depth": 6, "num_leaves": 31, "subsample": 0.88, "colsample_bytree": 0.89}
        ic_metrics = {"mean_ic": 0.0162, "rank_ic": 0.0175, "icir": 0.0994, "annualized_icir": 1.578, "annualized_rank_icir": 1.625, "daily_observations": 418}
        top_features = [{"feature": "CORD20", "gain": 1500.0, "name": "20-Day Price Return - Volume Change Correlation", "formula": "Corr(...)", "description": "Measures accumulation"}]
        artifact_paths = {"Model": Path("/tmp/model.pkl")}

        print_institutional_summary_banner("russell1000", universe_audit, segment_stats, model_stats, ic_metrics, top_features, artifact_paths)
        captured = capsys.readouterr().out

        assert "INSTITUTIONAL TRAINING & AUDIT REPORT" in captured
        assert "UNIVERSE DISCOVERY & TICKER ACCOUNTING" in captured
        assert "DATASET DIMENSIONS & CROSS-SECTIONAL BREADTH" in captured
        assert "MODEL TOPOLOGY & BOOSTING TREE METRICS" in captured
        assert "OUT-OF-SAMPLE TEST METRICS" in captured
        assert "TOP 5 ALPHA ATTRIBUTION FACTORS" in captured
        assert "PERSISTED PRODUCTION ARTIFACTS" in captured
