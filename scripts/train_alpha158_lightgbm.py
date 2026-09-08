#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LightGBM Alpha158 Training Pipeline for US Equities (Russell 1000)
=================================================================
Automates data initialization, pre-flight universe and physical feature store auditing,
low-memory dataset inspection, model training, Information Coefficient evaluation
(IC, Rank IC, Daily ICIR, Annualized ICIR), feature attribution with mathematical
formulas and financial semantics, artifact serialization, and cross-sectional score generation.

Reference Architecture:
    examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml
"""

import os
import sys
import json
import time
import uuid
import pickle
import hashlib
import logging
import argparse
import datetime
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Set, cast

import numpy as np
import pandas as pd
from ruamel.yaml import YAML

# Reconfigure stdout to UTF-8 for safe institutional console logging
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace") # pyright: ignore[reportAttributeAccessIssue]
    except Exception:
        pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TrainAlpha158LightGBM")

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import qlib
from qlib.config import C
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.model.trainer import task_train
from qlib.data.dataset.utils import convert_index_format
from qlib.contrib.validation.score_quality import (
    ScoreQualityGateError,
    validate_score_quality,
    extract_ic_metrics,
    extract_portfolio_metrics,
)

# Production artifact paths. These are the ONLY two locations any successful,
# gate-passing training run may write its final model/score artifacts to. A
# hyperparameter override (--num_boost_round today; any future equivalent)
# is structurally forced away from these paths -- see `_resolve_output_dirs`
# below -- closing the exact mechanism (Part 1.3.1 / 3.3 P0 of the
# implementation plan) that let a 1-round "quick test" run silently become
# the 2026-09-08 production incident.
PRODUCTION_MODEL_DIR = REPO_ROOT / "models" / "lightgbm"
PRODUCTION_SCORES_DIR = REPO_ROOT / "output" / "scores"
SMOKETEST_MODEL_DIR = REPO_ROOT / "models" / "lightgbm" / "_smoketest"
SMOKETEST_SCORES_DIR = REPO_ROOT / "output" / "scores" / "_smoketest"
PINNED_REFERENCE_PATH = REPO_ROOT / "models" / "lightgbm" / "pinned_reference" / "alpha158_russell1000_pinned_reference.parquet"
PINNED_REFERENCE_META_PATH = REPO_ROOT / "models" / "lightgbm" / "pinned_reference" / "alpha158_russell1000_pinned_reference_meta.json"


ALPHA158_FACTOR_ONTOLOGY: Dict[str, Dict[str, str]] = {
    "KMID": {
        "name": "Normalized Intraday Return",
        "description": "Candlestick body: (Close - Open) / Open. Reflects regular trading hours directional thrust.",
    },
    "KLEN": {
        "name": "Normalized Intraday Range / Volatility",
        "description": "Total candle range: (High - Low) / Open. Quantifies intraday dispersion and liquidity friction.",
    },
    "KMID2": {
        "name": "Body to Total Shadow Ratio",
        "description": "(Close - Open) / (High - Low). Measures directional conviction relative to total intraday noise.",
    },
    "KUP": {
        "name": "Normalized Upper Shadow",
        "description": "(High - Max(Open, Close)) / Open. Quantifies intraday seller rejection and overhead supply.",
    },
    "KLOW": {
        "name": "Normalized Lower Shadow",
        "description": "(Min(Open, Close) - Low) / Open. Quantifies dip-buying demand and intraday support elasticity.",
    },
    "KSFT": {
        "name": "Candle Body Asymmetry / Skew",
        "description": "(2*Close - High - Low) / Open. Captures close positioning within daily extreme range.",
    },
    "CORD": {
        "name": "{w}-Day Price Return - Volume Change Correlation",
        "description": "Correlation between daily return and log volume change over {w} days. Quantifies institutional accumulation vs distribution.",
    },
    "CORR": {
        "name": "{w}-Day Price - Volume Correlation",
        "description": "Direct rolling correlation between price level and volume over {w} days.",
    },
    "ROC": {
        "name": "{w}-Day Momentum / Rate of Change (Inverted)",
        "description": "Ref(Close, {w}) / Close. Intermediate/short-term momentum reversal signal over {w} trading days.",
    },
    "MA": {
        "name": "{w}-Day Moving Average Trend Ratio",
        "description": "Mean(Close, {w}) / Close. Identifies mean-reversion discount or extension relative to {w}-day trendline.",
    },
    "STD": {
        "name": "{w}-Day Realized Volatility",
        "description": "Std(Close, {w}) / Close. Normalized price dispersion over {w} trading days.",
    },
    "BETA": {
        "name": "{w}-Day Linear Trend Slope",
        "description": "Time-series linear regression slope of close prices over {w} trading days.",
    },
    "RSQR": {
        "name": "{w}-Day Trend Linearity (R-Squared)",
        "description": "Coefficient of determination for linear trend over {w} days. High values signal persistent, low-noise trends.",
    },
    "RESI": {
        "name": "{w}-Day Trend Residual Variance",
        "description": "Residual variance around linear regression over {w} days. Isolates idiosyncratic noise from secular trend.",
    },
    "CNTP": {
        "name": "{w}-Day Positive Day Ratio",
        "description": "Count(Close > Ref(Close,1), {w}) / {w}. Up-day breadth percentage indicating buying persistence.",
    },
    "CNTN": {
        "name": "{w}-Day Negative Day Ratio",
        "description": "Count(Close < Ref(Close,1), {w}) / {w}. Down-day breadth percentage indicating liquidation pressure.",
    },
    "CNTD": {
        "name": "{w}-Day Net Directional Day Breadth",
        "description": "CNTP{w} - CNTN{w}. Net positive vs negative trading day differential over {w} days.",
    },
    "SUMP": {
        "name": "{w}-Day Upward Price Accumulation Ratio",
        "description": "RSI-style ratio of upward price movement to total absolute movement over {w} trading days.",
    },
    "SUMN": {
        "name": "{w}-Day Downward Price Distribution Ratio",
        "description": "Ratio of downward price movement to total absolute movement over {w} trading days.",
    },
    "SUMD": {
        "name": "{w}-Day Net Directional Strength Differential",
        "description": "SUMP{w} - SUMN{w}. Directional oscillator quantifying net bullish vs bearish force over {w} days.",
    },
    "RSV": {
        "name": "{w}-Day Stochastic Oscillator (%K)",
        "description": "(Close - Min(Low, {w})) / (Max(High, {w}) - Min(Low, {w})). Quantifies location within {w}-day channel envelope.",
    },
    "MAX": {
        "name": "{w}-Day High Envelope Ratio",
        "description": "Max(High, {w}) / Close. Proximity to trailing {w}-day cyclical ceiling.",
    },
    "MIN": {
        "name": "{w}-Day Low Envelope Ratio",
        "description": "Min(Low, {w}) / Close. Proximity to trailing {w}-day cyclical floor.",
    },
    "QTLU": {
        "name": "{w}-Day Upper 80% Quantile Ratio",
        "description": "Quantile(Close, {w}, 0.8) / Close. Upper quantile resistance benchmark.",
    },
    "QTLD": {
        "name": "{w}-Day Lower 20% Quantile Ratio",
        "description": "Quantile(Close, {w}, 0.2) / Close. Lower quantile support benchmark.",
    },
    "RANK": {
        "name": "{w}-Day Rolling Percentile Rank",
        "description": "Percentile rank of current close over {w} trading days.",
    },
    "IMAX": {
        "name": "{w}-Day High Recency / Dip Age",
        "description": "Days since {w}-day high. Quantifies momentum exhaustion and pullback duration.",
    },
    "IMIN": {
        "name": "{w}-Day Low Recency / Rally Age",
        "description": "Days since {w}-day low. Quantifies breakout age.",
    },
    "IMXD": {
        "name": "{w}-Day High-Low Distance",
        "description": "Recency difference between {w}-day high and low.",
    },
    "VMA": {
        "name": "{w}-Day Volume Trend Ratio",
        "description": "Mean(Volume, {w}) / Volume. Volume expansion or contraction relative to {w}-day baseline.",
    },
    "VSTD": {
        "name": "{w}-Day Volume Dispersion",
        "description": "Std(Volume, {w}) / Volume. Volatility of turnover over {w} days.",
    },
    "WVMA": {
        "name": "{w}-Day Volume-Weighted Price Dispersion",
        "description": "Volume-weighted standard deviation of price changes over {w} days.",
    },
    "OPEN0": {
        "name": "Current Open Ratio",
        "description": "Open / Close. Overnight gap indicator.",
    },
    "HIGH0": {
        "name": "Current High Ratio",
        "description": "High / Close. Intraday upside excursion.",
    },
    "LOW0": {
        "name": "Current Low Ratio",
        "description": "Low / Close. Intraday downside excursion.",
    },
    "VWAP0": {
        "name": "Current VWAP Ratio",
        "description": "VWAP / Close. Intraday institutional benchmark premium/discount.",
    },
}


def ensure_universe_file(qlib_dir: Path, market_name: str = "russell1000") -> Path:
    """Ensure the specified instrument universe file exists in Qlib provider."""
    inst_dir = qlib_dir / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)
    inst_file = inst_dir / f"{market_name}.txt"

    if inst_file.exists() and inst_file.stat().st_size > 0:
        return inst_file

    # Check repository data/instruments/russell1000.txt
    repo_inst = REPO_ROOT / "data" / "instruments" / f"{market_name}.txt"
    if repo_inst.exists() and repo_inst.stat().st_size > 0:
        logger.info(f"Copying universe file from {repo_inst} to {inst_file}...")
        inst_file.write_text(repo_inst.read_text(encoding="utf-8"), encoding="utf-8")
        return inst_file

    # Check fallback all.txt
    all_file = inst_dir / "all.txt"
    if all_file.exists() and all_file.stat().st_size > 0:
        logger.warning(f"Universe {inst_file} not found. Linking all.txt as {market_name}.txt fallback...")
        inst_file.write_text(all_file.read_text(encoding="utf-8"), encoding="utf-8")
        return inst_file

    # Create dynamic seed instrument file if missing
    logger.info(f"Generating default {market_name}.txt seed universe...")
    from scripts.get_russell1000_symbols import get_curated_russell1000_universe, write_qlib_instrument_file
    symbols = get_curated_russell1000_universe(seed_only=True)
    write_qlib_instrument_file(symbols, inst_file)
    return inst_file


def audit_universe_and_features(
    qlib_dir: Path,
    market: str = "russell1000",
    start_date: str = "2020-01-01",
    required_features: Tuple[str, ...] = ("close.day.bin", "open.day.bin", "high.day.bin", "low.day.bin", "volume.day.bin"),
) -> Dict[str, Any]:
    """
    Audits the target instrument universe against local Qlib binary storage.
    Identifies valid, missing, corrupted, and delisted tickers before launching training.
    """
    inst_file = qlib_dir / "instruments" / f"{market}.txt"
    if not inst_file.exists():
        inst_file = ensure_universe_file(qlib_dir, market)

    targeted_tickers: List[str] = []
    ticker_spans: Dict[str, Tuple[str, str]] = {}

    with open(inst_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            sym = parts[0].strip().upper()
            targeted_tickers.append(sym)
            if len(parts) >= 3:
                ticker_spans[sym] = (parts[1], parts[2])
            else:
                ticker_spans[sym] = ("N/A", "N/A")

    features_dir = qlib_dir / "features"
    valid_tickers: List[str] = []
    missing_dir_tickers: List[str] = []
    corrupt_tickers: Dict[str, str] = {}
    delisted_tickers: Dict[str, str] = {}

    for sym in targeted_tickers:
        sym_dir = features_dir / sym.lower()

        # Check date span bounds if recorded in instrument file
        start_bound, end_bound = ticker_spans.get(sym, ("N/A", "N/A"))
        if end_bound != "N/A" and end_bound < start_date:
            delisted_tickers[sym] = f"Expired before start ({end_bound} < {start_date})"
            continue

        # Check directory existence
        if not sym_dir.exists() or not sym_dir.is_dir():
            missing_dir_tickers.append(sym)
            continue

        # Check existence and non-zero (> 4 bytes header) binary files
        corrupted_files = []
        for req_f in required_features:
            bin_path = sym_dir / req_f
            if not bin_path.exists() or bin_path.stat().st_size <= 4:
                corrupted_files.append(req_f)

        if corrupted_files:
            corrupt_tickers[sym] = f"Corrupt/Missing binaries: {', '.join(corrupted_files)}"
            continue

        valid_tickers.append(sym)

    audit_result = {
        "targeted_total": len(targeted_tickers),
        "valid_count": len(valid_tickers),
        "missing_count": len(missing_dir_tickers),
        "corrupt_count": len(corrupt_tickers),
        "delisted_count": len(delisted_tickers),
        "targeted_tickers": targeted_tickers,
        "valid_tickers": valid_tickers,
        "missing_tickers": missing_dir_tickers,
        "corrupt_tickers": corrupt_tickers,
        "delisted_tickers": delisted_tickers,
    }

    logger.info("=" * 80)
    logger.info(f"UNIVERSE DISCOVERY & STORAGE AUDIT: {market.upper()}")
    logger.info(f"  Targeted Universe Tickers: {audit_result['targeted_total']}")
    pct_valid = (audit_result['valid_count'] / audit_result['targeted_total'] * 100.0) if audit_result['targeted_total'] > 0 else 0.0
    logger.info(f"  Valid Binary Datasets:     {audit_result['valid_count']} ({pct_valid:.1f}%)")
    logger.info(f"  Missing Feature Folders:   {audit_result['missing_count']}")
    logger.info(f"  Corrupt/Empty Binaries:    {audit_result['corrupt_count']}")
    logger.info(f"  Pre-Start Delisted:        {audit_result['delisted_count']}")
    if missing_dir_tickers:
        logger.warning(f"  Sample Missing Tickers (first 5): {missing_dir_tickers[:5]}")
    if corrupt_tickers:
        logger.warning(f"  Sample Corrupted Tickers: {list(corrupt_tickers.keys())[:5]}")
    logger.info("=" * 80)

    return audit_result


def _empty_segment_stats(error: Optional[str] = None) -> Dict[str, Any]:
    """Zeroed stats block so the summary table always has every key it prints."""
    stats: Dict[str, Any] = {
        "rows": 0,
        "rows_before_label_dropna": 0,
        "label_nan_dropped": 0,
        "start_date": "N/A",
        "end_date": "N/A",
        "trading_days": 0,
        "active_ticker_count": 0,
        "active_tickers": [],
        "daily_breadth_min": 0,
        "daily_breadth_max": 0,
        "daily_breadth_mean": 0.0,
        "daily_breadth_median": 0.0,
    }
    if error is not None:
        stats["error"] = error
    return stats


def _open_label_only_loader(handler_config: Dict[str, Any]) -> Tuple[Any, Tuple[list, list], Dict[str, Any]]:
    """
    Instantiate the configured handler with ``init_data=False`` purely to borrow its
    fully-resolved data loader (freq, filter_pipe, inst_processors, label expression)
    without materialising a single row, then hand back the pieces needed to load the
    label group on its own.
    """
    handler_kwargs = dict(handler_config.get("kwargs", {}) or {})
    handler_kwargs["init_data"] = False
    probe = init_instance_by_config({**handler_config, "kwargs": handler_kwargs})

    loader = getattr(probe, "data_loader", None)
    if loader is None or not hasattr(loader, "load_group_df"):
        raise TypeError(f"{type(loader).__name__} does not support per-group loading")
    if not getattr(loader, "is_group", False) or "label" not in getattr(loader, "fields", {}):
        raise KeyError("data loader exposes no 'label' group to project onto")

    scope = {
        "instruments": probe.instruments,
        "start_time": probe.start_time,
        "end_time": probe.end_time,
    }
    return loader, loader.fields["label"], scope


def _slice_segment(df: pd.DataFrame, seg_range: Any) -> pd.DataFrame:
    """
    Slice on the (sorted) datetime level using the same inclusive-both-ends semantics
    that Qlib's ``fetch_df_by_index`` applies to ``segments`` entries.
    """
    if isinstance(seg_range, slice):
        start, end = seg_range.start, seg_range.stop
    elif isinstance(seg_range, (list, tuple)) and len(seg_range) == 2:
        start, end = seg_range
    else:
        start = end = seg_range

    # ``segments`` may still present a single value as a len-1 list/tuple in some
    # callers, so normalise to a scalar before constructing pandas timestamps.
    if isinstance(start, (list, tuple)):
        start = start[0] if start else None
    if isinstance(end, (list, tuple)):
        end = end[-1] if end else None

    lo = pd.Timestamp(start) if start is not None else None
    hi = pd.Timestamp(end) if end is not None else None
    return df.loc[lo:hi]


def audit_dataset_segments(handler_config: Dict[str, Any], segments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts row counts, date spans, active tickers, and daily cross-sectional breadth
    for each dataset segment.

    Loads ONLY the label expression through the handler's own data loader -- one column
    instead of 159 -- so the audit never re-materialises the feature matrix. ``DropnaLabel``
    is mirrored so ``rows`` matches the learn-side view (DK_L) the model is actually fitted
    on; the pre-drop count is reported alongside it.
    """
    segment_stats: Dict[str, Any] = {}

    if not segments:
        logger.warning("No dataset segments configured; skipping dimension audit.")
        return segment_stats
    if not handler_config:
        logger.warning("No handler config available; skipping dimension audit.")
        return {seg: _empty_segment_stats("handler config missing") for seg in segments}

    try:
        loader, (label_exprs, label_names), scope = _open_label_only_loader(handler_config)
        logger.info(f"Auditing segments via label-only projection: {list(label_exprs)}")
        label_df = loader.load_group_df(
            scope["instruments"],
            list(label_exprs),
            list(label_names),
            scope["start_time"],
            scope["end_time"],
            gp_name="label",
        )
        # load_group_df honours swap_level; normalise so datetime is always level 0
        label_df = convert_index_format(label_df, level="datetime").sort_index()
    except Exception as e:
        logger.warning(f"Label-only dataset audit unavailable: {type(e).__name__}: {e}")
        return {seg: _empty_segment_stats(f"{type(e).__name__}: {e}") for seg in segments}

    label_cols = [c for c in list(label_names) if c in label_df.columns]

    for seg_name, seg_range in segments.items():
        try:
            # Type narrowing: ensure DataFrame for downstream helpers
            if isinstance(label_df, pd.Series):
                label_df = label_df.to_frame(name=label_df.name or "label")
            else:
                label_df = cast(pd.DataFrame, label_df)

            seg_raw = _slice_segment(label_df, seg_range)
            seg_df = seg_raw.dropna(subset=label_cols) if label_cols else seg_raw

            if seg_df.empty:
                logger.warning(
                    f"Segment '{seg_name}' ({seg_range}) resolved to 0 usable rows "
                    f"({len(seg_raw)} rows before label dropna)."
                )
                stats = _empty_segment_stats()
                stats["rows_before_label_dropna"] = int(len(seg_raw))
                stats["label_nan_dropped"] = int(len(seg_raw))
                segment_stats[seg_name] = stats
                continue

            dt_idx = seg_df.index.get_level_values(0)
            inst_idx = seg_df.index.get_level_values(1)

            daily_counts = seg_df.groupby(level=0).size()
            unique_insts = sorted(set(inst_idx.unique()))

            segment_stats[seg_name] = {
                "rows": int(len(seg_df)),
                "rows_before_label_dropna": int(len(seg_raw)),
                "label_nan_dropped": int(len(seg_raw) - len(seg_df)),
                "start_date": dt_idx.min().strftime("%Y-%m-%d"),
                "end_date": dt_idx.max().strftime("%Y-%m-%d"),
                "trading_days": int(len(daily_counts)),
                "active_ticker_count": len(unique_insts),
                "active_tickers": unique_insts,
                "daily_breadth_min": int(daily_counts.min()),
                "daily_breadth_max": int(daily_counts.max()),
                "daily_breadth_mean": round(float(daily_counts.mean()), 1),
                "daily_breadth_median": round(float(daily_counts.median()), 1),
            }
        except Exception as e:
            logger.warning(f"Could not inspect segment {seg_name}: {type(e).__name__}: {e}")
            segment_stats[seg_name] = _empty_segment_stats(f"{type(e).__name__}: {e}")

    return segment_stats


def calculate_ic_metrics(pred_df: pd.DataFrame, label_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate daily Information Coefficient (IC), Rank IC, Daily ICIR,
    and Annualized ICIR (x sqrt(252)) between predicted score and actual forward return.

    This is an INFORMATIONAL/reporting computation only (feeds the console banner and
    ``_meta.json``'s "metrics" block) -- it is a separate, hand-rolled computation from
    Qlib's own ``SigAnaRecord``/``calc_ic`` mechanism. Per the implementation plan
    (Part 1.4.2 / 3.2.4), the actual quality-gate decision is fed exclusively by
    ``qlib.contrib.validation.score_quality.extract_ic_metrics(recorder)`` (Qlib's own
    logged ``IC``/``Rank IC``), so two disagreeing IC computations never coexist inside
    the gate. This function is retained only because it reports additional detail
    (annualized ICIR variants, daily observation count) SigAnaRecord does not log.

    NOTE (3.3 P1 fail-closed fix): this used to return an all-zero dict on both
    exception and empty-merge, making "the computation broke" indistinguishable from
    "IC is genuinely ~0". It now raises in both cases -- callers must not swallow this.
    """
    if pred_df is None or label_df is None:
        raise ValueError("calculate_ic_metrics: pred_df and label_df must both be provided (got a None).")

    # Align prediction and label on (datetime, instrument)
    merged = pd.concat([pred_df.rename(columns={pred_df.columns[0]: "pred"}),
                        label_df.rename(columns={label_df.columns[0]: "label"})], axis=1).dropna()
    if merged.empty:
        raise ValueError(
            "calculate_ic_metrics: prediction/label merge produced 0 rows after dropna() -- "
            "cannot compute IC metrics from empty data. This must not be silently reported as IC=0."
        )

    daily_ics = []
    daily_rank_ics = []

    for date, group in merged.groupby(level=0):
        if len(group) >= 3:
            ic = group["pred"].corr(group["label"], method="pearson")
            rank_ic = group["pred"].corr(group["label"], method="spearman")
            if not np.isnan(ic):
                daily_ics.append(ic)
            if not np.isnan(rank_ic):
                daily_rank_ics.append(rank_ic)

    if not daily_ics and not daily_rank_ics:
        raise ValueError(
            "calculate_ic_metrics: no date had >= 3 valid observations to correlate -- "
            "cannot compute IC metrics. This must not be silently reported as IC=0."
        )

    ic_mean = float(np.mean(daily_ics)) if daily_ics else 0.0
    ic_std = float(np.std(daily_ics)) if daily_ics else 1.0
    rank_ic_mean = float(np.mean(daily_rank_ics)) if daily_rank_ics else 0.0
    rank_ic_std = float(np.std(daily_rank_ics)) if daily_rank_ics else 1.0

    icir = float(ic_mean / (ic_std + 1e-12))
    rank_icir = float(rank_ic_mean / (rank_ic_std + 1e-12))
    annualized_icir = float(icir * (252.0 ** 0.5))
    annualized_rank_icir = float(rank_icir * (252.0 ** 0.5))

    return {
        "mean_ic": round(ic_mean, 5),
        "rank_ic": round(rank_ic_mean, 5),
        "icir": round(icir, 4),
        "annualized_icir": round(annualized_icir, 4),
        "rank_icir": round(rank_icir, 4),
        "annualized_rank_icir": round(annualized_rank_icir, 4),
        "daily_observations": len(daily_ics),
    }


def resolve_factor_attribution(raw_feature_names: List[str], importances: np.ndarray) -> List[Dict[str, Any]]:
    """
    Pairs raw booster feature names (Column_i or factor names) with canonical Alpha158
    mathematical formulas and institutional financial semantic descriptions.
    """
    from qlib.contrib.data.loader import Alpha158DL
    fields, canonical_names = Alpha158DL.get_feature_config()
    formula_map = dict(zip(canonical_names, fields))

    resolved = []
    for fn, gain in zip(raw_feature_names, importances):
        canonical_name = str(fn)
        if canonical_name.startswith("Column_"):
            try:
                col_idx = int(canonical_name.split("_")[1])
                if col_idx < len(canonical_names):
                    canonical_name = canonical_names[col_idx]
            except (ValueError, IndexError):
                pass

        formula = formula_map.get(canonical_name, "Expression defined in Alpha158DL")
        
        # Resolve financial name and description
        factor_title = "Alpha158 Technical Factor"
        description = "Quantitative statistical factor."

        if canonical_name in ALPHA158_FACTOR_ONTOLOGY:
            factor_title = ALPHA158_FACTOR_ONTOLOGY[canonical_name]["name"]
            description = ALPHA158_FACTOR_ONTOLOGY[canonical_name]["description"]
        else:
            # Prefix matching for parametrized rolling features (e.g. CORD20, ROC60, RSQR5)
            for prefix, meta in ALPHA158_FACTOR_ONTOLOGY.items():
                if canonical_name.startswith(prefix):
                    suffix = canonical_name[len(prefix):]
                    w = suffix if suffix.isdigit() else ""
                    factor_title = meta["name"].replace("{w}", w)
                    description = meta["description"].replace("{w}", w)
                    break

        resolved.append({
            "feature": canonical_name,
            "gain": float(gain),
            "name": factor_title,
            "formula": formula,
            "description": description,
        })

    resolved.sort(key=lambda x: x["gain"], reverse=True)
    return resolved


def print_institutional_summary_banner(
    market: str,
    universe_audit: Dict[str, Any],
    segment_stats: Dict[str, Any],
    model_stats: Dict[str, Any],
    ic_metrics: Dict[str, Any],
    top_features: List[Dict[str, Any]],
    artifact_paths: Dict[str, Path],
) -> None:
    """Renders an institutional-grade summary display conforming to Hedge Fund & Prop Trader standards."""
    b = "=" * 94
    sep = "-" * 94
    print("\n" + b)
    print(f"{'MICROSOFT QLIB LIGHTGBM ALPHA158 INSTITUTIONAL TRAINING & AUDIT REPORT':^94}")
    print(b)

    # 1. Universe Accounting
    print("1. UNIVERSE DISCOVERY & TICKER ACCOUNTING")
    print(f"   Targeted Instruments (Config):     {universe_audit.get('targeted_total', 0):>6}")
    pct_valid = (universe_audit.get('valid_count', 0) / universe_audit.get('targeted_total', 1) * 100.0) if universe_audit.get('targeted_total', 0) > 0 else 0.0
    print(f"   Valid Binary Datasets on Disk:     {universe_audit.get('valid_count', 0):>6}  ({pct_valid:.1f}%)")
    print(f"   Skipped / Missing Feature Folders: {universe_audit.get('missing_count', 0):>6}")
    print(f"   Corrupted / Empty Binaries:        {universe_audit.get('corrupt_count', 0):>6}")
    print(f"   Pre-Start Delisted Instruments:    {universe_audit.get('delisted_count', 0):>6}")

    # 2. Dataset Dimensions & Cross-Sectional Breadth
    print(sep)
    print("2. DATASET DIMENSIONS & CROSS-SECTIONAL BREADTH")
    print(f"   {'Segment':<8} | {'Rows':<9} | {'Date Span':<23} | {'Days':<5} | {'Active Tickers':<14} | {'Daily Breadth (Min / Avg / Max)'}")
    print(f"   {'-'*8}-|-{'-'*9}-|-{'-'*23}-|-{'-'*5}-|-{'-'*14}-|-{'-'*29}")

    for seg_k in ["TRAIN", "VALID", "TEST"]:
        s = segment_stats.get(seg_k.lower(), {})
        if s and s.get("rows", 0) > 0:
            breadth = f"{s.get('daily_breadth_min', 0):>3} / {s.get('daily_breadth_mean', 0.0):>5.1f} / {s.get('daily_breadth_max', 0):>3}"
            date_span = f"{s.get('start_date', 'N/A')} -> {s.get('end_date', 'N/A')}"
            print(f"   {seg_k:<8} | {s.get('rows', 0):<9,d} | {date_span:<23} | {s.get('trading_days', 0):<5} | {s.get('active_ticker_count', 0):<14} | {breadth}")
        else:
            print(f"   {seg_k:<8} | {'N/A':<9} | {'N/A':<23} | {'N/A':<5} | {'N/A':<14} | {'N/A'}")

    # 3. Model Architecture & Splitting
    print(sep)
    print("3. MODEL TOPOLOGY & BOOSTING TREE METRICS")
    print(f"   LightGBM Trees Split:              {model_stats.get('num_trees', 'N/A'):>6}")
    print(f"   Total Alpha158 Features:           {model_stats.get('features_count', 158):>6}")
    print(f"   Max Tree Depth / Num Leaves:       {model_stats.get('max_depth', 6):>2} / {model_stats.get('num_leaves', 31):>2}")
    print(f"   Subsample / Colsample by Tree:     {model_stats.get('subsample', 0.88):.2f} / {model_stats.get('colsample_bytree', 0.89):.2f}")

    # 4. Out-of-Sample Performance Metrics
    print(sep)
    print("4. OUT-OF-SAMPLE TEST METRICS (TEST SEGMENT)")
    print(f"   Mean Information Coefficient (IC): {ic_metrics.get('mean_ic', 0.0):>+8.5f}")
    print(f"   Rank Information Coefficient:      {ic_metrics.get('rank_ic', 0.0):>+8.5f}")
    print(f"   Daily ICIR (Unannualized):         {ic_metrics.get('icir', 0.0):>+8.4f}")
    print(f"   Annualized ICIR (x sqrt(252)):     {ic_metrics.get('annualized_icir', 0.0):>+8.4f}")
    print(f"   Annualized Rank ICIR:              {ic_metrics.get('annualized_rank_icir', 0.0):>+8.4f}")
    print(f"   Evaluated Daily Cross-Sections:    {ic_metrics.get('daily_observations', 0):>8d} trading days")

    # 5. Top Factor Attribution
    print(sep)
    print("5. TOP 5 ALPHA ATTRIBUTION FACTORS (ECONOMIC & MATHEMATICAL BREAKDOWN)")
    for i, f in enumerate(top_features[:5], 1):
        print(f"   #{i} {f['feature']:<8} [Gain: {f['gain']:11.1f}] - {f['name']}")
        print(f"       Formula: {f['formula']}")
        print(f"       Logic:   {f['description']}")

    # 6. Serialized Production Artifacts
    print(sep)
    print("6. PERSISTED PRODUCTION ARTIFACTS")
    for k, p in artifact_paths.items():
        print(f"   • {k:<26}: {p}")
    print(b + "\n")


def _atomic_write_via(dest: Path, write_fn) -> None:
    """
    Write an artifact atomically: ``write_fn(tmp_path)`` performs the actual
    (library-specific) write to a staging temp path in the SAME directory as
    ``dest``, then this helper renames it into place with ``os.replace``
    (atomic on both POSIX and Windows, and overwrites an existing ``dest``).

    This is the mechanism behind the implementation plan's P0 "atomic staged
    promotion" fix (3.3): every production artifact write goes through this
    function so a crash or exception mid-write can never leave ``dest``
    partially written, and the promotion-gate call site in
    ``train_alpha158_model`` never reaches any of these writes unless
    ``validate_score_quality`` has already returned successfully.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f".{dest.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    try:
        write_fn(tmp)
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _pickle_writer(obj: Any):
    """Return a `write_fn(tmp_path)` for `_atomic_write_via` that pickles `obj`, closing the handle."""
    def _write(tmp: Path) -> None:
        with open(tmp, "wb") as f:
            pickle.dump(obj, f)
    return _write


def _text_writer(text: str, encoding: str = "utf-8"):
    """Return a `write_fn(tmp_path)` for `_atomic_write_via` that writes `text`, closing the handle."""
    def _write(tmp: Path) -> None:
        with open(tmp, "w", encoding=encoding) as f:
            f.write(text)
    return _write


def _build_scores_df(pred_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Build the (date, symbol, score, rank, percentile) cross-sectional score
    table from a recorder's ``pred.pkl``.

    NOTE (3.3 P1 fail-closed fix): this used to be wrapped in a
    try/except that logged a warning and left ``scores_exported=False``
    silently dropped from the run summary on any failure. It now raises --
    an export failure must stop the run (and, downstream, block promotion),
    never be swallowed into "the run reported success but the score file is
    stale/missing".
    """
    if pred_df is None:
        raise ValueError("_build_scores_df: pred_df is None -- SignalRecord did not produce a usable pred.pkl.")

    scores_df = pred_df.copy()
    if isinstance(scores_df, pd.Series):
        scores_df = scores_df.to_frame("score")
    elif "score" not in scores_df.columns:
        scores_df.columns = ["score"]

    scores_df = scores_df.reset_index()
    if "datetime" in scores_df.columns:
        scores_df.rename(columns={"datetime": "date"}, inplace=True)
    if "instrument" in scores_df.columns:
        scores_df.rename(columns={"instrument": "symbol"}, inplace=True)

    if scores_df.empty:
        raise ValueError("_build_scores_df: prediction table is empty -- refusing to export an empty score file.")
    if "date" not in scores_df.columns or "symbol" not in scores_df.columns:
        raise ValueError(f"_build_scores_df: expected 'date'/'symbol' columns after reset_index(), got {list(scores_df.columns)}.")

    # Compute cross-sectional ranks and percentiles per date.
    #
    # NOTE on `method`: this model's score distribution has, on two separate
    # occasions (see .team-code/20260905-finance_team_review_alpha158_degenerate_score.md
    # and .team-code/20260908-alpha158_training_audit_and_score_degeneracy_implementation_plan.md),
    # been degenerate -- e.g. only 232 distinct scores across 908 names, with
    # ties up to 120-wide, or 7 distinct scores across 903 names. `method="dense"`
    # ranks *distinct values* (1, 2, 3, ... with no gaps for ties), so under this
    # much degeneracy it stops meaning "Nth best of the universe" at all -- it
    # becomes "Nth distinct score value", compressing the reported rank far below
    # where `percentile` (which correctly divides by the full universe size) puts
    # the same row. `method="min"` (standard competition ranking: a tied group all
    # takes the best rank in the group, next distinct value resumes at
    # group_size + previous_rank) keeps `rank` consistent with `percentile`
    # regardless of tie width, and rank 1 still means "the single best score" the
    # way a human expects. The quality gate's distinctness check (3.2.1) is what
    # actually prevents a degenerate distribution like this from being promoted at
    # all -- this ranking-method choice is a defense-in-depth consistency
    # property, not a substitute for that gate.
    scores_df["rank"] = scores_df.groupby("date")["score"].rank(ascending=False, method="min").astype(int)
    scores_df["percentile"] = (
        scores_df.groupby("date")["score"].rank(pct=True, ascending=True) * 100.0
    ).round(2)
    return scores_df


def _get_git_sha() -> str:
    """Best-effort git SHA for provenance stamping (3.3 P2). Never raises the training run."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL
        )
        return sha.decode("utf-8").strip()
    except Exception as e:
        logger.warning(f"Could not resolve git SHA for provenance stamping: {e}")
        return "unknown"


def _hash_config(config: Dict[str, Any]) -> str:
    """Stable sha256 hash of the fully-resolved task config, for provenance stamping (3.3 P2)."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _resolve_output_dirs(
    override_active: bool,
    model_output_dir: Optional[Path],
    scores_output_dir: Optional[Path],
) -> Tuple[Path, Path]:
    """
    Resolve the model/score output directories for this run.

    P0 root-cause fix (implementation plan 3.3 / 3.7 step 3): when a
    hyperparameter override (``--num_boost_round`` today) is active, this
    function UNCONDITIONALLY forces scratch output paths -- it deliberately
    ignores any caller-supplied ``model_output_dir``/``scores_output_dir`` --
    so no flag combination can ever redirect a "quick test" run's output back
    onto the production artifact paths. This is what makes the override
    "structurally incapable" of writing to production, not merely
    discouraged from it.
    """
    if override_active:
        logger.warning(
            "A hyperparameter override (--num_boost_round) is active. Per the implementation "
            "plan's P0 root-cause fix, this run's artifacts are FORCED to scratch paths "
            f"({SMOKETEST_MODEL_DIR}, {SMOKETEST_SCORES_DIR}) regardless of any other argument; "
            "production artifacts will NOT be touched by this run."
        )
        resolved_model_dir = SMOKETEST_MODEL_DIR
        resolved_scores_dir = SMOKETEST_SCORES_DIR
    else:
        resolved_model_dir = model_output_dir if model_output_dir is not None else PRODUCTION_MODEL_DIR
        resolved_scores_dir = scores_output_dir if scores_output_dir is not None else PRODUCTION_SCORES_DIR

    # Defensive, belt-and-braces assertion: an active override must never resolve
    # to the real production paths, no matter how this function is refactored later.
    assert not override_active or resolved_model_dir != PRODUCTION_MODEL_DIR, "override must not resolve to production model dir"
    assert not override_active or resolved_scores_dir != PRODUCTION_SCORES_DIR, "override must not resolve to production scores dir"
    return resolved_model_dir, resolved_scores_dir


def train_alpha158_model(
    config_path: Path,
    qlib_dir: Path,
    market: str = "russell1000",
    experiment_name: str = "lightgbm_alpha158_us_russell1000",
    model_output_dir: Optional[Path] = None,
    scores_output_dir: Optional[Path] = None,
    num_boost_round: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Execute full LightGBM Alpha158 model training, pre-flight auditing,
    metric evaluation, and artifact serialization.
    """
    start_time = time.time()
    # Configure MLflow execution flags locally
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

    logger.info(f"Loading configuration from: {config_path.resolve()}")
    yaml = YAML(typ="safe", pure=True)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f)

    # Override qlib_init if needed
    qlib_dir_str = str(qlib_dir.expanduser().resolve())
    config["qlib_init"]["provider_uri"] = qlib_dir_str
    config["qlib_init"]["region"] = "us"
    config["market"] = market

    # 1. Ensure universe file exists and perform Pre-Flight Universe Storage Audit
    ensure_universe_file(qlib_dir, market)
    start_date = config.get("data_handler_config", {}).get("start_time", "2020-01-01")
    universe_audit = audit_universe_and_features(qlib_dir, market, start_date=str(start_date))

    # 2. Initialize Qlib
    logger.info(f"Initializing Qlib (provider_uri={qlib_dir_str}, region=us)...")
    exp_uri = "file:" + str(REPO_ROOT / "mlruns")
    exp_manager = C["exp_manager"]
    exp_manager["kwargs"]["uri"] = exp_uri
    qlib.init(**config.get("qlib_init"), exp_manager=exp_manager)

    # Fast run override if requested. `override_active` gates BOTH the effective
    # num_boost_round used for training AND (via `_resolve_output_dirs`) whether
    # this run is even allowed to consider writing to the production artifact
    # paths at all -- see 3.3 P0 / 3.7 step 3 of the implementation plan.
    override_active = num_boost_round is not None
    if override_active:
        config["task"]["model"]["kwargs"]["num_boost_round"] = num_boost_round
    num_boost_round_effective = config["task"]["model"]["kwargs"].get("num_boost_round", 1000)

    resolved_model_dir, resolved_scores_dir = _resolve_output_dirs(override_active, model_output_dir, scores_output_dir)
    resolved_model_dir.mkdir(parents=True, exist_ok=True)
    resolved_scores_dir.mkdir(parents=True, exist_ok=True)

    # 3. Execute training via Qlib's task_train. This already runs the full
    #    `task.record` pipeline declared in the workflow YAML (SignalRecord ->
    #    SigAnaRecord -> PortAnaRecord), so IC/Rank IC and the portfolio
    #    backtest's information ratio / annualized return / max drawdown / fill
    #    rate are ALL already computed and logged to `recorder` by the time this
    #    call returns -- no separate computation or wiring is needed for the
    #    quality gate below.
    logger.info(f"Starting Qlib model training (experiment={experiment_name})...")
    recorder = task_train(config.get("task"), experiment_name=experiment_name)
    recorder.save_objects(config=config)

    # 4. Low-memory dataset dimension accounting.
    #    NOTE: the dataset saved to the recorder is dumped with `dump_all=False` (qlib/model/trainer.py),
    #    so its `_data`/`_infer`/`_learn` frames are stripped from the pickle and `prepare()` would raise.
    #    Audit from the config via a label-only projection instead of reloading/rebuilding the handler.
    dataset_kwargs = config.get("task", {}).get("dataset", {}).get("kwargs", {})
    segments_config = dataset_kwargs.get("segments", {})
    handler_config = dataset_kwargs.get("handler", {})
    segment_stats = audit_dataset_segments(handler_config, segments_config)

    # 5. Retrieve trained model and test predictions, and compute the COMPLETE
    #    evidence set (model + scores + all metrics) BEFORE anything is written
    #    to disk -- this is the 3.3 P0 "atomic staged promotion" restructure:
    #    the gate below runs exactly once against this complete set, and only
    #    on success does the write phase (Step 6+) begin.
    trained_model = recorder.load_object("params.pkl")
    pred_df = recorder.load_object("pred.pkl")
    try:
        label_df = recorder.load_object("label.pkl")
    except Exception:
        label_df = None

    # Informational IC/RankIC (banner + meta.json "metrics" block only). Kept
    # separate from the gate's IC input per Part 1.4.2 / 3.2.4's consolidation:
    # the gate is fed exclusively by Qlib's own SigAnaRecord-logged IC/Rank IC
    # (via `extract_ic_metrics`, below), never by this hand-rolled duplicate.
    ic_metrics_banner: Dict[str, Any] = {}
    if pred_df is not None and label_df is not None:
        ic_metrics_banner = calculate_ic_metrics(pred_df, label_df)
        logger.info(f"Validation IC Metrics (informational): {ic_metrics_banner}")
    else:
        logger.warning("pred.pkl/label.pkl unavailable -- informational IC banner metrics will show as empty.")

    scores_df = _build_scores_df(pred_df)

    ic_metrics_gate = extract_ic_metrics(recorder)
    portfolio_metrics_gate = extract_portfolio_metrics(recorder)
    logger.info(f"Gate input -- IC/Rank IC (from SigAnaRecord): {ic_metrics_gate}")
    logger.info(f"Gate input -- portfolio metrics (from PortAnaRecord): {portfolio_metrics_gate}")

    # Feature attribution / num_trees are read directly off the in-memory
    # booster -- NOT via `save_model()` to a real path -- so nothing is written
    # to disk before the gate has had a chance to run.
    feature_importances: List[Dict[str, Any]] = []
    num_trees = 0
    booster = getattr(trained_model, "model", None)
    if booster is not None:
        raw_feature_names = booster.feature_name()
        importances = booster.feature_importance(importance_type="gain")
        num_trees = booster.num_trees()
        feature_importances = resolve_factor_attribution(raw_feature_names, importances)
        logger.info(f"Model Quality Check: num_trees={num_trees}")
    else:
        logger.warning("Trained model has no native LightGBM booster -- num_trees/feature attribution unavailable.")

    # check 3.2.5 is sequenced last, per the implementation plan: it stays
    # dormant (pinned_reference=None) until a clean run has been produced and
    # explicitly blessed via `bless_pinned_reference` (see below main()).
    pinned_reference_df: Optional[pd.DataFrame] = None
    if PINNED_REFERENCE_PATH.exists():
        try:
            pinned_reference_df = pd.read_parquet(PINNED_REFERENCE_PATH)
            logger.info(f"Loaded pinned reference for check 3.2.5: {PINNED_REFERENCE_PATH}")
        except Exception as e:
            logger.warning(f"Could not load pinned reference at {PINNED_REFERENCE_PATH}: {e}. Check 3.2.5 will be skipped.")

    as_of_date = datetime.datetime.utcnow().date()

    # 6. Run the quality gate EXACTLY ONCE against the complete evidence set.
    #
    # `enforce_gate` is False only for an active hyperparameter-override
    # ("quick test") run: such a run is, by construction (`_resolve_output_dirs`
    # above), already incapable of reaching the production artifact paths, so
    # blocking it on the gate too would defeat its purpose as a fast dev-loop
    # tool (a deliberately tiny num_boost_round is EXPECTED to fail the
    # num_trees floor). The gate still runs and its result is still logged
    # loudly for a smoke-test run -- it just doesn't raise.
    enforce_gate = not override_active
    gate_report = None
    gate_exception: Optional[ScoreQualityGateError] = None
    try:
        gate_report = validate_score_quality(
            scores_df,
            ic_metrics_gate,
            portfolio_metrics_gate,
            trained_model,
            num_boost_round_cap=num_boost_round_effective,
            as_of_date=as_of_date,
            pinned_reference=pinned_reference_df,
        )
        for w in gate_report.warnings:
            logger.warning(f"Quality gate warning: {w}")
        logger.info("Quality gate PASSED.")
    except ScoreQualityGateError as e:
        gate_exception = e
        for f in e.report.failures:
            logger.error(f"Quality gate FAILURE: {f}")
        if enforce_gate:
            logger.error(
                "Quality gate FAILED -- refusing to promote this run. No production artifact "
                "(models/lightgbm/alpha158_russell1000_latest.* or "
                "output/scores/alpha158_russell1000_latest.*) has been touched."
            )
            raise
        else:
            logger.warning(
                "Quality gate FAILED but this is a hyperparameter-override (smoke-test) run -- "
                "writing to scratch paths only, as intended for a quick dev-loop iteration."
            )
        gate_report = gate_exception.report

    # ------------------------------------------------------------------
    # WRITE PHASE -- reached only if the gate passed (or this is a
    # non-enforced override/smoke-test run). Every write below goes through
    # `_atomic_write_via`: staged to a temp path in the destination directory,
    # then atomically renamed into place (3.3 P0).
    # ------------------------------------------------------------------
    date_tag = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    prod_model_pkl = resolved_model_dir / "alpha158_russell1000_latest.pkl"
    _atomic_write_via(prod_model_pkl, _pickle_writer(trained_model))
    logger.info(f"Saved production model pickle to: {prod_model_pkl.resolve()}")

    prod_model_txt = resolved_model_dir / "alpha158_russell1000_latest.txt"
    if booster is not None:
        _atomic_write_via(prod_model_txt, lambda tmp: booster.save_model(str(tmp)))
        logger.info(f"Saved native LightGBM booster text to: {prod_model_txt.resolve()}")

    # Versioned model checkpoint (pre-existing pattern).
    checkpoint_dir = resolved_model_dir / "checkpoints"
    checkpoint_pkl = checkpoint_dir / f"alpha158_russell1000_{date_tag}.pkl"
    _atomic_write_via(checkpoint_pkl, _pickle_writer(trained_model))

    # Versioned score-file checkpoint (NEW -- 3.3 P0 / 3.7 step 2 item 5:
    # the .pkl already had a rollback path via `checkpoints/`; the score file
    # -- the artifact type BOTH real incidents actually corrupted -- did not).
    scores_checkpoint_dir = resolved_scores_dir / "checkpoints"
    scores_checkpoint_parquet = scores_checkpoint_dir / f"alpha158_russell1000_scores_{date_tag}.parquet"
    _atomic_write_via(scores_checkpoint_parquet, lambda tmp: scores_df.to_parquet(tmp, index=False))

    parquet_path = resolved_scores_dir / "alpha158_russell1000_latest.parquet"
    csv_path = resolved_scores_dir / "alpha158_russell1000_latest.csv"
    _atomic_write_via(parquet_path, lambda tmp: scores_df.to_parquet(tmp, index=False))
    _atomic_write_via(csv_path, lambda tmp: scores_df.to_csv(tmp, index=False))
    logger.info(f"Exported {len(scores_df)} scores to {parquet_path.resolve()} and {csv_path.resolve()}")

    # Provenance stamping (3.3 P2): git SHA, config hash, and seed alongside
    # the hyperparameters already captured -- without this, "was this artifact
    # produced by the blessed config?" is unanswerable after the fact, and
    # check 3.2.5's pinned reference has nothing durable to pin to.
    provenance = {
        "git_sha": _get_git_sha(),
        "config_hash_sha256": _hash_config(config),
        "seed": config["task"]["model"]["kwargs"].get("seed", "not_set (lightgbm/qlib default)"),
    }

    meta_data = {
        "model_name": "LightGBM_Alpha158_Russell1000",
        "trained_at_utc": datetime.datetime.utcnow().isoformat(),
        "training_duration_seconds": round(time.time() - start_time, 2),
        "experiment_name": experiment_name,
        "recorder_id": recorder.id,
        "market": market,
        "provider_uri": qlib_dir_str,
        "provenance": provenance,
        "override_active": override_active,
        "universe_audit": {
            "targeted_total": universe_audit.get("targeted_total", 0),
            "valid_count": universe_audit.get("valid_count", 0),
            "missing_count": universe_audit.get("missing_count", 0),
            "corrupt_count": universe_audit.get("corrupt_count", 0),
            "delisted_count": universe_audit.get("delisted_count", 0),
        },
        "segment_dimensions": {
            k: {
                "rows": v.get("rows", 0),
                "date_span": f"{v.get('start_date', 'N/A')} -> {v.get('end_date', 'N/A')}",
                "trading_days": v.get("trading_days", 0),
                "active_tickers": v.get("active_ticker_count", 0),
                "daily_breadth_mean": v.get("daily_breadth_mean", 0.0),
            }
            for k, v in segment_stats.items()
        },
        "hyperparameters": config["task"]["model"]["kwargs"],
        "metrics": ic_metrics_banner,
        "gate_metrics": {"ic": ic_metrics_gate, "portfolio": portfolio_metrics_gate},
        "quality_gate": gate_report.to_dict() if gate_report is not None else None,
        "features_count": len(feature_importances),
        "num_trees": num_trees,
        "top_10_features": feature_importances[:10] if feature_importances else [],
    }
    prod_model_meta = resolved_model_dir / "alpha158_russell1000_latest_meta.json"
    _atomic_write_via(prod_model_meta, _text_writer(json.dumps(meta_data, indent=4, default=str)))
    logger.info(f"Saved model metadata to: {prod_model_meta.resolve()}")

    # 7. Render Institutional Summary Banner
    artifact_paths = {
        "Production Model (.pkl)": prod_model_pkl,
        "Booster Text (.txt)": prod_model_txt,
        "Model Metadata (.json)": prod_model_meta,
        "Latest Scores (.parquet)": parquet_path,
        "Latest Scores (.csv)": csv_path,
    }

    print_institutional_summary_banner(
        market=market,
        universe_audit=universe_audit,
        segment_stats=segment_stats,
        model_stats={
            "num_trees": num_trees,
            "features_count": len(feature_importances),
            "max_depth": config["task"]["model"]["kwargs"].get("max_depth", 6),
            "num_leaves": config["task"]["model"]["kwargs"].get("num_leaves", 31),
            "subsample": config["task"]["model"]["kwargs"].get("subsample", 0.88),
            "colsample_bytree": config["task"]["model"]["kwargs"].get("colsample_bytree", 0.89),
        },
        ic_metrics=ic_metrics_banner,
        top_features=feature_importances,
        artifact_paths=artifact_paths,
    )

    return {
        "status": "success",
        "model_path": str(prod_model_pkl),
        "metadata_path": str(prod_model_meta),
        "ic_metrics": ic_metrics_banner,
        "gate_report": gate_report.to_dict() if gate_report is not None else None,
        "gate_passed": bool(gate_report.passed) if gate_report is not None else None,
        "override_active": override_active,
        "recorder_id": recorder.id,
        "universe_audit": universe_audit,
        "segment_dimensions": segment_stats,
    }


def bless_pinned_reference(
    scores_parquet_path: Optional[Path] = None,
    meta_path: Optional[Path] = None,
) -> Path:
    """
    Explicitly bless a scores file as the immutable, pinned reference for
    quality-gate check 3.2.5 (rank-correlation-vs-pinned-reference).

    This is a DELIBERATE, manual, operator-invoked action (``--bless_pinned_reference``
    on the CLI) -- never automatic, even on a passing training run. Per the
    implementation plan's 3.2.5 "Critical correction from review": the pinned
    reference must never be implicitly equal to "whatever currently occupies
    the production path" (that was precisely how the original design would
    have anchored the check to an already-corrupt baseline). Call this only
    after independently confirming the source run passed
    ``validate_score_quality`` -- see the training run's own
    ``alpha158_russell1000_latest_meta.json["quality_gate"]["passed"]``.

    Defaults to blessing the current production
    ``output/scores/alpha158_russell1000_latest.parquet`` /
    ``models/lightgbm/alpha158_russell1000_latest_meta.json`` pair.
    """
    scores_parquet_path = scores_parquet_path or (PRODUCTION_SCORES_DIR / "alpha158_russell1000_latest.parquet")
    meta_path = meta_path or (PRODUCTION_MODEL_DIR / "alpha158_russell1000_latest_meta.json")

    if not scores_parquet_path.exists():
        raise FileNotFoundError(f"Cannot bless pinned reference: {scores_parquet_path} does not exist.")

    with open(meta_path, "r", encoding="utf-8") as f:
        source_meta = json.load(f)

    quality_gate = source_meta.get("quality_gate") or {}
    if not quality_gate.get("passed", False):
        raise ValueError(
            f"Refusing to bless {scores_parquet_path} as the pinned reference: its own metadata "
            f"({meta_path}) does not record a passing quality_gate result (quality_gate={quality_gate})."
        )

    scores_df = pd.read_parquet(scores_parquet_path)
    _atomic_write_via(PINNED_REFERENCE_PATH, lambda tmp: scores_df.to_parquet(tmp, index=False))

    blessing_meta = {
        "blessed_at_utc": datetime.datetime.utcnow().isoformat(),
        "source_scores_path": str(scores_parquet_path.resolve()),
        "source_recorder_id": source_meta.get("recorder_id"),
        "source_provenance": source_meta.get("provenance"),
        "source_quality_gate": quality_gate,
        "row_count": int(len(scores_df)),
        "date_span": [str(scores_df["date"].min()), str(scores_df["date"].max())] if "date" in scores_df.columns else None,
        "git_sha_at_blessing": _get_git_sha(),
    }
    _atomic_write_via(PINNED_REFERENCE_META_PATH, _text_writer(json.dumps(blessing_meta, indent=4, default=str)))
    logger.info(f"Blessed {scores_parquet_path} as the pinned reference -> {PINNED_REFERENCE_PATH}")
    return PINNED_REFERENCE_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train LightGBM on Alpha158 for US Russell 1000 equities."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_russell1000.yaml",
        help="Path to workflow YAML configuration.",
    )
    parser.add_argument(
        "--qlib_dir",
        type=str,
        default="~/.qlib/qlib_data/us_data",
        help="Qlib data directory path.",
    )
    parser.add_argument(
        "--market",
        type=str,
        default="russell1000",
        help="Market universe instrument file prefix.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="lightgbm_alpha158_us_russell1000",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--num_boost_round",
        type=int,
        default=None,
        help=(
            "Override number of boosting rounds for quick testing. WARNING: activating this "
            "override FORCES output to scratch paths (models/lightgbm/_smoketest/, "
            "output/scores/_smoketest/) -- no combination of arguments can redirect an override "
            "run's output back onto the production artifact paths (models/lightgbm/"
            "alpha158_russell1000_latest.* / output/scores/alpha158_russell1000_latest.*)."
        ),
    )
    parser.add_argument(
        "--bless_pinned_reference",
        action="store_true",
        help=(
            "Do not train. Instead, explicitly bless the CURRENT production scores file as the "
            "pinned, immutable reference used by quality-gate check 3.2.5 (rank-correlation-vs-"
            "pinned-reference). Refuses to run unless that production run's own metadata records "
            "a passing quality gate."
        ),
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.bless_pinned_reference:
        bless_pinned_reference()
        return

    config_p = Path(args.config)
    if not config_p.is_absolute():
        config_p = REPO_ROOT / config_p
    config_p = config_p.resolve()
    qlib_p = Path(args.qlib_dir).expanduser().resolve()

    train_alpha158_model(
        config_path=config_p,
        qlib_dir=qlib_p,
        market=args.market,
        experiment_name=args.experiment_name,
        num_boost_round=args.num_boost_round,
    )


if __name__ == "__main__":
    main()
