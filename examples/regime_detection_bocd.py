#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Bayesian Online Changepoint Detection (BOCD) & Market Regime Demonstration
==========================================================================
Demonstrates real-time, non-lagging market regime classification and changepoint
detection across financial time series using Adams & MacKay (2007) conjugate Student-t
BOCD combined with multi-horizon realized volatility surfaces and credit spreads.

Usage:
    python examples/regime_detection_bocd.py --symbol MSFT
    python examples/regime_detection_bocd.py --symbol NVDA --data_dir D:\trading\qlib
"""

import os
import sys
import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure qlib/contrib and scripts directories are accessible
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
CONTRIB_DIR = REPO_ROOT / "qlib" / "contrib"
SCRIPTS_DIR = REPO_ROOT / "scripts"

for p in [str(REPO_ROOT), str(CONTRIB_DIR), str(SCRIPTS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from qlib.contrib.regime import MarketRegimeClassifier
from stock_analysis_engine import load_stock_data


def run_regime_demo(
    symbol: str = "MSFT",
    data_dir: str = "~/.qlib/qlib_data/us_data",
    auto_download: bool = True,
    start: str = "2018-01-01",
    changepoint_threshold: float = 0.35,
):
    print("=" * 70)
    print("  BAYESIAN ONLINE CHANGEPOINT DETECTION (BOCD) & REGIME ANALYZER")
    print("  Model: Adams & MacKay (2007) Conjugate Student-t Prior")
    print("  Drivers: Multi-Horizon Vol Surface (5d/21d/63d) + Credit Spreads (HYG/IEI)")
    print("=" * 70)
    print(f"Symbol:           {symbol.upper()}")
    print(f"Data Directory:   {data_dir}")
    print(f"Historical Start: {start}")
    print(f"CP Threshold:     {changepoint_threshold:.2f} (P(r <= 1))\n")

    # 1. Load market data
    df = load_stock_data(
        symbol=symbol,
        data_dir=data_dir,
        auto_download=auto_download,
        start=start,
    )
    print(f"Loaded {len(df)} historical trading days for {symbol.upper()} from {df['date'].iloc[0]} to {df['date'].iloc[-1]}.")

    # 2. Check for credit ETFs in data directory
    hyg_df = None
    iei_df = None
    root_p = Path(data_dir).expanduser().resolve()
    for cand in [root_p / "HYG.csv", root_p / "source" / "HYG.csv", root_p / "normalize" / "HYG.csv"]:
        if cand.exists():
            print(f"Found High Yield credit proxy: {cand}")
            hyg_df = pd.read_csv(cand)
            break
    for cand in [root_p / "IEI.csv", root_p / "source" / "IEI.csv", root_p / "normalize" / "IEI.csv"]:
        if cand.exists():
            print(f"Found Treasury benchmark proxy: {cand}")
            iei_df = pd.read_csv(cand)
            break

    # 3. Execute Regime Classification & BOCD
    print("\nRunning online Bayesian recursive inference...")
    classifier = MarketRegimeClassifier(expected_run_length=63.0, bocd_threshold=changepoint_threshold)
    df_regime = classifier.analyze(df, hyg_df=hyg_df, iei_df=iei_df)
    summary = classifier.get_current_regime_summary(df_regime)

    # 4. Display Significant Historical Changepoints Detected
    print("\n----------------------------------------------------------------------")
    print(" SIGNIFICANT HISTORICAL REGIME CHANGEPOINTS DETECTED")
    print("----------------------------------------------------------------------")
    cp_events = df_regime[df_regime["changepoint_prob"] >= changepoint_threshold].copy()

    if not cp_events.empty:
        # Group closely spaced consecutive days to identify the onset of the structural shift
        cp_events["date_dt"] = pd.to_datetime(cp_events["date"])
        cp_events["day_diff"] = cp_events["date_dt"].diff().dt.days
        clusters = cp_events[(cp_events["day_diff"] > 5) | (cp_events["day_diff"].isna())].tail(10)

        print(f"{'Date':12s} | {'Close':9s} | {'CP Hazard':10s} | {'Vol Ratio':10s} | {'Regime Detected':30s}")
        print("-" * 80)
        for _, row in clusters.iterrows():
            print(
                f"{str(row['date'])[:10]:12s} | "
                f"${row['close']:8.2f} | "
                f"{row['changepoint_prob']*100:8.1f}% | "
                f"{row['vol_ratio']:8.2f}x | "
                f"State {row['regime_state']} - {row['regime_name']}"
            )
    else:
        print("No changepoints exceeded the threshold in the examined window.")

    # 5. Display Current Real-Time Regime Diagnosis
    print("\n======================================================================")
    print(" CURRENT REAL-TIME REGIME DIAGNOSIS & RISK GUIDANCE")
    print("======================================================================")
    print(f"Current State:           State {summary['state']} - {summary['name']}")
    print(f"Actionable Guidance:     {summary['action']}")
    print(f"Description:             {summary['description']}")
    print(f"Changepoint Hazard P(r): {summary['changepoint_prob_pct']}% (Active Run-Length: {summary['expected_run_length_days']} days)")
    print(f"Volatility Surface:      21d Vol: {summary['vol_21d_pct']}% | 5d Vol: {summary['vol_5d_pct']}% | Ratio: {summary['vol_ratio']}x")
    print(f"Credit Momentum:         {summary['credit_mom_pct']:+.2f}%")
    print(f"Portfolio Risk Sizing:   {summary['risk_multiplier']}x exposure ({int(summary['risk_multiplier']*100)}% standard size)")
    print(f"State Probabilities:     Bull: {summary['probabilities']['bull']*100:.1f}% | Neutral: {summary['probabilities']['neutral']*100:.1f}% | Risk-Off: {summary['probabilities']['risk_off']*100:.1f}% | Transition: {summary['probabilities']['transition']*100:.1f}%")
    print("======================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Demonstrate Bayesian Online Changepoint Detection and Market Regime Classification.")
    parser.add_argument("--symbol", "-s", type=str, default="MSFT", help="Stock ticker (e.g. MSFT, SPY, NVDA)")
    parser.add_argument("--data_dir", "-d", type=str, default="~/.qlib/qlib_data/us_data", help="Directory with CSVs or Qlib binary files")
    parser.add_argument("--start", type=str, default="2018-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--threshold", "-t", type=float, default=0.35, help="Changepoint probability threshold (0.0 to 1.0)")
    parser.add_argument("--no-auto_download", dest="auto_download", action="store_false", default=True, help="Disable auto-downloading missing data")
    args = parser.parse_args()

    run_regime_demo(
        symbol=args.symbol,
        data_dir=args.data_dir,
        auto_download=args.auto_download,
        start=args.start,
        changepoint_threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
