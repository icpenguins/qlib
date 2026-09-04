#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Institutional Microstructure: Anchored VWAP & Volume Profile (KDE) Demonstration
================================================================================
Demonstrates institutional order flow liquidity analysis using:
1. Multi-Anchor VWAP (YTD, 52W High, 52W Low) with volume-weighted dispersion bands (+/-1 sigma).
2. Continuous Volume-at-Price Profile via Gaussian Kernel Density Estimation (KDE),
   identifying Point of Control (POC), 70% Value Area (VAH/VAL), and Liquidity Voids.

Usage:
    python examples/microstructure_avwap_profile.py --symbol SMH
    python examples/microstructure_avwap_profile.py --symbol MSFT --data_dir D:\trading\qlib
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

from qlib.contrib.microstructure import (
    AnchoredVWAPCalculator,
    VolumeProfileKDE,
    compute_microstructure_features,
)
from stock_analysis_engine import load_stock_data


def run_microstructure_demo(
    symbol: str = "SMH",
    data_dir: str = "~/.qlib/qlib_data/us_data",
    auto_download: bool = True,
    start: str = "2020-01-01",
    lookback_vp: int = 63,
):
    print("=" * 75)
    print("  INSTITUTIONAL MICROSTRUCTURE: ANCHORED VWAP & VOLUME PROFILE (KDE)")
    print("  Models: Vectorized AVWAP +/-1s/2s Dispersion Bands + Gaussian Kernel Density")
    print("=" * 75)
    print(f"Symbol:               {symbol.upper()}")
    print(f"Data Directory:       {data_dir}")
    print(f"Historical Start:     {start}")
    print(f"VP Lookback Days:     {lookback_vp} (~1 quarter)\n")

    # 1. Load Market Data
    df = load_stock_data(
        symbol=symbol,
        data_dir=data_dir,
        auto_download=auto_download,
        start=start,
    )
    print(f"Loaded {len(df)} historical trading days for {symbol.upper()} from {df['date'].iloc[0]} to {df['date'].iloc[-1]}.\n")

    # 2. Compute Anchored VWAPs and Volume Profile
    enriched_df, summary = compute_microstructure_features(df)
    avwap_info = summary.get("avwap", {})
    vp_info = summary.get("volume_profile", {})

    ytd = avwap_info.get("ytd", {})
    h52 = avwap_info.get("high_52w", {})
    l52 = avwap_info.get("low_52w", {})

    current_price = float(df["close"].iloc[-1])

    # 3. Print Anchored VWAP Diagnostic
    print("---------------------------------------------------------------------------")
    print(" 1. ANCHORED VWAP (AVWAP) TRAJECTORIES & DISPERSION BANDS")
    print("---------------------------------------------------------------------------")
    print(f"Latest Close Price:   ${current_price:.2f}")

    if ytd.get("value") is not None:
        print(f"\n[YTD Anchor: {ytd.get('date')}]")
        print(f"  Anchored VWAP:      ${ytd['value']:.2f}")
        print(f"  Spread from Close:  {ytd.get('spread_pct'):+.2f}%")
        print(f"  Z-Score Deviation:  {ytd.get('zscore'):+.2f}s")
        print(f"  +/-1s Channel:      ${ytd.get('lower_1s'):.2f}  to  ${ytd.get('upper_1s'):.2f}")
        print(f"  Regime Diagnosis:   {ytd.get('regime')}")
        print(f"  Actionable Rule:    {ytd.get('action')}")

    if h52.get("value") is not None:
        print(f"\n[52-Week High Anchor: {h52.get('date')}]")
        print(f"  Anchored VWAP:      ${h52['value']:.2f} ({h52.get('spread_pct'):+.2f}% spread, {h52.get('zscore'):+.2f}s)")
        print(f"  Supply Overhead:    Institutions who bought the peak are at average price of ${h52['value']:.2f}")

    if l52.get("value") is not None:
        print(f"\n[52-Week Low Anchor: {l52.get('date')}]")
        print(f"  Anchored VWAP:      ${l52['value']:.2f} ({l52.get('spread_pct'):+.2f}% spread, {l52.get('zscore'):+.2f}s)")
        print(f"  Support Foundation: Institutions who accumulated the bottom are profitable above ${l52['value']:.2f}")

    # 4. Print Continuous Volume Profile KDE Diagnostic
    print("\n---------------------------------------------------------------------------")
    print(f" 2. CONTINUOUS VOLUME PROFILE (KDE) - TRAILING {lookback_vp} TRADING DAYS")
    print("---------------------------------------------------------------------------")
    if vp_info:
        print(f"Optimal Bandwidth h:  {vp_info.get('bandwidth_h')} (Silverman's rule)")
        print(f"Point of Control:     ${vp_info.get('poc'):.2f} (POC - Highest volume concentration)")
        print(f"Distance to POC:      {vp_info.get('dist_to_poc_pct'):+.2f}%")
        print(f"70% Value Area:       ${vp_info.get('val'):.2f} (VAL)  to  ${vp_info.get('vah'):.2f} (VAH)")
        print(f"Inside Value Area:    {'YES - In Balanced Fair-Value Zone' if vp_info.get('in_value_area') else 'NO - Outside Value Area'}")
        print(f"Liquidity Void:       {'ALERT: IN THIN BOOK / VOID' if vp_info.get('in_liquidity_void') else 'BALANCED DEPTH'}")
        print(f"Market Depth State:   {vp_info.get('void_status')}")

        if vp_info.get("hvn_levels"):
            print(f"High-Volume Nodes:    {', '.join(['$' + str(x) for x in vp_info['hvn_levels']])}")
        if vp_info.get("lvn_levels"):
            print(f"Low-Volume Nodes:     {', '.join(['$' + str(x) for x in vp_info['lvn_levels']])}")
    print("===========================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Demonstrate Institutional Anchored VWAP and Volume Profile KDE.")
    parser.add_argument("--symbol", "-s", type=str, default="SMH", help="Stock ticker (e.g. SMH, MSFT, NVDA)")
    parser.add_argument("--data_dir", "-d", type=str, default="~/.qlib/qlib_data/us_data", help="Directory with CSVs or Qlib data")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--lookback", "-l", type=int, default=63, help="Volume Profile lookback days (default 63)")
    parser.add_argument("--no-auto_download", dest="auto_download", action="store_false", default=True, help="Disable auto-downloading missing data")
    args = parser.parse_args()

    run_microstructure_demo(
        symbol=args.symbol,
        data_dir=args.data_dir,
        auto_download=args.auto_download,
        start=args.start,
        lookback_vp=args.lookback,
    )


if __name__ == "__main__":
    main()
