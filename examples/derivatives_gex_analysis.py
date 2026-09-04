#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Institutional Derivatives: Dealer Gamma Exposure (GEX) & Options Flow Analysis
================================================================================
Demonstrates institutional dealer positioning and options surface dynamics:
1. Dealer Gamma Exposure (GEX) aggregation across strike chains.
2. Gamma Flip Point (S*) root-finding separating positive gamma (stabilizer / mean-reverting)
   from negative gamma (volatility accelerant / trending) regimes.
3. Major option market structure levels: Call Gamma Wall, Put Gamma Wall, Absolute Wall, and Max Pain.
4. Volatility surface metrics: 25-Delta Risk Reversal (RR25) skew and Variance Risk Premium (VRP).

Usage:
    python examples/derivatives_gex_analysis.py --symbol SMH
    python examples/derivatives_gex_analysis.py --symbol NVDA --data_dir D:\trading\qlib
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

from qlib.contrib.derivatives import (
    DealerGammaEngine,
    OptionsDataLoader,
    SyntheticOptionSurfaceGenerator,
    VolatilitySurfaceFeatures,
    compute_dealer_gex_summary,
)
from stock_analysis_engine import load_stock_data


def run_gex_demo(
    symbol: str = "SMH",
    data_dir: str = "~/.qlib/qlib_data/us_data",
    auto_download: bool = True,
    r: float = 0.045,
):
    print("=" * 78)
    print("  INSTITUTIONAL DERIVATIVES: DEALER GAMMA EXPOSURE (GEX) & OPTIONS FLOW")
    print("  Models: Black-Scholes Dealer Net Gamma ($M/1%), Gamma Flip, & Vol Skew")
    print("=" * 78)
    print(f"Symbol:               {symbol.upper()}")
    print(f"Data Directory:       {data_dir}")
    print(f"Risk-Free Rate:       {r * 100:.2f}%\n")

    # 1. Load Market Data for Spot and Realized Volatility
    expanded_dir = Path(os.path.expanduser(data_dir))
    df = None
    spot = 200.0
    realized_vol_21d = 0.28

    if expanded_dir.exists():
        try:
            df = load_stock_data(
                symbol=symbol,
                data_dir=str(expanded_dir),
                start="2022-01-01",
                auto_download=auto_download,
            )
        except Exception as e:
            print(f"[*] Note: load_stock_data encountered: {e}. Falling back to default baseline.")

    if df is not None and not df.empty and "close" in df.columns:
        spot = float(df["close"].iloc[-1])
        log_ret = np.log(df["close"] / df["close"].shift(1))
        realized_vol_21d = float(log_ret.tail(21).std() * np.sqrt(252))
        if np.isnan(realized_vol_21d) or realized_vol_21d < 0.05:
            realized_vol_21d = 0.25
        print(f"[+] Loaded {len(df)} daily bars for {symbol.upper()}.")
    else:
        print(f"[*] Synthetic underlying baseline initialized.")

    print(f"[*] Current Spot Price (S):      ${spot:.2f}")
    print(f"[*] 21-Day Realized Vol (sigma): {realized_vol_21d * 100:.2f}%\n")

    # 2. Ingest or Generate Options Chain
    loader = OptionsDataLoader(data_dir=str(expanded_dir))
    options_df = loader.load_or_generate_chain(
        symbol=symbol,
        spot=spot,
        realized_vol_21d=realized_vol_21d,
        r=r,
    )
    print(f"[+] Active Options Chain: {len(options_df)} contracts ingested/calibrated.")

    # 3. Compute Dealer Gamma Exposure & Gamma Flip
    gex_res = compute_dealer_gex_summary(
        options_df=options_df,
        spot=spot,
        r=r,
        symbol=symbol,
    )

    # 4. Compute Volatility Surface Skew & VRP
    vol_feat = VolatilitySurfaceFeatures.compute_surface_metrics(
        options_df=options_df,
        spot=spot,
        realized_vol_21d=realized_vol_21d,
        r=r,
    )

    # 5. Display Institutional Executive Summary
    regime_str = gex_res["regime"]
    flip_price = gex_res["gamma_flip_price"]
    call_wall = gex_res["call_wall"]
    put_wall = gex_res["put_wall"]
    abs_wall = gex_res["absolute_wall"]
    max_pain = gex_res["max_pain"]
    net_gex = gex_res["net_gex_millions"]
    flip_dist = ((spot - flip_price) / spot) * 100 if flip_price > 0 else 0.0

    print("-" * 78)
    print("  INSTITUTIONAL DEALER POSITIONING REPORT")
    print("-" * 78)
    print(f"  Underlying Spot Price:        ${spot:.2f}")
    print(f"  Dealer Net GEX:               ${net_gex:+.2f}M per 1% Move")
    print(f"  Dealer Gamma Regime:          {regime_str}")
    print(f"  Gamma Flip Point (S*):        ${flip_price:.2f} (Distance: {flip_dist:+.2f}%)")
    print(f"  Call Gamma Wall:              ${call_wall:.2f} (Major Overhead Resistance)")
    print(f"  Put Gamma Wall:               ${put_wall:.2f} (Major Underlying Support)")
    print(f"  Absolute Gamma Wall:          ${abs_wall:.2f}")
    print(f"  Max Pain Strike:              ${max_pain:.2f}")
    print("-" * 78)
    print("  OPTIONS VOLATILITY SURFACE & RISK PREMIUM")
    print("-" * 78)
    print(f"  30-Day Implied Vol (ATM):     {vol_feat['iv_30d_atm'] * 100:.2f}%")
    print(f"  21-Day Realized Vol:          {realized_vol_21d * 100:.2f}%")
    print(f"  Variance Risk Premium (VRP):  {vol_feat['vrp_pct']:+.2f}%")
    print(f"  25-Delta Risk Reversal Skew:  {vol_feat['rr25_skew']:+.4f} (Put-Call Vol Spread)")
    print("-" * 78)

    # 6. Strike Gamma Profile (Top Strikes Around Spot)
    profile_list = gex_res.get("strike_profile", [])
    if profile_list:
        print("\n  STRIKE GAMMA DISTRIBUTION (Around Spot Price):")
        print(f"  {'Strike':>8} | {'Call GEX ($M)':>14} | {'Put GEX ($M)':>14} | {'Net GEX ($M)':>14} | {'Total OI':>10}")
        print("  " + "-" * 70)

        # Filter strikes within +/- 15% of spot
        near_strikes = [
            item for item in profile_list
            if spot * 0.85 <= item["strike"] <= spot * 1.15
        ]
        if not near_strikes:
            near_strikes = profile_list[:15]

        min_spot_dist = min(abs(item["strike"] - spot) for item in near_strikes)

        for item in near_strikes:
            marker = ""
            if abs(item["strike"] - spot) == min_spot_dist:
                marker = " <- SPOT"
            elif item["strike"] == call_wall:
                marker = " <- CALL WALL"
            elif item["strike"] == put_wall:
                marker = " <- PUT WALL"

            print(
                f"  {item['strike']:>8.2f} | "
                f"{item['call_gex_m']:>14.2f} | "
                f"{item['put_gex_m']:>14.2f} | "
                f"{item['net_gex_m']:>+14.2f} | "
                f"{int(item['open_interest']):>10,}{marker}"
            )
        print("  " + "-" * 70)

    # 7. Strategic Trading & Risk Mandate Insights
    print("\n  TACTICAL TRADING & RISK MANAGEMENT IMPLICATIONS:")
    if gex_res["regime_state"] >= 0:
        print("  [+] REGIME: POSITIVE GAMMA (Dealer Long Gamma).")
        print("      * Dealers counter-trade the market (sell rallies, buy dips) to maintain delta-neutrality.")
        print(f"      * Volatility is heavily dampened between Put Wall (${put_wall:.2f}) and Call Wall (${call_wall:.2f}).")
        print("      * Recommended Strategy: Mean-reversion, iron condors, fading breakout attempts at the walls.")
    else:
        print("  [!] REGIME: NEGATIVE GAMMA (Dealer Short Gamma).")
        print("      * Dealers amplify market moves (sell into drops, buy into rips) to dynamically hedge delta.")
        print(f"      * Volatility expansion is expected. Price has crossed below Gamma Flip (${flip_price:.2f}).")
        print("      * Recommended Strategy: Momentum breakout following, purchasing tail protection, trailing stops.")
    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Institutional Dealer Gamma Exposure (GEX) & Options Flow Analysis"
    )
    parser.add_argument("--symbol", type=str, default="SMH", help="Ticker symbol (e.g. SMH, NVDA, AAPL)")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="~/.qlib/qlib_data/us_data",
        help="Path to Qlib US market data directory",
    )
    parser.add_argument(
        "--no_auto_download",
        action="store_true",
        help="Disable automatic downloading of underlying market data",
    )
    parser.add_argument(
        "--r",
        type=float,
        default=0.045,
        help="Annualized risk-free interest rate (default: 0.045)",
    )
    args = parser.parse_args()

    run_gex_demo(
        symbol=args.symbol,
        data_dir=args.data_dir,
        auto_download=not args.no_auto_download,
        r=args.r,
    )


if __name__ == "__main__":
    main()
