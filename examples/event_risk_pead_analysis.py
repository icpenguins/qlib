#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone Event Risk, Catalyst Awareness & PEAD CLI Tool
=========================================================
Demonstrates corporate calendar tracking, risk de-grossing overlays,
Post-Earnings Announcement Drift (PEAD) alpha dynamics, and key momentum
events on any US equity ticker.

Usage:
    python examples/event_risk_pead_analysis.py --symbol NVDA --data_dir D:/trading/qlib
    python examples/event_risk_pead_analysis.py --symbol SMH --date 2026-09-03
"""

import argparse
import datetime
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from qlib.contrib.events import (
    EventCalendarEngine,
    PEADEngine,
    RiskDegrossingEngine,
    EventsDataLoader,
    compute_event_risk_features,
)
from stock_analysis_engine import load_stock_data


def main():
    parser = argparse.ArgumentParser(
        description="Inspect Corporate Catalysts, Event Risk De-Grossing, and PEAD Drift for US Equities."
    )
    parser.add_argument("--symbol", "-s", type=str, default="SMH", help="Ticker symbol (e.g. SMH, NVDA, MSFT).")
    parser.add_argument("--data_dir", "-d", type=str, default="data", help="Directory containing stock data.")
    parser.add_argument("--date", type=str, default=None, help="Evaluation date (YYYY-MM-DD). Defaults to today.")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    eval_date = args.date or datetime.date.today().strftime("%Y-%m-%d")

    print("=========================================================================")
    print(f" INSTITUTIONAL EVENT RISK & PEAD ANALYSIS: {symbol}")
    print("=========================================================================")
    print(f"Evaluation Date: {eval_date} | Data Directory: {args.data_dir}\n")

    # Load price history if available
    try:
        df = load_stock_data(symbol, args.data_dir, auto_download=False, request_date=eval_date)
    except Exception:
        # Generate dummy 1-year history if offline or CSV not present
        dates = pd.bdate_range(end=eval_date, periods=252).strftime("%Y-%m-%d")
        df = pd.DataFrame({
            "date": dates,
            "open": [100.0] * len(dates),
            "high": [105.0] * len(dates),
            "low": [95.0] * len(dates),
            "close": [102.0] * len(dates),
            "volume": [1_000_000] * len(dates),
            "symbol": symbol,
        })

    features = compute_event_risk_features(
        df=df,
        symbol=symbol,
        data_dir=args.data_dir,
        current_date=eval_date,
    )

    cat = features["catalyst"]
    pead = features["pead"]
    degross = features["degross_multiplier"]

    print("-------------------------------------------------------------------------")
    print(" 1. CORPORATE & MACROECONOMIC CATALYST SCHEDULE")
    print("-------------------------------------------------------------------------")
    earn_str = cat.get('next_earnings_date') or 'TBD'
    earn_days = f"({cat.get('earnings_days_away')} business days away)" if cat.get('earnings_days_away') is not None else ""
    print(f"Next Earnings Release:      {earn_str} {earn_days}")
    print(f"Earnings Threat Status:     {cat.get('earnings_proximity')}")
    print(f"Prev Earnings Announcement: {cat.get('prev_earnings_date')} ({cat.get('prev_earnings_days_ago')} business days ago)")
    print(f"Next FOMC Rate Decision:    {cat.get('next_fomc_date')} ({cat.get('fomc_days_away')} days away, {cat.get('fomc_proximity')})")
    print(f"Next BLS CPI Release:       {cat.get('next_cpi_date')} ({cat.get('cpi_days_away')} days away, {cat.get('cpi_proximity')})")
    print(f"\nComposite Threat Level:     {cat.get('composite_proximity')}")
    print(f"Guidance:                   {cat.get('status_description')}")
    print(f"Position Sizing Factor:     {degross:.2f}x standard capital allocation")

    print("\n-------------------------------------------------------------------------")
    print(" 2. POST-EARNINGS ANNOUNCEMENT DRIFT (PEAD) DYNAMICS")
    print("-------------------------------------------------------------------------")
    if pead.get("has_pead"):
        print(f"PEAD Regime State:          {pead['drift_regime']}")
        print(f"Standardized Surprise (SUE): {pead['sue_score']:+.2f} (Reported: ${pead['eps_actual']} vs Est: ${pead['eps_estimate']})")
        print(f"Announcement Reaction:      Day-1 Gap: {pead['announcement_gap_pct']:+.1f}% | 30d Drift: {pead['post_earnings_drift_pct']:+.1f}%")
        print(f"Current Drift Alpha Score:  {pead['pead_drift_score']:+.2f} (Exponential decay half-life: 21 days)")
        print(f"Summary:                    {pead['description']}")
    else:
        print("No historical PEAD records available for this ticker.")

    print("\n-------------------------------------------------------------------------")
    print(" 3. KEY MOMENTUM EVENTS (DISPLAYED ON MAIN CHART)")
    print("-------------------------------------------------------------------------")
    m_events = features.get("momentum_events", [])
    if m_events:
        print(f"{'Date':12s} | {'Type':15s} | {'Badge':12s} | {'Price':8s} | {'Impact / Details'}")
        print("-" * 75)
        for ev in m_events[-8:]:
            safe_badge = ev['badge'].replace('◆', '*').replace('▲', '^').replace('▼', 'v').replace('⚡', '!')
            print(f"{ev['date']:12s} | {ev['type']:15s} | {safe_badge:12s} | ${ev['price']:7.2f} | {ev['momentum_impact']}")
    else:
        print("No historical momentum events detected within this window.")

    print("\n=========================================================================")
    print(" TACTICAL TRADING RECOMMENDATION")
    print("=========================================================================")
    if degross <= 0.25:
        print("[RISK-OFF] Catalyst within 48h. FREEZE NEW ENTRIES and enforce pre-earnings position de-grossing.")
    elif degross <= 0.50:
        print("[CAUTIOUS] Imminent binary event. Reduce position size by 50% to limit overnight gap risk.")
    elif "Bullish" in pead.get("drift_regime", ""):
        print("[BULLISH] Active PEAD accumulation. SUE beat confirms institutional accumulation drift.")
    else:
        print("[NORMAL] Catalyst risk low. Standard systematic sizing and execution rules apply.")
    print("=========================================================================\n")


if __name__ == "__main__":
    main()
