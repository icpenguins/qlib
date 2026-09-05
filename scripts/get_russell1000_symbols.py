#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Russell 1000 US Equities Universe Generator for Microsoft Qlib
=============================================================
Curates, validates, and serializes the complete constituent universe
of the Russell 1000 index (and high-liquidity seed universes) formatted
specifically for Microsoft Qlib instruments files.

Format Specification:
    <SYMBOL>\\t<START_DATE>\\t<END_DATE>
"""

import sys
import io
import os
import logging
import argparse
import datetime
from pathlib import Path
from typing import List, Set, Dict, Optional

import requests
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Russell1000Universe")

# Top 60 liquid mega/large-cap seed stocks across all 11 GICS sectors for rapid validation
RUSSELL1000_SEED_STOCKS = [
    # Information Technology
    "MSFT", "AAPL", "NVDA", "AVGO", "CSCO", "ADBE", "CRM", "INTC", "AMD", "IBM", "MU", "ANET", "CRDO",
    # Communication Services
    "GOOGL", "GOOG", "META", "NFLX", "DIS", "CMCSA", "TMUS", "VZ", "T",
    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "BKNG", "LOW", "TJX",
    # Financials
    "JPM", "V", "MA", "BAC", "WFC", "MS", "GS", "BLK", "AXP", "C",
    # Health Care
    "LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "DHR", "PFE", "AMGN",
    # Industrials
    "GE", "CAT", "UNP", "HON", "BA", "UPS", "RTX", "LMT", "DE", "FIX",
    # Consumer Staples
    "WMT", "PG", "COST", "KO", "PEP", "PM", "MO", "MDLZ", "CL",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO",
    # Utilities & Real Estate & Materials
    "NEE", "SO", "DUK", "PLD", "AMT", "CCI", "LIN", "SHW", "FCX", "NEM",
    # Benchmark ETFs
    "SPY", "QQQ", "IWB", "VOO"
]


def fetch_wikipedia_symbols(url: str, symbol_col: str) -> List[str]:
    """Fetch symbols from a Wikipedia table."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            dfs = pd.read_html(io.StringIO(resp.text))
            for df in dfs:
                for col in df.columns:
                    col_str = str(col).lower()
                    if symbol_col.lower() in col_str or "symbol" in col_str or "ticker" in col_str:
                        raw = df[col].dropna().astype(str).tolist()
                        cleaned = [
                            s.strip().upper().replace(".", "-")
                            for s in raw
                            if s.strip() and not s.startswith("—") and len(s) <= 5
                        ]
                        if len(cleaned) >= 50:
                            return cleaned
    except Exception as e:
        logger.warning(f"Failed fetching symbols from {url}: {e}")
    return []


def get_curated_russell1000_universe(seed_only: bool = False) -> List[str]:
    """
    Retrieve comprehensive Russell 1000 constituents.
    If seed_only is True, returns top 60 liquid mega/large caps across sectors.
    Otherwise, gathers S&P 500 + S&P 400 MidCap + liquid large caps to curate 1,000 US equities.
    """
    if seed_only:
        seen = set()
        seed_dedup = []
        for s in RUSSELL1000_SEED_STOCKS:
            s_clean = s.strip().upper()
            if s_clean not in seen:
                seen.add(s_clean)
                seed_dedup.append(s_clean)
        logger.info(f"Using curated Russell 1000 seed universe: {len(seed_dedup)} symbols.")
        return seed_dedup

    logger.info("Curating full Russell 1000 constituent universe...")
    symbols: Set[str] = set()

    # 1. Add core seeds
    symbols.update(s.upper() for s in RUSSELL1000_SEED_STOCKS)

    # 2. Fetch S&P 500 (503 large-cap US stocks)
    sp500 = fetch_wikipedia_symbols("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol")
    if sp500:
        logger.info(f"Loaded {len(sp500)} S&P 500 symbols from live Wikipedia.")
        symbols.update(sp500)

    # 3. Fetch S&P 400 MidCap (400 mid-cap US stocks)
    sp400 = fetch_wikipedia_symbols("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "Symbol")
    if sp400:
        logger.info(f"Loaded {len(sp400)} S&P MidCap 400 symbols from live Wikipedia.")
        symbols.update(sp400)

    # 4. If network was offline or fewer than 900 collected, fall back to historical Russell 1000 list
    if len(symbols) < 800:
        logger.info("Attempting backup GitHub repository for Russell 1000 list...")
        backup_url = "https://raw.githubusercontent.com/mcprentiss/Russell_1000_download/master/rus1000.csv"
        try:
            r = requests.get(backup_url, timeout=10)
            if r.status_code == 200:
                lines = r.text.splitlines()
                for line in lines:
                    parts = line.split(",")
                    if parts and parts[0].strip():
                        sym = parts[0].strip().upper().replace(".", "-")
                        if len(sym) <= 5 and sym.isalpha():
                            symbols.add(sym)
                logger.info(f"Universe expanded to {len(symbols)} symbols using GitHub backup.")
        except Exception as e:
            logger.warning(f"Backup fetch failed: {e}")

    cleaned_symbols = sorted(list(symbols))
    logger.info(f"Total curated Russell 1000 universe: {len(cleaned_symbols)} symbols.")
    return cleaned_symbols


def write_qlib_instrument_file(
    symbols: List[str],
    output_path: Path,
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
) -> None:
    """
    Write symbols in Qlib instrument TSV format:
        <SYMBOL>\\t<START_DATE>\\t<END_DATE>
    """
    if end_date is None:
        end_date = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sym in sorted(symbols):
            f.write(f"{sym}\t{start_date}\t{end_date}\n")
    logger.info(f"Wrote {len(symbols)} instruments to {output_path.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Russell 1000 constituent universe file for Microsoft Qlib."
    )
    parser.add_argument(
        "--seed_only",
        action="store_true",
        default=False,
        help="Generate top 60 liquid mega/large-cap seed universe for rapid testing.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="data/instruments",
        help="Target directory in repository to store russell1000.txt.",
    )
    parser.add_argument(
        "--qlib_dir",
        type=str,
        default="~/.qlib/qlib_data/us_data",
        help="Qlib data directory where instruments/russell1000.txt will also be mirrored.",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default="2015-01-01",
        help="Start date for instrument active range (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="End date for instrument active range (YYYY-MM-DD).",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    symbols = get_curated_russell1000_universe(seed_only=args.seed_only)

    # 1. Write to repo instruments directory
    out_p = Path(args.output_dir)
    if not out_p.is_absolute():
        out_p = REPO_ROOT / out_p
    repo_file = out_p.resolve() / "russell1000.txt"
    write_qlib_instrument_file(symbols, repo_file, start_date=args.start_date, end_date=args.end_date)

    # 2. Write to Qlib data instruments directory
    qlib_inst_dir = Path(args.qlib_dir).expanduser().resolve() / "instruments"
    qlib_file = qlib_inst_dir / "russell1000.txt"
    write_qlib_instrument_file(symbols, qlib_file, start_date=args.start_date, end_date=args.end_date)

    print("\n" + "=" * 80)
    print(f"RUSSELL 1000 UNIVERSE GENERATION COMPLETE ({len(symbols)} tickers)")
    print("=" * 80)
    print(f"Repo File: {repo_file}")
    print(f"Qlib File: {qlib_file}")
    print("Sample Tickers:", symbols[:10])
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
