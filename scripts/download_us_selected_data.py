#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Targeted US Stock Data Downloader for Microsoft Qlib
===================================================

This script provides an end-to-end data pipeline to download, normalize,
and dump US stock data for a specific set of tickers:
    VOO, FIX, CRDO, MSFT, INTC, MU, ANET, IBM, TSLA, NVDA, SPY, QQQ

Pipeline Stages:
    1. Download raw OHLCV + adjusted close data from Yahoo Finance.
    2. Normalize data according to Qlib standards:
       - Split and dividend adjustment factor calculation
       - Volume adjustment
       - First-day close price standardization
       - Percentage change computation
    3. Dump normalized data to Qlib binary format (.bin) with trading
       calendars and instrument indices, making it immediately usable
       via `qlib.init(provider_uri=...)`.

Usage:
    # Run with default 12 tickers:
    python scripts/download_us_selected_data.py

    # Pass a custom list of tickers directly via CLI:
    python scripts/download_us_selected_data.py --symbols AAPL MSFT NVDA GOOGL AMZN
    python scripts/download_us_selected_data.py --symbols "AAPL, MSFT, NVDA"

    # Pass a path to a file containing tickers (txt, csv, or json):
    python scripts/download_us_selected_data.py --symbol_file my_tickers.txt
    python scripts/download_us_selected_data.py -f my_portfolio.csv

    # Customize start date and destination directories:
    # Specify storage destination directory (creates source, normalize, and qlib_data inside):
    python scripts/download_us_selected_data.py --target_dir ./my_market_data
    python scripts/download_us_selected_data.py -o /data/stocks --symbols MSFT NVDA

    # Customize start date and granular destination directories:
    python scripts/download_us_selected_data.py --symbols MSFT TSLA --start 2015-01-01 --qlib_dir ~/.qlib/qlib_data/us_data

    # Download raw CSVs only without binary dumping:
    python scripts/download_us_selected_data.py --symbols MSFT --no-dump_qlib
    python scripts/download_us_selected_data.py --symbols MSFT --target_dir ./raw_only --no-dump_qlib
"""

import os
import sys
import time
import json
import logging
import argparse
import datetime
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("USDataDownloader")

# Default targeted US symbols requested by user
DEFAULT_US_SYMBOLS: List[str] = [
    "VOO",
    "FIX",
    "CRDO",
    "MSFT",
    "INTC",
    "MU",
    "ANET",
    "IBM",
    "TSLA",
    "NVDA",
    "SPY",
    "QQQ",
]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
]


def load_symbols_from_file(file_path: Union[str, Path]) -> List[str]:
    """
    Load and parse stock tickers from an external file.

    Supported Formats:
    - Plain text (.txt): One symbol per line, or comma/space/tab-separated.
      Lines beginning with '#' or '//' are treated as comments and ignored.
    - CSV (.csv): If a column named 'symbol', 'ticker', 'code', or 'instrument'
      is present, extracts that column. Otherwise, splits rows by commas.
    - JSON (.json): Top-level array of strings, or a dictionary containing a
      'symbols' or 'tickers' list.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the file containing ticker symbols.

    Returns
    -------
    List[str]
        Cleaned, uppercase, deduplicated list of tickers maintaining appearance order.

    Raises
    ------
    FileNotFoundError
        If the specified file path does not exist.
    """
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Specified symbol file does not exist: {path}")

    logger.info(f"Loading ticker list from file: {path}")
    raw_symbols = []
    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                raw_symbols = [str(x) for x in data]
            elif isinstance(data, dict):
                for key in ["symbols", "tickers", "stocks", "instruments"]:
                    if key in data and isinstance(data[key], list):
                        raw_symbols = [str(x) for x in data[key]]
                        break
    elif suffix == ".csv":
        try:
            df = pd.read_csv(path)
            candidate_cols = [
                c for c in df.columns
                if str(c).strip().lower() in ("symbol", "ticker", "code", "instrument", "stock")
            ]
            if candidate_cols:
                raw_symbols = df[candidate_cols[0]].dropna().astype(str).tolist()
            else:
                text = path.read_text(encoding="utf-8")
                raw_symbols = [t.strip().strip('"\'') for t in text.replace("\n", ",").split(",") if t.strip()]
        except Exception as e:
            logger.debug(f"CSV dataframe parsing fallback: {e}")
            text = path.read_text(encoding="utf-8")
            raw_symbols = [t.strip().strip('"\'') for t in text.replace("\n", ",").split(",") if t.strip()]
    else:
        # Plain text or other delimited file
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("//"):
                    continue
                # Split inline comments if any
                line = line.split("#")[0].strip()
                parts = line.replace(",", " ").replace("\t", " ").split()
                for p in parts:
                    clean = p.strip().strip('"\'')
                    if clean:
                        raw_symbols.append(clean)

    symbols = parse_symbols(raw_symbols)
    logger.info(f"Loaded {len(symbols)} tickers from file: {symbols}")
    return symbols


def parse_symbols(symbol_input: Union[str, List[str], Path]) -> List[str]:
    """
    Parse and clean symbol list from comma-separated string, space-separated string,
    list, or existing file path.

    Parameters
    ----------
    symbol_input : Union[str, List[str], Path]
        String of symbols, list of symbol strings, or path to a symbol file.

    Returns
    -------
    List[str]
        Cleaned, uppercase, deduplicated list of tickers maintaining order.
    """
    if isinstance(symbol_input, Path):
        if symbol_input.is_file():
            return load_symbols_from_file(symbol_input)
        symbol_input = str(symbol_input)

    # Check if single string input is a file path
    if isinstance(symbol_input, str):
        trimmed = symbol_input.strip().strip('"\'')
        candidate_path = Path(trimmed).expanduser()
        if candidate_path.is_file():
            return load_symbols_from_file(candidate_path)
        raw_symbols = symbol_input.replace(",", " ").split()
    elif isinstance(symbol_input, (list, tuple)):
        # Check if list consists of a single file path string
        if len(symbol_input) == 1:
            candidate_path = Path(str(symbol_input[0]).strip().strip('"\'')).expanduser()
            if candidate_path.is_file():
                return load_symbols_from_file(candidate_path)
        raw_symbols = []
        for item in symbol_input:
            raw_symbols.extend(str(item).replace(",", " ").split())
    else:
        raw_symbols = list(symbol_input)

    seen = set()
    cleaned = []
    for s in raw_symbols:
        sym = s.strip().strip('"\'').upper()
        if sym and sym not in seen:
            seen.add(sym)
            cleaned.append(sym)
    return cleaned


def fetch_yahoo_v8_chart(
    symbol: str,
    start_ts: int,
    end_ts: int,
    interval: str = "1d",
    max_retries: int = 4,
    retry_delay: float = 1.0,
) -> Optional[pd.DataFrame]:
    """
    Fetch historical price data directly from Yahoo Finance v8 chart API.

    Parameters
    ----------
    symbol : str
        Stock or ETF ticker (e.g., 'MSFT', 'SPY').
    start_ts : int
        Unix timestamp for start date.
    end_ts : int
        Unix timestamp for end date.
    interval : str, optional
        Data frequency ('1d', '1m', etc.), by default '1d'.
    max_retries : int, optional
        Maximum number of retries on network failures, by default 4.
    retry_delay : float, optional
        Initial backoff delay in seconds, by default 1.0.

    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame with columns: date, open, high, low, close, adjclose, volume, symbol.
    """
    # Yahoo chart interval mapping: 1min -> 1m, 1d -> 1d
    api_interval = "1m" if interval in ("1m", "1min") else "1d"
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?"
        f"period1={start_ts}&period2={end_ts}&interval={api_interval}&events=history"
    )

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": USER_AGENTS[(attempt - 1) % len(USER_AGENTS)],
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status != 200:
                    raise urllib.error.HTTPError(
                        url, resp.status, f"HTTP status {resp.status}", resp.headers, None
                    )
                payload = json.loads(resp.read().decode("utf-8"))

            chart_data = payload.get("chart", {})
            results = chart_data.get("result")
            if not results:
                err = chart_data.get("error")
                logger.warning(f"Yahoo API returned no result for {symbol}: {err}")
                return None

            data = results[0]
            timestamps = data.get("timestamp")
            if not timestamps:
                logger.warning(f"No timestamp data available for {symbol}")
                return None

            indicators = data.get("indicators", {})
            quotes = indicators.get("quote", [{}])[0]
            adjclose_list = indicators.get("adjclose", [{}])[0].get("adjclose")

            opens = quotes.get("open", [])
            highs = quotes.get("high", [])
            lows = quotes.get("low", [])
            closes = quotes.get("close", [])
            volumes = quotes.get("volume", [])

            # If adjclose is missing, fallback to raw close
            if adjclose_list is None or len(adjclose_list) != len(timestamps):
                adjclose_list = closes

            # Construct DataFrame
            date_series = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(
                "America/New_York"
            )
            date_str = (
                date_series.strftime("%Y-%m-%d")
                if interval == "1d"
                else date_series.strftime("%Y-%m-%d %H:%M:%S")
            )

            df = pd.DataFrame(
                {
                    "date": date_str,
                    "open": opens,
                    "high": highs,
                    "low": lows,
                    "close": closes,
                    "adjclose": adjclose_list,
                    "volume": volumes,
                    "symbol": symbol,
                }
            )

            # Filter out records where essential price is NaN
            df = df.dropna(subset=["open", "high", "low", "close"]).copy()
            df["volume"] = df["volume"].fillna(0.0)

            # Sort and deduplicate by date
            df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

            return df

        except Exception as e:
            last_error = e
            if attempt < max_retries:
                sleep_time = retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"Attempt {attempt}/{max_retries} failed for {symbol}: {e}. Retrying in {sleep_time:.1f}s..."
                )
                time.sleep(sleep_time)

    logger.error(f"Failed to fetch data for {symbol} after {max_retries} attempts. Error: {last_error}")
    return None


def fetch_fallback_engines(
    symbol: str, start: str, end: str, interval: str = "1d"
) -> Optional[pd.DataFrame]:
    """
    Fallback data fetcher using `yfinance` or `yahooquery` if installed.
    """
    # 1. Try yfinance if installed
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        yf_interval = "1m" if interval in ("1m", "1min") else "1d"
        hist = ticker.history(start=start, end=end, interval=yf_interval, auto_adjust=False)
        if isinstance(hist, pd.DataFrame) and not hist.empty:
            hist = hist.reset_index()
            date_col = "Date" if "Date" in hist.columns else hist.columns[0]
            hist["date"] = pd.to_datetime(hist[date_col]).dt.strftime(
                "%Y-%m-%d" if interval == "1d" else "%Y-%m-%d %H:%M:%S"
            )
            hist = hist.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adjclose",
                    "Volume": "volume",
                }
            )
            hist["symbol"] = symbol
            cols = ["date", "open", "high", "low", "close", "adjclose", "volume", "symbol"]
            avail_cols = [c for c in cols if c in hist.columns]
            return hist[avail_cols].dropna(subset=["close"]).reset_index(drop=True)
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"yfinance fallback failed for {symbol}: {e}")

    # 2. Try yahooquery if installed
    try:
        from yahooquery import Ticker

        yq_interval = "1m" if interval in ("1m", "1min") else interval
        yq_resp = Ticker(symbol, asynchronous=False).history(
            interval=yq_interval, start=start, end=end
        )
        if isinstance(yq_resp, pd.DataFrame) and not yq_resp.empty:
            df = yq_resp.reset_index()
            date_col = "date" if "date" in df.columns else df.columns[1]
            df["date"] = pd.to_datetime(df[date_col]).dt.strftime(
                "%Y-%m-%d" if interval == "1d" else "%Y-%m-%d %H:%M:%S"
            )
            df["symbol"] = symbol
            if "adjclose" not in df.columns and "close" in df.columns:
                df["adjclose"] = df["close"]
            return df[["date", "open", "high", "low", "close", "adjclose", "volume", "symbol"]]
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"yahooquery fallback failed for {symbol}: {e}")

    return None


def download_raw_symbol(
    symbol: str,
    start: str,
    end: str,
    interval: str = "1d",
    delay: float = 0.5,
) -> Optional[pd.DataFrame]:
    """
    Download historical data for a single symbol using resilient multi-engine routing.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    start : str
        Start date 'YYYY-MM-DD'.
    end : str
        End date 'YYYY-MM-DD'.
    interval : str, optional
        Interval, default '1d'.
    delay : float, optional
        Delay in seconds, default 0.5.

    Returns
    -------
    Optional[pd.DataFrame]
        Raw OHLCV dataframe.
    """
    time.sleep(delay)

    # Convert start and end to unix timestamps
    start_dt = pd.Timestamp(start, tz="America/New_York")
    end_dt = pd.Timestamp(end, tz="America/New_York")
    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())

    # 1. Primary engine: Direct Yahoo v8 chart API
    df = fetch_yahoo_v8_chart(symbol, start_ts, end_ts, interval=interval)
    if df is not None and not df.empty:
        return df

    # 2. Secondary fallback engine (yfinance / yahooquery)
    logger.info(f"Attempting fallback data fetchers for {symbol}...")
    df_fallback = fetch_fallback_engines(symbol, start, end, interval=interval)
    if df_fallback is not None and not df_fallback.empty:
        return df_fallback

    return None


def normalize_symbol_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Normalize raw stock data following Microsoft Qlib 1D standards.

    Transformation Steps:
    1. Adjustment factor: factor = adjclose / close
    2. Adjusted prices: price = price * factor
    3. Adjusted volume: volume = volume / factor
    4. First-day price normalization: all prices scaled so first valid trading
       day close = 1.0 (price = price / first_close; volume = volume * first_close)
    5. Percentage change: change = (close - prev_close) / prev_close

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe containing [date, open, high, low, close, adjclose, volume, symbol].
    symbol : str
        Stock symbol.

    Returns
    -------
    pd.DataFrame
        Normalized dataframe ready for Qlib binary dumping.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 1. Calculate adjustment factor
    if "adjclose" in df.columns and "close" in df.columns:
        df["factor"] = (df["adjclose"] / df["close"]).replace([np.inf, -np.inf], np.nan).ffill().bfill()
        df["factor"] = df["factor"].fillna(1.0)
    else:
        df["factor"] = 1.0

    # 2. Adjust prices and volume by factor
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col] * df["factor"]

    if "volume" in df.columns:
        # Volume is divided by factor to reflect split adjustment
        df["volume"] = df["volume"] / df["factor"]

    # 3. Standardize by first day's close price (Qlib convention for 1D Yahoo data)
    valid_close_idx = df["close"].first_valid_index()
    if valid_close_idx is not None:
        first_close = df.loc[valid_close_idx, "close"]
        if first_close > 0 and not np.isnan(first_close):
            for col in price_cols:
                if col in df.columns:
                    df[col] = df[col] / first_close
            if "volume" in df.columns:
                df["volume"] = df["volume"] * first_close

    # 4. Calculate percentage change
    df["change"] = df["close"].pct_change()
    df.loc[df.index[0], "change"] = 0.0

    # Format date back to string
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df["symbol"] = symbol.upper()

    output_cols = ["date", "symbol", "open", "high", "low", "close", "volume", "factor", "change"]
    avail_cols = [c for c in output_cols if c in df.columns]
    return df[avail_cols]


def dump_to_qlib_format(
    normalized_dfs: Dict[str, pd.DataFrame],
    qlib_dir: Union[str, Path],
    freq: str = "day",
) -> None:
    """
    Dump normalized DataFrames into Qlib binary format (.bin) with calendars and instruments.

    File Layout:
        <qlib_dir>/
        ├── calendars/
        │   └── day.txt
        ├── instruments/
        │   └── all.txt
        └── features/
            └── <SYMBOL>/
                ├── open.day.bin
                ├── high.day.bin
                ├── low.day.bin
                ├── close.day.bin
                ├── volume.day.bin
                ├── factor.day.bin
                └── change.day.bin

    Parameters
    ----------
    normalized_dfs : Dict[str, pd.DataFrame]
        Mapping from symbol to normalized DataFrame.
    qlib_dir : Union[str, Path]
        Target directory for Qlib data provider.
    freq : str, optional
        Data frequency, by default 'day'.
    """
    qlib_path = Path(qlib_dir).expanduser().resolve()
    cal_dir = qlib_path.joinpath("calendars")
    inst_dir = qlib_path.joinpath("instruments")
    feat_dir = qlib_path.joinpath("features")

    cal_dir.mkdir(parents=True, exist_ok=True)
    inst_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build unified trading calendar
    all_dates = set()
    for symbol, df in normalized_dfs.items():
        if not df.empty and "date" in df.columns:
            all_dates.update(df["date"].dropna().unique())

    calendar_list = sorted(all_dates)
    if not calendar_list:
        logger.warning("No calendar dates found to dump.")
        return

    cal_file = cal_dir.joinpath(f"{freq}.txt")
    with open(cal_file, "w", encoding="utf-8") as f:
        for d in calendar_list:
            f.write(f"{d}\n")
    logger.info(f"Saved calendar with {len(calendar_list)} trading days to {cal_file}")

    # Map each date string to its index in calendar_list
    date_to_idx = {d: idx for idx, d in enumerate(calendar_list)}

    # 2. Dump features and instruments
    instruments_data = []
    feature_fields = ["open", "high", "low", "close", "volume", "factor", "change"]

    for symbol, df in normalized_dfs.items():
        if df.empty:
            continue

        symbol_upper = symbol.upper()
        sym_feat_dir = feat_dir.joinpath(symbol_upper)
        sym_feat_dir.mkdir(parents=True, exist_ok=True)

        df_sorted = df.drop_duplicates(subset=["date"]).sort_values("date").copy()
        df_sorted = df_sorted.set_index("date")

        start_date = df_sorted.index[0]
        end_date = df_sorted.index[-1]
        instruments_data.append((symbol_upper, start_date, end_date))

        # Align with calendar range between start_date and end_date
        start_idx = date_to_idx[start_date]
        end_idx = date_to_idx[end_date]
        symbol_calendar = calendar_list[start_idx : end_idx + 1]

        aligned_df = df_sorted.reindex(symbol_calendar)

        for field in feature_fields:
            if field not in aligned_df.columns:
                continue
            bin_path = sym_feat_dir.joinpath(f"{field}.{freq}.bin")
            values = aligned_df[field].to_numpy(dtype=np.float32)

            # Binary format specification:
            # First element: starting calendar index (int/float32)
            # Subsequent elements: array of float32 values
            header = np.array([start_idx], dtype="<f")
            payload = values.astype("<f")
            bin_array = np.hstack([header, payload])
            bin_array.tofile(str(bin_path.resolve()))

    # Save instruments/all.txt
    inst_file = inst_dir.joinpath("all.txt")
    with open(inst_file, "w", encoding="utf-8") as f:
        for sym, s_dt, e_dt in sorted(instruments_data):
            f.write(f"{sym}\t{s_dt}\t{e_dt}\n")
    logger.info(f"Saved {len(instruments_data)} instruments to {inst_file}")


def run_pipeline(
    symbols: List[str] = DEFAULT_US_SYMBOLS,
    start: str = "2000-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
    target_dir: Optional[Union[str, Path]] = None,
    source_dir: Optional[Union[str, Path]] = None,
    normalize_dir: Optional[Union[str, Path]] = None,
    qlib_dir: Optional[Union[str, Path]] = None,
    dump_qlib: bool = True,
    delay: float = 0.5,
    download_options: bool = False,
    download_events: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Execute the full end-to-end data acquisition and processing pipeline.

    Parameters
    ----------
    symbols : List[str]
        List of tickers to process.
    start : str
        Start date 'YYYY-MM-DD'.
    end : Optional[str]
        End date 'YYYY-MM-DD'. Defaults to tomorrow's date.
    interval : str
        Frequency ('1d').
    target_dir : Optional[Union[str, Path]]
        Root target directory to store downloaded and processed content.
        If supplied, default subdirectories 'source', 'normalize', and 'qlib_data'
        are created inside it unless explicitly overridden.
    source_dir : Optional[Union[str, Path]]
        Destination for raw CSV files.
    normalize_dir : Optional[Union[str, Path]]
        Destination for normalized CSV files.
    qlib_dir : Optional[Union[str, Path]]
        Destination for Qlib binary dataset.
    dump_qlib : bool
        Whether to generate Qlib binary dataset.
    delay : float
        Inter-request sleep delay in seconds.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping ticker symbols to normalized DataFrames.
    """
    if end is None:
        end = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    # Resolve target directory paths
    if target_dir is not None:
        target_path = Path(target_dir).expanduser().resolve()
        target_path.mkdir(parents=True, exist_ok=True)
        source_path = (target_path / "source") if source_dir is None else Path(source_dir).expanduser().resolve()
        norm_path = (target_path / "normalize") if normalize_dir is None else Path(normalize_dir).expanduser().resolve()
        qlib_path = (target_path / "qlib_data") if qlib_dir is None else Path(qlib_dir).expanduser().resolve()
    else:
        source_path = Path(source_dir if source_dir is not None else "~/.qlib/stock_data/source/us_data").expanduser().resolve()
        norm_path = Path(normalize_dir if normalize_dir is not None else "~/.qlib/stock_data/source/us_1d_nor").expanduser().resolve()
        qlib_path = Path(qlib_dir if qlib_dir is not None else "~/.qlib/qlib_data/us_data").expanduser().resolve()

    source_path.mkdir(parents=True, exist_ok=True)
    norm_path.mkdir(parents=True, exist_ok=True)

    cleaned_symbols = parse_symbols(symbols)
    logger.info(f"Starting US data collection for {len(cleaned_symbols)} symbols: {cleaned_symbols}")
    logger.info(f"Date Range: {start} -> {end} | Interval: {interval}")
    logger.info(f"Raw CSV Storage: {source_path}")
    logger.info(f"Normalized CSV Storage: {norm_path}")
    if dump_qlib:
        logger.info(f"Qlib Binary Storage: {qlib_path}")

    results: Dict[str, pd.DataFrame] = {}
    summary_rows = []

    for idx, sym in enumerate(cleaned_symbols, 1):
        logger.info(f"[{idx}/{len(cleaned_symbols)}] Downloading {sym}...")
        raw_df = download_raw_symbol(sym, start=start, end=end, interval=interval, delay=delay)

        if raw_df is None or raw_df.empty:
            logger.warning(f"[{idx}/{len(cleaned_symbols)}] Failed to retrieve data for {sym}")
            summary_rows.append({"Symbol": sym, "Status": "FAILED", "Days": 0, "Start": "-", "End": "-", "Latest": "-"})
            continue

        # Save raw CSV
        raw_file = source_path.joinpath(f"{sym.upper()}.csv")
        raw_df.to_csv(raw_file, index=False)

        # Normalize data
        norm_df = normalize_symbol_data(raw_df, sym)
        norm_file = norm_path.joinpath(f"{sym.upper()}.csv")
        norm_df.to_csv(norm_file, index=False)

        results[sym.upper()] = norm_df

        start_date = norm_df["date"].iloc[0]
        end_date = norm_df["date"].iloc[-1]
        latest_close = f"{raw_df['close'].iloc[-1]:.2f}"
        summary_rows.append({
            "Symbol": sym.upper(),
            "Status": "SUCCESS",
            "Days": len(norm_df),
            "Start": start_date,
            "End": end_date,
            "Latest": latest_close,
        })
        logger.info(f"[{idx}/{len(cleaned_symbols)}] {sym} processed: {len(norm_df)} trading days ({start_date} to {end_date})")

    # Dump to Qlib binary format if requested
    if dump_qlib and results:
        logger.info("Dumping normalized data into Qlib binary format...")
        dump_to_qlib_format(results, qlib_dir=qlib_path, freq="day" if interval == "1d" else "1min")
        logger.info(f"Qlib binary dump completed successfully at: {qlib_path}")

    # Download equity option chains if requested
    if download_options:
        logger.info("Downloading equity option chains for targeted tickers...")
        options_dir = (target_path / "options") if target_dir is not None else (source_path.parent / "options")
        options_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Ensure repository path is accessible
            repo_root = Path(__file__).resolve().parent.parent
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            from qlib.contrib.derivatives.options_data import OptionsDataLoader

            opt_loader = OptionsDataLoader(data_dir=options_dir.parent)
            for sym in cleaned_symbols:
                try:
                    logger.info(f"Downloading option chain for {sym} to {options_dir}...")
                    opt_loader.download_and_cache(sym, target_dir=options_dir)
                except Exception as e:
                    logger.warning(f"Failed downloading option chain for {sym}: {e}")
        except Exception as e:
            logger.warning(f"Error during option chain acquisition: {e}")

    # Download corporate earnings & event calendars if requested
    if download_events:
        logger.info("Downloading corporate event calendars & PEAD history for targeted tickers...")
        events_dir = (target_path / "events") if target_dir is not None else (source_path.parent / "events")
        events_dir.mkdir(parents=True, exist_ok=True)
        try:
            repo_root = Path(__file__).resolve().parent.parent
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            from qlib.contrib.events.events_data import EventsDataLoader

            ev_loader = EventsDataLoader(data_dir=events_dir.parent)
            for sym in cleaned_symbols:
                try:
                    logger.info(f"Downloading event calendar for {sym} to {events_dir}...")
                    ev_loader.load_or_generate_events(sym, force_download=True)
                except Exception as e:
                    logger.warning(f"Failed downloading event calendar for {sym}: {e}")
        except Exception as e:
            logger.warning(f"Error during event calendar acquisition: {e}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'DOWNLOAD & NORMALIZATION SUMMARY':^80}")
    print("=" * 80)
    print(f"{'Symbol':<10} {'Status':<10} {'Days':<10} {'Start Date':<15} {'End Date':<15} {'Latest Close':<12}")
    print("-" * 80)
    for row in summary_rows:
        print(f"{row['Symbol']:<10} {row['Status']:<10} {row['Days']:<10} {row['Start']:<15} {row['End']:<15} {row['Latest']:<12}")
    print("=" * 80 + "\n")

    return results


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Targeted US stock data downloader and Qlib binary dumper with support for custom ticker lists and custom storage locations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbols",
        "-s",
        nargs="*",
        default=None,
        help="Space or comma-separated ticker list, or path to a ticker file (e.g. --symbols AAPL MSFT or --symbols tickers.txt). Defaults to the 12 requested US tickers if not provided.",
    )
    parser.add_argument(
        "--symbol_file",
        "-f",
        type=str,
        default=None,
        help="Path to a text/CSV/JSON file containing a list of stock tickers (e.g. --symbol_file tickers.txt).",
    )
    parser.add_argument(
        "--target_dir",
        "--output_dir",
        "--dest",
        "-o",
        type=str,
        default=None,
        help="Root storage directory for all downloaded and processed content (automatically sub-divides into 'source', 'normalize', and 'qlib_data').",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2000-01-01",
        help="Start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date in YYYY-MM-DD format (defaults to tomorrow's date).",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        choices=["1d", "1m", "1min"],
        help="Data frequency.",
    )
    parser.add_argument(
        "--source_dir",
        "--raw_dir",
        type=str,
        default=None,
        help="Specific directory to save raw CSV files (defaults to <target_dir>/source or ~/.qlib/stock_data/source/us_data).",
    )
    parser.add_argument(
        "--normalize_dir",
        type=str,
        default=None,
        help="Specific directory to save normalized CSV files (defaults to <target_dir>/normalize or ~/.qlib/stock_data/source/us_1d_nor).",
    )
    parser.add_argument(
        "--qlib_dir",
        type=str,
        default=None,
        help="Specific target directory for dumped Qlib binary format data (defaults to <target_dir>/qlib_data or ~/.qlib/qlib_data/us_data).",
    )
    parser.add_argument(
        "--dump_qlib",
        action="store_true",
        default=True,
        help="Convert normalized data to Qlib binary format.",
    )
    parser.add_argument(
        "--no-dump_qlib",
        dest="dump_qlib",
        action="store_false",
        help="Skip Qlib binary dumping stage (only download raw and normalized CSVs).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay in seconds between ticker downloads to avoid rate limits.",
    )
    parser.add_argument(
        "--download_options",
        action="store_true",
        default=False,
        help="Download and cache equity option chains for targeted tickers into <target_dir>/options.",
    )
    parser.add_argument(
        "--download_events",
        action="store_true",
        default=False,
        help="Download and cache corporate earnings calendars and PEAD history for targeted tickers into <target_dir>/events.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Determine symbols source: --symbol_file, --symbols, or default list
    if args.symbol_file:
        symbols = load_symbols_from_file(args.symbol_file)
    elif args.symbols:
        symbols = parse_symbols(args.symbols)
    else:
        logger.info(
            f"No --symbols or --symbol_file specified. Using default 12 tickers: {DEFAULT_US_SYMBOLS}"
        )
        symbols = DEFAULT_US_SYMBOLS

    run_pipeline(
        symbols=symbols,
        start=args.start,
        end=args.end,
        interval=args.interval,
        target_dir=args.target_dir,
        source_dir=args.source_dir,
        normalize_dir=args.normalize_dir,
        qlib_dir=args.qlib_dir,
        dump_qlib=args.dump_qlib,
        delay=args.delay,
        download_options=args.download_options,
        download_events=args.download_events,
    )


if __name__ == "__main__":
    main()

