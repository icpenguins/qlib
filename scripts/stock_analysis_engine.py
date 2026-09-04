#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stock Performance & Predictive Buy Timing Engine
=================================================
Calculates 1-year, 3-year, and 5-year historical performance metrics,
detects optimal historical buy/entry points, and generates 3-month forward
predictive buy timing analysis from Qlib binary or CSV datasets.
"""

import sys
import math
import shutil
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd

logger = logging.getLogger("StockAnalysisEngine")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Ensure qlib/contrib is in sys.path for regime module
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
CONTRIB_DIR = REPO_ROOT / "qlib" / "contrib"
if str(CONTRIB_DIR) not in sys.path:
    sys.path.insert(0, str(CONTRIB_DIR))

try:
    from qlib.contrib.regime import MarketRegimeClassifier
except Exception:
    try:
        from regime import MarketRegimeClassifier
    except Exception:
        MarketRegimeClassifier = None

try:
    from qlib.contrib.microstructure import compute_microstructure_features
except Exception:
    try:
        from microstructure import compute_microstructure_features
    except Exception:
        compute_microstructure_features = None

try:
    from qlib.contrib.derivatives import (
        DealerGammaEngine,
        OptionsDataLoader,
        VolatilitySurfaceFeatures,
        compute_dealer_gex_summary,
    )
except Exception:
    try:
        from derivatives import (
            DealerGammaEngine,
            OptionsDataLoader,
            VolatilitySurfaceFeatures,
            compute_dealer_gex_summary,
        )
    except Exception:
        DealerGammaEngine = None
        OptionsDataLoader = None
        VolatilitySurfaceFeatures = None
        compute_dealer_gex_summary = None



# ----------------------------------------------------------------------
# 1. Data Ingestion Layer (Qlib Binary + CSV Discovery & Freshness)
# ----------------------------------------------------------------------

def is_data_up_to_date(
    df: pd.DataFrame,
    request_date: Optional[Union[str, datetime.date, datetime.datetime]] = None,
) -> Tuple[bool, str, str]:
    """
    Check if the loaded stock DataFrame is up-to-date for the requested date.

    Parameters
    ----------
    df : pd.DataFrame
        Stock DataFrame with a 'date' column.
    request_date : Optional[Union[str, datetime.date, datetime.datetime]]
        Date the report was requested. Defaults to today's date.

    Returns
    -------
    Tuple[bool, str, str]
        (is_up_to_date, latest_data_date, expected_trading_date)
    """
    if df.empty or "date" not in df.columns:
        return False, "", ""

    latest_data_date = str(df["date"].max())[:10]

    if request_date is None:
        req_dt = datetime.date.today()
    elif isinstance(request_date, datetime.datetime):
        req_dt = request_date.date()
    elif isinstance(request_date, datetime.date):
        req_dt = request_date
    else:
        req_dt = pd.to_datetime(str(request_date)).date()

    # Determine the expected most recent trading day on or before req_dt:
    # Saturday (5) -> Friday
    # Sunday (6) -> Friday
    if req_dt.weekday() == 5:
        expected_trading_dt = req_dt - datetime.timedelta(days=1)
    elif req_dt.weekday() == 6:
        expected_trading_dt = req_dt - datetime.timedelta(days=2)
    else:
        expected_trading_dt = req_dt

    expected_trading_date = expected_trading_dt.strftime("%Y-%m-%d")
    is_fresh = latest_data_date >= expected_trading_date
    return is_fresh, latest_data_date, expected_trading_date


def load_stock_data(
    symbol: str,
    data_dir: Union[str, Path],
    auto_download: bool = True,
    start: str = "2000-01-01",
    request_date: Optional[Union[str, datetime.date, datetime.datetime]] = None,
) -> pd.DataFrame:
    """
    Load stock OHLCV data for a symbol from a specified directory.
    Supports both Qlib binary format directories and CSV directories.
    Ensures that the loaded data is up-to-date for the day the report was requested.
    If the ticker is missing or stale, automatically downloads/updates it if auto_download=True.

    Parameters
    ----------
    symbol : str
        Stock ticker (e.g. 'MSFT', 'VOO', 'NVDA').
    data_dir : Union[str, Path]
        Directory path containing Qlib binaries, source CSVs, or normalized CSVs.
    auto_download : bool, optional
        Whether to invoke download pipeline if data is missing or stale, by default True.
    start : str, optional
        Start date if auto-download is triggered, by default '2000-01-01'.
    request_date : Optional[Union[str, datetime.date, datetime.datetime]], optional
        Date the report was requested. Defaults to today's date.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['date', 'open', 'high', 'low', 'close', 'volume', 'factor', 'change'],
        indexed by continuous integer or datetime.
    """
    symbol = symbol.upper().strip()
    root_path = Path(data_dir).expanduser().resolve()

    if not root_path.exists():
        raise FileNotFoundError(f"Specified data directory does not exist: {root_path}")

    # 1. Search for CSV formats
    candidate_csv_paths = [
        root_path / f"{symbol}.csv",
        root_path / f"{symbol.lower()}.csv",
        root_path / "source" / f"{symbol}.csv",
        root_path / "normalize" / f"{symbol}.csv",
        root_path / "qlib_data" / f"{symbol}.csv",
        root_path / "raw" / f"{symbol}.csv",
    ]

    # Search recursively for <symbol>.csv if not in standard locations
    found_csv = None
    for p in candidate_csv_paths:
        if p.exists() and p.is_file():
            found_csv = p
            break

    if found_csv is None:
        matches = list(root_path.glob(f"**/{symbol}.csv"))
        if matches:
            found_csv = matches[0]

    df = None
    if found_csv is not None:
        logger.info(f"Loading data for {symbol} from CSV: {found_csv}")
        raw_df = pd.read_csv(found_csv)
        df = _standardize_stock_df(raw_df, symbol)

    # 2. Search for Qlib binary format (<dir>/features/<SYMBOL>/*.bin and <dir>/calendars/day.txt)
    binary_dir = None
    cal_file = None

    if df is None:
        candidate_bin_dirs = [
            root_path / "features" / symbol,
            root_path / "qlib_data" / "features" / symbol,
        ]
        for b_dir in candidate_bin_dirs:
            if b_dir.exists() and b_dir.is_dir():
                binary_dir = b_dir
                cal_candidate = b_dir.parent.parent / "calendars" / "day.txt"
                if cal_candidate.exists():
                    cal_file = cal_candidate
                break

        if binary_dir is None:
            matches = list(root_path.glob(f"**/features/{symbol}"))
            if matches:
                binary_dir = matches[0]
                cal_candidate = binary_dir.parent.parent / "calendars" / "day.txt"
                if cal_candidate.exists():
                    cal_file = cal_candidate

        if binary_dir is not None and cal_file is not None and cal_file.exists():
            logger.info(f"Loading data for {symbol} from Qlib binary: {binary_dir}")
            df = _load_qlib_binary_data(binary_dir, cal_file, symbol)

    # 3. Check freshness against requested report date
    if df is not None:
        is_fresh, latest_date, expected_date = is_data_up_to_date(df, request_date=request_date)
        if is_fresh:
            logger.info(
                f"Data for {symbol} is up-to-date through {latest_date} "
                f"(expected {expected_date} for request date {request_date or datetime.date.today()})."
            )
            return df
        elif not auto_download:
            logger.warning(
                f"Data for {symbol} is from {latest_date}, which is older than expected {expected_date} "
                f"for request date {request_date or datetime.date.today()}, but auto_download is disabled."
            )
            return df
        else:
            logger.info(
                f"Data for {symbol} is from {latest_date}, which is older than expected {expected_date} "
                f"for request date {request_date or datetime.date.today()}. Updating data from provider..."
            )

    # 4. If data is missing or stale and auto_download is enabled, invoke the downloader pipeline
    if auto_download:
        if df is None:
            logger.info(
                f"Symbol '{symbol}' not found in '{root_path}'. "
                f"Invoking download script to acquire historical data..."
            )
        try:
            scripts_dir = Path(__file__).resolve().parent
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))
            from download_us_selected_data import run_pipeline

            # Set end parameter to capture request_date
            end_date = None
            if request_date:
                req_dt = pd.to_datetime(str(request_date)).date()
                end_date = (req_dt + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

            run_pipeline(
                symbols=[symbol],
                target_dir=root_path,
                start=start,
                end=end_date,
                dump_qlib=True,
            )

            # If an existing CSV was located outside source/normalize, update it too
            if found_csv is not None and found_csv.exists():
                source_file = root_path / "source" / f"{symbol}.csv"
                if source_file.exists() and source_file.resolve() != found_csv.resolve():
                    shutil.copy2(source_file, found_csv)

            # Re-load data with auto_download=False to prevent infinite loop
            updated_df = load_stock_data(symbol, data_dir=root_path, auto_download=False, request_date=request_date)
            _, new_latest, _ = is_data_up_to_date(updated_df, request_date=request_date)
            logger.info(f"Data for {symbol} successfully refreshed through {new_latest}.")
            return updated_df
        except Exception as e:
            logger.error(f"Auto-download / update failed for '{symbol}': {e}")
            if df is not None:
                logger.warning(f"Falling back to existing local data for {symbol} (latest: {df['date'].max()}).")
                return df
            raise FileNotFoundError(
                f"Could not find or download stock data for symbol '{symbol}' in '{root_path}': {e}"
            )

    raise FileNotFoundError(
        f"Could not find stock data for symbol '{symbol}' in '{root_path}'. "
        f"Searched for CSV files and Qlib binary features."
    )


def _load_qlib_binary_data(feat_dir: Path, cal_file: Path, symbol: str) -> pd.DataFrame:
    """Read Qlib binary float32 features into a DataFrame."""
    calendar_dates = [line.strip() for line in cal_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not calendar_dates:
        raise ValueError(f"Calendar file {cal_file} is empty.")

    fields = ["open", "high", "low", "close", "volume", "factor", "change"]
    records = {}
    start_idx = None
    length = None

    for field in fields:
        bin_path = feat_dir / f"{field}.day.bin"
        if not bin_path.exists():
            continue
        arr = np.fromfile(str(bin_path), dtype="<f")
        if len(arr) < 2:
            continue
        field_start_idx = int(arr[0])
        field_vals = arr[1:]
        records[field] = field_vals
        if start_idx is None:
            start_idx = field_start_idx
            length = len(field_vals)

    if not records or start_idx is None or length is None:
        raise ValueError(f"No valid binary features found in {feat_dir}")

    end_idx = start_idx + length
    sym_dates = calendar_dates[start_idx:end_idx]

    data = {"date": sym_dates}
    for field, vals in records.items():
        data[field] = vals[:len(sym_dates)]

    df = pd.DataFrame(data)
    return _standardize_stock_df(df, symbol)


def _standardize_stock_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Ensure standard column names, date formatting, and chronological sort."""
    df.columns = [c.lower().strip() for c in df.columns]

    # Map column aliases
    col_map = {
        "datetime": "date",
        "timestamp": "date",
        "dt": "date",
        "adj close": "adjclose",
        "adj_close": "adjclose",
        "vol": "volume",
    }
    df = df.rename(columns=col_map)

    if "date" not in df.columns:
        # Check if index is date
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "date"})
        else:
            raise KeyError("Stock data missing 'date' column.")

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Ensure required numeric columns exist
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            if col == "volume":
                df["volume"] = 1.0
            elif col in ["open", "high", "low"] and "close" in df.columns:
                df[col] = df["close"]
            else:
                raise KeyError(f"Missing required price column '{col}' for {symbol}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["close"]).reset_index(drop=True)
    df["symbol"] = symbol
    return df


# ----------------------------------------------------------------------
# 2. Historical Performance Analytics (1Y, 3Y, 5Y)
# ----------------------------------------------------------------------

def compute_performance_summary(
    df: pd.DataFrame,
    periods_years: List[int] = [1, 3, 5],
    risk_free_rate: float = 0.03,
) -> Dict[str, Any]:
    """
    Compute comprehensive historical performance metrics over 1Y, 3Y, and 5Y historical periods.

    Metrics include:
    - Total Return (%)
    - CAGR (%)
    - Annualized Volatility (%)
    - Maximum Drawdown (%)
    - Sharpe Ratio
    - Calmar Ratio
    - Win Rate (% positive trading days)
    - Start & End Dates and Prices
    """
    if df.empty or len(df) < 5:
        raise ValueError("Insufficient data points to calculate performance.")

    df = df.copy().sort_values("date").reset_index(drop=True)
    latest_date = df["date"].iloc[-1]
    latest_close = float(df["close"].iloc[-1])

    # Convert date to datetime for lookback calculations
    dt_series = pd.to_datetime(df["date"])
    latest_dt = dt_series.iloc[-1]

    results = {
        "symbol": df["symbol"].iloc[0] if "symbol" in df.columns else "UNKNOWN",
        "latest_date": latest_date,
        "latest_close": latest_close,
        "total_history_days": len(df),
        "periods": {},
    }

    # Trading days per year approximation: 252
    trading_days_map = {1: 252, 3: 756, 5: 1260}

    for y in periods_years:
        target_days = trading_days_map.get(y, y * 252)
        # Use calendar cutoff or trading day cutoff (whichever is available)
        cal_cutoff = latest_dt - pd.DateOffset(years=y)
        sub_df = df[dt_series >= cal_cutoff].copy()

        if sub_df.empty or len(sub_df) < 20:
            # Fallback to trading days from end if calendar offset has fewer points
            sub_df = df.tail(min(len(df), target_days)).copy()

        if len(sub_df) < 20:
            results["periods"][f"{y}Y"] = {
                "available": False,
                "reason": f"Insufficient data for {y}-year analysis (only {len(sub_df)} days)",
            }
            continue

        p_start_date = sub_df["date"].iloc[0]
        p_end_date = sub_df["date"].iloc[-1]
        p_start_close = float(sub_df["close"].iloc[0])
        p_end_close = float(sub_df["close"].iloc[-1])

        # Total Return
        total_return = (p_end_close - p_start_close) / p_start_close

        # Time elapsed in years
        n_days = len(sub_df)
        years_elapsed = max(0.1, n_days / 252.0)

        # CAGR
        if p_start_close > 0 and p_end_close > 0:
            cagr = (p_end_close / p_start_close) ** (1.0 / years_elapsed) - 1.0
        else:
            cagr = 0.0

        # Daily Returns & Volatility
        daily_ret = sub_df["close"].pct_change().dropna()
        daily_vol = float(daily_ret.std())
        annual_vol = daily_vol * math.sqrt(252)

        # Sharpe Ratio
        excess_return = cagr - risk_free_rate
        sharpe = (excess_return / annual_vol) if annual_vol > 1e-6 else 0.0

        # Drawdown calculation
        cum_max = sub_df["close"].cummax()
        drawdown_series = (sub_df["close"] - cum_max) / cum_max
        max_drawdown = float(drawdown_series.min())

        # Calmar Ratio
        calmar = (cagr / abs(max_drawdown)) if abs(max_drawdown) > 1e-6 else 0.0

        # Win rate
        win_rate = float((daily_ret > 0).mean())

        # Highest and lowest in period
        highest_price = float(sub_df["high"].max() if "high" in sub_df.columns else sub_df["close"].max())
        lowest_price = float(sub_df["low"].min() if "low" in sub_df.columns else sub_df["close"].min())

        results["periods"][f"{y}Y"] = {
            "available": True,
            "years": y,
            "trading_days": n_days,
            "start_date": p_start_date,
            "end_date": p_end_date,
            "start_price": p_start_close,
            "end_price": p_end_close,
            "highest_price": highest_price,
            "lowest_price": lowest_price,
            "total_return_pct": total_return * 100.0,
            "cagr_pct": cagr * 100.0,
            "annual_volatility_pct": annual_vol * 100.0,
            "max_drawdown_pct": max_drawdown * 100.0,
            "sharpe_ratio": round(sharpe, 2),
            "calmar_ratio": round(calmar, 2),
            "win_rate_pct": win_rate * 100.0,
        }

    return results


# ----------------------------------------------------------------------
# 3. Historical Best Buy Points Detection Engine
# ----------------------------------------------------------------------

def detect_historical_best_buys(
    df: pd.DataFrame,
    periods_years: List[int] = [1, 3, 5],
    min_gain_threshold: float = 0.15,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Identify the most optimal historical entry / buy points during 1Y, 3Y, and 5Y historical periods.

    Identifies:
    1. Cyclical Troughs: Major price bottoms preceding multi-month upward surges.
    2. Inflection Points: Extreme oversold pullbacks (RSI < 35) followed by sharp trend reversals.

    Each best buy point includes:
    - Entry Date & Price
    - Subsequent Peak Date & Price
    - Holding Days to Peak
    - Maximum Subsequent Gain (%)
    - Return to Present (%)
    - Rationale / Context
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    dt_series = pd.to_datetime(df["date"])
    latest_dt = dt_series.iloc[-1]
    latest_close = float(df["close"].iloc[-1])

    # Technical Indicators for Inflection Identification
    # 1. RSI (14)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi14"] = 100 - (100 / (1 + rs))

    # 2. Moving Averages
    df["sma50"] = df["close"].rolling(window=50, min_periods=10).mean()
    df["sma200"] = df["close"].rolling(window=200, min_periods=20).mean()

    # 3. Rolling Drawdown from 52-week high (252 days)
    df["roll_max252"] = df["close"].rolling(window=252, min_periods=30).max()
    df["drawdown252"] = (df["close"] - df["roll_max252"]) / df["roll_max252"]

    best_buys_by_period = {}

    for y in periods_years:
        cal_cutoff = latest_dt - pd.DateOffset(years=y)
        sub_indices = df[dt_series >= cal_cutoff].index
        if len(sub_indices) < 30:
            # Fallback to trading days
            target_len = min(len(df), y * 252)
            sub_indices = df.tail(target_len).index

        if len(sub_indices) < 30:
            best_buys_by_period[f"{y}Y"] = []
            continue

        start_idx = sub_indices[0]
        end_idx = sub_indices[-1]

        candidates = []

        # Scan for local minimums with a rolling window of 21 days (approx 1 trading month)
        window = 21
        for i in range(start_idx + window, end_idx - 5):
            curr_price = df.loc[i, "close"]
            prev_window_min = df.loc[i - window : i, "close"].min()

            # Check if this index is a local low
            if curr_price <= prev_window_min * 1.01:
                # Find subsequent highest peak within remaining period
                subsequent_slice = df.loc[i : end_idx, "close"]
                max_subsequent_price = float(subsequent_slice.max())
                max_subsequent_idx = subsequent_slice.idxmax()
                gain = (max_subsequent_price - curr_price) / curr_price

                if gain >= min_gain_threshold:
                    entry_date = df.loc[i, "date"]
                    peak_date = df.loc[max_subsequent_idx, "date"]
                    holding_days = int((pd.to_datetime(peak_date) - pd.to_datetime(entry_date)).days)
                    return_to_now = (latest_close - curr_price) / curr_price

                    # Determine rationale
                    rsi_val = df.loc[i, "rsi14"]
                    dd_val = df.loc[i, "drawdown252"]
                    if dd_val <= -0.20:
                        rationale = f"Major Market Correction Trough ({dd_val*100:.1f}% Drawdown)"
                    elif rsi_val <= 35:
                        rationale = f"Deep Oversold Rebound (RSI {rsi_val:.1f})"
                    elif curr_price < df.loc[i, "sma200"]:
                        rationale = "Value Accumulation Below 200-Day Moving Average"
                    else:
                        rationale = "Key Cyclical Support Bounce"

                    candidates.append({
                        "date": entry_date,
                        "price": round(float(curr_price), 2),
                        "peak_date": peak_date,
                        "peak_price": round(max_subsequent_price, 2),
                        "holding_days": holding_days,
                        "max_gain_pct": round(gain * 100.0, 2),
                        "return_to_now_pct": round(return_to_now * 100.0, 2),
                        "rationale": rationale,
                        "quality_score": gain * 1.5 + (return_to_now if return_to_now > 0 else 0),
                    })

        # Filter out clusters within 30 days of each other, keeping the highest quality / lowest price point
        filtered_buys = []
        candidates.sort(key=lambda x: x["date"])

        for cand in candidates:
            if not filtered_buys:
                filtered_buys.append(cand)
            else:
                last = filtered_buys[-1]
                day_diff = (pd.to_datetime(cand["date"]) - pd.to_datetime(last["date"])).days
                if day_diff < 25:
                    # Keep the one with lower price / higher gain
                    if cand["max_gain_pct"] > last["max_gain_pct"]:
                        filtered_buys[-1] = cand
                else:
                    filtered_buys.append(cand)

        # Sort by maximum gain descending and keep top opportunities
        filtered_buys.sort(key=lambda x: x["max_gain_pct"], reverse=True)
        # Re-sort chronologically for presentation
        top_buys = sorted(filtered_buys[:5], key=lambda x: x["date"])
        best_buys_by_period[f"{y}Y"] = top_buys

    return best_buys_by_period


# ----------------------------------------------------------------------
# 4. 3-Month Forward Predictive Buy Analysis Engine
# ----------------------------------------------------------------------

def predict_future_buy_timing(
    df: pd.DataFrame,
    forecast_days: int = 63,  # ~3 months (21 trading days / month)
    simulations: int = 1000,
    regime: Optional[Dict[str, Any]] = None,
    microstructure: Optional[Dict[str, Any]] = None,
    derivatives: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Perform quantitative and machine-learning predictive analysis on when the stock
    should be bought within the next 3 months (~63 trading days) from the current date,
    conditioned on Bayesian Online Changepoint Detection (BOCD) regime states,
    institutional microstructure (Anchored VWAP & Volume Profile), and
    Dealer Gamma Exposure (GEX) option market structure.

    Outputs:
    - Projected 3-month daily price path (10th percentile bear, 50th median, 90th bull).
    - BOCD regime state & forward changepoint hazard probabilities.
    - Dealer Gamma Exposure regime (+GEX stabilizer vs -GEX accelerant).
    - Call Gamma Wall, Put Gamma Wall, and Gamma Flip levels.
    - Optimal Entry Price Range (support level / pullback target).
    - Optimal Buy Window (estimated date range within the next 3 months).
    - 3-Month Target Price and Expected Upside %.
    - Stop-Loss Invalidation Price.
    - Risk/Reward Ratio.
    - Tactical Recommendation Rating conditioned on market regime and options flow.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    if len(df) < 50:
        raise ValueError("Insufficient data points for 3-month predictive forecasting (minimum 50 required).")

    latest_date_str = df["date"].iloc[-1]
    latest_dt = pd.to_datetime(latest_date_str)
    current_price = float(df["close"].iloc[-1])

    # 1. Feature Engineering
    # Trend indicators
    sma20 = float(df["close"].rolling(20).mean().iloc[-1])
    sma50 = float(df["close"].rolling(50).mean().iloc[-1])
    sma200 = float(df["close"].rolling(200, min_periods=30).mean().iloc[-1])

    # Volatility & Momentum
    recent_returns = df["close"].pct_change().dropna()
    daily_vol = float(recent_returns.tail(60).std())
    drift = float(recent_returns.tail(60).mean())

    # Bollinger Bands
    rolling_std20 = float(df["close"].rolling(20).std().iloc[-1])
    bb_upper = sma20 + 2 * rolling_std20
    bb_lower = sma20 - 2 * rolling_std20
    pct_b = (current_price - bb_lower) / (bb_upper - bb_lower + 1e-9)

    # RSI (14)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    current_rsi = float((100 - (100 / (1 + rs))).iloc[-1])

    # 2. Parse BOCD Regime Parameters
    regime_state = None
    regime_name = None
    cp_hazard_pct = None
    forward_cp_prob = None
    exp_run_length = 63.0
    risk_mult = 1.0

    if regime and isinstance(regime, dict):
        regime_state = regime.get("state")
        regime_name = regime.get("name")
        cp_hazard_pct = float(regime.get("changepoint_prob_pct", 0.0))
        exp_run_length = float(regime.get("expected_run_length_days", 63.0))
        risk_mult = float(regime.get("risk_multiplier", 1.0))
        vol_21d_pct = regime.get("vol_21d_pct")
        if vol_21d_pct is not None and vol_21d_pct > 0:
            daily_vol = (float(vol_21d_pct) / 100.0) / math.sqrt(252.0)

        vol_ratio = float(regime.get("vol_ratio", 1.0))
        if vol_ratio > 1.15:
            daily_vol *= 1.15

        # Bayesian Online Changepoint Detection cumulative hazard over the forecast horizon
        h_daily = 1.0 / max(10.0, exp_run_length)
        forward_cp_prob = (1.0 - (1.0 - h_daily) ** forecast_days) * 100.0
    else:
        h_daily = 0.0

    # 2.5 Parse Dealer Gamma Exposure (GEX) Parameters
    gex_regime_state = 0
    gex_vol_mult = 1.0
    call_wall = None
    put_wall = None
    gamma_flip = None
    max_pain = None
    net_gex_m = 0.0
    gex_regime_desc = None

    if derivatives and isinstance(derivatives, dict):
        gex_data = derivatives.get("gex", derivatives)
        if isinstance(gex_data, dict):
            net_gex_m = float(gex_data.get("net_gex_millions", gex_data.get("net_gex_dollar_per_1pct", 0.0) / 1e6))
            gex_regime_desc = gex_data.get("regime", "")
            call_wall = gex_data.get("call_wall", gex_data.get("call_wall_strike"))
            put_wall = gex_data.get("put_wall", gex_data.get("put_wall_strike"))
            gamma_flip = gex_data.get("gamma_flip_price")
            max_pain = gex_data.get("max_pain", gex_data.get("max_pain_strike"))

            # In +GEX (dealers long gamma), dealers counter-trade, suppressing volatility (0.85x)
            # In -GEX (dealers short gamma), dealers pro-cyclically hedge, amplifying volatility (1.25x)
            if "+GEX" in gex_regime_desc or net_gex_m > 0:
                gex_regime_state = 1
                gex_vol_mult = 0.85
            elif "-GEX" in gex_regime_desc or net_gex_m < 0:
                gex_regime_state = -1
                gex_vol_mult = 1.25
            else:
                gex_regime_state = gex_data.get("regime_state", 0)
                if gex_regime_state > 0:
                    gex_vol_mult = 0.85
                elif gex_regime_state < 0:
                    gex_vol_mult = 1.25

    # 3. Generate future business/trading dates
    future_dates = []
    curr = latest_dt
    while len(future_dates) < forecast_days:
        curr += pd.Timedelta(days=1)
        if curr.weekday() < 5:  # Monday to Friday
            future_dates.append(curr.strftime("%Y-%m-%d"))

    # 4. Monte Carlo & Trend Decomposition Simulation
    # Geometric Brownian Motion with Mean-Reversion Component + BOCD Regime Jump Shocks
    np.random.seed(42)
    daily_vol_clamped = max(0.005, min(0.045, daily_vol * gex_vol_mult))
    # Drift conditioned on regime risk multiplier
    adj_drift = max(-0.0015, min(0.0015, drift * risk_mult))

    price_paths = np.zeros((simulations, forecast_days))
    price_paths[:, 0] = current_price

    for t in range(1, forecast_days):
        z = np.random.standard_normal(simulations)
        # Pull toward 50-day / 200-day trend channel
        reversion = 0.02 * (sma50 - price_paths[:, t - 1]) / price_paths[:, t - 1]
        step_return = adj_drift + reversion + daily_vol_clamped * z

        # BOCD structural regime changepoint jump shocks:
        # With hazard probability h_daily, paths experience a fat-tailed structural regime break
        if h_daily > 0:
            jump_occurred = np.random.rand(simulations) < h_daily
            if np.any(jump_occurred):
                # Regime jump shock scale: 1.5x daily vol, with negative skew in Risk-Off
                jump_direction = -0.5 * daily_vol_clamped if regime_state == 2 else 0.0
                jump_shocks = np.random.laplace(loc=jump_direction, scale=1.5 * daily_vol_clamped, size=simulations)
                step_return += jump_occurred * jump_shocks

        price_paths[:, t] = price_paths[:, t - 1] * (1.0 + step_return)

    # Percentiles
    p10_bear = np.percentile(price_paths, 10, axis=0)
    p50_median = np.percentile(price_paths, 50, axis=0)
    p90_bull = np.percentile(price_paths, 90, axis=0)

    # 5. Optimal Buy Timing & Entry Zone Identification
    # Identify projected dip / trough in the median trajectory within next 3 months
    min_median_idx = int(np.argmin(p50_median[:40]))  # Look within the first ~2 months for entry
    min_median_price = float(p50_median[min_median_idx])

    # Dynamic Support Levels
    recent_low_60d = float(df["close"].tail(60).min())
    key_support = max(recent_low_60d, bb_lower, min(sma50, current_price * 0.96))
    resistance = max(bb_upper, float(df["close"].tail(60).max()), current_price * 1.05)

    if microstructure and isinstance(microstructure, dict):
        avwap_ytd = microstructure.get("avwap", {}).get("ytd", {})
        ytd_lower = avwap_ytd.get("lower_1s")
        ytd_upper = avwap_ytd.get("upper_1s")

        if ytd_lower and not pd.isna(ytd_lower):
            key_support = max(key_support, float(ytd_lower))
        if ytd_upper and not pd.isna(ytd_upper):
            resistance = max(resistance, float(ytd_upper))

        vp = microstructure.get("volume_profile", {})
        val = vp.get("val")
        vah = vp.get("vah")
        if val and not pd.isna(val) and val < current_price:
            key_support = max(key_support, float(val))
        if vah and not pd.isna(vah) and vah > current_price:
            resistance = max(resistance, float(vah))

    # Incorporate Institutional Dealer Gamma Exposure (GEX) Walls
    if derivatives and isinstance(derivatives, dict):
        if put_wall and not pd.isna(put_wall) and put_wall < current_price:
            key_support = max(key_support, float(put_wall))
        if max_pain and not pd.isna(max_pain) and max_pain < current_price * 0.98:
            key_support = max(key_support, float(max_pain))
        if call_wall and not pd.isna(call_wall) and call_wall > current_price:
            resistance = max(resistance, float(call_wall))

    # Check regime state first, then fallback to technical RSI/Bollinger Bands
    if regime_state == 2:
        # High-Vol Liquidation / Risk-Off
        recommendation = "RISK-OFF / CAPITAL PRESERVATION"
        action_summary = (
            f"Active High-Volatility Liquidation Regime (BOCD State 2, {cp_hazard_pct:.1f}% hazard). "
            f"Capital preservation is paramount. Delay large allocations until volatility structure normalizes."
        )
        entry_low = round(min(key_support * 0.94, current_price * 0.90), 2)
        entry_high = round(min(key_support * 0.98, current_price * 0.94), 2)
        # Delay entry window to allow liquidation phase to complete
        opt_window_start = future_dates[min(15, forecast_days - 1)]
        opt_window_end = future_dates[min(35, forecast_days - 1)]
    elif regime_state == 3 or (cp_hazard_pct is not None and cp_hazard_pct >= 35.0):
        # Regime Transition / Inflection Alert
        recommendation = "REGIME SHIFT ALERT / PAUSE ENTRIES"
        action_summary = (
            f"Bayesian changepoint alert active ({cp_hazard_pct:.1f}% instant hazard). "
            f"Market structure is in an inflection phase; pause new entries until run-length stabilizes."
        )
        entry_low = round(min(key_support * 0.96, current_price * 0.93), 2)
        entry_high = round(min(key_support * 0.99, current_price * 0.96), 2)
        opt_window_start = future_dates[min(10, forecast_days - 1)]
        opt_window_end = future_dates[min(25, forecast_days - 1)]
    elif regime_state == 0:
        # Low-Vol Trending Bull
        if current_rsi > 70 or pct_b > 0.85:
            recommendation = "BUY ON PULLBACK"
            action_summary = (
                f"Bull trend active (BOCD State 0), but short-term overbought (RSI {current_rsi:.1f}). "
                f"Wait for shallow pullback toward support before adding exposure."
            )
            entry_low = round(max(key_support, current_price * 0.96), 2)
            entry_high = round(current_price * 0.985, 2)
            opt_window_start = future_dates[min(3, forecast_days - 1)]
            opt_window_end = future_dates[min(15, forecast_days - 1)]
        else:
            recommendation = "STRONG BUY / TREND ACCUMULATION"
            action_summary = "Low-volatility expansion with supportive macro liquidity (BOCD State 0). Accumulate with high confidence."
            entry_low = round(current_price * 0.98, 2)
            entry_high = round(current_price * 1.01, 2)
            opt_window_start = future_dates[0]
            opt_window_end = future_dates[min(12, forecast_days - 1)]
    elif regime_state == 1:
        # Mean-Reverting Choppy Neutral
        recommendation = "RANGE ACCUMULATION / BUY SUPPORT"
        action_summary = "Range-bound consolidation regime (BOCD State 1). Accumulate near support and trim near resistance."
        entry_low = round(key_support * 0.98, 2)
        entry_high = round(key_support * 1.02, 2)
        opt_window_start = future_dates[5]
        opt_window_end = future_dates[min(20, forecast_days - 1)]
    else:
        # Fallback technical rules when regime is not provided
        if current_rsi < 35 or pct_b < 0.15:
            recommendation = "STRONG BUY"
            action_summary = "Stock is currently oversold near major technical support. Immediate entry recommended."
            entry_low = current_price * 0.985
            entry_high = current_price * 1.01
            opt_window_start = future_dates[0]
            opt_window_end = future_dates[min(10, forecast_days - 1)]
        elif current_rsi > 70 or pct_b > 0.85:
            recommendation = "BUY ON PULLBACK"
            action_summary = (
                f"Stock is currently in short-term overbought territory (RSI {current_rsi:.1f}). "
                f"Wait for a pullback toward the projected support zone before deploying capital."
            )
            entry_low = round(min(key_support, current_price * 0.94), 2)
            entry_high = round(current_price * 0.975, 2)
            dip_center = max(5, min_median_idx)
            opt_window_start = future_dates[max(0, dip_center - 5)]
            opt_window_end = future_dates[min(forecast_days - 1, dip_center + 7)]
        elif current_price > sma50 and sma50 > sma200:
            recommendation = "ACCUMULATE / DIP BUY"
            action_summary = "Healthy uptrend in place above major moving averages. Accumulate on any shallow dip."
            entry_low = round(max(key_support, current_price * 0.96), 2)
            entry_high = round(current_price * 0.995, 2)
            opt_window_start = future_dates[2]
            opt_window_end = future_dates[min(20, forecast_days - 1)]
        else:
            recommendation = "HOLD / CAUTIOUS BUY"
            action_summary = "Consolidation phase. Accumulate cautiously near tested support levels."
            entry_low = round(key_support * 0.98, 2)
            entry_high = round(key_support * 1.02, 2)
            opt_window_start = future_dates[5]
            opt_window_end = future_dates[min(25, forecast_days - 1)]

    # Incorporate Dealer Gamma Exposure into tactical action summary
    if derivatives and isinstance(derivatives, dict):
        if gex_regime_state < 0 and gamma_flip is not None:
            action_summary += (
                f" [GEX Alert: -GEX regime active ({net_gex_m:+.1f}M/1%). Dealer dynamic hedging accelerates drops below "
                f"Gamma Flip ${gamma_flip:.2f}; enforce strict stop-loss rules.]"
            )
        elif gex_regime_state > 0 and put_wall is not None and call_wall is not None:
            action_summary += (
                f" [GEX Note: +GEX regime active (+${net_gex_m:.1f}M/1%). Dealer counter-trading pins price between Put Wall "
                f"${put_wall:.2f} and Call Wall ${call_wall:.2f}.]"
            )

    target_price_3m = round(float(p50_median[-1]), 2)
    expected_gain_pct = round(((target_price_3m - current_price) / current_price) * 100.0, 2)
    stop_loss = round(float(min(key_support * 0.96, entry_low * 0.96)), 2)
    downside_risk = abs((stop_loss - current_price) / current_price)
    upside_reward = max(0.01, (target_price_3m - current_price) / current_price)
    risk_reward = round(upside_reward / (downside_risk + 1e-6), 2)

    # Forecast points array for charting
    forecast_points = []
    for i, f_date in enumerate(future_dates):
        forecast_points.append({
            "date": f_date,
            "bear_p10": round(float(p10_bear[i]), 2),
            "median_p50": round(float(p50_median[i]), 2),
            "bull_p90": round(float(p90_bull[i]), 2),
        })

    return {
        "current_price": current_price,
        "current_date": latest_date_str,
        "current_rsi": round(current_rsi, 1),
        "sma50": round(sma50, 2),
        "sma200": round(sma200, 2),
        "key_support": round(key_support, 2),
        "key_resistance": round(resistance, 2),
        "recommendation": recommendation,
        "action_summary": action_summary,
        "optimal_entry_range": [round(entry_low, 2), round(entry_high, 2)],
        "optimal_buy_window": {
            "start_date": opt_window_start,
            "end_date": opt_window_end,
            "description": f"Between {opt_window_start} and {opt_window_end}",
        },
        "target_price_3m": target_price_3m,
        "expected_return_pct": expected_gain_pct,
        "stop_loss": stop_loss,
        "risk_reward_ratio": risk_reward,
        "forecast_days": forecast_days,
        "forecast_series": forecast_points,
        "bocd_regime_state": regime_state,
        "bocd_regime_name": regime_name,
        "bocd_changepoint_hazard_pct": cp_hazard_pct,
        "bocd_forward_changepoint_prob_pct": round(forward_cp_prob, 1) if forward_cp_prob is not None else None,
        "bocd_expected_run_length_days": round(exp_run_length, 1) if regime else None,
        "dealer_gex_regime": gex_regime_desc,
        "gex_regime": gex_regime_desc,
        "dealer_net_gex_m": net_gex_m,
        "call_gamma_wall": round(float(call_wall), 2) if call_wall is not None and not pd.isna(call_wall) else None,
        "call_wall_price": round(float(call_wall), 2) if call_wall is not None and not pd.isna(call_wall) else None,
        "put_gamma_wall": round(float(put_wall), 2) if put_wall is not None and not pd.isna(put_wall) else None,
        "put_wall_price": round(float(put_wall), 2) if put_wall is not None and not pd.isna(put_wall) else None,
        "gamma_flip_price": round(float(gamma_flip), 2) if gamma_flip is not None and not pd.isna(gamma_flip) else None,
        "max_pain_price": round(float(max_pain), 2) if max_pain is not None and not pd.isna(max_pain) else None,
        "gex_vol_multiplier": gex_vol_mult,
    }



# ----------------------------------------------------------------------
# 5. Multi-Period Projections & Probability Scoring
# ----------------------------------------------------------------------

def compute_multi_period_projections(
    df: pd.DataFrame,
    horizons: Optional[Dict[str, int]] = None,
    regime: Optional[Dict[str, Any]] = None,
    microstructure: Optional[Dict[str, Any]] = None,
    derivatives: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Project future returns over multiple investment horizons:
    - 6 Months (~126 trading days)
    - 1 Year (~252 trading days)
    - 2 Years (~504 trading days)
    - 3 Years (~756 trading days)

    Dynamically conditions forward drift and volatility on:
    - Bayesian Online Changepoint Detection (BOCD) regime risk multipliers
    - Multi-horizon realized volatility surfaces (21d vol vs 5d inversion)
    - Anchored VWAP (AVWAP) standardized Z-score mean-reversion adjustments
    - Volume Profile liquidity void dispersion expansion
    - Dealer Gamma Exposure (GEX) regime compression vs expansion
    - Cumulative forward changepoint hazard probability per horizon

    For each horizon, calculates:
    - Expected Projected Return (%) & Projected CAGR (%)
    - Base Target Price (p50), Bull Target Price (p90), Bear Target Price (p10)
    - Probability Score (% chance of positive return via Gaussian erf)
    - Qualitative Confidence Level
    - Conditioned effective drift and volatility
    - Forward BOCD Changepoint Hazard Probability (%)
    """
    if horizons is None:
        horizons = {
            "6M": 126,
            "1Y": 252,
            "2Y": 504,
            "3Y": 756,
        }

    df = df.copy().sort_values("date").reset_index(drop=True)
    if len(df) < 30:
        raise ValueError("Insufficient data for multi-period projections (minimum 30 days required).")

    current_price = float(df["close"].iloc[-1])
    
    # Daily returns & annualized statistics
    daily_returns = df["close"].pct_change().dropna()
    daily_vol = float(daily_returns.std()) if len(daily_returns) > 1 else 0.015
    ann_vol = daily_vol * math.sqrt(252)

    # Estimate baseline secular drift:
    total_days = len(df)
    if total_days >= 756:  # >= 3 years
        years = total_days / 252.0
        cagr_historical = (current_price / float(df["close"].iloc[0])) ** (1.0 / years) - 1.0
    elif total_days >= 252:
        cagr_historical = (current_price / float(df["close"].iloc[-252])) - 1.0
    else:
        cagr_historical = float(daily_returns.mean() * 252)

    # Momentum over recent 252 days (or available)
    lookback = min(252, len(daily_returns))
    recent_annualized = float(daily_returns.tail(lookback).mean() * 252)

    # Baseline secular blended drift
    raw_drift = 0.50 * cagr_historical + 0.35 * recent_annualized + 0.15 * 0.10
    base_annual_drift = max(-0.15, min(0.35, raw_drift))

    # 1. Parse Regime Conditioning Factors
    regime_risk_mult = 1.0
    regime_vol_override = None
    vol_inversion_penalty = 0.0
    regime_state_name = None
    regime_state = None
    exp_run_length = 63.0

    if regime and isinstance(regime, dict):
        regime_state = regime.get("state")
        regime_state_name = regime.get("name", "Unknown")
        regime_risk_mult = float(regime.get("risk_multiplier", 1.0))
        exp_run_length = float(regime.get("expected_run_length_days", 63.0))
        vol_21d_pct = regime.get("vol_21d_pct")
        if vol_21d_pct is not None and vol_21d_pct > 0:
            regime_vol_override = float(vol_21d_pct) / 100.0

        vol_ratio = float(regime.get("vol_ratio", 1.0))
        if vol_ratio > 1.15:
            # Volatility surface inversion (short-term stress) adds temporary dispersion
            vol_inversion_penalty = 0.03

    # Daily hazard rate for BOCD forward changepoint probability
    h_daily = 1.0 / max(10.0, exp_run_length)

    # 2. Parse Microstructure (AVWAP & Volume Profile) Factors
    avwap_z = None
    in_liquidity_void = False
    if microstructure and isinstance(microstructure, dict):
        avwap_ytd = microstructure.get("avwap", {}).get("ytd", {})
        avwap_z = avwap_ytd.get("zscore")
        vp = microstructure.get("volume_profile", {})
        in_liquidity_void = bool(vp.get("in_liquidity_void", False))

    results = {}
    horizon_labels = {
        "6M": "6 Months",
        "1Y": "1 Year",
        "2Y": "2 Years",
        "3Y": "3 Years",
    }

    for key, days in horizons.items():
        t = days / 252.0  # Time in years

        # Horizon decay: Near-term horizons (6M, 1Y) are heavily influenced by current regime
        # and AVWAP extension; longer horizons (2Y, 3Y) gradually revert to long-term secular growth
        w_regime = math.exp(-0.75 * t)  # 6M: ~0.69, 1Y: ~0.47, 2Y: ~0.22, 3Y: ~0.10

        # Adjust drift based on regime risk multiplier
        effective_drift = base_annual_drift * (1.0 - w_regime * (1.0 - regime_risk_mult))

        # Adjust drift based on AVWAP Z-score (mean-reversion pull vs discount accumulation)
        delta_avwap = 0.0
        if avwap_z is not None:
            if avwap_z > 2.0:
                # Overbought extension above YTD AVWAP faces mean-reversion drag
                delta_avwap = -0.04 * w_regime
            elif -1.5 <= avwap_z < 0.0:
                # Institutional discount zone provides rebound support
                delta_avwap = 0.02 * w_regime
            elif avwap_z < -2.0:
                # Severe structural liquidation breakdown
                delta_avwap = -0.03 * w_regime
        effective_drift += delta_avwap
        effective_drift = max(-0.25, min(0.35, effective_drift))

        # Effective volatility
        if regime_vol_override is not None:
            # Blend long-term vol with current regime realized vol surface
            horizon_vol = (1.0 - w_regime * 0.70) * ann_vol + (w_regime * 0.70) * regime_vol_override
        else:
            horizon_vol = ann_vol

        horizon_vol += vol_inversion_penalty * w_regime
        if in_liquidity_void:
            horizon_vol *= (1.0 + 0.15 * w_regime)  # Thin market depth expands dispersion

        # GEX Volatility Conditioning for Near-Term Horizons
        if derivatives and isinstance(derivatives, dict):
            gex_state = derivatives.get("regime_state", 0)
            if gex_state > 0:
                horizon_vol *= (1.0 - 0.10 * w_regime)  # +GEX compresses dispersion
            elif gex_state < 0:
                horizon_vol *= (1.0 + 0.15 * w_regime)  # -GEX expands dispersion

        ann_vol_clamped = max(0.12, min(0.70, horizon_vol))

        # Geometric Brownian Motion log-normal parameters
        # ln(S_t / S_0) ~ Normal((mu - 0.5 * sigma^2) * t, sigma^2 * t)
        mu_log = (effective_drift - 0.5 * (ann_vol_clamped ** 2)) * t
        sigma_log = ann_vol_clamped * math.sqrt(t)

        # Median Base Target Price (p50)
        base_price = current_price * math.exp(mu_log)
        # Bear Target Price (10th percentile, z = -1.28155)
        bear_price = current_price * math.exp(mu_log - 1.28155 * sigma_log)
        # Bull Target Price (90th percentile, z = +1.28155)
        bull_price = current_price * math.exp(mu_log + 1.28155 * sigma_log)

        # Expected return %
        expected_return_pct = ((base_price - current_price) / current_price) * 100.0
        # Projected annualized CAGR
        projected_cagr_pct = ((base_price / current_price) ** (1.0 / t) - 1.0) * 100.0

        # Probability Score: P(S_t > S_0) = P(ln(S_t / S_0) > 0)
        # Using math.erf for exact normal CDF computation
        z_score = mu_log / (sigma_log + 1e-9)
        prob_positive = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0))) * 100.0
        prob_score = round(max(5.0, min(95.0, prob_positive)), 1)

        # Qualitative Confidence Label
        if prob_score >= 80.0:
            confidence = "High Confidence"
            conf_color = "emerald"
        elif prob_score >= 65.0:
            confidence = "Moderate Confidence"
            conf_color = "blue"
        elif prob_score >= 50.0:
            confidence = "Balanced Outlook"
            conf_color = "amber"
        else:
            confidence = "Downside Risk"
            conf_color = "red"

        # Forward BOCD changepoint hazard probability over this horizon
        cp_prob_horizon = None
        if regime and isinstance(regime, dict):
            cp_prob_horizon = round((1.0 - (1.0 - h_daily) ** days) * 100.0, 1)

        results[key] = {
            "key": key,
            "label": horizon_labels.get(key, key),
            "trading_days": days,
            "years": round(t, 2),
            "projected_return_pct": round(expected_return_pct, 1),
            "projected_cagr_pct": round(projected_cagr_pct, 1),
            "base_target_price": round(base_price, 2),
            "bear_price": round(bear_price, 2),
            "bull_price": round(bull_price, 2),
            "probability_score": prob_score,
            "confidence": confidence,
            "conf_color": conf_color,
            "effective_drift_pct": round(effective_drift * 100.0, 1),
            "effective_vol_pct": round(ann_vol_clamped * 100.0, 1),
            "regime_conditioned": regime is not None,
            "microstructure_conditioned": microstructure is not None,
            "derivatives_conditioned": derivatives is not None,
            "dealer_gex_regime": derivatives.get("regime") if derivatives else None,
            "bocd_changepoint_prob_pct": cp_prob_horizon,
            "regime_state": regime_state,
            "regime_name": regime_state_name,
        }

    return results



# ----------------------------------------------------------------------
# 6. Market Regime & Changepoint Detection
# ----------------------------------------------------------------------

def detect_market_regime(
    df: pd.DataFrame,
    data_dir: Optional[Union[str, Path]] = None,
    symbol: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], pd.DataFrame]:
    """
    Apply Bayesian Online Changepoint Detection (BOCD) and multi-horizon
    realized volatility surface + macro credit spread analysis to classify market regime.

    Parameters
    ----------
    df : pd.DataFrame
        Stock DataFrame with 'date', 'close', and other OHLCV fields.
    data_dir : Optional[Union[str, Path]]
        Path to market data directory to locate HYG/IEI credit proxy CSVs.
    symbol : Optional[str]
        Current symbol being analyzed.

    Returns
    -------
    Tuple[Optional[Dict[str, Any]], pd.DataFrame]
        (regime_summary_dict, df_with_regime_features)
    """
    if MarketRegimeClassifier is None or df.empty or len(df) < 10:
        return None, df

    hyg_df = None
    iei_df = None
    if data_dir is not None:
        try:
            root_p = Path(data_dir).expanduser().resolve()
            for cand in [root_p / "HYG.csv", root_p / "source" / "HYG.csv", root_p / "normalize" / "HYG.csv"]:
                if cand.exists() and cand.is_file():
                    hyg_df = pd.read_csv(cand)
                    break
            for cand in [root_p / "IEI.csv", root_p / "source" / "IEI.csv", root_p / "normalize" / "IEI.csv"]:
                if cand.exists() and cand.is_file():
                    iei_df = pd.read_csv(cand)
                    break
        except Exception as e:
            logger.debug(f"Note loading credit ETFs: {e}")

    try:
        classifier = MarketRegimeClassifier(expected_run_length=63.0)
        df_regime = classifier.analyze(df, hyg_df=hyg_df, iei_df=iei_df)
        summary = classifier.get_current_regime_summary(df_regime)
        return summary, df_regime
    except Exception as e:
        logger.warning(f"Market regime analysis encountered an exception: {e}")
        return None, df

# ----------------------------------------------------------------------
# 6.5 Institutional Derivatives & Dealer Gamma Exposure (GEX)
# ----------------------------------------------------------------------

def compute_dealer_gex_features(
    df: pd.DataFrame,
    symbol: str,
    data_dir: Optional[Union[str, Path]] = None,
    r: float = 0.045,
) -> Optional[Dict[str, Any]]:
    """
    Compute institutional Dealer Gamma Exposure (GEX), Gamma Flip level,
    Put/Call Gamma Walls, and options volatility surface metrics for the symbol.
    """
    if OptionsDataLoader is None or df.empty or "close" not in df.columns:
        return None

    try:
        spot = float(df["close"].iloc[-1])
        log_ret = np.log(df["close"] / df["close"].shift(1))
        realized_vol_21d = float(log_ret.tail(21).std() * np.sqrt(252))
        if np.isnan(realized_vol_21d) or realized_vol_21d < 0.05:
            realized_vol_21d = 0.25

        loader = OptionsDataLoader(data_dir=data_dir)
        options_df = loader.load_or_generate_chain(
            symbol=symbol,
            spot=spot,
            realized_vol_21d=realized_vol_21d,
            r=r,
        )
        if options_df is None or options_df.empty:
            return None

        # GEX calculation
        gex_summary = compute_dealer_gex_summary(
            options_df=options_df,
            spot=spot,
            r=r,
            symbol=symbol,
        )

        # Volatility surface metrics
        vol_surface = VolatilitySurfaceFeatures.compute_surface_metrics(
            options_df=options_df,
            spot=spot,
            realized_vol_21d=realized_vol_21d,
            r=r,
        )

        # Merge results into unified derivatives dict
        gex_summary["vol_surface"] = vol_surface
        gex_summary["atm_iv_pct"] = vol_surface.get("atm_iv_pct", 25.0)
        gex_summary["vrp_pct"] = vol_surface.get("vrp_pct", 0.0)
        gex_summary["rr25_skew"] = vol_surface.get("rr25_skew", -2.0)
        gex_summary["skew_regime"] = vol_surface.get("skew_regime", "Normal Equity Skew")
        gex_summary["realized_vol_21d_pct"] = round(realized_vol_21d * 100.0, 2)
        gex_summary["gex"] = dict(gex_summary)
        return gex_summary
    except Exception as e:
        logger.warning(f"Dealer Gamma Exposure analysis encountered an exception: {e}")
        return None

# ----------------------------------------------------------------------
# 7. Master Analysis Coordinator
# ----------------------------------------------------------------------

def run_stock_analysis(
    symbol: str,
    data_dir: Union[str, Path],
    forecast_days: int = 63,
    auto_download: bool = True,
    start: str = "2000-01-01",
    request_date: Optional[Union[str, datetime.date, datetime.datetime]] = None,
) -> Dict[str, Any]:
    """
    Execute full historical performance, optimal entry detection,
    3-month forward predictive analysis, multi-period return projections,
    Bayesian Online Changepoint Detection (BOCD) market regime classification,
    and Institutional Dealer Gamma Exposure (GEX) derivatives analysis.
    Ensures that market data is up-to-date for the requested date before analyzing.
    """
    if request_date is None:
        req_date_str = datetime.date.today().strftime("%Y-%m-%d")
    elif isinstance(request_date, (datetime.date, datetime.datetime)):
        req_date_str = request_date.strftime("%Y-%m-%d")
    else:
        req_date_str = str(request_date)[:10]

    df = load_stock_data(
        symbol,
        data_dir,
        auto_download=auto_download,
        start=start,
        request_date=req_date_str,
    )
    is_fresh, latest_date, expected_date = is_data_up_to_date(df, request_date=req_date_str)

    # 1. Market Regime & Bayesian Changepoint Analysis
    regime_summary, df_enriched = detect_market_regime(df, data_dir=data_dir, symbol=symbol)
    df = df_enriched

    # 2. Institutional Microstructure (AVWAP & Volume Profile KDE)
    micro_summary = None
    if compute_microstructure_features is not None and not df.empty:
        try:
            df_micro, micro_summary = compute_microstructure_features(df)
            df = df_micro
        except Exception as e:
            logger.warning(f"Microstructure analysis encountered an exception: {e}")

    # 2.5 Institutional Derivatives & Dealer Gamma Exposure (GEX)
    derivatives_summary = compute_dealer_gex_features(df, symbol=symbol, data_dir=data_dir)

    # 3. Performance, buy timing, and forward projections
    perf_summary = compute_performance_summary(df, periods_years=[1, 3, 5])
    best_buys = detect_historical_best_buys(df, periods_years=[1, 3, 5])
    predictive = predict_future_buy_timing(
        df,
        forecast_days=forecast_days,
        regime=regime_summary,
        microstructure=micro_summary,
        derivatives=derivatives_summary,
    )
    projections = compute_multi_period_projections(
        df,
        regime=regime_summary,
        microstructure=micro_summary,
        derivatives=derivatives_summary,
    )

    return {
        "symbol": symbol.upper(),
        "request_date": req_date_str,
        "is_up_to_date": is_fresh,
        "latest_data_date": latest_date,
        "expected_trading_date": expected_date,
        "data_points": len(df),
        "historical_data": df,
        "performance": perf_summary,
        "best_buys": best_buys,
        "predictive": predictive,
        "projections": projections,
        "regime": regime_summary,
        "microstructure": micro_summary,
        "derivatives": derivatives_summary,
    }
