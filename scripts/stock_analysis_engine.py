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

try:
    from qlib.contrib.events import (
        EventCalendarEngine,
        PEADEngine,
        RiskDegrossingEngine,
        EventsDataLoader,
        compute_event_risk_features,
    )
except Exception:
    try:
        from events import (
            EventCalendarEngine,
            PEADEngine,
            RiskDegrossingEngine,
            EventsDataLoader,
            compute_event_risk_features,
        )
    except Exception:
        EventCalendarEngine = None
        PEADEngine = None
        RiskDegrossingEngine = None
        EventsDataLoader = None
        compute_event_risk_features = None

try:
    from scripts.infer_alpha158 import Alpha158Scorer
except Exception:
    try:
        from infer_alpha158 import Alpha158Scorer
    except Exception:
        Alpha158Scorer = None

try:
    from scripts.indicators import compute_rsi, compute_bollinger_bands, compute_rolling_drawdown
except Exception:
    try:
        from indicators import compute_rsi, compute_bollinger_bands, compute_rolling_drawdown
    except Exception:
        compute_rsi = None
        compute_bollinger_bands = None
        compute_rolling_drawdown = None

# Named Constants for Model Mechanics & Volatility Bounds
GEX_POS_VOL_DAMPENER: float = 0.85
GEX_NEG_VOL_ACCELERATOR: float = 1.25
VOL_MIN_CLAMP: float = 0.005
VOL_MAX_CLAMP: float = 0.045
DRIFT_MEAN_REVERSION_COEFF: float = 0.02
BOCD_JUMP_SCALE_MULT: float = 1.5
EARNINGS_GAP_SCALE_MULT: float = 2.5
Z_90TH_PERCENTILE: float = 1.28155


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
    # 1. RSI (14) using shared indicator module
    if compute_rsi is not None:
        df["rsi14"] = compute_rsi(df["close"], period=14)
    else:
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df["rsi14"] = 100 - (100 / (1 + rs))

    # 2. Moving Averages
    df["sma50"] = df["close"].rolling(window=50, min_periods=10).mean()
    df["sma200"] = df["close"].rolling(window=200, min_periods=20).mean()

    # 3. Rolling Drawdown from 52-week high (252 days) using shared indicator module
    if compute_rolling_drawdown is not None:
        df["drawdown252"] = compute_rolling_drawdown(df["close"], window=252)
    else:
        df["roll_max252"] = df["close"].rolling(window=252, min_periods=30).max()
        df["drawdown252"] = (df["close"] - df["roll_max252"]) / df["roll_max252"]

    # 4. Pre-computed Rolling Minimum for Vectorized O(n) Local Trough Detection
    window = 21
    df["roll_min21"] = df["close"].rolling(window=window).min()

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

        # Scan for local minimums with O(1) rolling minimum lookup
        for i in range(start_idx + window, end_idx - 5):
            curr_price = df.loc[i, "close"]
            prev_window_min = df.loc[i, "roll_min21"]

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

try:
    from scripts.predictive_engine import predict_future_buy_timing as _delegated_predict_future_buy_timing
except Exception:
    try:
        from predictive_engine import predict_future_buy_timing as _delegated_predict_future_buy_timing
    except Exception:
        _delegated_predict_future_buy_timing = None


def predict_future_buy_timing(
    df: pd.DataFrame,
    forecast_days: int = 63,  # ~3 months (21 trading days / month)
    simulations: int = 1000,
    regime: Optional[Dict[str, Any]] = None,
    microstructure: Optional[Dict[str, Any]] = None,
    derivatives: Optional[Dict[str, Any]] = None,
    events: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Perform quantitative and machine-learning predictive analysis on when the stock
    should be bought within the next 3 months (~63 trading days) from the current date,
    conditioned on Bayesian Online Changepoint Detection (BOCD) regime states,
    institutional microstructure (Anchored VWAP & Volume Profile),
    Dealer Gamma Exposure (GEX) option market structure, and
    Corporate Catalyst Risk / Post-Earnings Announcement Drift (PEAD).
    """
    if _delegated_predict_future_buy_timing is not None:
        return _delegated_predict_future_buy_timing(
            df=df,
            forecast_days=forecast_days,
            simulations=simulations,
            regime=regime,
            microstructure=microstructure,
            derivatives=derivatives,
            events=events,
        )
    raise RuntimeError("Predictive engine service could not be loaded from scripts.predictive_engine")


# ----------------------------------------------------------------------
# 5. Multi-Period Projections & Probability Scoring
# ----------------------------------------------------------------------

def compute_multi_period_projections(
    df: pd.DataFrame,
    horizons: Optional[Dict[str, int]] = None,
    regime: Optional[Dict[str, Any]] = None,
    microstructure: Optional[Dict[str, Any]] = None,
    derivatives: Optional[Dict[str, Any]] = None,
    events: Optional[Dict[str, Any]] = None,
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
    - Corporate Catalyst / Post-Earnings Announcement Drift (PEAD) momentum & gap dispersion
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

    # 3. Parse Corporate Catalyst & PEAD Factors
    pead_drift_boost = 0.0
    pead_regime = None
    event_gap_sd = 0.0
    catalyst_status_code = None
    event_haircut = 1.0
    if events and isinstance(events, dict):
        pead_dict = events.get("pead", {})
        pead_drift_boost = float(pead_dict.get("pead_drift_boost", 0.0))
        pead_regime = pead_dict.get("drift_regime")
        degross_dict = events.get("degrossing", {})
        event_gap_sd = float(degross_dict.get("binary_gap_sd", 0.0))
        event_haircut = float(degross_dict.get("position_haircut", 1.0))
        cat_dict = events.get("catalyst_status", {})
        catalyst_status_code = cat_dict.get("status_code", "SAFE")

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

        # Corporate Catalyst & PEAD drift boost for near-term horizons
        if pead_drift_boost != 0.0:
            effective_drift += pead_drift_boost * w_regime

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

        # Event Binary Gap Risk dispersion expansion for near-term horizons
        if event_gap_sd > 0.0:
            horizon_vol += (event_gap_sd * 0.5) * w_regime

        ann_vol_clamped = max(0.12, min(0.70, horizon_vol))

        # Geometric Brownian Motion log-normal parameters
        # ln(S_t / S_0) ~ Normal((mu - 0.5 * sigma^2) * t, sigma^2 * t)
        mu_log = (effective_drift - 0.5 * (ann_vol_clamped ** 2)) * t
        sigma_log = ann_vol_clamped * math.sqrt(t)

        # Median Base Target Price (p50)
        base_price = current_price * math.exp(mu_log)
        # Bear Target Price (10th percentile, z = -Z_90TH_PERCENTILE)
        bear_price = current_price * math.exp(mu_log - Z_90TH_PERCENTILE * sigma_log)
        # Bull Target Price (90th percentile, z = +Z_90TH_PERCENTILE)
        bull_price = current_price * math.exp(mu_log + Z_90TH_PERCENTILE * sigma_log)

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
            "events_conditioned": events is not None,
            "dealer_gex_regime": derivatives.get("regime") if derivatives else None,
            "pead_drift_regime": pead_regime,
            "catalyst_status": catalyst_status_code,
            "event_haircut": round(event_haircut, 2),
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

        adtv = float(df["volume"].tail(20).mean()) if "volume" in df.columns and not df["volume"].dropna().empty else None
        loader = OptionsDataLoader(data_dir=data_dir)
        options_df = loader.load_or_generate_chain(
            symbol=symbol,
            spot=spot,
            realized_vol_21d=realized_vol_21d,
            r=r,
            adtv=adtv,
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


def compute_event_features(
    df: pd.DataFrame,
    symbol: str = "STOCK",
    data_dir: Optional[Union[str, Path]] = None,
    current_date: Optional[Union[str, pd.Timestamp]] = None,
    bocd_changepoints: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Compute corporate catalyst awareness, risk de-grossing factors,
    PEAD drift dynamics, and key momentum events for main charting.
    """
    if compute_event_risk_features is None:
        return None
    try:
        return compute_event_risk_features(
            df=df,
            symbol=symbol,
            data_dir=data_dir,
            current_date=current_date,
            bocd_changepoints=bocd_changepoints,
        )
    except Exception as e:
        logger.warning(f"Event risk & PEAD analysis encountered an exception: {e}")
        return None


def compute_alpha158_features(
    symbol: str,
    as_of_date: Optional[Union[str, datetime.date, datetime.datetime]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Retrieve or evaluate LightGBM Alpha158 machine learning predictive score
    and cross-sectional ranking across the Russell 1000 universe.
    """
    if Alpha158Scorer is None:
        return None
    try:
        scorer = Alpha158Scorer()
        dt_str = str(as_of_date)[:10] if as_of_date is not None else None
        return scorer.get_score(symbol, as_of_date=dt_str)
    except Exception as e:
        logger.warning(f"Alpha158 scoring encountered an exception: {e}")
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

    # 2.75 Corporate Catalyst Awareness & PEAD Models
    bocd_cps = []
    if regime_summary and "changepoints" in regime_summary:
        bocd_cps = regime_summary["changepoints"]
    elif not df.empty and "regime_state" in df.columns and "changepoint_prob" in df.columns:
        prev_st = None
        last_idx = -999
        for i_row, r_data in df.iterrows():
            st = int(r_data.get("regime_state", 1))
            prob = float(r_data.get("changepoint_prob", 0.0))
            if prev_st is not None and st != prev_st and (i_row - last_idx >= 30):
                st_name = MarketRegimeClassifier.REGIME_NAMES.get(st, f"State {st}") if MarketRegimeClassifier else f"State {st}"
                bocd_cps.append({
                    "date": str(r_data["date"])[:10],
                    "state": st,
                    "name": st_name,
                    "description": f"Bayesian regime transition to {st_name} (CP hazard: {prob*100:.0f}%)",
                    "prob": prob,
                })
                last_idx = i_row
            prev_st = st

    event_summary = compute_event_features(
        df,
        symbol=symbol,
        data_dir=data_dir,
        current_date=latest_date,
        bocd_changepoints=bocd_cps,
    )

    # 2.85 LightGBM Alpha158 Cross-Sectional Machine Learning Score
    alpha158_summary = compute_alpha158_features(symbol=symbol, as_of_date=latest_date)

    # 3. Performance, buy timing, and forward projections
    perf_summary = compute_performance_summary(df, periods_years=[1, 3, 5])
    best_buys = detect_historical_best_buys(df, periods_years=[1, 3, 5])
    predictive = predict_future_buy_timing(
        df,
        forecast_days=forecast_days,
        regime=regime_summary,
        microstructure=micro_summary,
        derivatives=derivatives_summary,
        events=event_summary,
    )
    projections = compute_multi_period_projections(
        df,
        regime=regime_summary,
        microstructure=micro_summary,
        derivatives=derivatives_summary,
        events=event_summary,
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
        "events": event_summary,
        "alpha158": alpha158_summary,
    }
