#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stock Analysis JSON Data Contract Engine & CLI
==============================================
Generates and manages canonical, fully-serializable JSON data contracts for
stock performance, technical indicators, predictive buy timing, multi-period
forward projections, Bayesian Online Changepoint Detection (BOCD) market regimes,
market microstructure (AVWAP & Volume Profile KDE), Dealer Gamma Exposure (GEX),
and Corporate Event / Post-Earnings Announcement Drift (PEAD) models.

Provides headless execution for quantitative research, backtesting harnesses,
risk analytics engines, and decoupled front-end visualization pipelines.
"""

import os
import sys
import json
import logging
import argparse
import datetime
from pathlib import Path
from typing import Dict, Any, Union, Optional

import numpy as np
import pandas as pd

# Setup logging
logger = logging.getLogger("StockAnalysisData")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Ensure scripts and qlib/contrib directories are in path
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
CONTRIB_DIR = REPO_ROOT / "qlib" / "contrib"
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(CONTRIB_DIR) not in sys.path:
    sys.path.insert(0, str(CONTRIB_DIR))

from stock_analysis_engine import (
    run_stock_analysis,
    compute_multi_period_projections,
    compute_dealer_gex_features,
)

try:
    from qlib.contrib.derivatives import (
        evaluate_earnings_gamma_squeeze,
        DataProvenance,
        DealerGammaEngine,
        SyntheticOptionSurfaceGenerator,
        VolatilitySurfaceFeatures,
    )
except Exception:
    try:
        from derivatives import (
            evaluate_earnings_gamma_squeeze,
            DataProvenance,
            DealerGammaEngine,
            SyntheticOptionSurfaceGenerator,
            VolatilitySurfaceFeatures,
        )
    except Exception:
        evaluate_earnings_gamma_squeeze = None
        DataProvenance = None
        DealerGammaEngine = None
        SyntheticOptionSurfaceGenerator = None
        VolatilitySurfaceFeatures = None

try:
    from scripts.infer_alpha158 import Alpha158Scorer
except Exception:
    try:
        from infer_alpha158 import Alpha158Scorer
    except Exception:
        Alpha158Scorer = None


def resolve_json_path(
    symbol: str,
    report_dir: Optional[Union[str, Path]] = None,
    output: Optional[Union[str, Path]] = None,
    report_date: Optional[Union[str, datetime.date, datetime.datetime]] = None,
) -> Path:
    """
    Resolve the final JSON data file path matching the report name convention.
    Always produces a canonical path ending in .json.

    - If output is specified:
        - If it ends in .json (case-insensitive), use as the exact file path.
        - If it ends in .html (case-insensitive), replace suffix with .json.
        - Otherwise, treat output as a target directory and construct
          <output>/<SYMBOL>_analysis_report_<DATE>.json.
    - If report_dir is specified (or defaults to 'reports'), use
      <report_dir>/<SYMBOL>_analysis_report_<DATE>.json.
    - Missing parent directories are created automatically.
    """
    sym = symbol.upper()
    if report_date is None:
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    elif isinstance(report_date, (datetime.date, datetime.datetime)):
        date_str = report_date.strftime("%Y-%m-%d")
    else:
        date_str = str(report_date)[:10]

    filename = f"{sym}_analysis_report_{date_str}.json"

    if output:
        out_p = Path(output).expanduser().resolve()
        if out_p.suffix.lower() == ".json":
            out_p.parent.mkdir(parents=True, exist_ok=True)
            return out_p
        elif out_p.suffix.lower() == ".html":
            out_p.parent.mkdir(parents=True, exist_ok=True)
            return out_p.with_suffix(".json")
        else:
            out_p.mkdir(parents=True, exist_ok=True)
            return out_p / filename

    target_dir = Path(report_dir if report_dir else "reports").expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / filename


def _sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize Python objects into native JSON-serializable types.
    Handles NaN, Inf, NumPy scalars, Pandas DataFrames/Series, and Timestamps.
    """
    if obj is None:
        return None
    if isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (datetime.date, datetime.datetime, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, np.ndarray)):
        return [_sanitize_for_json(item) for item in obj]
    if isinstance(obj, pd.DataFrame):
        return [_sanitize_for_json(row) for row in obj.to_dict(orient="records")]
    if isinstance(obj, pd.Series):
        return [_sanitize_for_json(val) for val in obj.tolist()]
    return str(obj)


def prepare_analysis_json_payload(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare the canonical, fully-serializable JSON payload from stock analysis results.
    Guarantees consistent contract schema v1.0.0 across all sub-models:
    metadata, historical_data, performance, best_buys, predictive,
    projections, regime (BOCD), microstructure (AVWAP/Volume Profile),
    derivatives (GEX), and events (PEAD).
    """
    perf = analysis_data.get("performance", {})
    symbol = analysis_data.get("symbol", analysis_data.get("metadata", {}).get("symbol", "UNKNOWN"))
    req_date = analysis_data.get(
        "request_date", analysis_data.get("metadata", {}).get("request_date", perf.get("latest_date", ""))
    )
    if not req_date:
        req_date = datetime.date.today().strftime("%Y-%m-%d")
    latest_data_date = analysis_data.get(
        "latest_data_date",
        analysis_data.get("metadata", {}).get("latest_data_date", perf.get("latest_date", req_date)),
    )
    is_up_to_date = analysis_data.get(
        "is_up_to_date", analysis_data.get("metadata", {}).get("is_up_to_date", True)
    )
    forecast_days = analysis_data.get(
        "forecast_days", analysis_data.get("metadata", {}).get("forecast_days", 63)
    )

    raw_hist = analysis_data.get("historical_data")
    if isinstance(raw_hist, pd.DataFrame):
        df = raw_hist.copy()
        if "close" in df.columns:
            if "sma50" not in df.columns:
                df["sma50"] = df["close"].rolling(window=50, min_periods=50).mean()
            if "sma200" not in df.columns:
                df["sma200"] = df["close"].rolling(window=200, min_periods=200).mean()
        else:
            df["sma50"] = None
            df["sma200"] = None

        history_payload = []
        for _, row in df.iterrows():
            history_payload.append({
                "date": str(row["date"])[:10] if not pd.isna(row.get("date")) else "",
                "open": round(float(row["open"]), 2) if not pd.isna(row.get("open")) else 0.0,
                "high": round(float(row["high"]), 2) if not pd.isna(row.get("high")) else 0.0,
                "low": round(float(row["low"]), 2) if not pd.isna(row.get("low")) else 0.0,
                "close": round(float(row["close"]), 2) if not pd.isna(row.get("close")) else 0.0,
                "volume": int(row["volume"]) if not pd.isna(row.get("volume")) else 0,
                "sma50": round(float(row["sma50"]), 2) if not pd.isna(row.get("sma50")) else None,
                "sma200": round(float(row["sma200"]), 2) if not pd.isna(row.get("sma200")) else None,
                "avwap_ytd": (
                    round(float(row["avwap_ytd"]), 2)
                    if "avwap_ytd" in row and not pd.isna(row.get("avwap_ytd"))
                    else None
                ),
                "avwap_ytd_upper_1s": (
                    round(float(row["avwap_ytd_upper_1s"]), 2)
                    if "avwap_ytd_upper_1s" in row and not pd.isna(row.get("avwap_ytd_upper_1s"))
                    else None
                ),
                "avwap_ytd_lower_1s": (
                    round(float(row["avwap_ytd_lower_1s"]), 2)
                    if "avwap_ytd_lower_1s" in row and not pd.isna(row.get("avwap_ytd_lower_1s"))
                    else None
                ),
            })
    elif isinstance(raw_hist, list):
        history_payload = raw_hist
    else:
        history_payload = []

    derivatives = analysis_data.get("derivatives")
    if not derivatives:
        spot_val = 0.0
        if isinstance(raw_hist, pd.DataFrame) and not raw_hist.empty and "close" in raw_hist.columns:
            spot_val = float(raw_hist["close"].iloc[-1])
        elif perf and "latest_price" in perf:
            spot_val = float(perf["latest_price"])
        elif perf and "latest_close" in perf:
            spot_val = float(perf["latest_close"])

        if spot_val > 0.0:
            if compute_dealer_gex_features is not None and isinstance(raw_hist, pd.DataFrame) and not raw_hist.empty:
                try:
                    derivatives = compute_dealer_gex_features(raw_hist, symbol=symbol)
                except Exception as e:
                    logger.debug(f"compute_dealer_gex_features fallback: {e}")
            if not derivatives and DealerGammaEngine is not None and SyntheticOptionSurfaceGenerator is not None:
                try:
                    adtv_val = float(raw_hist["volume"].tail(20).mean()) if isinstance(raw_hist, pd.DataFrame) and "volume" in raw_hist.columns else None
                    chain = SyntheticOptionSurfaceGenerator.generate_synthetic_chain(
                        spot_price=spot_val,
                        symbol=symbol,
                        adtv=adtv_val,
                    )
                    engine = DealerGammaEngine()
                    derivatives = engine.compute_gex(chain, spot_price=spot_val)
                    if VolatilitySurfaceFeatures is not None:
                        vol = VolatilitySurfaceFeatures.compute_surface_metrics(chain, spot=spot_val, realized_vol_21d=0.25, r=0.045)
                        derivatives["vol_surface"] = vol
                        derivatives["atm_iv_pct"] = vol.get("atm_iv_pct", 25.0)
                        derivatives["vrp_pct"] = vol.get("vrp_pct", 0.0)
                        derivatives["rr25_skew"] = vol.get("rr25_skew", -2.0)
                        derivatives["skew_regime"] = vol.get("skew_regime", "Normal Equity Skew")
                    derivatives["is_synthetic_surface"] = True
                except Exception as e:
                    logger.debug(f"SyntheticOptionSurfaceGenerator fallback: {e}")

    projections = analysis_data.get("projections")
    if not projections and isinstance(raw_hist, pd.DataFrame):
        projections = compute_multi_period_projections(
            raw_hist,
            regime=analysis_data.get("regime"),
            microstructure=analysis_data.get("microstructure"),
            derivatives=derivatives or analysis_data.get("derivatives"),
        )

    earnings_gamma_squeeze = analysis_data.get("earnings_gamma_squeeze")
    if not earnings_gamma_squeeze:
        if evaluate_earnings_gamma_squeeze is not None:
            last_price = 100.0
            if isinstance(raw_hist, pd.DataFrame) and not raw_hist.empty and "close" in raw_hist.columns:
                last_price = float(raw_hist["close"].iloc[-1])
            elif perf and "current_price" in perf:
                last_price = float(perf["current_price"])

            vol_mean = 1_000_000.0
            if isinstance(raw_hist, pd.DataFrame) and not raw_hist.empty and "volume" in raw_hist.columns:
                vol_mean = float(raw_hist["volume"].tail(20).mean())

            sue_val = 0.0
            events_info = analysis_data.get("events", {})
            if events_info and "pead" in events_info and events_info["pead"]:
                sue_val = float(events_info["pead"].get("sue_score", 0.0))

            flip_val = 0.0
            deriv_info = analysis_data.get("derivatives", {})
            if deriv_info and "gamma_flip_price" in deriv_info:
                flip_val = float(deriv_info["gamma_flip_price"])

            prov_val = DataProvenance.SYNTHETIC_RESEARCH_FALLBACK if DataProvenance else "synthetic_research_fallback"
            chain_to_pass = pd.DataFrame()
            if SyntheticOptionSurfaceGenerator is not None:
                try:
                    chain_to_pass = SyntheticOptionSurfaceGenerator.generate_synthetic_chain(
                        spot_price=last_price,
                        annual_vol=0.25,
                        dte_days=30,
                        symbol=symbol,
                        adtv=vol_mean,
                    )
                except Exception as e:
                    logger.warning(f"Could not generate synthetic chain: {e}")
                    chain_to_pass = pd.DataFrame()

            earnings_gamma_squeeze = evaluate_earnings_gamma_squeeze(
                spot=last_price,
                df_chain=chain_to_pass,
                adtv_20=vol_mean,
                sue_score=sue_val,
                short_interest_pct=0.05,
                gamma_flip_price=flip_val,
                provenance=prov_val,
                is_pit_timestamp=True,
                event_date=str(req_date),
                reporting_time="AMC",
            )
        else:
            earnings_gamma_squeeze = {
                "is_actionable": False,
                "provenance": "synthetic_research_fallback",
                "safety_status": "ACTION_SUPPRESSED",
                "recommended_action": "RESEARCH_ONLY_NO_ACTION",
            }

    # Extract backtesting_protocol and evaluation_matrix
    backtesting_protocol = analysis_data.get(
        "backtesting_protocol",
        earnings_gamma_squeeze.get("backtesting_protocol", {})
    )
    evaluation_matrix = analysis_data.get(
        "evaluation_matrix",
        earnings_gamma_squeeze.get("evaluation_matrix", {})
    )

    alpha158_data = analysis_data.get("alpha158")
    if not alpha158_data and Alpha158Scorer is not None:
        try:
            alpha158_data = Alpha158Scorer().get_score(symbol, as_of_date=str(latest_data_date))
        except Exception:
            pass

    payload = {
        "metadata": {
            "symbol": symbol,
            "request_date": str(req_date),
            "latest_data_date": str(latest_data_date),
            "is_up_to_date": bool(is_up_to_date),
            "forecast_days": int(forecast_days),
            "generated_at": datetime.datetime.now().isoformat(),
            "contract_version": "1.2.0",
        },
        "symbol": symbol,
        "historical_data": history_payload,
        "performance": analysis_data.get("performance", {}),
        "best_buys": analysis_data.get("best_buys", []),
        "predictive": analysis_data.get("predictive", {}),
        "projections": projections or {},
        "regime": analysis_data.get("regime", {}),
        "microstructure": analysis_data.get("microstructure", {}),
        "derivatives": derivatives or analysis_data.get("derivatives", {}),
        "events": analysis_data.get("events", {}),
        "earnings_gamma_squeeze": earnings_gamma_squeeze or {},
        "backtesting_protocol": backtesting_protocol or {},
        "evaluation_matrix": evaluation_matrix or {},
        "alpha158": alpha158_data or {},
    }
    return _sanitize_for_json(payload)


def export_analysis_json(
    analysis_data: Dict[str, Any],
    json_path: Union[str, Path],
    indent: int = 2,
) -> Path:
    """
    Save the full canonical analysis data contract to a .json file on disk.

    Args:
        analysis_data: Raw or pre-sanitized analysis data dictionary.
        json_path: Target path for the output JSON file.
        indent: Indentation level for formatting (default: 2).

    Returns:
        Path to the written JSON file.
    """
    json_file = Path(json_path).expanduser().resolve()
    json_file.parent.mkdir(parents=True, exist_ok=True)
    payload = prepare_analysis_json_payload(analysis_data)
    json_text = json.dumps(payload, indent=indent, ensure_ascii=False)
    json_file.write_text(json_text, encoding="utf-8")
    logger.info(f"Canonical analysis dataset successfully exported to: {json_file}")
    return json_file


def load_analysis_json(json_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read and deserialize the canonical analysis data from a .json file.

    Args:
        json_path: Target path to the existing JSON file.

    Returns:
        Deserialized analysis data dictionary.

    Raises:
        FileNotFoundError: If the specified JSON file does not exist.
    """
    json_file = Path(json_path).expanduser().resolve()
    if not json_file.exists():
        raise FileNotFoundError(f"Analysis JSON file not found: {json_file}")
    json_text = json_file.read_text(encoding="utf-8")
    data = json.loads(json_text)
    logger.info(f"Loaded analysis dataset from: {json_file}")
    return data


def generate_stock_analysis_data(
    symbol: str,
    data_dir: Optional[Union[str, Path]] = None,
    report_dir: Optional[Union[str, Path]] = None,
    output: Optional[Union[str, Path]] = None,
    forecast_days: int = 63,
    auto_download: bool = True,
    start: str = "2000-01-01",
    request_date: Optional[Union[str, datetime.date, datetime.datetime]] = None,
    indent: int = 2,
) -> Path:
    """
    High-level orchestration function that executes the multi-model analytical engine
    and exports the canonical JSON data contract to disk.

    Args:
        symbol: Stock ticker symbol (e.g., AAPL, MSFT, NVDA).
        data_dir: Path to Qlib binary data directory.
        report_dir: Directory where JSON files will be stored.
        output: Explicit path or filename override for the JSON file.
        forecast_days: Forward trading days forecast (default: 63).
        auto_download: Automatically fetch missing or updated market data.
        start: Historical data start date (default: '2000-01-01').
        request_date: Evaluation date for analysis (default: today).
        indent: JSON indentation formatting level (default: 2).

    Returns:
        Path to the written canonical JSON file.
    """
    if data_dir is None:
        data_dir = Path("~/.qlib/qlib_data/us_data").expanduser()

    sym = symbol.upper()
    req_date = (
        request_date
        if request_date
        else datetime.date.today().strftime("%Y-%m-%d")
    )
    json_path = resolve_json_path(sym, report_dir=report_dir, output=output, report_date=req_date)

    logger.info(f"Executing analytical engine for {sym} (requested: {req_date})...")
    raw_analysis = run_stock_analysis(
        symbol=sym,
        data_dir=data_dir,
        forecast_days=forecast_days,
        auto_download=auto_download,
        start=start,
        request_date=req_date,
    )

    exported_path = export_analysis_json(raw_analysis, json_path, indent=indent)
    logger.info(f"[Step 1 Complete] Analysis dataset exported to: {exported_path}")
    return exported_path


def main():
    """
    Command-line interface for the stock analysis JSON data contract engine.
    """
    parser = argparse.ArgumentParser(
        description="Institutional Stock Analysis JSON Data Contract Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbol", "-s",
        type=str,
        required=True,
        help="Stock ticker symbol to analyze (e.g., AAPL, MSFT, NVDA).",
    )
    parser.add_argument(
        "--data_dir", "-d",
        type=str,
        default=os.path.expanduser("~/.qlib/qlib_data/us_data"),
        help="Path to the Qlib binary data directory.",
    )
    parser.add_argument(
        "--report_dir", "-r",
        type=str,
        default="reports",
        help="Directory to save the canonical JSON file.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Explicit file path or custom destination for the output .json file.",
    )
    parser.add_argument(
        "--days_forecast",
        type=int,
        default=63,
        help="Number of forward trading days to forecast (~3 months).",
    )
    parser.add_argument(
        "--auto_download",
        dest="auto_download",
        action="store_true",
        default=True,
        help="Automatically download updated data if missing or outdated.",
    )
    parser.add_argument(
        "--no-auto_download",
        dest="auto_download",
        action="store_false",
        help="Disable automatic data downloading.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2000-01-01",
        help="Start date for historical data in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--request_date", "--report_date",
        type=str,
        default=None,
        help="Evaluation date for analysis in YYYY-MM-DD format (default: today).",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level for formatted output.",
    )
    parser.add_argument(
        "--provenance",
        type=str,
        default="historical_opra_eod",
        choices=["live_opra_verified", "historical_opra_eod", "synthetic_research_fallback"],
        help="Options chain data provenance tier (default: 'historical_opra_eod').",
    )
    parser.add_argument(
        "--simulate_jump",
        type=float,
        default=None,
        help="Simulate specific spot jump percentage for earnings squeeze evaluation (e.g. 0.08 for +8%%).",
    )
    parser.add_argument(
        "--custom_iv_crush",
        type=float,
        default=None,
        help="Override historical IV crush ratio (e.g. 0.45 for 45%% drop).",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        default=False,
        help="Suppress terminal summary banners and progress logs.",
    )

    args = parser.parse_args()

    if args.quiet:
        logger.setLevel(logging.WARNING)

    symbol = args.symbol.upper()
    req_date = args.request_date if args.request_date else datetime.date.today().strftime("%Y-%m-%d")

    exported_file = generate_stock_analysis_data(
        symbol=symbol,
        data_dir=args.data_dir,
        report_dir=args.report_dir,
        output=args.output,
        forecast_days=args.days_forecast,
        auto_download=args.auto_download,
        start=args.start,
        request_date=req_date,
        indent=args.indent,
    )

    data = load_analysis_json(exported_file)
    meta = data.get("metadata", {})
    perf = data.get("performance", {})
    latest_data_date = meta.get("latest_data_date", perf.get("latest_date", ""))
    is_up_to_date = meta.get("is_up_to_date", True)

    gamma_data = data.get("earnings_gamma_squeeze", {})
    vol_surf = gamma_data.get("calibrate_post_earnings_volatility_surface", {})
    backtest = data.get("backtesting_protocol", {})
    dsr_data = backtest.get("deflated_sharpe_ratio", {})
    panel = backtest.get("verifiable_replication_event_panel", {})

    if not args.quiet:
        print("\n=======================================================")
        print(" STOCK ANALYSIS JSON DATA CONTRACT GENERATOR ")
        print("=======================================================")
        print(f"Symbol:           {symbol}")
        print(f"Report Requested: {req_date}")
        print(f"Data Directory:   {args.data_dir}")
        print(f"Data Freshness:   Through {latest_data_date} ({'Up-to-Date' if is_up_to_date else 'Latest available'})")
        print(f"Auto Download:    {args.auto_download}")
        print(f"Contract Version: {meta.get('contract_version', '1.2.0')}")
        print(f"Squeeze Model:    Next-Day to Next-Week (t+1 to t+5) Active")
        if vol_surf:
            print(f"Vol Surface:      Expected Jump: +{vol_surf.get('expected_jump_pct', 0.0)}% | Post-Event IV: {vol_surf.get('post_earnings_iv', 0.0)}")
        if dsr_data:
            print(f"Backtest DSR:     Sharpe: {dsr_data.get('best_sharpe', 0.0)} | Hurdle: {dsr_data.get('expected_max_sharpe_hurdle', 0.0)} | DSR Prob: {dsr_data.get('dsr_probability', 0.0)} ({'Significant p<0.05' if dsr_data.get('is_statistically_significant') else 'Inconclusive'})")
        if panel:
            print(f"Replication Panel: {panel.get('n_events', 0):,} Events | Win Rate: {panel.get('win_rate', 0.0)*100:.1f}% | Profit Factor: {panel.get('profit_factor', 0.0)}")
        print(f"Output Dataset:   {exported_file.resolve()}")
        print("=======================================================")
        print(f"\n[SUCCESS] Canonical JSON analysis dataset generated at: {exported_file.resolve()}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

