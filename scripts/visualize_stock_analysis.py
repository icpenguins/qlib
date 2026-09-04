#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interactive Stock Performance & Predictive Buy Timing Visual Display
===================================================================
Generates a standalone, self-contained interactive visual dashboard
reporting on 1Y, 3Y, and 5Y performance, historical best buy points,
and a 3-month forward predictive buy analysis from a specified data directory.
"""

import os
import sys
import json
import logging
import argparse
import datetime
import webbrowser
from pathlib import Path
from typing import Dict, Any, Union, Optional

import pandas as pd

logger = logging.getLogger("VisualizeStockAnalysis")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Ensure scripts directory is in path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from stock_analysis_engine import run_stock_analysis, compute_multi_period_projections


def resolve_report_path(
    symbol: str,
    report_dir: Optional[Union[str, Path]] = None,
    output: Optional[Union[str, Path]] = None,
    report_date: Optional[Union[str, datetime.date, datetime.datetime]] = None,
) -> Path:
    """
    Resolve the final HTML report file path with the request date.
    - If output is specified:
        - If it has an .html suffix, use as the exact output file path.
        - Otherwise, treat output as the target report directory and include the request date.
    - If report_dir is specified (or defaults to 'reports'), use <report_dir>/<SYMBOL>_analysis_report_<DATE>.html.
    - Missing directories are automatically created.
    """
    sym = symbol.upper()
    if report_date is None:
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    elif isinstance(report_date, (datetime.date, datetime.datetime)):
        date_str = report_date.strftime("%Y-%m-%d")
    else:
        date_str = str(report_date)

    filename = f"{sym}_analysis_report_{date_str}.html"

    if output:
        out_p = Path(output).expanduser().resolve()
        if out_p.suffix.lower() == ".html":
            out_p.parent.mkdir(parents=True, exist_ok=True)
            return out_p
        else:
            out_p.mkdir(parents=True, exist_ok=True)
            return out_p / filename

    target_dir = Path(report_dir if report_dir else "reports").expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / filename


def generate_html_dashboard(
    analysis_data: Dict[str, Any],
    output_path: Union[str, Path],
) -> Path:
    """
    Generate an interactive, zero-dependency, self-contained HTML dashboard.
    """
    output_file = Path(output_path).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    symbol = analysis_data["symbol"]
    perf = analysis_data["performance"]
    best_buys = analysis_data["best_buys"]
    pred = analysis_data["predictive"]
    df = analysis_data["historical_data"].copy()
    req_date = analysis_data.get("request_date", perf.get("latest_date", ""))
    is_up_to_date = analysis_data.get("is_up_to_date", True)

    # Pre-calculate 50-day and 200-day moving averages across the ENTIRE historical dataset
    # so moving averages are always continuously available regardless of year selected or zoom level
    if "close" in df.columns:
        df["sma50"] = df["close"].rolling(window=50, min_periods=50).mean()
        df["sma200"] = df["close"].rolling(window=200, min_periods=200).mean()
    else:
        df["sma50"] = None
        df["sma200"] = None

    # Prepare JSON serializable payload for the interactive client-side charts
    history_payload = []
    for _, row in df.iterrows():
        history_payload.append({
            "date": row["date"],
            "open": round(float(row["open"]), 2),
            "high": round(float(row["high"]), 2),
            "low": round(float(row["low"]), 2),
            "close": round(float(row["close"]), 2),
            "volume": int(row["volume"]) if not pd.isna(row["volume"]) else 0,
            "sma50": round(float(row["sma50"]), 2) if not pd.isna(row.get("sma50")) else None,
            "sma200": round(float(row["sma200"]), 2) if not pd.isna(row.get("sma200")) else None,
            "avwap_ytd": round(float(row["avwap_ytd"]), 2) if "avwap_ytd" in row and not pd.isna(row["avwap_ytd"]) else None,
            "avwap_ytd_upper_1s": round(float(row["avwap_ytd_upper_1s"]), 2) if "avwap_ytd_upper_1s" in row and not pd.isna(row["avwap_ytd_upper_1s"]) else None,
            "avwap_ytd_lower_1s": round(float(row["avwap_ytd_lower_1s"]), 2) if "avwap_ytd_lower_1s" in row and not pd.isna(row["avwap_ytd_lower_1s"]) else None,
        })

    # Color palette based on recommendation
    rec_colors = {
        "STRONG BUY": ("#10b981", "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"),
        "STRONG BUY / TREND ACCUMULATION": ("#10b981", "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"),
        "BUY ON PULLBACK": ("#3b82f6", "bg-blue-500/10 text-blue-400 border-blue-500/30"),
        "ACCUMULATE / DIP BUY": ("#06b6d4", "bg-cyan-500/10 text-cyan-400 border-cyan-500/30"),
        "RANGE ACCUMULATION / BUY SUPPORT": ("#3b82f6", "bg-blue-500/10 text-blue-400 border-blue-500/30"),
        "HOLD / CAUTIOUS BUY": ("#f59e0b", "bg-amber-500/10 text-amber-400 border-amber-500/30"),
        "REGIME SHIFT ALERT / PAUSE ENTRIES": ("#f59e0b", "bg-amber-500/10 text-amber-400 border-amber-500/30"),
        "RISK-OFF / CAPITAL PRESERVATION": ("#ef4444", "bg-red-500/10 text-red-400 border-red-500/30"),
    }
    rec_color, rec_badge_class = rec_colors.get(pred["recommendation"], ("#3b82f6", "bg-blue-500/10 text-blue-400 border-blue-500/30"))

    # Multi-period forward projections (6M, 1Y, 2Y, 3Y)
    projections = analysis_data.get("projections")
    if not projections:
        projections = compute_multi_period_projections(
            df,
            regime=analysis_data.get("regime"),
            microstructure=analysis_data.get("microstructure"),
        )

    # Build projection cards HTML for 6M, 1Y, 2Y, 3Y
    proj_cards_html = ""
    for p_key in ["6M", "1Y", "2Y", "3Y"]:
        p_data = projections.get(p_key, {})
        if not p_data:
            continue
        p_ret = p_data.get("projected_return_pct", 0.0)
        ret_color = "text-emerald-400" if p_ret >= 0 else "text-red-400"
        ret_sign = "+" if p_ret >= 0 else ""
        prob = p_data.get("probability_score", 50.0)
        conf = p_data.get("confidence", "Moderate")

        if prob >= 80.0:
            badge_class = "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
            conf_text_color = "text-emerald-400"
            bar_gradient = "from-emerald-500 to-teal-300"
        elif prob >= 65.0:
            badge_class = "bg-blue-500/10 text-blue-400 border-blue-500/30"
            conf_text_color = "text-blue-400"
            bar_gradient = "from-blue-500 to-cyan-300"
        elif prob >= 50.0:
            badge_class = "bg-amber-500/10 text-amber-400 border-amber-500/30"
            conf_text_color = "text-amber-400"
            bar_gradient = "from-amber-500 to-yellow-300"
        else:
            badge_class = "bg-red-500/10 text-red-400 border-red-500/30"
            conf_text_color = "text-red-400"
            bar_gradient = "from-red-500 to-rose-400"

        proj_cards_html += f"""
        <div class="bg-gray-900/90 border border-gray-800 rounded-xl p-4 shadow-sm hover:border-purple-500/40 transition flex flex-col justify-between">
          <div>
            <div class="flex justify-between items-center mb-1.5">
              <span class="text-xs font-bold text-gray-300 uppercase tracking-wider">{p_data.get('label', p_key)} Projection</span>
              <span class="text-[10px] font-bold px-2 py-0.5 rounded-full border {badge_class}">
                {prob:.0f}% Prob
              </span>
            </div>
            <div class="text-2xl font-black {ret_color} mb-0.5">
              {ret_sign}{p_ret:.1f}%
            </div>
            <div class="text-xs font-semibold text-gray-200 mb-2">
              Target: ${p_data.get('base_target_price', 0):.2f}
              <span class="text-[10px] font-normal text-gray-400">({p_data.get('projected_cagr_pct', 0):.1f}% CAGR)</span>
            </div>

            <!-- Probability Score Meter -->
            <div class="mb-3">
              <div class="flex justify-between text-[10px] text-gray-400 mb-1">
                <span>Probability Score</span>
                <span class="font-bold text-white">{prob:.1f}%</span>
              </div>
              <div class="w-full bg-gray-800 rounded-full h-1.5 overflow-hidden">
                <div class="h-full bg-gradient-to-r {bar_gradient} rounded-full" style="width: {min(100, max(5, prob))}%;"></div>
              </div>
            </div>
          </div>

          <div class="text-[11px] border-t border-gray-800/80 pt-2 space-y-1">
            <div class="flex justify-between text-gray-400">
              <span>Bear &ndash; Bull:</span>
              <span class="font-mono text-gray-300">${p_data.get('bear_price', 0):.2f} &ndash; ${p_data.get('bull_price', 0):.2f}</span>
            </div>
            <div class="flex justify-between text-gray-400">
              <span>Confidence:</span>
              <span class="{conf_text_color} font-medium">{conf}</span>
            </div>
            <div class="flex justify-between text-gray-500 text-[10px] pt-0.5">
              <span>Conditioned &mu; / &sigma;:</span>
              <span class="font-mono text-gray-400">{p_data.get('effective_drift_pct', 0.0):+.1f}% / {p_data.get('effective_vol_pct', 0.0):.1f}%</span>
            </div>
            {f'''<div class="flex justify-between text-gray-500 text-[10px]">
              <span>BOCD Shift Risk:</span>
              <span class="font-mono text-purple-300 font-semibold">{p_data.get("bocd_changepoint_prob_pct"):.0f}%</span>
            </div>''' if p_data.get("bocd_changepoint_prob_pct") is not None else ''}
          </div>
        </div>
        """

    # Build market regime card HTML if available
    regime = analysis_data.get("regime")
    regime_html = ""
    if regime:
        cp_prob = regime.get("changepoint_prob_pct", 0.0)
        cp_color = "text-emerald-400" if cp_prob < 25.0 else ("text-amber-400" if cp_prob < 50.0 else "text-red-400")
        cp_bar_color = "bg-emerald-500" if cp_prob < 25.0 else ("bg-amber-500" if cp_prob < 50.0 else "bg-red-500")

        regime_html = f"""
    <!-- BAYESIAN ONLINE CHANGEPOINT DETECTION (BOCD) & MARKET REGIME ROW -->
    <div class="bg-gray-950/70 border border-teal-900/40 rounded-2xl p-5 shadow-sm">
      <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-1 mb-3 px-1">
        <div class="flex items-center gap-2">
          <span class="w-2.5 h-2.5 rounded-full bg-teal-400 animate-pulse"></span>
          <h2 class="text-xs font-bold text-teal-300 uppercase tracking-wider">Bayesian Online Changepoint Detection (BOCD) &amp; Macro Regime</h2>
        </div>
        <div class="text-[11px] text-gray-400 font-mono">
          Model: Adams &amp; MacKay (2007) Conjugate Student-t &bull; Non-Stationary Time Series
        </div>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
        <!-- REGIME STATE CARD -->
        <div class="bg-gray-900/90 border border-gray-800 rounded-xl p-4 shadow-sm flex flex-col justify-between">
          <div>
            <div class="flex justify-between items-center mb-1">
              <span class="text-[11px] font-bold text-gray-400 uppercase tracking-wider">Market State</span>
              <span class="text-[10px] font-bold px-2 py-0.5 rounded-full border {regime.get('badge_class', '')}">State {regime.get('state', 0)}</span>
            </div>
            <div class="text-lg font-black text-white mt-1">
              {regime.get('name', 'N/A')}
            </div>
            <div class="text-xs font-semibold text-teal-400 mt-1">
              {regime.get('action', '')}
            </div>
          </div>
          <div class="text-[11px] text-gray-400 border-t border-gray-800/80 pt-2 mt-3 leading-relaxed">
            {regime.get('description', '')}
          </div>
        </div>

        <!-- BOCD CHANGEPOINT RISK -->
        <div class="bg-gray-900/90 border border-gray-800 rounded-xl p-4 shadow-sm flex flex-col justify-between">
          <div>
            <div class="flex justify-between items-center mb-1">
              <span class="text-[11px] font-bold text-gray-400 uppercase tracking-wider">Changepoint Hazard P(r=0)</span>
              <span class="text-xs font-bold font-mono {cp_color}">{cp_prob:.1f}%</span>
            </div>
            <div class="text-2xl font-black text-white mt-1">
              {regime.get('expected_run_length_days', 0):.0f} <span class="text-xs font-normal text-gray-400">Days Active</span>
            </div>
            <div class="w-full bg-gray-800 rounded-full h-1.5 overflow-hidden my-2.5">
              <div class="h-full {cp_bar_color} rounded-full" style="width: {min(100, max(5, cp_prob))}%;"></div>
            </div>
          </div>
          <div class="text-[11px] text-gray-400 border-t border-gray-800/80 pt-2 space-y-1 font-mono">
            <div class="flex justify-between"><span>Bull Prob:</span> <span class="text-emerald-400">{(regime.get('probabilities', {}).get('bull', 0)*100):.1f}%</span></div>
            <div class="flex justify-between"><span>Risk-Off Prob:</span> <span class="text-red-400">{(regime.get('probabilities', {}).get('risk_off', 0)*100):.1f}%</span></div>
          </div>
        </div>

        <!-- VOLATILITY SURFACE -->
        <div class="bg-gray-900/90 border border-gray-800 rounded-xl p-4 shadow-sm flex flex-col justify-between">
          <div>
            <div class="flex justify-between items-center mb-1">
              <span class="text-[11px] font-bold text-gray-400 uppercase tracking-wider">Realized Vol Surface</span>
              <span class="text-xs font-bold font-mono {'text-red-400' if regime.get('vol_ratio', 1) > 1.15 else 'text-emerald-400'}">Ratio: {regime.get('vol_ratio', 1):.2f}x</span>
            </div>
            <div class="text-xl font-black text-white mt-1">
              {regime.get('vol_21d_pct', 0):.1f}% <span class="text-xs font-normal text-gray-400">21d Ann. Vol</span>
            </div>
            <div class="text-[11px] text-gray-400 mt-1">
              5d Vol: <span class="text-gray-200 font-mono">{regime.get('vol_5d_pct', 0):.1f}%</span> | Term: <span class="font-mono text-gray-200">{'Inverted (Stress)' if regime.get('vol_ratio', 1) > 1.15 else 'Normal / Contango'}</span>
            </div>
          </div>
          <div class="text-[11px] text-gray-400 border-t border-gray-800/80 pt-2">
            Status: <span class="text-gray-200 font-medium">{'Elevated short-term vol spike' if regime.get('vol_ratio', 1) > 1.15 else 'Stable volatility baseline'}</span>
          </div>
        </div>

        <!-- MACRO CREDIT & SIZING -->
        <div class="bg-gray-900/90 border border-gray-800 rounded-xl p-4 shadow-sm flex flex-col justify-between">
          <div>
            <div class="flex justify-between items-center mb-1">
              <span class="text-[11px] font-bold text-gray-400 uppercase tracking-wider">Macro Risk Appetite</span>
              <span class="text-xs font-bold font-mono {'text-emerald-400' if regime.get('credit_mom_pct', 0) >= 0 else 'text-amber-400'}">{'+' if regime.get('credit_mom_pct', 0) >= 0 else ''}{regime.get('credit_mom_pct', 0):.2f}%</span>
            </div>
            <div class="text-xl font-black text-white mt-1">
              {regime.get('risk_multiplier', 1.0):.1f}x <span class="text-xs font-normal text-gray-400">Sizing Factor</span>
            </div>
            <div class="text-[11px] text-gray-400 mt-1">
              Credit Momentum (HYG/IEI): <span class="font-mono {'text-emerald-400' if regime.get('credit_mom_pct', 0) >= 0 else 'text-amber-400'}">{'Expanding / Risk-On' if regime.get('credit_mom_pct', 0) >= 0 else 'Compressing / Defensive'}</span>
            </div>
          </div>
          <div class="text-[11px] text-gray-400 border-t border-gray-800/80 pt-2">
            Portfolio Allocation: <span class="text-gray-200 font-medium">{int(regime.get('risk_multiplier', 1.0)*100)}% standard exposure</span>
          </div>
        </div>
      </div>
    </div>
    """

    # Build institutional microstructure & AVWAP HTML if available
    micro = analysis_data.get("microstructure")
    micro_html = ""
    if micro:
        avwap_data = micro.get("avwap", {})
        ytd = avwap_data.get("ytd", {})
        h52 = avwap_data.get("high_52w", {})
        l52 = avwap_data.get("low_52w", {})
        vp = micro.get("volume_profile", {})

        ytd_val = ytd.get("value")
        ytd_str = f"${ytd_val:.2f}" if ytd_val is not None else "N/A"
        ytd_z = ytd.get("zscore")
        ytd_z_str = f"{ytd_z:+.2f}σ" if ytd_z is not None else "N/A"
        ytd_z_color = "text-emerald-400" if (ytd_z is not None and 0 <= ytd_z <= 1.5) else ("text-cyan-400" if (ytd_z is not None and -1.5 <= ytd_z < 0) else "text-amber-400")

        h52_val = h52.get("value")
        h52_str = f"${h52_val:.2f}" if h52_val is not None else "N/A"
        l52_val = l52.get("value")
        l52_str = f"${l52_val:.2f}" if l52_val is not None else "N/A"

        poc_val = vp.get("poc")
        poc_str = f"${poc_val:.2f}" if poc_val is not None else "N/A"
        vah_val = vp.get("vah")
        vah_str = f"${vah_val:.2f}" if vah_val is not None else "N/A"
        val_val = vp.get("val")
        val_str = f"${val_val:.2f}" if val_val is not None else "N/A"

        void_status = vp.get("void_status", "Balanced Liquidity")
        void_badge = "bg-amber-500/10 text-amber-400 border-amber-500/30" if vp.get("in_liquidity_void") else "bg-cyan-500/10 text-cyan-400 border-cyan-500/30"

        micro_html = f"""
    <!-- INSTITUTIONAL LIQUIDITY, ANCHORED VWAP & VOLUME PROFILE ROW -->
    <div class="bg-gray-950/70 border border-cyan-900/40 rounded-2xl p-5 shadow-sm">
      <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-1 mb-3 px-1">
        <div class="flex items-center gap-2">
          <span class="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse"></span>
          <h2 class="text-xs font-bold text-cyan-300 uppercase tracking-wider">Institutional Liquidity, Anchored VWAP (AVWAP) &amp; Volume Profile</h2>
        </div>
        <div class="text-[11px] text-gray-400 font-mono">
          Gaussian Kernel Density Estimation (KDE) &bull; Volume-Weighted Dispersion Bands (&plusmn;1&sigma;)
        </div>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
        <!-- YTD AVWAP CARD -->
        <div class="bg-gray-900/90 border border-gray-800 rounded-xl p-4 shadow-sm flex flex-col justify-between">
          <div>
            <div class="flex justify-between items-center mb-1">
              <span class="text-[11px] font-bold text-gray-400 uppercase tracking-wider">YTD Anchored VWAP</span>
              <span class="text-[10px] font-bold px-2 py-0.5 rounded-full border bg-cyan-500/10 text-cyan-400 border-cyan-500/30 font-mono">{ytd.get('date', 'N/A')}</span>
            </div>
            <div class="text-xl font-black text-white mt-1">
              {ytd_str} <span class="text-xs font-bold font-mono {ytd_z_color}">{ytd_z_str}</span>
            </div>
            <div class="text-[11px] text-gray-400 mt-1">
              &plusmn;1&sigma; Envelope: <span class="font-mono text-gray-200">${ytd.get('lower_1s', 0) if ytd.get('lower_1s') is not None else 0:.2f} &ndash; ${ytd.get('upper_1s', 0) if ytd.get('upper_1s') is not None else 0:.2f}</span>
            </div>
          </div>
          <div class="text-[11px] text-gray-400 border-t border-gray-800/80 pt-2 mt-3 leading-relaxed">
            {ytd.get('action', '')}
          </div>
        </div>

        <!-- 52-WEEK HIGH / LOW ANCHORS -->
        <div class="bg-gray-900/90 border border-gray-800 rounded-xl p-4 shadow-sm flex flex-col justify-between">
          <div>
            <div class="flex justify-between items-center mb-1">
              <span class="text-[11px] font-bold text-gray-400 uppercase tracking-wider">Cyclical AVWAP Anchors</span>
              <span class="text-[10px] font-bold px-2 py-0.5 rounded-full border bg-purple-500/10 text-purple-400 border-purple-500/30">52W Extremes</span>
            </div>
            <div class="space-y-1.5 mt-1">
              <div class="flex justify-between text-xs">
                <span class="text-gray-400">52W High ({h52.get('date', 'N/A')}):</span>
                <span class="font-bold text-white font-mono">{h52_str} <span class="text-[10px] text-gray-400">({'+' if (h52.get('spread_pct') or 0) >= 0 else ''}{h52.get('spread_pct', 0):.1f}%)</span></span>
              </div>
              <div class="flex justify-between text-xs">
                <span class="text-gray-400">52W Low ({l52.get('date', 'N/A')}):</span>
                <span class="font-bold text-white font-mono">{l52_str} <span class="text-[10px] text-gray-400">({'+' if (l52.get('spread_pct') or 0) >= 0 else ''}{l52.get('spread_pct', 0):.1f}%)</span></span>
              </div>
            </div>
          </div>
          <div class="text-[11px] text-gray-400 border-t border-gray-800/80 pt-2 mt-2">
            Anchor Memory: <span class="text-gray-200 font-medium">Overhead supply from peak vs. support from trough</span>
          </div>
        </div>

        <!-- VOLUME PROFILE POC & VALUE AREA -->
        <div class="bg-gray-900/90 border border-gray-800 rounded-xl p-4 shadow-sm flex flex-col justify-between">
          <div>
            <div class="flex justify-between items-center mb-1">
              <span class="text-[11px] font-bold text-gray-400 uppercase tracking-wider">Volume Profile (KDE)</span>
              <span class="text-[10px] font-bold px-2 py-0.5 rounded-full border bg-teal-500/10 text-teal-400 border-teal-500/30">70% Value Area</span>
            </div>
            <div class="text-xl font-black text-white mt-1">
              {poc_str} <span class="text-xs font-normal text-gray-400">Point of Control</span>
            </div>
            <div class="text-[11px] text-gray-400 mt-1">
              Value Area: <span class="font-mono text-gray-200">{val_str} &ndash; {vah_str}</span>
            </div>
          </div>
          <div class="text-[11px] text-gray-400 border-t border-gray-800/80 pt-2 mt-2">
            Distance to POC: <span class="font-mono text-gray-200">{'+' if (vp.get('dist_to_poc_pct') or 0) >= 0 else ''}{vp.get('dist_to_poc_pct', 0):.1f}%</span> &bull; <span class="text-gray-200">{'Inside Value Area' if vp.get('in_value_area') else 'Outside Value Area'}</span>
          </div>
        </div>

        <!-- LIQUIDITY VOID / DEPTH -->
        <div class="bg-gray-900/90 border border-gray-800 rounded-xl p-4 shadow-sm flex flex-col justify-between">
          <div>
            <div class="flex justify-between items-center mb-1">
              <span class="text-[11px] font-bold text-gray-400 uppercase tracking-wider">Market Depth &amp; Void</span>
              <span class="text-[10px] font-bold px-2 py-0.5 rounded-full border {void_badge}">{'VOID DETECTED' if vp.get('in_liquidity_void') else 'BALANCED'}</span>
            </div>
            <div class="text-sm font-bold text-white mt-1 leading-snug">
              {void_status}
            </div>
          </div>
          <div class="text-[11px] text-gray-400 border-t border-gray-800/80 pt-2 mt-2">
            Execution Rule: <span class="text-gray-200 font-medium">{'Fast traversal through thin book' if vp.get('in_liquidity_void') else 'Sustained institutional balance'}</span>
          </div>
        </div>
      </div>
    </div>
    """

    # Convert payloads to JSON
    json_history = json.dumps(history_payload)
    json_best_buys = json.dumps(best_buys)
    json_predictive = json.dumps(pred)
    json_performance = json.dumps(perf)
    json_projections = json.dumps(projections)
    json_regime = json.dumps(regime) if regime else "{}"
    json_micro = json.dumps(micro) if micro else "{}"

    html_content = f"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{symbol} - Performance & Predictive Buy Timing Report</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    :root {{
      --bg-main: #090d16;
      --bg-card: #111827;
      --border-color: #1f2937;
      --text-main: #f3f4f6;
      --text-muted: #9ca3af;
      --accent-green: #10b981;
      --accent-blue: #3b82f6;
      --accent-gold: #f59e0b;
      --accent-red: #ef4444;
    }}
    body {{
      background-color: var(--bg-main);
      color: var(--text-main);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}
    .chart-canvas {{
      width: 100%;
      height: 100%;
      display: block;
    }}
    .glow-gold {{
      box-shadow: 0 0 15px rgba(245, 158, 11, 0.25);
    }}
    .glow-green {{
      box-shadow: 0 0 15px rgba(16, 185, 129, 0.25);
    }}
    /* Custom Scrollbar */
    ::-webkit-scrollbar {{
      width: 6px;
      height: 6px;
    }}
    ::-webkit-scrollbar-track {{
      background: #111827;
    }}
    ::-webkit-scrollbar-thumb {{
      background: #374151;
      border-radius: 3px;
    }}
  </style>
</head>
<body class="min-h-screen antialiased p-4 md:p-8">
  <div class="max-w-7xl mx-auto space-y-6">

    <!-- TOP HEADER -->
    <header class="bg-gray-900/80 backdrop-blur border border-gray-800 rounded-2xl p-6 shadow-xl flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
      <div>
        <div class="flex flex-wrap items-center gap-2.5">
          <span class="text-3xl font-extrabold tracking-tight text-white">{symbol}</span>
          <span class="text-xs px-2.5 py-1 rounded-full font-semibold border {rec_badge_class}">
            {pred["recommendation"]}
          </span>
          <span class="text-xs font-semibold px-2.5 py-1 rounded-full border {'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' if is_up_to_date else 'bg-amber-500/10 text-amber-400 border-amber-500/30'} flex items-center gap-1.5">
            <span class="w-1.5 h-1.5 rounded-full {'bg-emerald-400' if is_up_to_date else 'bg-amber-400'}"></span>
            {'Data Up-to-Date' if is_up_to_date else 'Latest Market Data'}: {perf["latest_date"]}
          </span>
          <span class="text-xs text-gray-400 bg-gray-800/80 px-2.5 py-1 rounded-full border border-gray-700">
            Requested: {req_date}
          </span>
        </div>
        <p class="text-sm text-gray-400 mt-1">
          Quantitative historical return analysis, multi-period forward projections, and predictive timing forecast.
        </p>
      </div>
      <div class="flex items-center gap-6">
        <div class="text-right">
          <div class="text-xs text-gray-400 uppercase tracking-wider font-medium">Latest Close</div>
          <div class="text-3xl font-black text-white">${perf["latest_close"]:.2f}</div>
        </div>
        <div class="h-10 w-px bg-gray-800"></div>
        <div class="text-right">
          <div class="text-xs text-gray-400 uppercase tracking-wider font-medium">3M Target Price</div>
          <div class="text-2xl font-black text-emerald-400">${pred["target_price_3m"]:.2f}</div>
          <div class="text-xs font-semibold {'text-emerald-400' if pred['expected_return_pct'] >= 0 else 'text-red-400'}">
            {'+' if pred['expected_return_pct'] >= 0 else ''}{pred['expected_return_pct']:.1f}% Expected
          </div>
        </div>
      </div>
    </header>

    <!-- HISTORICAL PERFORMANCE CARDS -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
      <!-- 1-YEAR CARD -->
      <div class="bg-gray-900/90 border border-gray-800 rounded-xl p-5 shadow-sm hover:border-gray-700 transition">
        <div class="flex justify-between items-center mb-2">
          <span class="text-xs font-bold text-gray-400 uppercase tracking-wider">1-Year Historical</span>
          <span class="text-xs font-semibold {'text-emerald-400' if perf['periods'].get('1Y', {}).get('total_return_pct', 0) >= 0 else 'text-red-400'}">
            {'+' if perf['periods'].get('1Y', {}).get('total_return_pct', 0) >= 0 else ''}{perf['periods'].get('1Y', {}).get('total_return_pct', 0):.1f}%
          </span>
        </div>
        <div class="text-2xl font-bold text-white mb-3">
          ${perf['periods'].get('1Y', {}).get('start_price', 0):.2f} &rarr; ${perf['periods'].get('1Y', {}).get('end_price', 0):.2f}
        </div>
        <div class="grid grid-cols-2 gap-2 text-xs border-t border-gray-800/80 pt-3">
          <div><span class="text-gray-500">Max DD:</span> <span class="text-red-400 font-medium">{perf['periods'].get('1Y', {}).get('max_drawdown_pct', 0):.1f}%</span></div>
          <div><span class="text-gray-500">Sharpe:</span> <span class="text-gray-300 font-medium">{perf['periods'].get('1Y', {}).get('sharpe_ratio', 0)}</span></div>
          <div><span class="text-gray-500">Volatility:</span> <span class="text-gray-300 font-medium">{perf['periods'].get('1Y', {}).get('annual_volatility_pct', 0):.1f}%</span></div>
          <div><span class="text-gray-500">Win Rate:</span> <span class="text-gray-300 font-medium">{perf['periods'].get('1Y', {}).get('win_rate_pct', 0):.0f}%</span></div>
        </div>
      </div>

      <!-- 3-YEAR CARD -->
      <div class="bg-gray-900/90 border border-gray-800 rounded-xl p-5 shadow-sm hover:border-gray-700 transition">
        <div class="flex justify-between items-center mb-2">
          <span class="text-xs font-bold text-gray-400 uppercase tracking-wider">3-Year Historical</span>
          <span class="text-xs font-semibold {'text-emerald-400' if perf['periods'].get('3Y', {}).get('total_return_pct', 0) >= 0 else 'text-red-400'}">
            {'+' if perf['periods'].get('3Y', {}).get('total_return_pct', 0) >= 0 else ''}{perf['periods'].get('3Y', {}).get('total_return_pct', 0):.1f}%
          </span>
        </div>
        <div class="text-2xl font-bold text-white mb-3">
          ${perf['periods'].get('3Y', {}).get('start_price', 0):.2f} &rarr; ${perf['periods'].get('3Y', {}).get('end_price', 0):.2f}
        </div>
        <div class="grid grid-cols-2 gap-2 text-xs border-t border-gray-800/80 pt-3">
          <div><span class="text-gray-500">CAGR:</span> <span class="text-emerald-400 font-medium">{perf['periods'].get('3Y', {}).get('cagr_pct', 0):.1f}%</span></div>
          <div><span class="text-gray-500">Max DD:</span> <span class="text-red-400 font-medium">{perf['periods'].get('3Y', {}).get('max_drawdown_pct', 0):.1f}%</span></div>
          <div><span class="text-gray-500">Sharpe:</span> <span class="text-gray-300 font-medium">{perf['periods'].get('3Y', {}).get('sharpe_ratio', 0)}</span></div>
          <div><span class="text-gray-500">Volatility:</span> <span class="text-gray-300 font-medium">{perf['periods'].get('3Y', {}).get('annual_volatility_pct', 0):.1f}%</span></div>
        </div>
      </div>

      <!-- 5-YEAR CARD -->
      <div class="bg-gray-900/90 border border-gray-800 rounded-xl p-5 shadow-sm hover:border-gray-700 transition">
        <div class="flex justify-between items-center mb-2">
          <span class="text-xs font-bold text-gray-400 uppercase tracking-wider">5-Year Historical</span>
          <span class="text-xs font-semibold {'text-emerald-400' if perf['periods'].get('5Y', {}).get('total_return_pct', 0) >= 0 else 'text-red-400'}">
            {'+' if perf['periods'].get('5Y', {}).get('total_return_pct', 0) >= 0 else ''}{perf['periods'].get('5Y', {}).get('total_return_pct', 0):.1f}%
          </span>
        </div>
        <div class="text-2xl font-bold text-white mb-3">
          ${perf['periods'].get('5Y', {}).get('start_price', 0):.2f} &rarr; ${perf['periods'].get('5Y', {}).get('end_price', 0):.2f}
        </div>
        <div class="grid grid-cols-2 gap-2 text-xs border-t border-gray-800/80 pt-3">
          <div><span class="text-gray-500">CAGR:</span> <span class="text-emerald-400 font-medium">{perf['periods'].get('5Y', {}).get('cagr_pct', 0):.1f}%</span></div>
          <div><span class="text-gray-500">Max DD:</span> <span class="text-red-400 font-medium">{perf['periods'].get('5Y', {}).get('max_drawdown_pct', 0):.1f}%</span></div>
          <div><span class="text-gray-500">Sharpe:</span> <span class="text-gray-300 font-medium">{perf['periods'].get('5Y', {}).get('sharpe_ratio', 0)}</span></div>
          <div><span class="text-gray-500">Volatility:</span> <span class="text-gray-300 font-medium">{perf['periods'].get('5Y', {}).get('annual_volatility_pct', 0):.1f}%</span></div>
        </div>
      </div>

      <!-- 3-MONTH PREDICTIVE OUTLOOK CARD -->
      <div class="bg-gradient-to-br from-gray-900 to-blue-950/40 border border-blue-800/40 rounded-xl p-5 shadow-sm">
        <div class="flex justify-between items-center mb-2">
          <span class="text-xs font-bold text-blue-400 uppercase tracking-wider">3-Month Strategy</span>
          <span class="text-xs font-bold px-2 py-0.5 rounded bg-blue-500/20 text-blue-300">R:R {pred['risk_reward_ratio']}:1</span>
        </div>
        <div class="text-xl font-black text-white mb-1">
          ${pred['optimal_entry_range'][0]:.2f} - ${pred['optimal_entry_range'][1]:.2f}
        </div>
        <div class="text-xs text-gray-400 mb-3">Recommended Optimal Entry Range</div>
        <div class="text-xs border-t border-gray-800/80 pt-2 space-y-1">
          <div class="flex justify-between"><span class="text-gray-500">Optimal Window:</span> <span class="text-emerald-300 font-medium">{pred['optimal_buy_window']['start_date']} &rarr; {pred['optimal_buy_window']['end_date']}</span></div>
          <div class="flex justify-between"><span class="text-gray-500">Stop-Loss:</span> <span class="text-red-400 font-medium">${pred['stop_loss']:.2f}</span></div>
          <div class="flex justify-between"><span class="text-gray-500">Key Support:</span> <span class="text-gray-300 font-medium">${pred['key_support']:.2f}</span></div>
          {f'''<div class="flex justify-between"><span class="text-gray-500">BOCD Regime:</span> <span class="text-amber-300 font-medium">{pred.get("bocd_regime_name")}</span></div>''' if pred.get("bocd_regime_name") else ''}
          {f'''<div class="flex justify-between"><span class="text-gray-500">63d Changepoint Risk:</span> <span class="text-red-400 font-mono font-medium">{pred.get("bocd_forward_changepoint_prob_pct"):.1f}%</span></div>''' if pred.get("bocd_forward_changepoint_prob_pct") is not None else ''}
        </div>
      </div>
    </div>

    {regime_html}

    {micro_html}

    <!-- FORWARD RETURN PROJECTIONS & PROBABILITY SCORES ROW -->
    <div class="bg-gray-950/60 border border-purple-900/30 rounded-2xl p-5 shadow-sm">
      <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-1 mb-3 px-1">
        <div class="flex items-center gap-2">
          <span class="w-2.5 h-2.5 rounded-full bg-purple-400 animate-pulse"></span>
          <h2 class="text-xs font-bold text-purple-300 uppercase tracking-wider">Forward Return Projections &amp; Probability Analysis</h2>
        </div>
        <span class="text-[11px] text-gray-400 font-medium">Dynamically conditioned on BOCD regime risk, hazard probabilities &amp; microstructure</span>
      </div>
      <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
        {proj_cards_html}
      </div>
    </div>

    <!-- MAIN INTERACTIVE HISTORICAL CHART (1Y / 3Y / 5Y) -->
    <div class="bg-gray-900 border border-gray-800 rounded-2xl p-6 shadow-xl">
      <div class="flex flex-col md:flex-row justify-between items-start md:items-center mb-4 gap-3">
        <div>
          <h2 class="text-lg font-bold text-white flex items-center gap-2">
            <span>Historical Performance & Best Times to Buy</span>
            <span class="text-xs font-normal text-gray-400 bg-gray-800 px-2 py-0.5 rounded">Gold Stars = Optimal Entry Points</span>
          </h2>
          <p class="text-xs text-gray-400">
            Interactive chart showing price trajectory, 50/200-day moving averages, and historical buy opportunities.
          </p>
        </div>
        <div class="flex flex-wrap items-center gap-2">
          <!-- Zoom Controls -->
          <div class="flex items-center gap-1 bg-gray-800/80 p-1 rounded-lg border border-gray-700 text-xs">
            <button id="btn-zoom-in" onclick="zoomRelative(0.7)" title="Zoom In (or scroll wheel up)" class="px-2.5 py-1 rounded font-bold text-gray-300 hover:text-white hover:bg-gray-700/60 transition flex items-center gap-1">
              <span>+</span> <span>Zoom</span>
            </button>
            <button id="btn-zoom-out" onclick="zoomRelative(1.4)" title="Zoom Out (or scroll wheel down)" class="px-2.5 py-1 rounded font-bold text-gray-300 hover:text-white hover:bg-gray-700/60 transition flex items-center gap-1">
              <span>&minus;</span>
            </button>
            <button id="btn-zoom-reset" onclick="resetZoom()" title="Reset Zoom to full period" class="px-2.5 py-1 rounded font-medium text-gray-400 hover:text-white hover:bg-gray-700/60 transition flex items-center gap-1">
              <span>&#x21bb;</span> <span>Reset</span>
            </button>
          </div>

          <!-- Period Buttons -->
          <div class="flex items-center gap-1 bg-gray-800/80 p-1 rounded-lg border border-gray-700 text-xs">
            <button id="btn-1y" onclick="setPeriod('1Y')" class="px-3 py-1 rounded font-medium transition text-gray-400 hover:text-white">1 Year</button>
            <button id="btn-3y" onclick="setPeriod('3Y')" class="px-3 py-1 rounded font-medium transition text-gray-400 hover:text-white">3 Years</button>
            <button id="btn-5y" onclick="setPeriod('5Y')" class="px-3 py-1 rounded font-medium transition bg-blue-600 text-white font-semibold">5 Years</button>
            <button id="btn-all" onclick="setPeriod('ALL')" class="px-3 py-1 rounded font-medium transition text-gray-400 hover:text-white">Max</button>
          </div>
        </div>
      </div>

      <!-- Chart Container -->
      <div id="historicalChartWrapper" class="relative w-full h-96 bg-gray-950/60 rounded-xl border border-gray-800/80 overflow-hidden cursor-crosshair select-none">
        <canvas id="historicalChart" class="chart-canvas"></canvas>
        <div id="chartTooltip" class="absolute hidden pointer-events-none bg-gray-900/95 border border-gray-700 text-white text-xs rounded-lg p-3 shadow-2xl z-20 max-w-xs"></div>
        <!-- Drag-to-Zoom Selection Box -->
        <div id="zoomSelectionBox" class="absolute top-0 bottom-0 hidden pointer-events-none bg-blue-500/25 border-x-2 border-blue-400/90 z-10"></div>
        <!-- Zoom Active Status Badge -->
        <div id="zoomBadge" class="absolute top-3 left-4 hidden bg-blue-950/90 border border-blue-500/60 text-blue-200 text-xs px-3 py-1 rounded-full shadow-lg items-center gap-2 z-20">
          <span id="zoomBadgeText" class="font-medium">🔍 Zoomed View</span>
          <button onclick="resetZoom()" title="Reset Zoom" class="text-blue-400 hover:text-white font-bold ml-1 text-sm leading-none">&times;</button>
        </div>
        <!-- Drag Zoom Hint -->
        <div class="absolute bottom-2 left-3 text-[10px] text-gray-500 pointer-events-none bg-gray-900/70 px-2 py-0.5 rounded border border-gray-800">
          Tip: Click &amp; drag horizontally to zoom &bull; Mouse wheel to zoom &bull; Click card to zoom to trade
        </div>
      </div>

      <!-- Legend & Controls -->
      <div class="flex flex-wrap items-center justify-between text-xs text-gray-400 mt-3 pt-3 border-t border-gray-800">
        <div class="flex items-center gap-4">
          <span class="flex items-center gap-1.5"><span class="w-3 h-0.5 bg-blue-500 inline-block"></span> Close Price</span>
          <span class="flex items-center gap-1.5"><span class="w-3 h-0.5 bg-amber-400 inline-block"></span> 50-Day MA</span>
          <span class="flex items-center gap-1.5"><span class="w-3 h-0.5 bg-purple-400 inline-block"></span> 200-Day MA</span>
          <span class="flex items-center gap-1.5"><span class="w-3 h-0.5 bg-cyan-400 inline-block"></span> YTD AVWAP (&plusmn;1&sigma;)</span>
          <span class="flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-full bg-amber-400 inline-block"></span> Best Buy Point</span>
          <span class="flex items-center gap-1.5"><span class="w-3.5 h-2 bg-emerald-500/25 border border-emerald-500/50 inline-block"></span> Subsequent Rally Window</span>
        </div>
        <div>Drag across chart or scroll wheel to zoom into any period &bull; Click milestone below to zoom to trade.</div>
      </div>

      <!-- DEDICATED TIMELINE CHART OVERLAY -->
      <div class="mt-6 pt-5 border-t border-gray-800">
        <div class="flex items-center justify-between mb-3">
          <h3 class="text-sm font-bold text-white flex items-center gap-2">
            <span class="text-amber-400">&#9733;</span>
            <span>Historical Best Buy Timeline</span>
          </h3>
          <span class="text-xs text-gray-400">Chronological timeline of optimal entry points & subsequent profit surges</span>
        </div>
        <div id="buyTimelineTrack" class="relative w-full overflow-x-auto py-2">
          <!-- Dynamically populated via JavaScript -->
        </div>
      </div>
    </div>

    <!-- PREDICTIVE ANALYSIS: 3-MONTH FORWARD FORECAST -->
    <div class="bg-gray-900 border border-gray-800 rounded-2xl p-6 shadow-xl">
      <div class="flex flex-col md:flex-row justify-between items-start md:items-center mb-4 gap-3">
        <div>
          <div class="flex flex-wrap items-center gap-2 mb-1">
            <h2 class="text-lg font-bold text-white">3-Month Predictive Buy Analysis</h2>
            <span class="text-xs px-2.5 py-0.5 rounded-full font-bold border {rec_badge_class}">{pred['recommendation']}</span>
            {f'''<span class="text-xs px-2.5 py-0.5 rounded-full font-semibold border bg-purple-500/10 text-purple-300 border-purple-500/30">BOCD: {pred.get("bocd_regime_name")}</span>''' if pred.get("bocd_regime_name") else ''}
          </div>
          <p class="text-xs text-gray-400">
            BOCD jump-diffusion Monte Carlo path simulation with trend channels projecting 63 trading days forward from {perf['latest_date']}.
          </p>
        </div>
        <div class="flex flex-wrap items-center gap-3 text-xs">
          {f'''<div class="bg-purple-950/40 border border-purple-800/40 px-3 py-1.5 rounded-lg text-purple-300">
            <span class="text-gray-400">63d Changepoint Risk:</span> <strong>{pred.get("bocd_forward_changepoint_prob_pct"):.1f}%</strong>
          </div>''' if pred.get("bocd_forward_changepoint_prob_pct") is not None else ''}
          <div class="bg-emerald-950/40 border border-emerald-800/40 px-3 py-1.5 rounded-lg text-emerald-300">
            <span class="text-gray-400">Optimal Window:</span> <strong>{pred['optimal_buy_window']['start_date']} &rarr; {pred['optimal_buy_window']['end_date']}</strong>
          </div>
          <div class="bg-blue-950/40 border border-blue-800/40 px-3 py-1.5 rounded-lg text-blue-300">
            <span class="text-gray-400">Target Range:</span> <strong>${pred['optimal_entry_range'][0]:.2f} - ${pred['optimal_entry_range'][1]:.2f}</strong>
          </div>
        </div>
      </div>

      <!-- Forecast Chart Canvas -->
      <div class="relative w-full h-80 bg-gray-950/60 rounded-xl border border-gray-800/80 overflow-hidden">
        <canvas id="forecastChart" class="chart-canvas"></canvas>
        <div id="forecastTooltip" class="absolute hidden pointer-events-none bg-gray-900/95 border border-gray-700 text-white text-xs rounded-lg p-3 shadow-2xl z-20 max-w-xs"></div>
      </div>

      <!-- Strategy Callout -->
      <div class="mt-4 p-4 rounded-xl bg-gray-950 border border-gray-800 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 text-xs">
        <div class="space-y-1">
          <span class="font-bold text-white text-sm">Tactical Execution Guidance:</span>
          <p class="text-gray-300">{pred['action_summary']}</p>
        </div>
        <div class="flex items-center gap-6 shrink-0">
          <div>
            <div class="text-gray-500">Key Support</div>
            <div class="font-bold text-gray-200">${pred['key_support']:.2f}</div>
          </div>
          <div>
            <div class="text-gray-500">Key Resistance</div>
            <div class="font-bold text-gray-200">${pred['key_resistance']:.2f}</div>
          </div>
          <div>
            <div class="text-gray-500">Stop-Loss Invalidation</div>
            <div class="font-bold text-red-400">${pred['stop_loss']:.2f}</div>
          </div>
        </div>
      </div>
    </div>

    <!-- HISTORICAL BEST BUY OPPORTUNITIES TABLE -->
    <div class="bg-gray-900 border border-gray-800 rounded-2xl p-6 shadow-xl">
      <div class="mb-4">
        <h2 class="text-lg font-bold text-white">Historical Best Buy Opportunities Ranked</h2>
        <p class="text-xs text-gray-400">
          Optimal cyclical troughs and technical inflection points that generated maximum subsequent returns.
        </p>
      </div>

      <div class="overflow-x-auto">
        <table class="w-full text-left text-xs text-gray-300 border-collapse">
          <thead>
            <tr class="border-b border-gray-800 text-gray-400 uppercase tracking-wider bg-gray-950/50">
              <th class="py-3 px-4">Buy Date</th>
              <th class="py-3 px-4">Entry Price</th>
              <th class="py-3 px-4">Subsequent Peak</th>
              <th class="py-3 px-4">Peak Date</th>
              <th class="py-3 px-4">Holding Period</th>
              <th class="py-3 px-4">Max Gain</th>
              <th class="py-3 px-4">Return to Present</th>
              <th class="py-3 px-4">Trigger / Rationale</th>
            </tr>
          </thead>
          <tbody id="bestBuysTableBody" class="divide-y divide-gray-800/60">
            <!-- Populated via Javascript -->
          </tbody>
        </table>
      </div>
    </div>

  </div>

  <!-- INTERACTIVE JAVASCRIPT ENGINE -->
  <script>
    const RAW_HISTORY = {json_history};
    const BEST_BUYS = {json_best_buys};
    const PREDICTIVE = {json_predictive};
    const PERFORMANCE = {json_performance};

    let currentPeriod = '5Y';
    let filteredHistory = [];
    let currentBestBuys = [];

    // Helper: calculate Simple Moving Average
    function calculateSMA(data, period) {{
      const sma = [];
      for (let i = 0; i < data.length; i++) {{
        if (i < period - 1) {{
          sma.push(null);
        }} else {{
          let sum = 0;
          for (let j = 0; j < period; j++) {{
            sum += data[i - j].close;
          }}
          sma.push(sum / period);
        }}
      }}
      return sma;
    }}

    // Format date labels dynamically relative to zoom level
    function formatXAxisDate(dateStr, totalDays) {{
      if (!dateStr) return '';
      const parts = dateStr.split('-');
      if (parts.length < 3) return dateStr;
      const year = parts[0];
      const month = parts[1];
      const day = parts[2];
      const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
      const mIdx = parseInt(month, 10) - 1;
      const mName = monthNames[mIdx] || month;

      if (totalDays <= 35) {{
        return `${{mName}} ${{day}}`;
      }} else if (totalDays <= 140) {{
        return `${{mName}} ${{day}}`;
      }} else if (totalDays <= 500) {{
        return `${{mName}} '${{year.slice(2)}}`;
      }} else {{
        return `${{mName}} '${{year.slice(2)}}`;
      }}
    }}

    // Ensure moving averages are computed across the full historical dataset (RAW_HISTORY)
    // so moving averages are always visible regardless of year selected or zoom level
    function initMovingAverages() {{
      if (RAW_HISTORY.length === 0) return;
      if (RAW_HISTORY[0].sma50 === undefined || RAW_HISTORY[0].sma50 === null) {{
        const allSma50 = calculateSMA(RAW_HISTORY, 50);
        const allSma200 = calculateSMA(RAW_HISTORY, 200);
        for (let i = 0; i < RAW_HISTORY.length; i++) {{
          if (RAW_HISTORY[i].sma50 === undefined || RAW_HISTORY[i].sma50 === null) {{
            RAW_HISTORY[i].sma50 = allSma50[i];
          }}
          if (RAW_HISTORY[i].sma200 === undefined || RAW_HISTORY[i].sma200 === null) {{
            RAW_HISTORY[i].sma200 = allSma200[i];
          }}
        }}
      }}
    }}

    let periodHistory = [];
    let zoomRange = {{ start: 0, end: 0 }};
    let isZoomed = false;

    function filterData(period) {{
      const totalLen = RAW_HISTORY.length;
      if (totalLen === 0) return;

      let targetCount = totalLen;
      if (period === '1Y') targetCount = Math.min(totalLen, 252);
      else if (period === '3Y') targetCount = Math.min(totalLen, 756);
      else if (period === '5Y') targetCount = Math.min(totalLen, 1260);

      periodHistory = RAW_HISTORY.slice(totalLen - targetCount);
      zoomRange = {{ start: 0, end: periodHistory.length - 1 }};
      isZoomed = false;
      updateZoomBadge();

      filteredHistory = periodHistory.slice(zoomRange.start, zoomRange.end + 1);
      const minDate = filteredHistory[0].date;

      // Extract best buys within this date range
      const periodKey = period === 'ALL' ? '5Y' : period;
      const candidates = BEST_BUYS[periodKey] || BEST_BUYS['5Y'] || [];
      currentBestBuys = candidates.filter(b => b.date >= minDate);

      // Render table & timeline track
      updateBestBuysTable(currentBestBuys);
      updateTimelineTrack(currentBestBuys);
    }}

    function updateZoomBadge() {{
      const badge = document.getElementById('zoomBadge');
      const text = document.getElementById('zoomBadgeText');
      if (!badge || !text) return;
      if (isZoomed && filteredHistory.length > 0) {{
        text.textContent = `🔍 Zoomed: ${{filteredHistory[0].date}} \u2192 ${{filteredHistory[filteredHistory.length - 1].date}} (${{filteredHistory.length}} days)`;
        badge.classList.remove('hidden');
        badge.classList.add('flex');
      }} else {{
        badge.classList.add('hidden');
        badge.classList.remove('flex');
      }}
    }}

    function applyZoom(newStart, newEnd) {{
      if (periodHistory.length === 0) return;
      // Clamp bounds: at least 5 days, within periodHistory
      newStart = Math.max(0, Math.min(newStart, periodHistory.length - 6));
      newEnd = Math.min(periodHistory.length - 1, Math.max(newEnd, newStart + 5));

      zoomRange = {{ start: newStart, end: newEnd }};
      isZoomed = (zoomRange.start > 0 || zoomRange.end < periodHistory.length - 1);
      filteredHistory = periodHistory.slice(zoomRange.start, zoomRange.end + 1);

      updateZoomBadge();
      renderHistoricalChart();
    }}

    function zoomRelative(factor) {{
      if (periodHistory.length === 0) return;
      const currentLen = zoomRange.end - zoomRange.start + 1;
      const newLen = Math.round(currentLen * factor);
      const mid = Math.round((zoomRange.start + zoomRange.end) / 2);
      const half = Math.round(newLen / 2);
      applyZoom(mid - half, mid + half);
    }}

    function resetZoom() {{
      if (periodHistory.length === 0) return;
      zoomRange = {{ start: 0, end: periodHistory.length - 1 }};
      isZoomed = false;
      filteredHistory = periodHistory.slice(zoomRange.start, zoomRange.end + 1);
      updateZoomBadge();
      renderHistoricalChart();
    }}

    function zoomToTrade(buyDate, peakDate) {{
      const bIdx = periodHistory.findIndex(d => d.date === buyDate);
      if (bIdx < 0) return;
      const pIdx = peakDate ? periodHistory.findIndex(d => d.date === peakDate) : bIdx;
      const targetEnd = (pIdx >= bIdx) ? pIdx : bIdx;

      // Expand window by ~15 days before and ~25 days after
      const newStart = Math.max(0, bIdx - 15);
      const newEnd = Math.min(periodHistory.length - 1, targetEnd + 25);
      highlightedDate = buyDate;
      applyZoom(newStart, newEnd);
    }}

    function updateBestBuysTable(buys) {{
      const tbody = document.getElementById('bestBuysTableBody');
      tbody.innerHTML = '';
      if (!buys || buys.length === 0) {{
        tbody.innerHTML = '<tr><td colspan="8" class="text-center py-4 text-gray-500">No qualifying buy points detected in this timeframe.</td></tr>';
        return;
      }}
      buys.forEach(b => {{
        const tr = document.createElement('tr');
        tr.className = "hover:bg-gray-800/40 transition cursor-pointer";
        tr.onclick = () => zoomToTrade(b.date, b.peak_date);
        tr.title = "Click to zoom into this trade";
        tr.innerHTML = `
          <td class="py-3 px-4 font-semibold text-white flex items-center gap-1.5">
            <span class="text-amber-400">&#9733;</span> ${{b.date}}
          </td>
          <td class="py-3 px-4 font-bold text-gray-200">$${{b.price.toFixed(2)}}</td>
          <td class="py-3 px-4 text-emerald-400 font-bold">$${{b.peak_price.toFixed(2)}}</td>
          <td class="py-3 px-4 text-gray-400">${{b.peak_date}}</td>
          <td class="py-3 px-4 text-gray-400">${{b.holding_days}} days</td>
          <td class="py-3 px-4 text-emerald-400 font-bold">+${{b.max_gain_pct.toFixed(1)}}%</td>
          <td class="py-3 px-4 ${{b.return_to_now_pct >= 0 ? 'text-emerald-400' : 'text-red-400'}} font-semibold">
            ${{b.return_to_now_pct >= 0 ? '+' : ''}}${{b.return_to_now_pct.toFixed(1)}}%
          </td>
          <td class="py-3 px-4 text-gray-300">${{b.rationale}}</td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    // Dedicated Timeline Chart Renderer
    function updateTimelineTrack(buys) {{
      const container = document.getElementById('buyTimelineTrack');
      if (!container) return;
      container.innerHTML = '';

      if (!buys || buys.length === 0) {{
        container.innerHTML = '<div class="text-center py-4 text-gray-500 text-xs">No buy milestones detected in this timeframe.</div>';
        return;
      }}

      const sortedBuys = [...buys].sort((a, b) => a.date.localeCompare(b.date));
      const firstDate = periodHistory[0]?.date || sortedBuys[0].date;
      const lastDate = periodHistory[periodHistory.length - 1]?.date || sortedBuys[sortedBuys.length - 1].date;

      let html = `
        <div class="space-y-4">
          <!-- Horizontal Milestone Axis Rail -->
          <div class="relative w-full h-11 flex items-center px-6 bg-gray-950/60 rounded-xl border border-gray-800/80">
            <div class="absolute left-8 right-8 h-1 bg-gradient-to-r from-blue-600 via-amber-500 to-emerald-500 rounded-full opacity-50"></div>
            <div class="absolute left-2 text-[10px] text-gray-500 font-mono">${{firstDate}}</div>
            <div class="absolute right-2 text-[10px] text-gray-500 font-mono">${{lastDate}}</div>
      `;

      sortedBuys.forEach((b, i) => {{
        const idx = periodHistory.findIndex(d => d.date === b.date);
        const pct = (idx >= 0 && periodHistory.length > 1)
          ? Math.min(94, Math.max(6, (idx / (periodHistory.length - 1)) * 100))
          : ((i + 1) / (sortedBuys.length + 1)) * 100;

        html += `
          <div style="left: ${{pct}}%;" class="absolute -translate-x-1/2 flex flex-col items-center group cursor-pointer" onclick="zoomToTrade('${{b.date}}', '${{b.peak_date}}')" title="Click to zoom in on trade: ${{b.date}} at $${{b.price.toFixed(2)}} (+${{b.max_gain_pct.toFixed(1)}}%)">
            <span class="w-4 h-4 rounded-full bg-amber-400 border-2 border-white shadow-lg group-hover:scale-125 transition flex items-center justify-center text-[8px] text-black font-black">&#9733;</span>
            <span class="text-[9px] font-mono text-amber-300 opacity-80 group-hover:opacity-100 whitespace-nowrap mt-0.5">${{b.date.slice(2)}}</span>
          </div>
        `;
      }});

      html += `
          </div>
          <!-- Milestone Detail Cards Grid -->
          <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
      `;

      sortedBuys.forEach(b => {{
        html += `
          <div id="timeline-card-${{b.date}}" class="bg-gray-950/70 border border-gray-800 hover:border-amber-500/60 rounded-xl p-3 shadow-sm hover:shadow-amber-500/10 transition cursor-pointer" onclick="zoomToTrade('${{b.date}}', '${{b.peak_date}}')">
            <div class="flex items-center justify-between mb-1.5">
              <span class="text-xs font-bold text-white flex items-center gap-1">
                <span class="text-amber-400">&#9733;</span> ${{b.date}}
              </span>
              <span class="text-xs font-black text-emerald-400">+${{b.max_gain_pct.toFixed(1)}}%</span>
            </div>
            <div class="text-xs text-gray-300 font-semibold mb-1">
              $${{b.price.toFixed(2)}} &rarr; $${{b.peak_price.toFixed(2)}}
            </div>
            <div class="flex items-center justify-between text-[11px] text-gray-400 border-t border-gray-800/80 pt-1.5 mb-1.5">
              <span>Peak: ${{b.peak_date}}</span>
              <span class="bg-gray-800 px-1.5 py-0.5 rounded text-[10px] text-gray-300">${{b.holding_days}}d</span>
            </div>
            <div class="flex items-center justify-between gap-1">
              <span class="text-[10px] text-gray-400 truncate" title="${{b.rationale}}">${{b.rationale}}</span>
              <span class="text-[10px] text-blue-400 bg-blue-500/10 border border-blue-500/20 px-1.5 py-0.5 rounded shrink-0 font-medium">🔍 Zoom</span>
            </div>
          </div>
        `;
      }});

      html += `
          </div>
        </div>
      `;
      container.innerHTML = html;
    }}

    let highlightedDate = null;
    function highlightBuy(date) {{
      highlightedDate = (highlightedDate === date) ? null : date;
      renderHistoricalChart();

      // Highlight corresponding card
      document.querySelectorAll('[id^="timeline-card-"]').forEach(el => {{
        el.classList.remove('border-amber-400', 'bg-amber-950/20');
      }});
      if (highlightedDate) {{
        const card = document.getElementById('timeline-card-' + highlightedDate);
        if (card) {{
          card.classList.add('border-amber-400', 'bg-amber-950/20');
          card.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
        }}
      }}
    }}

    function setPeriod(period) {{
      currentPeriod = period;
      highlightedDate = null;
      ['1y', '3y', '5y', 'all'].forEach(p => {{
        const btn = document.getElementById('btn-' + p);
        if (p.toUpperCase() === period) {{
          btn.className = "px-3 py-1 rounded font-semibold transition bg-blue-600 text-white";
        }} else {{
          btn.className = "px-3 py-1 rounded font-medium transition text-gray-400 hover:text-white";
        }}
      }});
      filterData(period);
      renderHistoricalChart();
    }}

    // Canvas Chart Rendering Engine
    function renderHistoricalChart() {{
      const canvas = document.getElementById('historicalChart');
      const ctx = canvas.getContext('2d');
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();

      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);

      const width = rect.width;
      const height = rect.height;
      const padding = {{ top: 25, right: 60, bottom: 40, left: 20 }};
      const plotWidth = width - padding.left - padding.right;
      const plotHeight = height - padding.top - padding.bottom;

      ctx.clearRect(0, 0, width, height);

      if (filteredHistory.length === 0) return;

      const prices = filteredHistory.map(d => d.close);
      const sma50 = filteredHistory.map(d => (d.sma50 !== undefined ? d.sma50 : null));
      const sma200 = filteredHistory.map(d => (d.sma200 !== undefined ? d.sma200 : null));
      const avwapYtd = filteredHistory.map(d => (d.avwap_ytd !== undefined ? d.avwap_ytd : null));
      const avwapUpper = filteredHistory.map(d => (d.avwap_ytd_upper_1s !== undefined ? d.avwap_ytd_upper_1s : null));
      const avwapLower = filteredHistory.map(d => (d.avwap_ytd_lower_1s !== undefined ? d.avwap_ytd_lower_1s : null));

      // Calculate minPrice and maxPrice considering visible prices and MAs to avoid clipping
      const allVisibleValues = [
        ...prices,
        ...sma50.filter(v => v !== null && v !== undefined),
        ...sma200.filter(v => v !== null && v !== undefined),
        ...avwapYtd.filter(v => v !== null && v !== undefined),
      ];
      const minPrice = (allVisibleValues.length > 0 ? Math.min(...allVisibleValues) : 10) * 0.94;
      const maxPrice = (allVisibleValues.length > 0 ? Math.max(...allVisibleValues) : 100) * 1.06;

      // Coordinate mapping
      const getX = idx => padding.left + (idx / (filteredHistory.length - 1)) * plotWidth;
      const getY = price => padding.top + plotHeight - ((price - minPrice) / (maxPrice - minPrice)) * plotHeight;

      // Draw Grid & Axes
      ctx.strokeStyle = '#1f2937';
      ctx.lineWidth = 1;
      const gridCount = 5;
      for (let i = 0; i <= gridCount; i++) {{
        const y = padding.top + (i / gridCount) * plotHeight;
        const pVal = maxPrice - (i / gridCount) * (maxPrice - minPrice);
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(width - padding.right, y);
        ctx.stroke();

        ctx.fillStyle = '#6b7280';
        ctx.font = '10px monospace';
        ctx.textAlign = 'left';
        ctx.fillText('$' + pVal.toFixed(0), width - padding.right + 8, y + 3);
      }}

      // 1. Shaded Profit Rally Corridors (from Buy Date to Subsequent Peak Date)
      currentBestBuys.forEach(b => {{
        const bIdx = filteredHistory.findIndex(d => d.date === b.date);
        const pIdx = filteredHistory.findIndex(d => d.date === b.peak_date);
        if (bIdx >= 0 && pIdx > bIdx) {{
          const x1 = getX(bIdx);
          const x2 = getX(pIdx);
          const w = x2 - x1;

          const isHl = (highlightedDate === b.date);
          const gradRally = ctx.createLinearGradient(0, padding.top, 0, height - padding.bottom);
          gradRally.addColorStop(0, isHl ? 'rgba(16, 185, 129, 0.35)' : 'rgba(16, 185, 129, 0.18)');
          gradRally.addColorStop(1, isHl ? 'rgba(16, 185, 129, 0.08)' : 'rgba(16, 185, 129, 0.02)');

          ctx.fillStyle = gradRally;
          ctx.fillRect(x1, padding.top, w, plotHeight);

          // Top and border dashed lines
          ctx.strokeStyle = isHl ? 'rgba(16, 185, 129, 0.8)' : 'rgba(16, 185, 129, 0.4)';
          ctx.lineWidth = isHl ? 1.5 : 1;
          ctx.setLineDash([3, 3]);
          ctx.strokeRect(x1, padding.top, w, plotHeight);
          ctx.setLineDash([]);
        }}
      }});

      // Draw 200 SMA
      ctx.strokeStyle = '#c084fc';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < sma200.length; i++) {{
        if (sma200[i] !== null) {{
          const x = getX(i);
          const y = getY(sma200[i]);
          if (!started) {{ ctx.moveTo(x, y); started = true; }} else {{ ctx.lineTo(x, y); }}
        }}
      }}
      ctx.stroke();

      // Draw 50 SMA
      ctx.strokeStyle = '#fbbf24';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      started = false;
      for (let i = 0; i < sma50.length; i++) {{
        if (sma50[i] !== null) {{
          const x = getX(i);
          const y = getY(sma50[i]);
          if (!started) {{ ctx.moveTo(x, y); started = true; }} else {{ ctx.lineTo(x, y); }}
        }}
      }}
      ctx.stroke();

      // Draw YTD AVWAP & +/-1 sigma envelope
      let startedChannel = false;
      ctx.fillStyle = 'rgba(6, 182, 212, 0.08)';
      ctx.beginPath();
      for (let i = 0; i < avwapUpper.length; i++) {{
        if (avwapUpper[i] !== null && avwapLower[i] !== null) {{
          const x = getX(i);
          const y = getY(avwapUpper[i]);
          if (!startedChannel) {{ ctx.moveTo(x, y); startedChannel = true; }} else {{ ctx.lineTo(x, y); }}
        }}
      }}
      for (let i = avwapLower.length - 1; i >= 0; i--) {{
        if (avwapUpper[i] !== null && avwapLower[i] !== null) {{
          const x = getX(i);
          const y = getY(avwapLower[i]);
          ctx.lineTo(x, y);
        }}
      }}
      ctx.closePath();
      if (startedChannel) ctx.fill();

      // YTD AVWAP Line
      ctx.strokeStyle = '#06b6d4';
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      let startedAvwap = false;
      for (let i = 0; i < avwapYtd.length; i++) {{
        if (avwapYtd[i] !== null) {{
          const x = getX(i);
          const y = getY(avwapYtd[i]);
          if (!startedAvwap) {{ ctx.moveTo(x, y); startedAvwap = true; }} else {{ ctx.lineTo(x, y); }}
        }}
      }}
      ctx.stroke();

      // Draw Price Line with subtle gradient fill
      const grad = ctx.createLinearGradient(0, padding.top, 0, height - padding.bottom);
      grad.addColorStop(0, 'rgba(59, 130, 246, 0.25)');
      grad.addColorStop(1, 'rgba(59, 130, 246, 0.0)');

      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.moveTo(getX(0), getY(prices[0]));
      for (let i = 1; i < prices.length; i++) {{
        ctx.lineTo(getX(i), getY(prices[i]));
      }}
      ctx.lineTo(getX(prices.length - 1), height - padding.bottom);
      ctx.lineTo(getX(0), height - padding.bottom);
      ctx.closePath();
      ctx.fill();

      // Price Line Stroke
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(getX(0), getY(prices[0]));
      for (let i = 1; i < prices.length; i++) {{
        ctx.lineTo(getX(i), getY(prices[i]));
      }}
      ctx.stroke();

      // Highlight Historical Best Buy Points, Vertical Guidelines, and Overlay Callout Badges
      currentBestBuys.forEach((b, i) => {{
        const idx = filteredHistory.findIndex(d => d.date === b.date);
        if (idx >= 0) {{
          const x = getX(idx);
          const y = getY(b.price);
          const isHl = (highlightedDate === b.date);

          // Vertical timeline guideline
          ctx.strokeStyle = isHl ? 'rgba(245, 158, 11, 0.9)' : 'rgba(245, 158, 11, 0.45)';
          ctx.lineWidth = isHl ? 1.8 : 1.2;
          ctx.setLineDash([3, 3]);
          ctx.beginPath();
          ctx.moveTo(x, padding.top);
          ctx.lineTo(x, height - padding.bottom);
          ctx.stroke();
          ctx.setLineDash([]);

          // Outer Glow
          ctx.fillStyle = isHl ? 'rgba(245, 158, 11, 0.6)' : 'rgba(245, 158, 11, 0.35)';
          ctx.beginPath();
          ctx.arc(x, y, isHl ? 14 : 10, 0, Math.PI * 2);
          ctx.fill();

          // Star / Core Dot
          ctx.fillStyle = '#f59e0b';
          ctx.beginPath();
          ctx.arc(x, y, 5, 0, Math.PI * 2);
          ctx.fill();

          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 1.5;
          ctx.stroke();

          // Overlay Callout Flag Badge
          const tagText = `★ Buy $${{b.price.toFixed(0)}} (+${{b.max_gain_pct.toFixed(0)}}%)`;
          ctx.font = 'bold 9px monospace';
          const tagW = ctx.measureText(tagText).width + 12;
          const tagH = 17;
          // Stagger badges vertically so they don't collide
          const yOffset = (i % 2 === 0) ? 26 : 46;
          const tagY = Math.max(padding.top + 2, y - yOffset);
          const tagX = Math.min(width - padding.right - tagW - 2, Math.max(padding.left + 2, x - tagW / 2));

          // Pin stem
          ctx.strokeStyle = isHl ? '#f59e0b' : '#d97706';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(x, tagY + tagH);
          ctx.lineTo(x, y - 6);
          ctx.stroke();

          // Badge background
          ctx.fillStyle = isHl ? '#312e81' : '#1e1b4b';
          ctx.strokeStyle = isHl ? '#fbbf24' : '#f59e0b';
          ctx.lineWidth = isHl ? 1.5 : 1;
          ctx.beginPath();
          if (ctx.roundRect) {{
            ctx.roundRect(tagX, tagY, tagW, tagH, 4);
          }} else {{
            ctx.rect(tagX, tagY, tagW, tagH);
          }}
          ctx.fill();
          ctx.stroke();

          // Badge text
          ctx.fillStyle = isHl ? '#ffffff' : '#fef08a';
          ctx.textAlign = 'center';
          ctx.fillText(tagText, tagX + tagW / 2, tagY + 11);
        }}
      }});

      // Draw Bottom X-Axis with Dynamic Zoom-Relative Date Labels
      const numDateTicks = Math.min(8, Math.max(4, Math.floor(plotWidth / 85)));
      ctx.fillStyle = '#9ca3af';
      ctx.font = '10px monospace';
      ctx.textAlign = 'center';

      // X-Axis Baseline Line
      ctx.strokeStyle = '#374151';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padding.left, padding.top + plotHeight);
      ctx.lineTo(width - padding.right, padding.top + plotHeight);
      ctx.stroke();

      for (let i = 0; i < numDateTicks; i++) {{
        const idx = Math.round((i / (numDateTicks - 1)) * (filteredHistory.length - 1));
        const d = filteredHistory[idx];
        if (!d) continue;
        const x = getX(idx);

        // Vertical subtle dotted grid line
        ctx.strokeStyle = '#1e293b';
        ctx.beginPath();
        ctx.setLineDash([2, 3]);
        ctx.moveTo(x, padding.top);
        ctx.lineTo(x, padding.top + plotHeight);
        ctx.stroke();
        ctx.setLineDash([]);

        // Small tick mark
        ctx.strokeStyle = '#4b5563';
        ctx.beginPath();
        ctx.moveTo(x, padding.top + plotHeight);
        ctx.lineTo(x, padding.top + plotHeight + 5);
        ctx.stroke();

        // Formatted Date Label relative to current zoom level
        const dateLabel = formatXAxisDate(d.date, filteredHistory.length);
        ctx.fillText(dateLabel, x, padding.top + plotHeight + 18);
      }}
    }}

    // Setup interactive crosshair and tooltip on historicalChart
    const histCanvas = document.getElementById('historicalChart');
    const histTooltip = document.getElementById('chartTooltip');

    histCanvas.addEventListener('mousemove', (e) => {{
      if (filteredHistory.length === 0) return;
      const rect = histCanvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      const padding = {{ top: 25, right: 60, bottom: 40, left: 20 }};
      const plotWidth = rect.width - padding.left - padding.right;

      if (mouseX < padding.left || mouseX > rect.width - padding.right) {{
        histTooltip.classList.add('hidden');
        return;
      }}

      const ratio = (mouseX - padding.left) / plotWidth;
      const idx = Math.min(filteredHistory.length - 1, Math.max(0, Math.round(ratio * (filteredHistory.length - 1))));
      const d = filteredHistory[idx];

      // Check if near any Best Buy point
      const nearBuy = currentBestBuys.find(b => b.date === d.date);

      let tooltipHtml = `
        <div class="font-bold text-white mb-1">${{d.date}}</div>
        <div class="text-blue-400 font-semibold">Close: $${{d.close.toFixed(2)}}</div>
        ${{d.sma50 !== null && d.sma50 !== undefined ? `<div class="text-amber-400 text-xs font-medium mt-0.5">50 MA: $${{Number(d.sma50).toFixed(2)}}</div>` : ''}}
        ${{d.sma200 !== null && d.sma200 !== undefined ? `<div class="text-purple-400 text-xs font-medium mt-0.5">200 MA: $${{Number(d.sma200).toFixed(2)}}</div>` : ''}}
        ${{d.avwap_ytd !== null && d.avwap_ytd !== undefined ? `<div class="text-cyan-400 text-xs font-medium mt-0.5">YTD AVWAP: $${{Number(d.avwap_ytd).toFixed(2)}}</div>` : ''}}
      `;
      if (nearBuy) {{
        tooltipHtml += `
          <div class="mt-2 pt-2 border-t border-gray-700 text-amber-300">
            <div class="font-bold text-xs flex items-center gap-1"><span>&#9733;</span> Best Buy Point</div>
            <div class="text-emerald-400 font-bold mt-0.5">Peak: $${{nearBuy.peak_price.toFixed(2)}} (+${{nearBuy.max_gain_pct.toFixed(1)}}%)</div>
            <div class="text-gray-400 text-[10px]">Held: ${{nearBuy.holding_days}} days to ${{nearBuy.peak_date}}</div>
            <div class="text-gray-300 text-[10px] mt-1">${{nearBuy.rationale}}</div>
          </div>
        `;
      }}

      histTooltip.innerHTML = tooltipHtml;
      histTooltip.classList.remove('hidden');

      // Position tooltip
      let ttX = mouseX + 15;
      let ttY = mouseY - 20;
      if (ttX + 180 > rect.width) ttX = mouseX - 195;
      if (ttY < 10) ttY = 10;
      histTooltip.style.left = ttX + 'px';
      histTooltip.style.top = ttY + 'px';
    }});

    histCanvas.addEventListener('mouseleave', () => {{
      histTooltip.classList.add('hidden');
    }});

    // Drag-to-Zoom and Wheel Zoom Implementation
    let isDragging = false;
    let dragStartX = 0;
    const zoomBox = document.getElementById('zoomSelectionBox');

    histCanvas.addEventListener('mousedown', (e) => {{
      if (filteredHistory.length < 5) return;
      const rect = histCanvas.getBoundingClientRect();
      const padding = {{ top: 25, right: 60, bottom: 40, left: 20 }};
      const mouseX = e.clientX - rect.left;
      if (mouseX < padding.left || mouseX > rect.width - padding.right) return;

      isDragging = true;
      dragStartX = mouseX;
      zoomBox.style.left = dragStartX + 'px';
      zoomBox.style.width = '0px';
      zoomBox.classList.remove('hidden');
    }});

    window.addEventListener('mousemove', (e) => {{
      if (!isDragging) return;
      const rect = histCanvas.getBoundingClientRect();
      const padding = {{ top: 25, right: 60, bottom: 40, left: 20 }};
      const currentX = Math.max(padding.left, Math.min(rect.width - padding.right, e.clientX - rect.left));

      const left = Math.min(dragStartX, currentX);
      const width = Math.abs(currentX - dragStartX);
      zoomBox.style.left = left + 'px';
      zoomBox.style.width = width + 'px';
    }});

    window.addEventListener('mouseup', (e) => {{
      if (!isDragging) return;
      isDragging = false;
      zoomBox.classList.add('hidden');

      const rect = histCanvas.getBoundingClientRect();
      const padding = {{ top: 25, right: 60, bottom: 40, left: 20 }};
      const plotWidth = rect.width - padding.left - padding.right;
      const currentX = Math.max(padding.left, Math.min(rect.width - padding.right, e.clientX - rect.left));

      const minX = Math.min(dragStartX, currentX) - padding.left;
      const maxX = Math.max(dragStartX, currentX) - padding.left;

      if (maxX - minX < 15) return; // Ignore tiny accidental clicks

      const startRatio = Math.max(0, minX / plotWidth);
      const endRatio = Math.min(1, maxX / plotWidth);

      const subStart = Math.floor(startRatio * (filteredHistory.length - 1));
      const subEnd = Math.ceil(endRatio * (filteredHistory.length - 1));

      if (subEnd - subStart >= 4) {{
        const actualStart = zoomRange.start + subStart;
        const actualEnd = zoomRange.start + subEnd;
        applyZoom(actualStart, actualEnd);
      }}
    }});

    // Mouse wheel zoom
    histCanvas.addEventListener('wheel', (e) => {{
      e.preventDefault();
      if (periodHistory.length === 0) return;
      const rect = histCanvas.getBoundingClientRect();
      const padding = {{ top: 25, right: 60, bottom: 40, left: 20 }};
      const plotWidth = rect.width - padding.left - padding.right;
      const mouseX = e.clientX - rect.left;

      const factor = (e.deltaY < 0) ? 0.75 : 1.35;
      const ratio = Math.max(0, Math.min(1, (mouseX - padding.left) / plotWidth));
      const currentLen = zoomRange.end - zoomRange.start + 1;
      const focalIdx = zoomRange.start + Math.round(ratio * currentLen);
      const newLen = Math.max(6, Math.round(currentLen * factor));
      const newStart = Math.round(focalIdx - ratio * newLen);
      const newEnd = newStart + newLen - 1;
      applyZoom(newStart, newEnd);
    }}, {{ passive: false }});

    // Render 3-Month Forward Forecast Chart
    function renderForecastChart() {{
      const canvas = document.getElementById('forecastChart');
      const ctx = canvas.getContext('2d');
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();

      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);

      const width = rect.width;
      const height = rect.height;
      const padding = {{ top: 20, right: 60, bottom: 40, left: 20 }};
      const plotWidth = width - padding.left - padding.right;
      const plotHeight = height - padding.top - padding.bottom;

      ctx.clearRect(0, 0, width, height);

      // Join last 40 trading days with 63 forecast days
      const recentHistory = RAW_HISTORY.slice(-40);
      const forecastSeries = PREDICTIVE.forecast_series || [];
      const totalPoints = recentHistory.length + forecastSeries.length;

      const allPrices = [
        ...recentHistory.map(d => d.close),
        ...forecastSeries.map(d => d.bull_p90),
        ...forecastSeries.map(d => d.bear_p10),
      ];
      const minPrice = Math.min(...allPrices) * 0.95;
      const maxPrice = Math.max(...allPrices) * 1.05;

      const getX = idx => padding.left + (idx / (totalPoints - 1)) * plotWidth;
      const getY = price => padding.top + plotHeight - ((price - minPrice) / (maxPrice - minPrice)) * plotHeight;

      // Draw Grid & Axes
      ctx.strokeStyle = '#1f2937';
      ctx.lineWidth = 1;
      const gridCount = 4;
      for (let i = 0; i <= gridCount; i++) {{
        const y = padding.top + (i / gridCount) * plotHeight;
        const pVal = maxPrice - (i / gridCount) * (maxPrice - minPrice);
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(width - padding.right, y);
        ctx.stroke();

        ctx.fillStyle = '#6b7280';
        ctx.font = '10px monospace';
        ctx.textAlign = 'left';
        ctx.fillText('$' + pVal.toFixed(0), width - padding.right + 8, y + 3);
      }}

      // Shaded Forecast Corridor (10th to 90th percentile)
      if (forecastSeries.length > 0) {{
        const histLen = recentHistory.length;
        ctx.fillStyle = 'rgba(16, 185, 129, 0.12)';
        ctx.beginPath();
        ctx.moveTo(getX(histLen - 1), getY(recentHistory[histLen - 1].close));

        // Upper corridor (bull_p90)
        for (let i = 0; i < forecastSeries.length; i++) {{
          ctx.lineTo(getX(histLen + i), getY(forecastSeries[i].bull_p90));
        }}
        // Lower corridor (bear_p10) backwards
        for (let i = forecastSeries.length - 1; i >= 0; i--) {{
          ctx.lineTo(getX(histLen + i), getY(forecastSeries[i].bear_p10));
        }}
        ctx.closePath();
        ctx.fill();

        // Shaded Optimal Buy Zone Box
        const optStartIdx = forecastSeries.findIndex(f => f.date === PREDICTIVE.optimal_buy_window.start_date);
        const optEndIdx = forecastSeries.findIndex(f => f.date === PREDICTIVE.optimal_buy_window.end_date);
        if (optStartIdx >= 0 && optEndIdx >= 0) {{
          const bx1 = getX(histLen + optStartIdx);
          const bx2 = getX(histLen + optEndIdx);
          const by1 = getY(PREDICTIVE.optimal_entry_range[1]);
          const by2 = getY(PREDICTIVE.optimal_entry_range[0]);

          ctx.fillStyle = 'rgba(59, 130, 246, 0.25)';
          ctx.fillRect(bx1, by1, bx2 - bx1, by2 - by1);

          ctx.strokeStyle = '#3b82f6';
          ctx.lineWidth = 1;
          ctx.setLineDash([4, 4]);
          ctx.strokeRect(bx1, by1, bx2 - bx1, by2 - by1);
          ctx.setLineDash([]);
        }}
      }}

      // Draw Key Support Line
      if (PREDICTIVE.key_support) {{
        const ySupp = getY(PREDICTIVE.key_support);
        ctx.strokeStyle = 'rgba(16, 185, 129, 0.6)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(padding.left, ySupp);
        ctx.lineTo(width - padding.right, ySupp);
        ctx.stroke();
        ctx.setLineDash([]);
      }}

      // Draw Recent History Price Line
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(getX(0), getY(recentHistory[0].close));
      for (let i = 1; i < recentHistory.length; i++) {{
        ctx.lineTo(getX(i), getY(recentHistory[i].close));
      }}
      ctx.stroke();

      // Draw Median Forecast Trajectory Line
      const histLen = recentHistory.length;
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(getX(histLen - 1), getY(recentHistory[histLen - 1].close));
      for (let i = 0; i < forecastSeries.length; i++) {{
        ctx.lineTo(getX(histLen + i), getY(forecastSeries[i].median_p50));
      }}
      ctx.stroke();
      ctx.setLineDash([]);

      // Vertical line separating history and forecast
      const splitX = getX(histLen - 1);
      ctx.strokeStyle = '#4b5563';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(splitX, padding.top);
      ctx.lineTo(splitX, height - padding.bottom);
      ctx.stroke();

      ctx.fillStyle = '#9ca3af';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText('Current Date', splitX - 5, padding.top + 12);
      ctx.textAlign = 'left';
      ctx.fillText('3M Forecast \u2192', splitX + 5, padding.top + 12);

      // Draw Bottom X-Axis with Dynamic Dates for Forecast Chart
      const combinedDates = [
        ...recentHistory.map(d => d.date),
        ...forecastSeries.map(d => d.date)
      ];
      const numForecastDateTicks = Math.min(7, Math.max(4, Math.floor(plotWidth / 95)));
      ctx.fillStyle = '#9ca3af';
      ctx.font = '10px monospace';
      ctx.textAlign = 'center';

      // X-Axis Baseline Line
      ctx.strokeStyle = '#374151';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padding.left, padding.top + plotHeight);
      ctx.lineTo(width - padding.right, padding.top + plotHeight);
      ctx.stroke();

      for (let i = 0; i < numForecastDateTicks; i++) {{
        const idx = Math.round((i / (numForecastDateTicks - 1)) * (combinedDates.length - 1));
        const dt = combinedDates[idx];
        if (!dt) continue;
        const x = getX(idx);

        // Subtle vertical dotted line
        ctx.strokeStyle = '#1e293b';
        ctx.beginPath();
        ctx.setLineDash([2, 3]);
        ctx.moveTo(x, padding.top);
        ctx.lineTo(x, padding.top + plotHeight);
        ctx.stroke();
        ctx.setLineDash([]);

        // Small tick mark
        ctx.strokeStyle = '#4b5563';
        ctx.beginPath();
        ctx.moveTo(x, padding.top + plotHeight);
        ctx.lineTo(x, padding.top + plotHeight + 5);
        ctx.stroke();

        // Formatted Date Label
        const dateLabel = formatXAxisDate(dt, combinedDates.length);
        ctx.fillText(dateLabel, x, padding.top + plotHeight + 18);
      }}
    }}

    // Initial setup on load
    window.addEventListener('load', () => {{
      initMovingAverages();
      filterData('5Y');
      renderHistoricalChart();
      renderForecastChart();
    }});

    window.addEventListener('resize', () => {{
      renderHistoricalChart();
      renderForecastChart();
    }});
  </script>
</body>
</html>
"""
    output_file.write_text(html_content, encoding="utf-8")
    logger.info(f"Interactive visual display successfully generated at: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate visual performance and predictive buy timing dashboard for a stock using Qlib data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbol",
        "-s",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. MSFT, VOO, NVDA).",
    )
    parser.add_argument(
        "--data_dir",
        "-d",
        type=str,
        default="~/.qlib/qlib_data/us_data",
        help="Directory containing Qlib binary dataset or CSV files.",
    )
    parser.add_argument(
        "--report_dir",
        "-r",
        type=str,
        default="reports",
        help="Directory where the generated HTML report will be stored (defaults to 'reports').",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Specific output file path or custom directory for generated HTML report (overrides --report_dir).",
    )
    parser.add_argument(
        "--days_forecast",
        type=int,
        default=63,
        help="Number of forward trading days to simulate (~63 days = 3 months).",
    )
    parser.add_argument(
        "--auto_download",
        action="store_true",
        default=True,
        help="Automatically download stock data if missing from data_dir.",
    )
    parser.add_argument(
        "--no-auto_download",
        dest="auto_download",
        action="store_false",
        help="Do not auto-download; fail if stock data is not already in data_dir.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2000-01-01",
        help="Start date if auto-download is triggered (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--request_date",
        "--report_date",
        type=str,
        default=None,
        help="Date the report was requested (YYYY-MM-DD). Defaults to current date.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        default=False,
        help="Automatically open the generated visual display in the default web browser.",
    )

    args = parser.parse_args()

    symbol = args.symbol.upper()
    data_dir = args.data_dir
    req_date = args.request_date if args.request_date else datetime.date.today().strftime("%Y-%m-%d")
    output_path = resolve_report_path(symbol, report_dir=args.report_dir, output=args.output, report_date=req_date)

    # Run analytical engine (ensures data is up-to-date for req_date before running)
    analysis_data = run_stock_analysis(
        symbol=symbol,
        data_dir=data_dir,
        forecast_days=args.days_forecast,
        auto_download=args.auto_download,
        start=args.start,
        request_date=req_date,
    )

    print(f"\n=======================================================")
    print(f" STOCK PERFORMANCE & PREDICTIVE BUY TIMING ANALYZER ")
    print(f"=======================================================")
    print(f"Symbol:           {symbol}")
    print(f"Report Requested: {req_date}")
    print(f"Data Directory:   {data_dir}")
    print(f"Data Freshness:   Through {analysis_data['latest_data_date']} ({'Up-to-Date' if analysis_data['is_up_to_date'] else 'Latest available'})")
    print(f"Auto Download:    {args.auto_download}")
    print(f"Forecast Days:    {args.days_forecast} (~3 months)")
    print(f"Output Report:    {output_path}\n")

    # Generate visual dashboard
    report_file = generate_html_dashboard(analysis_data, output_path)

    # Print summary to terminal
    perf = analysis_data["performance"]
    pred = analysis_data["predictive"]

    print("\n-------------------------------------------------------")
    print(" HISTORICAL PERFORMANCE SUMMARY (1Y / 3Y / 5Y)")
    print("-------------------------------------------------------")
    for y in ["1Y", "3Y", "5Y"]:
        p_data = perf["periods"].get(y, {})
        if p_data.get("available"):
            print(f"[{y}] Return: {p_data['total_return_pct']:+.1f}% | CAGR: {p_data['cagr_pct']:.1f}% | Max DD: {p_data['max_drawdown_pct']:.1f}% | Sharpe: {p_data['sharpe_ratio']}")
        else:
            print(f"[{y}] {p_data.get('reason', 'N/A')}")

    regime = analysis_data.get("regime")
    if regime:
        print("\n-------------------------------------------------------")
        print(" BAYESIAN ONLINE CHANGEPOINT (BOCD) & MARKET REGIME")
        print("-------------------------------------------------------")
        print(f"Current State:      State {regime['state']} - {regime['name']}")
        print(f"Action Guidance:    {regime['action']}")
        print(f"Changepoint Hazard: {regime['changepoint_prob_pct']}% (Active Run-Length: {regime['expected_run_length_days']} days)")
        print(f"Volatility Surface: 21d Vol: {regime['vol_21d_pct']}% | Term Ratio (5d/21d): {regime['vol_ratio']}x")
        print(f"Macro Risk Sizing:  {regime['risk_multiplier']}x exposure (Credit momentum: {regime['credit_mom_pct']:+.2f}%)")

    micro = analysis_data.get("microstructure")
    if micro:
        av = micro.get("avwap", {})
        ytd_info = av.get("ytd", {})
        vp_info = micro.get("volume_profile", {})
        print("\n-------------------------------------------------------")
        print(" INSTITUTIONAL LIQUIDITY, AVWAP & VOLUME PROFILE (KDE)")
        print("-------------------------------------------------------")
        if ytd_info.get("value") is not None:
            print(f"YTD Anchored VWAP:  ${ytd_info['value']:.2f} ({ytd_info.get('spread_pct', 0):+.1f}%, {ytd_info.get('zscore', 0):+.2f}s)")
            print(f"AVWAP +/-1s Envelope: ${ytd_info.get('lower_1s', 0):.2f} - ${ytd_info.get('upper_1s', 0):.2f}")
        if vp_info.get("poc") is not None:
            print(f"Volume Profile POC: ${vp_info['poc']:.2f} ({vp_info.get('dist_to_poc_pct', 0):+.1f}% from close)")
            print(f"70% Value Area:     ${vp_info.get('val', 0):.2f} (VAL) - ${vp_info.get('vah', 0):.2f} (VAH)")
            print(f"Market Depth State: {vp_info.get('void_status')}")

    proj = analysis_data.get("projections", {})
    if proj:
        print("\n-------------------------------------------------------")
        print(" FORWARD RETURN PROJECTIONS & PROBABILITY SCORES")
        print("-------------------------------------------------------")
        for k in ["6M", "1Y", "2Y", "3Y"]:
            p_val = proj.get(k, {})
            if p_val:
                shift_str = f" | Shift Risk: {p_val['bocd_changepoint_prob_pct']:4.1f}%" if p_val.get('bocd_changepoint_prob_pct') is not None else ""
                print(f"[{p_val['label']:8s}] Expected: {p_val['projected_return_pct']:+5.1f}% | Target: ${p_val['base_target_price']:7.2f} | Range: ${p_val['bear_price']:.2f}-${p_val['bull_price']:.2f} | Prob: {p_val['probability_score']:4.1f}% ({p_val['confidence']}){shift_str}")

    print("\n-------------------------------------------------------")
    print(" 3-MONTH PREDICTIVE BUY ANALYSIS")
    print("-------------------------------------------------------")
    print(f"Recommendation:     {pred['recommendation']}")
    if pred.get("bocd_regime_name"):
        cp_fwd = pred.get("bocd_forward_changepoint_prob_pct")
        cp_fwd_str = f"{cp_fwd:.1f}%" if cp_fwd is not None else "N/A"
        print(f"BOCD Regime:        {pred['bocd_regime_name']} (63d Changepoint Risk: {cp_fwd_str})")
    print(f"Action:             {pred['action_summary']}")
    print(f"Optimal Entry Zone: ${pred['optimal_entry_range'][0]:.2f} - ${pred['optimal_entry_range'][1]:.2f}")
    print(f"Optimal Window:     {pred['optimal_buy_window']['description']}")
    print(f"3-Month Target:     ${pred['target_price_3m']:.2f} ({'+' if pred['expected_return_pct'] >= 0 else ''}{pred['expected_return_pct']:.1f}%)")
    print(f"Stop-Loss Level:    ${pred['stop_loss']:.2f}")
    print(f"Risk/Reward Ratio:  {pred['risk_reward_ratio']}:1")
    print("-------------------------------------------------------")
    print(f"\n[SUCCESS] Visual report generated at: {report_file.resolve()}")

    if args.open:
        print("Opening report in your web browser...")
        webbrowser.open(report_file.resolve().as_uri())


if __name__ == "__main__":
    main()
