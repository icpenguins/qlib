#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Russell 1000 Alpha158 Factor + Executive Verdict Cross-Sectional Screen
======================================================================
Builds ONE consolidated institutional dashboard covering every name in the
Russell 1000 universe with, per ticker:

  1. LightGBM Alpha158 factor score + cross-sectional rank + percentile
  2. Executive Investment Verdict  (repo-canonical taxonomy, reused verbatim
     from `scripts/verdict_taxonomy.py` -- the same ladder the single-ticker
     report's `build_buy_timing_verdict_banner_html` renders)
  3. Estimated best price (`optimal_entry_range`) and best buy date
     (`optimal_buy_window.start_date`) from `predict_future_buy_timing`

THIS IS A FAST CROSS-SECTIONAL SCREEN, NOT A SINGLE-TICKER DEEP DIVE.
See `SCREEN_LIMITATIONS` below for exactly which signals are absent; they are
rendered onto the report itself so a reader can never mistake one for the other.

PRICE SOURCE WARNING
--------------------
This module deliberately does NOT use `qlib.data.D.features`. The binary store
at `D:/trading/qlib/qlib_data` is misaligned: feature `.bin` files hold 1930
values against a 1500-entry calendar (857/909 tickers affected), so `D.features`
returns prices shifted ~430 trading days and labels 2024-12-16 data as
2026-09-04. Prices are therefore read from the upstream local source CSVs, which
are offline, complete for all 908 names, and denominated in real dollars.
Full evidence: `.team-code/20260905-russell1000_factor_verdict_screen-implementation_plan.md`.

Usage
-----
    python scripts/russell1000_factor_verdict_screen.py
    python scripts/russell1000_factor_verdict_screen.py --limit 25   # smoke test
"""

from __future__ import annotations

import sys
import json
import time
import html
import logging
import argparse
import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts"), str(REPO_ROOT / "qlib" / "contrib")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.predictive_engine import predict_future_buy_timing
from scripts.stock_analysis_engine import detect_market_regime
from scripts.verdict_taxonomy import (
    classify_executive_verdict,
    TIER_BUY,
    TIER_PULLBACK,
    TIER_CAUTION,
    TIER_NO_BUY,
)

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

DEFAULT_MARKET_DATA_ROOT = Path("D:/trading/qlib")
DEFAULT_UNIVERSE_FILE = REPO_ROOT / "data" / "instruments" / "russell1000.txt"
DEFAULT_SCORES_FILE = REPO_ROOT / "output" / "scores" / "alpha158_russell1000_latest.parquet"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports"

FORECAST_DAYS = 63
SIMULATIONS = 1000
MIN_BARS_REQUIRED = 50  # predict_future_buy_timing's own hard floor

INHIBITED = "ENTRIES INHIBITED"

# Signals present in the single-ticker report that this screen does NOT compute.
# Rendered verbatim onto the report. Adding a signal to the screen means
# deleting its line here -- never leave a stale claim.
SCREEN_LIMITATIONS: List[Tuple[str, str]] = [
    ("Live options chain / open interest",
     "No per-ticker chain is fetched. 908 live chain pulls would rate-limit (cf. the 404/429 fixes in this repo's history)."),
    ("Dealer Gamma Exposure (GEX)",
     "Net GEX, gamma flip, call/put gamma walls and max pain are all absent. GEXParams() defaults are used, so no gamma boundary contributes to support/resistance and no GEX volatility multiplier is applied (vol_multiplier = 1.0)."),
    ("Earnings calendar & PEAD conditioning",
     "No next-earnings date, no SUE, no post-earnings drift, no de-grossing haircut. The Monte Carlo carries no earnings gap jump and no buy-window event shift."),
    ("AVWAP & volume-profile microstructure",
     "Anchored VWAP bands and value-area high/low do not contribute to the support/resistance synthesis."),
    ("Borrow fees / hard-to-borrow state",
     "No short-availability or financing-cost screen is applied."),
    ("Gamma-squeeze 5-day spike engine",
     "GSI scores and calibrated squeeze probabilities are not computed."),
    ("Macro credit-spread regime input",
     "HYG/IEI credit-proxy CSVs are not present in the local market-data root, so the BOCD regime classifier runs with credit_mom_pct = 0.0."),
]

# Verdicts that CANNOT appear in this screen, and why. Stating this is
# mandatory: absent branches must not read as 'evaluated and rejected'.
UNREACHABLE_VERDICTS: List[Tuple[str, str]] = [
    ("IMMEDIATE BUY: HIGH-VELOCITY 5-DAY SPIKE DETECTED",
     "Requires the gamma-squeeze engine (no options chain in this screen)."),
    ("RESEARCH SPIKE PATTERN (ACTION SUPPRESSED: SYNTHETIC DATA)",
     "Requires the gamma-squeeze engine (no options chain in this screen)."),
    ("EVENT RISK / PRE-EARNINGS DE-GROSSING",
     "Requires the earnings calendar (no event conditioning in this screen)."),
    ("IMMINENT CATALYST / 50% DE-GROSSING",
     "Requires the earnings calendar (no event conditioning in this screen)."),
    ("PEAD POST-EARNINGS DRIFT ACCUMULATION",
     "Requires post-earnings drift metrics (no event conditioning in this screen)."),
    ("HOLD / CAUTIOUS BUY  (the amber tier)",
     "RecommendationEngine emits this only from its technical-fallback branch, which fires just when BOCD "
     "returns no regime state. BOCD resolved a state for all 908 names, so no name reached that branch and the "
     "amber tier is empty by construction -- not because every name was judged and found merely cautious."),
]


# ----------------------------------------------------------------------
# Row model
# ----------------------------------------------------------------------

@dataclass
class ScreenRow:
    """
    One fully-resolved ticker row.

    Every field the renderer consumes is declared here. The renderer reads
    attributes off this dataclass rather than `.get()`-ing a dict, so a
    key-name mismatch is an AttributeError at build time instead of a silent
    plausible-looking default -- the bug class remediated on 2026-09-05.
    """
    symbol: str

    # Alpha158 cross-section
    alpha_score: float
    alpha_rank: int
    alpha_percentile: float

    # Verdict
    verdict_badge: str
    verdict_short: str
    verdict_tier: str
    verdict_branch: str
    recommendation: str
    is_capital_preservation: bool

    # Pricing / timing
    current_price: float
    entry_low: Optional[float]
    entry_high: Optional[float]
    best_price_display: str
    best_buy_date: Optional[str]
    best_buy_date_display: str
    buy_window_end: Optional[str]
    buy_window_status: str

    # Supporting context
    target_price_3m: float
    expected_return_pct: float
    stop_loss: float
    risk_reward_ratio: float
    key_support: float
    rsi: float
    bocd_state: Optional[int]
    bocd_regime_name: Optional[str]
    changepoint_hazard_pct: float

    # Per-row integrity flags (rendered, never swallowed)
    integrity_flags: List[str]


# ----------------------------------------------------------------------
# Layer 1 -- local price loading
# ----------------------------------------------------------------------

def load_universe(universe_file: Path) -> List[str]:
    """Read the Russell 1000 instrument list (tab-separated: SYMBOL start end)."""
    symbols: List[str] = []
    with open(universe_file, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            symbols.append(line.split("\t")[0].split()[0].strip().upper())
    # Preserve order, drop duplicates
    seen = set()
    out = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def load_local_ohlcv(symbol: str, market_data_root: Path) -> pd.DataFrame:
    """
    Load one ticker's OHLCV from the local source CSV and apply this repo's own
    adjustment convention (`scripts/download_us_selected_data.py::normalize_symbol_data`):

        factor = adjclose / close
        open/high/low *= factor
        close        = adjclose
        volume      /= factor

    yfinance's `adjclose` is back-adjusted, so the resulting series is anchored
    at the right edge: the final bar equals the true traded dollar close. That
    is what makes "Estimated Best Price" a real, quotable dollar figure.
    """
    csv_path = market_data_root / "source" / f"{symbol}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No local price CSV for {symbol} at {csv_path}")

    raw = pd.read_csv(csv_path)
    required = {"date", "open", "high", "low", "close", "adjclose", "volume"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{symbol}: source CSV missing columns {sorted(missing)}")

    raw = raw.dropna(subset=["date", "close", "adjclose"]).sort_values("date").reset_index(drop=True)
    if raw.empty:
        raise ValueError(f"{symbol}: source CSV has no usable rows")

    close_safe = raw["close"].replace(0.0, np.nan)
    factor = (raw["adjclose"] / close_safe).replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(1.0)

    df = pd.DataFrame({
        "date": pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d"),
        "open": raw["open"] * factor,
        "high": raw["high"] * factor,
        "low": raw["low"] * factor,
        "close": raw["adjclose"],
        "volume": raw["volume"] / factor.replace(0.0, np.nan).fillna(1.0),
    })
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df


# ----------------------------------------------------------------------
# Layer 2 -- Alpha158 cross-section
# ----------------------------------------------------------------------

def load_latest_alpha_scores(scores_file: Path) -> Tuple[pd.DataFrame, str]:
    """
    Read the pre-computed Alpha158 scores and return the most recent date's
    cross-section. Scores are consumed AS STORED -- rank and percentile are not
    recomputed here, because the stored artifact was corrected on 2026-09-05 to
    use competition ranking (`method="min"`); recomputing with a different tie
    method is precisely how the rank/percentile mismatch (Issue 4) arose.
    """
    scores = pd.read_parquet(scores_file)
    for col in ("date", "symbol", "score", "rank", "percentile"):
        if col not in scores.columns:
            raise ValueError(f"Alpha158 score file missing required column '{col}'")

    latest_date = scores["date"].max()
    cs = scores[scores["date"] == latest_date].copy()
    cs["symbol"] = cs["symbol"].astype(str).str.upper()
    return cs, pd.Timestamp(latest_date).strftime("%Y-%m-%d")


# ----------------------------------------------------------------------
# Layer 3 -- per-ticker analysis
# ----------------------------------------------------------------------

def analyse_symbol(
    symbol: str,
    market_data_root: Path,
    alpha_row: pd.Series,
    as_of_date: str,
) -> ScreenRow:
    """
    Run the repo's real regime + predictive pipeline for one ticker off local
    price data, then classify with the canonical verdict taxonomy.
    """
    df = load_local_ohlcv(symbol, market_data_root)
    if len(df) < MIN_BARS_REQUIRED:
        raise ValueError(f"only {len(df)} bars (< {MIN_BARS_REQUIRED} required)")

    regime, _ = detect_market_regime(df, data_dir=market_data_root, symbol=symbol)

    pred = predict_future_buy_timing(
        df,
        forecast_days=FORECAST_DAYS,
        simulations=SIMULATIONS,
        regime=regime,
        microstructure=None,   # disclosed in SCREEN_LIMITATIONS
        derivatives=None,      # disclosed in SCREEN_LIMITATIONS
        events=None,           # disclosed in SCREEN_LIMITATIONS
    )

    verdict = classify_executive_verdict(pred, gamma_squeeze=None)

    entry_range = pred["optimal_entry_range"]
    entry_low = float(entry_range[0])
    entry_high = float(entry_range[1])
    window = pred["optimal_buy_window"]
    window_start = window["start_date"]
    window_end = window["end_date"]

    # --- Invariant I1: a suppressed state must suppress EVERY field describing it.
    # A live price/date beside a red DO-NOT-BUY badge is the exact
    # contradiction remediated on 2026-09-05 (audit Issue 3).
    if verdict.is_capital_preservation:
        best_price_display = INHIBITED
        best_buy_date_display = INHIBITED
        exposed_low: Optional[float] = None
        exposed_high: Optional[float] = None
        exposed_date: Optional[str] = None
    else:
        best_price_display = f"${entry_low:,.2f} &ndash; ${entry_high:,.2f}"
        best_buy_date_display = window_start
        exposed_low, exposed_high = entry_low, entry_high
        exposed_date = window_start

    flags: List[str] = []

    # --- Invariant I1b: a rose NO_BUY tier and the suppressed display must be
    # the same condition. If a verdict badge says stand-aside while a live
    # corridor/date is printed beside it, that is the audited contradiction.
    if (verdict.tier == TIER_NO_BUY) != verdict.is_capital_preservation:
        flags.append(
            f"VERDICT_TIER_VS_POSTURE_MISMATCH(tier={verdict.tier}, "
            f"capital_preservation={verdict.is_capital_preservation}, rec='{pred['recommendation']}')"
        )

    # --- Invariant I2: an actionable buy date is never in the past.
    if exposed_date is not None and exposed_date < as_of_date:
        flags.append(f"BUY_DATE_IN_PAST({exposed_date} < {as_of_date})")

    # --- Invariant I3: corridor is ordered.
    if exposed_low is not None and exposed_high is not None and exposed_low > exposed_high:
        flags.append(f"ENTRY_CORRIDOR_INVERTED({exposed_low} > {exposed_high})")

    # --- Invariant I4: an actionable BUY/PULLBACK verdict must not sit beside a
    # 3-month median target below spot.
    if verdict.tier in (TIER_BUY, TIER_PULLBACK) and pred["target_price_3m"] < pred["current_price"]:
        flags.append(
            f"BUY_VERDICT_WITH_NEGATIVE_TARGET(target {pred['target_price_3m']:.2f} < spot {pred['current_price']:.2f})"
        )

    return ScreenRow(
        symbol=symbol,
        alpha_score=float(alpha_row["score"]),
        alpha_rank=int(alpha_row["rank"]),
        alpha_percentile=float(alpha_row["percentile"]),
        verdict_badge=verdict.badge,
        verdict_short=verdict.short_label,
        verdict_tier=verdict.tier,
        verdict_branch=verdict.tier_key,
        recommendation=str(pred["recommendation"]),
        is_capital_preservation=bool(verdict.is_capital_preservation),
        current_price=round(float(pred["current_price"]), 2),
        entry_low=exposed_low,
        entry_high=exposed_high,
        best_price_display=best_price_display,
        best_buy_date=exposed_date,
        best_buy_date_display=best_buy_date_display,
        buy_window_end=window_end if not verdict.is_capital_preservation else None,
        buy_window_status=str(window["status"]),
        target_price_3m=float(pred["target_price_3m"]),
        expected_return_pct=float(pred["expected_return_pct"]),
        stop_loss=float(pred["stop_loss"]),
        risk_reward_ratio=float(pred["risk_reward_ratio"]),
        key_support=float(pred["key_support"]),
        rsi=float(pred["current_rsi"]),
        bocd_state=pred["bocd_regime_state"],
        bocd_regime_name=pred["bocd_regime_name"],
        changepoint_hazard_pct=float(pred["bocd_changepoint_hazard_pct"] or 0.0),
        integrity_flags=flags,
    )


def run_screen(
    market_data_root: Path = DEFAULT_MARKET_DATA_ROOT,
    universe_file: Path = DEFAULT_UNIVERSE_FILE,
    scores_file: Path = DEFAULT_SCORES_FILE,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the full cross-sectional screen. Returns a render-ready payload."""
    t0 = time.time()

    symbols = load_universe(universe_file)
    alpha_cs, alpha_as_of = load_latest_alpha_scores(scores_file)
    alpha_by_symbol = {r["symbol"]: r for _, r in alpha_cs.iterrows()}

    if limit is not None:
        symbols = symbols[:limit]

    rows: List[ScreenRow] = []
    skipped: List[Dict[str, str]] = []

    for i, sym in enumerate(symbols, 1):
        if sym not in alpha_by_symbol:
            skipped.append({"symbol": sym, "reason": "no Alpha158 score on the latest scoring date"})
            continue
        try:
            rows.append(analyse_symbol(sym, market_data_root, alpha_by_symbol[sym], alpha_as_of))
        except Exception as exc:  # noqa: BLE001 - every failure is recorded, never hidden
            skipped.append({"symbol": sym, "reason": f"{type(exc).__name__}: {exc}"})
        if i % 100 == 0:
            logger.info("screened %d/%d (%.1fs elapsed)", i, len(symbols), time.time() - t0)

    # --- Invariant I5 (cross-sectional): higher score => better (lower) rank.
    cross_flags: List[str] = []
    if rows:
        chk = sorted(rows, key=lambda r: -r.alpha_score)
        prev_rank = -1
        for r in chk:
            if r.alpha_rank < prev_rank:
                cross_flags.append(
                    f"RANK_NOT_MONOTONIC_IN_SCORE at {r.symbol} (rank {r.alpha_rank} after {prev_rank})"
                )
                break
            prev_rank = r.alpha_rank

    # Price as-of date, read back from the data actually used.
    price_as_of = None
    if rows:
        probe = load_local_ohlcv(rows[0].symbol, market_data_root)
        price_as_of = str(probe["date"].iloc[-1])

    elapsed = time.time() - t0

    tier_counts: Dict[str, int] = {}
    for r in rows:
        tier_counts[r.verdict_short] = tier_counts.get(r.verdict_short, 0) + 1

    return {
        "rows": rows,
        "skipped": skipped,
        "cross_flags": cross_flags,
        "alpha_as_of": alpha_as_of,
        "price_as_of": price_as_of,
        "universe_size": len(symbols),
        "succeeded": len(rows),
        "elapsed_sec": round(elapsed, 1),
        "tier_counts": tier_counts,
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "forecast_days": FORECAST_DAYS,
        "simulations": SIMULATIONS,
    }


# ----------------------------------------------------------------------
# Layer 4 -- render
# ----------------------------------------------------------------------

_TIER_PILL = {
    TIER_BUY: "bg-emerald-500/15 text-emerald-300 border-emerald-500/40",
    TIER_PULLBACK: "bg-blue-500/15 text-blue-300 border-blue-500/40",
    TIER_CAUTION: "bg-amber-500/15 text-amber-300 border-amber-500/40",
    TIER_NO_BUY: "bg-rose-500/15 text-rose-300 border-rose-500/40",
}


def _esc(v: Any) -> str:
    return html.escape(str(v), quote=True)


def build_screen_html(payload: Dict[str, Any]) -> str:
    """Render the consolidated dark institutional dashboard."""
    rows: List[ScreenRow] = payload["rows"]
    skipped: List[Dict[str, str]] = payload["skipped"]

    # Default sort: Alpha158 percentile descending (best names first).
    rows_sorted = sorted(rows, key=lambda r: (-r.alpha_percentile, r.alpha_rank, r.symbol))

    body_rows = []
    for r in rows_sorted:
        pill = _TIER_PILL[r.verdict_tier]
        inhibited = r.is_capital_preservation

        price_cell = (
            f'<span class="text-rose-400 font-bold text-[11px] tracking-wide">{INHIBITED}</span>'
            if inhibited else
            f'<span class="font-mono text-gray-100">{r.best_price_display}</span>'
        )
        date_cell = (
            f'<span class="text-rose-400 font-bold text-[11px] tracking-wide">{INHIBITED}</span>'
            if inhibited else
            f'<span class="font-mono text-gray-100">{_esc(r.best_buy_date_display)}</span>'
            f'<span class="block text-[10px] text-gray-500 font-mono">through {_esc(r.buy_window_end)}</span>'
        )

        flag_badge = ""
        if r.integrity_flags:
            flag_badge = (
                '<span class="ml-1 text-[10px] px-1.5 py-0.5 rounded bg-rose-900/50 '
                f'text-rose-300 border border-rose-700/60" title="{_esc("; ".join(r.integrity_flags))}">!</span>'
            )

        # A verdict whose own Monte Carlo median points the other way is shown
        # as a visible second badge, never as a tooltip a reader could miss.
        conflict_badge = ""
        if any(f.startswith("BUY_VERDICT_WITH_NEGATIVE_TARGET") for f in r.integrity_flags):
            conflict_badge = (
                '<span class="block mt-1 text-[9px] font-bold px-1.5 py-0.5 rounded border '
                'bg-rose-950/60 text-rose-300 border-rose-700/60" '
                'title="The verdict rules in RecommendationEngine.evaluate never consult the '
                'Monte Carlo median; here they disagree.">'
                '&#9888; MEDIAN PATH CONTRADICTS VERDICT</span>'
            )

        pct_bar = min(100.0, max(0.0, r.alpha_percentile))
        ret_col = "text-emerald-400" if r.expected_return_pct >= 0 else "text-rose-400"

        body_rows.append(f"""
      <tr class="border-b border-gray-800/60 hover:bg-gray-800/40 transition-colors"
          data-symbol="{_esc(r.symbol)}" data-tier="{_esc(r.verdict_tier)}"
          data-pct="{r.alpha_percentile}" data-rank="{r.alpha_rank}"
          data-score="{r.alpha_score}" data-price="{r.current_price}"
          data-entry="{r.entry_low if r.entry_low is not None else ''}"
          data-date="{_esc(r.best_buy_date or '')}">
        <td class="px-3 py-2 font-bold text-white whitespace-nowrap">{_esc(r.symbol)}{flag_badge}</td>
        <td class="px-3 py-2 text-right font-mono text-gray-300">{r.alpha_score:+.6f}</td>
        <td class="px-3 py-2 text-right font-mono text-gray-400">{r.alpha_rank}</td>
        <td class="px-3 py-2">
          <div class="flex items-center gap-2">
            <div class="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden min-w-[48px]">
              <div class="h-full bg-gradient-to-r from-cyan-500 to-emerald-400" style="width:{pct_bar:.1f}%"></div>
            </div>
            <span class="font-mono text-xs text-cyan-300 w-12 text-right">{r.alpha_percentile:.2f}</span>
          </div>
        </td>
        <td class="px-3 py-2 whitespace-nowrap">
          <span class="text-[11px] font-bold px-2 py-1 rounded-full border {pill}">{_esc(r.verdict_short)}</span>
          {conflict_badge}
        </td>
        <td class="px-3 py-2 text-right font-mono text-gray-400">${r.current_price:,.2f}</td>
        <td class="px-3 py-2 text-right whitespace-nowrap">{price_cell}</td>
        <td class="px-3 py-2 text-right whitespace-nowrap">{date_cell}</td>
        <td class="px-3 py-2 text-right font-mono {ret_col}">{r.expected_return_pct:+.2f}%</td>
        <td class="px-3 py-2 text-right font-mono text-gray-400">{r.rsi:.1f}</td>
        <td class="px-3 py-2 text-[10px] text-gray-500 whitespace-nowrap">{_esc(r.bocd_regime_name or 'n/a')}</td>
      </tr>""")

    regime_counts: Dict[str, int] = {}
    for r in rows:
        key = f"State {r.bocd_state} &mdash; {_esc(r.bocd_regime_name or 'unnamed')}"
        regime_counts[key] = regime_counts.get(key, 0) + 1
    regime_dist_html = "<br/>".join(
        f'<span class="text-cyan-300">{v:>4}</span> &nbsp;{k}'
        for k, v in sorted(regime_counts.items(), key=lambda kv: -kv[1])
    )

    limitations_html = "".join(
        f"""<div class="flex gap-3 py-2 border-b border-gray-800/50 last:border-0">
              <span class="text-rose-400 mt-0.5">&#10007;</span>
              <div><div class="text-xs font-bold text-gray-200">{_esc(t)}</div>
                   <div class="text-[11px] text-gray-500 leading-relaxed">{_esc(d)}</div></div>
            </div>"""
        for t, d in SCREEN_LIMITATIONS
    )

    unreachable_html = "".join(
        f"""<li class="text-[11px] text-gray-400 leading-relaxed py-1">
              <span class="font-mono text-amber-300">{_esc(v)}</span>
              <span class="text-gray-500"> &mdash; {_esc(why)}</span></li>"""
        for v, why in UNREACHABLE_VERDICTS
    )

    skipped_html = (
        "".join(
            f'<tr class="border-b border-gray-800/50"><td class="px-3 py-1.5 font-mono text-gray-300">{_esc(s["symbol"])}</td>'
            f'<td class="px-3 py-1.5 text-gray-500 text-[11px]">{_esc(s["reason"])}</td></tr>'
            for s in skipped
        )
        or '<tr><td colspan="2" class="px-3 py-3 text-emerald-400 text-xs">No tickers were skipped &mdash; full universe coverage.</td></tr>'
    )

    flagged_rows = [r for r in rows if r.integrity_flags]
    integrity_items = payload["cross_flags"] + [
        f"{r.symbol}: {f}" for r in flagged_rows for f in r.integrity_flags
    ]
    if integrity_items:
        integrity_html = (
            '<div class="text-xs text-rose-300 font-bold mb-2">'
            f'{len(integrity_items)} invariant violation(s) detected &mdash; listed, not suppressed:</div>'
            '<ul class="space-y-1 max-h-64 overflow-y-auto">'
            + "".join(f'<li class="text-[11px] font-mono text-rose-200">{_esc(i)}</li>' for i in integrity_items)
            + "</ul>"
        )
        integrity_border = "border-rose-700/60"
    else:
        integrity_html = (
            '<div class="text-xs text-emerald-300">All five consistency invariants hold across every rendered row:</div>'
            '<ul class="mt-2 space-y-1 text-[11px] text-gray-400">'
            '<li>I1 &mdash; capital-preservation rows suppress <em>both</em> best price and best buy date</li>'
            '<li>I2 &mdash; no actionable buy date precedes the report as-of date</li>'
            '<li>I3 &mdash; every entry corridor is correctly ordered (low &le; high)</li>'
            '<li>I4 &mdash; no BUY/PULLBACK verdict sits beside a 3-month target below spot</li>'
            '<li>I5 &mdash; Alpha158 rank is monotonic in score across the cross-section</li>'
            "</ul>"
        )
        integrity_border = "border-emerald-800/60"

    n_actionable = sum(1 for r in rows if r.verdict_tier in (TIER_BUY, TIER_PULLBACK))
    n_conflict = sum(
        1 for r in rows if any(f.startswith("BUY_VERDICT_WITH_NEGATIVE_TARGET") for f in r.integrity_flags)
    )
    conflict_pct = (100.0 * n_conflict / n_actionable) if n_actionable else 0.0

    tc = payload["tier_counts"]
    tile_defs = [
        ("Universe Screened", f'{payload["succeeded"]}', f'of {payload["universe_size"]} names', "text-white"),
        ("Strong Buy", str(tc.get("STRONG BUY", 0)), "emerald tier", "text-emerald-400"),
        ("Buy On Pullback", str(tc.get("BUY ON PULLBACK", 0)), "blue tier", "text-blue-400"),
        ("Hold / Cautious", str(tc.get("HOLD / CAUTIOUS BUY", 0)), "amber tier", "text-amber-400"),
        ("Do Not Buy", str(tc.get("DO NOT BUY", 0)), "capital preservation", "text-rose-400"),
        ("Verdict/Median Conflicts", f"{n_conflict}", f"{conflict_pct:.0f}% of {n_actionable} actionable", "text-rose-400"),
        ("Runtime", f'{payload["elapsed_sec"]}s', f'{payload["simulations"]} MC paths x {payload["forecast_days"]}d', "text-cyan-300"),
    ]
    tiles_html = "".join(
        f"""<div class="bg-gray-900/70 border border-gray-800 rounded-xl p-4">
              <div class="text-[10px] font-bold text-gray-500 uppercase tracking-wider">{_esc(lbl)}</div>
              <div class="text-2xl font-black {col} mt-1 font-mono">{_esc(val)}</div>
              <div class="text-[10px] text-gray-500 mt-0.5">{_esc(sub)}</div>
            </div>"""
        for lbl, val, sub, col in tile_defs
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Russell 1000 Alpha158 Factor &amp; Verdict Screen &mdash; {_esc(payload["alpha_as_of"])}</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  body {{ background:#030712; font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }}
  thead th {{ position: sticky; top: 0; z-index: 10; background:#0b1220; cursor: pointer; user-select:none; }}
  thead th:hover {{ background:#111c2e; }}
  th .arrow {{ opacity:.35; font-size:9px; margin-left:3px; }}
  ::-webkit-scrollbar {{ width:10px; height:10px; }}
  ::-webkit-scrollbar-track {{ background:#0b1220; }}
  ::-webkit-scrollbar-thumb {{ background:#1f2937; border-radius:5px; }}
</style>
</head>
<body class="text-gray-200">
<div class="max-w-[1600px] mx-auto p-6 space-y-6">

  <!-- HEADER -->
  <div class="bg-gradient-to-r from-gray-950 via-gray-900 to-gray-950 border-2 border-cyan-900/40 rounded-2xl p-6">
    <div class="flex flex-wrap items-start justify-between gap-4">
      <div>
        <div class="flex items-center gap-3 flex-wrap">
          <h1 class="text-2xl font-black text-white tracking-tight">Russell 1000 &mdash; Alpha158 Factor &amp; Executive Verdict Screen</h1>
          <span class="text-[10px] font-bold px-2.5 py-1 rounded-full border bg-amber-500/15 text-amber-300 border-amber-500/40">
            CROSS-SECTIONAL SCREEN &mdash; NOT A SINGLE-TICKER DEEP DIVE
          </span>
        </div>
        <p class="text-xs text-gray-400 mt-2 max-w-4xl leading-relaxed">
          LightGBM Alpha158 cross-sectional factor scores joined to the repo-canonical Executive Investment
          Verdict taxonomy and Monte&nbsp;Carlo buy-timing forecast. Verdicts are produced by the same
          <span class="font-mono text-gray-300">classify_executive_verdict</span> ladder that renders the
          single-ticker report's verdict banner, so the two can never disagree.
        </p>
      </div>
      <div class="text-right text-[11px] text-gray-500 font-mono leading-relaxed">
        <div>Alpha158 as-of: <span class="text-cyan-300">{_esc(payload["alpha_as_of"])}</span></div>
        <div>Price as-of: <span class="text-cyan-300">{_esc(payload["price_as_of"])}</span></div>
        <div>Generated: <span class="text-gray-400">{_esc(payload["generated_at"])}</span></div>
      </div>
    </div>
  </div>

  <!-- STAT TILES -->
  <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">{tiles_html}</div>

  <!-- VERDICT vs MEDIAN PATH CONFLICT ADVISORY -->
  <div class="bg-amber-950/25 border-2 border-amber-800/50 rounded-2xl p-5">
    <div class="flex items-start gap-3">
      <span class="text-amber-400 text-lg leading-none mt-0.5">&#9888;</span>
      <div>
        <div class="text-sm font-bold text-amber-300">
          {n_conflict} of {n_actionable} actionable names ({conflict_pct:.0f}%) carry a buy verdict whose own Monte Carlo median points down
        </div>
        <p class="text-[11px] text-gray-300 mt-1.5 leading-relaxed max-w-5xl">
          <span class="font-mono text-gray-400">RecommendationEngine.evaluate</span> decides the verdict from
          regime state, RSI and %B only &mdash; <strong class="text-amber-300">it never reads the Monte Carlo
          median path</strong> that produces the 3-month target in the very next column. Where the two disagree,
          the row carries a <span class="text-rose-300 font-bold">MEDIAN PATH CONTRADICTS VERDICT</span> badge and
          the 3M Expected Return cell is red. The driver is the simulator's mean-reversion term
          (<span class="font-mono">DRIFT_MEAN_REVERSION_COEFF</span> pulling toward SMA50): names extended well
          above their 50-day average get a bullish technical verdict and a mean-reverting median simultaneously.
          <strong class="text-gray-200">Neither number is suppressed and neither is adjusted</strong> &mdash; both are
          shown so the disagreement is visible rather than resolved by an undisclosed rule. Treat a flagged row as
          unresolved, not as a buy.
        </p>
      </div>
    </div>
  </div>

  <!-- PRICE SOURCE ADVISORY -->
  <div class="bg-rose-950/30 border-2 border-rose-800/50 rounded-2xl p-5">
    <div class="flex items-start gap-3">
      <span class="text-rose-400 text-lg leading-none mt-0.5">&#9888;</span>
      <div>
        <div class="text-sm font-bold text-rose-300">Price source advisory &mdash; the qlib binary store was not used</div>
        <p class="text-[11px] text-gray-300 mt-1.5 leading-relaxed max-w-5xl">
          The binary feature store at <span class="font-mono text-gray-400">D:/trading/qlib/qlib_data</span> is
          <strong class="text-rose-300">misaligned</strong>: each feature <span class="font-mono">.bin</span> holds
          1,930 values while <span class="font-mono">calendars/day.txt</span> lists only 1,500 trading days
          (<strong class="text-rose-300">857 of 909</strong> tickers affected). Because qlib maps bin index
          <em>i</em> to <span class="font-mono">calendar[start_index + i]</span>, every series is shifted by
          <strong class="text-rose-300">430 trading days</strong> and the most recent ~1.7 years is unreachable.
          Verified concretely: <span class="font-mono">D.features(AAPL, end_time='2026-09-04')</span> returns the
          normalized close <span class="font-mono">6.6574</span>, which is AAPL's <strong>2024-12-16</strong> bar &mdash;
          the true 2026-09-04 close is <span class="font-mono">$319.97</span>.
          This report therefore reads prices from the upstream local
          <span class="font-mono text-gray-400">source/*.csv</span> files (offline, 908/908 coverage, real dollar terms).
          <strong class="text-rose-300">Re-dumping the binary store is a separate outstanding fix.</strong>
        </p>
      </div>
    </div>
  </div>

  <!-- METHODOLOGY / LIMITATIONS -->
  <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
    <div class="bg-gray-900/60 border border-gray-800 rounded-2xl p-5">
      <div class="text-xs font-bold text-gray-200 uppercase tracking-wider mb-2">Signals excluded vs. the single-ticker report</div>
      <p class="text-[11px] text-gray-500 mb-2 leading-relaxed">
        This screen runs the real <span class="font-mono">detect_market_regime</span> (BOCD) and
        <span class="font-mono">predict_future_buy_timing</span> (GBM Monte&nbsp;Carlo) pipeline, but with the
        options / events / microstructure inputs set to <span class="font-mono">None</span>. Those extractors
        degrade to their dataclass defaults. What that removes:
      </p>
      <div class="divide-y divide-gray-800/50">{limitations_html}</div>
      <div class="mt-3 pt-3 border-t border-gray-800">
        <div class="text-[10px] font-bold text-gray-500 uppercase tracking-wider mb-1.5">BOCD regime distribution driving these verdicts</div>
        <div class="text-[11px] text-gray-400 font-mono leading-relaxed">{regime_dist_html}</div>
        <p class="text-[10px] text-gray-500 mt-2 leading-relaxed">
          The large DO-NOT-BUY count is driven by BOCD State&nbsp;2 (high-volatility liquidation), which forces
          capital preservation regardless of Alpha158 percentile. That override is intentional: a top-decile
          factor score does not license an entry into a liquidation regime.
        </p>
      </div>
    </div>

    <div class="space-y-4">
      <div class="bg-gray-900/60 border border-amber-800/40 rounded-2xl p-5">
        <div class="text-xs font-bold text-amber-300 uppercase tracking-wider mb-2">Verdicts structurally unreachable in this screen</div>
        <p class="text-[11px] text-gray-500 mb-2 leading-relaxed">
          Their absence below is <strong class="text-gray-300">not</strong> evidence they were evaluated and rejected.
        </p>
        <ul class="divide-y divide-gray-800/50">{unreachable_html}</ul>
      </div>

      <div class="bg-gray-900/60 border border-amber-800/40 rounded-2xl p-5">
        <div class="text-xs font-bold text-amber-300 uppercase tracking-wider mb-2">Alpha158 score degeneracy</div>
        <p class="text-[11px] text-gray-400 leading-relaxed">
          The stored model produces only <strong class="text-amber-300">232 distinct score values across 908 names</strong>
          on this date, with ties up to 120 wide. Percentile therefore discriminates far more coarsely than its two
          decimal places imply, and adjacent ranks within a tie block are arbitrary. Tracked separately in
          <span class="font-mono text-gray-500">.team-code/20260905-finance_team_review_alpha158_degenerate_score.md</span>.
          Rank/percentile are consumed exactly as stored (competition ranking, <span class="font-mono">method="min"</span>),
          never recomputed here.
        </p>
      </div>
    </div>
  </div>

  <!-- INTEGRITY -->
  <div class="bg-gray-900/60 border-2 {integrity_border} rounded-2xl p-5">
    <div class="text-xs font-bold text-gray-200 uppercase tracking-wider mb-2">Cross-field consistency invariants</div>
    {integrity_html}
  </div>

  <!-- CONTROLS -->
  <div class="bg-gray-900/60 border border-gray-800 rounded-2xl p-4 flex flex-wrap items-center gap-3">
    <input id="q" type="text" placeholder="Filter by ticker&hellip;"
           class="bg-gray-950 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 w-56 focus:outline-none focus:border-cyan-600"/>
    <select id="tier" class="bg-gray-950 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 focus:outline-none focus:border-cyan-600">
      <option value="">All verdicts</option>
      <option value="{TIER_BUY}">Strong Buy (emerald)</option>
      <option value="{TIER_PULLBACK}">Buy On Pullback (blue)</option>
      <option value="{TIER_CAUTION}">Hold / Cautious (amber)</option>
      <option value="{TIER_NO_BUY}">Do Not Buy (rose)</option>
    </select>
    <label class="flex items-center gap-2 text-xs text-gray-400">
      <input id="actionable" type="checkbox" class="accent-emerald-500"/> Actionable entries only
    </label>
    <span id="count" class="text-xs text-gray-500 font-mono ml-auto"></span>
  </div>

  <!-- TABLE -->
  <div class="bg-gray-900/40 border border-gray-800 rounded-2xl overflow-hidden">
    <div class="overflow-auto max-h-[75vh]">
      <table class="w-full text-xs" id="tbl">
        <thead class="text-[10px] uppercase tracking-wider text-gray-400 border-b-2 border-gray-800">
          <tr>
            <th class="px-3 py-3 text-left"   data-k="symbol" data-t="s">Symbol<span class="arrow">&#9650;&#9660;</span></th>
            <th class="px-3 py-3 text-right"  data-k="score"  data-t="n">Alpha158 Score<span class="arrow">&#9650;&#9660;</span></th>
            <th class="px-3 py-3 text-right"  data-k="rank"   data-t="n">Rank<span class="arrow">&#9650;&#9660;</span></th>
            <th class="px-3 py-3 text-left"   data-k="pct"    data-t="n">Percentile<span class="arrow">&#9650;&#9660;</span></th>
            <th class="px-3 py-3 text-left"   data-k="tier"   data-t="s">Executive Verdict<span class="arrow">&#9650;&#9660;</span></th>
            <th class="px-3 py-3 text-right"  data-k="price"  data-t="n">Last Close<span class="arrow">&#9650;&#9660;</span></th>
            <th class="px-3 py-3 text-right"  data-k="entry"  data-t="n">Est. Best Price<span class="arrow">&#9650;&#9660;</span></th>
            <th class="px-3 py-3 text-right"  data-k="date"   data-t="s">Est. Best Buy Date<span class="arrow">&#9650;&#9660;</span></th>
            <th class="px-3 py-3 text-right"  data-k="ret"    data-t="n">3M Exp. Return<span class="arrow">&#9650;&#9660;</span></th>
            <th class="px-3 py-3 text-right"  data-k="rsi"    data-t="n">RSI<span class="arrow">&#9650;&#9660;</span></th>
            <th class="px-3 py-3 text-left">BOCD Regime</th>
          </tr>
        </thead>
        <tbody id="tb">{"".join(body_rows)}</tbody>
      </table>
    </div>
  </div>

  <!-- SKIPPED -->
  <div class="bg-gray-900/60 border border-gray-800 rounded-2xl p-5">
    <div class="text-xs font-bold text-gray-200 uppercase tracking-wider mb-3">
      Skipped tickers ({len(skipped)})
    </div>
    <div class="overflow-auto max-h-56">
      <table class="w-full text-xs"><tbody>{skipped_html}</tbody></table>
    </div>
  </div>

  <div class="text-[10px] text-gray-600 text-center pb-6 leading-relaxed">
    Generated by <span class="font-mono">scripts/russell1000_factor_verdict_screen.py</span> &bull;
    Verdict taxonomy: <span class="font-mono">scripts/verdict_taxonomy.py</span> &bull;
    Forecast engine: <span class="font-mono">scripts/predictive_engine.py</span>
    ({payload["simulations"]} seeded Monte Carlo paths, {payload["forecast_days"]} trading-day horizon; deterministic, seed=42)
    <br/>Research screen for institutional use. Not investment advice. No live options, event, or order-flow data is reflected in these verdicts.
  </div>
</div>

<script>
(function() {{
  var tb = document.getElementById('tb');
  var all = Array.prototype.slice.call(tb.querySelectorAll('tr'));
  var q = document.getElementById('q'), tier = document.getElementById('tier'),
      actionable = document.getElementById('actionable'), count = document.getElementById('count');
  var sortKey = 'pct', sortDir = -1;

  function cellVal(tr, k, t) {{
    if (k === 'ret')  return parseFloat(tr.children[8].textContent) || 0;
    if (k === 'rsi')  return parseFloat(tr.children[9].textContent) || 0;
    var v = tr.getAttribute('data-' + k);
    if (t === 'n') {{ var f = parseFloat(v); return isNaN(f) ? -Infinity : f; }}
    return (v || '').toString();
  }}

  function render() {{
    var term = q.value.trim().toUpperCase(), tv = tier.value, act = actionable.checked;
    var vis = all.filter(function(tr) {{
      if (term && tr.getAttribute('data-symbol').indexOf(term) === -1) return false;
      if (tv && tr.getAttribute('data-tier') !== tv) return false;
      if (act && tr.getAttribute('data-tier') === '{TIER_NO_BUY}') return false;
      return true;
    }});
    var th = document.querySelector('th[data-k="' + sortKey + '"]');
    var t = th ? th.getAttribute('data-t') : 'n';
    vis.sort(function(a, b) {{
      var x = cellVal(a, sortKey, t), y = cellVal(b, sortKey, t);
      if (x < y) return -sortDir;
      if (x > y) return sortDir;
      return a.getAttribute('data-symbol') < b.getAttribute('data-symbol') ? -1 : 1;
    }});
    tb.innerHTML = '';
    vis.forEach(function(tr) {{ tb.appendChild(tr); }});
    count.textContent = vis.length + ' / ' + all.length + ' names shown';
  }}

  Array.prototype.forEach.call(document.querySelectorAll('th[data-k]'), function(th) {{
    th.addEventListener('click', function() {{
      var k = th.getAttribute('data-k');
      if (k === sortKey) {{ sortDir = -sortDir; }} else {{ sortKey = k; sortDir = (th.getAttribute('data-t') === 's') ? 1 : -1; }}
      render();
    }});
  }});
  q.addEventListener('input', render);
  tier.addEventListener('change', render);
  actionable.addEventListener('change', render);
  render();
}})();
</script>
</body>
</html>"""


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Russell 1000 Alpha158 + Executive Verdict screen")
    ap.add_argument("--market-data-root", default=str(DEFAULT_MARKET_DATA_ROOT))
    ap.add_argument("--universe", default=str(DEFAULT_UNIVERSE_FILE))
    ap.add_argument("--scores", default=str(DEFAULT_SCORES_FILE))
    ap.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    ap.add_argument("--limit", type=int, default=None, help="Screen only the first N tickers (smoke test)")
    args = ap.parse_args()

    payload = run_screen(
        market_data_root=Path(args.market_data_root),
        universe_file=Path(args.universe),
        scores_file=Path(args.scores),
        limit=args.limit,
    )

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = payload["alpha_as_of"]
    html_path = report_dir / f"russell1000_factor_verdict_screen_{stamp}.html"
    json_path = report_dir / f"russell1000_factor_verdict_screen_{stamp}.json"

    html_path.write_text(build_screen_html(payload), encoding="utf-8")

    json_payload = dict(payload)
    json_payload["rows"] = [asdict(r) for r in payload["rows"]]
    json_path.write_text(json.dumps(json_payload, indent=2, default=str), encoding="utf-8")

    logger.info("succeeded=%d skipped=%d elapsed=%ss", payload["succeeded"], len(payload["skipped"]), payload["elapsed_sec"])
    logger.info("report: %s", html_path)
    logger.info("json:   %s", json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
