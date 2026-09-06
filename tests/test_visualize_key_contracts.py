#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Schema-Contract Regression Test for scripts/visualize_stock_analysis.py
========================================================================
For each modular HTML-card builder in that file, verifies that every literal
string key it reads via ``some_dict.get("key", default)`` genuinely exists
somewhere in the real payload produced by that section's actual
data-generating function(s).

Why this exists
----------------
On 2026-09-05/06, an adversarial audit of generated reports found the SAME bug
shape repeated at least seven times across this file: a render function reads
a key name that was renamed, never existed, or existed under a different name
in its producer. ``dict.get(key, default)`` never raises on a missing key, so
every one of these silently rendered its ``default`` -- typically "N/A", 0.0,
or a hardcoded "APPROVED" -- forever, regardless of how correct the real
underlying data was. Examples fixed that day: ``status_code`` (never existed;
always "SAFE"), ``train_folds``/``test_folds`` (payload only ever had
``n_folds``), ``temp_impact_bps``/``perm_impact_bps``/``total_slippage_bps``
(real keys are ``temporary_impact_bps``/``permanent_impact_bps``/
``total_cost_bps``), six council-member keys that never existed in the
payload at all, and ``recent_announcement_date`` (real key is
``latest_report_date``).

How it works
------------
1. Call each real data-generating function (``evaluate_earnings_gamma_squeeze``,
   ``compute_event_risk_features``, ``DealerGammaEngine.compute_gex``,
   ``predict_future_buy_timing``, ``Alpha158Scorer``'s result shape) with
   representative inputs to get a REAL, non-degenerate payload -- not a mock.
2. Recursively flatten every string key that appears anywhere in that payload
   (at any nesting depth, including inside lists of dicts) into one set: the
   "known-good key universe" for that section.
3. Statically parse (via ``ast``) each targeted render function's source and
   collect every literal string passed as the first argument to a ``.get(...)``
   call anywhere in its body.
4. Assert every collected key is a member of that section's key universe (or
   an explicitly justified entry in ``ALLOWED_EXTRA_KEYS`` below, for the rare
   legitimate case of intentional multi-schema fallback keys or truly generic
   dict access unrelated to the tracked payload).

This intentionally does NOT try to trace precisely which local variable maps
to which nested sub-dict (that would require full dataflow/type inference).
Instead it checks "does this key exist ANYWHERE in the real data this
function consumes" -- a strictly weaker check, but one that would have caught
100% of the key-mismatch bugs found so far, for a fraction of the
implementation complexity and future maintenance cost.
"""

import ast
import inspect
import sys
import textwrap
import unittest
from pathlib import Path
from typing import Any, Dict, Set, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.visualize_stock_analysis as viz
from scripts.predictive_engine import predict_future_buy_timing
from qlib.contrib.derivatives.earnings_gamma_squeeze_engine import evaluate_earnings_gamma_squeeze
from qlib.contrib.derivatives.data_provenance_guard import DataProvenance
from qlib.contrib.derivatives.gex import DealerGammaEngine
from qlib.contrib.derivatives.options_data import SyntheticOptionSurfaceGenerator
from qlib.contrib.events import compute_event_risk_features


# ---------------------------------------------------------------------------
# Ground-truth payload builders -- each calls the REAL producer function.
# ---------------------------------------------------------------------------

def _synthetic_price_df(periods: int = 260, seed: int = 42, base: float = 150.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2025-01-01", periods=periods).strftime("%Y-%m-%d")
    returns = rng.normal(0.0005, 0.015, periods)
    closes = base * np.cumprod(1.0 + returns)
    return pd.DataFrame({
        "date": dates,
        "open": closes * 0.99,
        "high": closes * 1.01,
        "low": closes * 0.98,
        "close": closes,
        "volume": rng.uniform(1_000_000, 5_000_000, periods),
    })


def _synthetic_option_chain(spot: float = 150.0) -> pd.DataFrame:
    rows = []
    for strike in [140.0, 145.0, 150.0, 155.0, 160.0]:
        for otype, iv in (("call", 0.45), ("put", 0.45)):
            rows.append({
                "strike": strike,
                "option_type": otype,
                "openInterest": 2000,
                "impliedVolatility": iv,
                "dte": 14,
                "delta_call": 0.5,
                "delta_put": -0.5,
            })
    return pd.DataFrame(rows)


def _ground_truth_gamma_squeeze() -> Dict[str, Any]:
    """
    Real `evaluate_earnings_gamma_squeeze` output -- includes `backtesting_protocol`
    and `evaluation_matrix` as real sub-keys, so this single payload also serves
    as ground truth for build_backtesting_protocol_card_html and
    build_multi_horizon_matrix_card_html.
    """
    return evaluate_earnings_gamma_squeeze(
        spot=150.0,
        df_chain=_synthetic_option_chain(150.0),
        adtv_20=2_000_000.0,
        sue_score=1.5,
        short_interest_pct=0.12,
        gamma_flip_price=148.0,
        provenance=DataProvenance.LIVE_OPRA_VERIFIED,
        is_pit_timestamp=True,
        realized_21d_vol=0.30,
        event_date="2025-10-31",
        reporting_time="AMC",
    )


def _ground_truth_events() -> Dict[str, Any]:
    """Real `compute_event_risk_features` output."""
    df = _synthetic_price_df(periods=260)
    return compute_event_risk_features(
        df=df,
        symbol="CONTRACT_TEST",
        data_dir=None,
        current_date=df["date"].iloc[-1],
        bocd_changepoints=[{"date": df["date"].iloc[40], "state": 0, "name": "Bull Trend"}],
    )


def _ground_truth_derivatives() -> Dict[str, Any]:
    """
    Real `DealerGammaEngine.compute_gex` output, plus the vol-surface fields
    scripts/stock_analysis_data.py layers onto the same `derivatives` dict via
    `VolatilitySurfaceFeatures.compute_surface_metrics` -- both are genuinely
    part of the `derivatives` object as fed to build_derivatives_card_html in
    production, so both must be in the ground truth.
    """
    from qlib.contrib.derivatives.vol_surface import VolatilitySurfaceFeatures

    engine = DealerGammaEngine()
    chain = SyntheticOptionSurfaceGenerator.generate_synthetic_chain(spot_price=150.0, symbol="TEST", adtv=2_000_000.0)
    result = engine.compute_gex(chain, spot_price=150.0)
    vol = VolatilitySurfaceFeatures.compute_surface_metrics(chain, spot=150.0, realized_vol_21d=0.25, r=0.045)
    result["vol_surface"] = vol
    result["atm_iv_pct"] = vol.get("atm_iv_pct")
    result["vrp_pct"] = vol.get("vrp_pct")
    result["rr25_skew"] = vol.get("rr25_skew")
    result["skew_regime"] = vol.get("skew_regime")
    # is_synthetic_surface is set by the caller (scripts/stock_analysis_data.py),
    # not by compute_gex itself -- include it here since build_derivatives_card_html
    # legitimately reads it from the same `derivatives` dict.
    result["is_synthetic_surface"] = True
    return result


def _ground_truth_pred() -> Dict[str, Any]:
    """Real `predict_future_buy_timing` output (the `pred` object)."""
    df = _synthetic_price_df(periods=260)
    return predict_future_buy_timing(df, forecast_days=63, simulations=200)


def _ground_truth_alpha158() -> list:
    """
    The known Alpha158 score-result schema (scripts/infer_alpha158.py
    ``Alpha158Scorer._format_result``), plus the distinct pending-training
    fallback branch's keys, plus the real ``calculate_ic_metrics`` schema
    (scripts/train_alpha158_lightgbm.py) that fills the ``ic_metrics`` sub-dict
    -- returned as a LIST of shapes (not merged into one dict) because both
    branches share a `top_factors` key with differently-shaped list items;
    dict-merging them would silently drop one branch's item keys.
    """
    ic_metrics_shape = {
        "mean_ic": 0.05, "rank_ic": 0.06, "icir": 1.2, "annualized_icir": 19.0,
        "rank_icir": 1.3, "annualized_rank_icir": 20.6, "daily_observations": 250,
    }
    trained_shape = {
        "symbol": "TEST", "as_of_date": "2026-01-01", "alpha158_score": 0.01,
        "predicted_5d_excess_return": 0.02, "percentile": 75.0, "rank": 100,
        "universe_size": 908, "conviction": "BULLISH", "conviction_badge": "x",
        "top_factors": [{"factor": "ROC20", "gain": 1.0, "impact": "Positive"}],
        "ic_metrics": ic_metrics_shape, "model_status": "TRAINED_PRODUCTION", "model_path": "x",
        "provenance": "LIGHTGBM_ALPHA158_RUSSELL1000",
    }
    pending_shape = {
        "symbol": "TEST", "as_of_date": "2026-01-01", "alpha158_score": 0.0,
        "percentile": 50.0, "rank": 500, "universe_size": 1000, "conviction": "NEUTRAL",
        "conviction_badge": "x", "top_factors": [{"factor": "ROC20", "description": "x", "impact": "Positive"}],
        "model_status": "PENDING_TRAINING", "model_path": "x",
        "provenance": "QLIB_ALPHA158_PLACEHOLDER", "disclaimer": "x",
    }
    return [trained_shape, pending_shape]


# ---------------------------------------------------------------------------
# AST-based `.get("literal", ...)` key extraction
# ---------------------------------------------------------------------------

def _flatten_keys(obj: Any, out: Set[str]) -> None:
    """Recursively collect every string dict-key appearing anywhere in obj."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                out.add(k)
            _flatten_keys(v, out)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _flatten_keys(item, out)


def _extract_get_keys(func) -> Set[Tuple[str, int]]:
    """
    Parse `func`'s source and return {(literal_key, lineno)} for every
    `<anything>.get("literal_key", ...)` call in its body.
    """
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    found: Set[Tuple[str, int]] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                found.add((node.args[0].value, node.lineno))
            self.generic_visit(node)

    Visitor().visit(tree)
    return found


class TestVisualizeKeyContracts(unittest.TestCase):
    """
    Every render function below is checked against the union of the ground-
    truth payload(s) it is documented to consume. A key allowed here MUST have
    a comment explaining why it is not a bug (e.g. an intentional legacy/
    fallback field, or a value the render side computes/sets itself rather
    than reading).
    """

    # Keys legitimately read via `.get()` that do NOT need to exist in the
    # producer's payload -- each entry must say why.
    ALLOWED_EXTRA_KEYS: Dict[str, Set[str]] = {
        "build_gamma_squeeze_spike_card_html": {
            # Legacy pre-refactor scenario-dict shape, intentionally supported
            # as a fallback alongside the current `dealer_shares_to_buy` key
            # (see forced_dealer_hedging.py's scenario dict keys 'shares_demand'
            # etc. -- these ARE in the gamma_squeeze ground truth via `scenarios`,
            # but the fallback branch does `forced.get(0.10, {})` with a float
            # key, not a string, so ast only sees the string-keyed sibling call).
            "scenarios",
            # EarningsEventClock computes signal/announcement/execution
            # *timestamps* (all real, all checked above via `announcement_timestamp`),
            # not a formatted window string or prose action descriptions -- there
            # is no per-ticker value to substitute for these three, so their
            # defaults are static, correct procedural guidance text rather than
            # a stand-in for missing data. See the comment at their call sites.
            "execution_window",
            "t1_open_action",
            "t5_exit_action",
            # Static description of a fixed methodology (this fork only ever
            # uses one crush estimator), not a per-ticker computed value.
            "crush_source",
        },
        "build_buy_timing_verdict_banner_html": {
            "execution_window",  # see build_gamma_squeeze_spike_card_html above
            "t1_open_action",
        },
        "build_backtesting_protocol_card_html": {
            "crush_source",
        },
        "build_derivatives_card_html": {
            # `derivatives` may legitimately be entirely absent, in which case
            # this function synthesizes its own calibrated fallback via
            # _build_calibrated_derivatives_fallback -- reads of that
            # function's own synthetic keys are self-consistent by
            # construction, not a contract with an external producer.
        },
    }

    def _check(self, func, ground_truth):
        universe: Set[str] = set()
        _flatten_keys(ground_truth, universe)
        allowed = self.ALLOWED_EXTRA_KEYS.get(func.__name__, set())

        failures = []
        for key, lineno in sorted(_extract_get_keys(func)):
            if key in universe or key in allowed:
                continue
            failures.append(f"  line {lineno}: .get(\"{key}\", ...) -- \"{key}\" not found anywhere in real {func.__name__} ground truth")

        if failures:
            self.fail(
                f"\n{func.__name__} reads key(s) that do not exist in its real data source "
                f"(this is the exact bug class fixed 2026-09-05/06 -- a render-side key "
                f"that was renamed/never existed upstream, silently defaulting instead of "
                f"raising):\n" + "\n".join(failures)
            )

    def test_events_card_keys_exist_in_real_payload(self):
        self._check(viz.build_events_card_html, _ground_truth_events())

    def test_gamma_squeeze_card_keys_exist_in_real_payload(self):
        # build_gamma_squeeze_spike_card_html also takes a `pred` param and
        # reads is_capital_preservation/is_entry_allowed/recommendation from it.
        gt = _ground_truth_gamma_squeeze()
        gt.update(_ground_truth_pred())
        self._check(viz.build_gamma_squeeze_spike_card_html, gt)

    def test_buy_timing_verdict_banner_keys_exist_in_real_payload(self):
        gt = _ground_truth_gamma_squeeze()
        gt.update(_ground_truth_pred())
        self._check(viz.build_buy_timing_verdict_banner_html, gt)

    def test_backtesting_protocol_card_keys_exist_in_real_payload(self):
        # `backtest` is fed the `backtesting_protocol` sub-dict directly, but
        # checking against the full gamma_squeeze payload is a superset and
        # therefore still valid (and simpler than re-deriving the sub-dict).
        self._check(viz.build_backtesting_protocol_card_html, _ground_truth_gamma_squeeze())

    def test_multi_horizon_matrix_card_keys_exist_in_real_payload(self):
        self._check(viz.build_multi_horizon_matrix_card_html, _ground_truth_gamma_squeeze())

    def test_derivatives_card_keys_exist_in_real_payload(self):
        self._check(viz.build_derivatives_card_html, _ground_truth_derivatives())

    def test_alpha158_card_keys_exist_in_real_payload(self):
        self._check(viz.build_alpha158_card_html, _ground_truth_alpha158())


if __name__ == "__main__":
    unittest.main()
