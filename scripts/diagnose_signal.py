#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pre-training diagnostic gate: signal diagnostics library
==========================================================
Implements gate items #2-#7 (item #1's guard lives in ``ensemble_lib.py``; item #10 is a
CLI/``LABEL_VARIANTS`` concern in ``train_ensemble.py`` that reuses ``run_diagnostics()``
here unchanged) from
``.team-code/plans/pretraining-diagnostic-gate.md``.

This module imports ONLY from ``ensemble_lib`` -- never from ``train_ensemble.py`` -- so the
import graph stays acyclic: ``train_ensemble.py -> diagnose_signal.py -> ensemble_lib.py``.
Every function here is pure / qlib-read-only (no training): they consume already-trained
models, already-computed predictions, and read-only qlib data queries.

Item #8 (walk-forward / purged CV) is explicitly OUT OF SCOPE for this module and this pass
-- it is a separate, deferred milestone (its own compute-budget sizing decision) and is not
implemented here. Item #9 (sector/month attribution) is deferred per the plan and is not
referenced at all.
"""

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import pandas as pd
from statsmodels.stats.power import TTestPower
from statsmodels.stats.weightstats import DescrStatsW

from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.backtest.profit_attribution import get_stock_weight_df

import ensemble_lib

logger = logging.getLogger("DiagnoseSignal")

# Gate item #2's detection threshold: "|Rank_IC_tstat| >= ~2 (95%) to call a reported RankICIR
# 'detected, not noise'" (plan Section 8). A placeholder consistent with `confidence=0.95`
# (two-sided 95% critical t is ~1.96 for large-sample df), not a first-principles constraint --
# see plan Section 12, Open Question #3 (named there for `min_total_shrinkage`/`confidence`; the
# same "judgment call, needs team sign-off" framing applies here).
RANK_IC_TSTAT_THRESHOLD = 2.0

# Target statistical power (1 - beta) used for gate item #2's "minimum detectable mean RankIC"
# framing. The plan names `confidence` (the significance level) as an open parameter (Section 4)
# but does not specify a target power for the companion minimum-detectable-effect calculation;
# 0.80 is the conventional Cohen's-convention default, named explicitly here as a judgment call
# in the same spirit as the plan's other placeholder defaults.
DEFAULT_TARGET_POWER = 0.80

_REF_PATTERN = re.compile(r"Ref\(\s*\$([A-Za-z_][A-Za-z0-9_]*)\s*,\s*(-?\d+)\s*\)")


# ---------------------------------------------------------------------------
# Gate item #2: noise-floor framing
# ---------------------------------------------------------------------------
def compute_ic_with_power(
    pred_series: pd.Series,
    label_series: pd.Series,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """Gate item #2. Reuses ``ensemble_lib.calc_ic_metrics``'s groupby logic (via its additive
    ``return_daily_series`` parameter) rather than recomputing daily Rank IC, then frames the
    daily Rank IC series with a one-sample t-test against zero (Grinold & Kahn noise-floor
    framing) using ``statsmodels`` (scipy is not a core qlib dependency; see plan Section 6/12
    option (b)).

    Returns
    -------
    Dict[str, float]
        {"IC", "ICIR", "Rank IC", "Rank ICIR", "Rank_IC_daily_std", "n_days",
         "Rank_IC_tstat", "Rank_IC_pvalue", "min_detectable_mean_RankIC"}
    """
    ic_dict, daily_ric = ensemble_lib.calc_ic_metrics(pred_series, label_series, return_daily_series=True)
    clean = daily_ric.dropna()
    n_days = int(clean.shape[0])
    daily_std = float(clean.std(ddof=1)) if n_days >= 2 else float("nan")

    tstat, pvalue = float("nan"), float("nan")
    if n_days >= 2:
        try:
            tstat_, pvalue_, _df = DescrStatsW(clean.to_numpy()).ttest_mean(0.0)
            tstat, pvalue = float(tstat_), float(pvalue_)
        except Exception:
            logger.warning("compute_ic_with_power: one-sample t-test failed; reporting NaN.", exc_info=True)

    min_detectable = float("nan")
    if n_days >= 2 and daily_std == daily_std and daily_std > 0:
        try:
            alpha = 1.0 - confidence
            effect_size = TTestPower().solve_power(
                nobs=n_days, alpha=alpha, power=DEFAULT_TARGET_POWER, alternative="two-sided"
            )
            min_detectable = float(effect_size) * daily_std
        except Exception:
            logger.warning("compute_ic_with_power: min-detectable-effect solve failed; reporting NaN.", exc_info=True)

    result: Dict[str, float] = dict(ic_dict)
    result.update(
        {
            "Rank_IC_daily_std": round(daily_std, 5) if daily_std == daily_std else float("nan"),
            "n_days": n_days,
            "Rank_IC_tstat": round(tstat, 5) if tstat == tstat else float("nan"),
            "Rank_IC_pvalue": round(pvalue, 5) if pvalue == pvalue else float("nan"),
            "min_detectable_mean_RankIC": round(min_detectable, 5) if min_detectable == min_detectable else float("nan"),
        }
    )
    return result


# ---------------------------------------------------------------------------
# Gate item #3: linear baseline factors
# ---------------------------------------------------------------------------
def fetch_price_panel(
    instruments: Union[str, List[str]],
    start_time: str,
    end_time: str,
    fields: List[str],
    price_field: str = "close",
) -> pd.DataFrame:
    """Thin wrapper over ``qlib.data.D.features()``. Returns a (datetime, instrument)-
    MultiIndexed frame matching ``pred_series``'s index shape -- ``D.features()`` itself
    returns (instrument, datetime); this function swaps levels to match, mirroring
    ``QlibDataLoader``'s own default ``swap_level=True`` behavior (see
    ``qlib/data/dataset/loader.py``).

    ``instruments`` accepts the same shapes ``ensemble_lib.parse_instruments()`` produces: a
    market/universe name (str, e.g. "sp500") or an explicit ticker list. ``D.features()``
    itself only accepts a dict stockpool config or a list/tuple/Index/ndarray (see
    ``qlib.data.data.LocalDatasetProvider.get_instruments_d``) -- a bare string raises
    ``ValueError: Unsupported input type for param `instrument```. ``QlibDataLoader.
    load_group_df`` resolves this by calling ``D.instruments(market)`` first when given a
    string; this function does the same so callers can pass either shape uniformly.

    ``price_field`` ("close" default, "open" for the item #10 variant) is not used to build
    `fields` itself -- callers construct ``f"${price_field}"`` and include it in `fields` --
    it is used here only to sanity-check that the caller actually did so, so a caller cannot
    silently fetch `$close` while claiming to score against `$open`.
    """
    expected_field = f"${price_field}"
    if expected_field not in fields:
        logger.warning(
            f"fetch_price_panel: price_field='{price_field}' implies '{expected_field}' should be "
            f"among the requested fields, but fields={fields} does not include it."
        )
    resolved_instruments = D.instruments(instruments) if isinstance(instruments, str) else instruments
    panel = D.features(resolved_instruments, fields, start_time=start_time, end_time=end_time, freq="day")
    panel = panel.swaplevel().sort_index()
    return panel


def _by_instrument(series: pd.Series, func) -> pd.Series:
    """Apply `func` independently within each instrument's own time series, preserving the
    original (datetime, instrument) index (no extra group-key level added)."""
    return series.groupby(level="instrument", group_keys=False).apply(func)


def compute_linear_baseline_ic(
    price_panel: pd.DataFrame,
    label_series: pd.Series,
    windows: Tuple[int, ...] = (1, 5, 21),
    vol_window: int = 60,
    price_field: str = "close",
) -> Dict[str, Dict[str, float]]:
    """Gate item #3. One row per factor ("reversal_1d", "reversal_5d", "reversal_21d",
    "resid_vol_60d"), each scored via ``compute_ic_with_power`` against the SAME
    ``label_series`` the GBDT models used.

    "resid_vol_Nd" uses the simpler of the two definitions the plan's PM review considered
    (plain N-day realized volatility of daily returns) rather than a CAPM-style rolling-beta
    residual-to-benchmark volatility -- the plan defaults to this simpler definition and flags
    the CAPM variant as a future ``--residual_vol_method`` option, not required this pass
    (Section 6/12, Open Question #2).
    """
    price_col = f"${price_field}"
    if price_col not in price_panel.columns:
        raise ValueError(f"compute_linear_baseline_ic: price_panel is missing expected column '{price_col}'.")
    prices = price_panel[price_col].sort_index()

    results: Dict[str, Dict[str, float]] = {}
    for w in windows:
        factor = _by_instrument(prices, lambda s, w=w: s.pct_change(w))
        results[f"reversal_{w}d"] = compute_ic_with_power(factor, label_series)

    daily_return = _by_instrument(prices, lambda s: s.pct_change(1))
    resid_vol = _by_instrument(daily_return, lambda s: s.rolling(vol_window).std())
    results[f"resid_vol_{vol_window}d"] = compute_ic_with_power(resid_vol, label_series)

    return results


# ---------------------------------------------------------------------------
# Gate item #4: feature/label lag audit
# ---------------------------------------------------------------------------
def _resolve_feature_label_exprs(handler: Any) -> Tuple[Tuple[List[str], List[str]], Tuple[List[str], List[str]]]:
    """Resolve the ACTUAL (exprs, names) used for the "feature" and "label" groups.

    ``Alpha158.get_feature_config()``/``get_label_config()`` are static, class-level defaults
    -- they do NOT reflect a ``label=`` (or ``feature=``) kwarg override applied at
    construction time (see ``qlib/contrib/data/handler.py:122``:
    ``kwargs.pop("label", self.get_label_config())``). Gate item #10's ``open_shift`` variant
    is exactly such an override, so calling ``get_label_config()`` directly would silently
    audit the wrong (default, close-based) label expression and always misreport a false
    misalignment for that variant. qlib's own ``QlibDataLoader``/``DLWParser`` stores what was
    actually resolved per group in ``data_loader.fields[group]`` (see
    ``qlib/data/dataset/loader.py``'s ``DLWParser.__init__``/``_parse_fields_info``), so prefer
    that when available. Falls back to ``get_feature_config()``/``get_label_config()`` for
    handlers/test doubles that don't expose ``data_loader.fields`` (e.g. this module's own
    synthetic unit tests), which is also the plan's originally-documented lookup path for the
    common (no-override) case.
    """
    data_loader = getattr(handler, "data_loader", None)
    fields = getattr(data_loader, "fields", None) if data_loader is not None else None
    if isinstance(fields, dict) and "feature" in fields and "label" in fields:
        return fields["feature"], fields["label"]
    return handler.get_feature_config(), handler.get_label_config()


def audit_feature_label_lag(
    handler: Any,
    strategy_shift: int,
    deal_price: str,
) -> Dict[str, Any]:
    """Gate item #4. Regex-parses ``Ref($field, N)`` out of the handler's ACTUAL feature/label
    expressions (see ``_resolve_feature_label_exprs`` -- NOT necessarily
    ``handler.get_feature_config()[0]``/``handler.get_label_config()[0]`` verbatim, since those
    two methods don't reflect a runtime ``label=`` override); asserts every feature Ref uses
    N >= 0 (backward-looking) and that the label's two Ref offsets, combined with
    `strategy_shift`/`deal_price`, reproduce the entry/exit relationship team-finance verified
    by hand. This is a report, not a guard -- it never raises; callers read
    ``result["aligned"]``.
    """
    (feature_exprs, feature_names), (label_exprs, label_names) = _resolve_feature_label_exprs(handler)

    violations: List[str] = []

    forward_looking_features = []
    for name, expr in zip(feature_names, feature_exprs):
        for field, n_str in _REF_PATTERN.findall(expr):
            n = int(n_str)
            if n < 0:
                forward_looking_features.append({"name": name, "expr": expr, "field": field, "offset": n})
    if forward_looking_features:
        violations.append(
            f"{len(forward_looking_features)} feature(s) reference forward (look-ahead) Ref() "
            f"offsets: {[f['name'] for f in forward_looking_features]}"
        )

    label_expr = label_exprs[0] if label_exprs else ""
    label_refs = [(field, int(n)) for field, n in _REF_PATTERN.findall(label_expr)]
    label_detail: Dict[str, Any] = {"expr": label_expr, "refs": label_refs}

    if len(label_refs) != 2:
        violations.append(
            f"Label expression does not have exactly 2 Ref($field,N) terms (found {len(label_refs)}): {label_expr!r}"
        )
    else:
        fields_used = {f for f, _ in label_refs}
        if len(fields_used) != 1:
            violations.append(f"Label expression mixes different fields in its two Ref() terms: {sorted(fields_used)}")
        label_field = label_refs[0][0]
        offsets = sorted(n for _, n in label_refs)  # ascending: more negative (exit) first, then entry
        exit_offset, entry_offset = offsets[0], offsets[1]
        label_detail.update(label_field=label_field, exit_offset=exit_offset, entry_offset=entry_offset)

        if not (exit_offset < 0 and entry_offset < 0):
            violations.append(f"Label Ref offsets are not both forward-looking (future) offsets: {offsets}")
        if abs(entry_offset) != strategy_shift:
            violations.append(
                f"Label entry offset magnitude ({abs(entry_offset)}) does not match strategy_shift "
                f"({strategy_shift}) -- the strategy would trade at a different bar than the label assumes."
            )
        if label_field != deal_price:
            violations.append(
                f"Label field '${label_field}' does not match deal_price='{deal_price}' -- the label's "
                f"return basis and the strategy's execution price basis are misaligned."
            )

    return {
        "aligned": len(violations) == 0,
        "violations": violations,
        "forward_looking_features": forward_looking_features,
        "label_detail": label_detail,
        "strategy_shift": strategy_shift,
        "deal_price": deal_price,
    }


# ---------------------------------------------------------------------------
# Gate item #5: train/valid/test IC + dispersion
# ---------------------------------------------------------------------------
def _extract_best_iteration(model_wrapper: Any) -> Optional[int]:
    """Best-effort extraction of a GBDT model's early-stopping stop point across qlib's
    LGBModel/XGBModel/CatBoostModel wrappers, whose native booster objects expose this
    differently (LightGBM/XGBoost: `.best_iteration`; CatBoost: `.get_best_iteration()` or
    `.tree_count_`). Returns None rather than raising if none of these are available."""
    native = getattr(model_wrapper, "model", None)
    if native is None:
        return None
    for attr in ("best_iteration", "best_iteration_"):
        val = getattr(native, attr, None)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                pass
    get_best = getattr(native, "get_best_iteration", None)
    if callable(get_best):
        try:
            val = get_best()
            if val is not None:
                return int(val)
        except Exception:
            pass
    tree_count = getattr(native, "tree_count_", None)
    if tree_count is not None:
        try:
            return int(tree_count)
        except (TypeError, ValueError):
            pass
    return None


def _prediction_dispersion(pred_series: pd.Series, near_constant_eps: float = 1e-6) -> Dict[str, float]:
    """Cross-sectional dispersion of a prediction series, by day. A model producing nearly
    identical scores for every instrument on a given day (near-zero cross-sectional std) is
    degenerate for a top-K selector even before looking at IC."""
    daily_std = pred_series.groupby(level="datetime").std()
    if len(daily_std) == 0:
        return {"pred_dispersion_mean_daily_std": float("nan"), "pred_dispersion_frac_near_constant_days": float("nan")}
    return {
        "pred_dispersion_mean_daily_std": float(daily_std.mean()),
        "pred_dispersion_frac_near_constant_days": float((daily_std < near_constant_eps).mean()),
    }


def compute_segment_ic_and_dispersion(
    trained_models: Dict[str, Any],
    dataset: DatasetH,
    segments: Tuple[str, ...] = ("train", "valid", "test"),
) -> Dict[str, Dict[str, Any]]:
    """Gate item #5. Per model, per segment: ``compute_ic_with_power`` plus prediction
    dispersion stats and the model's ``best_iteration`` (surfaced once per model, repeated
    across each of that model's segment entries for convenience)."""
    segment_labels: Dict[str, pd.Series] = {}
    for segment in segments:
        seg_df = dataset.prepare(segment, col_set="label", data_key=DataHandlerLP.DK_L)
        segment_labels[segment] = seg_df.iloc[:, 0]

    results: Dict[str, Dict[str, Any]] = {}
    for model_name, model in trained_models.items():
        best_iteration = _extract_best_iteration(model)
        results[model_name] = {}
        for segment in segments:
            pred = model.predict(dataset, segment=segment)
            label = segment_labels[segment]
            ic_stats = compute_ic_with_power(pred, label)
            dispersion = _prediction_dispersion(pred)
            results[model_name][segment] = {**ic_stats, **dispersion, "best_iteration": best_iteration}
    return results


# ---------------------------------------------------------------------------
# Gate item #6: IC decay curve
# ---------------------------------------------------------------------------
def compute_ic_decay_curve(
    predictions: Dict[str, pd.Series],
    price_panel: pd.DataFrame,
    horizons: range = range(1, 11),
    price_field: str = "close",
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Gate item #6. Re-labels the SAME fixed test-segment `predictions` at h=1..10 trading
    days (no retraining) and scores each via ``compute_ic_with_power``. `price_field` must
    match whatever `fetch_price_panel()` was called with, so the re-labeling is built from the
    same price basis the model's own label used (never hardcode `$close` here).

    Horizon h's label is defined the same way Alpha158's own default label is (entry at the
    next tradable bar after the prediction, i.e. Ref(price,-1), matching TopkDropoutStrategy's
    shift=1): ``price(entry + h) / price(entry) - 1`` where ``entry = Ref(price, -1)``. h=1
    reproduces the default label's own return interval exactly.
    """
    price_col = f"${price_field}"
    prices = price_panel[price_col].sort_index()

    entry = _by_instrument(prices, lambda s: s.shift(-1))

    results: Dict[str, Dict[int, Dict[str, float]]] = {}
    for model_name, pred in predictions.items():
        results[model_name] = {}
        for h in horizons:
            exit_price = _by_instrument(prices, lambda s, h=h: s.shift(-(1 + h)))
            label_h = (exit_price / entry - 1.0).reindex(pred.index)
            results[model_name][int(h)] = compute_ic_with_power(pred, label_h)
    return results


# ---------------------------------------------------------------------------
# Gate item #7: turnover / cost stress
# ---------------------------------------------------------------------------
def _compute_daily_turnover(positions_normal: Any) -> pd.Series:
    """One-way daily turnover (0.5 * sum of absolute weight changes across instruments,
    treating an unheld instrument as weight 0), derived from qlib's own
    ``get_stock_weight_df`` (as used by ``qlib.contrib.report.analysis_position``)."""
    if not positions_normal:
        return pd.Series(dtype="float64")
    weight_df = get_stock_weight_df(positions_normal).fillna(0.0).sort_index()
    turnover = weight_df.diff().abs().sum(axis=1) * 0.5
    if len(turnover):
        turnover.iloc[0] = float("nan")  # no prior day to diff the first day against
    return turnover


def compute_turnover_and_cost_stress(
    pred_series: pd.Series,
    benchmark: str,
    codes: Union[str, List[str]],
    baseline_bp: float = 1.0,
    stress_bp: float = 10.0,
    deal_price: str = "close",
    topk: int = 50,
    n_drop: int = 5,
    annualization_n: int = 252,
) -> Dict[str, Any]:
    """Gate item #7. Consumes `positions_normal` (obtained via
    ``ensemble_lib.run_portfolio_backtest``'s additive ``return_positions=True``, previously
    discarded) for turnover, and calls the generalized ``run_portfolio_backtest(...,
    open_cost=, close_cost=)`` twice (baseline vs. stress bp) to compare
    `information_ratio_net` degradation. `deal_price` should match whichever label variant is
    being diagnosed (Section 8 of the plan).

    `codes` : Union[str, List[str]]
        The already-resolved trading universe (``ensemble_lib.parse_instruments()``'s return
        value from ``run_diagnostics()``) -- forwarded verbatim to both
        ``run_portfolio_backtest()`` calls below, which now require it explicitly (see that
        function's docstring for why: qlib's own unset-``codes`` default resolves to a market
        name, ``<data_dir>/instruments/all.txt``, not a wildcard).
    """
    baseline_cost = baseline_bp / 10000.0
    stress_cost = stress_bp / 10000.0

    baseline_result, positions_normal = ensemble_lib.run_portfolio_backtest(
        pred_series,
        benchmark=benchmark,
        codes=codes,
        topk=topk,
        n_drop=n_drop,
        annualization_n=annualization_n,
        open_cost=baseline_cost,
        close_cost=baseline_cost,
        deal_price=deal_price,
        return_positions=True,
    )
    stress_result = ensemble_lib.run_portfolio_backtest(
        pred_series,
        benchmark=benchmark,
        codes=codes,
        topk=topk,
        n_drop=n_drop,
        annualization_n=annualization_n,
        open_cost=stress_cost,
        close_cost=stress_cost,
        deal_price=deal_price,
    )

    mean_daily_turnover = float("nan")
    if baseline_result.get("status") == "ok" and positions_normal:
        turnover_series = _compute_daily_turnover(positions_normal).dropna()
        if len(turnover_series):
            mean_daily_turnover = float(turnover_series.mean())

    ir_baseline = baseline_result.get("information_ratio_net")
    ir_stress = stress_result.get("information_ratio_net")
    sign_flip = False
    pct_degradation = float("nan")
    if (
        ir_baseline is not None
        and ir_stress is not None
        and ir_baseline == ir_baseline
        and ir_stress == ir_stress
        and ir_baseline != 0
    ):
        sign_flip = (ir_baseline > 0) != (ir_stress > 0)
        pct_degradation = float((ir_baseline - ir_stress) / abs(ir_baseline))

    return {
        "mean_daily_turnover": round(mean_daily_turnover, 5) if mean_daily_turnover == mean_daily_turnover else float("nan"),
        "baseline_bp": baseline_bp,
        "stress_bp": stress_bp,
        "information_ratio_net_baseline": ir_baseline,
        "information_ratio_net_stress": ir_stress,
        "information_ratio_sign_flip": bool(sign_flip),
        "information_ratio_pct_degradation": round(pct_degradation, 5) if pct_degradation == pct_degradation else float("nan"),
        "baseline_backtest_status": baseline_result.get("status"),
        "stress_backtest_status": stress_result.get("status"),
    }


# ---------------------------------------------------------------------------
# Gate verdict heuristics (advisory only -- see plan Section 10: never sys.exit()s)
# ---------------------------------------------------------------------------
def _verdict_from_tstat(tstat: Optional[float], label: str) -> Dict[str, str]:
    if tstat is None or tstat != tstat:
        return {"status": "inconclusive", "rationale": f"{label}: Rank_IC_tstat unavailable (NaN)"}
    if abs(tstat) >= RANK_IC_TSTAT_THRESHOLD:
        return {"status": "pass", "rationale": f"{label}: Rank_IC_tstat={tstat:.2f}, |t|>={RANK_IC_TSTAT_THRESHOLD} -- signal detected, not noise"}
    return {"status": "fail", "rationale": f"{label}: Rank_IC_tstat={tstat:.2f}, |t|<{RANK_IC_TSTAT_THRESHOLD} -- statistically indistinguishable from noise"}


def _verdict_segment_pattern(model_name: str, train_t: Optional[float], test_t: Optional[float]) -> Dict[str, str]:
    if train_t is None or test_t is None or train_t != train_t or test_t != test_t:
        return {"status": "inconclusive", "rationale": f"{model_name}: train/test Rank_IC_tstat unavailable"}
    train_sig = abs(train_t) >= RANK_IC_TSTAT_THRESHOLD
    test_sig = abs(test_t) >= RANK_IC_TSTAT_THRESHOLD
    if not train_sig and not test_sig:
        return {
            "status": "fail",
            "rationale": f"{model_name}: train t={train_t:.2f}, test t={test_t:.2f} -- no signal even in-sample (underfit/no-signal-for-model)",
        }
    if train_sig and not test_sig:
        return {
            "status": "inconclusive",
            "rationale": f"{model_name}: train t={train_t:.2f} clears threshold but test t={test_t:.2f} does not -- classic overfit pattern",
        }
    return {"status": "pass", "rationale": f"{model_name}: train t={train_t:.2f}, test t={test_t:.2f} -- both clear the noise threshold"}


def _verdict_decay_curve(per_horizon: Dict[int, Dict[str, float]]) -> Dict[str, str]:
    if not per_horizon:
        return {"status": "not_run", "rationale": "no horizons computed"}
    tstats = {h: v.get("Rank_IC_tstat") for h, v in per_horizon.items()}
    ric = {h: v.get("Rank IC") for h, v in per_horizon.items()}
    valid = {h: t for h, t in tstats.items() if t is not None and t == t}
    if not valid:
        return {"status": "inconclusive", "rationale": "all horizon t-stats unavailable (NaN)"}
    hits = [h for h, t in valid.items() if abs(t) >= RANK_IC_TSTAT_THRESHOLD]
    if not hits:
        return {
            "status": "fail",
            "rationale": f"no horizon h=1..{max(valid)} clears |t|>={RANK_IC_TSTAT_THRESHOLD} (t-stats: {valid})",
        }
    coherent = []
    for h in hits:
        left = ric.get(h - 1)
        right = ric.get(h + 1)
        same_sign_left = left is not None and left == left and (left > 0) == (ric[h] > 0)
        same_sign_right = right is not None and right == right and (right > 0) == (ric[h] > 0)
        if same_sign_left or same_sign_right:
            coherent.append(h)
    if coherent:
        return {
            "status": "pass",
            "rationale": f"horizon(s) {coherent} clear |t|>={RANK_IC_TSTAT_THRESHOLD} with a same-signed neighbor -- label-horizon hypothesis worth pursuing",
        }
    return {
        "status": "inconclusive",
        "rationale": f"horizon(s) {hits} clear the threshold but without a coherent same-signed neighbor -- may be noise",
    }


def _verdict_turnover_cost(tc: Dict[str, Any]) -> Dict[str, str]:
    if tc.get("baseline_backtest_status") != "ok" or tc.get("stress_backtest_status") != "ok":
        return {"status": "inconclusive", "rationale": "baseline or stress backtest failed"}
    if tc.get("information_ratio_sign_flip"):
        return {
            "status": "fail",
            "rationale": f"IR sign flips between baseline ({tc.get('information_ratio_net_baseline')}) and "
            f"stress ({tc.get('information_ratio_net_stress')}) cost -- any net signal is cost-fragile",
        }
    degradation = tc.get("information_ratio_pct_degradation")
    if degradation is not None and degradation == degradation and degradation >= 0.5:
        return {"status": "fail", "rationale": f"IR degrades {degradation:.0%} under {tc.get('stress_bp')}bp stress cost -- cost-fragile"}
    return {
        "status": "pass",
        "rationale": f"IR degradation under stress cost is {degradation if degradation == degradation else 'n/a'}; no sign flip",
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_diagnostics(
    dataset: DatasetH,
    trained_models: Dict[str, Any],
    predictions: Dict[str, pd.Series],
    args: Any,
    price_field: str = "close",
) -> Dict[str, Any]:
    """Orchestrator ``train_ensemble.py`` calls when ``--diagnose`` is set -- for BOTH the
    default run and (per item #10) the ``open_shift`` variant run, passing
    ``price_field="open"`` in the latter case so every internal ``fetch_price_panel()``/
    ``compute_ic_decay_curve()`` call scores against the same price basis the variant's own
    label used. Returns the full ``diagnostics_report.json`` payload, including a top-level
    ``"gate_verdict"`` dict of ``{item: {status, rationale}}``.

    Advisory only: never raises or exits on a failing item (see plan Section 10).
    """
    report: Dict[str, Any] = {}
    gate_verdict: Dict[str, Dict[str, str]] = {}
    deal_price = price_field  # this pipeline's convention: price_field always mirrors deal_price

    # --- Item #1: budget guard (already ran, and either passed or was explicitly overridden,
    # before training started in train_ensemble.py's main()). Recorded here so the single
    # diagnostics_report.json is a complete artifact.
    learning_rate = ensemble_lib.DEFAULT_LEARNING_RATE
    total_shrinkage = args.num_boost_round * learning_rate
    allow_undertrained = bool(getattr(args, "allow_undertrained", False))
    budget_ok = total_shrinkage >= ensemble_lib.DEFAULT_MIN_TOTAL_SHRINKAGE
    report["budget_guard"] = {
        "num_boost_round": args.num_boost_round,
        "learning_rate": learning_rate,
        "total_shrinkage": total_shrinkage,
        "min_total_shrinkage": ensemble_lib.DEFAULT_MIN_TOTAL_SHRINKAGE,
        "allow_undertrained": allow_undertrained,
        "passed_without_override": budget_ok,
    }
    if budget_ok:
        gate_verdict["1"] = {
            "status": "pass",
            "rationale": f"total_shrinkage={total_shrinkage:.3f} >= {ensemble_lib.DEFAULT_MIN_TOTAL_SHRINKAGE} threshold",
        }
    else:
        # main() would have raised (and never reached here) unless --allow_undertrained was set.
        gate_verdict["1"] = {
            "status": "fail",
            "rationale": (
                f"total_shrinkage={total_shrinkage:.3f} < {ensemble_lib.DEFAULT_MIN_TOTAL_SHRINKAGE}; "
                f"--allow_undertrained explicitly overrode the guard -- readings below are a smoke "
                f"test, not a conclusive diagnostic run"
            ),
        }

    # --- Item #4: feature/label lag audit -- pure static/config introspection.
    lag_audit = audit_feature_label_lag(dataset.handler, strategy_shift=ensemble_lib.STRATEGY_SHIFT, deal_price=deal_price)
    report["lag_audit"] = lag_audit
    if lag_audit["aligned"]:
        gate_verdict["4"] = {"status": "pass", "rationale": "0 forward-looking features; label/shift/deal_price aligned"}
    else:
        gate_verdict["4"] = {"status": "fail", "rationale": "; ".join(lag_audit["violations"])}

    # Test-segment label, shared by items #2/#3/#6.
    test_label = dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_L).iloc[:, 0]

    # --- Item #2: noise-floor framing on each model's (already in-memory) test predictions.
    ic_power = {model_name: compute_ic_with_power(pred, test_label) for model_name, pred in predictions.items()}
    report["ic_power"] = ic_power
    primary_key = "Ensemble_Blended" if "Ensemble_Blended" in ic_power else next(iter(ic_power), None)
    if primary_key is not None:
        gate_verdict["2"] = _verdict_from_tstat(ic_power[primary_key].get("Rank_IC_tstat"), primary_key)
    else:
        gate_verdict["2"] = {"status": "not_run", "rationale": "no predictions available"}

    # --- Item #3: linear baseline factors on the same universe/dates/label.
    data_dir = Path(args.data_dir).expanduser().resolve()
    instruments = ensemble_lib.parse_instruments(args.market, data_dir)
    # 150 calendar days of buffer before test_start comfortably covers the 60-trading-day
    # rolling-vol window plus the longest (21d) reversal window with margin to spare.
    buffer_start = (pd.Timestamp(args.test_start) - pd.Timedelta(days=150)).strftime("%Y-%m-%d")
    price_panel = fetch_price_panel(
        instruments, buffer_start, args.test_end, fields=[f"${price_field}"], price_field=price_field
    )
    baseline_factors = compute_linear_baseline_ic(price_panel, test_label, price_field=price_field)
    report["baseline_factors"] = baseline_factors
    baseline_tstats = {k: v.get("Rank_IC_tstat") for k, v in baseline_factors.items()}
    valid_baseline_tstats = {k: t for k, t in baseline_tstats.items() if t is not None and t == t}
    if not valid_baseline_tstats:
        gate_verdict["3"] = {"status": "inconclusive", "rationale": "baseline factor t-stats unavailable (NaN)"}
    elif all(abs(t) < RANK_IC_TSTAT_THRESHOLD for t in valid_baseline_tstats.values()):
        gate_verdict["3"] = {
            "status": "inconclusive",
            "rationale": f"ALL baseline factors also fail |t|>={RANK_IC_TSTAT_THRESHOLD} ({valid_baseline_tstats}) "
            f"-- supports a window/universe noise-floor explanation over a model-specific one",
        }
    else:
        gate_verdict["3"] = {
            "status": "inconclusive",
            "rationale": f"at least one baseline factor clears the noise threshold ({valid_baseline_tstats}) "
            f"-- compare against item #2's model read to judge model vs. window",
        }

    # --- Item #5: train/valid/test IC + dispersion (extra prepare()/predict() calls).
    segment_ic = compute_segment_ic_and_dispersion(trained_models, dataset)
    report["segment_ic"] = segment_ic
    if segment_ic:
        first_model = next(iter(segment_ic))
        train_t = segment_ic[first_model].get("train", {}).get("Rank_IC_tstat")
        test_t = segment_ic[first_model].get("test", {}).get("Rank_IC_tstat")
        gate_verdict["5"] = _verdict_segment_pattern(first_model, train_t, test_t)
    else:
        gate_verdict["5"] = {"status": "not_run", "rationale": "no trained models available"}

    # --- Item #6: IC decay curve h=1..10 on the fixed test predictions (no retraining).
    ic_decay = compute_ic_decay_curve(predictions, price_panel, price_field=price_field)
    report["ic_decay"] = ic_decay
    decay_key = "Ensemble_Blended" if "Ensemble_Blended" in ic_decay else next(iter(ic_decay), None)
    gate_verdict["6"] = _verdict_decay_curve(ic_decay.get(decay_key, {})) if decay_key else {
        "status": "not_run",
        "rationale": "no predictions available",
    }

    # --- Item #7: turnover / cost stress, per prediction series.
    turnover_cost = {
        model_name: compute_turnover_and_cost_stress(
            pred, benchmark=args.benchmark, codes=instruments, deal_price=deal_price
        )
        for model_name, pred in predictions.items()
    }
    report["turnover_cost"] = turnover_cost
    tc_key = "Ensemble_Blended" if "Ensemble_Blended" in turnover_cost else next(iter(turnover_cost), None)
    gate_verdict["7"] = _verdict_turnover_cost(turnover_cost[tc_key]) if tc_key is not None else {
        "status": "not_run",
        "rationale": "no predictions available",
    }

    # --- Item #10: paired label/execution variant. A single run cannot verdict the *paired*
    # comparison alone (Section 10: no automated diff tool is built this pass) -- for the
    # default run this item simply did not apply; for the open_shift run, surface this run's
    # OWN item #2 read with a pointer to the comparison that still needs to happen by hand.
    label_variant = getattr(args, "label_variant", "default")
    if label_variant == "default":
        gate_verdict["10"] = {
            "status": "not_run",
            "rationale": "this is the default label/execution run; re-run with --label_variant open_shift for the paired variant",
        }
    else:
        base_entry = gate_verdict.get("2", {"status": "not_run", "rationale": "no item #2 result"})
        gate_verdict["10"] = {
            "status": base_entry["status"],
            "rationale": (
                f"[label_variant={label_variant}] {base_entry['rationale']} -- compare this run's "
                f"Rank_IC_tstat against the default run's diagnostics_report.json (same significance "
                f"framing, not a raw-number diff) to judge whether horizon/execution alignment helped"
            ),
        }

    report["gate_verdict"] = gate_verdict
    report["price_field"] = price_field
    report["label_variant"] = label_variant
    return report
