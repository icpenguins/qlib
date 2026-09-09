#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared library for the GBDT ensemble pipeline
==============================================
Extracted from ``scripts/train_ensemble.py`` so that both ``train_ensemble.py``
(the training CLI) and ``scripts/diagnose_signal.py`` (the pre-training diagnostic
gate, see ``.team-code/plans/pretraining-diagnostic-gate.md``) can import the same
dataset/model/backtest plumbing without either script importing the other -- that
would create a circular import (``train_ensemble -> diagnose_signal -> train_ensemble``).

This module owns:
- ``parse_instruments`` / ``audit_universe_bias``: universe resolution + survivorship
  bias audit (unchanged from the pre-extraction ``train_ensemble.py``).
- ``build_dataset``: newly extracted from ``train_ensemble.py``'s ``main()`` so the
  identical Alpha158/DatasetH construction is available to callers other than
  ``train_ensemble.py`` itself. Accepts an additive ``label_config`` override (used by
  ``train_ensemble.py --label_variant open_shift``, gate item #10) that leaves the
  default (``label_config=None``) behavior byte-for-byte identical to before.
- ``calc_ic_metrics``: unchanged default behavior; additive ``return_daily_series``
  parameter exposes the daily Rank IC series (previously computed then discarded) for
  gate item #2's noise-floor framing, without changing the return shape for existing
  callers that omit the new parameter.
- ``run_portfolio_backtest``: unchanged default behavior; additive ``open_cost``/
  ``close_cost``/``deal_price``/``return_positions`` parameters support gate item #7's
  cost-stress test and item #10's open-price execution variant.
- ``build_and_train_models`` / ``blend_predictions`` / ``_json_safe``: unchanged.
- ``assert_sufficient_training_budget``: new, gate item #1's guard. Lives here and only
  here (single home, per the plan's Section 3 circular-import correction).
"""

import sys
import time
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure repository root is in sys.path so `import qlib` resolves even when this module
# is imported directly (e.g. from a test file) rather than via a `scripts/*.py` entrypoint.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from qlib.data.dataset import DatasetH
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.evaluate import risk_analysis, backtest_daily
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.model.ens.ensemble import AverageEnsemble

logger = logging.getLogger("EnsembleAlpha")

# Default learning rate applied by build_and_train_models to all three learners. Named
# here (rather than left as a bare literal at each call site) so
# assert_sufficient_training_budget's caller in train_ensemble.py's main() can reuse the
# exact same value the training call itself uses, without a second, possibly-drifting copy.
DEFAULT_LEARNING_RATE = 0.05

# Gate item #1's default minimum total shrinkage (num_boost_round * learning_rate). This is
# a placeholder threshold pending explicit team/PM sign-off (see plan Section 12, Open
# Question #3) -- not a value derived from a first-principles constraint.
DEFAULT_MIN_TOTAL_SHRINKAGE = 5.0

# TopkDropoutStrategy's execution shift is a hardcoded literal inside `generate_trade_decision`
# (qlib/contrib/strategy/signal_strategy.py:141-142, :351-352), not a constructor parameter --
# see the pretraining-diagnostic-gate plan's Section 4/7. Named here once, shared by
# train_ensemble.py (console/label-variant docs) and diagnose_signal.py's gate item #4 lag audit,
# so neither script needs to reach into the other to reference this fact (which would violate the
# acyclic import rule) or repeat the literal `1` with no explanation.
STRATEGY_SHIFT = 1


def parse_instruments(instruments_arg: str, data_dir: Path) -> Union[str, List[str]]:
    """Parse instruments argument into universe name or symbol list."""
    inst_str = instruments_arg.strip()
    if "," in inst_str:
        return [t.strip().upper() for t in inst_str.split(",") if t.strip()]
    inst_file = data_dir / "instruments" / f"{inst_str}.txt"
    if inst_file.exists():
        return inst_str
    logger.warning(
        f"    [WARN] Market '{inst_str}' did not resolve to an instruments file "
        f"({inst_file}). Falling back to a single-ticker universe: ['{inst_str.upper()}']. "
        f"This is almost certainly not what you want -- check the --market spelling "
        f"and --data_dir path."
    )
    return [inst_str.upper()]


def audit_universe_bias(
    instruments: Union[str, List[str]],
    data_dir: Path,
    allow_survivorship_bias: bool = False,
    same_date_threshold: float = 0.90,
) -> Dict[str, Any]:
    """
    Inspect a named universe's point-in-time (start_date, end_date) columns for signs of
    survivorship / index-membership look-ahead bias, i.e. a constituent list that has been
    back-projected from today's membership rather than reconstructed from historical
    membership and delisting records.

    NOTE: This is a heuristic sanity check, not a fix for the underlying data problem.
    Qlib's instruments files already support point-in-time filtering via per-row
    (start_date, end_date) -- but that only helps if the dates in the file are true
    historical membership dates. Producing genuine historical Russell 1000 (or any index)
    membership/delisting history requires a licensed point-in-time data provider (e.g. the
    index vendor, CRSP, FactSet); this script cannot fabricate that data, so instead it
    fails loudly (or warns, if explicitly overridden) rather than silently reporting
    survivorship-inflated backtest metrics.
    """
    audit = {"checked": False, "n_instruments": 0, "flagged": False, "detail": None}

    if not isinstance(instruments, str):
        # Explicit ticker list from the user -- can't check it against a membership file, but a
        # hand-picked list of tickers is itself a common way to smuggle in survivorship bias
        # (e.g. picking names you know did well/still exist), so at least say so.
        logger.info(
            f"    Universe is an explicit ticker list ({instruments}) -- not audited for "
            f"survivorship bias. A hand-picked list can itself be survivorship-biased if the "
            f"tickers were chosen with hindsight."
        )
        return audit

    inst_file = data_dir / "instruments" / f"{instruments}.txt"
    if not inst_file.exists():
        return audit

    rows = []
    for line in inst_file.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("\t") if "\t" in line else line.strip().split()
        if len(parts) >= 3:
            rows.append((parts[0], parts[1], parts[2]))

    if not rows:
        return audit

    starts = [r[1] for r in rows]
    ends = [r[2] for r in rows]
    n = len(rows)
    max_end, min_start = max(ends), min(starts)
    frac_same_end = sum(1 for e in ends if e == max_end) / n
    frac_same_start = sum(1 for s in starts if s == min_start) / n
    n_delisted = sum(1 for e in ends if e != max_end)

    audit.update(
        checked=True,
        n_instruments=n,
        frac_same_end_date=round(frac_same_end, 4),
        frac_same_start_date=round(frac_same_start, 4),
        max_end_date=max_end,
        min_start_date=min_start,
        n_apparently_delisted=n_delisted,
    )

    # A near-universal shared end_date is, by itself, sufficient evidence of survivorship bias --
    # it means (almost) no name is ever recorded as having left the universe, which a real
    # historical membership file for a multi-year window would not exhibit. This is the sole
    # gating condition. `n_delisted == 0` is the exact special case `frac_same_end == 1.0`, so it
    # is deliberately NOT ANDed with the (much softer, and non-gating) start-date signal below --
    # a zero-delistings file with staggered add-dates must still be caught.
    severe = frac_same_end >= same_date_threshold
    audit["flagged"] = severe

    if severe:
        msg = (
            f"\n{'!' * 90}\n"
            f"SURVIVORSHIP / INDEX-MEMBERSHIP LOOK-AHEAD BIAS DETECTED in universe '{instruments}'\n"
            f"  {inst_file}\n"
            f"  {n} instruments; {frac_same_end:.1%} share the SAME end_date ({max_end}) -- i.e. only "
            f"{n_delisted} of {n} rows show any delisting/exit date at all.\n"
            f"  (For context, {frac_same_start:.1%} also share the same start_date {min_start}; this is "
            f"informational only and is not itself gating.)\n"
            f"  This looks like TODAY's index membership back-projected across the whole "
            f"backtest window, not reconstructed point-in-time membership. Training and testing\n"
            f"  a top-K selector on a universe of names *known* to have survived to the present\n"
            f"  will materially inflate every IC / return / Sharpe number this script reports.\n"
            f"  Fix: supply an instruments file with true historical add/drop and delisting dates\n"
            f"  (from your index vendor / CRSP / FactSet / similar), or pass\n"
            f"  --allow_survivorship_bias to proceed anyway (e.g. for a quick smoke test).\n"
            f"{'!' * 90}\n"
        )
        if allow_survivorship_bias:
            logger.warning(msg + "  [OVERRIDDEN] --allow_survivorship_bias was set; continuing anyway.\n")
        else:
            logger.error(msg)
            raise RuntimeError(
                f"Universe '{instruments}' appears to have survivorship/look-ahead bias "
                f"({frac_same_end:.1%} identical end dates, only {n_delisted}/{n} rows show a distinct "
                f"delisting date). Re-run with --allow_survivorship_bias to override once you understand the risk."
            )
    else:
        logger.info(
            f"    Universe bias audit for '{instruments}': {n} instruments, "
            f"{frac_same_end:.1%} share end_date={max_end}, {frac_same_start:.1%} share start_date={min_start}, "
            f"{n_delisted} show a distinct exit date. No severe bias flagged."
        )

    return audit


def assert_sufficient_training_budget(
    num_boost_round: int,
    learning_rate: float,
    min_total_shrinkage: float = DEFAULT_MIN_TOTAL_SHRINKAGE,
    allow_override: bool = False,
) -> None:
    """Gate item #1: fail loudly (mirrors ``audit_universe_bias``'s pattern) if the requested
    training budget is too small to produce a meaningful fit.

    ``num_boost_round * learning_rate`` ("total shrinkage") is a crude but cheap proxy for
    how much a GBDT ensemble has actually been allowed to fit: e.g. the 20-round smoke test
    that originally triggered this diagnostic gate had total shrinkage of ~1.0 (20 rounds *
    0.05 learning rate), which cannot support any conclusion about whether a real signal
    exists. ``min_total_shrinkage`` is a placeholder default (see plan Section 12, Open
    Question #3), not a value derived from a first-principles constraint -- it is expected to
    be revisited with explicit team/PM sign-off.

    Parameters
    ----------
    num_boost_round : int
        The requested number of boosting rounds.
    learning_rate : float
        The learning rate that will be applied to each learner.
    min_total_shrinkage : float
        The minimum acceptable ``num_boost_round * learning_rate`` product.
    allow_override : bool
        If True, log a warning and continue instead of raising (wired to the
        ``--allow_undertrained`` CLI flag).

    Raises
    ------
    RuntimeError
        If the total shrinkage is below ``min_total_shrinkage`` and ``allow_override`` is False.
    """
    total_shrinkage = num_boost_round * learning_rate
    if total_shrinkage >= min_total_shrinkage:
        logger.info(
            f"    Training budget check: num_boost_round({num_boost_round}) * "
            f"learning_rate({learning_rate}) = {total_shrinkage:.3f} >= "
            f"min_total_shrinkage({min_total_shrinkage}). OK."
        )
        return

    msg = (
        f"\n{'!' * 90}\n"
        f"INSUFFICIENT TRAINING BUDGET: num_boost_round({num_boost_round}) * "
        f"learning_rate({learning_rate}) = {total_shrinkage:.3f}, below the "
        f"min_total_shrinkage threshold of {min_total_shrinkage}.\n"
        f"  A model trained this lightly cannot support any conclusion about whether a real\n"
        f"  signal exists in this data -- near-zero IC from an undertrained run is\n"
        f"  indistinguishable from near-zero IC from a genuinely absent signal.\n"
        f"  Fix: increase --num_boost_round (or the learning rate), or pass\n"
        f"  --allow_undertrained to proceed anyway (e.g. for a quick smoke test where the\n"
        f"  diagnostic *readings themselves* are not meant to be trusted).\n"
        f"{'!' * 90}\n"
    )
    if allow_override:
        logger.warning(msg + "  [OVERRIDDEN] --allow_undertrained was set; continuing anyway.\n")
    else:
        logger.error(msg)
        raise RuntimeError(
            f"Training budget too small (num_boost_round({num_boost_round}) * "
            f"learning_rate({learning_rate}) = {total_shrinkage:.3f} < {min_total_shrinkage}). "
            f"Re-run with --allow_undertrained to override once you understand the risk."
        )


def calc_ic_metrics(
    pred_series: pd.Series,
    label_series: pd.Series,
    return_daily_series: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], pd.Series]]:
    """Calculate daily Information Coefficient (IC) and Rank IC.

    Parameters
    ----------
    pred_series, label_series : pd.Series
        (datetime, instrument)-MultiIndexed prediction and label series.
    return_daily_series : bool
        Additive parameter (default False preserves the exact prior return shape/behavior
        for existing callers). When True, also returns the daily Rank IC series that this
        function already computes internally -- previously discarded -- as a second return
        value, so gate item #2's ``compute_ic_with_power`` can reuse this function's groupby
        logic (one-sample t-test framing) rather than recomputing daily Rank IC itself.

    Returns
    -------
    Dict[str, float]
        When ``return_daily_series`` is False (default): unchanged from before --
        {"IC", "ICIR", "Rank IC", "Rank ICIR"}.
    Tuple[Dict[str, float], pd.Series]
        When ``return_daily_series`` is True: the same dict, plus the daily Rank IC series
        (indexed by datetime; NaN days included, not dropped).
    """
    df = pd.DataFrame({"pred": pred_series, "label": label_series}).dropna()
    if df.empty:
        # Same principle as the backtest failure handling below: "no usable data" is not the same
        # measurement as "IC of exactly zero" -- fabricating 0.0 here would silently read as a
        # genuinely uncorrelated (rather than absent) signal.
        logger.warning("    [WARN] calc_ic_metrics: no overlapping (pred, label) rows after dropna(); returning NaN.")
        result = {"IC": float("nan"), "ICIR": float("nan"), "Rank IC": float("nan"), "Rank ICIR": float("nan")}
        if return_daily_series:
            return result, pd.Series(dtype="float64", name="Rank IC")
        return result

    # Group by date to compute daily cross-sectional correlation. A day with a constant `pred`
    # or `label` column (e.g. very few names that day, or tied values) makes the correlation
    # genuinely undefined -- pandas/scipy correctly return NaN for it via a `ConstantInputWarning`,
    # which `.mean()`/`.std()` below already skip (skipna=True is the default), so this does not
    # corrupt the aggregate stats. Suppress only that specific, benign warning (never a blanket
    # warnings filter) so it doesn't spam the console on every such day, and log how many days
    # were actually degenerate so it's visible in a summary line instead of a wall of warnings.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="An input array is constant", category=Warning)
        daily_ic = df.groupby(level="datetime").apply(lambda d: d["pred"].corr(d["label"], method="pearson"))
        daily_ric = df.groupby(level="datetime").apply(lambda d: d["pred"].corr(d["label"], method="spearman"))

    n_degenerate_ic = int(daily_ic.isna().sum())
    n_degenerate_ric = int(daily_ric.isna().sum())
    if n_degenerate_ic or n_degenerate_ric:
        logger.info(
            f"    calc_ic_metrics: {n_degenerate_ic}/{len(daily_ic)} day(s) had an undefined "
            f"(constant-input) Pearson IC and {n_degenerate_ric}/{len(daily_ric)} an undefined "
            f"Spearman Rank IC (e.g. too few names or tied values that day) -- excluded via "
            f"skipna in the mean/std below, not treated as zero."
        )

    ic_mean = float(daily_ic.mean())
    ic_std = float(daily_ic.std())
    ric_mean = float(daily_ric.mean())
    ric_std = float(daily_ric.std())

    n_days = daily_ic.notna().sum()
    if n_days < 2:
        logger.warning(
            f"    [WARN] calc_ic_metrics: only {n_days} day(s) of daily IC available -- std/ICIR "
            f"is undefined with fewer than 2 days, reporting NaN rather than a fabricated value."
        )

    def _safe_ratio(mean: float, std: float, eps: float = 1e-8) -> float:
        # A std of NaN (fewer than 2 days), exactly 0, or merely floating-point noise around 0
        # (e.g. a daily IC series that is ~1.0 on every day due to a near-perfect signal, where
        # Pearson correlation's own float arithmetic leaves a ~1e-16 residual std instead of an
        # exact 0) all make the ratio effectively undefined. An `std > 0` check alone lets that
        # noise through and produces an absurd, arbitrarily large "ICIR" (observed: ~1e15) rather
        # than either the true undefined-ness or a sane bound -- guard with an epsilon instead.
        if not (std > eps):
            return float("nan")
        return mean / std

    result = {
        "IC": round(ic_mean, 5),
        "ICIR": round(_safe_ratio(ic_mean, ic_std), 5),
        "Rank IC": round(ric_mean, 5),
        "Rank ICIR": round(_safe_ratio(ric_mean, ric_std), 5),
    }
    if return_daily_series:
        return result, daily_ric
    return result


BACKTEST_METRIC_KEYS = (
    "annualized_return_gross",
    "information_ratio_gross",
    "annualized_return_net",
    "information_ratio_net",
    "max_relative_drawdown_net",
    "max_absolute_drawdown_net",
    "annualized_return_absolute_net",
)


def _failed_backtest_result(error: Exception) -> Dict[str, Any]:
    """A backtest failure is NOT the same thing as a flat/zero result -- use NaN sentinels
    plus an explicit status/error so a crash can never be silently reported as '0% return'."""
    result = {k: float("nan") for k in BACKTEST_METRIC_KEYS}
    result["status"] = "failed"
    result["error"] = str(error)
    return result


def run_portfolio_backtest(
    pred_series: pd.Series,
    benchmark: str,
    topk: int = 50,
    n_drop: int = 5,
    annualization_n: int = 252,
    open_cost: float = 0.0001,
    close_cost: float = 0.0001,
    deal_price: str = "close",
    return_positions: bool = False,
) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Optional[Any]]]:
    """Run simulated backtest using TopkDropoutStrategy and compute both gross-of-cost and
    net-of-cost risk metrics. Any failure is surfaced explicitly (status="failed" + the
    original exception, plus a full traceback in the log) rather than masked as zero metrics,
    since a crashed backtest is not observationally a flat/zero-return strategy.

    NOTE on `annualization_n`: qlib's own `risk_analysis` default for freq="day" is a scaler of
    238 (a China A-share trading-day convention baked into `qlib.contrib.evaluate`), not the more
    common 252 US trading days. This function passes 252 explicitly since this pipeline trades a
    US universe -- which means every annualized figure here will differ (~5-6% higher on return,
    ~3% higher on IR) from anything qlib's built-in `PortAnaRecord`/workflow reports on the same
    underlying data. That's intentional, but don't directly diff the two without accounting for it.

    Parameters
    ----------
    open_cost, close_cost : float
        Additive parameters (defaults match the previously-hardcoded 0.0001/0.0001) so gate
        item #7's cost-stress test can call this function twice at different cost levels
        without duplicating the backtest wiring.
    deal_price : str
        Additive parameter (default "close", matching prior hardcoded behavior). Gate item
        #10's ``open_shift`` label variant passes "open" here so the backtest's execution
        price basis matches the variant's own label basis.
    return_positions : bool
        Additive parameter (default False preserves the prior return shape/behavior exactly).
        When True, also returns the raw ``positions_normal`` object from ``backtest_daily``
        (or None if the backtest failed before positions were computed) as a second return
        value, so gate item #7 can compute turnover from it without a second backtest call.
    """
    dt_index = pred_series.index.get_level_values("datetime")
    start_time = str(dt_index.min())[:10]
    end_time = str(dt_index.max())[:10]

    try:
        strategy = TopkDropoutStrategy(
            signal=pred_series,
            topk=topk,
            n_drop=n_drop,
            risk_degree=0.95,
        )
        with warnings.catch_warnings():
            # Benign, high-volume noise: with a real point-in-time universe (vs. the old
            # back-projected one), many symbols legitimately are not part of the tradable set on
            # any given day (not yet added / already removed), which makes qlib's internal
            # qlib/utils/index_data.py:492 hit `np.nanmean` on an empty/all-NaN slice for that
            # day -- np.nanmean's own documented, correct behavior is to return NaN with this
            # warning, not an error. Scoped narrowly to this exact message so unrelated warnings
            # (e.g. real data-quality issues) still surface.
            warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
            report_normal, positions_normal = backtest_daily(
                start_time=start_time,
                end_time=end_time,
                strategy=strategy,
                benchmark=benchmark,
                account=100000000,
                exchange_kwargs={
                    "limit_threshold": None,
                    "deal_price": deal_price,
                    "open_cost": open_cost,
                    "close_cost": close_cost,
                    "min_cost": 0,
                },
            )
        # Qlib's own convention (see qlib.workflow.record_temp.PortAnaRecord): "return" is
        # gross of trading cost (cost is added back into account earnings), so the excess
        # return vs. benchmark must be reported both without and with cost subtracted --
        # reporting only the without-cost number as "the" excess return/IR silently hides
        # the exact costs this script configures (open/close costs).
        excess_gross = report_normal["return"] - report_normal["bench"]
        excess_net = report_normal["return"] - report_normal["bench"] - report_normal["cost"]
        # Absolute (non-benchmark-relative) net portfolio return, for a true capital-at-risk
        # drawdown. `risk_analysis`'s max_drawdown on `excess_net` is the drawdown of the
        # *benchmark-relative* curve (e.g. underperformance vs. SPY) -- in a rising benchmark
        # that can look far smaller than the actual peak-to-trough loss in the account itself.
        absolute_net = report_normal["return"] - report_normal["cost"]

        # freq=None (alongside N=annualization_n) avoids risk_analysis's own
        # "freq will be ignored" UserWarning -- its default `freq="day"` is not None, so passing
        # N without also nulling freq makes it warn on every single call otherwise.
        analysis_gross = risk_analysis(excess_gross, N=annualization_n, freq=None)
        analysis_net = risk_analysis(excess_net, N=annualization_n, freq=None)
        # mode="product" here (vs. the "sum"/arithmetic default used above for the benchmark-
        # relative series) is deliberate: qlib's "sum" mode computes max_drawdown as the min of
        # cumsum(r) - cummax(cumsum(r)), an *arithmetic* drawdown of daily returns, not a real
        # compounded peak-to-trough percentage. For the account's own absolute capital curve we
        # want the actual "% of capital lost from peak", which is only correct under compounding.
        analysis_absolute_net = risk_analysis(absolute_net, N=annualization_n, freq=None, mode="product")

        result = {
            "annualized_return_gross": round(float(analysis_gross.loc["annualized_return", "risk"]), 5),
            "information_ratio_gross": round(float(analysis_gross.loc["information_ratio", "risk"]), 5),
            "annualized_return_net": round(float(analysis_net.loc["annualized_return", "risk"]), 5),
            "information_ratio_net": round(float(analysis_net.loc["information_ratio", "risk"]), 5),
            "max_relative_drawdown_net": round(float(analysis_net.loc["max_drawdown", "risk"]), 5),
            "max_absolute_drawdown_net": round(float(analysis_absolute_net.loc["max_drawdown", "risk"]), 5),
            # Compounded (CAGR) absolute return, computed the same "product" way as the drawdown
            # above -- previously this was discarded despite being computed, leaving every return
            # column in the table as benchmark-excess only, with no genuine absolute-return figure.
            "annualized_return_absolute_net": round(float(analysis_absolute_net.loc["annualized_return", "risk"]), 5),
            "status": "ok",
            "error": None,
        }
        if return_positions:
            return result, positions_normal
        return result
    except Exception as e:
        logger.exception(f"Backtest calculation error for benchmark={benchmark}, window=({start_time}, {end_time})")
        failed = _failed_backtest_result(e)
        if return_positions:
            return failed, None
        return failed


def build_dataset(
    args: Any,
    instruments: Union[str, List[str]],
    label_config: Optional[Tuple[List[str], List[str]]] = None,
) -> DatasetH:
    """Build the shared Alpha158 handler + DatasetH used by the whole pipeline.

    Extracted from ``train_ensemble.py``'s ``main()`` so diagnostic/CV scripts can build
    the identical handler/dataset without copy-pasting handler_kwargs or importing
    ``train_ensemble.py`` itself (which would recreate the circular import this module
    exists to avoid).

    Parameters
    ----------
    args : argparse.Namespace
        Must provide ``train_start``, ``train_end``, ``valid_start``, ``valid_end``,
        ``test_start``, ``test_end`` (all date strings).
    instruments : Union[str, List[str]]
        Universe name or explicit ticker list (see ``parse_instruments``).
    label_config : Optional[Tuple[List[str], List[str]]]
        Additive parameter (default None preserves the exact prior behavior: Alpha158's own
        ``get_label_config()`` default, i.e. ``Ref($close,-2)/Ref($close,-1) - 1``). Gate item
        #10's ``open_shift`` variant passes ``LABEL_VARIANTS["open_shift"]["label_config"]``
        here -- a ``label=`` kwarg override at this call site, NOT a new Alpha158 subclass
        (``Alpha158.__init__`` already does ``kwargs.pop("label", self.get_label_config())``).
    """
    handler_kwargs: Dict[str, Any] = {
        "start_time": args.train_start,
        "end_time": args.test_end,
        "fit_start_time": args.train_start,
        "fit_end_time": args.train_end,
        "instruments": instruments,
        # Alpha158's own class default is `infer_processors=[]` -- i.e. the 158
        # raw features are never sanitized (this matches Qlib's canonical
        # CSI300 benchmark config too; see .team-code/train_alpha158_lightgbm.md
        # section 2.2, where this was independently ruled out as the cause of
        # a *different* issue, score-collapse). Left at [], a handful of
        # ratio-style features (e.g. dividing by a rolling std that is exactly
        # 0) can genuinely be +-inf for some (date, symbol) rows. LightGBM's
        # histogram binning tolerates raw inf silently; XGBoost's DMatrix
        # construction validates against it and raises
        # ("Input data contains `inf` ... while `missing` is not set to
        # `inf`") -- it isn't wrong to do so, it's XGBoost surfacing a data
        # problem LightGBM was quietly absorbing. `ProcessInf` is Qlib's own
        # mechanism for exactly this (qlib/data/dataset/processor.py) --
        # already qlib's own default for the Alpha360 handler -- and replaces
        # +-inf with that date's cross-sectional mean of the finite values,
        # rather than leaving raw infinities for only some learners to choke
        # on. This does not reintroduce the score-collapse hypothesis already
        # ruled out: that was about CSZScoreNorm rescaling the *label*, this
        # is strictly a raw-feature inf/data-hygiene fix and touches neither
        # the label nor the feature scale.
        "infer_processors": [{"class": "ProcessInf", "kwargs": {}}],
    }
    if label_config is not None:
        handler_kwargs["label"] = label_config
    handler = Alpha158(**handler_kwargs)
    dataset = DatasetH(
        handler=handler,
        segments={
            "train": [args.train_start, args.train_end],
            "valid": [args.valid_start, args.valid_end],
            "test": [args.test_start, args.test_end],
        },
    )
    return dataset


def build_and_train_models(
    dataset: DatasetH,
    models_to_run: List[str],
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 50,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    num_threads: int = 16,
    seed: int = 42,
) -> Tuple[Dict[str, Any], Dict[str, pd.Series], Dict[str, str]]:
    """Train requested models on dataset and return model objects, test predictions, and the
    resolved library versions actually used (for reproducibility provenance).

    `seed` is threaded into every learner's native RNG controls so runs are reproducible --
    without it, LightGBM/XGBoost/CatBoost's stochastic row/column subsampling (and CatBoost's
    bootstrap) make every run produce different metrics with no way to reproduce a reported
    result. Note this is necessary but not sufficient for bit-for-bit reproducibility:
    - LightGBM also needs `deterministic=True` + a fixed row/col-wise strategy to be reproducible
      across different `--num_threads` values (set below).
    - CatBoost auto-selects GPU when one is visible (qlib's CatBoostModel.fit hardcodes this and
      cannot be overridden from here); GPU CatBoost is not bit-reproducible even with a fixed
      seed, so that leg's numbers may still drift run-to-run on a GPU machine.
    """
    trained_models = {}
    predictions = {}
    library_versions = {}

    # 1. LightGBM
    if "lgb" in models_to_run:
        logger.info("--> [1/3] Training LightGBM Model (LGBModel)...")
        start_t = time.time()
        import lightgbm

        library_versions["lightgbm"] = lightgbm.__version__
        lgb_model = LGBModel(
            loss="mse",
            colsample_bytree=0.88,
            subsample=0.88,
            learning_rate=learning_rate,
            max_depth=6,
            num_leaves=31,
            num_threads=num_threads,
            lambda_l1=0.1,
            lambda_l2=1.0,
            seed=seed,
            # Required alongside `seed` for run-to-run reproducibility, including across
            # different --num_threads values (LightGBM docs: bagging_fraction/feature subsampling
            # is only bit-reproducible with `deterministic=True` + a fixed row/col-wise mode).
            deterministic=True,
            force_row_wise=True,
        )
        lgb_model.fit(
            dataset,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
        trained_models["LightGBM"] = lgb_model
        predictions["LightGBM"] = lgb_model.predict(dataset, segment="test")
        logger.info(f"    LightGBM training completed in {time.time() - start_t:.2f}s")

    # 2. XGBoost
    if "xgb" in models_to_run:
        try:
            from qlib.contrib.model.xgboost import XGBModel
            import xgboost

            library_versions["xgboost"] = xgboost.__version__
            logger.info(
                "    [NOTE] XGBoost gets a fixed `seed` but (unlike the LightGBM leg) no equivalent "
                "cross-`nthread` determinism setting -- treat this leg as 'seeded' rather than "
                "'guaranteed bit-reproducible across --num_threads values'."
            )
            logger.info("--> [2/3] Training XGBoost Model (XGBModel)...")
            start_t = time.time()
            xgb_model = XGBModel(
                eval_metric="rmse",
                colsample_bytree=0.88,
                subsample=0.88,
                eta=learning_rate,
                max_depth=6,
                nthread=num_threads,
                seed=seed,
            )
            xgb_model.fit(
                dataset,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False,
            )
            trained_models["XGBoost"] = xgb_model
            predictions["XGBoost"] = xgb_model.predict(dataset, segment="test")
            logger.info(f"    XGBoost training completed in {time.time() - start_t:.2f}s")
        except ImportError:
            logger.warning("    [SKIP] 'xgboost' not installed. Install with: pip install xgboost")

    # 3. CatBoost
    if "cat" in models_to_run:
        try:
            from qlib.contrib.model.catboost_model import CatBoostModel
            import catboost
            from catboost.utils import get_gpu_device_count

            library_versions["catboost"] = catboost.__version__
            if get_gpu_device_count() > 0:
                logger.warning(
                    "    [WARN] A GPU is visible and qlib's CatBoostModel will train on it "
                    "(task_type='GPU' is hardcoded in qlib/contrib/model/catboost_model.py). "
                    "GPU CatBoost is NOT bit-reproducible even with a fixed --seed."
                )
            logger.info("--> [3/3] Training CatBoost Model (CatBoostModel)...")
            start_t = time.time()
            cat_model = CatBoostModel(
                loss="RMSE",
                learning_rate=learning_rate,
                max_depth=6,
                thread_count=num_threads,
                verbose_eval=False,
                random_seed=seed,
            )
            cat_model.fit(
                dataset,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False,
            )
            trained_models["CatBoost"] = cat_model
            predictions["CatBoost"] = cat_model.predict(dataset, segment="test")
            logger.info(f"    CatBoost training completed in {time.time() - start_t:.2f}s")
        except ImportError:
            logger.warning("    [SKIP] 'catboost' not installed. Install with: pip install catboost")

    return trained_models, predictions, library_versions


def _json_safe(obj):
    """Recursively replace NaN/Inf floats with None so `json.dump` produces valid RFC 8259 JSON.
    Python's `json` module happily emits the bare (non-standard) tokens NaN/Infinity by default,
    which round-trip through `json.load` but are rejected by most other JSON consumers (jq,
    JavaScript's JSON.parse, many typed schema validators) -- so a run with any failed backtest
    would otherwise silently produce an artifact that looks fine but isn't actually valid JSON.
    """
    if isinstance(obj, float):
        return None if (obj != obj or obj in (float("inf"), float("-inf"))) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def blend_predictions(predictions: Dict[str, pd.Series]) -> pd.Series:
    """
    Blend predictions using Cross-Sectional Z-Score Standardization + Equal-Weight Mean.
    Implements Qlib's canonical AverageEnsemble.
    """
    if len(predictions) == 1:
        return list(predictions.values())[0]

    # Convert dictionary of Series into a DataFrame mapping
    formatted_dict = {name: pd.DataFrame(s) for name, s in predictions.items()}
    ensemble_op = AverageEnsemble()
    blended = ensemble_op(formatted_dict)
    return blended
