# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Production promotion gate for trained scoring models (LightGBM/Alpha158 today,
written generically enough to cover any regression model that emits a
cross-sectional (date, symbol) -> score table plus a Qlib recorder).

Why this module exists
-----------------------
Two incidents reached ``models/lightgbm/alpha158_russell1000_latest.*`` /
``output/scores/alpha158_russell1000_latest.*`` before this gate existed:

- 2026-09-05: over-regularized hyperparameters produced a single-leaf,
  zero-split tree. Every score was the same constant. Rank IC was ``nan``.
- 2026-09-08: ``--num_boost_round=1`` (a "quick test" CLI override) silently
  became the production run. The resulting single 31-leaf tree emitted only
  7 distinct scores across 903 names (858 names sharing one value). Headline
  IC (0.00718) looked superficially plausible; Rank IC (-0.00113) was
  sign-flipped versus every healthy historical run; the backtest could not
  fill the book (``ffr = nan``) and lost money at an information ratio of
  -2.018 with a -44% max drawdown.

Neither incident was caught because nothing ever branched on the metrics
Qlib's own ``SigAnaRecord``/``PortAnaRecord`` were already computing and
logging into ``mlruns/``. This module is that missing branch: a single
function, ``validate_score_quality``, that raises ``ScoreQualityGateError``
on any hard-fail condition and otherwise returns a ``ScoreQualityReport``.

Full design history, the independent ``@team-finance`` trading-desk-
credibility review, and the rationale behind every threshold below live in
``.team-code/20260908-alpha158_training_audit_and_score_degeneracy_implementation_plan.md``
(Part 3) and ``.team-code/validate_score_quality.md`` (this module's spec).

Design invariants
------------------
- **Fail closed, always.** A ``NaN``, a missing metric, or an exception while
  computing a check is a hard failure of that check -- never a skip, never a
  pass. This is the single most important rule in this module: the 2026-09-05
  incident's ``Rank IC = nan`` would have silently passed a naively written
  ``nan < floor`` comparison (Python evaluates that ``False``).
- **No silent partial promotion.** This module never writes any artifact. It
  only inspects in-memory evidence (a scores DataFrame, a metrics dict, a
  trained model, an optional pinned reference) and raises or returns. Callers
  are responsible for computing the complete evidence set *before* calling
  this function and for only writing production artifacts *after* it returns
  without raising -- see ``scripts/train_alpha158_lightgbm.py::train_alpha158_model``
  for the reference integration.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Thresholds -- named constants, not magic numbers. Every value here is
# derived and justified in .team-code/validate_score_quality.md; do not tune
# any of these to make a specific run pass without a documented, reviewed
# reason (per the endorsing @team-finance review, distinctness/portfolio
# checks are an explicit HARD STOP with no "legitimate market regime"
# exemption).
# ---------------------------------------------------------------------------

#: check 3.2.1 -- a healthy LightGBM regression ensemble emits effectively
#: unique float64 scores; n_distinct / n_names should sit near 1.0.
MIN_DISTINCT_FRACTION: float = 0.95

#: check 3.2.1 -- companion ceiling on the single most common value's share
#: of the cross-section on a given date.
MAX_DOMINANT_SHARE: float = 0.02

#: check 3.2.1 (revised scope) -- a date that fails the two thresholds above
#: is a "degenerate date". The 2026-09-08 incident was systemic (median
#: dominant_share 0.961 across ~420 dates), not a latest-date-only artifact,
#: so this gate hard-fails on the most recent date unconditionally, and
#: additionally hard-fails if more than this fraction of *all* dates in the
#: file are degenerate -- catching a systemic collapse even if the very last
#: date happens to look fine by chance.
MAX_DEGENERATE_DATE_FRACTION: float = 0.05

#: check 3.2.2 -- floor on the number of boosted trees actually produced.
#: Qlib's own LGBModel default is num_boost_round=1000 with
#: early_stopping_rounds=50; a healthy run terminates in the hundreds of
#: trees. 50 is comfortably below every observed healthy run locally while
#: being far above the 1-tree and 20-tree (single-leaf) incidents.
MIN_NUM_TREES: int = 50

#: check 3.2.3 -- portfolio metrics, sourced from PortAnaRecord/risk_analysis
#: output already logged to the recorder. Floors are derived from the
#: healthy-run population documented in the implementation plan (IR
#: +0.674..+1.495, ann. return +3.6%..+29.7%, max drawdown -7.1%..-14.2%)
#: versus the confirmed-bad run (IR -2.018, ann. return -23.5%, max drawdown
#: -44.0%, ffr=nan) -- zero distributional overlap in any of the four.
MIN_INFORMATION_RATIO: float = 0.5
MIN_ANNUALIZED_RETURN: float = 0.0  # strictly greater than zero
MIN_MAX_DRAWDOWN: float = -0.20  # max_drawdown must be >= -0.20 (i.e. no worse than -20%)
REQUIRED_FILL_RATE: float = 1.0  # ffr must equal this exactly; NaN is an automatic hard fail

#: which PortAnaRecord frequency / cost variant this gate reads. "with_cost"
#: is the conservative, capital-preservation-first choice an Institutional
#: Hedge Fund Manager actually underwrites (net of the exchange_kwargs cost
#: model), not the optimistic without-cost figure.
PORTFOLIO_METRICS_FREQ: str = "1day"
PORTFOLIO_METRICS_COST_VARIANT: str = "excess_return_with_cost"

#: check 3.2.5 -- minimum mean per-date cross-sectional Spearman rank
#: correlation between a candidate run's scores and the pinned reference's
#: scores, computed over overlapping (date, symbol) pairs. A legitimate
#: retrain sharing the same features/hyperparameters over materially
#: overlapping history stays highly rank-correlated with its predecessor
#: (observed empirically at >0.9 between same-config reruns of this
#: pipeline); a structural break (the failure mode this check exists to
#: catch) collapses toward zero. 0.5 is set well below the reproducible
#: same-config baseline and well above "near zero" so it actually
#: discriminates rather than rubber-stamping.
MIN_RANK_CORRELATION_VS_REFERENCE: float = 0.5
MIN_REFERENCE_OVERLAP_DATES: int = 5

#: check 3.3 P1 -- staleness. max(date) in the score file must fall within
#: this many business days of the run/as-of date.
MAX_STALENESS_BUSINESS_DAYS: int = 5

_REQUIRED_SCORE_COLUMNS: Tuple[str, ...] = ("date", "symbol", "score")


class ScoreQualityGateError(Exception):
    """
    Raised by ``validate_score_quality`` when one or more hard-fail checks
    trip. Carries the full ``ScoreQualityReport`` so callers can log/persist
    exactly which checks failed and why, rather than parsing the message.
    """

    def __init__(self, report: "ScoreQualityReport"):
        self.report = report
        header = f"Score quality gate FAILED ({len(report.failures)} hard failure(s)):"
        body = "\n".join(f"  - {f}" for f in report.failures)
        super().__init__(f"{header}\n{body}")


@dataclasses.dataclass
class ScoreQualityReport:
    """Full result of a ``validate_score_quality`` evaluation."""

    passed: bool
    failures: List[str] = dataclasses.field(default_factory=list)
    warnings: List[str] = dataclasses.field(default_factory=list)
    details: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "failures": list(self.failures),
            "warnings": list(self.warnings),
            "details": self.details,
        }


def _is_bad_number(x: Any) -> bool:
    """True if x is missing, not a real number, NaN, or +/-inf. Fail-closed helper."""
    if x is None:
        return True
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return True
    return math.isnan(xf) or math.isinf(xf)


# ---------------------------------------------------------------------------
# Check 1 -- cross-sectional distinctness / dominant-share (PRIMARY)
# ---------------------------------------------------------------------------

def _check_distinctness(scores_df: pd.DataFrame) -> Tuple[List[str], List[str], Dict[str, Any]]:
    failures: List[str] = []
    warnings: List[str] = []

    per_date = []
    for date, grp in scores_df.groupby("date"):
        n_names = len(grp)
        n_distinct = grp["score"].nunique(dropna=True)
        if n_names == 0:
            continue
        value_counts = grp["score"].value_counts(dropna=True)
        dominant_share = float(value_counts.max()) / n_names if len(value_counts) else 1.0
        distinct_fraction = n_distinct / n_names
        degenerate = (distinct_fraction < MIN_DISTINCT_FRACTION) or (dominant_share > MAX_DOMINANT_SHARE)
        per_date.append(
            {
                "date": date,
                "n_names": n_names,
                "n_distinct": int(n_distinct),
                "distinct_fraction": distinct_fraction,
                "dominant_share": dominant_share,
                "degenerate": degenerate,
            }
        )

    if not per_date:
        failures.append("Distinctness check: scores_df has no dates to evaluate (empty after grouping).")
        return failures, warnings, {"per_date": []}

    per_date.sort(key=lambda r: r["date"])
    total_dates = len(per_date)
    degenerate_dates = [r for r in per_date if r["degenerate"]]
    degenerate_fraction = len(degenerate_dates) / total_dates

    latest = per_date[-1]
    if latest["degenerate"]:
        failures.append(
            f"Distinctness check: most recent date {latest['date']} is degenerate "
            f"(distinct_fraction={latest['distinct_fraction']:.4f} < {MIN_DISTINCT_FRACTION} "
            f"or dominant_share={latest['dominant_share']:.4f} > {MAX_DOMINANT_SHARE})."
        )

    if degenerate_fraction > MAX_DEGENERATE_DATE_FRACTION:
        median_distinct = float(np.median([r["distinct_fraction"] for r in per_date]))
        median_dominant = float(np.median([r["dominant_share"] for r in per_date]))
        failures.append(
            f"Distinctness check: {len(degenerate_dates)}/{total_dates} dates "
            f"({degenerate_fraction:.1%}) are degenerate, exceeding the "
            f"{MAX_DEGENERATE_DATE_FRACTION:.0%} systemic-failure threshold "
            f"(median distinct_fraction={median_distinct:.4f}, median dominant_share={median_dominant:.4f})."
        )

    details = {
        "total_dates": total_dates,
        "degenerate_date_count": len(degenerate_dates),
        "degenerate_date_fraction": degenerate_fraction,
        "latest_date": latest["date"],
        "latest_distinct_fraction": latest["distinct_fraction"],
        "latest_dominant_share": latest["dominant_share"],
        "median_distinct_fraction": float(np.median([r["distinct_fraction"] for r in per_date])),
        "median_dominant_share": float(np.median([r["dominant_share"] for r in per_date])),
    }
    return failures, warnings, details


# ---------------------------------------------------------------------------
# Check 2 -- boosting-round / model-complexity sanity
# ---------------------------------------------------------------------------

def _get_num_trees(trained_model: Any) -> int:
    """
    Resolve the trained booster's tree count. Accepts either a Qlib
    ``LGBModel``-like wrapper (``.model.num_trees()``) or a raw LightGBM
    ``Booster`` (``.num_trees()``) directly. Fail-closed: any failure to
    resolve a tree count raises rather than silently reporting 0/skip, since
    "we could not determine how trained this model is" is itself a reason
    not to promote it.
    """
    booster = getattr(trained_model, "model", trained_model)
    if booster is None or not hasattr(booster, "num_trees"):
        raise ValueError(
            f"Could not resolve a LightGBM booster with num_trees() from trained_model "
            f"(type={type(trained_model).__name__}); refusing to treat this as 'passing'."
        )
    return int(booster.num_trees())


def _check_num_trees(trained_model: Any, num_boost_round_cap: Optional[int]) -> Tuple[List[str], List[str], Dict[str, Any]]:
    failures: List[str] = []
    warnings: List[str] = []

    try:
        num_trees = _get_num_trees(trained_model)
    except Exception as e:
        failures.append(f"Model-complexity check: could not determine num_trees(): {type(e).__name__}: {e}")
        return failures, warnings, {"num_trees": None}

    if num_trees < MIN_NUM_TREES:
        failures.append(
            f"Model-complexity check: num_trees={num_trees} is below the minimum floor of {MIN_NUM_TREES} "
            f"(a model this small cannot express meaningful cross-sectional dispersion)."
        )

    if num_boost_round_cap is not None and num_trees >= int(num_boost_round_cap):
        warnings.append(
            f"Model-complexity check: num_trees={num_trees} reached the configured num_boost_round cap "
            f"({num_boost_round_cap}) without early stopping ever triggering -- non-convergence warning, "
            f"not a hard failure. Consider raising num_boost_round."
        )

    return failures, warnings, {"num_trees": num_trees}


# ---------------------------------------------------------------------------
# Check 3 -- portfolio metrics (CO-PRIMARY)
# ---------------------------------------------------------------------------

def extract_portfolio_metrics(
    recorder: Any,
    freq: str = PORTFOLIO_METRICS_FREQ,
    cost_variant: str = PORTFOLIO_METRICS_COST_VARIANT,
) -> Dict[str, Optional[float]]:
    """
    Pull PortAnaRecord's already-logged information ratio / annualized
    return / max drawdown / fill rate straight from the live recorder's
    metrics (``recorder.list_metrics()``, backed by MLflow's
    ``run.data.metrics``) -- no new computation, no new recorder wiring, per
    the implementation plan's explicit instruction to reuse Qlib's existing
    mechanism rather than inventing a parallel one.

    A key that is genuinely absent from the recorder (e.g. the backtest
    could not run at all) is returned as ``None``, not omitted -- callers
    (``validate_score_quality``) must treat ``None`` as a hard fail, never
    as "nothing to check".
    """
    metrics = recorder.list_metrics() if hasattr(recorder, "list_metrics") else dict(recorder)
    prefix = f"{freq}.{cost_variant}"
    return {
        "information_ratio": metrics.get(f"{prefix}.information_ratio"),
        "annualized_return": metrics.get(f"{prefix}.annualized_return"),
        "max_drawdown": metrics.get(f"{prefix}.max_drawdown"),
        "ffr": metrics.get(f"{freq}.ffr"),
    }


def _check_portfolio_metrics(portfolio_metrics: Optional[Dict[str, Any]]) -> Tuple[List[str], List[str], Dict[str, Any]]:
    failures: List[str] = []
    warnings: List[str] = []
    portfolio_metrics = portfolio_metrics or {}

    def _get(key: str) -> Any:
        return portfolio_metrics.get(key)

    ir = _get("information_ratio")
    ann_ret = _get("annualized_return")
    mdd = _get("max_drawdown")
    ffr = _get("ffr")

    if _is_bad_number(ir):
        failures.append(f"Portfolio-metrics check: information_ratio is missing/NaN ({ir!r}); treated as hard fail.")
    elif float(ir) < MIN_INFORMATION_RATIO:
        failures.append(f"Portfolio-metrics check: information_ratio={float(ir):.4f} < floor {MIN_INFORMATION_RATIO}.")

    if _is_bad_number(ann_ret):
        failures.append(f"Portfolio-metrics check: annualized_return is missing/NaN ({ann_ret!r}); treated as hard fail.")
    elif float(ann_ret) <= MIN_ANNUALIZED_RETURN:
        failures.append(f"Portfolio-metrics check: annualized_return={float(ann_ret):.4%} is not > {MIN_ANNUALIZED_RETURN:.0%}.")

    if _is_bad_number(mdd):
        failures.append(f"Portfolio-metrics check: max_drawdown is missing/NaN ({mdd!r}); treated as hard fail.")
    elif float(mdd) < MIN_MAX_DRAWDOWN:
        failures.append(f"Portfolio-metrics check: max_drawdown={float(mdd):.4%} is worse than floor {MIN_MAX_DRAWDOWN:.0%}.")

    if _is_bad_number(ffr):
        failures.append(f"Portfolio-metrics check: ffr (fill rate) is missing/NaN ({ffr!r}); NaN/absent fill rate is an automatic hard fail (book could not be filled).")
    elif float(ffr) != REQUIRED_FILL_RATE:
        failures.append(f"Portfolio-metrics check: ffr={float(ffr)} != required {REQUIRED_FILL_RATE} exactly.")

    details = {"information_ratio": ir, "annualized_return": ann_ret, "max_drawdown": mdd, "ffr": ffr}
    return failures, warnings, details


# ---------------------------------------------------------------------------
# Check 4 -- IC / RankIC (secondary/corroborating)
# ---------------------------------------------------------------------------

def extract_ic_metrics(recorder: Any) -> Dict[str, Optional[float]]:
    """
    Pull SigAnaRecord's own logged ``IC`` / ``Rank IC`` (Qlib's canonical
    ``calc_ic``-based computation) directly from the recorder, per the
    implementation plan's consolidation instruction (1.4.2 / 3.2.4): the
    gate must be fed by a single source of truth, not the script's separate
    hand-rolled ``calculate_ic_metrics`` duplicate.
    """
    metrics = recorder.list_metrics() if hasattr(recorder, "list_metrics") else dict(recorder)
    return {
        "ic": metrics.get("IC"),
        "rank_ic": metrics.get("Rank IC"),
    }


def _check_ic_metrics(ic_metrics: Optional[Dict[str, Any]]) -> Tuple[List[str], List[str], Dict[str, Any]]:
    failures: List[str] = []
    warnings: List[str] = []
    ic_metrics = ic_metrics or {}

    ic = ic_metrics.get("ic")
    rank_ic = ic_metrics.get("rank_ic")

    if _is_bad_number(ic):
        failures.append(f"IC check: IC is missing/NaN ({ic!r}); missing or NaN IC is treated as a hard fail, not 'nothing to check'.")
    if _is_bad_number(rank_ic):
        failures.append(f"IC check: Rank IC is missing/NaN ({rank_ic!r}); missing or NaN Rank IC is treated as a hard fail, not 'nothing to check'.")

    return failures, warnings, {"ic": ic, "rank_ic": rank_ic}


# ---------------------------------------------------------------------------
# Additional checks required by the @team-finance review (3.3)
# ---------------------------------------------------------------------------

def _check_schema_integrity(scores_df: pd.DataFrame) -> Tuple[List[str], List[str], Dict[str, Any]]:
    failures: List[str] = []
    warnings: List[str] = []

    missing_cols = [c for c in _REQUIRED_SCORE_COLUMNS if c not in scores_df.columns]
    if missing_cols:
        failures.append(f"Schema check: scores_df is missing required column(s): {missing_cols}.")
        return failures, warnings, {"missing_columns": missing_cols}

    n_nan = int(scores_df["score"].isna().sum())
    n_inf = int(np.isinf(scores_df["score"].to_numpy(dtype=float, na_value=0.0)).sum())
    if n_nan > 0:
        failures.append(f"Schema check: {n_nan} row(s) have a NaN score.")
    if n_inf > 0:
        failures.append(f"Schema check: {n_inf} row(s) have an infinite score.")

    dup_mask = scores_df.duplicated(subset=["date", "symbol"], keep=False)
    n_dup = int(dup_mask.sum())
    if n_dup > 0:
        failures.append(f"Schema check: {n_dup} row(s) share a duplicate (date, symbol) key.")

    if "rank" in scores_df.columns:
        # rank 1 must be the best (highest) score within each date, per the
        # method="min" competition-ranking convention this pipeline uses.
        bad_rank_dates = []
        for date, grp in scores_df.groupby("date"):
            top_row_idx = grp["score"].idxmax()
            if grp.loc[top_row_idx, "rank"] != 1:
                bad_rank_dates.append(date)
        if bad_rank_dates:
            failures.append(
                f"Schema check: rank column inconsistent with score on {len(bad_rank_dates)} date(s) "
                f"(the highest-scoring row is not rank==1), e.g. {bad_rank_dates[:3]}."
            )

    if "percentile" in scores_df.columns:
        out_of_range = scores_df[(scores_df["percentile"] < 0.0) | (scores_df["percentile"] > 100.0)]
        if not out_of_range.empty:
            failures.append(f"Schema check: {len(out_of_range)} row(s) have a percentile outside [0, 100].")

    details = {"n_nan": n_nan, "n_inf": n_inf, "n_duplicate_keys": n_dup}
    return failures, warnings, details


def _check_staleness(
    scores_df: pd.DataFrame,
    as_of_date: Optional[Any],
    max_staleness_business_days: int = MAX_STALENESS_BUSINESS_DAYS,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    failures: List[str] = []
    warnings: List[str] = []

    if as_of_date is None:
        warnings.append("Staleness check: no as_of_date provided; skipped.")
        return failures, warnings, {"skipped": True}

    if "date" not in scores_df.columns or scores_df.empty:
        failures.append("Staleness check: scores_df has no 'date' column or is empty.")
        return failures, warnings, {"skipped": False}

    max_date = pd.Timestamp(scores_df["date"].max()).normalize()
    as_of = pd.Timestamp(as_of_date).normalize()

    if max_date > as_of:
        warnings.append(f"Staleness check: max(date)={max_date.date()} is after as_of_date={as_of.date()}.")

    business_days_stale = int(np.busday_count(max_date.date(), as_of.date()))
    if business_days_stale > max_staleness_business_days:
        failures.append(
            f"Staleness check: score file's most recent date ({max_date.date()}) is {business_days_stale} "
            f"business day(s) behind as_of_date ({as_of.date()}), exceeding the "
            f"{max_staleness_business_days}-business-day tolerance."
        )

    return failures, warnings, {"max_date": str(max_date.date()), "as_of_date": str(as_of.date()), "business_days_stale": business_days_stale}


def _check_pinned_reference(
    scores_df: pd.DataFrame,
    pinned_reference: Optional[pd.DataFrame],
    min_rank_correlation: float = MIN_RANK_CORRELATION_VS_REFERENCE,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Check 3.2.5, sequenced last. A no-op (returns no failures/warnings) when
    ``pinned_reference`` is ``None`` -- per the implementation plan, this
    check cannot be turned on until a clean, gate-passing run exists to
    bless as the pin. Passing ``pinned_reference`` is how a caller opts in.
    """
    failures: List[str] = []
    warnings: List[str] = []
    if pinned_reference is None:
        return failures, warnings, {"skipped": True}

    required = {"date", "symbol", "score"}
    if not required.issubset(pinned_reference.columns):
        failures.append(f"Pinned-reference check: reference is missing required column(s) {required - set(pinned_reference.columns)}.")
        return failures, warnings, {"skipped": False}

    merged = scores_df[["date", "symbol", "score"]].merge(
        pinned_reference[["date", "symbol", "score"]],
        on=["date", "symbol"],
        how="inner",
        suffixes=("_candidate", "_reference"),
    )

    overlap_dates = merged["date"].nunique()
    if overlap_dates < MIN_REFERENCE_OVERLAP_DATES:
        failures.append(
            f"Pinned-reference check: only {overlap_dates} overlapping date(s) between candidate and "
            f"pinned reference (< {MIN_REFERENCE_OVERLAP_DATES} required) -- insufficient evidence to "
            f"validate against the reference; treated as a hard fail rather than skipped."
        )
        return failures, warnings, {"overlap_dates": overlap_dates}

    daily_corrs = []
    for _, grp in merged.groupby("date"):
        if len(grp) < 3:
            continue
        corr = grp["score_candidate"].corr(grp["score_reference"], method="spearman")
        if not (corr is None or (isinstance(corr, float) and math.isnan(corr))):
            daily_corrs.append(float(corr))

    if not daily_corrs:
        failures.append("Pinned-reference check: could not compute any valid daily rank correlation against the reference.")
        return failures, warnings, {"overlap_dates": overlap_dates, "mean_rank_correlation": None}

    mean_corr = float(np.mean(daily_corrs))
    if mean_corr < min_rank_correlation:
        failures.append(
            f"Pinned-reference check: mean daily Spearman rank correlation against the pinned reference "
            f"is {mean_corr:.4f}, below the {min_rank_correlation} floor -- candidate looks like a "
            f"structural break from the blessed reference, not routine drift."
        )

    return failures, warnings, {"overlap_dates": overlap_dates, "mean_rank_correlation": mean_corr, "n_dates_correlated": len(daily_corrs)}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def validate_score_quality(
    scores_df: pd.DataFrame,
    ic_metrics: Dict[str, Any],
    portfolio_metrics: Dict[str, Any],
    trained_model: Any,
    *,
    num_boost_round_cap: Optional[int] = None,
    as_of_date: Optional[Any] = None,
    pinned_reference: Optional[pd.DataFrame] = None,
    min_rank_correlation: float = MIN_RANK_CORRELATION_VS_REFERENCE,
) -> ScoreQualityReport:
    """
    Evaluate every check in Part 3.2/3.3 of the implementation plan against a
    complete evidence set and raise ``ScoreQualityGateError`` if any hard
    failure is found. Intended to be called exactly once, after a model,
    its scores, and all metrics have been computed but *before* any
    production artifact is written -- see the module docstring's "Design
    invariants" and ``train_alpha158_lightgbm.py::train_alpha158_model`` for
    the reference call site.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Columns ``date``, ``symbol``, ``score`` (``rank``/``percentile``
        optional but validated for internal consistency if present).
    ic_metrics : dict
        ``{"ic": float, "rank_ic": float}`` -- source via
        ``extract_ic_metrics(recorder)``, not a hand-rolled duplicate.
    portfolio_metrics : dict
        ``{"information_ratio": float, "annualized_return": float,
        "max_drawdown": float, "ffr": float}`` -- source via
        ``extract_portfolio_metrics(recorder)``.
    trained_model : Any
        The fitted model object (Qlib ``LGBModel`` wrapper or raw booster).
    num_boost_round_cap : int, optional
        The configured ``num_boost_round`` the run was capped at, used only
        for the non-convergence *warning* (not a hard fail).
    as_of_date : date-like, optional
        The run/"today" date for the staleness check. Passing ``None`` skips
        staleness (with a warning), which is only appropriate for offline
        re-validation of a historical artifact, never for a live promotion.
    pinned_reference : pd.DataFrame, optional
        The blessed reference scores table (check 3.2.5). Leave ``None``
        until a clean run has been produced and explicitly blessed -- this
        check is sequenced last by design.

    Returns
    -------
    ScoreQualityReport
        Only returned when there are zero hard failures.

    Raises
    ------
    ScoreQualityGateError
        If any check hard-fails. The exception carries the full report.
    """
    all_failures: List[str] = []
    all_warnings: List[str] = []
    details: Dict[str, Any] = {}

    for name, fn, args in (
        ("distinctness", _check_distinctness, (scores_df,)),
        ("schema_integrity", _check_schema_integrity, (scores_df,)),
        ("staleness", _check_staleness, (scores_df, as_of_date)),
        ("num_trees", _check_num_trees, (trained_model, num_boost_round_cap)),
        ("portfolio_metrics", _check_portfolio_metrics, (portfolio_metrics,)),
        ("ic_metrics", _check_ic_metrics, (ic_metrics,)),
        ("pinned_reference", _check_pinned_reference, (scores_df, pinned_reference, min_rank_correlation)),
    ):
        try:
            failures, warnings, check_details = fn(*args)
        except Exception as e:  # fail-closed: an exception inside a check is itself a hard failure
            failures, warnings, check_details = (
                [f"{name} check raised {type(e).__name__}: {e}"],
                [],
                {"exception": str(e)},
            )
        all_failures.extend(failures)
        all_warnings.extend(warnings)
        details[name] = check_details

    report = ScoreQualityReport(
        passed=(len(all_failures) == 0),
        failures=all_failures,
        warnings=all_warnings,
        details=details,
    )
    if not report.passed:
        raise ScoreQualityGateError(report)
    return report
