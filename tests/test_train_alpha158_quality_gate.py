#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression-Test Twin for qlib/contrib/validation/score_quality.py
====================================================================
Follows the same pattern as ``tests/test_visualize_key_contracts.py``: real
payload shapes drawn from the actual incidents documented in
``.team-code/20260908-alpha158_training_audit_and_score_degeneracy_implementation_plan.md``,
not loosely-mocked stand-ins.

Four required fixtures (per the implementation plan's Part 3.4 / 3.7 step 4):

1. ``test_20260905_incident_fails_gate`` -- the 2026-09-05 shape: 232/908
   distinct scores on the single reported date, Rank IC = nan.
2. ``test_20260908_incident_fails_gate`` -- the 2026-09-08 shape: 7/903
   distinct scores, dominant value shared by 858/903 names, IR -2.018,
   max drawdown -44%, ffr = nan. Real numbers pulled directly from
   ``mlruns/306366047040812909/145fb75192024a80b0ad8e1f95c366f0``.
3. ``test_nan_rank_ic_alone_fails_gate`` -- a dedicated fixture isolating
   ONLY the fail-closed NaN-Rank-IC fix: every other check (distinctness,
   num_trees, portfolio metrics, IC) is healthy.
4. ``test_healthy_baseline_passes_gate`` -- a synthetic fixture mirroring the
   ``362c9610`` baseline's real metrics (pulled from
   ``mlruns/306366047040812909/362c96100cd34e468b0b302a70006469``), asserted
   to pass with zero hard failures.

Additional focused tests below cover the auxiliary checks (schema/integrity,
staleness, non-convergence warning, pinned-reference comparison) that are
part of the gate but not exercised end-to-end by the four incident fixtures
alone.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qlib.contrib.validation.score_quality import (
    ScoreQualityGateError,
    validate_score_quality,
    extract_ic_metrics,
    extract_portfolio_metrics,
    MIN_NUM_TREES,
    MIN_DISTINCT_FRACTION,
    MAX_DOMINANT_SHARE,
)


class _FakeBooster:
    """Minimal stand-in for a lightgbm.Booster exposing only num_trees()."""

    def __init__(self, num_trees: int):
        self._num_trees = num_trees

    def num_trees(self) -> int:
        return self._num_trees


class _FakeLGBModel:
    """Minimal stand-in for qlib.contrib.model.gbdt.LGBModel: `.model.num_trees()`."""

    def __init__(self, num_trees: int):
        self.model = _FakeBooster(num_trees)


class _FakeRecorder:
    """Minimal stand-in for qlib.workflow.recorder.Recorder: `.list_metrics()`."""

    def __init__(self, metrics: dict):
        self._metrics = metrics

    def list_metrics(self) -> dict:
        return dict(self._metrics)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _healthy_portfolio_metrics() -> dict:
    """Real 362c9610 (healthy baseline) PortAnaRecord numbers."""
    return {
        "information_ratio": 1.494665851055892,
        "annualized_return": 0.2971049171994215,
        "max_drawdown": -0.12497546292428685,
        "ffr": 1.0,
    }


def _healthy_ic_metrics() -> dict:
    """Real 362c9610 (healthy baseline) SigAnaRecord numbers."""
    return {"ic": 0.008004342501323756, "rank_ic": 0.010932338795831296}


def _make_scores_df(dates, n_names: int, distinctness_fn, seed: int = 7) -> pd.DataFrame:
    """
    Build a (date, symbol, score) table. `distinctness_fn(rng, n_names) -> np.ndarray`
    generates one date's score column so each fixture can control its own
    cross-sectional shape precisely.
    """
    rng = np.random.default_rng(seed)
    symbols = [f"SYM{i:04d}" for i in range(n_names)]
    rows = []
    for date in dates:
        scores = distinctness_fn(rng, n_names)
        for sym, sc in zip(symbols, scores):
            rows.append({"date": pd.Timestamp(date), "symbol": sym, "score": float(sc)})
    df = pd.DataFrame(rows)
    df["rank"] = df.groupby("date")["score"].rank(ascending=False, method="min").astype(int)
    df["percentile"] = (df.groupby("date")["score"].rank(pct=True, ascending=True) * 100.0).round(2)
    return df


def _degenerate_20260905_scores() -> np.ndarray:
    """232 distinct values spread (with heavy ties) across 908 names -- the 2026-09-05 shape."""
    def _fn(rng, n_names):
        assert n_names == 908
        n_distinct = 232
        distinct_values = rng.normal(0.0, 1e-8, n_distinct)
        # Distribute 908 names over 232 distinct values with uneven tie widths.
        assignment = rng.integers(0, n_distinct, size=n_names)
        return distinct_values[assignment]
    return _fn


def _degenerate_20260908_scores():
    """7 distinct values across 903 names, one value shared by 858 -- the 2026-09-08 shape."""
    def _fn(rng, n_names):
        assert n_names == 903
        dominant_value = -0.000156
        other_values = rng.normal(0.0, 1e-6, 6)
        counts = [858, 8, 8, 8, 7, 7, 7]  # 6 non-dominant groups, each count >= 1
        assert len(counts) == 7 and sum(counts) == n_names
        values = [dominant_value] + list(other_values)
        out = np.concatenate([np.full(c, v) for c, v in zip(counts, values)])
        rng.shuffle(out)
        return out
    return _fn


def _fully_dispersed_scores():
    """Effectively-unique float64 scores -- the healthy-model shape."""
    def _fn(rng, n_names):
        return rng.normal(0.0, 1.0, n_names)
    return _fn


# ---------------------------------------------------------------------------
# The four required fixtures
# ---------------------------------------------------------------------------

class TestFourRequiredFixtures:
    def test_20260905_incident_fails_gate(self):
        """232/908 distinct on the reported date; Rank IC = nan; weak/negative portfolio metrics."""
        scores_df = _make_scores_df(["2026-09-04"], n_names=908, distinctness_fn=_degenerate_20260905_scores())
        ic_metrics = {"ic": 0.0, "rank_ic": float("nan")}
        portfolio_metrics = {
            "information_ratio": 0.110,
            "annualized_return": 0.0154,
            "max_drawdown": -0.3344,
            "ffr": 1.0,
        }
        trained_model = _FakeLGBModel(num_trees=20)

        with pytest.raises(ScoreQualityGateError) as excinfo:
            validate_score_quality(
                scores_df, ic_metrics, portfolio_metrics, trained_model,
                num_boost_round_cap=20, as_of_date="2026-09-04",
            )

        report = excinfo.value.report
        assert not report.passed
        joined = "\n".join(report.failures)
        assert "Distinctness check" in joined
        assert "Rank IC is missing/NaN" in joined
        assert "information_ratio=0.1100" in joined
        assert "max_drawdown=-33.4400%" in joined
        assert "num_trees=20 is below the minimum floor" in joined

    def test_20260908_incident_fails_gate(self):
        """7/903 distinct, dominant 858/903, IR -2.018, MDD -44%, ffr=nan -- real mlruns numbers."""
        scores_df = _make_scores_df(["2026-09-04"], n_names=903, distinctness_fn=_degenerate_20260908_scores())
        ic_metrics = {"ic": 0.007178708157636706, "rank_ic": -0.0011313856415178381}
        portfolio_metrics = {
            "information_ratio": -2.017709435939294,
            "annualized_return": -0.23533292412757875,
            "max_drawdown": -0.44016551971435547,
            "ffr": float("nan"),
        }
        trained_model = _FakeLGBModel(num_trees=31)

        with pytest.raises(ScoreQualityGateError) as excinfo:
            validate_score_quality(
                scores_df, ic_metrics, portfolio_metrics, trained_model,
                num_boost_round_cap=1, as_of_date="2026-09-04",
            )

        report = excinfo.value.report
        assert not report.passed
        joined = "\n".join(report.failures)
        # Distinctness: dominant_share should be caught (858/903 ~= 0.9502)
        assert "Distinctness check" in joined
        assert report.details["distinctness"]["latest_dominant_share"] == pytest.approx(858 / 903, abs=1e-4)
        assert report.details["distinctness"]["latest_distinct_fraction"] == pytest.approx(7 / 903, abs=1e-4)
        # IC looks superficially plausible and must NOT independently trip the IC check
        assert "IC check" not in joined
        # Rank IC is negative but not NaN -- also must not trip the (NaN-only) IC check
        assert "num_trees=31 is below the minimum floor" in joined
        assert "information_ratio=-2.0177" in joined
        assert "max_drawdown=-44.0166%" in joined
        assert "ffr (fill rate) is missing/NaN" in joined

    def test_nan_rank_ic_alone_fails_gate(self):
        """Isolates ONLY the fail-closed NaN-Rank-IC fix: everything else is healthy."""
        scores_df = _make_scores_df(
            pd.bdate_range("2026-08-01", periods=10), n_names=500, distinctness_fn=_fully_dispersed_scores()
        )
        ic_metrics = {"ic": 0.012, "rank_ic": float("nan")}
        portfolio_metrics = _healthy_portfolio_metrics()
        trained_model = _FakeLGBModel(num_trees=150)

        with pytest.raises(ScoreQualityGateError) as excinfo:
            validate_score_quality(
                scores_df, ic_metrics, portfolio_metrics, trained_model,
                num_boost_round_cap=1000, as_of_date=scores_df["date"].max(),
            )

        report = excinfo.value.report
        assert not report.passed
        assert report.failures == [
            "IC check: Rank IC is missing/NaN (nan); missing or NaN Rank IC is treated as a hard fail, "
            "not 'nothing to check'."
        ]

    def test_healthy_baseline_passes_gate(self):
        """Mirrors the 362c9610 baseline's real metrics; must pass with zero hard failures."""
        scores_df = _make_scores_df(
            pd.bdate_range("2026-07-01", periods=15), n_names=900, distinctness_fn=_fully_dispersed_scores()
        )
        ic_metrics = _healthy_ic_metrics()
        portfolio_metrics = _healthy_portfolio_metrics()
        trained_model = _FakeLGBModel(num_trees=137)

        report = validate_score_quality(
            scores_df, ic_metrics, portfolio_metrics, trained_model,
            num_boost_round_cap=1000, as_of_date=scores_df["date"].max(),
        )

        assert report.passed
        assert report.failures == []
        assert report.details["num_trees"]["num_trees"] == 137


# ---------------------------------------------------------------------------
# Auxiliary checks -- schema/integrity, staleness, non-convergence warning,
# pinned-reference comparison, and metric-extraction helpers.
# ---------------------------------------------------------------------------

class TestAuxiliaryChecks:
    def _base_healthy_kwargs(self):
        scores_df = _make_scores_df(
            pd.bdate_range("2026-07-01", periods=10), n_names=400, distinctness_fn=_fully_dispersed_scores()
        )
        return dict(
            scores_df=scores_df,
            ic_metrics=_healthy_ic_metrics(),
            portfolio_metrics=_healthy_portfolio_metrics(),
            trained_model=_FakeLGBModel(num_trees=100),
            num_boost_round_cap=1000,
            as_of_date=scores_df["date"].max(),
        )

    def test_duplicate_date_symbol_key_fails_schema_check(self):
        kwargs = self._base_healthy_kwargs()
        dup_row = kwargs["scores_df"].iloc[[0]].copy()
        kwargs["scores_df"] = pd.concat([kwargs["scores_df"], dup_row], ignore_index=True)

        with pytest.raises(ScoreQualityGateError) as excinfo:
            validate_score_quality(**kwargs)
        assert any("duplicate (date, symbol) key" in f for f in excinfo.value.report.failures)

    def test_nan_score_fails_schema_check(self):
        kwargs = self._base_healthy_kwargs()
        df = kwargs["scores_df"].copy()
        df.loc[df.index[0], "score"] = float("nan")
        kwargs["scores_df"] = df

        with pytest.raises(ScoreQualityGateError) as excinfo:
            validate_score_quality(**kwargs)
        assert any("NaN score" in f for f in excinfo.value.report.failures)

    def test_stale_scores_fail_staleness_check(self):
        kwargs = self._base_healthy_kwargs()
        stale_as_of = kwargs["scores_df"]["date"].max() + pd.Timedelta(days=30)
        kwargs["as_of_date"] = stale_as_of

        with pytest.raises(ScoreQualityGateError) as excinfo:
            validate_score_quality(**kwargs)
        assert any("Staleness check" in f for f in excinfo.value.report.failures)

    def test_non_convergence_produces_warning_not_failure(self):
        """num_trees hitting the configured cap is a warning, not a hard fail, on its own."""
        kwargs = self._base_healthy_kwargs()
        kwargs["trained_model"] = _FakeLGBModel(num_trees=1000)
        kwargs["num_boost_round_cap"] = 1000

        report = validate_score_quality(**kwargs)
        assert report.passed
        assert any("non-convergence warning" in w for w in report.warnings)

    def test_below_min_trees_floor_fails_even_with_everything_else_healthy(self):
        kwargs = self._base_healthy_kwargs()
        kwargs["trained_model"] = _FakeLGBModel(num_trees=MIN_NUM_TREES - 1)

        with pytest.raises(ScoreQualityGateError) as excinfo:
            validate_score_quality(**kwargs)
        assert any("below the minimum floor" in f for f in excinfo.value.report.failures)

    def test_pinned_reference_none_is_skipped(self):
        """Check 3.2.5 must be inert until a caller explicitly supplies a pinned reference."""
        kwargs = self._base_healthy_kwargs()
        report = validate_score_quality(pinned_reference=None, **kwargs)
        assert report.passed
        assert report.details["pinned_reference"] == {"skipped": True}

    def test_pinned_reference_structural_break_fails(self):
        kwargs = self._base_healthy_kwargs()
        candidate_df = kwargs["scores_df"]
        # Build a reference with the SAME (date, symbol) keys but shuffled scores per date,
        # i.e. rank correlation with the candidate should collapse toward zero.
        rng = np.random.default_rng(99)
        reference_df = candidate_df.copy()
        reference_df["score"] = reference_df.groupby("date")["score"].transform(
            lambda s: rng.permutation(s.values)
        )

        with pytest.raises(ScoreQualityGateError) as excinfo:
            validate_score_quality(pinned_reference=reference_df, **kwargs)
        assert any("Pinned-reference check" in f for f in excinfo.value.report.failures)

    def test_pinned_reference_same_scores_passes(self):
        kwargs = self._base_healthy_kwargs()
        reference_df = kwargs["scores_df"].copy()  # identical -> perfect rank correlation

        report = validate_score_quality(pinned_reference=reference_df, **kwargs)
        assert report.passed
        assert report.details["pinned_reference"]["mean_rank_correlation"] == pytest.approx(1.0, abs=1e-9)

    def test_insufficient_reference_overlap_fails_closed(self):
        kwargs = self._base_healthy_kwargs()
        reference_df = kwargs["scores_df"][kwargs["scores_df"]["date"] == kwargs["scores_df"]["date"].min()].copy()
        # Shift symbols so there's zero (date, symbol) overlap at all except possibly none.
        reference_df["date"] = pd.Timestamp("2000-01-01")

        with pytest.raises(ScoreQualityGateError) as excinfo:
            validate_score_quality(pinned_reference=reference_df, **kwargs)
        assert any("overlapping date" in f for f in excinfo.value.report.failures)


class TestMetricExtractionHelpers:
    def test_extract_portfolio_metrics_reads_expected_keys(self):
        recorder = _FakeRecorder(
            {
                "1day.excess_return_with_cost.information_ratio": 1.5,
                "1day.excess_return_with_cost.annualized_return": 0.2,
                "1day.excess_return_with_cost.max_drawdown": -0.1,
                "1day.ffr": 1.0,
                "1day.excess_return_without_cost.information_ratio": 1.8,  # must NOT be picked up
            }
        )
        metrics = extract_portfolio_metrics(recorder)
        assert metrics == {
            "information_ratio": 1.5,
            "annualized_return": 0.2,
            "max_drawdown": -0.1,
            "ffr": 1.0,
        }

    def test_extract_portfolio_metrics_absent_key_is_none_not_omitted(self):
        recorder = _FakeRecorder({})
        metrics = extract_portfolio_metrics(recorder)
        assert set(metrics.keys()) == {"information_ratio", "annualized_return", "max_drawdown", "ffr"}
        assert all(v is None for v in metrics.values())

    def test_extract_ic_metrics_reads_sig_ana_record_keys(self):
        recorder = _FakeRecorder({"IC": 0.01, "Rank IC": 0.02, "ICIR": 0.5})
        metrics = extract_ic_metrics(recorder)
        assert metrics == {"ic": 0.01, "rank_ic": 0.02}

    def test_distinctness_thresholds_are_the_reviewed_values(self):
        """Pin the reviewed 0.95 / 0.02 thresholds so a future edit needs a deliberate change here too."""
        assert MIN_DISTINCT_FRACTION == 0.95
        assert MAX_DOMINANT_SHARE == 0.02
