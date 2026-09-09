# Architect Plan: Pre-Training Diagnostic Gate for the Ensemble Pipeline

**Request:** Produce an implementation plan (no code) for gating further GBDT ensemble model work
behind a 9-item diagnostic suite (team-finance's revised, priority-ordered list), following a
20-round smoke test that showed near-zero IC/RankIC and a team-finance review that found the
result is statistically indistinguishable from a linear-factor noise floor on this window.
**Repo Stack Found:** Python, qlib (region="us"), LightGBM/XGBoost/CatBoost via
`qlib.contrib.model.*`, `qlib.contrib.data.handler.Alpha158`, `qlib.contrib.strategy.TopkDropoutStrategy`,
`qlib.contrib.evaluate.{risk_analysis,backtest_daily}`, `qlib.model.ens.ensemble.AverageEnsemble`,
argparse CLI scripts under `scripts/`, `unittest`-based tests under `tests/`. **Correction (scipy):**
scipy is NOT a core dependency — `scipy<=1.15.3` in `pyproject.toml` sits under
`[project.optional-dependencies].docs` (pinned only for Sphinx builds), not the top-level
`dependencies` list. Separately, `qlib/data/ops.py:12` does an unconditional
`from scipy.stats import percentileofscore` in core runtime code — a pre-existing packaging gap
where qlib's own core already needs scipy at runtime without declaring it; it merely happens to be
present today via transitive installs or the docs/analysis extras. This affects item #2's stats
library choice — see Section 6/12. No CI job currently trains models (no GitHub Actions workflow
invokes `train_ensemble.py`), so this gate is a human-in-the-loop report, not an automated build
gate, unless the team later decides otherwise (see Open Questions).
**Constraints Applied:** Default engineering standards (`.team-code/requirements.md` is absent).

## 1. Problem Statement

`scripts/train_ensemble.py` reports near-zero IC/RankIC for LightGBM/XGBoost/CatBoost on a
20-round smoke test over the real point-in-time S&P 500 universe. team-finance ran an independent
linear-factor check on the same test window (2025-01-01 to 2026-09-04) and found that ALL of
1d/5d/21d reversal and 60d residual vol are themselves statistically indistinguishable from zero
(|t-stat| < 1.4) at this window's sample size (~420 days, ~488 names/day, noise floor ≈ 0.02 mean
RankIC) — while also independently confirming the Alpha158 default label
(`Ref($close,-2)/Ref($close,-1)-1`) is *currently* correctly synchronized with
`TopkDropoutStrategy`'s execution (`shift=1`, `deal_price="close"`). team-finance's verdict is
AGREE WITH CHANGES: before spending more effort on new models, run a 9-item diagnostic suite
(re-run at full training budget, statistical power framing, a linear baseline, a repeatable
lag/look-ahead audit, train/valid/test IC + dispersion, an IC-decay-by-horizon curve, a
turnover/cost-realism check, walk-forward CV, and a paired label/execution variant) that can
distinguish "this window/universe has too little signal to detect" from "the model/pipeline is
broken" from "the label horizon is miscalibrated."

**Active gate items: #1-#8 and #10 (nine items total).** Item #9 (sector/month attribution) is
explicitly deferred and is NOT part of this PR — see Non-Goals below. Any table or diagram in this
plan that lists items as `1,2,3,4,5,6,7,8,10` (skipping 9) is correct as written, not a typo.

## 2. Non-Goals

- Do not implement or run anything in this pass — this is a plan only.
- Do not implement gate item #9 (sector/month attribution) — explicitly deferred: this qlib US
  data bundle carries no GICS sector mapping, so it is not runnable without a new external
  data source. Track it as a future backlog item, not part of this gate.
- Do not change `Alpha158.get_label_config()` or any other shared/core `qlib/contrib/data/handler.py`
  default — item #10's open-price variant is a `label=` kwarg override at one call site (no new
  subclass, see Section 3/4/7), never a silent replacement of the current close-based default.
- Do not build a fully generalized walk-forward-CV framework for all of qlib; scope item #8 to
  this pipeline's own `DatasetH`/direct-object training style (see Section 5/8 tradeoff on
  `RollingGen`).
- Do not wire this gate into CI/GitHub Actions in this pass — no existing pipeline trains models
  in CI, so "gate" here means a human-readable report + JSON artifact, not a build-blocking check
  (flagged as an open question in Section 10, not decided by this plan).
- Do not weaken, remove, or bypass `audit_universe_bias()`'s existing survivorship-bias check —
  the new diagnostics are additive to the existing fail-loud pattern, not a replacement for it.

## 3. Proposed Architecture & Modules

**Module layout correction:** the first draft placed `assert_sufficient_training_budget()` in both
`diagnose_signal.py` (Section 3) and `train_ensemble.py` (Section 4), and had `train_ensemble.py`
call into `diagnose_signal.run_diagnostics()` while `diagnose_signal.py` would need to import
`build_dataset()`/`calc_ic_metrics()`/`run_portfolio_backtest()` back out of `train_ensemble.py` —
a circular import (`train_ensemble → diagnose_signal → train_ensemble`). Fixed by extracting all
shared, dependency-free logic into a new **`scripts/ensemble_lib.py`**, so `train_ensemble.py` and
`diagnose_signal.py` both import downward from it and never from each other:

| Module / File | Responsibility | Owner |
| :--- | :--- | :--- |
| `scripts/ensemble_lib.py` (new) | Shared library, imported by all three scripts below, imports none of them: `parse_instruments`, `audit_universe_bias`, `build_dataset()` (newly extracted), `calc_ic_metrics()` (refactored to optionally expose its daily series), `run_portfolio_backtest()` (generalized `open_cost`/`close_cost` params), `build_and_train_models()`, `blend_predictions()`, `_json_safe()`, `assert_sufficient_training_budget()` (#1 guard — lives here, its single location, since both `train_ensemble.py`'s `main()` and `scripts/walk_forward_cv.py`'s per-fold loop call it) | Principal Dev |
| `scripts/train_ensemble.py` (edit) | Becomes a thin CLI/orchestrator over `ensemble_lib`: argparse, `--diagnose`/`--allow_undertrained`/`--label_variant {default,open_shift}` flags, a `LABEL_VARIANTS` dict (Section 4) that overrides `Alpha158(label=...)` and `exchange_kwargs["deal_price"]` for `open_shift` — no new class — `dataset.prepare()`/`predict()` on `train`/`valid` segments when `--diagnose` is set, calls `ensemble_lib` for training/eval and `diagnose_signal.run_diagnostics()` (including for the `open_shift` run, with `price_field="open"`) for diagnostics, writes `diagnostics_report.json`, prints the gate summary table | Principal Dev |
| `scripts/diagnose_signal.py` (new) | Diagnostic library + standalone CLI, imports only from `ensemble_lib` (never from `train_ensemble.py`): `compute_ic_with_power()` (#2), `fetch_price_panel()` + `compute_linear_baseline_ic()` (#3), `audit_feature_label_lag()` (#4), `compute_segment_ic_and_dispersion()` (#5), `compute_ic_decay_curve()` (#6, `price_field`-aware), `compute_turnover_and_cost_stress()` (#7), `run_diagnostics()` orchestrator (`price_field`-aware, reused unchanged for item #10's variant run) | Principal Dev + Senior Dev |
| `scripts/walk_forward_cv.py` (new) | Item #8 only, imports only from `ensemble_lib`: purged-fold generator + per-fold `Alpha158`/`DatasetH` construction + repeated `build_and_train_models()`/`calc_ic_metrics()` calls, own `walk_forward_report.json` | Principal Dev |
| `qlib/contrib/data/handler.py` | **No changes, and no new subclass anywhere.** Item #10's label variant is a `label=` kwarg override at the `Alpha158(...)` call site inside `build_dataset()` — `Alpha158.__init__` (`qlib/contrib/data/handler.py:122`) already does `"label": kwargs.pop("label", self.get_label_config())`, so overriding `label=` needs no subclass at all (the earlier draft's `Alpha158OpenLabel` subclass/file is dropped from scope entirely). Contrast with the existing `Alpha158vwap` pattern, which IS a subclass meant for general reuse — item #10's override is a single call-site kwarg, an even smaller footprint | Principal Dev |
| `tests/test_ensemble_lib.py` (new) | Unit tests for `ensemble_lib.py`'s functions, including the refactored `calc_ic_metrics` return shape, `parse_instruments`, `_json_safe`, and the new `assert_sufficient_training_budget` guard | Senior Dev / QA |
| `tests/test_diagnose_signal.py` (new) | Unit tests for every pure function in `diagnose_signal.py` (synthetic data, no qlib data dir required) | Senior Dev / QA |
| `models/gbdt_ensemble/diagnostics_report.json` (new artifact) | Machine-readable gate result, sibling to existing `ensemble_metadata.json` | (generated) |
| `.team-code/walkthroughs/` | Versioned report once implementation actually happens (future pass) | (future) |

Resulting import direction (acyclic): `train_ensemble.py → ensemble_lib.py`;
`train_ensemble.py → diagnose_signal.py → ensemble_lib.py`; `walk_forward_cv.py → ensemble_lib.py`.
Neither `diagnose_signal.py` nor `walk_forward_cv.py` ever imports `train_ensemble.py`, and
`ensemble_lib.py` imports neither of them.

## 4. Public API Specification

```python
# scripts/ensemble_lib.py -- new module; shared library, imports nothing from the scripts below
def build_dataset(args: argparse.Namespace, instruments: Union[str, List[str]]) -> DatasetH: ...
    # Extracted from train_ensemble.py's main() so diagnose_signal.py's and walk_forward_cv.py's
    # standalone CLI modes can build the identical handler/dataset without copy-pasting
    # handler_kwargs or importing train_ensemble.py itself (would recreate the circular import).

def assert_sufficient_training_budget(
    num_boost_round: int, learning_rate: float, min_total_shrinkage: float = 5.0,
    allow_override: bool = False,
) -> None: ...
    # Gate item #1. Single home for this function (corrected -- the prior draft listed it in both
    # diagnose_signal.py and train_ensemble.py). Raises RuntimeError (mirrors audit_universe_bias's
    # fail-loud + override-flag pattern) unless num_boost_round * learning_rate >= min_total_shrinkage,
    # or allow_override=True (wired to a new --allow_undertrained CLI flag). Lives in ensemble_lib.py
    # because both train_ensemble.py's main() and walk_forward_cv.py's per-fold loop call it. Named
    # threshold is a placeholder for PM/team sign-off, not a physical law -- see Open Questions.

# calc_ic_metrics(), run_portfolio_backtest(), build_and_train_models(), blend_predictions(),
# parse_instruments(), audit_universe_bias(), _json_safe() also move here unchanged (or, for
# calc_ic_metrics/run_portfolio_backtest, additively extended -- see Section 5) from
# train_ensemble.py's current top level.

# scripts/diagnose_signal.py -- new module, all functions pure / qlib-read-only (no training);
# imports build_dataset/calc_ic_metrics/run_portfolio_backtest etc. from ensemble_lib, never from
# train_ensemble.py
def compute_ic_with_power(
    pred_series: pd.Series, label_series: pd.Series, confidence: float = 0.95,
) -> Dict[str, float]: ...
    # Returns {"IC","ICIR","Rank IC","Rank ICIR","Rank_IC_daily_std","n_days","Rank_IC_tstat",
    # "Rank_IC_pvalue","min_detectable_mean_RankIC"} using a one-sample t-test against zero on the
    # (reused, now-exposed) daily Rank IC series -- reuses ensemble_lib.calc_ic_metrics's groupby
    # logic rather than recomputing it. The exact library (scipy.stats.ttest_1samp vs. statsmodels)
    # is an open decision, NOT settled by this plan -- scipy is not a core qlib dependency as
    # previously (incorrectly) stated here; see Section 6/12.

def fetch_price_panel(
    instruments: Union[str, List[str]], start_time: str, end_time: str, fields: List[str],
    price_field: str = "close",
) -> pd.DataFrame: ...
    # Thin wrapper over qlib.data.D.features(), returns a (datetime, instrument)-MultiIndexed
    # frame matching pred_series's index shape. Shared by baseline factors (#3) and IC-decay (#6).
    # `price_field` ("close" default, "open" for the item #10 variant -- see Section 8) selects
    # which price series `$<price_field>` the factors/decay labels below are built from, so this
    # function is reusable as-is against the open-label variant rather than hardcoding `$close`.

def compute_linear_baseline_ic(
    price_panel: pd.DataFrame, label_series: pd.Series, windows: Tuple[int, ...] = (1, 5, 21),
    vol_window: int = 60,
) -> Dict[str, Dict[str, float]]: ...
    # Gate item #3. One row per factor ("reversal_1d","reversal_5d","reversal_21d","resid_vol_60d"),
    # each scored via compute_ic_with_power against the SAME label_series the GBDT models used.

def audit_feature_label_lag(
    handler: DataHandlerLP, strategy_shift: int, deal_price: str,
) -> Dict[str, Any]: ...
    # Gate item #4. Regex-parses Ref($field, N) out of handler.get_feature_config()[0] and
    # handler.get_label_config()[0]; asserts every feature Ref uses N >= 0 (backward-looking) and
    # the label's two Ref offsets, combined with strategy_shift/deal_price, reproduce the
    # entry/exit relationship team-finance verified by hand. Returns a dict with "aligned": bool
    # and a human-readable "violations" list; raises nothing -- this is a report, not a guard,
    # by design (see Section 10).

def compute_segment_ic_and_dispersion(
    trained_models: Dict[str, Any], dataset: DatasetH, segments: Tuple[str, ...] = ("train", "valid", "test"),
) -> Dict[str, Dict[str, Any]]: ...
    # Gate item #5. Per model, per segment: compute_ic_with_power(...) plus
    # {"pred_dispersion_mean_daily_std", "pred_dispersion_frac_near_constant_days",
    # "best_iteration"} (best_iteration surfaced from each model's native API where available).

def compute_ic_decay_curve(
    predictions: Dict[str, pd.Series], price_panel: pd.DataFrame, horizons: range = range(1, 11),
    price_field: str = "close",
) -> Dict[str, Dict[int, Dict[str, float]]]: ...
    # Gate item #6. Re-labels the SAME fixed test-segment predictions at h=1..10 trading days
    # (no retraining) and scores each via compute_ic_with_power. `price_field` must match whatever
    # `fetch_price_panel()` was called with (i.e. "close" for the default run, "open" for the item
    # #10 variant) so the re-labeling is built from the same price basis the model's own label used
    # -- never hardcode `$close` here, or item #10's re-run of this function would silently score
    # against the wrong price series.

def compute_turnover_and_cost_stress(
    pred_series: pd.Series, benchmark: str, baseline_bp: float = 1.0, stress_bp: float = 10.0,
) -> Dict[str, Any]: ...
    # Gate item #7. Consumes `positions_normal` (currently discarded by run_portfolio_backtest) for
    # turnover, and calls a generalized run_portfolio_backtest(..., open_cost=, close_cost=) twice
    # (baseline vs. stress bp) to compare information_ratio_net degradation.

def run_diagnostics(
    dataset: DatasetH, trained_models: Dict[str, Any], predictions: Dict[str, pd.Series],
    args: argparse.Namespace, price_field: str = "close",
) -> Dict[str, Any]: ...
    # Orchestrator train_ensemble.py calls when --diagnose is set -- for BOTH the default run and
    # (per item #10, see the train_ensemble.py block below) the open-label variant run, passing
    # price_field="open" in the latter case so its internal fetch_price_panel()/
    # compute_ic_decay_curve() calls score against the same price basis the variant's own label
    # used, not $close. Returns the full diagnostics_report.json payload, including a top-level
    # "gate_verdict" per item ("pass" | "fail" | "inconclusive" | "not_run", plus a one-line
    # rationale string).

# scripts/train_ensemble.py -- new --label_variant CLI surface (item #10; no new class/subclass --
# see Section 3/7 correction)
LABEL_VARIANTS = {
    "default": {
        "label_config": None,  # None -> use Alpha158's own get_label_config() default unchanged:
                                # (["Ref($close,-2)/Ref($close,-1) - 1"], ["LABEL0"])
        "deal_price": "close",
    },
    "open_shift": {
        "label_config": (["Ref($open,-2)/Ref($open,-1) - 1"], ["LABEL0"]),  # open(s+2)/open(s+1)-1
        "deal_price": "open",
    },
}
# Exact, complete change for --label_variant open_shift, nothing else:
#   1. Alpha158(..., label=LABEL_VARIANTS["open_shift"]["label_config"]) -- a kwarg override at the
#      call site inside build_dataset(), NOT a new Alpha158OpenLabel subclass/file. Alpha158.__init__
#      (qlib/contrib/data/handler.py:122) already does
#      `"label": kwargs.pop("label", self.get_label_config())`, so passing `label=` directly
#      overrides the default without any new class. Dropped from scope: the previously-planned
#      Alpha158OpenLabel subclass and its file.
#   2. exchange_kwargs["deal_price"] = LABEL_VARIANTS["open_shift"]["deal_price"] in the
#      backtest_daily(...) call inside ensemble_lib.run_portfolio_backtest() -- every other
#      exchange_kwargs entry (open_cost=0.0001, close_cost=0.0001, min_cost=0,
#      limit_threshold=None) is UNCHANGED.
#   3. Nothing else changes. In particular, TopkDropoutStrategy's `shift=1` is NOT a variant knob
#      and cannot be exposed as one: qlib/contrib/strategy/signal_strategy.py:141-142 and :351-352
#      hardcode `self.trade_calendar.get_step_time(trade_step, shift=1)` as a literal inside
#      `generate_trade_decision`'s method body -- it is not a constructor parameter on
#      TopkDropoutStrategy or BaseSignalStrategy (neither __init__ exposes it). State this so an
#      implementer does not go looking for a `shift=` kwarg that does not exist.
# Rejected alternative -- name it explicitly so it is not independently "discovered" later:
#   Ref($open,-1)/$close - 1 (i.e. open(s+1)/close(s) - 1) is REJECTED. It is anchored at close(s)
#   -- the day the features are computed -- but with shift=1 fixed, the earliest price the strategy
#   can ever actually transact at is open(s+1); it can never enter at close(s). This expression is
#   therefore look-ahead/non-executable in exactly the way the earlier-rejected
#   Ref($close,-1)/$close-1 "naive fix" was rejected: it would inflate IC with a return the backtest
#   cannot realize. Ref($open,-2)/Ref($open,-1)-1 + deal_price="open" is the only pairing where the
#   label's return interval (open(s+1) -> open(s+2)) matches what shift=1 execution can actually
#   achieve.

# scripts/walk_forward_cv.py -- new script, item #8 only; imports build_and_train_models/
# calc_ic_metrics/assert_sufficient_training_budget from ensemble_lib, never from train_ensemble.py
def generate_purged_folds(
    start: str, end: str, train_years: int, test_months: int, step_months: int, embargo_days: int,
) -> List[Dict[str, Tuple[str, str]]]: ...
    # Returns each fold's (train_start, train_end, valid_start, valid_end, test_start, test_end)
    # date strings only -- NOT dataset/handler objects. `embargo_days` truncates train/valid
    # backward from each fold's test start (label-horizon leakage control, per Lopez de Prado --
    # see Section 6). This is a separate, ALSO-required control from the per-fold refit below, not
    # a substitute for it.

def build_fold_dataset(fold: Dict[str, Tuple[str, str]], args: argparse.Namespace) -> DatasetH: ...
    # HARD REQUIREMENT (see Section 5, "Per-Fold Refit Invariant"): constructs a BRAND NEW
    # `Alpha158(..., fit_start_time=fold["train_start"], fit_end_time=fold["train_end"], ...)`
    # handler and a brand new `DatasetH(handler=..., segments=...)` for THIS fold only. Never reuse
    # one `Alpha158`/`DatasetH` object across folds by re-slicing its `segments` while leaving
    # `fit_start_time`/`fit_end_time` pinned to the first fold (or the full sample) -- that would
    # fit any stateful `infer_processors`/`learn_processors` (e.g. `ZScoreNorm`, `RobustZScoreNorm`,
    # `MinMaxNorm` -- see `qlib/data/dataset/processor.py`) outside that fold's own train window,
    # a real look-ahead bug. Not a no-op with THIS pipeline's current processor list (Alpha158's own
    # `infer_processors=[]` default, plus stateless `CSZScoreNorm`/`DropnaLabel` in
    # `learn_processors`, none of which read `fit_start_time`/`fit_end_time`), but the plan must not
    # rely on that happenstance staying true -- `_DEFAULT_INFER_PROCESSORS` in the same file already
    # includes stateful `ZScoreNorm` and is one config change away from being used here too.

def run_walk_forward(args: argparse.Namespace) -> Dict[str, Any]: ...
    # For each fold: build_fold_dataset(fold, args) -> build_and_train_models(...) -> calc_ic_metrics(...).
    # Writes walk_forward_report.json.
```

## 5. Technical Risk Matrix

- **Concurrency:** None new — everything here is single-process, sequential, read-heavy analysis
  on already-computed pandas objects; no shared mutable state introduced beyond what
  `train_ensemble.py` already has.
- **Data Safety:** None — pure computation/reporting on existing dataset/model objects; no writes
  outside the existing `--output_dir` plus one new sibling script's own output dir.
- **Performance / Compute Cost:** Item #8 (walk-forward CV) is the one real scaling risk: N folds
  × 3 models × 300 boosting rounds × the full ~887-name, multi-year universe could run from tens
  of minutes to several hours depending on hardware and fold count — this must be sized and
  budgeted explicitly before running (see Section 10), and is why it is scoped to its own script
  and sequenced last/optional rather than folded into every `--diagnose` run.
- **Statistical Validity:** The `min_total_shrinkage` threshold in item #1's guard, the
  `confidence` level in item #2's power calc, and the exact definition of "60-day residual
  volatility" in item #3 (see Open Questions) are judgment calls that need explicit team/PM
  sign-off before implementation — the plan proposes defaults but they are not derived from a
  first-principles constraint.
- **Regression Risk on Shared Code:** `calc_ic_metrics()` (moving to `ensemble_lib.py`, see
  Section 3) needs to expose its internal daily Rank IC series (currently computed then discarded)
  without changing its existing return shape or behavior for the two existing call sites in
  `train_ensemble.py`'s `main()` — implement as an additive `return_daily_series: bool = False`
  parameter (or a new sibling function that `calc_ic_metrics` itself calls), never a breaking
  signature change, and add regression tests first since no test currently covers this function at
  all. The `ensemble_lib.py` extraction itself is a pure move-and-import-fix for this function and
  `run_portfolio_backtest()`/`build_and_train_models()` — no logic change — but should land as its
  own first commit, verified against the existing (pre-change) console output byte-for-byte, before
  any new diagnostic behavior is added on top.
- **Circular-Import Risk:** the module split in Section 3 (`ensemble_lib.py` as the sole shared
  base, with `train_ensemble.py` and `diagnose_signal.py` both importing downward from it and never
  from each other) is what avoids a `train_ensemble → diagnose_signal → train_ensemble` cycle;
  any future addition to either script must preserve that one-directional import rule rather than
  reaching back into `train_ensemble.py` for "just one more helper."
- **Per-Fold Refit Invariant (item #8, hard requirement):** `scripts/walk_forward_cv.py` MUST
  construct a brand-new `Alpha158(fit_start_time=fold_train_start, fit_end_time=fold_train_end, ...)`
  handler and a brand-new `DatasetH` for every fold (`build_fold_dataset()`, Section 4) — never one
  shared, globally-fit handler object re-sliced by segment across folds. A shared handler would fit
  any stateful `infer_processors`/`learn_processors` entry (`ZScoreNorm`, `RobustZScoreNorm`,
  `MinMaxNorm` in `qlib/data/dataset/processor.py`, all of which call
  `fetch_df_by_index(df, slice(self.fit_start_time, self.fit_end_time), ...)` in their `.fit()`)
  outside that fold's own train window — a real look-ahead bug, not cosmetic. This pipeline's
  current processors are stateless (`Alpha158.__init__`'s own `infer_processors=[]` default;
  `CSZScoreNorm`'s `__call__` is a pure per-day `groupby("datetime")` transform with no `.fit()`
  override; `DropnaLabel` is a stateless per-row filter) so a shared handler would not leak *today*
  — but `_DEFAULT_INFER_PROCESSORS` in that same module already includes stateful `ZScoreNorm`, so
  this must not be relied on as a permanent property of the pipeline. **This is separate from, and
  does not replace,** the `embargo_days` truncation in `generate_purged_folds()`: embargo prevents
  the 2-day label horizon from leaking across a fold's train/test *boundary*; per-fold handler
  construction prevents a stateful processor from being fit on data outside that fold's train
  *window* in the first place. Both controls are required; neither substitutes for the other.
  Note also that `RollingGen.trunc_days` (`qlib/workflow/task/gen.py`) does NOT solve this even if
  `RollingGen` were adopted later: `trunc_segments()` only truncates train/valid backward from each
  fold's test start (the same embargo concern above), and `RollingGen`'s default
  `ds_extra_mod_func=handler_mod` only extends the handler's `end_time` per fold — neither touches
  `fit_start_time`/`fit_end_time`. A `RollingGen`-based implementation would need a custom
  `ds_extra_mod_func` that also advances `fit_start_time`/`fit_end_time` per fold to be safe; the
  manual per-fold `Alpha158`/`DatasetH` construction in `build_fold_dataset()` is simpler to get
  right than retrofitting that hook, which is itself part of why Section 6 defers `RollingGen`
  adoption.

## 6. Program Manager Review & Approved Plan

| PM Recommendation | Decision (Keep / Drop) | Rationale & Industry Practice |
| :--- | :---: | :--- |
| Use a tested one-sample t-test primitive for the item #2 t-stat/p-value instead of hand-rolling `mean/std` arithmetic | Keep, library TBD | A one-sample t-test against zero is the standard way to frame "is this daily IC series distinguishable from noise" (Grinold & Kahn, *Active Portfolio Management*, information-ratio/noise-floor framing) — no reason to reimplement a tested primitive. **Correction:** the prior draft justified this with "scipy is already a core dependency," which is false — `scipy<=1.15.3` in `pyproject.toml` is pinned under `[project.optional-dependencies].docs` only, not the top-level `dependencies` list. Separately, `qlib/data/ops.py:12`'s unconditional `from scipy.stats import percentileofscore` means qlib's own core already has a de facto, undeclared runtime dependency on scipy — it is merely present today via transitive/extras installs, not because it is actually core. Two options, left as an explicit pre-implementation decision (Section 12): (a) formalize the existing de facto dependency by adding `scipy` to `pyproject.toml`'s core `dependencies` as part of this change (defensible, since `qlib/data/ops.py` already needs it unconditionally), or (b) use `statsmodels` instead (already present in the `dev` and `analysis` optional-dependency groups, though also not core) to avoid touching core deps, while leaving scipy's undeclared use elsewhere in qlib unresolved. |
| Extract shared logic (`calc_ic_metrics`, `build_dataset`, `run_portfolio_backtest`, `build_and_train_models`, `assert_sufficient_training_budget`, etc.) into a dependency-free `ensemble_lib.py` rather than having the diagnostic scripts import back into `train_ensemble.py` | Keep | Standard layering to avoid a circular import (`train_ensemble → diagnose_signal → train_ensemble`, present in the prior draft) — treats each CLI script (`train_ensemble.py`, `diagnose_signal.py`, `walk_forward_cv.py`) as a thin entry point over one shared library module, consistent with this project's own "Small Public APIs" / "Clean Architecture" standards; a plain dependency-inversion fix, not a novel pattern. |
| Implement item #10's open-price label as a `label=` kwarg override at the `Alpha158(...)` call site inside `build_dataset()`, not a new `Alpha158OpenLabel` subclass | Keep | `Alpha158.__init__` (`qlib/contrib/data/handler.py:122`) already does `"label": kwargs.pop("label", self.get_label_config())` — the override is a one-line kwarg, so a subclass/new file adds a public class with no behavior a kwarg can't already express; drops a whole file from scope. Also confirms `TopkDropoutStrategy`'s `shift=1` (`qlib/contrib/strategy/signal_strategy.py:141-142`,`:351-352`) is a hardcoded literal inside `generate_trade_decision`, not a constructor parameter on `TopkDropoutStrategy`/`BaseSignalStrategy` — it is NOT a variant knob and must not be exposed as one. |
| Run the full #2-#7 diagnostic suite (`run_diagnostics()`) on item #10's `open_shift` variant, not a hand-diff of its `ensemble_metadata.json` against the default run | Keep | The prior draft's "compared by hand" is exactly the eyeball-point-estimate failure mode item #2's noise-floor framing exists to prevent everywhere else in this gate — the variant's own `Rank_IC_tstat`/noise-floor read is what actually answers "did this help," not a raw-number diff. Requires `fetch_price_panel()`/`compute_ic_decay_curve()` to take a `price_field` parameter (Section 4) instead of hardcoding `$close`, so they're correctly reusable against `$open` for this run. |
| Frame item #8 explicitly as **purged/embargoed** walk-forward CV (Lopez de Prado, *Advances in Financial Machine Learning*), not plain rolling CV | Keep | The label horizon (2 days) means naive fold boundaries leak label information across the train/test split; an explicit `embargo_days` parameter names and fixes this rather than leaving it implicit. This is the kind of "recognized industry practice, named, with tradeoff documented" the Plan-Change Rule requires. |
| Evaluate adopting qlib's own `RollingGen` / `MultiHorizonGenBase` (`qlib/workflow/task/gen.py`) for items #6/#8 instead of hand-written loops | Keep (evaluate, decision deferred to implementation time) | These already exist in-repo and implement rolling-window generation with fold-boundary embargo truncation (`trunc_days`) and multi-horizon label task generation (`label_leak_n`) — don't reinvent them blind. **But**: they are built around qlib's task-dict + `Recorder`/`Experiment` execution model, not the direct `DatasetH`/model-object style `train_ensemble.py` already uses; adopting them means a materially bigger refactor than this diagnostic task needs. **Correction:** `trunc_days`/`trunc_segments()` only truncates train/valid backward from each fold's test start (embargo, for label-horizon leakage across the fold boundary) — it does NOT solve, and was never claimed here to solve, the separate per-fold-handler-refit requirement in Section 5 (`fit_start_time`/`fit_end_time` must also advance per fold for any stateful processor to be fit correctly); `RollingGen`'s default `ds_extra_mod_func=handler_mod` likewise only extends the handler's `end_time` per fold and does not touch `fit_start_time`/`fit_end_time` either. Adopting `RollingGen` later would still require a custom `ds_extra_mod_func` for that. Recommendation: start with a lightweight custom loop (`generate_purged_folds()` for embargo + `build_fold_dataset()` for per-fold handler construction, Section 4) reusing `build_and_train_models()`/`calc_ic_metrics()` as-is; revisit consolidating onto `RollingGen` (with a custom `ds_extra_mod_func`) only if walk-forward CV becomes a permanent, recurring pipeline stage rather than a one-off diagnostic. |
| Surface each GBDT model's `best_iteration` (early-stopping stop point) alongside item #5's train IC | Keep | Standard GBDT diagnostic practice: a model that stops at iteration ~5 of 300 is itself evidence "no exploitable train-set relationship was found," independent of the IC arithmetic — cheap to add (LightGBM/XGBoost/CatBoost all expose this natively) and materially sharpens the underfit-vs-overfit read. |
| Add a hard `pytest`/CI gate that fails the build if any diagnostic item reports "fail" | Drop (for now) | No existing CI workflow trains models at all (verified: no GitHub Actions job invokes `train_ensemble.py`); wiring a training-dependent gate into CI is a much larger, separate infrastructure change (self-hosted runner with a real qlib data bundle, multi-hour job budget) that this diagnostic-reporting task should not silently absorb. Keep this as a human-read JSON + console report; revisit as its own proposal if/when model retraining becomes a recurring, automatable job. |
| Fold sector/month attribution (item #9) into this pass using free SIC-code proxies scraped ad hoc | Drop | team-finance already scoped this out explicitly as deferred/not-runnable without a licensed sector-mapping source; substituting an ad hoc scrape would itself be a new, unaudited data-quality risk of exactly the kind `audit_universe_bias()` exists to catch elsewhere in this pipeline. Leave it deferred, as directed. |
| Compute item #3's "60-day residual volatility" as true CAPM-style residual-to-market volatility (rolling beta regression against SPY, then vol of residuals) | Drop (default to the simpler definition; flag as configurable) | A full rolling-beta residual calculation adds a second modeling step (its own window-length and estimator choices) to what is meant to be a *simple*, hard-to-argue-with linear baseline. Default to plain 60-day realized volatility of daily returns (or cross-sectionally demeaned daily returns, to strip out common market moves without a beta estimate) and expose the market-model version as a `--residual_vol_method {simple,capm}` flag for later use if team-finance specifically wants the CAPM variant. Named as an explicit open question, not silently decided (Section 10). |

## 7. Developer Briefs

* **Principal Dev Brief:** Own the core plumbing and every stateful/expensive path: create
  `scripts/ensemble_lib.py` and move `parse_instruments`, `audit_universe_bias`, `calc_ic_metrics`,
  `run_portfolio_backtest`, `build_and_train_models`, `blend_predictions`, `_json_safe` into it
  unchanged (pure move-and-import-fix commit, verified byte-for-byte against current console
  output before layering anything new on top — see Section 5); extract `build_dataset()` into the
  same module; implement `assert_sufficient_training_budget()` (item #1) there too, using the
  existing `audit_universe_bias` fail-loud/override pattern as a template; implement
  `scripts/walk_forward_cv.py` in full (item #8), including the purged-fold generator, the
  per-fold `build_fold_dataset()` refit (Section 5 invariant), and the build-vs-`RollingGen`
  decision from Section 6, importing only from `ensemble_lib.py`; implement item #10's
  `LABEL_VARIANTS`/`--label_variant {default,open_shift}` CLI wiring in `train_ensemble.py`
  (Section 4) — a `label=` kwarg override at the `Alpha158(...)` call site inside `build_dataset()`
  plus the matched `exchange_kwargs["deal_price"]` override in `run_portfolio_backtest()`; **no
  new class** (the earlier-planned `Alpha158OpenLabel` subclass is dropped from scope); confirm
  `TopkDropoutStrategy`'s hardcoded `shift=1` (`qlib/contrib/strategy/signal_strategy.py:141-142`,
  `:351-352`) is left untouched and is not exposed as a variant knob; generalize
  `run_portfolio_backtest()` to accept `open_cost`/`close_cost` parameters (needed by item #7)
  without changing its existing default behavior for current callers.
* **Senior Dev Brief:** Own `scripts/diagnose_signal.py`'s pure-function library and its tests —
  importing from `ensemble_lib.py` only, never from `train_ensemble.py`:
  `compute_ic_with_power()` (build on `ensemble_lib.calc_ic_metrics`, wire in whichever one-sample
  t-test primitive is decided per Section 6/12), `fetch_price_panel()` + `compute_linear_baseline_ic()`
  (item #3, both `price_field`-parameterized per Section 4 so item #10 can reuse them against
  `$open` rather than hardcoding `$close`), `audit_feature_label_lag()` (item #4, including unit
  tests with a deliberately-misaligned mock handler config to prove the check actually fails when
  it should), `compute_segment_ic_and_dispersion()` (item #5, plus surfacing `best_iteration` per
  PM's recommendation), `compute_ic_decay_curve()` (item #6, also `price_field`-parameterized), and
  `compute_turnover_and_cost_stress()`'s pure turnover math (item #7, consuming `positions_normal`).
  Ensure `run_diagnostics()` is called identically for item #10's `open_shift` run (only
  `price_field="open"` differs) so the variant gets the full #2-#7 gate treatment, not a hand-diff
  (Section 8/10). Write `tests/test_ensemble_lib.py` and `tests/test_diagnose_signal.py`
  (all-synthetic, no qlib data dir dependency) covering the refactored `calc_ic_metrics` shape and
  the new CLI flags/guard. Wire `diagnostics_report.json` output and the console "GATE SUMMARY"
  table (PASS/FAIL/INCONCLUSIVE/NOT-RUN per item + one-line rationale), matching the existing
  table's `TABLE_WIDTH`/NOTE-line style.

## 8. Per-Item Diagnostic Specification

*(Active items only: #1-#8 and #10 — nine items total. #9, sector/month attribution, is deferred
per Section 1/2 and intentionally absent from this table.)*

| # | Item | Input source | New computation | Output location | Human pass/fail read |
| :-: | :--- | :--- | :--- | :--- | :--- |
| 1 | Full 300-round rerun + budget guard | Existing CLI (`--num_boost_round` default is already 300) | `assert_sufficient_training_budget()` | Raises/warns at run start; recorded in `ensemble_metadata.json["args"]` (already captured) | Rerun did not raise / `--allow_undertrained` wasn't silently needed |
| 2 | Noise-floor framing | `test_label` + each model's `pred_series` (already in memory) | `compute_ic_with_power()` (refactor of `calc_ic_metrics`) | `diagnostics_report.json["ic_power"][model][segment]` + console gate row | `|Rank_IC_tstat| >= ~2` (95%) to call a reported RankICIR "detected, not noise" |
| 3 | Linear baseline factors | `fetch_price_panel()` via `D.features()` on the same universe/dates | `compute_linear_baseline_ic()` | `diagnostics_report.json["baseline_factors"]` + console table beneath model results | If baseline factors ALSO fail #2's threshold on this window → evidence points at window/universe, not model family |
| 4 | Feature/label lag audit | `handler.get_feature_config()`, `handler.get_label_config()`, strategy `shift`/`deal_price` constants | `audit_feature_label_lag()` (regex over `Ref($x, N)`) | `diagnostics_report.json["lag_audit"]` + console PASS/FAIL | `aligned: true` and empty `violations` list |
| 5 | Train/valid/test IC + dispersion | `dataset.prepare()` on all 3 segments + `trained_models` | `compute_segment_ic_and_dispersion()` | `diagnostics_report.json["segment_ic"][model][segment]` | train IC≈0 & test IC≈0 ⇒ underfit/no-signal-for-model; train IC≫test IC ⇒ overfit |
| 6 | IC decay curve h=1..10 | Fixed test-segment `predictions` (no retrain) + `fetch_price_panel()` | `compute_ic_decay_curve()` | `diagnostics_report.json["ic_decay"][model][h]` + console/optional chart | Any h with `|t-stat| >= ~2` and a coherent (non-noisy-looking) shape across neighboring h ⇒ label-horizon hypothesis worth pursuing |
| 7 | Turnover / cost realism | `positions_normal` (already computed, currently discarded) + generalized `run_portfolio_backtest()` | `compute_turnover_and_cost_stress()` | `diagnostics_report.json["turnover_cost"][model]` | IR sign flip or large % degradation under stress bp ⇒ any net signal is cost-fragile |
| 8 | Walk-forward / purged CV | Own, per-fold `Alpha158`/`DatasetH` built by `build_fold_dataset()` (never a shared, globally-fit handler — see Section 5 "Per-Fold Refit Invariant"); `build_and_train_models()` + `calc_ic_metrics()` reused per fold | `scripts/walk_forward_cv.py` (new): `generate_purged_folds()` for embargo + `build_fold_dataset()` for per-fold handler/dataset construction | Separate `walk_forward_report.json` | Consistent RankIC sign/magnitude and `|t-stat|` across folds ⇒ genuine (if small) signal vs. single-window artifact |
| 10 | Paired label/execution variant | `Alpha158(label=(["Ref($open,-2)/Ref($open,-1) - 1"], ["LABEL0"]))` kwarg override (no subclass) + `exchange_kwargs["deal_price"]="open"`; `shift=1` unchanged/not exposed | Full separate `train_ensemble.py --label_variant open_shift` run, THEN the full `run_diagnostics()` suite (#2-#7, with `price_field="open"`) on that run's own trained models/predictions | Separate `ensemble_metadata.json` + its own `diagnostics_report.json` in a distinct `--output_dir` — compared against the default run's `diagnostics_report.json`, not a hand-diff | Variant's own `Rank_IC_tstat`/noise-floor read (Section 6) clears the detection threshold where the default run's didn't ⇒ horizon-alignment hypothesis worth pursuing; equally indistinguishable from noise ⇒ rules it out too |

## 9. Sequencing & Dependencies

*(Same nine active items as Section 8: #1-#8 and #10. #9 is deferred and does not appear below.)*

```
Phase 0 (no training needed, run first/cheapest):
  #4 lag/look-ahead audit   -- pure static + config introspection
  #2 (utility only)         -- write compute_ic_with_power(), not yet applied to real data

Phase 1 (expensive, required before anything else is meaningful):
  #1  full 300-round rerun (lgb, xgb, cat)         --\
  #10 paired label/execution variant, full rerun    --+-- can run in parallel with each other
                                                        (independent output dirs)

Phase 2 (consumes Phase 1's trained models/predictions; can run concurrently with each other):
  #2 applied to #1's real predictions
  #3 baseline factors        (independent of #1's models; compare its own result against #1's once both exist)
  #5 train/valid/test IC + dispersion
  #6 IC decay curve
  #7 turnover / cost stress

Phase 3 (conditional -- only if Phase 2 does not already explain the flat result, given its cost):
  #8 walk-forward / purged CV (depends on #1's budget guard, #4's embargo-length correctness, AND
     Section 5's Per-Fold Refit Invariant: each fold gets its own freshly-fit Alpha158/DatasetH,
     never one shared handler re-sliced by segment)
```

Rationale: #1 is the explicit non-negotiable prerequisite team-finance named — the 20-round result
has total shrinkage ≈ 1.0 and cannot support any diagnosis. #4 and #2's utility function have zero
dependencies and are the cheapest possible first steps. #3 can start immediately in parallel with
#1's rerun (it never touches the trained models) but its *interpretation* ("model vs. universe")
needs #1's results to compare against. #10 is architecturally independent of #1-#8 but shares the
same "must be a full, properly-trained run" requirement, so it is scheduled alongside #1. #8 is
sequenced last both because it is the most expensive item and because team-finance's own ordering
places it after the cheaper single-split diagnostics that might already fully explain the result.

## 10. Reporting Format

- **Console:** Extend the existing summary table (same `TABLE_WIDTH`/`NOTE:`-line style already in
  `train_ensemble.py`) with a new "GATE SUMMARY" block printed after it when `--diagnose` is set:
  one row per active item (1,2,3,4,5,6,7,10 — 8 only when `walk_forward_cv.py` was also run),
  status in `{PASS, FAIL, INCONCLUSIVE, NOT_RUN}`, and a one-line rationale string, e.g.
  `[4] Lag/label audit .......... PASS   (0 forward-looking features; label/shift/deal_price aligned)`.
- **JSON artifact:** `diagnostics_report.json`, written next to `ensemble_metadata.json` in the
  same `--output_dir`, using the existing `_json_safe()` NaN-sanitization helper. Contains one
  top-level key per item plus a `"gate_verdict"` object of `{item: {status, rationale}}` mirroring
  the console block, so the same file is both human-readable (via `jq`) and script-consumable.
- **Item #8** gets its own `walk_forward_report.json` from `scripts/walk_forward_cv.py`, kept
  separate given its different execution model/cost rather than merged into the main artifact.
- **Item #10** is a second, complete `ensemble_metadata.json` **plus its own `diagnostics_report.json`**
  (from running the full `run_diagnostics()` suite, items #2-#7, on the variant's own trained
  models/predictions with `price_field="open"`) in its own `--output_dir`. The "did the open-price
  variant actually help" question is answered by comparing the two runs' `Rank_IC_tstat`/noise-floor
  reads against each other (Section 6/8) — not a raw-number hand-diff of point estimates, which was
  the prior draft's spec and is exactly the eyeball-comparison failure mode item #2 exists to
  prevent elsewhere in this gate. No automated diff tool is built in this pass (a future
  `scripts/compare_runs.py` could do that), but the comparison itself must use the same
  significance framing as every other item, not two numbers read side by side.
- No automated pass/fail exit code by default (see PM review, Section 6) — `--diagnose` reports;
  it does not `sys.exit(1)` on a failing item. This matches "human decides," per the request.

## 11. Effort / Complexity Estimate

| Item | Estimate | Notes |
| :-: | :--- | :--- |
| `ensemble_lib.py` extraction (prerequisite, not a numbered item) | S (~2-3 hrs) | Pure move-and-import-fix of `parse_instruments`/`audit_universe_bias`/`calc_ic_metrics`/`run_portfolio_backtest`/`build_and_train_models`/`blend_predictions`/`_json_safe` out of `train_ensemble.py`; must land first and be verified byte-for-byte against current output before any new diagnostic logic is added |
| #4 Lag/label audit | S (~2-4 hrs) | Regex parsing + assertion function + tests; zero training dependency |
| #1 Budget guard + rerun | S (~1-2 hrs dev) | Rerun itself is compute time (minutes-hours), not dev time |
| #2 Noise-floor utility | S-M (~3-5 hrs) | Additive refactor of `calc_ic_metrics`; scipy t-test wiring; tests |
| #3 Baseline factors | M (~1 dev-day) | New price-panel fetch + 4 factor definitions + train/test scoring + tests |
| #5 Train/valid IC + dispersion | M (~4-6 hrs) | Extra `prepare()`/`predict()` calls ×3 segments; dispersion + `best_iteration` |
| #6 IC decay curve | M (~1 dev-day) | Reuses #3's price-panel helper; 10 label variants × 3 models |
| #7 Turnover / cost stress | S-M (~4-6 hrs) | Consume existing-but-discarded `positions_normal`; generalize cost kwargs |
| #10 Label/execution variant | S (~3-5 hrs dev) | `LABEL_VARIANTS` dict + CLI flag + `label=`/`deal_price` kwarg wiring (no subclass) + tests; running the full `run_diagnostics()` suite on the variant is not extra dev cost (same code path, `price_field="open"`) but does add a second full training run's compute time |
| #8 Walk-forward / purged CV | L (~2-3 dev-days + open-ended compute time) | New script; embargo logic (`generate_purged_folds`) PLUS per-fold `Alpha158`/`DatasetH` construction (`build_fold_dataset`, see Section 5 invariant — not optional, not extra scope beyond what correctness requires); N-fold × 3-model × 300-round compute cost is the dominant risk, not the code |
| **Total (items 1-7, 10)** | **~5-7 dev-days** | Reasonable as a single implementation pass / PR |
| **Item #8** | **Separate milestone** | Scope, budget compute time, and possibly restrict to LightGBM-only or fewer folds before committing calendar time |

## 12. Risks & Open Questions

1. **Compute cost of #8** is unbounded until a fold count/step size is chosen — recommend sizing
   this (e.g., 6-8 folds, 12-month step, LightGBM-only first pass) before writing the script, not
   after.
2. **"60-day residual volatility" definition (#3)** — plain realized vol, cross-sectionally
   demeaned vol, or true CAPM residual-to-SPY vol? PM review defaults to the simplest (plain
   realized vol) with a `--residual_vol_method` escape hatch; needs team-finance confirmation.
3. **`min_total_shrinkage` threshold (#1)** and **confidence level (#2)** are placeholder defaults
   (5.0 and 95%) pending explicit team sign-off, not derived values.
4. **Early-stopping semantics complicate #5's "train IC":** all three models are fit with
   `early_stopping_rounds` against the `valid` segment, so a very small `best_iteration` is itself
   diagnostic (see Section 6 PM recommendation) — report it, don't let train-IC-alone drive the
   underfit/overfit call.
5. **Whether to adopt `qlib.workflow.task.gen.RollingGen`/`MultiHorizonGenBase` for #6/#8** — they
   exist and cover similar ground, but assume qlib's task-dict + Recorder/Experiment execution
   model, not this pipeline's direct `DatasetH`/model-object style; Section 6 recommends deferring
   that consolidation rather than forcing it into this pass.
6. **No existing tests cover `train_ensemble.py` or its functions at all** (`tests/test_train_alpha158.py`
   only covers the separate `train_alpha158.py` qrun-based script) — this plan's Senior Dev brief
   includes writing the first tests for `ensemble_lib.py` (post-extraction) and every new
   `diagnose_signal.py` function, which is incremental scope beyond the nine active diagnostic
   items (#1-#8, #10) themselves but necessary to keep the "no unweakened/deleted assertions" and
   regression-safety standards intact.
7. **Whether the gate should eventually block, not just report** — no CI currently trains models,
   so this plan treats the gate as advisory (console + JSON) rather than a build-failing check;
   confirm with the team whether a future automated retrain job should consume
   `diagnostics_report.json`'s `gate_verdict` programmatically (e.g., refuse to promote a new model
   if any item is `"fail"`).
8. **scipy vs. statsmodels for item #2's t-test (correction from the prior draft):** scipy is not
   a core qlib dependency (`scipy<=1.15.3` sits under `[project.optional-dependencies].docs`), even
   though `qlib/data/ops.py:12` already imports `scipy.stats` unconditionally in core runtime code —
   a pre-existing, undeclared dependency this plan did not introduce and is not scoped to fix
   wholesale. Before implementing `ensemble_lib`'s/`diagnose_signal`'s one-sample t-test, decide
   between (a) adding `scipy` to `pyproject.toml`'s core `dependencies` (formalizes the existing de
   facto need) or (b) using `statsmodels` (already in the `dev`/`analysis` extras, also not core,
   avoids a core-dependency change but leaves scipy's undeclared core usage elsewhere untouched).
   Either is defensible; this plan does not pick one.
