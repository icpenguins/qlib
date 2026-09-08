# validate_score_quality Specification

## Purpose
`qlib/contrib/validation/score_quality.py::validate_score_quality()` is the
production-promotion gate for the Alpha158/LightGBM Russell1000 training
pipeline (and, by design, any future model that emits a cross-sectional
`(date, symbol) -> score` table plus a Qlib recorder). It is the concrete
implementation of Part 3 (post-`@team-finance`-review) of
`.team-code/20260908-alpha158_training_audit_and_score_degeneracy_implementation_plan.md`.

It exists because two incidents reached
`models/lightgbm/alpha158_russell1000_latest.*` /
`output/scores/alpha158_russell1000_latest.*` before this gate did:

- **2026-09-05**: over-regularized hyperparameters (CSI300-tuned `lambda_l1`/
  `lambda_l2` applied to a much smaller US universe) produced a single-leaf,
  zero-split tree. Every score was the same constant; Rank IC was `nan`.
- **2026-09-08**: `--num_boost_round=1` (a "quick test" CLI override) silently
  became the production run. The resulting single 31-leaf tree emitted 7
  distinct scores across 903 names (858 names sharing one value). Headline IC
  (0.00718) looked superficially plausible; Rank IC (-0.00113) was
  sign-flipped versus every healthy historical run; the backtest could not
  fill the book (`ffr = nan`) and lost money at an information ratio of
  -2.018 with a -44% max drawdown.

Neither incident was caught because nothing ever branched on the metrics
Qlib's own `SigAnaRecord`/`PortAnaRecord` were already computing and logging
into `mlruns/` on every run. This gate is that missing branch.

## Design history and review
Drafted by `team-code`, then submitted to `@team-finance` for an actual
(not simulated) trading-desk-credibility review. Verdict:
**ENDORSED-WITH-CHANGES**. All required changes from that review are folded
directly into this implementation (see the implementation plan's Part 3.2-3.4
for the full, itemized before/after). The headline changes the review forced:

1. Portfolio metrics (information ratio, annualized return, max drawdown,
   fill rate) were promoted from "not checked at all" to **co-primary** with
   distinctness -- they are the single strongest discriminator found (zero
   distributional overlap between the bad run and every healthy run), and the
   data already exists in `mlruns/` at zero marginal computation cost.
2. Distinctness thresholds were tightened from an original 0.30/0.20 pair
   (which left a dead zone wide enough to admit the next incident) to
   0.95/0.02 -- calibrated to where a genuinely healthy regression ensemble's
   float64 scores actually sit (`n_distinct/n_names ~= 1.0`,
   `dominant_share ~= 1/n`).
3. The pinned-reference comparison (check 5) was redesigned to compare
   against a deliberately-blessed, immutable reference artifact instead of
   "whatever currently occupies the production path" (which, un-redesigned,
   would have anchored the check to the already-corrupt 2026-09-08 artifact).
4. `num_trees() <= 1` was replaced with a `>= 50` floor (too permissive;
   only caught the exact incident already seen).
5. Mandatory fail-closed fixes: NaN must never silently pass a numeric
   comparison (`nan < floor` is `False` in Python); an absent metric must
   never be treated as "nothing to check".
6. Additional gaps the review required beyond re-ranking checks: the P0
   override-cannot-write-production control, atomic staged promotion with
   score-file versioning, fail-closed silent-failure fixes in the caller, and
   provenance stamping. All implemented -- see "Integration" below.

Unanimous **hard stop**, no exemption for "legitimate low-dispersion market
regimes" was the review's explicit position on every hard-fail check: a
market regime compresses returns, not the count of distinct float64 values a
regression model emits.

## Public API

```python
from qlib.contrib.validation.score_quality import (
    ScoreQualityGateError,
    ScoreQualityReport,
    validate_score_quality,
    extract_ic_metrics,
    extract_portfolio_metrics,
)

report = validate_score_quality(
    scores_df,            # DataFrame: date, symbol, score (rank/percentile optional)
    ic_metrics,           # {"ic": float, "rank_ic": float} -- from extract_ic_metrics(recorder)
    portfolio_metrics,    # {"information_ratio", "annualized_return", "max_drawdown", "ffr"}
                          # -- from extract_portfolio_metrics(recorder)
    trained_model,        # fitted LGBModel wrapper (or raw booster)
    num_boost_round_cap=1000,     # optional: configured num_boost_round, for the
                                   # non-convergence *warning* only (not a hard fail)
    as_of_date=date.today(),      # optional: run date, for the staleness check
    pinned_reference=None,        # optional: blessed reference scores DataFrame (check 5)
)
# raises ScoreQualityGateError (report attached) on any hard failure;
# otherwise returns a ScoreQualityReport with .passed / .failures / .warnings / .details
```

`ScoreQualityGateError.report` and a successful `validate_score_quality()`
return value are both `ScoreQualityReport` instances -- callers get the same
structured detail either way, they just differ in whether an exception was
raised.

## Checks implemented (thresholds are named constants in the module, not
magic numbers -- see the module docstring for full derivation of each)

| # | Check | Kind | Threshold | Fail-closed rule |
| :- | :--- | :--- | :--- | :--- |
| 3.2.1 | Cross-sectional distinctness / dominant-share, evaluated across **all** dates | PRIMARY | `n_distinct/n_names >= 0.95` and `dominant_share <= 0.02`, per date | Hard-fails if the **latest** date is degenerate, OR if more than **5%** of all dates are degenerate (systemic collapse) |
| 3.2.2 | Boosting-round / model-complexity | Hard floor + warning | `num_trees() >= 50` | Warning (not hard-fail) if `num_trees() >= num_boost_round_cap` (never early-stopped) |
| 3.2.3 | Portfolio metrics (IR / ann. return / max drawdown / fill rate), sourced from `PortAnaRecord`/`risk_analysis()` via `recorder.list_metrics()` | CO-PRIMARY | IR >= 0.5, ann. return > 0, max drawdown >= -0.20, `ffr == 1.0` exactly | Missing or NaN on ANY of the four is an automatic hard fail (NaN fill rate = book could not be filled) |
| 3.2.4 | IC / Rank IC, sourced from `SigAnaRecord`'s own logged metrics via `recorder.list_metrics()` | Secondary/corroborating | n/a (no magnitude threshold imposed -- see rationale below) | Missing or NaN IC or Rank IC is a hard fail |
| 3.2.5 | Rank correlation vs. a pinned, explicitly-blessed reference | Sequenced last, opt-in | mean daily Spearman rank correlation >= 0.5, requires >= 5 overlapping dates | `None` (no reference yet) = the check is a no-op; an explicitly-passed reference with insufficient overlap is a hard fail, never silently skipped |
| 3.3 P1 | Schema/integrity: no NaN/inf scores, no duplicate `(date, symbol)` keys, `rank`/`percentile` internally consistent with `score` | Hard | n/a | Any violation is a hard fail |
| 3.3 P1 | Staleness: `max(date)` in the score file must be within 5 business days of `as_of_date` | Hard (when `as_of_date` given) | 5 business days | `as_of_date=None` skips with a warning -- appropriate only for offline re-validation of a historical artifact, never a live promotion |

**Why IC/Rank IC carry no magnitude threshold**: the 2026-09-08 incident's IC
(0.00718) sat inside the range of every healthy historical run -- "IC alone
looks superficially plausible" is literally the finding that motivated
demoting IC/Rank IC to secondary in the first place. Inventing a magnitude or
sign threshold not specified by the reviewed design would itself be a
material change to that design; the two mandatory fixes the review actually
required (NaN-is-fail, absent-is-fail) are what's implemented. Distinctness
and portfolio metrics are what actually gate the two known incidents, and
they do so independently of any IC threshold.

## Integration (`scripts/train_alpha158_lightgbm.py`)

`train_alpha158_model()` was restructured (implementation plan 3.7 step 2)
so that:

1. `task_train()` runs first -- this already executes the workflow YAML's
   full `record` pipeline (`SignalRecord` -> `SigAnaRecord` -> `PortAnaRecord`),
   so IC/Rank IC and the portfolio backtest metrics are already computed and
   logged to the recorder by the time it returns. No new computation or
   recorder wiring was added for this gate.
2. The complete evidence set (trained model, `scores_df`, `ic_metrics_gate`
   via `extract_ic_metrics`, `portfolio_metrics_gate` via
   `extract_portfolio_metrics`) is assembled entirely in memory.
3. `validate_score_quality()` is called **exactly once** against that
   complete set.
4. Only if it does not raise (or the run is a non-enforced smoke-test
   override -- see below) does the write phase begin. Every artifact write
   goes through `_atomic_write_via()`: staged to a temp file in the
   destination directory, then `os.replace()`d into place -- so a crash or
   exception mid-write can never leave a production path partially written,
   and a failed gate leaves every production artifact (`.pkl`, `.txt`,
   `.json`, `.parquet`/`.csv`) completely untouched.
5. Score-file versioning was added under `output/scores/checkpoints/`,
   mirroring the pre-existing `.pkl` versioning under
   `models/lightgbm/checkpoints/` -- closing the gap where the artifact type
   both real incidents actually corrupted had no rollback path.
6. Provenance (git SHA, a sha256 hash of the fully-resolved task config, and
   the configured seed, or `"not_set"` if none) is stamped into
   `alpha158_russell1000_latest_meta.json` alongside the existing
   hyperparameters, plus the full `quality_gate` report.

### The `--num_boost_round` override is structurally incapable of reaching production
`_resolve_output_dirs()` unconditionally forces `models/lightgbm/_smoketest/`
and `output/scores/_smoketest/` whenever `num_boost_round` is passed on the
CLI -- it ignores any caller-supplied output directories. This closes, by
construction rather than by detection, the exact mechanism Part 2 of the
implementation plan confirmed caused the 2026-09-08 incident. A defensive
`assert` backs this up so a future refactor cannot silently regress it.

For such a run, the gate still executes and every failure is logged loudly
(`enforce_gate = not override_active`), but it does not raise -- a
deliberately tiny `--num_boost_round` for a fast dev-loop iteration is
*expected* to fail the `num_trees >= 50` floor, and since the run is already
confined to scratch paths, blocking the write too would defeat the flag's
purpose as a developer convenience (Part 1.3.1 of the implementation plan
explicitly preserves this use case).

### Blessing the pinned reference (check 3.2.5)
`bless_pinned_reference()` (invoked via `--bless_pinned_reference` on the
CLI, never automatically) copies a scores parquet file into
`models/lightgbm/pinned_reference/alpha158_russell1000_pinned_reference.parquet`
and writes a companion `_meta.json` recording the source recorder id,
provenance, and the source run's own recorded `quality_gate` result. It
**refuses to run** unless the source run's own metadata records
`quality_gate.passed == true`. `train_alpha158_model()` loads this file (if
present) and passes it as `pinned_reference` on every subsequent run -- so
check 3.2.5 stays a no-op until a human has explicitly blessed a reference,
per the implementation plan's sequencing requirement.

## Test coverage
`tests/test_train_alpha158_quality_gate.py`, following the
`tests/test_visualize_key_contracts.py` pattern (real payload shapes, not
loosely-mocked). Required fixtures per the implementation plan's Part 3.4/3.7:

1. `test_20260905_incident_fails_gate` -- 232/908 distinct, Rank IC = nan.
2. `test_20260908_incident_fails_gate` -- 7/903 distinct, dominant 858/903,
   real IR/MDD/ffr numbers pulled directly from
   `mlruns/306366047040812909/145fb75192024a80b0ad8e1f95c366f0`.
3. `test_nan_rank_ic_alone_fails_gate` -- isolates only the fail-closed
   NaN-Rank-IC fix; every other check is healthy.
4. `test_healthy_baseline_passes_gate` -- mirrors the real
   `362c9610cd34e468b0b302a70006469` baseline's metrics; asserted to pass.

Plus auxiliary coverage for schema/integrity violations, staleness, the
non-convergence warning (vs. hard fail), the pinned-reference check's three
states (skipped / structural break / matching reference), insufficient
reference overlap, and the two `extract_*` helpers reading the correct
recorder metric keys.

## Files
- `qlib/contrib/validation/__init__.py` -- package init, re-exports.
- `qlib/contrib/validation/score_quality.py` -- the gate itself.
- `scripts/train_alpha158_lightgbm.py` -- integration (restructured
  `train_alpha158_model`, new `_atomic_write_via`/`_build_scores_df`/
  `_resolve_output_dirs`/`_get_git_sha`/`_hash_config`/`bless_pinned_reference`
  helpers, hardened `calculate_ic_metrics`).
- `tests/test_train_alpha158_quality_gate.py` -- regression-test twin.
- `.gitignore` -- excludes `models/lightgbm/_smoketest/` and
  `output/scores/_smoketest/`.
