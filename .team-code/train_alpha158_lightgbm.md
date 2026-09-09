# Technical Specification: `train_alpha158_lightgbm.py`

## 1. Overview & Purpose
`train_alpha158_lightgbm.py` is the production training runner for Microsoft Qlib's **LightGBM Alpha158** predictive model tailored for US Equities (Russell 1000 universe).

It executes an automated machine learning workflow:
1. Initializes Qlib with US market configuration (`provider_uri=~/.qlib/qlib_data/us_data`, `region=us`).
2. Validates and loads the Russell 1000 instrument universe (`data/instruments/russell1000.txt`).
3. Computes the 158-factor technical feature set (`Alpha158`) across historical daily bars using pure Python/NumPy rolling and expanding operator fallbacks.
4. Splits the timeline into purged walk-forward segments:
   - **Train**: 2015-01-01 → 2022-12-31 (8-year modern regime)
   - **Validation**: 2023-01-01 → 2023-12-31 (early stopping with L1/L2 regularization)
   - **Test / Out-of-Sample**: 2024-01-01 → Present
5. Trains `LGBModel` with tree-based regularization (`lambda_l1=205.7`, `lambda_l2=581.0`, `num_leaves=128`, `learning_rate=0.05`).
6. Tracks experiments and evaluation metrics in MLflow (`lightgbm_alpha158_us_russell1000`).
7. Serializes production artifacts into `models/lightgbm/` and exports out-of-sample cross-sectional score parquets into `output/scores/`.

## 2. Storage Locations & Artifacts

| Component | Filesystem Path | Purpose |
| :--- | :--- | :--- |
| **Production Model Binary** | `models/lightgbm/alpha158_russell1000_latest.pkl` | Pickled model ready for high-speed inference in production. |
| **Native Booster Text** | `models/lightgbm/alpha158_russell1000_latest.txt` | LightGBM text dump readable without full Python environment. |
| **Model Metadata JSON** | `models/lightgbm/alpha158_russell1000_latest_meta.json` | Contains IC, Rank IC, ICIR, hyperparameters, and top 10 feature gains. |
| **Versioned Checkpoints** | `models/lightgbm/checkpoints/alpha158_russell1000_<YYYYMMDD_HHMMSS>.pkl` | Timestamped checkpoints for backtesting historical model versions. |
| **Cross-Sectional Scores** | `output/scores/alpha158_russell1000_latest.parquet`<br>and `.csv` | Pre-computed daily scores, ranks, and percentiles for all Russell 1000 stocks. |
| **Versioned Score Checkpoints** (2026-09-08) | `output/scores/checkpoints/alpha158_russell1000_scores_<YYYYMMDD_HHMMSS>.parquet` | Timestamped score-file snapshots -- gives the score artifact the same rollback path the `.pkl` already had. |
| **Pinned Reference** (2026-09-08) | `models/lightgbm/pinned_reference/alpha158_russell1000_pinned_reference.parquet`<br>+ `_meta.json` | The explicitly-blessed, immutable baseline used by the gate's rank-correlation check. Updated only via `--bless_pinned_reference`, never implicitly. |
| **MLflow Experiment** | `mlruns/<exp_id>/<run_id>/artifacts/` | Detailed training traces, loss curves, and evaluation tables. |

## 2.1 Cross-Sectional Rank/Percentile Consistency (2026-09-05 fix)

The model's score distribution is degenerate on many dates (e.g. only 232 distinct
score values across 908 Russell 1000 names on 2026-09-04, with ties up to 120-wide
-- see [audit_dataset_segments.md](audit_dataset_segments.md)'s sibling doc
[20260905-finance_team_review_alpha158_degenerate_score.md](20260905-finance_team_review_alpha158_degenerate_score.md)
for the root cause of the degeneracy itself). `rank` is computed with
`.rank(ascending=False, method="min")` (standard competition ranking), **not**
`method="dense"`. With this much tie-degeneracy, `"dense"` ranks *distinct score
values* rather than *cross-sectional position* -- it silently stops meaning "Nth
best of the universe" and starts meaning "Nth distinct value", which can diverge
enormously from `percentile` (computed independently via `.rank(pct=True)`, which
correctly divides by the full universe size regardless of ties). An adversarial
audit of a report using the old `"dense"` rank caught exactly this: FIX showed
"Rank 179 of 908" (implying ~80th percentile) alongside a stored `percentile` of
51.8% for the same row -- a rank/percentile pair that cannot both be right for the
same score. `method="min"` keeps `rank` consistent with `percentile` (small
residual differences remain expected and legitimate: `min` assigns a tied group
its best rank, while `percentile`'s default tie-handling averages within the tied
group -- these are two standard, differently-defined ranking conventions, not a
bug against each other).

## 2.2 Handler processor defaults are inherited from Qlib, not a repo override (2026-09-08)

The workflow YAML does not set `infer_processors`/`learn_processors` on the
`Alpha158` handler kwargs -- it inherits Qlib's own defaults verbatim:
`infer_processors=[]` (no processor applied to the 158 raw features at
inference) and `learn_processors=[DropnaLabel, CSZScoreNorm(fields_group="label")]`
(`CSZScoreNorm` applied only to the label, cross-sectionally, during
learning -- never to the features). This is Qlib's canonical CSI300 benchmark
config's own behavior too (`examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml`),
which does not exhibit the score-degeneracy failure mode on its own reference
data. **Processor misconfiguration is therefore ruled out as the cause of the
2026-09-05/2026-09-08 score-collapse incidents** -- both were caused by the
model's own hyperparameters/boosting-round count (see
[20260908-alpha158_training_audit_and_score_degeneracy_implementation_plan.md](20260908-alpha158_training_audit_and_score_degeneracy_implementation_plan.md),
Part 1.2), not the feature/label preprocessing pipeline. A future reviewer
should check this note before re-opening the processor hypothesis.

## 2.3 Production promotion gate (2026-09-08)

`train_alpha158_model()` no longer writes any production artifact
unconditionally. It now:

1. Runs `task_train()` (which already executes `SignalRecord` ->
   `SigAnaRecord` -> `PortAnaRecord` per the workflow YAML's `task.record`
   list) and assembles the complete evidence set (trained model, cross-
   sectional scores, IC/Rank IC, and portfolio backtest metrics) entirely in
   memory.
2. Calls `qlib.contrib.validation.score_quality.validate_score_quality()`
   **exactly once** against that complete set.
3. Only if the gate does not raise does it proceed to write
   `alpha158_russell1000_latest.{pkl,txt,json}` /
   `alpha158_russell1000_latest.{parquet,csv}` -- and even then, every write
   is staged to a temp file and atomically renamed into place, so a failed or
   interrupted run can never leave a partially-promoted artifact.

Full design, thresholds, and rationale:
[validate_score_quality.md](validate_score_quality.md).

## 2.4 Root cause of the 2026-09-07/08 catastrophic failure: a stale/truncated
qlib binary-store calendar, not LightGBM nondeterminism (2026-09-08, follow-up)

The 2026-09-08 unmodified-config retrain (`.team-code/walkthroughs/walkthrough-r3.md`,
Section 4) failed the new gate with `num_trees=8`, `information_ratio=-2.02`,
`max_drawdown=-44%`, and `ffr=nan`, and that walkthrough recorded an
**unverified working hypothesis** that unpinned LightGBM seeds under
`num_threads=16` were responsible. A dedicated follow-up investigation
disconfirmed that hypothesis with direct evidence and found the real cause:

**Nondeterminism, disconfirmed.** Three independent process invocations of the
exact unmodified retrain command, run back-to-back on 2026-09-08, produced
**bit-identical** results to 16 significant figures (`ic=0.008401536470462791`,
`rank_ic=0.0023225113642203507`, `num_trees=8`, identical portfolio metrics
down to the last digit) every time. LightGBM's own seed defaults
(`seed`/`bagging_seed`/`feature_fraction_seed`/`data_random_seed`) are fixed
constants in its C++ `Config`, not randomized, when left unset, so this
environment was never actually nondeterministic. This also explains why the
three 2026-09-05 "healthy baseline" runs (`362c9610`/`605c0c64`/`b3537e20`)
were themselves bit-identical to each other -- the same determinism, just at a
different (correct) data state.

**Real root cause: `D:/trading/qlib/qlib_data/calendars/day.txt` was stale and
truncated relative to the per-ticker binary feature files.** Confirmed via a
timing/mtime audit of the external qlib data store (not tracked by this repo's
git): every ticker's `.day.bin` files carry `start_index=0.0` and `1930`
values (spanning 2019-01-02 -> 2026-09-04, matching `source/*.csv` and
`normalize/*.csv` exactly), but `calendars/day.txt` had only **1500** entries
starting **2020-09-16** -- the exact misalignment already documented (but not
repaired) in
[20260905-russell1000_factor_verdict_screen-walkthrough.md](20260905-russell1000_factor_verdict_screen-walkthrough.md)'s
Finding 1. Per-file mtimes on the store showed the corrupted calendar (plus
`instruments/all.txt` and a newly-added `features/SNOW/*`) was written at
2026-09-06 00:57 UTC -- **after** the last "healthy" baseline run (`362c9610`,
ended 2026-09-05 23:33 UTC) and **before** every subsequent run, including the
2026-09-08 failure. `SNOW`'s own post-fix `start_index=430` against the
corrected 1930-entry calendar lands exactly on the old calendar's start date,
indicating whatever process added `SNOW` to the store rebuilt `day.txt` from a
single ticker's date range instead of the union across all 909 tickers --
silently truncating the calendar 430 trading days forward for every other
name and shifting `D.features()`'s date-index mapping for 857/909 tickers.

**Fix applied**: re-ran Qlib's own `scripts/dump_bin.py dump_all` against the
already-correct, already-verified `normalize/*.csv` files (which were never
themselves corrupted -- only the derived binary calendar was), writing a fresh
`calendars/day.txt` (1930 entries, 2019-01-02 -> 2026-09-04, now consistent
with every `.day.bin` file's `start_index`/length) with `--backup_dir`
preserving the pre-fix store byte-for-byte at
`D:/trading/qlib/qlib_data_backup_20260908_broken_calendar/`. This only
rewrites `calendars/`, `features/`, and `instruments/all.txt`; the
manually-curated `instruments/russell1000.txt` (already correct -- it recorded
each ticker's true 2019-01-02 -> 2026-09-04 span all along) and the unrelated
`events/`/`options/` subdirectories were untouched. Verified post-fix:
`D.features(['AAPL'], ['$close'], end_time='2026-09-04')` now returns
`8.546910` (the true value) instead of the pre-fix `6.6574` (which actually
belonged to 2024-12-16).

**Outcome after the fix**: three more repeated invocations of the exact
unmodified retrain (still no config/hyperparameter change) were, again,
bit-identical to each other, and reproduced the 2026-09-05 "healthy baseline"
numbers exactly (`ic=0.008004342501323756`, `rank_ic=0.010932338795831296`,
`information_ratio=1.4947`, `annualized_return=+29.71%`,
`max_drawdown=-12.5%`, `ffr=1.0`, `num_trees=38`). The catastrophic failure
modes (NaN fill rate, -44% drawdown, IR -2.02, 8-tree model) are gone and do
not recur. **The gate still fails** on two checks that were never evaluated
against the historical baseline before this session's new gate existed:
`num_trees=38 < 50` and score distinctness (`distinct_fraction=0.2555`,
`dominant_share=0.1322`, both short of the `>=0.95`/`<=0.02` bar) -- vastly
improved from the pre-fix run (`distinct_fraction=0.05`, `dominant_share=0.79`)
but not passing. This appears to be a pre-existing characteristic of the
current hyperparameters (already reduced from the extreme
`lambda_l1=205.7`/`lambda_l2=580.98` documented in
[20260905-finance_team_review_alpha158_degenerate_score.md](20260905-finance_team_review_alpha158_degenerate_score.md)
down to `lambda_l1=0.1`/`lambda_l2=1.0`) applied to a real, low-signal daily
return-prediction task, not a new bug -- but changing hyperparameters further
to clear this bar is a model-design decision, not a bug fix, and per this
session's own constraints was correctly left to a dedicated, `@team-finance`-
reviewed follow-up rather than iterated on unilaterally to force a pass. No
reference has been blessed and production remains the pre-existing
`145fb751` artifact; see
[walkthrough-r4.md](walkthroughs/walkthrough-r4.md) for the full record.

## 3. Usage & CLI Options
```bash
# Standard training using default US Russell 1000 workflow. Only a run that
# PASSES the quality gate (see 2.3 above) is promoted to
# models/lightgbm/alpha158_russell1000_latest.* / output/scores/alpha158_russell1000_latest.*.
# A failing run raises ScoreQualityGateError and touches no production artifact.
python scripts/train_alpha158_lightgbm.py

# Fast run override for quick smoke testing (10 boosting rounds). WARNING:
# activating --num_boost_round FORCES output to scratch paths
# (models/lightgbm/_smoketest/, output/scores/_smoketest/) -- no argument
# combination can redirect this back onto the production artifact paths. The
# quality gate still runs and every failure is logged, but does not block the
# scratch-path write (a deliberately tiny round count is expected to fail the
# num_trees floor; that is not itself a bug to fix).
python scripts/train_alpha158_lightgbm.py --num_boost_round 10

# Custom configuration file and data directory:
python scripts/train_alpha158_lightgbm.py \
    --config examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_russell1000.yaml \
    --qlib_dir ~/.qlib/qlib_data/us_data \
    --market russell1000

# Explicitly bless the CURRENT production scores file as the pinned reference
# used by the gate's rank-correlation-vs-reference check. Refuses to run
# unless that production run's own metadata records a passing quality gate.
# Never automatic -- a deliberate, auditable, operator-invoked action only.
python scripts/train_alpha158_lightgbm.py --bless_pinned_reference
```

