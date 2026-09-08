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

