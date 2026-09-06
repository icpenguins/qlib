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
| **Versioned Checkpoints** | `models/lightgbm/checkpoints/alpha158_russell1000_<YYYYMMDD>.pkl` | Timestamped checkpoints for backtesting historical model versions. |
| **Cross-Sectional Scores** | `output/scores/alpha158_russell1000_latest.parquet`<br>and `.csv` | Pre-computed daily scores, ranks, and percentiles for all Russell 1000 stocks. |
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

## 3. Usage & CLI Options
```bash
# Standard training using default US Russell 1000 workflow:
python scripts/train_alpha158_lightgbm.py

# Fast run override for quick smoke testing (10 boosting rounds):
python scripts/train_alpha158_lightgbm.py --num_boost_round 10

# Custom configuration file and data directory:
python scripts/train_alpha158_lightgbm.py \
    --config examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_russell1000.yaml \
    --qlib_dir ~/.qlib/qlib_data/us_data \
    --market russell1000
```

