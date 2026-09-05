# Implementation Plan: Option A - Rapid Alpha158 Calibration & Seed Universe Training

## 1. Goal Description
Resolve the degenerate `-0.00000` factor score, zero Rank IC, and generic `Column_0` feature names on the LightGBM Alpha158 card in `visualize_stock_analysis.py` by:
1. Populating Qlib binary features for the top 60 liquid US large-cap seed equities across all 11 GICS sectors.
2. Calibrating LightGBM regularization hyperparameters (`lambda_l1`, `lambda_l2`, `min_child_samples`) in the YAML config to allow tree splitting and non-zero alpha generation.
3. Fixing feature column name preservation during training and adding a robust index-to-factor fallback map in `infer_alpha158.py`.
4. Retraining the model with quality gates (`num_trees > 1`, `std(predictions) > 0`, `rank_ic != 0.0`).
5. Regenerating `reports/MSFT_analysis_report_2026-09-05.html` and verifying non-zero scores, realistic percentiles, and named factor drivers (`ROC20`, `MA60`, etc.).

---

## 2. User Review Required
> [!IMPORTANT]
> **Data Download Time**: Downloading and Qlib binary dumping for the 60 seed equities takes approximately 2 to 3 minutes via Yahoo Finance.
> 
> **Universe Scope**: This 60-stock universe covers all 11 GICS sectors (Tech, Healthcare, Financials, Consumer, Energy, Industrials, Utilities, Real Estate, Materials) plus benchmark ETFs (`SPY`, `QQQ`, `IWB`, `VOO`), providing genuine cross-sectional dispersion while establishing the verified foundation for later scaling to the full 909-stock Russell 1000.

---

## 3. Proposed Changes

### Component 1: Market Data Pipeline (60-Stock Seed Universe)
Populate `~/.qlib/qlib_data/us_data/features` with normalized daily binary data for the 60 liquid seed tickers defined in `scripts/get_russell1000_symbols.py`.

#### [EXECUTE] Data Downloader CLI
- Run `python scripts/download_us_selected_data.py --symbols <60_symbols>` to fetch, normalize, and binary dump to `~/.qlib/qlib_data/us_data`.
- Update `~/.qlib/qlib_data/us_data/instruments/russell1000_seed.txt` and ensure `russell1000.txt` or a dedicated instrument file contains these active symbols.

---

### Component 2: LightGBM Hyperparameter Calibration
#### [MODIFY] [`examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_russell1000.yaml`](file:///e:/SRC/GITHUB/my-qlib/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_russell1000.yaml)
- Lower L1 regularization from `205.6999` to `0.1`.
- Lower L2 regularization from `580.9768` to `1.0`.
- Set `min_child_samples: 5` (appropriate for seed universe size).
- Set `num_leaves: 31`, `max_depth: 6`.
- Set `num_boost_round: 50`.

---

### Component 3: Feature Name Preservation & Fallback Mapping
#### [MODIFY] [`scripts/train_alpha158_lightgbm.py`](file:///e:/SRC/GITHUB/my-qlib/scripts/train_alpha158_lightgbm.py)
- In `train_alpha158_model()`, inspect `dataset.prepare("train").columns` or `Alpha158DL.get_feature_config()` to extract true factor column names (`KMID`, `KLEN`, `ROC5`, `ROC20`, `MA5`, `MA60`, `STD20`, `VWAP0`, etc.).
- Supply or map feature names in `top_10_features` metadata rather than recording raw `Column_0`...`Column_9`.
- Add an automated Quality Gate asserting:
  - `num_trees >= 10`
  - `leaf_count > 1`
  - `std(predictions) > 1e-6`
  - Warning if `rank_ic == 0.0`

#### [MODIFY] [`scripts/infer_alpha158.py`](file:///e:/SRC/GITHUB/my-qlib/scripts/infer_alpha158.py)
- Add canonical `ALPHA158_FEATURE_NAMES` dictionary mapping `Column_i` or integer indices to real factor names and economic descriptions.
- In `_format_result()`, if a feature name starts with `Column_` or is missing, translate it using the canonical dictionary so no generic column index is ever surfaced.

---

### Component 4: Verification & Report Regeneration
- Run `python scripts/train_alpha158_lightgbm.py` to train and serialize new artifacts.
- Inspect `alpha158_russell1000_latest_meta.json` and `output/scores/alpha158_russell1000_latest.csv` to confirm non-zero scores and named factors.
- Run `python scripts/visualize_stock_analysis.py MSFT`.
- Run `python scripts/run_all_tests.py` to verify that all test suites remain 100% passing.

---

## 4. Verification Plan

### Automated Tests
```powershell
python scripts/run_all_tests.py
```
- Validates that `test_lightgbm_alpha158_us.py`, `test_stock_analysis_engine.py`, `test_visualize_stock_analysis_refactor.py`, and all 15 core suites pass.

### Manual / Visual Verification
- Inspect generated `reports/MSFT_analysis_report_2026-09-05.html`:
  1. Alpha158 Raw Score is non-zero (e.g., `+0.0182` or `-0.0094`).
  2. Predicted 5-Day Excess Return is non-zero.
  3. Russell 1000 Percentile reflects genuine rank distribution.
  4. Top Contributing Factors display named quantitative indicators (`ROC20`, `MA60`, `KMID`, `STD20`) with positive/negative gain values.
