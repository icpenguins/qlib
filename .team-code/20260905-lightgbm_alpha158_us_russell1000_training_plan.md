# Team-Code Implementation Plan: LightGBM Alpha158 Training Pipeline for US Equities (Russell 1000)

---

## Priority Requirement -1: End-User Requirements Acknowledgement
The engineering team acknowledges and certifies full understanding of the project's core end-users:
1. **The Profitable Stock Trader** (*Veteran Discretionary & Quantitative Prop Trader*):
   - Mandate: Consistent alpha, capital preservation, avoiding catastrophic drawdown.
   - Core Warning: Qlib must not operate in a sterile academic vacuum. Alpha158 scores must be actionable, cross-sectionally ranked (0-100th percentile across the Russell 1000 universe), aligned with market regimes and dealer gamma walls, and accessible for pre-market execution without retrain latency.
2. **The Institutional Hedge Fund Manager** (*Chief Investment Officer & Head of Quantitative Research*):
   - Mandate: Double-digit annualized return, Sharpe ratio > 2.0, net zero market/factor beta.
   - Core Standard: Purged walk-forward temporal splits (Train: 2015-2022, Valid: 2023, Test: 2024-present), tracking of daily Spearman Rank Information Coefficient (Rank IC) and ICIR, rigorous L1/L2 regularization (`lambda_l1: 205.7`, `lambda_l2: 581.0`) against factor collinearity, and auditable storage of all model weights, features, and evaluation records.

---

## Executive Summary & Architectural Overview
Users have observed that Microsoft Qlib's primary native capability—training machine learning models (specifically LightGBM) on the standard 158-factor technical library (`Alpha158`) to generate predictive cross-sectional stock scores—is missing for US equities in the Russell 1000 universe.

This implementation plan defines the complete, production-grade engineering architecture to:
1. **Curate and Download the Russell 1000 US Equities Universe**: Implement automated universe acquisition for the ~1,000 constituent members, generating standard Qlib trading calendars and binary feature stores (`open`, `high`, `low`, `close`, `volume`, `factor`, `change`, and `vwap`).
2. **Configure US-Specific LightGBM Alpha158 Workflow**: Adapt `examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml` into a dedicated US configuration (`workflow_config_lightgbm_Alpha158_us_russell1000.yaml`) with US market parameters, SPY/IWB benchmarks, realistic US execution costs, and purged rolling time splits.
3. **Establish Concrete Storage Locations**: Define deterministic paths for raw market data, feature binary stores, Qlib expression caches, MLflow experiment records, production model binaries (`models/lightgbm/`), and cross-sectional score parquets (`output/scores/`).
4. **Implement Training & Inference CLI Runners**: Build `scripts/train_alpha158_lightgbm.py` for headless model training and `scripts/infer_alpha158.py` for daily scoring.
5. **Integrate into Stock Analysis Engine**: Connect the trained Alpha158 model directly into `scripts/stock_analysis_engine.py` and `scripts/stock_analysis_data.py` to embed predictive alpha scores, percentile rankings, and feature importance into HTML/JSON reports.

---

## Data and Model Storage Specification

| Data Category | Exact Filesystem Path | Format / Specification | Description |
| :--- | :--- | :--- | :--- |
| **Universe Definition** | `~/.qlib/qlib_data/us_data/instruments/russell1000.txt`<br>and `data/instruments/russell1000.txt` | TSV (`SYMBOL\tSTART\tEND`) | Complete Russell 1000 constituent universe list with valid date spans. |
| **Raw Market Data** | `~/.qlib/stock_data/source/us_data/<SYMBOL>.csv` | CSV | Raw daily OHLCV + Adj Close data fetched from Yahoo Finance. |
| **Normalized Data** | `~/.qlib/stock_data/source/us_1d_nor/<SYMBOL>.csv` | CSV | Qlib-normalized data with split/dividend factor adjustments. |
| **Trading Calendar** | `~/.qlib/qlib_data/us_data/calendars/day.txt` | Text (1 date per line) | Unified US exchange trading days (NYSE/NASDAQ). |
| **Feature Binaries** | `~/.qlib/qlib_data/us_data/features/<SYMBOL>/<field>.day.bin` | Float32 binary with header | `open.day.bin`, `high.day.bin`, `low.day.bin`, `close.day.bin`, `volume.day.bin`, `factor.day.bin`, `change.day.bin`, `vwap.day.bin`. |
| **Qlib Expression Cache** | `~/.qlib/qlib_data/us_data/cache/` | Mmap binary cache | Accelerated storage of precomputed Alpha158 factor expressions. |
| **MLflow Runs & Artifacts** | `<repo_root>/mlruns/<exp_id>/<run_id>/artifacts/` | Pickles & text | `params.pkl`, `pred.pkl`, `label.pkl`, `sig_ana.pkl`, `port_analysis.pkl`, `task`. |
| **Production Model Binary** | `<repo_root>/models/lightgbm/alpha158_russell1000_latest.pkl` | Pickle | Serialized `LGBModel` ready for instant inference. |
| **Production Model Text** | `<repo_root>/models/lightgbm/alpha158_russell1000_latest.txt` | LightGBM native text dump | Portable booster model file readable without full Python Qlib stack. |
| **Model Metadata** | `<repo_root>/models/lightgbm/alpha158_russell1000_latest_meta.json` | JSON | Model training parameters, feature names, train/valid/test dates, IC/Rank IC metrics. |
| **Historical Checkpoints** | `<repo_root>/models/lightgbm/checkpoints/alpha158_russell1000_<YYYYMMDD>.pkl` | Pickle | Historical archive of quarterly/annual retrained models. |
| **Daily Inference Scores** | `<repo_root>/output/scores/alpha158_russell1000_latest.parquet`<br>and `.csv` | Parquet / CSV | Cross-sectional scores, percentile ranks, and 5-day predicted returns for all 1,000 stocks. |

---

## Phase 1: Structural Architecture & Benchmark Configuration

### 1. Benchmark Comparison: China CSI300 Reference vs. US Russell 1000 Specification

| Parameter | Reference: `workflow_config_lightgbm_Alpha158.yaml` (China) | Proposed: `workflow_config_lightgbm_Alpha158_us_russell1000.yaml` (US) | Rationale |
| :--- | :--- | :--- | :--- |
| `qlib_init.provider_uri` | `~/.qlib/qlib_data/cn_data` | `~/.qlib/qlib_data/us_data` | Isolated US market binary store. |
| `qlib_init.region` | `cn` | `us` | Enables US calendar and exchange conventions. |
| `market` | `csi300` | `russell1000` | Russell 1000 constituent universe (~1,000 tickers). |
| `benchmark` | `SH000300` | `SPY` (or `^GSPC` / `IWB`) | Institutional US market cap benchmark. |
| `train segment` | `2008-01-01` to `2014-12-31` | `2015-01-01` to `2022-12-31` | Modern 8-year market cycle training window. |
| `valid segment` | `2015-01-01` to `2016-12-31` | `2023-01-01` to `2023-12-31` | 1-year early stopping validation window. |
| `test segment` | `2017-01-01` to `2020-08-01` | `2024-01-01` to `2026-09-04` | 2.5-year true out-of-sample test window. |
| `exchange_kwargs.limit_threshold` | `0.095` (China 10% limit-up/down) | `null` (No daily price bands) | US equity markets do not have 10% daily halt bands. |
| `exchange_kwargs.open_cost` | `0.0005` (5 bps) | `0.0001` (1 bps) | Reflects US institutional electronic trading costs. |
| `exchange_kwargs.close_cost` | `0.0015` (15 bps stamp duty) | `0.0001` (1 bps) | Eliminates Chinese stamp tax assumption. |
| `exchange_kwargs.min_cost` | `5` | `0` | Zero minimum ticket fee for modern US brokers. |
| `model.kwargs.loss` | `mse` | `mse` | Mean squared error on relative excess forward return. |
| `model.kwargs.learning_rate` | `0.2` | `0.05` | Lower learning rate for more stable gradient boosting on noisy US cross-section. |
| `model.kwargs.num_leaves` | `210` | `128` | Prevents tree overfitting across 1,000 cross-sectional stocks. |
| `model.kwargs.lambda_l1` / `l2` | `205.6999` / `580.9768` | `205.6999` / `580.9768` | Strong regularization preserved to prevent factor collinearity overfitting. |

---

## Phase 2: Component Breakdown & Implementation Steps

### Step 1: Environment & Dependency Validation
- Verify and install `lightgbm` in `.venv`:
  - Target: Python 3.11 (`e:\SRC\GITHUB\my-qlib\.venv\Scripts\python.exe`).
  - Command: `& "e:\SRC\GITHUB\my-qlib\.venv\Scripts\python.exe" -m pip install lightgbm pyarrow`.
  - Validate import of `lightgbm` and `qlib.contrib.model.gbdt.LGBModel`.

### Step 2: Russell 1000 Universe Curation (`scripts/get_russell1000_symbols.py`)
- Purpose: Retrieve, validate, and write the 1,000 constituent tickers of the Russell 1000 index.
- Data Sources:
  1. Primary: S&P 500 (503 stocks) + S&P MidCap 400 (400 stocks) from verified Wikipedia / SEC tables, supplemented with top Russell 1000 constituents to reach 1,000 distinct liquid US equities.
  2. Fallback: Pre-packaged constituent list `data/instruments/russell1000.txt` with verified large-cap and mid-cap tickers.
- Output: Writes `data/instruments/russell1000.txt` and `~/.qlib/qlib_data/us_data/instruments/russell1000.txt`.

### Step 3: US Data Downloader VWAP Enhancement (`scripts/download_us_selected_data.py`)
- Purpose: Ensure Qlib binary dumper generates `vwap.day.bin` for every ticker so `Alpha158DL` does not encounter missing factor errors.
- Modification:
  - In `normalize_symbol_data`: Compute `df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0` (or `(df["open"] + df["high"] + df["low"] + df["close"]) / 4.0`).
  - In `dump_to_qlib_format`: Include `"vwap"` in `feature_fields`.
  - Run downloader with `--symbol_file data/instruments/russell1000.txt --qlib_dir ~/.qlib/qlib_data/us_data`.

### Step 4: US Workflow Configuration YAML (`examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_us_russell1000.yaml`)
- Create the official production configuration conforming to Qlib's standard format:
  - `qlib_init`: `provider_uri: "~/.qlib/qlib_data/us_data"`, `region: us`.
  - `market: russell1000`, `benchmark: SPY`.
  - `data_handler_config`: Segments 2015-2022 (train), 2023 (val), 2024-present (test).
  - `task`: `model` (`LGBModel`), `dataset` (`DatasetH` with `Alpha158`), `record` (`SignalRecord`, `SigAnaRecord`, `PortAnaRecord`).

### Step 5: Automated Training Script (`scripts/train_alpha158_lightgbm.py`)
- Purpose: End-to-end executable that:
  1. Initializes Qlib with US provider URI.
  2. Loads or generates `workflow_config_lightgbm_Alpha158_us_russell1000.yaml`.
  3. Executes `task_train` under experiment `lightgbm_alpha158_us_russell1000`.
  4. Extracts the fitted model and saves production copies to `models/lightgbm/alpha158_russell1000_latest.pkl` and `.txt`.
  5. Computes and saves `models/lightgbm/alpha158_russell1000_latest_meta.json` with IC, Rank IC, and feature importance rankings.
  6. Exports the latest predictions to `output/scores/alpha158_russell1000_latest.parquet`.

### Step 6: Scoring & Inference Module (`scripts/infer_alpha158.py` / `qlib/contrib/model/alpha_scorer.py`)
- Purpose: Lightweight scoring utility that can score any individual stock or the entire Russell 1000 universe on any given date.
- Methods:
  - `load_latest_model()`: Loads `models/lightgbm/alpha158_russell1000_latest.pkl`.
  - `predict_stock(symbol, date)`: Computes Alpha158 features for `symbol` up to `date`, evaluates the LightGBM model, and returns:
    - Raw Alpha158 predicted score
    - Cross-sectional percentile rank (0% - 100%)
    - Top 3 positive/negative factor drivers for the prediction.

### Step 7: Integration into `stock_analysis_engine.py` & `stock_analysis_data.py`
- Modify `scripts/stock_analysis_engine.py`:
  - Add `load_alpha158_score(symbol, as_of_date)` helper.
  - When analyzing a stock (e.g. MSFT), query the Alpha158 model / latest score cache.
  - Add an **"Alpha158 Machine Learning Score"** card to the report:
    - Alpha158 Predicted Score: e.g. `+0.0284` (Predicted 5-Day Excess Return)
    - Universe Percentile: `89.2%` (Top Quintile / Strong Long Factor Signal)
    - Factor Conviction Badge: Bullish / Neutral / Bearish based on score percentile.
    - Factor Driver Attribution: Primary contributing Alpha158 sub-factors.
- Modify `scripts/stock_analysis_data.py`:
  - Include `"alpha158"` block in the JSON output schema under `"ml_model"`.

---

## Phase 3: Comprehensive Verification & Quality Assurance Plan

### Automated Unit & Integration Tests (`tests/test_lightgbm_alpha158_us.py`)
1. **Test 1: Configuration Schema & Parsing**:
   - Verify `workflow_config_lightgbm_Alpha158_us_russell1000.yaml` parses valid YAML and maps all parameters correctly (`provider_uri`, `region: us`, `market: russell1000`).
2. **Test 2: Universe & Instrument Resolution**:
   - Verify `data/instruments/russell1000.txt` has valid formatting, non-empty tickers, and proper date ranges.
3. **Test 3: Feature Loading with VWAP**:
   - Verify `Alpha158` feature loader constructs expressions without missing variable errors when binary store includes `vwap.day.bin`.
4. **Test 4: LightGBM Model Fit & Inference**:
   - Train a fast mini-LGBModel on a sample segment of US stocks; verify convergence, non-zero weights, and valid predictions.
5. **Test 5: Model Artifact & Metadata Persistence**:
   - Verify model saves properly to `models/lightgbm/alpha158_russell1000_latest.pkl` and `alpha158_russell1000_latest_meta.json` contains required keys (`metrics`, `trained_at`, `features_count`).
6. **Test 6: Scoring Engine & Percentile Ranking**:
   - Verify `infer_alpha158.py` correctly calculates cross-sectional percentiles (0 to 100) and integrates with `stock_analysis_engine.py`.

---

## Team Perspectives & Review Sign-Off
- **Architect**: Standardized on Qlib native `LGBModel`, `DatasetH`, and `Alpha158` architecture to maintain complete ecosystem compatibility while tailoring to US institutional markets.
- **Program Manager**: Directly fulfills high-priority user request for US Alpha158 LightGBM training, positioning our repo as the definitive US-equities Qlib fork.
- **Principal Developer**: Enforces strong L1/L2 regularization (`lambda_l1=205.7`, `lambda_l2=581.0`), controlled tree depth (`max_depth=8`, `num_leaves=128`), and thread bounds to ensure production stability.
- **Senior Developer**: Validates VWAP binary generation, full type hints, docstrings, and clean separation between training scripts, configs, and analysis reports.
- **QA Tester**: Comprehensive 6-point test suite ensuring zero regressions across existing 85 unit tests.
- **CI/CD Developer**: Headless execution script `scripts/train_alpha158_lightgbm.py` for automated scheduled retrains.

