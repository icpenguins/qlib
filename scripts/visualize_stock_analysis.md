# Usage Guide: `visualize_stock_analysis.py`

Interactive stock performance visualizer and 3-month predictive buy timing dashboard generator.

---

## 1. Overview & Capabilities

`visualize_stock_analysis.py` generates a self-contained, zero-dependency, interactive HTML dashboard combining historical performance evaluation, technical buy milestones, and forward predictive quantitative forecasting:

1. **Pre-Execution Data Freshness Verification**:
   - Compares the latest available trade date in local data against the requested report date (taking US trading schedules and weekends into account).
   - If data is missing or stale, automatically invokes `download_us_selected_data.py` to acquire up-to-date prices from Yahoo Finance before running calculations.
2. **Historical Performance Summary (1Y / 3Y / 5Y Historical)**:
   - Evaluates realized performance cards reporting Total Realized Return %, Compound Annual Growth Rate (CAGR %), Maximum Drawdown (MDD %), and Annualized Sharpe Ratio.
3. **Multi-Period Forward Projections & Probability Scoring (6M / 1Y / 2Y / 3Y)**:
   - Models Geometric Brownian Motion (GBM) drift and volatility distributions across 4 investment horizons (6 Months, 1 Year, 2 Years, 3 Years).
   - Displays Expected Median Return %, Annualized CAGR %, Base Target Price ($P_{50}$), Bear-to-Bull Corridor ($P_{10}$ to $P_{90}$), exact statistical Probability Score %, color-coded probability meter, and qualitative Confidence Badge.
4. **3-Month Forward Predictive Buy Timing Analysis**:
   - Simulates forward price trajectories over 63 trading days (~3 months) using Monte Carlo methods.
   - Provides clear Actionable Recommendation rating (`STRONG BUY`, `BUY ON PULLBACK`, `ACCUMULATE / DIP BUY`, or `HOLD / CAUTIOUS BUY`), Optimal Entry Price Zone, Optimal Buy Window timeline, 3-Month Target Price, Stop-Loss Risk Level, and Risk/Reward Ratio.
5. **Interactive Canvas Timeline & Moving Averages**:
   - Continuous 50-day (amber) and 200-day (purple) Simple Moving Averages computed across the entire historical series—always visible regardless of timeframe or zoom depth.
   - Golden dashed milestone markers with green rally corridors for historical best buying points.
   - Dynamic zoom-adaptive bottom X-axis date formatting (< 35 days: daily; 35–140 days: month/day; > 140 days: month/year).
   - Multi-mode zooming: drag-to-zoom range selection, mouse wheel zoom, header buttons (`1Y`, `3Y`, `5Y`, `Max`), and reset button.
   - 3-Month predictive Monte Carlo confidence cone chart ($P_{10}$, $P_{50}$, $P_{90}$).
6. **Self-Contained & Date-Stamped Output**:
   - Generates `<SYMBOL>_analysis_report_<YYYY-MM-DD>.html` (e.g. `reports/MSFT_analysis_report_2026-09-03.html`).
   - Self-contained file with embedded styles and JavaScript; opens in any modern web browser without internet access or local servers.

---

## 2. Quick Start & Common Usage Examples

### A. Basic Report Generation
Generates a report for `MSFT` using default data and saves to `reports/`:
```powershell
python scripts/visualize_stock_analysis.py --symbol MSFT
```

### B. Open Automatically in Default Web Browser
Use `--open` to launch the generated HTML report immediately upon completion:
```powershell
python scripts/visualize_stock_analysis.py --symbol NVDA --open
```

### C. Specify Custom Data Directory
Point to an existing data directory containing CSVs or Qlib binaries (e.g. `D:\trading\qlib`):
```powershell
python scripts/visualize_stock_analysis.py --symbol SMH --data_dir D:\trading\qlib
```

### D. Specify Custom Report Output Directory
Store the generated HTML report in a designated folder (e.g. `D:\trading\custom_reports`):
```powershell
python scripts/visualize_stock_analysis.py --symbol AAPL --report_dir D:\trading\custom_reports
```

### E. Specify an Exact File Path
Override directory and naming conventions with `--output` (`-o`):
```powershell
python scripts/visualize_stock_analysis.py --symbol TSLA --output ./my_reports/tesla_special.html
```

### F. Specify Request Date (Historical Backfill or Specific Date Simulation)
Verify data freshness and generate analysis as of a specific date:
```powershell
python scripts/visualize_stock_analysis.py --symbol MSFT --request_date 2026-01-15
```

### G. Disable Automatic Download
Enforce strict offline operation (fails if ticker is not present locally):
```powershell
python scripts/visualize_stock_analysis.py --symbol VOO --no-auto_download
```

---

## 3. CLI Options Reference

| Option | Short Flag | Default | Description |
|---|---|---|---|
| `--symbol` | `-s` | *Required* | Stock ticker symbol (e.g. `MSFT`, `VOO`, `NVDA`, `SMH`). |
| `--data_dir` | `-d` | `~/.qlib/qlib_data/us_data` | Directory containing Qlib binary dataset or CSV files (`source/`, `normalize/`, or root). |
| `--report_dir` | `-r` | `reports` | Directory where generated HTML report is saved. Automatically created if missing. |
| `--output` | `-o` | `None` | Specific output file path (`.html`) or custom target directory (overrides `--report_dir`). |
| `--days_forecast` | | `63` | Forward trading days for predictive simulation (~63 days = 3 months). |
| `--auto_download` | | `True` | Automatically download and refresh market data if missing or stale. |
| `--no-auto_download` | | | Disable automatic download; run only on existing local data. |
| `--start` | | `2000-01-01` | Start date if auto-download pipeline is triggered (`YYYY-MM-DD`). |
| `--request_date` | `--report_date` | Today's date | Date the report was requested (`YYYY-MM-DD`). Drives filename date stamp and freshness checks. |
| `--open` | | `False` | Automatically launch the generated HTML report in your default web browser. |

---

## 4. End-to-End Workflow Example

Run a complete automated workflow from the root directory:

```powershell
# 1. Generate an interactive report for semiconductor ETF (SMH)
#    - Verifies freshness against D:\trading\qlib
#    - Auto-downloads missing data from Yahoo Finance
#    - Stores date-stamped HTML under D:\trading\custom_reports
#    - Automatically launches the report in the browser

python scripts/visualize_stock_analysis.py `
  --symbol SMH `
  --data_dir D:\trading\qlib `
  --report_dir D:\trading\custom_reports `
  --open
```

**Terminal Output Summary Example**:
```
=======================================================
 STOCK PERFORMANCE & PREDICTIVE BUY TIMING ANALYZER 
=======================================================
Symbol:           SMH
Report Requested: 2026-09-03
Data Directory:   D:\trading\qlib
Data Freshness:   Through 2026-09-03 (Up-to-Date)
Auto Download:    True
Forecast Days:    63 (~3 months)
Output Report:    D:\trading\custom_reports\SMH_analysis_report_2026-09-03.html

-------------------------------------------------------
 HISTORICAL PERFORMANCE SUMMARY (1Y / 3Y / 5Y)
-------------------------------------------------------
[1Y] Return: +48.2% | CAGR: 48.2% | Max DD: -18.4% | Sharpe: 1.85
[3Y] Return: +132.5% | CAGR: 32.5% | Max DD: -28.1% | Sharpe: 1.42
[5Y] Return: +245.8% | CAGR: 28.1% | Max DD: -35.2% | Sharpe: 1.28

-------------------------------------------------------
 FORWARD RETURN PROJECTIONS & PROBABILITY SCORES
-------------------------------------------------------
[6 Months] Expected: +12.4% | Target: $ 285.40 | Range: $242.10-$328.50 | Prob: 76.2% (Moderate)
[1 Year  ] Expected: +26.8% | Target: $ 322.10 | Range: $265.00-$391.20 | Prob: 81.4% (High)
[2 Years ] Expected: +58.5% | Target: $ 402.60 | Range: $310.20-$522.40 | Prob: 84.1% (High)
[3 Years ] Expected: +98.2% | Target: $ 503.40 | Range: $360.50-$701.80 | Prob: 86.5% (High)

-------------------------------------------------------
 3-MONTH PREDICTIVE BUY ANALYSIS
-------------------------------------------------------
Recommendation:     BUY ON PULLBACK
Action:             Wait for minor consolidation near 50 SMA before executing entry.
Optimal Entry Zone: $248.50 - $254.00
Optimal Window:     Next 10-15 trading days
3-Month Target:     $285.40 (+12.4%)
Stop-Loss Level:    $238.00
Risk/Reward Ratio:  2.8:1
-------------------------------------------------------

[SUCCESS] Visual report generated at: D:\trading\custom_reports\SMH_analysis_report_2026-09-03.html
Opening report in your web browser...
```

