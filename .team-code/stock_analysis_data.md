# Stock Analysis JSON Data Contract Engine (`stock_analysis_data.py`)

## 1. Executive Summary & Dual End-User Alignment
The `stock_analysis_data.py` module is the dedicated analytical data serialization engine and headless CLI for institutional stock analysis in Qlib. It decouples data transformation and contract generation from front-end visual rendering, enabling headless quantitative pipelines, algorithmic trading strategies, and automated reporting.

### End-User Value Proposition
- **The Profitable Stock Trader**:
  Provides a ultra-fast terminal CLI to generate structured quantitative datasets without launching a browser or rendering HTML. Enables direct programmatic piping of critical trading metrics (AVWAP support, volume profile POC, GEX gamma flip, PEAD drift, and optimal entry zones) into execution bots, risk watchdogs, and proprietary alerts.
- **The Institutional Hedge Fund Manager**:
  Enforces a strict, versioned JSON data contract (`contract_version: "1.0.0"`). Guarantees reproducible mathematical inputs across quantitative factor models, eliminates cross-contamination between presentation and calculation layers, and supports distributed CI/CD batch processing.

---

## 2. Architecture & Pipeline Decoupling

```
┌───────────────────────────────────────────────────────────┐
│              scripts/stock_analysis_engine.py             │
│   (BOCD Regimes, AVWAP KDE, Dealer GEX, PEAD Drift, Qlib) │
└─────────────────────────────┬─────────────────────────────┘
                              │ Raw Dict & Pandas DataFrames
                              ▼
┌───────────────────────────────────────────────────────────┐
│               scripts/stock_analysis_data.py              │
│  - resolve_json_path()                                    │
│  - _sanitize_for_json()                                   │
│  - prepare_analysis_json_payload()                        │
│  - export_analysis_json()                                 │
│  - load_analysis_json()                                   │
│  - generate_stock_analysis_data()                         │
│  - CLI: python stock_analysis_data.py -s AAPL ...         │
└─────────────────────────────┬─────────────────────────────┘
                              │ Canonical .json File (Contract v1.0.0)
                              ▼
┌───────────────────────────────────────────────────────────┐
│            scripts/visualize_stock_analysis.py            │
│  - Consumes canonical JSON dataset                        │
│  - Injects JSON into <script id="report-data"> (CORS-safe)│
│  - Renders standalone interactive HTML dashboard          │
└───────────────────────────────────────────────────────────┘
```

---

## 3. Canonical JSON Schema Contract (v1.0.0)

Every generated JSON dataset complies with the following root schema:

```json
{
  "metadata": {
    "symbol": "AAPL",
    "request_date": "2025-10-31",
    "latest_data_date": "2025-10-31",
    "is_up_to_date": true,
    "forecast_days": 63,
    "generated_at": "2026-09-04T11:36:32.123456",
    "contract_version": "1.0.0"
  },
  "symbol": "AAPL",
  "historical_data": [
    {
      "date": "2025-10-30",
      "open": 224.50,
      "high": 226.10,
      "low": 223.80,
      "close": 225.20,
      "volume": 45200000,
      "sma50": 221.40,
      "sma200": 205.80,
      "avwap_ytd": 218.30,
      "avwap_ytd_upper_1s": 226.50,
      "avwap_ytd_lower_1s": 210.10
    }
  ],
  "performance": {
    "latest_date": "2025-10-31",
    "latest_close": 225.20,
    "latest_price": 225.20,
    "periods": {
      "1Y": { "total_return_pct": 28.5, "cagr_pct": 28.5, "max_drawdown_pct": -10.5, "sharpe_ratio": 1.62 },
      "3Y": { "total_return_pct": 68.2, "cagr_pct": 18.9, "max_drawdown_pct": -22.1, "sharpe_ratio": 1.21 },
      "5Y": { "total_return_pct": 125.0, "cagr_pct": 17.6, "max_drawdown_pct": -28.4, "sharpe_ratio": 1.08 }
    }
  },
  "best_buys": [
    {
      "rank": 1,
      "date": "2024-04-19",
      "entry_price": 165.0,
      "peak_price": 220.0,
      "holding_days": 180,
      "max_gain_pct": 33.3,
      "trigger_type": "AVWAP Value Area Bounce"
    }
  ],
  "predictive": {
    "recommendation": "STRONG BUY",
    "optimal_entry_range": [215.0, 222.0],
    "target_price_3m": 245.0,
    "expected_return_pct": 11.5,
    "stop_loss": 208.0,
    "risk_reward_ratio": 3.4
  },
  "projections": {
    "6M": { "projected_return_pct": 9.2, "probability_score": 75.0, "confidence": "High" },
    "1Y": { "projected_return_pct": 16.5, "probability_score": 70.0, "confidence": "High" },
    "2Y": { "projected_return_pct": 32.0, "probability_score": 65.0, "confidence": "Moderate" },
    "3Y": { "projected_return_pct": 48.0, "probability_score": 60.0, "confidence": "Moderate" }
  },
  "regime": {
    "state": 1,
    "name": "Bull Trend",
    "action": "Accumulate dips",
    "changepoint_prob_pct": 8.5,
    "vol_ratio": 0.92,
    "vol_21d_pct": 14.8,
    "credit_mom_pct": 1.5,
    "risk_multiplier": 1.0
  },
  "microstructure": {
    "avwap": {
      "ytd": { "value": 218.0, "zscore": 0.6, "lower_1s": 210.0, "upper_1s": 226.0 },
      "high_52w": { "value": 235.0, "spread_pct": -4.0 },
      "low_52w": { "value": 170.0, "spread_pct": 32.0 }
    },
    "volume_profile": {
      "poc": 220.0,
      "val": 212.0,
      "vah": 228.0,
      "void_status": "Balanced Liquidity"
    }
  },
  "derivatives": {
    "net_gex_millions": 45.2,
    "regime": "+GEX (Dealer Long Gamma / Volatility Dampening)",
    "gamma_flip_price": 214.0,
    "call_wall": 240.0,
    "put_wall": 210.0,
    "max_pain": 225.0
  },
  "events": {
    "catalyst_status": { "status_code": "SAFE", "days_to_earnings": 20, "days_to_macro": 12 },
    "pead": { "drift_regime": "Bullish Post-Earnings Drift", "sue_score": 1.40, "post_earnings_drift_pct": 5.0 },
    "degrossing": { "position_haircut": 1.0, "risk_advice": "Normal position sizing." }
  }
}
```

---

## 4. Command-Line Interface (CLI) Specification

```bash
python scripts/stock_analysis_data.py [OPTIONS]
```

### Options & Arguments
| Flag | Short | Default | Description |
| :--- | :--- | :--- | :--- |
| `--symbol` | `-s` | *Required* | Stock ticker symbol to analyze (e.g. `AAPL`, `NVDA`). |
| `--data_dir` | `-d` | `~/.qlib/qlib_data/us_data` | Path to Qlib binary data repository. |
| `--report_dir` | `-r` | `reports` | Target directory for generated `.json` files. |
| `--output` | `-o` | `None` | Explicit path or filename override for the `.json` output. |
| `--days_forecast` | | `63` | Forward trading days forecast (~3 months). |
| `--auto_download` | | `True` | Automatically fetch missing/outdated market data. |
| `--no-auto_download`| | | Disable automatic downloading. |
| `--start` | | `2000-01-01` | Historical data start date (`YYYY-MM-DD`). |
| `--request_date` | `--report_date` | Today | Evaluation date for point-in-time backtesting. |
| `--indent` | | `2` | JSON indentation level for output formatting. |
| `--quiet` | `-q` | `False` | Suppress stdout summary banner and non-error logs. |

### CLI Example Usage
```bash
# Standard execution
python scripts/stock_analysis_data.py --symbol AAPL

# Custom destination with quiet mode for shell automation
python scripts/stock_analysis_data.py -s NVDA -o ./data/nvda_contract.json --quiet

# Point-in-time backtest evaluation
python scripts/stock_analysis_data.py -s MSFT --request_date 2025-06-30 --no-auto_download
```

