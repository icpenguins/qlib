# Usage Guide: `download_us_selected_data.py`

Targeted US stock data downloader, Qlib-compliant normalizer, and binary dumper pipeline.

---

## 1. Overview & Capabilities

`download_us_selected_data.py` provides an end-to-end pipeline to acquire, normalize, and serialize US stock market data into Qlib-compatible datasets:

1. **Multi-Source Yahoo Finance Ingestion**:
   - Queries Yahoo Finance v8 chart API with cookie/crumb session handling.
   - Fallbacks to multi-source endpoints and user-agent rotation.
   - Automatic retries with exponential backoff.
2. **Qlib-Compliant Data Normalization**:
   - Computes split and dividend adjustment factor: $\text{factor} = \frac{\text{adj\_close}}{\text{close}}$.
   - Adjusts volume: $\text{adjusted\_volume} = \frac{\text{volume}}{\text{factor}}$.
   - Calculates daily percentage change: $\text{change} = \frac{\text{close}_t - \text{close}_{t-1}}{\text{close}_{t-1}}$.
   - Standardizes timestamps and sorts chronologically.
3. **Qlib Binary Serialization**:
   - Writes float32 little-endian (`<f`) arrays to `features/<SYMBOL>/<field>.day.bin`.
   - Generates unified market calendars (`calendars/day.txt`).
   - Updates instrument index metadata (`instruments/all.txt`).
   - Makes datasets immediately ready for `qlib.init(provider_uri=...)` or `stock_analysis_engine.py`.

---

## 2. Quick Start & Common Usage Examples

### A. Download Default 12 US Tickers
Default symbol universe: `VOO`, `FIX`, `CRDO`, `MSFT`, `INTC`, `MU`, `ANET`, `IBM`, `TSLA`, `NVDA`, `SPY`, `QQQ`.
```powershell
python scripts/download_us_selected_data.py
```

### B. Custom Symbols via Command Line
Pass one or more tickers separated by spaces or commas:
```powershell
# Space-separated
python scripts/download_us_selected_data.py --symbols AAPL MSFT NVDA GOOGL AMZN

# Comma-separated string
python scripts/download_us_selected_data.py --symbols "SMH, AMD, AVGO"
```

### C. Load Symbols from a File
Supports plain text (`.txt`), CSV (`.csv`), or JSON (`.json`):
```powershell
# Plain text file (one per line or comma-separated)
python scripts/download_us_selected_data.py --symbol_file my_watchlist.txt

# Using short flag -f
python scripts/download_us_selected_data.py -f my_portfolio.csv

# Directly passed to --symbols
python scripts/download_us_selected_data.py --symbols universe.json
```

### D. Custom Destination Directory
Consolidate all downloaded and processed data under a single root directory:
```powershell
python scripts/download_us_selected_data.py --symbols SMH NVDA --target_dir D:\trading\qlib
```
This automatically creates and populates:
- `D:\trading\qlib\source\` (Raw Yahoo CSVs)
- `D:\trading\qlib\normalize\` (Normalized CSVs with factors)
- `D:\trading\qlib\qlib_data\` (Binary `.bin` features and calendars)

### E. Custom Date Ranges
```powershell
python scripts/download_us_selected_data.py --symbols MSFT --start 2015-01-01 --end 2026-09-03
```

### F. Download Raw CSVs Only (Skip Qlib Binary Dumping)
If you only need raw and normalized CSV files:
```powershell
python scripts/download_us_selected_data.py --symbols AAPL MSFT --target_dir ./csv_data --no-dump_qlib
```

---

## 3. CLI Options Reference

| Option | Short Flag | Default | Description |
|---|---|---|---|
| `--symbols` | `-s` | `None` (defaults to 12 targeted US stocks) | Space or comma-separated tickers, or path to a file containing tickers. |
| `--symbol_file` | `-f` | `None` | Path to `.txt`, `.csv`, or `.json` file containing stock symbols. |
| `--target_dir` | `-o`, `--dest` | `None` | Root destination folder. Automatically partitions into `source`, `normalize`, and `qlib_data`. |
| `--start` | | `2000-01-01` | Start date for historical price data (`YYYY-MM-DD`). |
| `--end` | | Tomorrow's date | End date for historical price data (`YYYY-MM-DD`). |
| `--interval` | | `1d` | Data frequency (`1d`, `1m`, `1min`). |
| `--source_dir` | `--raw_dir` | `~/.qlib/stock_data/source/us_data` | Granular override for raw CSV output directory. |
| `--normalize_dir` | | `~/.qlib/stock_data/source/us_1d_nor` | Granular override for normalized CSV output directory. |
| `--qlib_dir` | | `~/.qlib/qlib_data/us_data` | Granular override for dumped Qlib binary output directory. |
| `--dump_qlib` | | `True` | Convert normalized data into Qlib binary format (`.bin`). |
| `--no-dump_qlib` | | | Skip binary dumping stage (saves raw/normalized CSVs only). |
| `--delay` | | `0.5` | Delay in seconds between ticker downloads to avoid rate limits. |

---

## 4. Generated Directory Layout

When running with `--target_dir <TARGET_DIR>`:

```
<TARGET_DIR>/
├── source/                          # Raw CSV files
│   ├── MSFT.csv
│   └── NVDA.csv
├── normalize/                       # Normalized CSV files
│   ├── MSFT.csv
│   └── NVDA.csv
└── qlib_data/                       # Ready-to-use Qlib binaries
    ├── calendars/
    │   └── day.txt                  # Trading day calendar sequence
    ├── instruments/
    │   └── all.txt                  # Symbol, start_date, end_date
    └── features/
        └── MSFT/
            ├── open.day.bin         # float32 binary arrays
            ├── high.day.bin
            ├── low.day.bin
            ├── close.day.bin
            ├── volume.day.bin
            ├── factor.day.bin
            └── change.day.bin
```

---

## 5. Python API Usage

You can also import and invoke the pipeline programmatically in your own scripts:

```python
from pathlib import Path
from download_us_selected_data import run_pipeline

results = run_pipeline(
    symbols=["MSFT", "NVDA", "SMH"],
    target_dir=Path("D:/trading/qlib"),
    start="2010-01-01",
    dump_qlib=True,
)

print(f"Successfully processed {len(results)} symbols.")
```

