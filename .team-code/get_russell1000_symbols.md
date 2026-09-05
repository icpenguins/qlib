# Technical Specification: `get_russell1000_symbols.py`

## 1. Overview & Purpose
`get_russell1000_symbols.py` automates the curation, validation, and serialization of the constituent universe for the **Russell 1000 Index** formatted specifically for Microsoft Qlib's instrument files.

In quantitative equity trading, maintaining a point-in-time, standardized stock universe is critical to avoid survivorship bias and ensure cross-sectional models (like LightGBM on Alpha158) train across representative large-cap and mid-cap US equities.

## 2. Universe Composition
The generator supports two modes:
1. **Full Russell 1000 Universe (~1,000 tickers)**:
   - Primary: S&P 500 (503 large-cap stocks) + S&P MidCap 400 (400 mid-cap stocks) scraped from live authoritative tables.
   - Backup/Supplement: FTSE Russell 1000 historical constituents to guarantee ~1,000 distinct liquid US equities across all 11 GICS sectors.
2. **Curated Seed Universe (`--seed_only`, 60 tickers)**:
   - Covers top-tier mega-cap and sector leaders across Tech, Communication, Discretionary, Financials, Health Care, Industrials, Staples, Energy, Utilities, and Benchmark ETFs (SPY, QQQ, IWB, VOO).
   - Designed for rapid test execution, local verification, and fast continuous integration.

## 3. Qlib File Format Specification
The output files strictly conform to the Microsoft Qlib instrument format (`<provider_uri>/instruments/russell1000.txt`):
```text
AAPL	2015-01-01	2026-09-06
AMZN	2015-01-01	2026-09-06
GOOGL	2015-01-01	2026-09-06
MSFT	2015-01-01	2026-09-06
NVDA	2015-01-01	2026-09-06
...
```
Each row contains:
- `SYMBOL`: Ticker symbol (uppercase, hyphen-separated for multi-class shares like `BRK-B`).
- `START_DATE`: Active start trading date (default `2015-01-01`).
- `END_DATE`: Active end trading date.

## 4. Destination File Paths
The script automatically writes and mirrors the instrument file to two locations:
1. **Repository Versioned Path**: `<repo_root>/data/instruments/russell1000.txt`
2. **Qlib Active Data Provider**: `~/.qlib/qlib_data/us_data/instruments/russell1000.txt`

## 5. Usage & Command Line Interface
```bash
# Generate curated 60-stock seed universe for rapid testing:
python scripts/get_russell1000_symbols.py --seed_only

# Generate full 1,000-stock Russell 1000 universe:
python scripts/get_russell1000_symbols.py

# Custom date ranges and destinations:
python scripts/get_russell1000_symbols.py --start_date 2010-01-01 --output_dir ./my_instruments
```

