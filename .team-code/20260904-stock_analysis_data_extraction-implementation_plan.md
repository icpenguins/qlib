# Implementation Plan: Extraction of `stock_analysis_data.py` & JSON Pipeline Decoupling

## Overview & Background
Following the initial implementation of the two-step reporting pipeline in `scripts/visualize_stock_analysis.py`, the JSON serialization and data export engine is currently co-located with the visualization and HTML dashboard generation logic.

This plan details the surgical extraction of the canonical JSON data contract and export functions into a dedicated, standalone module: `scripts/stock_analysis_data.py`. This decoupling enables headless data pipelines, quantitative backtesting harnesses, and programmatic consumers (such as institutional analytics engines) to generate and consume the comprehensive stock analysis data contract without incurring HTML rendering overhead. `scripts/visualize_stock_analysis.py` will then import these capabilities directly from `stock_analysis_data.py`, preserving 100% backward compatibility and visual fidelity.

---

## Stakeholder & End-User Guarantees

### Dual End-User Alignment
Per `.team-code/requirements.md` (Priority -1), the team explicitly acknowledges and designs for our dual primary end-users:
1. **The Profitable Stock Trader**: Demands immediate, reliable access to actionable alpha signals, AVWAP anchored volume profiles, post-earnings drift (PEAD) states, and predictive target levels with zero runtime lag or dependency bloat. A standalone JSON CLI allows fast terminal piping and automated alerts.
2. **The Institutional Hedge Fund Manager**: Mandates strict factor data contracts, reproducible schema versioning (`contract_version: "1.0.0"`), clean separation of concerns between data transformation and presentation layers, and seamless CI/CD orchestration for quantitative risk models without browser or web server dependencies.

### Team Collaboration
- **Senior Refactoring Specialist**: Ensures zero regression, clean code separation, single-responsibility principle adherence, and DRY architecture.
- **The Architect**: Enforces canonical schema contracts, decoupled pipeline topology, and strict interface stability across modules.
- **The Principal Developer**: Verifies complete cross-platform execution (Windows/Linux), UTF-8 encoding integrity, path sanitization, and exhaustive test coverage across the 60+ institutional test suite.

---

## User Review Required

> [!IMPORTANT]
> **Zero Breaking Changes to Existing APIs**:
> `scripts/visualize_stock_analysis.py` will re-export `resolve_json_path`, `_sanitize_for_json`, `prepare_analysis_json_payload`, `export_analysis_json`, and `load_analysis_json` so that external callers or existing tests importing from `visualize_stock_analysis` will experience zero disruption.

> [!NOTE]
> **Standalone CLI Support**:
> `scripts/stock_analysis_data.py` will feature an institutional-grade CLI with flags: `--symbol` (`-s`), `--data_dir` (`-d`), `--report_dir` (`-r`), `--output` (`-o`), `--days_forecast`, `--auto_download`, `--start`, `--request_date`, `--indent`, and `--quiet`.

---

## Open Questions
None. The scope is well-defined and aligns directly with the established two-step pipeline architecture.

---

## Proposed Changes

```
┌───────────────────────────────────────────────────────────┐
│              scripts/stock_analysis_engine.py             │
│        (Multi-model analytics, GEX, PEAD, BOCD, AVWAP)    │
└─────────────────────────────┬─────────────────────────────┘
                              │ Returns raw dict & DataFrames
                              ▼
┌───────────────────────────────────────────────────────────┐
│               scripts/stock_analysis_data.py              │
│  - resolve_json_path()                                    │
│  - _sanitize_for_json()                                   │
│  - prepare_analysis_json_payload()                        │
│  - export_analysis_json()                                 │
│  - load_analysis_json()                                   │
│  - generate_stock_analysis_data()                         │
│  - CLI: python stock_analysis_data.py -s AAPL -o ...      │
└─────────────────────────────┬─────────────────────────────┘
                              │ Canonical .json file & dict
                              ▼
┌───────────────────────────────────────────────────────────┐
│            scripts/visualize_stock_analysis.py            │
│  - Imports data functions from stock_analysis_data        │
│  - Reads canonical .json                                  │
│  - Embeds JSON inside <script id="report-data">           │
│  - Renders interactive HTML report                        │
└───────────────────────────────────────────────────────────┘
```

---

### Component 1: Data Contract Engine & Standalone CLI
#### [NEW] `scripts/stock_analysis_data.py`
- Core functions to implement:
  - `resolve_json_path(symbol, report_dir=None, output=None, report_date=None) -> Path`: Resolves standard destination path (`<dir>/<SYMBOL>_analysis_report_<DATE>.json`).
  - `_sanitize_for_json(obj: Any) -> Any`: Recursively handles `NaN`, `Inf`, NumPy scalars, Pandas DataFrames/Series, and ISO timestamps.
  - `prepare_analysis_json_payload(analysis_data: Dict[str, Any]) -> Dict[str, Any]`: Assembles the structured dictionary matching schema v1.0.0 (metadata, historical OHLCV + indicators, performance periods, predictive buy timing, multi-period projections, regime, microstructure, derivatives, events).
  - `export_analysis_json(analysis_data: Dict[str, Any], json_path: Union[str, Path], indent: int = 2) -> Path`: Writes sanitized payload to disk with UTF-8 encoding.
  - `load_analysis_json(json_path: Union[str, Path]) -> Dict[str, Any]`: Reads and validates JSON payload from disk.
  - `generate_stock_analysis_data(symbol, data_dir=None, report_dir=None, output=None, forecast_days=63, auto_download=True, start="2000-01-01", request_date=None, indent=2) -> Path`: High-level wrapper executing `stock_analysis_engine.run_stock_analysis` and exporting the JSON dataset.
- CLI Entrypoint:
  - Full `argparse` suite supporting `--symbol`, `--data_dir`, `--report_dir`, `--output`, `--days_forecast`, `--auto_download`, `--start`, `--request_date`, `--indent`, `--quiet`.
  - Institutional summary banner output upon successful completion.

---

### Component 2: Visualization Engine Refactoring
#### [MODIFY] `scripts/visualize_stock_analysis.py`
- Remove redundant implementations of lines 77–224 (`resolve_json_path`, `_sanitize_for_json`, `prepare_analysis_json_payload`, `export_analysis_json`, `load_analysis_json`).
- Import these functions from `stock_analysis_data`:
  ```python
  from stock_analysis_data import (
      resolve_json_path,
      _sanitize_for_json,
      prepare_analysis_json_payload,
      export_analysis_json,
      load_analysis_json,
      generate_stock_analysis_data,
  )
  ```
- Expose them in `__all__` or module namespace for backward compatibility.
- Streamline `main()` in `visualize_stock_analysis.py` to leverage `generate_stock_analysis_data` or the imported `export_analysis_json` function cleanly.

---

### Component 3: Test Suites & Verification
#### [NEW] `tests/test_stock_analysis_data.py`
- Dedicated unit test suite verifying:
  1. `resolve_json_path` resolution with default dirs, custom dirs, and explicit output paths.
  2. `_sanitize_for_json` edge cases (`np.nan`, `np.inf`, `np.int64`, `pd.Timestamp`, `pd.DataFrame`).
  3. `prepare_analysis_json_payload` schema compliance and versioning.
  4. `export_analysis_json` and `load_analysis_json` round-trip serialization.
  5. Standalone CLI execution via `subprocess.run([sys.executable, "scripts/stock_analysis_data.py", ...])`.

#### [MODIFY] `scripts/run_all_tests.py`
- Register `"data": ("Stock Analysis JSON Data Contract Pipeline", "tests.test_stock_analysis_data")` in `CORE_SUITES`.
- Ensure all test suites pass with 0 failures and 0 errors.

---

### Component 4: Architecture & Documentation (Part 2 Requirement)
#### [NEW] `.team-code/20260904-stock_analysis_data_extraction-implementation_plan.md`
- Committed record of this implementation plan within `.team-code/`.

#### [NEW] `.team-code/stock_analysis_data.md`
- Comprehensive specification covering function signatures, data contracts, schema v1.0.0 dictionary fields, CLI usage examples, and integration guidelines.

#### [MODIFY] `.team-code/visualize_stock_analysis_two_step_pipeline.md`
- Update architecture diagrams and call flows to reflect the extraction of `stock_analysis_data.py`.

#### [MODIFY] `walkthrough.md`
- Detailed walkthrough summarizing changes, verification runs, and CLI demonstrations.

---

## Verification Plan

### Automated Tests
- Syntax compilation check:
  ```powershell
  & "E:\SRC\GITHUB\my-qlib\.venv\Scripts\python.exe" -m py_compile scripts/stock_analysis_data.py scripts/visualize_stock_analysis.py
  ```
- Run new dedicated data tests:
  ```powershell
  & "E:\SRC\GITHUB\my-qlib\.venv\Scripts\python.exe" -m pytest tests/test_stock_analysis_data.py -v
  ```
- Run visualizer refactor tests:
  ```powershell
  & "E:\SRC\GITHUB\my-qlib\.venv\Scripts\python.exe" -m pytest tests/test_visualize_stock_analysis_refactor.py -v
  ```
- Run institutional test suite runner:
  ```powershell
  & "E:\SRC\GITHUB\my-qlib\.venv\Scripts\python.exe" scripts/run_all_tests.py
  ```

### CLI Verification
- Execute `stock_analysis_data.py` CLI in a temporary directory and inspect the resulting `.json` file structure:
  ```powershell
  & "E:\SRC\GITHUB\my-qlib\.venv\Scripts\python.exe" scripts/stock_analysis_data.py --help
  ```

