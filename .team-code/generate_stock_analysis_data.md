# Function Specification: `generate_stock_analysis_data`

## Location
- Defined in: [`scripts/stock_analysis_data.py`](file:///E:/SRC/GITHUB/my-qlib/scripts/stock_analysis_data.py)
- Re-exported in: [`scripts/visualize_stock_analysis.py`](file:///E:/SRC/GITHUB/my-qlib/scripts/visualize_stock_analysis.py)

---

## Signature
```python
def generate_stock_analysis_data(
    symbol: str,
    data_dir: Optional[Union[str, Path]] = None,
    report_dir: Optional[Union[str, Path]] = None,
    output: Optional[Union[str, Path]] = None,
    forecast_days: int = 63,
    auto_download: bool = True,
    start: str = "2000-01-01",
    request_date: Optional[Union[str, datetime.date, datetime.datetime]] = None,
    indent: int = 2,
) -> Path
```

---

## Purpose & Description
High-level programmatic interface that orchestrates the entire multi-model analytical engine and exports the canonical JSON dataset to disk in a single call.

---

## Processing Flow
1. **Directory & Parameter Resolution**:
   - Expands `data_dir` (defaults to `~/.qlib/qlib_data/us_data`).
   - Resolves target JSON path via `resolve_json_path(symbol, report_dir, output, request_date)`.
2. **Analytical Engine Execution**:
   - Calls `stock_analysis_engine.run_stock_analysis()` with specified horizons, date bounds, and auto-download options.
3. **Canonical Export**:
   - Calls `export_analysis_json(raw_analysis, json_path, indent=indent)`.
4. **Audit Logging**:
   - Emits structured completion log with resolved file path.

---

## Return Value
- `Path`: The resolved `pathlib.Path` pointing to the generated canonical JSON dataset.

