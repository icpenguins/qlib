# Function Specification: `resolve_json_path`

## Location
- Defined in: [`scripts/stock_analysis_data.py`](file:///E:/SRC/GITHUB/my-qlib/scripts/stock_analysis_data.py)
- Re-exported in: [`scripts/visualize_stock_analysis.py`](file:///E:/SRC/GITHUB/my-qlib/scripts/visualize_stock_analysis.py)

---

## Signature
```python
def resolve_json_path(
    symbol: str,
    report_dir: Optional[Union[str, Path]] = None,
    output: Optional[Union[str, Path]] = None,
    report_date: Optional[Union[str, datetime.date, datetime.datetime]] = None,
) -> Path
```

---

## Purpose & Description
Resolves the canonical on-disk file path for the stock analysis JSON data contract. It ensures consistent naming conventions matching companion HTML visualization reports (`<SYMBOL>_analysis_report_<DATE>.json`), handles relative/absolute directory expansions, and creates parent directories as needed.

---

## Behavior & Resolution Rules
1. **Ticker & Date Formatting**:
   - The symbol is capitalized (e.g., `aapl` $\rightarrow$ `AAPL`).
   - The report date defaults to the current UTC/local day (`YYYY-MM-DD`). If a `datetime` or `date` instance is passed, it is formatted to `YYYY-MM-DD`.
2. **Explicit Output Handling (`output`)**:
   - If `output` ends in `.json` (case-insensitive), it is treated as an explicit file path and resolved directly.
   - If `output` ends in `.html` (case-insensitive), the suffix is replaced with `.json`.
   - If `output` is a directory or path without extension, it creates `<output>/<SYMBOL>_analysis_report_<DATE>.json`.
3. **Directory Fallback (`report_dir`)**:
   - If `output` is not specified, defaults to `<report_dir>/<SYMBOL>_analysis_report_<DATE>.json` (where `report_dir` defaults to `'reports'`).
4. **Directory Auto-Creation**:
   - Automatically executes `mkdir(parents=True, exist_ok=True)` on parent directories to prevent `FileNotFoundError` during export.

---

## Return Value
- `Path`: Fully resolved `pathlib.Path` pointing to the destination JSON file.

