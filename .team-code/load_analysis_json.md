# Function Specification: `load_analysis_json`

## Location
- Defined in: [`scripts/stock_analysis_data.py`](file:///E:/SRC/GITHUB/my-qlib/scripts/stock_analysis_data.py)
- Re-exported in: [`scripts/visualize_stock_analysis.py`](file:///E:/SRC/GITHUB/my-qlib/scripts/visualize_stock_analysis.py)

---

## Signature
```python
def load_analysis_json(json_path: Union[str, Path]) -> Dict[str, Any]
```

---

## Purpose & Description
Reads, decodes, and deserializes a previously exported analysis `.json` file from disk into Python memory. Implements Step 2 consumption for downstream rendering, backtesting metrics extraction, or batch algorithmic execution.

---

## Parameters
- `json_path`: File destination path (`str` or `pathlib.Path`).

---

## Error Handling & Exceptions
- Checks file existence via `json_file.exists()`.
- Raises `FileNotFoundError(f"Analysis JSON file not found: {json_file}")` if the target path does not exist on disk.
- Decodes with explicit `utf-8` encoding.

---

## Return Value
- `Dict[str, Any]`: Fully deserialized analysis dictionary matching contract v1.0.0.

