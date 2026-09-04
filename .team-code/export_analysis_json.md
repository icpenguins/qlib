# Function Specification: `export_analysis_json`

## Location
- Defined in: [`scripts/stock_analysis_data.py`](file:///E:/SRC/GITHUB/my-qlib/scripts/stock_analysis_data.py)
- Re-exported in: [`scripts/visualize_stock_analysis.py`](file:///E:/SRC/GITHUB/my-qlib/scripts/visualize_stock_analysis.py)

---

## Signature
```python
def export_analysis_json(
    analysis_data: Dict[str, Any],
    json_path: Union[str, Path],
    indent: int = 2,
) -> Path
```

---

## Purpose & Description
Serializes the canonical stock analysis payload and exports it to a `.json` file on disk. Implements Step 1 of the decoupled analytical workflow, persisting data independently of HTML generation.

---

## Parameters
- `analysis_data`: Raw or pre-sanitized analysis dictionary.
- `json_path`: File destination path (`str` or `pathlib.Path`).
- `indent`: JSON indentation level for pretty-printing (default: `2`).

---

## Behavior
1. Resolves and expands target `json_path`.
2. Creates parent directories automatically (`parents=True, exist_ok=True`).
3. Executes `prepare_analysis_json_payload(analysis_data)` to enforce schema v1.0.0.
4. Serializes via `json.dumps(..., ensure_ascii=False)`.
5. Writes to disk using `utf-8` encoding.
6. Emits an institutional log message indicating successful export.

---

## Return Value
- `Path`: The fully resolved `pathlib.Path` to the exported JSON file.

