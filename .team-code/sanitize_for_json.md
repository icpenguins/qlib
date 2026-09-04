# Function Specification: `_sanitize_for_json`

## Location
- Defined in: [`scripts/stock_analysis_data.py`](file:///E:/SRC/GITHUB/my-qlib/scripts/stock_analysis_data.py)
- Re-exported in: [`scripts/visualize_stock_analysis.py`](file:///E:/SRC/GITHUB/my-qlib/scripts/visualize_stock_analysis.py)

---

## Signature
```python
def _sanitize_for_json(obj: Any) -> Any
```

---

## Purpose & Description
Performs deep recursive sanitization of arbitrarily nested Python data structures, converting proprietary mathematical types (NumPy scalars, Pandas DataFrames/Series, Timestamps) and non-finite IEEE 754 floating point numbers (`NaN`, `Inf`, `-Inf`) into native JSON-serializable primitives.

---

## Type Conversion Rules
| Input Type | Output Type / Value | Notes |
| :--- | :--- | :--- |
| `None` | `None` (`null`) | Preserved |
| `bool`, `str` | `bool`, `str` | Preserved |
| `int`, `np.integer` | `int` | Casts `np.int32`, `np.int64` to Python `int` |
| `float`, `np.floating` | `float` or `None` | Non-finite values (`np.isnan`, `np.isinf`) convert to `None` (`null`) to comply with strict JSON specs |
| `datetime.date`, `datetime.datetime`, `pd.Timestamp` | `str` | Formatted via `.isoformat()` |
| `dict` | `dict` | Keys coerced to `str`, values recursively sanitized |
| `list`, `tuple`, `set`, `np.ndarray` | `list` | Elements recursively sanitized |
| `pd.DataFrame` | `list` of `dict` | Transformed via `.to_dict(orient="records")` and recursively sanitized |
| `pd.Series` | `list` | Transformed via `.tolist()` and recursively sanitized |
| Other | `str` | Coerced to string via `str(obj)` |

---

## Return Value
- `Any`: Clean JSON-serializable Python data tree ready for `json.dumps()`.

