# Function Specification: `prepare_analysis_json_payload`

## Location
- Defined in: [`scripts/stock_analysis_data.py`](file:///E:/SRC/GITHUB/my-qlib/scripts/stock_analysis_data.py)
- Re-exported in: [`scripts/visualize_stock_analysis.py`](file:///E:/SRC/GITHUB/my-qlib/scripts/visualize_stock_analysis.py)

---

## Signature
```python
def prepare_analysis_json_payload(analysis_data: Dict[str, Any]) -> Dict[str, Any]
```

---

## Purpose & Description
Constructs the canonical schema v1.0.0 payload from raw stock analysis outputs. Unifies disparate models—market microstructure (AVWAP & Volume Profile KDE), Bayesian changepoint regimes (BOCD), derivatives positioning (GEX), forward multi-period projections, and earnings event risk (PEAD)—into a standardized dictionary contract.

---

## Key Processing Operations
1. **Metadata Construction**:
   - Injects canonical `metadata` block containing `symbol`, `request_date`, `latest_data_date`, `is_up_to_date`, `forecast_days`, `generated_at` (ISO timestamp), and `contract_version: "1.0.0"`.
2. **Historical OHLCV Normalization**:
   - If `historical_data` is a pandas DataFrame, computes rolling 50-day and 200-day Simple Moving Averages (`sma50`, `sma200`) if not already present.
   - Extracts date, open, high, low, close, volume, moving averages, and anchored VWAP envelopes (`avwap_ytd`, `avwap_ytd_upper_1s`, `avwap_ytd_lower_1s`).
3. **Multi-Period Projections Generation**:
   - If forward projections (`projections`) are missing, invokes `compute_multi_period_projections()` conditioning on regime, microstructure, and derivatives states.
4. **Deep Sanitization**:
   - Passes the final composite payload through `_sanitize_for_json` to guarantee serialization safety.

---

## Return Value
- `Dict[str, Any]`: Canonical schema dictionary adhering to contract v1.0.0.

