# Function Specification: `evaluate_earnings_gamma_squeeze`

## 1. Overview & Single Responsibility
- **File**: [`qlib/contrib/derivatives/earnings_gamma_squeeze_engine.py`](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/derivatives/earnings_gamma_squeeze_engine.py)
- **Module Classification**: `# my-qlib Fork Extension: Institutional Microstructure & Derivatives Layer`
- **Single Responsibility**: Master orchestration engine uniting discrete options Greeks, SUE normalization, Winsorized crush estimation, and provenance safety gates to produce Contract Schema v1.1.0 payloads.

---

## 2. Architectural Pipeline & Output Schema
1. **Safety Gate Verification**: Invokes `validate_data_provenance`.
2. **IV Crush Calibration**: Invokes `calculate_historical_iv_crush`.
3. **Vol & Jump Decomposition**: Invokes `calibrate_post_earnings_volatility`.
4. **Forced Hedging Simulation**: Invokes `calculate_forced_dealer_hedging_demand`.
5. **GSI Computation**: Invokes `compute_positive_gamma_squeeze_index` and `compute_negative_gamma_squeeze_index`.
6. **Probability & Conformal Calibration**: Invokes `calibrate_squeeze_probability` and `calculate_conformal_bounds`.
7. **Acceleration Corridor Construction**: Bounds upper squeeze wall and lower liquidation trapdoor.

---

## 3. Function Signature
```python
def evaluate_earnings_gamma_squeeze(
    spot: float,
    df_chain: pd.DataFrame,
    adtv_20: float,
    sue_score: float = 0.0,
    short_interest_pct: Optional[float] = 0.05,
    gamma_flip_price: float = 0.0,
    provenance: DataProvenance = DataProvenance.HISTORICAL_OPRA_EOD,
    is_pit_timestamp: bool = True,
    observed_iv_pairs: Optional[List[Tuple[float, float]]] = None,
    month1_iv: Optional[float] = None,
    month2_iv: Optional[float] = None,
    realized_21d_vol: float = 0.25,
    atm_straddle_price: Optional[float] = None,
    in_liquidity_void: bool = False,
) -> Dict[str, Any]:
```

---

## 4. Invariants & Testing Accuracy
- Tested in [`tests/test_earnings_gamma_squeeze_engine.py`](file:///e:/SRC/GITHUB/my-qlib/tests/test_earnings_gamma_squeeze_engine.py).
- Invariant: If provenance is `SYNTHETIC_RESEARCH_FALLBACK`, output sets `is_actionable = False`, sets `recommended_action = "RESEARCH_ONLY_NO_ACTION"`, and leaves probability fields `None`.
- Invariant: Generates fully validated Contract Schema v1.1.0 payloads with all required sub-dictionaries.

