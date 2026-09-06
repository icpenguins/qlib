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

## 5. `council_interrogation_outcomes` -- real per-member verdicts (added 2026-09-05)

`backtest.council_interrogation_outcomes` previously contained only five keys
(`high_earning_trader`, `quant_developer`, `top_hedge_fund_manager`,
`global_finance_manager`, `council_multi_horizon_consensus`) holding static
illustrative numbers. The rendered report (`build_backtesting_protocol_card_html`
in `scripts/visualize_stock_analysis.py`) displays six *named* council members
(Dr. Victoria Vance, Marcus Reynolds, Dr. Elena Rostova, Julian Montgomery,
Sophia Chen, Arthur Pendelton III) looked up by keys (`dr_vance`,
`marcus_reynolds`, `dr_rostova`, `julian_montgomery`, `sophia_chen`,
`arthur_pendelton`) that **did not exist in this payload at all** -- every
`council.get(<name>, {})` returned `{}`, so every member's verdict/notes
unconditionally hit the render side's hardcoded `"APPROVED"` /
`"Quantitative standards validated. Invariants enforced."` defaults, regardless
of any real invariant violation (including the ones an adversarial audit found
elsewhere in the same report -- the dealer-hedging OI-ceiling violation and the
zero-slippage Almgren-Chriss anomaly among them). This made the council section a
structurally-incapable-of-failing rubber stamp.

`_build_council_verdicts(...)` (module-private helper in this file) now computes
each of the six member keys from a real, specific check against that member's
stated audit focus, using values already computed earlier in this function:

| Member | Focus | Verdict driven by |
|---|---|---|
| `dr_vance` | Derivatives & Vol Surface | `guard_result.safety_status` / `is_actionable` |
| `marcus_reynolds` | Execution & Slippage | `forced_dealer_payload.invariant_ok` (the OI-ceiling check, see [calculate_forced_dealer_hedging_demand.md](calculate_forced_dealer_hedging_demand.md)) and `impact_meta.total_cost_bps` being implausibly zero |
| `dr_rostova` | Isotonic Calibration & Ortho | `bounds_bull` well-ordered; `factor_ortho_payload.idiosyncratic_alpha_ratio` in `[0, 1]` |
| `julian_montgomery` | Short Locate & HTB Borrow | `borrow_meta.is_hard_to_borrow` |
| `sophia_chen` | SUE Score & Accounting | `sue_score` not saturated at the +/-10 clip ceiling |
| `arthur_pendelton` | Bottom-Line Capital Allocation | `is_actionable` (ties directly back to the P(Squeeze)-vs-verdict contradiction the same audit flagged) |

This computation must run **after** `forced_dealer_payload` is built (a few
sections later in the function) -- it is merged into
`backtesting_protocol_payload["council_interrogation_outcomes"]` via `.update()`
just before the function's final `return`, not inlined into the dict literal
where the five legacy keys are defined.

The render side (`scripts/visualize_stock_analysis.py`) also no longer hardcodes
"100% Invariant Validation" in the council panel header -- it now counts real
`APPROVED` verdicts out of the six and reports `{n_approved}/{6} Approved`,
switching to amber when fewer than all six approve.

