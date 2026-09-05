# Team-Code Implementation Plan: Resolving Internal Inconsistencies & Flaws in the Gamma Squeeze Card

**Document ID**: `20260904-fix_gamma_card_internal_inconsistencies_plan`  
**Date**: 2026-09-04 / 2026-09-05  
**Author**: `team-code-architect` & Multi-Agent Team  
**Status**: PENDING REVIEW & USER APPROVAL  

---

## 1. Printed Acknowledgement of End-User Requirements (Priority Requirement -1)

The entire `team-code` software development pipeline hereby formally acknowledges and attests full understanding of the requirements of our primary end-users:

1. **The Profitable Stock Trader** (*Veteran Discretionary & Quantitative Prop Trader*):
   - Mandate: Consistent alpha, capital preservation, exploiting asymmetric risk/reward setups, avoiding catastrophic drawdowns.
   - Requirement: Zero tolerance for contradictory, broken, or physically inverted strike corridors. When a card presents a squeeze setup, the trigger strike must sit strictly *below* the upper squeeze wall so forced dealer delta buying accelerates *into* the wall. Displayed probabilities, GSI scores, dealer shares, and LIR must accurately reflect model calculations rather than collapsing to $0.0\% / 0.0$ while holding an $83.91$ raw score.
2. **The Institutional Hedge Fund Manager** (*CIO / Head of Quantitative Research*):
   - Mandate: Double-digit net annualized returns, Sharpe ratio $> 2.0$, net zero market/factor beta, zero catastrophic drawdown tolerance.
   - Requirement: Mathematical rigor across derivatives surfaces, monotone corridor invariants, strict separation between *Research Simulation Transparency* and *Live Order Authorization*, and reproducible unit-tested schemas.

---

## 2. Multi-Agent Team Review & Role Perspectives

### Team Architect (`team-code-architect`)
- **Diagnosis**: 
  1. *Corridor Inversion*: In `earnings_gamma_squeeze_engine.py`, `acceleration_corridors` emitted `upper_squeeze_wall = S_0 * (1 + \Delta_{\text{jump}})` ($517.39), but did not emit `trigger_strike`. In `visualize_stock_analysis.py`, `trigger_strike` defaulted to `spot_price * 1.05` ($524.69). Since $524.69 > $517.39, the trigger strike was placed *above* the upper squeeze wall.
  2. *Display Metric Zeroing*: When `is_actionable` was `False` (due to synthetic provenance), `earnings_gamma_squeeze_engine.py` set `p_squeeze_bull = None`. Furthermore, the engine emitted `"gsi_positive_raw": 83.91` without `"gsi_positive"`, and `visualize_stock_analysis.py` only looked for `"gsi_positive"` and `"calibrated_prob_squeeze"`.
  3. *Zero Dealer Demand & Zero LIR*: In `stock_analysis_data.py`, line 289 passed an empty DataFrame `df_chain=pd.DataFrame()`. With no option chain, `calculate_forced_dealer_hedging_demand` naturally returned zeros across all jump scenarios.
- **Architectural Solution**:
  - Enforce the **Corridor Geometry Invariant**:
    $$\text{Spot } (S_0) < \text{Trigger Strike } (K_{\text{trigger}}) < \text{Upper Squeeze Wall } (K_{\text{wall}})$$
  - Decouple **Quantitative Transparency** from **Execution Authorization**:
    - Under synthetic provenance, compute and emit theoretical calibrated probabilities ($P(\text{squeeze}) = \text{calibrate}(GSI^+)$), conformal bounds, and full GSI metrics.
    - Suppress live orders via `is_actionable: False`, `safety_status: "ACTION_SUPPRESSED"`, and `recommended_action: "RESEARCH_ONLY_NO_ACTION"`.
  - Ingest calibrated synthetic option chains via `SyntheticOptionSurfaceGenerator.generate_synthetic_chain(spot_price=spot, annual_vol=vol_21d)` whenever `df_chain` is empty, ensuring dealer demand and LIR are quantitatively evaluated.
  - Standardize schema keys across `earnings_gamma_squeeze_engine.py`, `stock_analysis_data.py`, and `visualize_stock_analysis.py`.

### Program Manager (`team-code-pm`)
- **Product Value**:
  - Eliminates confusion for traders reading the gamma radar.
  - Provides quants and researchers with full simulation metrics without compromising live capital safety invariants.
  - Aligns with institutional risk management standards (e.g. Goldman Sachs/Morgan Stanley derivatives desk risk monitors).

### Principal Developer (`team-code-principal-dev`)
- **Algorithmic Correctness**:
  - Compute jump envelope: $\Delta_{\text{jump}} = \max(\text{expected\_jump\_pct} / 100.0, 0.05)$.
  - Squeeze wall: $K_{\text{wall}} = \max(K_{\text{call\_wall}} \text{ if } K_{\text{call\_wall}} > S_0 \text{ else } 0.0, \text{round}(S_0 \times (1.0 + \Delta_{\text{jump}}), 2))$.
  - Trigger strike: $K_{\text{trigger}} = \text{round}(S_0 + 0.35 \times (K_{\text{wall}} - S_0), 2)$.
  - Assertion check: `assert spot < trigger_strike < upper_squeeze_wall`.
  - Hedging summary payload: Extract top-level `dealer_shares_to_buy`, `dealer_dollar_demand`, `pct_adtv_demand`, and `dealer_hedging_velocity` from scenario grid.

### Senior Developer (`team-code-dev`)
- **Code Standards & Clean Types**:
  - Dual-key compatibility for both raw and normalized keys (`gsi_positive` and `gsi_positive_raw`, `p_positive_squeeze` and `calibrated_prob_squeeze`).
  - Clear comments on corridor invariants and synthetic vs live safety gates.
  - Comprehensive docstrings with typing annotations.

### Quality Assurance Tester (`team-code-qa-tester`)
- **Testing Battery**:
  - Unit tests in `test_earnings_gamma_squeeze_engine.py` verifying:
    1. Corridor invariant $S_0 < K_{\text{trigger}} < K_{\text{wall}}$ across wide spot ranges.
    2. Synthetic provenance retains theoretical $P(\text{squeeze}) > 0$, conformal bounds, and non-zero GSI while asserting `is_actionable == False` and `safety_status == "ACTION_SUPPRESSED"`.
    3. Dealer hedging demand is non-zero when evaluated on synthetic or live chains.
  - Integration tests in `test_visualize_stock_analysis_refactor.py` ensuring HTML render displays theoretical metrics in simulation mode and enforces geometric corridor ordering.
  - Verification run via `run_all_tests.py` ensuring all 82+ tests pass.

### Continuous Integration Developer (`team-code-cicd`)
- **Build & Pipeline Integrity**:
  - Zero external dependency additions.
  - Pure deterministic mock and synthetic chain fixtures in CI.
  - Rapid execution time maintained (< 10s for entire test battery).

---

## 3. Proposed Code Changes

### Component 1: `qlib/contrib/derivatives/earnings_gamma_squeeze_engine.py`
1. **Corridor Calculation**:
   - Compute `upper_squeeze_wall` using call wall or jump envelope.
   - Compute `trigger_strike = round(spot + 0.35 * (upper_squeeze_wall - spot), 2)`.
   - Compute `lower_trapdoor` and `downside_trigger`.
   - Invariant assertion: `assert spot < trigger_strike < upper_squeeze_wall`.
2. **Model Transparency under Synthetic Provenance**:
   - Compute `p_squeeze_bull = calibrate_squeeze_probability(gsi_pos)` and `p_squeeze_bear = calibrate_squeeze_probability(gsi_neg)` regardless of provenance tier.
   - Compute conformal bounds `bounds_bull` and `bounds_bear`.
   - If `not is_actionable`, set `rec_action = "RESEARCH_ONLY_NO_ACTION"`.
3. **Structured Summary & Dual Keys**:
   - `forced_dealer_hedging`: Include `dealer_shares_to_buy`, `dealer_dollar_demand`, `pct_adtv_demand`, `dealer_hedging_velocity`, and `scenarios`.
   - `liquidity_impact`: Include `expected_spread_widening_bps`, `expected_slippage_bps`, and `liquidity_regime`.
   - `gsi_scores`: Include `gsi_positive`, `gsi_positive_raw`, `gsi_negative`, `gsi_negative_raw`, `is_positive_squeeze_candidate`.
   - `calibrated_probabilities`: Include `p_positive_squeeze`, `calibrated_prob_squeeze`, `probability_positive_spike`, and conformal bounds.
   - `acceleration_corridors`: Include `trigger_strike`, `upper_squeeze_wall`, `lower_trapdoor`, `lower_gamma_trap`, and `trigger_distance_pct`.

### Component 2: `scripts/stock_analysis_data.py`
1. **Chain Generation Fallback**:
   - In `prepare_analysis_json_payload`: When `df_chain` is empty, generate calibrated synthetic chain using `SyntheticOptionSurfaceGenerator.generate_synthetic_chain(spot_price=last_price, annual_vol=vol_21d)` so that dealer hedging demand and LIR are quantitatively evaluated instead of collapsing to zero.

### Component 3: `scripts/visualize_stock_analysis.py`
1. **Defensive Parsing & Fallbacks**:
   - Parse `gsi_pos` using `gsi.get("gsi_positive") if gsi.get("gsi_positive") is not None else gsi.get("gsi_positive_raw", 0.0)`.
   - Parse `prob_squeeze` using `calib.get("calibrated_prob_squeeze") or (calib.get("p_positive_squeeze", 0.0) * 100.0) or calib.get("probability_positive_spike", 0.0)`.
   - Parse dealer demand metrics from `forced_dealer_hedging` summary or scenario fallback.
2. **Corridor UI Clamp Invariant**:
   - Enforce: if `trigger_strike >= upper_wall` or `trigger_strike <= spot_price`, recompute `trigger_strike = round(spot_price + 0.35 * (upper_wall - spot_price), 2)` so an inverted corridor can never be rendered in the browser.

### Component 4: Test Suite Updates
1. **`tests/test_earnings_gamma_squeeze_engine.py`**:
   - Update `test_synthetic_suppression_locks_action` to assert `res["is_actionable"] is False`, `res["safety_status"] == "ACTION_SUPPRESSED"`, `res["recommended_action"] == "RESEARCH_ONLY_NO_ACTION"`, while asserting `res["calibrated_probabilities"]["p_positive_squeeze"] > 0` and `res["gsi_scores"]["gsi_positive"] > 0`.
   - Add test `test_corridor_geometric_ordering_invariant` asserting `spot < trigger_strike < upper_squeeze_wall`.
2. **`tests/test_visualize_stock_analysis_refactor.py`**:
   - Verify HTML output renders non-zero research metrics and properly ordered corridors.

---

## 4. Verification Plan
- Run automated unit test suite: `& "e:\SRC\GITHUB\my-qlib\.venv\Scripts\python.exe" "e:\SRC\GITHUB\my-qlib\scripts\run_all_tests.py"`.
- Run sample end-to-end pipeline: generate JSON and HTML for `NVDA` / `TEST` and verify the gamma card displays non-zero theoretical metrics ($P(\text{squeeze}) \approx 84.5\%$, GSI+ $\approx 83.9$), non-zero dealer demand, and valid corridor geometry ($S_0 < K_{\text{trigger}} < K_{\text{wall}}$).

