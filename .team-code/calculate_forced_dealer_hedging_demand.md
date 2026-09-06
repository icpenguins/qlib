# Function Specification: `calculate_forced_dealer_hedging_demand`

## 1. Overview & Single Responsibility
- **File**: [`qlib/contrib/derivatives/forced_dealer_hedging.py`](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/derivatives/forced_dealer_hedging.py)
- **Module Classification**: `# my-qlib Fork Extension: Institutional Microstructure & Derivatives Layer`
- **Single Responsibility**: Vectorized evaluation of option Greeks across a discrete spot-vol jump scenario grid, quantifying the exact net share re-hedging demand $\mathcal{D}(\Delta S)$ dealers must execute to maintain delta neutrality following an earnings gap.

---

## 2. Mathematical Formulation
$$\Delta_{\text{eff}}(K, \Delta S) = \Delta_{\text{BS}}\left(S_0(1+\Delta S), K, \tau - \Delta t, \sigma(1-\alpha_{\text{crush}}), r\right) - \Delta_{\text{BS}}\left(S_0, K, \tau, \sigma, r\right)$$
$$\mathcal{D}(\Delta S) = \sum_K 100 \cdot \left[ \text{OI}_{\text{call}}(K) \cdot \Delta_{\text{eff}}^{\text{call}}(K, \Delta S) - \text{OI}_{\text{put}}(K) \cdot \Delta_{\text{eff}}^{\text{put}}(K, \Delta S) \right]$$

---

## 3. Function Signature & Parameters
```python
def calculate_forced_dealer_hedging_demand(
    spot: float,
    df_chain: pd.DataFrame,
    adtv_20: float,
    jump_scenarios: List[float] = None,
    iv_crush_ratio: float = 0.40,
    depth_factor: float = 0.10,
) -> Dict[float, Dict[str, float]]:
```

---

## 4. Invariants & Testing Accuracy
- Tested in [`tests/test_forced_dealer_hedging.py`](file:///e:/SRC/GITHUB/my-qlib/tests/test_forced_dealer_hedging.py).
- Invariant: When empty options chain or spot $\le 0$ is passed, returns zero demand across all scenarios.
- Invariant: For call-dominated open interest, positive spot jumps produce strictly positive net share buying demand.
- **Physical open-interest ceiling invariant (added 2026-09-05)**: since
  `BlackScholesGreeks.calc_delta` returns exact-CDF deltas strictly bounded to
  `[0, 1]` for calls and `[-1, 0]` for puts, the largest possible `|delta change|`
  for any single leg is `1.0`. Aggregate `shares_demand` can therefore never
  physically exceed `100 * (total call OI + total put OI)`, even in the most
  extreme single-scenario case. Each scenario result now includes
  `max_physical_shares_demand` (that ceiling) and `invariant_ok` (whether
  `shares_demand` respected it); a violation is also logged via
  `logging.getLogger(__name__).warning(...)`. This was added after an adversarial
  audit found a report where `shares_demand` (7,369,303) was ~4x the ceiling
  implied by the report's own displayed open interest -- the formula itself was
  verified correct (each leg's delta change was individually bounded), so the
  likely cause was a data-source inconsistency upstream (see
  [dealer_gamma_exposure.md](dealer_gamma_exposure.md) and
  `scripts/stock_analysis_data.py`'s `synthetic_chain_for_report` sharing fix),
  not this function's math -- this invariant exists to make that class of
  inconsistency loud instead of silently rendering an impossible number.

