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

