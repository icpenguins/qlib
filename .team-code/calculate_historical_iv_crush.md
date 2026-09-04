# Function Specification: `calculate_historical_iv_crush`

## 1. Overview & Single Responsibility
- **File**: [`qlib/contrib/derivatives/historical_iv_crush.py`](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/derivatives/historical_iv_crush.py)
- **Module Classification**: `# my-qlib Fork Extension: Institutional Microstructure & Derivatives Layer`
- **Single Responsibility**: Computes firm-specific historical IV crush using a 20% Winsorized Median across a minimum of 4 verified observed implied volatility pairs, with term structure slope proxy fallback.

---

## 2. Mathematical Formulation
$$\text{Crush}_q = \frac{\sigma_{\text{pre}, q} - \sigma_{\text{post}, q}}{\sigma_{\text{pre}, q}}$$
$$\widehat{\alpha}_{\text{crush}, i} = \text{WinsorizedMedian}\left( \{\text{Crush}_q\}_{q=1}^{N_{\text{observed}}}, \text{trim}=0.20 \right) \quad (\text{for } N_{\text{observed}} \ge 4)$$
$$\text{Fallback (if } N_{\text{observed}} < 4): \quad \text{Proxy} = \max\left(0.20, \min\left(0.70, 1.0 - \frac{\text{IV}_{M2}}{\text{IV}_{M1}}\right)\right)$$

---

## 3. Function Signature
```python
def calculate_historical_iv_crush(
    observed_iv_pairs: Optional[List[Tuple[float, float]]] = None,
    month1_iv: Optional[float] = None,
    month2_iv: Optional[float] = None,
    trim_pct: float = 0.20,
    min_observed_pairs: int = 4,
) -> Dict[str, Any]:
```

---

## 4. Invariants & Testing Accuracy
- Tested in [`tests/test_historical_iv_crush.py`](file:///e:/SRC/GITHUB/my-qlib/tests/test_historical_iv_crush.py).
- Invariant: If $N_{\text{observed}} < 4$, output explicitly tags `crush_source: "term_structure_proxy"`, rejecting empirical claims.
- Invariant: Winsorization insulates against single-quarter black-swan spikes (e.g. COVID, corporate restructuring).

