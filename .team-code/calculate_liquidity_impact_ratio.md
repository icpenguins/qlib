# Function Specification: `calculate_liquidity_impact_ratio`

## 1. Overview & Single Responsibility
- **File**: [`qlib/contrib/derivatives/liquidity_impact_ratio.py`](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/derivatives/liquidity_impact_ratio.py)
- **Module Classification**: `# my-qlib Fork Extension: Institutional Microstructure & Derivatives Layer`
- **Single Responsibility**: Computes the Liquidity Impact Ratio (LIR), benchmarking forced dealer hedging share volume against the physical liquidity depth of the underlying equity.

---

## 2. Mathematical Formulation
$$\text{LIR}(\Delta S) = \frac{|\mathcal{D}(\Delta S)|}{\text{ADTV}_{20} \times \lambda_{\text{depth}}}$$

---

## 3. Function Signature
```python
def calculate_liquidity_impact_ratio(
    shares_demand: float,
    adtv_20: float,
    depth_factor: float = 0.10,
) -> float:
```

---

## 4. Invariants & Testing Accuracy
- Tested in [`tests/test_liquidity_impact_ratio.py`](file:///e:/SRC/GITHUB/my-qlib/tests/test_liquidity_impact_ratio.py).
- Invariant: Strictly monotonic with absolute shares demand.
- Invariant: Handles zero ADTV safely without crash, returning infinity for non-zero demand or 0.0 for zero demand.

