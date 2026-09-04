# Function Specification: `compute_negative_gamma_squeeze_index`

## 1. Overview & Single Responsibility
- **File**: [`qlib/contrib/derivatives/negative_gamma_squeeze.py`](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/derivatives/negative_gamma_squeeze.py)
- **Module Classification**: `# my-qlib Fork Extension: Institutional Microstructure & Derivatives Layer`
- **Single Responsibility**: Computes the raw continuous Negative Gamma Squeeze / Liquidation Cascade Index score (GSI-) in $[0.0, 100.0]$ predicting next-day to next-week forced dealer dumping pressure following earnings misses.

---

## 2. Mathematical Formulation
$$\text{Logit}^- = 1.6 \cdot \text{LIR}_{\text{bear}} + 1.3 \cdot \tanh\left(\frac{-\text{SUE}}{2}\right) + 1.5 \cdot \mathbb{I}_{S < S^*} + 1.2 \cdot \mathbb{I}_{\text{void}}$$
$$\text{GSI}^- = \frac{100.0}{1.0 + \exp(-\text{Logit}^-)}$$

---

## 3. Function Signature
```python
def compute_negative_gamma_squeeze_index(
    lir_bear: float,
    sue_score: float,
    spot: float,
    gamma_flip_price: float,
    in_liquidity_void: bool = False,
) -> Dict[str, Any]:
```

---

## 4. Invariants & Testing Accuracy
- Tested in [`tests/test_negative_gamma_squeeze.py`](file:///e:/SRC/GITHUB/my-qlib/tests/test_negative_gamma_squeeze.py).
- Invariant: Continuous output is strictly bounded within $[0.0, 100.0]$.
- Invariant: Breaching the gamma flip point ($S < S^*$) or volume profile void dramatically accelerates the cascade score, triggering `LIQUIDATION_CASCADE_ALERT` at $\ge 75.0$.

