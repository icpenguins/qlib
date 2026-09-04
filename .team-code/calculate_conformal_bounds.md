# Function Specification: `calculate_conformal_bounds`

## 1. Overview & Single Responsibility
- **File**: [`qlib/contrib/derivatives/conformal_prediction_bounds.py`](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/derivatives/conformal_prediction_bounds.py)
- **Module Classification**: `# my-qlib Fork Extension: Institutional Microstructure & Derivatives Layer`
- **Single Responsibility**: Computes non-parametric conformal prediction uncertainty intervals $[p_{\text{lower}}, p_{\text{upper}}]$ providing distribution-free finite-sample coverage guarantees for calibrated squeeze probabilities.

---

## 2. Mathematical Formulation
$$s_i = |y_i - \widehat{p}_i|$$
$$\widehat{q}_{1-\alpha} = \text{Quantile}\left(\{s_i\}_{i=1}^n, \, \frac{\lceil (n+1)(1-\alpha) \rceil}{n}\right)$$
$$C(X_{n+1}) = \left[ \max\left(0.0, \, \widehat{p} - \widehat{q}_{1-\alpha} \cdot \frac{1-\alpha}{0.90}\right), \; \min\left(1.0, \, \widehat{p} + \widehat{q}_{1-\alpha} \cdot \frac{1-\alpha}{0.90}\right) \right]$$

---

## 3. Function Signature
```python
def calculate_conformal_bounds(
    calibrated_prob: float,
    confidence_level: float = 0.90,
    residual_quantile: float = 0.08,
) -> Tuple[float, float]:
```

---

## 4. Invariants & Testing Accuracy
- Tested in [`tests/test_conformal_prediction_bounds.py`](file:///e:/SRC/GITHUB/my-qlib/tests/test_conformal_prediction_bounds.py).
- Invariant: Bounds always satisfy $0.0 \le p_{\text{lower}} \le \widehat{p} \le p_{\text{upper}} \le 1.0$.
- Invariant: Interval width widens monotonically as target confidence level increases.

