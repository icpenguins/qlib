# Function Specification: `compute_positive_gamma_squeeze_index`

## 1. Overview & Single Responsibility
- **File**: [`qlib/contrib/derivatives/positive_gamma_squeeze.py`](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/derivatives/positive_gamma_squeeze.py)
- **Module Classification**: `# my-qlib Fork Extension: Institutional Microstructure & Derivatives Layer`
- **Single Responsibility**: Computes the raw continuous Positive Gamma Squeeze Index score (GSI+) in $[0.0, 100.0]$ predicting next-day to next-week forced dealer buying pressure following positive earnings surprises.

---

## 2. Mathematical Formulation
$$\text{Logit}^+ = 1.5 \cdot \text{LIR}_{\text{bull}} + 1.2 \cdot \tanh\left(\frac{\text{SUE}}{2}\right) + 0.6 \cdot \min\left(5.0, \frac{\text{OI}_{\text{call, OTM}}}{\max(1, \text{OI}_{\text{put, ATM}})}\right) + 3.0 \cdot (\text{SI} - 0.05)$$
$$\text{GSI}^+ = \frac{100.0}{1.0 + \exp(-\text{Logit}^+)}$$

---

## 3. Function Signature
```python
def compute_positive_gamma_squeeze_index(
    lir_bull: float,
    sue_score: float,
    call_oi_otm: float,
    put_oi_atm: float,
    short_interest_pct: float,
) -> Dict[str, Any]:
```

---

## 4. Invariants & Testing Accuracy
- Tested in [`tests/test_positive_gamma_squeeze.py`](file:///e:/SRC/GITHUB/my-qlib/tests/test_positive_gamma_squeeze.py).
- Invariant: Continuous output is strictly bounded within $[0.0, 100.0]$.
- Invariant: Alerts trigger only when $\text{GSI}^+ \ge 75.0$, activating the `AGGRESSIVE_BULL_GAMMA_SQUEEZE` action flag for $t+1$ to $t+5$ holding horizons.

