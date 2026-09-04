# Function Specification: `calibrate_post_earnings_volatility`

## 1. Overview & Single Responsibility
- **File**: [`qlib/contrib/derivatives/post_earnings_volatility.py`](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/derivatives/post_earnings_volatility.py)
- **Module Classification**: `# my-qlib Fork Extension: Institutional Microstructure & Derivatives Layer`
- **Single Responsibility**: Decomposes pre-earnings ATM straddle pricing into expected jump magnitude and post-event residual implied volatility using jump-plus-crush variance extraction.

---

## 2. Mathematical Formulation
$$\mathbb{E}[|\Delta S|] \approx \sqrt{\frac{\pi}{2}} \cdot (C_{\text{ATM}} + P_{\text{ATM}}) \approx 0.79788 \cdot \frac{\text{Straddle}}{S_0}$$
$$\sigma^2_{\text{event}} = \frac{(\mathbb{E}[|\Delta S| / S_0])^2}{\tau}$$
$$\sigma^2_{\text{post}} = \max\left(\sigma^2_{\text{realized, 21d}}, \, \sigma^2_{\text{pre}} - \sigma^2_{\text{event}}\right)$$

---

## 3. Function Signature
```python
def calibrate_post_earnings_volatility(
    spot: float,
    atm_straddle_price: float,
    pre_earnings_iv: float,
    realized_21d_vol: float,
    dte_days: int = 7,
) -> Tuple[float, float]:
```

---

## 4. Invariants & Testing Accuracy
- Tested in [`tests/test_post_earnings_volatility.py`](file:///e:/SRC/GITHUB/my-qlib/tests/test_post_earnings_volatility.py).
- Invariant: Post-event implied volatility is strictly bounded below by 21-day realized historical volatility ($\sigma_{\text{post}} \ge \sigma_{\text{realized, 21d}}$).
- Invariant: Degenerate spot ($S_0 \le 0$) or straddle price returns 0.0 expected jump and safe baseline volatility.

