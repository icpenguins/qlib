# Function Specification: `calibrate_post_earnings_volatility_surface`

## 1. Overview & Single Responsibility
- **File**: [`qlib/contrib/derivatives/post_earnings_volatility.py`](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/derivatives/post_earnings_volatility.py)
- **Module Classification**: `# my-qlib Fork Extension: Institutional Microstructure & Derivatives Layer`
- **Single Responsibility**: Computes the complete post-earnings volatility surface from pre-announcement ATM straddle pricing, extracting event jump variance, post-earnings residual implied volatility, dollar implied move, and historical volatility crush ratio.

---

## 2. Mathematical Formulation
$$\mathbb{E}[|\Delta S_{\text{jump}}|] \approx \sqrt{\frac{\pi}{2}} \cdot V_{\text{straddle}} \approx 0.79788 \cdot (C_{\text{ATM}} + P_{\text{ATM}})$$
$$\text{Expected Jump \%} = \frac{\mathbb{E}[|\Delta S_{\text{jump}}|]}{S_0} \times 100$$
$$\sigma^2_{\text{event}} = \frac{(\mathbb{E}[|\Delta S_{\text{jump}}| / S_0])^2}{\tau} \quad \left(\tau = \max\left(\frac{1}{365}, \frac{\text{DTE}}{365}\right)\right)$$
$$\sigma_{\text{post}} = \sqrt{\max\left(\sigma^2_{\text{realized, 21d}}, \, \sigma^2_{\text{pre}} - \sigma^2_{\text{event}}\right)}$$
$$\text{Crush Ratio} = \max\left(0.0, \, \frac{\sigma_{\text{pre}} - \sigma_{\text{post}}}{\sigma_{\text{pre}}}\right)$$

---

## 3. Function Signature & Return Schema
```python
def calibrate_post_earnings_volatility_surface(
    spot: float,
    atm_straddle_price: float,
    pre_earnings_iv: float,
    realized_21d_vol: float,
    dte_days: int = 7,
) -> Dict[str, Any]:
```
**Return Payload**:
- `spot`: float
- `atm_straddle_price`: float
- `pre_earnings_iv`: float
- `realized_21d_vol`: float
- `dte_days`: int
- `expected_jump_pct`: float
- `event_variance`: float
- `post_earnings_iv`: float
- `implied_move_dollars`: float
- `volatility_crush_pct`: float
- `volatility_crush_ratio`: float

---

## 4. Invariants & Testing Accuracy
- Tested in [`tests/test_post_earnings_volatility.py`](file:///e:/SRC/GITHUB/my-qlib/tests/test_post_earnings_volatility.py).
- Invariant: Post-earnings implied volatility is bounded below by 21-day realized historical volatility ($\sigma_{\text{post}} \ge \sigma_{\text{realized, 21d}}$).
- Invariant: Degenerate spot ($S_0 \le 0$) or straddle price safely yields 0.0 expected jump and baseline floor volatility without runtime exceptions.

