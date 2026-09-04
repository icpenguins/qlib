# Function Specification: `calculate_deflated_sharpe_ratio`

## 1. Overview & Single Responsibility
- **File**: [`qlib/contrib/backtest/deflated_sharpe_ratio.py`](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/backtest/deflated_sharpe_ratio.py)
- **Module Classification**: `# my-qlib Fork Extension: Institutional Microstructure & Derivatives Layer`
- **Single Responsibility**: Computes the Bailey & López de Prado (2014) Deflated Sharpe Ratio (DSR), penalizing for multiple testing selection bias, non-normal returns (skewness/kurtosis), and sample track record length with dynamically computed hurdles.

---

## 2. Mathematical Formulation
$$\mathbb{E}\left[\max_{n=1 \dots N} \text{SR}_n\right] \approx \sigma_{\text{SR}} \left( \sqrt{2 \ln N} + \frac{\gamma_{\text{EM}}}{\sqrt{2 \ln N}} \right) \quad (\gamma_{\text{EM}} \approx 0.5772)$$
$$\widehat{\sigma}_{\text{SR}} = \sqrt{\frac{1 - \gamma_3 \cdot \text{SR} + \frac{\gamma_4 - 1}{4} \cdot \text{SR}^2}{T - 1}}$$
$$z = \frac{\text{SR} - \mathbb{E}[\max \text{SR}_0]}{\widehat{\sigma}_{\text{SR}}}$$
$$\text{DSR} = \Phi(z)$$

---

## 3. Function Signature
```python
def calculate_deflated_sharpe_ratio(
    trial_matrix: Union[np.ndarray, list],
    benchmark_sharpe: float = 0.0,
    annualization_factor: float = 252.0,
) -> Dict[str, Any]:
```

---

## 4. Invariants & Testing Accuracy
- Tested in [`tests/test_deflated_sharpe_ratio.py`](file:///e:/SRC/GITHUB/my-qlib/tests/test_deflated_sharpe_ratio.py).
- Invariant: Hurdle $\mathbb{E}[\max(\text{SR}_0)]$ is computed dynamically from the trial matrix dimension $N$ and empirical Sharpe variance, without pre-written scalars.
- Invariant: High trial variance collapses DSR significance, penalizing selection bias.

