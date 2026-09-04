# Function Specification: `calculate_empirical_sue`

## 1. Overview & Single Responsibility
- **File**: [`qlib/contrib/events/empirical_sue.py`](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/events/empirical_sue.py)
- **Module Classification**: `# my-qlib Fork Extension: Institutional Microstructure & Derivatives Layer`
- **Single Responsibility**: Computes company-specific Standardized Unexpected Earnings (SUE) normalized by the firm's trailing 12-quarter analyst forecast error standard deviation, eliminating arbitrary scaling heuristics.

---

## 2. Mathematical Formulation
$$\text{Surprise}_i = \text{EPS}_{\text{actual}, i} - \text{EPS}_{\text{consensus}, i}$$
$$\sigma_{\text{error}, i} = \sqrt{\frac{1}{N-1} \sum_{q=1}^N (\text{Surprise}_{i, q} - \overline{\text{Surprise}}_i)^2} \quad (N \ge 3)$$
$$\text{SUE}_i = \text{clip}\left( \frac{\text{Surprise}_i}{\max(\sigma_{\text{floor}}, \sigma_{\text{error}, i})}, \, -10.0, \, 10.0 \right)$$

---

## 3. Function Signature
```python
def calculate_empirical_sue(
    actual_eps: float,
    consensus_eps: float,
    historical_forecast_errors: Optional[List[float]] = None,
    min_std_floor: float = 0.02,
) -> float:
```

---

## 4. Invariants & Testing Accuracy
- Tested in [`tests/test_empirical_sue.py`](file:///e:/SRC/GITHUB/my-qlib/tests/test_empirical_sue.py).
- Invariant: Zero surprise ($\text{EPS}_{\text{actual}} = \text{EPS}_{\text{consensus}}$) strictly outputs $0.0$.
- Invariant: Output is bounded within $[-10.0, 10.0]$ with variance floor protection against penny stocks and near-zero forecast dispersion.

