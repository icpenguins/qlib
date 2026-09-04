# Function Specification: `orthogonalize_gsi_factors`

## 1. Overview & Single Responsibility
- **File**: [`qlib/contrib/derivatives/factor_orthogonalization.py`](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/derivatives/factor_orthogonalization.py)
- **Module Classification**: `# my-qlib Fork Extension: Institutional Microstructure & Derivatives Layer`
- **Single Responsibility**: Orthogonalizes cross-sectional Gamma Squeeze Index scores against confounding equity risk factors (Size, Momentum, Volatility, Short Interest) using Weighted Least Squares (WLS) projection to isolate pure idiosyncratic dealer hedging alpha.

---

## 2. Mathematical Formulation
$$\mathbf{GSI}_{\text{orth}} = \left( \mathbf{I} - \mathbf{X} (\mathbf{X}^T \mathbf{\Omega}^{-1} \mathbf{X})^{-1} \mathbf{X}^T \mathbf{\Omega}^{-1} \right) \mathbf{GSI}$$
$$\mathbf{X}^T \mathbf{\Omega}^{-1} \mathbf{GSI}_{\text{orth}} \equiv \mathbf{0}$$

---

## 3. Function Signature
```python
def orthogonalize_gsi_factors(
    gsi_series: Union[np.ndarray, list],
    factor_matrix: Union[np.ndarray, list],
    residual_variances: Optional[Union[np.ndarray, list]] = None,
) -> np.ndarray:
```

---

## 4. Invariants & Testing Accuracy
- Tested in [`tests/test_factor_orthogonalization.py`](file:///e:/SRC/GITHUB/my-qlib/tests/test_factor_orthogonalization.py).
- Invariant: Orthogonalized residual vector has zero inner product with all columns of $\mathbf{X}$ within machine precision ($|\mathbf{x}_k^T \mathbf{e}| < 10^{-4}$).
- Invariant: Degenerate cross-sections ($N \le K$) gracefully fall back to zero-mean demeaning without matrix singular errors.

