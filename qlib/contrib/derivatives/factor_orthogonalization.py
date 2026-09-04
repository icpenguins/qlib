# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Factor Orthogonalization Module (WLS Matrix Projection)
======================================================
# my-qlib Fork Extension: Institutional Microstructure & Derivatives Layer

Orthogonalizes cross-sectional Gamma Squeeze Index (GSI) scores against confounding
equity risk factors (MarketCap, Momentum, Volatility, Short Interest) using
Weighted Least Squares (WLS) projection to isolate pure idiosyncratic dealer hedging alpha.
"""

from typing import Optional, Union
import numpy as np


def orthogonalize_gsi_factors(
    gsi_series: Union[np.ndarray, list],
    factor_matrix: Union[np.ndarray, list],
    residual_variances: Optional[Union[np.ndarray, list]] = None,
) -> np.ndarray:
    """
    Applies Weighted Least Squares (WLS) projection:
        GSI_orth = (I - X (X^T Omega^-1 X)^-1 X^T Omega^-1) GSI

    Parameters
    ----------
    gsi_series : Union[np.ndarray, list]
        Vector of raw GSI scores for N assets at timestamp t (shape: (N,)).
    factor_matrix : Union[np.ndarray, list]
        Matrix of cross-sectional factor exposures (e.g. [1, Size, Mom, Vol, SI]) (shape: (N, K)).
    residual_variances : Optional[Union[np.ndarray, list]], optional
        Diagonal of residual covariance matrix Omega. If None, assumes homoscedastic OLS (Omega = I).

    Returns
    -------
    np.ndarray
        Orthogonalized GSI residual vector (shape: (N,)) having zero inner product with X.
    """
    y = np.asarray(gsi_series, dtype=float).ravel()
    X = np.asarray(factor_matrix, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_samples, n_factors = X.shape
    if n_samples != len(y):
        raise ValueError(f"Shape mismatch: y has {len(y)} samples, X has {n_samples} rows.")

    if n_samples <= n_factors:
        # Cannot invert X^T X with insufficient degrees of freedom; demean as fallback
        return y - np.mean(y)

    # Add intercept column if not present
    if not np.allclose(X[:, 0], 1.0):
        X = np.column_stack([np.ones(n_samples), X])
        n_factors += 1

    if residual_variances is not None:
        omega_diag = np.asarray(residual_variances, dtype=float).ravel()
        # Invert variances for weights: W = Omega^-1
        w = 1.0 / np.maximum(1e-6, omega_diag)
        # Scale X and y by sqrt(w)
        W_sqrt = np.sqrt(w)[:, np.newaxis]
        X_w = X * W_sqrt
        y_w = y * np.sqrt(w)
        # Solve WLS: beta = (X^T W X)^-1 X^T W y
        beta, _, _, _ = np.linalg.lstsq(X_w, y_w, rcond=None)
    else:
        # Standard OLS projection
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    # Compute fitted factor component and subtract to isolate orthogonal alpha
    fitted = X @ beta
    residuals = y - fitted

    return np.round(residuals, 6)

