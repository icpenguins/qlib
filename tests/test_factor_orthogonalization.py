"""Unit tests for orthogonalize_gsi_factors."""

import unittest
import numpy as np
from qlib.contrib.derivatives.factor_orthogonalization import orthogonalize_gsi_factors


class TestFactorOrthogonalization(unittest.TestCase):
    def test_orthogonality_condition(self):
        np.random.seed(42)
        n = 100
        # Simulated factors: Size, Momentum, Volatility, Short Interest
        factor_size = np.random.randn(n)
        factor_mom = np.random.randn(n)
        factor_vol = np.abs(np.random.randn(n))
        X = np.column_stack([factor_size, factor_mom, factor_vol])

        # GSI deliberately constructed with factor confounding
        true_alpha = np.random.randn(n)
        gsi_confounded = 2.0 * factor_size - 1.5 * factor_mom + 0.8 * factor_vol + true_alpha

        # Orthogonalize
        gsi_orth = orthogonalize_gsi_factors(gsi_confounded, X)

        # Invariant: Inner product with each factor column must be zero (within floating tolerance)
        for col_idx in range(X.shape[1]):
            corr = np.corrcoef(gsi_orth, X[:, col_idx])[0, 1]
            self.assertAlmostEqual(corr, 0.0, places=5)

    def test_wls_weights(self):
        n = 50
        X = np.random.randn(n, 2)
        y = np.random.randn(n)
        variances = np.random.uniform(0.5, 2.0, size=n)

        residuals = orthogonalize_gsi_factors(y, X, residual_variances=variances)
        self.assertEqual(len(residuals), n)


if __name__ == "__main__":
    unittest.main()

