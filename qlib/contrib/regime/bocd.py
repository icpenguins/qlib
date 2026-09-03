# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Bayesian Online Changepoint Detection (BOCD)
============================================
Exact recursive implementation of the Adams & MacKay (2007) algorithm for online
changepoint detection in financial time series without lookahead or lag.

Paper:
    Ryan Prescott Adams, David J.C. MacKay (2007)
    "Bayesian Online Changepoint Detection", arXiv:0710.3742.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

def _student_t_pdf(x: float, df: np.ndarray, loc: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """
    Zero-dependency Student-t PDF using pure NumPy and math.lgamma, with scipy fallback if available.
    """
    try:
        from scipy import stats
        return stats.t.pdf(x, df=df, loc=loc, scale=scale)
    except ImportError:
        pass

    nu = np.asarray(df, dtype=np.float64)
    mu = np.asarray(loc, dtype=np.float64)
    sig = np.maximum(np.asarray(scale, dtype=np.float64), 1e-12)

    log_gamma_num = np.array([math.lgamma((v + 1.0) / 2.0) for v in nu], dtype=np.float64)
    log_gamma_den = np.array([math.lgamma(v / 2.0) for v in nu], dtype=np.float64)

    z = (x - mu) / sig
    log_pdf = (
        log_gamma_num
        - log_gamma_den
        - 0.5 * np.log(np.pi * nu)
        - np.log(sig)
        - 0.5 * (nu + 1.0) * np.log1p((z ** 2) / nu)
    )
    return np.exp(np.clip(log_pdf, -700.0, 700.0))


class StudentTConjugatePrior:
    """
    Normal-Inverse-Gamma conjugate prior for Gaussian observation likelihood
    with unknown mean and unknown variance. Yields a Student-t predictive distribution.

    Parameters
    ----------
    mu0 : float
        Prior mean.
    kappa0 : float
        Prior precision scale (effective prior sample size for mean).
    alpha0 : float
        Prior shape parameter for inverse-gamma variance (effective degrees of freedom / 2).
    beta0 : float
        Prior scale parameter for inverse-gamma variance (effective sum of squares / 2).
    """

    def __init__(
        self,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
    ):
        self.mu0 = float(mu0)
        self.kappa0 = float(kappa0)
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)

        # Vectors for active run-lengths r = 0, 1, 2, ...
        self.mu = np.array([self.mu0], dtype=np.float64)
        self.kappa = np.array([self.kappa0], dtype=np.float64)
        self.alpha = np.array([self.alpha0], dtype=np.float64)
        self.beta = np.array([self.beta0], dtype=np.float64)

    def reset(self):
        """Reset prior parameters to initial state."""
        self.mu = np.array([self.mu0], dtype=np.float64)
        self.kappa = np.array([self.kappa0], dtype=np.float64)
        self.alpha = np.array([self.alpha0], dtype=np.float64)
        self.beta = np.array([self.beta0], dtype=np.float64)

    def pdf(self, x: float) -> np.ndarray:
        """
        Evaluate predictive Student-t probability density for observation x
        across all current run-lengths.

        Parameters
        ----------
        x : float
            Current observation scalar.

        Returns
        -------
        np.ndarray
            Vector of predictive probabilities P(x_t | r_{t-1}, x^{(r)}).
        """
        df = 2.0 * self.alpha
        scale_sq = (self.beta * (self.kappa + 1.0)) / (self.alpha * self.kappa)
        scale_sq = np.maximum(scale_sq, 1e-12)
        scale = np.sqrt(scale_sq)

        # Compute Student-t density using zero-dependency _student_t_pdf
        probs = _student_t_pdf(x, df=df, loc=self.mu, scale=scale)
        # Numerical guard against underflow
        return np.maximum(probs, 1e-300)

    def update(self, x: float):
        """
        Update conjugate parameters for observation x:
        Appends new parameters for run-length r+1, and prepends prior for r=0.
        """
        new_kappa = self.kappa + 1.0
        new_mu = (self.kappa * self.mu + x) / new_kappa
        new_alpha = self.alpha + 0.5
        new_beta = self.beta + (self.kappa * (x - self.mu) ** 2) / (2.0 * new_kappa)

        # Prepend initial prior for newly initiated run-length r=0
        self.kappa = np.concatenate(([self.kappa0], new_kappa))
        self.mu = np.concatenate(([self.mu0], new_mu))
        self.alpha = np.concatenate(([self.alpha0], new_alpha))
        self.beta = np.concatenate(([self.beta0], new_beta))

    def prune(self, keep_indices: np.ndarray):
        """Prune low-probability run-length states for O(1) step complexity."""
        self.mu = self.mu[keep_indices]
        self.kappa = self.kappa[keep_indices]
        self.alpha = self.alpha[keep_indices]
        self.beta = self.beta[keep_indices]


class ConstantHazard:
    """
    Constant hazard function corresponding to a geometric prior on run-length.

    Parameters
    ----------
    expected_run_length : float
        Expected segment duration in time steps (lambda).
    """

    def __init__(self, expected_run_length: float = 63.0):
        if expected_run_length <= 0:
            raise ValueError("expected_run_length must be positive.")
        self.hazard_rate = 1.0 / float(expected_run_length)

    def __call__(self, r: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        if isinstance(r, np.ndarray):
            return np.full_like(r, self.hazard_rate, dtype=np.float64)
        return self.hazard_rate


class BayesianOnlineChangepointDetector:
    """
    Bayesian Online Changepoint Detector (Adams & MacKay 2007).

    Recursively computes the full posterior distribution over the current run-length
    P(r_t | x_{1:t}) and the instant probability of a structural break P(r_t = 0 | x_{1:t}).

    Parameters
    ----------
    expected_run_length : float, optional
        Expected interval between changepoints in trading days (default 63 = ~1 quarter).
    prior_params : Optional[Dict[str, float]], optional
        Dictionary specifying mu0, kappa0, alpha0, beta0 for Student-t conjugate prior.
    prune_threshold : float, optional
        Minimum posterior mass below which run-lengths are dropped for efficiency (default 1e-5).
    max_run_length : int, optional
        Maximum run-length history to maintain (default 500).
    """

    def __init__(
        self,
        expected_run_length: float = 63.0,
        prior_params: Optional[Dict[str, float]] = None,
        prune_threshold: float = 1e-5,
        max_run_length: int = 500,
    ):
        self.expected_run_length = expected_run_length
        self.prune_threshold = prune_threshold
        self.max_run_length = max_run_length

        if prior_params is None:
            prior_params = {"mu0": 0.0, "kappa0": 1.0, "alpha0": 1.0, "beta0": 1.0}

        self.prior = StudentTConjugatePrior(**prior_params)
        self.hazard = ConstantHazard(expected_run_length=expected_run_length)

        # Initial run-length distribution: P(r_0 = 0) = 1.0
        self.R = np.array([1.0], dtype=np.float64)
        self.step_count = 0

    def reset(self):
        """Reset detector to initial state."""
        self.prior.reset()
        self.R = np.array([1.0], dtype=np.float64)
        self.step_count = 0

    def update(self, x_t: float) -> Tuple[float, float, np.ndarray]:
        """
        Process a single new scalar observation online.

        Parameters
        ----------
        x_t : float
            New observation value.

        Returns
        -------
        Tuple[float, float, np.ndarray]
            (changepoint_prob, expected_run_length, run_length_posterior)
            - changepoint_prob: P(r_t = 0 | x_{1:t}), probability that time t is a changepoint.
            - expected_run_length: Expected duration since last changepoint sum(r * P(r)).
            - run_length_posterior: Full posterior vector across active run lengths.
        """
        # 1. Evaluate predictive probabilities under current run-lengths
        pred_probs = self.prior.pdf(x_t)

        # 2. Evaluate hazard rate
        H = self.hazard(np.arange(len(self.R)))

        # 3. Calculate growth probabilities P(r_t = r_{t-1} + 1, x_{1:t})
        growth_probs = self.R * pred_probs * (1.0 - H)

        # 4. Calculate changepoint probability P(r_t = 0, x_{1:t})
        cp_prob_joint = np.sum(self.R * pred_probs * H)

        # 5. Form joint distribution vector
        R_new = np.concatenate(([cp_prob_joint], growth_probs))

        # 6. Normalize to compute posterior P(r_t | x_{1:t})
        evidence = np.sum(R_new)
        if evidence > 0.0 and not np.isnan(evidence):
            R_new = R_new / evidence
        else:
            R_new = np.zeros_like(R_new)
            R_new[0] = 1.0

        # 7. Update conjugate parameters
        self.prior.update(x_t)

        # 8. Prune negligible tail run-lengths for performance
        if len(R_new) > self.max_run_length or np.any(R_new[1:] < self.prune_threshold):
            keep_mask = R_new >= self.prune_threshold
            keep_mask[0] = True  # Always preserve r=0
            if len(R_new) > self.max_run_length:
                keep_mask[self.max_run_length :] = False

            keep_indices = np.where(keep_mask)[0]
            R_new = R_new[keep_indices]
            R_new = R_new / np.sum(R_new)  # Re-normalize
            self.prior.prune(keep_indices)

        self.R = R_new
        self.step_count += 1

        # Changepoint probability represents the posterior mass of a fresh or newborn run (r_t <= 1)
        cp_prob = float(np.sum(self.R[:2])) if len(self.R) >= 2 else float(self.R[0])
        exp_run_length = float(np.sum(np.arange(len(self.R)) * self.R))

        return cp_prob, exp_run_length, self.R

    def batch_process(
        self,
        series: Union[List[float], np.ndarray, pd.Series],
    ) -> pd.DataFrame:
        """
        Run BOCD sequentially across an entire time series.

        Parameters
        ----------
        series : Union[List[float], np.ndarray, pd.Series]
            Historical time series.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['changepoint_prob', 'expected_run_length', 'max_prob_run_length'].
        """
        self.reset()
        values = np.asarray(series, dtype=np.float64)
        n = len(values)

        cp_probs = np.zeros(n, dtype=np.float64)
        exp_rls = np.zeros(n, dtype=np.float64)
        map_rls = np.zeros(n, dtype=np.int64)

        for i in range(n):
            val = values[i]
            if np.isnan(val):
                # Impute previous or 0
                val = 0.0 if i == 0 else values[i - 1]
            cp_p, exp_rl, post = self.update(val)
            cp_probs[i] = cp_p
            exp_rls[i] = exp_rl
            map_rls[i] = int(np.argmax(post))

        df = pd.DataFrame(
            {
                "changepoint_prob": cp_probs,
                "expected_run_length": exp_rls,
                "map_run_length": map_rls,
            }
        )

        if isinstance(series, pd.Series):
            df.index = series.index

        return df
