# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Volume Profile Kernel Density Estimation (KDE) Engine
=====================================================
Computes continuous institutional Volume-at-Price profiles, replacing discrete
price bucket heuristics with continuous Gaussian Kernel Density Estimation (KDE).

Key Deliverables:
- Continuous Volume-Weighted Probability Density Function f(p)
- Point of Control (POC) - Global volume mode / institutional fair-value magnet
- 70% Value Area Envelope (VAH - Value Area High, VAL - Value Area Low)
- High-Volume Nodes (HVN) - Liquidity clusters / dynamic support & resistance
- Low-Volume Nodes (LVN / Liquidity Voids) - Fast breakout / air-pocket zones
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd


class VolumeProfileKDE:
    """
    Continuous Volume-at-Price Profiler using Gaussian Kernel Density Estimation.

    Parameters
    ----------
    lookback : int, optional
        Number of trailing trading days to include in the profile (default 63, ~1 quarter).
        Set to None to evaluate over the entire available dataset.
    grid_size : int, optional
        Number of price evaluation points along the continuous density curve (default 250).
    value_area_pct : float, optional
        Percentage of total volume to include in the Value Area (default 0.70 for 70%).
    price_col : str, optional
        Column name for closing price.
    volume_col : str, optional
        Column name for volume.
    high_col : Optional[str], optional
        Column name for high price.
    low_col : Optional[str], optional
        Column name for low price.
    """

    def __init__(
        self,
        lookback: Optional[int] = 63,
        grid_size: int = 250,
        value_area_pct: float = 0.70,
        price_col: str = "close",
        volume_col: str = "volume",
        high_col: Optional[str] = "high",
        low_col: Optional[str] = "low",
    ):
        self.lookback = lookback
        self.grid_size = grid_size
        self.value_area_pct = value_area_pct
        self.price_col = price_col
        self.volume_col = volume_col
        self.high_col = high_col
        self.low_col = low_col

    def _get_price_series(self, df: pd.DataFrame) -> np.ndarray:
        """Extract representative price for volume allocation."""
        if (
            self.high_col in df.columns
            and self.low_col in df.columns
            and self.price_col in df.columns
        ):
            h = df[self.high_col].astype(float).values
            l = df[self.low_col].astype(float).values
            c = df[self.price_col].astype(float).values
            return (h + l + c) / 3.0
        return df[self.price_col].astype(float).values

    def compute_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute continuous volume profile, POC, Value Area, and liquidity voids.

        Parameters
        ----------
        df : pd.DataFrame
            Stock history DataFrame.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'poc': Point of Control price
            - 'vah': Value Area High price (70% boundary)
            - 'val': Value Area Low price (70% boundary)
            - 'current_price': Latest price
            - 'in_value_area': Boolean whether current price is inside Value Area
            - 'dist_to_poc_pct': Percentage distance from current price to POC
            - 'in_liquidity_void': Boolean whether current price is inside an LVN
            - 'hvn_levels': List of High-Volume Node price peaks
            - 'lvn_levels': List of Low-Volume Node troughs
            - 'grid_prices': Evaluated price array (for chart rendering)
            - 'grid_density': Normalized density curve array (for chart rendering)
        """
        if df.empty or len(df) < 5:
            return {}

        sub_df = df.tail(self.lookback) if self.lookback is not None else df
        prices = self._get_price_series(sub_df)
        volumes = sub_df[self.volume_col].astype(float).values
        volumes = np.maximum(volumes, 1.0)
        total_volume = float(np.sum(volumes))

        if total_volume <= 0.0 or len(prices) == 0:
            return {}

        weights = volumes / total_volume
        current_price = float(df[self.price_col].iloc[-1])

        # 1. Optimal Gaussian Kernel Bandwidth via Silverman's Rule of Thumb
        vw_mean = float(np.sum(weights * prices))
        vw_var = float(np.sum(weights * (prices - vw_mean) ** 2))
        vw_std = np.sqrt(max(vw_var, 1e-6))
        n_obs = len(prices)

        # h = 1.06 * sigma_V * N^(-1/5), bounded to avoid overfitting / delta spikes
        h = 1.06 * vw_std * (n_obs ** (-0.2))
        min_h = max(0.005 * vw_mean, 0.05)
        h = max(h, min_h)

        # 2. Continuous Evaluation Price Grid
        if self.low_col in sub_df.columns and self.high_col in sub_df.columns:
            p_min = float(sub_df[self.low_col].min())
            p_max = float(sub_df[self.high_col].max())
        else:
            p_min = float(prices.min())
            p_max = float(prices.max())

        padding = 0.02 * (p_max - p_min if p_max > p_min else 1.0)
        grid = np.linspace(p_min - padding, p_max + padding, self.grid_size)

        # 3. Vectorized Gaussian Kernel Evaluation:
        # f(p) = sum(w_i * (1/(h*sqrt(2pi))) * exp(-0.5 * ((p - P_i)/h)^2))
        diff = grid[:, None] - prices[None, :]  # Shape: (grid_size, n_obs)
        kernel_vals = np.exp(-0.5 * (diff / h) ** 2) / (h * np.sqrt(2.0 * np.pi))
        density = np.sum(weights[None, :] * kernel_vals, axis=1)

        # Normalize density to integrate to 1.0 across grid step
        dp = (grid[-1] - grid[0]) / (self.grid_size - 1)
        total_integral = float(np.sum(density) * dp)
        if total_integral > 0:
            density = density / total_integral

        # 4. Point of Control (POC): Global Density Maximum
        max_idx = int(np.argmax(density))
        poc = float(grid[max_idx])
        max_density = float(density[max_idx])

        # 5. Value Area (70% Cumulative Density Mass)
        # Sort grid cells by descending density to capture the highest-density region
        sort_order = np.argsort(-density)
        mass_accum = 0.0
        target_mass = self.value_area_pct
        va_mask = np.zeros(self.grid_size, dtype=bool)

        for idx in sort_order:
            mass_accum += density[idx] * dp
            va_mask[idx] = True
            if mass_accum >= target_mass:
                break

        va_prices = grid[va_mask]
        val = float(np.min(va_prices))
        vah = float(np.max(va_prices))

        # 6. High-Volume Nodes (HVN) & Low-Volume Nodes (LVN)
        # Identify local extrema in the density curve
        hvn_levels = []
        lvn_levels = []
        for i in range(1, self.grid_size - 1):
            if density[i] > density[i - 1] and density[i] > density[i + 1]:
                # Peak (HVN) if density is above the 50th percentile
                if density[i] > np.median(density):
                    hvn_levels.append(round(float(grid[i]), 2))
            elif density[i] < density[i - 1] and density[i] < density[i + 1]:
                # Trough (LVN) if density is below the 25th percentile
                if density[i] < np.percentile(density, 25):
                    lvn_levels.append(round(float(grid[i]), 2))

        # 7. Current Price Microstructure State
        in_value_area = bool(val <= current_price <= vah)
        dist_to_poc_pct = float(((current_price - poc) / poc) * 100.0) if poc > 0 else 0.0

        # Evaluate density at current price
        curr_diff = current_price - prices
        curr_kernel = np.exp(-0.5 * (curr_diff / h) ** 2) / (h * np.sqrt(2.0 * np.pi))
        current_density = float(np.sum(weights * curr_kernel)) / (total_integral if total_integral > 0 else 1.0)

        # Liquidity void occurs if current density is less than 20% of peak density
        in_liquidity_void = bool(current_density < 0.20 * max_density)

        if in_liquidity_void:
            void_status = "In Liquidity Void (Low Market Depth - High Expected Breakout Velocity)"
        elif in_value_area:
            void_status = "Inside Institutional Value Area (Balanced Fair-Value Zone)"
        elif current_price > vah:
            void_status = "Trading Above Value Area High (Bullish Expansion)"
        else:
            void_status = "Trading Below Value Area Low (Discount / Liquidation Zone)"

        return {
            "lookback_days": len(sub_df),
            "poc": round(poc, 2),
            "vah": round(vah, 2),
            "val": round(val, 2),
            "current_price": round(current_price, 2),
            "in_value_area": in_value_area,
            "dist_to_poc_pct": round(dist_to_poc_pct, 2),
            "in_liquidity_void": in_liquidity_void,
            "void_status": void_status,
            "hvn_levels": hvn_levels[:5],
            "lvn_levels": lvn_levels[:5],
            "bandwidth_h": round(float(h), 3),
        }

