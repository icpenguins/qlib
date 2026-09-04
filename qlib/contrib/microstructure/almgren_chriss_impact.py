# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Almgren-Chriss Market Impact Module
===================================
# my-qlib Fork Extension: Institutional Microstructure & Derivatives Layer

Calculates temporary and permanent market impact for large-scale institutional order execution
using the non-linear Almgren-Chriss framework:
    Total Impact = Permanent Impact + Temporary Impact
    Permanent Impact = gamma * (v / V)
    Temporary Impact = eta * (v / V)^alpha
"""

from typing import Dict, Any
import numpy as np


class AlmgrenChrissImpactModel:
    """
    Implements Almgren-Chriss institutional market impact modeling.
    """

    def __init__(
        self,
        gamma_perm: float = 0.10,
        eta_temp: float = 0.15,
        alpha: float = 0.50,
        fixed_bps: float = 0.0005,  # 5 bps exchange/clearing
    ):
        self.gamma_perm = gamma_perm
        self.eta_temp = eta_temp
        self.alpha = alpha
        self.fixed_bps = fixed_bps

    def calculate_impact(
        self,
        trade_volume: float,
        adtv: float,
        daily_vol: float,
        spot_price: float = 100.0,
    ) -> Dict[str, float]:
        """
        Calculates price slippage and total execution friction.

        Parameters
        ----------
        trade_volume : float
            Total shares to execute.
        adtv : float
            20-day average daily trading volume.
        daily_vol : float
            Daily price volatility sigma (e.g. 0.02).
        spot_price : float, optional
            Reference spot price, by default 100.0.

        Returns
        -------
        Dict[str, float]
            Dictionary containing:
            - 'participation_rate': v / V
            - 'permanent_impact_bps': Basis points of permanent adverse price move
            - 'temporary_impact_bps': Basis points of temporary execution slippage
            - 'total_cost_bps': Total cost in basis points
            - 'effective_fill_price': Adjusted execution price
        """
        if adtv <= 0 or trade_volume <= 0:
            return {
                "participation_rate": 0.0,
                "permanent_impact_bps": 0.0,
                "temporary_impact_bps": 0.0,
                "total_cost_bps": round(self.fixed_bps * 10000.0, 2),
                "effective_fill_price": spot_price,
            }

        # Participation rate: v / V
        participation = trade_volume / adtv

        # Permanent impact (linear in volume, scales with daily volatility)
        perm_impact_fraction = self.gamma_perm * daily_vol * participation

        # Temporary impact (sub-linear / square-root law: alpha ~ 0.5)
        temp_impact_fraction = self.eta_temp * daily_vol * (participation ** self.alpha)

        # Total slippage fraction
        total_fraction = self.fixed_bps + perm_impact_fraction + temp_impact_fraction
        total_bps = total_fraction * 10000.0

        effective_fill = spot_price * (1.0 + total_fraction)

        return {
            "participation_rate": round(float(participation), 4),
            "permanent_impact_bps": round(float(perm_impact_fraction * 10000.0), 2),
            "temporary_impact_bps": round(float(temp_impact_fraction * 10000.0), 2),
            "total_cost_bps": round(float(total_bps), 2),
            "effective_fill_price": round(float(effective_fill), 4),
        }


def calculate_market_impact(
    trade_volume: float,
    adtv: float,
    daily_vol: float,
    spot_price: float = 100.0,
    gamma_perm: float = 0.10,
    eta_temp: float = 0.15,
    alpha: float = 0.50,
) -> Dict[str, float]:
    """Convenience functional wrapper for AlmgrenChrissImpactModel."""
    model = AlmgrenChrissImpactModel(gamma_perm=gamma_perm, eta_temp=eta_temp, alpha=alpha)
    return model.calculate_impact(
        trade_volume=trade_volume,
        adtv=adtv,
        daily_vol=daily_vol,
        spot_price=spot_price,
    )

