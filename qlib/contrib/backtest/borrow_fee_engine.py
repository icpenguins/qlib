# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Hard-To-Borrow (HTB) & Locate Capacity Engine
============================================
# my-qlib Fork Extension: Institutional Microstructure & Derivatives Layer

Models institutional short equity borrow costs, locate capacity checks, and recall risks.
Prohibits execution of short liquidation strategies when locate availability is zero.
"""

from typing import Dict, Any, Optional


class ZeroLocateCapacityError(ValueError):
    """Raised when an institutional short order cannot be located or borrowed."""
    pass


class BorrowFeeEngine:
    """
    Simulates institutional locate checking and daily borrow fee accrual.
    """

    def __init__(
        self,
        default_annual_rate: float = 0.0050,  # 50 bps general collateral
        htb_threshold: float = 0.10,          # 10% annualized is hard-to-borrow
    ):
        self.default_annual_rate = default_annual_rate
        self.htb_threshold = htb_threshold

    def calculate_borrow_cost(
        self,
        short_value: float,
        annual_fee_rate: Optional[float] = None,
        days_held: int = 1,
        locate_available: bool = True,
    ) -> Dict[str, Any]:
        """
        Calculates accrued borrow cost for short positioning.

        Parameters
        ----------
        short_value : float
            Total dollar value of short equity position.
        annual_fee_rate : Optional[float], optional
            Annualized borrow fee rate (e.g. 0.25 for 25% HTB rate), by default None.
        days_held : int, optional
            Holding period in calendar days, by default 1.
        locate_available : bool, optional
            Whether institutional prime broker has shares available to locate, by default True.

        Returns
        -------
        Dict[str, Any]
            Cost metrics and locate status.

        Raises
        ------
        ZeroLocateCapacityError
            If locate_available is False and short_value > 0.
        """
        if short_value > 0 and not locate_available:
            raise ZeroLocateCapacityError(
                f"Cannot execute short order of ${short_value:,.2f}: "
                "Prime broker reported ZERO locate capacity. Trade rejected."
            )

        rate = annual_fee_rate if annual_fee_rate is not None else self.default_annual_rate
        is_htb = rate >= self.htb_threshold

        # Daily borrow fee = Value * Rate * (days / 360) (money market convention)
        accrued_cost = short_value * rate * (max(1, days_held) / 360.0)

        return {
            "short_value": round(float(short_value), 2),
            "annual_fee_rate": round(float(rate), 4),
            "days_held": days_held,
            "accrued_cost_dollars": round(float(accrued_cost), 2),
            "is_hard_to_borrow": is_htb,
            "locate_granted": locate_available,
        }


def calculate_borrow_cost(
    short_value: float,
    annual_fee_rate: Optional[float] = None,
    days_held: int = 1,
    locate_available: bool = True,
) -> Dict[str, Any]:
    """Convenience functional wrapper for BorrowFeeEngine."""
    engine = BorrowFeeEngine()
    return engine.calculate_borrow_cost(
        short_value=short_value,
        annual_fee_rate=annual_fee_rate,
        days_held=days_held,
        locate_available=locate_available,
    )

