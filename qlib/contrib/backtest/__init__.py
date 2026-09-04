# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Quantitative Backtesting & Microstructure Evaluation
====================================================
# my-qlib Fork Extension: Institutional Microstructure & Derivatives Layer

Modular institutional backtesting suite providing:
- Purged Walk-Forward Cross-Validation with Event Embargo (PurgedWalkForwardCV)
- Hard-To-Borrow fee accrual and locate verification (BorrowFeeEngine)
- Deflated Sharpe Ratio multiple testing correction (calculate_deflated_sharpe_ratio)
"""

from .purged_walk_forward_cv import PurgedWalkForwardCV
from .borrow_fee_engine import BorrowFeeEngine, ZeroLocateCapacityError, calculate_borrow_cost
from .deflated_sharpe_ratio import calculate_deflated_sharpe_ratio

__all__ = [
    "PurgedWalkForwardCV",
    "BorrowFeeEngine",
    "ZeroLocateCapacityError",
    "calculate_borrow_cost",
    "calculate_deflated_sharpe_ratio",
]

