# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Data Provenance Guard & Institutional Safety Gatekeeper
======================================================
# my-qlib Fork Extension: Institutional Microstructure & Derivatives Layer

Enforces strict data provenance rules to prevent automated execution algorithms
from deploying capital on synthetic surfaces, missing short interest, or non-PIT timestamps.
"""

from enum import Enum
from typing import Dict, Any, Optional


class DataProvenance(str, Enum):
    """Enumeration of validated market data provenance tiers."""
    LIVE_OPRA_VERIFIED = "live_opra_verified"
    HISTORICAL_OPRA_EOD = "historical_opra_eod"
    SYNTHETIC_RESEARCH_FALLBACK = "synthetic_research_fallback"


class DataProvenanceGuard:
    """
    Institutional production safety gatekeeper.
    Refuses to emit actionable gamma squeeze signals on unverified or synthetic surfaces.
    """

    @staticmethod
    def validate_provenance(
        provenance: DataProvenance,
        short_interest_pct: Optional[float],
        is_pit_timestamp: bool,
    ) -> Dict[str, Any]:
        """
        Validates data sufficiency and asserts actionable permissions.

        Parameters
        ----------
        provenance : DataProvenance
            Source of the options chain and volatility surface.
        short_interest_pct : Optional[float]
            Short interest float percentage. Must not be None or negative.
        is_pit_timestamp : bool
            True if earnings timestamps and price bars are point-in-time verified.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'is_actionable': bool (True ONLY if live/historical verified, PIT, and short float present)
            - 'provenance_tier': str
            - 'gate_violations': List[str]
            - 'safety_status': "PRODUCTION_CLEAR" or "ACTION_SUPPRESSED"
        """
        violations = []

        if provenance == DataProvenance.SYNTHETIC_RESEARCH_FALLBACK:
            violations.append(
                "Synthetic option surface detected. Actionable gamma squeeze flags suppressed to prevent synthetic smile bias."
            )

        if short_interest_pct is None or short_interest_pct < 0.0:
            violations.append(
                "Missing point-in-time short interest float. Squeeze threshold evaluation incomplete."
            )

        if not is_pit_timestamp:
            violations.append(
                "Non-PIT earnings announcement timestamp. Risk of AMC/BMO lookahead bias."
            )

        is_actionable = len(violations) == 0

        return {
            "is_actionable": is_actionable,
            "provenance_tier": provenance.value if isinstance(provenance, DataProvenance) else str(provenance),
            "gate_violations": violations,
            "safety_status": "PRODUCTION_CLEAR" if is_actionable else "ACTION_SUPPRESSED",
        }


def validate_data_provenance(
    provenance: DataProvenance,
    short_interest_pct: Optional[float],
    is_pit_timestamp: bool = True,
) -> Dict[str, Any]:
    """Convenience functional wrapper for DataProvenanceGuard."""
    return DataProvenanceGuard.validate_provenance(
        provenance=provenance,
        short_interest_pct=short_interest_pct,
        is_pit_timestamp=is_pit_timestamp,
    )

