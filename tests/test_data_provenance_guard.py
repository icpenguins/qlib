"""Unit tests for DataProvenanceGuard."""

import unittest
from qlib.contrib.derivatives.data_provenance_guard import DataProvenance, DataProvenanceGuard


class TestDataProvenanceGuard(unittest.TestCase):
    def test_live_opra_verified_passes(self):
        res = DataProvenanceGuard.validate_provenance(
            provenance=DataProvenance.LIVE_OPRA_VERIFIED,
            short_interest_pct=0.15,
            is_pit_timestamp=True,
        )
        self.assertTrue(res["is_actionable"])
        self.assertEqual(res["safety_status"], "PRODUCTION_CLEAR")
        self.assertEqual(len(res["gate_violations"]), 0)

    def test_synthetic_fallback_strictly_suppressed(self):
        # INVARIANT CHECK: Synthetic provenance must NEVER permit actionable flags
        res = DataProvenanceGuard.validate_provenance(
            provenance=DataProvenance.SYNTHETIC_RESEARCH_FALLBACK,
            short_interest_pct=0.25,
            is_pit_timestamp=True,
        )
        self.assertFalse(res["is_actionable"])
        self.assertEqual(res["safety_status"], "ACTION_SUPPRESSED")
        self.assertTrue(any("Synthetic" in v for v in res["gate_violations"]))

    def test_missing_short_interest_suppressed(self):
        res = DataProvenanceGuard.validate_provenance(
            provenance=DataProvenance.LIVE_OPRA_VERIFIED,
            short_interest_pct=None,
            is_pit_timestamp=True,
        )
        self.assertFalse(res["is_actionable"])
        self.assertTrue(any("short interest" in v for v in res["gate_violations"]))

    def test_non_pit_timestamp_suppressed(self):
        res = DataProvenanceGuard.validate_provenance(
            provenance=DataProvenance.HISTORICAL_OPRA_EOD,
            short_interest_pct=0.10,
            is_pit_timestamp=False,
        )
        self.assertFalse(res["is_actionable"])
        self.assertTrue(any("PIT" in v for v in res["gate_violations"]))


if __name__ == "__main__":
    unittest.main()

