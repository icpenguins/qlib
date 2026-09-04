# Function Specification: `validate_data_provenance` & Guard

## 1. Overview & Single Responsibility
- **File**: [`qlib/contrib/derivatives/data_provenance_guard.py`](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/derivatives/data_provenance_guard.py)
- **Module Classification**: `# my-qlib Fork Extension: Institutional Microstructure & Derivatives Layer`
- **Single Responsibility**: Enforces institutional data provenance rules to safeguard automated execution systems from deploying capital based on synthetic volatility surfaces, missing short interest, or non-PIT timestamps.

---

## 2. Structural & Architectural Logic
- **Provenance Hierarchy**:
  - `LIVE_OPRA_VERIFIED`
  - `HISTORICAL_OPRA_EOD`
  - `SYNTHETIC_RESEARCH_FALLBACK`
- **Safety Gate Invariant**:
  $$\text{Actionable} \iff (\text{Provenance} \ne \text{SYNTHETIC}) \land (\text{SI} \ge 0) \land (\text{PIT} = \text{True})$$

---

## 3. Class & Function Signatures
```python
class DataProvenance(str, Enum):
    LIVE_OPRA_VERIFIED = "live_opra_verified"
    HISTORICAL_OPRA_EOD = "historical_opra_eod"
    SYNTHETIC_RESEARCH_FALLBACK = "synthetic_research_fallback"

class DataProvenanceGuard:
    @staticmethod
    def validate_provenance(
        provenance: DataProvenance,
        short_interest_pct: Optional[float],
        is_pit_timestamp: bool,
    ) -> Dict[str, Any]:

def validate_data_provenance(
    provenance: DataProvenance,
    short_interest_pct: Optional[float],
    is_pit_timestamp: bool = True,
) -> Dict[str, Any]:
```

---

## 4. Invariants & Testing Accuracy
- Tested in [`tests/test_data_provenance_guard.py`](file:///e:/SRC/GITHUB/my-qlib/tests/test_data_provenance_guard.py).
- Invariant: `SYNTHETIC_RESEARCH_FALLBACK` unconditionally sets `is_actionable = False` and sets `safety_status = "ACTION_SUPPRESSED"`.
- Invariant: Missing or negative short interest suppresses actionable flags and emits explicit gate violations.

