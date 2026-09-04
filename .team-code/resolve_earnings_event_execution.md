# Function Specification: `resolve_earnings_event_execution` & Event Clock

## 1. Overview & Single Responsibility
- **File**: [`qlib/contrib/events/earnings_event_clock.py`](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/events/earnings_event_clock.py)
- **Module Classification**: `# my-qlib Fork Extension: Institutional Microstructure & Derivatives Layer`
- **Single Responsibility**: Enforces discrete point-in-time event timing for corporate earnings releases (AMC vs. BMO), explicitly prohibiting physically impossible fills at $T_0$ close to eliminate forward-looking survival and lookahead bias.

---

## 2. Structural & Architectural Logic
- **AMC Rule**: Signal formed at $T_0$ 15:55 MOC; announcement occurs $T_0$ 16:01; earliest compliant execution is $T_1$ 09:30 Open or $T_1$ 10:00 Morning VWAP.
- **BMO Rule**: Signal formed at $T_{-1}$ 15:55 MOC; announcement occurs $T_0$ 07:00; earliest compliant execution is $T_0$ 09:30 Open or $T_0$ 10:00 Morning VWAP.
- **Microstructure Invariant**: Attempting to fill an AMC announcement at $T_0$ close immediately raises `InvalidEventExecutionError`.

---

## 3. Class & Function Signatures
```python
class InvalidEventExecutionError(ValueError):
    pass

class EarningsEventClock:
    @staticmethod
    def resolve_event_execution(
        event_date: str,
        reporting_time: str,
        requested_fill_target: str = "T1_OPEN",
    ) -> Dict[str, Any]:

def resolve_earnings_event_execution(
    event_date: str,
    reporting_time: str,
    requested_fill_target: str = "T1_OPEN",
) -> Dict[str, Any]:
```

---

## 4. Invariants & Testing Accuracy
- Tested in [`tests/test_earnings_event_clock.py`](file:///e:/SRC/GITHUB/my-qlib/tests/test_earnings_event_clock.py).
- Invariant: AMC with fill request `T0_CLOSE` raises `InvalidEventExecutionError`.
- Invariant: Weekend rollover is handled accurately (Friday AMC executes Monday $T_1$ Open).

