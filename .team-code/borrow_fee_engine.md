# Function Specification: `BorrowFeeEngine` & Hard-To-Borrow

## 1. Overview & Single Responsibility
- **File**: [`qlib/contrib/backtest/borrow_fee_engine.py`](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/backtest/borrow_fee_engine.py)
- **Module Classification**: `# my-qlib Fork Extension: Institutional Microstructure & Derivatives Layer`
- **Single Responsibility**: Simulates institutional stock borrow costs, locate capacity checks, and recall risks, physically prohibiting short execution when locate availability is zero.

---

## 2. Mathematical Formulation
$$\text{BorrowCost}(\$) = V_{\text{short}} \cdot r_{\text{borrow}} \cdot \left(\frac{\Delta t_{\text{days}}}{360}\right)$$
$$\text{LocateCheck}: \text{if } V_{\text{short}} > 0 \land \neg \text{LocateAvailable} \implies \text{Raise } \texttt{ZeroLocateCapacityError}$$

---

## 3. Class & Function Signatures
```python
class ZeroLocateCapacityError(ValueError):
    pass

class BorrowFeeEngine:
    def __init__(
        self,
        default_annual_rate: float = 0.0050,
        htb_threshold: float = 0.10,
    ):
    def calculate_borrow_cost(
        self,
        short_value: float,
        annual_fee_rate: Optional[float] = None,
        days_held: int = 1,
        locate_available: bool = True,
    ) -> Dict[str, Any]:

def calculate_borrow_cost(
    short_value: float,
    annual_fee_rate: Optional[float] = None,
    days_held: int = 1,
    locate_available: bool = True,
) -> Dict[str, Any]:
```

---

## 4. Invariants & Testing Accuracy
- Tested in [`tests/test_htb_borrow_fees.py`](file:///e:/SRC/GITHUB/my-qlib/tests/test_htb_borrow_fees.py).
- Invariant: Zero locate capacity ($\text{locate\_available} = \text{False}$) raises `ZeroLocateCapacityError` on non-zero short value.
- Invariant: Accurately tags positions with borrow rate $\ge 10\%$ as hard-to-borrow (`is_hard_to_borrow = True`).

