# Function Specification: `calculate_market_impact` & Almgren-Chriss

## 1. Overview & Single Responsibility
- **File**: [`qlib/contrib/microstructure/almgren_chriss_impact.py`](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/microstructure/almgren_chriss_impact.py)
- **Module Classification**: `# my-qlib Fork Extension: Institutional Microstructure & Derivatives Layer`
- **Single Responsibility**: Quantifies non-linear temporary and permanent price impact for institutional order execution, replacing flat execution fee assumptions with market microstructure realities.

---

## 2. Mathematical Formulation
$$\text{Permanent Impact} = \gamma_{\text{perm}} \cdot \sigma_{\text{daily}} \cdot \left(\frac{v}{V}\right)$$
$$\text{Temporary Impact} = \eta_{\text{temp}} \cdot \sigma_{\text{daily}} \cdot \left(\frac{v}{V}\right)^\alpha \quad (\alpha \approx 0.50)$$
$$\text{Total Cost} = \text{Fixed Fee} + \text{Permanent Impact} + \text{Temporary Impact}$$

---

## 3. Class & Function Signatures
```python
class AlmgrenChrissImpactModel:
    def __init__(
        self,
        gamma_perm: float = 0.10,
        eta_temp: float = 0.15,
        alpha: float = 0.50,
        fixed_bps: float = 0.0005,
    ):
    def calculate_impact(
        self,
        trade_volume: float,
        adtv: float,
        daily_vol: float,
        spot_price: float = 100.0,
    ) -> Dict[str, float]:

def calculate_market_impact(
    trade_volume: float,
    adtv: float,
    daily_vol: float,
    spot_price: float = 100.0,
    gamma_perm: float = 0.10,
    eta_temp: float = 0.15,
    alpha: float = 0.50,
) -> Dict[str, float]:
```

---

## 4. Invariants & Testing Accuracy
- Tested in [`tests/test_almgren_chriss_market_impact.py`](file:///e:/SRC/GITHUB/my-qlib/tests/test_almgren_chriss_market_impact.py).
- Invariant: Zero volume order incurs only the fixed fee floor.
- Invariant: Total slippage cost is strictly monotonic and concave with respect to trade size ($v$).

