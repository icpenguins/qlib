# Function Specification: `calibrate_squeeze_probability` & Platt Module

## 1. Overview & Single Responsibility
- **File**: [`qlib/contrib/derivatives/squeeze_probability_calibration.py`](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/derivatives/squeeze_probability_calibration.py)
- **Module Classification**: `# my-qlib Fork Extension: Institutional Microstructure & Derivatives Layer`
- **Single Responsibility**: Implements Platt scaling logistic calibration to map continuous GSI scores into statistically valid posterior probabilities $\mathbb{P}(Y=1 \mid \text{GSI}) \in [0.0, 1.0]$, fitted strictly on Market Open / Morning VWAP execution fills with dual-condition ground truth labeling.

---

## 2. Mathematical Formulation
### Dual-Condition Ground Truth Labeling
$$y_i = \mathbb{I}\left( |\text{AR}_{\text{open}, i}| \ge 1.5 \cdot \sigma_{\text{daily}, i} \right) \times \mathbb{I}\left( \text{sgn}(\mathcal{D}_i) \cdot \text{AR}_{\text{open}, i} > 0 \right)$$
### Platt Logistic Posterior Calibration
$$\mathbb{P}(Y = 1 \mid \text{GSI}) = \frac{1}{1 + \exp(A \cdot \text{GSI} + B)}$$
$$(\widehat{A}, \widehat{B}) = \arg\min_{A, B} \frac{1}{N} \sum_{i=1}^N \left( \mathbb{P}(Y=1 \mid \text{GSI}_i) - y_i \right)^2$$

---

## 3. Function Signatures
```python
def generate_dual_squeeze_label(
    ar_open: float,
    dealer_shares_demand: float,
    daily_vol: float,
    threshold_mult: float = 1.5,
) -> int:

def fit_platt_calibrator(
    gsi_scores: Union[List[float], np.ndarray],
    labels_open: Union[List[int], np.ndarray],
) -> Tuple[float, float]:

def calibrate_squeeze_probability(
    raw_score: float,
    platt_a: float = -0.085,
    platt_b: float = 4.20,
) -> float:
```

---

## 4. Invariants & Testing Accuracy
- Tested in [`tests/test_squeeze_probability_calibration.py`](file:///e:/SRC/GITHUB/my-qlib/tests/test_squeeze_probability_calibration.py).
- Invariant: Dual ground truth label $y_i=0$ if abnormal open return conflicts with dealer demand direction ($\text{sgn}(\mathcal{D}) \cdot \text{AR} \le 0$).
- Invariant: Calibrated probability output is monotonically increasing with GSI and bounded strictly in $[0.0001, 0.9999]$.

