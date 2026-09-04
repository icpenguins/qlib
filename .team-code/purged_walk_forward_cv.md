# Function Specification: `PurgedWalkForwardCV`

## 1. Overview & Single Responsibility
- **File**: [`qlib/contrib/backtest/purged_walk_forward_cv.py`](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/backtest/purged_walk_forward_cv.py)
- **Module Classification**: `# my-qlib Fork Extension: Institutional Microstructure & Derivatives Layer`
- **Single Responsibility**: Generates purged walk-forward cross-validation train/test splits with explicit event embargo horizons, asserting zero temporal or event-label overlap.

---

## 2. Mathematical Formulation & Partitioning Logic
$$\text{Train Window}: [t_{\text{start}}, \, t_{\text{train\_end}})$$
$$\text{Embargo Buffer}: [t_{\text{train\_end}}, \, t_{\text{train\_end}} + \tau_{\text{embargo}})$$
$$\text{Test Window}: [t_{\text{train\_end}} + \tau_{\text{embargo}}, \, t_{\text{test\_end}})$$
$$\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{test}} = \emptyset$$

---

## 3. Class & Method Signatures
```python
class PurgedWalkForwardCV:
    def __init__(
        self,
        train_window_days: int = 756,
        test_window_days: int = 252,
        embargo_days: int = 10,
        step_days: int = 252,
    ):
    def split(
        self,
        df_or_dates: List[pd.Timestamp],
        event_dates: List[pd.Timestamp] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
```

---

## 4. Invariants & Testing Accuracy
- Tested in [`tests/test_purged_walk_forward_cv.py`](file:///e:/SRC/GITHUB/my-qlib/tests/test_purged_walk_forward_cv.py).
- Invariant: Hard asserts zero overlap between training and testing dates ($\text{len}(\text{train} \cap \text{test}) == 0$).
- Invariant: Minimum distance between last training observation and first test observation is $\ge \tau_{\text{embargo}}$.

