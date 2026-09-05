# Implementation Plan: Architecture & Code Quality Phase 1 (Foundation Stabilization)

**Evaluation Context**: Derived directly from the Multi-Agent Architectural Audit ([`20260905-code_review_and_architectural_audit.md`](file:///e:/SRC/GITHUB/my-qlib/.team-code/20260905-code_review_and_architectural_audit.md)) under [`team-code.md`](file:///c:/Users/BrianRogers/.gemini/config/rules/team-code.md).

---

## Printed Priority -1 End-User Acknowledgement
> We explicitly acknowledge the end-user requirements defined in `.team-code/requirements.md`:
> 1. **The Profitable Stock Trader**: Demands non-stationarity and regime awareness, dealer gamma exposure (GEX) dynamics, volume profiling and institutional microstructure (AVWAP), realistic execution impact, and avoidance of academic vacuum assumptions.
> 2. **The Institutional Hedge Fund Manager (CIO)**: Demands double-digit net returns with Sharpe ratio $> 2.0$, cross-sectional factor orthogonalization, rigorous multiple-testing correction (Deflated Sharpe Ratio), purged walk-forward cross-validation, and zero catastrophic drawdown tolerance.
>
> All architectural and algorithmic refactorings in this implementation plan preserve 100% mathematical fidelity and operational contract integrity while eliminating technical debt, non-thread-safe global state, and runtime bottlenecks.

---

## 1. Problem Statement & Scope

The architectural audit revealed immediate foundational risks across `scripts/stock_analysis_engine.py`, `scripts/infer_alpha158.py`, `scripts/train_alpha158_lightgbm.py`, and `scripts/stock_analysis_data.py`:
1. **Algorithmic Inefficiency**: An $O(n^2)$ rolling window scan in [`detect_historical_best_buys()`](file:///e:/SRC/GITHUB/my-qlib/scripts/stock_analysis_engine.py#L598-L638) that slows down lookback analysis over 1,260 trading bars.
2. **Global Mutable Random State**: [`predict_future_buy_timing()`](file:///e:/SRC/GITHUB/my-qlib/scripts/stock_analysis_engine.py#L824) mutates global `np.random.seed(42)`, making Monte Carlo simulation non-thread-safe.
3. **Duplicated RSI Logic**: Identical 5-line rolling Wilder RSI calculation copy-pasted across two distinct analytical functions ([lines 563–567](file:///e:/SRC/GITHUB/my-qlib/scripts/stock_analysis_engine.py#L563-L567) and [lines 712–717](file:///e:/SRC/GITHUB/my-qlib/scripts/stock_analysis_engine.py#L712-L717)).
4. **Library-Level Logging Hijack**: [`infer_alpha158.py`](file:///e:/SRC/GITHUB/my-qlib/scripts/infer_alpha158.py#L35-L39) calls `logging.basicConfig()` at top-level import, overriding root logging for any external caller.
5. **Top-Level `os.environ` Mutation**: [`train_alpha158_lightgbm.py`](file:///e:/SRC/GITHUB/my-qlib/scripts/train_alpha158_lightgbm.py#L41-L42) sets environment variables at import time rather than inside execution functions.
6. **Undocumented Magic Numbers & Duplicate Aliases**: Numerous undocumented float literals in Monte Carlo and GEX volatility scaling, plus redundant duplicated keys in the predictive output dictionary.

Phase 1 addresses these **Foundation Stabilization** items with zero feature regressions, maintaining complete backward compatibility with all 15 core test suites (85 passing tests).

---

## 2. Proposed Changes & Component Architecture

### Component 1: Shared Technical Indicators Module (`scripts/indicators.py`)
Create a dedicated, reusable, vector-optimized technical indicator utility module to eliminate code duplication and provide standard, tested financial math.

#### [NEW] [indicators.py](file:///e:/SRC/GITHUB/my-qlib/scripts/indicators.py)
- `compute_rsi(series: pd.Series, period: int = 14) -> pd.Series`: Vectorized Wilder RSI calculation.
- `compute_bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]`: Standardized Upper, Lower, and %B series.
- `compute_rolling_drawdown(series: pd.Series, window: int = 252) -> pd.Series`: Vectorized peak-to-trough drawdown calculation.

---

### Component 2: Engine Vectorization & Thread Safety (`scripts/stock_analysis_engine.py`)

#### [MODIFY] [stock_analysis_engine.py](file:///e:/SRC/GITHUB/my-qlib/scripts/stock_analysis_engine.py)
1. **Vectorize Best Buys Search**:
   - Replace line 600 `prev_window_min = df.loc[i - window : i, "close"].min()` with a precomputed rolling minimum: `df["roll_min21"] = df["close"].rolling(window=21).min()`.
   - Read directly `prev_window_min = df.loc[i, "roll_min21"]` in $O(1)$ time per loop iteration.
2. **Import Shared RSI**:
   - Replace inline RSI at lines 563–567 and lines 712–717 with calls to `compute_rsi()`.
3. **Thread-Safe Localized Monte Carlo Generator**:
   - Replace line 824 `np.random.seed(42)` with `rng = np.random.default_rng(seed=42 if deterministic else None)`.
   - Update `rng.standard_normal(...)`, `rng.random(...)`, and `rng.laplace(...)` calls to use the local generator instance.
4. **Document & Extract Named Constants**:
   - `GEX_POS_VOL_DAMPENER = 0.85`
   - `GEX_NEG_VOL_ACCELERATOR = 1.25`
   - `VOL_MIN_CLAMP = 0.005`
   - `VOL_MAX_CLAMP = 0.045`
   - `DRIFT_MEAN_REVERSION_COEFF = 0.02`
   - `BOCD_JUMP_SCALE_MULT = 1.5`
   - `EARNINGS_GAP_SCALE_MULT = 2.5`
   - `Z_90TH_PERCENTILE = 1.28155`

---

### Component 3: Clean Module Hygiene (`scripts/infer_alpha158.py` & `scripts/train_alpha158_lightgbm.py`)

#### [MODIFY] [infer_alpha158.py](file:///e:/SRC/GITHUB/my-qlib/scripts/infer_alpha158.py)
- Remove module-level `logging.basicConfig()` (lines 35–39).
- Replace with standard `logger = logging.getLogger(__name__)`.
- Configure stream handlers only if `__name__ == "__main__"` (inside `main()`).

#### [MODIFY] [train_alpha158_lightgbm.py](file:///e:/SRC/GITHUB/my-qlib/scripts/train_alpha158_lightgbm.py)
- Remove top-level `os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"` and `os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"` (lines 41–42).
- Move these environment assignments inside `train_alpha158_model()` and `main()`, ensuring importing the module does not alter host environment variables.

---

### Component 4: Test Coverage & Regression Safety (`tests/`)

#### [NEW] [test_indicators.py](file:///e:/SRC/GITHUB/my-qlib/tests/test_indicators.py)
- Unit tests for `compute_rsi()`, `compute_bollinger_bands()`, and `compute_rolling_drawdown()`.
- Numerical validation against known series and edge cases (all zeros, flat series, single element).

#### [MODIFY] [run_all_tests.py](file:///e:/SRC/GITHUB/my-qlib/scripts/run_all_tests.py)
- Register `"indicators": ("Shared Technical Indicators", "tests.test_indicators")` into `CORE_SUITES`.

---

## 3. Verification Plan

### Automated Unit & Core Test Suites
Execute the unified test runner to ensure all existing and new tests pass:
```powershell
$env:PYTHONPATH = "e:\SRC\GITHUB\my-qlib"
e:\SRC\GITHUB\my-qlib\.venv\Scripts\python.exe scripts/run_all_tests.py
```
*Acceptance Criteria*: All 16 core test suites pass (85 existing tests + new indicator tests).

### Algorithmic Benchmark
Verify execution time speedup in `detect_historical_best_buys()` using a benchmark script on 1,260 bars of synthetic data to confirm $O(n)$ speedup over the previous slicing loop.
