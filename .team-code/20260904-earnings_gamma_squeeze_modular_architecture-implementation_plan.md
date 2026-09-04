# Implementation Plan: Modular Earnings Gamma Squeeze Architecture ($t+1$ to $t+5$)

## Executive Overview & Architectural Mandate

This implementation plan translates the institutional council review from [`.team-code/20260904-finance_team_review_stock_analysis_data.md`](file:///e:/SRC/GITHUB/my-qlib/.team-code/20260904-finance_team_review_stock_analysis_data.md) into concrete, production-grade software, incorporating the four critical governance invariants demanded during review:

1. **Invariants Over Target Metrics**: All test suites and production modules assert strict structural, physical, and timing invariants (e.g., zero AMC fills at $T_0$ close, synthetic provenance lockout, zero overlap across purged CV folds, dynamic DSR computation from the actual trial matrix) rather than pre-written target Sharpe figures.
2. **Winsorized Median Historical IV Crush**: Requires a minimum of four verified, observed pre- and post-earnings option implied volatility pairs ($\sigma_{\text{pre}}, \sigma_{\text{post}}$) and calculates a Winsorized Median (trimming extreme outliers). Fallbacks based on $1 - \text{IV}_{M2}/\text{IV}_{M1}$ are strictly tagged in metadata as a *term-structure proxy*, not an observed earnings crush.
3. **Dual-Condition Ground Truth Squeeze Labeling**: Ground truth labels for squeeze calibration require BOTH (a) abnormal jump magnitude at the open/morning VWAP and (b) sign agreement between dealer hedging demand $\mathcal{D}(\Delta S)$ and the opening print $(\text{sgn}(\mathcal{D}(\Delta S)) \times (P_{T_1, \text{open}} - P_{T_0, \text{close}}) > 0)$, filtering out pure cash-session gap-and-crap events.
4. **Calibration Fit Strictly on Open / Morning VWAP Fills**: $P(Y_i = 1 \mid \text{GSI}_i)$ is fitted exclusively on fills realized at market open ($P_{T_1, \text{open}}$) or the first 30-minute VWAP ($P_{T_1, \text{vwap30}}$), never on close-to-close marks.

---

## Stakeholder & End-User Guarantees

### Dual End-User Printed Acknowledgement
Per `.team-code/requirements.md` (Project Priority Requirement -1), all algorithms and system designs explicitly address:
1. **The Profitable Stock Trader**:
   - Demands actionable, high-velocity $t+1$ to $t+5$ asymmetric setups around earnings events.
   - Requires clear directional squeeze indices ($\text{GSI}^+$ and $\text{GSI}^-$) with target price acceleration corridors (Upper Squeeze Wall vs. Lower Trapdoor).
   - Relies on realistic trade timing that prevents execution at unachievable $T_0$ closing prices for After-Market-Close (AMC) announcements.
2. **The Institutional Hedge Fund Manager**:
   - Demands mathematical rigor: replacement of arbitrary sigmoids with Platt-calibrated probabilities ($P(Y=1 \mid \text{GSI})$) and $90\%$ conformal coverage bounds.
   - Enforces cross-sectional factor orthogonalization via Weighted Least Squares (WLS) to strip out size, momentum, volatility, and short interest collinearity.
   - Mandates strict production safety gates (`DataProvenanceGuard` suppressing automated execution on synthetic surfaces), Almgren-Chriss market impact modeling, hard-to-borrow (HTB) fee accrual, purged walk-forward cross-validation, and Bailey & López de Prado Deflated Sharpe Ratio (DSR) reporting.

---

## User Review Required

> [!IMPORTANT]
> **Hard Invariant Assertions Across Test Modules**:
> Rather than testing for synthetic backtest Sharpe numbers, tests must assert verifiable execution, microstructure, and mathematical invariants:
> - **Event Timing Invariant**: For AMC announcements, any execution request targeting $T_0$ close must raise an explicit exception (`InvalidEventExecutionError`). Execution is invariant to $T_1$ Open / Morning VWAP.
> - **Provenance Invariant**: Whenever `data_provenance == DataProvenance.SYNTHETIC_RESEARCH_FALLBACK`, `is_actionable` MUST assert `False` invariant to factor strength.
> - **Purging Invariant**: In `PurgedWalkForwardCV`, the intersection between training label event horizons $[t_{\text{event}}, t_{\text{event}} + \text{embargo}]$ and test evaluation windows MUST assert $\emptyset$ (zero overlap).
> - **DSR Invariant**: DSR MUST be computed dynamically from the trial returns matrix $\mathbf{R} \in \mathbb{R}^{T \times N_{\text{trials}}}$ actually executed, asserting that the hurdle $\mathbb{E}[\max(\text{SR}_0)]$ monotonically increases with $N_{\text{trials}}$.

> [!WARNING]
> **Dual Ground-Truth Squeeze Label Formulation**:
> Squeeze ground-truth labels for probability calibration ($y_i \in \{0, 1\}$) require BOTH:
> 1. Abnormal return hurdle at $T_1$ open: $|AR[0, 1_{\text{open}}]| > 1.5 \cdot \sigma_{\text{daily}}$ or $|AR[0, 5_{\text{open}}]| > 2.5 \cdot \sigma_{\text{daily}}$
> 2. Directional sign agreement with forced dealer demand: $\text{sgn}\left(\mathcal{D}(\Delta S)\right) \times (P_{T_1, \text{open}} - P_{T_0, \text{close}}) > 0$
> A gap that immediately mean-reverts or contradicts dealer demand is labeled $y_i = 0$.

> [!NOTE]
> **Winsorized IV Crush & Data Sufficiency**:
> `calculate_historical_iv_crush` enforces a strict minimum threshold of 4 verified, observed implied volatility pairs ($N_{\text{observed}} \ge 4$) and applies a 20% Winsorization to insulate against corporate actions or COVID-style volatility spikes. Fallbacks are explicitly labeled as `term_structure_proxy`.

---

## Open Questions
None. All 4 feedback items have been translated into formal mathematical specifications and engineering constraints.

---

## Proposed Changes: Single-Responsibility Modular Architecture

```
                                  [Market Data & Inputs]
                   (Option Chains, Spot, ADTV20, EPS Consensus, Short Float)
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │  qlib/contrib/derivatives/data_provenance_    │
                    │  guard.py (DataProvenanceGuard)               │
                    └───────┬───────────────────────────────┬───────┘
            [Synthetic / Non-PIT]                           │ [Live / Historical OPRA Verified]
                    │                                       ▼
                    │               ┌───────────────────────────────────────────────┐
                    │               │  qlib/contrib/derivatives/historical_iv_      │
                    │               │  crush.py (Winsorized 8Q Median Crush >= 4)   │
                    │               └───────────────────────┬───────────────────────┘
                    │                                       │
                    │                                       ▼
                    │               ┌───────────────────────────────────────────────┐
                    │               │  qlib/contrib/derivatives/forced_dealer_      │
                    │               │  hedging.py (Scenario Greeks & Delta Demand)  │
                    │               └───────────────────────┬───────────────────────┘
                    │                                       │
                    │                                       ▼
                    │               ┌───────────────────────────────────────────────┐
                    │               │  qlib/contrib/derivatives/liquidity_impact_   │
                    │               │  ratio.py (LIR Calculation)                   │
                    │               └───────────────────────┬───────────────────────┘
                    │                                       │
                    │                                       ▼
                    │               ┌───────────────────────────────────────────────┐
                    │               │  qlib/contrib/events/empirical_sue.py         │
                    │               │  (12Q Forecast Error Normalized SUE)          │
                    │               └───────────────────────┬───────────────────────┘
                    │                                       │
                    │                                       ▼
                    │               ┌───────────────────────────────────────────────┐
                    │               │  qlib/contrib/derivatives/positive_gamma_     │
                    │               │  squeeze.py & negative_gamma_squeeze.py       │
                    │               └───────────────────────┬───────────────────────┘
                    │                                       │
                    │                                       ▼
                    │               ┌───────────────────────────────────────────────┐
                    │               │  qlib/contrib/derivatives/squeeze_            │
                    │               │  probability_calibration.py (Dual-Label Platt)│
                    │               └───────────────────────┬───────────────────────┘
                    │                                       │
                    │                                       ▼
                    │               ┌───────────────────────────────────────────────┐
                    │               │  qlib/contrib/derivatives/factor_             │
                    │               │  orthogonalization.py (WLS Matrix Projection) │
                    │               └───────────────────────┬───────────────────────┘
                    │                                       │
                    ▼                                       ▼
    ┌───────────────────────────────────────────────────────────────────────────────┐
    │  qlib/contrib/derivatives/earnings_gamma_squeeze_engine.py                    │
    │  (Unified Orchestrator: Assembles Contract Payload v1.1.0)                    │
    └───────────────────────────────────────┬───────────────────────────────────────┘
                                            │
                                            ▼
    ┌───────────────────────────────────────────────────────────────────────────────┐
    │  scripts/stock_analysis_data.py (Schema v1.1.0 Export & CLI)                  │
    └───────────────────────────────────────────────────────────────────────────────┘
```

---

### Component 1: Derivatives Analytics Engine (`qlib/contrib/derivatives/`)

Every capability is placed into its own dedicated file:

#### [NEW] [forced_dealer_hedging.py](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/derivatives/forced_dealer_hedging.py)
- **Responsibility**: Vectorized evaluation of Black-Scholes Greeks under spot-vol jump scenarios with IV crush, calculating net dealer hedging share demand:
  $$\mathcal{D}(\Delta S) = \sum_K 100 \cdot \left[ \text{OI}_{\text{call}}(K) \cdot \Delta_{\text{eff}}^{\text{call}}(K, \Delta S) - \text{OI}_{\text{put}}(K) \cdot \Delta_{\text{eff}}^{\text{put}}(K, \Delta S) \right]$$
- **Exported Function**: `calculate_forced_dealer_hedging_demand(...) -> Dict[float, Dict[str, float]]`

#### [NEW] [liquidity_impact_ratio.py](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/derivatives/liquidity_impact_ratio.py)
- **Responsibility**: Evaluates dealer demand relative to available market depth:
  $$\text{LIR}(\Delta S) = \frac{|\mathcal{D}(\Delta S)|}{\text{ADTV}_{20} \times \lambda_{\text{depth}}}$$
- **Exported Function**: `calculate_liquidity_impact_ratio(shares_demand: float, adtv_20: float, depth_factor: float = 0.10) -> float`

#### [NEW] [historical_iv_crush.py](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/derivatives/historical_iv_crush.py)
- **Responsibility**: Computes ticker-specific historical Winsorized median IV crush over trailing 8 earnings quarters with $N_{\text{observed}} \ge 4$ minimum requirement. If $N_{\text{observed}} < 4$, uses term-structure proxy fallback explicitly tagged in metadata:
  $$\widehat{\alpha}_{\text{crush}, i} = \text{WinsorizedMedian}\left( \left\{ \frac{\sigma_{\text{pre}, q} - \sigma_{\text{post}, q}}{\sigma_{\text{pre}, q}} \right\}_{q=1}^{N_{\text{observed}}}, \text{trim}=0.20 \right)$$
- **Exported Function**: `calculate_historical_iv_crush(observed_iv_pairs: List[Tuple[float, float]], month1_iv: float = None, month2_iv: float = None) -> Dict[str, Any]`

#### [NEW] [post_earnings_volatility.py](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/derivatives/post_earnings_volatility.py)
- **Responsibility**: Implements jump-plus-crush volatility decomposition from ATM straddles:
  $$\mathbb{E}[|\Delta S_{\text{jump}}|] \approx 0.798 \cdot (C_{\text{ATM}} + P_{\text{ATM}}), \quad \sigma_{\text{post}} = \sqrt{\max\left( \sigma_{21d}^2, \sigma_{\text{pre}}^2 - \frac{\mathbb{E}[\Delta S^2]}{\tau} \right)}$$
- **Exported Function**: `calibrate_post_earnings_volatility(spot: float, atm_straddle_price: float, pre_earnings_iv: float, realized_21d_vol: float, dte_days: int) -> Tuple[float, float]`

#### [NEW] [positive_gamma_squeeze.py](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/derivatives/positive_gamma_squeeze.py)
- **Responsibility**: Synthesizes the continuous positive gamma squeeze index $\text{GSI}^+$ from bullish LIR, normalized SUE, call asymmetry, and short interest float.
- **Exported Function**: `compute_positive_gamma_squeeze_index(lir_bull: float, sue_score: float, call_oi_otm: float, put_oi_atm: float, short_interest_pct: float) -> Dict[str, Any]`

#### [NEW] [negative_gamma_squeeze.py](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/derivatives/negative_gamma_squeeze.py)
- **Responsibility**: Synthesizes the continuous negative gamma liquidation index $\text{GSI}^-$ from bearish LIR, miss SUE, gamma flip boundary status, and liquidity void penalty.
- **Exported Function**: `compute_negative_gamma_squeeze_index(lir_bear: float, sue_score: float, spot: float, gamma_flip_price: float, in_liquidity_void: bool) -> Dict[str, Any]`

#### [NEW] [squeeze_probability_calibration.py](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/derivatives/squeeze_probability_calibration.py)
- **Responsibility**: Platt-scaling logistic calibration fitted strictly on Open / Morning VWAP fills with dual-condition ground truth labels ($y_i=1 \iff \text{Abnormal Jump} \land \text{Sign Agreement}$):
  $$P(Y_i = 1 \mid \text{GSI}_i) = \frac{1}{1 + \exp(A \cdot \text{GSI}_i + B)}$$
- **Exported Functions**: `generate_dual_squeeze_label(...) -> int`, `calibrate_squeeze_probability(...) -> float`, `fit_platt_calibrator(...) -> Tuple[float, float]`

#### [NEW] [conformal_prediction_bounds.py](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/derivatives/conformal_prediction_bounds.py)
- **Responsibility**: Computes non-parametric conformal prediction coverage intervals $[p_{\text{lower}}, p_{\text{upper}}]$ at a $90\%$ confidence level.
- **Exported Function**: `calculate_conformal_bounds(calibrated_prob: float, confidence_level: float = 0.90, residual_quantile: float = 0.08) -> Tuple[float, float]`

#### [NEW] [data_provenance_guard.py](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/derivatives/data_provenance_guard.py)
- **Responsibility**: Strict safety gatekeeper enforcing provenance checks. Enumerates `DataProvenance` and asserts `is_actionable=False` whenever synthetic or missing inputs are present.
- **Exported Class/Functions**: `DataProvenance` (Enum), `DataProvenanceGuard`, `validate_data_provenance(...) -> Dict[str, Any]`

#### [NEW] [factor_orthogonalization.py](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/derivatives/factor_orthogonalization.py)
- **Responsibility**: Weighted Least Squares (WLS) factor projection matrix isolating pure idiosyncratic dealer gamma pressure:
  $$\mathbf{GSI}_{\text{orth}} = \left( \mathbf{I} - \mathbf{X} \left( \mathbf{X}^T \mathbf{\Omega}^{-1} \mathbf{X} \right)^{-1} \mathbf{X}^T \mathbf{\Omega}^{-1} \right) \mathbf{GSI}$$
- **Exported Function**: `orthogonalize_gsi_factors(gsi_series: np.ndarray, factor_matrix: np.ndarray, residual_variances: np.ndarray = None) -> np.ndarray`

#### [NEW] [earnings_gamma_squeeze_engine.py](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/derivatives/earnings_gamma_squeeze_engine.py)
- **Responsibility**: High-level orchestrator composing all the individual modules into a structured dictionary matching Contract Schema v1.1.0.
- **Exported Function**: `evaluate_earnings_gamma_squeeze(...) -> Dict[str, Any]`

#### [MODIFY] [qlib/contrib/derivatives/\_\_init\_\_.py](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/derivatives/__init__.py)
- Re-export all newly created individual functions cleanly.

---

### Component 2: Corporate Events Engine (`qlib/contrib/events/`)

#### [NEW] [empirical_sue.py](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/events/empirical_sue.py)
- **Responsibility**: Normalizes Standardized Unexpected Earnings by company-specific trailing 12-quarter analyst forecast standard error:
  $$\text{SUE}_i = \frac{\text{EPS}_{\text{actual}, i} - \text{EPS}_{\text{consensus}, i}}{\sqrt{\frac{1}{11} \sum_{q=1}^{12} (\text{EPS}_{\text{actual}, q} - \text{EPS}_{\text{consensus}, q} - \bar{\delta}_i)^2}}$$
- **Exported Function**: `calculate_empirical_sue(actual_eps: float, consensus_eps: float, historical_forecast_errors: List[float]) -> float`

#### [NEW] [earnings_event_clock.py](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/events/earnings_event_clock.py)
- **Responsibility**: Real-world execution timing separating AMC and BMO announcements. Formulates signal at $T_0$ MOC (15:55 EST), enforces fill at $T_1$ Market Open (09:30 EST), and physically rejects lookahead $T_0$ close fills with `InvalidEventExecutionError`.
- **Exported Class/Functions**: `EarningsEventClock`, `InvalidEventExecutionError`, `resolve_earnings_event_execution(announcement_timestamp: str, reporting_time: str, signal_moc_time: str = "15:55") -> Dict[str, Any]`

#### [MODIFY] [qlib/contrib/events/\_\_init\_\_.py](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/events/__init__.py)
- Re-export `calculate_empirical_sue`, `EarningsEventClock`, and `resolve_earnings_event_execution`.

---

### Component 3: Institutional Microstructure & Backtesting Layer (`qlib/contrib/microstructure/` & `qlib/contrib/backtest/`)

#### [NEW] [almgren_chriss_impact.py](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/microstructure/almgren_chriss_impact.py)
- **Responsibility**: Implements Almgren-Chriss market impact modeling for non-linear execution cost estimation:
  $$\text{Impact} = \gamma \left( \frac{v}{V} \right) + \eta \left( \frac{v}{V} \right)^\alpha$$
- **Exported Class/Functions**: `AlmgrenChrissImpactModel`, `calculate_market_impact(trade_volume: float, adtv: float, daily_vol: float, ...) -> Dict[str, float]`

#### [NEW] [purged_walk_forward_cv.py](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/backtest/purged_walk_forward_cv.py)
- **Responsibility**: Implements rolling walk-forward cross-validation with an explicit 10-day embargo purging window around quarterly earnings boundaries.
- **Exported Class**: `PurgedWalkForwardCV` (with invariant assertion of zero training/testing event label overlap).

#### [NEW] [borrow_fee_engine.py](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/backtest/borrow_fee_engine.py)
- **Responsibility**: Validates institutional short locate capacity, models daily hard-to-borrow (HTB) fee accrual, and triggers short recall events.
- **Exported Class/Functions**: `BorrowFeeEngine`, `calculate_borrow_cost(short_value: float, annual_fee_rate: float, days_held: int, locate_available: bool = True) -> Dict[str, Any]`

#### [NEW] [deflated_sharpe_ratio.py](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/backtest/deflated_sharpe_ratio.py)
- **Responsibility**: Implements Bailey & López de Prado Deflated Sharpe Ratio (DSR) to statistically correct for selection bias under multiple testing, computing the hurdle $\mathbb{E}[\max(\text{SR}_0)]$ dynamically from the empirical trial returns matrix:
  $$\mathbb{E}[\max(\text{SR}_0)] = \sqrt{2 \ln N_{\text{trials}}} + \frac{0.5772}{\sqrt{2 \ln N_{\text{trials}}}}$$
- **Exported Function**: `calculate_deflated_sharpe_ratio(trial_matrix: np.ndarray, benchmark_sharpe: float = 0.0) -> Dict[str, float]`

#### [NEW] [qlib/contrib/backtest/\_\_init\_\_.py](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/backtest/__init__.py)
- Module initialization and clean re-exports.

---

### Component 4: Data Contract & CLI Pipeline Layer (`scripts/`)

#### [MODIFY] [scripts/stock_analysis_data.py](file:///e:/SRC/GITHUB/my-qlib/scripts/stock_analysis_data.py)
- Upgrade `contract_version` to `"1.1.0"`.
- Integrate `evaluate_earnings_gamma_squeeze` into `prepare_analysis_json_payload`.
- Add CLI parameters: `--provenance`, `--simulate_jump`, and `--custom_iv_crush`.
- Incorporate safety suppression if provenance is synthetic.

#### [MODIFY] [scripts/stock_analysis_engine.py](file:///e:/SRC/GITHUB/my-qlib/scripts/stock_analysis_engine.py)
- Wire `evaluate_earnings_gamma_squeeze` into the primary multi-model orchestration pipeline.

---

### Component 5: Dedicated Unit & Institutional Test Battery (`tests/`)

Every single function has its own dedicated test file asserting exact physical and mathematical invariants:

1. #### [NEW] [tests/test_forced_dealer_hedging.py](file:///e:/SRC/GITHUB/my-qlib/tests/test_forced_dealer_hedging.py)
   - Invariant: Hedging demand $\mathcal{D}(\Delta S)$ is strictly positive for large positive spot jumps when call open interest dominates.
2. #### [NEW] [tests/test_liquidity_impact_ratio.py](file:///e:/SRC/GITHUB/my-qlib/tests/test_liquidity_impact_ratio.py)
   - Invariant: LIR increases strictly monotonically with absolute hedging shares; zero ADTV raises ZeroDivision error or returns infinity.
3. #### [NEW] [tests/test_historical_iv_crush.py](file:///e:/SRC/GITHUB/my-qlib/tests/test_historical_iv_crush.py)
   - Invariant: If $N_{\text{observed}} < 4$, crush calculation rejects empirical classification and returns `term_structure_proxy`; Winsorized median is invariant to single extreme outlier shocks.
4. #### [NEW] [tests/test_post_earnings_volatility.py](file:///e:/SRC/GITHUB/my-qlib/tests/test_post_earnings_volatility.py)
   - Invariant: Post-earnings variance is strictly bounded below by 21-day realized variance.
5. #### [NEW] [tests/test_empirical_sue.py](file:///e:/SRC/GITHUB/my-qlib/tests/test_empirical_sue.py)
   - Invariant: SUE scales inversely with forecast standard error; zero-variance histories apply minimum standard error floor.
6. #### [NEW] [tests/test_positive_gamma_squeeze.py](file:///e:/SRC/GITHUB/my-qlib/tests/test_positive_gamma_squeeze.py)
   - Invariant: $\text{GSI}^+$ is strictly bounded in $[0.0, 100.0]$ and monotonic with respect to LIR and short interest float.
7. #### [NEW] [tests/test_negative_gamma_squeeze.py](file:///e:/SRC/GITHUB/my-qlib/tests/test_negative_gamma_squeeze.py)
   - Invariant: $\text{GSI}^-$ increases monotonically with bearish LIR and triggers cascade alerts only below the gamma flip price.
8. #### [NEW] [tests/test_squeeze_probability_calibration.py](file:///e:/SRC/GITHUB/my-qlib/tests/test_squeeze_probability_calibration.py)
   - Invariants: Dual ground-truth label strictly requires sign agreement $(\text{sgn}(\mathcal{D}) \cdot \Delta S_{\text{open}} > 0)$; calibration is fitted strictly on Open/Morning VWAP returns; posterior probability $P \in [0.0, 1.0]$.
9. #### [NEW] [tests/test_conformal_prediction_bounds.py](file:///e:/SRC/GITHUB/my-qlib/tests/test_conformal_prediction_bounds.py)
   - Invariant: Interval $[p_{\text{lower}}, p_{\text{upper}}]$ always contains $p_{\text{calibrated}}$ and is strictly clipped to $[0.0, 1.0]$.
10. #### [NEW] [tests/test_data_provenance_guard.py](file:///e:/SRC/GITHUB/my-qlib/tests/test_data_provenance_guard.py)
    - Invariant: When `provenance == SYNTHETIC_RESEARCH_FALLBACK`, `is_actionable` asserts `False` without exception.
11. #### [NEW] [tests/test_factor_orthogonalization.py](file:///e:/SRC/GITHUB/my-qlib/tests/test_factor_orthogonalization.py)
    - Invariant: Orthogonalized residuals have zero inner product (correlation $< 1e-10$) with each factor column in $\mathbf{X}$.
12. #### [NEW] [tests/test_earnings_gamma_squeeze_engine.py](file:///e:/SRC/GITHUB/my-qlib/tests/test_earnings_gamma_squeeze_engine.py)
    - Invariant: Orchestrator assembles complete Contract Schema v1.1.0 dictionary with all provenance gates enforced.

#### The 5 Institutional Invariant Battery Modules:
13. #### [NEW] [tests/test_purged_walk_forward_cv.py](file:///e:/SRC/GITHUB/my-qlib/tests/test_purged_walk_forward_cv.py)
    - Invariant: Train and test splits across earnings boundaries assert $\text{len}(\text{train\_events} \cap \text{test\_events}) == 0$ with an enforced 10-day purging window.
14. #### [NEW] [tests/test_almgren_chriss_market_impact.py](file:///e:/SRC/GITHUB/my-qlib/tests/test_almgren_chriss_market_impact.py)
    - Invariant: Market impact costs increase non-linearly when trade size exceeds $10\%$ of ADTV.
15. #### [NEW] [tests/test_htb_borrow_fees.py](file:///e:/SRC/GITHUB/my-qlib/tests/test_htb_borrow_fees.py)
    - Invariant: If locate capacity is zero, short-side execution requests raise an immediate locate rejection error.
16. #### [NEW] [tests/test_deflated_sharpe_ratio.py](file:///e:/SRC/GITHUB/my-qlib/tests/test_deflated_sharpe_ratio.py)
    - Invariant: DSR is computed dynamically from the trial matrix run; hurdle $\mathbb{E}[\max(\text{SR}_0)]$ monotonically increases with $N_{\text{trials}}$. Zero hardcoded numbers.
17. #### [NEW] [tests/test_earnings_event_clock.py](file:///e:/SRC/GITHUB/my-qlib/tests/test_earnings_event_clock.py)
    - Invariant: For AMC events, any order execution requesting $T_0$ close asserts `InvalidEventExecutionError`; fills must execute at $T_1$ Open / Morning VWAP.

#### Test Suite Integration:
18. #### [MODIFY] [tests/test_stock_analysis_data.py](file:///e:/SRC/GITHUB/my-qlib/tests/test_stock_analysis_data.py)
    - Update test suite to validate Contract Schema v1.1.0 including `earnings_gamma_squeeze`.
19. #### [MODIFY] [scripts/run_all_tests.py](file:///e:/SRC/GITHUB/my-qlib/scripts/run_all_tests.py)
    - Register all new modular suites into `CORE_SUITES` and verify $100\%$ pass rate.

---

### Component 6: Part 2 Documentation Specifications (`.team-code/`)

Per `.team-code/requirements.md` (Part 2), create dedicated markdown specifications for every newly created function:
- `.team-code/calculate_forced_dealer_hedging_demand.md`
- `.team-code/calculate_liquidity_impact_ratio.md`
- `.team-code/calculate_historical_iv_crush.md`
- `.team-code/calibrate_post_earnings_volatility.md`
- `.team-code/calculate_empirical_sue.md`
- `.team-code/compute_positive_gamma_squeeze_index.md`
- `.team-code/compute_negative_gamma_squeeze_index.md`
- `.team-code/calibrate_squeeze_probability.md`
- `.team-code/calculate_conformal_bounds.md`
- `.team-code/validate_data_provenance.md`
- `.team-code/resolve_earnings_event_execution.md`
- `.team-code/orthogonalize_gsi_factors.md`
- `.team-code/evaluate_earnings_gamma_squeeze.md`
- `.team-code/almgren_chriss_impact.md`
- `.team-code/purged_walk_forward_cv.md`
- `.team-code/borrow_fee_engine.md`
- `.team-code/deflated_sharpe_ratio.md`

---

## Verification Plan

### Automated Tests
1. **Compilation Check**:
   ```powershell
   & "E:\SRC\GITHUB\my-qlib\.venv\Scripts\python.exe" -m py_compile (Get-ChildItem -Path "qlib/contrib/derivatives/*.py", "qlib/contrib/events/*.py", "qlib/contrib/microstructure/*.py", "qlib/contrib/backtest/*.py" | ForEach-Object { $_.FullName })
   ```
2. **Dedicated Modular Tests**:
   ```powershell
   & "E:\SRC\GITHUB\my-qlib\.venv\Scripts\python.exe" -m pytest tests/test_forced_dealer_hedging.py tests/test_liquidity_impact_ratio.py tests/test_historical_iv_crush.py tests/test_post_earnings_volatility.py tests/test_empirical_sue.py tests/test_positive_gamma_squeeze.py tests/test_negative_gamma_squeeze.py tests/test_squeeze_probability_calibration.py tests/test_conformal_prediction_bounds.py tests/test_data_provenance_guard.py tests/test_factor_orthogonalization.py tests/test_earnings_gamma_squeeze_engine.py -v
   ```
3. **Institutional Invariant Battery Validation**:
   ```powershell
   & "E:\SRC\GITHUB\my-qlib\.venv\Scripts\python.exe" -m pytest tests/test_purged_walk_forward_cv.py tests/test_almgren_chriss_market_impact.py tests/test_htb_borrow_fees.py tests/test_deflated_sharpe_ratio.py tests/test_earnings_event_clock.py -v
   ```
4. **Institutional Core Suite Runner**:
   ```powershell
   & "E:\SRC\GITHUB\my-qlib\.venv\Scripts\python.exe" scripts/run_all_tests.py
   ```

### Functional & CLI Verification
1. **Execute CLI with Simulated Parameters**:
   ```powershell
   & "E:\SRC\GITHUB\my-qlib\.venv\Scripts\python.exe" scripts/stock_analysis_data.py --symbol AAPL --output ./output_data/AAPL_test_v110.json
   ```
2. **Inspect JSON Output**:
   Verify `"contract_version": "1.1.0"` and `"earnings_gamma_squeeze"` dictionary payload with provenance enforcement.
