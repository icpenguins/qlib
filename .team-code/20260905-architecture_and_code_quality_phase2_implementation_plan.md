# Implementation Plan: Architecture & Code Quality Phase 2 (Domain Modeling & Structural Decomposition)

**Evaluation Context**: Derived directly from the Multi-Agent Architectural Audit ([`20260905-code_review_and_architectural_audit.md`](file:///e:/SRC/GITHUB/my-qlib/.team-code/20260905-code_review_and_architectural_audit.md)) under [`team-code.md`](file:///c:/Users/BrianRogers/.gemini/config/rules/team-code.md).

---

## Printed Priority -1 End-User Acknowledgement
> We explicitly acknowledge the end-user requirements defined in `.team-code/requirements.md`:
> 1. **The Profitable Stock Trader**: Demands real-market regime awareness, derivatives flow, microstructure awareness (AVWAP), realistic execution, and non-stationarity conditioning.
> 2. **The Institutional Hedge Fund Manager (CIO)**: Mandates Sharpe $> 2.0$, cross-sectional factor orthogonalization, rigorous risk management, and zero catastrophic drawdown tolerance.
>
> Phase 2 transforms untyped procedural dictionary dictionaries into strongly typed, immutable domain models and decomposes the ~500-line `predict_future_buy_timing()` God Function into focused, decoupled, testable collaborating services without altering existing JSON contracts or visual display outputs.

---

## 1. Problem Statement & Scope

Following the successful completion of Phase 1 (Foundation Stabilization: vectorized $O(n)$ best buys, shared `indicators.py`, thread-safe `rng`, module hygiene), the platform still carries major structural debt:
1. **No Domain Model / DTOs**: The entire application communicates via untyped, nested dictionaries (`Dict[str, Any]`), making data flow opaque and error-prone.
2. **The God Function (`predict_future_buy_timing()`)**: At ~500 lines, this single routine still intermingles parameter extraction, Monte Carlo path simulation, support/resistance synthesis, tactical recommendation branching, event window shifting, and dictionary output formatting.
3. **Absence of Shared Interface Protocols**: Signal providers (`Regime`, `GEX`, `PEAD`, `Microstructure`, `Alpha158`) have disparate calling conventions with no common protocol.

Phase 2 introduces:
- Strongly typed Domain Transfer Objects (DTOs) in a new `scripts/domain_models.py` module.
- Decomposed collaborating services for Monte Carlo simulation, parameter extraction, and recommendation decisions.
- Complete backward compatibility with existing tests and dictionary output contracts.

---

## 2. Proposed Changes & Component Architecture

### Component 1: Typed Domain Models (`scripts/domain_models.py`)

#### [NEW] [domain_models.py](file:///e:/SRC/GITHUB/my-qlib/scripts/domain_models.py)
Defines strongly typed, dataclass-based domain representations:
- `RegimeParams`: Regime state, name, changepoint hazard, expected run length, risk multiplier, volatility override.
- `GEXParams`: Net GEX millions, regime state description, call wall, put wall, gamma flip price, max pain, vol multiplier.
- `PEADParams`: Next earnings date, days away, proximity status code, position haircut, drift regime, SUE score, announcement gap, drift score.
- `BuyWindow`: `start_date`, `end_date`, `is_active`, `status`, `description`, `modeled_window_dates`.
- `ForecastSeriesPoint`: `date`, `bear_p10`, `median_p50`, `bull_p90`.
- `PredictiveForecastResult`: Complete structured output containing price targets, recommendation, posture, entry corridor, and forecast points. Provides `.to_dict()` for 100% contract fidelity with existing JSON contracts.

Companion documentation: [`.team-code/domain_models.md`](file:///e:/SRC/GITHUB/my-qlib/.team-code/domain_models.md) and [`scripts/domain_models.md`](file:///e:/SRC/GITHUB/my-qlib/scripts/domain_models.md).

---

### Component 2: Decomposed Predictive Buy Timing Services (`scripts/predictive_engine.py`)

#### [NEW] [predictive_engine.py](file:///e:/SRC/GITHUB/my-qlib/scripts/predictive_engine.py)
Decomposes the ~500-line God Function into focused classes:
1. `RegimeParameterExtractor`: Parses and calculates daily hazard rate and forward changepoint probability from BOCD output.
2. `GEXParameterExtractor`: Extracts dealer GEX walls, gamma flip, and volatility dampening/accelerating multipliers.
3. `EventParameterExtractor`: Extracts PEAD drift scores and position haircut multipliers.
4. `MonteCarloSimulator`: Runs thread-safe Geometric Brownian Motion simulation with Laplace jump shocks and earnings gap shocks using `np.random.default_rng()`.
5. `SupportResistanceSynthesizer`: Blends technical support/resistance, AVWAP envelopes, volume profile VAL/VAH, and GEX Call/Put walls.
6. `RecommendationEngine`: Evaluates the 7-branch regime-conditional decision tree, capital preservation posture, and tactical GEX narrative.
7. `BuyWindowOptimizer`: Adjusts buy window start and end dates based on corporate earnings announcements via `RiskDegrossingEngine`.
8. `predict_future_buy_timing()`: Orchestrates the collaborating classes and returns the canonical result dictionary.

Companion documentation: [`.team-code/predictive_engine.md`](file:///e:/SRC/GITHUB/my-qlib/.team-code/predictive_engine.md) and [`scripts/predictive_engine.md`](file:///e:/SRC/GITHUB/my-qlib/scripts/predictive_engine.md).

---

### Component 3: Integration into Master Engine (`scripts/stock_analysis_engine.py`)

#### [MODIFY] [stock_analysis_engine.py](file:///e:/SRC/GITHUB/my-qlib/scripts/stock_analysis_engine.py)
- Import `predict_future_buy_timing` and supporting extractors from `scripts.predictive_engine`.
- Delegate `predict_future_buy_timing()` in `stock_analysis_engine.py` to the modular implementation, preserving complete backward compatibility for external callers.

---

### Component 4: Test Coverage & Regression Safety

#### [NEW] [test_domain_models.py](file:///e:/SRC/GITHUB/my-qlib/tests/test_domain_models.py)
- Unit tests verifying typed DTO instantiations, immutability, validation, and `.to_dict()` conversion.

#### [NEW] [test_predictive_engine.py](file:///e:/SRC/GITHUB/my-qlib/tests/test_predictive_engine.py)
- Isolated unit tests for `MonteCarloSimulator`, `RecommendationEngine`, `SupportResistanceSynthesizer`, and `BuyWindowOptimizer`.

#### [MODIFY] [run_all_tests.py](file:///e:/SRC/GITHUB/my-qlib/scripts/run_all_tests.py)
- Register `"models"` and `"predictive"` suites into `CORE_SUITES`.

---

## 3. Verification Plan

### Automated Core Test Suite Execution
Run the unified test runner across all 18 core suites:
```powershell
& 'e:\SRC\GITHUB\my-qlib\.venv\Scripts\python.exe' 'e:\SRC\GITHUB\my-qlib\scripts\run_all_tests.py'
```
*Acceptance Criteria*: All 92 existing tests pass, plus new domain model and predictive engine unit tests.

### Functional Report Verification
Generate an end-to-end stock analysis report on a benchmark ticker (e.g. MSFT or TEST) to confirm that the generated JSON contract and HTML dashboard match 100% of the existing card requirements.
