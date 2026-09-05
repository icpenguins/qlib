# Walkthrough: Institutional Buy Timing & 5-Day Upward Spike Radar Integration

In collaboration with the project end-users (**The Profitable Stock Trader** and **The Institutional Hedge Fund Manager**) and the 6-member **@team-finance** council (Dr. Victoria Vance, Marcus Reynolds, Dr. Elena Rostova, Julian Montgomery, Sophia Chen, and Arthur Pendelton III), we have integrated the full Contract Schema v1.2.0 analytical suite into [`scripts/visualize_stock_analysis.py`](file:///e:/SRC/GITHUB/my-qlib/scripts/visualize_stock_analysis.py).

The interactive HTML dashboard now provides:
1. **Unambiguous Buy Recommendation**: Clear executive verdict on **whether the stock should be bought** (direction, conviction score, safety provenance check).
2. **Precise Buy Timing Protocol**: Explicit execution instructions on **when it should be bought** (calendar window, market open hours, optimal price corridor, stop-loss floor, and profit targets).
3. **5-Trading-Day Upward Spike Radar**: A dedicated high-velocity visual callout whenever a $t+1 \to t+5$ positive gamma squeeze, post-earnings jump acceleration, or short squeeze setup is detected.
4. **Institutional Quantitative Risk Audit**: Deflated Sharpe Ratio (DSR), Purged Walk-Forward Cross-Validation, Almgren-Chriss market impact, HTB borrow fee engine, verifiable event panel, and Council audit sign-offs.

---

## 1. Modular Components Added

### 1.1 Executive Buy Timing Verdict Banner
- **Function**: [`build_buy_timing_verdict_banner_html`](file:///e:/SRC/GITHUB/my-qlib/scripts/visualize_stock_analysis.py#L861-L970)
- **Visual Position**: Prominently placed at the very top of the dashboard, directly beneath the ticker header.
- **Key Features**:
  - **Verdict Headline**:
    - `⚡ IMMEDIATE BUY: HIGH-VELOCITY 5-DAY SPIKE DETECTED` (if active 5-day positive gamma squeeze setup)
    - `🟢 STRONG BUY: STRATEGIC MULTI-HORIZON ACCUMULATION`
    - `🔵 BUY ON PULLBACK: WAIT FOR ENTRY CORRIDOR`
    - `🟡 HOLD / CAUTIOUS BUY: IMMINENT CATALYST & REGIME HAZARD`
    - `🔴 DO NOT BUY / CAPITAL PRESERVATION MODE`
  - **Safety Invariant Enforcement**:
    - If data provenance is `synthetic_research_fallback` or `safety_status == "ACTION_SUPPRESSED"`, displays an amber badge: `SAFETY INVARIANT: SYNTHETIC RESEARCH DATA` and marks verdict as `RESEARCH ONLY`.
  - **Actionable Execution Protocol**:
    - **Should It Be Bought?**: Clear verdict (`YES - BUY NOW`, `BUY ON DIP`, `STAND ASIDE`, or `RESEARCH ONLY`).
    - **When Should It Be Bought?**: Exact window and hour (e.g. `Immediate T+1 Open through T+5 Close` at `09:30 AM EST`).
    - **Optimal Entry Corridor**: Explicit price range (e.g. `$148.50 - $152.00`).
    - **Invalidation Stop-Loss**: Structural floor peg (e.g. Put Wall / Lower Gamma Trap).
    - **Upper Target / R:R**: Upper Squeeze Wall with risk-reward ratio (e.g. `3.8:1`).

### 1.2 5-Trading-Day Upward Spike Radar & Gamma Squeeze Card
- **Function**: [`build_gamma_squeeze_spike_card_html`](file:///e:/SRC/GITHUB/my-qlib/scripts/visualize_stock_analysis.py#L973-L1130)
- **Visual Position**: Directly below the Executive Buy Timing Banner.
- **Hero Visual Box**:
  - Pulsing emerald radar indicator: `⚡ Next-Day to Next-Week (t+1 to t+5) Gamma Squeeze & 5-Day Upward Spike Radar`.
  - Badge: `ACTIVE 5-DAY UPWARD SPIKE DETECTED` (or `THEORETICAL SPIKE SETUP / ACTION SUPPRESSED`).
  - Metrics Bar: Calibrated Squeeze Probability $P(\text{Squeeze}) = 89.2\%$, Positive GSI ($GSI^+ = 88.2/100$), Residual GSI ($+81.5$), Expected 5d Jump ($+9.2\%$), Trigger Strike, and Upper Squeeze Wall.
- **4 Quantitative Sub-Cards**:
  1. **Post-Earnings Vol Surface & Jump Calibration** (Dr. Vance): Expected jump %, event variance, post-event IV, winsorized IV crush ratio (-44.0%), and crush estimator source.
  2. **Forced Dealer Delta/Gamma Hedging** (Dr. Vance & Marcus Reynolds): Shares demand (2.45M shares), dollar demand ($380M), hedging velocity, and % of ADTV (32.5%).
  3. **Microstructure & Liquidity Impact** (Marcus Reynolds & Julian Montgomery): Expected spread widening (+22.5 bps), Almgren-Chriss slippage (28.0 bps), liquidity regime, and lower gamma trap.
  4. **Actionable 5-Day Execution Clock**: $T_0$ AMC Announcement $\to$ $T_1$ 09:30 AM Open Limit Entry Window $\to$ $T_1 \to T_5$ Holding Horizon $\to$ $T_5$ Exit into Upper Squeeze Wall.

### 1.3 Multi-Horizon Institutional Conviction Matrix
- **Function**: [`build_multi_horizon_matrix_card_html`](file:///e:/SRC/GITHUB/my-qlib/scripts/visualize_stock_analysis.py#L1133-L1220)
- **Visual Position**: Below the Events & Catalysts card, above forward projections.
- **Horizons Evaluated**:
  1. `t_plus_1_to_5` (Next-Day to Next-Week / 5 Trading Days) - Highlighted with a distinct `5-DAY RADAR` badge.
  2. `1M` (1 Month / 21 Trading Days)
  3. `6M` (6 Months / 126 Trading Days)
  4. `1Y` (1 Year / 252 Trading Days)
  5. `3Y` (3 Years / 756 Trading Days)
  6. `10Y` (10 Years / 2520 Trading Days)
- **Table Visuals**: Direction badge, visual conviction score bar (0-100), expected return %, Sharpe ratio, primary quantitative driver, and optimal action.

### 1.4 Institutional Backtesting Protocol & Quantitative Risk Audit Card
- **Function**: [`build_backtesting_protocol_card_html`](file:///e:/SRC/GITHUB/my-qlib/scripts/visualize_stock_analysis.py#L1223-L1410)
- **Visual Position**: Below forward projections, above interactive charts.
- **Components**:
  - **Deflated Sharpe Ratio (DSR)**: Best Sharpe (1.92) vs Hurdle (1.34), DSR Probability ($97.4\%$, Significant $p < 0.05$), Trial count $N=240$.
  - **Purged Walk-Forward Cross-Validation**: 5 Folds, 10-day embargo, 0% event label overlap invariant.
  - **Almgren-Chriss Slippage Breakdown**: Temporary impact (15.2 bps), permanent impact (11.0 bps), total slippage (26.2 bps), participation cap ($\le 2.5\%$ POV).
  - **Securities Lending / Borrow Fee Engine**: Borrow fee (75 bps), general collateral status, utilization (42.0%).
  - **Verifiable Replication Event Panel**: 128 verified events, Win Rate (70.2%), Profit Factor (2.65), Calmar Ratio (3.5), Max Drawdown (-8.9%).
  - **Council Interrogation Sign-Offs**: Dedicated cards with audit status and comments from Dr. Vance, Marcus Reynolds, Dr. Rostova, Julian Montgomery, Sophia Chen, and Arthur Pendelton III.

---

## 2. Verification & Testing Results

### 2.1 Unit & Integration Tests
We executed the updated test suite in [`tests/test_visualize_stock_analysis_refactor.py`](file:///e:/SRC/GITHUB/my-qlib/tests/test_visualize_stock_analysis_refactor.py):
- All 10 visualizer tests PASSED.
- Verified that all 4 modular builder functions handle edge cases, empty dictionaries, and `None` defensively.
- Verified that the rendered HTML contains all new visual cards and client-side JavaScript bindings.

### 2.2 Full Institutional Test Battery
Ran the unified institutional test runner [`scripts/run_all_tests.py`](file:///e:/SRC/GITHUB/my-qlib/scripts/run_all_tests.py):
```text
======================================================================
SUMMARY: Ran 81 tests in 7.76s
Passed:   81
Failures: 0
Errors:   0
Status:   ALL PASSED [OK]
======================================================================
```

### 2.3 End-to-End Report Generation
Generated a full sample report with active 5-day spike setup at `test_reports/NVDA_analysis_report_2025-10-31.html`:
- The HTML file is 100% self-contained with zero CORS issues.
- The Executive Buy Timing Verdict Banner renders with glowing emerald styling and immediate actionability.
- The 5-Day Upward Spike Radar displays the calibrated squeeze probability, dealer hedging demand, and $T+1 \to T+5$ execution clock.
- The Multi-Horizon Matrix and Institutional Backtesting Protocol cards render seamlessly with interactive charts.
