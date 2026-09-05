# Institutional Buy Timing & 5-Day Upward Spike Radar Integration in HTML Report

Integrate institutional Contract Schema v1.2.0 analytical engines into [`scripts/visualize_stock_analysis.py`](file:///e:/SRC/GITHUB/my-qlib/scripts/visualize_stock_analysis.py) HTML visualizer. This plan addresses the dual requirements of our project end-users (**The Profitable Stock Trader** and **The Institutional Hedge Fund Manager**) and the 6-member **@team-finance** council (Dr. Victoria Vance, Marcus Reynolds, Dr. Elena Rostova, Julian Montgomery, Sophia Chen, and Arthur Pendelton III).

The core focus is:
1. **Actionable Buy Verdict**: An unmistakable answer to **whether the stock should be bought** (conviction level, directional bias, and signal concordance).
2. **Actionable Buy Timing**: An explicit answer to **when it should be bought** (exact trading window dates, market session timing, optimal entry price corridor, and stop-loss support peg).
3. **5-Trading-Day Upward Spike Radar**: A dedicated high-visibility visual callout whenever a $t+1 \to t+5$ positive gamma squeeze, earnings jump acceleration, or short squeeze setup is detected.

---

## User Review Required

> [!IMPORTANT]
> **Zero CORS & Self-Contained Integrity**:
> All new data structures (`earnings_gamma_squeeze`, `backtesting_protocol`, `evaluation_matrix`) are already embedded in `<script id="report-data" type="application/json">`. The HTML visual cards must render cleanly both on the server (Python string generation) and in browser client-side scripts, maintaining zero external runtime dependencies and opening directly via `file:///`.

> [!IMPORTANT]
> **Safety Invariant Enforcement on UI**:
> If data provenance is `synthetic_research_fallback` or `safety_status == 'ACTION_SUPPRESSED'`, the UI must explicitly display a warning badge (`ACTION SUPPRESSED: SYNTHETIC DATA`) and forbid displaying "ACTIVE LIVE EXECUTION", satisfying Chief Risk Officer Marcus Reynolds' and Prime Broker Julian Montgomery's safety protocols.

---

## Open Questions

> [!NOTE]
> **Open Question 1 (Spike Threshold Sensitivity)**:
> Current institutional default for triggering the prominent glowing Emerald Spike Badge is $GSI^+ \ge 60.0$ and calibrated squeeze probability $P(\text{Squeeze}) \ge 60.0\%$ (or an expected jump $\ge +5.0\%$). We propose this multi-condition threshold to prevent false alarms during low-volatility drift regimes.
>
> **Open Question 2 (Council Interrogation Presentation)**:
> We plan to present the 5 Council Member audit verdicts in an expandable, modern tabbed card with green/amber/red status pills and specific mathematical notes.

---

## Proposed Changes

### Visualization Engine & Modular Card Builders

#### [MODIFY] [visualize_stock_analysis.py](file:///e:/SRC/GITHUB/my-qlib/scripts/visualize_stock_analysis.py)

Add 4 new modular single-responsibility HTML card builder functions and update the layout in `generate_html_dashboard`:

1. **`build_buy_timing_verdict_banner_html(pred, gamma_squeeze, eval_matrix, spot_price) -> str`**:
   - Prominently placed immediately below the top header.
   - Answers: **Should it be bought?**
     - Big bold verdict badge:
       - `⚡ IMMEDIATE BUY: HIGH-VELOCITY 5-DAY SPIKE DETECTED` (if $t+1 \to t+5$ spike active)
       - `🟢 STRONG BUY: STRATEGIC MULTI-HORIZON ACCUMULATION` (if multi-horizon $> 70\%$ bullish)
       - `🔵 BUY ON PULLBACK: WAIT FOR DIP ENTRY CORRIDOR` (if extended above AVWAP)
       - `🟡 HOLD / CAUTIOUS BUY: IMMINENT CATALYST RISK` (if earnings $< 5$ days with haircut)
       - `🔴 DO NOT BUY / CAPITAL PRESERVATION` (if changepoint hazard high or bear regime)
   - Answers: **When should it be bought?**
     - Actionable Window: Exact dates (`Start Date → End Date`) and market time (e.g. `T+1 09:30 AM Market Open`).
     - Optimal Entry Price Range: `${low} - ${high}`.
     - Invalidation Stop-Loss: `${stop}` (Put Wall / AVWAP $-1\sigma$).
     - Risk-Reward Ratio: `${rr}:1`.

2. **`build_gamma_squeeze_spike_card_html(gamma_squeeze, spot_price) -> str`**:
   - Dedicated 5-Trading-Day Upward Spike Radar & Earnings Gamma Squeeze Card.
   - Prominent Glowing Emerald/Cyan Alert Header:
     - `⚡ 5-TRADING-DAY HIGH-VELOCITY UPWARD SPIKE RADAR`
     - Calibrated Squeeze Probability $P(\text{Squeeze})$ with isotonic confidence band.
     - Positive Gamma Squeeze Index ($GSI^+$) & Residual Orthogonalized Squeeze Signal.
     - Expected 5-Day Move / Jump Corridor ($+\%$) and Upper Squeeze Wall ($).
   - 4-Column Sub-Grid:
     - **Col 1: Post-Earnings Volatility Surface & Jump Calibration** (Dr. Vance):
       - Expected Jump %, Event Variance, Post-Event IV, Historical Winsorized IV Crush Ratio, and Sample Depth.
     - **Col 2: Forced Dealer Delta/Gamma Hedging** (Dr. Vance & Marcus Reynolds):
       - Dealer Shares to Buy, Dollar Hedging Demand ($M), Hedging Velocity, % of ADTV demand.
     - **Col 3: Microstructure & Liquidity Impact** (Marcus Reynolds & Julian Montgomery):
       - Expected Spread Widening (bps), Almgren-Chriss Slippage (bps), Liquidity Void Warning, Short Interest % & HTB Borrow Fee (bps).
     - **Col 4: Actionable 5-Day Execution Clock**:
       - $T_0$ AMC Announcement (No close fill invariant)
       - $T_1$ 09:30 AM Open Limit Entry Window
       - $T_1 \to T_5$ Holding Horizon & Trailing Stop Peg
       - $T_5$ Exit / De-Grossing Protocol into Upper Squeeze Wall

3. **`build_multi_horizon_matrix_card_html(eval_matrix) -> str`**:
   - Institutional 6-Horizon Asset Allocation Table:
     - $t+1 \to t+5$ (Tactical Gamma Squeeze / 5-Day Spike)
     - 1M (Post-Earnings Announcement Drift / PEAD)
     - 6M (Cyclical Momentum & Value Area)
     - 1Y (Fundamental Growth & Compound Return)
     - 3Y (Structural Trend & Regime Transitions)
     - 10Y (Secular Compounding & Moat Durability)
   - Visual progress bars for Conviction Score (0-100), direction pill, expected return %, Sharpe ratio, primary driver, and optimal action.

4. **`build_backtesting_protocol_card_html(backtest) -> str`**:
   - Institutional Quantitative Risk Audit Card for Institutional Managers & Council:
     - **Deflated Sharpe Ratio (DSR)**: Best Trial Sharpe, Bailey-López de Prado Hurdle, DSR Probability (e.g. $96.2\%$), Trial Count $N=240$, Statistical Significance badge.
     - **Purged Walk-Forward Cross-Validation**: 5 Folds, 10-day embargo, 0% event label leakage invariant.
     - **Almgren-Chriss Slippage Breakdown**: Temporary impact (bps), permanent impact (bps), half-life decay.
     - **Securities Lending / Borrow Fee Engine**: Borrow fee (bps), utilization %, HTB status.
     - **Verifiable Event Replication Panel**: 128 verified earnings events, Win Rate, Profit Factor, Calmar Ratio, Max Drawdown.
     - **Council Interrogation Sign-Offs**: Audit verdicts and commentary from Dr. Vance, Marcus Reynolds, Dr. Rostova, Julian Montgomery, Sophia Chen, and Arthur Pendelton III.

5. **`generate_html_dashboard` Layout Updates**:
   - Assemble the new cards in optimal hierarchy:
     1. Top Header
     2. **Executive Buy Timing Verdict Banner** (NEW)
     3. **5-Trading-Day Upward Spike & Earnings Gamma Squeeze Radar Card** (NEW)
     4. Historical Performance Cards (1Y, 3Y, 5Y, 3M Strategy)
     5. Regime Card (BOCD)
     6. Microstructure Card (AVWAP / Volume Profile)
     7. Derivatives Card (GEX / Flip / Skew)
     8. Events Card (PEAD / SUE / Catalyst)
     9. **Multi-Horizon Evaluation Matrix Card** (NEW)
     10. Forward Return Projections & Probability Scores (6M, 1Y, 2Y, 3Y)
     11. **Institutional Backtesting Protocol & Risk Audit Card** (NEW)
     12. Main Interactive Historical Chart (1Y/3Y/5Y with zoom and event markers)
     13. 3-Month Predictive Forecast Chart
     14. Best Buy Opportunities Ranked Table

---

### Test Suite Updates

#### [MODIFY] [test_visualize_stock_analysis_refactor.py](file:///e:/SRC/GITHUB/my-qlib/tests/test_visualize_stock_analysis_refactor.py)

1. Add unit test assertions for the 4 new modular builder functions:
   - `test_build_buy_timing_verdict_banner_html`
   - `test_build_gamma_squeeze_spike_card_html`
   - `test_build_multi_horizon_matrix_card_html`
   - `test_build_backtesting_protocol_card_html`
2. Validate defensive handling when optional fields are empty or `None`.
3. Validate full HTML dashboard generation contains the 5-day upward spike radar, buy timing window, and DSR backtesting audit.

---

## Verification Plan

### Automated Tests
1. Run updated visualizer tests:
   ```powershell
   .venv\Scripts\python.exe -m unittest tests/test_visualize_stock_analysis_refactor.py -v
   ```
2. Run full regression test suite (all 81+ tests):
   ```powershell
   .venv\Scripts\python.exe scripts/run_all_tests.py
   ```

### Manual Verification & Visual Report Inspection
1. Generate an end-to-end report for a test stock with earnings squeeze setup:
   ```powershell
   .venv\Scripts\python.exe scripts/stock_analysis_data.py --symbol AAPL --report_dir test_reports
   .venv\Scripts\python.exe scripts/visualize_stock_analysis.py --from_json test_reports/AAPL_analysis_report_*.json --report_dir test_reports
   ```
2. Verify HTML output:
   - Check that the **Executive Buy Timing Verdict Banner** is prominently visible at the top.
   - Check that the **5-Trading-Day Upward Spike Radar** renders with high visual contrast, highlighting spike probability, upper squeeze wall, and $T+1 \to T+5$ execution clock.
   - Check that the **Multi-Horizon Evaluation Matrix** clearly shows $t+1 \to t+5$ alongside 1M, 6M, 1Y, 3Y, 10Y.
   - Check that the **Backtesting Protocol & Risk Audit Card** displays the Deflated Sharpe Ratio, Purged CV, and Council sign-offs.
   - Verify that all charts, toggles, and drag-to-zoom functions continue to work without JavaScript console errors.
