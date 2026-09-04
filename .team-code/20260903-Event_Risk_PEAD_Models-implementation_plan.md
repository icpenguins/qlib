# Implementation Plan: Trader Demand 4 &ndash; Event Risk & PEAD Models

Integrate an institutional **Corporate Catalyst & Event Risk Architecture** into Qlib, featuring an **Event Awareness Engine** (Earnings, FOMC, CPI), **Automated Risk De-Grossing** ($\pm 48\text{h}$ binary gap risk reduction), and **Post-Earnings Announcement Drift (PEAD)** quantitative factor modeling (Standardized Unexpected Earnings - SUE, Cumulative Abnormal Return - CAR).

---

## User Review Required

> [!IMPORTANT]
> **Dual End-User Perspective**:
> - **The Profitable Stock Trader**: Demands that Qlib stop blindly holding equities through quarterly earnings releases and FOMC rate decisions where unexpected -25% gaps destroy months of alpha. Demands automated pre-event de-grossing (50%–100% position reduction within 48 hours) and systematic exploitation of positive Post-Earnings Announcement Drift (PEAD).
> - **The Institutional Hedge Fund Manager**: Confirms that binary gap risk cannot be diversified away cross-sectionally. Mandates formal econometric modeling: Standardized Unexpected Earnings ($\text{SUE}_q$), Cumulative Abnormal Return ($\text{CAR}[0, 1]$ announcement gap), and a 30-to-60 day post-earnings drift factor with decaying half-life.

> [!WARNING]
> **Data Independence & Zero External Dependencies**:
> Like the Derivatives (GEX) and Microstructure (AVWAP) modules, the Event Risk and PEAD module will operate 100% self-contained in pure Python/NumPy/Pandas. It will ingest live corporate calendar data when available, load local event files, and feature a deterministic **Corporate Calendar & Earnings Surprise Generator** calibrated to real-world SEC 10-Q cadence for offline environments and automated regression testing.

### How Live Corporate Calendar Data Will Be Downloaded
1. **Primary REST API Endpoint (Pure Standard Library - Zero External Dependencies)**:
   - Uses `urllib.request` with browser User-Agent headers to query Yahoo Finance's corporate quoteSummary endpoint:
     `https://query2.finance.yahoo.com/v10/finance/quoteSummary/{SYMBOL}?modules=calendarEvents,earningsHistory,earningsTrend`
   - **Extracted Fields**:
     - `calendarEvents.earnings.earningsDate`: Unix timestamps of the upcoming earnings release date window.
     - `earningsHistory.history`: Historical quarterly reports (past 4 quarters) containing:
       - `quarter`: Date of quarter end (e.g. `2026-06-30`).
       - `epsActual`: Reported EPS.
       - `epsEstimate`: Consensus analyst estimated EPS.
       - `epsDifference`: Surprise magnitude.
       - `surprisePercent`: Standardized percentage surprise.
2. **Optional Accelerator (`yfinance`)**:
   - If `yfinance` is available in the user's environment, `EventsDataLoader` optionally queries `ticker.calendar` and `ticker.earnings_dates` as a fast accelerator.
3. **Macro Event Calendars (FOMC & CPI)**:
   - Federal Reserve FOMC interest rate meetings and Bureau of Labor Statistics (BLS) CPI inflation release schedules are known 12–24 months in advance.
   - We embed the **Official 2024–2027 FOMC & CPI Release Calendar** directly into `event_calendar.py`, ensuring 100% uptime without requiring external API calls for macro dates.
4. **Local Disk Caching**:
   - Live downloads are saved to `<data_dir>/events/<SYMBOL>_events.json`.
   - The engine checks file modification times and caches results for 24 hours, preventing redundant network requests.
5. **Data Downloader Integration**:
   - Added `--download_events` CLI flag to `scripts/download_us_selected_data.py`. When run with `--download_events`, it automatically downloads and caches event files for all requested tickers.
6. **Graceful Offline Fallback**:
   - If network connectivity is unavailable, the engine generates an SEC 10-Q calibrated synthetic earnings schedule (Feb, May, Aug, Nov quarterly cycle) with realistic consensus surprise statistics.

---

## Open Questions

> [!NOTE]
> 1. **Default Risk De-Grossing Policy**:
>    - *Recommendation*: Within 48 hours prior to an earnings announcement, reduce target position sizing exposure by **50%** (risk multiplier $0.5\times$), while extending the optimal buy timing window to post-announcement ($t_{\text{earnings}} + 1\text{ to }3$ days). For extreme volatility regimes (BOCD State 2), de-gross by **100%** (zero new exposure).
> 2. **Macro Event Inclusion**:
>    - *Recommendation*: Include FOMC Rate Decision and CPI Release calendars in the event engine, applying a macro risk buffer to portfolio sizing.

---

## Proposed Changes

```mermaid
flowchart TD
    subgraph Data [Event Ingestion & Calendar Layer]
        D1[download_us_selected_data.py --download_events] --> EL[EventsDataLoader & Local Cache]
        D2[Local Events JSON/CSV] --> EL
        D3[Synthetic SEC 10-Q Schedule Generator Fallback] --> EL
    end

    subgraph Core [qlib.contrib.events - Standalone Package]
        EL --> EC[EventCalendarEngine: Earnings, FOMC, CPI proximity]
        EL --> PEAD[PEADEngine: SUE, CAR[0,1], 30d Drift Alpha]
        EC --> RD[RiskDegrossingEngine: 48h Haircut w_event]
    end

    subgraph Predictive [Predictive Modeling Impact]
        EC --> PFB[predict_future_buy_timing: Pre-event delay + Binary Jump Monte Carlo]
        PEAD --> PFB
        RD --> PFB
        PEAD --> CMP[compute_multi_period_projections: SUE drift boost & Event risk penalty]
    end

    subgraph Visual [Visual Report & Dashboard]
        EC --> EVD[Corporate Catalyst & Event Risk Card]
        PEAD --> EVD
        EC --> FC[forecastChart Canvas: Vertical Earnings/FOMC Event Line & Cone]
        RD --> ST[3-Month Strategy Callout: Event De-Grossing Badge]
    end
```

---

### Component 1: Standalone Package `qlib.contrib.events`

#### [NEW] [`qlib/contrib/events/__init__.py`](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/events/__init__.py)
- Public exports for `EventCalendarEngine`, `PEADEngine`, `RiskDegrossingEngine`, `EventsDataLoader`, and convenience orchestrator `compute_event_risk_features()`.

#### [NEW] [`qlib/contrib/events/event_calendar.py`](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/events/event_calendar.py)
- Tracks corporate earnings dates, FOMC interest rate meetings, and US CPI release dates.
- Calculates trading days to next catalyst ($\Delta t = t_{\text{event}} - t_{\text{now}}$).
- Categorizes catalyst proximity:
  - `SAFE`: Catalyst $> 10$ trading days away.
  - `APPROACHING`: Catalyst within $5\text{ to }10$ trading days.
  - `IMMINENT_DEGROSS`: Catalyst within $2\text{ to }4$ trading days (50% risk haircut).
  - `CRITICAL_EVENT`: Catalyst within $24\text{ to }48$ hours (100% entry freeze / capital de-grossing).

#### [NEW] [`qlib/contrib/events/pead.py`](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/events/pead.py)
- **Standardized Unexpected Earnings (SUE)**:
  $$\text{SUE}_q = \frac{\text{Actual EPS}_q - \text{Consensus EPS}_q}{\sigma(\text{EPS Surprise History})}$$
- **Cumulative Abnormal Return ($\text{CAR}[0, 1]$)**: Initial announcement price reaction adjusted for market benchmark (SPY/QQQ).
- **Post-Earnings Announcement Drift (PEAD) Factor**:
  - Quantifies 30-to-60 day expected continuation drift:
    $$\text{PEAD\_Score}_t = \text{SUE}_q \cdot \exp\left(-\frac{\Delta t_{\text{since\_earnings}}}{\tau_{\text{drift}}}\right)$$
    where $\tau_{\text{drift}} \approx 21$ trading days (~1 month half-life).
  - Identifies whether the asset is in an active **Bullish Drift** or **Bearish Drift** regime.

#### [NEW] [`qlib/contrib/events/risk_degrossing.py`](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/events/risk_degrossing.py)
- Calculates dynamic event position sizing factor $w_{\text{event}} \in [0.0, 1.0]$.
- Models binary gap volatility shock magnitude:
  $$\sigma_{\text{gap}} = \sqrt{\text{Historical Gap Variance} + \text{Implied Straddle Move}}$$

#### [NEW] [`qlib/contrib/events/events_data.py`](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/events/events_data.py)
- Ingests earnings and macro calendars with local disk caching at `<data_dir>/events/<SYMBOL>_events.json`.
- Includes a calibrated fallback generator that reconstructs historical quarterly reporting dates (Feb, May, Aug, Nov) and realistic consensus surprise distributions for any equity ticker.

---

### Component 2: Standalone CLI Demonstration Script

#### [NEW] [`examples/event_risk_pead_analysis.py`](file:///e:/SRC/GITHUB/my-qlib/examples/event_risk_pead_analysis.py)
- Standalone CLI utility to inspect any symbol's upcoming corporate catalysts, historical SUE surprises, PEAD continuation momentum, and risk de-grossing status.

---

### Component 3: Data Ingestion Pipeline

#### [MODIFY] [`scripts/download_us_selected_data.py`](file:///e:/SRC/GITHUB/my-qlib/scripts/download_us_selected_data.py)
- Add `--download_events` CLI argument.
- Download corporate earnings calendars and FOMC/CPI schedules into `<data_dir>/events/`.

---

### Component 4: Predictive Engine Integration

#### [MODIFY] [`scripts/stock_analysis_engine.py`](file:///e:/SRC/GITHUB/my-qlib/scripts/stock_analysis_engine.py)
- **`compute_event_features(df, symbol, data_dir)`**: Ingests corporate calendar and computes SUE, PEAD drift, and event proximity.
- **`predict_future_buy_timing()` Integration**:
  - **Optimal Buy Window Adjustment**: If earnings is scheduled within the next 2-5 days, the optimal entry window start date is delayed to $t_{\text{event}} + 2\text{ days}$ to prevent holding through binary gap risk.
  - **Tactical Recommendation Upgrades**:
    - `EVENT RISK / PRE-EARNINGS DE-GROSSING`: When within 48 hours of earnings.
    - `PEAD POST-EARNINGS DRIFT ACCUMULATION`: When recent earnings (<30d) showed high positive SUE and sustained gap-and-go continuation.
  - **Monte Carlo Binary Gap Shock**: Inject an asymmetric jump shock at $t_{\text{earnings}}$ along simulated paths if within the 63-day horizon.
- **`compute_multi_period_projections()` Integration**:
  - Incorporates PEAD drift momentum into 6M projection target and confidence scoring.
- **`run_stock_analysis()`**: Coordinates event features alongside BOCD, Microstructure, and Derivatives.

---

### Component 5: Interactive Visual Dashboard

#### [MODIFY] [`scripts/visualize_stock_analysis.py`](file:///e:/SRC/GITHUB/my-qlib/scripts/visualize_stock_analysis.py)
- **Main Interactive Price Chart (`historicalChart`) Key Momentum Event Markers**:
  - **PEAD Earnings Surprise Inflections (`E` Badges)**:
    - Placed directly along the historical price trajectory at earnings announcement dates:
      - **Green Pin (`E ▲`)**: Bullish Earnings Beat ($SUE > 0$, positive announcement gap) that sparked sustained upward momentum runs.
      - **Red Pin (`E ▼`)**: Bearish Earnings Miss ($SUE < 0$, negative gap) that triggered severe breakdowns.
    - Displays callout tag with surprise magnitude (e.g. `E: +8.4% EPS Beat (+14.2% 30d Drift)`).
  - **BOCD Structural Regime Shift Pivots (`⚡` Pins)**:
    - Highlights major Bayesian changepoints where market structure transitioned (e.g. into State 2 Liquidation or into State 0 Bullish Trend).
  - **FOMC Macro Rate Pivots (`◆ FOMC` Diamonds)**:
    - Marks key Federal Reserve interest rate decision dates that reversed broader equity momentum.
  - **Interactive Canvas Hover Tooltip**:
    - Hovering over an event marker on the chart displays an interactive card showing: Event Date, Catalyst Type, SUE Score / Surprise %, Initial Gap %, and 30-day Post-Event Drift (PEAD) return.
  - **Toolbar Toggle Button**:
    - Added `[✓ Key Momentum Events]` button in the chart header toolbar so traders can instantly toggle event markers on or off.
- **Dedicated Corporate Catalyst & Event Risk Card**:
  - Next Earnings Date & Countdown pill.
  - Next FOMC Interest Rate Meeting and CPI release countdowns.
  - Active Risk De-Grossing Recommendation ($w_{\text{event}}$ exposure haircut).
  - Historical Earnings & PEAD Track Record Table (last 4 quarters: Date, EPS actual vs estimate, SUE score, Initial Gap %, 30-Day Drift %).
- **Interactive 3-Month Forecast Canvas (`forecastChart`)**:
  - Vertical dashed **Earnings Announcement Line** plotted at the exact upcoming reporting date.
  - Demarcated post-event volatility cone illustrating binary outcome dispersion.
- **3-Month Strategy Callout**:
  - Event alert badges (`Earnings in X days - De-Grossing Active` or `PEAD Bullish Drift Active`).
- **Console Terminal Output**:
  - Dedicated `CORPORATE CATALYST, EVENT RISK & PEAD DYNAMICS` printout.

---

## Verification Plan

### Automated Unit & Integration Tests
Run full suite including newly created event tests:
```powershell
python tests/test_events_pead.py
python tests/test_derivatives_gex.py
python tests/test_stock_analysis_engine.py
python tests/test_microstructure.py
python tests/test_bocd_regime.py
```
- Verify SUE calculation matches known EPS surprise distributions.
- Verify PEAD exponential decay function produces monotonic drift attenuation.
- Verify de-grossing factor scales correctly from 1.0 down to 0.0 inside the 48h event window.
- Verify `predict_future_buy_timing` delays the optimal entry window past imminent earnings dates.

### End-to-End Live Validation on `SMH` and `MSFT`
```powershell
python scripts/visualize_stock_analysis.py --symbol SMH --data_dir D:\trading\qlib --report_dir D:\trading\custom_reports
python scripts/visualize_stock_analysis.py --symbol MSFT --data_dir D:\trading\qlib --report_dir D:\trading\custom_reports
```
- Verify console printout displays the Corporate Catalyst & PEAD section.
- Open generated HTML report to verify the visual rendering of the Catalyst card and the vertical earnings line on the 3-Month Forecast Canvas.

