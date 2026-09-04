# Technical Specification: Event Risk & Post-Earnings Announcement Drift (PEAD) Models

**Document Reference**: .team-code/event_risk_pead.md  
**Implemented Modules**: qlib.contrib.events ([vent_calendar.py](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/events/event_calendar.py), [pead.py](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/events/pead.py), [
isk_degrossing.py](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/events/risk_degrossing.py), [vents_data.py](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/events/events_data.py), [__init__.py](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/events/__init__.py))  
**Integration Points**: [scripts/stock_analysis_engine.py](file:///e:/SRC/GITHUB/my-qlib/scripts/stock_analysis_engine.py), [scripts/visualize_stock_analysis.py](file:///e:/SRC/GITHUB/my-qlib/scripts/visualize_stock_analysis.py), [scripts/download_us_selected_data.py](file:///e:/SRC/GITHUB/my-qlib/scripts/download_us_selected_data.py), [xamples/event_risk_pead_analysis.py](file:///e:/SRC/GITHUB/my-qlib/examples/event_risk_pead_analysis.py)  
**Test Suite**: [	ests/test_events_pead.py](file:///e:/SRC/GITHUB/my-qlib/tests/test_events_pead.py), [	ests/test_stock_analysis_engine.py](file:///e:/SRC/GITHUB/my-qlib/tests/test_stock_analysis_engine.py)

---

## 1. End-User Requirements Verification

### The Profitable Stock Trader
- **Demand Fulfilled**: Integrated institutional **Corporate Catalyst Awareness**, **Pre-Earnings Risk De-Grossing**, and **Post-Earnings Announcement Drift (PEAD)** quantitative modeling into Qlib.
- **Microstructure Rationale**: Holding full unhedged equity delta into binary corporate earnings reports exposes capital to catastrophic overnight gap risk (+/-10% to +/-25%) that violates risk limits. Conversely, systematic post-earnings drift reflects well-documented institutional underreaction to earnings surprises, providing a durable 30-to-60 trading day post-announcement trend.
- **Predictive Buy Timing Impact**:
  - Automatically shifts the **Optimal Buy Window** past imminent binary earnings dates (t > t_earnings + 2 business days), preventing traders from buying into binary risk.
  - Generates explicit **Tactical Recommendations**:
    - EVENT RISK / PRE-EARNINGS DE-GROSSING (<= 2 days to report: 100% position freeze / de-gross).
    - IMMINENT CATALYST / 50% DE-GROSSING (3 to 4 days to report: 50% capital sizing haircut).
    - PEAD POST-EARNINGS DRIFT ACCUMULATION (Positive SUE beat: aggressive dip-buying in the post-earnings drift window).
  - Visually renders **Interactive Momentum Pins** directly on the main historical canvas (E ▲ earnings beat, E ▼ earnings miss, ⚡ BOCD shift, ◆ FOMC pivot) with detailed hover tooltips and an on/off toolbar toggle.
  - Draws a vertical dashed event line and shaded volatility corridor on the 3-Month Forward Forecast Canvas (orecastChart) at the next scheduled earnings date.

### The Institutional Hedge Fund Manager
- **Methodological Rigor**: Built closed-form, robust econometric models for **Standardized Unexpected Earnings (SUE)**, **Announcement Gap %**, and **Cumulative Abnormal Return (CAR)** with exponential decay (tau = 21d half-life).
- **Macro Catalyst Awareness**: Embedded verified Federal Reserve FOMC rate decision dates and BLS CPI inflation release schedules for 2024-2027 to monitor systemic macro factor risk alongside single-stock corporate reporting dates.
- **Automated Risk De-Grossing**: Replaced static delta assumptions with dynamic event multipliers:
  - <= 2 business days away: 0% allocation (Mandatory pre-event freeze).
  - 3 to 4 business days away: 50% allocation (Pre-event haircut).
  - >= 5 business days away: 100% allocation (Normal risk budget).
- **Forward Horizon Conditioning**: Dynamically conditions 6-Month forward drift and jump volatility on active post-earnings drift and imminent gap variance.

---

## 2. Mathematical Foundation & Analytical Derivations

### Standardized Unexpected Earnings (SUE)
For actual quarterly diluted EPS e_t, consensus analyst expectation e_hat_t, and historical surprise volatility sigma_surprise:
\\text{SUE}_t = \\frac{e_t - \\hat{e}_t}{\\sigma_{\\text{surprise}}}
Where sigma_surprise is estimated from the standard deviation of recent earnings forecast errors.
- SUE > +1.0: Strong positive surprise (Top decile beat).
- SUE < -1.0: Severe negative surprise (Bottom decile miss).

### Announcement Gap Jump %
\\text{Gap}_t = \\frac{O_t - C_{t-1}}{C_{t-1}}
Captures the immediate overnight binary repricing upon market open following earnings release.

### Cumulative Abnormal Return (CAR) & 30-Day Drift Momentum
\\text{CAR}_{[+1, +30]} = \\sum_{\\tau=1}^{30} \\left( R_{i,\\tau} - R_{m,\\tau} \\right) e^{-\\lambda \\tau}
Where lambda = ln(2) / tau_half with half-life tau_half = 21 trading days (~1 calendar month).

### PEAD Forward Drift Score & Alpha Boost
\\text{Drift Score} = 0.50 \\tanh(0.7 \\times \\text{SUE}) + 0.30 \\tanh(10 \\times \\text{Gap}) + 0.20 \\tanh(10 \\times \\text{Drift}_{30d})
\\alpha_{\\text{PEAD}} = \\text{Drift Score} \\times 0.08 \\quad (\\text{up to } \\pm 8\\% \\text{ annualized forward drift conditioning})

### Monte Carlo Binary Gap Jump Injection
In the 3-month daily Monte Carlo simulation (S_t = S_{t-1} * (1 + R_t)), at the forecasted trading day t = t_earn:
R_{t_{\\text{earn}}} \\sim \\mathcal{N}\\left(\\mu_{\\text{SUE}}, \\left(2.5 \\times \\sigma_{\\text{daily}}\\right)^2\\right)
Where mu_SUE = 0.003 * SUE reflects earnings drift asymmetry and 2.5 * sigma_daily models the high-variance binary gap jump shock.

---

## 3. Class & Function Specifications

### 1. EventCalendarEngine
- **Location**: [qlib/contrib/events/event_calendar.py](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/events/event_calendar.py)
- Evaluates calendar catalyst proximity, counting business days to earnings and macro events.

### 2. PEADEngine
- **Location**: [qlib/contrib/events/pead.py](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/events/pead.py)
- Computes SUE, announcement gap %, 30-day exponential drift, and extracts historical momentum events for charting.

### 3. RiskDegrossingEngine
- **Location**: [qlib/contrib/events/risk_degrossing.py](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/events/risk_degrossing.py)
- Calculates position haircut multiplier (w_event) and shifts optimal buy window past imminent event dates.

### 4. EventsDataLoader & SyntheticEventScheduleGenerator
- **Location**: [qlib/contrib/events/events_data.py](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/events/events_data.py)
- Downloads live corporate calendars from Yahoo Finance with 24-hour disk caching, backed by deterministic SEC 10-Q schedule generator.

### 5. compute_event_risk_features
- **Location**: [qlib/contrib/events/__init__.py](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/events/__init__.py)
- Master orchestrator returning catalyst status, degrossing factors, PEAD dynamics, and momentum events.

---

## 4. Visual Dashboard & UI Features

1. **Corporate Catalyst Awareness & PEAD Models Card**:
   - 4 analytical summary widgets (Catalyst Proximity, Macro Catalyst, PEAD Drift Status, Position Haircut Multiplier).
   - Historical quarterly earnings surprise and post-announcement drift table.

2. **Interactive Main Historical Chart (historicalChart)**:
   - **Pins Supported**:
     - `E ▲` (Earnings Beat): Emerald `#10b981` badge with announcement Day-1 gap and 30d drift.
     - `E ▼` (Earnings Miss): Rose `#f43f5e` badge with gap markdown and post-miss drift.
     - `⚡` (BOCD Shift): Amber `#f59e0b` badge indicating structural regime transition (Bull, Compression, Risk-Off).
     - `◆` (FOMC Macro Pivot): Cyan `#06b6d4` badge identifying rate decision market volatility.
   - **Rendering & Collision Avoidance**:
     - 3-tier staggered stem heights (42px, 65px, 88px) to eliminate badge overlap on contiguous catalysts.
     - Dynamic ceiling boundary avoidance: if the price line approaches the chart ceiling (`y < padding.top + 70`), the catalyst stem and badge extend downward rather than upward, preventing clipping.
     - High-contrast white-bordered anchor dots placed directly on the price curve.
   - **Multi-Priority Hit-Testing & Hover Tooltips**:
     - Hit testing checks badge bounding box (+/-4px hit area), vertical stem proximity (+/-8px), and adjacent date indices (+/-1 trading bar).
     - Interactive tooltip renders the event category, announcement gap %, 30-day forward drift %, SUE score / reported vs. consensus figures, and structural regime transition notes.
   - **Interactive Control**:
     - Toolbar toggle button `[⚡ Key Events: ON/OFF]` allows instantaneous toggling of all event pins and guidelines.

3. **3-Month Forward Forecast Canvas (forecastChart)**:
   - Vertical dashed red guideline at scheduled earnings date with shaded event risk corridor.
