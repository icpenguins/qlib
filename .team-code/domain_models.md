# Domain Models Specification (`scripts/domain_models.py`)

## 1. Overview
The `scripts/domain_models.py` module defines strongly typed, immutable dataclasses for all financial and predictive parameters within the stock analysis platform. It eliminates untyped dictionary proliferation, guarantees parameter integrity, and provides transparent `.to_dict()` serialization for complete backward compatibility with existing JSON schemas.

## 2. Core Data Transfer Objects (DTOs)

### `RegimeParams`
Holds parsed state and statistical parameters from Bayesian Online Changepoint Detection (BOCD):
- `state`: Optional integer regime state ($0, 1, 2, 3$).
- `name`: Descriptive name (e.g., "Bull Trend", "Risk-Off Liquidation").
- `changepoint_hazard_pct`: Instantaneous changepoint hazard probability percentage.
- `forward_changepoint_prob_pct`: Cumulative probability of a regime shift over the forecast horizon.
- `expected_run_length_days`: Historical expectation of regime persistence in trading days.
- `risk_multiplier`: Factor exposure scaling multiplier ($0.5$ to $1.2$).
- `daily_hazard`: Daily hazard rate: $h = \frac{1}{\max(10, \text{run\_length})}$.
- `vol_21d_pct`: Optional 21-day realized volatility surface anchor.

### `GEXParams`
Parameters extracted from options chains and Dealer Gamma Exposure (GEX) market structure:
- `net_gex_millions`: Dollar gamma exposure per 1% move in millions.
- `regime_state`: $+1$ (long gamma / dampening), $-1$ (short gamma / accelerating), $0$ (neutral).
- `regime_desc`: Formatted regime string.
- `call_wall`: Major upper dealer resistance / pinning strike.
- `put_wall`: Major lower dealer support strike.
- `gamma_flip`: Price boundary where net gamma flips from positive to negative.
- `max_pain`: Expiration pinning strike.
- `vol_multiplier`: Volatility multiplier ($0.85$ in $+GEX$, $1.25$ in $-GEX$).

### `PEADParams`
Parameters extracted from corporate event calendars and Post-Earnings Announcement Drift:
- `next_earnings_date`: Upcoming earnings announcement date string (YYYY-MM-DD).
- `earnings_days_away`: Trading days remaining before the announcement.
- `earnings_proximity`: Proximity classification (`SAFE`, `IMMINENT_DEGROSS`, `CRITICAL_EVENT`).
- `catalyst_status`: Operational status code.
- `event_degross_multiplier`: Capital haircut multiplier ($0.0$ to $1.0$).
- `pead_regime`: Trend classification (e.g., "Bullish Post-Earnings Drift").
- `sue_score`: Standardized Unexpected Earnings metric.
- `pead_gap_pct`: Immediate announcement price gap percentage.
- `pead_drift_pct`: Cumulative post-announcement drift percentage.
- `pead_drift_score`: Composite drift momentum score.

### `BuyWindow`
Encapsulates execution timing and liquidity buffers:
- `start_date`: First actionable entry trading date.
- `end_date`: Final actionable entry trading date.
- `is_active`: Boolean execution state flag.
- `status`: Execution status string (`ACTIVE`, `SUSPENDED`).
- `description`: Actionable summary for traders.
- `modeled_window_dates`: List containing `[start_date, end_date]`.

### `ForecastSeriesPoint`
Defines a single forward trading day trajectory:
- `date`: Forward business date string.
- `bear_p10`: 10th percentile price (bear case).
- `median_p50`: 50th percentile price (base case).
- `bull_p90`: 90th percentile price (bull case).

### `PredictiveForecastResult`
The canonical composite output object encompassing the entire 3-month predictive analysis. Provides `.to_dict()` for 100% contract fidelity with existing JSON contracts.
