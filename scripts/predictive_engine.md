# Predictive Buy Timing Engine Specification (`scripts/predictive_engine.py`)

## 1. Overview
The `scripts/predictive_engine.py` module decomposes the historical ~500-line monolithic predictive buy timing function into focused, single-responsibility collaborating services. It manages the forward projection of asset prices over a 63-day (~3-month) investment horizon, integrating market regimes, options market structure, and corporate earnings catalysts.

## 2. Collaborating Classes & Responsibilities

### `RegimeParameterExtractor`
Parses regime analysis dictionaries into typed `RegimeParams`, computing daily hazard rates and cumulative horizon changepoint probabilities from Bayesian Online Changepoint Detection run-lengths.

### `GEXParameterExtractor`
Extracts dealer gamma exposure levels, detecting positive gamma pinning versus negative gamma volatility expansion regimes, and calculating strike boundaries (Call Wall, Put Wall, Gamma Flip, Max Pain).

### `EventParameterExtractor`
Extracts upcoming corporate earnings announcement dates and determines execution proximity status codes (`SAFE`, `IMMINENT_DEGROSS`, `CRITICAL_EVENT`) and SUE post-earnings drift scores.

### `SupportResistanceSynthesizer`
Synthesizes multi-source dynamic price boundaries, fusing:
- Technical moving averages and Bollinger Bands
- Anchored VWAP $\pm 1\sigma$ envelopes
- Volume Profile Value Area Low (VAL) and High (VAH)
- Dealer Put and Call gamma walls

### `MonteCarloSimulator`
Executes thread-safe Geometric Brownian Motion simulation ($1,000$ paths $\times$ $63$ forward days) with:
- Mean-reversion drift pull toward 50-day moving average
- Regime-conditioned volatility scaling
- Laplace-distributed changepoint jump shocks triggered by BOCD hazard rates
- Asymmetric binary earnings announcement jump shocks

### `RecommendationEngine`
Implements the 7-branch institutional regime decision tree:
1. State 2: High-Vol Liquidation / Capital Preservation
2. State 3 / Hazard $\ge 35\%$: Regime Transition Alert / Pause Entries
3. State 0: Low-Vol Bull Trend Momentum / Pullback Accumulation
4. State 1: Range-Bound Consolidation / Mean-Reversion Support Buying
5. Pre-Earnings Event Risk: 100% de-grossing / entry suspension
6. Imminent Catalyst: 50% capital haircut
7. Bullish PEAD Underreaction Drift Accumulation

### `predict_future_buy_timing()`
Orchestrates parameter extraction, simulation, boundary synthesis, and recommendation rules, outputting a typed `PredictiveForecastResult` serialized to dictionary format.
