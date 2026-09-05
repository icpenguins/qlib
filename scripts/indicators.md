# Technical Indicators Specification (`scripts/indicators.py`)

## 1. Overview
The `scripts/indicators.py` module centralizes common, vector-optimized technical indicator calculations across the stock analysis engine and predictive modeling pipelines. It guarantees mathematically consistent definitions of technical indicators, prevents copy-paste drift, and provides high-performance pandas vector operations.

## 2. Exported Functions

### `compute_rsi(series: pd.Series, period: int = 14) -> pd.Series`
Computes the Relative Strength Index (RSI) using standard Wilder rolling average smoothing.
- **Parameters**:
  - `series`: `pd.Series` of prices (e.g. daily closing prices).
  - `period`: Lookback window in trading bars (default 14).
- **Formula**:
  $$\text{Gain} = \max(\Delta P, 0), \quad \text{Loss} = \max(-\Delta P, 0)$$
  $$\text{RS} = \frac{\text{SMA}(\text{Gain}, \text{period})}{\text{SMA}(\text{Loss}, \text{period}) + 10^{-9}}$$
  $$\text{RSI} = 100 - \frac{100}{1 + \text{RS}}$$
- **Returns**: `pd.Series` with values clamped to $[0.0, 100.0]$.

### `compute_bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]`
Computes standard Bollinger Bands and the %B oscillator.
- **Parameters**:
  - `series`: Price series.
  - `window`: Moving average window (default 20).
  - `num_std`: Multiplier for rolling standard deviation (default 2.0).
- **Returns**:
  - `middle_band`: 20-period simple moving average.
  - `upper_band`: $\text{Middle} + 2\sigma$.
  - `lower_band`: $\text{Middle} - 2\sigma$.
  - `pct_b`: $\frac{\text{Price} - \text{Lower}}{\text{Upper} - \text{Lower} + 10^{-9}}$.

### `compute_rolling_drawdown(series: pd.Series, window: int = 252) -> pd.Series`
Computes peak-to-trough rolling drawdown over a lookback window.
- **Parameters**:
  - `series`: Price series.
  - `window`: Lookback window in trading days (default 252).
- **Returns**: `pd.Series` where values represent fractional drawdown $\le 0.0$.
