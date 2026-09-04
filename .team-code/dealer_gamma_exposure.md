# Technical Specification: Dealer Gamma Exposure (GEX) & Options Flow

**Document Reference**: `.team-code/dealer_gamma_exposure.md`  
**Implemented Modules**: `qlib.contrib.derivatives` ([`gex.py`](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/derivatives/gex.py), [`vol_surface.py`](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/derivatives/vol_surface.py), [`options_data.py`](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/derivatives/options_data.py), [`__init__.py`](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/derivatives/__init__.py))  
**Integration Points**: [`scripts/stock_analysis_engine.py`](file:///e:/SRC/GITHUB/my-qlib/scripts/stock_analysis_engine.py), [`scripts/visualize_stock_analysis.py`](file:///e:/SRC/GITHUB/my-qlib/scripts/visualize_stock_analysis.py), [`scripts/download_us_selected_data.py`](file:///e:/SRC/GITHUB/my-qlib/scripts/download_us_selected_data.py), [`examples/derivatives_gex_analysis.py`](file:///e:/SRC/GITHUB/my-qlib/examples/derivatives_gex_analysis.py)  
**Test Suite**: [`tests/test_derivatives_gex.py`](file:///e:/SRC/GITHUB/my-qlib/tests/test_derivatives_gex.py), [`tests/test_stock_analysis_engine.py`](file:///e:/SRC/GITHUB/my-qlib/tests/test_stock_analysis_engine.py)

---

## 1. End-User Requirements Verification

### The Profitable Stock Trader
- **Demand Fulfilled**: Integrated institutional **Dealer Gamma Exposure (GEX)**, **Call Gamma Wall**, **Put Gamma Wall**, **Absolute Wall**, and **Gamma Flip Point ($S^*$)** into Qlib.
- **Microstructure Rationale**: Market makers (dealers) must dynamically delta-hedge their option books. When the market is in **Positive Gamma ($+GEX$)**, dealers buy dips and sell rips, pinning price between the Call Wall and Put Wall and suppressing volatility. In **Negative Gamma ($-GEX$)**, dealers are forced to pro-cyclically sell declines and buy rallies, amplifying volatility and triggering cascades.
- **Predictive Buy Timing Impact**:
  - Anchors tactical support to the **Put Gamma Wall** and **Max Pain Strike**.
  - Anchors upside resistance to the **Call Gamma Wall** (major pin/ceiling).
  - Triggers actionable volatility warnings when spot crosses below the **Gamma Flip Point**.
  - Visually renders Call Wall, Put Wall, and Gamma Flip lines on the interactive 3-Month Forecast Canvas (`forecastChart`).

### The Institutional Hedge Fund Manager
- **Methodological Rigor**: Built a vectorized, closed-form Black-Scholes analytical Greeks engine with zero external dependencies (pure NumPy with `math.erf` standard normal CDF).
- **Vol Surface Modeling**: Derived the **25-Delta Risk Reversal Skew ($\text{RR}_{25}$)**, 30-day ATM implied volatility ($\text{IV}_{30d}$), and the **Variance Risk Premium ($\text{VRP} = \text{IV}_{30d} - \sigma_{21d}$)**.
- **Non-Linear Risk Control**: Replaced static Gaussian assumptions with GEX-modulated Monte Carlo volatility:
  $$\sigma_{\text{eff}} = \sigma_{\text{realized}} \times w_{\text{GEX}}$$
  where $w_{\text{GEX}} = 0.85$ in $+GEX$ (mean-reverting regime) and $w_{\text{GEX}} = 1.25$ in $-GEX$ (high-volatility cascading regime).

---

## 2. Mathematical Foundation & Analytical Derivations

### Black-Scholes Option Gamma ($\Gamma$)
For underlying price $S$, strike $K$, time-to-expiration $\tau$, implied volatility $\sigma$, risk-free rate $r$, and dividend yield $q$:
$$d_1 = \frac{\ln(S/K) + \left(r - q + \frac{1}{2}\sigma^2\right)\tau}{\sigma \sqrt{\tau}}$$
$$\Gamma = \frac{e^{-q\tau}}{S \sigma \sqrt{\tau}} \phi(d_1) \quad \text{where} \quad \phi(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2}x^2}$$

### Dealer Net Dollar Gamma per 1% Move
$$\text{GEX}_{\text{call}}(K) = +\text{OI}_{\text{call}}(K) \times 100 \times S \times \Gamma(K) \times (0.01 \times S)$$
$$\text{GEX}_{\text{put}}(K) = -\text{OI}_{\text{put}}(K) \times 100 \times S \times \Gamma(K) \times (0.01 \times S)$$
$$\text{Net GEX} = \sum_K \left( \text{GEX}_{\text{call}}(K) + \text{GEX}_{\text{put}}(K) \right)$$

### Gamma Flip Root-Finding ($S^*$)
The Gamma Flip point $S^*$ is defined by the zero-crossing condition:
$$\text{Net GEX}(S^*) = 0$$
Solved via a 120-point dense evaluation grid spanning $[0.65 S_0, 1.35 S_0]$ followed by continuous linear interpolation across the root boundary.

### Max Pain Strike
$$K_{\text{pain}} = \arg\min_{K^*} \sum_{K} \left( \mathbb{I}_{\text{call}} \max(0, K^* - K) \text{OI}_{\text{call}}(K) + \mathbb{I}_{\text{put}} \max(0, K - K^*) \text{OI}_{\text{put}}(K) \right) \times 100$$

---

## 3. Class & Function Specifications

### 1. `BlackScholesGreeks`
- **Location**: [`qlib/contrib/derivatives/gex.py`](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/derivatives/gex.py)
- **Methods**:
  - `calc_gamma(spot, strike, t_years, sigma, r, q)`: Vectorized calculation of analytical gamma.
  - `calc_delta(spot, strike, t_years, sigma, is_call, r, q)`: Vectorized calculation of option delta ($\Delta$).

### 2. `DealerGammaEngine`
- **Location**: [`qlib/contrib/derivatives/gex.py`](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/derivatives/gex.py)
- **Methods**:
  - `compute_gex(options_df, spot_price)`: Computes dollar GEX, strike profiles, Call/Put/Absolute Walls, Max Pain strike, and Gamma Flip price $S^*$.
  - `_solve_gamma_flip(df, spot)`: Numerical solver identifying the zero-gamma inflection boundary.
  - `_calculate_max_pain(df)`: Evaluates cash expiration value across unique strike prices.

### 3. `VolatilitySurfaceFeatures`
- **Location**: [`qlib/contrib/derivatives/vol_surface.py`](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/derivatives/vol_surface.py)
- **Methods**:
  - `calc_25d_risk_reversal(options_df, spot)`: Interpolates $\text{IV}_{25\Delta \text{ Call}} - \text{IV}_{25\Delta \text{ Put}}$.
  - `calc_variance_risk_premium(iv_30d, realized_vol_21d)`: Computes $\text{VRP} = \text{IV}_{30d} - \sigma_{21d}$.
  - `compute_surface_metrics(...)`: Generates comprehensive term-structure and skew feature vectors.

### 4. `OptionsDataLoader` & `SyntheticOptionSurfaceGenerator`
- **Location**: [`qlib/contrib/derivatives/options_data.py`](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/derivatives/options_data.py)
- **Capabilities**:
  - Ingests institutional CSV option chains if provided on disk.
  - Automatically synthesizes realistic, calibrated option chains (volatility smile, negative equity skew, expiration term structure, open interest clustering) when options are unavailable or during synthetic testing.

---

## 4. Visual Dashboard & Predictive Integration

### 1. Institutional Derivatives & Dealer Gamma Exposure Card
- **HTML/CSS Dashboard**: Added a dedicated glassmorphism card presenting:
  - **Net Dealer GEX Metric**: Color-coded pill (`+$M` in emerald, `-$M` in rose) per 1% move.
  - **Regime Badge**: Active $+GEX$ (Stabilizer / Mean-Reversion) or $-GEX$ (Accelerant / Directional Trend).
  - **Gamma Flip Trigger**: Distance to flip ($S^*$) with breach alerts.
  - **Key Levels**: Call Wall (upside ceiling), Put Wall (downside floor), Max Pain strike.
  - **Surface Features**: 30d ATM IV, Variance Risk Premium (VRP), and 25-Delta Risk Reversal skew.
  - **Strike Profile Distribution**: Visual table with horizontal delta/gamma bars and markers for `CALL WALL`, `PUT WALL`, and `SPOT`.

### 2. 3-Month Forecast Canvas (`forecastChart`)
- Renders **Call Gamma Wall** (cyan dash line), **Put Gamma Wall** (amber dash line), and **Gamma Flip Point** (fuchsia dash line) across the predictive 63-day horizon.
- Price auto-scaling dynamically incorporates all active option walls into the y-axis boundaries.

### 3. Predictive Buy Timing & Multi-Period Projections
- Anchors support and resistance levels to the Put Wall and Call Wall.
- Modulates Monte Carlo volatility simulation path generation by $w_{\text{GEX}}$ ($0.85\times$ for $+GEX$, $1.25\times$ for $-GEX$).
- Enriches action guidance with explicit GEX alerts and stop-loss invalidation thresholds.

---

## 5. Verification & Test Suite Results

All 34 automated unit and integration tests across the repository pass cleanly:

```powershell
python tests/test_derivatives_gex.py
# Ran 5 tests in 0.012s -> OK

python tests/test_stock_analysis_engine.py
# Ran 15 tests in 3.277s -> OK

python tests/test_microstructure.py
# Ran 6 tests in 0.021s -> OK

python tests/test_bocd_regime.py
# Ran 8 tests in 0.113s -> OK
```

### Live End-to-End Execution
Generated production interactive visual report for `SMH` with live Qlib dataset:
```powershell
python scripts/visualize_stock_analysis.py --symbol SMH --data_dir D:\trading\qlib --report_dir D:\trading\custom_reports
# Generated report: D:\trading\custom_reports\SMH_analysis_report_2026-09-03.html
```
- Successfully parsed 1260 historical daily bars for `SMH`.
- Evaluated BOCD state: State 2 (High-Vol Liquidation / Risk-Off).
- Evaluated Dealer GEX: `$-3.31M / 1% move` ($-GEX$ Regime, Accelerant / Squeeze).
- Anchored Put Wall: `$539.50` | Call Wall: `$556.10` | Gamma Flip: `$598.13`.
- Generated 3-month predictive forecast with volatility acceleration and gamma wall lines.

