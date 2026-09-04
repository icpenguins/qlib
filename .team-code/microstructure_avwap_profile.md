# Technical Specification: Anchored VWAP & Continuous Volume Profile (KDE)

**Document Reference**: `.team-code/microstructure_avwap_profile.md`  
**Implemented Modules**: `qlib.contrib.microstructure` (`anchored_vwap.py`, `volume_profile.py`, `__init__.py`)  
**Integration Points**: `scripts/stock_analysis_engine.py`, `scripts/visualize_stock_analysis.py`, `examples/microstructure_avwap_profile.py`  
**Test Suite**: `tests/test_microstructure.py`

---

## 1. End-User Requirements Verification

### The Profitable Stock Trader
- **Demand Fulfilled**: Added institutional **Anchored VWAP** with multiple anchor points (YTD, QTD, 52W High, 52W Low), volume-weighted standard deviation dispersion bands ($\pm 1\sigma, \pm 2\sigma$), and **Volume Profile (KDE)** delivering the exact Point of Control (POC), 70% Value Area (VAH/VAL), and Liquidity Void detection.
- **Trading Value**: Equips traders to spot institutional accumulation near AVWAP dynamic support and anticipate rapid breakout velocity across thin liquidity voids.

### The Institutional Hedge Fund Manager
- **Methodological Rigor**: Replaced discrete histogram binning heuristics with continuous **Volume-Weighted Gaussian Kernel Density Estimation (KDE)** using Silverman's adaptive bandwidth rule.
- **Statistical Features**: Derived continuous standardized distance metrics ($Z_{\text{AVWAP}}$) and highest-density region integrations suitable for cross-sectional factor ranking and portfolio optimization.

---

## 2. Key Functions & Class Specifications

### `AnchoredVWAPCalculator`
* Location: [`qlib/contrib/microstructure/anchored_vwap.py`](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/microstructure/anchored_vwap.py)
* Methods:
  - `calculate_single_anchor(df, anchor_date, prefix='avwap')`: Vectorized calculation of cumulative price-volume, volume-weighted variance, $\pm 1\sigma, \pm 2\sigma$ dispersion bands, and $Z$-scores.
  - `identify_anchor_dates(df)`: Dynamically extracts YTD, QTD, 52W High, and 52W Low anchor dates from historical price bars without external calendar files.
  - `compute_all_institutional_anchors(df)`: Computes multi-anchor trajectories and formats actionable regime diagnosis summaries.

### `VolumeProfileKDE`
* Location: [`qlib/contrib/microstructure/volume_profile.py`](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/microstructure/volume_profile.py)
* Methods:
  - `compute_profile(df)`: Evaluates continuous volume-weighted Gaussian density over a 250-point price grid:
    - Global maximum density $\rightarrow$ **Point of Control (POC)**
    - Highest-density region sorting for 70% cumulative volume mass $\rightarrow$ **Value Area Low (VAL) & Value Area High (VAH)**
    - Local density extrema $\rightarrow$ **High-Volume Nodes (HVN)** and **Low-Volume Nodes (LVN)**
    - Local density thresholding ($< 20\%$ of peak) $\rightarrow$ **Liquidity Void / High Breakout Velocity Alert**

### `compute_microstructure_features`
* Location: [`qlib/contrib/microstructure/__init__.py`](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/microstructure/__init__.py)
* Convenience entrypoint coordinating AVWAP and Volume Profile calculations, returning enriched DataFrames and metadata dictionaries.

---

## 3. Verification & Validation Metrics

All unit tests pass with zero errors:
```powershell
python tests/test_microstructure.py
# Ran 6 tests in 0.020s -> OK
```
- **Hand-Calculated Benchmark Match**: Analytically verified against a known 3-day dataset down to 4 decimal places.
- **KDE Normalization**: Integral of density $\hat{f}(p) \, dp \approx 1.0$.
- **Regression Passes**: `test_bocd_regime.py` (8 tests), `test_stock_analysis_engine.py` (11 tests), and `test_download_us_selected_data.py` (9 tests) all pass cleanly (34 total tests).

