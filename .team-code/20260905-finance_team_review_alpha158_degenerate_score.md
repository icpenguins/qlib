# Formal Council Review: LightGBM Alpha158 Degenerate Score (-0.00000) & Generic Factor Names (`Column_0`...`Column_3`)

**Document Reference**: [`.team-code/20260905-finance_team_review_alpha158_degenerate_score.md`](file:///e:/SRC/GITHUB/my-qlib/.team-code/20260905-finance_team_review_alpha158_degenerate_score.md)  
**Convening Body**: **`@team-finance`** (The Alpha-Review Framework: 5 Specialized Council Members & The Principal)  
**Repository Compliance**: `.team-code/requirements.md` (Priority -1 & Part 2) and `c:\Users\BrianRogers\.gemini\config\rules\team-finance.md`  
**Evaluation Target**: LightGBM Alpha158 Predictive Machine Learning Card in `visualize_stock_analysis.py`, `infer_alpha158.py`, `train_alpha158_lightgbm.py`, and `reports/MSFT_analysis_report_2026-09-05.html`.

---

## 1. Priority -1: Dual End-User Printed Acknowledgement

Per `.team-code/requirements.md` (Project Priority Requirement -1), all agents confirm compliance with the dual end-user mandates before evaluating the visualizer and model outputs:

1. **The Profitable Stock Trader** (*Veteran Discretionary & Quantitative Prop Trader*):
   - **Mandate**: Consistent alpha, capital preservation, exploiting asymmetric risk/reward setups, avoiding catastrophic drawdowns.
   - **Trader Stance**: *"Seeing an Alpha158 Raw Score of `-0.00000`, a Predicted 5-Day Excess Return of `+0.00%`, and top drivers labeled `Column_0`, `Column_1`, `Column_2`, `Column_3` with `Gain: 0.0` is an instantaneous stop-work condition. You cannot allocate risk, size a bracket, or take an entry on statistical flatline. An opaque column name tells me nothing about whether the algorithm is leaning into short-term mean-reversion, volume expansion, or momentum trend. If the model isn't splitting trees and has no named factor drivers, it is worse than useless—it is dead capital posing as quantitative intelligence."*

2. **The Institutional Hedge Fund Manager** (*CIO / Head of Quantitative Research*):
   - **Mandate**: Deliver double-digit net annualized returns with a Sharpe ratio $> 2.0$, net zero market/factor beta, and zero catastrophic drawdown tolerance.
   - **Manager Stance**: *"An institutional long/short equity book depends entirely on cross-sectional rank separation and factor attribution. In `alpha158_russell1000_latest.csv`, every single stock receives the identical prediction: `-3.5453779018694515e-09`. That yields Rank IC = 0.0, ICIR = 0.0, and an arbitrary tie-break percentile rank of 66.67%. If we run a quintile rebalancing on this output, the portfolio weights collapse into random noise. Furthermore, no institutional Risk Committee or LP will permit capital deployment when factor attribution displays `Column_0` instead of standardized Barra/Qlib style factors."*

---

## 2. Executive Summary of the Finding

During the review of the newly generated institutional HTML report (`reports/MSFT_analysis_report_2026-09-05.html`), two critical defects were identified in the **LightGBM Alpha158 Factor Score Card**:

1. **Degenerate / Flat Prediction Value (`score = -0.00000`, `pred_5d = +0.00%`, `Rank IC = 0.0`)**:
   The trained LightGBM booster (`models/lightgbm/alpha158_russell1000_latest.pkl`) contains **1 tree with exactly 1 leaf and zero splits** (`num_leaves=1`, `leaf_value=-3.5453779018694515e-09`). The tree failed to make any partition during training, resulting in a constant output across every trading day and every instrument.
2. **Loss of Factor Names (`Column_0`, `Column_1`, `Column_2`, `Column_3`, `Gain: 0.0`)**:
   In `qlib.contrib.model.gbdt.LGBModel._prepare_data()`, the DataFrame features `x` are passed to `lgb.Dataset(x.values, ...)` as an unnamed NumPy array rather than retaining `x.columns`. LightGBM defaults to auto-generating `Column_i`. When `train_alpha158_lightgbm.py` serialized `alpha158_russell1000_latest_meta.json`, it captured these generic column placeholders.

---

## 3. The 5 Specialized Council Members: Independent Evaluations

```
                     +---------------------------------------------------+
                     |           THE ALPHA-REVIEW FRAMEWORK              |
                     |  LightGBM Alpha158 Report Card Evaluation         |
                     +---------------------------------------------------+
                                               |
         +------------------+------------------+------------------+------------------+
         |                  |                  |                  |                  |
         v                  v                  v                  v                  v
+------------------+ +------------------+ +------------------+ +------------------+ +------------------+
| 1. TRADER        | | 2. HF MANAGER    | | 3. CHIEF ANALYST | | 4. GLOBAL FINANCE| | 5. QUANT DEV     |
| (1M & 6M)        | | (6M, 1Y, 3Y)     | | (1Y & 3Y)        | | (3Y & 10Y)       | | (All Horizons)   |
| Status: REJECT   | | Status: REJECT   | | Status: REJECT   | | Status: REJECT   | | Status: DIAGNOSED|
| Zero momentum,   | | Rank IC: 0.0,    | | Zero macro /     | | Structural drag, | | Hyperparam over- |
| opaque factors,  | | zero Sharpe,     | | fundamental link,| | LP audit hazard, | | reg & x.values   |
| dead capital.    | | ties in ranking. | | uninterpretable. | | broken contract. | | numpy stripping. |
+------------------+ +------------------+ +------------------+ +------------------+ +------------------+
         |                  |                  |                  |                  |
         +------------------+------------------+------------------+------------------+
                                               |
                                               v
                     +---------------------------------------------------+
                     |       6. THE PRINCIPAL (THE BILLIONAIRE)          |
                     | Verdict: CAPITAL ALLOCATION FROZEN                |
                     | Requires: Split Calibration & Named Factor Map   |
                     +---------------------------------------------------+
```

### 3.1 The High-Earning Trader
- **Core Mandate**: Short-term execution, market timing, and liquidity exploitation (1-Month & 6-Month Horizons).
- **Evaluation & Verdict**: **REJECTED**
- **Statement**:
  > *"Software outputs must provide actionable, high-conviction entry and exit points. A score of `-0.00000` gives me zero edge. I cannot execute a breakout or a pullback trade when the ML factor model is dead flat.
  > 
  > More damning is the factor driver section: `Column_0`, `Column_1`, `Column_2`, `Column_3` with `Gain: 0.0`. In fast-moving markets, I need to know the 'why'. If MSFT is flashing an entry, is it driven by `ROC5` momentum, `VWAP0` institutional mean-reversion, or `KLEN` volatility expansion? `Column_0` is useless in the trading pit. It provides zero visibility into market mechanics and exposes our book to unhedged regime transitions. If the 5-day predicted alpha is 0.00%, my capital sits idle or gets chopped up by fees. Fix the trees, name the factors, or turn off the card."*

### 3.2 The Top Hedge Fund Manager
- **Core Mandate**: Portfolio construction, risk-adjusted returns, alpha generation (6-Month, 1-Year, & 3-Year Horizons).
- **Evaluation & Verdict**: **REJECTED**
- **Statement**:
  > *"We construct market-neutral, factor-orthogonalized portfolios. Alpha158 is supposed to be our quantitative engine for cross-sectional ranking across the Russell 1000. 
  >
  > Looking at `output/scores/alpha158_russell1000_latest.csv`, every single stock—`MSFT`, `NVDA`, `SPY`—has the exact same score of `-3.545e-09`. When we feed this to an optimizer, the rank correlation (Rank IC) is precisely `0.0000` and ICIR is `0.0000`. The optimizer cannot form long/short quintiles because there is zero cross-sectional dispersion. It's a degenerate distribution. If we applied 3x or 5x institutional leverage to this signal, we would be leveraging pure unmodeled idiosyncratic risk with zero expected return. We cannot accept a model with zero splits and zero information coefficient."*

### 3.3 The Chief Analyst
- **Core Mandate**: Fundamental validation and macroeconomic contextualization (1-Year & 3-Year Horizons).
- **Evaluation & Verdict**: **REJECTED**
- **Statement**:
  > *"The algorithm's revenue projections must be anchored in reality. When an executive or investment committee looks at this dashboard, the Alpha158 card is supposed to synthesize 158 quantitative indicators into a coherent technical and macro thesis.
  > 
  > Presenting `Column_0`, `Column_1`, `Column_2`, and `Column_3` destroys analytical credibility. Alpha158 features have distinct financial meanings: price volume trend (`PVT`), volume-weighted average price (`VWAP`), moving average convergence, and volatility ratios. In an earnings season or a shifting Fed rate environment, I need to verify whether the model is penalizing high-debt capital structures or rewarding operating leverage. Unlabeled column indices make fundamental stress-testing impossible. The factor attribution must be mapped to human-readable quantitative metrics."*

### 3.4 The Global Finance Manager
- **Core Mandate**: Capital allocation, liquidity management, structural compounding (3-Year & 10-Year Horizons).
- **Evaluation & Verdict**: **REJECTED**
- **Statement**:
  > *"Our mandate is sustainable, long-term capital preservation and compounded returns. A model producing flat `-0.00000` scores introduces insidious structural drag. If downstream execution algorithms or automated rebalancing scripts rely on this score without threshold validation, capital will be misallocated or trapped in cash equivalents.
  > 
  > Furthermore, regulatory compliance, client reporting, and LP due diligence require rigorous model governance. Exporting unmapped `Column_0` features into official analysis reports violates model explainability standards (SR 11-7 / OCC guidelines for model risk management). Every feature entering production reports must have complete lineage, description, and economic intuition."*

### 3.5 The Quant Developer
- **Core Mandate**: Model integrity, statistical arbitrage, and algorithmic backtesting (All Horizons: 1-Month to 10-Year).
- **Evaluation & Verdict**: **ROOT CAUSE DIAGNOSED & TECHNICAL BLUEPRINT PREPARED**
- **Statement & Technical Diagnosis**:
  > *"The math and the code reveal two distinct, clean bugs that caused this behavior:*
  > 
  > #### Root Cause 1: Severe Hyperparameter Over-Regularization on a Small Universe
  > * In `workflow_config_lightgbm_Alpha158_us_russell1000.yaml`, lines 40-41:
  >   ```yaml
  >   lambda_l1: 205.6999
  >   lambda_l2: 580.9768
  >   ```
  > * These hyperparameter values were copied directly from an academic benchmark configuration originally tuned for the Chinese CSI300 universe (3,000+ instruments over 10+ years with unnormalized features).
  > * In our current local dataset environment (`~/.qlib/qlib_data/us_data`), only 3 instruments are actively populated (`MSFT`, `NVDA`, `SPY`).
  > * Alpha158 applies `CSZScoreNorm` (cross-sectional Z-score normalization) daily across all universe tickers:
  >   $$\tilde{x}_{i,t} = \frac{x_{i,t} - \mu_t}{\sigma_t}$$
  > * Because the normalized gradients $G$ and Hessians $H$ across 3 tickers are small ($O(1)$), the LightGBM split gain criterion:
  >   $$\Delta \text{Loss} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$$
  >   with $\lambda = 580.9768$ yields a gain of essentially **zero** for every candidate split.
  > * LightGBM therefore terminated at tree 0 without splitting:
  >   ```text
  >   Tree=0
  >   num_leaves=1
  >   leaf_value=-3.5453779018694515e-09
  >   ```
  > * **Solution**: Calibrate `lambda_l1` and `lambda_l2` to reasonable levels for US equity data (e.g. `lambda_l1: 0.1`, `lambda_l2: 1.0`, `min_child_samples: 5`, `num_leaves: 31`).
  > 
  > #### Root Cause 2: Feature Name Stripping via `.values` in Qlib Dataset Creation
  > * In `qlib/contrib/model/gbdt.py` (`LGBModel._prepare_data`):
  >   ```python
  >   x, y = df["feature"], df["label"]
  >   ds_l.append((lgb.Dataset(x.values, label=y, ...)))
  >   ```
  > * Passing `x.values` converts the pandas DataFrame into a raw 2D NumPy array. Because LightGBM is not supplied with the column names, it assigns default names `Column_0`, `Column_1`, ..., `Column_157`.
  > * When `train_alpha158_lightgbm.py` executes:
  >   ```python
  >   feature_names = trained_model.model.feature_name()
  >   ```
  >   it extracts these generic string tokens (`Column_0` through `Column_157`) and writes them directly to `alpha158_russell1000_latest_meta.json`.
  > * **Solution**: 
  >   1. Provide the true feature names (`feature_names=list(x.columns)`) during training, or map `Column_i` indices directly to the canonical Alpha158 feature definition names from `Alpha158DL.get_feature_config()`.
  >   2. Update `infer_alpha158.py` to map any remaining `Column_i` tokens to their standard technical names (`KMID`, `KLEN`, `ROC5`, `ROC20`, `MA5`, `MA60`, `STD20`, `VWAP0`, etc.) with human-readable descriptions."*

---

## 4. The Synthesis & Interrogation Engine: The Billionaire (The Principal)

As the conflict-resolution and capital-deployment node, **The Principal** interrogates the council:

### Interrogations
1. **To the Trader**:
   - *Question*: *"If we execute trades based on this software's current 1-month output, how much does my liquid net worth grow, and what is the probability of loss?"*
   - *Trader Response*: *"Zero dollar growth. The 5-day excess return prediction is +0.00%, and the score is -0.00000. It provides zero directional bias, meaning trades executed on it are pure random coin flips after paying bid-ask spread and commissions."*
2. **To the Quant Developer**:
   - *Question*: *"Why was a model with 1 tree and 0 splits promoted to production status (`alpha158_russell1000_latest.pkl`) without an automated gate rejecting it?"*
   - *Quant Response*: *"The training script `train_alpha158_lightgbm.py` checked for file creation but lacked a minimum validation threshold (e.g. `num_trees > 1`, `num_leaves > 1`, and `rank_ic != 0.0`). The script wrote the artifact to disk regardless of split depth. We must introduce a CI/CD Model Gate: any model with zero splits or zero IC must fail the build immediately."*
3. **To the Top Hedge Fund Manager**:
   - *Question*: *"If we feed this data into the multi-period projection engine or the executive buy-timing verdict, does it corrupt other cards?"*
   - *Manager Response*: *"Fortunately, our modular architecture protected the other engines: the Dealer GEX card, BOCD regime card, AVWAP microstructure card, and Monte Carlo engine ran on their independent pipelines. However, the Alpha158 card itself is an embarrassment and cannot be presented to capital partners."*
4. **To the Chief Analyst**:
   - *Question*: *"Once the Quant fixes the column mapping, will you be able to explain the factor attribution directly to our investment committee?"*
   - *Analyst Response*: *"Yes. When `Column_0` is correctly rendered as `ROC20` (20-Day Rate of Change) or `MA60` (60-Day Trend), we can immediately cross-validate whether the algorithm's bullish conviction aligns with MSFT's cloud growth and enterprise capex cycle."*

### Final Capital Deployment Verdict
> **THE PRINCIPAL'S EXECUTIVE ORDER**:
> **CAPITAL ALLOCATION TO ALPHA158 SIGNALS IS FROZEN.**  
> The LightGBM Alpha158 model is declared **UNFIT FOR LIVE TRADING** in its current degenerate state.  
> `team-code` is instructed to execute an immediate 2-part remediation:
> 1. **Calibrate Training Hyperparameters**: Lower L1/L2 regularization (`lambda_l1=0.1`, `lambda_l2=1.0`, `min_child_samples=5`) so trees split dynamically and generate non-zero alpha predictions and non-zero Rank IC.
> 2. **Restore Feature Name Integrity**: Map all 158 factor columns from Qlib's `Alpha158DL` specification (`KMID`, `ROC20`, `MA60`, `VWAP0`, `STD20`, etc.) both at training time and in `infer_alpha158.py`, eliminating all traces of `Column_0`.
> 3. **Regenerate Artifacts**: Retrain the model, re-score the universe, and re-compile `MSFT_analysis_report_2026-09-05.html` with verified non-zero alpha and named factors.

---

## 5. Probability and Earnings Evaluation Matrix

| Time Horizon | Primary Evaluating Agents | Optimization Focus | Minimum Probability Threshold | Model Status | Council Finding |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1-Month** | High-Earning Trader, Quant | Momentum / Mean Reversion | **> 75%** | **FAILED (0%)** | Score = -0.00000; zero directional velocity |
| **6-Month** | Trader, HF Manager, Quant | Event-driven / Trend following | **> 70%** | **FAILED (0%)** | Zero split trees; no trend capture |
| **1-Year** | HF Manager, Analyst, Quant | Macro regime capture | **> 80%** | **FAILED (0%)** | Rank IC = 0.0000; zero cross-sectional dispersion |
| **3-Year** | Analyst, Finance Mgr, Quant | Fundamental compounding | **> 85%** | **FAILED (0%)** | `Column_0` opacity; zero fundamental lineage |
| **10-Year** | Finance Mgr, Quant | Capital preservation / Growth | **> 90%** | **FAILED (0%)** | Model governance violation (SR 11-7 non-compliance) |

---

## 6. Actionable Implementation Mandates for `team-code`

1. **`workflow_config_lightgbm_Alpha158_us_russell1000.yaml`**:
   - Replace CSI300 over-regularization (`lambda_l1: 205.6999`, `lambda_l2: 580.9768`) with US equity parameters:
     - `lambda_l1: 0.1`
     - `lambda_l2: 1.0`
     - `min_child_samples: 5`
     - `num_leaves: 31`
     - `learning_rate: 0.05`
2. **`train_alpha158_lightgbm.py`**:
   - Extract actual Alpha158 feature names from `dataset.prepare("train").columns` or `Alpha158DL.get_feature_config()` and pass them to LightGBM or store the explicit mapping in `alpha158_russell1000_latest_meta.json`.
   - Add a Quality Gate: Assert that `num_trees > 1` and `num_leaves > 1` before writing `_latest.*` artifacts.
3. **`infer_alpha158.py`**:
   - Implement a fallback feature name lookup mapping index $i \to \text{factor\_name}_i$ (e.g. `0 -> KMID`, `1 -> KLEN`, ..., `ROC20`, `MA60`) so that even legacy metadata gracefully displays real quantitative names.
4. **`visualize_stock_analysis.py`**:
   - Ensure the card formats non-zero scores, non-zero percentiles, and positive/negative gains properly with full tooltips and descriptions.

---
*Signed by the Council of Six:*  
- **The High-Earning Trader**  
- **The Top Hedge Fund Manager**  
- **The Chief Analyst**  
- **The Global Finance Manager**  
- **The Quant Developer**  
- **The Billionaire (The Principal)**
