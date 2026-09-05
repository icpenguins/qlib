# Team-Code Implementation Plan: Realistic Mega-Cap GEX Scale & Strike Dispersion Engine

**Document ID**: `20260904-realistic_mega_cap_gex_and_strike_dispersion_plan`  
**Date**: 2026-09-04 / 2026-09-05  
**Author**: `team-code-architect` & Multi-Agent Quantitative Team  
**Status**: PENDING USER APPROVAL  

---

## 1. Printed Acknowledgement of End-User Requirements (Priority Requirement -1)

The entire `team-code` software development pipeline hereby formally acknowledges and attests full understanding of the requirements of our primary end-users:

1. **The Profitable Stock Trader** (*Veteran Discretionary & Quantitative Prop Trader*):
   - **Mandate**: Consistent alpha, capital preservation, exploiting asymmetric risk/reward setups, avoiding catastrophic drawdowns.
   - **Requirement**: "For the ticker MSFT Call Wall = Put Wall = Max Pain, all exactly $510.00. In real options-chain gamma exposure math these three levels are independently derived from open interest and almost never land on the identical strike. That's a strong signal the gamma-wall engine is defaulting to 'nearest round strike to spot' rather than doing a genuine OI-weighted calculation — I would not trust this level for pinning/breakout trades without independent verification against live options data. Total call open interest 7,660 and put open interest 11,008 are not Mega-Cap figures. Live MSFT open interest is orders of magnitude larger. Net GEX of −$5.1 million per 1 percent is likewise a toy book."
   - **Trader Imperative**: Strike collapse ($K_{\text{call\_wall}} == K_{\text{put\_wall}} == K_{\text{max\_pain}}$) must be eradicated. Mega-cap volume and GEX must represent institutional capital reality (millions of contracts, hundreds of millions in GEX). A clear provenance disclosure must be shown so discretionary traders know when levels are derived from calibrated synthetic models vs live OPRA exchange data.

2. **The Institutional Hedge Fund Manager** (*CIO / Head of Quantitative Research*):
   - **Mandate**: Double-digit net annualized returns, Sharpe ratio $> 2.0$, net zero market/factor beta, zero catastrophic drawdown tolerance.
   - **Requirement**: Mathematical rigor across derivatives surfaces. Asymmetric volatility skew and open interest distributions reflecting empirical market microstructure (institutional covered call selling above spot, downside tail-risk hedging below spot). Clean exchange-standard discrete strike increments ($5.00 for $250+, $2.50 for $100–$250, $1.00 for $25–$100). Mathematical non-degeneracy guarantees ($K_{\text{put\_wall}} < S_0 < K_{\text{call\_wall}}$).

---

## 2. Problem Diagnosis & Mathematical Root Causes

1. **Hardcoded Fixed Base Open Interest in `SyntheticOptionSurfaceGenerator`**:
   In `qlib/contrib/derivatives/options_data.py`:
   ```python
   base_call_oi = 500.0 * math.exp(...) * pin_multiplier
   base_put_oi = 650.0 * math.exp(...) * pin_multiplier
   ```
   Constants of 500 and 650 with 25 strikes yielded total open interest of ~7,660 calls and 11,008 puts regardless of whether the ticker was an illiquid micro-cap or Microsoft ($3.8T market cap, ~22M shares daily volume). This produced Net GEX of only -$5.1M/1%, which is two orders of magnitude smaller than institutional reality.

2. **Symmetric ATM Pinning on a Fractional Strike Grid**:
   - Strike grid was generated with `step = round(max(1.0, spot * 0.015), 1)` (yielding $7.60 for spot $505).
   - A single strike ($510.00) was divisible by 10, receiving a 2.5x multiplier for *both* calls and puts, while neighboring strikes received 1.0x.
   - Since $510.00 was also the closest strike to spot, Black-Scholes Gamma ($\Gamma$) peaked at $510.00$.
   - Consequently, $510.00$ simultaneously maximized Call GEX ($\text{OI} \times \Gamma \times S^2 \times 0.01$), maximized negative Put GEX ($-\text{OI} \times \Gamma \times S^2 \times 0.01$), and minimized intrinsic option seller payouts (Max Pain). All three metrics collapsed to $510.00$.

3. **Absence of Economic Asymmetry & Invariant Bounds in `DealerGammaEngine`**:
   - Institutional open interest is economically asymmetric:
     - Covered call writing and upside speculative call buying concentrate open interest **above spot** ($m = \ln(K/S) > 0$).
     - Downside portfolio put protection concentrates open interest **below spot** ($m < 0$).
   - `DealerGammaEngine.compute_gex` performed a global unconstrained `idxmax()` and `idxmin()` without enforcing the structural boundary conditions:
     $$K_{\text{put\_wall}} < S_0 < K_{\text{call\_wall}}$$

---

## 3. Detailed Proposed Implementation

### A. Strike Grid Standardization (`SyntheticOptionSurfaceGenerator`)
Replace continuous/fractional step sizes with standard US exchange-listed strike increments:
- For $S_0 \ge \$250$: strike step = $\$5.00$ (e.g. MSFT, NVDA, META)
- For $\$100 \le S_0 < \$250$: strike step = $\$2.50$
- For $\$25 \le S_0 < \$100$: strike step = $\$1.00$
- For $S_0 < \$25$: strike step = $\$0.50$
Generate grid spanning $\pm 18\%$ to $\pm 22\%$ around spot, anchored to round strike multiples.

### B. Liquidity-Aware Open Interest Scaling (`SyntheticOptionSurfaceGenerator`)
Accept `adtv: Optional[float] = None` and `symbol: Optional[str] = None`.
1. Identify Mega-Cap universe: `{"MSFT", "NVDA", "AAPL", "AMZN", "GOOGL", "GOOG", "META", "TSLA"}` or tickers with $\text{ADTV} \ge 10{,}000{,}000$ shares.
2. Calculate base open interest per strike:
   $$\text{base\_oi\_scale} = \max\left(2500.0, \frac{\text{adtv} \times 0.08}{\text{num\_strikes}}\right)$$
   - For MSFT ($\text{ADTV} \approx 22\text{M}$): $\text{base\_oi\_scale} \approx 58{,}000$ contracts.
   - Total open interest across the chain: $\approx 1.5\text{M} - 3.0\text{M}$ contracts.
   - Net GEX: hundreds of millions of dollars per 1% move, matching institutional options book scale.

### C. Asymmetric Open Interest & Pin Mechanics
1. **Calls**:
   - Centered out-of-the-money above spot: $m_{\text{call}} \approx +0.035$ to $+0.045$.
   - Round pin multiplier ($2.5\times$ for multiples of 10, $1.8\times$ for multiples of 5) applies with primary weight to strikes **above spot** ($K > S_0$).
2. **Puts**:
   - Centered out-of-the-money below spot: $m_{\text{put}} \approx -0.045$ to $-0.065$.
   - Round pin multiplier applies with primary weight to strikes **below spot** ($K < S_0$).
3. **Natural Dispersion**:
   - Call Wall will naturally emerge at a major round strike above spot (e.g. $\$520.00$).
   - Put Wall will naturally emerge at a major round strike below spot (e.g. $\$490.00$).
   - Max Pain will balance between them (e.g. $\$505.00$ or $\$510.00$).

### D. Structural Invariants in `DealerGammaEngine.compute_gex`
Add structural domain filters and non-degeneracy assertions:
```python
# Call Wall: Resistance at or above spot
otm_calls = strike_groups[strike_groups["strike"] >= spot]
if not otm_calls.empty:
    call_wall = float(otm_calls.loc[otm_calls["call_gex_dollar"].idxmax()]["strike"])
else:
    call_wall = float(strike_groups.loc[strike_groups["call_gex_dollar"].idxmax()]["strike"])

# Put Wall: Support at or below spot
otm_puts = strike_groups[strike_groups["strike"] <= spot]
if not otm_puts.empty:
    put_wall = float(otm_puts.loc[otm_puts["put_gex_dollar"].idxmin()]["strike"])
else:
    put_wall = float(strike_groups.loc[strike_groups["put_gex_dollar"].idxmin()]["strike"])

# Guaranteed non-degeneracy invariant:
if call_wall <= put_wall:
    # Separate to next distinct strikes strictly above and below spot
    ...
```

### E. Pipeline Data Propagation
1. `qlib/contrib/derivatives/options_data.py`:
   - Update `OptionsDataLoader.get_options_chain` and `load_or_generate_chain` to accept `adtv: Optional[float] = None`.
2. `scripts/stock_analysis_engine.py`:
   - Compute 20-day mean volume (`adtv = float(df["volume"].tail(20).mean())`) and pass to `loader.load_or_generate_chain`.
3. `scripts/stock_analysis_data.py`:
   - Pass `vol_mean` and `symbol` to `SyntheticOptionSurfaceGenerator.generate_synthetic_chain`.
4. `scripts/visualize_stock_analysis.py`:
   - In `_build_calibrated_derivatives_fallback`: accept `symbol` and `adtv` and pass to generator.
   - In `build_derivatives_card_html`:
     - Render prominent provenance badge: `PROVENANCE: SYNTHETIC RESEARCH CHAIN (UNVERIFIED LIVE OPTIONS)`.
     - Render caution banner:
       `⚠️ Research Provenance: Model-calibrated synthetic surface. Call/Put walls and Max Pain reflect theoretical open interest distributions scaled to ticker liquidity. For real-capital pinning or breakout execution, independent verification against live OPRA exchange data is required.`
     - Enforce UI clamp invariant `assert put_wall < call_wall`.

---

## 4. Verification Plan

### A. Automated Unit & Integration Tests
1. `tests/test_derivatives_gex.py`:
   - Add `test_mega_cap_gex_scale_and_strike_dispersion`:
     - Test MSFT (spot $505, ADTV 22M):
     - Assert total call OI $> 500{,}000$ and total put OI $> 500{,}000$.
     - Assert $| \text{Net GEX} | > \$20\text{M}$ per 1% move.
     - Assert strike dispersion invariant: $K_{\text{put\_wall}} < S_0 < K_{\text{call\_wall}}$.
     - Assert Max Pain is distinct and sits between or at boundary: $K_{\text{put\_wall}} \le K_{\text{max\_pain}} \le K_{\text{call\_wall}}$.
     - Assert all strikes adhere to $\$5.00$ discrete increments.
2. Full Test Suite:
   - Run `python scripts/run_all_tests.py` ensuring all 84+ tests pass.

### B. End-to-End Report Verification
Generate MSFT and TEST reports and verify:
- HTML contains proper mega-cap open interest figures and Net GEX in tens/hundreds of millions.
- Call Wall, Put Wall, and Max Pain are distinct and properly spaced.
- Provenance badge and trader caution banner are prominently displayed.

