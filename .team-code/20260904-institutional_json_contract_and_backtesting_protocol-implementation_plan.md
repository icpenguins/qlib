# Implementation Plan: Full Institutional JSON Contract & Backtesting Protocol Integration

## Executive Overview & Problem Statement
In the previous implementation phase, all 17 modular mathematical and microstructure algorithms were separated into single-responsibility modules across `qlib/contrib/` and validated by isolated unit test suites. However, as identified in the review of [`20260904-finance_team_review_stock_analysis_data.md`](file:///e:/SRC/GITHUB/my-qlib/.team-code/20260904-finance_team_review_stock_analysis_data.md), the serialized JSON output produced by [`scripts/stock_analysis_data.py`](file:///e:/SRC/GITHUB/my-qlib/scripts/stock_analysis_data.py) omitted the full results of this work.

Specifically, the emitted JSON object is missing:
1. **`calibrate_post_earnings_volatility_surface`**: The explicitly requested post-earnings volatility surface calibration function and its comprehensive parameters (expected jump percentage, event variance, post-earnings IV, dollar implied moves, and crush ratios).
2. **`backtesting_protocol`**: The exhaustive institutional backtesting section requested throughout Sections 5, 6, 7, 9.6, and 9.7 of the finance team review, including:
   - **`purged_walk_forward_cv`**: 3Y train (756d), 1Y test (252d), 10d post-earnings embargo, fold metrics, and hard invariant assertion.
   - **`almgren_chriss_market_impact`**: Non-linear permanent ($\gamma$) and temporary ($\eta, \alpha=0.5$) market impact, fixed fees, and effective fill prices.
   - **`borrow_fee_engine`**: Short locate capacity checks, hard-to-borrow (HTB) rate thresholds ($\ge 10\%$), daily borrow fee accruals, and zero-locate order rejection enforcement.
   - **`deflated_sharpe_ratio`**: Bailey & López de Prado dynamic multiple-testing hurdle $\mathbb{E}[\max(\text{SR}_0)]$, empirical Sharpe variance across trials ($N_{\text{trials}} = 144$), skewness, kurtosis, and statistical significance ($p < 0.05$).
   - **`verifiable_replication_event_panel`**: The 10-year S&P 500 survivorship-bias-free replication panel ($N_{\text{events}} = 18,420$, win rate $83.6\%$, avg jump $+8.4\%$, loss prob $> 2\%$ of $4.2\%$, profit factor $3.45$).
   - **`strategy_rules`**: Detailed programmatic entry, exit, stop-loss, and profit-taking rules for $\text{GSI}^+$ (Earnings Beat), $\text{GSI}^-$ (Earnings Miss), and Variance Risk Premium (VRP) Harvest.
3. **`factor_orthogonalization`**: WLS factor projection matrix results isolating idiosyncratic dealer gamma alpha from Size, Momentum, Volatility, and Short Interest.
4. **`earnings_event_clock`**: Point-in-Time phase scheduling (AMC vs. BMO), signal formation timestamps ($T_0$ 15:55 MOC), news release timestamps ($T_0$ 16:01 or $T_1$ 07:00), execution fill timestamps ($T_1$ 09:30 Open / 10:00 VWAP), and prohibition of $T_0$ close fills.
5. **`council_interrogation_outcomes` & `evaluation_matrix`**: The quantitative answers to The Billionaire's interrogation for all 5 council members and the 6-horizon probability evaluation matrix.

This plan establishes the architecture to integrate all of these institutional components directly into the JSON object emitted by `stock_analysis_data.py`.

---

## User Review Required

> [!IMPORTANT]
> **Schema Enrichment & Backward Compatibility**:
> All new data structures will be added under well-defined, versioned keys in the JSON contract:
> - `earnings_gamma_squeeze`: Enriched with `calibrate_post_earnings_volatility_surface`, `factor_orthogonalization`, and `earnings_event_clock`.
> - `backtesting_protocol`: Added both as a top-level section in the JSON document (for direct ingestion by institutional risk systems) and referenced inside `earnings_gamma_squeeze` for seamless discoverability.
> - `evaluation_matrix`: Added to the JSON document to capture multi-horizon institutional targets ($t+1 \to t+5$, 1M, 6M, 1Y, 3Y, 10Y).
> Existing front-end visualizer consumers and tests reading `stock_analysis_data.py` will remain 100% compatible.

---

## Target JSON Output Specification

When `stock_analysis_data.py` executes (programmatically or via CLI), the resulting `.json` file will contain the following enriched structure:

```json
{
  "metadata": {
    "symbol": "AAPL",
    "request_date": "2025-10-31",
    "latest_data_date": "2025-10-31",
    "is_up_to_date": true,
    "forecast_days": 63,
    "generated_at": "2026-09-04T15:30:00.000000",
    "contract_version": "1.2.0"
  },
  "symbol": "AAPL",
  "historical_data": [ ... ],
  "performance": { ... },
  "best_buys": [ ... ],
  "predictive": { ... },
  "projections": { ... },
  "regime": { ... },
  "microstructure": { ... },
  "derivatives": { ... },
  "events": { ... },
  "earnings_gamma_squeeze": {
    "is_actionable": true,
    "provenance": "historical_opra_eod",
    "safety_status": "PRODUCTION_CLEAR",
    "gate_violations": [],
    "calibrate_post_earnings_volatility_surface": {
      "spot": 100.0,
      "atm_straddle_price": 7.15,
      "pre_earnings_iv": 0.45,
      "realized_21d_vol": 0.25,
      "dte_days": 7,
      "expected_jump_pct": 8.56,
      "event_variance": 0.0382,
      "post_earnings_iv": 0.3204,
      "implied_move_dollars": 8.56,
      "volatility_crush_pct": 28.8
    },
    "iv_crush_model": {
      "crush_ratio": 0.288,
      "crush_source": "empirical_winsorized_median",
      "is_empirical": true,
      "observed_quarters": 8
    },
    "forced_dealer_hedging": {
      "-0.15": { "shares_demand": -2450000.0, "lir": 2.45 },
      "-0.1": { "shares_demand": -1650000.0, "lir": 1.65 },
      "-0.05": { "shares_demand": -780000.0, "lir": 0.78 },
      "0.05": { "shares_demand": 820000.0, "lir": 0.82 },
      "0.1": { "shares_demand": 1840000.0, "lir": 1.84 },
      "0.15": { "shares_demand": 2950000.0, "lir": 2.95 }
    },
    "liquidity_impact": {
      "bullish_lir_10pct": 1.84,
      "bearish_lir_10pct": 1.65
    },
    "gsi_scores": {
      "gsi_positive_raw": 82.4,
      "gsi_negative_raw": 28.5,
      "is_positive_alert": true,
      "is_negative_alert": false
    },
    "factor_orthogonalization": {
      "is_orthogonalized": true,
      "gsi_raw": 82.4,
      "gsi_orthogonal": 76.8,
      "factor_exposures": {
        "size_market_cap": 0.12,
        "momentum_12m": 0.08,
        "volatility_21d": -0.05,
        "short_interest": 0.15
      },
      "idiosyncratic_alpha_ratio": 0.932,
      "projection_method": "WLS projection against [1, ln(Size), Mom12M, Vol21D, ShortInterestFloat]"
    },
    "calibrated_probabilities": {
      "p_positive_squeeze": 0.815,
      "p_negative_cascade": 0.124,
      "conformal_bounds_positive": [0.735, 0.895],
      "conformal_bounds_negative": [0.044, 0.204]
    },
    "earnings_event_clock": {
      "reporting_time": "AMC",
      "signal_timestamp": "2025-10-31 15:55:00",
      "announcement_timestamp": "2025-10-31 16:01:00",
      "execution_timestamp": "2025-11-03 09:30:00",
      "execution_fill_type": "T1_OPEN",
      "disallowed_fill_rule": "T0_CLOSE physically prohibited by EarningsEventClock",
      "is_compliant": true
    },
    "recommended_action": "STRONG_POSITIVE_GAMMA_SQUEEZE",
    "acceleration_corridors": {
      "upper_squeeze_wall": 108.56,
      "lower_trapdoor": 91.44
    }
  },
  "backtesting_protocol": {
    "purged_walk_forward_cv": {
      "train_window_days": 756,
      "test_window_days": 252,
      "embargo_days": 10,
      "step_days": 252,
      "n_folds": 7,
      "zero_overlap_invariant_asserted": true,
      "validation_sharpe_mean": 2.15,
      "out_of_sample_sharpe": 2.42
    },
    "almgren_chriss_market_impact": {
      "participation_rate": 0.05,
      "permanent_impact_bps": 1.25,
      "temporary_impact_bps": 3.42,
      "fixed_fee_bps": 5.0,
      "total_slippage_bps": 9.67,
      "effective_fill_price": 100.97
    },
    "borrow_fee_engine": {
      "short_value_tested": 1000000.0,
      "annual_borrow_rate": 0.0050,
      "is_hard_to_borrow": false,
      "locate_granted": true,
      "daily_accrued_cost_dollars": 13.89,
      "zero_locate_rejection_rule_active": true
    },
    "deflated_sharpe_ratio": {
      "best_sharpe": 2.42,
      "expected_max_sharpe_hurdle": 2.31,
      "dsr_probability": 0.962,
      "is_statistically_significant": true,
      "n_trials": 144,
      "sample_length_days": 2520,
      "skewness": -0.42,
      "kurtosis": 3.85,
      "bailey_lopez_de_prado_hurdle_formula": "E[max(SR_0)] = sqrt(2*ln(N)) + gamma_EM/sqrt(2*ln(N))"
    },
    "verifiable_replication_event_panel": {
      "sample_period": "2015-01-01 to 2024-12-31",
      "universe": "S&P 500 Survivorship-Bias-Free Point-In-Time",
      "n_events": 18420,
      "win_rate": 0.836,
      "avg_trade_jump_pct": 8.4,
      "loss_probability_gt_2pct": 0.042,
      "profit_factor": 3.45,
      "max_drawdown_pct": 8.2
    },
    "strategy_rules": {
      "gsi_bull_entry": "GSI+ >= 75.0 and SUE > 0.5 -> Buy equity or front-week calls at T1 Open",
      "gsi_bull_exit": "Trail stop at Major Call Wall or exit at T5 Close",
      "gsi_bear_entry": "GSI- >= 75.0 and SUE < -0.5 -> Short equity or front-week puts at T1 Open",
      "gsi_bear_exit": "Cover at Major Put Wall or exit at T3 Close",
      "vrp_harvest_entry": "GSI+ < 40 and GSI- < 40 -> Sell ATM strangles at T0 Close, buy back at T1 Open"
    },
    "council_interrogation_outcomes": {
      "high_earning_trader": {
        "allocation_tested": 10000000,
        "gross_profit_per_trade": 840000,
        "loss_probability_gt_2pct": 0.042,
        "win_rate": 0.836
      },
      "quant_developer": {
        "alpha_decay_annual_pct": 3.1,
        "half_life_years": 6.2,
        "dealer_hedging_mandate": "FINRA/OCC Delta Neutrality"
      },
      "top_hedge_fund_manager": {
        "unconstrained_3x_margin_call_risk": 0.184,
        "degrossed_3x_margin_call_risk": 0.0008,
        "sharpe_ratio": 2.42
      },
      "global_finance_manager": {
        "net_compounded_growth_pct": 15.8,
        "principal_doubling_years": 4.7,
        "tax_structure": "Section 1256 60/40 blended capital gains"
      },
      "council_multi_horizon_consensus": {
        "net_annualized_cagr": 16.4,
        "bootstrap_95ci": [14.2, 18.9]
      }
    }
  },
  "evaluation_matrix": {
    "t_plus_1_to_t_plus_5": {
      "evaluating_agents": "High-Earning Trader, Quant",
      "focus": "Earnings Gamma Squeeze / Liquidation Cascade",
      "min_probability_threshold": 0.78,
      "target_output": "Immediate cash velocity ($840k per $10M trade) exploiting forced dealer re-hedging"
    },
    "1_month": {
      "evaluating_agents": "High-Earning Trader, Quant",
      "focus": "PEAD Momentum / AVWAP Rebound",
      "min_probability_threshold": 0.75,
      "target_output": "Rapid monthly cash generation without capital lockup"
    },
    "6_month": {
      "evaluating_agents": "Trader, HF Manager, Quant",
      "focus": "Event-driven / Trend following",
      "min_probability_threshold": 0.70,
      "target_output": "Scalable quarterly alpha via BOCD regime transitions"
    },
    "1_year": {
      "evaluating_agents": "HF Manager, Analyst, Quant",
      "focus": "Macro regime capture",
      "min_probability_threshold": 0.80,
      "target_output": "Maximum risk-adjusted Annual Recurring Revenue (Sharpe > 2.0)"
    },
    "3_year": {
      "evaluating_agents": "Analyst, Finance Mgr, Quant",
      "focus": "Fundamental compounding",
      "min_probability_threshold": 0.85,
      "target_output": "Structural market share, secular earnings growth"
    },
    "10_year": {
      "evaluating_agents": "Finance Mgr, Quant",
      "focus": "Capital preservation / Growth",
      "min_probability_threshold": 0.90,
      "target_output": "Legacy wealth compounding and structural tax shielding"
    }
  }
}
```

---

## Proposed Changes

### Component 1: Derivatives Layer ([`qlib/contrib/derivatives/`](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/derivatives/))

#### [MODIFY] [`post_earnings_volatility.py`](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/derivatives/post_earnings_volatility.py)
- Export `calibrate_post_earnings_volatility_surface` matching the exact signature from Section 5.4 of `20260904-finance_team_review_stock_analysis_data.md`.
- Return detailed surface metrics dictionary including:
  - `expected_jump_pct`
  - `post_earnings_iv`
  - `event_variance`
  - `pre_earnings_iv`
  - `realized_21d_vol`
  - `atm_straddle_price`
  - `implied_move_dollars`
  - `volatility_crush_pct`
- Maintain `calibrate_post_earnings_volatility` as a backwards-compatible alias returning `Tuple[float, float]`.

#### [MODIFY] [`earnings_gamma_squeeze_engine.py`](file:///e:/SRC/GITHUB/my-qlib/qlib/contrib/derivatives/earnings_gamma_squeeze_engine.py)
- Integrate:
  - `calibrate_post_earnings_volatility_surface`
  - `resolve_earnings_event_execution`
  - `orthogonalize_gsi_factors`
  - `calculate_market_impact`
  - `calculate_borrow_cost`
  - `calculate_deflated_sharpe_ratio`
  - `PurgedWalkForwardCV`
- Synthesize:
  - `calibrate_post_earnings_volatility_surface`
  - `factor_orthogonalization`
  - `earnings_event_clock`
  - Full institutional `backtesting_protocol` block
  - Full institutional `evaluation_matrix` block

---

### Component 2: Data Pipeline & Serialization Layer ([`scripts/`](file:///e:/SRC/GITHUB/my-qlib/scripts/))

#### [MODIFY] [`stock_analysis_data.py`](file:///e:/SRC/GITHUB/my-qlib/scripts/stock_analysis_data.py)
- Update `prepare_analysis_json_payload`:
  - Ensure `contract_version` is bumped to `"1.2.0"`.
  - Expose `calibrate_post_earnings_volatility_surface` inside `earnings_gamma_squeeze`.
  - Expose `factor_orthogonalization` and `earnings_event_clock` inside `earnings_gamma_squeeze`.
  - Expose `backtesting_protocol` as a top-level section in the JSON payload and cross-referenced in `earnings_gamma_squeeze`.
  - Expose `evaluation_matrix` as a top-level section in the JSON payload.
- Update CLI summary to display `Vol Surface Calibration`, `Backtest DSR (p-value)`, and `Replication Panel Win Rate`.

---

### Component 3: Test Suites ([`tests/`](file:///e:/SRC/GITHUB/my-qlib/tests/))

#### [MODIFY] [`test_post_earnings_volatility.py`](file:///e:/SRC/GITHUB/my-qlib/tests/test_post_earnings_volatility.py)
- Add unit tests for `calibrate_post_earnings_volatility_surface` testing both tuple and dictionary outputs, variance floor bounds, and crash handling.

#### [MODIFY] [`test_earnings_gamma_squeeze_engine.py`](file:///e:/SRC/GITHUB/my-qlib/tests/test_earnings_gamma_squeeze_engine.py)
- Assert that `evaluate_earnings_gamma_squeeze` output contains:
  - `calibrate_post_earnings_volatility_surface`
  - `backtesting_protocol` (with `purged_walk_forward_cv`, `almgren_chriss_market_impact`, `borrow_fee_engine`, `deflated_sharpe_ratio`, `verifiable_replication_event_panel`)
  - `factor_orthogonalization`
  - `earnings_event_clock`

#### [MODIFY] [`test_stock_analysis_data.py`](file:///e:/SRC/GITHUB/my-qlib/tests/test_stock_analysis_data.py)
- Update contract validation tests to assert that `prepare_analysis_json_payload` and exported `.json` files strictly contain:
  - Top-level `backtesting_protocol`
  - Top-level `evaluation_matrix`
  - Nested `calibrate_post_earnings_volatility_surface`
  - Nested `factor_orthogonalization`
  - Nested `earnings_event_clock`

---

### Component 4: Part 2 Governance Documentation ([`.team-code/`](file:///e:/SRC/GITHUB/my-qlib/.team-code/))

#### [NEW] [`.team-code/calibrate_post_earnings_volatility_surface.md`](file:///e:/SRC/GITHUB/my-qlib/.team-code/calibrate_post_earnings_volatility_surface.md)
- Dedicated specification document for `calibrate_post_earnings_volatility_surface`.

#### [NEW] [`.team-code/20260904-institutional_json_contract_and_backtesting_protocol-implementation_plan.md`](file:///e:/SRC/GITHUB/my-qlib/.team-code/20260904-institutional_json_contract_and_backtesting_protocol-implementation_plan.md)
- Synchronized repository implementation plan for version control and auditability.

---

## Verification Plan

### Automated Tests
1. **Targeted Volatility Surface Test**:
   ```powershell
   & .venv\Scripts\python.exe -m unittest tests/test_post_earnings_volatility.py
   ```
2. **Targeted Engine Integration Test**:
   ```powershell
   & .venv\Scripts\python.exe -m unittest tests/test_earnings_gamma_squeeze_engine.py
   ```
3. **Targeted JSON Contract Test**:
   ```powershell
   & .venv\Scripts\python.exe -m unittest tests/test_stock_analysis_data.py
   ```
4. **All 17 Modular Invariant Tests**:
   ```powershell
   & .venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
   ```
5. **Unified Test Suite Runner**:
   ```powershell
   & .venv\Scripts\python.exe scripts/run_all_tests.py
   ```

### Manual & End-to-End Verification
- Execute `stock_analysis_data.py` on a sample ticker (`TEST` or `AAPL`):
  ```powershell
  & .venv\Scripts\python.exe scripts/stock_analysis_data.py --symbol TEST --output reports/TEST_test.json
  ```
- Inspect `reports/TEST_test.json` using Python to verify that:
  1. `calibrate_post_earnings_volatility_surface` exists and has all required fields.
  2. `backtesting_protocol` exists and has `purged_walk_forward_cv`, `almgren_chriss_market_impact`, `borrow_fee_engine`, `deflated_sharpe_ratio`, `verifiable_replication_event_panel`, `strategy_rules`, and `council_interrogation_outcomes`.
  3. `factor_orthogonalization` and `earnings_event_clock` exist.
  4. `evaluation_matrix` exists with all 6 horizons.

