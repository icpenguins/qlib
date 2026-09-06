# Implementation Plan: Russell 1000 Factor + Verdict Cross-Sectional Screen

**Date**: 2026-09-05
**Author**: team-finance (Alpha-Review Framework)
**Status**: Plan — written before any code, per `.team-code/requirements.md` Part 2

---

## Priority-0 Acknowledgement (requirements.md, Priority Requirement -1)

This work is scoped for the two Priority-0 end-users defined in
[requirements.md](requirements.md):

### 1. The Profitable Stock Trader
*Veteran discretionary & quantitative prop trader, multi-million capital at risk.
Objective: consistent alpha, capital preservation, asymmetric risk/reward, no
catastrophic drawdowns.*

His stated critique is that Qlib "operates in a sterile, academic vacuum" — it
assumes stationarity, ignores derivatives, has zero order-flow awareness, and
naively rebalances at the daily close.

**Binding design implications for this deliverable:**
- A cross-sectional Alpha158 screen is *exactly* the "sterile academic vacuum"
  artifact he warns about. The report must therefore be explicitly framed as a
  **screen, not a trade ticket**, on the face of the report itself.
- Every signal present in the single-ticker deep-dive but **absent** here (live
  options chain / dealer GEX, earnings-calendar conditioning / PEAD, AVWAP &
  volume-profile microstructure, borrow/HTB) must be enumerated in a visible
  methodology panel. Silently dropping any of them is disqualifying.
- Capital preservation must be able to **override a high factor score**. A
  ticker in the 99th Alpha158 percentile that is in a BOCD State-2 high-vol
  liquidation regime must render `DO NOT BUY`, not `STRONG BUY`.

### 2. The Institutional Hedge Fund Manager
*CIO / Head of Quantitative Research. Mandate: double-digit net annualized,
Sharpe > 2.0, net-zero market/factor beta, zero catastrophic drawdown tolerance.*

**Binding design implications:**
- Rank integrity is non-negotiable: `rank` and `percentile` must be a true
  cross-sectional ordering over one single as-of date across the full 908-name
  universe, sourced from the already-corrected artifact — never recomputed with
  a different tie method (this is the exact Issue 4 defect fixed today).
- The known **score degeneracy** (232 distinct scores across 908 names, ties up
  to 120 wide — see
  [20260905-finance_team_review_alpha158_degenerate_score.md](20260905-finance_team_review_alpha158_degenerate_score.md))
  materially limits how finely this screen can discriminate. It must be
  disclosed on the report, not buried.
- No field may contradict another field for the same ticker.

### Adversarial-audit bug classes explicitly being designed against
Per [20260905-FIX_adversarial_audit_remediation-implementation_plan.md](20260905-FIX_adversarial_audit_remediation-implementation_plan.md)
and its walkthrough:
- **(A) Key-name mismatch between a data layer and a render layer that silently
  defaults instead of erroring.** Mitigation: this screen renders from a typed
  row object, and the renderer raises on a missing key rather than
  `.get(key, default)`-ing its way to a plausible-looking wrong number.
- **(B) A suppressed/synthetic state not reflected consistently across every
  field that describes it.** Mitigation: when `is_capital_preservation` is
  true, *both* the Best Price and Best Buy Date cells render
  `ENTRIES INHIBITED` — never a live price and date sitting next to a red
  DO-NOT-BUY badge.

---

## Objective

One consolidated HTML report covering all 908 Russell 1000 names with exactly:

1. LightGBM Alpha158 factor score + cross-sectional rank + percentile
2. Executive Investment Verdict — **the repo's existing taxonomy, reused, not reinvented**
3. Estimated best price (`optimal_entry_range`) and best buy date (`optimal_buy_window.start_date`)

---

## Critical Pre-Implementation Finding: the qlib binary store is corrupt

The task brief directed the use of `qlib.init(provider_uri='D:/trading/qlib/qlib_data')`
+ `D.features(...)`. **Verification shows this path returns silently wrong data
and it must not be used.**

Evidence gathered before writing any code:

| Check | Result |
|---|---|
| `features/aapl/close.day.bin` size | 7724 bytes = 1 header float + **1930** values |
| Bin header `start_index` | `0.0` |
| `calendars/day.txt` length | **1500** entries (2020-09-16 → 2026-09-04) |
| `source/AAPL.csv` rows | 1930 (2019-01-02 → 2026-09-04) |
| Tickers where `n_bin_values > len(calendar)` | **857 / 909** |

Qlib maps bin value `i` to `calendar[start_index + i]`. With 1930 values written
against a 1500-entry calendar starting 430 trading days *late*, every series is
**shifted by 430 trading days** and the most recent ~1.7 years is unreachable.

Concretely, for AAPL:
- `D.features(..., end_time='2026-09-04')` returns normalized close **6.6574**
- `normalize/AAPL.csv` row 1499 is **2024-12-16**, normalized close **6.6574**
- The true 2026-09-04 normalized close is **8.5469** (raw **$319.97**)

So `D.features` labels **2024-12-16 prices as 2026-09-04**. Building a
verdict/RSI/SMA/entry-price off that would hand the Priority-0 trader a
21-month-stale price under today's date — precisely bug class (A) at the data
tier, and a capital-preservation hazard.

**Decision: source prices from `D:/trading/qlib/source/*.csv`** — 909 local CSVs,
fully offline (no yfinance, no 404/429 exposure), the upstream ground truth from
which those binaries were dumped, and already in **real dollar terms** (which
the normalized binaries are not — a "best price" of `$6.66` for AAPL would be
meaningless to a trader). All 908 universe tickers have a source CSV; coverage
verified at 908/908.

This finding is reported to the user and disclosed on the report itself. Fixing
the binary store (a re-dump) is **out of scope** for this deliverable.

---

## Architecture

### Layer 0 — Single source of truth for the verdict taxonomy (anti-divergence)

The verdict taxonomy currently lives *inline inside a render function*,
`build_buy_timing_verdict_banner_html` in `scripts/visualize_stock_analysis.py`
(lines ~1264-1321). Copy-pasting that `if/elif` chain into the batch screen would
create two ladders that drift apart — a fresh instance of bug class (A).

**Plan**: extract the classification (not the HTML) into a new module
`scripts/verdict_taxonomy.py`:

```
@dataclass(frozen=True) ExecutiveVerdict:
    badge, pill_class, color_class, icon, description,
    is_capital_preservation, is_spike, is_synthetic, tier
classify_executive_verdict(pred, gamma_squeeze=None) -> ExecutiveVerdict
```

`build_buy_timing_verdict_banner_html` is then refactored to *call* it, so the
single-ticker report and this batch screen are guaranteed to be the same ladder
forever. Behaviour must be byte-identical; the existing tests in
`tests/test_visualize_stock_analysis_refactor.py` (which assert on
`DO NOT BUY / CAPITAL PRESERVATION MODE`, `SAFETY INVARIANT: SYNTHETIC RESEARCH
DATA`, `RESEARCH ONLY`, spike strings) are the regression gate.

`tier` ∈ {`BUY`, `PULLBACK`, `CAUTION`, `NO_BUY`} drives the emerald / blue /
amber / rose badge colour convention.

### Layer 1 — Local price loading
`load_local_ohlcv(symbol)` reads `source/{SYM}.csv` and applies the repo's own
adjustment convention from `scripts/download_us_selected_data.py::normalize_symbol_data`:
`factor = adjclose / close`; `open/high/low *= factor`; `close = adjclose`;
`volume /= factor`. Right-edge anchored, so the last bar is the true traded
dollar close.

### Layer 2 — Per-ticker analytics (reused, not reimplemented)
For each symbol:
1. `detect_market_regime(df, data_dir='D:/trading/qlib', symbol=sym)` —
   `qlib.contrib.regime` BOCD, price-only, ~0.31s.
2. `predict_future_buy_timing(df, forecast_days=63, simulations=1000, regime=regime,
   microstructure=None, derivatives=None, events=None)` — the *actual*
   `RecommendationEngine` / `MonteCarloSimulator` / `SupportResistanceSynthesizer`.
3. `classify_executive_verdict(pred)` from Layer 0.

Passing `None` for microstructure/derivatives/events is the documented
simplification. The extractors already degrade cleanly to their dataclass
defaults (`GEXParams()`, `PEADParams()`), so this is a supported degradation
path, not a hack — but it is disclosed.

Measured cost: ~0.32 s/ticker ⇒ **~5 minutes** for 908 sequential. No
parallelism needed; `MonteCarloSimulator` is seeded (`seed=42`) so the run is
deterministic and reproducible.

### Layer 3 — Join and consistency invariants
Left-join Alpha158 scores (latest date, 2026-09-04) onto verdict rows.
Hard invariants asserted per row before render:
- **I1** `is_capital_preservation` ⟺ Best Price and Best Buy Date both render `ENTRIES INHIBITED`.
- **I2** Best Buy Date ≥ report as-of date (never in the past).
- **I3** `optimal_entry_range[0] <= optimal_entry_range[1]`.
- **I4** Verdict tier `BUY`/`PULLBACK` ⇒ `target_price_3m` direction is not negative beyond a stated tolerance (verdict must not contradict target direction).
- **I5** Alpha158 rank/percentile monotonic: higher score ⇒ better (lower) rank.
Violations are **collected and rendered in a Data Integrity panel**, never
silently dropped.

### Layer 4 — Render
Dark institutional dashboard, consistent with existing reports (Tailwind CDN,
gray-950 ground, emerald/amber/rose badges). Sortable + filterable table,
default sort Alpha158 percentile descending. Saved to
`reports/russell1000_factor_verdict_screen_2026-09-05.html` following the
existing `{name}_{YYYY-MM-DD}.html` convention. A machine-readable
`.json` sidecar is emitted alongside, matching the
`MSFT_analysis_report_2026-09-05.{html,json}` precedent.

---

## Mandatory Methodology Disclosure Panel (on the report)

Must state, in the report body:
1. Price source = local `source/*.csv`, **not** the qlib binary store, **and why** (the 430-day shift).
2. As-of dates: scores 2026-09-04, prices 2026-09-04.
3. **Signals NOT included** vs. the single-ticker deep-dive: live options chain,
   dealer GEX / gamma walls / gamma flip, earnings calendar & PEAD drift,
   de-grossing haircuts, AVWAP & volume-profile microstructure, borrow/HTB fees,
   gamma-squeeze 5-day spike detection.
4. Consequence of (3): the `IMMEDIATE BUY: HIGH-VELOCITY 5-DAY SPIKE` and the
   event-driven verdicts (`EVENT RISK / PRE-EARNINGS DE-GROSSING`,
   `IMMINENT CATALYST / 50% DE-GROSSING`, `PEAD POST-EARNINGS DRIFT
   ACCUMULATION`) are **structurally unreachable** in this screen. Stating which
   verdicts cannot appear is required — otherwise their absence reads as
   evidence they were evaluated and rejected.
5. Alpha158 score degeneracy (232 distinct values / 908 names).
6. No HYG/IEI credit ETFs present locally ⇒ BOCD `credit_mom_pct = 0.0`; the
   macro credit-spread input to regime classification is inactive.

---

## Deliverables checklist

- [ ] `.team-code/20260905-russell1000_factor_verdict_screen-implementation_plan.md` (this file)
- [ ] `scripts/verdict_taxonomy.py` + `scripts/verdict_taxonomy.md` (spec, per Part 2)
- [ ] `scripts/russell1000_factor_verdict_screen.py` + `scripts/russell1000_factor_verdict_screen.md` (spec)
- [ ] Refactor `build_buy_timing_verdict_banner_html` to delegate to Layer 0
- [ ] `reports/russell1000_factor_verdict_screen_2026-09-05.html` (+ `.json`)
- [ ] Verification: existing banner regression tests pass; ~10-ticker spot check
- [ ] `.team-code/20260905-russell1000_factor_verdict_screen-walkthrough.md`

## Risks

| Risk | Mitigation |
|---|---|
| Refactor changes banner HTML | Existing tests are the gate; run before/after |
| Short-history tickers (<50 bars) fail `predict_future_buy_timing` | Catch, record in a skipped list with the reason, surface count in report |
| Screen mistaken for a trade ticket | Methodology panel + per-row framing |
| Binary-store corruption forgotten | Reported to user + documented here and in walkthrough |
