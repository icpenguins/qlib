# Walkthrough: Russell 1000 Factor + Verdict Cross-Sectional Screen (Revision 1)

**Date**: 2026-09-05
**Implements**: [20260905-russell1000_factor_verdict_screen-implementation_plan.md](20260905-russell1000_factor_verdict_screen-implementation_plan.md)
**Status**: Complete. Full 908-name universe screened, 0 skipped.

## Priority-0 Acknowledgement

Scoped for the **Profitable Stock Trader** and **Institutional Hedge Fund
Manager** per [requirements.md](requirements.md). Concretely honoured by:

- The report is labelled a **cross-sectional screen, not a trade ticket**, in the
  header badge, and enumerates every signal it lacks versus the single-ticker
  deep dive.
- **Capital preservation overrides factor rank.** 560 of 908 names carry
  DO-NOT-BUY because BOCD placed them in State 2 (high-volatility liquidation),
  irrespective of their Alpha158 percentile. AAL sits in the 84th percentile and
  is still suppressed.
- **Rank integrity**: rank/percentile are consumed exactly as stored (competition
  ranking) and never recomputed. Verified monotonic in score across all 908 rows.
- **No field contradicts another** — verified by five invariants, four of which
  are now structurally impossible to violate; the fifth is disclosed per-row.

## Deliverable

- **`E:\SRC\GITHUB\my-qlib\reports\russell1000_factor_verdict_screen_2026-09-04.html`** (1.77 MB)
- `E:\SRC\GITHUB\my-qlib\reports\russell1000_factor_verdict_screen_2026-09-04.json` (0.98 MB sidecar)

908 rows: Symbol, Alpha158 Score, Rank, Percentile (with bar), Executive Verdict
(badge-styled emerald/blue/amber/rose), Last Close, Est. Best Price, Est. Best
Buy Date, 3M Expected Return, RSI, BOCD Regime. Sortable on every column,
filterable by ticker / verdict tier / actionable-only. Default sort: Alpha158
percentile descending.

## Final counts

| Metric | Value |
|---|---|
| Universe | 908 |
| **Succeeded** | **908** |
| **Skipped** | **0** |
| Runtime | 278.8 s (~0.31 s/ticker, single-threaded) |
| STRONG BUY (emerald) | 324 |
| BUY ON PULLBACK (blue) | 24 |
| HOLD / CAUTIOUS (amber) | 0 — unreachable, disclosed |
| DO NOT BUY (rose) | 560 |
| Structural invariant violations | **0** |
| Disclosed verdict/median conflicts | 144 |

BOCD regime distribution: State 2 (high-vol liquidation) 557, State 0 (bullish
markup) 348, State 3 (changepoint alert) 3.

---

## Finding 1 (blocking, worked around): the qlib binary store is misaligned

The task directed the use of `qlib.init` + `D.features`. **That path returns
silently wrong data and was not used.**

| Check | Result |
|---|---|
| `features/aapl/close.day.bin` | 7724 bytes = 1 header float + **1930** values |
| bin header `start_index` | `0.0` |
| `calendars/day.txt` | **1500** entries (2020-09-16 → 2026-09-04) |
| `source/AAPL.csv` | 1930 rows (2019-01-02 → 2026-09-04) |
| Tickers where `n_values > len(calendar)` | **857 / 909** |

Qlib maps bin index `i` to `calendar[start_index + i]`. With 1930 values against
a 1500-entry calendar that starts 430 trading days late, every series is shifted
by 430 trading days and the most recent ~1.7 years is unreachable.

Reproduced concretely:
- `D.features(['AAPL'], ['$close'], end_time='2026-09-04')` → `6.6574`
- `normalize/AAPL.csv` row 1499 = **2024-12-16**, close `6.6574`
- True 2026-09-04 normalized close = `8.5469`, raw = **$319.97**

So `D.features` labels 2024-12-16 prices as 2026-09-04. Building a verdict, RSI,
or entry price off that would hand the Priority-0 trader a 21-month-stale price
under today's date.

**Resolution**: prices are read from `D:/trading/qlib/source/*.csv` — fully
offline (no yfinance, no 404/429 exposure), 908/908 coverage, and in real dollar
terms, which the normalized binaries are not (a "best price" of `$6.66` for AAPL
would be unusable). The repo's own adjustment convention from
`download_us_selected_data.py::normalize_symbol_data` is applied.

**Not fixed here.** Re-dumping the binary store is a separate change with its own
blast radius (it is the store `train_alpha158_lightgbm.py` and every qlib
workflow reads). It is disclosed in a red advisory panel on the report and is the
recommended next action.

## Finding 2 (fixed): three constructive verdicts rendered as rose DO-NOT-BUY

Running the verdict ladder across 908 names exposed a pre-existing defect **in
the single-ticker report**, not just the screen.

`build_buy_timing_verdict_banner_html` matched the exact token `"ACCUMULATE"`.
Three `RecommendationEngine` branches emit `"ACCUMULATION"`, which does not
contain `"ACCUMULATE"`:

- `BULLISH MOMENTUM / DIP ACCUMULATION` (BOCD state 0) — **74 of 908 names**
- `RANGE ACCUMULATION / BUY SUPPORT` (BOCD state 1)
- `PEAD POST-EARNINGS DRIFT ACCUMULATION`

All three carry `is_entry_allowed=True` and an **ACTIVE** buy window, yet fell
through to the rose `DO NOT BUY / CAPITAL PRESERVATION MODE` fallback — a red
stand-aside badge rendered directly above a live entry corridor and buy window.
Same bug class as the 2026-09-05 audit: a render layer contradicting the data
layer it renders.

Observed live on **ACN** in the first smoke run: verdict `DO NOT BUY`, best price
`$179.25 – $183.92`, best buy date `2026-09-10`.

**Fixed** in `scripts/verdict_taxonomy.py`:
1. `"ACCUMULATE"` → the stem `"ACCUMULAT"`.
2. `IMMINENT CATALYST / 50% DE-GROSSING` routed to amber — a haircut, not a stop.
3. New closing invariant: **a rose DO-NOT-BUY badge is never shown for a posture
   whose buy window is ACTIVE.** An entry-allowed recommendation with no better
   label now renders amber; the rose fallback is reserved for genuinely
   inhibited postures. This makes the taxonomy closed under future additions to
   `RecommendationEngine`.

## Finding 3 (fixed): entry corridors could invert

`RecommendationEngine.evaluate`'s BOCD State-0 branches build the corridor from
two independently-derived bounds:

```
entry_low  = max(key_support, current_price * 0.96)
entry_high = current_price * 0.985
```

These invert whenever `key_support > 0.985 * spot`. Observed on **AES**
(`$14.69 – $14.57`) and **SLAB** (`$217.38 – $217.27`) — nonsensical corridors
shipped to a trader.

**Fixed** in `scripts/predictive_engine.py` by ordering the pair at the return
site. This is a no-op for every well-formed corridor and provably cannot change
a valid result; only the two broken rows changed. Re-run confirms 0 inverted
corridors.

## Finding 4 (disclosed, not fixed): verdicts ignore their own Monte Carlo median

**144 of 348 actionable names (41%)** carry a BUY/PULLBACK verdict whose 3-month
Monte Carlo median target is *below* spot. Worst: CRM −16.6%, CNH −14.1%,
VEEV −11.3%. Median across the conflicted set: −2.75%.

Root cause: `RecommendationEngine.evaluate` decides the verdict from regime
state, RSI and %B only — it never reads `p50_median`, which is computed in the
same function and produces the target rendered in the adjacent column. The
simulator's mean-reversion term (`DRIFT_MEAN_REVERSION_COEFF` pulling toward
SMA50) drags the median down for names extended above their 50-day average,
exactly the names the technical rules call bullish.

**Deliberately not "fixed".** Reconciling them means either suppressing a number
or inventing a new gating rule outside the canonical taxonomy — the task
explicitly required reusing the existing taxonomy, and silently adjusting either
value would be the very failure mode being guarded against. Instead:
- each affected row carries a visible **`⚠ MEDIAN PATH CONTRADICTS VERDICT`**
  badge under its verdict (a badge, not a tooltip),
- a dedicated stat tile shows the count and percentage,
- an amber advisory panel explains the mechanism and instructs the reader to
  treat a flagged row as unresolved rather than as a buy.

This is a genuine model-design question for the Council, not a rendering bug.

## Finding 5 (noted only): `compute_rsi` is not Wilder's RSI

`scripts/indicators.py::compute_rsi` docstring says "standard Wilder rolling
average" but the implementation uses a simple `.rolling(period).mean()` — that is
Cutler's RSI. Values differ materially (CRM: 88.2 vs 76.6 Wilder). It is applied
consistently across the single-ticker report and this screen, so nothing is
internally inconsistent; only the docstring is wrong. Not changed — altering RSI
would shift verdicts repo-wide.

---

## Files changed

**New**
- `scripts/verdict_taxonomy.py` — canonical verdict classifier
- `scripts/verdict_taxonomy.md` — spec (Part 2)
- `scripts/russell1000_factor_verdict_screen.py` — batch screen + renderer
- `scripts/russell1000_factor_verdict_screen.md` — spec (Part 2)
- `.team-code/20260905-russell1000_factor_verdict_screen-implementation_plan.md`
- `.team-code/20260905-russell1000_factor_verdict_screen-walkthrough.md` — this file
- `reports/russell1000_factor_verdict_screen_2026-09-04.{html,json}`

**Modified**
- `scripts/visualize_stock_analysis.py` — `build_buy_timing_verdict_banner_html`
  now delegates classification to `verdict_taxonomy`; the inline ladder is gone.
  Import added.
- `scripts/predictive_engine.py` — entry-corridor ordering invariant (Finding 3).

## Anti-regression measures for the audited bug classes

**Class (A) — key-name mismatch defaulting silently.** The renderer consumes a
typed `ScreenRow` dataclass, so a wrong field name is an `AttributeError` at
build time, not a plausible-looking default. `load_local_ohlcv` and
`load_latest_alpha_scores` raise on missing columns rather than substituting.
The verdict ladder now has exactly one definition site shared by both consumers.

**Class (B) — suppressed state not reflected everywhere.** When
`is_capital_preservation` is true, `best_price_display`, `best_buy_date_display`,
`entry_low`, `entry_high`, `best_buy_date` and `buy_window_end` are **all**
suppressed — in the JSON sidecar as well as the HTML, so a downstream consumer
cannot read a corridor the report refuses to show. Verified: 560 suppressed rows
× 2 cells = 1120 `ENTRIES INHIBITED` renders, 0 mismatches.

## Verification performed

**Invariants across all 908 rows — 0 violations:**

| ID | Invariant | Violations |
|---|---|---|
| I1 | capital-preservation ⟺ price cell suppressed | 0 |
| I1 | capital-preservation ⟺ date cell suppressed | 0 |
| I1b | verdict tier `NO_BUY` ⟺ capital preservation | 0 |
| I2 | best buy date never before as-of date | 0 |
| I2b | window start ≤ window end | 0 |
| I3 | `entry_low ≤ entry_high` | 0 |
| I5 | rank monotonic in score | 0 |
| — | `stop_loss < entry_low` | 0 |
| — | price > 0; percentile ∈ [0,100]; rank ∈ [1,908] | 0 |
| I4 | BUY verdict vs. median direction | **144 — disclosed per-row** |

**10-ticker sample cross-check** (AAPL, MSFT, NVDA, JPM, XOM, WMT, AES, SLAB,
CRM, ABT):
- `current_price` matched the source CSV's last `adjclose` to the cent — 10/10
  (AAPL $319.97, MSFT $499.70, NVDA $230.36).
- `alpha_score` / `alpha_rank` matched the parquet exactly — 10/10.
- `rsi` matched an independent call to the repo's own `compute_rsi` — 10/10.
- **Cross-pipeline**: for each ticker the badge rendered by the single-ticker
  `build_buy_timing_verdict_banner_html` was confirmed to contain the same
  verdict as the screen's table cell — **10/10**. This is the end-to-end proof
  that the shared-classifier refactor keeps the two reports consistent.

**HTML structural checks**: 908 `<tr>` data rows, 2817 `<div>` = 2817 `</div>`,
all seven disclosure panels present, 0 skipped-ticker rows.

**Test suite**: `test_visualize_stock_analysis_refactor` (the behaviour-preservation
gate for the banner refactor), `test_predictive_engine`, `test_domain_models`,
`test_indicators`, `test_stock_analysis_data`, `test_stock_analysis_engine`,
`test_bocd_regime`, `test_earnings_gamma_squeeze_engine`,
`test_lightgbm_alpha158_us` — **75 passed, 0 failed.** The banner suite was run
green before the refactor and green after.

## Skipped tickers

**None.** All 908 names in `data/instruments/russell1000.txt` had a local source
CSV, ≥ 50 bars, a resolvable BOCD regime, and an Alpha158 score on 2026-09-04.

## Items not completed / out of scope

- **The qlib binary-store misalignment is not repaired** (Finding 1) — worked
  around and disclosed. This is the highest-value follow-up: any other consumer
  of `D.features` in this repo is currently reading 430-day-shifted data.
- **The Alpha158 score degeneracy** (232 distinct values / 908 names) is
  unchanged; disclosed on the report, tracked in
  [20260905-finance_team_review_alpha158_degenerate_score.md](20260905-finance_team_review_alpha158_degenerate_score.md).
- **The verdict/median-path conflict** (Finding 4) is disclosed, not resolved.
- **`compute_rsi` docstring/implementation mismatch** (Finding 5) not changed.
- **No live end-user review has occurred** — this walkthrough is the artifact to
  present to the Trader and the Hedge Fund Manager per Team-Code Phase 4.

## Suggested next steps

1. Re-dump the qlib binary store so `calendars/day.txt` covers the full 1930-bar
   history, then re-verify `D.features` against `source/*.csv` for a sample.
2. Council decision on Finding 4: should `RecommendationEngine` consult
   `p50_median` before issuing a BUY, or is the mean-reversion coefficient
   mis-calibrated for trending names?
3. Regenerate this screen once the Alpha158 score degeneracy is addressed — the
   percentile column is currently far coarser than its two decimals suggest.
