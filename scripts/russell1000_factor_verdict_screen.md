# Specification: `scripts/russell1000_factor_verdict_screen.py`

**Created**: 2026-09-05
**Purpose**: Build one consolidated HTML dashboard covering every Russell 1000
name with its Alpha158 factor score, Executive Investment Verdict, and estimated
best entry price / best buy date.

## Scope statement

This is a **fast cross-sectional screen, not a single-ticker deep dive.** It runs
the repo's real regime and forecasting engines but with the options, events, and
microstructure inputs absent. The report renders that fact prominently; see
"Disclosure contract" below.

## Public API

### `load_universe(universe_file: Path) -> List[str]`
Reads `data/instruments/russell1000.txt` (tab-separated `SYMBOL start end`).
De-duplicates, preserves order. Returns 908 symbols.

### `load_local_ohlcv(symbol: str, market_data_root: Path) -> pd.DataFrame`
Loads `{market_data_root}/source/{SYMBOL}.csv` and applies this repo's own
adjustment convention from `download_us_selected_data.py::normalize_symbol_data`:

```
factor      = adjclose / close
open/high/low *= factor
close        = adjclose
volume      /= factor
```

Returns `date, open, high, low, close, volume`. Because yfinance's `adjclose` is
back-adjusted, the series is anchored at the right edge: the final bar equals the
true traded dollar close. This is what makes "Estimated Best Price" a real,
quotable figure.

Raises `FileNotFoundError` if no CSV exists, `ValueError` on missing columns or
an empty file. Never silently substitutes a default.

**Why not `qlib.data.D.features`?** See "Price source" below. It returns
silently wrong data in this environment.

### `load_latest_alpha_scores(scores_file: Path) -> Tuple[pd.DataFrame, str]`
Reads `output/scores/alpha158_russell1000_latest.parquet`, returns the most
recent date's cross-section plus that date as `YYYY-MM-DD`.

Rank and percentile are consumed **exactly as stored** and are never recomputed.
The stored artifact was corrected on 2026-09-05 to use competition ranking
(`method="min"`); recomputing with a different tie method is precisely how the
rank/percentile mismatch (audit Issue 4) arose. Raises `ValueError` if any of
`date, symbol, score, rank, percentile` is missing.

### `analyse_symbol(symbol, market_data_root, alpha_row, as_of_date) -> ScreenRow`
Per-ticker pipeline:
1. `load_local_ohlcv` (raises if `< 50` bars, `predict_future_buy_timing`'s floor)
2. `stock_analysis_engine.detect_market_regime(df, data_dir, symbol)` -- real BOCD
3. `predictive_engine.predict_future_buy_timing(df, forecast_days=63,
   simulations=1000, regime=regime, microstructure=None, derivatives=None,
   events=None)` -- the real `MonteCarloSimulator`, `SupportResistanceSynthesizer`
   and `RecommendationEngine`
4. `verdict_taxonomy.classify_executive_verdict(pred, gamma_squeeze=None)`
5. Applies the suppression rule and evaluates invariants I1-I4

Passing `None` for microstructure/derivatives/events is a supported degradation
path: `GEXParameterExtractor` and `EventParameterExtractor` return their
dataclass defaults (`GEXParams()`, `PEADParams()`). It is not a hack -- but it is
disclosed on the report.

### `run_screen(...) -> Dict[str, Any]`
Orchestrates the full universe. Returns a render-ready payload with `rows`,
`skipped`, `cross_flags`, as-of dates, counts and runtime. Every per-ticker
failure is captured into `skipped` with its exception type and message; nothing
is swallowed.

### `build_screen_html(payload) -> str`
Renders the dashboard. Default sort: Alpha158 percentile descending.

## `ScreenRow` dataclass

The renderer reads **attributes off this dataclass**, not `.get()` calls on a
dict. A key-name mismatch is therefore an `AttributeError` at build time rather
than a silent plausible-looking default -- directly targeting audit bug class (A).

Fields: `symbol`, `alpha_score`, `alpha_rank`, `alpha_percentile`,
`verdict_badge`, `verdict_short`, `verdict_tier`, `verdict_branch`,
`recommendation`, `is_capital_preservation`, `current_price`, `entry_low`,
`entry_high`, `best_price_display`, `best_buy_date`, `best_buy_date_display`,
`buy_window_end`, `buy_window_status`, `target_price_3m`, `expected_return_pct`,
`stop_loss`, `risk_reward_ratio`, `key_support`, `rsi`, `bocd_state`,
`bocd_regime_name`, `changepoint_hazard_pct`, `integrity_flags`.

## Suppression rule (audit bug class B)

When `verdict.is_capital_preservation` is true:

- `best_price_display` = `"ENTRIES INHIBITED"`, `entry_low` / `entry_high` = `None`
- `best_buy_date_display` = `"ENTRIES INHIBITED"`, `best_buy_date` = `None`
- `buy_window_end` = `None`

A suppressed state suppresses **every** field that describes it. A live price or
date beside a red DO-NOT-BUY badge is the exact contradiction remediated on
2026-09-05 (audit Issue 3). The suppression is applied to the JSON sidecar too,
not just the HTML, so a downstream consumer cannot read a corridor the report
refuses to display.

## Consistency invariants

| ID | Invariant | On violation |
|---|---|---|
| I1 | `is_capital_preservation` ⟺ both price and date cells render `ENTRIES INHIBITED` | enforced by construction |
| I1b | verdict tier `NO_BUY` ⟺ `is_capital_preservation` | row flag `VERDICT_TIER_VS_POSTURE_MISMATCH` |
| I2 | an actionable best-buy date is never before the report as-of date | row flag `BUY_DATE_IN_PAST` |
| I3 | `entry_low <= entry_high` | row flag `ENTRY_CORRIDOR_INVERTED` |
| I4 | a `BUY`/`PULLBACK` verdict does not sit beside a 3-month median target below spot | row flag `BUY_VERDICT_WITH_NEGATIVE_TARGET` + a visible in-row badge |
| I5 | Alpha158 rank is monotonic in score across the cross-section | payload `cross_flags` |

Violations are **collected and rendered** in a Data Integrity panel and, for I4,
as a per-row badge. They are never dropped, and no row is hidden because it
failed a check.

## Price source

`D:/trading/qlib/source/*.csv`, **not** the qlib binary store.

The binary store at `D:/trading/qlib/qlib_data` is misaligned: each feature
`.bin` holds 1,930 values against a 1,500-entry `calendars/day.txt`
(**857 of 909** tickers affected). Since qlib maps bin index `i` to
`calendar[start_index + i]`, every series is shifted by **430 trading days** and
the most recent ~1.7 years is unreachable. Verified concretely:
`D.features(AAPL, end_time='2026-09-04')` returns normalized close `6.6574`,
which is AAPL's **2024-12-16** bar; the true 2026-09-04 close is `$319.97`.

The source CSVs are offline (no yfinance, no 404/429 exposure), cover 908/908
universe names, and are denominated in real dollars -- which the normalized
binaries are not.

## Disclosure contract

Two module-level constants are rendered verbatim onto the report and MUST be
kept truthful:

- `SCREEN_LIMITATIONS` -- signals present in the single-ticker report but absent
  here (options chain, dealer GEX, earnings calendar & PEAD, AVWAP/volume
  profile, borrow fees, gamma-squeeze engine, macro credit spreads).
- `UNREACHABLE_VERDICTS` -- verdicts that cannot appear, and why. Their absence
  must not read as "evaluated and rejected".

**If a signal is later added to this screen, delete its line from
`SCREEN_LIMITATIONS`. Never leave a stale claim.**

## Outputs

- `reports/russell1000_factor_verdict_screen_{alpha_as_of}.html`
- `reports/russell1000_factor_verdict_screen_{alpha_as_of}.json` (machine-readable sidecar)

Naming follows the existing `{name}_{YYYY-MM-DD}.{html,json}` convention set by
`MSFT_analysis_report_2026-09-05.{html,json}`.

## CLI

```
python scripts/russell1000_factor_verdict_screen.py
python scripts/russell1000_factor_verdict_screen.py --limit 25        # smoke test
python scripts/russell1000_factor_verdict_screen.py --market-data-root D:/trading/qlib
```

## Performance

908 tickers in **281.5 s** (~0.31 s/ticker), single-threaded. Dominated by BOCD.
Deterministic: `MonteCarloSimulator` is seeded (`seed=42`), so re-running the
same inputs reproduces the report byte-for-byte apart from the generation
timestamp.
