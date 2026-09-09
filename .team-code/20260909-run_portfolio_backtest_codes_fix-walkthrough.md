# Walkthrough: `run_portfolio_backtest()` Missing `codes` -> `instrument not exists: .../all.txt` (2026-09-09)

## Priority-0 acknowledgement
Reviewed `.team-code/requirements.md` before starting. This fix serves the
**Profitable Stock Trader** and **Institutional Hedge Fund Manager** end-users:
a backtest that silently trades/reports against the wrong universe (or crashes
opaquely deep inside qlib after a full multi-minute training run has already
completed) is exactly the "sterile academic vacuum" / non-production-grade
failure mode both end-users called out. The fix makes the resolved universe an
explicit, required, loudly-enforced input to the backtest step rather than an
implicit qlib default.

## Constraints applied
`.team-code/requirements.md` (Part 1 end-users above; Part 2 documentation
requirements followed via this walkthrough plus inline docstring updates in
the changed functions -- a standalone architecture/plan doc was judged
unnecessary for this small, single-root-cause bugfix with no new subsystem,
per the task's own guidance to use judgment here).

## Trigger
User-reported error:
```
ValueError: instrument not exists: D:\trading\qlib2\instruments\all.txt
```

## Phase notes
- **Architect**: Restated the problem as "an already-resolved instrument universe never
  reaches the exchange constructed for backtesting" with the non-goal "do not touch
  parallelism/`kernels` speculatively." Traced the exact qlib call chain
  (`run_portfolio_backtest` -> `backtest_daily` -> `get_exchange` -> `Exchange.get_quote_from_qlib`
  -> `D.features`) and identified all three call sites needing the same fix.
- **Program Manager**: Benchmarked against this file's own established pattern
  (`audit_universe_bias`/`assert_sufficient_training_budget`: fail loudly with an
  actionable message rather than silently do the wrong thing) rather than inventing a
  new convention; recommended a hand-written `ValueError` guard over a bare required
  positional argument, for a more actionable message. Flagged the `--kernels` CLI idea
  as premature without reproducible evidence -- kept as "evaluate, don't implement
  speculatively," dropped from this change's scope after the real run showed no
  parallelism-related failure.
- **Principal Dev**: Implemented the `codes` parameter and guard in
  `run_portfolio_backtest()`, threaded it into `exchange_kwargs`, and updated the three
  call sites to pass the already-resolved `instruments` (never re-resolved).
- **Senior Dev**: Wrote full docstrings for the new/changed parameters, updated the
  module-level docstring's summary of `run_portfolio_backtest`, and wrote the 4
  regression tests in `tests/test_ensemble_lib.py` against real qlib behavior (no mocks).
- **QA Tester**: Behavior matrix -- "omit codes" (raises loudly), "explicit codes='all'"
  (reproduces the exact reported bug, pinning the diagnosis), "resolved ticker list"
  and "resolved market-name string" (both complete a real backtest without `all.txt`) --
  all 4 covered by new tests, all passing against real qlib. Confirmed no O(n^2) or
  scaling concern introduced (this is a parameter plumbing fix, no new loops).
- **CI/CD**: See CI/CD check section below.

## Root cause (verified, not re-derived on faith)
`run_portfolio_backtest()` (`scripts/ensemble_lib.py`) calls
`backtest_daily(..., exchange_kwargs={...})` with a dict that never sets
`codes`. `qlib.backtest.get_exchange()`'s own signature defaults
`codes: Union[list, str] = "all"` when the caller omits it. Confirmed directly
in `qlib/backtest/__init__.py:33-46`. `Exchange.get_quote_from_qlib()`
(`qlib/backtest/exchange.py:201-206`) then passes that literal string straight
into `D.features(self.codes, ...)`:
```python
def get_quote_from_qlib(self) -> None:
    if len(self.codes) == 0:
        self.codes = D.instruments()
    self.quote_df = D.features(self.codes, self.all_fields, self.start_time, self.end_time, freq=self.freq, disk_cache=True)
```
`"all"` has `len() == 3` (truthy), so the `D.instruments()` fallback is never
reached -- `"all"` is forwarded verbatim to `D.features()`, which resolves a
bare string as a MARKET NAME, i.e. `<data_dir>/instruments/all.txt`. This
session's real `D:/trading/qlib2` directory no longer has that file --
confirmed directly:
```
$ ls D:/trading/qlib2/instruments/
benchmark.txt
sp500.txt
```
-- it was renamed/replaced with `sp500.txt` + a new `benchmark.txt` earlier
this session, which reproduces the exact reported `ValueError` with certainty,
not conjecture.

The already-resolved instrument universe (`parse_instruments()`, computed once
per run specifically to avoid this exact class of market-name mismatch) was
simply never threaded down to the exchange that needed it. Three call sites
had this gap: `train_ensemble.py:300`, and `diagnose_signal.py`'s
`compute_turnover_and_cost_stress()` (called from `run_diagnostics()`), which
itself calls `run_portfolio_backtest()` twice.

## Fix
1. **`scripts/ensemble_lib.py` -- `run_portfolio_backtest()`**: added a
   `codes: Optional[Union[str, List[str]]] = None` parameter, threaded into
   `exchange_kwargs["codes"]`. Chose a **required-in-effect, loud-failure**
   design over silently keeping qlib's `"all"` default: if `codes` is `None`,
   the function raises `ValueError` immediately with an actionable message
   ("pass the already-resolved `instruments` variable... do not re-resolve it
   a second time") rather than letting the call fall through to
   `get_exchange()`'s own default. This was chosen over making `codes` a bare
   required positional argument because a hand-written `ValueError` message
   naming the exact fix is far more actionable than Python's generic "missing
   1 required positional argument" `TypeError` would have been -- consistent
   with this file's existing fail-loudly pattern (`audit_universe_bias`,
   `assert_sufficient_training_budget`). Either a real crash (no `all.txt`) or
   a silently-wrong backtest (an unrelated `all.txt` existing) is unacceptable
   for the end-users above; this makes both impossible by construction.
2. **`scripts/diagnose_signal.py` -- `compute_turnover_and_cost_stress()`**:
   added a required `codes: Union[str, List[str]]` parameter, forwarded to
   both of its internal `run_portfolio_backtest()` calls.
3. **Three call sites updated to thread the already-resolved `instruments`
   variable through (never re-resolved a second time)**:
   - `scripts/train_ensemble.py:300-302` -- `run_portfolio_backtest(pred_series, benchmark=args.benchmark, codes=instruments, deal_price=variant["deal_price"])`
   - `scripts/diagnose_signal.py` (`run_diagnostics()`, item #7 block) --
     `compute_turnover_and_cost_stress(pred, benchmark=args.benchmark, codes=instruments, deal_price=deal_price)`,
     where `instruments` is the same `ensemble_lib.parse_instruments(args.market, data_dir)`
     result already computed earlier in `run_diagnostics()` for item #3.

## Item 3: is the `disk_cache`/`inst_processors` `TypeError` the same root cause?
**No -- verified as a separate, independent, self-recovering qlib-internal
quirk, not a manifestation of the `codes` bug.** Directly reproduced against a
real local (non-server) qlib provider:
```
>>> qd.DatasetD.dataset(['A'], ['$close'], dates[0], dates[-1], 'day', True, inst_processors=[])
TypeError: LocalDatasetProvider.dataset() got multiple values for argument 'inst_processors'
```
`LocalDatasetProvider.dataset()` (`qlib/data/data.py:902-910`) has no
`disk_cache` parameter at all, so `BaseProvider.features()`'s first, optimistic
call (`qlib/data/data.py:1183-1190`, which always passes `disk_cache`
positionally) raises this `TypeError` on **every single** `D.features(...)`
call against a local provider -- entirely independent of what `codes` is set
to -- and is silently caught and retried without `disk_cache` by
`BaseProvider.features()`'s own `except TypeError:` clause, which succeeds.
This happens on every real run today, fix or no fix, and is not blocking (it
self-recovers by design). It is a real, pre-existing, low-priority
qlib-internal quirk (positional-arg collision in `BaseProvider.features()`
when `DatasetD` resolves to a `LocalDatasetProvider`), tracked here as
**separate, still-open, non-blocking technical debt** -- explicitly NOT
conflated with, caused by, or fixed by the `codes` change in this walkthrough.

## Item 4: `--kernels` / parallelism reduction
**Not implemented.** The `codes` fix directly and fully explains the only
crash this session could actually reproduce (`ValueError: instrument not
exists: .../all.txt`). The real verification run below (default
`NUM_USABLE_CPU`-derived kernel count, no `--kernels` flag) completed the full
dataset-preparation, training, and backtest pipeline without any
`send_bytes()`/`MemoryError` symptom recurring. Per the task's own guidance
("don't implement it speculatively if the `codes` fix alone resolves what you
can actually reproduce"), no `--kernels` CLI flag was added. If a genuine,
reproducible parallelism-related crash resurfaces independently of this fix,
it should be scoped and fixed in its own follow-up, not bundled here.

## Regression tests added
`tests/test_ensemble_lib.py` -- new `TestRunPortfolioBacktestCodes`, following
the existing `TestFindAvailableBenchmark` pattern (a real qlib data directory
built via `download_us_selected_data.dump_to_qlib_format`, `qlib.init()`
against it -- no mocks):
- Builds a 6-symbol, 90-business-day synthetic qlib data directory, then
  **deliberately renames `instruments/all.txt` to `instruments/sp500.txt`** --
  reproducing the real `D:/trading/qlib2` scenario verbatim inside the test.
- `test_missing_codes_raises_clear_error_instead_of_silent_qlib_default`:
  omitting `codes` raises `ValueError` immediately (no qlib call is even
  attempted).
- `test_explicit_codes_all_reproduces_the_exact_reported_bug`: an explicit
  `codes="all"` call (simulating pre-fix/qlib-default behavior) against this
  directory hits `status="failed"` with the real, verbatim
  `ValueError: instrument not exists: .../all.txt` from the actual qlib
  exchange code path -- pinning the diagnosis, not just asserting the fix.
- `test_resolved_codes_list_completes_the_backtest_without_all_txt`: passing
  the resolved ticker list completes a real `backtest_daily()` run
  (`status="ok"`, non-NaN metrics) with no `all.txt` present at all.
- `test_resolved_codes_as_market_name_string_also_completes`: `codes="sp500"`
  (the market-name-string shape `parse_instruments()` also returns) likewise
  completes, resolving against the renamed `sp500.txt`.

## Verification

### Regression tests (real qlib, no mocks)
```
$ python -m pytest tests/test_ensemble_lib.py -q
........
8 passed in 16.18s
```
(4 pre-existing `TestFindAvailableBenchmark` tests + 4 new
`TestRunPortfolioBacktestCodes` tests, all passing.)

### Full suite
```
$ python -m pytest tests/ -q --ignore=tests/rl --ignore=tests/test_pit.py
...
15 failed, 260 passed, 1 skipped in 554.20s (0:09:14)
```
All 15 failures are the documented pre-existing, unrelated failures (MLflow
filesystem-backend maintenance-mode exceptions, CN-data-dependent tests, a
pandas-version API deprecation in `test_processor.py`/`test_dataloader.py`,
etc.) -- none touch `ensemble_lib.py`, `train_ensemble.py`, or
`diagnose_signal.py`. No regressions introduced.

### Real reproduction run against `D:/trading/qlib2`
Exact command (matching this session's current universe filenames -- `sp500.txt`
+ `benchmark.txt`, no `all.txt`), run to completion, not truncated:
```
$ python scripts/train_ensemble.py --market sp500 --data_dir "D:/trading/qlib2" \
    --num_boost_round 300 --output_dir models/gbdt_ensemble_verify_codes_fix
```
Exit code: `0`. `grep -c "instrument not exists\|Traceback\|status.*failed"` over
the full run log: `0` matches. Universe resolved to 582 real S&P 500
instruments (`universe_audit.n_instruments: 582`, `flagged: false`); training,
IC evaluation, and backtest completed for all four rows (LightGBM, XGBoost,
CatBoost, Ensemble_Blended) with `"Backtest Status": "ok"` / `"Backtest Error":
null` for every one of them (from the saved `ensemble_metadata.json`):
```
======================================================================================================================
           QLIB MULTI-MODEL GBDT ENSEMBLE PERFORMANCE EVALUATION           
======================================================================================================================
Market Universe: sp500 | Benchmark: SPY | Seed: 42
Train: 2020-01-01 -> 2023-12-31 | Test: 2025-01-01 -> 2026-09-04
Label variant: default (deal_price='close')
----------------------------------------------------------------------------------------------------------------------
Model              |      IC |  RankIC | RankICIR | AnnRet(g) |   IR(g) | AnnRet(n) |   IR(n) | RelDD(n) | AbsDD(n)
----------------------------------------------------------------------------------------------------------------------
   LightGBM        | -0.0014 | -0.0024 |  -0.0265 |   -0.0898 | -0.8023 |   -0.0935 | -0.8357 |  -0.2504 |  -0.1739
   XGBoost         | -0.0040 |  0.0040 |   0.0370 |    0.0208 |  0.2009 |    0.0157 |  0.1513 |  -0.1100 |  -0.2101
   CatBoost        |  0.0065 | -0.0014 |  -0.0141 |   -0.0762 | -0.8242 |   -0.0788 | -0.8515 |  -0.2035 |  -0.1923
>> Ensemble_Blended |  0.0024 |  0.0030 |   0.0308 |   -0.0195 | -0.1848 |   -0.0244 | -0.2304 |  -0.1157 |  -0.2119
======================================================================================================================
```
`ensemble_metadata.json` per-model confirmation (excerpt):
```json
{"Model": "LightGBM",  "Backtest Status": "ok", "Backtest Error": null},
{"Model": "XGBoost",   "Backtest Status": "ok", "Backtest Error": null},
{"Model": "CatBoost",  "Backtest Status": "ok", "Backtest Error": null},
{"Model": "Ensemble_Blended", "Backtest Status": "ok", "Backtest Error": null}
```
The `disk_cache`/`inst_processors` `TypeError` (Item 3) was independently
reproduced (see above) as occurring on every `D.features()` call regardless of
this fix -- consistent with its self-recovering, non-blocking nature, this
full real run completed cleanly with exit code 0 despite it.

This is a real out-of-sample result on genuinely noisy signal (small,
sometimes-negative IC/IR across legs) -- it is reported here strictly to
confirm the *backtest mechanics* now complete without crashing, not as a claim
that this particular model/window combination is trading-ready. That
signal-quality question is squarely what `--diagnose` (the pre-training
diagnostic gate) exists to interrogate, and is out of scope for this bugfix.

## CI/CD check
No dedicated lint/type-check CI gate targets `scripts/*.py` in this repo (no
`pyrightconfig.json`; `pyright` is not installed in this environment, though
prior commits reference manual pyright passes). Ran `black --check` and
`pyflakes` against all touched files as a sanity check:
- `black --check`: all three touched `scripts/*.py` files "would reformat" --
  confirmed via `git stash` to be the exact same pre-existing condition before
  this change (not a regression; this repo is not black-formatted at
  baseline).
- `pyflakes`: only pre-existing, unrelated warnings (`numpy as np` imported
  but unused in `ensemble_lib.py`/`diagnose_signal.py`, both present before
  this change) -- no new warnings from this diff.
- `python -c "import ast; ast.parse(...)"` on all three edited scripts:
  syntax OK.
- No `package.json`/`pyproject.toml` dependency changes needed -- no new
  imports were added.

## Files changed
- `scripts/ensemble_lib.py` -- `run_portfolio_backtest()`: added required
  `codes` parameter + loud-failure guard; threaded into `exchange_kwargs`.
- `scripts/diagnose_signal.py` -- `compute_turnover_and_cost_stress()`: added
  required `codes` parameter, forwarded to both internal
  `run_portfolio_backtest()` calls; its caller in `run_diagnostics()` now
  passes the already-resolved `instruments`.
- `scripts/train_ensemble.py` -- its `run_portfolio_backtest()` call site now
  passes `codes=instruments`.
- `tests/test_ensemble_lib.py` -- new `TestRunPortfolioBacktestCodes` (4 tests).

## Open items / technical debt
- The `disk_cache`/`inst_processors` `TypeError` (Item 3 above) remains a
  real, low-priority, self-recovering qlib-internal quirk, independent of
  this fix. Not blocking; not addressed here (out of scope for this bugfix).
- `--kernels` parallelism reduction (Item 4) was evaluated and deliberately
  not implemented -- no reproducible evidence it is still needed after this
  fix (see Item 4 above).

## End-user feedback
Pending.

## Changes to the original request
None.
