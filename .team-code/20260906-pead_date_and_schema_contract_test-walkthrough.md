# Walkthrough: PEAD Announcement-Date Fix + Schema-Contract Regression Test (2026-09-06)

## Trigger
User ran `scripts/visualize_stock_analysis.py --symbol INTC` and found the PEAD
card showing "Reported on N/A." despite real Gap/Drift numbers being displayed.
Asked to fix it and build a test to catch this bug class going forward.

## Root cause (same family as 2026-09-05's audit)
`PEADEngine.evaluate_recent_pead` (qlib/contrib/events/pead.py) returns the
report date as `latest_report_date`. `build_events_card_html`
(scripts/visualize_stock_analysis.py:893) read `recent_announcement_date` --
a key that never existed anywhere. Confirmed via the real INTC JSON:
`latest_report_date: "2026-08-18"` was present and correct the whole time.

**Fix**: `recent_earn_date = pead_info.get("latest_report_date", "N/A")`.
Regenerated the real INTC report; confirmed "Reported on 2026-08-18." renders.

## Schema-contract regression test
Built `tests/test_visualize_key_contracts.py`: for each of 7 modular HTML-card
builders in `visualize_stock_analysis.py`, statically extracts (via `ast`)
every literal-keyed `.get("key", default)` call in the function's source, and
asserts each key exists somewhere in the REAL payload produced by that
section's actual producer function (called directly with representative
inputs, not mocked) -- flattened recursively across the whole payload tree.
An explicit `ALLOWED_EXTRA_KEYS` registry documents the few legitimate
exceptions (intentional generic fallback text with no per-ticker data to
substitute).

## Additional bugs the new test immediately surfaced and fixed
Running it against real producer output (not the repo's existing mocks, which
turned out to have drifted to encode the same bugs) found:

1. **Multi-Horizon Conviction Matrix rendered zero rows on every report ever
   generated.** The horizon keys the render loop checked
   (`t_plus_1_to_5`/`1M`/`6M`/...) matched NONE of the real payload's keys
   (`t_plus_1_to_t_plus_5`/`1_month`/`6_month`/...) -- `eval_matrix.get(key)`
   returned `None` every time. The per-horizon fields it wanted
   (direction/conviction_score/expected_return_pct/sharpe_ratio/
   primary_driver/optimal_action) also don't exist in the producer, which
   emits a qualitative brief instead (evaluating_agents/focus/
   min_probability_threshold/target_output). Fixed both the keys and the
   columns to match reality rather than fabricate a quant score with no model
   behind it.
2. **Borrow Fee Engine card always showed 0.0 bps / a fabricated 0.0%
   "Lendable Utilization."** Real key is `annual_borrow_rate` (a fraction,
   needs `*10000` for bps), not `borrow_fee_bps`. `utilization_pct` never
   existed anywhere in this pipeline -- no such metric is computed -- so the
   stat was replaced with the real `locate_granted` field rather than left
   fabricating a number.
3. **"Winsorized IV Crush" always showed -0.0%.** Real key is
   `volatility_crush_ratio`, not `historical_crush_ratio`.
4. **5-Day Execution Clock's T0 timestamp always showed the generic "Post-Close
   AMC" default.** Real key is `announcement_timestamp`, not `t0_timestamp` --
   the same bug class as the original PEAD trigger, in an adjacent card.
5. Removed a dead `cagr_pct` read (key never existed, and the resulting
   variable was never referenced in any rendered HTML anyway).

`execution_window`/`t1_open_action`/`t5_exit_action`/`crush_source` were
investigated and confirmed to be intentional static fallback text (no
per-ticker producer field exists to substitute) -- documented in
`ALLOWED_EXTRA_KEYS`, not fixed as bugs.

## Why existing tests never caught any of this
`tests/test_visualize_stock_analysis_refactor.py`'s mock fixtures had
independently drifted to encode the SAME wrong key names the render code used
(`train_folds`/`test_folds`, `temp_impact_bps`/`total_slippage_bps`,
`borrow_fee_bps`/`utilization_pct`, `t0_timestamp`, the old evaluation_matrix
shape, and a `dsr_probability` mock value already pre-scaled to a percentage).
The mocks were written to match the bug, not the real producers, so every
assertion passed while production reports were wrong. Corrected all of these
mocks to the real schemas.

## Verification
- New test: 7/7 pass against real (non-mocked) producer output.
- `test_visualize_stock_analysis_refactor.py`: 12/12 pass after mock corrections.
- Combined touched-module suite: 48/48 pass.
- Full repo suite (excluding `tests/rl/`, `tests/test_pit.py` -- pre-existing
  missing-dependency collection errors): 186 passed, 15 failed (same 15
  pre-existing, confirmed unrelated failures as 2026-09-05's run -- MLflow
  filesystem-backend maintenance mode, network/CN-data tests, a pandas-version
  API deprecation), 1 skipped.
- Regenerated the real INTC report end-to-end via the user's exact command;
  confirmed the PEAD date, borrow fee, and Multi-Horizon Matrix headers all
  render correctly in the actual output file.

## Files changed
- `scripts/visualize_stock_analysis.py` -- 6 key-mismatch fixes, 1 dead-code
  removal, Multi-Horizon Matrix column redesign
- `tests/test_visualize_key_contracts.py` -- new schema-contract regression test
- `tests/test_visualize_stock_analysis_refactor.py` -- corrected mock fixtures
  to match real producer schemas (evaluation_matrix, purged_walk_forward_cv,
  almgren_chriss_market_impact, borrow_fee_engine, deflated_sharpe_ratio,
  calibrate_post_earnings_volatility_surface, earnings_event_clock)

## Open item
No further known instances of this bug class in the 7 render functions this
test covers. It does not yet cover every card-building function in
`visualize_stock_analysis.py` (e.g. performance/projection cards) -- extending
coverage there is a reasonable next step if the same pattern is suspected
elsewhere.
