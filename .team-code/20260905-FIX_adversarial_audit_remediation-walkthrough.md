# Walkthrough: FIX Adversarial Audit Remediation (Revision 1)

**Implements**: [20260905-FIX_adversarial_audit_remediation-implementation_plan.md](20260905-FIX_adversarial_audit_remediation-implementation_plan.md)
**Status**: Phases A-C complete and verified. Phase D (design decisions) resolved
during implementation -- see "Decisions Made" below.

## Priority-0 Acknowledgement

Per [requirements.md](requirements.md), this work is scoped for the **Profitable
Stock Trader** and **Institutional Hedge Fund Manager** end-users. Every fix below
either removes a place where the rendered HTML contradicted its own embedded JSON
canonical data, or removes a place where a stat could contradict another stat on
the same page. Nothing here changes the underlying models' methodology.

## What Was Fixed

### Issue 1 -- Dealer hedging demand
- **Corrected the audit's own math**: `dollar_demand = shares * spot * 1.10`
  (priced at the +10% scenario, not current spot) -- not a bug. Card now labels it
  `"Dollar Demand (at +10% Spot Scenario)"` instead of implying current-spot pricing.
- **Added a physical invariant**: `qlib/contrib/derivatives/forced_dealer_hedging.py`
  now computes `max_physical_shares_demand = 100 * total_chain_OI` per scenario and
  flags `invariant_ok=False` (logged as a warning) if `shares_demand` exceeds it --
  since each option leg's delta change is mathematically bounded to `[0,1]`/`[-1,0]`,
  this can never be legitimate. The report card now shows an "INVARIANT VIOLATED"
  badge and a red-bordered warning line instead of silently rendering an impossible
  number.
- **Hardened chain sourcing**: `scripts/stock_analysis_data.py` previously called
  `SyntheticOptionSurfaceGenerator.generate_synthetic_chain(...)` twice
  independently (once for the GEX/OI card, once for the gamma-squeeze engine).
  Verified these happened to produce identical chains today (same seed, same
  effective parameters) but nothing enforced that. Now the first-built chain is
  captured and reused by the second call site whenever the spot price matches, so
  the two can never silently diverge under a future parameter change.
- New Council member Marcus Reynolds' verdict is now tied directly to this
  invariant (see Issue 7).

### Issue 2 -- Provenance badge vs. safety gate
- The header PROVENANCE badge and the gamma-squeeze section's own safety gate were
  computed by two independent flags (`derivatives.is_synthetic_surface` vs.
  `gamma_squeeze.provenance`/`safety_status`) that could disagree.
- `build_derivatives_card_html` now accepts `is_synthetic_or_suppressed` and the
  call site computes one canonical value from the gamma-squeeze gate, passed to
  both the header badge and (already) the squeeze radar section.

### Issue 3 -- P(Squeeze) 98.7% next to "DO NOT BUY"
- The `Calibrated P(Squeeze)` and `Positive GSI` stat tiles now render `SUPPRESSED`
  in place of the raw number specifically when `is_capital_preservation` is true --
  the exact contradiction the audit flagged (an executive DO-NOT-BUY verdict next
  to a bright bullish stat).
- **Scope note**: gating was deliberately *not* extended to plain
  `is_synthetic` (no capital-preservation trigger) -- an existing test,
  `test_gamma_squeeze_card_synthetic_and_corridor_invariants`, encodes an
  intentional prior design decision that synthetic-but-not-suppressed numbers
  should still render (framed by the card's existing amber "THEORETICAL SPIKE
  SETUP" badge) for research visibility. Caught this via the test suite before
  it shipped; see "How This Was Verified" below.

### Issue 4 -- Alpha158 rank/percentile mismatch
- **The audit's proposed fix (rewrite the percentile formula) would have been a
  no-op** -- the formula was already correct. Traced the actual bug by reading
  `output/scores/alpha158_russell1000_latest.parquet` directly: on 2026-09-04 the
  model's score distribution has only 232 distinct values across 908 names (ties
  up to 120-wide -- a known, separately-tracked issue, see
  [20260905-finance_team_review_alpha158_degenerate_score.md](20260905-finance_team_review_alpha158_degenerate_score.md)).
  `rank` was computed with `method="dense"`, which under this much tie-degeneracy
  ranks *distinct values* (max dense rank was 232, not 908) rather than
  cross-sectional position -- for FIX this produced rank 179, while the
  independently-correct `percentile` (51.8%) implied a true rank near 430-490.
  Recomputing with `method="min"` (standard competition ranking) gives FIX rank
  389, consistent with the stored 51.8th percentile.
- Fixed in `scripts/train_alpha158_lightgbm.py`; also regenerated
  `output/scores/alpha158_russell1000_latest.{parquet,csv}` in place with the
  corrected ranks so the fix is visible without a full retrain.
- Does **not** fix the underlying score degeneracy itself -- that is a separate,
  already-tracked model-quality issue, out of scope for this audit's Issue 4.

### Issue 5 -- Backtesting Protocol card ignoring its own JSON
All three confirmed and fixed at the exact source:
- `dsr_probability` is a fraction (0.8507) but was rendered without `*100`
  (`{dsr_prob:.1f}%` -> "0.9%" instead of "85.1%") -- the same scaling `win_rate`
  already gets two lines below. Fixed.
- `purged_walk_forward_cv` only ever contained `n_folds` (a single expanding-window
  fold count); the template read nonexistent `train_folds`/`test_folds` keys,
  always defaulting to "5 Train / 5 Test Folds" regardless of the real `n_folds=7`.
  Replaced the fabricated train/test split with the real `n_folds` and the real
  per-fold `train_window_days`/`test_window_days`.
- `almgren_chriss_market_impact` returns `temporary_impact_bps`/
  `permanent_impact_bps`/`total_cost_bps`; the template read
  `temp_impact_bps`/`perm_impact_bps`/`total_slippage_bps` (none of which exist),
  always defaulting to 0.0 bps. Fixed to the real key names.

### Issue 6 -- Catalyst calendar contradictions
- **6a**: `evaluate_catalyst_status`'s `status_code` key never existed (only
  `composite_proximity` did) -- every consumer reading it, including the
  earnings card's own badge color and `stock_analysis_engine.py`'s near-term
  event-driven volatility sizing, silently always saw "SAFE". Added `status_code`
  as a real alias. Separately, the composite `status_description` ("Catalyst in 5
  days") is about whichever of {earnings, FOMC, CPI} is nearest, with no
  indication of which -- rendering it under the earnings-specific card produced
  exactly the audit's contradiction. Added `earnings_status_code`/
  `earnings_status_description` and `macro_status_code`/`macro_status_description`
  (event-specific), and rewired both cards to their own event's fields including
  the macro card's real `next_fomc_date`/`fomc_days_away`/`next_cpi_date`/
  `cpi_days_away` (previously read nonexistent `next_macro_event`/`next_macro_date`
  keys, always showing "FOMC / CPI" / "TBD").
- **6b**: Historical earnings table read `actual_eps`/`estimated_eps`/`sue_score`/
  `announcement_gap_pct`/`drift_30d_pct` -- none of which were ever computed in
  `events_data.py`'s per-quarter records (only `eps_actual`/`eps_estimate`/
  `surprise_pct` exist there), so every row always showed N/A regardless of data.
  Added `PEADEngine.evaluate_earnings_history`/`compute_report_reaction`
  (`qlib/contrib/events/pead.py`) to annotate each historical quarter with real
  SUE/gap/a fixed 21-trading-day drift, using the *same* methodology the "most
  recent report" summary card already used -- so the two can never disagree about
  the same event again. Table header corrected from "30D POST DRIFT" to "21D POST
  DRIFT" to match the actual fixed window.

### Issue 7 -- Rubber-stamp council sign-offs
- The six named members the report renders (`dr_vance`, `marcus_reynolds`, ...)
  never existed as keys in `council_interrogation_outcomes` (only five differently-
  named, generic keys did) -- every member's verdict/notes unconditionally hit the
  render side's hardcoded "APPROVED" defaults, incapable of ever flagging a real
  violation.
- Added `_build_council_verdicts(...)` in `earnings_gamma_squeeze_engine.py`,
  deriving each of the six members' verdict from a real check against their
  stated focus (see the plan/spec for the full mapping) -- including Marcus
  Reynolds' verdict now failing when Issue 1's OI invariant is violated, and
  Arthur Pendelton's verdict now failing when `is_actionable` is false (directly
  encoding the Issue 3 contradiction check at the data layer, not just display).
- The panel header no longer hardcodes "100% Invariant Validation" -- it counts
  real approvals and shows `{n}/6 Approved`.

## Decisions Made (resolving Plan Phase D without further user input)

- **Issue 7 fix path**: chose "add the six named-member keys with real checks"
  over "rename the render loop to the five generic keys" -- the six named
  personas' stated audit focuses map cleanly onto data already computed in
  `evaluate_earnings_gamma_squeeze`, while the five generic keys' existing content
  (e.g. `quant_developer.alpha_decay_annual_pct`) has no relationship to the
  focuses the render side displays for them.
- **Issue 1 invariant severity**: chose "log a warning + flag `invariant_ok=False`
  + show a red UI badge" over "hard-fail report generation" -- consistent with how
  every other data-quality problem in this pipeline (e.g. synthetic fallback) is
  surfaced as a visible flag rather than an exception, so one bad chain doesn't
  take down an otherwise-renderable report.

## How This Was Verified

- Every root cause was traced to an exact file/line by reading source, not by
  trusting the audit's narrative -- two of the audit's seven claims turned out to
  be factually wrong (Issue 1's "secondary calculation error" and Issue 4's
  proposed fix formula) and are corrected above rather than "fixed".
- `output/scores/alpha158_russell1000_latest.parquet` was read directly to
  reproduce Issue 4 against real persisted data (not a synthetic repro).
- `tests/test_forced_dealer_hedging.py`: added
  `test_shares_demand_never_exceeds_physical_oi_ceiling` and
  `test_invariant_flag_present_and_consistent_with_ceiling_formula`.
- Full existing suite for every touched module re-run: `test_almgren_chriss_market_impact.py`,
  `test_deflated_sharpe_ratio.py`, `test_earnings_event_clock.py`,
  `test_earnings_gamma_squeeze_engine.py`, `test_events_pead.py`,
  `test_negative_gamma_squeeze.py`, `test_positive_gamma_squeeze.py`,
  `test_stock_analysis_data.py`, `test_visualize_stock_analysis_refactor.py`,
  `test_forced_dealer_hedging.py` -- **43/43 passed**. One pre-existing test
  (`test_gamma_squeeze_card_synthetic_and_corridor_invariants`) initially broke
  from an over-broad Issue 3 gate; narrowed the gate to match its documented
  intent rather than changing the test.
- Full repo test suite run (excluding `tests/rl/` and `tests/test_pit.py`, which
  fail to collect in this environment for unrelated missing-dependency reasons --
  `tianshou`, `baostock`): **179 passed, 15 failed, 1 skipped**. All 15 failures
  confirmed pre-existing and unrelated (MLflow filesystem-backend maintenance-mode
  exceptions, CN-data/network-dependent tests, a pandas-version API deprecation) --
  none touch any file this work modified.

## Items Not Completed / Explicitly Out of Scope

- **The Alpha158 score degeneracy itself** (232 distinct values across 908 names)
  is not fixed here -- only its downstream rank/percentile inconsistency is. That
  is tracked separately in
  [20260905-finance_team_review_alpha158_degenerate_score.md](20260905-finance_team_review_alpha158_degenerate_score.md).
- **Issue 1's root chain-source question** ("were the two chains ever actually
  different for the FIX report that triggered this audit?") could not be
  conclusively answered without re-running the exact original FIX report request
  (network-dependent, not attempted here) -- verified the two call sites use
  identical seed/parameters *today* and hardened them to share one object, which
  closes the failure mode regardless of the original trigger.
- No live end-user (Trader / Hedge Fund Manager) review has occurred yet -- this
  walkthrough is the artifact to present for that review per Team-Code Phase 4.

## Files Changed

- `scripts/visualize_stock_analysis.py` -- rendering fixes for Issues 1, 2, 3, 5, 6, 7
- `scripts/stock_analysis_data.py` -- Issue 1 chain-sharing hardening
- `scripts/train_alpha158_lightgbm.py` -- Issue 4 rank method fix
- `qlib/contrib/derivatives/earnings_gamma_squeeze_engine.py` -- Issue 1 payload fields, Issue 7 council verdicts
- `qlib/contrib/derivatives/forced_dealer_hedging.py` -- Issue 1 physical invariant
- `qlib/contrib/events/event_calendar.py` -- Issue 6a per-event status fields
- `qlib/contrib/events/pead.py` -- Issue 6b per-quarter history annotation
- `qlib/contrib/events/__init__.py` -- wires the above into `recent_earnings_history`
- `tests/test_forced_dealer_hedging.py` -- new invariant regression tests
- `output/scores/alpha158_russell1000_latest.{parquet,csv}` -- regenerated with corrected ranks
- `.team-code/calculate_forced_dealer_hedging_demand.md`,
  `evaluate_earnings_gamma_squeeze.md`, `event_risk_pead.md`,
  `train_alpha158_lightgbm.md`, `audit_dataset_segments.md` -- spec updates
- `.team-code/20260905-FIX_adversarial_audit_remediation-implementation_plan.md` -- this work's plan (prior revision)

## Suggested Next Step for the User

Regenerate the FIX report end-to-end (`scripts/stock_analysis_data.py` then
`scripts/visualize_stock_analysis.py`, network access required for live/refreshed
data) and visually confirm: no card contradicts its own embedded JSON, the P(Squeeze)
banner and executive verdict never disagree, and the council panel shows real,
non-uniform verdicts.
