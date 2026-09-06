# Implementation Plan: FIX Analysis Report — Adversarial Audit Remediation

**Target Artifact:** `FIX_analysis_report_2026-09-05.html` (Comfort Systems USA, Inc.)
**Source Audit:** Adversarial Financial Audit & Quantitative Model Verification Report (7 issues, 4 proposed code fixes)
**Author of this plan:** Claude Code, verifying the audit against `e:\SRC\GITHUB\my-qlib` source before scoping fixes.

## Priority-0 Acknowledgement (per `.team-code/requirements.md`)

This plan is scoped for the two end-users defined in [requirements.md](requirements.md):
the **Profitable Stock Trader** and the **Institutional Hedge Fund Manager**. Both
end-users' core complaint is the same: the report presents *numbers that cannot all be
true at once*, which is disqualifying for anyone trading off it. Every fix below is
justified by restoring **internal consistency between the rendered HTML and the
embedded canonical JSON data contract** — not by picking a "nicer" number.

## Verification Method

Before writing this plan I read the actual source for every claim in the audit:
`scripts/visualize_stock_analysis.py` (rendering), `qlib/contrib/derivatives/
earnings_gamma_squeeze_engine.py` (data generation), `qlib/contrib/microstructure/
almgren_chriss_impact.py`, `qlib/contrib/backtest/deflated_sharpe_ratio.py`,
`qlib/contrib/events/event_calendar.py` and `events_data.py`, and
`scripts/train_alpha158_lightgbm.py` / `scripts/infer_alpha158.py`.

**Headline finding: this is one systemic bug wearing seven costumes.** Issues 2, 5, 6,
and 7 all share the identical mechanism: the data-generation layer
(`earnings_gamma_squeeze_engine.py`, `events_data.py`) emits a dict under one set of
key names; the rendering layer (`visualize_stock_analysis.py`) reads a *different*
set of key names via `.get(key, hardcoded_default)`. Because `.get()` never raises,
every one of these mismatches silently falls through to a plausible-looking
**hardcoded fallback value** instead of erroring — which is exactly why QA never
caught it and why Issue 7's council sign-offs are rubber-stamped: the code path that
would show real per-member verdicts can never execute.

## Corrections to the Audit (read before implementing)

1. **Issue 1's "secondary calculation error" is not a bug.** The audit claims
   `$7,369,303 × $1,610.34 = $11.87B ≠ $13,053.8M`. I verified the actual formula in
   `qlib/contrib/derivatives/forced_dealer_hedging.py:88` (`dollar_demand =
   net_shares_demand * S_new`, where `S_new = spot * (1 + dS)` for the +10% bull
   scenario): `7,369,303 × ($1,610.34 × 1.10) = $13,053.79M` — matches the report
   exactly to the dollar. The dollar figure is deliberately computed at the
   **post-jump** price, not current spot. This is a **labeling/clarity defect**
   (the card doesn't say "at +10% scenario price"), not an arithmetic bug. Do not
   "fix" the formula per the audit's Required Fix #1 second clause — that would
   break a value that is already internally correct. Fix the *label* instead.

2. **Issue 4's proposed percentile formula is already what the code does.** I read
   `scripts/train_alpha158_lightgbm.py:834-836`:
   `scores_df.groupby("date")["score"].rank(pct=True, ascending=True) * 100` — for
   rank 179 of 908 (dense rank, 1=best), this formula correctly produces ≈80.3%,
   identical to the audit's "Required Fix #4". **The training-time formula is not
   inverted.** The bug is downstream: either a stale/mismatched cache row in
   `infer_alpha158.py`'s `get_score()`, or the `rank`/`percentile` columns being read
   from different rows/dates. Applying the audit's proposed formula change to
   `train_alpha158_lightgbm.py` would be a no-op that doesn't fix anything and risks
   the reviewer believing it's resolved when it isn't. This needs a runtime trace
   (Task 4 below), not a formula rewrite.

## Root Cause Per Issue (verified) and Fix Plan

### Issue 1 — Dealer Hedging Demand exceeds physical OI bound (CRITICAL)
**Confirmed:** `calculate_forced_dealer_hedging_demand()` bounds each option's delta
change to at most 1.0 (calls ∈[0,1], puts ∈[-1,0] via exact `erf`-based normal CDF —
`qlib/contrib/derivatives/gex.py:70-96`, mathematically sound, no clamping bug). The
theoretical absolute ceiling is therefore `100 × total_OI` = 1,788,800 shares for
17,888 contracts. The reported 7,369,303 shares is ~4.1× that ceiling.
**Root cause (needs one more trace step, flagged below):** `earnings_gamma_squeeze_engine.py:77-86`
falls back to `SyntheticOptionSurfaceGenerator.generate_synthetic_chain(...)` whenever
the real `df_chain` passed in is empty. The "Total Call/Put OI: 17,888" figure shown
in Card 2 is most likely sourced from a *different* chain object (`gex.py:328-329`
`total_call_oi`/`total_put_oi`) than the one actually fed into
`calculate_forced_dealer_hedging_demand`. If the synthetic generator produces a much
larger/unnormalized OI-per-strike distribution than the real chain's true OI, the two
numbers displayed on the same page describe two different chains entirely.
**Tasks:**
1. Instrument `earnings_gamma_squeeze_engine.py` to log/assert
   `sum(df_chain["openInterest"])` at the point it's passed into
   `calculate_forced_dealer_hedging_demand`, and separately trace what chain object
   feeds Card 2's `total_call_oi`/`total_put_oi` in `gex.py`. Confirm whether they are
   the same DataFrame instance.
2. If they differ: make `earnings_gamma_squeeze_engine.py` and the GEX/OI summary
   consume **one single chain object** per report run (real chain if available,
   else the *same* synthetic chain for both), so OI totals and hedging demand can
   never diverge.
3. Add a hard invariant in `calculate_forced_dealer_hedging_demand` (or its caller):
   `assert abs(net_shares_demand) <= 100.0 * (ois_call.sum() + ois_put.sum())`,
   raising/logging loudly rather than rendering an impossible number. This is the
   single highest-leverage guardrail in this whole plan — it converts a silent
   institutional embarrassment into a visible, debuggable failure.
4. Relabel the dollar-demand card to state the scenario price basis explicitly, e.g.
   `"Dollar Hedging Demand (at +10% Spot Scenario)"` — see Correction #1 above.

### Issue 2 — Provenance badge contradicts the gamma-squeeze safety gate (CRITICAL)
**Confirmed root cause:** `visualize_stock_analysis.py:621-624` derives the header
PROVENANCE badge from `derivatives.get('is_synthetic_surface')`
(`scripts/stock_analysis_data.py:263`), while the squeeze-radar section and safety
gate three hundred lines later correctly use
`gamma_squeeze.get("provenance") == "synthetic_research_fallback"` /
`safety_status == "ACTION_SUPPRESSED"` (`visualize_stock_analysis.py:1183-1187`).
These are two independently-computed flags from two different pipeline stages that
can and did disagree.
**Fix:** There must be exactly one canonical "is this surface synthetic/suppressed"
signal per report. Compute it once (prefer the stricter, more specific
`gamma_squeeze.safety_status`/`provenance` gate, since it already accounts for
downstream validation the raw chain-fetch flag doesn't) and thread that single
boolean into both the header badge and the squeeze radar. Delete the independent
`derivatives.is_synthetic_surface` badge-selection branch or make it defer to the
canonical flag when both are present.

### Issue 3 — P(Squeeze) 98.7% shown alongside DO NOT BUY (HIGH)
**Confirmed:** `build_gamma_squeeze_spike_card_html` already computes `is_synthetic`
and `is_cap_pres` and correctly changes the verdict badge/spike-callout text
(lines 1266-1273). But the raw stat readout — `Calibrated P(Squeeze): {prob:.1f}%` and
`Positive GSI: {gsi_pos:.1f}/100` near line 1618-1622 — is rendered from
`prob_squeeze`/`gsi_pos` unconditionally, with no gating.
**Fix:** Once Issue 2's single canonical suppression flag exists, gate these two stat
tiles on it: when `is_capital_preservation or is_synthetic`, render
`"SUPPRESSED (Regime Invalidation)"` in the muted/rose style instead of the raw
percentage, matching the audit's Required Fix #3. Do not delete the underlying
computed value from the JSON — only change how it's displayed when suppressed, so a
downstream consumer reading the JSON contract directly can still see the
unsuppressed sub-model output plus the gate flags together.

### Issue 4 — Alpha158 Percentile / Rank inversion (HIGH)
**Correction applied:** see "Corrections to the Audit" #2 — the training formula is
already correct. Do not touch `train_alpha158_lightgbm.py:834-836`.
**Tasks (investigation, not a known fix yet):**
1. Reproduce with the actual FIX ticker: load the persisted
   `output/scores/alpha158_russell1000_latest.parquet`, filter to FIX and the report's
   `as_of_date`, and manually confirm what `rank` and `percentile` are stored for that
   row.
2. If the stored row itself already shows a mismatched rank/percentile pair, trace
   backward into the `groupby("date")` call at training time for an index-alignment
   bug (e.g., a duplicate-date or duplicate-symbol row silently corrupting one
   groupby's ordering relative to the other).
3. If the stored row is correct but the *report* shows something else, the bug is in
   `infer_alpha158.py::get_score()`'s cache lookup (e.g. `date_match` picking a stale
   date, or `universe_size` from a mismatched date's row count) or in
   `visualize_stock_analysis.py:1013` reading the wrong sub-dict.
4. Only after locating the actual divergent step should a one-line fix be applied —
   given the training formula is already right, this is very likely a plumbing bug,
   not a math bug.

### Issue 5 — Backtesting Protocol card ignores its own JSON (HIGH) — fully confirmed
All three sub-findings traced to exact, distinct root causes in
`visualize_stock_analysis.py::build_backtesting_protocol_card_html`:

| Field | Template reads (wrong) | Actual payload key (`earnings_gamma_squeeze_engine.py`) | Bug type |
|---|---|---|---|
| DSR probability | `dsr.get("dsr_probability")` then `{dsr_prob:.1f}%` | `dsr_probability` (0.8507, a **fraction**) | Missing `*100` — same pattern as the correct `{win_rate*100:.1f}%` two lines below it |
| Purged CV folds | `purged_cv.get("train_folds", 5)` / `get("test_folds", 5)` | Payload only has `"n_folds": 7` — `train_folds`/`test_folds` **do not exist anywhere** in the payload | Reads nonexistent keys, silently uses the `, 5` defaults for both |
| Almgren-Chriss impact | `impact.get("temp_impact_bps")`, `get("perm_impact_bps")`, `get("total_slippage_bps")` | `calculate_market_impact()` returns `temporary_impact_bps`, `permanent_impact_bps`, `total_cost_bps` (`qlib/contrib/microstructure/almgren_chriss_impact.py:62-64`) | Reads nonexistent keys, silently uses `0.0` for all three |

**Fix (line-level, `visualize_stock_analysis.py` ~1904-1928):**
```python
dsr_prob = float(dsr.get("dsr_probability", 0.0)) * 100.0          # was missing *100
train_folds = ...  # n_folds is a single count; render "7 Folds (Purged, Zero-Overlap)"
                    # instead of a Train/Test split that doesn't exist in the schema —
                    # do not invent a train/test split value that was never computed.
temp_bps  = float(impact.get("temporary_impact_bps", 0.0))         # was temp_impact_bps
perm_bps  = float(impact.get("permanent_impact_bps", 0.0))         # was perm_impact_bps
tot_slip  = float(impact.get("total_cost_bps", 0.0))                # was total_slippage_bps
```
Also update the "5 Train / 5 Test Folds" HTML string at line ~2022 to match whatever
the corrected `n_folds`-based display becomes.

### Issue 6 — Catalyst Calendar contradictions (MEDIUM) — fully confirmed
**6a (date mismatch):** `qlib/contrib/events/event_calendar.py::evaluate_catalyst_status`
computes one **composite** `status_description` from whichever of
{earnings, FOMC, CPI} is nearest ("highest threat wins", lines 187-206) — e.g.
`"APPROACHING EVENT: Catalyst in 5 days..."` when CPI is 5 days out. But
`visualize_stock_analysis.py::build_events_card_html` renders this composite
description inside the **earnings-specific** "Catalyst Proximity" card, whose
headline (`days_earn_display`, `next_earn_date`) is 50 days / 2026-11-13. The
composite description was never designed to be event-specific, yet it's placed under
a card whose number is specific to a different event.
**Fix:** Either (a) make `evaluate_catalyst_status` return a *per-event* description
map (`earnings_status_desc`, `macro_status_desc`) and render each under its own card,
or (b) move the composite description to whichever card corresponds to
`composite_proximity`'s actual triggering event (it already tracks `earn_prox`,
`fomc_prox`, `cpi_prox` individually — use those to route the text, not just the
severity level).

**6b (earnings history table all-N/A vs. Card 3 real numbers):** `events_data.py`'s
`generate_...()` (~line 88-107) emits each history record as
`{"date", "quarter", "eps_actual", "eps_estimate", "eps_difference", "surprise_pct"}`.
`visualize_stock_analysis.py::build_events_card_html` (~line 831-845) reads
`h.get("actual_eps")`, `h.get("estimated_eps")` (word order swapped from the real
`eps_actual`/`eps_estimate`), plus `h.get("sue_score")`, `h.get("announcement_gap_pct")`,
`h.get("drift_30d_pct")` — **none of the last three fields are ever computed anywhere**
in `events_data.py`'s synthetic history generator. Card 3's real SUE/Gap/Drift numbers
come from a completely separate calculation (`pead_info`, populated elsewhere,
presumably via `calculate_empirical_sue.py`) that was never wired into the per-quarter
history list.
**Fix:**
1. Rename `eps_actual`→read as `actual_eps` or vice-versa (pick one convention,
   `actual_eps`/`estimated_eps` matches the render side, so fix the smaller surface:
   the two `.get()` calls in `visualize_stock_analysis.py`).
2. Either compute real per-quarter SUE/gap/drift in `events_data.py`'s history
   generator (calling the same SUE/drift functions Card 3 already uses, once per
   historical quarter) so the table has real data, or — if per-quarter values are
   genuinely not computed by design — remove those three columns from the rendered
   table rather than showing four rows of fabricated-looking "N/A" next to a card
   that confidently asserts real numbers for what should be the same underlying
   events.

### Issue 7 — Council sign-offs are structurally incapable of failing (MEDIUM) — fully confirmed
**Confirmed root cause:** `earnings_gamma_squeeze_engine.py:340-365`'s
`council_interrogation_outcomes` payload has keys `high_earning_trader`,
`quant_developer`, `top_hedge_fund_manager`, `global_finance_manager`,
`council_multi_horizon_consensus` — **none of which match** the keys
`visualize_stock_analysis.py:1937-1942` looks up (`dr_vance`, `marcus_reynolds`,
`dr_rostova`, `julian_montgomery`, `sophia_chen`, `arthur_pendelton`). Every
`council.get(key, {})` call returns `{}`, so `audit.get("verdict", "APPROVED")` and
`audit.get("notes", "Quantitative standards validated. Invariants enforced.")`
**always** hit their hardcoded defaults. This is not a design choice to rubber-stamp;
the wiring to real per-member verdicts was never connected, so the section cannot
currently render anything else regardless of the underlying numbers.
**Fix:** Decide the actual design intent, then implement real assertions:
1. Either rename the six members in the render loop to match the payload's actual
   five entries (`high_earning_trader`, etc.) and derive `verdict` programmatically
   from real thresholds (e.g., Marcus Reynolds / Chief Risk Officer's verdict should
   be `CAUTION`/`REJECTED` when `impact.total_cost_bps` or Issue 1's dealer-demand
   invariant is violated — this is exactly the kind of check that should have caught
   Issues 1 and 5), **or**
2. Add the six named-member keys to `council_interrogation_outcomes` in
   `earnings_gamma_squeeze_engine.py`, each computed from a real programmatic check
   against that member's stated audit focus (e.g., Dr. Vance's "Derivatives & Vol
   Surface" verdict should fail when `is_synthetic_surface` is true; Marcus Reynolds'
   "Execution & Slippage" verdict should fail when Issue 1's OI-bound invariant or
   the Almgren-Chriss zero-slippage anomaly fires).
Either path converts this section from decorative to load-bearing — which is the
actual institutional-integrity gap the audit is pointing at.

## Systemic Guardrail (apply once, benefits every issue above)

Add a lightweight schema-contract check that runs wherever
`earnings_gamma_squeeze_engine.py`'s payload is handed to
`visualize_stock_analysis.py` (e.g., in the report-generation entrypoint, before
rendering): for each `.get("key", default)` call site currently reading gamma-squeeze
/ backtesting-protocol data, replace silent `.get(..., default)` fallbacks with a
strict lookup that raises or `logger.error()`s on a missing key when the parent dict
is non-empty (i.e., "the section exists but this specific field doesn't" should never
look identical to "the section is legitimately absent"). This is what should have
caught Issues 2, 5, and 7 as unit-test or smoke-test failures well before a real
ticker's report reached a reviewer.

## Phased Execution Order

1. **Phase A (mechanical key-name fixes — lowest risk, highest confidence):**
   Issue 5 (all three sub-items), Issue 6b's `actual_eps`/`estimated_eps` swap,
   Issue 2's single-flag consolidation. These are confirmed, surgical, single-file
   changes in `visualize_stock_analysis.py`.
2. **Phase B (labeling/display-only, no data changes):** Issue 1's dollar-demand
   label, Issue 3's suppression-aware stat tiles (depends on Phase A's Issue 2 flag).
3. **Phase C (needs a short investigation before a fix can be written):** Issue 4
   (runtime trace through `infer_alpha158.py`), Issue 1's OI-source reconciliation
   (trace whether GEX-card OI and hedging-engine OI share a chain object), Issue 6a
   (decide per-event vs. composite description routing).
4. **Phase D (design decision required from the user before implementing):** Issue 7
   — needs a decision on which of the two fix paths (rename-and-derive vs.
   add-real-member-keys) matches intent, plus Issue 1's new hard invariant assertion
   (confirm the assertion should hard-fail report generation vs. log-and-flag).

## Verification Plan

- For each Phase A/B fix, regenerate the FIX report and diff the specific rendered
  numbers against the embedded JSON `<script>` block's canonical values — they must
  match exactly (mirrors the equivalence-testing approach already used for
  `audit_dataset_segments`, see [audit_dataset_segments.md](audit_dataset_segments.md)).
- Add the OI-bound invariant (Issue 1, Task 3) as a unit test in
  `tests/test_forced_dealer_hedging.py`, asserting
  `abs(shares_demand) <= 100 * total_oi` across a range of synthetic chains,
  so this class of bug cannot silently regress.
- Add a schema-contract test that walks `council_interrogation_outcomes`,
  `purged_walk_forward_cv`, and `almgren_chriss_market_impact` payload keys against
  the exact set the template reads, failing CI if they diverge — this directly
  prevents Issues 5 and 7 from recurring under a different ticker's data shape.
- Per `.team-code/requirements.md` Part 2, create/update a same-named spec markdown
  file in `.team-code/` for every function touched (e.g.
  `evaluate_catalyst_status.md`, `calculate_market_impact.md`,
  `build_backtesting_protocol_card_html.md` if one doesn't already exist).

## Files Expected to Change

- `scripts/visualize_stock_analysis.py` (majority of fixes — rendering layer)
- `qlib/contrib/derivatives/earnings_gamma_squeeze_engine.py` (Issue 7 payload keys,
  Issue 1 chain-sourcing)
- `qlib/contrib/events/event_calendar.py` (Issue 6a per-event description)
- `qlib/contrib/events/events_data.py` (Issue 6b, if per-quarter SUE/drift is added)
- `scripts/infer_alpha158.py` and/or `scripts/train_alpha158_lightgbm.py` (Issue 4,
  pending Phase C investigation — do not touch the percentile formula itself)
- `tests/test_forced_dealer_hedging.py` (new invariant test)
- New: a schema-contract test module for the gamma-squeeze/backtesting payload

## End-User Sign-Off (Team-Code Phase 4)

Per the protocol, after Phases A–C land, the Profitable Stock Trader and
Institutional Hedge Fund Manager end-users must be presented with a regenerated FIX
report and asked to confirm: (1) no rendered number contradicts its own JSON
contract, (2) the P(Squeeze)/verdict banners never disagree, and (3) the council
sign-off section now reflects real computed verdicts rather than boilerplate. Their
feedback and any remaining open items go into the walkthrough document for this
plan's implementation.
