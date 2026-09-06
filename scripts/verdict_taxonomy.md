# Specification: `scripts/verdict_taxonomy.py`

**Created**: 2026-09-05
**Purpose**: Single source of truth for the Executive Investment Verdict taxonomy.

## Why this module exists

Before 2026-09-05 the verdict ladder lived inline inside the render function
`scripts/visualize_stock_analysis.py::build_buy_timing_verdict_banner_html`.
It had exactly one consumer, so that was tolerable. Adding a second consumer
(the Russell 1000 cross-sectional screen) would have required copy-pasting the
`if/elif` chain, producing two ladders that drift apart under future edits --
the same "data layer and render layer silently disagree" failure mode
remediated in `20260905-FIX_adversarial_audit_remediation-implementation_plan.md`.

The classification is therefore extracted here. `build_buy_timing_verdict_banner_html`
now calls it and formats the result; it no longer decides anything.

**Rule: every consumer that needs an Executive Verdict MUST call
`classify_executive_verdict`. Do not re-derive a verdict from `recommendation`
text anywhere else.**

## Public API

### `ExecutiveVerdict` (frozen dataclass)

| Field | Type | Meaning |
|---|---|---|
| `badge` | `str` | Full display badge, e.g. `"🟢 STRONG BUY: STRATEGIC MULTI-HORIZON ACCUMULATION"` |
| `pill_class` | `str` | Tailwind classes for the badge pill |
| `color_class` | `str` | Tailwind text colour |
| `icon` | `str` | Glyph shown beside the badge |
| `description` | `str` | One-paragraph rationale |
| `tier` | `str` | `BUY` \| `PULLBACK` \| `CAUTION` \| `NO_BUY` -- drives the emerald/blue/amber/rose convention |
| `is_capital_preservation` | `bool` | Entries inhibited |
| `is_spike` | `bool` | 5-day gamma spike setup detected |
| `is_synthetic` | `bool` | Gamma payload is synthetic / action-suppressed |
| `short_label` (property) | `str` | Dense-table label, e.g. `"STRONG BUY"` |
| `tier_key` (property) | `str` | Stable machine key naming the branch that fired |

### `classify_executive_verdict(pred, gamma_squeeze=None) -> ExecutiveVerdict`

`pred` is a `PredictiveForecastResult.to_dict()` payload. `gamma_squeeze` is the
gamma-squeeze engine payload; when omitted, `is_spike` and `is_synthetic` are
both `False`, which makes the two spike branches unreachable. **A caller that
omits it must disclose that to the reader** (the Russell 1000 screen does so in
its "Verdicts structurally unreachable" panel).

### Helper predicates
`detect_capital_preservation(pred)`, `detect_spike(gamma)`, `detect_synthetic(gamma)`
are exported so callers can interrogate a single gate without re-classifying.

## Ladder (precedence order, first match wins)

| # | Branch key | Condition | Badge | Tier |
|---|---|---|---|---|
| 1 | `CAPITAL_PRESERVATION` | `detect_capital_preservation(pred)` | 🔴 DO NOT BUY / CAPITAL PRESERVATION MODE | `NO_BUY` |
| 2 | `SPIKE_LIVE` | spike and not synthetic | ⚡ IMMEDIATE BUY: HIGH-VELOCITY 5-DAY SPIKE DETECTED | `BUY` |
| 3 | `SPIKE_SYNTHETIC` | spike and synthetic | ⚡ RESEARCH SPIKE PATTERN (ACTION SUPPRESSED) | `CAUTION` |
| 4 | `STRONG_BUY` | `"STRONG BUY"` or `"ACCUMULAT"` in recommendation | 🟢 STRONG BUY: STRATEGIC MULTI-HORIZON ACCUMULATION | `BUY` |
| 5 | `PULLBACK` | `"PULLBACK"` in recommendation | 🔵 BUY ON PULLBACK: WAIT FOR ENTRY CORRIDOR | `PULLBACK` |
| 6 | `HOLD_CAUTIOUS` | `"HOLD"`, `"DE-GROSSING"` or `"CATALYST"` in recommendation | 🟡 HOLD / CAUTIOUS BUY | `CAUTION` |
| 7 | `HOLD_CAUTIOUS` | otherwise, **and** `is_entry_allowed` | 🟡 HOLD / CAUTIOUS BUY | `CAUTION` |
| 8 | `FALLBACK_NO_BUY` | otherwise | 🔴 DO NOT BUY / CAPITAL PRESERVATION MODE | `NO_BUY` |

### Capital-preservation gate (branch 1)
True when **any** of: `is_capital_preservation`, `not is_entry_allowed`, or the
recommendation contains `DO NOT BUY` / `CAPITAL PRESERVATION` / `RISK-OFF` /
`REGIME SHIFT` / `PAUSE` / `EVENT RISK`, or `bocd_regime_state == 2`.
Deliberately redundant with `predict_future_buy_timing`'s own flag so a stale or
absent flag cannot leak an actionable verdict.

## Behaviour changes made on extraction (2026-09-05)

These are **bug fixes**, not refactors. All were found by running the ladder
across all 908 Russell 1000 names.

### 1. `"ACCUMULATE"` widened to the stem `"ACCUMULAT"`
The old exact-token test missed every `RecommendationEngine` branch emitting
`ACCUMULATION`:

- `BULLISH MOMENTUM / DIP ACCUMULATION` (BOCD state 0)
- `RANGE ACCUMULATION / BUY SUPPORT` (BOCD state 1)
- `PEAD POST-EARNINGS DRIFT ACCUMULATION`

All three carry `is_entry_allowed=True` and an **ACTIVE** buy window, yet fell
through to the rose `DO NOT BUY / CAPITAL PRESERVATION MODE` fallback. The
single-ticker report was rendering a red stand-aside badge directly above a
live entry corridor and buy window. Observed on 74 of 908 names
(`BULLISH MOMENTUM / DIP ACCUMULATION`) in the 2026-09-04 cross-section.

### 2. De-grossing / catalyst recommendations routed to amber
`IMMINENT CATALYST / 50% DE-GROSSING` permits entries at halved size, but was
also hitting the rose fallback. It is a haircut, not a stop.

### 3. Entry-allowed fallback routed to amber, not rose (branch 7)
New invariant: **a rose DO-NOT-BUY badge is never shown for a posture whose buy
window is ACTIVE.** If the ladder has no better label for an entry-allowed
recommendation, amber caution is the honest rendering; the rose fallback is now
reserved for genuinely inhibited or unknown postures. This makes the taxonomy
closed under future additions to `RecommendationEngine` -- a new recommendation
string can no longer silently render as a capital-preservation stand-aside.

## Consumers

- `scripts/visualize_stock_analysis.py::build_buy_timing_verdict_banner_html`
- `scripts/russell1000_factor_verdict_screen.py::analyse_symbol`

## Verification

- `tests/test_visualize_stock_analysis_refactor.py` (12 tests) passes unchanged
  before and after extraction -- it is the behaviour-preservation gate.
- Full touched-module suite: 75 passed.
- End-to-end: for a 10-ticker sample the badge rendered by
  `build_buy_timing_verdict_banner_html` was confirmed to match the verdict in
  the cross-sectional screen's table for the same `pred` (10/10).
