#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Executive Investment Verdict Taxonomy (Single Source of Truth)
==============================================================
Canonical classification of a `PredictiveForecastResult` (plus optional gamma
squeeze payload) into the repo's Executive Investment Verdict.

WHY THIS MODULE EXISTS
----------------------
The verdict ladder previously lived inline inside the render function
`scripts/visualize_stock_analysis.py::build_buy_timing_verdict_banner_html`.
Any second consumer (e.g. the Russell 1000 cross-sectional screen) would have
had to copy that if/elif chain, creating two ladders that silently drift apart
-- the exact "data layer vs render layer disagree" bug class remediated in
`.team-code/20260905-FIX_adversarial_audit_remediation-implementation_plan.md`.

Every consumer MUST call `classify_executive_verdict`. The badge strings, pill
classes, and gating precedence are defined here once.

TAXONOMY (precedence order -- first match wins)
-----------------------------------------------
1. DO NOT BUY / CAPITAL PRESERVATION MODE      (rose)    tier=NO_BUY
2. IMMEDIATE BUY: HIGH-VELOCITY 5-DAY SPIKE    (emerald) tier=BUY
3. RESEARCH SPIKE PATTERN (ACTION SUPPRESSED)  (amber)   tier=CAUTION
4. STRONG BUY: STRATEGIC MULTI-HORIZON ACCUM.  (emerald) tier=BUY
5. BUY ON PULLBACK: WAIT FOR ENTRY CORRIDOR    (blue)    tier=PULLBACK
6. HOLD / CAUTIOUS BUY                         (amber)   tier=CAUTION
7. (fallback) DO NOT BUY / CAPITAL PRESERVATION (rose)   tier=NO_BUY
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

# Verdict tiers -- drive the report colour convention
TIER_BUY = "BUY"
TIER_PULLBACK = "PULLBACK"
TIER_CAUTION = "CAUTION"
TIER_NO_BUY = "NO_BUY"


@dataclass(frozen=True)
class ExecutiveVerdict:
    """Canonical executive verdict classification for a single instrument."""

    badge: str
    pill_class: str
    color_class: str
    icon: str
    description: str
    tier: str
    is_capital_preservation: bool
    is_spike: bool
    is_synthetic: bool

    # Short label for dense tabular contexts (the cross-sectional screen).
    @property
    def short_label(self) -> str:
        return _SHORT_LABELS[self.tier_key]

    @property
    def tier_key(self) -> str:
        """Stable machine key identifying which ladder branch fired."""
        return self._branch

    _branch: str = "UNKNOWN"


_SHORT_LABELS = {
    "CAPITAL_PRESERVATION": "DO NOT BUY",
    "SPIKE_LIVE": "IMMEDIATE BUY",
    "SPIKE_SYNTHETIC": "RESEARCH SPIKE",
    "STRONG_BUY": "STRONG BUY",
    "PULLBACK": "BUY ON PULLBACK",
    "HOLD_CAUTIOUS": "HOLD / CAUTIOUS BUY",
    "FALLBACK_NO_BUY": "DO NOT BUY",
    "UNKNOWN": "UNKNOWN",
}


def detect_capital_preservation(pred: Dict[str, Any]) -> bool:
    """
    Canonical capital-preservation gate.

    Mirrors `predict_future_buy_timing`'s own `is_capital_preservation`
    computation and additionally defends against a stale/absent flag by
    re-deriving it from the recommendation text and BOCD state.
    """
    rec_upper = str(pred.get("recommendation", "")).upper()
    return bool(
        pred.get("is_capital_preservation", False)
        or not pred.get("is_entry_allowed", True)
        or "DO NOT BUY" in rec_upper
        or "CAPITAL PRESERVATION" in rec_upper
        or "RISK-OFF" in rec_upper
        or "REGIME SHIFT" in rec_upper
        or "PAUSE" in rec_upper
        or "EVENT RISK" in rec_upper
        or pred.get("bocd_regime_state") == 2
    )


def detect_spike(gamma: Dict[str, Any]) -> bool:
    """Canonical 5-trading-day positive gamma spike detection."""
    calib = gamma.get("calibrated_probabilities", {})
    gsi = gamma.get("gsi_scores", {})
    vol_surf = gamma.get("calibrate_post_earnings_volatility_surface", {})

    prob_val = calib.get("calibrated_prob_squeeze")
    if prob_val is None:
        p_raw = calib.get("p_positive_squeeze")
        if p_raw is not None:
            prob_val = float(p_raw) * 100.0
        else:
            prob_val = calib.get("probability_positive_spike", 0.0)
    prob_spike = float(prob_val)

    gsi_pos = float(
        gsi.get("gsi_positive")
        if gsi.get("gsi_positive") is not None
        else gsi.get("gsi_positive_raw", 0.0)
    )
    exp_jump = float(vol_surf.get("expected_jump_pct", 0.0))
    is_pos_candidate = bool(
        gsi.get("is_positive_squeeze_candidate", False)
        or gsi.get("is_positive_alert", False)
        or (gsi_pos >= 60.0)
    )
    return (prob_spike >= 60.0 or gsi_pos >= 60.0 or exp_jump >= 5.0) and is_pos_candidate


def detect_synthetic(gamma: Dict[str, Any]) -> bool:
    """Canonical synthetic / action-suppressed provenance gate."""
    return bool(
        gamma.get("provenance") == "synthetic_research_fallback"
        or gamma.get("safety_status") == "ACTION_SUPPRESSED"
        or not gamma.get("is_actionable", True)
    )


def classify_executive_verdict(
    pred: Optional[Dict[str, Any]],
    gamma_squeeze: Optional[Dict[str, Any]] = None,
) -> ExecutiveVerdict:
    """
    Classify a predictive forecast result into the canonical Executive Verdict.

    Parameters
    ----------
    pred : Optional[Dict[str, Any]]
        `PredictiveForecastResult.to_dict()` payload.
    gamma_squeeze : Optional[Dict[str, Any]]
        Gamma-squeeze engine payload. When absent (e.g. the cross-sectional
        screen, which has no options chain), spike and synthetic gates are
        both False -- meaning the spike branches are unreachable. Callers in
        that situation MUST disclose that to the reader.
    """
    pred = pred or {}
    gamma = gamma_squeeze or {}

    rec_upper = str(pred.get("recommendation", "HOLD / CAUTIOUS BUY")).upper()

    is_capital_preservation = detect_capital_preservation(pred)
    is_spike = detect_spike(gamma)
    is_synthetic = detect_synthetic(gamma)

    if is_capital_preservation:
        branch = "CAPITAL_PRESERVATION"
        badge = "🔴 DO NOT BUY / CAPITAL PRESERVATION MODE"
        pill = "bg-rose-500/15 text-rose-400 border-rose-500/30"
        color = "text-rose-400"
        icon = "▼"
        desc = (
            "Unfavorable technical structure, negative gamma trap, or macroeconomic regime stress. "
            "Maintain capital preservation and inhibit all buy entries."
        )
        tier = TIER_NO_BUY
    elif is_spike and not is_synthetic:
        branch = "SPIKE_LIVE"
        badge = "⚡ IMMEDIATE BUY: HIGH-VELOCITY 5-DAY SPIKE DETECTED"
        pill = "bg-emerald-500/20 text-emerald-300 border-emerald-500/50 glow-green"
        color = "text-emerald-400"
        icon = "⚡"
        desc = (
            "High-conviction convex gamma expansion triggered. Dealer hedging demand expected to "
            "accelerate spot price above trigger strike over the next 5 trading days."
        )
        tier = TIER_BUY
    elif is_spike and is_synthetic:
        branch = "SPIKE_SYNTHETIC"
        badge = "⚡ RESEARCH SPIKE PATTERN (ACTION SUPPRESSED: SYNTHETIC DATA)"
        pill = "bg-amber-500/20 text-amber-300 border-amber-500/50"
        color = "text-amber-400"
        icon = "⚠️"
        desc = (
            "Theoretical 5-day gamma spike modeled on synthetic fallback. Action suppressed until "
            "real-time options chain and live borrow data verify."
        )
        tier = TIER_CAUTION
    elif "STRONG BUY" in rec_upper or "ACCUMULAT" in rec_upper:
        # NOTE: matches the stem "ACCUMULAT" deliberately. Matching the exact
        # token "ACCUMULATE" (the pre-2026-09-05 behaviour) missed every
        # RecommendationEngine branch that emits "ACCUMULATION":
        #   BULLISH MOMENTUM / DIP ACCUMULATION
        #   RANGE ACCUMULATION / BUY SUPPORT
        #   PEAD POST-EARNINGS DRIFT ACCUMULATION
        # All three carry is_entry_allowed=True and an ACTIVE buy window, yet
        # fell through to the rose "DO NOT BUY / CAPITAL PRESERVATION MODE"
        # fallback -- a verdict badge contradicting the execution posture on
        # the same page. See the walkthrough for this fix.
        branch = "STRONG_BUY"
        badge = "🟢 STRONG BUY: STRATEGIC MULTI-HORIZON ACCUMULATION"
        pill = "bg-emerald-500/15 text-emerald-400 border-emerald-500/30"
        color = "text-emerald-400"
        icon = "▲"
        desc = (
            "Favorable multi-horizon risk-reward profile backed by positive drift, institutional "
            "AVWAP support, and low changepoint hazard."
        )
        tier = TIER_BUY
    elif "PULLBACK" in rec_upper:
        branch = "PULLBACK"
        badge = "🔵 BUY ON PULLBACK: WAIT FOR ENTRY CORRIDOR"
        pill = "bg-blue-500/15 text-blue-400 border-blue-500/30"
        color = "text-blue-400"
        icon = "⏳"
        desc = (
            "Stock is currently extended above near-term value. Place limit orders inside the "
            "optimal entry corridor to capture favorable asymmetry."
        )
        tier = TIER_PULLBACK
    elif "HOLD" in rec_upper or "DE-GROSSING" in rec_upper or "CATALYST" in rec_upper:
        # De-grossing / catalyst-haircut recommendations that were NOT caught by
        # the capital-preservation gate above (e.g. "IMMINENT CATALYST / 50%
        # DE-GROSSING") still permit entries at reduced size. Amber, not rose.
        branch = "HOLD_CAUTIOUS"
        badge = "🟡 HOLD / CAUTIOUS BUY: IMMINENT CATALYST & REGIME HAZARD"
        pill = "bg-amber-500/15 text-amber-400 border-amber-500/30"
        color = "text-amber-400"
        icon = "◼"
        desc = (
            "Approaching binary earnings announcement or elevated changepoint hazard. Enforce "
            "position haircuts until catalyst resolution."
        )
        tier = TIER_CAUTION
    elif pred.get("is_entry_allowed", True):
        # INVARIANT: a rose DO-NOT-BUY badge must never be shown for a posture
        # whose buy window is ACTIVE and whose entry corridor is live. If the
        # ladder has no better label for an entry-allowed recommendation, the
        # honest rendering is amber caution, not a red stand-aside that
        # contradicts the corridor printed beside it.
        branch = "HOLD_CAUTIOUS"
        badge = "🟡 HOLD / CAUTIOUS BUY: IMMINENT CATALYST & REGIME HAZARD"
        pill = "bg-amber-500/15 text-amber-400 border-amber-500/30"
        color = "text-amber-400"
        icon = "◼"
        desc = (
            "Approaching binary earnings announcement or elevated changepoint hazard. Enforce "
            "position haircuts until catalyst resolution."
        )
        tier = TIER_CAUTION
    else:
        branch = "FALLBACK_NO_BUY"
        badge = "🔴 DO NOT BUY / CAPITAL PRESERVATION MODE"
        pill = "bg-rose-500/15 text-rose-400 border-rose-500/30"
        color = "text-rose-400"
        icon = "▼"
        desc = (
            "Unfavorable technical structure, negative gamma trap, or macroeconomic regime stress. "
            "Maintain capital preservation."
        )
        tier = TIER_NO_BUY

    return ExecutiveVerdict(
        badge=badge,
        pill_class=pill,
        color_class=color,
        icon=icon,
        description=desc,
        tier=tier,
        is_capital_preservation=is_capital_preservation,
        is_spike=is_spike,
        is_synthetic=is_synthetic,
        _branch=branch,
    )


__all__ = [
    "ExecutiveVerdict",
    "classify_executive_verdict",
    "detect_capital_preservation",
    "detect_spike",
    "detect_synthetic",
    "TIER_BUY",
    "TIER_PULLBACK",
    "TIER_CAUTION",
    "TIER_NO_BUY",
]
