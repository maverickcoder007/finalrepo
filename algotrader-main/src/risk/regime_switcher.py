"""
Regime-Aware Strategy Switcher
================================

Combines three market regime signals to auto-select the optimal
F&O strategy type:

  1. **OI flow direction** — from intraday OI analysis (bullish/bearish/neutral)
  2. **Volatility regime** — from VIX / ATM IV (low/medium/high/extreme)
  3. **Index trend state** — from price action (trending-up/trending-down/range/breakout)

Strategy mapping:
  • Range / low-vol       → Iron Condor
  • Range / medium-vol    → Credit Spreads (put or call based on bias)
  • Trending + breakout   → Debit Spreads (directional)
  • Compression → extreme → Straddle / Strangle (vol expansion)
  • High-vol + range      → Short Straddle / Short Strangle

The switcher produces a recommendation that the agent can execute
through the OI → FNO bridge or F&O builder.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger("regime_switcher")


class MarketDirection(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH_BREAKOUT = "bullish_breakout"
    BEARISH_BREAKOUT = "bearish_breakout"


class VolatilityRegime(str, Enum):
    LOW = "low"           # VIX < 13 or IV < 15%
    MEDIUM = "medium"     # VIX 13-18 or IV 15-22%
    HIGH = "high"         # VIX 18-25 or IV 22-35%
    EXTREME = "extreme"   # VIX > 25 or IV > 35%


class TrendState(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGE_BOUND = "range_bound"
    BREAKOUT_UP = "breakout_up"
    BREAKOUT_DOWN = "breakout_down"
    COMPRESSION = "compression"


class RecommendedStrategy(str, Enum):
    IRON_CONDOR = "iron_condor"
    PUT_CREDIT_SPREAD = "put_credit_spread"
    CALL_CREDIT_SPREAD = "call_credit_spread"
    CALL_DEBIT_SPREAD = "call_debit_spread"
    PUT_DEBIT_SPREAD = "put_debit_spread"
    SHORT_STRADDLE = "short_straddle"
    SHORT_STRANGLE = "short_strangle"
    LONG_STRADDLE = "long_straddle"
    LONG_STRANGLE = "long_strangle"
    NO_TRADE = "no_trade"


@dataclass
class RegimeSnapshot:
    """Current market regime assessment."""
    timestamp: str = ""
    underlying: str = "NIFTY"

    # Input signals
    oi_direction: str = "neutral"
    oi_confidence: float = 0.0
    volatility_regime: str = "medium"
    vix_value: float = 0.0
    atm_iv: float = 0.0
    trend_state: str = "range_bound"
    trend_strength: float = 0.0

    # Recommendation
    recommended_strategy: str = "no_trade"
    recommendation_confidence: float = 0.0
    reasoning: list[str] = field(default_factory=list)

    # Alternative strategies
    alternatives: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in d:
            if isinstance(d[k], float):
                d[k] = round(d[k], 4)
        return d


# ══════════════════════════════════════════════════════════════
# Strategy selection matrix
# ══════════════════════════════════════════════════════════════

# Key: (trend_state, vol_regime, oi_direction) → (strategy, confidence_boost)
# Wildcards use "*"

_STRATEGY_MATRIX: list[tuple[tuple[str, str, str], str, float, str]] = [
    # (trend, vol, direction) → strategy, conf_boost, reason

    # ── Range markets ────────────────────────────────
    (("range_bound", "low", "*"),        "iron_condor",         0.85, "Low-vol range: ideal for iron condors"),
    (("range_bound", "medium", "neutral"), "iron_condor",       0.80, "Medium-vol neutral range: iron condor with wider wings"),
    (("range_bound", "medium", "bullish"), "put_credit_spread", 0.75, "Range with bullish bias: sell puts"),
    (("range_bound", "medium", "bearish"), "call_credit_spread", 0.75, "Range with bearish bias: sell calls"),
    (("range_bound", "high", "neutral"),  "short_strangle",     0.70, "High-vol range: sell premium via strangle"),
    (("range_bound", "high", "bullish"),  "put_credit_spread",  0.70, "High-vol bullish range: sell puts for premium"),
    (("range_bound", "high", "bearish"),  "call_credit_spread", 0.70, "High-vol bearish range: sell calls for premium"),
    (("range_bound", "extreme", "*"),     "short_straddle",     0.65, "Extreme vol in range: sell premium aggressively"),

    # ── Compression → breakout expected ──────────────
    (("compression", "low", "*"),         "long_strangle",      0.70, "Compression in low vol: buy vol for breakout"),
    (("compression", "medium", "*"),      "long_straddle",      0.65, "Compression in medium vol: buy straddle for expansion"),
    (("compression", "high", "*"),        "iron_condor",        0.60, "Compression in high vol: sell premium, breakout priced in"),
    (("compression", "extreme", "*"),     "no_trade",           0.50, "Extreme vol compression: unclear direction, wait"),

    # ── Trending up ──────────────────────────────────
    (("trending_up", "low", "bullish"),   "call_debit_spread",  0.80, "Bullish trend + low vol: buy call spreads cheaply"),
    (("trending_up", "low", "*"),         "put_credit_spread",  0.75, "Uptrend low vol: sell puts underlyingly"),
    (("trending_up", "medium", "bullish"), "call_debit_spread", 0.75, "Bullish trend: directional call spread"),
    (("trending_up", "medium", "*"),      "put_credit_spread",  0.70, "Uptrend: sell put credit spread"),
    (("trending_up", "high", "bullish"),  "put_credit_spread",  0.65, "High-vol uptrend: sell puts for premium"),
    (("trending_up", "high", "*"),        "put_credit_spread",  0.60, "High-vol uptrend: cautious put credit spread"),
    (("trending_up", "extreme", "*"),     "no_trade",           0.40, "Extreme vol in trend: wait for clarity"),

    # ── Trending down ────────────────────────────────
    (("trending_down", "low", "bearish"), "put_debit_spread",   0.80, "Bearish trend + low vol: buy put spreads"),
    (("trending_down", "low", "*"),       "call_credit_spread", 0.75, "Downtrend low vol: sell calls"),
    (("trending_down", "medium", "bearish"), "put_debit_spread", 0.75, "Bearish trend: directional put spread"),
    (("trending_down", "medium", "*"),    "call_credit_spread", 0.70, "Downtrend: sell call credit spread"),
    (("trending_down", "high", "bearish"), "call_credit_spread", 0.65, "High-vol downtrend: sell calls for premium"),
    (("trending_down", "high", "*"),      "call_credit_spread", 0.60, "High-vol downtrend: cautious call credit spread"),
    (("trending_down", "extreme", "*"),   "no_trade",           0.40, "Extreme vol crash: wait for stabilization"),

    # ── Breakout up ──────────────────────────────────
    (("breakout_up", "*", "bullish"),     "call_debit_spread",  0.85, "Bullish breakout: aggressive call debit spread"),
    (("breakout_up", "*", "*"),           "call_debit_spread",  0.70, "Breakout up: directional long calls"),

    # ── Breakout down ───────────────────────────────
    (("breakout_down", "*", "bearish"),   "put_debit_spread",   0.85, "Bearish breakout: aggressive put debit spread"),
    (("breakout_down", "*", "*"),         "put_debit_spread",   0.70, "Breakout down: directional long puts"),
]


class RegimeStrategySwithcer:
    """
    Combines multiple regime signals to recommend optimal F&O strategy.
    """

    def __init__(self) -> None:
        self._history: list[RegimeSnapshot] = []

    def assess_regime(
        self,
        underlying: str = "NIFTY",
        oi_direction: str = "neutral",
        oi_confidence: float = 0.0,
        vix_value: float = 0.0,
        atm_iv: float = 0.0,
        spot_price: float = 0.0,
        sma_20: float = 0.0,
        sma_50: float = 0.0,
        recent_high: float = 0.0,
        recent_low: float = 0.0,
        atr_pct: float = 0.0,
        bollinger_width: float = 0.0,
    ) -> RegimeSnapshot:
        """
        Assess current regime and recommend strategy.

        Args:
            underlying: Index name (NIFTY / BANKNIFTY / SENSEX).
            oi_direction: From OI analysis — bullish/bearish/neutral/bullish_breakout/bearish_breakout.
            oi_confidence: Confidence 0-1 from OI analysis.
            vix_value: India VIX value (e.g. 14.5).
            atm_iv: ATM implied volatility as decimal (e.g. 0.18 for 18%).
            spot_price: Current spot price.
            sma_20: 20-period SMA.
            sma_50: 50-period SMA.
            recent_high: Recent swing high (20-day).
            recent_low: Recent swing low (20-day).
            atr_pct: ATR as percentage of price.
            bollinger_width: Bollinger Band width (current / 20-period average).

        Returns:
            RegimeSnapshot with recommendation.
        """
        snap = RegimeSnapshot(
            timestamp=datetime.now().isoformat(),
            underlying=underlying,
            oi_direction=oi_direction,
            oi_confidence=oi_confidence,
            vix_value=vix_value,
            atm_iv=atm_iv,
        )

        # ── Determine volatility regime ──────────────────────
        snap.volatility_regime = self._classify_volatility(vix_value, atm_iv)

        # ── Determine trend state ────────────────────────────
        snap.trend_state, snap.trend_strength = self._classify_trend(
            spot_price, sma_20, sma_50, recent_high, recent_low,
            atr_pct, bollinger_width, oi_direction,
        )

        # ── Match strategy from matrix ───────────────────────
        best_strategy, best_conf, best_reason = self._match_strategy(
            snap.trend_state, snap.volatility_regime, snap.oi_direction,
        )

        # Adjust confidence by OI confidence
        final_conf = best_conf * (0.5 + 0.5 * oi_confidence)

        snap.recommended_strategy = best_strategy
        snap.recommendation_confidence = final_conf
        snap.reasoning = [best_reason]

        # ── Add reasoning details ────────────────────────────
        snap.reasoning.append(f"Volatility regime: {snap.volatility_regime} (VIX={vix_value:.1f}, IV={atm_iv*100:.1f}%)")
        snap.reasoning.append(f"Trend: {snap.trend_state} (strength={snap.trend_strength:.2f})")
        snap.reasoning.append(f"OI direction: {oi_direction} (confidence={oi_confidence:.2f})")

        # ── Find alternatives ────────────────────────────────
        alternatives = self._find_alternatives(
            snap.trend_state, snap.volatility_regime, snap.oi_direction,
            exclude=best_strategy,
        )
        snap.alternatives = alternatives

        # Store
        self._history.append(snap)
        if len(self._history) > 200:
            self._history = self._history[-200:]

        logger.info(
            "regime_assessed",
            underlying=underlying,
            trend=snap.trend_state,
            vol=snap.volatility_regime,
            oi=oi_direction,
            strategy=best_strategy,
            confidence=round(final_conf, 2),
        )

        return snap

    def get_current_regime(self, underlying: str = "NIFTY") -> Optional[dict[str, Any]]:
        """Get the most recent regime assessment for an underlying."""
        for snap in reversed(self._history):
            if snap.underlying == underlying:
                return snap.to_dict()
        return None

    def get_history(self, underlying: str = "", limit: int = 50) -> list[dict[str, Any]]:
        """Get regime assessment history."""
        filtered = self._history
        if underlying:
            filtered = [s for s in filtered if s.underlying == underlying]
        return [s.to_dict() for s in filtered[-limit:]]

    # ── Classification helpers ───────────────────────────────

    @staticmethod
    def _classify_volatility(vix: float, atm_iv: float) -> str:
        """Classify volatility regime from VIX or ATM IV."""
        # Prefer VIX if available
        if vix > 0:
            if vix < 13:
                return "low"
            elif vix < 18:
                return "medium"
            elif vix < 25:
                return "high"
            else:
                return "extreme"

        # Fall back to ATM IV
        if atm_iv > 0:
            if atm_iv < 0.15:
                return "low"
            elif atm_iv < 0.22:
                return "medium"
            elif atm_iv < 0.35:
                return "high"
            else:
                return "extreme"

        return "medium"  # Default

    @staticmethod
    def _classify_trend(
        spot: float, sma_20: float, sma_50: float,
        high: float, low: float,
        atr_pct: float, bb_width: float,
        oi_direction: str,
    ) -> tuple[str, float]:
        """
        Classify trend state and strength.

        Returns:
            (trend_state, trend_strength 0-1)
        """
        if spot <= 0:
            return ("range_bound", 0.0)

        signals: dict[str, float] = {}

        # SMA crossover
        if sma_20 > 0 and sma_50 > 0:
            if sma_20 > sma_50 * 1.01:
                signals["sma_bullish"] = min(1.0, (sma_20 / sma_50 - 1) * 50)
            elif sma_20 < sma_50 * 0.99:
                signals["sma_bearish"] = min(1.0, (1 - sma_20 / sma_50) * 50)

        # Price vs SMAs
        if sma_20 > 0 and spot > sma_20 * 1.01:
            signals["price_above_sma20"] = min(1.0, (spot / sma_20 - 1) * 30)
        elif sma_20 > 0 and spot < sma_20 * 0.99:
            signals["price_below_sma20"] = min(1.0, (1 - spot / sma_20) * 30)

        # Range breakout
        if high > 0 and low > 0 and high > low:
            range_pct = (high - low) / low
            if spot > high * 0.998:
                signals["breakout_up"] = min(1.0, (spot - high) / (high * 0.01) + 0.5)
            elif spot < low * 1.002:
                signals["breakout_down"] = min(1.0, (low - spot) / (low * 0.01) + 0.5)

        # Bollinger compression
        if bb_width > 0 and bb_width < 0.7:
            signals["compression"] = 1.0 - bb_width

        # Determine state
        bullish_score = sum(v for k, v in signals.items() if "bullish" in k or "above" in k or "breakout_up" in k)
        bearish_score = sum(v for k, v in signals.items() if "bearish" in k or "below" in k or "breakout_down" in k)
        compression_score = signals.get("compression", 0)

        # Factor in OI direction
        if "bullish_breakout" in oi_direction:
            bullish_score += 0.5
        elif "bearish_breakout" in oi_direction:
            bearish_score += 0.5
        elif oi_direction == "bullish":
            bullish_score += 0.3
        elif oi_direction == "bearish":
            bearish_score += 0.3

        # Classify
        if "breakout_up" in signals and signals["breakout_up"] > 0.6:
            return ("breakout_up", min(1.0, signals["breakout_up"]))
        if "breakout_down" in signals and signals["breakout_down"] > 0.6:
            return ("breakout_down", min(1.0, signals["breakout_down"]))
        if compression_score > 0.5:
            return ("compression", compression_score)
        if bullish_score > bearish_score + 0.3:
            return ("trending_up", min(1.0, bullish_score))
        if bearish_score > bullish_score + 0.3:
            return ("trending_down", min(1.0, bearish_score))

        return ("range_bound", max(0, 1 - bullish_score - bearish_score))

    @staticmethod
    def _match_strategy(
        trend: str, vol: str, direction: str,
    ) -> tuple[str, float, str]:
        """Match against strategy matrix. Returns (strategy, confidence, reason)."""
        for (t_pat, v_pat, d_pat), strat, conf, reason in _STRATEGY_MATRIX:
            t_match = t_pat == "*" or t_pat == trend
            v_match = v_pat == "*" or v_pat == vol
            d_match = d_pat == "*" or d_pat == direction
            if t_match and v_match and d_match:
                return (strat, conf, reason)

        return ("no_trade", 0.3, "No matching strategy for current regime")

    @staticmethod
    def _find_alternatives(
        trend: str, vol: str, direction: str,
        exclude: str = "",
        max_alts: int = 3,
    ) -> list[dict[str, Any]]:
        """Find alternative strategies sorted by confidence."""
        alts = []
        for (t_pat, v_pat, d_pat), strat, conf, reason in _STRATEGY_MATRIX:
            if strat == exclude:
                continue
            t_match = t_pat == "*" or t_pat == trend
            v_match = v_pat == "*" or v_pat == vol
            d_match = d_pat == "*" or d_pat == direction
            # Allow partial matches (2 of 3 match)
            match_count = sum([t_match, v_match, d_match])
            if match_count >= 2 and strat not in [a["strategy"] for a in alts]:
                alts.append({
                    "strategy": strat,
                    "confidence": round(conf * (match_count / 3), 2),
                    "reason": reason,
                })
        alts.sort(key=lambda x: x["confidence"], reverse=True)
        return alts[:max_alts]
