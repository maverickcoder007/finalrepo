"""
Execution Quality Scorer
========================

Auto-scores every executed trade on four dimensions:
  1. Slippage score     — expected vs actual fill price
  2. Timing score       — entry relative to bar extremes (MAE/MFE)
  3. Spread efficiency  — fill vs bid-ask spread  
  4. Market impact      — estimated price impact of the order

Each dimension yields 0-100.  The composite score is a weighted average.

Scores are stored per trade in the journal store for later analytics.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger("execution_quality")


@dataclass
class ExecutionQualityScore:
    """Quality score for a single execution."""
    trade_id: str = ""
    instrument: str = ""
    timestamp: str = ""

    # Raw metrics
    expected_price: float = 0.0
    actual_price: float = 0.0
    quantity: int = 0
    direction: str = ""  # BUY / SELL

    # Bar context (the candle around execution time)
    bar_high: float = 0.0
    bar_low: float = 0.0
    bar_open: float = 0.0
    bar_close: float = 0.0

    # Bid-ask context
    best_bid: float = 0.0
    best_ask: float = 0.0

    # Scores (0-100, higher = better)
    slippage_score: float = 0.0
    timing_score: float = 0.0
    spread_efficiency_score: float = 0.0
    market_impact_score: float = 0.0
    composite_score: float = 0.0

    # Raw slippage in absolute and bps
    slippage_abs: float = 0.0
    slippage_bps: float = 0.0
    market_impact_bps: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExecutionQualityScorer:
    """
    Computes execution quality scores for each trade.

    Weights:
      slippage       35%
      timing         25%
      spread_eff     20%
      market_impact  20%
    """

    WEIGHTS = {
        "slippage": 0.35,
        "timing": 0.25,
        "spread_efficiency": 0.20,
        "market_impact": 0.20,
    }

    def __init__(self) -> None:
        self._scores: list[ExecutionQualityScore] = []
        # Rolling averages
        self._total_scored: int = 0
        self._sum_composite: float = 0.0

    # ── Public API ──────────────────────────────────────────

    def score_execution(
        self,
        trade_id: str,
        instrument: str,
        direction: str,
        quantity: int,
        expected_price: float,
        actual_price: float,
        bar_high: float = 0.0,
        bar_low: float = 0.0,
        bar_open: float = 0.0,
        bar_close: float = 0.0,
        best_bid: float = 0.0,
        best_ask: float = 0.0,
    ) -> ExecutionQualityScore:
        """
        Score a single execution on all four dimensions.

        Args:
            trade_id: Unique trade / order identifier.
            instrument: Tradingsymbol.
            direction: "BUY" or "SELL".
            quantity: Number of shares / lots.
            expected_price: Signal / expected fill price.
            actual_price: Actual fill price.
            bar_high/low/open/close: OHLC of the bar around execution.
            best_bid/best_ask: Top-of-book at order time (if available).

        Returns:
            ExecutionQualityScore with all four sub-scores and composite.
        """
        qs = ExecutionQualityScore(
            trade_id=trade_id,
            instrument=instrument,
            timestamp=datetime.now().isoformat(),
            expected_price=expected_price,
            actual_price=actual_price,
            quantity=quantity,
            direction=direction.upper(),
            bar_high=bar_high,
            bar_low=bar_low,
            bar_open=bar_open,
            bar_close=bar_close,
            best_bid=best_bid,
            best_ask=best_ask,
        )

        qs.slippage_score, qs.slippage_abs, qs.slippage_bps = self._score_slippage(
            expected_price, actual_price, direction
        )
        qs.timing_score = self._score_timing(
            actual_price, direction, bar_high, bar_low, bar_open, bar_close
        )
        qs.spread_efficiency_score = self._score_spread_efficiency(
            actual_price, direction, best_bid, best_ask
        )
        qs.market_impact_score, qs.market_impact_bps = self._score_market_impact(
            actual_price, quantity, bar_high, bar_low
        )
        qs.composite_score = (
            self.WEIGHTS["slippage"] * qs.slippage_score
            + self.WEIGHTS["timing"] * qs.timing_score
            + self.WEIGHTS["spread_efficiency"] * qs.spread_efficiency_score
            + self.WEIGHTS["market_impact"] * qs.market_impact_score
        )

        # Store
        self._scores.append(qs)
        if len(self._scores) > 5000:
            self._scores = self._scores[-5000:]
        self._total_scored += 1
        self._sum_composite += qs.composite_score

        logger.info(
            "execution_quality_scored",
            trade_id=trade_id,
            instrument=instrument,
            composite=round(qs.composite_score, 1),
            slippage=round(qs.slippage_score, 1),
            timing=round(qs.timing_score, 1),
        )
        return qs

    def get_recent_scores(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent execution quality scores."""
        return [s.to_dict() for s in self._scores[-limit:]]

    def get_summary(self) -> dict[str, Any]:
        """Aggregate execution quality summary."""
        if not self._scores:
            return {
                "total_scored": 0,
                "avg_composite": 0,
                "avg_slippage": 0,
                "avg_timing": 0,
                "avg_spread_efficiency": 0,
                "avg_market_impact": 0,
                "avg_slippage_bps": 0,
            }
        n = len(self._scores)
        return {
            "total_scored": self._total_scored,
            "avg_composite": round(sum(s.composite_score for s in self._scores) / n, 2),
            "avg_slippage": round(sum(s.slippage_score for s in self._scores) / n, 2),
            "avg_timing": round(sum(s.timing_score for s in self._scores) / n, 2),
            "avg_spread_efficiency": round(sum(s.spread_efficiency_score for s in self._scores) / n, 2),
            "avg_market_impact": round(sum(s.market_impact_score for s in self._scores) / n, 2),
            "avg_slippage_bps": round(sum(s.slippage_bps for s in self._scores) / n, 2),
            "recent_count": n,
        }

    # ── Scoring methods ─────────────────────────────────────

    @staticmethod
    def _score_slippage(
        expected: float, actual: float, direction: str
    ) -> tuple[float, float, float]:
        """
        Score slippage: 0 bps = 100, ≥50 bps = 0.

        For BUY: slippage = actual - expected (positive = bad).
        For SELL: slippage = expected - actual (positive = bad).
        """
        if expected <= 0:
            return (80.0, 0.0, 0.0)  # No expected price → assume decent

        if direction == "BUY":
            slip = actual - expected
        else:
            slip = expected - actual

        slip_bps = (slip / expected) * 10000
        # Score: 100 at 0 bps, decays linearly to 0 at 50 bps
        score = max(0, min(100, 100 - (abs(slip_bps) * 2)))
        return (round(score, 2), round(slip, 4), round(slip_bps, 2))

    @staticmethod
    def _score_timing(
        actual: float, direction: str,
        high: float, low: float, _open: float, close: float,
    ) -> float:
        """
        Score timing relative to bar range.

        BUY: best at bar low → 100, worst at bar high → 0.
        SELL: best at bar high → 100, worst at bar low → 0.
        """
        if high <= 0 or low <= 0 or high == low:
            return 50.0  # No bar context

        bar_range = high - low
        if direction == "BUY":
            # Lower in bar range = better for buy
            position_in_range = (actual - low) / bar_range
            score = (1 - position_in_range) * 100
        else:
            # Higher in bar range = better for sell
            position_in_range = (actual - low) / bar_range
            score = position_in_range * 100

        return round(max(0, min(100, score)), 2)

    @staticmethod
    def _score_spread_efficiency(
        actual: float, direction: str,
        best_bid: float, best_ask: float,
    ) -> float:
        """
        Score fill relative to bid-ask spread.

        BUY at ask = 50 (mid = 100, below mid = 100).
        SELL at bid = 50 (mid = 100, above mid = 100).
        """
        if best_bid <= 0 or best_ask <= 0 or best_ask <= best_bid:
            return 70.0  # No spread data → assume OK

        mid = (best_bid + best_ask) / 2
        spread = best_ask - best_bid

        if direction == "BUY":
            # Better if fill is closer to bid
            if actual <= best_bid:
                return 100.0
            elif actual >= best_ask:
                return 50.0
            else:
                return round(50 + 50 * (best_ask - actual) / spread, 2)
        else:
            # Better if fill is closer to ask
            if actual >= best_ask:
                return 100.0
            elif actual <= best_bid:
                return 50.0
            else:
                return round(50 + 50 * (actual - best_bid) / spread, 2)

    @staticmethod
    def _score_market_impact(
        actual: float, quantity: int,
        bar_high: float, bar_low: float,
    ) -> tuple[float, float]:
        """
        Estimate market impact based on order size vs bar range.

        Larger orders relative to bar range → higher impact → lower score.
        """
        if bar_high <= 0 or bar_low <= 0 or bar_high == bar_low or actual <= 0:
            return (75.0, 0.0)

        bar_range = bar_high - bar_low
        notional = abs(quantity * actual)

        # Rough impact: assume 1 lot ~ 0.5 bps impact, scale with size
        # This is a heuristic; real impact requires order book depth
        impact_bps = min(50, abs(quantity) * 0.5)
        score = max(0, 100 - impact_bps * 2)

        return (round(score, 2), round(impact_bps, 2))
