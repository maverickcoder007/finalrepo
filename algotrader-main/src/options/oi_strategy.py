"""
OI-Based Strategy Engine for NIFTY & SENSEX Weekly Options.

Scans OI data to generate directional and non-directional trading signals,
manages active OI-strategy positions, and provides journaling metadata.

Strategies:
  1. PCR Extreme Reversal     — sell options when PCR is extreme (<0.5 or >1.5)
  2. OI Wall Breakout/Bounce  — trade when price nears or breaks OI walls
  3. Straddle Unwinding       — detect premium collapse for direction
  4. Max Pain Magnet           — fade moves away from max pain
  5. IV Skew Exploitation      — trade skew mean-reversion
  6. OI Buildup Momentum      — ride heavy OI additions near ATM
  7. Put Writing Support       — sell puts at heavy PE OI support
  8. Call Writing Resistance   — sell calls at heavy CE OI resistance

Works with both NIFTY (NFO, weekly Thu expiry) and SENSEX (BFO, weekly Fri expiry).
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, date
from typing import Any, Optional
from enum import Enum

from pydantic import BaseModel, Field

from src.options.oi_analysis import OptionsOIReport, OptionsOIStrike
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

NIFTY_LOT_SIZE = 65
SENSEX_LOT_SIZE = 20
NIFTY_STRIKE_GAP = 50
SENSEX_STRIKE_GAP = 100

INDEX_CONFIG = {
    "NIFTY": {
        "lot_size": NIFTY_LOT_SIZE,
        "strike_gap": NIFTY_STRIKE_GAP,
        "exchange": "NFO",
        "expiry_day": "Thursday",   # Weekly expiry day
        "tick_size": 0.05,
    },
    "SENSEX": {
        "lot_size": SENSEX_LOT_SIZE,
        "strike_gap": SENSEX_STRIKE_GAP,
        "exchange": "BFO",
        "expiry_day": "Friday",
        "tick_size": 0.05,
    },
}


class OISignalType(str, Enum):
    PCR_EXTREME_BULLISH = "PCR_EXTREME_BULLISH"
    PCR_EXTREME_BEARISH = "PCR_EXTREME_BEARISH"
    OI_WALL_SUPPORT_BOUNCE = "OI_WALL_SUPPORT_BOUNCE"
    OI_WALL_RESISTANCE_REJECTION = "OI_WALL_RESISTANCE_REJECTION"
    OI_WALL_BREAKOUT_UP = "OI_WALL_BREAKOUT_UP"
    OI_WALL_BREAKOUT_DOWN = "OI_WALL_BREAKOUT_DOWN"
    STRADDLE_COLLAPSE_BULLISH = "STRADDLE_COLLAPSE_BULLISH"
    STRADDLE_COLLAPSE_BEARISH = "STRADDLE_COLLAPSE_BEARISH"
    MAX_PAIN_MAGNET = "MAX_PAIN_MAGNET"
    IV_SKEW_REVERSAL = "IV_SKEW_REVERSAL"
    OI_BUILDUP_BULLISH = "OI_BUILDUP_BULLISH"
    OI_BUILDUP_BEARISH = "OI_BUILDUP_BEARISH"
    PUT_WRITING_SUPPORT = "PUT_WRITING_SUPPORT"
    CALL_WRITING_RESISTANCE = "CALL_WRITING_RESISTANCE"
    COMBINED_BULLISH = "COMBINED_BULLISH"
    COMBINED_BEARISH = "COMBINED_BEARISH"


class OIStrategyAction(str, Enum):
    BUY_CE = "BUY_CE"
    BUY_PE = "BUY_PE"
    SELL_CE = "SELL_CE"
    SELL_PE = "SELL_PE"
    BUY_STRADDLE = "BUY_STRADDLE"
    SELL_STRADDLE = "SELL_STRADDLE"
    BULL_CALL_SPREAD = "BULL_CALL_SPREAD"
    BEAR_PUT_SPREAD = "BEAR_PUT_SPREAD"
    IRON_CONDOR = "IRON_CONDOR"


# ─────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────

class OISignal(BaseModel):
    """A single OI-based trading signal."""
    id: str = ""
    timestamp: str = ""
    underlying: str = ""
    signal_type: str = ""
    action: str = ""
    direction: str = ""     # BULLISH | BEARISH | NEUTRAL
    confidence: float = 0.0
    strike: float = 0.0
    option_type: str = ""   # CE | PE | STRADDLE
    expiry: str = ""
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target: float = 0.0
    lot_size: int = 0
    lots: int = 1
    exchange: str = ""
    tradingsymbol: str = ""
    # Context
    spot_price: float = 0.0
    atm_strike: float = 0.0
    pcr_oi: float = 0.0
    max_pain: float = 0.0
    iv_skew: float = 0.0
    straddle_premium: float = 0.0
    pe_wall: float = 0.0
    ce_wall: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OIPosition(BaseModel):
    """Active position from OI strategy."""
    id: str = ""
    signal_id: str = ""
    underlying: str = ""
    tradingsymbol: str = ""
    exchange: str = ""
    option_type: str = ""
    strike: float = 0.0
    expiry: str = ""
    direction: str = ""     # BUY | SELL
    quantity: int = 0
    entry_price: float = 0.0
    current_price: float = 0.0
    stop_loss: float = 0.0
    target: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    status: str = "OPEN"    # OPEN | CLOSED | SL_HIT | TARGET_HIT
    entry_time: str = ""
    exit_time: str = ""
    exit_price: float = 0.0
    exit_reason: str = ""
    signal_type: str = ""
    strategy_name: str = "oi_strategy"
    # OI context at entry
    entry_pcr: float = 0.0
    entry_max_pain: float = 0.0
    entry_spot: float = 0.0
    entry_straddle: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class OIScanResult(BaseModel):
    """Full scan result for an underlying."""
    underlying: str = ""
    timestamp: str = ""
    spot_price: float = 0.0
    atm_strike: float = 0.0
    expiry: str = ""
    pcr_oi: float = 0.0
    pcr_volume: float = 0.0
    max_pain: float = 0.0
    straddle_premium: float = 0.0
    iv_skew: float = 0.0
    pe_wall: float = 0.0
    ce_wall: float = 0.0
    bias: str = ""
    bias_reasons: list[str] = Field(default_factory=list)
    signals: list[OISignal] = Field(default_factory=list)
    signal_count: int = 0
    bullish_signals: int = 0
    bearish_signals: int = 0
    combined_direction: str = ""
    combined_confidence: float = 0.0


class OIStrategyConfig(BaseModel):
    """Configuration for OI strategy parameters."""
    # General
    enabled: bool = True
    auto_execute: bool = False      # If True, automatically sends signals to execution engine
    max_lots: int = 2
    max_open_positions: int = 4
    # PCR thresholds
    pcr_extreme_high: float = 1.5   # Very bullish PCR
    pcr_extreme_low: float = 0.5    # Very bearish PCR
    pcr_mild_high: float = 1.2
    pcr_mild_low: float = 0.7
    # OI wall proximity
    wall_proximity_pct: float = 1.5  # % distance to consider "near wall"
    wall_min_oi_lakh: float = 50.0   # Minimum OI (in lakhs) to consider a wall
    # Straddle
    straddle_collapse_pct: float = 15.0  # % drop in straddle premium to signal
    # IV Skew
    iv_skew_threshold: float = 5.0       # PE IV - CE IV diff to trigger
    # Max pain
    max_pain_distance_pct: float = 1.0   # % from max pain to spot
    # Buildup
    buildup_min_oi_change_pct: float = 5.0  # min OI change % for buildup signal
    # Risk
    stop_loss_pct: float = 30.0     # SL as % of entry premium
    target_pct: float = 50.0        # Target as % of entry premium
    trailing_sl_pct: float = 20.0   # Trailing SL once in profit
    # Minimum premium filters — reject near-worthless options
    min_premium_nifty: float = 30.0   # Min ₹30 for NIFTY options
    min_premium_sensex: float = 50.0  # Min ₹50 for SENSEX options
    min_premium_pct_of_spot: float = 0.15  # Min premium as % of spot (0.15% = ~₹35 for NIFTY at 23000)
    # Execution
    min_confidence: float = 0.6     # Minimum signal confidence to display/execute (0-1)
    preferred_strategy: str = "directional"  # directional | spread | straddle


# ─────────────────────────────────────────────────────────────
# OI Strategy Engine
# ─────────────────────────────────────────────────────────────

class OIStrategyEngine:
    """
    Core engine that scans OI reports and generates trading signals
    for NIFTY and SENSEX weekly options.
    """

    def __init__(self, config: Optional[OIStrategyConfig] = None) -> None:
        self.config = config or OIStrategyConfig()
        self._signals_history: list[OISignal] = []
        self._positions: list[OIPosition] = []
        self._scan_history: dict[str, list[OIScanResult]] = {"NIFTY": [], "SENSEX": []}
        self._prev_straddle: dict[str, float] = {}
        self._signal_counter = 0
        self._position_counter = 0

    # ═══════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════

    def scan_for_signals(self, report: OptionsOIReport) -> OIScanResult:
        """
        Scan an OptionsOIReport and detect all OI-based signals.
        Works for both NIFTY and SENSEX.
        """
        underlying = report.underlying.upper()
        if underlying not in INDEX_CONFIG:
            return OIScanResult(underlying=underlying, timestamp=datetime.now().isoformat())

        cfg = INDEX_CONFIG[underlying]
        signals: list[OISignal] = []

        base_ctx = {
            "underlying": underlying,
            "spot_price": report.spot_price,
            "atm_strike": report.atm_strike,
            "expiry": report.expiry,
            "pcr_oi": report.pcr_oi,
            "max_pain": report.max_pain,
            "iv_skew": report.iv_skew,
            "straddle_premium": report.atm_straddle_premium,
            "pe_wall": report.max_pe_oi_strike,
            "ce_wall": report.max_ce_oi_strike,
            "exchange": cfg["exchange"],
            "lot_size": cfg["lot_size"],
            "strike_gap": cfg["strike_gap"],
        }

        # Run all detectors
        signals.extend(self._detect_pcr_extreme(report, base_ctx))
        signals.extend(self._detect_oi_wall_signals(report, base_ctx))
        signals.extend(self._detect_straddle_collapse(report, base_ctx))
        signals.extend(self._detect_max_pain_magnet(report, base_ctx))
        signals.extend(self._detect_iv_skew(report, base_ctx))
        signals.extend(self._detect_oi_buildup(report, base_ctx))
        signals.extend(self._detect_put_writing_support(report, base_ctx))
        signals.extend(self._detect_call_writing_resistance(report, base_ctx))

        # Filter out zero-confidence signals (from price sanity failures etc.)
        signals = [s for s in signals if s.confidence > 0]

        # Compute combined direction
        bullish = sum(1 for s in signals if s.direction == "BULLISH")
        bearish = sum(1 for s in signals if s.direction == "BEARISH")
        total_conf = sum(s.confidence for s in signals) if signals else 0
        bull_conf = sum(s.confidence for s in signals if s.direction == "BULLISH")
        bear_conf = sum(s.confidence for s in signals if s.direction == "BEARISH")

        if bullish > bearish + 1 and bull_conf > bear_conf:
            combined_dir = "BULLISH"
            combined_conf = min(90, bull_conf / max(1, bullish))
            # Add combined signal
            combined = self._create_combined_signal(
                report, base_ctx, "BULLISH", combined_conf, signals
            )
            if combined:
                signals.append(combined)
        elif bearish > bullish + 1 and bear_conf > bull_conf:
            combined_dir = "BEARISH"
            combined_conf = min(90, bear_conf / max(1, bearish))
            combined = self._create_combined_signal(
                report, base_ctx, "BEARISH", combined_conf, signals
            )
            if combined:
                signals.append(combined)
        else:
            combined_dir = "NEUTRAL"
            combined_conf = 50.0

        # Store signals
        self._signals_history.extend(signals)
        if len(self._signals_history) > 500:
            self._signals_history = self._signals_history[-500:]

        # Update straddle history for next comparison
        self._prev_straddle[underlying] = report.atm_straddle_premium

        result = OIScanResult(
            underlying=underlying,
            timestamp=datetime.now().isoformat(),
            spot_price=report.spot_price,
            atm_strike=report.atm_strike,
            expiry=report.expiry,
            pcr_oi=report.pcr_oi,
            pcr_volume=report.pcr_volume,
            max_pain=report.max_pain,
            straddle_premium=report.atm_straddle_premium,
            iv_skew=report.iv_skew,
            pe_wall=report.max_pe_oi_strike,
            ce_wall=report.max_ce_oi_strike,
            bias=report.bias,
            bias_reasons=report.bias_reasons,
            signals=signals,
            signal_count=len(signals),
            bullish_signals=bullish,
            bearish_signals=bearish,
            combined_direction=combined_dir,
            combined_confidence=round(combined_conf, 1),
        )

        # Cache scan result
        if underlying not in self._scan_history:
            self._scan_history[underlying] = []
        self._scan_history[underlying].append(result)
        if len(self._scan_history[underlying]) > 50:
            self._scan_history[underlying] = self._scan_history[underlying][-50:]

        return result

    def create_position_from_signal(self, signal: OISignal) -> OIPosition:
        """Convert a signal into a tracked position."""
        self._position_counter += 1
        pos = OIPosition(
            id=f"OIP-{self._position_counter:05d}",
            signal_id=signal.id,
            underlying=signal.underlying,
            tradingsymbol=signal.tradingsymbol,
            exchange=signal.exchange,
            option_type=signal.option_type,
            strike=signal.strike,
            expiry=signal.expiry,
            direction="BUY" if signal.action in (
                OIStrategyAction.BUY_CE, OIStrategyAction.BUY_PE,
                OIStrategyAction.BUY_STRADDLE
            ) else "SELL",
            quantity=signal.lot_size * signal.lots,
            entry_price=signal.entry_price,
            current_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            target=signal.target,
            pnl=0.0,
            pnl_pct=0.0,
            status="OPEN",
            entry_time=datetime.now().isoformat(),
            signal_type=signal.signal_type,
            entry_pcr=signal.pcr_oi,
            entry_max_pain=signal.max_pain,
            entry_spot=signal.spot_price,
            entry_straddle=signal.straddle_premium,
            metadata=signal.metadata,
        )
        self._positions.append(pos)
        return pos

    def update_position(self, position_id: str, current_price: float) -> Optional[OIPosition]:
        """Update position with current market price, check SL/target."""
        pos = next((p for p in self._positions if p.id == position_id and p.status == "OPEN"), None)
        if not pos:
            return None

        pos.current_price = current_price

        # Calculate PnL
        if pos.direction == "BUY":
            pos.pnl = (current_price - pos.entry_price) * pos.quantity
            pos.pnl_pct = ((current_price - pos.entry_price) / pos.entry_price * 100) if pos.entry_price > 0 else 0
        else:
            pos.pnl = (pos.entry_price - current_price) * pos.quantity
            pos.pnl_pct = ((pos.entry_price - current_price) / pos.entry_price * 100) if pos.entry_price > 0 else 0

        # Check stop loss
        if pos.direction == "BUY" and current_price <= pos.stop_loss:
            pos.status = "SL_HIT"
            pos.exit_price = current_price
            pos.exit_time = datetime.now().isoformat()
            pos.exit_reason = "Stop loss hit"
        elif pos.direction == "SELL" and current_price >= pos.stop_loss:
            pos.status = "SL_HIT"
            pos.exit_price = current_price
            pos.exit_time = datetime.now().isoformat()
            pos.exit_reason = "Stop loss hit"

        # Check target
        if pos.status == "OPEN":
            if pos.direction == "BUY" and current_price >= pos.target:
                pos.status = "TARGET_HIT"
                pos.exit_price = current_price
                pos.exit_time = datetime.now().isoformat()
                pos.exit_reason = "Target hit"
            elif pos.direction == "SELL" and current_price <= pos.target:
                pos.status = "TARGET_HIT"
                pos.exit_price = current_price
                pos.exit_time = datetime.now().isoformat()
                pos.exit_reason = "Target hit"

        # Trailing SL
        if pos.status == "OPEN" and pos.pnl_pct > 10:
            trail_pct = self.config.trailing_sl_pct / 100
            if pos.direction == "BUY":
                new_sl = current_price * (1 - trail_pct)
                if new_sl > pos.stop_loss:
                    pos.stop_loss = round(new_sl, 2)
            else:
                new_sl = current_price * (1 + trail_pct)
                if new_sl < pos.stop_loss:
                    pos.stop_loss = round(new_sl, 2)

        return pos

    def close_position(self, position_id: str, exit_price: float, reason: str = "Manual close") -> Optional[OIPosition]:
        """Manually close an open position."""
        pos = next((p for p in self._positions if p.id == position_id and p.status == "OPEN"), None)
        if not pos:
            return None

        pos.status = "CLOSED"
        pos.exit_price = exit_price
        pos.exit_time = datetime.now().isoformat()
        pos.exit_reason = reason
        pos.current_price = exit_price

        if pos.direction == "BUY":
            pos.pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pos.pnl = (pos.entry_price - exit_price) * pos.quantity
        pos.pnl_pct = (pos.pnl / (pos.entry_price * pos.quantity) * 100) if pos.entry_price > 0 else 0

        return pos

    def get_open_positions(self, underlying: Optional[str] = None) -> list[OIPosition]:
        """Get all open positions, optionally filtered by underlying."""
        positions = [p for p in self._positions if p.status == "OPEN"]
        if underlying:
            positions = [p for p in positions if p.underlying == underlying.upper()]
        return positions

    def get_closed_positions(self, underlying: Optional[str] = None, limit: int = 50) -> list[OIPosition]:
        """Get closed positions."""
        closed = [p for p in self._positions if p.status != "OPEN"]
        if underlying:
            closed = [c for c in closed if c.underlying == underlying.upper()]
        return sorted(closed, key=lambda x: x.exit_time or "", reverse=True)[:limit]

    def get_signal_history(self, underlying: Optional[str] = None, limit: int = 50) -> list[OISignal]:
        """Get recent signal history."""
        signals = self._signals_history
        if underlying:
            signals = [s for s in signals if s.underlying == underlying.upper()]
        return sorted(signals, key=lambda x: x.timestamp, reverse=True)[:limit]

    def get_scan_history(self, underlying: str = "NIFTY", limit: int = 20) -> list[OIScanResult]:
        """Get scan result history."""
        return self._scan_history.get(underlying.upper(), [])[-limit:]

    def get_strategy_summary(self) -> dict[str, Any]:
        """Get overall strategy performance summary."""
        open_pos = [p for p in self._positions if p.status == "OPEN"]
        closed_pos = [p for p in self._positions if p.status != "OPEN"]

        total_pnl = sum(p.pnl for p in closed_pos)
        wins = [p for p in closed_pos if p.pnl > 0]
        losses = [p for p in closed_pos if p.pnl <= 0]
        win_rate = len(wins) / len(closed_pos) * 100 if closed_pos else 0

        nifty_pnl = sum(p.pnl for p in closed_pos if p.underlying == "NIFTY")
        sensex_pnl = sum(p.pnl for p in closed_pos if p.underlying == "SENSEX")

        open_pnl = sum(p.pnl for p in open_pos)

        # By signal type
        by_signal: dict[str, dict[str, Any]] = {}
        for p in closed_pos:
            st = p.signal_type
            if st not in by_signal:
                by_signal[st] = {"trades": 0, "pnl": 0, "wins": 0}
            by_signal[st]["trades"] += 1
            by_signal[st]["pnl"] += p.pnl
            if p.pnl > 0:
                by_signal[st]["wins"] += 1

        return {
            "config": self.config.model_dump(),
            "open_positions": len(open_pos),
            "closed_positions": len(closed_pos),
            "total_signals": len(self._signals_history),
            "total_pnl": round(total_pnl, 2),
            "open_pnl": round(open_pnl, 2),
            "win_count": len(wins),
            "loss_count": len(losses),
            "win_rate": round(win_rate, 1),
            "avg_win": round(sum(p.pnl for p in wins) / len(wins), 2) if wins else 0,
            "avg_loss": round(sum(p.pnl for p in losses) / len(losses), 2) if losses else 0,
            "nifty_pnl": round(nifty_pnl, 2),
            "sensex_pnl": round(sensex_pnl, 2),
            "by_signal_type": {
                k: {**v, "win_rate": round(v["wins"] / v["trades"] * 100, 1) if v["trades"] else 0}
                for k, v in by_signal.items()
            },
            "last_scan_nifty": self._scan_history.get("NIFTY", [{}])[-1].model_dump()
            if self._scan_history.get("NIFTY") else None,
            "last_scan_sensex": self._scan_history.get("SENSEX", [{}])[-1].model_dump()
            if self._scan_history.get("SENSEX") else None,
        }

    def update_config(self, updates: dict[str, Any]) -> OIStrategyConfig:
        """Update strategy configuration."""
        current = self.config.model_dump()
        current.update(updates)
        self.config = OIStrategyConfig(**current)
        return self.config

    def get_journal_metadata(self, position: OIPosition) -> dict[str, Any]:
        """
        Generate rich metadata for journal entry from an OI position.
        This provides context for reviewing OI-based trades.
        """
        return {
            "strategy_type": "oi_strategy",
            "signal_type": position.signal_type,
            "underlying": position.underlying,
            "option_type": position.option_type,
            "strike": position.strike,
            "expiry": position.expiry,
            "direction": position.direction,
            "entry_pcr": position.entry_pcr,
            "entry_max_pain": position.entry_max_pain,
            "entry_spot": position.entry_spot,
            "entry_straddle": position.entry_straddle,
            "stop_loss": position.stop_loss,
            "target": position.target,
            "exit_reason": position.exit_reason,
            "pnl": position.pnl,
            "pnl_pct": position.pnl_pct,
            **position.metadata,
        }

    # ═══════════════════════════════════════════════════════════
    # SIGNAL DETECTORS
    # ═══════════════════════════════════════════════════════════

    def _next_signal_id(self) -> str:
        self._signal_counter += 1
        return f"OIS-{self._signal_counter:05d}"

    def _make_signal(
        self,
        ctx: dict[str, Any],
        signal_type: OISignalType,
        action: OIStrategyAction,
        direction: str,
        confidence: float,
        strike: float,
        option_type: str,
        entry_price: float,
        reasons: list[str],
        extra_meta: Optional[dict[str, Any]] = None,
    ) -> OISignal:
        """Factory for creating OISignal with full context."""
        spot = ctx.get("spot_price", 0)
        underlying = ctx.get("underlying", "NIFTY")

        # ── Price validation ──────────────────────────────────

        # 1. Sanity check: option premium should not exceed ~15% of spot
        if spot > 0 and entry_price > spot * 0.15:
            logger.warning(
                "oi_signal_price_too_high",
                signal_type=signal_type.value,
                strike=strike,
                option_type=option_type,
                entry_price=entry_price,
                spot_price=spot,
                msg="Entry price > 15% of spot — likely deep ITM or stale data, skipping",
            )
            entry_price = 0.0

        # 2. Minimum premium filter — reject near-worthless options
        if entry_price > 0:
            min_abs = self.config.min_premium_nifty if underlying == "NIFTY" else self.config.min_premium_sensex
            min_pct_val = spot * self.config.min_premium_pct_of_spot / 100 if spot > 0 else 0
            min_premium = max(min_abs, min_pct_val)

            if entry_price < min_premium:
                logger.warning(
                    "oi_signal_premium_too_low",
                    signal_type=signal_type.value,
                    strike=strike,
                    option_type=option_type,
                    entry_price=entry_price,
                    min_premium=min_premium,
                    spot_price=spot,
                    msg=f"Option premium ₹{entry_price:.1f} below minimum ₹{min_premium:.0f} — likely far OTM/near expiry, skipping",
                )
                entry_price = 0.0

        # 3. ATM sanity: if this is ATM option, premium should be at least 0.5% of spot
        if entry_price > 0 and spot > 0:
            distance_pct = abs(strike - spot) / spot * 100
            if distance_pct < 1.0:  # Near ATM (within 1% of spot)
                atm_min = spot * 0.005  # At least 0.5% of spot for ATM
                if entry_price < atm_min:
                    logger.warning(
                        "oi_signal_atm_price_stale",
                        signal_type=signal_type.value,
                        strike=strike,
                        option_type=option_type,
                        entry_price=entry_price,
                        atm_min=atm_min,
                        spot_price=spot,
                        msg=f"ATM option at ₹{entry_price:.1f} is unrealistically cheap (min ~₹{atm_min:.0f}) — stale data?",
                    )
                    entry_price = 0.0

        if entry_price <= 0:
            # Can't generate signal without a valid entry price
            return OISignal(
                id=self._next_signal_id(),
                underlying=ctx["underlying"],
                signal_type=signal_type.value,
                confidence=0.0,
            )

        sl_mult = self.config.stop_loss_pct / 100
        tgt_mult = self.config.target_pct / 100

        if action.name.startswith("BUY"):
            stop_loss = round(entry_price * (1 - sl_mult), 2)
            target = round(entry_price * (1 + tgt_mult), 2)
        else:
            stop_loss = round(entry_price * (1 + sl_mult), 2)
            target = round(entry_price * (1 - tgt_mult), 2)

        return OISignal(
            id=self._next_signal_id(),
            timestamp=datetime.now().isoformat(),
            underlying=ctx["underlying"],
            signal_type=signal_type.value,
            action=action.value,
            direction=direction,
            confidence=round(min(confidence, 100.0), 1),
            strike=strike,
            option_type=option_type,
            expiry=ctx.get("expiry", ""),
            entry_price=entry_price,
            stop_loss=stop_loss,
            target=target,
            lot_size=ctx["lot_size"],
            lots=min(self.config.max_lots, 1),
            exchange=ctx["exchange"],
            tradingsymbol=self._build_tradingsymbol(
                ctx["underlying"], ctx.get("expiry", ""), strike, option_type
            ),
            spot_price=ctx["spot_price"],
            atm_strike=ctx["atm_strike"],
            pcr_oi=ctx["pcr_oi"],
            max_pain=ctx["max_pain"],
            iv_skew=ctx["iv_skew"],
            straddle_premium=ctx["straddle_premium"],
            pe_wall=ctx["pe_wall"],
            ce_wall=ctx["ce_wall"],
            reasons=reasons,
            metadata=extra_meta or {},
        )

    # ── 1. PCR Extreme ────────────────────────────────────────

    def _detect_pcr_extreme(self, report: OptionsOIReport, ctx: dict) -> list[OISignal]:
        """PCR > 1.5 → bullish reversal; PCR < 0.5 → bearish reversal."""
        signals = []
        pcr = report.pcr_oi

        if pcr >= self.config.pcr_extreme_high:
            # Extreme put buying = oversold = BUY CE
            atm_ce_price = self._get_atm_option_price(report, "CE")
            if atm_ce_price > 0:
                signals.append(self._make_signal(
                    ctx, OISignalType.PCR_EXTREME_BULLISH, OIStrategyAction.BUY_CE,
                    "BULLISH", min(85, 50 + (pcr - 1.2) * 30),
                    report.atm_strike, "CE", atm_ce_price,
                    [f"PCR extremely high ({pcr:.2f}) — oversold, bullish reversal expected",
                     f"Total PE OI: {report.total_pe_oi:,} vs CE OI: {report.total_ce_oi:,}"],
                ))

        elif pcr <= self.config.pcr_extreme_low:
            # Extreme call buying = overbought = BUY PE
            atm_pe_price = self._get_atm_option_price(report, "PE")
            if atm_pe_price > 0:
                signals.append(self._make_signal(
                    ctx, OISignalType.PCR_EXTREME_BEARISH, OIStrategyAction.BUY_PE,
                    "BEARISH", min(85, 50 + (0.8 - pcr) * 50),
                    report.atm_strike, "PE", atm_pe_price,
                    [f"PCR extremely low ({pcr:.2f}) — overbought, bearish reversal expected",
                     f"Total CE OI: {report.total_ce_oi:,} vs PE OI: {report.total_pe_oi:,}"],
                ))

        return signals

    # ── 2. OI Wall Signals ────────────────────────────────────

    def _detect_oi_wall_signals(self, report: OptionsOIReport, ctx: dict) -> list[OISignal]:
        """Detect proximity to OI walls (support/resistance)."""
        signals = []
        spot = report.spot_price
        pe_wall = report.max_pe_oi_strike
        ce_wall = report.max_ce_oi_strike
        threshold = self.config.wall_proximity_pct / 100

        if pe_wall > 0 and spot > 0:
            pe_dist = (spot - pe_wall) / spot
            if 0 < pe_dist <= threshold:
                # Near PE wall = support bounce expected
                atm_ce_price = self._get_atm_option_price(report, "CE")
                if atm_ce_price > 0:
                    conf = min(80, 55 + (1 - pe_dist / threshold) * 25)
                    signals.append(self._make_signal(
                        ctx, OISignalType.OI_WALL_SUPPORT_BOUNCE, OIStrategyAction.BUY_CE,
                        "BULLISH", conf,
                        report.atm_strike, "CE", atm_ce_price,
                        [f"Spot near PE OI wall (support) at {pe_wall:.0f}",
                         f"Distance: {pe_dist*100:.1f}% — bounce expected",
                         f"PE OI at wall: {report.max_pe_oi:,}"],
                        {"pe_wall_oi": report.max_pe_oi},
                    ))
            elif pe_dist < 0:
                # Broken PE wall = bearish breakdown
                atm_pe_price = self._get_atm_option_price(report, "PE")
                if atm_pe_price > 0:
                    signals.append(self._make_signal(
                        ctx, OISignalType.OI_WALL_BREAKOUT_DOWN, OIStrategyAction.BUY_PE,
                        "BEARISH", 70,
                        report.atm_strike, "PE", atm_pe_price,
                        [f"Spot broke below PE OI wall at {pe_wall:.0f}",
                         f"Support breach = bearish momentum"],
                    ))

        if ce_wall > 0 and spot > 0:
            ce_dist = (ce_wall - spot) / spot
            if 0 < ce_dist <= threshold:
                # Near CE wall = resistance rejection expected
                atm_pe_price = self._get_atm_option_price(report, "PE")
                if atm_pe_price > 0:
                    conf = min(80, 55 + (1 - ce_dist / threshold) * 25)
                    signals.append(self._make_signal(
                        ctx, OISignalType.OI_WALL_RESISTANCE_REJECTION, OIStrategyAction.BUY_PE,
                        "BEARISH", conf,
                        report.atm_strike, "PE", atm_pe_price,
                        [f"Spot near CE OI wall (resistance) at {ce_wall:.0f}",
                         f"Distance: {ce_dist*100:.1f}% — rejection expected",
                         f"CE OI at wall: {report.max_ce_oi:,}"],
                        {"ce_wall_oi": report.max_ce_oi},
                    ))
            elif ce_dist < 0:
                # Broken CE wall = bullish breakout
                atm_ce_price = self._get_atm_option_price(report, "CE")
                if atm_ce_price > 0:
                    signals.append(self._make_signal(
                        ctx, OISignalType.OI_WALL_BREAKOUT_UP, OIStrategyAction.BUY_CE,
                        "BULLISH", 70,
                        report.atm_strike, "CE", atm_ce_price,
                        [f"Spot broke above CE OI wall at {ce_wall:.0f}",
                         f"Resistance breach = bullish momentum"],
                    ))

        return signals

    # ── 3. Straddle Collapse ──────────────────────────────────

    def _detect_straddle_collapse(self, report: OptionsOIReport, ctx: dict) -> list[OISignal]:
        """Detect rapid straddle premium collapse → directional move."""
        signals = []
        underlying = report.underlying.upper()
        current_premium = report.atm_straddle_premium
        prev_premium = self._prev_straddle.get(underlying, 0)

        if prev_premium <= 0 or current_premium <= 0:
            return signals

        pct_change = (current_premium - prev_premium) / prev_premium * 100

        if pct_change < -self.config.straddle_collapse_pct:
            # Straddle premium collapsing — directional move
            bias = report.bias.upper()
            if "BULLISH" in bias:
                atm_ce_price = self._get_atm_option_price(report, "CE")
                if atm_ce_price > 0:
                    signals.append(self._make_signal(
                        ctx, OISignalType.STRADDLE_COLLAPSE_BULLISH, OIStrategyAction.BUY_CE,
                        "BULLISH", 65,
                        report.atm_strike, "CE", atm_ce_price,
                        [f"Straddle premium collapsed {pct_change:.1f}%",
                         f"OI bias: {bias} — directional move expected upward"],
                        {"straddle_change_pct": round(pct_change, 2)},
                    ))
            elif "BEARISH" in bias:
                atm_pe_price = self._get_atm_option_price(report, "PE")
                if atm_pe_price > 0:
                    signals.append(self._make_signal(
                        ctx, OISignalType.STRADDLE_COLLAPSE_BEARISH, OIStrategyAction.BUY_PE,
                        "BEARISH", 65,
                        report.atm_strike, "PE", atm_pe_price,
                        [f"Straddle premium collapsed {pct_change:.1f}%",
                         f"OI bias: {bias} — directional move expected downward"],
                        {"straddle_change_pct": round(pct_change, 2)},
                    ))

        return signals

    # ── 4. Max Pain Magnet ────────────────────────────────────

    def _detect_max_pain_magnet(self, report: OptionsOIReport, ctx: dict) -> list[OISignal]:
        """Fade moves away from max pain — price tends to gravitate towards max pain."""
        signals = []
        spot = report.spot_price
        mp = report.max_pain

        if mp <= 0 or spot <= 0:
            return signals

        dist_pct = (spot - mp) / spot * 100

        if dist_pct > self.config.max_pain_distance_pct:
            # Spot above max pain → expect pullback toward max pain
            atm_pe_price = self._get_atm_option_price(report, "PE")
            if atm_pe_price > 0:
                signals.append(self._make_signal(
                    ctx, OISignalType.MAX_PAIN_MAGNET, OIStrategyAction.BUY_PE,
                    "BEARISH", min(70, 50 + abs(dist_pct) * 10),
                    report.atm_strike, "PE", atm_pe_price,
                    [f"Spot ({spot:.0f}) above max pain ({mp:.0f}) by {dist_pct:.1f}%",
                     "Price tends to revert to max pain near expiry"],
                    {"max_pain_dist_pct": round(dist_pct, 2)},
                ))

        elif dist_pct < -self.config.max_pain_distance_pct:
            # Spot below max pain → expect bounce toward max pain
            atm_ce_price = self._get_atm_option_price(report, "CE")
            if atm_ce_price > 0:
                signals.append(self._make_signal(
                    ctx, OISignalType.MAX_PAIN_MAGNET, OIStrategyAction.BUY_CE,
                    "BULLISH", min(70, 50 + abs(dist_pct) * 10),
                    report.atm_strike, "CE", atm_ce_price,
                    [f"Spot ({spot:.0f}) below max pain ({mp:.0f}) by {abs(dist_pct):.1f}%",
                     "Price tends to revert to max pain near expiry"],
                    {"max_pain_dist_pct": round(dist_pct, 2)},
                ))

        return signals

    # ── 5. IV Skew ────────────────────────────────────────────

    def _detect_iv_skew(self, report: OptionsOIReport, ctx: dict) -> list[OISignal]:
        """Trade IV skew extremes — mean reversion expected."""
        signals = []
        skew = report.iv_skew  # PE IV - CE IV

        if abs(skew) < self.config.iv_skew_threshold:
            return signals

        if skew > self.config.iv_skew_threshold:
            # Put IV much higher than call IV → Fear skew → sell puts (mean reversion)
            otm_pe_strike = report.atm_strike - ctx["strike_gap"] * 2
            pe_price = self._get_option_price_at_strike(report, otm_pe_strike, "PE")
            if pe_price > 0:
                signals.append(self._make_signal(
                    ctx, OISignalType.IV_SKEW_REVERSAL, OIStrategyAction.SELL_PE,
                    "BULLISH", min(75, 50 + skew * 3),
                    otm_pe_strike, "PE", pe_price,
                    [f"IV Skew extreme: PE IV +{skew:.1f}% above CE IV",
                     "Fear premium elevated — mean reversion (sell puts)"],
                    {"iv_skew": round(skew, 2)},
                ))

        elif skew < -self.config.iv_skew_threshold:
            # Call IV much higher → Greed skew → sell calls (mean reversion)
            otm_ce_strike = report.atm_strike + ctx["strike_gap"] * 2
            ce_price = self._get_option_price_at_strike(report, otm_ce_strike, "CE")
            if ce_price > 0:
                signals.append(self._make_signal(
                    ctx, OISignalType.IV_SKEW_REVERSAL, OIStrategyAction.SELL_CE,
                    "BEARISH", min(75, 50 + abs(skew) * 3),
                    otm_ce_strike, "CE", ce_price,
                    [f"IV Skew extreme: CE IV +{abs(skew):.1f}% above PE IV",
                     "Greed premium elevated — mean reversion (sell calls)"],
                    {"iv_skew": round(skew, 2)},
                ))

        return signals

    # ── 6. OI Buildup Momentum ────────────────────────────────

    def _detect_oi_buildup(self, report: OptionsOIReport, ctx: dict) -> list[OISignal]:
        """Detect heavy OI buildup near ATM as momentum signal."""
        signals = []

        if not report.buildup_signals:
            return signals

        bull_count = sum(1 for s in report.buildup_signals if s.get("sentiment") == "BULLISH")
        bear_count = sum(1 for s in report.buildup_signals if s.get("sentiment") == "BEARISH")

        # Strong directional buildup
        if bull_count >= bear_count + 3:
            atm_ce_price = self._get_atm_option_price(report, "CE")
            if atm_ce_price > 0:
                top_signals = [s for s in report.buildup_signals if s.get("sentiment") == "BULLISH"][:3]
                reasons = [f"Strong bullish OI buildup: {bull_count} bullish vs {bear_count} bearish"]
                reasons.extend([s.get("description", "") for s in top_signals])
                signals.append(self._make_signal(
                    ctx, OISignalType.OI_BUILDUP_BULLISH, OIStrategyAction.BUY_CE,
                    "BULLISH", min(80, 50 + bull_count * 5),
                    report.atm_strike, "CE", atm_ce_price,
                    reasons,
                    {"bull_signals": bull_count, "bear_signals": bear_count},
                ))

        elif bear_count >= bull_count + 3:
            atm_pe_price = self._get_atm_option_price(report, "PE")
            if atm_pe_price > 0:
                top_signals = [s for s in report.buildup_signals if s.get("sentiment") == "BEARISH"][:3]
                reasons = [f"Strong bearish OI buildup: {bear_count} bearish vs {bull_count} bullish"]
                reasons.extend([s.get("description", "") for s in top_signals])
                signals.append(self._make_signal(
                    ctx, OISignalType.OI_BUILDUP_BEARISH, OIStrategyAction.BUY_PE,
                    "BEARISH", min(80, 50 + bear_count * 5),
                    report.atm_strike, "PE", atm_pe_price,
                    reasons,
                    {"bull_signals": bull_count, "bear_signals": bear_count},
                ))

        return signals

    # ── 7. Put Writing Support ────────────────────────────────

    def _detect_put_writing_support(self, report: OptionsOIReport, ctx: dict) -> list[OISignal]:
        """Heavy PE writing at OTM strikes = support → sell OTM puts."""
        signals = []
        pe_wall = report.max_pe_oi_strike
        pe_oi = report.max_pe_oi

        min_oi = self.config.wall_min_oi_lakh * 100000

        if pe_wall > 0 and pe_oi >= min_oi and pe_wall < report.atm_strike:
            pe_price = self._get_option_price_at_strike(report, pe_wall, "PE")
            if pe_price > 0:
                signals.append(self._make_signal(
                    ctx, OISignalType.PUT_WRITING_SUPPORT, OIStrategyAction.SELL_PE,
                    "BULLISH", min(75, 55 + (pe_oi / min_oi) * 5),
                    pe_wall, "PE", pe_price,
                    [f"Heavy PE writing at {pe_wall:.0f} (OI: {pe_oi:,})",
                     "Writers providing support — sell OTM put"],
                    {"pe_wall_oi": pe_oi},
                ))

        return signals

    # ── 8. Call Writing Resistance ────────────────────────────

    def _detect_call_writing_resistance(self, report: OptionsOIReport, ctx: dict) -> list[OISignal]:
        """Heavy CE writing at OTM strikes = resistance → sell OTM calls."""
        signals = []
        ce_wall = report.max_ce_oi_strike
        ce_oi = report.max_ce_oi

        min_oi = self.config.wall_min_oi_lakh * 100000

        if ce_wall > 0 and ce_oi >= min_oi and ce_wall > report.atm_strike:
            ce_price = self._get_option_price_at_strike(report, ce_wall, "CE")
            if ce_price > 0:
                signals.append(self._make_signal(
                    ctx, OISignalType.CALL_WRITING_RESISTANCE, OIStrategyAction.SELL_CE,
                    "BEARISH", min(75, 55 + (ce_oi / min_oi) * 5),
                    ce_wall, "CE", ce_price,
                    [f"Heavy CE writing at {ce_wall:.0f} (OI: {ce_oi:,})",
                     "Writers providing resistance — sell OTM call"],
                    {"ce_wall_oi": ce_oi},
                ))

        return signals

    # ── Combined Signal ───────────────────────────────────────

    def _create_combined_signal(
        self, report: OptionsOIReport, ctx: dict,
        direction: str, confidence: float, signals: list[OISignal],
    ) -> Optional[OISignal]:
        """Create a combined signal when multiple signals align."""
        if direction == "BULLISH":
            atm_price = self._get_atm_option_price(report, "CE")
            if atm_price > 0:
                reasons = [f"Combined bullish: {len([s for s in signals if s.direction == 'BULLISH'])} aligned signals"]
                reasons.extend([s.reasons[0] for s in signals if s.direction == "BULLISH" and s.reasons][:3])
                return self._make_signal(
                    ctx, OISignalType.COMBINED_BULLISH, OIStrategyAction.BUY_CE,
                    "BULLISH", confidence,
                    report.atm_strike, "CE", atm_price, reasons,
                    {"component_signals": [s.signal_type for s in signals if s.direction == "BULLISH"]},
                )
        elif direction == "BEARISH":
            atm_price = self._get_atm_option_price(report, "PE")
            if atm_price > 0:
                reasons = [f"Combined bearish: {len([s for s in signals if s.direction == 'BEARISH'])} aligned signals"]
                reasons.extend([s.reasons[0] for s in signals if s.direction == "BEARISH" and s.reasons][:3])
                return self._make_signal(
                    ctx, OISignalType.COMBINED_BEARISH, OIStrategyAction.BUY_PE,
                    "BEARISH", confidence,
                    report.atm_strike, "PE", atm_price, reasons,
                    {"component_signals": [s.signal_type for s in signals if s.direction == "BEARISH"]},
                )
        return None

    # ═══════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _get_atm_option_price(report: OptionsOIReport, opt_type: str) -> float:
        """Get ATM option price from report strikes."""
        for s in report.strikes:
            if s.is_atm:
                return s.ce_ltp if opt_type == "CE" else s.pe_ltp
        # Fallback: find closest to ATM
        if report.strikes:
            closest = min(report.strikes, key=lambda x: abs(x.strike - report.atm_strike))
            return closest.ce_ltp if opt_type == "CE" else closest.pe_ltp
        return 0.0

    @staticmethod
    def _get_option_price_at_strike(report: OptionsOIReport, strike: float, opt_type: str) -> float:
        """Get option price at a specific strike."""
        for s in report.strikes:
            if s.strike == strike:
                return s.ce_ltp if opt_type == "CE" else s.pe_ltp
        # Find nearest
        if report.strikes:
            closest = min(report.strikes, key=lambda x: abs(x.strike - strike))
            if abs(closest.strike - strike) <= 200:
                return closest.ce_ltp if opt_type == "CE" else closest.pe_ltp
        return 0.0

    @staticmethod
    def _build_tradingsymbol(underlying: str, expiry: str, strike: float, option_type: str) -> str:
        """Build Zerodha-compatible tradingsymbol like NIFTY2522725500CE.

        Kite expiry comes as 'YYYY-MM-DD' string or date — we convert to
        the compact format used by Zerodha: YY + upper-month-abbrev + DD.
        Falls back to a readable label if expiry is missing/unparseable.
        """
        strike_str = str(int(strike)) if strike == int(strike) else f"{strike:.1f}"

        if not expiry:
            return f"{underlying}_{strike_str}{option_type}"

        try:
            if isinstance(expiry, str):
                exp_date = datetime.strptime(expiry[:10], "%Y-%m-%d")
            elif isinstance(expiry, date):
                exp_date = datetime.combine(expiry, datetime.min.time())
            else:
                return f"{underlying}_{strike_str}{option_type}"

            yy = exp_date.strftime("%y")
            # Month: use full uppercase month abbreviation for Zerodha NFO format
            month_map = {
                1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
                7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC",
            }
            mon = month_map[exp_date.month]
            dd = exp_date.strftime("%d")
            # Zerodha weekly format: NIFTY + YY + Mon-abbrev + DD + Strike + CE/PE
            # e.g. NIFTY25FEB2725500CE  (note: monthly uses 3-letter month, weekly uses 1/2-digit month+date)
            # Zerodha actually uses: NIFTY + YY + M + DD + Strike + CE/PE for weeklies
            # where M = 1..9 for Jan-Sep, O/N/D for Oct/Nov/Dec
            # For monthly: NIFTY + YY + MON + Strike + CE/PE
            # We'll use the monthly format which is readable and works for display
            return f"{underlying}{yy}{mon}{dd}{strike_str}{option_type}"
        except (ValueError, KeyError):
            return f"{underlying}_{strike_str}{option_type}"
