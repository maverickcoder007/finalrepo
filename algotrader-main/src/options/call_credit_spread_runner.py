"""
Call Credit Spread with Trailing Stop — "Risk-Free Runner" Strategy
====================================================================

Entry:
    • Sell ATM/OTM Call (e.g. 25500 CE)
    • Buy higher-strike Call (e.g. 25600 CE) — 100-point wide spread

Stop-Loss on sold call:
    • +40 premium points above sold-call entry price

Scenario 1 — SL Hit (market moves against):
    • Close the short call (at loss)
    • Keep the long call open (now profitable since market went UP)
    • Move long call SL to break-even → "Risk-free runner"

Scenario 2 — Market moves 60 pts in favour (spread profits):
    • Move the entire spread SL to break-even
    • Let the trade run to expiry or profit target

The strategy is implemented as a proper FnO class that plugs into the
existing OptionStrategyBase → ExecutionEngine → Journal pipeline.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import pandas as pd

from src.data.models import (
    Exchange,
    OrderType,
    ProductType,
    Signal,
    Tick,
    TransactionType,
)
from src.options.chain import OptionChainData, OptionContract
from src.options.strategies import OptionStrategyBase
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Path for strategy state persistence
_STATE_DIR = os.path.join("data", "strategy_state")


# ────────────────────────────────────────────────────────────────
# State Machine
# ────────────────────────────────────────────────────────────────

class SpreadState(str, Enum):
    """Lifecycle states for the call credit spread."""
    IDLE = "idle"                       # No position
    PENDING_ENTRY = "pending_entry"     # Signals sent, awaiting order confirmation
    SPREAD_OPEN = "spread_open"         # Both legs open, initial SL active
    RUNNER_LONG_ONLY = "runner_long"    # Short leg closed (SL hit), long running
    SPREAD_BE = "spread_be"             # Spread SL moved to break-even
    CLOSED = "closed"                   # All legs exited


class CallCreditSpreadRunnerStrategy(OptionStrategyBase):
    """Call Credit Spread with trailing stop management.

    Params (all overridable):
        spread_width        — strike gap between short & long call (default 100)
        sl_premium_points   — SL distance on the sold call premium  (default 40)
        be_trigger_points   — spot move in favour to shift SL to BE (default 60)
        profit_target_pct   — close spread at this % of max-profit  (default 80)
        short_call_delta    — delta of the short call for entry      (default 0.30)
        underlying          — "NIFTY" or "SENSEX"
    """

    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        defaults = {
            "spread_width": 100,            # 100-point wide spread
            "sl_premium_points": 40,        # +40 on sold call
            "be_trigger_points": 60,        # move to BE after 60 pts in favour
            "profit_target_pct": 80.0,      # close at 80% of max profit
            "short_call_delta": 0.30,       # ~30 delta short call
            "max_risk_pct": 0.02,           # 2% of capital per spread
            "capital": 500000.0,            # default capital for risk calc
            "min_credit": 30.0,             # minimum net credit per lot (covers round-trip costs)
            "disable_persistence": False,   # disable state persistence for backtest
        }
        merged = {**defaults, **(params or {})}
        # Calculate max_risk_per_spread from capital * max_risk_pct
        if "max_risk_per_spread" not in (params or {}):
            merged["max_risk_per_spread"] = merged["capital"] * merged["max_risk_pct"]
        super().__init__("call_credit_spread_runner", merged)
        self._user_params = params  # Store to check user overrides later

        # ── Internal state ──
        self._state = SpreadState.IDLE
        self._short_leg: Optional[dict[str, Any]] = None   # sold call info
        self._long_leg: Optional[dict[str, Any]] = None    # bought call info
        self._entry_credit: float = 0.0                    # net premium received
        self._entry_spot: float = 0.0                      # spot at entry
        self._short_entry_premium: float = 0.0             # premium at which short call was sold
        self._long_entry_premium: float = 0.0              # premium paid for long call
        self._sl_triggered_at: Optional[str] = None        # ISO timestamp
        self._last_skip_log_time: float = 0.0              # throttle skip logs

        # Fix 3: Spread linkage — unique ID per spread trade
        self._spread_id: Optional[str] = None

        # Fix 5: Re-entry throttle after rejection/failure
        self._entry_cooldown_until: float = 0.0            # monotonic time
        self._ENTRY_COOLDOWN_SECS: float = 60.0            # 60s cooldown after failure

        # Fix 1: Track pending order IDs to detect fill/rejection
        self._pending_order_ids: list[str] = []
        self._confirmed_fills: int = 0

        # IV regime filter: rolling ATM IV buffer used to avoid selling options
        # when IV is cyclically depressed (low-IV entry compresses credit and
        # widens breakeven, making the spread structurally unattractive).
        self._iv_history: list[float] = []
        self._IV_HISTORY_MAX: int = 30    # keep at most 30 observations
        self._IV_MIN_SAMPLES: int = 10    # need ≥10 samples before gating
        self._IV_PERCENTILE_GATE: float = 0.30  # block if IV < 30th percentile

        # Fix 4: Attempt to restore persisted state on init
        self._load_persisted_state()

    # ── Properties ──────────────────────────────────────────

    @property
    def state(self) -> SpreadState:
        return self._state

    @property
    def is_in_trade(self) -> bool:
        return self._state not in (SpreadState.IDLE, SpreadState.CLOSED)

    @property
    def spread_id(self) -> Optional[str]:
        return self._spread_id

    # ────────────────────────────────────────────────────────
    # On-Tick / On-Bar hooks (called by the live loop)
    # ────────────────────────────────────────────────────────

    async def on_tick(self, ticks: list[Tick]) -> list[Signal]:
        exit_signals = self._evaluate_exit(ticks)
        if exit_signals:
            return exit_signals
        if self._state not in (SpreadState.IDLE,):
            return []
        if not self._option_chain:
            now = time.monotonic()
            if now - self._last_skip_log_time > 30:
                logger.debug("ccs_runner_tick_skip_no_chain", hint="waiting for option chain refresh")
                self._last_skip_log_time = now
            return []
        # Fix 5: Respect entry cooldown
        if time.monotonic() < self._entry_cooldown_until:
            return []
        return self._evaluate_entry()

    async def on_bar(self, instrument_token: int, bar: pd.Series) -> list[Signal]:
        exit_signals = self._evaluate_exit([])
        if exit_signals:
            return exit_signals
        if self._state != SpreadState.IDLE or not self._option_chain:
            return []
        self._record_iv()
        # Fix 5: Respect entry cooldown
        if time.monotonic() < self._entry_cooldown_until:
            return []
        return self._evaluate_entry()

    def _record_iv(self) -> None:
        """Append current ATM IV to the rolling regime buffer."""
        chain = self._option_chain
        if chain and chain.atm_iv > 0:
            self._iv_history.append(chain.atm_iv)
            if len(self._iv_history) > self._IV_HISTORY_MAX:
                self._iv_history = self._iv_history[-self._IV_HISTORY_MAX:]

    def generate_signal(self, data: pd.DataFrame, instrument_token: int) -> Optional[Signal]:
        """Backtesting entry point — returns a single entry signal when conditions met."""
        return None  # Multi-leg entry handled by on_tick / on_bar

    # ────────────────────────────────────────────────────────
    # ENTRY LOGIC
    # ────────────────────────────────────────────────────────

    def _evaluate_entry(self) -> list[Signal]:
        if self._state != SpreadState.IDLE:
            logger.debug("ccs_runner_skip_not_idle", state=self._state.value)
            return []

        chain = self._option_chain
        if not chain or not chain.entries:
            logger.debug("ccs_runner_skip_no_chain", has_chain=bool(chain), entries=len(chain.entries) if chain else 0)
            return []

        logger.info(
            "ccs_runner_evaluating_entry",
            spot=chain.spot_price,
            entries=len(chain.entries),
            atm_iv=chain.atm_iv,
        )

        # IV regime check: skip entry if current ATM IV is below the 30th percentile
        # of recent observations. Selling options in a depressed-IV regime reduces
        # credit received, widens breakeven, and increases adverse-move risk.
        if chain.atm_iv > 0 and len(self._iv_history) >= self._IV_MIN_SAMPLES:
            sorted_ivs = sorted(self._iv_history)
            p30_idx = max(0, int(len(sorted_ivs) * self._IV_PERCENTILE_GATE) - 1)
            p30_iv = sorted_ivs[p30_idx]
            if chain.atm_iv < p30_iv:
                logger.info(
                    "ccs_runner_skip_low_iv_regime",
                    current_iv=round(chain.atm_iv, 4),
                    iv_p30=round(p30_iv, 4),
                    samples=len(self._iv_history),
                )
                return []

        # Dynamic spread width: 300 for SENSEX (~80k), 100 for NIFTY (~24k)
        if "spread_width" in (self._user_params or {}):
            spread_width = self.params["spread_width"]  # User override
        elif chain.spot_price > 50000:
            spread_width = 300  # SENSEX
        else:
            spread_width = 100  # NIFTY

        # 1. Find the short call — use delta-based selection, with ATM fallback
        short_call = self._find_strike_by_delta(
            self.params["short_call_delta"], "CE", chain
        )
        spot = chain.spot_price
        # Validate: short call must be OTM (strike > spot).  When all deltas are 0
        # (market closed, quote failure, or near-zero DTE), the selector can return
        # a deep-ITM call as the "closest" match — we must reject that and fall back.
        if not short_call or short_call.last_price <= 0 or short_call.strike <= spot:
            # Fallback: find OTM strike ~1-2 strikes above ATM with meaningful premium
            candidates = []
            for entry in chain.entries:
                if entry.ce and entry.ce.strike > spot and entry.ce.last_price > 1.0:
                    candidates.append(entry.ce)
            # Pick the one with highest premium (closest to ATM with liquidity)
            if candidates:
                short_call = max(candidates, key=lambda c: c.last_price)
        if not short_call or short_call.last_price <= 1.0:
            logger.info(
                "ccs_runner_no_short_call",
                target_delta=self.params["short_call_delta"],
                found=bool(short_call),
                price=short_call.last_price if short_call else 0,
                spot=chain.spot_price,
            )
            return []

        logger.info(
            "ccs_runner_short_selected",
            strike=short_call.strike,
            premium=round(short_call.last_price, 2),
            delta=round(short_call.delta, 4) if short_call.delta else 0,
        )

        # 2. Find long call — short_strike + spread_width
        target_long_strike = short_call.strike + spread_width
        long_call = self._find_strike_nearest(target_long_strike, "CE", chain)
        if not long_call or long_call.last_price <= 0:
            logger.info(
                "ccs_runner_no_long_call",
                target_strike=target_long_strike,
                found=bool(long_call),
                price=long_call.last_price if long_call else 0,
            )
            return []

        # Ensure correct ordering
        if long_call.strike <= short_call.strike:
            logger.info("ccs_runner_strikes_invalid", short=short_call.strike, long=long_call.strike)
            return []

        logger.info(
            "ccs_runner_long_selected",
            strike=long_call.strike,
            premium=round(long_call.last_price, 2),
        )

        # 3. Credit / risk validation
        net_credit = short_call.last_price - long_call.last_price
        if net_credit < self.params["min_credit"]:
            logger.info(
                "ccs_runner_credit_insufficient",
                short_strike=short_call.strike,
                long_strike=long_call.strike,
                credit=round(net_credit, 2),
                min_required=self.params["min_credit"],
                short_premium=round(short_call.last_price, 2),
                long_premium=round(long_call.last_price, 2),
            )
            return []

        actual_width = long_call.strike - short_call.strike
        max_loss = actual_width - net_credit  # per unit (multiply by lot for total)
        max_loss_per_lot = max_loss * self.params["lot_size"]

        if max_loss_per_lot > self.params["max_risk_per_spread"]:
            logger.warning(
                "ccs_runner_risk_exceeded",
                max_loss=max_loss_per_lot,
                limit=self.params["max_risk_per_spread"],
            )
            return []

        # 4. Build metadata
        meta = {
            "strategy_type": "call_credit_spread_runner",
            "short_call_strike": short_call.strike,
            "short_call_premium": round(short_call.last_price, 2),
            "long_call_strike": long_call.strike,
            "long_call_premium": round(long_call.last_price, 2),
            "net_credit": round(net_credit, 2),
            "max_profit_per_lot": round(net_credit * self.params["lot_size"], 2),
            "max_loss_per_lot": round(max_loss_per_lot, 2),
            "spread_width": actual_width,
            "sl_premium_points": self.params["sl_premium_points"],
            "be_trigger_points": self.params["be_trigger_points"],
            "spot_at_entry": chain.spot_price,
            "atm_iv": chain.atm_iv,
        }

        # 5. Compute premium-based stop-loss for the short call
        #    SL = short_call entry premium + sl_premium_points
        sl_trigger = short_call.last_price + self.params["sl_premium_points"]

        # Generate spread_id for linking legs together
        spread_id = f"ccs_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"

        short_signal = self._create_signal(
            short_call, TransactionType.SELL, confidence=65.0,
            metadata={**meta, "leg": "short_call", "spread_id": spread_id},
        )
        # NOTE: Do NOT set short_signal.stop_loss here.
        # The strategy manages SL internally via chain premium monitoring.
        # Setting stop_loss would cause the ExecutionEngine to place a
        # competing broker-side protective SL order → risk of double execution.

        long_signal = self._create_signal(
            long_call, TransactionType.BUY, confidence=65.0,
            metadata={**meta, "leg": "long_call", "spread_id": spread_id},
        )

        # Fix 1: Transition to PENDING_ENTRY, not SPREAD_OPEN.
        # SPREAD_OPEN is confirmed only after orders are filled via notify_order_result().
        self._state = SpreadState.PENDING_ENTRY
        self._spread_id = spread_id
        self._entry_credit = net_credit
        self._entry_spot = chain.spot_price
        self._short_entry_premium = short_call.last_price
        self._long_entry_premium = long_call.last_price
        self._pending_order_ids = []  # Populated by notify_order_result
        self._confirmed_fills = 0
        self._short_leg = {
            "tradingsymbol": short_call.tradingsymbol,
            "strike": short_call.strike,
            "entry_premium": short_call.last_price,
            "sl_premium": sl_trigger,
            "quantity": self.params["lot_size"] * self.params["lots"],
        }
        self._long_leg = {
            "tradingsymbol": long_call.tradingsymbol,
            "strike": long_call.strike,
            "entry_premium": long_call.last_price,
            "quantity": self.params["lot_size"] * self.params["lots"],
        }
        self._active_legs = [
            short_signal.model_dump(),
            long_signal.model_dump(),
        ]

        logger.info(
            "ccs_runner_entry",
            short=short_call.tradingsymbol,
            long=long_call.tradingsymbol,
            credit=net_credit,
            sl_trigger=sl_trigger,
            spot=chain.spot_price,
            spread_id=spread_id,
            state="PENDING_ENTRY",
        )

        return [short_signal, long_signal]

    # ────────────────────────────────────────────────────────
    # EXIT / ADJUSTMENT LOGIC
    # ────────────────────────────────────────────────────────

    def _evaluate_exit(self, ticks: list[Tick]) -> list[Signal]:
        """Evaluate exit / adjustment conditions based on current chain+tick data."""
        if self._state in (SpreadState.IDLE, SpreadState.CLOSED, SpreadState.PENDING_ENTRY):
            return []
        if not self._option_chain:
            return []

        chain = self._option_chain
        spot = chain.spot_price

        # ── State: SPREAD_OPEN — both legs active ───────────
        if self._state == SpreadState.SPREAD_OPEN:
            return self._check_spread_open_exits(chain, spot)

        # ── State: RUNNER_LONG_ONLY — short exited, long running ─
        if self._state == SpreadState.RUNNER_LONG_ONLY:
            return self._check_runner_exits(chain, spot)

        # ── State: SPREAD_BE — spread at break-even SL ────────
        if self._state == SpreadState.SPREAD_BE:
            return self._check_spread_be_exits(chain, spot)

        return []

    def _check_spread_open_exits(self, chain: OptionChainData, spot: float) -> list[Signal]:
        """While spread is open: check short-call SL and BE-trigger."""
        if not self._short_leg or not self._long_leg:
            return []

        short_strike = self._short_leg["strike"]
        sl_premium = self._short_leg["sl_premium"]

        # Get current premium of the short call from chain
        current_short_premium = self._get_option_price_from_chain(
            chain, short_strike, "CE"
        )

        # ── Condition 1: Short-call SL hit (premium rose beyond SL point) ──
        if current_short_premium is not None and current_short_premium >= sl_premium:
            logger.warning(
                "ccs_runner_short_sl_hit",
                short_strike=short_strike,
                current_premium=current_short_premium,
                sl_trigger=sl_premium,
            )
            return self._handle_short_sl_hit(chain)

        # ── Condition 2: Spot moved 60 pts in favour (down for credit spread) ──
        spot_move = self._entry_spot - spot  # positive = moved in our favour
        be_trigger = self.params["be_trigger_points"]
        if spot_move >= be_trigger:
            logger.info(
                "ccs_runner_be_trigger",
                spot_move=round(spot_move, 2),
                be_trigger=be_trigger,
            )
            self._state = SpreadState.SPREAD_BE
            logger.info("ccs_runner_state_change", new_state="SPREAD_BE")
            # No signals — just state change; SL is now effectively at entry credit
            return []

        # ── Condition 3: Profit target ──
        current_long_premium = self._get_option_price_from_chain(
            chain, self._long_leg["strike"], "CE"
        )
        if current_short_premium is not None and current_long_premium is not None:
            current_spread_cost = current_short_premium - current_long_premium
            profit_captured = self._entry_credit - current_spread_cost
            max_possible = self._entry_credit
            target_pct = self.params["profit_target_pct"] / 100
            if max_possible > 0 and profit_captured >= max_possible * target_pct:
                logger.info(
                    "ccs_runner_profit_target",
                    profit_pct=round((profit_captured / max_possible) * 100, 1),
                    captured=round(profit_captured, 2),
                )
                return self._close_full_spread("profit_target_hit")

        return []

    def _handle_short_sl_hit(self, chain: OptionChainData) -> list[Signal]:
        """Scenario 1: Short-call SL hit.
        → Close short call.
        → Keep long call open with SL at break-even.
        → Transition to RUNNER_LONG_ONLY state.
        """
        signals: list[Signal] = []

        # Close the short call (buy back)
        short_sym = self._short_leg["tradingsymbol"]
        qty = self._short_leg["quantity"]
        signals.append(Signal(
            tradingsymbol=short_sym,
            exchange=Exchange(self.params["exchange"]),
            transaction_type=TransactionType.BUY,  # buy-back the sold call
            quantity=qty,
            order_type=OrderType.MARKET,
            product=ProductType(self.params["product"]),
            strategy_name=self.name,
            confidence=80.0,
            metadata={
                "leg": "short_call_exit",
                "reason": "sl_hit",
                "signal_type": "exit",
                "spread_id": self._spread_id,
                "sl_premium": self._short_leg["sl_premium"],
                "strategy_type": "call_credit_spread_runner",
            },
        ))

        # Transition: long call is now a runner
        self._state = SpreadState.RUNNER_LONG_ONLY
        self._sl_triggered_at = datetime.now().isoformat()

        # The long call's BE = its own entry premium
        # (We don't generate a signal here — the execution engine's
        #  TrackedPosition handles trailing SL updates)
        logger.info(
            "ccs_runner_state_transition",
            new_state="RUNNER_LONG_ONLY",
            long_call=self._long_leg["tradingsymbol"],
            long_be=self._long_entry_premium,
        )

        # Update _active_legs to only contain the long call
        self._active_legs = [
            leg for leg in self._active_legs
            if leg.get("metadata", {}).get("leg") != "short_call"
        ]

        return signals

    def _check_runner_exits(self, chain: OptionChainData, spot: float) -> list[Signal]:
        """State: Long call running as risk-free runner.
        Exit if premium drops back to entry premium (break-even SL).
        """
        if not self._long_leg:
            return []

        long_strike = self._long_leg["strike"]
        current_premium = self._get_option_price_from_chain(chain, long_strike, "CE")

        if current_premium is None:
            return []

        # BE stop: close if premium falls back to entry
        entry_premium = self._long_entry_premium
        if current_premium <= entry_premium:
            logger.info(
                "ccs_runner_long_be_exit",
                current_premium=current_premium,
                entry_premium=entry_premium,
            )
            return self._close_long_call("runner_be_exit")

        return []

    def _check_spread_be_exits(self, chain: OptionChainData, spot: float) -> list[Signal]:
        """State: Spread SL moved to break-even.
        Close if we'd lose the net credit (i.e. spread cost rises to entry credit).
        """
        if not self._short_leg or not self._long_leg:
            return []

        short_premium = self._get_option_price_from_chain(
            chain, self._short_leg["strike"], "CE"
        )
        long_premium = self._get_option_price_from_chain(
            chain, self._long_leg["strike"], "CE"
        )

        if short_premium is None or long_premium is None:
            return []

        # Current spread value = short - long (what we'd pay to close)
        current_spread_cost = short_premium - long_premium

        # If cost to close >= entry credit, we've lost our break-even cushion
        if current_spread_cost >= self._entry_credit:
            logger.info(
                "ccs_runner_spread_be_exit",
                spread_cost=round(current_spread_cost, 2),
                entry_credit=round(self._entry_credit, 2),
            )
            return self._close_full_spread("spread_be_exit")

        # Profit target still applies
        profit_captured = self._entry_credit - current_spread_cost
        target_pct = self.params["profit_target_pct"] / 100
        if self._entry_credit > 0 and profit_captured >= self._entry_credit * target_pct:
            return self._close_full_spread("profit_target_hit")

        return []

    # ────────────────────────────────────────────────────────
    # Exit Helpers
    # ────────────────────────────────────────────────────────

    def _close_full_spread(self, reason: str) -> list[Signal]:
        """Close both legs of the spread."""
        signals: list[Signal] = []

        if self._short_leg:
            signals.append(Signal(
                tradingsymbol=self._short_leg["tradingsymbol"],
                exchange=Exchange(self.params["exchange"]),
                transaction_type=TransactionType.BUY,
                quantity=self._short_leg["quantity"],
                order_type=OrderType.MARKET,
                product=ProductType(self.params["product"]),
                strategy_name=self.name,
                confidence=80.0,
                metadata={
                    "leg": "short_call_exit",
                    "reason": reason,
                    "signal_type": "exit",
                    "spread_id": self._spread_id,
                    "strategy_type": "call_credit_spread_runner",
                },
            ))

        if self._long_leg:
            signals.append(Signal(
                tradingsymbol=self._long_leg["tradingsymbol"],
                exchange=Exchange(self.params["exchange"]),
                transaction_type=TransactionType.SELL,
                quantity=self._long_leg["quantity"],
                order_type=OrderType.MARKET,
                product=ProductType(self.params["product"]),
                strategy_name=self.name,
                confidence=80.0,
                metadata={
                    "leg": "long_call_exit",
                    "reason": reason,
                    "signal_type": "exit",
                    "spread_id": self._spread_id,
                    "strategy_type": "call_credit_spread_runner",
                },
            ))

        self._reset_state(reason)
        return signals

    def _close_long_call(self, reason: str) -> list[Signal]:
        """Close only the long call (used in runner state)."""
        if not self._long_leg:
            return []

        signal = Signal(
            tradingsymbol=self._long_leg["tradingsymbol"],
            exchange=Exchange(self.params["exchange"]),
            transaction_type=TransactionType.SELL,
            quantity=self._long_leg["quantity"],
            order_type=OrderType.MARKET,
            product=ProductType(self.params["product"]),
            strategy_name=self.name,
            confidence=80.0,
            metadata={
                "leg": "long_call_exit",
                "reason": reason,
                "signal_type": "exit",
                "spread_id": self._spread_id,
                "strategy_type": "call_credit_spread_runner",
            },
        )
        self._reset_state(reason)
        return [signal]

    def _reset_state(self, reason: str) -> None:
        """Reset internal state after full exit."""
        logger.info(
            "ccs_runner_position_closed",
            reason=reason,
            entry_credit=self._entry_credit,
            entry_spot=self._entry_spot,
            state_before=self._state.value,
            spread_id=self._spread_id,
        )
        self._short_leg = None
        self._long_leg = None
        self._active_legs = []
        self._entry_credit = 0
        self._entry_spot = 0
        self._short_entry_premium = 0
        self._long_entry_premium = 0
        self._sl_triggered_at = None
        self._spread_id = None
        self._pending_order_ids = []
        self._confirmed_fills = 0
        # Transition directly to IDLE — ready for re-entry on next cycle
        self._state = SpreadState.IDLE
        # Persist clean state
        self._persist_state()

    # ────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────

    def _get_option_price_from_chain(
        self, chain: OptionChainData, strike: float, opt_type: str
    ) -> Optional[float]:
        """Look up current option price from the live chain."""
        for entry in chain.entries:
            contract = entry.ce if opt_type == "CE" else entry.pe
            if contract and abs(contract.strike - strike) < 0.01 and contract.last_price > 0:
                return contract.last_price
        return None

    def get_status(self) -> dict[str, Any]:
        """Return current strategy status for dashboard / API."""
        return {
            "strategy": self.name,
            "state": self._state.value,
            "in_trade": self.is_in_trade,
            "entry_credit": self._entry_credit,
            "entry_spot": self._entry_spot,
            "short_leg": self._short_leg,
            "long_leg": self._long_leg,
            "short_entry_premium": self._short_entry_premium,
            "long_entry_premium": self._long_entry_premium,
            "sl_premium_points": self.params["sl_premium_points"],
            "be_trigger_points": self.params["be_trigger_points"],
            "sl_triggered_at": self._sl_triggered_at,
            "spread_id": self._spread_id,
            "params": self.params,
        }

    # ────────────────────────────────────────────────────────
    # Fix 1: Order result notification (called by TradingService)
    # ────────────────────────────────────────────────────────

    def notify_order_placed(self, order_id: str, tradingsymbol: str) -> None:
        """Called by TradingService after an entry order is successfully placed."""
        if self._state == SpreadState.PENDING_ENTRY:
            self._pending_order_ids.append(order_id)
            logger.info(
                "ccs_runner_order_placed",
                order_id=order_id,
                symbol=tradingsymbol,
                pending_count=len(self._pending_order_ids),
                spread_id=self._spread_id,
            )

    def notify_order_filled(self, order_id: str, fill_price: float) -> None:
        """Called by TradingService when an entry order fills.
        Transitions PENDING_ENTRY → SPREAD_OPEN when both legs are confirmed."""
        if self._state == SpreadState.PENDING_ENTRY:
            self._confirmed_fills += 1
            logger.info(
                "ccs_runner_fill_confirmed",
                order_id=order_id,
                fill_price=fill_price,
                fills=self._confirmed_fills,
                spread_id=self._spread_id,
            )
            if self._confirmed_fills >= 2:  # Both legs filled
                self._state = SpreadState.SPREAD_OPEN
                self._persist_state()
                logger.info(
                    "ccs_runner_spread_confirmed",
                    state="SPREAD_OPEN",
                    spread_id=self._spread_id,
                )

    def notify_order_failed(self, order_id: str, reason: str) -> None:
        """Called by TradingService when an entry order is rejected/fails.
        Reverts from PENDING_ENTRY → IDLE and activates cooldown."""
        if self._state == SpreadState.PENDING_ENTRY:
            logger.warning(
                "ccs_runner_order_failed_reverting",
                order_id=order_id,
                reason=reason,
                spread_id=self._spread_id,
            )
            self._reset_state(f"order_failed:{reason}")
            # Fix 5: Activate cooldown to prevent immediate re-entry spam
            self._entry_cooldown_until = time.monotonic() + self._ENTRY_COOLDOWN_SECS
            logger.info(
                "ccs_runner_entry_cooldown_set",
                cooldown_secs=self._ENTRY_COOLDOWN_SECS,
            )

    # ────────────────────────────────────────────────────────
    # Fix 4: State Persistence
    # ────────────────────────────────────────────────────────

    def _state_file_path(self) -> str:
        return os.path.join(_STATE_DIR, f"{self.name}.json")

    def _persist_state(self) -> None:
        """Save strategy state to disk for crash recovery."""
        try:
            os.makedirs(_STATE_DIR, exist_ok=True)
            state_data = {
                "state": self._state.value,
                "spread_id": self._spread_id,
                "entry_credit": self._entry_credit,
                "entry_spot": self._entry_spot,
                "short_entry_premium": self._short_entry_premium,
                "long_entry_premium": self._long_entry_premium,
                "short_leg": self._short_leg,
                "long_leg": self._long_leg,
                "sl_triggered_at": self._sl_triggered_at,
                "pending_order_ids": self._pending_order_ids,
                "confirmed_fills": self._confirmed_fills,
                "updated_at": datetime.now().isoformat(),
            }
            with open(self._state_file_path(), "w") as f:
                json.dump(state_data, f, indent=2)
        except Exception as e:
            logger.error("ccs_runner_persist_state_error", error=str(e))

    def _load_persisted_state(self) -> None:
        """Load strategy state from disk on startup for crash recovery."""
        if self.params.get("disable_persistence", False):
            return  # Skip loading for backtest
        try:
            fp = self._state_file_path()
            if not os.path.exists(fp):
                return
            with open(fp, "r") as f:
                data = json.load(f)

            saved_state = data.get("state", "idle")

            # Only restore if there was an active position
            if saved_state in (SpreadState.SPREAD_OPEN.value,
                               SpreadState.RUNNER_LONG_ONLY.value,
                               SpreadState.SPREAD_BE.value):
                self._state = SpreadState(saved_state)
                self._spread_id = data.get("spread_id")
                self._entry_credit = data.get("entry_credit", 0)
                self._entry_spot = data.get("entry_spot", 0)
                self._short_entry_premium = data.get("short_entry_premium", 0)
                self._long_entry_premium = data.get("long_entry_premium", 0)
                self._short_leg = data.get("short_leg")
                self._long_leg = data.get("long_leg")
                self._sl_triggered_at = data.get("sl_triggered_at")
                self._pending_order_ids = data.get("pending_order_ids", [])
                self._confirmed_fills = data.get("confirmed_fills", 0)
                logger.warning(
                    "ccs_runner_state_restored",
                    state=saved_state,
                    spread_id=self._spread_id,
                    short_leg=self._short_leg.get("tradingsymbol") if self._short_leg else None,
                    long_leg=self._long_leg.get("tradingsymbol") if self._long_leg else None,
                )
            elif saved_state == SpreadState.PENDING_ENTRY.value:
                # PENDING_ENTRY on restart means orders may be orphaned;
                # revert to IDLE and let reconciliation handle orphaned orders
                logger.warning(
                    "ccs_runner_pending_on_restart_reverting",
                    spread_id=data.get("spread_id"),
                )
                self._state = SpreadState.IDLE
                self._persist_state()
        except Exception as e:
            logger.error("ccs_runner_load_state_error", error=str(e))
