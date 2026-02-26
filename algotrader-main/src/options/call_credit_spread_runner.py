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

from datetime import datetime
from enum import Enum
from typing import Any, Optional

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


# ────────────────────────────────────────────────────────────────
# State Machine
# ────────────────────────────────────────────────────────────────

class SpreadState(str, Enum):
    """Lifecycle states for the call credit spread."""
    IDLE = "idle"                       # No position
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
            "max_risk_per_spread": 10000.0, # max loss cap per spread
            "min_credit": 15.0,             # minimum net credit per lot
        }
        merged = {**defaults, **(params or {})}
        super().__init__("call_credit_spread_runner", merged)

        # ── Internal state ──
        self._state = SpreadState.IDLE
        self._short_leg: Optional[dict[str, Any]] = None   # sold call info
        self._long_leg: Optional[dict[str, Any]] = None    # bought call info
        self._entry_credit: float = 0.0                    # net premium received
        self._entry_spot: float = 0.0                      # spot at entry
        self._short_entry_premium: float = 0.0             # premium at which short call was sold
        self._long_entry_premium: float = 0.0              # premium paid for long call
        self._sl_triggered_at: Optional[str] = None        # ISO timestamp

    # ── Properties ──────────────────────────────────────────

    @property
    def state(self) -> SpreadState:
        return self._state

    @property
    def is_in_trade(self) -> bool:
        return self._state not in (SpreadState.IDLE, SpreadState.CLOSED)

    # ────────────────────────────────────────────────────────
    # On-Tick / On-Bar hooks (called by the live loop)
    # ────────────────────────────────────────────────────────

    async def on_tick(self, ticks: list[Tick]) -> list[Signal]:
        exit_signals = self._evaluate_exit(ticks)
        if exit_signals:
            return exit_signals
        if self._state != SpreadState.IDLE or not self._option_chain:
            return []
        return self._evaluate_entry()

    async def on_bar(self, instrument_token: int, bar: pd.Series) -> list[Signal]:
        exit_signals = self._evaluate_exit([])
        if exit_signals:
            return exit_signals
        if self._state != SpreadState.IDLE or not self._option_chain:
            return []
        return self._evaluate_entry()

    def generate_signal(self, data: pd.DataFrame, instrument_token: int) -> Optional[Signal]:
        """Backtesting entry point — returns a single entry signal when conditions met."""
        return None  # Multi-leg entry handled by on_tick / on_bar

    # ────────────────────────────────────────────────────────
    # ENTRY LOGIC
    # ────────────────────────────────────────────────────────

    def _evaluate_entry(self) -> list[Signal]:
        if self._state != SpreadState.IDLE:
            return []

        chain = self._option_chain
        if not chain or not chain.entries:
            return []

        spread_width = self.params["spread_width"]

        # 1. Find the short call — use delta-based selection
        short_call = self._find_strike_by_delta(
            self.params["short_call_delta"], "CE", chain
        )
        if not short_call or short_call.last_price <= 0:
            return []

        # 2. Find long call — short_strike + spread_width
        target_long_strike = short_call.strike + spread_width
        long_call = self._find_strike_nearest(target_long_strike, "CE", chain)
        if not long_call or long_call.last_price <= 0:
            return []

        # Ensure correct ordering
        if long_call.strike <= short_call.strike:
            logger.debug("ccs_runner_strikes_invalid", short=short_call.strike, long=long_call.strike)
            return []

        # 3. Credit / risk validation
        net_credit = short_call.last_price - long_call.last_price
        if net_credit < self.params["min_credit"]:
            logger.debug("ccs_runner_credit_insufficient", credit=net_credit)
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

        short_signal = self._create_signal(
            short_call, TransactionType.SELL, confidence=65.0,
            metadata={**meta, "leg": "short_call"},
        )
        # NOTE: Do NOT set short_signal.stop_loss here.
        # The strategy manages SL internally via chain premium monitoring.
        # Setting stop_loss would cause the ExecutionEngine to place a
        # competing broker-side protective SL order → risk of double execution.

        long_signal = self._create_signal(
            long_call, TransactionType.BUY, confidence=65.0,
            metadata={**meta, "leg": "long_call"},
        )

        # 6. Update internal state
        self._state = SpreadState.SPREAD_OPEN
        self._entry_credit = net_credit
        self._entry_spot = chain.spot_price
        self._short_entry_premium = short_call.last_price
        self._long_entry_premium = long_call.last_price
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
        )

        return [short_signal, long_signal]

    # ────────────────────────────────────────────────────────
    # EXIT / ADJUSTMENT LOGIC
    # ────────────────────────────────────────────────────────

    def _evaluate_exit(self, ticks: list[Tick]) -> list[Signal]:
        """Evaluate exit / adjustment conditions based on current chain+tick data."""
        if self._state == SpreadState.IDLE or self._state == SpreadState.CLOSED:
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
        )
        self._short_leg = None
        self._long_leg = None
        self._active_legs = []
        self._entry_credit = 0
        self._entry_spot = 0
        self._short_entry_premium = 0
        self._long_entry_premium = 0
        self._sl_triggered_at = None
        # Transition directly to IDLE — ready for re-entry on next cycle
        self._state = SpreadState.IDLE

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
            "params": self.params,
        }
