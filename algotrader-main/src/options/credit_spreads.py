"""Credit spread strategies for options trading."""

from __future__ import annotations

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


class BullCallCreditSpreadStrategy(OptionStrategyBase):
    """Sell OTM call spread - collects premium, profits if price stays below short strike."""

    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        defaults = {
            "short_call_delta": 0.20,  # Sell ~20 delta call (OTM)
            "long_call_delta": 0.10,   # Buy ~10 delta call (further OTM, protection)
            "min_credit_per_spread": 20.0,  # Minimum credit to collect
            "max_risk_per_spread": 500.0,    # Max loss per spread
            "profit_target_pct": 50.0,   # Close at 50% of max profit
            "adjustment_threshold_pct": 20.0,  # Adjust if price moves this %
        }
        merged = {**defaults, **(params or {})}
        super().__init__("bull_call_credit_spread", merged)

    async def on_tick(self, ticks: list[Tick]) -> list[Signal]:
        exit_signals = await self._evaluate_exit(ticks)
        if exit_signals:
            return exit_signals
        if not self._option_chain or self._active_legs:
            return []
        return self._evaluate_entry()

    async def on_bar(self, instrument_token: int, bar: pd.Series) -> list[Signal]:
        exit_signals = await self._evaluate_exit([])
        if exit_signals:
            return exit_signals
        if not self._option_chain or self._active_legs:
            return []
        return self._evaluate_entry()

    def generate_signal(self, data: pd.DataFrame, instrument_token: int) -> Optional[Signal]:
        return None

    async def _evaluate_exit(self, ticks: list[Tick]) -> list[Signal]:
        """Check profit target and adjustment conditions."""
        if not self._active_legs or not self._option_chain:
            return []

        chain = self._option_chain
        first_leg = self._active_legs[0]
        meta = first_leg.get("metadata", {})

        # Get the short call strike (sold higher strike)
        short_call_strike = meta.get("short_call_strike", 0)
        long_call_strike = meta.get("long_call_strike", 0)
        entry_credit = meta.get("net_credit", 0)
        spot = chain.spot_price

        # Condition 1: Price moved up significantly (assignment risk)
        if spot > short_call_strike:
            distance_pct = ((spot - short_call_strike) / short_call_strike) * 100
            if distance_pct > self.params.get("adjustment_threshold_pct", 20.0):
                logger.warning(
                    "bull_call_credit_adjustment_needed",
                    spot=spot,
                    short_strike=short_call_strike,
                    distance_pct=distance_pct,
                )
                return self._create_close_signals("adjustment_needed_up_move")

        # Condition 2: Profit target check based on premium decay
        # Look up current option prices from chain to compute current spread value
        if entry_credit > 0 and short_call_strike > 0 and long_call_strike > 0:
            short_call_current = self._get_option_price(chain, short_call_strike, "CE")
            long_call_current = self._get_option_price(chain, long_call_strike, "CE")
            if short_call_current is not None and long_call_current is not None:
                current_spread_value = short_call_current - long_call_current
                # Profit = entry_credit - current_spread_value (cost to close)
                profit_captured = entry_credit - current_spread_value
                target_profit = entry_credit * (self.params["profit_target_pct"] / 100)
                if profit_captured >= target_profit:
                    logger.info(
                        "bull_call_credit_profit_target",
                        profit_captured=round(profit_captured, 2),
                        target=round(target_profit, 2),
                    )
                    return self._create_close_signals("profit_target_hit")

        return []

    def _get_option_price(self, chain, strike: float, opt_type: str) -> float | None:
        """Get option price from chain by strike and type."""
        if not hasattr(chain, 'strikes') and hasattr(chain, 'options'):
            for opt in chain.options:
                if opt.strike == strike and opt.instrument_type == opt_type:
                    return opt.last_price
        elif hasattr(chain, 'strikes'):
            key = f"{strike}_{opt_type}"
            if key in chain.strikes:
                return chain.strikes[key].price
        return None

    def _evaluate_entry(self) -> list[Signal]:
        chain = self._option_chain
        if not chain:
            return []

        # Find short call (OTM)
        short_call = self._find_strike_by_delta(self.params["short_call_delta"], "CE", chain)
        # Find long call (further OTM, protection)
        long_call = self._find_strike_by_delta(self.params["long_call_delta"], "CE", chain)

        if not short_call or not long_call or long_call.strike <= short_call.strike:
            return []

        # Credit = premium collected (sell) - premium paid (buy)
        net_credit = short_call.last_price - long_call.last_price
        if net_credit < self.params["min_credit_per_spread"]:
            logger.debug("bull_call_credit_insufficient", credit=net_credit)
            return []

        max_loss = (long_call.strike - short_call.strike) - net_credit
        if max_loss > self.params["max_risk_per_spread"]:
            logger.warning("bull_call_credit_max_risk_exceeded", max_loss=max_loss)
            return []

        profit_target = net_credit * (self.params["profit_target_pct"] / 100)

        stop_loss = round(0.3 * max_loss, 2)
        meta = {
            "strategy_type": "bull_call_credit_spread",
            "net_credit": round(net_credit, 2),
            "max_profit": round(net_credit * self.params["lot_size"], 2),
            "max_loss_per_spread": round(max_loss, 2),
            "profit_target": round(profit_target, 2),
            "short_call_strike": short_call.strike,
            "long_call_strike": long_call.strike,
            "spot_price": chain.spot_price,
            "atm_iv": chain.atm_iv,
            "stop_loss": stop_loss,
        }

        signals = [
            self._create_signal(short_call, TransactionType.SELL, 60.0, {**meta, "leg": "short_call"}),
            self._create_signal(long_call, TransactionType.BUY, 60.0, {**meta, "leg": "long_call"}),
        ]
        self._active_legs = [s.model_dump() for s in signals]
        logger.info("bull_call_credit_spread_entry", meta=meta)
        return signals

    def _create_close_signals(self, reason: str) -> list[Signal]:
        """Create closing signals for active legs."""
        if not self._active_legs:
            return []

        signals = []
        for leg in self._active_legs:
            meta = leg.get("metadata", {})
            reverse_type = (
                TransactionType.BUY
                if leg["transaction_type"] == TransactionType.SELL
                else TransactionType.SELL
            )

            signals.append(
                Signal(
                    tradingsymbol=leg["tradingsymbol"],
                    exchange=Exchange(self.params["exchange"]),
                    transaction_type=reverse_type,
                    quantity=leg["quantity"],
                    order_type=OrderType.MARKET,
                    product=ProductType(self.params["product"]),
                    strategy_name=self.name,
                    confidence=75.0,
                    metadata={
                        **meta,
                        "signal_type": "exit",
                        "reason": reason,
                    },
                )
            )

        self._active_legs = []
        return signals


class BearPutCreditSpreadStrategy(OptionStrategyBase):
    """Sell OTM put spread - collects premium, profits if price stays above short strike."""

    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        defaults = {
            "short_put_delta": 0.20,  # Sell ~20 delta put (OTM)
            "long_put_delta": 0.10,   # Buy ~10 delta put (further OTM, protection)
            "min_credit_per_spread": 20.0,  # Minimum credit to collect
            "max_risk_per_spread": 500.0,    # Max loss per spread
            "profit_target_pct": 50.0,   # Close at 50% of max profit
            "adjustment_threshold_pct": 20.0,  # Adjust if price moves this %
        }
        merged = {**defaults, **(params or {})}
        super().__init__("bear_put_credit_spread", merged)

    async def on_tick(self, ticks: list[Tick]) -> list[Signal]:
        exit_signals = await self._evaluate_exit(ticks)
        if exit_signals:
            return exit_signals
        if not self._option_chain or self._active_legs:
            return []
        return self._evaluate_entry()

    async def on_bar(self, instrument_token: int, bar: pd.Series) -> list[Signal]:
        exit_signals = await self._evaluate_exit([])
        if exit_signals:
            return exit_signals
        if not self._option_chain or self._active_legs:
            return []
        return self._evaluate_entry()

    def generate_signal(self, data: pd.DataFrame, instrument_token: int) -> Optional[Signal]:
        return None

    async def _evaluate_exit(self, ticks: list[Tick]) -> list[Signal]:
        """Check profit target and adjustment conditions."""
        if not self._active_legs or not self._option_chain:
            return []

        chain = self._option_chain
        first_leg = self._active_legs[0]
        meta = first_leg.get("metadata", {})

        # Get the short put strike (sold lower strike)
        short_put_strike = meta.get("short_put_strike", 0)
        long_put_strike = meta.get("long_put_strike", 0)
        entry_credit = meta.get("net_credit", 0)
        spot = chain.spot_price

        # Condition 1: Price moved down significantly (assignment risk)
        if spot < short_put_strike:
            distance_pct = ((short_put_strike - spot) / short_put_strike) * 100
            if distance_pct > self.params.get("adjustment_threshold_pct", 20.0):
                logger.warning(
                    "bear_put_credit_adjustment_needed",
                    spot=spot,
                    short_strike=short_put_strike,
                    distance_pct=distance_pct,
                )
                return self._create_close_signals("adjustment_needed_down_move")

        # Condition 2: Profit target check based on premium decay
        if entry_credit > 0 and short_put_strike > 0 and long_put_strike > 0:
            short_put_current = self._get_option_price(chain, short_put_strike, "PE")
            long_put_current = self._get_option_price(chain, long_put_strike, "PE")
            if short_put_current is not None and long_put_current is not None:
                current_spread_value = short_put_current - long_put_current
                profit_captured = entry_credit - current_spread_value
                target_profit = entry_credit * (self.params["profit_target_pct"] / 100)
                if profit_captured >= target_profit:
                    logger.info(
                        "bear_put_credit_profit_target",
                        profit_captured=round(profit_captured, 2),
                        target=round(target_profit, 2),
                    )
                    return self._create_close_signals("profit_target_hit")

        return []

    def _get_option_price(self, chain, strike: float, opt_type: str) -> float | None:
        """Get option price from chain by strike and type."""
        if not hasattr(chain, 'strikes') and hasattr(chain, 'options'):
            for opt in chain.options:
                if opt.strike == strike and opt.instrument_type == opt_type:
                    return opt.last_price
        elif hasattr(chain, 'strikes'):
            key = f"{strike}_{opt_type}"
            if key in chain.strikes:
                return chain.strikes[key].price
        return None

    def _evaluate_entry(self) -> list[Signal]:
        chain = self._option_chain
        if not chain:
            return []

        # Find short put (OTM - lower strike)
        short_put = self._find_strike_by_delta(self.params["short_put_delta"], "PE", chain)
        # Find long put (further OTM - even lower strike)
        long_put = self._find_strike_by_delta(self.params["long_put_delta"], "PE", chain)

        if not short_put or not long_put or long_put.strike >= short_put.strike:
            return []

        # Credit = premium collected (sell) - premium paid (buy)
        net_credit = short_put.last_price - long_put.last_price
        if net_credit < self.params["min_credit_per_spread"]:
            logger.debug("bear_put_credit_insufficient", credit=net_credit)
            return []

        max_loss = (short_put.strike - long_put.strike) - net_credit
        if max_loss > self.params["max_risk_per_spread"]:
            logger.warning("bear_put_credit_max_risk_exceeded", max_loss=max_loss)
            return []

        profit_target = net_credit * (self.params["profit_target_pct"] / 100)

        stop_loss = round(0.3 * max_loss, 2)
        meta = {
            "strategy_type": "bear_put_credit_spread",
            "net_credit": round(net_credit, 2),
            "max_profit": round(net_credit * self.params["lot_size"], 2),
            "max_loss_per_spread": round(max_loss, 2),
            "profit_target": round(profit_target, 2),
            "short_put_strike": short_put.strike,
            "long_put_strike": long_put.strike,
            "spot_price": chain.spot_price,
            "atm_iv": chain.atm_iv,
            "stop_loss": stop_loss,
        }

        signals = [
            self._create_signal(short_put, TransactionType.SELL, 60.0, {**meta, "leg": "short_put"}),
            self._create_signal(long_put, TransactionType.BUY, 60.0, {**meta, "leg": "long_put"}),
        ]
        self._active_legs = [s.model_dump() for s in signals]
        logger.info("bear_put_credit_spread_entry", meta=meta)
        return signals

    def _create_close_signals(self, reason: str) -> list[Signal]:
        """Create closing signals for active legs."""
        if not self._active_legs:
            return []

        signals = []
        for leg in self._active_legs:
            meta = leg.get("metadata", {})
            reverse_type = (
                TransactionType.BUY
                if leg["transaction_type"] == TransactionType.SELL
                else TransactionType.SELL
            )

            signals.append(
                Signal(
                    tradingsymbol=leg["tradingsymbol"],
                    exchange=Exchange(self.params["exchange"]),
                    transaction_type=reverse_type,
                    quantity=leg["quantity"],
                    order_type=OrderType.MARKET,
                    product=ProductType(self.params["product"]),
                    strategy_name=self.name,
                    confidence=75.0,
                    metadata={
                        **meta,
                        "signal_type": "exit",
                        "reason": reason,
                    },
                )
            )

        self._active_legs = []
        return signals
