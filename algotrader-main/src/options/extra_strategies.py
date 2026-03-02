"""
Additional F&O Live Trading Strategies
=======================================

Contains live trading strategy classes for:
- BullPutSpreadStrategy (credit spread, bullish)
- BearCallSpreadStrategy (credit spread, bearish)
- ShortStraddleStrategy (sell ATM call + put)
- ShortStrangleStrategy (sell OTM call + put)
- IronButterflyStrategy (ATM short strikes with wings)
- CoveredCallStrategy (long stock + short call)
- ProtectivePutStrategy (long stock + long put)
- CalendarSpreadStrategy (different expiry, same strike)
"""

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
from src.options.strategies import OptionStrategyBase
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ────────────────────────────────────────────────────────────────
# Bull Put Spread (Credit Spread, Bullish)
# ────────────────────────────────────────────────────────────────

class BullPutSpreadStrategy(OptionStrategyBase):
    """
    Bull Put Spread: Sell higher strike put, buy lower strike put.
    
    Profit when underlying stays above short put strike.
    Max profit = net credit received.
    Max loss = spread width - net credit.
    """
    
    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        defaults = {
            "sell_strike_offset": -100,    # Sell put slightly OTM
            "buy_strike_offset": -300,     # Buy put further OTM (protection)
            "min_credit": 30.0,            # Minimum net credit per lot
            "profit_target_pct": 50.0,     # Close at 50% of max profit
            "stop_loss_pct": 100.0,        # Stop loss at 100% of max profit (double the credit)
        }
        merged = {**defaults, **(params or {})}
        super().__init__("bull_put_spread", merged)

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
        if not self._active_legs or not self._option_chain:
            return []
        
        chain = self._option_chain
        sell_leg = next((l for l in self._active_legs if l.get("metadata", {}).get("leg") == "sell_pe"), None)
        
        if not sell_leg:
            return []
        
        sell_strike = sell_leg.get("metadata", {}).get("sell_strike", 0)
        net_credit = sell_leg.get("metadata", {}).get("net_credit", 0)
        
        # Profit target: if spot is well above short strike
        if chain.spot_price > sell_strike + (sell_strike * 0.01):  # 1% above short strike
            profit_pct = (sell_strike + net_credit - chain.spot_price) / net_credit * 100 if net_credit > 0 else 0
            if profit_pct >= self.params["profit_target_pct"]:
                logger.info("bull_put_profit_target_hit", spot=chain.spot_price, strike=sell_strike)
                return self._create_close_signals("profit_target_hit")
        
        # Stop loss: if spot drops below buy strike
        buy_strike = sell_leg.get("metadata", {}).get("buy_strike", 0)
        if chain.spot_price < buy_strike:
            logger.warning("bull_put_stop_loss_hit", spot=chain.spot_price, strike=buy_strike)
            return self._create_close_signals("stop_loss_hit")
        
        return []

    def _evaluate_entry(self) -> list[Signal]:
        chain = self._option_chain
        if not chain:
            return []

        sell_strike = chain.atm_strike + self.params["sell_strike_offset"]
        buy_strike = chain.atm_strike + self.params["buy_strike_offset"]

        sell_pe = self._find_strike_nearest(sell_strike, "PE", chain)
        buy_pe = self._find_strike_nearest(buy_strike, "PE", chain)

        if not sell_pe or not buy_pe or buy_pe.strike >= sell_pe.strike:
            return []

        net_credit = sell_pe.last_price - buy_pe.last_price
        if net_credit < self.params["min_credit"] / self.params["lot_size"]:
            return []

        spread_width = sell_pe.strike - buy_pe.strike
        max_loss = spread_width - net_credit
        
        stop_loss = round(net_credit * (self.params["stop_loss_pct"] / 100), 2)
        meta = {
            "strategy_type": "bull_put_spread",
            "net_credit": round(net_credit, 2),
            "max_loss": round(max_loss, 2),
            "spread_width": spread_width,
            "profit_target": round(net_credit * (self.params["profit_target_pct"] / 100), 2),
            "sell_strike": sell_pe.strike,
            "buy_strike": buy_pe.strike,
            "spot_price": chain.spot_price,
            "stop_loss": stop_loss,
        }

        signals = [
            self._create_signal(sell_pe, TransactionType.SELL, 55.0, {**meta, "leg": "sell_pe"}),
            self._create_signal(buy_pe, TransactionType.BUY, 55.0, {**meta, "leg": "buy_pe"}),
        ]
        self._active_legs = [s.model_dump() for s in signals]
        logger.info("bull_put_spread_entry", meta=meta)
        return signals

    def _create_close_signals(self, reason: str) -> list[Signal]:
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
                    metadata={**meta, "signal_type": "exit", "reason": reason},
                )
            )
        self._active_legs = []
        return signals


# ────────────────────────────────────────────────────────────────
# Bear Call Spread (Credit Spread, Bearish)
# ────────────────────────────────────────────────────────────────

class BearCallSpreadStrategy(OptionStrategyBase):
    """
    Bear Call Spread: Sell lower strike call, buy higher strike call.
    
    Profit when underlying stays below short call strike.
    Max profit = net credit received.
    Max loss = spread width - net credit.
    """
    
    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        defaults = {
            "sell_strike_offset": 100,     # Sell call slightly OTM
            "buy_strike_offset": 300,      # Buy call further OTM (protection)
            "min_credit": 30.0,            # Minimum net credit per lot
            "profit_target_pct": 50.0,
            "stop_loss_pct": 100.0,
        }
        merged = {**defaults, **(params or {})}
        super().__init__("bear_call_spread", merged)

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
        if not self._active_legs or not self._option_chain:
            return []
        
        chain = self._option_chain
        sell_leg = next((l for l in self._active_legs if l.get("metadata", {}).get("leg") == "sell_ce"), None)
        
        if not sell_leg:
            return []
        
        sell_strike = sell_leg.get("metadata", {}).get("sell_strike", 0)
        net_credit = sell_leg.get("metadata", {}).get("net_credit", 0)
        
        # Profit target: if spot is well below short strike
        if chain.spot_price < sell_strike - (sell_strike * 0.01):
            profit_pct = (chain.spot_price - sell_strike + net_credit) / net_credit * 100 if net_credit > 0 else 0
            if profit_pct >= self.params["profit_target_pct"]:
                logger.info("bear_call_profit_target_hit", spot=chain.spot_price, strike=sell_strike)
                return self._create_close_signals("profit_target_hit")
        
        # Stop loss: if spot rises above buy strike
        buy_strike = sell_leg.get("metadata", {}).get("buy_strike", 0)
        if chain.spot_price > buy_strike:
            logger.warning("bear_call_stop_loss_hit", spot=chain.spot_price, strike=buy_strike)
            return self._create_close_signals("stop_loss_hit")
        
        return []

    def _evaluate_entry(self) -> list[Signal]:
        chain = self._option_chain
        if not chain:
            return []

        sell_strike = chain.atm_strike + self.params["sell_strike_offset"]
        buy_strike = chain.atm_strike + self.params["buy_strike_offset"]

        sell_ce = self._find_strike_nearest(sell_strike, "CE", chain)
        buy_ce = self._find_strike_nearest(buy_strike, "CE", chain)

        if not sell_ce or not buy_ce or buy_ce.strike <= sell_ce.strike:
            return []

        net_credit = sell_ce.last_price - buy_ce.last_price
        if net_credit < self.params["min_credit"] / self.params["lot_size"]:
            return []

        spread_width = buy_ce.strike - sell_ce.strike
        max_loss = spread_width - net_credit
        
        stop_loss = round(net_credit * (self.params["stop_loss_pct"] / 100), 2)
        meta = {
            "strategy_type": "bear_call_spread",
            "net_credit": round(net_credit, 2),
            "max_loss": round(max_loss, 2),
            "spread_width": spread_width,
            "profit_target": round(net_credit * (self.params["profit_target_pct"] / 100), 2),
            "sell_strike": sell_ce.strike,
            "buy_strike": buy_ce.strike,
            "spot_price": chain.spot_price,
            "stop_loss": stop_loss,
        }

        signals = [
            self._create_signal(sell_ce, TransactionType.SELL, 55.0, {**meta, "leg": "sell_ce"}),
            self._create_signal(buy_ce, TransactionType.BUY, 55.0, {**meta, "leg": "buy_ce"}),
        ]
        self._active_legs = [s.model_dump() for s in signals]
        logger.info("bear_call_spread_entry", meta=meta)
        return signals

    def _create_close_signals(self, reason: str) -> list[Signal]:
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
                    metadata={**meta, "signal_type": "exit", "reason": reason},
                )
            )
        self._active_legs = []
        return signals


# ────────────────────────────────────────────────────────────────
# Short Straddle (Sell ATM Call + Put)
# ────────────────────────────────────────────────────────────────

class ShortStraddleStrategy(OptionStrategyBase):
    """
    Short Straddle: Sell ATM call + sell ATM put.
    
    Profit from time decay when underlying stays near ATM.
    Max profit = total premium received.
    Max loss = unlimited (for practical purposes, limited by adjustments).
    Best in high IV environments expecting mean reversion.
    """
    
    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        defaults = {
            "min_iv": 10.0,  # Lowered from 20 - typical NIFTY IV is 10-30%
            "max_iv": 80.0,
            "adjustment_threshold_pct": 20.0,  # Adjust if spot moves 20% from center
            "profit_target_pct": 50.0,
            "stop_loss_pct": 100.0,
        }
        merged = {**defaults, **(params or {})}
        super().__init__("short_straddle", merged)

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
        if not self._active_legs or not self._option_chain:
            return []
        
        chain = self._option_chain
        first_leg = self._active_legs[0]
        meta = first_leg.get("metadata", {})
        
        atm_strike = meta.get("atm_strike", 0)
        total_premium = meta.get("total_premium", 0)
        
        # Check adjustment threshold
        if atm_strike > 0:
            distance_pct = abs(chain.spot_price - atm_strike) / atm_strike * 100
            if distance_pct > self.params["adjustment_threshold_pct"]:
                logger.warning("short_straddle_adjustment_needed", distance_pct=distance_pct)
                return self._create_close_signals("adjustment_threshold_exceeded")
        
        return []

    def _evaluate_entry(self) -> list[Signal]:
        chain = self._option_chain
        if not chain:
            logger.debug("short_straddle_no_chain")
            return []

        if chain.atm_iv < self.params["min_iv"] or chain.atm_iv > self.params["max_iv"]:
            logger.info(
                "short_straddle_iv_out_of_range",
                atm_iv=round(chain.atm_iv, 2),
                min_iv=self.params["min_iv"],
                max_iv=self.params["max_iv"],
            )
            return []

        ce = self._find_strike_nearest(chain.atm_strike, "CE", chain)
        pe = self._find_strike_nearest(chain.atm_strike, "PE", chain)

        if not ce or not pe:
            logger.info("short_straddle_no_atm_strikes", atm=chain.atm_strike)
            return []

        total_premium = ce.last_price + pe.last_price
        
        stop_loss = round(total_premium * (self.params["stop_loss_pct"] / 100), 2)
        meta = {
            "strategy_type": "short_straddle",
            "total_premium": round(total_premium, 2),
            "profit_target": round(total_premium * (self.params["profit_target_pct"] / 100), 2),
            "atm_strike": chain.atm_strike,
            "ce_strike": ce.strike,
            "pe_strike": pe.strike,
            "spot_price": chain.spot_price,
            "atm_iv": chain.atm_iv,
            "stop_loss": stop_loss,
        }

        signals = [
            self._create_signal(ce, TransactionType.SELL, 55.0, {**meta, "leg": "sell_ce"}),
            self._create_signal(pe, TransactionType.SELL, 55.0, {**meta, "leg": "sell_pe"}),
        ]
        self._active_legs = [s.model_dump() for s in signals]
        logger.info("short_straddle_entry", premium=total_premium, meta=meta)
        return signals

    def _create_close_signals(self, reason: str) -> list[Signal]:
        if not self._active_legs:
            return []
        
        signals = []
        for leg in self._active_legs:
            meta = leg.get("metadata", {})
            signals.append(
                Signal(
                    tradingsymbol=leg["tradingsymbol"],
                    exchange=Exchange(self.params["exchange"]),
                    transaction_type=TransactionType.BUY,  # Buy back to close
                    quantity=leg["quantity"],
                    order_type=OrderType.MARKET,
                    product=ProductType(self.params["product"]),
                    strategy_name=self.name,
                    confidence=75.0,
                    metadata={**meta, "signal_type": "exit", "reason": reason},
                )
            )
        self._active_legs = []
        return signals


# ────────────────────────────────────────────────────────────────
# Short Strangle (Sell OTM Call + Put)
# ────────────────────────────────────────────────────────────────

class ShortStrangleStrategy(OptionStrategyBase):
    """
    Short Strangle: Sell OTM call + sell OTM put.
    
    Wider profit zone than straddle, lower premium received.
    Profit from time decay when underlying stays between strikes.
    """
    
    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        defaults = {
            "strangle_offset_pct": 3.0,    # OTM percentage from spot
            "min_iv": 20.0,
            "max_iv": 80.0,
            "adjustment_threshold_pct": 25.0,
            "profit_target_pct": 50.0,
            "stop_loss_pct": 100.0,
        }
        merged = {**defaults, **(params or {})}
        super().__init__("short_strangle", merged)

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
        if not self._active_legs or not self._option_chain:
            return []
        
        chain = self._option_chain
        first_leg = self._active_legs[0]
        meta = first_leg.get("metadata", {})
        
        ce_strike = meta.get("ce_strike", 0)
        pe_strike = meta.get("pe_strike", 0)
        
        # Check if spot breached either strike
        if chain.spot_price > ce_strike or chain.spot_price < pe_strike:
            logger.warning("short_strangle_breach", spot=chain.spot_price, ce=ce_strike, pe=pe_strike)
            return self._create_close_signals("strike_breach")
        
        return []

    def _evaluate_entry(self) -> list[Signal]:
        chain = self._option_chain
        if not chain:
            return []

        if chain.atm_iv < self.params["min_iv"] or chain.atm_iv > self.params["max_iv"]:
            return []

        offset = chain.spot_price * self.params["strangle_offset_pct"] / 100
        ce = self._find_strike_nearest(chain.spot_price + offset, "CE", chain)
        pe = self._find_strike_nearest(chain.spot_price - offset, "PE", chain)

        if not ce or not pe:
            return []

        total_premium = ce.last_price + pe.last_price
        
        stop_loss = round(total_premium * (self.params["stop_loss_pct"] / 100), 2)
        meta = {
            "strategy_type": "short_strangle",
            "total_premium": round(total_premium, 2),
            "profit_target": round(total_premium * (self.params["profit_target_pct"] / 100), 2),
            "ce_strike": ce.strike,
            "pe_strike": pe.strike,
            "spot_price": chain.spot_price,
            "atm_iv": chain.atm_iv,
            "stop_loss": stop_loss,
        }

        signals = [
            self._create_signal(ce, TransactionType.SELL, 55.0, {**meta, "leg": "sell_ce"}),
            self._create_signal(pe, TransactionType.SELL, 55.0, {**meta, "leg": "sell_pe"}),
        ]
        self._active_legs = [s.model_dump() for s in signals]
        logger.info("short_strangle_entry", premium=total_premium, meta=meta)
        return signals

    def _create_close_signals(self, reason: str) -> list[Signal]:
        if not self._active_legs:
            return []
        
        signals = []
        for leg in self._active_legs:
            meta = leg.get("metadata", {})
            signals.append(
                Signal(
                    tradingsymbol=leg["tradingsymbol"],
                    exchange=Exchange(self.params["exchange"]),
                    transaction_type=TransactionType.BUY,
                    quantity=leg["quantity"],
                    order_type=OrderType.MARKET,
                    product=ProductType(self.params["product"]),
                    strategy_name=self.name,
                    confidence=75.0,
                    metadata={**meta, "signal_type": "exit", "reason": reason},
                )
            )
        self._active_legs = []
        return signals


# ────────────────────────────────────────────────────────────────
# Iron Butterfly (ATM Short Strikes with Wings)
# ────────────────────────────────────────────────────────────────

class IronButterflyStrategy(OptionStrategyBase):
    """
    Iron Butterfly: Sell ATM call + ATM put, buy OTM call + OTM put.
    
    Similar to short straddle but with defined risk.
    Max profit = net credit received.
    Max loss = wing spread width - net credit.
    """
    
    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        defaults = {
            "wing_width": 200,          # Distance to OTM wings
            "min_iv": 20.0,
            "max_iv": 80.0,
            "min_premium": 80.0,        # Minimum net credit per lot
            "profit_target_pct": 50.0,
            "adjustment_threshold_pct": 15.0,
        }
        merged = {**defaults, **(params or {})}
        super().__init__("iron_butterfly", merged)

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
        if not self._active_legs or not self._option_chain:
            return []
        
        chain = self._option_chain
        first_leg = self._active_legs[0]
        meta = first_leg.get("metadata", {})
        
        center_strike = meta.get("center_strike", 0)
        
        # Check adjustment threshold
        if center_strike > 0:
            distance_pct = abs(chain.spot_price - center_strike) / center_strike * 100
            if distance_pct > self.params["adjustment_threshold_pct"]:
                logger.warning("iron_butterfly_adjustment_needed", distance_pct=distance_pct)
                return self._create_close_signals("adjustment_threshold_exceeded")
        
        return []

    def _evaluate_entry(self) -> list[Signal]:
        chain = self._option_chain
        if not chain:
            return []

        if chain.atm_iv < self.params["min_iv"] or chain.atm_iv > self.params["max_iv"]:
            return []

        center = chain.atm_strike
        wing_width = self.params["wing_width"]

        # ATM short strikes
        sell_ce = self._find_strike_nearest(center, "CE", chain)
        sell_pe = self._find_strike_nearest(center, "PE", chain)
        # OTM long wings
        buy_ce = self._find_strike_nearest(center + wing_width, "CE", chain)
        buy_pe = self._find_strike_nearest(center - wing_width, "PE", chain)

        if not all([sell_ce, sell_pe, buy_ce, buy_pe]):
            return []

        net_credit = (sell_ce.last_price + sell_pe.last_price) - (buy_ce.last_price + buy_pe.last_price)
        if net_credit < self.params["min_premium"] / self.params["lot_size"]:
            return []

        max_loss = wing_width - net_credit
        
        stop_loss = round(net_credit * 0.5, 2)  # 50% of credit as stop
        meta = {
            "strategy_type": "iron_butterfly",
            "net_credit": round(net_credit, 2),
            "max_loss": round(max_loss, 2),
            "profit_target": round(net_credit * (self.params["profit_target_pct"] / 100), 2),
            "center_strike": center,
            "buy_ce_strike": buy_ce.strike,
            "buy_pe_strike": buy_pe.strike,
            "spot_price": chain.spot_price,
            "atm_iv": chain.atm_iv,
            "stop_loss": stop_loss,
        }

        signals = [
            self._create_signal(sell_ce, TransactionType.SELL, 55.0, {**meta, "leg": "sell_ce"}),
            self._create_signal(sell_pe, TransactionType.SELL, 55.0, {**meta, "leg": "sell_pe"}),
            self._create_signal(buy_ce, TransactionType.BUY, 55.0, {**meta, "leg": "buy_ce"}),
            self._create_signal(buy_pe, TransactionType.BUY, 55.0, {**meta, "leg": "buy_pe"}),
        ]
        self._active_legs = [s.model_dump() for s in signals]
        logger.info("iron_butterfly_entry", premium=net_credit, meta=meta)
        return signals

    def _create_close_signals(self, reason: str) -> list[Signal]:
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
                    metadata={**meta, "signal_type": "exit", "reason": reason},
                )
            )
        self._active_legs = []
        return signals


# ────────────────────────────────────────────────────────────────
# Covered Call (Long Stock + Short Call)
# ────────────────────────────────────────────────────────────────

class CoveredCallStrategy(OptionStrategyBase):
    """
    Covered Call: Long stock + sell OTM call.
    
    Generate income from stock holdings.
    Max profit = premium + (call strike - stock price).
    Risk = full downside on stock minus premium received.
    """
    
    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        defaults = {
            "call_delta": 0.30,          # Sell 30-delta OTM call
            "min_premium_pct": 1.0,      # Minimum 1% of stock price
            "profit_target_pct": 80.0,   # Close call at 80% profit
            "roll_dte": 7,               # Roll when 7 DTE remaining
        }
        merged = {**defaults, **(params or {})}
        super().__init__("covered_call", merged)

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
        if not self._active_legs or not self._option_chain:
            return []
        
        chain = self._option_chain
        call_leg = next((l for l in self._active_legs if l.get("metadata", {}).get("leg") == "sell_ce"), None)
        
        if not call_leg:
            return []
        
        call_strike = call_leg.get("metadata", {}).get("call_strike", 0)
        
        # If stock price exceeds call strike, consider rolling or closing
        if chain.spot_price > call_strike:
            logger.info("covered_call_itm", spot=chain.spot_price, strike=call_strike)
            return self._create_close_signals("call_itm_close")
        
        return []

    def _evaluate_entry(self) -> list[Signal]:
        chain = self._option_chain
        if not chain:
            return []

        # Find OTM call by delta
        call = self._find_strike_by_delta(self.params["call_delta"], "CE", chain)
        if not call:
            return []

        premium_pct = call.last_price / chain.spot_price * 100
        if premium_pct < self.params["min_premium_pct"]:
            return []

        stop_loss = round(call.last_price * 0.8, 2)  # 80% of premium as mental stop
        meta = {
            "strategy_type": "covered_call",
            "premium": round(call.last_price, 2),
            "premium_pct": round(premium_pct, 2),
            "call_strike": call.strike,
            "spot_price": chain.spot_price,
            "profit_target": round(call.last_price * (self.params["profit_target_pct"] / 100), 2),
            "stop_loss": stop_loss,
        }

        # Note: Stock leg would need separate handling in execution
        signals = [
            self._create_signal(call, TransactionType.SELL, 55.0, {**meta, "leg": "sell_ce"}),
        ]
        self._active_legs = [s.model_dump() for s in signals]
        logger.info("covered_call_entry", meta=meta)
        return signals

    def _create_close_signals(self, reason: str) -> list[Signal]:
        if not self._active_legs:
            return []
        
        signals = []
        for leg in self._active_legs:
            meta = leg.get("metadata", {})
            signals.append(
                Signal(
                    tradingsymbol=leg["tradingsymbol"],
                    exchange=Exchange(self.params["exchange"]),
                    transaction_type=TransactionType.BUY,
                    quantity=leg["quantity"],
                    order_type=OrderType.MARKET,
                    product=ProductType(self.params["product"]),
                    strategy_name=self.name,
                    confidence=75.0,
                    metadata={**meta, "signal_type": "exit", "reason": reason},
                )
            )
        self._active_legs = []
        return signals


# ────────────────────────────────────────────────────────────────
# Protective Put (Long Stock + Long Put)
# ────────────────────────────────────────────────────────────────

class ProtectivePutStrategy(OptionStrategyBase):
    """
    Protective Put: Long stock + buy OTM put.
    
    Insurance for stock holdings.
    Max loss = stock price - put strike + premium paid.
    Unlimited upside minus premium cost.
    """
    
    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        defaults = {
            "put_delta": 0.30,           # Buy 30-delta OTM put
            "max_premium_pct": 2.0,      # Maximum 2% of stock price for insurance
            "protection_level_pct": 5.0, # Protect against 5% drop
        }
        merged = {**defaults, **(params or {})}
        super().__init__("protective_put", merged)

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
        if not self._active_legs or not self._option_chain:
            return []
        
        chain = self._option_chain
        put_leg = next((l for l in self._active_legs if l.get("metadata", {}).get("leg") == "buy_pe"), None)
        
        if not put_leg:
            return []
        
        put_strike = put_leg.get("metadata", {}).get("put_strike", 0)
        
        # If stock drops below put strike, the put is ITM - could exercise or close
        if chain.spot_price < put_strike * 0.98:  # 2% below strike
            logger.info("protective_put_activated", spot=chain.spot_price, strike=put_strike)
            return self._create_close_signals("protection_activated")
        
        return []

    def _evaluate_entry(self) -> list[Signal]:
        chain = self._option_chain
        if not chain:
            return []

        # Calculate target put strike based on protection level
        protection_strike = chain.spot_price * (1 - self.params["protection_level_pct"] / 100)
        
        put = self._find_strike_nearest(protection_strike, "PE", chain)
        if not put:
            return []

        premium_pct = put.last_price / chain.spot_price * 100
        if premium_pct > self.params["max_premium_pct"]:
            return []

        meta = {
            "strategy_type": "protective_put",
            "premium": round(put.last_price, 2),
            "premium_pct": round(premium_pct, 2),
            "put_strike": put.strike,
            "spot_price": chain.spot_price,
            "protection_level_pct": self.params["protection_level_pct"],
        }

        signals = [
            self._create_signal(put, TransactionType.BUY, 55.0, {**meta, "leg": "buy_pe"}),
        ]
        self._active_legs = [s.model_dump() for s in signals]
        logger.info("protective_put_entry", meta=meta)
        return signals

    def _create_close_signals(self, reason: str) -> list[Signal]:
        if not self._active_legs:
            return []
        
        signals = []
        for leg in self._active_legs:
            meta = leg.get("metadata", {})
            signals.append(
                Signal(
                    tradingsymbol=leg["tradingsymbol"],
                    exchange=Exchange(self.params["exchange"]),
                    transaction_type=TransactionType.SELL,
                    quantity=leg["quantity"],
                    order_type=OrderType.MARKET,
                    product=ProductType(self.params["product"]),
                    strategy_name=self.name,
                    confidence=75.0,
                    metadata={**meta, "signal_type": "exit", "reason": reason},
                )
            )
        self._active_legs = []
        return signals


# ────────────────────────────────────────────────────────────────
# Calendar Spread (Different Expiry, Same Strike)
# ────────────────────────────────────────────────────────────────

class CalendarSpreadStrategy(OptionStrategyBase):
    """
    Calendar Spread: Sell near-term option, buy far-term option (same strike).
    
    Profit from time decay differential between expirations.
    Max profit when underlying at strike at near-term expiry.
    """
    
    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        defaults = {
            "option_type": "CE",         # CE or PE
            "strike_offset": 0,          # ATM by default
            "min_time_spread_days": 7,   # Minimum days between expiries
            "profit_target_pct": 30.0,   # Lower target due to complexity
            "adjustment_threshold_pct": 10.0,
        }
        merged = {**defaults, **(params or {})}
        super().__init__("calendar_spread", merged)

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
        if not self._active_legs or not self._option_chain:
            return []
        
        chain = self._option_chain
        first_leg = self._active_legs[0]
        meta = first_leg.get("metadata", {})
        
        strike = meta.get("strike", 0)
        
        # Check if spot moved significantly from strike
        if strike > 0:
            distance_pct = abs(chain.spot_price - strike) / strike * 100
            if distance_pct > self.params["adjustment_threshold_pct"]:
                logger.warning("calendar_adjustment_needed", distance_pct=distance_pct)
                return self._create_close_signals("adjustment_threshold_exceeded")
        
        return []

    def _evaluate_entry(self) -> list[Signal]:
        chain = self._option_chain
        if not chain:
            return []

        target_strike = chain.atm_strike + self.params["strike_offset"]
        option_type = self.params["option_type"]
        
        # Find near-term option to sell
        opt = self._find_strike_nearest(target_strike, option_type, chain)
        if not opt:
            return []

        # Note: Calendar spreads require two different expiries
        # The far-term leg would need separate chain data
        # For now, we signal the near-term sell with metadata indicating calendar intent
        
        meta = {
            "strategy_type": "calendar_spread",
            "option_type": option_type,
            "strike": opt.strike,
            "near_premium": round(opt.last_price, 2),
            "spot_price": chain.spot_price,
            "calendar_note": "Requires manual far-term leg setup",
        }

        signals = [
            self._create_signal(opt, TransactionType.SELL, 50.0, {**meta, "leg": f"sell_{option_type.lower()}"}),
        ]
        self._active_legs = [s.model_dump() for s in signals]
        logger.info("calendar_spread_entry", meta=meta)
        return signals

    def _create_close_signals(self, reason: str) -> list[Signal]:
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
                    metadata={**meta, "signal_type": "exit", "reason": reason},
                )
            )
        self._active_legs = []
        return signals
