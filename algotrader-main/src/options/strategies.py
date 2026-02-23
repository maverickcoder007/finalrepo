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
from src.options.greeks import BlackScholes
from src.strategy.base import BaseStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OptionStrategyBase(BaseStrategy):
    def __init__(self, name: str, params: Optional[dict[str, Any]] = None) -> None:
        defaults = {
            "underlying": "NIFTY",
            "exchange": "NFO",
            "product": "NRML",
            "lot_size": 50,
            "lots": 1,
            "max_iv_entry": 50.0,
            "min_iv_entry": 10.0,
        }
        merged = {**defaults, **(params or {})}
        super().__init__(name, merged)
        self._option_chain: Optional[OptionChainData] = None
        self._active_legs: list[dict[str, Any]] = []

    def update_chain(self, chain: OptionChainData) -> None:
        self._option_chain = chain

    def _find_strike_by_delta(
        self, target_delta: float, option_type: str, chain: OptionChainData
    ) -> Optional[OptionContract]:
        best: Optional[OptionContract] = None
        best_diff = float("inf")
        for entry in chain.entries:
            contract = entry.ce if option_type == "CE" else entry.pe
            if contract and contract.last_price > 0:
                diff = abs(abs(contract.delta) - abs(target_delta))
                if diff < best_diff:
                    best_diff = diff
                    best = contract
        return best

    def _find_strike_nearest(
        self, target_strike: float, option_type: str, chain: OptionChainData
    ) -> Optional[OptionContract]:
        best: Optional[OptionContract] = None
        best_diff = float("inf")
        for entry in chain.entries:
            contract = entry.ce if option_type == "CE" else entry.pe
            if contract and contract.last_price > 0:
                diff = abs(contract.strike - target_strike)
                if diff < best_diff:
                    best_diff = diff
                    best = contract
        return best

    def _create_signal(
        self,
        contract: OptionContract,
        transaction_type: TransactionType,
        confidence: float = 50.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Signal:
        quantity = self.params["lot_size"] * self.params["lots"]
        return Signal(
            tradingsymbol=contract.tradingsymbol,
            exchange=Exchange(self.params["exchange"]),
            transaction_type=transaction_type,
            quantity=quantity,
            price=contract.last_price,
            order_type=OrderType.LIMIT,
            product=ProductType(self.params["product"]),
            strategy_name=self.name,
            confidence=confidence,
            metadata=metadata or {},
        )


class IronCondorStrategy(OptionStrategyBase):
    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        defaults = {
            "sell_ce_delta": 0.20,
            "buy_ce_delta": 0.10,
            "sell_pe_delta": 0.20,
            "buy_pe_delta": 0.10,
            "min_premium": 50.0,
            "max_loss_per_lot": 5000.0,
            "iv_rank_threshold": 30.0,
            "profit_target_pct": 50.0,  # Close at 50% of max profit
        }
        merged = {**defaults, **(params or {})}
        super().__init__("iron_condor", merged)

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
        """Check profit target and exit conditions based on time decay."""
        if not self._active_legs or not self._option_chain:
            return []
        
        chain = self._option_chain
        if not self._active_legs:
            return []
        
        # Get position metadata to check profit target
        first_leg = self._active_legs[0]
        meta = first_leg.get("metadata", {})
        max_profit = meta.get("profit_target", 0)
        
        # Simplified profit check: if significant time decay or price moved to profit zone
        # In real implementation, would track entry prices and current mark prices
        # For now, use spot price distance to strangle strikes
        sell_ce_strike = meta.get("sell_ce_strike", 0)
        sell_pe_strike = meta.get("sell_pe_strike", 0)
        spot = chain.spot_price
        
        # If spot is between short strikes, we're profitable
        if sell_pe_strike < spot < sell_ce_strike:
            distance_to_nearest = min(spot - sell_pe_strike, sell_ce_strike - spot)
            # Close if we have reasonable profit cushion
            if distance_to_nearest > 50:  # At least 50 points profit buildup
                logger.info("iron_condor_profit_target_hit", spot=spot, profit_margin=distance_to_nearest)
                return self._create_close_signals("profit_target_hit")
        
        return []

    def _create_close_signals(self, reason: str) -> list[Signal]:
        """Create closing signals for all legs."""
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

    def _evaluate_entry(self) -> list[Signal]:
        chain = self._option_chain
        if not chain or chain.atm_iv < self.params.get("iv_rank_threshold", 0):
            return []

        sell_ce = self._find_strike_by_delta(self.params["sell_ce_delta"], "CE", chain)
        buy_ce = self._find_strike_by_delta(self.params["buy_ce_delta"], "CE", chain)
        sell_pe = self._find_strike_by_delta(self.params["sell_pe_delta"], "PE", chain)
        buy_pe = self._find_strike_by_delta(self.params["buy_pe_delta"], "PE", chain)

        if not all([sell_ce, buy_ce, sell_pe, buy_pe]):
            return []

        if buy_ce.strike <= sell_ce.strike or buy_pe.strike >= sell_pe.strike:
            return []

        net_premium = (sell_ce.last_price + sell_pe.last_price) - (buy_ce.last_price + buy_pe.last_price)
        if net_premium < self.params["min_premium"] / self.params["lot_size"]:
            return []

        # Calculate and validate maximum loss
        call_spread_width = sell_ce.strike - buy_ce.strike
        put_spread_width = sell_pe.strike - buy_pe.strike
        max_loss = min(call_spread_width, put_spread_width) - net_premium
        max_loss_per_lot = max_loss * self.params["lot_size"]

        if max_loss_per_lot > self.params.get("max_loss_per_lot", float("inf")):
            logger.warning(
                "iron_condor_max_loss_exceeded",
                max_loss=max_loss_per_lot,
                threshold=self.params.get("max_loss_per_lot"),
                spread_width=min(call_spread_width, put_spread_width),
            )
            return []

        meta = {
            "strategy_type": "iron_condor",
            "net_premium": round(net_premium, 2),
            "max_loss_per_lot": round(max_loss_per_lot, 2),
            "max_loss_total": round(max_loss_per_lot * self.params["lots"], 2),
            "profit_target": round(net_premium * self.params["lot_size"] * 0.5, 2),  # 50% of max profit
            "sell_ce_strike": sell_ce.strike,
            "buy_ce_strike": buy_ce.strike,
            "sell_pe_strike": sell_pe.strike,
            "buy_pe_strike": buy_pe.strike,
            "spot_price": chain.spot_price,
            "atm_iv": chain.atm_iv,
        }

        signals = [
            self._create_signal(sell_ce, TransactionType.SELL, 60.0, {**meta, "leg": "sell_ce"}),
            self._create_signal(buy_ce, TransactionType.BUY, 60.0, {**meta, "leg": "buy_ce"}),
            self._create_signal(sell_pe, TransactionType.SELL, 60.0, {**meta, "leg": "sell_pe"}),
            self._create_signal(buy_pe, TransactionType.BUY, 60.0, {**meta, "leg": "buy_pe"}),
        ]
        self._active_legs = [s.model_dump() for s in signals]
        logger.info("iron_condor_entry", premium=net_premium, meta=meta)
        return signals


class StraddleStrangleStrategy(OptionStrategyBase):
    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        defaults = {
            "mode": "straddle",
            "strangle_offset_pct": 2.0,
            "direction": "sell",
            "min_iv": 15.0,
            "max_iv": 80.0,
            "adjustment_threshold_pct": 30.0,
            "profit_target_pct": 50.0,  # Close at 50% of max profit for short
            "max_loss_pct": 100.0,      # Stop loss at 100% of premium for long
            "time_decay_days": 2,       # Days before expiry to close
        }
        merged = {**defaults, **(params or {})}
        super().__init__("straddle_strangle", merged)

    async def on_tick(self, ticks: list[Tick]) -> list[Signal]:
        # Check exit conditions first
        exit_signals = await self._evaluate_exit(ticks)
        if exit_signals:
            return exit_signals
        
        if not self._option_chain or self._active_legs:
            return []
        return self._evaluate_entry()

    async def on_bar(self, instrument_token: int, bar: pd.Series) -> list[Signal]:
        # Check exit conditions first
        exit_signals = await self._evaluate_exit([])
        if exit_signals:
            return exit_signals
            
        if not self._option_chain or self._active_legs:
            return []
        return self._evaluate_entry()

    def generate_signal(self, data: pd.DataFrame, instrument_token: int) -> Optional[Signal]:
        return None

    async def _evaluate_exit(self, ticks: list[Tick]) -> list[Signal]:
        """Generate exit signals based on various conditions."""
        if not self._active_legs or not self._option_chain:
            return []

        signals = []
        chain = self._option_chain
        
        # Exit condition 1: Profit target reached (for short positions)
        if self.params["direction"] == "sell":
            # Check if price moved too far (adjustment threshold)
            current_spot = chain.spot_price
            ce = self._active_legs[0] if self._active_legs else None
            if ce and "ce_strike" in ce.get("metadata", {}):
                ce_strike = ce["metadata"]["ce_strike"]
                distance_pct = abs((current_spot - ce_strike) / ce_strike) * 100
                
                if distance_pct > self.params["adjustment_threshold_pct"]:
                    # Close at loss or adjust
                    logger.warning(
                        "straddle_adjustment_needed",
                        distance_pct=distance_pct,
                        threshold=self.params["adjustment_threshold_pct"],
                    )
                    # For now, close the position
                    self._active_legs = []
                    return self._create_close_signals("adjustment_threshold_exceeded")
        
        # Exit condition 2: Max loss for long positions
        if self.params["direction"] == "buy" and self._active_legs:
            entry_premium = sum(
                leg.get("metadata", {}).get("total_premium", 0)
                for leg in self._active_legs[:1]  # stored once in metadata
            )
            if entry_premium > 0:
                # Estimate current premium from chain
                current_premium = 0.0
                for leg in self._active_legs:
                    meta = leg.get("metadata", {})
                    leg_type = meta.get("leg", "ce")
                    option_type = "CE" if leg_type == "ce" else "PE"
                    strike_val = meta.get(f"{leg_type}_strike", meta.get("ce_strike", 0))
                    opt = self._find_strike_nearest(strike_val, option_type, chain) if strike_val else None
                    if opt:
                        current_premium += opt.last_price
                # For long positions, loss = entry_premium - current_premium
                unrealized_loss = entry_premium - current_premium
                max_loss_limit = entry_premium * (self.params["max_loss_pct"] / 100)
                if unrealized_loss >= max_loss_limit:
                    logger.warning(
                        "straddle_max_loss_hit",
                        unrealized_loss=unrealized_loss,
                        max_loss_limit=max_loss_limit,
                    )
                    self._active_legs = []
                    return self._create_close_signals("max_loss_exceeded")
        
        return signals

    def _create_close_signals(self, reason: str) -> list[Signal]:
        """Create closing signals for active legs."""
        if not self._active_legs:
            return []
        
        signals = []
        for leg in self._active_legs:
            meta = leg.get("metadata", {})
            leg_type = meta.get("leg", "ce")
            
            # Reverse the transaction type to close
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

    def _evaluate_entry(self) -> list[Signal]:
        chain = self._option_chain
        if not chain:
            return []

        if chain.atm_iv < self.params["min_iv"] or chain.atm_iv > self.params["max_iv"]:
            return []

        direction = (
            TransactionType.SELL
            if self.params["direction"] == "sell"
            else TransactionType.BUY
        )

        if self.params["mode"] == "straddle":
            ce = self._find_strike_nearest(chain.atm_strike, "CE", chain)
            pe = self._find_strike_nearest(chain.atm_strike, "PE", chain)
        else:
            offset = chain.spot_price * self.params["strangle_offset_pct"] / 100
            ce = self._find_strike_nearest(chain.spot_price + offset, "CE", chain)
            pe = self._find_strike_nearest(chain.spot_price - offset, "PE", chain)

        if not ce or not pe:
            return []

        total_premium = ce.last_price + pe.last_price
        
        # For short positions, calculate profit target and max loss
        if self.params["direction"] == "sell":
            max_profit = total_premium
            profit_target = max_profit * (self.params["profit_target_pct"] / 100)
            max_loss = max_profit * (self.params["max_loss_pct"] / 100)
        else:
            max_loss = total_premium
            profit_target = total_premium * (self.params["profit_target_pct"] / 100)
            max_profit = float("inf")
        
        meta = {
            "strategy_type": self.params["mode"],
            "direction": self.params["direction"],
            "total_premium": round(total_premium, 2),
            "profit_target": round(profit_target, 2),
            "max_loss": round(max_loss, 2),
            "adjustment_threshold_pct": self.params["adjustment_threshold_pct"],
            "ce_strike": ce.strike,
            "pe_strike": pe.strike,
            "spot_price": chain.spot_price,
            "atm_iv": chain.atm_iv,
        }

        signals = [
            self._create_signal(ce, direction, 55.0, {**meta, "leg": "ce"}),
            self._create_signal(pe, direction, 55.0, {**meta, "leg": "pe"}),
        ]
        self._active_legs = [s.model_dump() for s in signals]
        logger.info(f"{self.params['mode']}_entry", premium=total_premium, meta=meta)
        return signals


class BullCallSpreadStrategy(OptionStrategyBase):
    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        defaults = {
            "buy_strike_offset": 0,
            "sell_strike_offset": 200,
            "min_risk_reward": 1.5,
            "trend_ema_period": 20,
            "profit_target_pct": 50.0,  # Close at 50% of max profit
        }
        merged = {**defaults, **(params or {})}
        super().__init__("bull_call_spread", merged)

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
        """Check profit target and exit conditions."""
        if not self._active_legs or not self._option_chain:
            return []
        
        chain = self._option_chain
        # Find current prices of leg options to estimate P&L
        buy_leg = next((l for l in self._active_legs if l.get("metadata", {}).get("leg") == "buy_ce"), None)
        sell_leg = next((l for l in self._active_legs if l.get("metadata", {}).get("leg") == "sell_ce"), None)
        
        if not buy_leg or not sell_leg:
            return []
        
        buy_strike = buy_leg.get("metadata", {}).get("buy_strike", 0)
        sell_strike = sell_leg.get("metadata", {}).get("sell_strike", 0)
        max_profit = sell_leg.get("metadata", {}).get("max_profit", 0)
        
        # Simple check: if current spot moved favorably beyond profit target %
        profit_target = max_profit * (self.params["profit_target_pct"] / 100)
        distance_from_buy = chain.spot_price - buy_strike
        
        # If price moved up significantly (closer to max profit), close position
        if distance_from_buy >= profit_target / (sell_strike - buy_strike) * (sell_strike - buy_strike):
            logger.info("bull_call_profit_target_hit", distance=distance_from_buy, target=profit_target)
            return self._create_close_signals("profit_target_hit")
        
        return []

    def _evaluate_entry(self) -> list[Signal]:
        chain = self._option_chain
        if not chain:
            return []

        buy_strike = chain.atm_strike + self.params["buy_strike_offset"]
        sell_strike = chain.atm_strike + self.params["sell_strike_offset"]

        buy_ce = self._find_strike_nearest(buy_strike, "CE", chain)
        sell_ce = self._find_strike_nearest(sell_strike, "CE", chain)

        if not buy_ce or not sell_ce or sell_ce.strike <= buy_ce.strike:
            return []

        net_debit = buy_ce.last_price - sell_ce.last_price
        max_profit = (sell_ce.strike - buy_ce.strike) - net_debit
        if net_debit <= 0 or max_profit <= 0:
            return []

        risk_reward = max_profit / net_debit
        if risk_reward < self.params["min_risk_reward"]:
            return []

        meta = {
            "strategy_type": "bull_call_spread",
            "net_debit": round(net_debit, 2),
            "max_profit": round(max_profit, 2),
            "profit_target": round(max_profit * (self.params["profit_target_pct"] / 100), 2),
            "risk_reward": round(risk_reward, 2),
            "buy_strike": buy_ce.strike,
            "sell_strike": sell_ce.strike,
            "spot_price": chain.spot_price,
        }

        signals = [
            self._create_signal(buy_ce, TransactionType.BUY, 55.0, {**meta, "leg": "buy_ce"}),
            self._create_signal(sell_ce, TransactionType.SELL, 55.0, {**meta, "leg": "sell_ce"}),
        ]
        self._active_legs = [s.model_dump() for s in signals]
        logger.info("bull_call_spread_entry", meta=meta)
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


class BearPutSpreadStrategy(OptionStrategyBase):
    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        defaults = {
            "buy_strike_offset": 0,
            "sell_strike_offset": -200,
            "min_risk_reward": 1.5,
            "trend_ema_period": 20,
            "profit_target_pct": 50.0,  # Close at 50% of max profit
        }
        merged = {**defaults, **(params or {})}
        super().__init__("bear_put_spread", merged)

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
        """Check profit target and exit conditions."""
        if not self._active_legs or not self._option_chain:
            return []
        
        chain = self._option_chain
        buy_leg = next((l for l in self._active_legs if l.get("metadata", {}).get("leg") == "buy_pe"), None)
        sell_leg = next((l for l in self._active_legs if l.get("metadata", {}).get("leg") == "sell_pe"), None)
        
        if not buy_leg or not sell_leg:
            return []
        
        buy_strike = buy_leg.get("metadata", {}).get("buy_strike", 0)
        sell_strike = sell_leg.get("metadata", {}).get("sell_strike", 0)
        max_profit = sell_leg.get("metadata", {}).get("max_profit", 0)
        
        # Simple check: if current spot moved favorably (down) beyond profit target %
        profit_target = max_profit * (self.params["profit_target_pct"] / 100)
        distance_from_buy = buy_strike - chain.spot_price
        
        # If price moved down significantly (closer to max profit), close position
        if distance_from_buy >= profit_target / (buy_strike - sell_strike) * (buy_strike - sell_strike):
            logger.info("bear_put_profit_target_hit", distance=distance_from_buy, target=profit_target)
            return self._create_close_signals("profit_target_hit")
        
        return []

    def _evaluate_entry(self) -> list[Signal]:
        chain = self._option_chain
        if not chain:
            return []

        buy_strike = chain.atm_strike + self.params["buy_strike_offset"]
        sell_strike = chain.atm_strike + self.params["sell_strike_offset"]

        buy_pe = self._find_strike_nearest(buy_strike, "PE", chain)
        sell_pe = self._find_strike_nearest(sell_strike, "PE", chain)

        if not buy_pe or not sell_pe or sell_pe.strike >= buy_pe.strike:
            return []

        net_debit = buy_pe.last_price - sell_pe.last_price
        max_profit = (buy_pe.strike - sell_pe.strike) - net_debit
        if net_debit <= 0 or max_profit <= 0:
            return []

        risk_reward = max_profit / net_debit
        if risk_reward < self.params["min_risk_reward"]:
            return []

        meta = {
            "strategy_type": "bear_put_spread",
            "net_debit": round(net_debit, 2),
            "max_profit": round(max_profit, 2),
            "profit_target": round(max_profit * (self.params["profit_target_pct"] / 100), 2),
            "risk_reward": round(risk_reward, 2),
            "buy_strike": buy_pe.strike,
            "sell_strike": sell_pe.strike,
            "spot_price": chain.spot_price,
        }

        signals = [
            self._create_signal(buy_pe, TransactionType.BUY, 55.0, {**meta, "leg": "buy_pe"}),
            self._create_signal(sell_pe, TransactionType.SELL, 55.0, {**meta, "leg": "sell_pe"}),
        ]
        self._active_legs = [s.model_dump() for s in signals]
        logger.info("bear_put_spread_entry", meta=meta)
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
