"""
F&O Execution Simulator — Realistic fill modelling for Indian derivatives.

Models: bid-ask slippage, liquidity impact, partial fills, freeze quantity
splitting, and order rejection scenarios.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from src.derivatives.contracts import (
    DerivativeContract,
    MultiLegPosition,
    OptionLeg,
)


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class FnOOrderStatus(str, Enum):
    NEW = "NEW"
    OPEN = "OPEN"
    PARTIAL = "PARTIAL"
    COMPLETE = "COMPLETE"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


@dataclass
class FnOFillResult:
    """Result of a simulated F&O fill."""
    order_id: str = ""
    status: FnOOrderStatus = FnOOrderStatus.COMPLETE
    side: OrderSide = OrderSide.BUY
    contract: DerivativeContract | None = None
    requested_lots: int = 0
    filled_lots: int = 0
    fill_price: float = 0.0
    slippage: float = 0.0          # price impact from bid-ask + liquidity
    total_premium: float = 0.0     # fill_price × lots × lot_size
    timestamp: datetime | None = None
    child_orders: int = 1          # how many orders after freeze splitting
    reject_reason: str = ""
    latency_ms: float = 0.0        # simulated latency

    @property
    def is_filled(self) -> bool:
        return self.status == FnOOrderStatus.COMPLETE

    @property
    def fill_rate(self) -> float:
        return self.filled_lots / max(self.requested_lots, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "status": self.status.value,
            "side": self.side.value,
            "tradingsymbol": self.contract.tradingsymbol if self.contract else "",
            "requested_lots": self.requested_lots,
            "filled_lots": self.filled_lots,
            "fill_price": round(self.fill_price, 2),
            "slippage": round(self.slippage, 2),
            "total_premium": round(self.total_premium, 2),
            "child_orders": self.child_orders,
            "reject_reason": self.reject_reason,
            "latency_ms": round(self.latency_ms, 1),
        }


class FnOExecutionSimulator:
    """Simulate realistic F&O order execution.

    Features:
    • Bid-ask spread based fill pricing
    • Liquidity/volume impact model
    • NSE freeze quantity splitting
    • Partial fills for illiquid strikes
    • Realistic rejection scenarios (margin, price band, freeze)
    • Configurable slippage model
    """

    _order_counter: int = 0

    def __init__(
        self,
        slippage_model: str = "realistic",  # "zero", "fixed", "realistic"
        fixed_slippage_pct: float = 0.1,
        rejection_probability: float = 0.005,  # 0.5% random rejection
    ) -> None:
        self.slippage_model = slippage_model
        self.fixed_slippage_pct = fixed_slippage_pct
        self.rejection_probability = rejection_probability

    def _next_order_id(self) -> str:
        FnOExecutionSimulator._order_counter += 1
        return f"SIMFNO{FnOExecutionSimulator._order_counter:06d}"

    def simulate_fill(
        self,
        contract: DerivativeContract,
        side: OrderSide,
        lots: int,
        market_price: float,
        bid: float = 0.0,
        ask: float = 0.0,
        volume: int = 0,
        available_margin: float = float("inf"),
        timestamp: datetime | None = None,
    ) -> FnOFillResult:
        """Simulate order execution for a single contract."""
        result = FnOFillResult(
            order_id=self._next_order_id(),
            side=side,
            contract=contract,
            requested_lots=lots,
            timestamp=timestamp or datetime.now(),
        )

        # ── Rejection checks ───
        # 1. Minimum price check (options can't trade below tick_size)
        if market_price < contract.tick_size:
            result.status = FnOOrderStatus.REJECTED
            result.reject_reason = f"Price {market_price} below tick size {contract.tick_size}"
            return result

        # 2. Random rejection (simulates exchange/broker glitches)
        if random.random() < self.rejection_probability:
            result.status = FnOOrderStatus.REJECTED
            result.reject_reason = "Order rejected by exchange (simulated)"
            return result

        # ── Freeze quantity splitting ───
        freeze_qty = contract.freeze_quantity or 1800
        total_qty = lots * contract.lot_size
        child_orders = math.ceil(total_qty / freeze_qty) if total_qty > freeze_qty else 1
        result.child_orders = child_orders

        # ── Fill price computation ───
        fill_price = self._compute_fill_price(
            contract, side, lots, market_price, bid, ask, volume
        )
        result.fill_price = fill_price
        result.slippage = abs(fill_price - market_price)

        # ── Partial fill check (illiquid strikes) ───
        if volume > 0:
            # If order is >20% of bar volume, partial fill likely
            participation_rate = (lots * contract.lot_size) / max(volume, 1)
            if participation_rate > 0.5:
                # Fill only 60-90% for very large orders
                fill_rate = random.uniform(0.6, 0.9)
                result.filled_lots = max(1, int(lots * fill_rate))
                result.status = FnOOrderStatus.PARTIAL
            elif participation_rate > 0.2:
                # Fill 85-100%
                fill_rate = random.uniform(0.85, 1.0)
                result.filled_lots = max(1, min(lots, int(lots * fill_rate)))
                result.status = (
                    FnOOrderStatus.COMPLETE
                    if result.filled_lots == lots
                    else FnOOrderStatus.PARTIAL
                )
            else:
                result.filled_lots = lots
                result.status = FnOOrderStatus.COMPLETE
        else:
            # No volume data — assume full fill
            result.filled_lots = lots
            result.status = FnOOrderStatus.COMPLETE

        result.total_premium = fill_price * result.filled_lots * contract.lot_size

        # ── Margin check for sells ───
        if side == OrderSide.SELL and result.total_premium * 5 > available_margin:
            # Rough check: short option margin ≈ 5× premium (simplified)
            result.status = FnOOrderStatus.REJECTED
            result.reject_reason = "Insufficient margin for short position"
            result.filled_lots = 0
            result.total_premium = 0

        # ── Simulated latency ───
        result.latency_ms = self._simulate_latency(child_orders)

        return result

    def _compute_fill_price(
        self,
        contract: DerivativeContract,
        side: OrderSide,
        lots: int,
        market_price: float,
        bid: float,
        ask: float,
        volume: int,
    ) -> float:
        """Compute realistic fill price based on slippage model."""
        if self.slippage_model == "zero":
            return market_price

        if self.slippage_model == "fixed":
            slip = market_price * self.fixed_slippage_pct / 100
            return (market_price + slip) if side == OrderSide.BUY else (market_price - slip)

        # ── Realistic model ───
        # Component 1: Bid-ask spread
        if bid > 0 and ask > 0:
            if side == OrderSide.BUY:
                base_price = ask  # buy at ask
            else:
                base_price = bid  # sell at bid
        else:
            # Estimate spread from contract characteristics
            spread_pct = self._estimate_spread(contract, market_price)
            half_spread = market_price * spread_pct / 200  # half spread
            if side == OrderSide.BUY:
                base_price = market_price + half_spread
            else:
                base_price = market_price - half_spread

        # Component 2: Liquidity impact (large orders move price)
        if volume > 0:
            participation = (lots * contract.lot_size) / max(volume, 1)
            # Square-root impact model (Kyle's lambda)
            impact_bps = 2.0 * math.sqrt(participation) * 100  # basis points
            impact = base_price * impact_bps / 10000
            if side == OrderSide.BUY:
                base_price += impact
            else:
                base_price -= impact

        # Component 3: Random noise (±0.1% uniform)
        noise = base_price * random.uniform(-0.001, 0.001)
        base_price += noise

        # Ensure price is non-negative and tick-aligned
        base_price = max(contract.tick_size, base_price)
        tick = contract.tick_size
        base_price = round(round(base_price / tick) * tick, 2)

        return base_price

    def _estimate_spread(self, contract: DerivativeContract, price: float) -> float:
        """Estimate bid-ask spread percentage for options."""
        if not contract.is_option or contract.strike is None:
            return 0.05  # Futures: tight spread

        # Options: wider spread for OTM and near-expiry
        dte = max(contract.dte, 0)
        if price < 5:
            return 10.0   # Very cheap options: 10% spread
        elif price < 20:
            return 5.0
        elif price < 100:
            return 2.0
        elif dte < 2:
            return 3.0    # Expiry day: wider
        else:
            return 1.0    # ATM reasonable

    def _simulate_latency(self, child_orders: int) -> float:
        """Simulate order execution latency in milliseconds."""
        # Base latency: 50-200ms per order
        base = random.uniform(50, 200)
        # Additional latency for freeze splits
        total = base * child_orders
        # Network jitter
        total += random.uniform(0, 50)
        return total

    def execute_multi_leg(
        self,
        legs: list[dict],
        available_margin: float = float("inf"),
        timestamp: datetime | None = None,
    ) -> list[FnOFillResult]:
        """Execute a multi-leg order (e.g., iron condor = 4 legs).

        Each dict in legs should have:
        - contract: DerivativeContract
        - side: OrderSide
        - lots: int
        - market_price: float
        - bid: float (optional)
        - ask: float (optional)
        - volume: int (optional)
        """
        results = []
        remaining_margin = available_margin

        for leg_info in legs:
            fill = self.simulate_fill(
                contract=leg_info["contract"],
                side=leg_info["side"],
                lots=leg_info["lots"],
                market_price=leg_info["market_price"],
                bid=leg_info.get("bid", 0),
                ask=leg_info.get("ask", 0),
                volume=leg_info.get("volume", 0),
                available_margin=remaining_margin,
                timestamp=timestamp,
            )
            results.append(fill)

            # Deduct premium for buys, track margin for sells
            if fill.is_filled:
                if fill.side == OrderSide.BUY:
                    remaining_margin -= fill.total_premium
                else:
                    # Short margin requirement (simplified: 5× premium)
                    remaining_margin -= fill.total_premium * 5

        return results

    def build_option_leg(
        self,
        fill: FnOFillResult,
        iv: float = 0.0,
    ) -> OptionLeg | None:
        """Convert a fill result into an OptionLeg for position tracking."""
        if not fill.is_filled or fill.contract is None:
            return None

        signed_qty = fill.filled_lots if fill.side == OrderSide.BUY else -fill.filled_lots

        return OptionLeg(
            contract=fill.contract,
            quantity=signed_qty,
            entry_price=fill.fill_price,
            current_price=fill.fill_price,
            iv_at_entry=iv,
            iv_current=iv,
        )
