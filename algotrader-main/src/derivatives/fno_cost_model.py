"""
Indian F&O Cost Model — Accurate transaction cost computation.

Indian derivatives have a complex fee structure:
  • STT: Different rates for options vs futures, buy vs sell
  • Exchange transaction charges (NSE/BSE)
  • SEBI turnover fees
  • GST on brokerage + transaction charges
  • Stamp duty (state-level, applied on buy side only)
  • Brokerage (flat ₹20 per order for discount brokers)

Reference: NSE circulars, Zerodha brokerage calculator
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.derivatives.contracts import DerivativeContract, InstrumentType


class TransactionSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class FnOTransactionCost:
    """Detailed breakdown of F&O transaction costs."""
    brokerage: float = 0.0
    stt: float = 0.0                      # Securities Transaction Tax
    exchange_txn_charge: float = 0.0       # NSE transaction charge
    sebi_fee: float = 0.0                  # SEBI turnover fee
    gst: float = 0.0                       # 18% GST on (brokerage + exchange + SEBI)
    stamp_duty: float = 0.0               # State stamp duty (buy side only)
    total: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "brokerage": round(self.brokerage, 2),
            "stt": round(self.stt, 2),
            "exchange_txn_charge": round(self.exchange_txn_charge, 4),
            "sebi_fee": round(self.sebi_fee, 4),
            "gst": round(self.gst, 2),
            "stamp_duty": round(self.stamp_duty, 4),
            "total": round(self.total, 2),
        }


class IndianFnOCostModel:
    """Compute realistic transaction costs for Indian F&O.

    Fee structure (as of 2024):
    ┌─────────────┬───────────────────┬──────────────────────┐
    │ Component   │ Futures           │ Options              │
    ├─────────────┼───────────────────┼──────────────────────┤
    │ STT         │ 0.0125% buy+sell  │ 0.0625% sell only    │
    │ Exchange    │ 0.0019%           │ 0.0495% (NSE)        │
    │ SEBI fee    │ ₹10/crore         │ ₹10/crore            │
    │ Stamp duty  │ 0.002% buy only   │ 0.003% buy only      │
    │ GST         │ 18%               │ 18%                  │
    │ Brokerage   │ ₹20/order or 0.03%│ ₹20/order            │
    └─────────────┴───────────────────┴──────────────────────┘

    Notes:
    - STT on options: charged ONLY on sell side at 0.0625% of premium
    - STT on exercised options: 0.125% of settlement price × qty (costly!)
    - Brokerage: flat ₹20 per executed order (discount broker model)
    """

    # STT rates
    FUTURES_STT_RATE = 0.0125 / 100       # 0.0125% on both sides
    OPTIONS_STT_SELL_RATE = 0.0625 / 100  # 0.0625% on sell side only
    OPTIONS_STT_EXERCISE_RATE = 0.125 / 100  # 0.125% on exercised intrinsic

    # Exchange transaction charges (NSE)
    FUTURES_EXCHANGE_RATE = 0.0019 / 100   # 0.0019%
    OPTIONS_EXCHANGE_RATE = 0.0495 / 100   # 0.0495% (for premium)

    # SEBI turnover fee
    SEBI_FEE_PER_CRORE = 10.0            # ₹10 per crore turnover

    # Stamp duty (on buy side only)
    FUTURES_STAMP_DUTY_RATE = 0.002 / 100  # 0.002%
    OPTIONS_STAMP_DUTY_RATE = 0.003 / 100  # 0.003%

    # GST rate
    GST_RATE = 0.18  # 18%

    # Brokerage
    FLAT_BROKERAGE_PER_ORDER = 20.0       # ₹20 per order (Zerodha model)
    BROKERAGE_PCT_CAP = 0.03 / 100        # max 0.03% of turnover

    def __init__(
        self,
        brokerage_per_order: float = 20.0,
        use_percentage_brokerage: bool = False,
        brokerage_pct: float = 0.03,
    ) -> None:
        self.brokerage_per_order = brokerage_per_order
        self.use_percentage_brokerage = use_percentage_brokerage
        self.brokerage_pct = brokerage_pct / 100

    def calculate(
        self,
        contract: DerivativeContract,
        side: TransactionSide,
        price: float,
        lots: int,
        is_exercise: bool = False,
        child_orders: int = 1,
    ) -> FnOTransactionCost:
        """Calculate complete transaction cost for an F&O trade.

        Args:
            contract: the derivative contract
            side: BUY or SELL
            price: fill price (premium for options, futures price for futures)
            lots: number of lots traded
            is_exercise: True if this is an option exercise/assignment at expiry
            child_orders: number of child orders (after freeze splitting)
        """
        total_qty = lots * contract.lot_size
        turnover = price * total_qty  # premium turnover for options

        cost = FnOTransactionCost()

        # ── Brokerage ───
        if self.use_percentage_brokerage:
            cost.brokerage = min(
                turnover * self.brokerage_pct,
                self.brokerage_per_order * child_orders
            )
        else:
            cost.brokerage = self.brokerage_per_order * child_orders

        # ── STT ───
        if contract.is_future:
            cost.stt = turnover * self.FUTURES_STT_RATE
        elif contract.is_option:
            if is_exercise:
                # Exercise STT: 0.125% of settlement value (very expensive!)
                cost.stt = turnover * self.OPTIONS_STT_EXERCISE_RATE
            elif side == TransactionSide.SELL:
                cost.stt = turnover * self.OPTIONS_STT_SELL_RATE
            else:
                cost.stt = 0.0  # No STT on option buy

        # ── Exchange transaction charge ───
        if contract.is_future:
            cost.exchange_txn_charge = turnover * self.FUTURES_EXCHANGE_RATE
        elif contract.is_option:
            cost.exchange_txn_charge = turnover * self.OPTIONS_EXCHANGE_RATE

        # ── SEBI fee ───
        cost.sebi_fee = turnover * self.SEBI_FEE_PER_CRORE / 1e7

        # ── Stamp duty (buy side only) ───
        if side == TransactionSide.BUY:
            if contract.is_future:
                cost.stamp_duty = turnover * self.FUTURES_STAMP_DUTY_RATE
            elif contract.is_option:
                cost.stamp_duty = turnover * self.OPTIONS_STAMP_DUTY_RATE

        # ── GST: 18% on (brokerage + exchange + SEBI) ───
        gst_base = cost.brokerage + cost.exchange_txn_charge + cost.sebi_fee
        cost.gst = gst_base * self.GST_RATE

        # ── Total ───
        cost.total = (
            cost.brokerage
            + cost.stt
            + cost.exchange_txn_charge
            + cost.sebi_fee
            + cost.gst
            + cost.stamp_duty
        )

        return cost

    def round_trip_cost(
        self,
        contract: DerivativeContract,
        entry_price: float,
        exit_price: float,
        lots: int,
        is_exercise: bool = False,
        child_orders_entry: int = 1,
        child_orders_exit: int = 1,
    ) -> FnOTransactionCost:
        """Calculate total round-trip cost (entry + exit)."""
        entry_cost = self.calculate(
            contract, TransactionSide.BUY, entry_price, lots,
            child_orders=child_orders_entry,
        )
        exit_cost = self.calculate(
            contract, TransactionSide.SELL, exit_price, lots,
            is_exercise=is_exercise,
            child_orders=child_orders_exit,
        )

        return FnOTransactionCost(
            brokerage=entry_cost.brokerage + exit_cost.brokerage,
            stt=entry_cost.stt + exit_cost.stt,
            exchange_txn_charge=entry_cost.exchange_txn_charge + exit_cost.exchange_txn_charge,
            sebi_fee=entry_cost.sebi_fee + exit_cost.sebi_fee,
            gst=entry_cost.gst + exit_cost.gst,
            stamp_duty=entry_cost.stamp_duty + exit_cost.stamp_duty,
            total=entry_cost.total + exit_cost.total,
        )

    def multi_leg_cost(
        self,
        legs: list[dict],
    ) -> FnOTransactionCost:
        """Calculate combined cost for multi-leg strategy.

        Each dict: {contract, side, price, lots, child_orders?}
        """
        combined = FnOTransactionCost()

        for leg in legs:
            cost = self.calculate(
                contract=leg["contract"],
                side=leg["side"],
                price=leg["price"],
                lots=leg["lots"],
                child_orders=leg.get("child_orders", 1),
            )
            combined.brokerage += cost.brokerage
            combined.stt += cost.stt
            combined.exchange_txn_charge += cost.exchange_txn_charge
            combined.sebi_fee += cost.sebi_fee
            combined.gst += cost.gst
            combined.stamp_duty += cost.stamp_duty
            combined.total += cost.total

        return combined

    def breakeven_move(
        self,
        contract: DerivativeContract,
        entry_price: float,
        lots: int,
    ) -> float:
        """Calculate minimum price move needed to break even after costs."""
        costs = self.round_trip_cost(contract, entry_price, entry_price, lots)
        total_qty = lots * contract.lot_size
        return costs.total / max(total_qty, 1)
