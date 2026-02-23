"""
Expiry & Assignment Engine — Handles option expiry, settlement, and assignment.

Indian F&O:
  • Index options: cash-settled (European)
  • Stock options: physically settled since Oct 2019 (European)
  • Futures: physically settled (stocks), cash-settled (index)
  • Weekly expiry: Tuesday (shifted if holiday)
  • Monthly expiry: last Tuesday
  • ITM options auto-exercised; OTM expire worthless
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Any

from src.derivatives.contracts import (
    DerivativeContract,
    InstrumentType,
    MultiLegPosition,
    OptionLeg,
    SettlementType,
)


class ExpiryEventType(str, Enum):
    ITM_EXERCISE = "ITM_EXERCISE"
    OTM_EXPIRE = "OTM_EXPIRE"
    PHYSICAL_DELIVERY = "PHYSICAL_DELIVERY"
    CASH_SETTLEMENT = "CASH_SETTLEMENT"
    ASSIGNMENT = "ASSIGNMENT"          # short option assigned
    FUTURES_SETTLEMENT = "FUTURES_SETTLEMENT"


@dataclass
class ExpiryEvent:
    """Record of what happened at expiry for a single leg."""
    event_type: ExpiryEventType
    contract: DerivativeContract
    settlement_price: float        # exchange closing/settlement price
    intrinsic_value: float         # max(0, S-K) for CE, max(0, K-S) for PE
    pnl: float = 0.0              # realized P&L from this event
    quantity: int = 0              # signed lots (positive = long, negative = short)
    delivery_obligation: float = 0.0  # ₹ value of physical delivery
    delivery_quantity: int = 0        # shares to deliver/receive
    margin_released: float = 0.0
    timestamp: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.event_type.value,
            "tradingsymbol": self.contract.tradingsymbol,
            "settlement_price": round(self.settlement_price, 2),
            "intrinsic_value": round(self.intrinsic_value, 2),
            "pnl": round(self.pnl, 2),
            "quantity": self.quantity,
            "delivery_obligation": round(self.delivery_obligation, 2),
            "delivery_quantity": self.delivery_quantity,
            "margin_released": round(self.margin_released, 2),
        }


class ExpiryEngine:
    """Process option and futures expiry.

    Responsibilities:
    • Determine if a contract expires on a given date
    • Compute settlement/intrinsic values
    • Handle ITM exercise (long) and assignment (short)
    • Physical vs cash settlement logic
    • Margin release on expiry
    • Penalty calculation for physical settlement default
    """

    # NSE settlement time: 3:30 PM IST
    SETTLEMENT_HOUR = 15
    SETTLEMENT_MINUTE = 30

    # Physical delivery margin increase (4 days before expiry)
    DELIVERY_MARGIN_DAYS = 4
    DELIVERY_MARGIN_MULTIPLIER = 1.4  # 40% extra margin

    def __init__(self) -> None:
        self._events: list[ExpiryEvent] = []

    def is_expiry_day(self, contract: DerivativeContract, current_date: date) -> bool:
        """Check if today is expiry day for this contract."""
        if contract.expiry is None:
            return False

        expiry_date = contract.expiry
        if isinstance(expiry_date, datetime):
            expiry_date = expiry_date.date()

        return current_date == expiry_date

    def is_near_expiry(self, contract: DerivativeContract, current_date: date, days: int = 4) -> bool:
        """Check if within N days of expiry (delivery margin period)."""
        if contract.expiry is None:
            return False

        expiry_date = contract.expiry
        if isinstance(expiry_date, datetime):
            expiry_date = expiry_date.date()

        dte = (expiry_date - current_date).days
        return 0 <= dte <= days

    def compute_settlement_value(
        self,
        contract: DerivativeContract,
        settlement_price: float,
    ) -> float:
        """Compute intrinsic / settlement value at expiry."""
        if contract.is_future:
            return settlement_price

        if contract.strike is None:
            return 0.0

        if contract.is_call:
            return max(0.0, settlement_price - contract.strike)
        elif contract.is_put:
            return max(0.0, contract.strike - settlement_price)
        return 0.0

    def process_expiry(
        self,
        position: MultiLegPosition,
        settlement_price: float,
        current_date: date,
        timestamp: datetime | None = None,
    ) -> list[ExpiryEvent]:
        """Process all legs of a position at expiry.

        Returns list of ExpiryEvents describing what happened.
        """
        events: list[ExpiryEvent] = []
        ts = timestamp or datetime.now()

        for leg in position.legs:
            if leg.is_closed:
                continue

            contract = leg.contract
            if not self.is_expiry_day(contract, current_date):
                continue

            intrinsic = self.compute_settlement_value(contract, settlement_price)
            is_itm = intrinsic > 0

            if contract.is_future:
                event = self._process_futures_expiry(leg, settlement_price, intrinsic, ts)
            elif is_itm:
                event = self._process_itm_option(leg, settlement_price, intrinsic, ts)
            else:
                event = self._process_otm_option(leg, settlement_price, ts)

            events.append(event)
            leg.close(intrinsic)  # Close leg at intrinsic value

        self._events.extend(events)
        return events

    def _process_futures_expiry(
        self,
        leg: OptionLeg,
        settlement_price: float,
        intrinsic: float,
        timestamp: datetime,
    ) -> ExpiryEvent:
        """Process futures contract at expiry."""
        contract = leg.contract
        pnl = (settlement_price - leg.entry_price) * leg.quantity * contract.lot_size

        if contract.settlement == SettlementType.PHYSICAL:
            delivery_qty = abs(leg.quantity) * contract.lot_size
            delivery_value = settlement_price * delivery_qty
            return ExpiryEvent(
                event_type=ExpiryEventType.FUTURES_SETTLEMENT,
                contract=contract,
                settlement_price=settlement_price,
                intrinsic_value=intrinsic,
                pnl=pnl,
                quantity=leg.quantity,
                delivery_obligation=delivery_value,
                delivery_quantity=delivery_qty,
                timestamp=timestamp,
            )
        else:
            return ExpiryEvent(
                event_type=ExpiryEventType.CASH_SETTLEMENT,
                contract=contract,
                settlement_price=settlement_price,
                intrinsic_value=intrinsic,
                pnl=pnl,
                quantity=leg.quantity,
                timestamp=timestamp,
            )

    def _process_itm_option(
        self,
        leg: OptionLeg,
        settlement_price: float,
        intrinsic: float,
        timestamp: datetime,
    ) -> ExpiryEvent:
        """Process in-the-money option at expiry."""
        contract = leg.contract
        is_long = leg.quantity > 0

        # P&L = (intrinsic - premium_paid) × lots × lot_size for long
        # P&L = (premium_received - intrinsic) × lots × lot_size for short
        if is_long:
            pnl = (intrinsic - leg.entry_price) * abs(leg.quantity) * contract.lot_size
            event_type = ExpiryEventType.ITM_EXERCISE
        else:
            pnl = (leg.entry_price - intrinsic) * abs(leg.quantity) * contract.lot_size
            event_type = ExpiryEventType.ASSIGNMENT

        # Physical delivery for stock options
        if contract.settlement == SettlementType.PHYSICAL:
            delivery_qty = abs(leg.quantity) * contract.lot_size
            delivery_value = (contract.strike or settlement_price) * delivery_qty
            return ExpiryEvent(
                event_type=ExpiryEventType.PHYSICAL_DELIVERY,
                contract=contract,
                settlement_price=settlement_price,
                intrinsic_value=intrinsic,
                pnl=pnl,
                quantity=leg.quantity,
                delivery_obligation=delivery_value,
                delivery_quantity=delivery_qty,
                timestamp=timestamp,
            )
        else:
            # Cash settlement (index options)
            return ExpiryEvent(
                event_type=event_type,
                contract=contract,
                settlement_price=settlement_price,
                intrinsic_value=intrinsic,
                pnl=pnl,
                quantity=leg.quantity,
                timestamp=timestamp,
            )

    def _process_otm_option(
        self,
        leg: OptionLeg,
        settlement_price: float,
        timestamp: datetime,
    ) -> ExpiryEvent:
        """Process out-of-the-money option at expiry."""
        contract = leg.contract
        is_long = leg.quantity > 0

        # Long OTM: lose entire premium
        # Short OTM: keep entire premium
        if is_long:
            pnl = -leg.entry_price * abs(leg.quantity) * contract.lot_size
        else:
            pnl = leg.entry_price * abs(leg.quantity) * contract.lot_size

        return ExpiryEvent(
            event_type=ExpiryEventType.OTM_EXPIRE,
            contract=contract,
            settlement_price=settlement_price,
            intrinsic_value=0.0,
            pnl=pnl,
            quantity=leg.quantity,
            margin_released=leg.entry_price * abs(leg.quantity) * contract.lot_size * 0.5,
            timestamp=timestamp,
        )

    def compute_delivery_margin(
        self,
        position: MultiLegPosition,
        spot_price: float,
        current_date: date,
    ) -> float:
        """Compute additional margin needed for physical delivery period."""
        extra_margin = 0.0

        for leg in position.legs:
            if leg.is_closed:
                continue
            contract = leg.contract
            if contract.settlement != SettlementType.PHYSICAL:
                continue
            if not self.is_near_expiry(contract, current_date, self.DELIVERY_MARGIN_DAYS):
                continue

            # Extra margin = delivery value × multiplier
            notional = spot_price * abs(leg.quantity) * contract.lot_size
            extra_margin += notional * (self.DELIVERY_MARGIN_MULTIPLIER - 1)

        return extra_margin

    def get_expiry_events(self) -> list[dict[str, Any]]:
        """Get all expiry events for journaling."""
        return [e.to_dict() for e in self._events]
