"""
Derivative Contract Models — Foundation of the F&O layer.

Canonical data structures for options and futures contracts,
multi-leg positions, and strategy intents.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Optional


class InstrumentType(str, Enum):
    FUT = "FUT"
    CE = "CE"
    PE = "PE"
    EQ = "EQ"


class SettlementType(str, Enum):
    CASH = "CASH"       # Index options/futures in India
    PHYSICAL = "PHYSICAL"  # Stock options/futures in India


class OptionStyle(str, Enum):
    EUROPEAN = "EUROPEAN"  # Index options
    AMERICAN = "AMERICAN"  # (Unused in NSE but kept for completeness)


class StructureType(str, Enum):
    """Multi-leg option structure types."""
    SINGLE = "SINGLE"
    BULL_CALL_SPREAD = "BULL_CALL_SPREAD"
    BEAR_PUT_SPREAD = "BEAR_PUT_SPREAD"
    BULL_PUT_SPREAD = "BULL_PUT_SPREAD"      # credit spread
    BEAR_CALL_SPREAD = "BEAR_CALL_SPREAD"    # credit spread
    IRON_CONDOR = "IRON_CONDOR"
    IRON_BUTTERFLY = "IRON_BUTTERFLY"
    STRADDLE = "STRADDLE"
    LONG_STRADDLE = "LONG_STRADDLE"
    SHORT_STRADDLE = "SHORT_STRADDLE"
    STRANGLE = "STRANGLE"
    LONG_STRANGLE = "LONG_STRANGLE"
    SHORT_STRANGLE = "SHORT_STRANGLE"
    CALENDAR_SPREAD = "CALENDAR_SPREAD"
    RATIO_SPREAD = "RATIO_SPREAD"
    COVERED_CALL = "COVERED_CALL"
    PROTECTIVE_PUT = "PROTECTIVE_PUT"
    NAKED_CALL = "NAKED_CALL"
    NAKED_PUT = "NAKED_PUT"
    FUTURES = "FUTURES"
    CUSTOM = "CUSTOM"


@dataclass
class DerivativeContract:
    """Canonical derivative contract object.

    This is the foundation of the F&O layer. Every option or futures
    position references a DerivativeContract.
    """
    symbol: str                 # e.g. "NIFTY", "RELIANCE"
    tradingsymbol: str          # e.g. "NIFTY2530620000CE"
    instrument_type: InstrumentType
    exchange: str = "NFO"
    strike: float | None = None
    expiry: date | None = None
    lot_size: int = 1
    tick_size: float = 0.05
    underlying: str = ""        # underlying tradingsymbol
    instrument_token: int = 0
    freeze_quantity: int = 900  # max qty per order (exchange limit)

    # Settlement
    settlement: SettlementType = SettlementType.CASH
    option_style: OptionStyle = OptionStyle.EUROPEAN

    @property
    def is_option(self) -> bool:
        return self.instrument_type in (InstrumentType.CE, InstrumentType.PE)

    @property
    def is_call(self) -> bool:
        return self.instrument_type == InstrumentType.CE

    @property
    def is_put(self) -> bool:
        return self.instrument_type == InstrumentType.PE

    @property
    def is_future(self) -> bool:
        return self.instrument_type == InstrumentType.FUT

    @property
    def dte(self) -> int:
        """Days to expiry from today."""
        if not self.expiry:
            return 0
        return max(0, (self.expiry - date.today()).days)

    @property
    def tte(self) -> float:
        """Time to expiry in years (annualized)."""
        return max(self.dte / 365.0, 1 / 365.0)

    def moneyness(self, spot: float) -> str:
        """ITM / ATM / OTM classification."""
        if not self.is_option or self.strike is None:
            return "N/A"
        if self.is_call:
            if spot > self.strike * 1.005:
                return "ITM"
            elif spot < self.strike * 0.995:
                return "OTM"
            return "ATM"
        else:
            if spot < self.strike * 0.995:
                return "ITM"
            elif spot > self.strike * 1.005:
                return "OTM"
            return "ATM"

    def intrinsic_value(self, spot: float) -> float:
        """Intrinsic value of the option."""
        if not self.is_option or self.strike is None:
            return 0.0
        if self.is_call:
            return max(0.0, spot - self.strike)
        return max(0.0, self.strike - spot)


@dataclass
class OptionLeg:
    """A single leg in a multi-leg option position."""
    contract: DerivativeContract
    quantity: int                   # positive = long, negative = short
    entry_price: float = 0.0
    current_price: float = 0.0
    iv_at_entry: float = 0.0
    iv_current: float = 0.0

    # Greeks (per unit)
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0

    # Tracking
    entry_time: datetime | None = None
    exit_price: float = 0.0
    exit_time: datetime | None = None
    is_closed: bool = False

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def lots(self) -> int:
        return abs(self.quantity) // max(self.contract.lot_size, 1)

    @property
    def net_quantity(self) -> int:
        """Signed quantity (positive=long, negative=short)."""
        return self.quantity

    @property
    def premium_paid(self) -> float:
        """Total premium paid/received at entry (per unit × quantity)."""
        return self.entry_price * abs(self.quantity)

    @property
    def current_value(self) -> float:
        return self.current_price * abs(self.quantity)

    @property
    def unrealised_pnl(self) -> float:
        if self.is_closed:
            return 0.0
        if self.quantity > 0:  # long
            return (self.current_price - self.entry_price) * abs(self.quantity)
        else:  # short
            return (self.entry_price - self.current_price) * abs(self.quantity)

    @property
    def realised_pnl(self) -> float:
        if not self.is_closed:
            return 0.0
        if self.quantity > 0:
            return (self.exit_price - self.entry_price) * abs(self.quantity)
        else:
            return (self.entry_price - self.exit_price) * abs(self.quantity)

    # Portfolio-level Greeks (position-weighted)
    @property
    def position_delta(self) -> float:
        return self.delta * self.quantity * self.contract.lot_size

    @property
    def position_gamma(self) -> float:
        return self.gamma * abs(self.quantity) * self.contract.lot_size

    @property
    def position_theta(self) -> float:
        return self.theta * self.quantity * self.contract.lot_size

    @property
    def position_vega(self) -> float:
        return self.vega * abs(self.quantity) * self.contract.lot_size

    def close(self, exit_price: float, exit_time: datetime | None = None) -> float:
        self.exit_price = exit_price
        self.exit_time = exit_time or datetime.now()
        self.is_closed = True
        return self.realised_pnl


@dataclass
class MultiLegPosition:
    """A multi-leg option/futures position (e.g., Iron Condor, Spread)."""
    position_id: str
    structure: StructureType
    underlying: str
    legs: list[OptionLeg] = field(default_factory=list)
    entry_time: datetime | None = None
    exit_time: datetime | None = None
    strategy_name: str = ""

    # Margin tracking
    margin_required: float = 0.0
    margin_peak: float = 0.0
    margin_at_entry: float = 0.0

    # Exit reason (profit_target, stop_loss, expiry, session_end, final_exit)
    exit_reason: str = ""

    # Journal metadata
    iv_percentile_at_entry: float = 0.0
    regime_at_entry: str = ""
    notes: str = ""

    @property
    def is_closed(self) -> bool:
        return all(leg.is_closed for leg in self.legs)

    @property
    def net_premium(self) -> float:
        """Net premium: positive = credit, negative = debit."""
        return sum(
            leg.entry_price * (-leg.quantity)  # short = +credit, long = -debit
            for leg in self.legs
        )

    @property
    def net_delta(self) -> float:
        return sum(leg.position_delta for leg in self.legs)

    @property
    def net_gamma(self) -> float:
        return sum(leg.position_gamma for leg in self.legs)

    @property
    def net_theta(self) -> float:
        return sum(leg.position_theta for leg in self.legs)

    @property
    def net_vega(self) -> float:
        return sum(leg.position_vega for leg in self.legs)

    @property
    def total_lots(self) -> int:
        return sum(leg.lots for leg in self.legs)

    @property
    def unrealised_pnl(self) -> float:
        return sum(leg.unrealised_pnl for leg in self.legs)

    @property
    def realised_pnl(self) -> float:
        return sum(leg.realised_pnl for leg in self.legs)

    @property
    def max_profit(self) -> float:
        """For credit structures, max profit = net premium received."""
        if self.net_premium > 0:
            return self.net_premium * max(leg.contract.lot_size for leg in self.legs)
        return float("inf")

    @property
    def max_loss(self) -> float:
        """Estimated max loss for defined-risk structures."""
        if len(self.legs) < 2:
            return float("inf")
        # For vertical spreads: width of strikes − net premium
        strikes = sorted(set(
            leg.contract.strike for leg in self.legs
            if leg.contract.strike is not None
        ))
        if len(strikes) >= 2:
            width = strikes[-1] - strikes[0]
            lot_size = max(leg.contract.lot_size for leg in self.legs)
            lots = max(leg.lots for leg in self.legs)
            return (width - abs(self.net_premium)) * lot_size * lots
        return float("inf")

    def close_all(self, price_map: dict[str, float], exit_time: datetime | None = None) -> float:
        """Close all legs. price_map: {tradingsymbol: exit_price}."""
        total_pnl = 0.0
        for leg in self.legs:
            if not leg.is_closed:
                price = price_map.get(leg.contract.tradingsymbol, leg.current_price)
                total_pnl += leg.close(price, exit_time)
        self.exit_time = exit_time or datetime.now()
        return total_pnl

    def to_journal_dict(self) -> dict[str, Any]:
        """Export for F&O Journal."""
        return {
            "position_id": self.position_id,
            "structure": self.structure.value,
            "underlying": self.underlying,
            "strategy": self.strategy_name,
            "legs": [
                {
                    "symbol": l.contract.tradingsymbol,
                    "type": l.contract.instrument_type.value,
                    "strike": l.contract.strike,
                    "expiry": str(l.contract.expiry) if l.contract.expiry else None,
                    "qty": l.quantity,
                    "entry": round(l.entry_price, 2),
                    "exit": round(l.exit_price, 2) if l.is_closed else None,
                    "pnl": round(l.realised_pnl if l.is_closed else l.unrealised_pnl, 2),
                    "iv_entry": round(l.iv_at_entry, 4),
                    "delta": round(l.delta, 4),
                    "theta": round(l.theta, 4),
                    "vega": round(l.vega, 4),
                }
                for l in self.legs
            ],
            "net_premium": round(self.net_premium, 2),
            "net_delta": round(self.net_delta, 4),
            "net_theta": round(self.net_theta, 4),
            "net_vega": round(self.net_vega, 4),
            "margin_required": round(self.margin_required, 2),
            "margin_peak": round(self.margin_peak, 2),
            "total_pnl": round(self.realised_pnl if self.is_closed else self.unrealised_pnl, 2),
            "iv_percentile": round(self.iv_percentile_at_entry, 2),
            "regime": self.regime_at_entry,
            "entry_time": str(self.entry_time) if self.entry_time else None,
            "exit_time": str(self.exit_time) if self.exit_time else None,
        }


@dataclass
class OptionIntent:
    """Strategy outputs intent, not raw orders.

    The Derivatives Engine translates intent → specific strikes,
    lot counts, expiry selection, hedge legs.

    This is how institutional systems operate.
    """
    structure: StructureType
    underlying: str
    direction: str = "SELL"           # SELL for premium collection, BUY for directional
    delta_target: float = 0.25        # target delta for strike selection
    max_risk_pct: float = 1.5         # max capital risk %
    min_premium: float = 0.0          # minimum premium per lot
    expiry_preference: str = "weekly"  # weekly / monthly / next
    dte_min: int = 0
    dte_max: int = 45
    iv_rank_min: float = 0.0          # minimum IV percentile
    iv_rank_max: float = 100.0
    adjustment_trigger_pct: float = 50.0  # % of max loss → adjust
    profit_target_pct: float = 50.0       # % of max credit → close
    metadata: dict[str, Any] = field(default_factory=dict)
