"""
Journal Data Models — Production-Grade 3-Layer Structure
=========================================================

Layer 1: ExecutionRecord  — Order lifecycle, fill quality, slippage, costs
Layer 2: StrategyContext   — Market environment, signal quality, position Greeks
Layer 3: PortfolioSnapshot — Capital, risk, drawdown, system stability

All models are dataclasses with to_dict()/from_dict() for SQLite JSON storage.
Timestamps are ISO-8601 strings for cross-platform compatibility.
"""

from __future__ import annotations
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List


# ── Enums ────────────────────────────────────────────────────

class JournalEventType(str, Enum):
    TRADE_OPEN = "trade_open"
    TRADE_CLOSE = "trade_close"
    TRADE_UPDATE = "trade_update"
    POSITION_ADJUST = "position_adjust"
    PORTFOLIO_SNAPSHOT = "portfolio_snapshot"
    SYSTEM_EVENT = "system_event"
    RECONCILIATION = "reconciliation"


class OrderEventType(str, Enum):
    PLACED = "placed"
    ACKNOWLEDGED = "acknowledged"
    PARTIAL_FILL = "partial_fill"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    MODIFIED = "modified"
    EXPIRED = "expired"


class TradeDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class SessionType(str, Enum):
    OPENING = "opening"       # 09:15 – 09:45
    MID_MORNING = "mid_morning"  # 09:45 – 11:30
    MIDDAY = "midday"         # 11:30 – 13:30
    AFTERNOON = "afternoon"   # 13:30 – 14:45
    CLOSING = "closing"       # 14:45 – 15:30


class RegimeType(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"
    EVENT_RISK = "event_risk"
    UNKNOWN = "unknown"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LAYER 1: TRADE EXECUTION QUALITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class OrderEvent:
    """Single order lifecycle event — placed/ack/fill/reject."""
    event_type: str              # OrderEventType value
    timestamp: str = ""          # ISO-8601
    order_id: str = ""
    status: str = ""
    price: float = 0.0
    filled_qty: int = 0
    pending_qty: int = 0
    reject_reason: str = ""
    latency_ms: float = 0.0     # time from previous event
    raw_response: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "OrderEvent":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class LiquiditySnapshot:
    """Market microstructure at trade execution."""
    bid_price: float = 0.0
    ask_price: float = 0.0
    spread_pct: float = 0.0      # (ask - bid) / mid * 100
    market_volume: int = 0
    oi: int = 0                   # open interest (F&O)
    depth_buy_qty: int = 0        # total buy-side depth
    depth_sell_qty: int = 0       # total sell-side depth
    imbalance_ratio: float = 0.0  # buy_depth / (buy + sell)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LiquiditySnapshot":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class CostBreakdown:
    """Indian market transaction costs (applicable to both equity and F&O)."""
    brokerage: float = 0.0
    stt: float = 0.0
    exchange_txn_charge: float = 0.0
    gst: float = 0.0
    sebi_fee: float = 0.0
    stamp_duty: float = 0.0
    slippage_cost: float = 0.0
    total_cost: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CostBreakdown":
        d2 = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**d2)

    def compute_total(self):
        self.total_cost = (self.brokerage + self.stt + self.exchange_txn_charge +
                           self.gst + self.sebi_fee + self.stamp_duty + self.slippage_cost)


@dataclass
class ExcursionMetrics:
    """MAE/MFE — the most powerful trade diagnostics."""
    mae: float = 0.0             # Max Adverse Excursion (₹)
    mfe: float = 0.0             # Max Favorable Excursion (₹)
    mae_pct: float = 0.0
    mfe_pct: float = 0.0
    time_to_mae_bars: int = 0    # bars until worst point
    time_to_mfe_bars: int = 0    # bars until best point
    edge_ratio: float = 0.0      # MFE / MAE — > 1.0 is good

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExcursionMetrics":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class FillRecord:
    """Individual fill event within an order."""
    fill_id: str = ""
    order_id: str = ""
    fill_timestamp: str = ""
    fill_price: float = 0.0
    fill_qty: int = 0
    exchange_trade_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FillRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ExecutionRecord:
    """
    LAYER 1 — Complete execution quality record for a single trade leg.
    Tracks the full order lifecycle, price quality, liquidity, and costs.
    """
    # ── Identification ──
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    group_id: str = ""               # links multi-leg trades
    strategy_name: str = ""
    instrument: str = ""             # e.g. "NIFTY", "RELIANCE"
    tradingsymbol: str = ""          # full Kite symbol
    exchange: str = "NSE"
    expiry: str = ""                 # ISO date if F&O
    strike: float = 0.0
    option_type: str = ""            # CE / PE / FUT / EQ
    direction: str = ""              # BUY / SELL
    quantity: int = 0
    lot_size: int = 1

    # ── Timing ──
    signal_timestamp: str = ""       # when strategy generated signal
    order_sent_timestamp: str = ""   # when order was placed
    exchange_ack_timestamp: str = "" # exchange acknowledgement
    fill_timestamp: str = ""         # final fill time
    exit_timestamp: str = ""         # position close time

    # ── Price Quality (THE CRITICAL THREE) ──
    expected_entry_price: float = 0.0   # signal price
    actual_fill_price: float = 0.0      # what we actually got
    mid_price_at_fill: float = 0.0      # market mid at fill

    # Derived (computed on save)
    entry_slippage: float = 0.0         # actual_fill - expected_entry
    entry_slippage_pct: float = 0.0
    execution_alpha: float = 0.0        # mid_price - fill_price (+ = good)

    # ── Exit Price Quality ──
    expected_exit_price: float = 0.0
    actual_exit_price: float = 0.0
    mid_price_at_exit: float = 0.0
    exit_slippage: float = 0.0
    exit_slippage_pct: float = 0.0

    # ── Order Behavior ──
    order_type: str = "MARKET"       # MARKET / LIMIT / SL / SL-M
    partial_fill_count: int = 0
    rejected_orders: int = 0
    hedge_triggered: bool = False
    emergency_hedge_used: bool = False

    # ── Sub-records ──
    order_events: List[Dict] = field(default_factory=list)
    fills: List[Dict] = field(default_factory=list)
    liquidity_at_entry: Dict = field(default_factory=dict)
    liquidity_at_exit: Dict = field(default_factory=dict)
    costs: Dict = field(default_factory=dict)

    # ── Outcome ──
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    excursion: Dict = field(default_factory=dict)  # ExcursionMetrics.to_dict()

    # ── Derived Timing ──
    signal_to_fill_ms: float = 0.0     # total latency
    order_to_ack_ms: float = 0.0
    ack_to_fill_ms: float = 0.0
    hold_duration_seconds: float = 0.0

    def compute_derived(self):
        """Calculate slippage, alpha, and timing metrics."""
        if self.expected_entry_price and self.actual_fill_price:
            self.entry_slippage = self.actual_fill_price - self.expected_entry_price
            self.entry_slippage_pct = (self.entry_slippage / self.expected_entry_price * 100
                                       if self.expected_entry_price else 0)
        if self.mid_price_at_fill and self.actual_fill_price:
            self.execution_alpha = self.mid_price_at_fill - self.actual_fill_price
            if self.direction == "SELL":
                self.execution_alpha = -self.execution_alpha

        if self.expected_exit_price and self.actual_exit_price:
            self.exit_slippage = self.actual_exit_price - self.expected_exit_price
            self.exit_slippage_pct = (self.exit_slippage / self.expected_exit_price * 100
                                      if self.expected_exit_price else 0)

    def to_dict(self) -> dict:
        self.compute_derived()
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExecutionRecord":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LAYER 2: STRATEGY EDGE QUALITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class StrategyContext:
    """
    LAYER 2 — Market environment and signal quality at trade entry.
    This is what lets you discover: "Strategy only works when IV > 60 pctl."
    """
    # ── Market Context at Entry ──
    underlying_price: float = 0.0
    iv_level: float = 0.0
    iv_percentile: float = 0.0       # 0-100
    vix_level: float = 0.0
    trend_regime: str = ""            # RegimeType value
    realized_volatility_20d: float = 0.0
    realized_volatility_5d: float = 0.0
    dte: int = 0                      # days to expiry
    day_of_week: int = 0              # 0=Mon → 4=Fri
    session: str = ""                 # SessionType value
    is_expiry_day: bool = False
    is_weekly_expiry: bool = False
    is_monthly_expiry: bool = False

    # ── ADX / trend indicators ──
    adx_14: float = 0.0
    rsi_14: float = 0.0
    ema_9: float = 0.0
    ema_21: float = 0.0
    vwap: float = 0.0
    atr_14: float = 0.0

    # ── Signal Quality ──
    signal_confidence: float = 0.0    # 0.0 – 1.0
    probability_of_profit: float = 0.0
    expected_reward_risk: float = 0.0
    indicator_values: Dict[str, float] = field(default_factory=dict)

    # ── Market Context at Exit ──
    exit_underlying_price: float = 0.0
    exit_iv_level: float = 0.0
    exit_vix_level: float = 0.0
    exit_regime: str = ""
    underlying_move_pct: float = 0.0  # % move during trade

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyContext":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SignalQuality:
    """Signal generation metadata — was the signal good?"""
    signal_id: str = ""
    strategy_name: str = ""
    signal_type: str = ""       # "entry" / "exit" / "adjustment"
    confidence: float = 0.0     # 0-1
    expected_pnl: float = 0.0   # model's predicted P&L
    actual_pnl: float = 0.0     # realized P&L
    signal_was_correct: bool = False  # expected_pnl and actual_pnl same sign
    edge_captured_pct: float = 0.0    # actual / expected * 100

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SignalQuality":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PositionStructure:
    """
    Options position structure — Greeks, margins, breakevens.
    For spreads and multi-leg structures.
    """
    structure_type: str = ""       # StructureType value
    legs_count: int = 0
    net_credit_debit: float = 0.0  # + credit, - debit
    max_profit: float = 0.0
    max_loss: float = 0.0
    breakeven_points: List[float] = field(default_factory=list)

    # ── Greeks at Entry ──
    entry_delta: float = 0.0
    entry_gamma: float = 0.0
    entry_theta: float = 0.0
    entry_vega: float = 0.0

    # ── Greeks at Exit ──
    exit_delta: float = 0.0
    exit_gamma: float = 0.0
    exit_theta: float = 0.0
    exit_vega: float = 0.0

    # ── Margin ──
    initial_margin: float = 0.0
    peak_margin: float = 0.0
    margin_utilization_pct: float = 0.0

    # ── Per-leg details ──
    legs: List[Dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PositionStructure":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LAYER 3: PORTFOLIO / SYSTEM HEALTH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class CapitalMetrics:
    """Capital state before and after trade."""
    equity_before: float = 0.0
    equity_after: float = 0.0
    capital_at_risk_pct: float = 0.0
    margin_utilization_pct: float = 0.0
    leverage_ratio: float = 0.0
    available_margin: float = 0.0
    used_margin: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CapitalMetrics":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RiskMetrics:
    """Risk state at the time of this trade."""
    rolling_drawdown_pct: float = 0.0
    peak_to_valley_drawdown_pct: float = 0.0
    daily_loss: float = 0.0
    weekly_loss: float = 0.0
    monthly_loss: float = 0.0
    open_positions_count: int = 0
    exposure_by_symbol: Dict[str, float] = field(default_factory=dict)
    exposure_by_strategy: Dict[str, float] = field(default_factory=dict)
    correlation_risk: float = 0.0     # portfolio correlation
    max_single_position_pct: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RiskMetrics":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SystemEvent:
    """Operational health — tracks system-level issues."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: str = ""
    event_type: str = ""             # "ws_disconnect", "broker_rejection", "crash_recovery", etc.
    severity: str = "info"           # "info", "warning", "error", "critical"
    execution_latency_ms: float = 0.0
    websocket_disconnects: int = 0
    reconciliation_events: int = 0
    broker_rejections: int = 0
    recovery_events: int = 0
    crash_recoveries: int = 0
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SystemEvent":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PortfolioSnapshot:
    """
    LAYER 3 — Point-in-time portfolio state.
    Taken at trade events AND on a periodic schedule (every N minutes).
    """
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: str = ""
    trigger: str = ""                 # "trade_open", "trade_close", "periodic", "eod"

    capital: Dict = field(default_factory=dict)    # CapitalMetrics.to_dict()
    risk: Dict = field(default_factory=dict)       # RiskMetrics.to_dict()

    # ── Portfolio Greeks (F&O) ──
    portfolio_delta: float = 0.0
    portfolio_gamma: float = 0.0
    portfolio_theta: float = 0.0
    portfolio_vega: float = 0.0

    # ── Aggregate Performance ──
    total_equity: float = 0.0
    day_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_trades_today: int = 0
    win_rate_today: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PortfolioSnapshot":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FULL JOURNAL ENTRY — COMBINES ALL 3 LAYERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class JournalEntry:
    """
    Complete journal entry — one per trade (round-trip: open → close).
    Combines execution quality, strategy context, and portfolio state.
    
    For multi-leg F&O trades, one JournalEntry per MultiLegPosition.
    For equity trades, one JournalEntry per round-trip (entry+exit).
    """
    # ── Identity ──
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    trade_id: str = ""                # links to ExecutionRecord.trade_id
    group_id: str = ""                # multi-leg group
    strategy_name: str = ""
    instrument: str = ""
    tradingsymbol: str = ""
    exchange: str = ""
    direction: str = ""               # LONG / SHORT
    trade_type: str = "equity"        # "equity" / "fno" / "futures"
    is_closed: bool = False

    # ── Timestamps ──
    entry_time: str = ""
    exit_time: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = ""

    # ── Layer 1: Execution Quality ──
    execution: Dict = field(default_factory=dict)       # ExecutionRecord.to_dict()
    execution_legs: List[Dict] = field(default_factory=list)  # multi-leg: list of ExecutionRecord dicts

    # ── Layer 2: Strategy Edge ──
    strategy_context: Dict = field(default_factory=dict)  # StrategyContext.to_dict()
    signal_quality: Dict = field(default_factory=dict)    # SignalQuality.to_dict()
    position_structure: Dict = field(default_factory=dict) # PositionStructure.to_dict()

    # ── Layer 3: Portfolio Health ──
    portfolio_at_entry: Dict = field(default_factory=dict)  # PortfolioSnapshot.to_dict()
    portfolio_at_exit: Dict = field(default_factory=dict)   # PortfolioSnapshot.to_dict()

    # ── Outcome ──
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    total_costs: float = 0.0
    return_pct: float = 0.0
    return_on_margin: float = 0.0    # for F&O: net_pnl / margin_used

    # ── Excursion ──
    mae: float = 0.0
    mfe: float = 0.0
    mae_pct: float = 0.0
    mfe_pct: float = 0.0
    edge_ratio: float = 0.0         # MFE / MAE

    # ── Tags & Notes ──
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    review_status: str = "unreviewed"  # "unreviewed", "reviewed", "flagged"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Backtest Reference ──
    source: str = "live"             # "live", "paper", "backtest"
    backtest_expected_pnl: float = 0.0
    live_vs_backtest_diff: float = 0.0  # edge decay detection

    def compute_outcome(self):
        """Calculate return metrics."""
        if self.execution:
            ex = self.execution
            entry_val = ex.get("actual_fill_price", 0) * ex.get("quantity", 0)
            if entry_val:
                self.return_pct = (self.net_pnl / entry_val) * 100
        # F&O return on margin
        ps = self.position_structure
        if ps and ps.get("initial_margin", 0) > 0:
            self.return_on_margin = (self.net_pnl / ps["initial_margin"]) * 100
        # Edge ratio
        if self.mae and self.mae != 0:
            self.edge_ratio = abs(self.mfe / self.mae) if self.mae else 0
        # Edge decay
        if self.backtest_expected_pnl:
            self.live_vs_backtest_diff = self.net_pnl - self.backtest_expected_pnl

    def to_dict(self) -> dict:
        self.updated_at = datetime.now().isoformat()
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "JournalEntry":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def close_trade(self, net_pnl: float, gross_pnl: float, total_costs: float,
                    exit_time: str = "", exit_execution: dict = None,
                    portfolio_at_exit: dict = None, exit_context: dict = None):
        """Mark trade as closed with final P&L and exit context."""
        self.is_closed = True
        self.net_pnl = net_pnl
        self.gross_pnl = gross_pnl
        self.total_costs = total_costs
        self.exit_time = exit_time or datetime.now().isoformat()
        if exit_execution:
            if self.execution:
                self.execution.update({
                    "actual_exit_price": exit_execution.get("actual_exit_price", 0),
                    "mid_price_at_exit": exit_execution.get("mid_price_at_exit", 0),
                    "exit_timestamp": self.exit_time,
                })
        if portfolio_at_exit:
            self.portfolio_at_exit = portfolio_at_exit
        if exit_context and self.strategy_context:
            self.strategy_context.update({
                "exit_underlying_price": exit_context.get("exit_underlying_price", 0),
                "exit_iv_level": exit_context.get("exit_iv_level", 0),
                "exit_vix_level": exit_context.get("exit_vix_level", 0),
                "exit_regime": exit_context.get("exit_regime", ""),
                "underlying_move_pct": exit_context.get("underlying_move_pct", 0),
            })
        self.compute_outcome()
