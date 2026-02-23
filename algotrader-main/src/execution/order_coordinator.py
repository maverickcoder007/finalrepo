"""
Order Coordinator — Broker Microstructure Layer

Sits between ExecutionEngine and KiteClient to handle:
- Exchange freeze quantity splitting (BEFORE margin simulation)
- Market state awareness (pre-open, auction, circuit freeze)
- Execution queue with rate pacing (~10 orders/sec ceiling)
- Liquidity checks before MARKET fallbacks
- Timestamp precedence resolution (WS vs REST race condition)

Architecture:
    ExecutionEngine
           ↓
    OrderCoordinator  ← THIS MODULE
           ↓
    KiteClient
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from src.api.client import KiteClient
from src.data.models import (
    Exchange,
    OrderRequest,
    OrderType,
    OrderVariety,
    ProductType,
    Signal,
    TransactionType,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# 1. Freeze Quantity Manager
# ─────────────────────────────────────────────────────────────

# NSE F&O freeze limits per underlying (updated periodically by NSE)
# Source: NSE circular on quantity freeze limits
FREEZE_LIMITS: dict[str, int] = {
    "NIFTY": 1800,
    "BANKNIFTY": 900,
    "FINNIFTY": 1800,
    "MIDCPNIFTY": 4200,
    "SENSEX": 1000,
    "BANKEX": 1500,
    # Default for stock options / futures
    "_DEFAULT_NFO": 1800,
    "_DEFAULT_MCX": 1800,
}


class FreezeQuantityManager:
    """Split orders exceeding exchange freeze limits BEFORE margin simulation.

    NSE FO rejects any single order exceeding freeze quantity.
    Must split FIRST so margin simulation runs on actual order sizes.

    Pipeline order:
        Signal → freeze_split → margin_simulation → execution
    """

    def __init__(self, custom_limits: Optional[dict[str, int]] = None) -> None:
        self._limits = {**FREEZE_LIMITS, **(custom_limits or {})}

    def get_freeze_qty(self, tradingsymbol: str, exchange: Exchange) -> int:
        """Get freeze quantity for a given instrument.

        Extracts underlying from tradingsymbol (e.g. NIFTY24FEB22000CE → NIFTY).
        """
        if exchange not in (Exchange.NFO, Exchange.BFO, Exchange.MCX):
            return 999_999_999  # No freeze limit for equity

        underlying = self._extract_underlying(tradingsymbol)
        limit = self._limits.get(underlying)
        if limit:
            return limit

        # Fallback defaults by exchange
        if exchange == Exchange.MCX:
            return self._limits.get("_DEFAULT_MCX", 1800)
        return self._limits.get("_DEFAULT_NFO", 1800)

    def split_order(self, order: OrderRequest) -> list[OrderRequest]:
        """Split a single OrderRequest into freeze-compliant chunks.

        Returns list of 1+ OrderRequest objects, each ≤ freeze limit.
        """
        freeze_qty = self.get_freeze_qty(order.tradingsymbol, order.exchange)

        if order.quantity <= freeze_qty:
            return [order]

        chunks: list[OrderRequest] = []
        remaining = order.quantity

        while remaining > 0:
            chunk_qty = min(remaining, freeze_qty)
            chunk = order.model_copy(update={"quantity": chunk_qty})
            chunks.append(chunk)
            remaining -= chunk_qty

        logger.info(
            "order_freeze_split",
            tradingsymbol=order.tradingsymbol,
            original_qty=order.quantity,
            chunks=len(chunks),
            freeze_limit=freeze_qty,
        )
        return chunks

    def split_signal(self, signal: Signal) -> list[Signal]:
        """Split a Signal into freeze-compliant chunks.

        Used in multi-leg flow before margin simulation.
        """
        freeze_qty = self.get_freeze_qty(signal.tradingsymbol, signal.exchange)

        if signal.quantity <= freeze_qty:
            return [signal]

        chunks: list[Signal] = []
        remaining = signal.quantity

        while remaining > 0:
            chunk_qty = min(remaining, freeze_qty)
            chunk = signal.model_copy(update={"quantity": chunk_qty})
            chunks.append(chunk)
            remaining -= chunk_qty

        logger.info(
            "signal_freeze_split",
            tradingsymbol=signal.tradingsymbol,
            original_qty=signal.quantity,
            chunks=len(chunks),
        )
        return chunks

    @staticmethod
    def _extract_underlying(tradingsymbol: str) -> str:
        """Extract underlying name from options/futures tradingsymbol.

        Examples:
            NIFTY24FEB22000CE → NIFTY
            BANKNIFTY24FEB48000PE → BANKNIFTY
            RELIANCE24FEBFUT → RELIANCE
            MIDCPNIFTY24FEB8000CE → MIDCPNIFTY
        """
        known_underlyings = [
            "BANKNIFTY", "MIDCPNIFTY", "FINNIFTY", "NIFTY",
            "SENSEX", "BANKEX",
        ]
        upper = tradingsymbol.upper()
        for u in known_underlyings:
            if upper.startswith(u):
                return u

        # For stock F&O: extract alphabetic prefix
        name = ""
        for ch in tradingsymbol:
            if ch.isalpha():
                name += ch
            else:
                break
        return name or tradingsymbol


# ─────────────────────────────────────────────────────────────
# 2. Market State Guard
# ─────────────────────────────────────────────────────────────

class MarketState(str, Enum):
    """Indian market session states."""
    PRE_OPEN = "PRE_OPEN"
    NORMAL = "NORMAL"
    POST_CLOSE = "POST_CLOSE"
    CLOSED = "CLOSED"
    CIRCUIT_FREEZE = "CIRCUIT_FREEZE"
    AUCTION = "AUCTION"
    UNKNOWN = "UNKNOWN"


class MarketStateGuard:
    """Detect market microstructure states before order placement.

    Indian markets have hidden states (pre-open, auction, circuit freeze)
    where orders may remain OPEN indefinitely. This guard prevents
    placing orders into illiquid/frozen markets.

    Detection methods:
    1. Time-based session detection (pre-open, post-close)
    2. Quote-based liquidity detection (empty depth = frozen)
    3. Circuit limit detection (LTP at upper/lower circuit)
    """

    # IST market hours (as UTC offsets for comparison)
    PRE_OPEN_START_HOUR = 9   # 9:00 IST
    PRE_OPEN_START_MIN = 0
    PRE_OPEN_END_HOUR = 9     # 9:15 IST
    PRE_OPEN_END_MIN = 15
    MARKET_CLOSE_HOUR = 15    # 3:30 IST
    MARKET_CLOSE_MIN = 30

    def __init__(self, client: KiteClient) -> None:
        self._client = client
        self._circuit_cache: dict[str, float] = {}  # symbol → last check time
        self._state_cache: dict[str, MarketState] = {}
        self._cache_ttl = 5.0  # seconds

    async def check_market_state(
        self, tradingsymbol: str, exchange: Exchange
    ) -> MarketState:
        """Determine current market state for an instrument.

        Returns MarketState indicating whether it's safe to place orders.
        """
        # Check time-based state first (cheapest)
        time_state = self._check_time_based_state()
        if time_state != MarketState.NORMAL:
            return time_state

        # Check cache
        cache_key = f"{exchange.value}:{tradingsymbol}"
        cached = self._state_cache.get(cache_key)
        if cached and (time.time() - self._circuit_cache.get(cache_key, 0)) < self._cache_ttl:
            return cached

        # Quote-based check for circuit/liquidity
        state = await self._check_quote_state(tradingsymbol, exchange)
        self._state_cache[cache_key] = state
        self._circuit_cache[cache_key] = time.time()
        return state

    async def is_safe_to_execute(
        self, tradingsymbol: str, exchange: Exchange
    ) -> tuple[bool, str]:
        """Check if it's safe to place orders for this instrument.

        Returns:
            (True, "ok") if safe
            (False, reason) if not safe
        """
        state = await self.check_market_state(tradingsymbol, exchange)

        if state == MarketState.NORMAL:
            return True, "ok"
        elif state == MarketState.PRE_OPEN:
            return False, "Market in pre-open session. Orders may not execute."
        elif state == MarketState.POST_CLOSE:
            return False, "Market closed. Only AMO orders allowed."
        elif state == MarketState.CLOSED:
            return False, "Market is closed."
        elif state == MarketState.CIRCUIT_FREEZE:
            return False, "Instrument hit circuit limit. No liquidity available."
        elif state == MarketState.AUCTION:
            return False, "Instrument in auction mode. Execution uncertain."
        else:
            # UNKNOWN — allow execution but log warning
            logger.warning("market_state_unknown", symbol=tradingsymbol)
            return True, "Market state unknown — proceeding with caution."

    def _check_time_based_state(self) -> MarketState:
        """Check market state based on current time (IST)."""
        now = datetime.now()  # Assumes server is in IST
        hour, minute = now.hour, now.minute

        # Before 9:00 or after 15:30
        if hour < self.PRE_OPEN_START_HOUR:
            return MarketState.CLOSED
        if hour == self.PRE_OPEN_START_HOUR and minute < self.PRE_OPEN_START_MIN:
            return MarketState.CLOSED
        if hour > self.MARKET_CLOSE_HOUR:
            return MarketState.CLOSED
        if hour == self.MARKET_CLOSE_HOUR and minute >= self.MARKET_CLOSE_MIN:
            return MarketState.POST_CLOSE

        # Pre-open: 9:00 — 9:15
        if hour == self.PRE_OPEN_START_HOUR and minute < self.PRE_OPEN_END_MIN:
            return MarketState.PRE_OPEN

        # Normal trading hours: 9:15 — 15:30
        return MarketState.NORMAL

    async def _check_quote_state(
        self, tradingsymbol: str, exchange: Exchange
    ) -> MarketState:
        """Check quote depth for circuit/liquidity issues."""
        try:
            instrument_key = f"{exchange.value}:{tradingsymbol}"
            quotes = await self._client.get_quote([instrument_key])
            quote = quotes.get(instrument_key)

            if not quote:
                return MarketState.UNKNOWN

            # Check circuit limits
            if quote.upper_circuit_limit > 0 and quote.lower_circuit_limit > 0:
                ltp = quote.last_price
                # At upper circuit: no sellers
                if ltp >= quote.upper_circuit_limit:
                    logger.warning(
                        "upper_circuit_detected",
                        symbol=tradingsymbol,
                        ltp=ltp,
                        upper_circuit=quote.upper_circuit_limit,
                    )
                    return MarketState.CIRCUIT_FREEZE
                # At lower circuit: no buyers
                if ltp <= quote.lower_circuit_limit:
                    logger.warning(
                        "lower_circuit_detected",
                        symbol=tradingsymbol,
                        ltp=ltp,
                        lower_circuit=quote.lower_circuit_limit,
                    )
                    return MarketState.CIRCUIT_FREEZE

            # Check depth for liquidity
            depth = quote.depth
            if depth:
                has_bids = any(d.quantity > 0 for d in depth.buy)
                has_asks = any(d.quantity > 0 for d in depth.sell)

                if not has_bids and not has_asks:
                    logger.warning("no_liquidity_detected", symbol=tradingsymbol)
                    return MarketState.CIRCUIT_FREEZE

            return MarketState.NORMAL

        except Exception as e:
            logger.error("market_state_check_failed", symbol=tradingsymbol, error=str(e))
            return MarketState.UNKNOWN


# ─────────────────────────────────────────────────────────────
# 3. Execution Queue with Rate Pacing
# ─────────────────────────────────────────────────────────────

@dataclass
class QueuedOrder:
    """An order waiting in the execution queue."""
    order: OrderRequest
    future: asyncio.Future
    priority: int = 0  # Lower = higher priority. 0=hedge, 1=risk, 2=normal
    enqueued_at: float = field(default_factory=time.time)
    signal: Optional[Signal] = None


class ExecutionQueue:
    """Priority FIFO queue with rate pacing for Zerodha's practical order limit.

    Zerodha practical ceiling: ~10 orders/second.
    This queue prevents burst-flooding at market open or when multiple
    strategies fire simultaneously.

    Priority levels:
        0 = Emergency hedges (immediate, skip queue)
        1 = Protection legs (high priority)
        2 = Risk legs (normal)
        3 = Regular single-leg orders
    """

    MAX_ORDERS_PER_SECOND = 8  # Conservative vs Zerodha's ~10/s practical limit
    MIN_INTERVAL = 1.0 / MAX_ORDERS_PER_SECOND  # ~125ms between orders

    def __init__(self, client: KiteClient) -> None:
        self._client = client
        self._queue: list[QueuedOrder] = []  # Sorted by priority then time
        self._lock = asyncio.Lock()
        self._last_order_time: float = 0.0
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        self._orders_this_second: int = 0
        self._second_start: float = 0.0

    async def start(self) -> None:
        """Start the queue processing worker."""
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._process_loop())
        logger.info("execution_queue_started")

    async def stop(self) -> None:
        """Stop the queue worker."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("execution_queue_stopped")

    async def enqueue(
        self, order: OrderRequest, priority: int = 3, signal: Optional[Signal] = None
    ) -> str:
        """Add order to queue and wait for execution result.

        Args:
            order: OrderRequest to place
            priority: 0=emergency, 1=protection, 2=risk, 3=normal
            signal: Optional associated Signal for logging

        Returns:
            order_id from broker

        Raises:
            OrderError, NetworkError on placement failure
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()

        queued = QueuedOrder(
            order=order,
            future=future,
            priority=priority,
            signal=signal,
        )

        async with self._lock:
            self._queue.append(queued)
            # Sort: lower priority number first, then by enqueue time
            self._queue.sort(key=lambda q: (q.priority, q.enqueued_at))

        logger.debug(
            "order_enqueued",
            symbol=order.tradingsymbol,
            priority=priority,
            queue_depth=len(self._queue),
        )

        return await future

    async def enqueue_immediate(self, order: OrderRequest) -> str:
        """Place order immediately, bypassing queue. For emergency hedges only."""
        await self._pace()
        order_id = await self._client.place_order(order)
        return order_id

    async def _process_loop(self) -> None:
        """Worker loop: dequeue and place orders respecting rate limits."""
        while self._running:
            try:
                queued = await self._dequeue()
                if not queued:
                    await asyncio.sleep(0.01)
                    continue

                # Rate pacing
                await self._pace()

                # Place order
                try:
                    order_id = await self._client.place_order(queued.order)
                    queued.future.set_result(order_id)
                except Exception as e:
                    if not queued.future.done():
                        queued.future.set_exception(e)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("execution_queue_error", error=str(e))
                await asyncio.sleep(0.1)

        # Drain remaining items on shutdown
        async with self._lock:
            for queued in self._queue:
                if not queued.future.done():
                    queued.future.set_exception(
                        RuntimeError("Execution queue shutting down")
                    )
            self._queue.clear()

    async def _dequeue(self) -> Optional[QueuedOrder]:
        """Pop highest-priority item from queue."""
        async with self._lock:
            if self._queue:
                return self._queue.pop(0)
        return None

    async def _pace(self) -> None:
        """Enforce rate limit: max N orders per second."""
        now = time.time()

        # Reset counter each second
        if now - self._second_start >= 1.0:
            self._orders_this_second = 0
            self._second_start = now

        # If we've hit the limit this second, wait
        if self._orders_this_second >= self.MAX_ORDERS_PER_SECOND:
            wait_time = 1.0 - (now - self._second_start)
            if wait_time > 0:
                logger.debug("rate_pacing_wait", wait_ms=round(wait_time * 1000))
                await asyncio.sleep(wait_time)
            self._orders_this_second = 0
            self._second_start = time.time()

        # Minimum interval between orders
        elapsed = now - self._last_order_time
        if elapsed < self.MIN_INTERVAL:
            await asyncio.sleep(self.MIN_INTERVAL - elapsed)

        self._orders_this_second += 1
        self._last_order_time = time.time()

    @property
    def queue_depth(self) -> int:
        return len(self._queue)


# ─────────────────────────────────────────────────────────────
# 4. Liquidity Checker
# ─────────────────────────────────────────────────────────────

class LiquidityChecker:
    """Check bid-ask spread before placing MARKET orders.

    In options, MARKET can slip catastrophically:
        Bid: 100, Next bid: 65 → 35% slippage on MARKET sell

    Pre-check:
        If spread > threshold → use aggressive LIMIT instead of MARKET
        This single check reduces tail loss dramatically.
    """

    # Max acceptable spread as % of mid-price
    DEFAULT_SPREAD_THRESHOLD_PCT = 5.0  # 5% spread = switch to LIMIT
    # Absolute spread threshold in rupees (for low-priced options)
    DEFAULT_SPREAD_THRESHOLD_ABS = 5.0  # ₹5 absolute

    def __init__(
        self,
        client: KiteClient,
        spread_threshold_pct: float = DEFAULT_SPREAD_THRESHOLD_PCT,
        spread_threshold_abs: float = DEFAULT_SPREAD_THRESHOLD_ABS,
    ) -> None:
        self._client = client
        self._spread_threshold_pct = spread_threshold_pct
        self._spread_threshold_abs = spread_threshold_abs

    async def check_liquidity(
        self, tradingsymbol: str, exchange: Exchange
    ) -> dict[str, Any]:
        """Check bid-ask spread and depth for an instrument.

        Returns:
            {
                "liquid": True/False,
                "best_bid": float,
                "best_ask": float,
                "spread": float,
                "spread_pct": float,
                "recommendation": "MARKET" | "LIMIT" | "ABORT"
            }
        """
        try:
            instrument_key = f"{exchange.value}:{tradingsymbol}"
            quotes = await self._client.get_quote([instrument_key])
            quote = quotes.get(instrument_key)

            if not quote or not quote.depth:
                return {
                    "liquid": False,
                    "best_bid": 0,
                    "best_ask": 0,
                    "spread": 0,
                    "spread_pct": 0,
                    "recommendation": "ABORT",
                    "reason": "No quote data available",
                }

            depth = quote.depth
            best_bid = depth.buy[0].price if depth.buy else 0
            best_ask = depth.sell[0].price if depth.sell else 0
            bid_qty = depth.buy[0].quantity if depth.buy else 0
            ask_qty = depth.sell[0].quantity if depth.sell else 0

            # No bids or asks
            if best_bid <= 0 or best_ask <= 0:
                return {
                    "liquid": False,
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "spread": 0,
                    "spread_pct": 0,
                    "recommendation": "ABORT",
                    "reason": "Empty order book",
                }

            spread = best_ask - best_bid
            mid_price = (best_ask + best_bid) / 2
            spread_pct = (spread / mid_price * 100) if mid_price > 0 else 999

            # Wide spread check
            is_wide = (
                spread_pct > self._spread_threshold_pct
                or spread > self._spread_threshold_abs
            )

            if is_wide:
                recommendation = "LIMIT"
                reason = (
                    f"Wide spread: {spread:.2f} ({spread_pct:.1f}%). "
                    f"Bid={best_bid}, Ask={best_ask}. Using aggressive LIMIT."
                )
                logger.warning(
                    "wide_spread_detected",
                    symbol=tradingsymbol,
                    spread=spread,
                    spread_pct=round(spread_pct, 2),
                    bid=best_bid,
                    ask=best_ask,
                )
            else:
                recommendation = "MARKET"
                reason = "Spread acceptable"

            return {
                "liquid": not is_wide,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "bid_qty": bid_qty,
                "ask_qty": ask_qty,
                "spread": round(spread, 2),
                "spread_pct": round(spread_pct, 2),
                "recommendation": recommendation,
                "reason": reason,
            }

        except Exception as e:
            logger.error("liquidity_check_failed", symbol=tradingsymbol, error=str(e))
            # Fail open: assume liquid to avoid blocking execution
            return {
                "liquid": True,
                "best_bid": 0,
                "best_ask": 0,
                "spread": 0,
                "spread_pct": 0,
                "recommendation": "MARKET",
                "reason": f"Check failed ({e}), defaulting to MARKET",
            }

    def convert_to_aggressive_limit(
        self,
        order: OrderRequest,
        liquidity: dict[str, Any],
    ) -> OrderRequest:
        """Convert a MARKET order to aggressive LIMIT based on liquidity data.

        For BUY: limit price = best_ask + small buffer (ensures fill)
        For SELL: limit price = best_bid - small buffer (ensures fill)
        """
        if order.order_type != OrderType.MARKET:
            return order  # Already LIMIT, don't change

        best_bid = liquidity.get("best_bid", 0)
        best_ask = liquidity.get("best_ask", 0)

        if order.transaction_type == TransactionType.BUY:
            # Aggressive BUY: at best ask + 0.5% buffer
            limit_price = round(best_ask * 1.005, 2) if best_ask > 0 else None
        else:
            # Aggressive SELL: at best bid - 0.5% buffer
            limit_price = round(best_bid * 0.995, 2) if best_bid > 0 else None

        if limit_price and limit_price > 0:
            updated = order.model_copy(update={
                "order_type": OrderType.LIMIT,
                "price": limit_price,
            })
            logger.info(
                "market_to_aggressive_limit",
                symbol=order.tradingsymbol,
                direction=order.transaction_type.value,
                limit_price=limit_price,
                best_bid=best_bid,
                best_ask=best_ask,
            )
            return updated

        return order  # Fallback: keep as MARKET


# ─────────────────────────────────────────────────────────────
# 5. Timestamp Precedence Resolver
# ─────────────────────────────────────────────────────────────

class TimestampResolver:
    """Resolve WebSocket vs REST race conditions using broker timestamps.

    Problem:
        REST poll returns OPEN (stale snapshot)
        WS delivers COMPLETE (newer)
        REST overwrites state with older data

    Rule:
        newer(exchange_update_timestamp) wins — ALWAYS.

    Stores latest known timestamp per order_id to reject stale updates.
    """

    def __init__(self) -> None:
        self._order_timestamps: dict[str, str] = {}  # order_id → latest timestamp
        self._order_statuses: dict[str, str] = {}  # order_id → latest status
        self._lock = asyncio.Lock()

    async def should_update(
        self,
        order_id: str,
        new_status: str,
        new_timestamp: Optional[str],
    ) -> bool:
        """Check if this update is newer than what we already have.

        Args:
            order_id: Broker order ID
            new_status: Incoming status (OPEN, COMPLETE, REJECTED, etc.)
            new_timestamp: Incoming exchange_update_timestamp

        Returns:
            True if this update should be applied (newer or same)
            False if stale (older timestamp — reject)
        """
        async with self._lock:
            existing_ts = self._order_timestamps.get(order_id)
            existing_status = self._order_statuses.get(order_id)

            # No existing record — always accept
            if not existing_ts:
                if new_timestamp:
                    self._order_timestamps[order_id] = new_timestamp
                self._order_statuses[order_id] = new_status
                return True

            # Terminal statuses should never be overwritten by non-terminal
            terminal_statuses = {"COMPLETE", "REJECTED", "CANCELLED"}
            if existing_status in terminal_statuses and new_status not in terminal_statuses:
                logger.warning(
                    "stale_update_rejected_terminal",
                    order_id=order_id,
                    existing=existing_status,
                    incoming=new_status,
                )
                return False

            # Compare timestamps if both available
            if new_timestamp and existing_ts:
                try:
                    # Kite timestamps: "2024-02-21 10:30:45" or ISO format
                    new_dt = self._parse_timestamp(new_timestamp)
                    existing_dt = self._parse_timestamp(existing_ts)

                    if new_dt < existing_dt:
                        logger.warning(
                            "stale_update_rejected_timestamp",
                            order_id=order_id,
                            existing_ts=existing_ts,
                            incoming_ts=new_timestamp,
                            existing_status=existing_status,
                            incoming_status=new_status,
                        )
                        return False

                    # Newer or equal — accept
                    self._order_timestamps[order_id] = new_timestamp
                    self._order_statuses[order_id] = new_status
                    return True
                except (ValueError, TypeError):
                    # Can't parse timestamps — fallback to accepting
                    pass

            # No timestamp comparison possible — accept
            if new_timestamp:
                self._order_timestamps[order_id] = new_timestamp
            self._order_statuses[order_id] = new_status
            return True

    def clear_order(self, order_id: str) -> None:
        """Remove tracking for a completed/settled order."""
        self._order_timestamps.pop(order_id, None)
        self._order_statuses.pop(order_id, None)

    @staticmethod
    def _parse_timestamp(ts: str) -> datetime:
        """Parse Kite Connect timestamp formats."""
        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S.%f",
        ):
            try:
                return datetime.strptime(ts, fmt)
            except ValueError:
                continue
        raise ValueError(f"Cannot parse timestamp: {ts}")


# ─────────────────────────────────────────────────────────────
# 6. Order Coordinator (Main Orchestrator)
# ─────────────────────────────────────────────────────────────

class OrderCoordinator:
    """Central coordinator for all broker microstructure concerns.

    Responsibilities:
        1. Freeze quantity splitting (BEFORE margin simulation)
        2. Market state guard (pre-open, circuit, auction)
        3. Rate-paced execution queue (~8 orders/sec)
        4. Liquidity checks before MARKET fallbacks
        5. Timestamp arbitration (WS vs REST)

    Architecture:
        ExecutionEngine → OrderCoordinator → KiteClient

    This layer isolates broker microstructure logic from strategy & execution.
    """

    def __init__(self, client: KiteClient) -> None:
        self._client = client
        self.freeze_mgr = FreezeQuantityManager()
        self.market_guard = MarketStateGuard(client)
        self.exec_queue = ExecutionQueue(client)
        self.liquidity = LiquidityChecker(client)
        self.timestamp_resolver = TimestampResolver()

    async def start(self) -> None:
        """Start background workers (queue processor)."""
        await self.exec_queue.start()

    async def stop(self) -> None:
        """Stop background workers."""
        await self.exec_queue.stop()

    # ── Primary API ──────────────────────────────────────────

    async def place_order(
        self,
        order: OrderRequest,
        priority: int = 3,
        check_market_state: bool = True,
        check_liquidity: bool = True,
    ) -> list[str]:
        """Place order through the full coordinator pipeline.

        Pipeline:
            1. Market state check
            2. Freeze split
            3. Liquidity check (for MARKET orders)
            4. Queue for rate-paced execution

        Args:
            order: OrderRequest to place
            priority: Queue priority (0=emergency, 1=protection, 2=risk, 3=normal)
            check_market_state: Whether to check market state
            check_liquidity: Whether to check liquidity for MARKET orders

        Returns:
            list of order_ids (1 if no split, N if freeze-split)
        """
        # Step 1: Market state check
        if check_market_state:
            safe, reason = await self.market_guard.is_safe_to_execute(
                order.tradingsymbol, order.exchange
            )
            if not safe:
                logger.warning(
                    "order_blocked_market_state",
                    symbol=order.tradingsymbol,
                    reason=reason,
                )
                raise MarketStateError(reason)

        # Step 2: Freeze split
        chunks = self.freeze_mgr.split_order(order)

        # Step 3: Liquidity check (for MARKET orders)
        if check_liquidity and order.order_type == OrderType.MARKET:
            liq = await self.liquidity.check_liquidity(
                order.tradingsymbol, order.exchange
            )
            if liq["recommendation"] == "ABORT":
                raise LiquidityError(
                    f"No liquidity for {order.tradingsymbol}: {liq['reason']}"
                )
            if liq["recommendation"] == "LIMIT":
                # Convert all chunks to aggressive LIMIT
                chunks = [
                    self.liquidity.convert_to_aggressive_limit(c, liq)
                    for c in chunks
                ]

        # Step 4: Enqueue each chunk
        order_ids: list[str] = []
        for chunk in chunks:
            order_id = await self.exec_queue.enqueue(chunk, priority=priority)
            order_ids.append(order_id)

        return order_ids

    async def place_order_immediate(self, order: OrderRequest) -> str:
        """Place order immediately, bypassing queue. Emergency hedges only."""
        return await self.exec_queue.enqueue_immediate(order)

    async def check_and_update_order_status(
        self,
        order_id: str,
        new_status: str,
        new_timestamp: Optional[str],
    ) -> bool:
        """Check if an order status update should be applied.

        Uses timestamp resolver to prevent WS/REST race conditions.
        """
        return await self.timestamp_resolver.should_update(
            order_id, new_status, new_timestamp
        )

    def split_signals_for_freeze(self, signals: list[Signal]) -> list[Signal]:
        """Split signals into freeze-compliant chunks BEFORE margin simulation.

        This is the critical fix: freeze split must happen BEFORE margin check.
        """
        result: list[Signal] = []
        for signal in signals:
            result.extend(self.freeze_mgr.split_signal(signal))
        return result

    def get_status(self) -> dict[str, Any]:
        """Get coordinator status for monitoring."""
        return {
            "queue_depth": self.exec_queue.queue_depth,
            "queue_running": self.exec_queue._running,
        }


# ─────────────────────────────────────────────────────────────
# Custom Exceptions
# ─────────────────────────────────────────────────────────────

class MarketStateError(Exception):
    """Raised when market state prevents order execution."""
    pass


class LiquidityError(Exception):
    """Raised when instrument has no liquidity."""
    pass
