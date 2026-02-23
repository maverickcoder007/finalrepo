from __future__ import annotations

import asyncio
import json
import struct
from datetime import datetime
from typing import Any, Callable, Coroutine, Optional

import websockets
from websockets.asyncio.client import ClientConnection as WebSocketClientProtocol

from src.data.models import (
    DepthItem,
    MarketDepth,
    OHLC,
    Tick,
    TickMode,
)
from src.auth.authenticator import KiteAuthenticator
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

OnTickCallback = Callable[[list[Tick]], Coroutine[Any, Any, None]]
OnConnectCallback = Callable[[], Coroutine[Any, Any, None]]
OnCloseCallback = Callable[[int, str], Coroutine[Any, Any, None]]
OnErrorCallback = Callable[[Exception], Coroutine[Any, Any, None]]
OnOrderCallback = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]


class KiteTicker:
    EXCHANGE_MAP = {
        1: "NSE", 2: "NFO", 3: "CDS", 4: "BSE",
        5: "BFO", 6: "BCD", 7: "MCX", 8: "MCXSX",
        9: "INDICES",
    }

    def __init__(self, authenticator: Optional[KiteAuthenticator] = None) -> None:
        self._settings = get_settings()
        self._auth = authenticator or KiteAuthenticator()
        self._ws: Optional[WebSocketClientProtocol] = None
        self._subscribed_tokens: dict[int, str] = {}
        self._tick_cache: dict[int, Tick] = {}
        self._running = False
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 50
        self._reconnect_delay = 5

        self.on_ticks: Optional[OnTickCallback] = None
        self.on_connect: Optional[OnConnectCallback] = None
        self.on_close: Optional[OnCloseCallback] = None
        self.on_error: Optional[OnErrorCallback] = None
        self.on_order_update: Optional[OnOrderCallback] = None

    @property
    def is_connected(self) -> bool:
        if self._ws is None:
            return False
        try:
            # websockets v14+: use .state.name or check close_code
            state = getattr(self._ws, 'state', None)
            if state is not None:
                return str(state.name) == 'OPEN' if hasattr(state, 'name') else True
            # websockets v10-v13: .open attribute
            if hasattr(self._ws, 'open'):
                return bool(self._ws.open)
            # Fallback: if close_code is set, the connection is closed
            return getattr(self._ws, 'close_code', None) is None
        except Exception:
            return False

    def get_tick_cache(self) -> dict[int, Tick]:
        return self._tick_cache.copy()

    def get_cached_tick(self, token: int) -> Optional[Tick]:
        return self._tick_cache.get(token)

    async def connect(self) -> None:
        self._running = True
        while self._running and self._reconnect_attempts < self._max_reconnect_attempts:
            try:
                ws_url = (
                    f"{self._settings.kite_ws_url}"
                    f"?api_key={self._auth.api_key}"
                    f"&access_token={self._auth.access_token}"
                )
                self._ws = await websockets.connect(ws_url, ping_interval=20, ping_timeout=10)
                self._reconnect_attempts = 0

                logger.info("websocket_connected")
                if self.on_connect:
                    await self.on_connect()

                if self._subscribed_tokens:
                    await self._resubscribe()

                await self._listen()

            except websockets.ConnectionClosed as e:
                logger.warning("websocket_closed", code=e.code, reason=e.reason)
                if self.on_close:
                    await self.on_close(e.code, e.reason)
            except Exception as e:
                logger.error("websocket_error", error=str(e))
                if self.on_error:
                    await self.on_error(e)

            if self._running:
                self._reconnect_attempts += 1
                delay = min(self._reconnect_delay * self._reconnect_attempts, 60)
                logger.info("websocket_reconnecting", attempt=self._reconnect_attempts, delay=delay)
                await asyncio.sleep(delay)

    async def disconnect(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
        logger.info("websocket_disconnected")

    async def subscribe(self, tokens: list[int], mode: TickMode = TickMode.QUOTE) -> None:
        for token in tokens:
            self._subscribed_tokens[token] = mode.value

        if self.is_connected and self._ws:
            msg = json.dumps({"a": "subscribe", "v": tokens})
            await self._ws.send(msg)
            await self._set_mode(tokens, mode)

    async def unsubscribe(self, tokens: list[int]) -> None:
        for token in tokens:
            self._subscribed_tokens.pop(token, None)
            self._tick_cache.pop(token, None)

        if self.is_connected and self._ws:
            msg = json.dumps({"a": "unsubscribe", "v": tokens})
            await self._ws.send(msg)

    async def _set_mode(self, tokens: list[int], mode: TickMode) -> None:
        if self.is_connected and self._ws:
            msg = json.dumps({"a": "mode", "v": [mode.value, tokens]})
            await self._ws.send(msg)

    async def _resubscribe(self) -> None:
        mode_groups: dict[str, list[int]] = {}
        for token, mode in self._subscribed_tokens.items():
            mode_groups.setdefault(mode, []).append(token)

        all_tokens = list(self._subscribed_tokens.keys())
        if all_tokens and self._ws:
            msg = json.dumps({"a": "subscribe", "v": all_tokens})
            await self._ws.send(msg)

            for mode, tokens in mode_groups.items():
                await self._set_mode(tokens, TickMode(mode))

    async def _listen(self) -> None:
        if not self._ws:
            return
        async for message in self._ws:
            if isinstance(message, bytes):
                ticks = self._parse_binary(message)
                if ticks:
                    for tick in ticks:
                        self._tick_cache[tick.instrument_token] = tick
                    if self.on_ticks:
                        await self.on_ticks(ticks)
            elif isinstance(message, str):
                try:
                    data = json.loads(message)
                    if data.get("type") == "order" and self.on_order_update:
                        await self.on_order_update(data.get("data", {}))
                except json.JSONDecodeError:
                    pass

    def _parse_binary(self, data: bytes) -> list[Tick]:
        if len(data) < 2:
            return []

        num_packets = struct.unpack(">H", data[:2])[0]
        offset = 2
        ticks: list[Tick] = []

        for _ in range(num_packets):
            if offset + 2 > len(data):
                break
            packet_len = struct.unpack(">H", data[offset:offset + 2])[0]
            offset += 2

            if offset + packet_len > len(data):
                break

            packet = data[offset:offset + packet_len]
            offset += packet_len

            tick = self._parse_packet(packet)
            if tick:
                ticks.append(tick)

        return ticks

    def _parse_packet(self, packet: bytes) -> Optional[Tick]:
        packet_len = len(packet)
        if packet_len < 8:
            return None

        instrument_token = struct.unpack(">I", packet[0:4])[0]
        segment = instrument_token & 0xFF
        # Divisor per official Kite SDK: CDS=10M, BCD=10K, else=100
        if segment == 3:      # CDS
            divisor = 10000000.0
        elif segment == 6:    # BCD
            divisor = 10000.0
        else:
            divisor = 100.0
        is_index = (segment == 9)  # Indices segment

        # ── LTP mode (8 bytes) ──────────────────────────────
        if packet_len == 8:
            last_price = struct.unpack(">i", packet[4:8])[0] / divisor
            return Tick(
                instrument_token=instrument_token,
                mode=TickMode.LTP.value,
                tradable=not is_index,
                last_price=last_price,
            )

        # ── Index packets (28 = full, 32 = quote with timestamp) ──
        if is_index and (packet_len == 28 or packet_len == 32):
            values = struct.unpack(">IIIII", packet[4:24])
            last_price = values[0] / divisor
            high = values[1] / divisor
            low = values[2] / divisor
            open_ = values[3] / divisor
            close = values[4] / divisor

            change = (last_price - close) if close != 0 else 0.0

            tick = Tick(
                instrument_token=instrument_token,
                mode=TickMode.QUOTE.value if packet_len == 28 else TickMode.FULL.value,
                tradable=False,
                last_price=last_price,
                ohlc=OHLC(open=open_, high=high, low=low, close=close),
                change=change,
            )
            if packet_len == 32:
                try:
                    ts = struct.unpack(">I", packet[28:32])[0]
                    if ts:
                        tick.exchange_timestamp = datetime.fromtimestamp(ts)
                except Exception:
                    pass
            return tick

        # ── Non-index Quote (44 bytes) / Full (184 bytes) ───
        if packet_len >= 44:
            values = struct.unpack(">IIIIII", packet[4:28])
            ohlc_values = struct.unpack(">IIII", packet[28:44])

            last_price = values[0] / divisor
            close = ohlc_values[3] / divisor

            tick = Tick(
                instrument_token=instrument_token,
                mode=TickMode.QUOTE.value if packet_len == 44 else TickMode.FULL.value,
                last_price=last_price,
                last_traded_quantity=values[1],
                average_traded_price=values[2] / divisor,
                volume_traded=values[3],
                total_buy_quantity=values[4],
                total_sell_quantity=values[5],
                ohlc=OHLC(
                    open=ohlc_values[0] / divisor,
                    high=ohlc_values[1] / divisor,
                    low=ohlc_values[2] / divisor,
                    close=close,
                ),
                # Compute change from OHLC (not read from packet)
                change=(last_price - close) if close != 0 else 0.0,
            )

            # Full mode: parse extra fields after OHLC (offset 44+)
            if packet_len >= 184:
                # Offset 44-47: last_trade_time (uint32 unix timestamp)
                try:
                    ltt = struct.unpack(">I", packet[44:48])[0]
                    if ltt:
                        tick.last_trade_time = datetime.fromtimestamp(ltt)
                except Exception:
                    pass

                # Offset 48-51: oi, 52-55: oi_day_high, 56-59: oi_day_low
                oi_vals = struct.unpack(">III", packet[48:60])
                tick.oi = oi_vals[0]
                tick.oi_day_high = oi_vals[1]
                tick.oi_day_low = oi_vals[2]

                # Offset 60-63: exchange_timestamp (uint32 unix timestamp)
                try:
                    ets = struct.unpack(">I", packet[60:64])[0]
                    if ets:
                        tick.exchange_timestamp = datetime.fromtimestamp(ets)
                except Exception:
                    pass

                # Offset 64-183: market depth (120 bytes)
                if packet_len >= 184:
                    tick.depth = self._parse_depth(packet[64:184], divisor)

            return tick

        return None

    def _parse_depth(self, data: bytes, divisor: float) -> MarketDepth:
        buy: list[DepthItem] = []
        sell: list[DepthItem] = []

        for i in range(5):
            base = i * 12
            if base + 12 > len(data):
                break
            qty, price, orders = struct.unpack(">IIH", data[base:base + 10])
            buy.append(DepthItem(
                quantity=qty,
                price=price / divisor,
                orders=orders,
            ))

        for i in range(5):
            base = 60 + i * 12
            if base + 12 > len(data):
                break
            qty, price, orders = struct.unpack(">IIH", data[base:base + 10])
            sell.append(DepthItem(
                quantity=qty,
                price=price / divisor,
                orders=orders,
            ))

        return MarketDepth(buy=buy, sell=sell)
