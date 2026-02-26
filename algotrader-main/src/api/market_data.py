"""
Market Data Service — Historical Charts, Live Quotes, Reports
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Provides:
• Historical OHLCV chart data in multiple intervals
• Live market quotes (LTP, OHLC, full depth)
• Portfolio reports (P&L, holdings summary, trade log)
• Instrument search with smart caching
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.api.client import KiteClient
from src.data.market_data_store import MarketDataStore
from src.data.models import Interval
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ────────────────────────────────────────────────────────────────
# Interval Mapping  (string → Interval enum for KiteClient)
# ────────────────────────────────────────────────────────────────

INTERVAL_MAP: dict[str, Interval] = {
    "minute": Interval("minute"),
    "3minute": Interval("3minute"),
    "5minute": Interval("5minute"),
    "10minute": Interval("10minute"),
    "15minute": Interval("15minute"),
    "30minute": Interval("30minute"),
    "60minute": Interval("60minute"),
    "day": Interval("day"),
}

# ────────────────────────────────────────────────────────────────
# Kite API max-days limits per interval (enforced server-side)
# We use slightly conservative values to avoid edge-case rejections.
# ────────────────────────────────────────────────────────────────
KITE_MAX_DAYS: dict[str, int] = {
    "minute": 55,       # official 60
    "3minute": 95,      # official 100
    "5minute": 95,      # official 100
    "10minute": 95,     # official 100
    "15minute": 195,    # official 200
    "30minute": 195,    # official 200
    "60minute": 395,    # official 400
    "day": 1995,        # official 2000
}

# ────────────────────────────────────────────────────────────────
# Market Data Provider
# ────────────────────────────────────────────────────────────────

class MarketDataProvider:
    """Unified market data provider for historical and live data."""

    def __init__(self, client: KiteClient, store: MarketDataStore | None = None) -> None:
        self._client = client
        self._store = store
        self._instrument_cache: dict[str, list[dict]] = {}
        self._token_map: dict[str, int] = {}  # symbol -> instrument_token
        self._chart_cache: dict[str, dict] = {}  # cache key -> OHLCV data

    # ────────────────────────────────────────────────────────
    # Instrument Resolution
    # ────────────────────────────────────────────────────────

    async def resolve_token(self, symbol: str, exchange: str = "NSE") -> int | None:
        """Resolve tradingsymbol to instrument token."""
        key = f"{exchange}:{symbol}"
        if key in self._token_map:
            return self._token_map[key]

        # Try DB cache first
        if self._store:
            cached = self._store.get_instruments(exchange)
            if cached:
                for inst in cached:
                    ck = f"{inst['exchange']}:{inst['tradingsymbol']}"
                    self._token_map[ck] = inst["instrument_token"]
                if key in self._token_map:
                    return self._token_map[key]

        try:
            instruments = await self._client.get_instruments(exchange)
            for inst in instruments:
                cache_key = f"{inst.exchange}:{inst.tradingsymbol}"
                self._token_map[cache_key] = inst.instrument_token
            # Store to DB
            if self._store:
                self._store.store_instruments(exchange, [
                    {"instrument_token": i.instrument_token, "exchange_token": i.exchange_token,
                     "tradingsymbol": i.tradingsymbol, "name": i.name,
                     "last_price": i.last_price, "expiry": i.expiry,
                     "strike": i.strike, "tick_size": i.tick_size,
                     "lot_size": i.lot_size, "instrument_type": i.instrument_type,
                     "segment": i.segment}
                    for i in instruments
                ])
            return self._token_map.get(key)
        except Exception as e:
            logger.error("token_resolve_failed", symbol=symbol, error=str(e))
            return None

    async def search_instruments(
        self, query: str, exchange: str = "NSE", limit: int = 20
    ) -> list[dict[str, Any]]:
        """Search instruments by name/symbol.  Checks DB cache first."""
        # Try DB cache first
        if self._store:
            cached = self._store.search_instruments(query, exchange=exchange, limit=limit)
            if cached:
                return cached

        try:
            instruments = await self._client.get_instruments(exchange)
            q = query.upper()
            matches = []
            for inst in instruments:
                if q in inst.tradingsymbol.upper() or q in (inst.name or "").upper():
                    matches.append({
                        "instrument_token": inst.instrument_token,
                        "tradingsymbol": inst.tradingsymbol,
                        "name": inst.name,
                        "exchange": inst.exchange,
                        "segment": inst.segment,
                        "instrument_type": inst.instrument_type,
                        "lot_size": inst.lot_size,
                        "tick_size": inst.tick_size,
                        "expiry": inst.expiry,
                        "strike": inst.strike,
                    })
                    if len(matches) >= limit:
                        break
            # Cache full instrument set to DB while we have it
            if self._store:
                self._store.store_instruments(exchange, [
                    {"instrument_token": i.instrument_token, "exchange_token": i.exchange_token,
                     "tradingsymbol": i.tradingsymbol, "name": i.name,
                     "last_price": i.last_price, "expiry": i.expiry,
                     "strike": i.strike, "tick_size": i.tick_size,
                     "lot_size": i.lot_size, "instrument_type": i.instrument_type,
                     "segment": i.segment}
                    for i in instruments
                ])
            return matches
        except Exception as e:
            logger.error("instrument_search_failed", error=str(e))
            return []

    async def get_instruments(self, exchange: str = "NSE") -> list[dict[str, Any]]:
        """Get all instruments for an exchange.  DB-cached (refreshed daily).

        Always returns a list of *dicts* (not Pydantic models) so callers
        get a uniform interface whether data came from the cache or API.
        """
        # 1. Try DB cache (max 1 day old)
        if self._store:
            cached = self._store.get_instruments(exchange)
            if cached:
                logger.info("instruments_cache_hit", exchange=exchange, count=len(cached))
                return cached

        # 2. Fetch from Kite API
        try:
            raw = await self._client.get_instruments(exchange)
            dicts = [
                {
                    "instrument_token": i.instrument_token,
                    "exchange_token": getattr(i, "exchange_token", 0),
                    "tradingsymbol": i.tradingsymbol,
                    "name": i.name,
                    "last_price": i.last_price,
                    "expiry": i.expiry,
                    "strike": i.strike,
                    "tick_size": i.tick_size,
                    "lot_size": i.lot_size,
                    "instrument_type": i.instrument_type,
                    "segment": i.segment,
                    "exchange": exchange,
                }
                for i in raw
            ]
            # 3. Persist to DB
            if self._store:
                self._store.store_instruments(exchange, dicts)
                logger.info("instruments_stored", exchange=exchange, count=len(dicts))
            # Also populate in-memory token map
            for d in dicts:
                key = f"{d['exchange']}:{d['tradingsymbol']}"
                self._token_map[key] = d["instrument_token"]
            return dicts
        except Exception as e:
            logger.error("instruments_fetch_failed", exchange=exchange, error=str(e))
            return []

    async def fetch_historical_candles(
        self,
        instrument_token: int,
        interval: str = "5minute",
        from_date: str | None = None,
        to_date: str | None = None,
        days: int = 5,
    ) -> list:
        """Public wrapper around _fetch_historical_chunked returning raw HistoricalCandle objects.

        This is the preferred entry point for code that needs candle
        objects rather than a DataFrame.  It goes through the DB cache.
        """
        kite_interval = INTERVAL_MAP.get(interval, Interval("5minute"))
        if to_date is None:
            to_date = datetime.now().strftime("%Y-%m-%d")
        if from_date is None:
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        return await self._fetch_historical_chunked(
            instrument_token, kite_interval, interval, from_date, to_date
        )

    # ────────────────────────────────────────────────────────
    # Historical Chart Data
    # ────────────────────────────────────────────────────────

    async def get_historical_chart(
        self,
        instrument_token: int,
        interval: str = "day",
        days: int = 365,
        from_date: str | None = None,
        to_date: str | None = None,
        include_indicators: bool = False,
    ) -> dict[str, Any]:
        """
        Fetch historical OHLCV data for charting.
        Returns data formatted for candlestick chart rendering.
        """
        kite_interval = INTERVAL_MAP.get(interval)
        if not kite_interval:
            return {"error": f"Invalid interval: {interval}. Valid: {list(INTERVAL_MAP.keys())}"}

        if to_date is None:
            to_date = datetime.now().strftime("%Y-%m-%d")
        if from_date is None:
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        try:
            candles = await self._fetch_historical_chunked(
                instrument_token, kite_interval, interval, from_date, to_date
            )

            if not candles:
                return {"error": "No data returned for the given parameters"}

            # Build OHLCV arrays for charting
            timestamps = []
            opens = []
            highs = []
            lows = []
            closes = []
            volumes = []

            for c in candles:
                timestamps.append(c.timestamp)
                opens.append(c.open)
                highs.append(c.high)
                lows.append(c.low)
                closes.append(c.close)
                volumes.append(c.volume)

            # Build array-of-objects for JS candlestick rendering
            candle_list = []
            for i in range(len(timestamps)):
                candle_list.append({
                    "timestamp": timestamps[i],
                    "open": opens[i],
                    "high": highs[i],
                    "low": lows[i],
                    "close": closes[i],
                    "volume": volumes[i],
                })

            result: dict[str, Any] = {
                "instrument_token": instrument_token,
                "interval": interval,
                "from_date": from_date,
                "to_date": to_date,
                "data_points": len(candles),
                "candles": candle_list,
            }

            if include_indicators and len(closes) >= 20:
                result["indicators"] = self._compute_chart_indicators(
                    closes, highs, lows, volumes
                )

            return result

        except Exception as e:
            logger.error("historical_chart_failed", error=str(e))
            return {"error": str(e)}

    async def _fetch_historical_chunked(
        self,
        instrument_token: int,
        kite_interval: Interval,
        interval_str: str,
        from_date: str,
        to_date: str,
    ) -> list:
        """
        Fetch historical candles in Kite-compliant date-range chunks.

        Checks the SQLite cache first.  Any candles fetched from the API
        are written to the cache for future use.

        Kite Connect enforces per-interval day limits.  When the requested
        range exceeds that limit we split into successive sub-ranges,
        fetch each one, and concatenate the results.
        """
        # ── 1. Try DB cache first ────────────────────────────────
        if self._store and self._store.has_candles(instrument_token, interval_str, from_date, to_date):
            cached = self._store.get_candles(instrument_token, interval_str, from_date, to_date)
            if cached:
                logger.info(
                    "historical_cache_hit",
                    token=instrument_token,
                    interval=interval_str,
                    bars=len(cached),
                )
                # Convert dicts back to HistoricalCandle-like objects
                from src.data.models import HistoricalCandle
                return [
                    HistoricalCandle(
                        timestamp=c["ts"],
                        open=c["open"],
                        high=c["high"],
                        low=c["low"],
                        close=c["close"],
                        volume=c["volume"],
                        oi=c.get("oi"),
                    )
                    for c in cached
                ]

        # ── 2. Fetch from Kite API ───────────────────────────────
        max_days = KITE_MAX_DAYS.get(interval_str, 195)
        fmt = "%Y-%m-%d"
        start = datetime.strptime(from_date, fmt)
        end = datetime.strptime(to_date, fmt)

        all_candles: list = []
        total_days = (end - start).days
        if total_days <= max_days:
            # Single request is within limits
            all_candles = await self._client.get_historical_data(
                instrument_token=instrument_token,
                interval=kite_interval,
                from_date=from_date,
                to_date=to_date,
            )
        else:
            # --- Chunk the date range ---
            chunk_start = start
            while chunk_start < end:
                chunk_end = min(chunk_start + timedelta(days=max_days), end)
                logger.info(
                    "historical_chunk_fetch",
                    token=instrument_token,
                    interval=interval_str,
                    chunk_from=chunk_start.strftime(fmt),
                    chunk_to=chunk_end.strftime(fmt),
                )
                try:
                    candles = await self._client.get_historical_data(
                        instrument_token=instrument_token,
                        interval=kite_interval,
                        from_date=chunk_start.strftime(fmt),
                        to_date=chunk_end.strftime(fmt),
                    )
                    if candles:
                        all_candles.extend(candles)
                except Exception as e:
                    logger.error(
                        "historical_chunk_error",
                        chunk_from=chunk_start.strftime(fmt),
                        chunk_to=chunk_end.strftime(fmt),
                        error=str(e),
                    )
                # Next chunk starts from the day after the current chunk end
                chunk_start = chunk_end + timedelta(days=1)
                # Small delay to be kind to the API rate-limiter
                await asyncio.sleep(0.3)

        # ── 3. Persist fetched candles to DB cache ───────────────
        if self._store and all_candles:
            store_rows = [
                {
                    "ts": c.timestamp,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                    "oi": c.oi,
                }
                for c in all_candles
            ]
            stored = self._store.store_candles(instrument_token, interval_str, store_rows)
            logger.info(
                "historical_cache_stored",
                token=instrument_token,
                interval=interval_str,
                bars=stored,
            )

        return all_candles

    async def get_historical_df(
        self,
        instrument_token: int,
        interval: str = "day",
        days: int = 365,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> pd.DataFrame:
        """Fetch historical data as a DataFrame for backtesting."""
        kite_interval = INTERVAL_MAP.get(interval, Interval("day"))

        if to_date is None:
            to_date = datetime.now().strftime("%Y-%m-%d")
        if from_date is None:
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        candles = await self._fetch_historical_chunked(
            instrument_token, kite_interval, interval, from_date, to_date
        )

        if not candles:
            return pd.DataFrame()

        data = []
        for c in candles:
            data.append({
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            })

        df = pd.DataFrame(data)
        if not df.empty and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
        return df

    def _compute_chart_indicators(
        self,
        closes: list, highs: list, lows: list, volumes: list
    ) -> dict[str, Any]:
        """Compute basic chart overlay indicators."""
        c = np.array(closes, dtype=float)
        h = np.array(highs, dtype=float)
        lo = np.array(lows, dtype=float)

        indicators: dict[str, Any] = {}

        # SMA 20, 50, 200
        for period in [20, 50, 200]:
            if len(c) >= period:
                sma_vals = pd.Series(c).rolling(period).mean().tolist()
                indicators[f"sma_{period}"] = [
                    round(v, 2) if v == v and v is not None else None for v in sma_vals
                ]

        # EMA 9, 21
        for period in [9, 21]:
            if len(c) >= period:
                ema_vals = pd.Series(c).ewm(span=period, adjust=False).mean().tolist()
                indicators[f"ema_{period}"] = [
                    round(v, 2) if v == v and v is not None else None for v in ema_vals
                ]

        # RSI 14
        if len(c) >= 15:
            delta = np.diff(c)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = pd.Series(gain).rolling(14).mean()
            avg_loss = pd.Series(loss).rolling(14).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            indicators["rsi_14"] = [None] + [
                round(v, 2) if pd.notna(v) else None for v in rsi.tolist()
            ]

        # Bollinger Bands
        if len(c) >= 20:
            sma20 = pd.Series(c).rolling(20).mean()
            std20 = pd.Series(c).rolling(20).std()
            indicators["bb_upper"] = [
                round(v, 2) if pd.notna(v) else None for v in (sma20 + 2 * std20).tolist()
            ]
            indicators["bb_lower"] = [
                round(v, 2) if pd.notna(v) else None for v in (sma20 - 2 * std20).tolist()
            ]

        # VWAP (intraday)
        if len(volumes) > 0 and sum(volumes) > 0:
            typical = (h + lo + c) / 3
            cum_tp_vol = np.cumsum(typical * np.array(volumes, dtype=float))
            cum_vol = np.cumsum(np.array(volumes, dtype=float))
            vwap = np.where(cum_vol > 0, cum_tp_vol / cum_vol, 0)
            indicators["vwap"] = [round(float(v), 2) for v in vwap]

        # Volume SMA 20
        if len(volumes) >= 20:
            vol_sma = pd.Series(volumes, dtype=float).rolling(20).mean().tolist()
            indicators["volume_sma_20"] = [
                round(v, 0) if pd.notna(v) else None for v in vol_sma
            ]

        return indicators

    # ────────────────────────────────────────────────────────
    # Live Market Quotes
    # ────────────────────────────────────────────────────────

    async def get_quote(self, instruments: list[str]) -> dict[str, Any]:
        """Get full market quote for instruments (e.g., ['NSE:RELIANCE', 'NSE:TCS'])."""
        quotes = {}
        try:
            quotes = await self._client.get_quote(instruments)
            result = {}
            for key, q in quotes.items():
                result[key] = {
                    "instrument_token": q.instrument_token,
                    "timestamp": str(q.timestamp) if q.timestamp else None,
                    "last_price": q.last_price,
                    "last_quantity": q.last_quantity,
                    "average_price": q.average_price,
                    "volume": q.volume,
                    "buy_quantity": q.buy_quantity,
                    "sell_quantity": q.sell_quantity,
                    "ohlc": {
                        "open": q.ohlc.open if q.ohlc else None,
                        "high": q.ohlc.high if q.ohlc else None,
                        "low": q.ohlc.low if q.ohlc else None,
                        "close": q.ohlc.close if q.ohlc else None,
                    },
                    "net_change": q.net_change,
                    "lower_circuit_limit": q.lower_circuit_limit,
                    "upper_circuit_limit": q.upper_circuit_limit,
                    "oi": q.oi,
                    "oi_day_high": q.oi_day_high,
                    "oi_day_low": q.oi_day_low,
                    "depth": {
                        "buy": [
                            {"price": d.price, "quantity": d.quantity, "orders": d.orders}
                            for d in (q.depth.buy if q.depth and q.depth.buy else [])
                        ],
                        "sell": [
                            {"price": d.price, "quantity": d.quantity, "orders": d.orders}
                            for d in (q.depth.sell if q.depth and q.depth.sell else [])
                        ],
                    } if q.depth else None,
                }
            return {"quotes": result}
        except Exception as e:
            logger.error("quote_fetch_failed", error=str(e))
            return {"error": str(e)}
        finally:
            # Store-behind: persist quote snapshots to DB
            if self._store and quotes:
                try:
                    snaps = [
                        {
                            "instrument_token": q.instrument_token,
                            "ts": str(q.timestamp) if q.timestamp else "",
                            "last_price": q.last_price,
                            "volume": q.volume or 0,
                            "oi": q.oi or 0,
                            "buy_quantity": q.buy_quantity or 0,
                            "sell_quantity": q.sell_quantity or 0,
                            "open": q.ohlc.open if q.ohlc else 0,
                            "high": q.ohlc.high if q.ohlc else 0,
                            "low": q.ohlc.low if q.ohlc else 0,
                            "close": q.ohlc.close if q.ohlc else 0,
                            "net_change": q.net_change or 0,
                        }
                        for q in quotes.values()
                    ]
                    self._store.store_quote_snapshots(snaps)
                except Exception:
                    pass

    async def get_ltp(self, instruments: list[str]) -> dict[str, Any]:
        """Get last traded price for instruments."""
        try:
            ltps = await self._client.get_ltp(instruments)
            result = {}
            for key, q in ltps.items():
                result[key] = {
                    "instrument_token": q.instrument_token,
                    "last_price": q.last_price,
                }
            return {"ltp": result}
        except Exception as e:
            logger.error("ltp_fetch_failed", error=str(e))
            return {"error": str(e)}

    async def get_ohlc(self, instruments: list[str]) -> dict[str, Any]:
        """Get OHLC data for instruments."""
        try:
            ohlcs = await self._client.get_ohlc(instruments)
            result = {}
            for key, q in ohlcs.items():
                result[key] = {
                    "instrument_token": q.instrument_token,
                    "last_price": q.last_price,
                    "ohlc": {
                        "open": q.ohlc.open if q.ohlc else None,
                        "high": q.ohlc.high if q.ohlc else None,
                        "low": q.ohlc.low if q.ohlc else None,
                        "close": q.ohlc.close if q.ohlc else None,
                    },
                }
            return {"ohlc": result}
        except Exception as e:
            logger.error("ohlc_fetch_failed", error=str(e))
            return {"error": str(e)}

    async def get_market_overview(self, symbols: list[str] | None = None) -> dict[str, Any]:
        """Get a quick market overview for watchlist symbols."""
        if not symbols:
            symbols = [
                "NIFTY 50", "NIFTY BANK", "RELIANCE", "TCS", "HDFCBANK",
                "INFY", "ICICIBANK", "SBIN", "BHARTIARTL", "ITC",
            ]

        instruments = [f"NSE:{s}" for s in symbols]
        try:
            quotes = await self._client.get_quote(instruments)
            overview = []
            for key, q in quotes.items():
                change = 0.0
                change_pct = 0.0
                if q.ohlc and q.ohlc.close and q.ohlc.close > 0:
                    change = q.last_price - q.ohlc.close
                    change_pct = (change / q.ohlc.close) * 100

                overview.append({
                    "symbol": key.split(":")[-1] if ":" in key else key,
                    "exchange": key.split(":")[0] if ":" in key else "NSE",
                    "last_price": q.last_price,
                    "change": round(change, 2),
                    "change_pct": round(change_pct, 2),
                    "volume": q.volume,
                    "open": q.ohlc.open if q.ohlc else None,
                    "high": q.ohlc.high if q.ohlc else None,
                    "low": q.ohlc.low if q.ohlc else None,
                    "close": q.ohlc.close if q.ohlc else None,
                    "buy_quantity": q.buy_quantity,
                    "sell_quantity": q.sell_quantity,
                })

            overview.sort(key=lambda x: abs(x.get("change_pct", 0)), reverse=True)
            return {"market_overview": overview, "count": len(overview)}
        except Exception as e:
            logger.error("market_overview_failed", error=str(e))
            return {"error": str(e)}


# ────────────────────────────────────────────────────────────────
# Portfolio Reports Service
# ────────────────────────────────────────────────────────────────

class PortfolioReporter:
    """Portfolio analysis and reporting service."""

    def __init__(self, client: KiteClient) -> None:
        self._client = client

    async def get_holdings_report(self) -> dict[str, Any]:
        """Comprehensive holdings analysis."""
        try:
            holdings = await self._client.get_holdings()
            if not holdings:
                return {"holdings": [], "summary": {"total_investment": 0, "current_value": 0, "total_pnl": 0}}

            items = []
            total_investment = 0.0
            current_value = 0.0
            total_pnl = 0.0

            for h in holdings:
                investment = h.average_price * h.quantity
                value = h.last_price * h.quantity
                pnl = value - investment
                pnl_pct = (pnl / investment * 100) if investment > 0 else 0.0
                day_change = (h.last_price - h.close_price) * h.quantity if h.close_price else 0.0

                total_investment += investment
                current_value += value
                total_pnl += pnl

                items.append({
                    "tradingsymbol": h.tradingsymbol,
                    "exchange": h.exchange,
                    "quantity": h.quantity,
                    "average_price": round(h.average_price, 2),
                    "last_price": round(h.last_price, 2),
                    "close_price": round(h.close_price, 2) if h.close_price else None,
                    "investment": round(investment, 2),
                    "current_value": round(value, 2),
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "day_change": round(day_change, 2),
                    "weight_pct": 0.0,  # filled below
                })

            # Calculate weights
            for item in items:
                item["weight_pct"] = round(
                    (item["current_value"] / current_value * 100) if current_value > 0 else 0, 2
                )

            # Sort by value
            items.sort(key=lambda x: x["current_value"], reverse=True)

            # Top performers / losers
            by_pnl = sorted(items, key=lambda x: x["pnl_pct"], reverse=True)

            return {
                "holdings": items,
                "summary": {
                    "total_investment": round(total_investment, 2),
                    "current_value": round(current_value, 2),
                    "total_pnl": round(total_pnl, 2),
                    "total_pnl_pct": round((total_pnl / total_investment * 100) if total_investment > 0 else 0, 2),
                    "holdings_count": len(items),
                    "profitable_count": sum(1 for i in items if i["pnl"] > 0),
                    "loss_count": sum(1 for i in items if i["pnl"] <= 0),
                },
                "top_gainers": by_pnl[:5],
                "top_losers": by_pnl[-5:][::-1] if len(by_pnl) > 5 else [],
            }
        except Exception as e:
            logger.error("holdings_report_failed", error=str(e))
            return {"error": str(e)}

    async def get_positions_report(self) -> dict[str, Any]:
        """Comprehensive positions analysis."""
        try:
            positions = await self._client.get_positions()
            net = positions.net
            day = positions.day

            items = []
            total_pnl = 0.0
            realized_pnl = 0.0
            unrealized_pnl = 0.0

            for p in net:
                pnl = p.pnl
                total_pnl += pnl
                is_open = p.quantity != 0

                m2m = p.m2m if hasattr(p, 'm2m') else 0.0
                if is_open:
                    unrealized_pnl += pnl
                else:
                    realized_pnl += pnl

                items.append({
                    "tradingsymbol": p.tradingsymbol,
                    "exchange": p.exchange,
                    "product": p.product,
                    "quantity": p.quantity,
                    "average_price": round(p.average_price, 2),
                    "last_price": round(p.last_price, 2),
                    "pnl": round(pnl, 2),
                    "m2m": round(m2m, 2) if m2m else 0.0,
                    "buy_quantity": p.buy_quantity,
                    "sell_quantity": p.sell_quantity,
                    "buy_price": round(p.buy_price, 2),
                    "sell_price": round(p.sell_price, 2),
                    "is_open": is_open,
                    "direction": "LONG" if p.quantity > 0 else "SHORT" if p.quantity < 0 else "CLOSED",
                })

            items.sort(key=lambda x: abs(x["pnl"]), reverse=True)

            return {
                "positions": items,
                "summary": {
                    "total_pnl": round(total_pnl, 2),
                    "realized_pnl": round(realized_pnl, 2),
                    "unrealized_pnl": round(unrealized_pnl, 2),
                    "open_positions": sum(1 for i in items if i["is_open"]),
                    "closed_positions": sum(1 for i in items if not i["is_open"]),
                    "total_positions": len(items),
                    "profitable": sum(1 for i in items if i["pnl"] > 0),
                    "loss_making": sum(1 for i in items if i["pnl"] < 0),
                },
            }
        except Exception as e:
            logger.error("positions_report_failed", error=str(e))
            return {"error": str(e)}

    async def get_trades_report(self) -> dict[str, Any]:
        """Today's executed trades with analytics."""
        try:
            trades = await self._client.get_trades()
            items = []
            total_turnover = 0.0
            buy_value = 0.0
            sell_value = 0.0

            for t in trades:
                value = t.average_price * t.filled_quantity
                total_turnover += value

                if t.transaction_type == "BUY":
                    buy_value += value
                else:
                    sell_value += value

                items.append({
                    "trade_id": t.trade_id,
                    "order_id": t.order_id,
                    "tradingsymbol": t.tradingsymbol,
                    "exchange": t.exchange,
                    "transaction_type": t.transaction_type,
                    "quantity": t.filled_quantity,
                    "average_price": round(t.average_price, 2),
                    "value": round(value, 2),
                    "product": t.product,
                    "fill_timestamp": str(t.fill_timestamp) if hasattr(t, 'fill_timestamp') and t.fill_timestamp else str(t.exchange_timestamp) if hasattr(t, 'exchange_timestamp') else None,
                    "exchange_order_id": t.exchange_order_id if hasattr(t, 'exchange_order_id') else None,
                })

            items.sort(key=lambda x: x.get("fill_timestamp", ""), reverse=True)

            return {
                "trades": items,
                "summary": {
                    "total_trades": len(items),
                    "total_turnover": round(total_turnover, 2),
                    "buy_value": round(buy_value, 2),
                    "sell_value": round(sell_value, 2),
                    "net_value": round(buy_value - sell_value, 2),
                    "buy_trades": sum(1 for i in items if i["transaction_type"] == "BUY"),
                    "sell_trades": sum(1 for i in items if i["transaction_type"] == "SELL"),
                },
            }
        except Exception as e:
            logger.error("trades_report_failed", error=str(e))
            return {"error": str(e)}

    async def get_orders_report(self) -> dict[str, Any]:
        """Comprehensive orders report with status breakdown."""
        try:
            orders = await self._client.get_orders()
            items = []
            status_counts: dict[str, int] = {}

            for o in orders:
                status = o.status or "UNKNOWN"
                status_counts[status] = status_counts.get(status, 0) + 1

                items.append({
                    "order_id": o.order_id,
                    "tradingsymbol": o.tradingsymbol,
                    "exchange": o.exchange,
                    "transaction_type": o.transaction_type,
                    "order_type": o.order_type,
                    "quantity": o.quantity,
                    "price": o.price,
                    "trigger_price": o.trigger_price,
                    "average_price": o.average_price,
                    "filled_quantity": o.filled_quantity,
                    "pending_quantity": o.pending_quantity,
                    "status": status,
                    "status_message": o.status_message if hasattr(o, 'status_message') else None,
                    "variety": o.variety,
                    "product": o.product,
                    "validity": o.validity,
                    "placed_by": o.placed_by if hasattr(o, 'placed_by') else None,
                    "order_timestamp": str(o.order_timestamp) if hasattr(o, 'order_timestamp') and o.order_timestamp else None,
                    "exchange_timestamp": str(o.exchange_timestamp) if hasattr(o, 'exchange_timestamp') and o.exchange_timestamp else None,
                    "tag": o.tag if hasattr(o, 'tag') else None,
                })

            return {
                "orders": items,
                "summary": {
                    "total_orders": len(items),
                    "status_breakdown": status_counts,
                    "filled": status_counts.get("COMPLETE", 0),
                    "pending": status_counts.get("OPEN", 0) + status_counts.get("TRIGGER PENDING", 0),
                    "cancelled": status_counts.get("CANCELLED", 0),
                    "rejected": status_counts.get("REJECTED", 0),
                },
            }
        except Exception as e:
            logger.error("orders_report_failed", error=str(e))
            return {"error": str(e)}

    async def get_pnl_report(self) -> dict[str, Any]:
        """Combined P&L report from positions and holdings."""
        try:
            positions_report, holdings_report, trades_report = await asyncio.gather(
                self.get_positions_report(),
                self.get_holdings_report(),
                self.get_trades_report(),
            )

            pos_pnl = positions_report.get("summary", {}).get("total_pnl", 0)
            hold_pnl = holdings_report.get("summary", {}).get("total_pnl", 0)

            return {
                "day_pnl": round(pos_pnl, 2),
                "investment_pnl": round(hold_pnl, 2),
                "combined_pnl": round(pos_pnl + hold_pnl, 2),
                "positions_summary": positions_report.get("summary", {}),
                "holdings_summary": holdings_report.get("summary", {}),
                "trades_summary": trades_report.get("summary", {}),
                "report_time": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error("pnl_report_failed", error=str(e))
            return {"error": str(e)}

    async def get_margins_report(self) -> dict[str, Any]:
        """Detailed margin report."""
        try:
            margins = await self._client.get_margins()
            result: dict[str, Any] = {}

            for segment_name, segment in [("equity", margins.equity), ("commodity", margins.commodity)]:
                if segment:
                    result[segment_name] = {
                        "available": {
                            "cash": round(segment.available.cash, 2) if segment.available else 0,
                            "collateral": round(segment.available.collateral, 2) if segment.available else 0,
                            "intraday_payin": round(segment.available.intraday_payin, 2) if segment.available else 0,
                            "live_balance": round(segment.available.live_balance, 2) if segment.available else 0,
                        },
                        "utilised": {
                            "debits": round(segment.utilised.debits, 2) if segment.utilised else 0,
                            "exposure": round(segment.utilised.exposure, 2) if segment.utilised else 0,
                            "span": round(segment.utilised.span, 2) if segment.utilised else 0,
                            "option_premium": round(segment.utilised.option_premium, 2) if segment.utilised else 0,
                            "holding_sales": round(segment.utilised.holding_sales, 2) if segment.utilised else 0,
                            "turnover": round(segment.utilised.turnover, 2) if segment.utilised else 0,
                        },
                        "net": round(segment.net, 2) if hasattr(segment, 'net') else 0,
                    }

            return {"margins": result}
        except Exception as e:
            logger.error("margins_report_failed", error=str(e))
            return {"error": str(e)}
