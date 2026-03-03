"""
OI Snapshot Scheduler — Periodic 15-minute option chain snapshots
==================================================================

Captures NIFTY and SENSEX option chain data every 15 minutes during
market hours (9:15 AM – 3:30 PM IST).  Stores snapshots in the
FnODataStore (SQLite) for intraday analysis.

Usage:
    scheduler = OISnapshotScheduler(kite_client)
    await scheduler.start()   # runs as background task
    await scheduler.stop()
    await scheduler.take_snapshot()  # manual trigger
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, date, time, timedelta, timezone
from typing import Any, Optional, TYPE_CHECKING

from src.data.fno_data_fetcher import FnODataFetcher, get_fno_data_fetcher, INDEX_TOKENS
from src.data.fno_data_store import FnODataStore, get_fno_data_store

if TYPE_CHECKING:
    from src.api.client import KiteClient

logger = logging.getLogger("oi_snapshot_scheduler")

# IST = UTC+5:30
IST = timezone(timedelta(hours=5, minutes=30))

# Market hours (IST)
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

# Underlyings to capture
SNAPSHOT_UNDERLYINGS = ["NIFTY", "SENSEX"]

# Capture interval (minutes)
SNAPSHOT_INTERVAL_MINUTES = 15


class OISnapshotScheduler:
    """
    Background scheduler that takes option chain snapshots every 15 minutes
    for NIFTY and SENSEX during market hours.
    """

    def __init__(
        self,
        kite_client: Optional["KiteClient"] = None,
        fetcher: Optional[FnODataFetcher] = None,
        store: Optional[FnODataStore] = None,
        interval_minutes: int = SNAPSHOT_INTERVAL_MINUTES,
    ) -> None:
        self._kite = kite_client
        self._fetcher = fetcher or get_fno_data_fetcher()
        self._store = store or get_fno_data_store()
        self._interval = interval_minutes * 60  # seconds
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._last_snapshot_time: Optional[datetime] = None
        self._snapshot_count = 0
        self._errors: list[dict[str, Any]] = []

    # ─── Lifecycle ──────────────────────────────────────────────

    def set_kite_client(self, client: "KiteClient") -> None:
        """Set Kite client (needed before start)."""
        self._kite = client
        self._fetcher.set_kite_client(client)

    async def start(self) -> dict[str, Any]:
        """Start the background snapshot scheduler."""
        if self._running:
            return {"status": "already_running", "interval_minutes": self._interval // 60}

        if not self._kite:
            return {"status": "error", "message": "Kite client not set"}

        self._fetcher.set_kite_client(self._kite)
        self._running = True
        self._task = asyncio.create_task(self._scheduler_loop())
        logger.info("OI snapshot scheduler started (interval=%dm)", self._interval // 60)
        return {
            "status": "started",
            "interval_minutes": self._interval // 60,
            "underlyings": SNAPSHOT_UNDERLYINGS,
        }

    async def stop(self) -> dict[str, Any]:
        """Stop the background scheduler."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("OI snapshot scheduler stopped (snapshots=%d)", self._snapshot_count)
        return {
            "status": "stopped",
            "total_snapshots": self._snapshot_count,
        }

    @property
    def is_running(self) -> bool:
        return self._running

    def get_status(self) -> dict[str, Any]:
        """Get scheduler status."""
        now_ist = datetime.now(IST)
        in_market_hours = self._is_market_hours(now_ist)
        return {
            "running": self._running,
            "interval_minutes": self._interval // 60,
            "underlyings": SNAPSHOT_UNDERLYINGS,
            "total_snapshots": self._snapshot_count,
            "last_snapshot": self._last_snapshot_time.isoformat() if self._last_snapshot_time else None,
            "in_market_hours": in_market_hours,
            "current_time_ist": now_ist.strftime("%H:%M:%S"),
            "recent_errors": self._errors[-5:],
        }

    # ─── Manual Trigger ─────────────────────────────────────────

    async def take_snapshot(self, force: bool = False) -> dict[str, Any]:
        """
        Take a snapshot now (manual trigger).
        If force=False, only runs during market hours.
        """
        now_ist = datetime.now(IST)
        if not force and not self._is_market_hours(now_ist):
            return {
                "status": "skipped",
                "reason": "Outside market hours",
                "current_time_ist": now_ist.strftime("%H:%M:%S"),
            }

        if not self._kite or not self._fetcher.is_available:
            return {"status": "error", "message": "Kite client not available"}

        results = await self._capture_all_underlyings()
        return {
            "status": "completed",
            "timestamp": now_ist.isoformat(),
            "results": results,
        }

    # ─── Scheduler Loop ─────────────────────────────────────────

    async def _scheduler_loop(self) -> None:
        """Main loop: sleep until next 15-min boundary, then snapshot."""
        try:
            while self._running:
                now_ist = datetime.now(IST)

                if self._is_market_hours(now_ist):
                    # Take snapshot
                    try:
                        results = await self._capture_all_underlyings()
                        logger.info(
                            "Snapshot #%d completed: %s",
                            self._snapshot_count, results,
                        )
                    except Exception as e:
                        logger.error("Snapshot error: %s", e)
                        self._errors.append({
                            "time": now_ist.isoformat(),
                            "error": str(e),
                        })
                        if len(self._errors) > 50:
                            self._errors = self._errors[-50:]

                # Sleep until next interval
                sleep_secs = self._seconds_until_next_slot()
                logger.debug("Next snapshot in %.0f seconds", sleep_secs)
                await asyncio.sleep(sleep_secs)

        except asyncio.CancelledError:
            logger.info("Scheduler loop cancelled")
        except Exception as e:
            logger.error("Scheduler loop fatal error: %s", e)
            self._running = False

    # ─── Capture Logic ──────────────────────────────────────────

    async def _capture_all_underlyings(self) -> dict[str, Any]:
        """Capture snapshots for all configured underlyings."""
        results: dict[str, Any] = {}
        ts = datetime.now(IST)

        for underlying in SNAPSHOT_UNDERLYINGS:
            try:
                count = await self._capture_one(underlying)
                results[underlying] = {"rows": count, "status": "ok"}
            except Exception as e:
                logger.error("Snapshot error for %s: %s", underlying, e)
                results[underlying] = {"rows": 0, "status": "error", "error": str(e)}

        self._last_snapshot_time = ts
        self._snapshot_count += 1
        return results

    async def _capture_one(self, underlying: str) -> int:
        """Capture option chain snapshot for one underlying."""
        ul = underlying.upper()

        # Get spot price
        spot = await self._get_spot(ul)
        if spot <= 0:
            logger.warning("Could not get spot price for %s", ul)
            return 0

        # Get nearest expiry
        expiries = self._fetcher.get_fno_expiries(ul)
        if not expiries:
            logger.warning("No expiries found for %s", ul)
            return 0

        # Use nearest weekly/monthly expiry
        nearest_expiry = expiries[0]

        # Fetch full quotes with OI
        count = await self._fetcher.fetch_option_chain_with_quotes(
            underlying=ul,
            expiry=nearest_expiry,
            spot_price=spot,
        )

        logger.info(
            "Captured %d option rows for %s (exp=%s, spot=%.2f)",
            count, ul, nearest_expiry, spot,
        )
        return count

    async def _get_spot(self, underlying: str) -> float:
        """Fetch current spot price for an underlying."""
        try:
            idx_map = {
                "NIFTY": "NSE:NIFTY 50",
                "NIFTY 50": "NSE:NIFTY 50",
                "SENSEX": "BSE:SENSEX",
                "BANKNIFTY": "NSE:NIFTY BANK",
            }
            key = idx_map.get(underlying, f"NSE:{underlying}")
            ltp_data = await self._kite.get_ltp([key])
            for k, v in ltp_data.items():
                price = v.last_price if hasattr(v, "last_price") else v.get("last_price", 0)
                if price > 0:
                    return price
        except Exception as e:
            logger.error("Spot price error for %s: %s", underlying, e)
        return 0.0

    # ─── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _is_market_hours(now: datetime) -> bool:
        """Check if current IST time is within market hours."""
        if now.weekday() >= 5:  # Saturday/Sunday
            return False
        t = now.time()
        return MARKET_OPEN <= t <= MARKET_CLOSE

    def _seconds_until_next_slot(self) -> float:
        """Calculate seconds until next 15-min snapshot slot."""
        now = datetime.now(IST)
        minute = now.minute
        interval = self._interval // 60

        # Next slot minute
        next_slot_minute = ((minute // interval) + 1) * interval
        if next_slot_minute >= 60:
            next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        else:
            next_time = now.replace(minute=next_slot_minute, second=0, microsecond=0)

        delta = (next_time - now).total_seconds()
        return max(delta, 5)  # At least 5 seconds

    # ─── Today's snapshots query ────────────────────────────────

    def get_todays_snapshots(self, underlying: str = "NIFTY") -> list[str]:
        """Get all snapshot timestamps for today."""
        today = date.today().isoformat()
        return self._store.get_distinct_timestamps(underlying, today, today)

    def get_snapshot_data(self, underlying: str, ts: str) -> list[dict[str, Any]]:
        """Get option chain data at a specific timestamp."""
        return self._store.get_option_chain_at_timestamp(underlying, ts)

    def get_todays_all_data(self, underlying: str = "NIFTY") -> list[dict[str, Any]]:
        """Get all today's option chain data for an underlying."""
        today = date.today().isoformat()
        return self._store.get_option_chain_snapshots(underlying, today, today)


# ─── Singleton ──────────────────────────────────────────────────

_scheduler_instance: Optional[OISnapshotScheduler] = None


def get_oi_snapshot_scheduler() -> OISnapshotScheduler:
    """Get or create singleton scheduler instance."""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = OISnapshotScheduler()
    return _scheduler_instance
