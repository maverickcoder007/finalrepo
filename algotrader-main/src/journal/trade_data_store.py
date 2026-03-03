"""
Trade Data Store — Instrument OHLCV Capture + Daily Trade Analysis
===================================================================

Two capabilities:

1. **Per-trade data capture**
   When a trade opens, record the instrument, exchange, entry time, and
   instrument_token.  When it closes, fetch the full OHLCV history from
   entry-to-exit and persist it.  This gives exhaustive post-trade replay
   data — the agent or user can chart every bar the position was live.

2. **Daily trade analysis**
   Aggregates all trades executed on a given day:
     • instrument-wise P&L, win rate, avg duration
     • strategy-wise breakdown
     • best / worst trades
     • intraday P&L curve

All data is stored in ``data/journal.db`` alongside the existing
JournalStore tables.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from typing import Any, Optional

logger = logging.getLogger("trade_data_store")

_DB_PATH = os.path.join("data", "journal.db")

_TRADE_DATA_SCHEMA = """
-- Open/pending trade registrations (entry recorded immediately, exit filled later)
CREATE TABLE IF NOT EXISTS trade_data_registry (
    trade_id          TEXT    PRIMARY KEY,
    order_id          TEXT    NOT NULL DEFAULT '',
    instrument        TEXT    NOT NULL,
    tradingsymbol     TEXT    NOT NULL,
    exchange          TEXT    NOT NULL DEFAULT 'NSE',
    instrument_token  INTEGER NOT NULL DEFAULT 0,
    strategy_name     TEXT    NOT NULL DEFAULT '',
    direction         TEXT    NOT NULL DEFAULT '',
    quantity          INTEGER NOT NULL DEFAULT 0,
    entry_price       REAL    NOT NULL DEFAULT 0,
    exit_price        REAL    NOT NULL DEFAULT 0,
    entry_time        TEXT    NOT NULL,
    exit_time         TEXT    NOT NULL DEFAULT '',
    pnl               REAL    NOT NULL DEFAULT 0,
    is_closed         INTEGER NOT NULL DEFAULT 0,
    interval          TEXT    NOT NULL DEFAULT '5minute',
    candle_count      INTEGER NOT NULL DEFAULT 0,
    metadata          TEXT    NOT NULL DEFAULT '{}',
    created_at        TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_tdr_instrument ON trade_data_registry(instrument);
CREATE INDEX IF NOT EXISTS idx_tdr_entry_time ON trade_data_registry(entry_time);
CREATE INDEX IF NOT EXISTS idx_tdr_closed     ON trade_data_registry(is_closed);
CREATE INDEX IF NOT EXISTS idx_tdr_strategy   ON trade_data_registry(strategy_name);

-- Per-trade OHLCV bars (entry→exit window)
CREATE TABLE IF NOT EXISTS trade_data_candles (
    trade_id  TEXT    NOT NULL,
    ts        TEXT    NOT NULL,
    open      REAL    NOT NULL,
    high      REAL    NOT NULL,
    low       REAL    NOT NULL,
    close     REAL    NOT NULL,
    volume    INTEGER NOT NULL DEFAULT 0,
    oi        INTEGER,
    PRIMARY KEY (trade_id, ts)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_tdc_trade ON trade_data_candles(trade_id);

-- Daily trade analysis cache
CREATE TABLE IF NOT EXISTS daily_trade_analysis (
    analysis_date  TEXT    NOT NULL,
    analysis_type  TEXT    NOT NULL DEFAULT 'full',   -- 'full', 'instrument', 'strategy'
    generated_at   TEXT    NOT NULL,
    data           TEXT    NOT NULL,   -- JSON blob
    PRIMARY KEY (analysis_date, analysis_type)
);
"""


@dataclass
class TradeRegistration:
    """A registered trade awaiting or completed with OHLCV data."""
    trade_id: str = ""
    order_id: str = ""
    instrument: str = ""
    tradingsymbol: str = ""
    exchange: str = "NSE"
    instrument_token: int = 0
    strategy_name: str = ""
    direction: str = ""
    quantity: int = 0
    entry_price: float = 0.0
    exit_price: float = 0.0
    entry_time: str = ""
    exit_time: str = ""
    pnl: float = 0.0
    is_closed: bool = False
    interval: str = "5minute"
    candle_count: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("entry_price", "exit_price", "pnl"):
            d[k] = round(d[k], 2)
        return d


@dataclass
class DailyAnalysis:
    """Aggregated daily trade analysis."""
    analysis_date: str = ""
    total_trades: int = 0
    closed_trades: int = 0
    open_trades: int = 0
    total_pnl: float = 0.0
    winners: int = 0
    losers: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_trade_pnl: float = 0.0
    best_trade_symbol: str = ""
    worst_trade_pnl: float = 0.0
    worst_trade_symbol: str = ""
    avg_duration_minutes: float = 0.0
    by_instrument: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_strategy: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_tradingsymbol: dict[str, dict[str, Any]] = field(default_factory=dict)
    pnl_curve: list[dict[str, Any]] = field(default_factory=list)
    trades: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("total_pnl", "avg_win", "avg_loss", "best_trade_pnl",
                   "worst_trade_pnl", "win_rate", "avg_duration_minutes"):
            if isinstance(d.get(k), float):
                d[k] = round(d[k], 2)
        return d


class TradeDataStore:
    """
    SQLite-backed store for per-trade OHLCV capture and daily analysis.
    """

    def __init__(self, db_path: str = _DB_PATH) -> None:
        self._db_path = db_path
        self._local = threading.local()
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
            self._local.conn = sqlite3.connect(self._db_path)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
        return self._local.conn

    def _init_schema(self) -> None:
        conn = self._get_conn()
        conn.executescript(_TRADE_DATA_SCHEMA)
        conn.commit()

    # ──────────────────────────────────────────────────────────
    # Trade Registration (on fill)
    # ──────────────────────────────────────────────────────────

    def register_trade_open(
        self,
        trade_id: str,
        order_id: str,
        instrument: str,
        tradingsymbol: str,
        exchange: str,
        instrument_token: int,
        strategy_name: str,
        direction: str,
        quantity: int,
        entry_price: float,
        entry_time: str = "",
        interval: str = "5minute",
        metadata: dict | None = None,
    ) -> TradeRegistration:
        """Register a new trade when it opens (on fill callback)."""
        if not trade_id:
            trade_id = str(uuid.uuid4())[:16]
        if not entry_time:
            entry_time = datetime.now().isoformat()
        meta_json = json.dumps(metadata or {})

        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO trade_data_registry
               (trade_id, order_id, instrument, tradingsymbol, exchange,
                instrument_token, strategy_name, direction, quantity,
                entry_price, entry_time, interval, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (trade_id, order_id, instrument, tradingsymbol, exchange,
             instrument_token, strategy_name, direction, quantity,
             entry_price, entry_time, interval, meta_json),
        )
        conn.commit()

        logger.info("trade_registered", trade_id=trade_id, symbol=tradingsymbol)
        return TradeRegistration(
            trade_id=trade_id, order_id=order_id, instrument=instrument,
            tradingsymbol=tradingsymbol, exchange=exchange,
            instrument_token=instrument_token, strategy_name=strategy_name,
            direction=direction, quantity=quantity, entry_price=entry_price,
            entry_time=entry_time, interval=interval,
            metadata=metadata or {},
        )

    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        exit_time: str = "",
        pnl: float = 0.0,
    ) -> bool:
        """Mark a trade as closed when it exits."""
        if not exit_time:
            exit_time = datetime.now().isoformat()
        conn = self._get_conn()
        cur = conn.execute(
            """UPDATE trade_data_registry
               SET exit_price = ?, exit_time = ?, pnl = ?, is_closed = 1
               WHERE trade_id = ?""",
            (exit_price, exit_time, pnl, trade_id),
        )
        conn.commit()
        if cur.rowcount > 0:
            logger.info("trade_closed", trade_id=trade_id, pnl=round(pnl, 2))
            return True
        return False

    # ──────────────────────────────────────────────────────────
    # OHLCV Data Storage (fetched after trade closes)
    # ──────────────────────────────────────────────────────────

    def store_trade_candles(
        self, trade_id: str, candles: list[dict[str, Any]],
    ) -> int:
        """Store OHLCV bars for a trade's duration."""
        if not candles:
            return 0
        conn = self._get_conn()
        inserted = 0
        for c in candles:
            try:
                conn.execute(
                    """INSERT OR IGNORE INTO trade_data_candles
                       (trade_id, ts, open, high, low, close, volume, oi)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (trade_id, c["ts"], c["open"], c["high"], c["low"],
                     c["close"], c["volume"], c.get("oi")),
                )
                inserted += 1
            except Exception:
                pass
        # Update candle count
        conn.execute(
            "UPDATE trade_data_registry SET candle_count = ? WHERE trade_id = ?",
            (inserted, trade_id),
        )
        conn.commit()
        logger.info("trade_candles_stored", trade_id=trade_id, bars=inserted)
        return inserted

    def get_trade_candles(self, trade_id: str) -> list[dict[str, Any]]:
        """Retrieve OHLCV bars for a specific trade."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM trade_data_candles WHERE trade_id = ? ORDER BY ts",
            (trade_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ──────────────────────────────────────────────────────────
    # Trade Queries
    # ──────────────────────────────────────────────────────────

    def get_trade(self, trade_id: str) -> Optional[TradeRegistration]:
        """Get a single trade by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM trade_data_registry WHERE trade_id = ?", (trade_id,),
        ).fetchone()
        return self._row_to_reg(row) if row else None

    def get_open_trades(self) -> list[TradeRegistration]:
        """Get all currently open trades."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM trade_data_registry WHERE is_closed = 0 ORDER BY entry_time DESC",
        ).fetchall()
        return [self._row_to_reg(r) for r in rows]

    def get_trades_for_date(self, trade_date: str = "") -> list[TradeRegistration]:
        """Get all trades for a specific date (by entry_time)."""
        if not trade_date:
            trade_date = date.today().isoformat()
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM trade_data_registry WHERE entry_time LIKE ? ORDER BY entry_time",
            (f"{trade_date}%",),
        ).fetchall()
        return [self._row_to_reg(r) for r in rows]

    def get_trades_by_instrument(
        self, instrument: str, limit: int = 50,
    ) -> list[TradeRegistration]:
        """Get trades for a specific instrument."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT * FROM trade_data_registry
               WHERE instrument = ? OR tradingsymbol = ?
               ORDER BY entry_time DESC LIMIT ?""",
            (instrument, instrument, limit),
        ).fetchall()
        return [self._row_to_reg(r) for r in rows]

    def get_recent_trades(self, limit: int = 50) -> list[TradeRegistration]:
        """Get recent trades."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM trade_data_registry ORDER BY entry_time DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_reg(r) for r in rows]

    def get_trades_pending_candles(self) -> list[TradeRegistration]:
        """Get closed trades that don't have candle data yet."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT * FROM trade_data_registry
               WHERE is_closed = 1 AND candle_count = 0
               ORDER BY exit_time DESC LIMIT 100""",
        ).fetchall()
        return [self._row_to_reg(r) for r in rows]

    def find_trade_by_order_id(self, order_id: str) -> Optional[TradeRegistration]:
        """Find a trade by its broker order_id."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM trade_data_registry WHERE order_id = ? LIMIT 1",
            (order_id,),
        ).fetchone()
        return self._row_to_reg(row) if row else None

    # ──────────────────────────────────────────────────────────
    # Daily Analysis
    # ──────────────────────────────────────────────────────────

    def compute_daily_analysis(self, analysis_date: str = "") -> DailyAnalysis:
        """
        Compute full daily analysis for all trades on a given date.

        Produces:
          • Summary stats (total trades, P&L, win rate, avg duration)
          • Instrument-wise breakdown (per underlying / tradingsymbol)
          • Strategy-wise breakdown
          • Intraday P&L curve
        """
        if not analysis_date:
            analysis_date = date.today().isoformat()

        trades = self.get_trades_for_date(analysis_date)
        analysis = DailyAnalysis(analysis_date=analysis_date)
        analysis.total_trades = len(trades)

        if not trades:
            return analysis

        closed = [t for t in trades if t.is_closed]
        analysis.closed_trades = len(closed)
        analysis.open_trades = analysis.total_trades - analysis.closed_trades

        # ── P&L stats ────────────────────────────────────────
        pnls = [t.pnl for t in closed]
        analysis.total_pnl = sum(pnls)
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p < 0]
        analysis.winners = len(winners)
        analysis.losers = len(losers)
        analysis.win_rate = (len(winners) / len(closed) * 100) if closed else 0.0
        analysis.avg_win = (sum(winners) / len(winners)) if winners else 0.0
        analysis.avg_loss = (sum(losers) / len(losers)) if losers else 0.0

        # Best / worst
        if closed:
            best = max(closed, key=lambda t: t.pnl)
            worst = min(closed, key=lambda t: t.pnl)
            analysis.best_trade_pnl = best.pnl
            analysis.best_trade_symbol = best.tradingsymbol
            analysis.worst_trade_pnl = worst.pnl
            analysis.worst_trade_symbol = worst.tradingsymbol

        # Average duration
        durations = []
        for t in closed:
            if t.entry_time and t.exit_time:
                try:
                    entry_dt = datetime.fromisoformat(t.entry_time)
                    exit_dt = datetime.fromisoformat(t.exit_time)
                    durations.append((exit_dt - entry_dt).total_seconds() / 60)
                except Exception:
                    pass
        analysis.avg_duration_minutes = (
            sum(durations) / len(durations) if durations else 0.0
        )

        # ── By instrument (underlying) ───────────────────────
        inst_map: dict[str, list[TradeRegistration]] = {}
        for t in trades:
            key = self._extract_underlying(t.tradingsymbol)
            inst_map.setdefault(key, []).append(t)

        for inst, inst_trades in inst_map.items():
            inst_closed = [t for t in inst_trades if t.is_closed]
            inst_pnls = [t.pnl for t in inst_closed]
            inst_winners = [p for p in inst_pnls if p > 0]
            analysis.by_instrument[inst] = {
                "total_trades": len(inst_trades),
                "closed_trades": len(inst_closed),
                "total_pnl": round(sum(inst_pnls), 2),
                "winners": len(inst_winners),
                "losers": len(inst_pnls) - len(inst_winners),
                "win_rate": round(len(inst_winners) / max(len(inst_closed), 1) * 100, 1),
                "symbols": list(set(t.tradingsymbol for t in inst_trades)),
            }

        # ── By tradingsymbol ─────────────────────────────────
        sym_map: dict[str, list[TradeRegistration]] = {}
        for t in trades:
            sym_map.setdefault(t.tradingsymbol, []).append(t)

        for sym, sym_trades in sym_map.items():
            sym_closed = [t for t in sym_trades if t.is_closed]
            sym_pnls = [t.pnl for t in sym_closed]
            sym_winners = [p for p in sym_pnls if p > 0]
            analysis.by_tradingsymbol[sym] = {
                "total_trades": len(sym_trades),
                "closed_trades": len(sym_closed),
                "total_pnl": round(sum(sym_pnls), 2),
                "winners": len(sym_winners),
                "losers": len(sym_pnls) - len(sym_winners),
                "win_rate": round(len(sym_winners) / max(len(sym_closed), 1) * 100, 1),
                "avg_pnl": round(sum(sym_pnls) / max(len(sym_closed), 1), 2),
                "directions": list(set(t.direction for t in sym_trades if t.direction)),
            }

        # ── By strategy ──────────────────────────────────────
        strat_map: dict[str, list[TradeRegistration]] = {}
        for t in trades:
            strat_map.setdefault(t.strategy_name or "unknown", []).append(t)

        for strat, strat_trades in strat_map.items():
            strat_closed = [t for t in strat_trades if t.is_closed]
            strat_pnls = [t.pnl for t in strat_closed]
            strat_winners = [p for p in strat_pnls if p > 0]
            analysis.by_strategy[strat] = {
                "total_trades": len(strat_trades),
                "closed_trades": len(strat_closed),
                "total_pnl": round(sum(strat_pnls), 2),
                "winners": len(strat_winners),
                "losers": len(strat_pnls) - len(strat_winners),
                "win_rate": round(len(strat_winners) / max(len(strat_closed), 1) * 100, 1),
                "instruments": list(set(t.tradingsymbol for t in strat_trades)),
            }

        # ── Intraday P&L curve (cumulative) ──────────────────
        closed_sorted = sorted(closed, key=lambda t: t.exit_time or t.entry_time)
        cumulative = 0.0
        for t in closed_sorted:
            cumulative += t.pnl
            analysis.pnl_curve.append({
                "time": t.exit_time or t.entry_time,
                "symbol": t.tradingsymbol,
                "trade_pnl": round(t.pnl, 2),
                "cumulative_pnl": round(cumulative, 2),
            })

        # ── Trade list (compact) ─────────────────────────────
        for t in trades:
            analysis.trades.append({
                "trade_id": t.trade_id,
                "symbol": t.tradingsymbol,
                "strategy": t.strategy_name,
                "direction": t.direction,
                "quantity": t.quantity,
                "entry_price": round(t.entry_price, 2),
                "exit_price": round(t.exit_price, 2),
                "pnl": round(t.pnl, 2),
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "is_closed": t.is_closed,
                "candle_count": t.candle_count,
            })

        # Cache the analysis
        self._cache_analysis(analysis_date, "full", analysis.to_dict())

        return analysis

    def get_daily_analysis(self, analysis_date: str = "") -> Optional[dict[str, Any]]:
        """Get cached daily analysis, or compute fresh."""
        if not analysis_date:
            analysis_date = date.today().isoformat()
        cached = self._get_cached_analysis(analysis_date, "full")
        if cached:
            return cached
        return self.compute_daily_analysis(analysis_date).to_dict()

    def get_daily_analysis_range(
        self, from_date: str = "", to_date: str = "", days: int = 7,
    ) -> list[dict[str, Any]]:
        """Get daily analyses for a date range."""
        if not to_date:
            to_date = date.today().isoformat()
        if not from_date:
            from_date = (date.today() - timedelta(days=days)).isoformat()

        results = []
        current = datetime.strptime(from_date, "%Y-%m-%d").date()
        end = datetime.strptime(to_date, "%Y-%m-%d").date()

        while current <= end:
            day_str = current.isoformat()
            analysis = self.get_daily_analysis(day_str)
            if analysis and analysis.get("total_trades", 0) > 0:
                results.append(analysis)
            current += timedelta(days=1)

        return results

    # ──────────────────────────────────────────────────────────
    # Trade Replay Data
    # ──────────────────────────────────────────────────────────

    def get_trade_replay(self, trade_id: str) -> Optional[dict[str, Any]]:
        """
        Get full trade replay data: trade info + OHLCV bars.
        This is the primary post-analysis endpoint — shows every bar
        from entry to exit with trade overlay.
        """
        trade = self.get_trade(trade_id)
        if not trade:
            return None

        candles = self.get_trade_candles(trade_id)

        return {
            "trade": trade.to_dict(),
            "candles": candles,
            "candle_count": len(candles),
            "has_data": len(candles) > 0,
            "analysis": self._analyze_trade_candles(trade, candles) if candles else {},
        }

    def _analyze_trade_candles(
        self, trade: TradeRegistration, candles: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Analyze OHLCV data relative to the trade entry/exit."""
        if not candles:
            return {}

        closes = [c["close"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]

        entry = trade.entry_price
        is_long = trade.direction in ("BUY", "LONG", "long")

        # Max favorable / adverse excursion from candle data
        if is_long:
            mfe = max(highs) - entry if highs else 0
            mae = entry - min(lows) if lows else 0
        else:
            mfe = entry - min(lows) if lows else 0
            mae = max(highs) - entry if highs else 0

        # Bar-by-bar unrealized P&L
        bar_pnl = []
        for c in candles:
            if is_long:
                unrealized = (c["close"] - entry) * trade.quantity
            else:
                unrealized = (entry - c["close"]) * trade.quantity
            bar_pnl.append({
                "ts": c["ts"],
                "close": c["close"],
                "unrealized_pnl": round(unrealized, 2),
            })

        # Price range during trade
        price_high = max(highs) if highs else 0
        price_low = min(lows) if lows else 0
        price_range = price_high - price_low

        # Volatility (simple: std of returns)
        returns = []
        for i in range(1, len(closes)):
            if closes[i - 1] > 0:
                returns.append((closes[i] - closes[i - 1]) / closes[i - 1])
        volatility = (sum(r ** 2 for r in returns) / max(len(returns), 1)) ** 0.5 if returns else 0

        # Volume profile
        volumes = [c["volume"] for c in candles if c.get("volume", 0) > 0]
        avg_volume = sum(volumes) / max(len(volumes), 1) if volumes else 0

        return {
            "mfe": round(mfe, 2),
            "mae": round(mae, 2),
            "mfe_pct": round(mfe / entry * 100, 2) if entry else 0,
            "mae_pct": round(mae / entry * 100, 2) if entry else 0,
            "edge_ratio": round(mfe / mae, 2) if mae > 0 else float("inf"),
            "price_high": round(price_high, 2),
            "price_low": round(price_low, 2),
            "price_range": round(price_range, 2),
            "bar_count": len(candles),
            "volatility": round(volatility * 100, 4),
            "avg_volume": round(avg_volume),
            "bar_pnl": bar_pnl,
        }

    # ──────────────────────────────────────────────────────────
    # Stats
    # ──────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Overall store statistics."""
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) FROM trade_data_registry").fetchone()[0]
        closed = conn.execute(
            "SELECT COUNT(*) FROM trade_data_registry WHERE is_closed = 1"
        ).fetchone()[0]
        with_candles = conn.execute(
            "SELECT COUNT(*) FROM trade_data_registry WHERE candle_count > 0"
        ).fetchone()[0]
        pending = conn.execute(
            "SELECT COUNT(*) FROM trade_data_registry WHERE is_closed = 1 AND candle_count = 0"
        ).fetchone()[0]
        total_candles = conn.execute(
            "SELECT COUNT(*) FROM trade_data_candles"
        ).fetchone()[0]
        return {
            "total_trades": total,
            "closed_trades": closed,
            "open_trades": total - closed,
            "trades_with_candles": with_candles,
            "pending_candle_fetch": pending,
            "total_candle_bars": total_candles,
        }

    # ──────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_reg(row: sqlite3.Row) -> TradeRegistration:
        d = dict(row)
        meta = d.get("metadata", "{}")
        try:
            meta = json.loads(meta) if isinstance(meta, str) else meta
        except Exception:
            meta = {}
        return TradeRegistration(
            trade_id=d.get("trade_id", ""),
            order_id=d.get("order_id", ""),
            instrument=d.get("instrument", ""),
            tradingsymbol=d.get("tradingsymbol", ""),
            exchange=d.get("exchange", "NSE"),
            instrument_token=d.get("instrument_token", 0),
            strategy_name=d.get("strategy_name", ""),
            direction=d.get("direction", ""),
            quantity=d.get("quantity", 0),
            entry_price=d.get("entry_price", 0.0),
            exit_price=d.get("exit_price", 0.0),
            entry_time=d.get("entry_time", ""),
            exit_time=d.get("exit_time", ""),
            pnl=d.get("pnl", 0.0),
            is_closed=bool(d.get("is_closed", 0)),
            interval=d.get("interval", "5minute"),
            candle_count=d.get("candle_count", 0),
            metadata=meta,
        )

    @staticmethod
    def _extract_underlying(tradingsymbol: str) -> str:
        """Extract underlying name from tradingsymbol."""
        known = ["BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX", "NIFTY"]
        upper = tradingsymbol.upper()
        for k in known:
            if upper.startswith(k):
                return k
        return tradingsymbol.split()[0] if tradingsymbol else "UNKNOWN"

    def _cache_analysis(self, analysis_date: str, analysis_type: str, data: dict) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO daily_trade_analysis
               (analysis_date, analysis_type, generated_at, data)
               VALUES (?, ?, ?, ?)""",
            (analysis_date, analysis_type, datetime.now().isoformat(), json.dumps(data)),
        )
        conn.commit()

    def _get_cached_analysis(self, analysis_date: str, analysis_type: str) -> Optional[dict]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT data FROM daily_trade_analysis WHERE analysis_date = ? AND analysis_type = ?",
            (analysis_date, analysis_type),
        ).fetchone()
        if row:
            try:
                return json.loads(row["data"])
            except Exception:
                pass
        return None
