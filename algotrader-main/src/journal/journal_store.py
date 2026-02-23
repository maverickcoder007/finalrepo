"""
Journal Storage Engine — SQLite-backed event-sourced storage
=============================================================

Replaces the old JSON-rewrite-on-every-trade approach with proper SQLite.
Event-sourced: every mutation is an event; current state is derived.

Tables:
  journal_entries  — Full JournalEntry records (one per round-trip trade)
  system_events    — Operational events (ws disconnect, crash, reconciliation)
  portfolio_snapshots — Periodic + trade-triggered portfolio state

Indexes:
  By strategy, instrument, date, is_closed, source, tags
"""

from __future__ import annotations
import json
import os
import sqlite3
import threading
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple

from src.journal.journal_models import (
    JournalEntry, SystemEvent, PortfolioSnapshot,
    ExecutionRecord, StrategyContext, SignalQuality,
    PositionStructure, CostBreakdown, ExcursionMetrics,
    CapitalMetrics, RiskMetrics, LiquiditySnapshot,
)

logger = logging.getLogger("journal_store")


class JournalStore:
    """
    Production SQLite journal store.
    Thread-safe, append-optimized, with rich querying.
    """

    def __init__(self, db_path: str = "data/journal.db"):
        self._db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._local = threading.local()
        self._init_db()
        logger.info("JournalStore initialized: %s", db_path)

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path, timeout=10)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
        return self._local.conn

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS journal_entries (
                entry_id        TEXT PRIMARY KEY,
                trade_id        TEXT,
                group_id        TEXT DEFAULT '',
                strategy_name   TEXT DEFAULT '',
                instrument      TEXT DEFAULT '',
                tradingsymbol   TEXT DEFAULT '',
                exchange        TEXT DEFAULT '',
                direction       TEXT DEFAULT '',
                trade_type      TEXT DEFAULT 'equity',
                is_closed       INTEGER DEFAULT 0,
                source          TEXT DEFAULT 'live',
                entry_time      TEXT DEFAULT '',
                exit_time       TEXT DEFAULT '',
                created_at      TEXT DEFAULT '',
                updated_at      TEXT DEFAULT '',
                gross_pnl       REAL DEFAULT 0,
                net_pnl         REAL DEFAULT 0,
                total_costs     REAL DEFAULT 0,
                return_pct      REAL DEFAULT 0,
                return_on_margin REAL DEFAULT 0,
                mae             REAL DEFAULT 0,
                mfe             REAL DEFAULT 0,
                mae_pct         REAL DEFAULT 0,
                mfe_pct         REAL DEFAULT 0,
                edge_ratio      REAL DEFAULT 0,
                review_status   TEXT DEFAULT 'unreviewed',
                notes           TEXT DEFAULT '',
                tags            TEXT DEFAULT '[]',
                data            TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS system_events (
                event_id    TEXT PRIMARY KEY,
                timestamp   TEXT DEFAULT '',
                event_type  TEXT DEFAULT '',
                severity    TEXT DEFAULT 'info',
                description TEXT DEFAULT '',
                data        TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                timestamp   TEXT DEFAULT '',
                trigger     TEXT DEFAULT '',
                total_equity REAL DEFAULT 0,
                day_pnl     REAL DEFAULT 0,
                data        TEXT DEFAULT '{}'
            );

            -- Indexes for fast queries
            CREATE INDEX IF NOT EXISTS idx_je_strategy ON journal_entries(strategy_name);
            CREATE INDEX IF NOT EXISTS idx_je_instrument ON journal_entries(instrument);
            CREATE INDEX IF NOT EXISTS idx_je_entry_time ON journal_entries(entry_time);
            CREATE INDEX IF NOT EXISTS idx_je_exit_time ON journal_entries(exit_time);
            CREATE INDEX IF NOT EXISTS idx_je_is_closed ON journal_entries(is_closed);
            CREATE INDEX IF NOT EXISTS idx_je_source ON journal_entries(source);
            CREATE INDEX IF NOT EXISTS idx_je_trade_type ON journal_entries(trade_type);
            CREATE INDEX IF NOT EXISTS idx_je_direction ON journal_entries(direction);
            CREATE INDEX IF NOT EXISTS idx_se_type ON system_events(event_type);
            CREATE INDEX IF NOT EXISTS idx_se_timestamp ON system_events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_ps_timestamp ON portfolio_snapshots(timestamp);
            CREATE INDEX IF NOT EXISTS idx_ps_trigger ON portfolio_snapshots(trigger);
        """)
        conn.commit()

    # ─── JOURNAL ENTRIES ────────────────────────────────────────

    def record_entry(self, entry: JournalEntry) -> str:
        """Insert or update a journal entry."""
        conn = self._get_conn()
        d = entry.to_dict()
        # Store the full object as JSON in 'data', plus extract indexed columns
        data_json = json.dumps(d, default=str)
        tags_json = json.dumps(d.get("tags", []))

        conn.execute("""
            INSERT OR REPLACE INTO journal_entries
            (entry_id, trade_id, group_id, strategy_name, instrument, tradingsymbol,
             exchange, direction, trade_type, is_closed, source,
             entry_time, exit_time, created_at, updated_at,
             gross_pnl, net_pnl, total_costs, return_pct, return_on_margin,
             mae, mfe, mae_pct, mfe_pct, edge_ratio,
             review_status, notes, tags, data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            d["entry_id"], d["trade_id"], d["group_id"], d["strategy_name"],
            d["instrument"], d["tradingsymbol"], d["exchange"], d["direction"],
            d["trade_type"], 1 if d["is_closed"] else 0, d["source"],
            d["entry_time"], d["exit_time"], d["created_at"], d["updated_at"],
            d["gross_pnl"], d["net_pnl"], d["total_costs"],
            d["return_pct"], d["return_on_margin"],
            d["mae"], d["mfe"], d["mae_pct"], d["mfe_pct"], d["edge_ratio"],
            d["review_status"], d["notes"], tags_json, data_json,
        ))
        conn.commit()
        return entry.entry_id

    def get_entry(self, entry_id: str) -> Optional[JournalEntry]:
        """Get a single journal entry by ID."""
        conn = self._get_conn()
        row = conn.execute("SELECT data FROM journal_entries WHERE entry_id = ?",
                           (entry_id,)).fetchone()
        if row:
            return JournalEntry.from_dict(json.loads(row["data"]))
        return None

    def close_entry(self, entry_id: str, net_pnl: float, gross_pnl: float,
                    total_costs: float, exit_time: str = "",
                    exit_execution: dict = None, portfolio_at_exit: dict = None,
                    exit_context: dict = None, mae: float = 0, mfe: float = 0):
        """Close an open trade with outcome data."""
        entry = self.get_entry(entry_id)
        if not entry:
            logger.warning("Cannot close entry %s — not found", entry_id)
            return
        entry.close_trade(net_pnl, gross_pnl, total_costs, exit_time,
                          exit_execution, portfolio_at_exit, exit_context)
        entry.mae = mae
        entry.mfe = mfe
        if entry.execution:
            qty = entry.execution.get("quantity", 0)
            price = entry.execution.get("actual_fill_price", 0)
            basis = qty * price if qty and price else 1
            entry.mae_pct = (mae / basis * 100) if basis else 0
            entry.mfe_pct = (mfe / basis * 100) if basis else 0
        self.record_entry(entry)

    def query_entries(
        self,
        strategy: str = "",
        instrument: str = "",
        trade_type: str = "",
        source: str = "",
        is_closed: Optional[bool] = None,
        direction: str = "",
        review_status: str = "",
        from_date: str = "",
        to_date: str = "",
        min_pnl: Optional[float] = None,
        max_pnl: Optional[float] = None,
        tags: List[str] = None,
        limit: int = 500,
        offset: int = 0,
        order_by: str = "entry_time DESC",
    ) -> List[JournalEntry]:
        """Rich query with filtering, sorting, pagination."""
        conn = self._get_conn()
        conditions = []
        params = []

        if strategy:
            conditions.append("strategy_name = ?")
            params.append(strategy)
        if instrument:
            conditions.append("instrument = ?")
            params.append(instrument)
        if trade_type:
            conditions.append("trade_type = ?")
            params.append(trade_type)
        if source:
            conditions.append("source = ?")
            params.append(source)
        if is_closed is not None:
            conditions.append("is_closed = ?")
            params.append(1 if is_closed else 0)
        if direction:
            conditions.append("direction = ?")
            params.append(direction)
        if review_status:
            conditions.append("review_status = ?")
            params.append(review_status)
        if from_date:
            conditions.append("entry_time >= ?")
            params.append(from_date)
        if to_date:
            conditions.append("entry_time <= ?")
            params.append(to_date)
        if min_pnl is not None:
            conditions.append("net_pnl >= ?")
            params.append(min_pnl)
        if max_pnl is not None:
            conditions.append("net_pnl <= ?")
            params.append(max_pnl)

        where = " AND ".join(conditions) if conditions else "1=1"
        # Whitelist order_by columns
        allowed_order = {"entry_time DESC", "entry_time ASC", "net_pnl DESC",
                         "net_pnl ASC", "created_at DESC", "return_pct DESC",
                         "mae DESC", "mfe DESC", "total_costs DESC"}
        if order_by not in allowed_order:
            order_by = "entry_time DESC"

        sql = f"SELECT data FROM journal_entries WHERE {where} ORDER BY {order_by} LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = conn.execute(sql, params).fetchall()
        entries = []
        for row in rows:
            try:
                entries.append(JournalEntry.from_dict(json.loads(row["data"])))
            except Exception as e:
                logger.error("Failed to parse journal entry: %s", e)
        return entries

    def count_entries(self, strategy: str = "", instrument: str = "",
                      is_closed: Optional[bool] = None, source: str = "",
                      direction: str = "", trade_type: str = "",
                      review_status: str = "",
                      from_date: str = "", to_date: str = "") -> int:
        conn = self._get_conn()
        conditions, params = [], []
        if strategy:
            conditions.append("strategy_name = ?"); params.append(strategy)
        if instrument:
            conditions.append("instrument = ?"); params.append(instrument)
        if is_closed is not None:
            conditions.append("is_closed = ?"); params.append(1 if is_closed else 0)
        if source:
            conditions.append("source = ?"); params.append(source)
        if direction:
            conditions.append("direction = ?"); params.append(direction)
        if trade_type:
            conditions.append("trade_type = ?"); params.append(trade_type)
        if review_status:
            conditions.append("review_status = ?"); params.append(review_status)
        if from_date:
            conditions.append("entry_time >= ?"); params.append(from_date)
        if to_date:
            conditions.append("entry_time <= ?"); params.append(to_date)
        where = " AND ".join(conditions) if conditions else "1=1"
        row = conn.execute(f"SELECT COUNT(*) as cnt FROM journal_entries WHERE {where}", params).fetchone()
        return row["cnt"] if row else 0

    def get_strategies(self) -> List[str]:
        conn = self._get_conn()
        rows = conn.execute("SELECT DISTINCT strategy_name FROM journal_entries ORDER BY strategy_name").fetchall()
        return [r["strategy_name"] for r in rows if r["strategy_name"]]

    def get_instruments(self) -> List[str]:
        conn = self._get_conn()
        rows = conn.execute("SELECT DISTINCT instrument FROM journal_entries ORDER BY instrument").fetchall()
        return [r["instrument"] for r in rows if r["instrument"]]

    def get_pnl_by_date(self, days: int = 30, strategy: str = "",
                         instrument: str = "", source: str = "",
                         direction: str = "", trade_type: str = "") -> List[Dict]:
        """Daily P&L for the last N days, with optional filters."""
        conn = self._get_conn()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        conditions = ["is_closed = 1", "exit_time >= ?"]
        params: list = [cutoff]
        if strategy:
            conditions.append("strategy_name = ?"); params.append(strategy)
        if instrument:
            conditions.append("instrument = ?"); params.append(instrument)
        if source:
            conditions.append("source = ?"); params.append(source)
        if direction:
            conditions.append("direction = ?"); params.append(direction)
        if trade_type:
            conditions.append("trade_type = ?"); params.append(trade_type)
        where = " AND ".join(conditions)
        rows = conn.execute(f"""
            SELECT DATE(exit_time) as trade_date, 
                   SUM(net_pnl) as daily_pnl,
                   COUNT(*) as trade_count,
                   SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) as wins
            FROM journal_entries
            WHERE {where}
            GROUP BY DATE(exit_time)
            ORDER BY trade_date
        """, params).fetchall()
        return [{"date": r["trade_date"], "pnl": r["daily_pnl"],
                 "trades": r["trade_count"], "wins": r["wins"]} for r in rows]

    def get_pnl_by_strategy(self, source: str = "", days: int = 0,
                             instrument: str = "", direction: str = "",
                             trade_type: str = "", strategy: str = "") -> List[Dict]:
        conn = self._get_conn()
        conditions = ["is_closed = 1"]
        params: list = []
        if strategy:
            conditions.append("strategy_name = ?"); params.append(strategy)
        if source:
            conditions.append("source = ?"); params.append(source)
        if instrument:
            conditions.append("instrument = ?"); params.append(instrument)
        if direction:
            conditions.append("direction = ?"); params.append(direction)
        if trade_type:
            conditions.append("trade_type = ?"); params.append(trade_type)
        if days:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            conditions.append("exit_time >= ?"); params.append(cutoff)
        where = " AND ".join(conditions)
        rows = conn.execute(f"""
            SELECT strategy_name,
                   COUNT(*) as trades,
                   SUM(net_pnl) as total_pnl,
                   AVG(net_pnl) as avg_pnl,
                   SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN net_pnl > 0 THEN net_pnl ELSE 0 END) as total_profit,
                   SUM(CASE WHEN net_pnl < 0 THEN ABS(net_pnl) ELSE 0 END) as total_loss,
                   SUM(total_costs) as total_costs,
                   AVG(return_pct) as avg_return_pct
            FROM journal_entries
            WHERE {where}
            GROUP BY strategy_name
            ORDER BY total_pnl DESC
        """, params).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["strategy"] = d["strategy_name"]  # alias for JS
            d["win_rate"] = round(d["wins"] / d["trades"], 4) if d["trades"] else 0
            tp = d.get("total_profit", 0) or 0
            tl = d.get("total_loss", 0) or 0
            d["profit_factor"] = round(tp / tl, 4) if tl > 0 else 0
            result.append(d)
        return result

    # ─── SYSTEM EVENTS ──────────────────────────────────────────

    def record_system_event(self, event: SystemEvent) -> str:
        conn = self._get_conn()
        d = event.to_dict()
        conn.execute("""
            INSERT OR REPLACE INTO system_events (event_id, timestamp, event_type, severity, description, data)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (d["event_id"], d["timestamp"], d["event_type"], d["severity"],
              d["description"], json.dumps(d, default=str)))
        conn.commit()
        return event.event_id

    def get_system_events(self, event_type: str = "", severity: str = "",
                          limit: int = 100) -> List[SystemEvent]:
        conn = self._get_conn()
        conditions, params = [], []
        if event_type:
            conditions.append("event_type = ?"); params.append(event_type)
        if severity:
            conditions.append("severity = ?"); params.append(severity)
        where = " AND ".join(conditions) if conditions else "1=1"
        rows = conn.execute(
            f"SELECT data FROM system_events WHERE {where} ORDER BY timestamp DESC LIMIT ?",
            params + [limit]).fetchall()
        return [SystemEvent.from_dict(json.loads(r["data"])) for r in rows]

    # ─── PORTFOLIO SNAPSHOTS ─────────────────────────────────────

    def record_portfolio_snapshot(self, snapshot: PortfolioSnapshot) -> str:
        conn = self._get_conn()
        d = snapshot.to_dict()
        conn.execute("""
            INSERT OR REPLACE INTO portfolio_snapshots
            (snapshot_id, timestamp, trigger, total_equity, day_pnl, data)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (d["snapshot_id"], d["timestamp"], d["trigger"],
              d["total_equity"], d["day_pnl"], json.dumps(d, default=str)))
        conn.commit()
        return snapshot.snapshot_id

    def get_portfolio_snapshots(self, days: int = 30, trigger: str = "",
                                limit: int = 500) -> List[PortfolioSnapshot]:
        conn = self._get_conn()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        conditions = ["timestamp >= ?"]
        params = [cutoff]
        if trigger:
            conditions.append("trigger = ?"); params.append(trigger)
        where = " AND ".join(conditions)
        rows = conn.execute(
            f"SELECT data FROM portfolio_snapshots WHERE {where} ORDER BY timestamp DESC LIMIT ?",
            params + [limit]).fetchall()
        return [PortfolioSnapshot.from_dict(json.loads(r["data"])) for r in rows]

    def get_equity_curve(self, days: int = 365) -> List[Dict]:
        """Get equity curve from portfolio snapshots."""
        conn = self._get_conn()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        rows = conn.execute("""
            SELECT timestamp, total_equity FROM portfolio_snapshots
            WHERE timestamp >= ? ORDER BY timestamp
        """, (cutoff,)).fetchall()
        return [{"time": r["timestamp"], "equity": r["total_equity"]} for r in rows]

    # ─── AGGREGATE QUERIES ───────────────────────────────────────

    def get_summary(self, source: str = "", days: int = 0,
                     strategy: str = "", instrument: str = "",
                     direction: str = "", trade_type: str = "") -> Dict[str, Any]:
        """Comprehensive journal summary with full filter support."""
        conn = self._get_conn()
        conditions = ["is_closed = 1"]
        params: list = []
        if source:
            conditions.append("source = ?"); params.append(source)
        if days:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            conditions.append("exit_time >= ?"); params.append(cutoff)
        if strategy:
            conditions.append("strategy_name = ?"); params.append(strategy)
        if instrument:
            conditions.append("instrument = ?"); params.append(instrument)
        if direction:
            conditions.append("direction = ?"); params.append(direction)
        if trade_type:
            conditions.append("trade_type = ?"); params.append(trade_type)
        where = " AND ".join(conditions)

        row = conn.execute(f"""
            SELECT COUNT(*) as total_trades,
                   SUM(net_pnl) as total_pnl,
                   SUM(gross_pnl) as total_gross_pnl,
                   SUM(total_costs) as total_costs,
                   AVG(net_pnl) as avg_pnl,
                   AVG(return_pct) as avg_return_pct,
                   AVG(mae) as avg_mae,
                   AVG(mfe) as avg_mfe,
                   AVG(edge_ratio) as avg_edge_ratio,
                   SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                   SUM(CASE WHEN net_pnl <= 0 THEN 1 ELSE 0 END) as losing_trades,
                   SUM(CASE WHEN net_pnl > 0 THEN net_pnl ELSE 0 END) as total_profit,
                   SUM(CASE WHEN net_pnl < 0 THEN ABS(net_pnl) ELSE 0 END) as total_loss,
                   AVG(CASE WHEN net_pnl > 0 THEN net_pnl END) as avg_win,
                   AVG(CASE WHEN net_pnl < 0 THEN net_pnl END) as avg_loss,
                   MAX(net_pnl) as best_trade,
                   MIN(net_pnl) as worst_trade,
                   COUNT(DISTINCT strategy_name) as strategy_count,
                   COUNT(DISTINCT instrument) as instrument_count
            FROM journal_entries WHERE {where}
        """, params).fetchone()

        r = dict(row) if row else {}
        total = r.get("total_trades", 0) or 0
        wins = r.get("winning_trades", 0) or 0
        total_profit = r.get("total_profit", 0) or 0
        total_loss = r.get("total_loss", 0) or 0

        r["win_rate"] = round(wins / total, 4) if total > 0 else 0
        r["profit_factor"] = round(total_profit / total_loss, 4) if total_loss > 0 else 0
        r["expectancy"] = r.get("avg_pnl", 0) or 0

        # Open trades
        open_count = conn.execute(
            "SELECT COUNT(*) as c FROM journal_entries WHERE is_closed = 0"
        ).fetchone()
        r["open_trades"] = open_count["c"] if open_count else 0

        # System events summary
        events = conn.execute("""
            SELECT event_type, COUNT(*) as cnt FROM system_events
            GROUP BY event_type ORDER BY cnt DESC LIMIT 10
        """).fetchall()
        r["system_events"] = {e["event_type"]: e["cnt"] for e in events}

        return r

    def get_regime_performance_matrix(self, strategy: str = "",
                                       instrument: str = "", source: str = "",
                                       direction: str = "", trade_type: str = "",
                                       days: int = 0) -> List[Dict]:
        """Auto-build regime → strategy → PnL matrix with filters."""
        conn = self._get_conn()
        conditions = ["is_closed = 1"]
        params: list = []
        if strategy:
            conditions.append("strategy_name = ?"); params.append(strategy)
        if instrument:
            conditions.append("instrument = ?"); params.append(instrument)
        if source:
            conditions.append("source = ?"); params.append(source)
        if direction:
            conditions.append("direction = ?"); params.append(direction)
        if trade_type:
            conditions.append("trade_type = ?"); params.append(trade_type)
        if days:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            conditions.append("exit_time >= ?"); params.append(cutoff)
        where = " AND ".join(conditions)
        rows = conn.execute(f"""
            SELECT data FROM journal_entries WHERE {where}
        """, params).fetchall()

        matrix = {}  # regime -> {trades, pnl, wins}
        for row in rows:
            try:
                d = json.loads(row["data"])
                ctx = d.get("strategy_context", {})
                regime = ctx.get("trend_regime", "unknown") or "unknown"
                if regime not in matrix:
                    matrix[regime] = {"regime": regime, "trades": 0, "total_pnl": 0,
                                      "wins": 0, "avg_pnl": 0}
                matrix[regime]["trades"] += 1
                pnl = d.get("net_pnl", 0)
                matrix[regime]["total_pnl"] += pnl
                if pnl > 0:
                    matrix[regime]["wins"] += 1
            except Exception:
                continue

        for v in matrix.values():
            if v["trades"]:
                v["win_rate"] = round(v["wins"] / v["trades"], 4)
                v["avg_pnl"] = round(v["total_pnl"] / v["trades"], 2)
        return list(matrix.values())

    def get_slippage_drift(self, window_days: int = 30,
                            strategy: str = "", instrument: str = "",
                            source: str = "", direction: str = "",
                            trade_type: str = "") -> Dict:
        """Track slippage over time with filters."""
        conn = self._get_conn()
        cutoff = (datetime.now() - timedelta(days=window_days)).isoformat()
        conditions = ["is_closed = 1", "entry_time >= ?"]
        params: list = [cutoff]
        if strategy:
            conditions.append("strategy_name = ?"); params.append(strategy)
        if instrument:
            conditions.append("instrument = ?"); params.append(instrument)
        if source:
            conditions.append("source = ?"); params.append(source)
        if direction:
            conditions.append("direction = ?"); params.append(direction)
        if trade_type:
            conditions.append("trade_type = ?"); params.append(trade_type)
        where = " AND ".join(conditions)
        rows = conn.execute(f"""
            SELECT data FROM journal_entries
            WHERE {where}
            ORDER BY entry_time
        """, params).fetchall()

        slippages = []
        for row in rows:
            try:
                d = json.loads(row["data"])
                ex = d.get("execution", {})
                slip = ex.get("entry_slippage_pct", 0)
                if slip:
                    slippages.append({"time": d.get("entry_time", ""), "slippage_pct": slip})
            except Exception:
                continue

        if not slippages:
            return {"avg_slippage_pct": 0, "recent_slippage_pct": 0,
                    "drift": 0, "data_points": 0}

        avg = sum(s["slippage_pct"] for s in slippages) / len(slippages)
        recent = slippages[-min(10, len(slippages)):]
        recent_avg = sum(s["slippage_pct"] for s in recent) / len(recent)

        return {
            "avg_slippage_pct": round(avg, 4),
            "recent_slippage_pct": round(recent_avg, 4),
            "drift": round(recent_avg - avg, 4),  # + = slippage increasing
            "data_points": len(slippages),
            "history": slippages[-50:]  # last 50 points for charts
        }

    # ─── MIGRATION / COMPAT ─────────────────────────────────────

    def import_legacy_journal(self, legacy_entries: List[Dict]):
        """Import from old TradeJournal JSON format."""
        for le in legacy_entries:
            entry = JournalEntry(
                strategy_name=le.get("strategy", ""),
                tradingsymbol=le.get("tradingsymbol", ""),
                instrument=le.get("tradingsymbol", ""),
                exchange=le.get("exchange", ""),
                direction=le.get("transaction_type", ""),
                entry_time=le.get("timestamp", ""),
                exit_time=le.get("timestamp", ""),
                is_closed=True,
                net_pnl=le.get("pnl", 0),
                gross_pnl=le.get("pnl", 0),
                source="legacy",
                notes=le.get("notes", ""),
            )
            entry.execution = {
                "actual_fill_price": le.get("price", 0),
                "quantity": le.get("quantity", 0),
                "order_type": le.get("metadata", {}).get("order_type", "MARKET"),
            }
            self.record_entry(entry)
        logger.info("Imported %d legacy journal entries", len(legacy_entries))

    def export_all(self) -> List[Dict]:
        """Export all entries as dicts (for backup/analysis)."""
        conn = self._get_conn()
        rows = conn.execute("SELECT data FROM journal_entries ORDER BY entry_time").fetchall()
        return [json.loads(r["data"]) for r in rows]

    def get_db_stats(self) -> Dict:
        """Database health metrics."""
        conn = self._get_conn()
        stats = {}
        stats["entries"] = conn.execute("SELECT COUNT(*) as c FROM journal_entries").fetchone()["c"]
        stats["open_entries"] = conn.execute("SELECT COUNT(*) as c FROM journal_entries WHERE is_closed=0").fetchone()["c"]
        stats["closed_entries"] = conn.execute("SELECT COUNT(*) as c FROM journal_entries WHERE is_closed=1").fetchone()["c"]
        stats["system_events"] = conn.execute("SELECT COUNT(*) as c FROM system_events").fetchone()["c"]
        stats["snapshots"] = conn.execute("SELECT COUNT(*) as c FROM portfolio_snapshots").fetchone()["c"]
        stats["db_size_bytes"] = os.path.getsize(self._db_path) if os.path.exists(self._db_path) else 0
        stats["db_size_mb"] = round(stats["db_size_bytes"] / 1048576, 2)
        return stats
