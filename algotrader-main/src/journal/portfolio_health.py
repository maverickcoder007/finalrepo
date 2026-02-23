"""
Portfolio Health Tracker — Layer 3 System Health Monitor
========================================================

Tracks long-term portfolio survival metrics:
  - Capital metrics (equity, margin, leverage)
  - Risk metrics (drawdown, exposure, daily/weekly loss)
  - System stability (latency, disconnects, rejections, crashes)
  - Periodic snapshots (configurable interval)
  - EOD auto-snapshot
"""

from __future__ import annotations
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from collections import defaultdict

from src.journal.journal_models import (
    PortfolioSnapshot, CapitalMetrics, RiskMetrics, SystemEvent,
)
from src.journal.journal_store import JournalStore

logger = logging.getLogger("portfolio_health")


class PortfolioHealthTracker:
    """
    Continuous portfolio health monitoring.
    Records snapshots at trade events + periodic intervals.
    Tracks system stability metrics.
    """

    def __init__(self, store: JournalStore, initial_capital: float = 0):
        self._store = store
        self._initial_capital = initial_capital
        self._peak_equity = initial_capital
        self._current_equity = initial_capital

        # Running counters (reset daily)
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._daily_wins = 0
        self._daily_costs = 0.0

        # Weekly/monthly counters
        self._weekly_pnl = 0.0
        self._monthly_pnl = 0.0

        # System stability counters
        self._ws_disconnects = 0
        self._broker_rejections = 0
        self._reconciliation_events = 0
        self._recovery_events = 0
        self._crash_recoveries = 0
        self._latency_samples: List[float] = []

        # Exposure tracking
        self._exposure_by_symbol: Dict[str, float] = {}
        self._exposure_by_strategy: Dict[str, float] = {}
        self._open_positions = 0

        self._last_snapshot_time = time.time()
        self._day_start = datetime.now().date()

    # ─── TRADE EVENT HOOKS ──────────────────────────────────────

    def on_trade_open(self, trade_value: float, symbol: str = "",
                      strategy: str = "", margin_used: float = 0):
        """Called when a new trade is opened."""
        self._open_positions += 1
        if symbol:
            self._exposure_by_symbol[symbol] = (
                self._exposure_by_symbol.get(symbol, 0) + trade_value
            )
        if strategy:
            self._exposure_by_strategy[strategy] = (
                self._exposure_by_strategy.get(strategy, 0) + trade_value
            )
        self._take_snapshot("trade_open")

    def on_trade_close(self, pnl: float, costs: float = 0,
                       symbol: str = "", strategy: str = "",
                       margin_freed: float = 0):
        """Called when a trade is closed."""
        self._daily_pnl += pnl
        self._weekly_pnl += pnl
        self._monthly_pnl += pnl
        self._daily_trades += 1
        self._daily_costs += costs
        if pnl > 0:
            self._daily_wins += 1

        self._current_equity += pnl - costs
        self._peak_equity = max(self._peak_equity, self._current_equity)

        if symbol and symbol in self._exposure_by_symbol:
            self._exposure_by_symbol[symbol] -= margin_freed
            if self._exposure_by_symbol[symbol] <= 0:
                del self._exposure_by_symbol[symbol]

        self._open_positions = max(0, self._open_positions - 1)
        self._take_snapshot("trade_close")

    def on_day_end(self):
        """End-of-day snapshot and reset."""
        self._take_snapshot("eod")
        self._daily_pnl = 0
        self._daily_trades = 0
        self._daily_wins = 0
        self._daily_costs = 0
        self._exposure_by_symbol.clear()
        self._exposure_by_strategy.clear()
        self._open_positions = 0
        self._day_start = datetime.now().date()

    def on_week_end(self):
        self._weekly_pnl = 0

    def on_month_end(self):
        self._monthly_pnl = 0

    # ─── SYSTEM STABILITY HOOKS ──────────────────────────────────

    def record_ws_disconnect(self):
        self._ws_disconnects += 1
        self._record_system_event("ws_disconnect", "warning",
                                  f"WebSocket disconnect #{self._ws_disconnects}")

    def record_broker_rejection(self, reason: str = ""):
        self._broker_rejections += 1
        self._record_system_event("broker_rejection", "warning",
                                  f"Order rejected: {reason}")

    def record_reconciliation(self, details: str = ""):
        self._reconciliation_events += 1
        self._record_system_event("reconciliation", "info", details)

    def record_crash_recovery(self, details: str = ""):
        self._crash_recoveries += 1
        self._record_system_event("crash_recovery", "error", details)

    def record_latency(self, latency_ms: float):
        self._latency_samples.append(latency_ms)
        if len(self._latency_samples) > 1000:
            self._latency_samples = self._latency_samples[-500:]

    def _record_system_event(self, event_type: str, severity: str, description: str):
        event = SystemEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            severity=severity,
            description=description,
            execution_latency_ms=(
                self._latency_samples[-1] if self._latency_samples else 0
            ),
            websocket_disconnects=self._ws_disconnects,
            broker_rejections=self._broker_rejections,
            reconciliation_events=self._reconciliation_events,
            crash_recoveries=self._crash_recoveries,
        )
        try:
            self._store.record_system_event(event)
        except Exception as e:
            logger.error("Failed to record system event: %s", e)

    # ─── SNAPSHOT MANAGEMENT ─────────────────────────────────────

    def _take_snapshot(self, trigger: str):
        """Create and store a portfolio snapshot."""
        now = datetime.now()
        drawdown = self._peak_equity - self._current_equity
        drawdown_pct = (drawdown / self._peak_equity * 100) if self._peak_equity > 0 else 0

        # Total exposure
        total_exposure = sum(abs(v) for v in self._exposure_by_symbol.values())
        leverage = (total_exposure / self._current_equity
                    if self._current_equity > 0 else 0)

        # Max single position
        max_single_pct = 0
        if self._exposure_by_symbol and self._current_equity > 0:
            max_single = max(abs(v) for v in self._exposure_by_symbol.values())
            max_single_pct = max_single / self._current_equity * 100

        capital = CapitalMetrics(
            equity_before=self._current_equity - self._daily_pnl + self._daily_costs,
            equity_after=self._current_equity,
            capital_at_risk_pct=round(total_exposure / self._current_equity * 100, 2)
                                if self._current_equity > 0 else 0,
            margin_utilization_pct=0,  # updated by broker integration
            leverage_ratio=round(leverage, 4),
            available_margin=0,
            used_margin=total_exposure,
        )

        risk = RiskMetrics(
            rolling_drawdown_pct=round(drawdown_pct, 4),
            peak_to_valley_drawdown_pct=round(drawdown_pct, 4),
            daily_loss=round(min(0, self._daily_pnl), 2),
            weekly_loss=round(min(0, self._weekly_pnl), 2),
            monthly_loss=round(min(0, self._monthly_pnl), 2),
            open_positions_count=self._open_positions,
            exposure_by_symbol=dict(self._exposure_by_symbol),
            exposure_by_strategy=dict(self._exposure_by_strategy),
            max_single_position_pct=round(max_single_pct, 2),
        )

        snapshot = PortfolioSnapshot(
            timestamp=now.isoformat(),
            trigger=trigger,
            capital=capital.to_dict(),
            risk=risk.to_dict(),
            total_equity=round(self._current_equity, 2),
            day_pnl=round(self._daily_pnl, 2),
            unrealized_pnl=0,  # updated by position tracker integration
            realized_pnl=round(self._daily_pnl, 2),
            total_trades_today=self._daily_trades,
            win_rate_today=round(
                self._daily_wins / self._daily_trades * 100, 1
            ) if self._daily_trades > 0 else 0,
        )

        try:
            self._store.record_portfolio_snapshot(snapshot)
        except Exception as e:
            logger.error("Failed to record portfolio snapshot: %s", e)

    def take_periodic_snapshot(self, interval_seconds: int = 300):
        """Call periodically (e.g., every 5 minutes during market hours)."""
        now = time.time()
        if now - self._last_snapshot_time >= interval_seconds:
            self._take_snapshot("periodic")
            self._last_snapshot_time = now

    # ─── QUERY METHODS ───────────────────────────────────────────

    def get_current_health(self) -> Dict[str, Any]:
        """Real-time portfolio health dashboard data."""
        drawdown = self._peak_equity - self._current_equity
        drawdown_pct = (drawdown / self._peak_equity * 100) if self._peak_equity else 0

        total_exposure = sum(abs(v) for v in self._exposure_by_symbol.values())

        avg_latency = (
            sum(self._latency_samples[-50:]) / min(50, len(self._latency_samples))
            if self._latency_samples else 0
        )
        max_latency = max(self._latency_samples[-50:]) if self._latency_samples else 0

        return {
            "timestamp": datetime.now().isoformat(),
            # Capital
            "current_equity": round(self._current_equity, 2),
            "initial_capital": round(self._initial_capital, 2),
            "peak_equity": round(self._peak_equity, 2),
            "total_return_pct": round(
                (self._current_equity - self._initial_capital) / self._initial_capital * 100, 2
            ) if self._initial_capital > 0 else 0,
            # Risk
            "current_drawdown": round(drawdown, 2),
            "current_drawdown_pct": round(drawdown_pct, 2),
            "daily_pnl": round(self._daily_pnl, 2),
            "weekly_pnl": round(self._weekly_pnl, 2),
            "monthly_pnl": round(self._monthly_pnl, 2),
            "open_positions": self._open_positions,
            "total_exposure": round(total_exposure, 2),
            "exposure_by_symbol": dict(self._exposure_by_symbol),
            "exposure_by_strategy": dict(self._exposure_by_strategy),
            # Today's stats
            "daily_trades": self._daily_trades,
            "daily_wins": self._daily_wins,
            "daily_win_rate": round(
                self._daily_wins / self._daily_trades * 100, 1
            ) if self._daily_trades > 0 else 0,
            "daily_costs": round(self._daily_costs, 2),
            # System stability
            "system_stability": {
                "avg_latency_ms": round(avg_latency, 1),
                "max_latency_ms": round(max_latency, 1),
                "ws_disconnects": self._ws_disconnects,
                "broker_rejections": self._broker_rejections,
                "reconciliation_events": self._reconciliation_events,
                "crash_recoveries": self._crash_recoveries,
            },
        }

    def get_health_summary(self, days: int = 30) -> Dict[str, Any]:
        """Historical health summary from snapshots."""
        snapshots = self._store.get_portfolio_snapshots(days=days, limit=1000)
        events = self._store.get_system_events(limit=100)

        if not snapshots:
            return {"message": "No portfolio snapshots yet", "current": self.get_current_health()}

        equities = [s.total_equity for s in snapshots if s.total_equity > 0]
        daily_pnls = [s.day_pnl for s in snapshots if s.trigger == "eod"]

        peak = max(equities) if equities else 0
        trough = min(equities) if equities else 0
        max_dd = peak - trough if peak > 0 else 0
        avg_daily_pnl = sum(daily_pnls) / len(daily_pnls) if daily_pnls else 0

        # System events breakdown
        event_summary = defaultdict(int)
        for e in events:
            event_summary[e.event_type] += 1

        return {
            "current": self.get_current_health(),
            "period_days": days,
            "snapshots_count": len(snapshots),
            "equity_range": {
                "peak": round(peak, 2),
                "trough": round(trough, 2),
                "max_drawdown": round(max_dd, 2),
            },
            "avg_daily_pnl": round(avg_daily_pnl, 2),
            "profitable_days": sum(1 for p in daily_pnls if p > 0),
            "total_days": len(daily_pnls),
            "system_events": dict(event_summary),
            "recent_events": [e.to_dict() for e in events[:20]],
        }

    def update_equity(self, equity: float, margin_available: float = 0,
                      margin_used: float = 0):
        """Update current equity from broker data."""
        self._current_equity = equity
        self._peak_equity = max(self._peak_equity, equity)

    def set_initial_capital(self, capital: float):
        self._initial_capital = capital
        if self._current_equity == 0:
            self._current_equity = capital
            self._peak_equity = capital
