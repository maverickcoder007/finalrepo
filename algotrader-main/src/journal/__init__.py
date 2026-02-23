"""
Production-Grade 3-Layer Trade Journal System
==============================================

Layer 1: Trade Execution Quality   — Did we execute correctly?
Layer 2: Strategy Edge Quality     — Does the idea actually work?
Layer 3: Portfolio Health          — Will we survive long-term?

Architecture:
  journal_models.py   — All dataclasses for 3 layers + system events
  journal_store.py    — SQLite-backed event-sourced storage engine
  journal_analytics.py — Rolling metrics, regime matrix, edge decay
  portfolio_health.py — Capital, risk, drawdown, system stability
"""

from src.journal.journal_models import (
    # Layer 1 — Execution
    ExecutionRecord,
    OrderEvent,
    FillRecord,
    LiquiditySnapshot,
    CostBreakdown,
    ExcursionMetrics,
    # Layer 2 — Strategy Edge
    StrategyContext,
    SignalQuality,
    PositionStructure,
    # Layer 3 — Portfolio Health
    PortfolioSnapshot,
    CapitalMetrics,
    RiskMetrics,
    SystemEvent,
    # Full trade journal entry
    JournalEntry,
)

from src.journal.journal_store import JournalStore
from src.journal.journal_analytics import JournalAnalytics
from src.journal.portfolio_health import PortfolioHealthTracker

__all__ = [
    # Models
    "ExecutionRecord", "OrderEvent", "FillRecord", "LiquiditySnapshot",
    "CostBreakdown", "ExcursionMetrics",
    "StrategyContext", "SignalQuality", "PositionStructure",
    "PortfolioSnapshot", "CapitalMetrics", "RiskMetrics", "SystemEvent",
    "JournalEntry",
    # Engines
    "JournalStore", "JournalAnalytics", "PortfolioHealthTracker",
]
