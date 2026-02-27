from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, Optional

from src.analysis.scanner import StockScanner
from src.api.client import KiteClient
from src.api.market_data import MarketDataProvider, PortfolioReporter
from src.api.ticker import KiteTicker
from src.auth.authenticator import KiteAuthenticator
from src.data.backtest import BacktestEngine, WalkForwardOptimizer
from src.data.journal import TradeJournal
from src.data.models import (
    Interval,
    Signal,
    Tick,
    TickMode,
)
from src.data.paper_trader import PaperTradingEngine
from src.data.synthetic import generate_synthetic_ohlcv
from src.data.market_data_store import MarketDataStore
from src.derivatives.fno_backtest import FnOBacktestEngine
from src.derivatives.fno_paper_trader import FnOPaperTradingEngine
from src.execution.engine import ExecutionEngine
from src.journal.journal_store import JournalStore
from src.journal.journal_analytics import JournalAnalytics
from src.journal.portfolio_health import PortfolioHealthTracker
from src.journal.journal_models import (
    JournalEntry, ExecutionRecord, StrategyContext,
    PositionStructure, CostBreakdown, ExcursionMetrics,
    PortfolioSnapshot, CapitalMetrics, RiskMetrics, SystemEvent,
    SignalQuality, LiquiditySnapshot,
)
from src.options.chain import OptionChainBuilder
from src.options.oi_analysis import FuturesOIAnalyzer, OptionsOIAnalyzer
from src.options.oi_tracker import NIFTY_TOKEN, SENSEX_TOKEN, OITracker
from src.options.oi_strategy import OIStrategyEngine, OIStrategyConfig
from src.options.call_credit_spread_runner import CallCreditSpreadRunnerStrategy
from src.options.put_credit_spread_runner import PutCreditSpreadRunnerStrategy
from src.options.strategies import (
    BearPutSpreadStrategy,
    BullCallSpreadStrategy,
    IronCondorStrategy,
    OptionStrategyBase,
    StraddleStrangleStrategy,
)
from src.risk.manager import RiskManager
from src.risk.preflight import PreflightChecker, PreflightConfig, TradingMode, set_preflight_checker
from src.strategy.base import BaseStrategy
from src.strategy.custom_builder import (
    CustomStrategy,
    CustomStrategyStore,
    get_available_indicators,
)
from src.strategy.ema_crossover import EMACrossoverStrategy
from src.strategy.mean_reversion import MeanReversionStrategy
from src.strategy.rsi_strategy import RSIStrategy
from src.strategy.scanner_strategy import ScannerStrategy
from src.strategy.vwap_breakout import VWAPBreakoutStrategy
from src.strategy.health_engine import compute_health_report, HealthReport, PillarResult, get_health_grade
from src.options.fno_builder import (
    FnOStrategyStore,
    FnOStrategyConfig,
    FnOLeg,
    resolve_legs,
    compute_strategy_payoff,
    UNDERLYING_LOT_SIZES,
    UNDERLYING_STRIKE_GAPS,
)
from src.utils.config import get_settings
from src.utils.exceptions import KillSwitchError, RiskLimitError
from src.utils.logger import get_logger

import pandas as pd

logger = get_logger(__name__)


class TradingService:
    _instance: Optional[TradingService] = None

    def __init__(self, api_key: str = "", api_secret: str = "", zerodha_id: str = "") -> None:
        import os
        self._settings = get_settings()
        self._zerodha_id = zerodha_id
        self._auth = KiteAuthenticator(api_key=api_key, api_secret=api_secret, zerodha_id=zerodha_id)
        self._client = KiteClient(self._auth)
        self._ticker = KiteTicker(self._auth)
        self._risk = RiskManager()

        # ── DB initialization: create files if missing ──
        os.makedirs("data", exist_ok=True)
        db_files = ["data/journal.db", "data/market_data.db"]
        for db_file in db_files:
            if not os.path.exists(db_file):
                open(db_file, "a").close()

        # Pre-flight checker: consolidated gate before any order
        self._preflight = PreflightChecker(
            client=self._client,
            ticker=self._ticker,
            risk_manager=self._risk,
            chain_builder=None,  # Set after chain_builder init
            config=PreflightConfig(),
        )
        set_preflight_checker(self._preflight)

        self._execution = ExecutionEngine(self._client, self._risk, preflight=self._preflight)
        self._journal = TradeJournal()
        self._journal_store = JournalStore(db_path="data/journal.db")
        self._journal_analytics = JournalAnalytics(self._journal_store)
        self._portfolio_health = PortfolioHealthTracker(self._journal_store)

        # Wire execution engine → journal callbacks
        self._execution.set_on_fill_callback(self._on_execution_fill)
        self._execution.set_on_exit_callback(self._on_execution_exit)
        self._chain_builder = OptionChainBuilder()
        self._preflight._chain = self._chain_builder  # Wire chain builder for F&O validation
        self._oi_tracker = OITracker()
        self._oi_streaming = False
        self._futures_oi = FuturesOIAnalyzer()
        self._options_oi = OptionsOIAnalyzer()
        self._oi_strategy = OIStrategyEngine()
        self._scanner = StockScanner()
        self._strategies: list[BaseStrategy] = []
        self._instruments_cache: dict[str, list] = {}
        self._running = False
        self._subscribed_tokens: list[int] = []
        self._live_ticks: dict[int, dict[str, Any]] = {}
        # Token → display name map (pre-seeded with common instruments)
        self._token_name_map: dict[int, str] = {
            256265: "NIFTY 50", 260105: "NIFTY BANK", 257801: "NIFTY FIN SERVICE",
            265: "SENSEX", 738561: "RELIANCE", 341249: "HDFCBANK",
            2953217: "TCS", 408065: "INFY", 2885377: "SBIN",
            424961: "ICICIBANK", 1270529: "KOTAKBANK", 3861249: "AXISBANK",
            2714625: "LT", 3465729: "TATAMOTORS", 3456769: "TATASTEEL",
            969473: "MARUTI", 2760193: "BAJFINANCE", 895745: "HINDUNILVR",
            81153: "ADANIENT", 3001089: "SUNPHARMA", 2815745: "WIPRO",
            779521: "POWERGRID", 2929921: "TECHM", 5215745: "ASIANPAINT",
            54273: "BHARTIARTL", 315393: "ITC", 261889: "NIFTY FIN SERVICE",
        }
        self._signal_log: list[dict[str, Any]] = []
        self._ws_clients: list[Any] = []
        self._lock = asyncio.Lock()
        self._custom_store = CustomStrategyStore()
        self._fno_strategy_store = FnOStrategyStore()
        self._paper_results: dict[str, Any] = {}  # cache latest paper trade results
        self._data_store = MarketDataStore(db_path="data/market_data.db")
        self._market_data = MarketDataProvider(self._client, store=self._data_store)
        self._portfolio_reporter = PortfolioReporter(self._client)
        self._backtest_results: dict[str, Any] = {}  # cache latest backtest results
        self._last_health_report: Optional[dict[str, Any]] = None  # cache latest health report
        self._tested_strategies: set[str] = self._data_store.load_tested_strategies()

    # ─── Tested Strategy Persistence ────────────────────────────

    def _mark_strategy_tested(self, name: str) -> None:
        """Add a strategy to the tested set and persist to DB."""
        self._tested_strategies.add(name)
        self._data_store.add_tested_strategy(name)

    # ─── Preflight API ──────────────────────────────────────────

    async def run_preflight_check(self, mode: str = "live") -> dict[str, Any]:
        """Run full preflight checklist and return report for dashboard."""
        trading_mode = {
            "live": TradingMode.LIVE,
            "paper": TradingMode.PAPER,
            "backtest": TradingMode.BACKTEST,
        }.get(mode, TradingMode.LIVE)

        report = await self._preflight.run_all_checks(mode=trading_mode)
        return report.to_dict()

    def get_last_preflight_report(self) -> Optional[dict[str, Any]]:
        """Get the last preflight report (cached)."""
        return self._preflight.get_last_report()

    @staticmethod
    def _task_exception_handler(task: asyncio.Task) -> None:
        """Log exceptions from fire-and-forget background tasks."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("background_task_failed", task=task.get_name(), error=str(exc), exc_info=exc)

    @classmethod
    def get_instance(cls) -> TradingService:
        if cls._instance is None:
            cls._instance = TradingService()
        return cls._instance

    @property
    def zerodha_id(self) -> str:
        return self._zerodha_id

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_authenticated(self) -> bool:
        try:
            return self._auth.is_authenticated
        except Exception:
            return False

    def get_login_url(self) -> str:
        return self._auth.get_login_url()

    async def authenticate(self, request_token: str) -> dict[str, Any]:
        session = await self._auth.generate_session(request_token)
        return {"user_id": session.user_id, "user_name": session.user_name}

    async def search_instruments(self, query: str, exchange: str = "NSE") -> list[dict[str, Any]]:
        if exchange not in self._instruments_cache:
            self._instruments_cache[exchange] = await self._market_data.get_instruments(exchange)

        results = []
        q = query.upper()
        eq_only = exchange in ("NSE", "BSE")
        for inst in self._instruments_cache[exchange]:
            itype = inst.get("instrument_type", "") if isinstance(inst, dict) else getattr(inst, "instrument_type", "")
            tsym = inst.get("tradingsymbol", "") if isinstance(inst, dict) else getattr(inst, "tradingsymbol", "")
            iname = inst.get("name", "") if isinstance(inst, dict) else getattr(inst, "name", "")
            itoken = inst.get("instrument_token", 0) if isinstance(inst, dict) else getattr(inst, "instrument_token", 0)
            iexch = inst.get("exchange", exchange) if isinstance(inst, dict) else getattr(inst, "exchange", exchange)
            ilot = inst.get("lot_size", 0) if isinstance(inst, dict) else getattr(inst, "lot_size", 0)
            iexpiry = inst.get("expiry", "") if isinstance(inst, dict) else getattr(inst, "expiry", "")
            istrike = inst.get("strike", 0) if isinstance(inst, dict) else getattr(inst, "strike", 0)

            if eq_only and itype not in ("EQ", ""):
                continue
            if not q or q in tsym.upper() or q in (iname or "").upper():
                results.append({
                    "token": itoken,
                    "symbol": tsym,
                    "name": iname or tsym,
                    "exchange": iexch,
                    "type": itype,
                    "lot_size": ilot,
                    "expiry": iexpiry or "",
                    "strike": istrike,
                })
            if len(results) >= 50:
                break
        return results

    async def get_profile(self) -> dict[str, Any]:
        profile = await self._client.get_profile()
        return profile.model_dump()

    async def get_margins(self) -> dict[str, Any]:
        margins = await self._client.get_margins()
        return margins.model_dump()

    async def get_orders(self) -> list[dict[str, Any]]:
        orders = await self._client.get_orders()
        order_dicts = [o.model_dump() for o in orders]
        # Persist to store
        if self._data_store:
            try:
                self._data_store.store_orders(order_dicts)
            except Exception:
                pass
        return order_dicts

    async def get_positions(self) -> dict[str, Any]:
        positions = await self._client.get_positions()
        pos_dump = positions.model_dump()
        # Persist net positions to store
        if self._data_store and pos_dump.get("net"):
            try:
                self._data_store.store_positions(pos_dump["net"])
            except Exception:
                pass
        return pos_dump

    async def get_holdings(self) -> list[dict[str, Any]]:
        holdings = await self._client.get_holdings()
        return [h.model_dump() for h in holdings]

    def get_risk_summary(self) -> dict[str, Any]:
        return self._risk.get_risk_summary()

    def get_journal_summary(self) -> dict[str, Any]:
        return self._journal.get_summary()

    # ── New 3-Layer Journal API ──

    def get_pro_journal_summary(self, source: str = "", days: int = 0,
                                 strategy: str = "", instrument: str = "",
                                 direction: str = "", trade_type: str = "") -> dict[str, Any]:
        return self._journal_store.get_summary(
            source=source, days=days, strategy=strategy,
            instrument=instrument, direction=direction, trade_type=trade_type)

    def get_journal_analytics(self, strategy: str = "", source: str = "",
                               days: int = 0, instrument: str = "",
                               direction: str = "", trade_type: str = "") -> dict[str, Any]:
        return self._journal_analytics.compute_full_analytics(
            strategy=strategy, source=source, days=days,
            instrument=instrument, direction=direction, trade_type=trade_type)

    def get_fno_journal_analytics(self, strategy: str = "",
                                   days: int = 0, instrument: str = "",
                                   direction: str = "", trade_type: str = "",
                                   source: str = "") -> dict[str, Any]:
        return self._journal_analytics.compute_fno_analytics(
            strategy=strategy, days=days)

    def get_journal_entries(self, strategy: str = "", instrument: str = "",
                            trade_type: str = "", source: str = "",
                            direction: str = "", review_status: str = "",
                            is_closed: bool | None = None,
                            from_date: str = "", to_date: str = "",
                            limit: int = 100, offset: int = 0) -> dict:
        entries = self._journal_store.query_entries(
            strategy=strategy, instrument=instrument, trade_type=trade_type,
            source=source, direction=direction, review_status=review_status,
            is_closed=is_closed,
            from_date=from_date, to_date=to_date,
            limit=limit, offset=offset)
        total = self._journal_store.count_entries(
            strategy=strategy, instrument=instrument, source=source,
            direction=direction, trade_type=trade_type,
            review_status=review_status,
            from_date=from_date, to_date=to_date)
        return {"entries": [e.to_dict() for e in entries], "total": total}

    def get_journal_entry(self, entry_id: str) -> dict[str, Any] | None:
        entry = self._journal_store.get_entry(entry_id)
        return entry.to_dict() if entry else None

    def get_regime_matrix(self, strategy: str = "", instrument: str = "",
                           source: str = "", direction: str = "",
                           trade_type: str = "", days: int = 0) -> list[dict]:
        return self._journal_store.get_regime_performance_matrix(
            strategy=strategy, instrument=instrument, source=source,
            direction=direction, trade_type=trade_type, days=days)

    def get_slippage_drift(self, days: int = 30, strategy: str = "",
                            instrument: str = "", source: str = "",
                            direction: str = "", trade_type: str = "") -> dict:
        return self._journal_store.get_slippage_drift(
            window_days=days, strategy=strategy, instrument=instrument,
            source=source, direction=direction, trade_type=trade_type)

    def get_daily_pnl_history(self, days: int = 30, strategy: str = "",
                               instrument: str = "", source: str = "",
                               direction: str = "", trade_type: str = "") -> list[dict]:
        return self._journal_store.get_pnl_by_date(
            days=days, strategy=strategy, instrument=instrument,
            source=source, direction=direction, trade_type=trade_type)

    def get_strategy_breakdown(self, source: str = "", days: int = 0,
                                instrument: str = "", direction: str = "",
                                trade_type: str = "", strategy: str = "") -> list[dict]:
        return self._journal_store.get_pnl_by_strategy(
            source=source, days=days, instrument=instrument,
            direction=direction, trade_type=trade_type, strategy=strategy)

    def get_portfolio_health(self) -> dict[str, Any]:
        return self._portfolio_health.get_current_health()

    def get_portfolio_health_history(self, days: int = 30) -> dict[str, Any]:
        return self._portfolio_health.get_health_summary(days=days)

    def get_equity_curve(self, days: int = 365) -> list[dict]:
        return self._journal_store.get_equity_curve(days=days)

    def get_system_events(self, event_type: str = "",
                           limit: int = 100) -> list[dict]:
        events = self._journal_store.get_system_events(
            event_type=event_type, limit=limit)
        return [e.to_dict() for e in events]

    def get_journal_db_stats(self) -> dict:
        return self._journal_store.get_db_stats()

    def get_journal_strategies(self) -> list[str]:
        return self._journal_store.get_strategies()

    def get_journal_instruments(self) -> list[str]:
        return self._journal_store.get_instruments()

    def record_journal_entry(self, entry_data: dict) -> str:
        """Record a new journal entry (from UI or integrations)."""
        entry = JournalEntry.from_dict(entry_data)
        return self._journal_store.record_entry(entry)

    def update_journal_entry(self, entry_id: str, updates: dict) -> bool:
        """Update an existing journal entry (notes, tags, review status)."""
        entry = self._journal_store.get_entry(entry_id)
        if not entry:
            return False
        if "notes" in updates:
            entry.notes = updates["notes"]
        if "tags" in updates:
            entry.tags = updates["tags"]
        if "review_status" in updates:
            entry.review_status = updates["review_status"]
        self._journal_store.record_entry(entry)
        return True

    def export_journal(self) -> list[dict]:
        return self._journal_store.export_all()

    def get_recent_signals(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._signal_log[-limit:]

    def get_strategies_info(self) -> list[dict[str, Any]]:
        from src.strategy.analysis_strategies import ANALYSIS_STRATEGIES
        # Build full strategy map (equity/options)
        strategy_map = {
            "ema_crossover": EMACrossoverStrategy,
            "vwap_breakout": VWAPBreakoutStrategy,
            "mean_reversion": MeanReversionStrategy,
            "rsi": RSIStrategy,
            "rsi_strategy": RSIStrategy,
            "scanner_strategy": ScannerStrategy,
            "iron_condor": IronCondorStrategy,
            "straddle_strangle": StraddleStrangleStrategy,
            "bull_call_spread": BullCallSpreadStrategy,
            "bear_put_spread": BearPutSpreadStrategy,
            "call_credit_spread_runner": CallCreditSpreadRunnerStrategy,
            "put_credit_spread_runner": PutCreditSpreadRunnerStrategy,
        }
        strategy_map.update(ANALYSIS_STRATEGIES)

        # Active/running strategies
        active_names = {s.name for s in self._strategies if getattr(s, 'is_live', True)}
        active = {s.name: s for s in self._strategies if getattr(s, 'is_live', True)}

        # Health report cache
        health = self.get_last_health_report() or {}
        tested = self._tested_strategies

        all_strats = []
        # Equity/options strategies
        for name, cls in strategy_map.items():
            strat_info = {
                "name": name,
                "is_active": name in active_names,
                "params": active[name].params if name in active else {},
                "type": "option" if issubclass(cls, OptionStrategyBase) else "equity",
                "source": "builtin",
                "tested": name in tested,
                "health_score": health.get("overall_score") if health.get("strategy_name") == name else None,
                "execution_ready": health.get("execution_ready") if health.get("strategy_name") == name else None,
            }
            all_strats.append(strat_info)

        # Custom strategies from builder
        for cs in self._custom_store.list_strategies():
            name = cs["name"]
            strat_info = {
                "name": name,
                "is_active": name in active_names,
                "params": active[name].params if name in active else {},
                "type": "custom",
                "source": "custom_builder",
                "description": cs.get("description", ""),
                "entry_rules": cs.get("entry_rules", 0),
                "exit_rules": cs.get("exit_rules", 0),
                "tested": name in tested,
                "health_score": health.get("overall_score") if health.get("strategy_name") == name else None,
                "execution_ready": health.get("execution_ready") if health.get("strategy_name") == name else None,
            }
            all_strats.append(strat_info)

        # FNO strategies (built-in)
        for fname in self.FNO_STRATEGIES:
            strat_info = {
                "name": fname,
                "is_active": fname in active_names,
                "params": active[fname].params if fname in active else {},
                "type": "fno",
                "source": "fno_builtin",
                "tested": fname in tested,
                "health_score": health.get("overall_score") if health.get("strategy_name") == fname else None,
                "execution_ready": health.get("execution_ready") if health.get("strategy_name") == fname else None,
            }
            all_strats.append(strat_info)

        # Custom FNO builder strategies
        for fno_cs in self._fno_strategy_store.list_all():
            name = fno_cs.get("name", "")
            strat_info = {
                "name": name,
                "is_active": name in active_names,
                "params": active[name].params if name in active else {},
                "type": "fno",
                "source": "fno_custom_builder",
                "description": fno_cs.get("description", ""),
                "legs": fno_cs.get("legs", []),
                "tested": name in tested,
                "health_score": health.get("overall_score") if health.get("strategy_name") == name else None,
                "execution_ready": health.get("execution_ready") if health.get("strategy_name") == name else None,
            }
            all_strats.append(strat_info)

        return all_strats

    def get_execution_summary(self) -> dict[str, Any]:
        return self._execution.get_execution_summary()

    def get_live_ticks(self) -> dict[int, dict[str, Any]]:
        return self._live_ticks.copy()

    async def get_option_chain(
        self, underlying: str = "NIFTY", expiry: Optional[str] = None
    ) -> dict[str, Any]:
        cached = self._chain_builder.get_cached_chain(underlying, expiry or "")
        if cached:
            return cached.model_dump()
        return {"error": "No option chain data available. Subscribe to live data first."}

    @property
    def oi_tracker(self) -> OITracker:
        return self._oi_tracker

    def get_oi_summary(self, underlying: str = "NIFTY") -> dict[str, Any]:
        return self._oi_tracker.get_summary(underlying).model_dump()

    def get_oi_all(self) -> dict[str, Any]:
        return self._oi_tracker.get_all_tracked_data()

    async def get_futures_oi_report(self) -> dict[str, Any]:
        """Run full futures OI analysis."""
        try:
            report = await self._futures_oi.analyze(self._client)
            return report.model_dump()
        except Exception as e:
            logger.error("futures_oi_analysis_error", error=str(e))
            return {"error": str(e)}

    def get_futures_oi_cached(self) -> dict[str, Any]:
        """Return last computed futures OI report."""
        report = self._futures_oi.get_last_report()
        return report.model_dump() if report else {"error": "No data yet. Run scan first."}

    async def get_options_oi_report(self, underlying: str = "NIFTY", expiry: Optional[str] = None) -> dict[str, Any]:
        """Run near-ATM options OI analysis for an index."""
        try:
            report = await self._options_oi.analyze(underlying, self._client, expiry_filter=expiry)
            return report.model_dump()
        except Exception as e:
            logger.error("options_oi_analysis_error", underlying=underlying, error=str(e))
            return {"error": str(e)}

    def get_options_oi_cached(self, underlying: str = "NIFTY") -> dict[str, Any]:
        """Return last computed options OI report."""
        report = self._options_oi.get_last_report(underlying)
        return report.model_dump() if report else {"error": "No data yet. Run scan first."}

    async def get_options_oi_comparison(self, underlying: str = "NIFTY") -> dict[str, Any]:
        """Compare current vs next expiry options OI."""
        try:
            comp = await self._options_oi.analyze_comparison(underlying, self._client)
            return comp.model_dump()
        except Exception as e:
            logger.error("options_oi_comparison_error", underlying=underlying, error=str(e))
            return {"error": str(e)}

    def get_pcr_history(self, underlying: str = "NIFTY") -> list[dict[str, Any]]:
        return self._options_oi.get_pcr_history(underlying)

    def get_straddle_history(self, underlying: str = "NIFTY") -> list[dict[str, Any]]:
        return self._options_oi.get_straddle_history(underlying)

    async def start_oi_tracking(
        self,
        nifty_spot: float = 0.0,
        sensex_spot: float = 0.0,
        instruments: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        # Auto-discover instruments if not provided
        if not instruments and not self._oi_tracker.get_tracked_tokens():
            try:
                nfo_instruments = await self._market_data.get_instruments("NFO")
                inst_dicts = [
                    i if isinstance(i, dict) else (i.model_dump() if hasattr(i, "model_dump") else i)
                    for i in nfo_instruments
                    if (i.get("instrument_type", "") if isinstance(i, dict) else getattr(i, "instrument_type", "")) in ("CE", "PE")
                ]
                self._oi_tracker.register_instruments(inst_dicts)
                logger.info("oi_instruments_auto_loaded", count=len(inst_dicts))
            except Exception as e:
                logger.error("oi_instruments_auto_load_error", error=str(e))

        if instruments:
            self._oi_tracker.register_instruments(instruments)

        # Auto-fetch spot prices if not provided
        if nifty_spot <= 0:
            try:
                ltp_data = await self._client.get_ltp(["NSE:NIFTY 50"])
                for k, v in ltp_data.items():
                    price = v.last_price if hasattr(v, "last_price") else 0
                    if price > 0:
                        nifty_spot = price
            except Exception:
                pass
        if sensex_spot <= 0:
            try:
                ltp_data = await self._client.get_ltp(["BSE:SENSEX"])
                for k, v in ltp_data.items():
                    price = v.last_price if hasattr(v, "last_price") else 0
                    if price > 0:
                        sensex_spot = price
            except Exception:
                pass

        if nifty_spot > 0:
            self._oi_tracker.set_spot_price("NIFTY", nifty_spot)
        if sensex_spot > 0:
            self._oi_tracker.set_spot_price("SENSEX", sensex_spot)

        tokens = self._oi_tracker.get_tracked_tokens()
        if not tokens:
            nifty_tokens = self._oi_tracker.get_nifty_option_tokens(nifty_spot) if nifty_spot else []
            sensex_tokens = self._oi_tracker.get_sensex_option_tokens(sensex_spot) if sensex_spot else []
            tokens = nifty_tokens + sensex_tokens

        # ── Attempt to connect ticker WebSocket for live ticks ──
        ticker_connected = False
        if tokens:
            # Ensure on_ticks callback is wired so OI data flows
            if not self._ticker.on_ticks:
                self._ticker.on_ticks = self._on_ticks
            try:
                await self._ticker.subscribe(tokens, TickMode.FULL)
                # If ticker is not already connected, try to connect it
                if not self._ticker.is_connected and not self._running:
                    self._ticker.on_connect = self._on_connect
                    self._ticker.on_close = self._on_close
                    self._ticker.on_error = self._on_error
                    task = asyncio.create_task(self._ticker.connect())
                    task.add_done_callback(self._task_exception_handler)
                    logger.info("oi_ticker_connecting")
                ticker_connected = self._ticker.is_connected
            except Exception as e:
                logger.warning("oi_ticker_subscribe_error", error=str(e))

        # ── Start broadcast loop ──
        if not self._oi_streaming:
            self._oi_streaming = True
            task = asyncio.create_task(self._oi_broadcast_loop())
            task.add_done_callback(self._task_exception_handler)

        # ── Start REST polling fallback (always, for reliability) ──
        if not getattr(self, '_oi_rest_polling', False):
            self._oi_rest_polling = True
            # Use near-ATM tokens for REST polling (much smaller than full token list)
            near_atm_tokens = []
            if nifty_spot > 0:
                near_atm_tokens += self._oi_tracker.get_nifty_option_tokens(nifty_spot)
            if sensex_spot > 0:
                near_atm_tokens += self._oi_tracker.get_sensex_option_tokens(sensex_spot)
            poll_tokens = near_atm_tokens if near_atm_tokens else tokens[:200]
            # Do an immediate REST poll to seed data right away
            await self._oi_rest_poll_once(poll_tokens)
            task = asyncio.create_task(self._oi_rest_poll_loop(poll_tokens))
            task.add_done_callback(self._task_exception_handler)

        logger.info("oi_tracking_started",
                    token_count=len(tokens), nifty_spot=nifty_spot,
                    ticker_connected=ticker_connected,
                    nifty_expiry=self._oi_tracker._nearest_expiry.get("NIFTY", ""),
                    sensex_expiry=self._oi_tracker._nearest_expiry.get("SENSEX", ""))
        return {"success": True, "tracked_tokens": len(tokens), "nifty_spot": nifty_spot, "sensex_spot": sensex_spot}

    async def _oi_rest_poll_once(self, tokens: list[int]) -> None:
        """Fetch OI data via REST API and feed into tracker."""
        if not tokens:
            return
        inst_map = self._oi_tracker._instrument_map
        # Build NFO:TRADINGSYMBOL keys for the tokens (batch of max 500 at a time)
        quote_keys: list[str] = []
        token_to_key: dict[str, int] = {}
        for t in tokens:
            inst = inst_map.get(t)
            if inst:
                exchange = inst.get("exchange", "NFO")
                symbol = inst.get("tradingsymbol", "")
                if symbol:
                    key = f"{exchange}:{symbol}"
                    quote_keys.append(key)
                    token_to_key[key] = t

        if not quote_keys:
            return

        BATCH_SIZE = 200  # Kite quote API limit
        for i in range(0, len(quote_keys), BATCH_SIZE):
            batch = quote_keys[i:i + BATCH_SIZE]
            try:
                quotes = await self._client.get_quote(batch)
                snap_batch: list[dict] = []
                for key, q in quotes.items():
                    token = token_to_key.get(key, q.instrument_token)
                    if token and (q.oi or q.oi_day_high):
                        self._oi_tracker.update_from_tick({
                            "instrument_token": token,
                            "last_price": q.last_price,
                            "volume_traded": q.volume,
                            "change": q.net_change,
                            "oi": q.oi or 0,
                            "oi_day_high": q.oi_day_high or 0,
                            "oi_day_low": q.oi_day_low or 0,
                        })
                    # Store quote snapshot
                    snap_batch.append({
                        "instrument_token": token or q.instrument_token,
                        "ts": str(q.timestamp) if q.timestamp else datetime.now().isoformat(),
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
                    })
                if self._data_store and snap_batch:
                    try:
                        self._data_store.store_quote_snapshots(snap_batch)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning("oi_rest_poll_batch_error", error=str(e), batch_start=i)
                await asyncio.sleep(0.5)
            # Small delay between batches to avoid rate-limiting
            await asyncio.sleep(0.3)

    async def _oi_rest_poll_loop(self, tokens: list[int]) -> None:
        """Periodically fetch OI data via REST as fallback/supplement to ticker."""
        poll_interval = 30  # seconds between REST polls
        while self._oi_streaming:
            await asyncio.sleep(poll_interval)
            try:
                # Re-fetch spot prices periodically
                try:
                    ltp_data = await self._client.get_ltp(["NSE:NIFTY 50"])
                    for k, v in ltp_data.items():
                        if v.last_price > 0:
                            self._oi_tracker.set_spot_price("NIFTY", v.last_price)
                except Exception:
                    pass
                await self._oi_rest_poll_once(tokens)
                logger.debug("oi_rest_poll_complete", tokens=len(tokens))
            except Exception as e:
                logger.error("oi_rest_poll_error", error=str(e))

    async def _oi_broadcast_loop(self) -> None:
        while self._oi_streaming:
            try:
                data = self._oi_tracker.get_all_tracked_data()
                await self._oi_tracker.broadcast_oi_update(data)
            except Exception as e:
                logger.error("oi_broadcast_error", error=str(e))
            await asyncio.sleep(2)

    # ── OI Strategy Methods ───────────────────────────────────

    async def oi_strategy_scan(self, underlying: str = "NIFTY") -> dict[str, Any]:
        """Run OI strategy scan: fetch options OI → detect trading signals."""
        try:
            report = await self._options_oi.analyze(underlying, self._client)
            result = self._oi_strategy.scan_for_signals(report)

            # Enrich signals with live LTP from Kite API
            if result.signals:
                await self._enrich_oi_signals_with_live_prices(result.signals)

            # Auto-execute if configured
            if self._oi_strategy.config.auto_execute and result.signals:
                for sig in result.signals:
                    if sig.confidence >= 65:
                        pos = self._oi_strategy.create_position_from_signal(sig)
                        # Record in journal
                        self._record_oi_journal_entry(pos, "ENTRY")
            return result.model_dump()
        except Exception as e:
            logger.error("oi_strategy_scan_error", underlying=underlying, error=str(e))
            return {"error": str(e)}

    async def _enrich_oi_signals_with_live_prices(self, signals: list) -> None:
        """Fetch live LTP for OI strategy signals and update entry_price/SL/target."""
        try:
            # Collect tradingsymbols that need live price lookup
            symbols_to_fetch = []
            for sig in signals:
                if sig.tradingsymbol and sig.confidence > 0:
                    exchange = sig.exchange or "NFO"
                    symbols_to_fetch.append(f"{exchange}:{sig.tradingsymbol}")

            if not symbols_to_fetch:
                return

            # Fetch live quotes in batches
            quotes = await self._client.get_quote(symbols_to_fetch)

            # Store-behind: persist quote snapshots
            if self._data_store and quotes:
                try:
                    snap_batch = [
                        {
                            "instrument_token": q.instrument_token,
                            "ts": str(q.timestamp) if q.timestamp else datetime.now().isoformat(),
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
                    self._data_store.store_quote_snapshots(snap_batch)
                except Exception:
                    pass

            sl_pct = self._oi_strategy.config.stop_loss_pct / 100
            tgt_pct = self._oi_strategy.config.target_pct / 100

            for sig in signals:
                if sig.confidence <= 0 or not sig.tradingsymbol:
                    continue

                exchange = sig.exchange or "NFO"
                key = f"{exchange}:{sig.tradingsymbol}"
                q = quotes.get(key)
                if not q:
                    continue

                live_price = q.last_price if hasattr(q, "last_price") else 0.0
                if live_price <= 0:
                    continue

                # Update entry price with live LTP
                old_price = sig.entry_price
                sig.entry_price = round(live_price, 2)

                # Recalculate SL and target based on live price
                if sig.action in ("BUY_CE", "BUY_PE", "BUY_STRADDLE"):
                    sig.stop_loss = round(live_price * (1 - sl_pct), 2)
                    sig.target = round(live_price * (1 + tgt_pct), 2)
                else:
                    sig.stop_loss = round(live_price * (1 + sl_pct), 2)
                    sig.target = round(live_price * (1 - tgt_pct), 2)

                if old_price != sig.entry_price:
                    sig.metadata["oi_report_price"] = old_price
                    sig.metadata["live_price_updated"] = True

                logger.info(
                    "oi_signal_price_enriched",
                    symbol=sig.tradingsymbol,
                    old_price=old_price,
                    live_price=live_price,
                )

        except Exception as e:
            logger.warning("oi_signal_price_enrichment_failed", error=str(e))

    async def oi_strategy_scan_both(self) -> dict[str, Any]:
        """Scan both NIFTY and SENSEX for OI signals."""
        nifty = await self.oi_strategy_scan("NIFTY")
        sensex = await self.oi_strategy_scan("SENSEX")
        return {"nifty": nifty, "sensex": sensex}

    def oi_strategy_execute_signal(self, signal_id: str) -> dict[str, Any]:
        """Execute a specific OI signal by creating a position."""
        sig = next((s for s in self._oi_strategy._signals_history if s.id == signal_id), None)
        if not sig:
            return {"error": f"Signal {signal_id} not found"}
        if len(self._oi_strategy.get_open_positions()) >= self._oi_strategy.config.max_open_positions:
            return {"error": "Max open positions reached"}
        pos = self._oi_strategy.create_position_from_signal(sig)
        self._record_oi_journal_entry(pos, "ENTRY")
        return {"success": True, "position": pos.model_dump()}

    def oi_strategy_close_position(self, position_id: str, exit_price: float = 0.0) -> dict[str, Any]:
        """Close an OI strategy position."""
        if exit_price <= 0:
            return {"error": "exit_price must be greater than 0"}
        pos = self._oi_strategy.close_position(position_id, exit_price, "Manual close")
        if not pos:
            return {"error": f"Position {position_id} not found or not open"}
        self._record_oi_journal_entry(pos, "EXIT")
        return {"success": True, "position": pos.model_dump()}

    def oi_strategy_get_positions(self, underlying: Optional[str] = None) -> dict[str, Any]:
        """Get open and recent closed positions."""
        open_pos = self._oi_strategy.get_open_positions(underlying)
        closed_pos = self._oi_strategy.get_closed_positions(underlying, limit=30)
        return {
            "open": [p.model_dump() for p in open_pos],
            "closed": [p.model_dump() for p in closed_pos],
            "open_count": len(open_pos),
            "closed_count": len(closed_pos),
        }

    def oi_strategy_get_signals(self, underlying: Optional[str] = None, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent signal history."""
        signals = self._oi_strategy.get_signal_history(underlying, limit)
        return [s.model_dump() for s in signals]

    def oi_strategy_get_summary(self) -> dict[str, Any]:
        """Get overall OI strategy performance summary."""
        return self._oi_strategy.get_strategy_summary()

    def oi_strategy_update_config(self, updates: dict[str, Any]) -> dict[str, Any]:
        """Update OI strategy configuration."""
        cfg = self._oi_strategy.update_config(updates)
        return cfg.model_dump()

    def oi_strategy_get_config(self) -> dict[str, Any]:
        """Get current OI strategy configuration."""
        return self._oi_strategy.config.model_dump()

    def _record_oi_journal_entry(self, position: "OIPosition", action: str) -> None:
        """Record OI strategy entry/exit in the journal system."""
        try:
            meta = self._oi_strategy.get_journal_metadata(position)
            pnl = position.pnl if action == "EXIT" else 0.0
            txn = "BUY" if position.direction == "BUY" else "SELL"
            if action == "EXIT":
                txn = "SELL" if position.direction == "BUY" else "BUY"
            self._journal.record_trade(
                strategy=f"oi_strategy_{position.signal_type}",
                tradingsymbol=position.tradingsymbol or f"{position.underlying}_{position.option_type}_{position.strike}",
                exchange=position.exchange,
                transaction_type=txn,
                quantity=position.quantity,
                price=position.entry_price if action == "ENTRY" else position.exit_price,
                order_id=position.id,
                status="COMPLETE",
                pnl=pnl,
                notes=f"OI Strategy: {position.signal_type} | {action}",
                metadata=meta,
            )
            # Also record in JournalStore (pro journal)
            if action == "EXIT":
                entry = JournalEntry(
                    entry_id=f"oi-{position.id}",
                    trade_type="fno",
                    instrument=position.tradingsymbol or f"{position.underlying}_{position.option_type}_{position.strike}",
                    tradingsymbol=position.tradingsymbol or "",
                    exchange=position.exchange,
                    strategy_name=f"oi_strategy_{position.signal_type}",
                    direction=position.direction,
                    is_closed=True,
                    gross_pnl=position.pnl,
                    net_pnl=position.pnl,
                    entry_time=position.entry_time,
                    exit_time=position.exit_time or "",
                    tags=["oi_strategy", position.signal_type, position.underlying],
                    notes=f"OI Signal: {position.signal_type}",
                    metadata=meta,
                )
                self._journal_store.record_entry(entry)
        except Exception as e:
            logger.error("oi_journal_record_error", error=str(e))

    # Valid timeframe choices for live trading
    VALID_TIMEFRAMES = [
        "minute", "3minute", "5minute", "10minute",
        "15minute", "30minute", "60minute", "day",
    ]

    async def add_strategy(
        self, name: str, params: Optional[dict[str, Any]] = None,
        timeframe: str = "5minute", require_tested: bool = True,
    ) -> dict[str, Any]:
        from src.strategy.analysis_strategies import ANALYSIS_STRATEGIES

        # ── Validate timeframe ──────────────────────────────────
        if timeframe not in self.VALID_TIMEFRAMES:
            return {
                "error": f"Invalid timeframe: {timeframe}",
                "valid_timeframes": self.VALID_TIMEFRAMES,
            }


        # ── Backtest / paper-trade + health benchmark gate ───────────────
        if require_tested:
            if name not in self._tested_strategies:
                return {
                    "error": f"Strategy '{name}' has not been backtested or paper-traded yet.",
                    "hint": "Run a backtest or paper trade for this strategy first, then add it for live trading.",
                    "tested_strategies": sorted(self._tested_strategies),
                }
            # Enforce health benchmark: must have execution_ready==True (score >= 60)
            health = self.get_last_health_report()
            if not health or not health.get("execution_ready", False):
                score = health.get("overall_score") if health else None
                return {
                    "error": f"Strategy '{name}' does not meet the minimum health benchmark (score >= 60 required).",
                    "score": score,
                    "hint": "Improve backtest/paper-trade results until health score is at least 60.",
                }

        strategy_map = {
            "ema_crossover": EMACrossoverStrategy,
            "vwap_breakout": VWAPBreakoutStrategy,
            "mean_reversion": MeanReversionStrategy,
            "rsi": RSIStrategy,
            "rsi_strategy": RSIStrategy,
            "scanner_strategy": ScannerStrategy,
            "iron_condor": IronCondorStrategy,
            "straddle_strangle": StraddleStrangleStrategy,
            "bull_call_spread": BullCallSpreadStrategy,
            "bear_put_spread": BearPutSpreadStrategy,
            "call_credit_spread_runner": CallCreditSpreadRunnerStrategy,
            "put_credit_spread_runner": PutCreditSpreadRunnerStrategy,
        }
        # Merge all analysis strategies (cci_extreme, mfi_strategy, bollinger, etc.)
        strategy_map.update(ANALYSIS_STRATEGIES)
        cls = strategy_map.get(name)
        if cls:
            strategy = cls(params)
        else:
            # Check custom strategies
            custom = self._custom_store.build_strategy(name)
            if custom:
                strategy = custom
            else:
                return {"error": f"Unknown strategy: {name}", "available": list(strategy_map.keys())}

        # ── Apply timeframe ─────────────────────────────────────
        strategy.set_timeframe(timeframe)
        logger.info("strategy_timeframe_set", name=name, timeframe=timeframe)

        # Mark as live strategy (for UI tracking/removal)
        setattr(strategy, "is_live", True)
        self._strategies.append(strategy)
        logger.info("strategy_added", name=name, timeframe=timeframe)

        # Push the current token→symbol map so signals use real symbols
        self._sync_tradingsymbol_map()

        # If live trading is already running, seed bar data for the new strategy
        if self._running and self._subscribed_tokens:
            try:
                for token in self._subscribed_tokens:
                    df = await self._market_data.get_historical_df(
                        instrument_token=token, interval="5minute", days=5
                    )
                    if not df.empty:
                        strategy.update_bar_data(token, df)
                logger.info("strategy_bar_data_seeded_mid_live", name=name)
            except Exception as e:
                logger.warning("strategy_bar_data_seed_failed_mid_live", name=name, error=str(e))

        return {"success": True, "name": strategy.name, "timeframe": timeframe}

    def remove_strategy(self, name: str) -> dict[str, Any]:
        self._strategies = [s for s in self._strategies if s.name != name]
        return {"success": True, "removed": name}

    def toggle_strategy(self, name: str) -> dict[str, Any]:
        for s in self._strategies:
            if s.name == name:
                if s.is_active:
                    s.deactivate()
                else:
                    s.activate()
                return {"name": name, "is_active": s.is_active}
        return {"error": f"Strategy {name} not found"}

    async def start_live_with_defaults(self, mode: str = "full") -> dict[str, Any]:
        """Start live data with default watchlist symbols (convenience method)."""
        default_symbols = [
            "NIFTY 50", "NIFTY BANK", "RELIANCE", "TCS", "HDFCBANK",
            "INFY", "ICICIBANK", "SBIN", "BHARTIARTL", "ITC",
        ]
        
        tokens = []
        for symbol in default_symbols:
            try:
                token = await self._market_data.resolve_token(symbol, "NSE")
                if token:
                    tokens.append(token)
                    self._token_name_map[token] = symbol
            except Exception as e:
                logger.warning(f"Failed to resolve token for {symbol}: {e}")
        
        if not tokens:
            return {"error": "Could not resolve any default tokens"}
        
        return await self.start_live(tokens, mode)

    async def start_live(self, tokens: list[int], mode: str = "quote") -> dict[str, Any]:
        if self._running:
            return {"error": "Already running"}

        if not self._strategies:
            logger.warning("start_live_no_strategies", msg="No strategies registered. Add at least one strategy before starting live trading.")
            return {
                "error": "No strategies added. Please add at least one strategy before starting live trading.",
                "hint": "Click '+ Strategy' to add a strategy first.",
            }

        self._running = True
        self._subscribed_tokens = list(tokens)  # copy so we can extend

        # ── Auto-add NIFTY / SENSEX futures for options strategies ──
        options_strat_names = {
            "iron_condor", "straddle_strangle", "bull_call_spread", "bear_put_spread",
            "call_credit_spread_runner", "put_credit_spread_runner",
        }
        has_options = any(s.name in options_strat_names for s in self._strategies)
        if has_options:
            futures_tokens = await self._resolve_index_futures_tokens()
            for ft, fn in futures_tokens:
                if ft not in self._subscribed_tokens:
                    self._subscribed_tokens.append(ft)
                    self._token_name_map[ft] = fn
                    logger.info("auto_subscribed_index_future", token=ft, name=fn)

        self._ticker.on_ticks = self._on_ticks
        self._ticker.on_connect = self._on_connect
        self._ticker.on_close = self._on_close
        self._ticker.on_error = self._on_error
        self._ticker.on_order_update = self._on_order_update

        await self._seed_bar_data(self._subscribed_tokens)
        # Inject token→symbol map into all strategies so signals use real trading symbols
        self._sync_tradingsymbol_map()
        await self._ticker.subscribe(self._subscribed_tokens, TickMode(mode))

        for coro in [
            self._ticker.connect(),
            self._execution.start_reconciliation_loop(),
            self._position_update_loop(),
            self._index_quote_enrichment_loop(),
            self._option_chain_refresh_loop(),
        ]:
            task = asyncio.create_task(coro)
            task.add_done_callback(self._task_exception_handler)

        strategy_names = [s.name for s in self._strategies]
        logger.info("live_trading_started", tokens=self._subscribed_tokens, strategies=strategy_names, count=len(strategy_names))
        return {"success": True, "tokens": self._subscribed_tokens, "strategies": strategy_names}

    async def stop_live(self) -> dict[str, Any]:
        self._running = False
        await self._execution.stop_reconciliation_loop()
        await self._ticker.disconnect()
        logger.info("live_trading_stopped")
        return {"success": True}

    async def _resolve_index_futures_tokens(self) -> list[tuple[int, str]]:
        """Resolve current-month NIFTY and SENSEX futures instrument tokens.

        Returns list of (token, tradingsymbol) tuples for the nearest expiry
        futures contracts. These are needed for options strategies that require
        the underlying futures price for hedging/Greeks.
        """
        results: list[tuple[int, str]] = []
        today = datetime.now().date()

        def _parse_expiry(raw: str | None):
            """Parse expiry string (YYYY-MM-DD) to date, return None on failure."""
            if not raw:
                return None
            try:
                return datetime.strptime(raw, "%Y-%m-%d").date()
            except (ValueError, TypeError):
                return None

        for underlying, exchange in [("NIFTY", "NFO"), ("SENSEX", "BFO")]:
            try:
                instruments = await self._market_data.get_instruments(exchange)
                # Find FUT instruments for this underlying
                fut_candidates = []
                for inst in instruments:
                    # Works with both dicts (from cache) and Pydantic models
                    _get = (lambda k, d="": inst.get(k, d)) if isinstance(inst, dict) else (lambda k, d="": getattr(inst, k, d))
                    exp_date = _parse_expiry(_get("expiry"))
                    if (
                        _get("name") == underlying
                        and _get("instrument_type") == "FUT"
                        and exp_date
                        and exp_date >= today
                    ):
                        fut_candidates.append((inst, exp_date))

                if fut_candidates:
                    # Pick nearest expiry future
                    nearest_inst, nearest_exp = min(fut_candidates, key=lambda t: t[1])
                    _g = (lambda k, d=0: nearest_inst.get(k, d)) if isinstance(nearest_inst, dict) else (lambda k, d=0: getattr(nearest_inst, k, d))
                    results.append((_g("instrument_token"), _g("tradingsymbol", "")))
                    logger.info(
                        "index_future_resolved",
                        underlying=underlying,
                        tradingsymbol=_g("tradingsymbol", ""),
                        token=_g("instrument_token"),
                        expiry=str(nearest_exp),
                    )
                else:
                    logger.warning("index_future_not_found", underlying=underlying)
            except Exception as e:
                logger.warning("index_future_resolve_failed", underlying=underlying, error=str(e))

        return results

    def register_ws_client(self, ws: Any) -> None:
        self._ws_clients.append(ws)

    def unregister_ws_client(self, ws: Any) -> None:
        if ws in self._ws_clients:
            self._ws_clients.remove(ws)

    async def _broadcast_ws(self, data: dict[str, Any]) -> None:
        dead: list[Any] = []
        for ws in self._ws_clients:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._ws_clients.remove(ws)

    async def _seed_bar_data(self, tokens: list[int]) -> None:
        for token in tokens:
            try:
                df = await self._market_data.get_historical_df(
                    instrument_token=token, interval="5minute", days=5
                )
                if not df.empty:
                    for strategy in self._strategies:
                        strategy.update_bar_data(token, df)
            except Exception as e:
                logger.warning("bar_data_seed_failed", token=token, error=str(e))

    def _sync_tradingsymbol_map(self) -> None:
        """Push the service's token→name map into every strategy's params.

        Strategies use ``params["tradingsymbol_map"][token]`` to resolve
        instrument tokens to real trading symbols (e.g. "RELIANCE").
        Without this, they fall back to "TOKEN_738561" which makes the
        execution layer's quote/liquidity lookups fail.
        """
        # Build the map with both int and str keys (some strategies use str keys)
        sym_map: dict = {}
        for token, name in self._token_name_map.items():
            sym_map[token] = name
            sym_map[str(token)] = name

        for strategy in self._strategies:
            strategy.params["tradingsymbol_map"] = sym_map

    async def _on_ticks(self, ticks: list[Tick]) -> None:
        for tick in ticks:
            is_index = (tick.instrument_token & 0xFF) == 9
            # For index tokens, WS doesn't send volume/buy/sell — preserve cached values
            prev = self._live_ticks.get(tick.instrument_token, {})
            tick_data = {
                "token": tick.instrument_token,
                "name": self._token_name_map.get(tick.instrument_token, ""),
                "ltp": tick.last_price,
                "volume": tick.volume_traded if (tick.volume_traded or not is_index) else prev.get("volume", 0),
                "high": tick.ohlc.high,
                "low": tick.ohlc.low,
                "open": tick.ohlc.open,
                "close": tick.ohlc.close,
                "change": tick.change,
                "buy_qty": tick.total_buy_quantity if (tick.total_buy_quantity or not is_index) else prev.get("buy_qty", 0),
                "sell_qty": tick.total_sell_quantity if (tick.total_sell_quantity or not is_index) else prev.get("sell_qty", 0),
            }
            self._live_ticks[tick.instrument_token] = tick_data

            if tick.instrument_token == NIFTY_TOKEN:
                self._oi_tracker.set_spot_price("NIFTY", tick.last_price)
            elif tick.instrument_token == SENSEX_TOKEN:
                self._oi_tracker.set_spot_price("SENSEX", tick.last_price)

            if tick.oi is not None or tick.oi_day_high is not None:
                self._oi_tracker.update_from_tick({
                    "instrument_token": tick.instrument_token,
                    "last_price": tick.last_price,
                    "volume_traded": tick.volume_traded,
                    "change": tick.change,
                    "oi": tick.oi or 0,
                    "oi_day_high": tick.oi_day_high or 0,
                    "oi_day_low": tick.oi_day_low or 0,
                })

            for strategy in self._strategies:
                df = strategy.get_bar_data(tick.instrument_token)
                if df is not None and not df.empty:
                    new_row = pd.DataFrame([{
                        "timestamp": str(tick.exchange_timestamp or ""),
                        "open": tick.ohlc.open if tick.ohlc.open else tick.last_price,
                        "high": max(tick.ohlc.high, tick.last_price) if tick.ohlc.high else tick.last_price,
                        "low": min(tick.ohlc.low, tick.last_price) if tick.ohlc.low else tick.last_price,
                        "close": tick.last_price,
                        "volume": tick.volume_traded,
                    }])
                    updated = pd.concat([df, new_row], ignore_index=True).tail(500)
                    strategy.update_bar_data(tick.instrument_token, updated)

        for strategy in self._strategies:
            if not strategy.is_active:
                continue
            try:
                signals = await strategy.on_tick(ticks)
                for sig in signals:
                    await self._process_signal(sig)
            except Exception as e:
                logger.error("strategy_tick_error", strategy=strategy.name, error=str(e))

        await self._broadcast_ws({
            "type": "ticks",
            "data": self._live_ticks,
        })

    async def _on_connect(self) -> None:
        logger.info("ticker_connected")
        await self._broadcast_ws({"type": "status", "data": {"connected": True}})
        # Record system event
        try:
            from src.journal.journal_models import SystemEvent
            self._journal_store.record_system_event(SystemEvent(
                timestamp=datetime.now().isoformat(),
                event_type="ws_connect",
                severity="info",
                description="WebSocket ticker connected",
            ))
        except Exception:
            pass

    async def _on_close(self, code: int, reason: str) -> None:
        logger.warning("ticker_closed", code=code, reason=reason)
        await self._broadcast_ws({"type": "status", "data": {"connected": False}})
        # Record system event
        try:
            from src.journal.journal_models import SystemEvent
            self._journal_store.record_system_event(SystemEvent(
                timestamp=datetime.now().isoformat(),
                event_type="ws_disconnect",
                severity="warning",
                description=f"WebSocket closed: code={code} reason={reason}",
                metadata={"code": code, "reason": reason},
            ))
        except Exception:
            pass

    async def _on_error(self, error: Exception) -> None:
        logger.error("ticker_error", error=str(error))
        # Record system event
        try:
            from src.journal.journal_models import SystemEvent
            self._journal_store.record_system_event(SystemEvent(
                timestamp=datetime.now().isoformat(),
                event_type="ws_error",
                severity="error",
                description=f"WebSocket error: {str(error)}",
            ))
        except Exception:
            pass

    async def _on_order_update(self, data: dict[str, Any]) -> None:
        self._journal.record_trade(
            strategy="live",
            tradingsymbol=data.get("tradingsymbol", ""),
            exchange=data.get("exchange", ""),
            transaction_type=data.get("transaction_type", ""),
            quantity=data.get("quantity", 0),
            price=data.get("average_price", 0),
            order_id=data.get("order_id", ""),
            status=data.get("status", ""),
        )
        # Also record in JournalStore as system event for audit trail
        try:
            from src.journal.journal_models import SystemEvent
            self._journal_store.record_system_event(SystemEvent(
                timestamp=datetime.now().isoformat(),
                event_type="order_update",
                severity="info",
                description=f"{data.get('tradingsymbol', '')} {data.get('transaction_type', '')} {data.get('status', '')} @ {data.get('average_price', 0)}",
                metadata=data,
            ))
        except Exception as e:
            logger.error("order_update_system_event_error", error=str(e))
        await self._broadcast_ws({"type": "order_update", "data": data})

    # ────────────────────────────────────────────────────────────
    # Execution → Journal Callbacks
    # ────────────────────────────────────────────────────────────

    async def _on_execution_fill(self, order_id: str, broker_order: Any, signal: Any) -> None:
        """Called by ExecutionEngine when an entry order fills.
        Records fill in both TradeJournal and JournalStore (pro journal).
        """
        fill_price = getattr(broker_order, "average_price", 0) or 0.0
        tradingsymbol = getattr(broker_order, "tradingsymbol", "") or (signal.tradingsymbol if signal else "")
        exchange = getattr(broker_order, "exchange", "") or (signal.exchange.value if signal and hasattr(signal.exchange, "value") else "")
        txn_type = getattr(broker_order, "transaction_type", "") or (signal.transaction_type.value if signal else "")
        qty = getattr(broker_order, "filled_quantity", 0) or (signal.quantity if signal else 0)
        strategy_name = signal.strategy_name if signal else "live"

        # 1. Record in simple TradeJournal (JSON file)
        self._journal.record_trade(
            strategy=strategy_name,
            tradingsymbol=tradingsymbol,
            exchange=exchange,
            transaction_type=txn_type,
            quantity=qty,
            price=fill_price,
            order_id=order_id,
            status="COMPLETE",
            metadata=signal.metadata if signal else {},
        )

        # 2. Record in JournalStore (SQLite pro journal)
        try:
            from src.journal.journal_models import (
                JournalEntry, ExecutionRecord, StrategyContext, TradeDirection
            )
            direction = TradeDirection.LONG if txn_type == "BUY" else TradeDirection.SHORT
            exec_rec = ExecutionRecord(
                trade_id=order_id,
                instrument=tradingsymbol,
                tradingsymbol=tradingsymbol,
                exchange=exchange,
                direction=txn_type,
                quantity=qty,
                actual_fill_price=fill_price,
                expected_entry_price=signal.price if signal and signal.price else fill_price,
                order_type=signal.order_type.value if signal and hasattr(signal.order_type, "value") else "MARKET",
            )
            strat_ctx = StrategyContext(
                signal_confidence=signal.confidence if signal else 0,
            )
            entry = JournalEntry(
                trade_id=order_id,
                instrument=tradingsymbol,
                tradingsymbol=tradingsymbol,
                exchange=exchange,
                direction=direction.value if hasattr(direction, 'value') else str(direction),
                strategy_name=strategy_name,
                source="execution_engine",
                entry_time=datetime.now().isoformat(),
                execution=exec_rec.to_dict(),
                strategy_context=strat_ctx.to_dict(),
                tags=["auto", strategy_name],
            )
            self._journal_store.record_entry(entry)
            logger.info("journal_entry_recorded_on_fill", order_id=order_id, symbol=tradingsymbol)
        except Exception as e:
            logger.error("journal_store_fill_error", error=str(e), order_id=order_id)

        # 3. Notify portfolio health tracker
        try:
            trade_value = fill_price * qty
            self._portfolio_health.on_trade_open(
                trade_value=trade_value,
                symbol=tradingsymbol,
                strategy=strategy_name,
            )
        except Exception as e:
            logger.error("portfolio_health_open_error", error=str(e))

        logger.info(
            "execution_fill_journaled",
            order_id=order_id,
            symbol=tradingsymbol,
            price=fill_price,
            strategy=strategy_name,
        )

    async def _on_execution_exit(self, tracked: Any, exit_order_id: str) -> None:
        """Called by ExecutionEngine when a position exits (SL, trailing, MIS squareoff, manual).
        Records the exit in TradeJournal and updates JournalStore.
        """
        exit_reason = getattr(tracked, "exit_reason", "unknown")
        
        # Record in simple TradeJournal
        exit_txn = "SELL" if tracked.is_long else "BUY"
        self._journal.record_trade(
            strategy=tracked.strategy_name,
            tradingsymbol=tracked.tradingsymbol,
            exchange=tracked.exchange,
            transaction_type=exit_txn,
            quantity=tracked.quantity,
            price=tracked.exit_price or 0.0,
            order_id=exit_order_id or "",
            status="EXIT",
            pnl=tracked.pnl,
            notes=f"Exit reason: {exit_reason}",
        )

        # Close the entry in JournalStore (pro journal)
        try:
            entry_order_id = getattr(tracked, "order_id", "")
            if entry_order_id:
                # Try to find and close the matching open entry
                open_entries = self._journal_store.query_entries(
                    is_closed=False, instrument=tracked.tradingsymbol, limit=10
                )
                for oe in open_entries:
                    if oe.trade_id == entry_order_id:
                        self._journal_store.close_entry(
                            entry_id=oe.entry_id,
                            net_pnl=tracked.pnl,
                            gross_pnl=tracked.pnl,
                            total_costs=0.0,
                            exit_time=datetime.now().isoformat(),
                            exit_execution={
                                "actual_exit_price": tracked.exit_price or 0.0,
                                "exit_order_id": exit_order_id,
                            },
                        )
                        logger.info("journal_entry_closed_on_exit", entry_id=oe.entry_id, pnl=tracked.pnl)
                        break
                else:
                    # No matching open entry found — create a closed entry directly
                    from src.journal.journal_models import JournalEntry, ExecutionRecord, TradeDirection
                    direction = TradeDirection.LONG if tracked.is_long else TradeDirection.SHORT
                    exec_rec = ExecutionRecord(
                        trade_id=entry_order_id or exit_order_id,
                        instrument=tracked.tradingsymbol,
                        exchange=tracked.exchange,
                        direction=direction.value,
                        quantity=tracked.quantity,
                        actual_fill_price=tracked.entry_price or 0.0,
                        actual_exit_price=tracked.exit_price or 0.0,
                        net_pnl=tracked.pnl,
                        gross_pnl=tracked.pnl,
                    )
                    je = JournalEntry(
                        trade_id=entry_order_id or exit_order_id,
                        instrument=tracked.tradingsymbol,
                        tradingsymbol=tracked.tradingsymbol,
                        exchange=tracked.exchange,
                        direction=direction.value,
                        strategy_name=tracked.strategy_name,
                        source="execution_engine",
                        is_closed=True,
                        net_pnl=tracked.pnl,
                        gross_pnl=tracked.pnl,
                        entry_time=getattr(tracked, "entry_time", "") or "",
                        exit_time=datetime.now().isoformat(),
                        execution=exec_rec.to_dict(),
                        tags=["live", tracked.strategy_name, exit_reason],
                        notes=f"Live exit: {exit_reason} | PnL: {tracked.pnl:.2f}",
                    )
                    self._journal_store.record_entry(je)
                    logger.info("journal_entry_created_on_exit", trade_id=je.trade_id, pnl=tracked.pnl)
        except Exception as e:
            logger.error("journal_store_exit_error", error=str(e), order_id=exit_order_id)

        # Record system event for significant exits
        try:
            from src.journal.journal_models import SystemEvent
            self._journal_store.record_system_event(SystemEvent(
                timestamp=datetime.now().isoformat(),
                event_type="trade_exit",
                severity="info",
                description=f"{tracked.tradingsymbol} {exit_reason} PnL: {tracked.pnl:.2f}",
                metadata={"symbol": tracked.tradingsymbol, "pnl": tracked.pnl, "reason": exit_reason},
            ))
        except Exception as e:
            logger.error("system_event_exit_error", error=str(e))

        # Update risk PnL
        try:
            self._risk.update_daily_pnl(tracked.pnl)
        except Exception as e:
            logger.error("risk_pnl_update_error", error=str(e))

        # Notify portfolio health tracker
        try:
            self._portfolio_health.on_trade_close(
                pnl=tracked.pnl,
                symbol=tracked.tradingsymbol,
                strategy=tracked.strategy_name,
            )
        except Exception as e:
            logger.error("portfolio_health_close_error", error=str(e))

        # Broadcast exit to dashboard
        await self._broadcast_ws({
            "type": "position_exit",
            "data": {
                "order_id": tracked.order_id,
                "symbol": tracked.tradingsymbol,
                "direction": "LONG" if tracked.is_long else "SHORT",
                "entry_price": tracked.entry_price,
                "exit_price": tracked.exit_price,
                "pnl": tracked.pnl,
                "reason": exit_reason,
                "exit_order_id": exit_order_id,
            },
        })

        logger.info(
            "execution_exit_journaled",
            symbol=tracked.tradingsymbol,
            reason=exit_reason,
            pnl=tracked.pnl,
            exit_order=exit_order_id,
        )

    async def _process_signal(self, signal: Signal) -> None:
        signal_data = {
            "timestamp": datetime.now().isoformat(),
            "strategy": signal.strategy_name,
            "tradingsymbol": signal.tradingsymbol,
            "transaction_type": signal.transaction_type.value,
            "quantity": signal.quantity,
            "price": signal.price,
            "confidence": signal.confidence,
            "metadata": signal.metadata,
        }
        self._signal_log.append(signal_data)
        await self._broadcast_ws({"type": "signal", "data": signal_data})

        try:
            order_id = await self._execution.execute_signal(signal)
            if order_id:
                self._journal.record_trade(
                    strategy=signal.strategy_name,
                    tradingsymbol=signal.tradingsymbol,
                    exchange=signal.exchange.value,
                    transaction_type=signal.transaction_type.value,
                    quantity=signal.quantity,
                    price=signal.price or 0.0,
                    order_id=order_id,
                    status="PLACED",
                    metadata=signal.metadata,
                )
        except KillSwitchError:
            logger.critical("kill_switch_triggered")
            for s in self._strategies:
                s.deactivate()
            # Record critical system event
            try:
                from src.journal.journal_models import SystemEvent
                self._journal_store.record_system_event(SystemEvent(
                    timestamp=datetime.now().isoformat(),
                    event_type="kill_switch",
                    severity="critical",
                    description=f"Kill switch triggered for signal: {signal.tradingsymbol}",
                    metadata={"symbol": signal.tradingsymbol, "strategy": signal.strategy_name},
                ))
            except Exception:
                pass
        except RiskLimitError as e:
            logger.warning("risk_limit_hit", error=str(e))
            # Record warning system event
            try:
                from src.journal.journal_models import SystemEvent
                self._journal_store.record_system_event(SystemEvent(
                    timestamp=datetime.now().isoformat(),
                    event_type="risk_limit_hit",
                    severity="warning",
                    description=f"Risk limit: {str(e)}",
                    metadata={"symbol": signal.tradingsymbol, "strategy": signal.strategy_name, "error": str(e)},
                ))
            except Exception:
                pass
        except Exception as e:
            logger.error("signal_processing_error", error=str(e))

    # ────────────────────────────────────────────────────────────
    # Paper Trading
    # ────────────────────────────────────────────────────────────

    async def run_paper_trade(
        self,
        strategy_name: str,
        instrument_token: int,
        tradingsymbol: str,
        exchange: str = "NSE",
        days: int = 60,
        interval: str = "5minute",
        capital: float = 100000.0,
        commission_pct: float = 0.03,
        slippage_pct: float = 0.05,
        strategy_params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Run paper trade with a built-in or custom strategy on historical data."""
        # Resolve strategy
        strategy = self._resolve_strategy(strategy_name, strategy_params)
        if isinstance(strategy, dict):  # error dict
            return strategy

        # Fetch historical data (through cache)
        try:
            df = await self._market_data.get_historical_df(
                instrument_token=instrument_token,
                interval=interval,
                days=days,
            )
        except Exception as e:
            logger.error("paper_trade_data_fetch_failed", error=str(e))
            return {"error": f"Failed to fetch data: {str(e)}"}

        if df.empty:
            return {"error": "No historical data returned. Check instrument token and date range."}

        # Run paper trade
        engine = PaperTradingEngine(
            strategy=strategy,
            initial_capital=capital,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
        )
        result = engine.run(
            data=df,
            instrument_token=instrument_token,
            tradingsymbol=tradingsymbol,
            exchange=exchange,
            timeframe=interval,
        )

        # BacktestEngine.run() returns a plain dict (not an object with to_dict_safe)
        result_dict = result if isinstance(result, dict) else result.to_dict_safe()
        self._paper_results[strategy_name] = result_dict
        # Mark strategy as tested so it can be added for live trading
        self._mark_strategy_tested(strategy_name)
        logger.info("strategy_marked_tested", name=strategy_name, via="paper_trade")

        # Compute strategy health report and embed in result
        try:
            health = self.compute_strategy_health(result_dict, strategy_type="equity", strategy_name=strategy_name)
            result_dict["health_report"] = health
        except Exception as e:
            logger.error("health_compute_error", error=str(e))
            result_dict.setdefault("_warnings", []).append(f"Health computation failed: {e}")

        # Record paper trades to journal for analytics
        try:
            self._journal_record_paper_trades(result_dict, strategy_name, tradingsymbol)
        except Exception as e:
            logger.error("paper_trade_journal_record_error", error=str(e))
            result_dict.setdefault("_warnings", []).append(f"Journal recording failed: {e}")

        logger.info("paper_trade_complete", strategy=strategy_name, trades=result_dict.get("total_trades", len(result_dict.get("trades", []))))
        return result_dict

    async def run_paper_trade_with_sample_data(
        self,
        strategy_name: str,
        tradingsymbol: str = "SAMPLE",
        bars: int = 500,
        capital: float = 100000.0,
        interval: str = "5minute",
        strategy_params: Optional[dict[str, Any]] = None,
        allow_synthetic: bool = False,
    ) -> dict[str, Any]:
        """Run paper trade with sample data — Live → DB cache → Synthetic (with confirmation)."""
        strategy = self._resolve_strategy(strategy_name, strategy_params)
        if isinstance(strategy, dict):
            return strategy

        df, data_source, err = await self._resolve_sample_data(
            instrument_token=256265,
            interval=interval,
            bars=bars,
            allow_synthetic=allow_synthetic,
            base_price=100.0,
            style="equity",
        )
        if err:
            return {"error": err, "needs_synthetic_confirmation": True}

        symbol = "NIFTY 50" if data_source in ("zerodha", "db_cache") else tradingsymbol

        engine = PaperTradingEngine(strategy=strategy, initial_capital=capital)
        result = engine.run(df, instrument_token=0, tradingsymbol=symbol, timeframe=interval)
        # BacktestEngine.run() returns a plain dict (not an object with to_dict_safe)
        result_dict = result if isinstance(result, dict) else result.to_dict_safe()
        result_dict["data_source"] = data_source
        self._paper_results[strategy_name] = result_dict
        self._mark_strategy_tested(strategy_name)
        logger.info("strategy_marked_tested", name=strategy_name, via="paper_trade_sample")

        # Compute strategy health report and embed in result
        try:
            health = self.compute_strategy_health(result_dict, strategy_type="equity", strategy_name=strategy_name)
            result_dict["health_report"] = health
        except Exception as e:
            logger.error("health_compute_error", error=str(e))
            result_dict.setdefault("_warnings", []).append(f"Health computation failed: {e}")

        # Record paper trades to journal
        try:
            self._journal_record_paper_trades(result_dict, strategy_name, symbol)
        except Exception as e:
            logger.error("paper_trade_sample_journal_error", error=str(e))
            result_dict.setdefault("_warnings", []).append(f"Journal recording failed: {e}")

        return result_dict

    def get_paper_results(self) -> dict[str, Any]:
        return self._paper_results.copy()

    def _resolve_strategy(
        self, name: str, params: Optional[dict[str, Any]] = None,
    ) -> BaseStrategy | dict[str, Any]:
        """Get a strategy instance by name (built-in or custom)."""
        from src.strategy.analysis_strategies import ANALYSIS_STRATEGIES

        strategy_map = {
            "ema_crossover": EMACrossoverStrategy,
            "vwap_breakout": VWAPBreakoutStrategy,
            "mean_reversion": MeanReversionStrategy,
            "rsi": RSIStrategy,
            "rsi_strategy": RSIStrategy,
            "scanner_strategy": ScannerStrategy,
            "call_credit_spread_runner": CallCreditSpreadRunnerStrategy,
            "put_credit_spread_runner": PutCreditSpreadRunnerStrategy,
        }
        # Merge analysis strategies into the map
        strategy_map.update(ANALYSIS_STRATEGIES)

        if name in strategy_map:
            return strategy_map[name](params)

        # Check custom strategies
        custom = self._custom_store.build_strategy(name)
        if custom:
            return custom

        return {
            "error": f"Unknown strategy: {name}",
            "available": list(strategy_map.keys()) + [
                s["name"] for s in self._custom_store.list_strategies()
            ],
        }

    # ────────────────────────────────────────────────────────────
    # Custom Strategy Builder
    # ────────────────────────────────────────────────────────────

    def get_custom_strategy_store(self) -> CustomStrategyStore:
        return self._custom_store

    def save_custom_strategy(self, config: dict[str, Any]) -> dict[str, Any]:
        return self._custom_store.save_strategy(config)

    def delete_custom_strategy(self, name: str) -> dict[str, Any]:
        return self._custom_store.delete_strategy(name)

    def list_custom_strategies(self) -> list[dict[str, Any]]:
        return self._custom_store.list_strategies()

    def get_custom_strategy(self, name: str) -> Optional[dict[str, Any]]:
        return self._custom_store.get_strategy(name)

    def get_builder_indicators(self) -> list[dict[str, Any]]:
        return get_available_indicators()

    # ────────────────────────────────────────────────────────────
    # F&O Custom Strategy Builder
    # ────────────────────────────────────────────────────────────

    def save_fno_custom_strategy(self, config: dict[str, Any]) -> dict[str, Any]:
        cfg = FnOStrategyConfig.from_dict(config)
        if not cfg.name:
            return {"error": "Strategy name is required"}
        if not cfg.legs:
            return {"error": "At least one leg is required"}
        # Auto-set lot size
        cfg.lot_size = UNDERLYING_LOT_SIZES.get(cfg.underlying, 50)
        return self._fno_strategy_store.save(cfg)

    def delete_fno_custom_strategy(self, name: str) -> dict[str, Any]:
        return self._fno_strategy_store.delete(name)

    def list_fno_custom_strategies(self) -> list[dict[str, Any]]:
        return self._fno_strategy_store.list_all()

    def get_fno_custom_strategy(self, name: str) -> Optional[dict[str, Any]]:
        cfg = self._fno_strategy_store.get(name)
        return cfg.to_dict() if cfg else None

    def get_fno_builder_templates(self) -> list[dict[str, Any]]:
        return self._fno_strategy_store.get_templates()

    def compute_fno_payoff(self, config: dict[str, Any], spot_price: float) -> dict[str, Any]:
        """Compute payoff diagram for a custom F&O strategy."""
        cfg = FnOStrategyConfig.from_dict(config)
        cfg.lot_size = UNDERLYING_LOT_SIZES.get(cfg.underlying, 50)
        resolved = resolve_legs(cfg, spot_price, chain=None)
        return compute_strategy_payoff(resolved, spot_price, cfg.lot_size)

    def get_analysis_strategy_list(self) -> list[dict[str, Any]]:
        """Return list of all individual analysis strategies for UI dropdowns."""
        from src.strategy.analysis_strategies import ANALYSIS_STRATEGIES, ANALYSIS_STRATEGY_LABELS
        return [
            {"value": k, "label": v}
            for k, v in ANALYSIS_STRATEGY_LABELS.items()
        ]

    async def run_analysis_scan(self, symbols: Optional[list[str]] = None,
                                universe: str = "nifty50") -> dict[str, Any]:
        return await self._scanner.run_scan(self._client, symbols, universe=universe)

    def get_analysis_results(self) -> dict[str, Any]:
        return self._scanner.get_cached_results()

    def get_scanner_status(self) -> dict[str, Any]:
        return {
            "scanning": self._scanner.is_scanning,
            "last_scan_time": self._scanner.last_scan_time,
            "has_data": bool(self._scanner._analysis_cache),
        }

    def get_deep_profile(self, symbol: str) -> dict[str, Any]:
        """Get deep analysis profile for a single stock."""
        return self._scanner.get_deep_profile(symbol)

    async def analyze_stock_on_demand(self, symbol: str) -> dict[str, Any]:
        """Fetch data & run full analysis for any NSE stock on-demand."""
        return await self._scanner.analyze_stock_on_demand(self._client, symbol)

    def get_sector_analysis(self) -> dict[str, Any]:
        """Get sector-level analysis from cached scan data."""
        return self._scanner.get_sector_analysis()

    # ────────────────────────────────────────────────────────────
    # Historical Chart Data
    # ────────────────────────────────────────────────────────────

    async def get_historical_chart(
        self, instrument_token: int, interval: str = "day", days: int = 365,
        from_date: str | None = None, to_date: str | None = None,
        include_indicators: bool = False,
    ) -> dict[str, Any]:
        return await self._market_data.get_historical_chart(
            instrument_token, interval, days, from_date, to_date, include_indicators
        )

    async def get_historical_df(
        self, instrument_token: int, interval: str = "day", days: int = 365,
        from_date: str | None = None, to_date: str | None = None,
    ) -> Any:
        return await self._market_data.get_historical_df(
            instrument_token, interval, days, from_date, to_date
        )

    # ────────────────────────────────────────────────────────────
    # Live Market Quotes
    # ────────────────────────────────────────────────────────────

    async def get_market_quote(self, instruments: list[str]) -> dict[str, Any]:
        return await self._market_data.get_quote(instruments)

    async def get_market_ltp(self, instruments: list[str]) -> dict[str, Any]:
        return await self._market_data.get_ltp(instruments)

    async def get_market_ohlc(self, instruments: list[str]) -> dict[str, Any]:
        return await self._market_data.get_ohlc(instruments)

    async def get_market_overview(self, symbols: list[str] | None = None) -> dict[str, Any]:
        return await self._market_data.get_market_overview(symbols)

    async def resolve_instrument_token(self, symbol: str, exchange: str = "NSE") -> int | None:
        return await self._market_data.resolve_token(symbol, exchange)

    # ────────────────────────────────────────────────────────────
    # Portfolio Reports
    # ────────────────────────────────────────────────────────────

    async def get_holdings_report(self) -> dict[str, Any]:
        return await self._portfolio_reporter.get_holdings_report()

    async def get_positions_report(self) -> dict[str, Any]:
        return await self._portfolio_reporter.get_positions_report()

    async def get_trades_report(self) -> dict[str, Any]:
        return await self._portfolio_reporter.get_trades_report()

    async def get_orders_report(self) -> dict[str, Any]:
        return await self._portfolio_reporter.get_orders_report()

    async def get_pnl_report(self) -> dict[str, Any]:
        return await self._portfolio_reporter.get_pnl_report()

    async def get_margins_report(self) -> dict[str, Any]:
        return await self._portfolio_reporter.get_margins_report()

    # ────────────────────────────────────────────────────────────
    # Advanced Backtesting
    # ────────────────────────────────────────────────────────────

    def _journal_record_backtest_trades(
        self, result: dict[str, Any], strategy_name: str, tradingsymbol: str
    ) -> None:
        """Record backtest round-trip trades into JournalStore for analytics.
        
        Each EXIT trade now contains entry_price, entry_time, quantity, direction
        as self-contained round-trip info. We still also support ENTRY/EXIT pairing
        as a fallback.
        """
        trades = result.get("trades", [])
        if not trades:
            return

        from src.journal.journal_models import JournalEntry, ExecutionRecord, TradeDirection

        recorded = 0
        pending_entry: dict[str, Any] | None = None

        for t in trades:
            ttype = t.get("type", "")

            if ttype.endswith("_ENTRY"):
                pending_entry = t
                continue

            # Process EXIT trades (LONG_EXIT, SHORT_EXIT, SL_LONG_EXIT, SL_SHORT_EXIT, FINAL_EXIT)
            if "EXIT" not in ttype:
                continue

            # Get entry info from the exit trade itself (self-contained) or from pending_entry
            entry_price = t.get("entry_price", 0) or (pending_entry.get("entry_price", 0) if pending_entry else 0)
            entry_time = t.get("entry_time", "") or (pending_entry.get("entry_time", "") if pending_entry else "")
            qty = t.get("quantity", 0) or (pending_entry.get("quantity", 1) if pending_entry else 1)
            direction_str = t.get("direction", "")
            if not direction_str:
                if pending_entry:
                    direction_str = "LONG" if "LONG" in pending_entry.get("type", "") else "SHORT"
                elif "LONG" in ttype:
                    direction_str = "LONG"
                elif "SHORT" in ttype:
                    direction_str = "SHORT"
                else:
                    direction_str = "LONG"  # default fallback

            exit_price = t.get("exit_price", 0)
            pnl = t.get("pnl", 0)
            costs = t.get("costs", 0)

            exec_rec = ExecutionRecord(
                trade_id=f"BT-{strategy_name}-{recorded}",
                instrument=tradingsymbol,
                exchange="NSE",
                direction=direction_str,
                quantity=qty,
                actual_fill_price=entry_price,
                actual_exit_price=exit_price,
                expected_entry_price=entry_price,
                net_pnl=pnl,
                gross_pnl=pnl + costs,
            )
            entry = JournalEntry(
                trade_id=f"BT-{strategy_name}-{recorded}",
                instrument=tradingsymbol,
                tradingsymbol=tradingsymbol,
                exchange="NSE",
                direction=direction_str,
                is_closed=True,
                net_pnl=pnl,
                gross_pnl=pnl + costs,
                total_costs=costs,
                strategy_name=strategy_name,
                source="backtest",
                entry_time=entry_time,
                exit_time=t.get("exit_time", ""),
                mae=t.get("mae_pct", 0),
                mfe=t.get("mfe_pct", 0),
                execution=exec_rec.to_dict(),
                tags=["backtest", strategy_name, ttype.replace("_EXIT", "").replace("SL_", "sl_").lower()],
                notes=f"Backtest {strategy_name} on {tradingsymbol} | Exit: {ttype}",
            )
            try:
                self._journal_store.record_entry(entry)
                recorded += 1
            except Exception as exc:
                logger.error("backtest_journal_entry_error", error=str(exc), trade=recorded)
            pending_entry = None

        if recorded > 0:
            logger.info("backtest_trades_journaled", count=recorded, strategy=strategy_name)

    def _journal_record_paper_trades(
        self, result: dict[str, Any], strategy_name: str, tradingsymbol: str
    ) -> None:
        """Record paper trade round-trips into JournalStore for analytics.

        Paper trading uses BacktestEngine under the hood, so result format
        is the same: result["trades"] contains alternating ENTRY / EXIT dicts.
        Each EXIT trade now contains self-contained entry info.
        """
        trades = result.get("trades", [])
        if not trades:
            return

        from src.journal.journal_models import JournalEntry, ExecutionRecord, TradeDirection

        recorded = 0
        pending_entry: dict[str, Any] | None = None

        for t in trades:
            ttype = t.get("type", "")

            if ttype.endswith("_ENTRY"):
                pending_entry = t
                continue

            if "EXIT" not in ttype:
                continue

            # Get entry info from the exit trade itself (self-contained) or from pending_entry
            entry_price = t.get("entry_price", 0) or (pending_entry.get("entry_price", 0) if pending_entry else 0)
            entry_time = t.get("entry_time", "") or (pending_entry.get("entry_time", "") if pending_entry else "")
            qty = t.get("quantity", 0) or (pending_entry.get("quantity", 1) if pending_entry else 1)
            direction_str = t.get("direction", "")
            if not direction_str:
                if pending_entry:
                    direction_str = "LONG" if "LONG" in pending_entry.get("type", "") else "SHORT"
                elif "LONG" in ttype:
                    direction_str = "LONG"
                elif "SHORT" in ttype:
                    direction_str = "SHORT"
                else:
                    direction_str = "LONG"

            exit_price = t.get("exit_price", 0)
            pnl = t.get("pnl", 0)
            costs = t.get("costs", 0)

            exec_rec = ExecutionRecord(
                trade_id=f"PT-{strategy_name}-{recorded}",
                instrument=tradingsymbol,
                exchange="NSE",
                direction=direction_str,
                quantity=qty,
                actual_fill_price=entry_price,
                actual_exit_price=exit_price,
                expected_entry_price=entry_price,
                net_pnl=pnl,
                gross_pnl=pnl + costs,
            )
            entry = JournalEntry(
                trade_id=f"PT-{strategy_name}-{recorded}",
                instrument=tradingsymbol,
                tradingsymbol=tradingsymbol,
                exchange="NSE",
                direction=direction_str,
                is_closed=True,
                net_pnl=pnl,
                gross_pnl=pnl + costs,
                total_costs=costs,
                strategy_name=strategy_name,
                source="paper_trade",
                entry_time=entry_time,
                exit_time=t.get("exit_time", ""),
                mae=t.get("mae_pct", 0),
                mfe=t.get("mfe_pct", 0),
                execution=exec_rec.to_dict(),
                tags=["paper_trade", strategy_name, ttype.replace("_EXIT", "").replace("SL_", "sl_").lower()],
                notes=f"Paper trade {strategy_name} on {tradingsymbol} | Exit: {ttype}",
            )
            try:
                self._journal_store.record_entry(entry)
                recorded += 1
            except Exception as exc:
                logger.error("paper_journal_entry_error", error=str(exc), trade=recorded)
            pending_entry = None

        if recorded > 0:
            logger.info("paper_trades_journaled", count=recorded, strategy=strategy_name)

    def _journal_record_fno_backtest_trades(
        self, result: dict[str, Any], strategy_name: str, underlying: str
    ) -> None:
        """Record F&O backtest trades into JournalStore.

        F&O trades have a different format from equity:
          ENTRY → {type: ENTRY, structure, legs, net_premium, costs, position_id, ...}
          EXIT  → {type: PROFIT_TARGET|STOP_LOSS|FINAL_EXIT|EXPIRY, pnl, costs, position_id, ...}
        We pair ENTRY with its corresponding EXIT by position_id for a round-trip.
        """
        trades = result.get("trades", [])
        if not trades:
            return

        from src.journal.journal_models import JournalEntry, ExecutionRecord

        # Group trades by position_id: pair entries with exits
        entries_by_pos: dict[str, dict] = {}
        exits_by_pos: dict[str, dict] = {}
        for t in trades:
            pid = t.get("position_id", "")
            if not pid:
                continue
            ttype = t.get("type", "")
            if ttype == "ENTRY":
                entries_by_pos[pid] = t
            elif ttype in ("PROFIT_TARGET", "STOP_LOSS", "FINAL_EXIT", "EXPIRY"):
                exits_by_pos[pid] = t

        recorded = 0
        structure = result.get("structure", strategy_name)
        for pid, entry_t in entries_by_pos.items():
            exit_t = exits_by_pos.get(pid)
            if not exit_t:
                continue  # position not closed yet — skip

            pnl = exit_t.get("pnl", 0)
            costs = entry_t.get("costs", 0) + exit_t.get("costs", 0)
            net_premium = entry_t.get("net_premium", 0)
            margin = entry_t.get("margin", 0)
            exit_type = exit_t.get("type", "")

            # Determine direction from net_premium: credit strategies are SHORT, debit are LONG
            direction = "SHORT" if net_premium > 0 else "LONG"

            exec_rec = ExecutionRecord(
                trade_id=f"FNO-BT-{strategy_name}-{recorded}",
                instrument=underlying,
                exchange="NFO",
                direction=direction,
                quantity=entry_t.get("legs", 1),
                actual_fill_price=abs(net_premium),
                actual_exit_price=0,
                expected_entry_price=abs(net_premium),
                net_pnl=pnl,
                gross_pnl=pnl + costs,
            )
            entry = JournalEntry(
                trade_id=f"FNO-BT-{strategy_name}-{recorded}",
                instrument=underlying,
                tradingsymbol=underlying,
                exchange="NFO",
                direction=direction,
                trade_type="fno",
                is_closed=True,
                net_pnl=pnl,
                gross_pnl=pnl + costs,
                total_costs=costs,
                strategy_name=f"fno_{strategy_name}",
                source="backtest",
                entry_time=entry_t.get("date", ""),
                exit_time=exit_t.get("date", ""),
                execution=exec_rec.to_dict(),
                position_structure={
                    "structure": structure,
                    "legs": entry_t.get("legs", 0),
                    "net_premium": net_premium,
                    "initial_margin": margin,
                    "spot_at_entry": entry_t.get("spot", 0),
                },
                return_on_margin=round(pnl / margin * 100, 2) if margin > 0 else 0,
                tags=["backtest", "fno", strategy_name, structure, exit_type.lower()],
                notes=f"F&O Backtest {structure} on {underlying} | {exit_type} | PnL: {pnl:.2f}",
            )
            try:
                self._journal_store.record_entry(entry)
                recorded += 1
            except Exception as exc:
                logger.error("fno_backtest_journal_entry_error", error=str(exc), trade=recorded)

        if recorded > 0:
            logger.info("fno_backtest_trades_journaled", count=recorded, strategy=strategy_name)

    def _journal_record_fno_paper_trades(
        self, result: dict[str, Any], strategy_name: str, underlying: str
    ) -> None:
        """Record F&O paper trade positions into JournalStore.

        F&O paper trade positions have format:
          {id, structure, legs, net_premium, pnl, entry, exit, regime}
        """
        positions = result.get("positions", [])
        if not positions:
            return

        from src.journal.journal_models import JournalEntry, ExecutionRecord

        recorded = 0
        for i, pos in enumerate(positions):
            pnl = pos.get("pnl", 0)
            net_premium = pos.get("net_premium", 0)
            structure = pos.get("structure", strategy_name)
            direction = "SHORT" if net_premium > 0 else "LONG"

            exec_rec = ExecutionRecord(
                trade_id=f"FNO-PT-{strategy_name}-{i}",
                instrument=underlying,
                exchange="NFO",
                direction=direction,
                quantity=pos.get("legs", 1),
                actual_fill_price=abs(net_premium),
                actual_exit_price=0,
                expected_entry_price=abs(net_premium),
                net_pnl=pnl,
                gross_pnl=pnl,
            )
            entry = JournalEntry(
                trade_id=f"FNO-PT-{strategy_name}-{i}",
                instrument=underlying,
                tradingsymbol=underlying,
                exchange="NFO",
                direction=direction,
                trade_type="fno",
                is_closed=True,
                net_pnl=pnl,
                gross_pnl=pnl,
                strategy_name=f"fno_{strategy_name}",
                source="paper_trade",
                entry_time=pos.get("entry", ""),
                exit_time=pos.get("exit", ""),
                execution=exec_rec.to_dict(),
                position_structure={
                    "structure": structure,
                    "legs": pos.get("legs", 0),
                    "net_premium": net_premium,
                    "regime_at_entry": pos.get("regime", ""),
                },
                tags=["paper_trade", "fno", strategy_name, structure],
                notes=f"F&O Paper {structure} on {underlying} | PnL: {pnl:.2f}",
            )
            try:
                self._journal_store.record_entry(entry)
                recorded += 1
            except Exception as exc:
                logger.error("fno_paper_journal_entry_error", error=str(exc), trade=i)

        if recorded > 0:
            logger.info("fno_paper_trades_journaled", count=recorded, strategy=strategy_name)

    async def run_backtest(
        self,
        strategy_name: str,
        instrument_token: int = 0,
        tradingsymbol: str = "BACKTEST",
        interval: str = "day",
        days: int = 365,
        from_date: str | None = None,
        to_date: str | None = None,
        initial_capital: float = 100_000.0,
        position_sizing: str = "fixed",
        risk_per_trade: float = 0.02,
        capital_fraction: float = 0.10,
        slippage_pct: float = 0.05,
        use_indian_costs: bool = True,
        is_intraday: bool = True,
        strategy_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run advanced backtest on historical Zerodha data."""
        strategy = self._resolve_strategy(strategy_name, strategy_params)
        if isinstance(strategy, dict):
            return strategy  # error dict

        # Fetch historical data
        df = await self._market_data.get_historical_df(
            instrument_token, interval, days, from_date, to_date
        )
        if df.empty:
            return {"error": "No historical data available. Check instrument token and date range."}

        engine = BacktestEngine(
            strategy=strategy,
            initial_capital=initial_capital,
            slippage_pct=slippage_pct,
            position_sizing=position_sizing,
            risk_per_trade=risk_per_trade,
            capital_fraction=capital_fraction,
            use_indian_costs=use_indian_costs,
            is_intraday=is_intraday,
        )

        result = engine.run(df, instrument_token, tradingsymbol, timeframe=interval)
        self._backtest_results = result
        self._mark_strategy_tested(strategy_name)
        logger.info("strategy_marked_tested", name=strategy_name, via="backtest")

        # Compute strategy health report and embed in result
        try:
            stype = "intraday" if is_intraday else "equity"
            health = self.compute_strategy_health(result, strategy_type=stype, strategy_name=strategy_name)
            result["health_report"] = health
        except Exception as e:
            logger.error("health_compute_error", error=str(e))
            result.setdefault("_warnings", []).append(f"Health computation failed: {e}")

        # Record backtest trades to journal for analysis
        try:
            self._journal_record_backtest_trades(result, strategy_name, tradingsymbol)
        except Exception as e:
            logger.error("backtest_journal_record_error", error=str(e))
            result.setdefault("_warnings", []).append(f"Journal recording failed: {e}")

        return result

    # Interval → pandas freq mapping for synthetic data generation
    _INTERVAL_FREQ: dict[str, str] = {
        "day": "D", "60minute": "60min", "30minute": "30min",
        "15minute": "15min", "10minute": "10min", "5minute": "5min",
        "3minute": "3min", "minute": "1min",
    }

    # Bars-per-day lookup used to estimate how many calendar days to fetch
    _BARS_PER_DAY: dict[str, int] = {
        "day": 1, "60minute": 6, "30minute": 13,
        "15minute": 25, "10minute": 38, "5minute": 75,
        "3minute": 125, "minute": 375,
    }

    async def _resolve_sample_data(
        self,
        *,
        instrument_token: int = 256265,
        interval: str = "day",
        bars: int = 500,
        allow_synthetic: bool = False,
        base_price: float = 100.0,
        style: str = "equity",
    ) -> tuple[pd.DataFrame, str, str | None]:
        """Centralised three-tier data resolver for sample / demo flows.

        Priority:
            1. **Live** – Zerodha historical via ``get_historical_df``
               (which itself checks the DB cache first, then the Kite API).
            2. **DB cache** – If not authenticated, query ``MarketDataStore``
               directly for any previously-cached candles.
            3. **Synthetic** – Only if ``allow_synthetic=True``; otherwise
               returns an empty DataFrame so the caller can signal the user.

        Returns
        -------
        (df, data_source, error_msg | None)
            *data_source* is one of ``"zerodha"``, ``"db_cache"``, ``"synthetic"``.
            *error_msg* is ``None`` on success; when non-``None`` the caller
            should return it as the API response asking the user to confirm
            synthetic usage.
        """
        df = pd.DataFrame()
        data_source = "synthetic"

        # ── Tier 1: Live / cached via MarketDataProvider ───────────
        if self._auth.is_authenticated:
            try:
                _bpd = self._BARS_PER_DAY.get(interval, 75)
                _days = max(int(bars / max(_bpd, 1)) + 10, 30)
                if interval == "day":
                    _days = max(bars * 2, 365)
                df = await self._market_data.get_historical_df(
                    instrument_token, interval, days=_days,
                )
                if not df.empty and len(df) >= bars:
                    df = df.tail(bars)
                    data_source = "zerodha"
                    return df, data_source, None
                df = pd.DataFrame()  # not enough rows
            except Exception as exc:
                logger.warning("resolve_data_live_fallback", error=str(exc))
                df = pd.DataFrame()

        # ── Tier 2: DB cache (even when not authenticated) ────────
        if df.empty and self._data_store:
            try:
                _bpd = self._BARS_PER_DAY.get(interval, 75)
                _days = max(int(bars / max(_bpd, 1)) + 10, 30)
                if interval == "day":
                    _days = max(bars * 2, 730)
                to_date = datetime.now().strftime("%Y-%m-%d")
                from_date = (datetime.now() - timedelta(days=_days)).strftime("%Y-%m-%d")
                cached = self._data_store.get_candles(
                    instrument_token, interval, from_date, to_date,
                )
                if cached and len(cached) >= bars:
                    rows = [{
                        "timestamp": c["ts"],
                        "open": c["open"],
                        "high": c["high"],
                        "low": c["low"],
                        "close": c["close"],
                        "volume": c.get("volume", 0),
                    } for c in cached]
                    df = pd.DataFrame(rows)
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df.set_index("timestamp", inplace=True)
                    df = df.tail(bars)
                    data_source = "db_cache"
                    return df, data_source, None
            except Exception as exc:
                logger.warning("resolve_data_db_cache_fallback", error=str(exc))

        # ── Tier 3: Synthetic (only if explicitly allowed) ────────
        if not allow_synthetic:
            return pd.DataFrame(), "none", (
                "No live or cached market data available. "
                "Set allow_synthetic=true to run with synthetic data."
            )

        freq = self._INTERVAL_FREQ.get(interval, "D")
        df = generate_synthetic_ohlcv(
            bars=bars, base_price=base_price, freq=freq, style=style,
        )
        data_source = "synthetic"
        return df, data_source, None

    async def run_backtest_sample(
        self,
        strategy_name: str,
        bars: int = 500,
        initial_capital: float = 100_000.0,
        position_sizing: str = "fixed",
        slippage_pct: float = 0.05,
        use_indian_costs: bool = True,
        is_intraday: bool = False,
        risk_per_trade: float = 0.02,
        capital_fraction: float = 0.10,
        interval: str = "day",
        strategy_params: dict[str, Any] | None = None,
        allow_synthetic: bool = False,
    ) -> dict[str, Any]:
        """Run backtest on sample data — Live → DB cache → Synthetic (with confirmation)."""
        strategy = self._resolve_strategy(strategy_name, strategy_params)
        if isinstance(strategy, dict):
            return strategy

        df, data_source, err = await self._resolve_sample_data(
            instrument_token=256265,
            interval=interval,
            bars=bars,
            allow_synthetic=allow_synthetic,
            base_price=100.0,
            style="equity",
        )
        if err:
            return {"error": err, "needs_synthetic_confirmation": True}

        tradingsymbol = "NIFTY 50" if data_source in ("zerodha", "db_cache") else "SAMPLE"

        engine = BacktestEngine(
            strategy=strategy, initial_capital=initial_capital,
            slippage_pct=slippage_pct, position_sizing=position_sizing,
            risk_per_trade=risk_per_trade, capital_fraction=capital_fraction,
            use_indian_costs=use_indian_costs, is_intraday=is_intraday,
        )
        result = engine.run(df, tradingsymbol=tradingsymbol, timeframe=interval)
        result["data_source"] = data_source
        self._backtest_results = result
        self._mark_strategy_tested(strategy_name)
        logger.info("strategy_marked_tested", name=strategy_name, via="backtest_sample")

        # Compute strategy health report and embed in result
        try:
            stype = "intraday" if is_intraday else "equity"
            health = self.compute_strategy_health(result, strategy_type=stype, strategy_name=strategy_name)
            result["health_report"] = health
        except Exception as e:
            logger.error("health_compute_error", error=str(e))
            result.setdefault("_warnings", []).append(f"Health computation failed: {e}")

        # Record backtest trades to journal for analysis
        try:
            self._journal_record_backtest_trades(result, strategy_name, tradingsymbol)
        except Exception as e:
            logger.error("backtest_sample_journal_record_error", error=str(e))
            result.setdefault("_warnings", []).append(f"Journal recording failed: {e}")

        return result

    def get_backtest_results(self) -> dict[str, Any]:
        return self._backtest_results.copy() if self._backtest_results else {"error": "No backtest results"}

    def compute_strategy_health(
        self,
        result: Optional[dict[str, Any]] = None,
        strategy_type: str = "",
        strategy_name: str = "",
    ) -> dict[str, Any]:
        """Compute strategy health report from backtest/paper-trade results.
        
        If result is None, uses the last cached backtest result.
        """
        if result is None:
            result = self._backtest_results
        if not result or "error" in result:
            return {"error": "No backtest/paper-trade results available. Run a backtest first."}

        # Auto-detect strategy type
        if not strategy_type:
            engine = result.get("engine", "")
            if "fno" in engine.lower():
                strategy_type = "fno"
            else:
                strategy_type = "equity"

        if not strategy_name:
            strategy_name = result.get("strategy_name", "unknown")

        report = compute_health_report(result, strategy_type=strategy_type, strategy_name=strategy_name)
        report_dict = report.to_dict()
        report_dict["grade"] = get_health_grade(report.overall_score)

        # Sanitize numpy types (numpy.bool_, numpy.float64 etc.) for JSON serialization
        report_dict = BacktestEngine._sanitize(report_dict)

        # Cache
        self._last_health_report = report_dict

        # Record health score in journal metadata
        try:
            self._record_health_to_journal(report, result)
        except Exception as e:
            logger.error("health_journal_record_error", error=str(e))

        return report_dict

    def get_last_health_report(self) -> dict[str, Any]:
        """Return the last computed health report."""
        if self._last_health_report:
            return self._last_health_report
        return {"error": "No health report computed yet. Run a backtest first."}

    def _record_health_to_journal(self, report: HealthReport, result: dict) -> None:
        """Store strategy health score as a journal metadata entry."""
        try:
            strategy_name = report.strategy_name
            entry = JournalEntry(
                entry_id=f"health-{strategy_name}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                trade_type=report.strategy_type or "equity",
                strategy_name=strategy_name,
                instrument=result.get("underlying", result.get("tradingsymbol", "")),
                source="health_report",
                notes=f"Health Score: {report.overall_score:.1f} ({report.overall_verdict})",
                tags=["health_report", report.overall_verdict.lower(), strategy_name],
                metadata={
                    "health_score": round(report.overall_score, 1),
                    "health_grade": get_health_grade(report.overall_score),
                    "health_verdict": report.overall_verdict,
                    "execution_ready": report.execution_ready,
                    "pillar_scores": {k: round(v.score, 1) for k, v in report.pillars.items()},
                    "pillar_verdicts": {k: v.verdict for k, v in report.pillars.items()},
                    "blockers": report.blockers,
                    "warnings": report.warnings,
                    "strategy_type": report.strategy_type,
                    "total_trades": result.get("total_trades", 0),
                    "expectancy": report.pillars.get("profitability", PillarResult(name="")).metrics.get("expectancy", 0),
                    "profit_factor": report.pillars.get("profitability", PillarResult(name="")).metrics.get("profit_factor", 0),
                    "sharpe_ratio": report.pillars.get("profitability", PillarResult(name="")).metrics.get("sharpe_ratio", 0),
                    "max_drawdown_pct": report.pillars.get("drawdown", PillarResult(name="")).metrics.get("max_drawdown_pct", 0),
                },
            )
            self._journal_store.record_entry(entry)
        except Exception as e:
            logger.error("health_journal_error", error=str(e))


    # ────────────────────────────────────────────────────────────
    # F&O Derivatives Backtesting & Paper Trading
    # ────────────────────────────────────────────────────────────

    FNO_STRATEGIES = [
        "iron_condor", "bull_call_spread", "bear_put_spread",
        "bull_put_spread", "bear_call_spread", "straddle",
        "short_straddle", "strangle", "short_strangle",
        "iron_butterfly", "covered_call", "protective_put",
        "calendar_spread",
        "call_credit_spread_runner", "put_credit_spread_runner",
    ]

    def _is_valid_fno_strategy(self, name: str) -> bool:
        """Check if strategy name is a built-in or saved custom F&O strategy."""
        if name.lower() in self.FNO_STRATEGIES:
            return True
        # Also accept saved custom strategies from the builder
        return self.get_fno_custom_strategy(name) is not None

    # Well-known index tokens for F&O backtesting
    INDEX_TOKENS: dict[str, tuple[int, str, str]] = {
        "NIFTY":     (256265, "NIFTY 50",  "NSE"),
        "NIFTY 50":  (256265, "NIFTY 50",  "NSE"),
        "BANKNIFTY": (260105, "NIFTY BANK", "NSE"),
        "SENSEX":    (265,    "SENSEX",    "BSE"),
        "FINNIFTY":  (257801, "NIFTY FIN SERVICE", "NSE"),
    }

    async def _resolve_underlying_token(self, underlying: str, instrument_token: int) -> int:
        """Resolve instrument_token for an underlying index."""
        if instrument_token and instrument_token > 0:
            return instrument_token
        # Try well-known index tokens first
        entry = self.INDEX_TOKENS.get(underlying.upper())
        if entry:
            return entry[0]
        # Fallback: server-side resolution
        token = await self.resolve_instrument_token(underlying, "NSE")
        if token:
            return token
        token = await self.resolve_instrument_token(underlying, "BSE")
        return token or 0

    async def run_fno_backtest(
        self,
        strategy_name: str = "iron_condor",
        underlying: str = "NIFTY",
        instrument_token: int = 0,
        interval: str = "day",
        days: int = 365,
        from_date: str | None = None,
        to_date: str | None = None,
        initial_capital: float = 500_000.0,
        max_positions: int = 3,
        profit_target_pct: float = 50.0,
        stop_loss_pct: float = 100.0,
        entry_dte_min: int = 15,
        entry_dte_max: int = 45,
        delta_target: float = 0.16,
        slippage_model: str = "realistic",
        use_regime_filter: bool = True,
    ) -> dict[str, Any]:
        """Run F&O derivatives backtest on underlying OHLCV data."""
        if not self._is_valid_fno_strategy(strategy_name):
            return {
                "error": f"Unknown F&O strategy: {strategy_name}",
                "available": self.FNO_STRATEGIES,
            }

        # Auto-resolve instrument token from underlying name
        instrument_token = await self._resolve_underlying_token(underlying, instrument_token)
        if not instrument_token:
            return {"error": f"Cannot resolve instrument token for '{underlying}'. Provide a valid underlying name."}

        # Fetch underlying data
        df = await self._market_data.get_historical_df(
            instrument_token, interval, days, from_date, to_date
        )
        if df.empty:
            return {"error": "No historical data. Check instrument token and date range."}

        engine = FnOBacktestEngine(
            strategy_name=strategy_name,
            underlying=underlying,
            initial_capital=initial_capital,
            max_positions=max_positions,
            profit_target_pct=profit_target_pct,
            stop_loss_pct=stop_loss_pct,
            entry_dte_min=entry_dte_min,
            entry_dte_max=entry_dte_max,
            delta_target=delta_target,
            slippage_model=slippage_model,
            use_regime_filter=use_regime_filter,
        )

        result = engine.run(df, tradingsymbol=underlying)
        self._backtest_results = result
        self._mark_strategy_tested(strategy_name)
        logger.info("strategy_marked_tested", name=strategy_name, via="fno_backtest")

        # Compute strategy health report and embed in result
        try:
            health = self.compute_strategy_health(result, strategy_type="fno", strategy_name=strategy_name)
            result["health_report"] = health
        except Exception as e:
            logger.error("health_compute_error", error=str(e))
            result.setdefault("_warnings", []).append(f"Health computation failed: {e}")

        # Record F&O backtest trades to journal
        try:
            self._journal_record_fno_backtest_trades(result, strategy_name, underlying)
        except Exception as e:
            logger.error("fno_backtest_journal_record_error", error=str(e))
            result.setdefault("_warnings", []).append(f"Journal recording failed: {e}")

        return result

    async def run_fno_backtest_sample(
        self,
        strategy_name: str = "iron_condor",
        underlying: str = "NIFTY",
        bars: int = 500,
        initial_capital: float = 500_000.0,
        max_positions: int = 3,
        profit_target_pct: float = 50.0,
        stop_loss_pct: float = 100.0,
        delta_target: float = 0.16,
        allow_synthetic: bool = False,
    ) -> dict[str, Any]:
        """Run F&O backtest — Live → DB cache → Synthetic (with confirmation)."""
        if not self._is_valid_fno_strategy(strategy_name):
            return {
                "error": f"Unknown F&O strategy: {strategy_name}",
                "available": self.FNO_STRATEGIES,
            }

        df, data_source, err = await self._resolve_sample_data(
            instrument_token=256265,
            interval="day",
            bars=bars,
            allow_synthetic=allow_synthetic,
            base_price=20000.0,
            style="index",
        )
        if err:
            return {"error": err, "needs_synthetic_confirmation": True}

        engine = FnOBacktestEngine(
            strategy_name=strategy_name,
            underlying=underlying,
            initial_capital=initial_capital,
            max_positions=max_positions,
            profit_target_pct=profit_target_pct,
            stop_loss_pct=stop_loss_pct,
            delta_target=delta_target,
        )
        result = engine.run(df, tradingsymbol=underlying)
        result["data_source"] = data_source
        self._backtest_results = result
        self._mark_strategy_tested(strategy_name)
        logger.info("strategy_marked_tested", name=strategy_name, via="fno_backtest_sample")

        # Compute strategy health report and embed in result
        try:
            health = self.compute_strategy_health(result, strategy_type="fno", strategy_name=strategy_name)
            result["health_report"] = health
        except Exception as e:
            logger.error("health_compute_error", error=str(e))
            result.setdefault("_warnings", []).append(f"Health computation failed: {e}")

        # Record F&O backtest sample trades to journal
        try:
            self._journal_record_fno_backtest_trades(result, strategy_name, underlying)
        except Exception as e:
            logger.error("fno_backtest_sample_journal_record_error", error=str(e))
            result.setdefault("_warnings", []).append(f"Journal recording failed: {e}")

        return result

    async def run_fno_paper_trade(
        self,
        strategy_name: str = "iron_condor",
        underlying: str = "NIFTY",
        instrument_token: int = 0,
        interval: str = "day",
        days: int = 60,
        initial_capital: float = 500_000.0,
        max_positions: int = 3,
        profit_target_pct: float = 50.0,
        stop_loss_pct: float = 100.0,
        entry_dte_min: int = 15,
        entry_dte_max: int = 45,
        delta_target: float = 0.16,
        slippage_model: str = "realistic",
    ) -> dict[str, Any]:
        """Run F&O paper trading simulation."""
        if not self._is_valid_fno_strategy(strategy_name):
            return {
                "error": f"Unknown F&O strategy: {strategy_name}",
                "available": self.FNO_STRATEGIES,
            }

        # Auto-resolve instrument token from underlying name
        instrument_token = await self._resolve_underlying_token(underlying, instrument_token)
        if not instrument_token:
            return {"error": f"Cannot resolve instrument token for '{underlying}'."}

        df = await self._market_data.get_historical_df(
            instrument_token, interval, days
        )
        if df.empty:
            return {"error": "No historical data. Check instrument token and date range."}

        engine = FnOPaperTradingEngine(
            strategy_name=strategy_name,
            underlying=underlying,
            initial_capital=initial_capital,
            max_positions=max_positions,
            profit_target_pct=profit_target_pct,
            stop_loss_pct=stop_loss_pct,
            entry_dte_min=entry_dte_min,
            entry_dte_max=entry_dte_max,
            delta_target=delta_target,
            slippage_model=slippage_model,
        )
        result = engine.run(df, tradingsymbol=underlying, timeframe=interval)
        result_dict = result.to_dict_safe()
        self._paper_results[f"fno_{strategy_name}"] = result_dict

        # Compute strategy health report and embed in result
        try:
            health = self.compute_strategy_health(result_dict, strategy_type="fno", strategy_name=strategy_name)
            result_dict["health_report"] = health
        except Exception as e:
            logger.error("health_compute_error", error=str(e))
            result_dict.setdefault("_warnings", []).append(f"Health computation failed: {e}")

        # Record F&O paper trades to journal
        try:
            self._journal_record_fno_paper_trades(result_dict, strategy_name, underlying)
        except Exception as e:
            logger.error("fno_paper_trade_journal_record_error", error=str(e))
            result_dict.setdefault("_warnings", []).append(f"Journal recording failed: {e}")

        return result_dict

    async def run_fno_paper_trade_sample(
        self,
        strategy_name: str = "iron_condor",
        underlying: str = "NIFTY",
        bars: int = 500,
        initial_capital: float = 500_000.0,
        max_positions: int = 3,
        profit_target_pct: float = 50.0,
        stop_loss_pct: float = 100.0,
        delta_target: float = 0.16,
        allow_synthetic: bool = False,
    ) -> dict[str, Any]:
        """Run F&O paper trade — Live → DB cache → Synthetic (with confirmation)."""
        if not self._is_valid_fno_strategy(strategy_name):
            return {
                "error": f"Unknown F&O strategy: {strategy_name}",
                "available": self.FNO_STRATEGIES,
            }

        df, data_source, err = await self._resolve_sample_data(
            instrument_token=256265,
            interval="day",
            bars=bars,
            allow_synthetic=allow_synthetic,
            base_price=20000.0,
            style="index",
        )
        if err:
            return {"error": err, "needs_synthetic_confirmation": True}

        engine = FnOPaperTradingEngine(
            strategy_name=strategy_name,
            underlying=underlying,
            initial_capital=initial_capital,
            max_positions=max_positions,
            profit_target_pct=profit_target_pct,
            stop_loss_pct=stop_loss_pct,
            delta_target=delta_target,
        )
        result = engine.run(df, tradingsymbol=underlying, timeframe="day")
        result_dict = result.to_dict_safe()
        result_dict["data_source"] = data_source
        self._paper_results[f"fno_{strategy_name}"] = result_dict

        # Compute strategy health report and embed in result
        try:
            health = self.compute_strategy_health(result_dict, strategy_type="fno", strategy_name=strategy_name)
            result_dict["health_report"] = health
        except Exception as e:
            logger.error("health_compute_error", error=str(e))
            result_dict.setdefault("_warnings", []).append(f"Health computation failed: {e}")

        # Record F&O paper trade sample to journal
        try:
            self._journal_record_fno_paper_trades(result_dict, strategy_name, underlying)
        except Exception as e:
            logger.error("fno_paper_sample_journal_record_error", error=str(e))
            result_dict.setdefault("_warnings", []).append(f"Journal recording failed: {e}")

        return result_dict

    def get_fno_strategies(self) -> list[str]:
        """Get available F&O strategy names."""
        return self.FNO_STRATEGIES

    # Well-known index instrument_token → NSE quote key mapping
    _INDEX_QUOTE_MAP: dict[int, str] = {
        256265: "NSE:NIFTY 50",       # NIFTY 50
        260105: "NSE:NIFTY BANK",     # NIFTY BANK
        257801: "NSE:NIFTY FIN SERVICE",  # FINNIFTY
        265:    "BSE:SENSEX",         # SENSEX
    }

    # ── Option Chain Refresh (feeds live chain data to option strategies) ──

    async def _option_chain_refresh_loop(self) -> None:
        """Periodically build option chains and feed them to OptionStrategyBase strategies.

        Without this, option strategies (credit spreads etc.) never receive
        chain data in the live loop and can neither enter nor manage positions.
        Runs every 10 seconds; uses REST quotes (≤500 instruments per call batch).
        """
        await asyncio.sleep(5)  # Let ticks start flowing first

        # Determine which underlyings our option strategies care about
        underlyings: set[str] = set()
        option_strategies: list[OptionStrategyBase] = []
        for s in self._strategies:
            if isinstance(s, OptionStrategyBase):
                option_strategies.append(s)
                underlyings.add(s.params.get("underlying", "NIFTY"))

        if not option_strategies:
            logger.debug("option_chain_refresh_loop_skipped", reason="no option strategies")
            return

        logger.info(
            "option_chain_refresh_loop_started",
            underlyings=list(underlyings),
            strategies=[s.name for s in option_strategies],
        )

        # Pre-fetch NFO/BFO instruments once (they don't change intra-day)
        instruments_by_underlying: dict[str, list[dict[str, Any]]] = {}
        try:
            nfo_instruments = await self._market_data.get_instruments("NFO")
            nfo_list = [
                i if isinstance(i, dict) else (i.model_dump() if hasattr(i, "model_dump") else vars(i))
                for i in nfo_instruments
            ]
            for underlying in underlyings:
                instruments_by_underlying[underlying] = [
                    i for i in nfo_list
                    if i.get("instrument_type") in ("CE", "PE")
                    and i.get("name", "").upper() == underlying.upper()
                ]
            # Also try BFO for SENSEX
            if "SENSEX" in underlyings:
                try:
                    bfo_instruments = await self._market_data.get_instruments("BFO")
                    bfo_list = [
                        i if isinstance(i, dict) else (i.model_dump() if hasattr(i, "model_dump") else vars(i))
                        for i in bfo_instruments
                    ]
                    sensex_opts = [
                        i for i in bfo_list
                        if i.get("instrument_type") in ("CE", "PE")
                        and i.get("name", "").upper() == "SENSEX"
                    ]
                    instruments_by_underlying["SENSEX"] = sensex_opts or instruments_by_underlying.get("SENSEX", [])
                except Exception:
                    pass
            logger.info("option_chain_instruments_loaded", counts={u: len(v) for u, v in instruments_by_underlying.items()})
        except Exception as e:
            logger.error("option_chain_instruments_load_failed", error=str(e))
            return

        while self._running:
            for underlying in underlyings:
                try:
                    # 1. Get spot price from cached live ticks
                    spot = 0.0
                    spot_token = {"NIFTY": 256265, "SENSEX": 265, "BANKNIFTY": 260105}.get(underlying, 0)
                    if spot_token and spot_token in self._live_ticks:
                        spot = self._live_ticks[spot_token].get("ltp", 0.0)
                    if spot <= 0:
                        # Fallback: try REST LTP
                        try:
                            ltp_key = {"NIFTY": "NSE:NIFTY 50", "SENSEX": "BSE:SENSEX", "BANKNIFTY": "NSE:NIFTY BANK"}.get(underlying)
                            if ltp_key:
                                ltp_data = await self._client.get_ltp([ltp_key])
                                for v in ltp_data.values():
                                    p = v.last_price if hasattr(v, "last_price") else 0
                                    if p > 0:
                                        spot = p
                        except Exception:
                            pass
                    if spot <= 0:
                        logger.debug("option_chain_skip_no_spot", underlying=underlying)
                        continue

                    # 2. Filter instruments to near-ATM strikes (±10 strikes) to limit quote calls
                    all_instruments = instruments_by_underlying.get(underlying, [])
                    if not all_instruments:
                        continue

                    # Find nearest expiry
                    from datetime import date as _date
                    today = _date.today()
                    expiries: set[str] = set()
                    for inst in all_instruments:
                        exp = inst.get("expiry", "")
                        if exp:
                            try:
                                exp_dt = datetime.strptime(exp, "%Y-%m-%d").date()
                                if exp_dt >= today:
                                    expiries.add(exp)
                            except (ValueError, TypeError):
                                pass
                    if not expiries:
                        continue
                    nearest_expiry = min(expiries)

                    expiry_instruments = [
                        i for i in all_instruments if i.get("expiry") == nearest_expiry
                    ]

                    # Narrow to ±10 strikes around ATM to keep quote counts small
                    strikes = sorted(set(i.get("strike", 0) for i in expiry_instruments if i.get("strike", 0) > 0))
                    if not strikes:
                        continue
                    atm_strike = min(strikes, key=lambda s: abs(s - spot))
                    atm_idx = strikes.index(atm_strike)
                    lo = max(0, atm_idx - 10)
                    hi = min(len(strikes), atm_idx + 11)
                    near_strikes = set(strikes[lo:hi])

                    near_instruments = [
                        i for i in expiry_instruments if i.get("strike", 0) in near_strikes
                    ]

                    # 3. Fetch quotes for these instruments
                    quote_keys = []
                    for inst in near_instruments:
                        exch = inst.get("exchange", "NFO")
                        tsym = inst.get("tradingsymbol", "")
                        if tsym:
                            quote_keys.append(f"{exch}:{tsym}")

                    quotes: dict[str, dict[str, Any]] = {}
                    # Zerodha allows max ~500 instruments per quote call
                    for batch_start in range(0, len(quote_keys), 400):
                        batch = quote_keys[batch_start:batch_start + 400]
                        try:
                            raw = await self._client.get_quote(batch)
                            for k, q in raw.items():
                                quotes[k] = {
                                    "last_price": q.last_price if hasattr(q, "last_price") else 0,
                                    "volume": q.volume if hasattr(q, "volume") else 0,
                                    "oi": q.oi if hasattr(q, "oi") else 0,
                                }
                        except Exception as e:
                            logger.warning("option_chain_quote_batch_error", underlying=underlying, error=str(e))

                    # 4. Build chain
                    chain = self._chain_builder.build_chain(
                        underlying=underlying,
                        spot_price=spot,
                        instruments=near_instruments,
                        quotes=quotes,
                        expiry_filter=nearest_expiry,
                    )

                    # 5. Feed chain to all option strategies that trade this underlying
                    for strat in option_strategies:
                        if strat.params.get("underlying", "NIFTY") == underlying:
                            strat.update_chain(chain)

                    logger.debug(
                        "option_chain_refreshed",
                        underlying=underlying,
                        expiry=nearest_expiry,
                        entries=len(chain.entries),
                        spot=round(spot, 2),
                        atm_iv=chain.atm_iv,
                    )
                except Exception as e:
                    logger.error("option_chain_refresh_error", underlying=underlying, error=str(e))

            await asyncio.sleep(10)

    async def _index_quote_enrichment_loop(self) -> None:
        """Periodically fetch REST quotes for index tokens to enrich volume/buy/sell data.

        The Kite WebSocket binary protocol doesn't include volume, buy_quantity,
        or sell_quantity for index instruments (segment 9). This loop supplements
        that data via the REST quote API every 15 seconds.
        """
        # Initial delay to let ticks start flowing
        await asyncio.sleep(3)
        while self._running:
            try:
                # Collect index tokens from subscribed tokens + any in live ticks
                index_keys: list[str] = []
                token_for_key: dict[str, int] = {}
                all_tokens = set(self._subscribed_tokens) | set(self._live_ticks.keys())
                for token in all_tokens:
                    if (token & 0xFF) == 9:  # index segment
                        key = self._INDEX_QUOTE_MAP.get(token)
                        if key:
                            index_keys.append(key)
                            token_for_key[key] = token

                if index_keys:
                    try:
                        quotes = await self._client.get_quote(index_keys)
                        snap_batch: list[dict] = []
                        for key, q in quotes.items():
                            token = token_for_key.get(key, q.instrument_token)
                            if token in self._live_ticks:
                                # Enrich existing tick data with REST-sourced volume/buy/sell
                                self._live_ticks[token]["volume"] = q.volume or 0
                                self._live_ticks[token]["buy_qty"] = q.buy_quantity or 0
                                self._live_ticks[token]["sell_qty"] = q.sell_quantity or 0
                            else:
                                # Seed tick data from REST if not yet in cache
                                change = 0.0
                                if q.ohlc and q.ohlc.close and q.ohlc.close > 0:
                                    change = q.last_price - q.ohlc.close
                                self._live_ticks[token] = {
                                    "token": token,
                                    "name": self._token_name_map.get(token, ""),
                                    "ltp": q.last_price,
                                    "volume": q.volume or 0,
                                    "high": q.ohlc.high if q.ohlc else 0,
                                    "low": q.ohlc.low if q.ohlc else 0,
                                    "open": q.ohlc.open if q.ohlc else 0,
                                    "close": q.ohlc.close if q.ohlc else 0,
                                    "change": change,
                                    "buy_qty": q.buy_quantity or 0,
                                    "sell_qty": q.sell_quantity or 0,
                                }
                            # Collect for DB snapshot
                            snap_batch.append({
                                "instrument_token": token or q.instrument_token,
                                "ts": str(q.timestamp) if q.timestamp else datetime.now().isoformat(),
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
                            })
                        if self._data_store and snap_batch:
                            try:
                                self._data_store.store_quote_snapshots(snap_batch)
                            except Exception:
                                pass
                        logger.debug("index_quote_enrichment_done", count=len(quotes))
                    except Exception as e:
                        logger.warning("index_quote_enrichment_error", error=str(e))
            except Exception as e:
                logger.error("index_enrichment_loop_error", error=str(e))
            await asyncio.sleep(15)

    async def _position_update_loop(self) -> None:
        while self._running:
            try:
                positions = await self._client.get_positions()
                self._risk.update_positions(positions.net)
                daily_pnl = sum(p.pnl for p in positions.net)
                self._risk.update_daily_pnl(daily_pnl)
                # Persist position snapshot to store
                if self._data_store:
                    try:
                        pos_dicts = [
                            {
                                "tradingsymbol": p.tradingsymbol,
                                "exchange": p.exchange,
                                "product": p.product,
                                "quantity": p.quantity,
                                "average_price": p.average_price,
                                "last_price": p.last_price,
                                "pnl": p.pnl,
                                "buy_quantity": p.buy_quantity,
                                "sell_quantity": p.sell_quantity,
                            }
                            for p in positions.net
                        ]
                        self._data_store.store_positions(pos_dicts)
                    except Exception:
                        pass
                await self._broadcast_ws({
                    "type": "positions",
                    "data": positions.model_dump(),
                    "pnl": daily_pnl,
                })
            except Exception as e:
                logger.error("position_update_error", error=str(e))
            await asyncio.sleep(30)


class TradingServiceManager:
    _instance: Optional[TradingServiceManager] = None

    def __init__(self) -> None:
        self._services: dict[str, TradingService] = {}

    @classmethod
    def get_instance(cls) -> TradingServiceManager:
        if cls._instance is None:
            cls._instance = TradingServiceManager()
        return cls._instance

    def get_service(self, zerodha_id: str, api_key: str = "", api_secret: str = "") -> TradingService:
        uid = zerodha_id.upper()
        if uid not in self._services:
            self._services[uid] = TradingService(api_key=api_key, api_secret=api_secret, zerodha_id=uid)
            logger.info("service_created", zerodha_id=uid)
        return self._services[uid]

    def has_service(self, zerodha_id: str) -> bool:
        return zerodha_id.upper() in self._services

    def remove_service(self, zerodha_id: str) -> None:
        uid = zerodha_id.upper()
        if uid in self._services:
            del self._services[uid]
