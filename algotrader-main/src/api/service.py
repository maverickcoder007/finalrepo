from __future__ import annotations

import asyncio
from datetime import datetime
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
        self._settings = get_settings()
        self._zerodha_id = zerodha_id
        self._auth = KiteAuthenticator(api_key=api_key, api_secret=api_secret, zerodha_id=zerodha_id)
        self._client = KiteClient(self._auth)
        self._ticker = KiteTicker(self._auth)
        self._risk = RiskManager()
        
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
        self._signal_log: list[dict[str, Any]] = []
        self._ws_clients: list[Any] = []
        self._lock = asyncio.Lock()
        self._custom_store = CustomStrategyStore()
        self._fno_strategy_store = FnOStrategyStore()
        self._paper_results: dict[str, Any] = {}  # cache latest paper trade results
        self._market_data = MarketDataProvider(self._client)
        self._portfolio_reporter = PortfolioReporter(self._client)
        self._backtest_results: dict[str, Any] = {}  # cache latest backtest results
        self._last_health_report: Optional[dict[str, Any]] = None  # cache latest health report

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
            instruments = await self._client.get_instruments(exchange)
            self._instruments_cache[exchange] = instruments

        results = []
        q = query.upper()
        eq_only = exchange in ("NSE", "BSE")
        for inst in self._instruments_cache[exchange]:
            if eq_only and inst.instrument_type not in ("EQ", ""):
                continue
            if not q or q in inst.tradingsymbol.upper() or q in (inst.name or "").upper():
                results.append({
                    "token": inst.instrument_token,
                    "symbol": inst.tradingsymbol,
                    "name": inst.name or inst.tradingsymbol,
                    "exchange": inst.exchange,
                    "type": inst.instrument_type,
                    "lot_size": inst.lot_size,
                    "expiry": inst.expiry or "",
                    "strike": inst.strike,
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
        return [o.model_dump() for o in orders]

    async def get_positions(self) -> dict[str, Any]:
        positions = await self._client.get_positions()
        return positions.model_dump()

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
        return [
            {
                "name": s.name,
                "is_active": s.is_active,
                "params": s.params,
                "type": "option" if isinstance(s, OptionStrategyBase) else "equity",
            }
            for s in self._strategies
        ]

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
                nfo_instruments = await self._client.get_instruments("NFO")
                inst_dicts = [
                    i.model_dump() if hasattr(i, "model_dump") else i
                    for i in nfo_instruments
                    if (i.instrument_type if hasattr(i, "instrument_type") else i.get("instrument_type", "")) in ("CE", "PE")
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

    def add_strategy(self, name: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        strategy_map = {
            "ema_crossover": EMACrossoverStrategy,
            "vwap_breakout": VWAPBreakoutStrategy,
            "mean_reversion": MeanReversionStrategy,
            "rsi": RSIStrategy,
            "scanner_strategy": ScannerStrategy,
            "iron_condor": IronCondorStrategy,
            "straddle_strangle": StraddleStrangleStrategy,
            "bull_call_spread": BullCallSpreadStrategy,
            "bear_put_spread": BearPutSpreadStrategy,
        }
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
        self._strategies.append(strategy)
        logger.info("strategy_added", name=name)
        return {"success": True, "name": strategy.name}

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
            except Exception as e:
                logger.warning(f"Failed to resolve token for {symbol}: {e}")
        
        if not tokens:
            return {"error": "Could not resolve any default tokens"}
        
        return await self.start_live(tokens, mode)

    async def start_live(self, tokens: list[int], mode: str = "quote") -> dict[str, Any]:
        if self._running:
            return {"error": "Already running"}

        self._running = True
        self._subscribed_tokens = tokens

        self._ticker.on_ticks = self._on_ticks
        self._ticker.on_connect = self._on_connect
        self._ticker.on_close = self._on_close
        self._ticker.on_error = self._on_error
        self._ticker.on_order_update = self._on_order_update

        await self._seed_bar_data(tokens)
        await self._ticker.subscribe(tokens, TickMode(mode))

        for coro in [
            self._ticker.connect(),
            self._execution.start_reconciliation_loop(),
            self._position_update_loop(),
        ]:
            task = asyncio.create_task(coro)
            task.add_done_callback(self._task_exception_handler)

        logger.info("live_trading_started", tokens=tokens)
        return {"success": True, "tokens": tokens}

    async def stop_live(self) -> dict[str, Any]:
        self._running = False
        await self._execution.stop_reconciliation_loop()
        await self._ticker.disconnect()
        logger.info("live_trading_stopped")
        return {"success": True}

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
        from datetime import timedelta
        to_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        from_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S")
        for token in tokens:
            try:
                candles = await self._client.get_historical_data(
                    instrument_token=token,
                    interval=Interval.MINUTE_5,
                    from_date=from_date,
                    to_date=to_date,
                )
                if candles:
                    rows = [
                        {"timestamp": c.timestamp, "open": c.open, "high": c.high,
                         "low": c.low, "close": c.close, "volume": c.volume}
                        for c in candles
                    ]
                    df = pd.DataFrame(rows)
                    for strategy in self._strategies:
                        strategy.update_bar_data(token, df)
            except Exception as e:
                logger.warning("bar_data_seed_failed", token=token, error=str(e))

    async def _on_ticks(self, ticks: list[Tick]) -> None:
        for tick in ticks:
            tick_data = {
                "token": tick.instrument_token,
                "ltp": tick.last_price,
                "volume": tick.volume_traded,
                "high": tick.ohlc.high,
                "low": tick.ohlc.low,
                "open": tick.ohlc.open,
                "close": tick.ohlc.close,
                "change": tick.change,
                "buy_qty": tick.total_buy_quantity,
                "sell_qty": tick.total_sell_quantity,
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

        # Fetch historical data
        from datetime import timedelta
        to_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")

        try:
            candles = await self._client.get_historical_data(
                instrument_token=instrument_token,
                interval=Interval(interval),
                from_date=from_date,
                to_date=to_date,
            )
        except Exception as e:
            logger.error("paper_trade_data_fetch_failed", error=str(e))
            return {"error": f"Failed to fetch data: {str(e)}"}

        if not candles:
            return {"error": "No historical data returned. Check instrument token and date range."}

        rows = [
            {"timestamp": c.timestamp, "open": c.open, "high": c.high,
             "low": c.low, "close": c.close, "volume": c.volume}
            for c in candles
        ]
        df = pd.DataFrame(rows)

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

        # Compute strategy health report and embed in result
        try:
            health = self.compute_strategy_health(result_dict, strategy_type="equity", strategy_name=strategy_name)
            result_dict["health_report"] = health
        except Exception as e:
            logger.error("health_compute_error", error=str(e))

        # Record paper trades to journal for analytics
        try:
            self._journal_record_paper_trades(result_dict, strategy_name, tradingsymbol)
        except Exception as e:
            logger.error("paper_trade_journal_record_error", error=str(e))

        logger.info("paper_trade_complete", strategy=strategy_name, trades=result_dict.get("total_trades", len(result_dict.get("trades", []))))
        return result_dict

    async def run_paper_trade_with_sample_data(
        self,
        strategy_name: str,
        tradingsymbol: str = "SAMPLE",
        bars: int = 500,
        capital: float = 100000.0,
        strategy_params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Run paper trade with synthetic data (no auth required)."""
        import numpy as np

        strategy = self._resolve_strategy(strategy_name, strategy_params)
        if isinstance(strategy, dict):
            return strategy

        # Generate realistic synthetic OHLCV data with trending regimes
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=bars, freq="5min")
        price = 100.0
        opens, highs, lows, closes, volumes = [], [], [], [], []
        regime_length = max(20, bars // 10)
        trend = 0.0
        vol_scale = 0.5
        for i in range(bars):
            # Switch regime periodically for trending & ranging moves
            if i % regime_length == 0:
                trend = np.random.choice([-0.15, -0.05, 0.0, 0.05, 0.15])
                vol_scale = np.random.choice([0.3, 0.5, 0.8])
            change = np.random.normal(trend, vol_scale)
            o = price
            c = price + change
            h = max(o, c) + abs(np.random.normal(0, vol_scale * 0.4))
            l = min(o, c) - abs(np.random.normal(0, vol_scale * 0.4))
            # Volume with occasional spikes
            v = int(np.random.uniform(20000, 80000))
            if np.random.random() < 0.15:
                v = int(v * np.random.uniform(2.0, 4.0))
            opens.append(o)
            highs.append(h)
            lows.append(l)
            closes.append(c)
            volumes.append(v)
            price = c

        df = pd.DataFrame({
            "timestamp": dates, "open": opens, "high": highs,
            "low": lows, "close": closes, "volume": volumes,
        })

        engine = PaperTradingEngine(strategy=strategy, initial_capital=capital)
        result = engine.run(df, instrument_token=0, tradingsymbol=tradingsymbol, timeframe="5min")
        # BacktestEngine.run() returns a plain dict (not an object with to_dict_safe)
        result_dict = result if isinstance(result, dict) else result.to_dict_safe()
        self._paper_results[strategy_name] = result_dict

        # Compute strategy health report and embed in result
        try:
            health = self.compute_strategy_health(result_dict, strategy_type="equity", strategy_name=strategy_name)
            result_dict["health_report"] = health
        except Exception as e:
            logger.error("health_compute_error", error=str(e))

        # Record paper trades to journal
        try:
            self._journal_record_paper_trades(result_dict, strategy_name, tradingsymbol)
        except Exception as e:
            logger.error("paper_trade_sample_journal_error", error=str(e))

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

        result = engine.run(df, instrument_token, tradingsymbol)
        self._backtest_results = result

        # Compute strategy health report and embed in result
        try:
            stype = "intraday" if is_intraday else "equity"
            health = self.compute_strategy_health(result, strategy_type=stype, strategy_name=strategy_name)
            result["health_report"] = health
        except Exception as e:
            logger.error("health_compute_error", error=str(e))

        # Record backtest trades to journal for analysis
        try:
            self._journal_record_backtest_trades(result, strategy_name, tradingsymbol)
        except Exception as e:
            logger.error("backtest_journal_record_error", error=str(e))

        return result

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
        strategy_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run backtest on synthetic sample data."""
        strategy = self._resolve_strategy(strategy_name, strategy_params)
        if isinstance(strategy, dict):
            return strategy

        import numpy as np
        np.random.seed(42)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=bars, freq="D")
        base = 100.0

        # Generate realistic price data with trend regimes and mean-reversion
        # Instead of pure random walk, create alternating trending/ranging periods
        prices = np.zeros(bars)
        prices[0] = base
        regime_length = max(20, bars // 10)  # ~10 regime changes
        trend = 0.0
        volatility = 0.02
        for i in range(1, bars):
            # Switch regime every regime_length bars
            if i % regime_length == 0:
                trend = np.random.choice([-0.003, -0.001, 0.0, 0.001, 0.003])
                volatility = np.random.choice([0.015, 0.02, 0.03])
            noise = np.random.normal(trend, volatility)
            prices[i] = prices[i - 1] * (1 + noise)

        # Volume with spikes: base volume + occasional spikes for breakout signals
        base_vol = np.random.randint(500_000, 2_000_000, bars).astype(float)
        # Add volume spikes (~20% of bars get 1.5-3x volume)
        spike_mask = np.random.random(bars) < 0.20
        base_vol[spike_mask] *= np.random.uniform(1.5, 3.0, spike_mask.sum())
        volumes = base_vol.astype(int)

        df = pd.DataFrame({
            "timestamp": dates,
            "open": prices * (1 + np.random.uniform(-0.008, 0.008, bars)),
            "high": prices * (1 + np.abs(np.random.normal(0, 0.012, bars))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.012, bars))),
            "close": prices,
            "volume": volumes,
        }, index=dates)

        engine = BacktestEngine(
            strategy=strategy, initial_capital=initial_capital,
            slippage_pct=slippage_pct, position_sizing=position_sizing,
            risk_per_trade=risk_per_trade, capital_fraction=capital_fraction,
            use_indian_costs=use_indian_costs, is_intraday=is_intraday,
        )
        result = engine.run(df, tradingsymbol="SAMPLE")
        self._backtest_results = result

        # Compute strategy health report and embed in result
        try:
            stype = "intraday" if is_intraday else "equity"
            health = self.compute_strategy_health(result, strategy_type=stype, strategy_name=strategy_name)
            result["health_report"] = health
        except Exception as e:
            logger.error("health_compute_error", error=str(e))

        # Record backtest trades to journal for analysis
        try:
            self._journal_record_backtest_trades(result, strategy_name, "SAMPLE")
        except Exception as e:
            logger.error("backtest_sample_journal_record_error", error=str(e))

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

        # Compute strategy health report and embed in result
        try:
            health = self.compute_strategy_health(result, strategy_type="fno", strategy_name=strategy_name)
            result["health_report"] = health
        except Exception as e:
            logger.error("health_compute_error", error=str(e))

        # Record F&O backtest trades to journal
        try:
            self._journal_record_fno_backtest_trades(result, strategy_name, underlying)
        except Exception as e:
            logger.error("fno_backtest_journal_record_error", error=str(e))

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
    ) -> dict[str, Any]:
        """Run F&O backtest on synthetic data (no auth required)."""
        if not self._is_valid_fno_strategy(strategy_name):
            return {
                "error": f"Unknown F&O strategy: {strategy_name}",
                "available": self.FNO_STRATEGIES,
            }

        import numpy as np
        np.random.seed(42)
        # Simulate index-like data (NIFTY ~20000)
        base = 20000.0
        dates = pd.date_range(end=pd.Timestamp.now(), periods=bars, freq="D")
        returns = np.random.normal(0.0003, 0.012, bars)
        prices = base * np.cumprod(1 + returns)
        df = pd.DataFrame({
            "open": prices * (1 + np.random.uniform(-0.003, 0.003, bars)),
            "high": prices * (1 + np.abs(np.random.normal(0, 0.008, bars))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.008, bars))),
            "close": prices,
            "volume": np.random.randint(100000, 5000000, bars),
        }, index=dates)

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
        self._backtest_results = result

        # Compute strategy health report and embed in result
        try:
            health = self.compute_strategy_health(result, strategy_type="fno", strategy_name=strategy_name)
            result["health_report"] = health
        except Exception as e:
            logger.error("health_compute_error", error=str(e))

        # Record F&O backtest sample trades to journal
        try:
            self._journal_record_fno_backtest_trades(result, strategy_name, underlying)
        except Exception as e:
            logger.error("fno_backtest_sample_journal_record_error", error=str(e))

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

        # Record F&O paper trades to journal
        try:
            self._journal_record_fno_paper_trades(result_dict, strategy_name, underlying)
        except Exception as e:
            logger.error("fno_paper_trade_journal_record_error", error=str(e))

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
    ) -> dict[str, Any]:
        """Run F&O paper trade on synthetic data (no auth required)."""
        if not self._is_valid_fno_strategy(strategy_name):
            return {
                "error": f"Unknown F&O strategy: {strategy_name}",
                "available": self.FNO_STRATEGIES,
            }

        import numpy as np
        np.random.seed(42)
        base = 20000.0
        dates = pd.date_range(end=pd.Timestamp.now(), periods=bars, freq="D")
        returns = np.random.normal(0.0003, 0.012, bars)
        prices = base * np.cumprod(1 + returns)
        df = pd.DataFrame({
            "open": prices * (1 + np.random.uniform(-0.003, 0.003, bars)),
            "high": prices * (1 + np.abs(np.random.normal(0, 0.008, bars))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.008, bars))),
            "close": prices,
            "volume": np.random.randint(100000, 5000000, bars),
        }, index=dates)

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
        self._paper_results[f"fno_{strategy_name}"] = result_dict

        # Compute strategy health report and embed in result
        try:
            health = self.compute_strategy_health(result_dict, strategy_type="fno", strategy_name=strategy_name)
            result_dict["health_report"] = health
        except Exception as e:
            logger.error("health_compute_error", error=str(e))

        # Record F&O paper trade sample to journal
        try:
            self._journal_record_fno_paper_trades(result_dict, strategy_name, underlying)
        except Exception as e:
            logger.error("fno_paper_sample_journal_record_error", error=str(e))

        return result_dict

    def get_fno_strategies(self) -> list[str]:
        """Get available F&O strategy names."""
        return self.FNO_STRATEGIES

    async def _position_update_loop(self) -> None:
        while self._running:
            try:
                positions = await self._client.get_positions()
                self._risk.update_positions(positions.net)
                daily_pnl = sum(p.pnl for p in positions.net)
                self._risk.update_daily_pnl(daily_pnl)
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
