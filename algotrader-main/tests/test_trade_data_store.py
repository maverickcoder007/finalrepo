"""
Tests for TradeDataStore — per-trade OHLCV capture + daily analysis.
"""

import os
import sys
import tempfile
import pytest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.journal.trade_data_store import TradeDataStore, TradeRegistration, DailyAnalysis


@pytest.fixture
def store(tmp_path):
    """Create a TradeDataStore with a temp DB."""
    db = str(tmp_path / "test_trade_data.db")
    return TradeDataStore(db_path=db)


@pytest.fixture
def sample_candles():
    """Generate sample OHLCV candles."""
    base = datetime(2025, 1, 15, 9, 30)
    candles = []
    for i in range(20):
        ts = (base + timedelta(minutes=i * 5)).isoformat()
        price = 100 + i * 0.5
        candles.append({
            "ts": ts,
            "open": price,
            "high": price + 1,
            "low": price - 1,
            "close": price + 0.3,
            "volume": 10000 + i * 100,
            "oi": None,
        })
    return candles


class TestTradeRegistration:
    """Test trade registration lifecycle."""

    def test_register_trade_open(self, store):
        reg = store.register_trade_open(
            trade_id="T001",
            order_id="ORD001",
            instrument="RELIANCE",
            tradingsymbol="RELIANCE",
            exchange="NSE",
            instrument_token=738561,
            strategy_name="ema_crossover",
            direction="BUY",
            quantity=10,
            entry_price=2500.0,
        )
        assert reg.trade_id == "T001"
        assert reg.instrument == "RELIANCE"
        assert reg.entry_price == 2500.0
        assert not reg.is_closed

    def test_get_trade(self, store):
        store.register_trade_open(
            trade_id="T002", order_id="ORD002", instrument="TCS",
            tradingsymbol="TCS", exchange="NSE", instrument_token=2953217,
            strategy_name="rsi", direction="BUY", quantity=5, entry_price=3800.0,
        )
        trade = store.get_trade("T002")
        assert trade is not None
        assert trade.tradingsymbol == "TCS"

    def test_get_trade_not_found(self, store):
        assert store.get_trade("NONEXISTENT") is None

    def test_close_trade(self, store):
        store.register_trade_open(
            trade_id="T003", order_id="ORD003", instrument="INFY",
            tradingsymbol="INFY", exchange="NSE", instrument_token=408065,
            strategy_name="mean_rev", direction="BUY", quantity=15,
            entry_price=1500.0,
        )
        closed = store.close_trade("T003", exit_price=1550.0, pnl=750.0)
        assert closed is True

        trade = store.get_trade("T003")
        assert trade.is_closed
        assert trade.exit_price == 1550.0
        assert trade.pnl == 750.0
        assert trade.exit_time != ""

    def test_close_nonexistent_trade(self, store):
        assert store.close_trade("FAKE", 100.0, pnl=0) is False

    def test_find_trade_by_order_id(self, store):
        store.register_trade_open(
            trade_id="T004", order_id="ORD-ABC-123", instrument="SBIN",
            tradingsymbol="SBIN", exchange="NSE", instrument_token=779521,
            strategy_name="vwap", direction="BUY", quantity=20,
            entry_price=600.0,
        )
        found = store.find_trade_by_order_id("ORD-ABC-123")
        assert found is not None
        assert found.trade_id == "T004"

    def test_to_dict(self, store):
        reg = store.register_trade_open(
            trade_id="T005", order_id="ORD005", instrument="HDFC",
            tradingsymbol="HDFC", exchange="NSE", instrument_token=341249,
            strategy_name="test", direction="SELL", quantity=8,
            entry_price=1700.50,
        )
        d = reg.to_dict()
        assert d["trade_id"] == "T005"
        assert d["entry_price"] == 1700.50
        assert isinstance(d["metadata"], dict)


class TestOpenClosedQueries:
    """Test open/closed trade queries."""

    def test_get_open_trades(self, store):
        store.register_trade_open(
            trade_id="OPEN1", order_id="O1", instrument="A",
            tradingsymbol="A", exchange="NSE", instrument_token=1,
            strategy_name="s1", direction="BUY", quantity=1, entry_price=100.0,
        )
        store.register_trade_open(
            trade_id="OPEN2", order_id="O2", instrument="B",
            tradingsymbol="B", exchange="NSE", instrument_token=2,
            strategy_name="s2", direction="SELL", quantity=2, entry_price=200.0,
        )
        store.register_trade_open(
            trade_id="CLOSED1", order_id="O3", instrument="C",
            tradingsymbol="C", exchange="NSE", instrument_token=3,
            strategy_name="s1", direction="BUY", quantity=3, entry_price=300.0,
        )
        store.close_trade("CLOSED1", exit_price=310.0, pnl=30.0)

        open_trades = store.get_open_trades()
        assert len(open_trades) == 2
        ids = {t.trade_id for t in open_trades}
        assert "OPEN1" in ids
        assert "OPEN2" in ids

    def test_get_trades_for_date(self, store):
        today = datetime.now().strftime("%Y-%m-%d")
        store.register_trade_open(
            trade_id="TODAY1", order_id="OT1", instrument="X",
            tradingsymbol="X", exchange="NSE", instrument_token=10,
            strategy_name="s1", direction="BUY", quantity=1, entry_price=100.0,
            entry_time=f"{today}T10:00:00",
        )
        trades = store.get_trades_for_date(today)
        assert len(trades) >= 1

    def test_get_trades_by_instrument(self, store):
        store.register_trade_open(
            trade_id="INS1", order_id="OI1", instrument="NIFTY",
            tradingsymbol="NIFTY25JAN25000CE", exchange="NFO", instrument_token=100,
            strategy_name="oi_strategy", direction="BUY", quantity=50, entry_price=150.0,
        )
        trades = store.get_trades_by_instrument("NIFTY25JAN25000CE")
        assert len(trades) == 1
        assert trades[0].trade_id == "INS1"


class TestCandleStorage:
    """Test OHLCV candle storage and retrieval."""

    def test_store_and_retrieve_candles(self, store, sample_candles):
        store.register_trade_open(
            trade_id="TC1", order_id="OC1", instrument="RELIANCE",
            tradingsymbol="RELIANCE", exchange="NSE", instrument_token=738561,
            strategy_name="test", direction="BUY", quantity=10, entry_price=100.0,
        )
        stored = store.store_trade_candles("TC1", sample_candles)
        assert stored == 20

        candles = store.get_trade_candles("TC1")
        assert len(candles) == 20
        assert candles[0]["open"] == 100.0
        assert candles[0]["volume"] == 10000

    def test_candle_count_updated(self, store, sample_candles):
        store.register_trade_open(
            trade_id="TC2", order_id="OC2", instrument="TCS",
            tradingsymbol="TCS", exchange="NSE", instrument_token=2953217,
            strategy_name="test", direction="BUY", quantity=5, entry_price=100.0,
        )
        store.store_trade_candles("TC2", sample_candles)
        trade = store.get_trade("TC2")
        assert trade.candle_count == 20

    def test_empty_candles(self, store):
        store.register_trade_open(
            trade_id="TC3", order_id="OC3", instrument="INFY",
            tradingsymbol="INFY", exchange="NSE", instrument_token=408065,
            strategy_name="test", direction="BUY", quantity=1, entry_price=100.0,
        )
        stored = store.store_trade_candles("TC3", [])
        assert stored == 0
        candles = store.get_trade_candles("TC3")
        assert len(candles) == 0

    def test_pending_candles(self, store):
        store.register_trade_open(
            trade_id="PC1", order_id="OP1", instrument="A",
            tradingsymbol="A", exchange="NSE", instrument_token=1,
            strategy_name="s1", direction="BUY", quantity=1, entry_price=100.0,
        )
        store.close_trade("PC1", exit_price=110.0, pnl=10.0)
        # No candles stored — should show as pending
        pending = store.get_trades_pending_candles()
        assert len(pending) >= 1
        assert any(t.trade_id == "PC1" for t in pending)


class TestTradeReplay:
    """Test trade replay with analysis."""

    def test_replay_with_candles(self, store, sample_candles):
        store.register_trade_open(
            trade_id="TR1", order_id="OR1", instrument="RELIANCE",
            tradingsymbol="RELIANCE", exchange="NSE", instrument_token=738561,
            strategy_name="ema_crossover", direction="BUY", quantity=10,
            entry_price=100.0,
        )
        store.close_trade("TR1", exit_price=110.0, pnl=100.0)
        store.store_trade_candles("TR1", sample_candles)

        replay = store.get_trade_replay("TR1")
        assert replay is not None
        assert replay["has_data"] is True
        assert replay["candle_count"] == 20

        # Check analysis fields
        analysis = replay["analysis"]
        assert "mfe" in analysis
        assert "mae" in analysis
        assert "edge_ratio" in analysis
        assert "bar_pnl" in analysis
        assert "volatility" in analysis
        assert "avg_volume" in analysis
        assert len(analysis["bar_pnl"]) == 20

    def test_replay_without_candles(self, store):
        store.register_trade_open(
            trade_id="TR2", order_id="OR2", instrument="TCS",
            tradingsymbol="TCS", exchange="NSE", instrument_token=2953217,
            strategy_name="test", direction="BUY", quantity=5, entry_price=100.0,
        )
        replay = store.get_trade_replay("TR2")
        assert replay["has_data"] is False
        assert replay["analysis"] == {}

    def test_replay_not_found(self, store):
        assert store.get_trade_replay("NONEXISTENT") is None

    def test_mfe_mae_long_trade(self, store):
        """Verify MFE/MAE calculation for a long trade."""
        candles = [
            {"ts": "2025-01-15T09:30:00", "open": 100, "high": 105, "low": 98, "close": 103, "volume": 1000},
            {"ts": "2025-01-15T09:35:00", "open": 103, "high": 110, "low": 101, "close": 108, "volume": 1200},
            {"ts": "2025-01-15T09:40:00", "open": 108, "high": 112, "low": 95, "close": 107, "volume": 800},
        ]
        store.register_trade_open(
            trade_id="MFE1", order_id="OMFE1", instrument="TEST",
            tradingsymbol="TEST", exchange="NSE", instrument_token=1,
            strategy_name="test", direction="BUY", quantity=10, entry_price=100.0,
        )
        store.close_trade("MFE1", exit_price=107.0, pnl=70.0)
        store.store_trade_candles("MFE1", candles)

        replay = store.get_trade_replay("MFE1")
        analysis = replay["analysis"]
        # MFE: max high (112) - entry (100) = 12
        assert analysis["mfe"] == 12.0
        # MAE: entry (100) - min low (95) = 5
        assert analysis["mae"] == 5.0
        # Edge ratio: 12/5 = 2.4
        assert analysis["edge_ratio"] == 2.4

    def test_mfe_mae_short_trade(self, store):
        """Verify MFE/MAE calculation for a short trade."""
        candles = [
            {"ts": "2025-01-15T09:30:00", "open": 100, "high": 103, "low": 96, "close": 97, "volume": 1000},
            {"ts": "2025-01-15T09:35:00", "open": 97, "high": 105, "low": 92, "close": 93, "volume": 1200},
        ]
        store.register_trade_open(
            trade_id="SHORT1", order_id="OSHORT1", instrument="TEST",
            tradingsymbol="TEST", exchange="NSE", instrument_token=1,
            strategy_name="test", direction="SELL", quantity=10, entry_price=100.0,
        )
        store.close_trade("SHORT1", exit_price=93.0, pnl=70.0)
        store.store_trade_candles("SHORT1", candles)

        replay = store.get_trade_replay("SHORT1")
        analysis = replay["analysis"]
        # MFE for short: entry (100) - min low (92) = 8
        assert analysis["mfe"] == 8.0
        # MAE for short: max high (105) - entry (100) = 5
        assert analysis["mae"] == 5.0


class TestDailyAnalysis:
    """Test daily trade analysis."""

    def _create_day_trades(self, store, trade_date: str, prefix: str = ""):
        """Helper: create several trades for a given date."""
        pfx = prefix or trade_date.replace("-", "")
        trades = [
            (f"{pfx}_D1", "RELIANCE", "NSE", "ema_crossover", "BUY", 10, 2500.0, 2550.0, 500.0),
            (f"{pfx}_D2", "TCS", "NSE", "rsi", "SELL", 5, 3800.0, 3750.0, 250.0),
            (f"{pfx}_D3", "RELIANCE", "NSE", "ema_crossover", "BUY", 8, 2510.0, 2490.0, -160.0),
            (f"{pfx}_D4", "NIFTY25JAN25000CE", "NFO", "oi_strategy", "BUY", 50, 200.0, 220.0, 1000.0),
            (f"{pfx}_D5", "INFY", "NSE", "vwap", "BUY", 12, 1800.0, 1780.0, -240.0),
        ]
        for tid, sym, exch, strat, dirn, qty, entry, exit_p, pnl in trades:
            store.register_trade_open(
                trade_id=tid, order_id=f"O{tid}", instrument=sym,
                tradingsymbol=sym, exchange=exch, instrument_token=0,
                strategy_name=strat, direction=dirn, quantity=qty,
                entry_price=entry,
                entry_time=f"{trade_date}T10:00:00",
            )
            store.close_trade(
                tid, exit_price=exit_p, pnl=pnl,
                exit_time=f"{trade_date}T14:00:00",
            )

    def test_daily_analysis_summary(self, store):
        today = datetime.now().strftime("%Y-%m-%d")
        self._create_day_trades(store, today)

        analysis = store.compute_daily_analysis(today)
        assert analysis.total_trades == 5
        assert analysis.closed_trades == 5
        assert analysis.winners == 3  # D1, D2, D4 positive
        assert analysis.losers == 2   # D3, D5 negative
        assert analysis.total_pnl == pytest.approx(500 + 250 - 160 + 1000 - 240, abs=0.01)
        assert analysis.win_rate == pytest.approx(60.0, abs=0.1)

    def test_daily_analysis_by_instrument(self, store):
        today = datetime.now().strftime("%Y-%m-%d")
        self._create_day_trades(store, today)

        analysis = store.compute_daily_analysis(today)
        assert "RELIANCE" in analysis.by_instrument
        rel = analysis.by_instrument["RELIANCE"]
        assert rel["total_trades"] == 2
        assert rel["total_pnl"] == pytest.approx(340.0, abs=0.01)

    def test_daily_analysis_by_strategy(self, store):
        today = datetime.now().strftime("%Y-%m-%d")
        self._create_day_trades(store, today)

        analysis = store.compute_daily_analysis(today)
        assert "ema_crossover" in analysis.by_strategy
        ema = analysis.by_strategy["ema_crossover"]
        assert ema["total_trades"] == 2

    def test_daily_analysis_by_tradingsymbol(self, store):
        today = datetime.now().strftime("%Y-%m-%d")
        self._create_day_trades(store, today)

        analysis = store.compute_daily_analysis(today)
        assert "TCS" in analysis.by_tradingsymbol
        assert "NIFTY25JAN25000CE" in analysis.by_tradingsymbol

    def test_daily_pnl_curve(self, store):
        today = datetime.now().strftime("%Y-%m-%d")
        self._create_day_trades(store, today)

        analysis = store.compute_daily_analysis(today)
        assert len(analysis.pnl_curve) == 5  # 5 closed trades
        # Last cumulative PnL should equal total
        assert analysis.pnl_curve[-1]["cumulative_pnl"] == pytest.approx(analysis.total_pnl, abs=0.01)

    def test_daily_analysis_empty_date(self, store):
        analysis = store.compute_daily_analysis("2020-01-01")
        assert analysis.total_trades == 0
        assert analysis.total_pnl == 0.0

    def test_daily_analysis_caching(self, store):
        today = datetime.now().strftime("%Y-%m-%d")
        self._create_day_trades(store, today)

        # First call computes and caches
        store.compute_daily_analysis(today)
        # Second call should return cached
        cached = store.get_daily_analysis(today)
        assert cached is not None
        assert cached["total_trades"] == 5

    def test_daily_analysis_range(self, store):
        d1 = "2025-01-13"
        d2 = "2025-01-14"
        d3 = "2025-01-15"

        self._create_day_trades(store, d1)
        self._create_day_trades(store, d2)
        # d3 has no trades

        results = store.get_daily_analysis_range(from_date=d1, to_date=d3)
        assert len(results) == 2  # d1 and d2 have trades, d3 does not

    def test_daily_analysis_to_dict(self, store):
        today = datetime.now().strftime("%Y-%m-%d")
        self._create_day_trades(store, today)

        analysis = store.compute_daily_analysis(today)
        d = analysis.to_dict()
        assert isinstance(d, dict)
        assert "total_pnl" in d
        assert "by_instrument" in d
        assert "pnl_curve" in d

    def test_best_worst_trade(self, store):
        today = datetime.now().strftime("%Y-%m-%d")
        self._create_day_trades(store, today)

        analysis = store.compute_daily_analysis(today)
        assert analysis.best_trade_pnl == 1000.0
        assert analysis.best_trade_symbol == "NIFTY25JAN25000CE"
        assert analysis.worst_trade_pnl == -240.0
        assert analysis.worst_trade_symbol == "INFY"


class TestStats:
    """Test store statistics."""

    def test_stats(self, store, sample_candles):
        store.register_trade_open(
            trade_id="S1", order_id="OS1", instrument="A",
            tradingsymbol="A", exchange="NSE", instrument_token=1,
            strategy_name="s", direction="BUY", quantity=1, entry_price=100.0,
        )
        store.register_trade_open(
            trade_id="S2", order_id="OS2", instrument="B",
            tradingsymbol="B", exchange="NSE", instrument_token=2,
            strategy_name="s", direction="SELL", quantity=2, entry_price=200.0,
        )
        store.close_trade("S2", exit_price=190.0, pnl=-20.0)
        store.store_trade_candles("S2", sample_candles)

        stats = store.get_stats()
        assert stats["total_trades"] == 2
        assert stats["closed_trades"] == 1
        assert stats["open_trades"] == 1
        assert stats["trades_with_candles"] == 1
        assert stats["pending_candle_fetch"] == 0  # S2 now has candles
        assert stats["total_candle_bars"] == 20


class TestExtractUnderlying:
    """Test underlying extraction from tradingsymbol."""

    def test_nifty_option(self, store):
        assert store._extract_underlying("NIFTY25JAN25000CE") == "NIFTY"

    def test_banknifty_option(self, store):
        assert store._extract_underlying("BANKNIFTY25JAN50000PE") == "BANKNIFTY"

    def test_sensex(self, store):
        assert store._extract_underlying("SENSEX25JAN75000CE") == "SENSEX"

    def test_equity(self, store):
        assert store._extract_underlying("RELIANCE") == "RELIANCE"

    def test_finnifty(self, store):
        assert store._extract_underlying("FINNIFTY25JAN22000CE") == "FINNIFTY"
