"""
Tests for the 3-tier data source resolver:
  Tier 1: Live Zerodha → Tier 2: DB cache → Tier 3: Synthetic (with confirmation)

Tests verify:
  - When live data is available, it's used.
  - When not authenticated but DB cache has data, cache is used.
  - When no data and allow_synthetic=False → error with confirmation prompt.
  - When no data and allow_synthetic=True → synthetic is used.
  - All 4 sample endpoints honour the resolver.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ─── Helpers ────────────────────────────────────────────────

def _make_ohlcv_df(bars: int = 600, base: float = 100.0) -> pd.DataFrame:
    """Create a minimal OHLCV DataFrame mimicking get_historical_df output."""
    rng = np.random.default_rng(42)
    dates = pd.date_range(end=datetime.now(), periods=bars, freq="D")
    closes = base * np.exp(np.cumsum(rng.normal(0, 0.01, bars)))
    return pd.DataFrame({
        "open": closes * 0.999,
        "high": closes * 1.005,
        "low": closes * 0.995,
        "close": closes,
        "volume": rng.integers(1_000_000, 10_000_000, bars),
    }, index=dates).rename_axis("timestamp")


def _make_candle_rows(bars: int = 600, base: float = 100.0) -> list[dict[str, Any]]:
    """Create candle rows as returned by MarketDataStore.get_candles()."""
    rng = np.random.default_rng(42)
    start = datetime.now() - timedelta(days=bars + 50)
    rows = []
    price = base
    for i in range(bars):
        price *= np.exp(rng.normal(0, 0.01))
        ts = (start + timedelta(days=i)).strftime("%Y-%m-%dT09:15:00")
        rows.append({
            "ts": ts,
            "open": round(price * 0.999, 2),
            "high": round(price * 1.005, 2),
            "low": round(price * 0.995, 2),
            "close": round(price, 2),
            "volume": int(rng.integers(1_000_000, 10_000_000)),
            "oi": 0,
        })
    return rows


# ─── Fake TradingService builder ─────────────────────────────

def _build_service(*, authenticated: bool = False, live_df: pd.DataFrame | None = None,
                   cached_candles: list[dict] | None = None):
    """Build a minimal TradingService-like object with mocked deps.

    This avoids importing the real TradingService (which pulls heavy deps),
    instead we test _resolve_sample_data directly via the class.
    """
    from src.api.service import TradingService

    with patch.object(TradingService, "__init__", lambda self, *a, **kw: None):
        svc = TradingService.__new__(TradingService)

    # Wire up mocks
    svc._auth = MagicMock()
    svc._auth.is_authenticated = authenticated

    svc._market_data = AsyncMock()
    if live_df is not None:
        svc._market_data.get_historical_df = AsyncMock(return_value=live_df)
    else:
        svc._market_data.get_historical_df = AsyncMock(return_value=pd.DataFrame())

    svc._data_store = MagicMock()
    if cached_candles is not None:
        svc._data_store.get_candles = MagicMock(return_value=cached_candles)
    else:
        svc._data_store.get_candles = MagicMock(return_value=[])

    return svc


# ─── Tier 1: Live data ──────────────────────────────────────

class TestTier1LiveData:
    """When authenticated and Kite returns enough bars → data_source='zerodha'."""

    @pytest.mark.asyncio
    async def test_live_data_returned_when_authenticated(self):
        live = _make_ohlcv_df(600)
        svc = _build_service(authenticated=True, live_df=live)

        df, source, err = await svc._resolve_sample_data(bars=500)
        assert err is None
        assert source == "zerodha"
        assert len(df) == 500

    @pytest.mark.asyncio
    async def test_live_data_too_few_bars_falls_through(self):
        """If live returns fewer bars than needed, fall to tier 2/3."""
        live = _make_ohlcv_df(100)  # not enough for bars=500
        svc = _build_service(authenticated=True, live_df=live,
                             cached_candles=_make_candle_rows(600))

        df, source, err = await svc._resolve_sample_data(bars=500)
        assert err is None
        assert source == "db_cache"  # fell through to tier 2
        assert len(df) == 500

    @pytest.mark.asyncio
    async def test_live_exception_falls_through(self):
        """If Kite API throws, fall to tier 2/3."""
        svc = _build_service(authenticated=True,
                             cached_candles=_make_candle_rows(600))
        svc._market_data.get_historical_df = AsyncMock(side_effect=Exception("API error"))

        df, source, err = await svc._resolve_sample_data(bars=500)
        assert err is None
        assert source == "db_cache"
        assert len(df) == 500


# ─── Tier 2: DB cache ───────────────────────────────────────

class TestTier2DBCache:
    """When not authenticated but DB has cached candles → data_source='db_cache'."""

    @pytest.mark.asyncio
    async def test_db_cache_used_when_not_authenticated(self):
        candles = _make_candle_rows(600)
        svc = _build_service(authenticated=False, cached_candles=candles)

        df, source, err = await svc._resolve_sample_data(bars=500)
        assert err is None
        assert source == "db_cache"
        assert len(df) == 500

    @pytest.mark.asyncio
    async def test_db_cache_too_few_candles_no_synthetic(self):
        """DB has some candles but not enough, and allow_synthetic=False → error."""
        candles = _make_candle_rows(100)
        svc = _build_service(authenticated=False, cached_candles=candles)

        df, source, err = await svc._resolve_sample_data(bars=500, allow_synthetic=False)
        assert err is not None
        assert "allow_synthetic" in err
        assert df.empty

    @pytest.mark.asyncio
    async def test_db_cache_too_few_candles_with_synthetic(self):
        """DB cache not enough + allow_synthetic=True → falls to synthetic."""
        candles = _make_candle_rows(100)
        svc = _build_service(authenticated=False, cached_candles=candles)

        df, source, err = await svc._resolve_sample_data(
            bars=500, allow_synthetic=True, base_price=100.0, style="equity",
        )
        assert err is None
        assert source == "synthetic"
        assert len(df) >= 500


# ─── Tier 3: Synthetic ──────────────────────────────────────

class TestTier3Synthetic:
    """When no live and no cache → synthetic depends on allow_synthetic flag."""

    @pytest.mark.asyncio
    async def test_no_data_deny_synthetic(self):
        svc = _build_service(authenticated=False, cached_candles=[])

        df, source, err = await svc._resolve_sample_data(bars=500, allow_synthetic=False)
        assert err is not None
        assert "No live or cached market data" in err
        assert "allow_synthetic" in err
        assert source == "none"
        assert df.empty

    @pytest.mark.asyncio
    async def test_no_data_allow_synthetic(self):
        svc = _build_service(authenticated=False, cached_candles=[])

        df, source, err = await svc._resolve_sample_data(
            bars=500, allow_synthetic=True, base_price=100.0, style="equity",
        )
        assert err is None
        assert source == "synthetic"
        assert len(df) == 500

    @pytest.mark.asyncio
    async def test_synthetic_index_style(self):
        svc = _build_service(authenticated=False, cached_candles=[])

        df, source, err = await svc._resolve_sample_data(
            bars=200, allow_synthetic=True, base_price=20000.0, style="index",
        )
        assert err is None
        assert source == "synthetic"
        assert len(df) == 200
        # Index-style data should have values near base_price
        assert df["close"].iloc[0] > 10000


# ─── Tier priority ordering ─────────────────────────────────

class TestTierPriority:
    """Verify that tier 1 > tier 2 > tier 3."""

    @pytest.mark.asyncio
    async def test_live_beats_cache(self):
        """If both live and cache available, live wins."""
        live = _make_ohlcv_df(600)
        candles = _make_candle_rows(600)
        svc = _build_service(authenticated=True, live_df=live, cached_candles=candles)

        df, source, err = await svc._resolve_sample_data(bars=500)
        assert source == "zerodha"
        # DB cache should NOT be queried
        svc._data_store.get_candles.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_beats_synthetic(self):
        """If cache available but not authenticated, cache wins over synthetic."""
        candles = _make_candle_rows(600)
        svc = _build_service(authenticated=False, cached_candles=candles)

        df, source, err = await svc._resolve_sample_data(
            bars=500, allow_synthetic=True,
        )
        assert source == "db_cache"  # NOT synthetic


# ─── Integration: sample methods honour resolver ─────────────

class TestSampleMethodsIntegration:
    """Test that the 4 sample methods properly use _resolve_sample_data."""

    def _patch_resolver(self, svc, df, source, err=None):
        """Patch _resolve_sample_data on a service to return fixed values."""
        svc._resolve_sample_data = AsyncMock(return_value=(df, source, err))

    @pytest.mark.asyncio
    async def test_paper_trade_sample_deny_synthetic(self):
        from src.api.service import TradingService
        with patch.object(TradingService, "__init__", lambda self, *a, **kw: None):
            svc = TradingService.__new__(TradingService)

        svc._resolve_sample_data = AsyncMock(
            return_value=(pd.DataFrame(), "none", "No live or cached market data available. Set allow_synthetic=true to run with synthetic data.")
        )
        svc._resolve_strategy = MagicMock(return_value=MagicMock())  # fake strategy
        svc._paper_results = {}
        svc._tested_strategies = set()

        result = await svc.run_paper_trade_with_sample_data("ema_crossover", allow_synthetic=False)
        assert "error" in result
        assert result["needs_synthetic_confirmation"] is True

    @pytest.mark.asyncio
    async def test_backtest_sample_deny_synthetic(self):
        from src.api.service import TradingService
        with patch.object(TradingService, "__init__", lambda self, *a, **kw: None):
            svc = TradingService.__new__(TradingService)

        svc._resolve_sample_data = AsyncMock(
            return_value=(pd.DataFrame(), "none", "No data. Set allow_synthetic=true.")
        )
        svc._resolve_strategy = MagicMock(return_value=MagicMock())
        svc._backtest_results = {}

        result = await svc.run_backtest_sample("ema_crossover", allow_synthetic=False)
        assert "error" in result
        assert result["needs_synthetic_confirmation"] is True

    @pytest.mark.asyncio
    async def test_fno_backtest_sample_deny_synthetic(self):
        from src.api.service import TradingService
        with patch.object(TradingService, "__init__", lambda self, *a, **kw: None):
            svc = TradingService.__new__(TradingService)

        svc._resolve_sample_data = AsyncMock(
            return_value=(pd.DataFrame(), "none", "No data. Set allow_synthetic=true.")
        )
        svc._is_valid_fno_strategy = MagicMock(return_value=True)
        svc.FNO_STRATEGIES = ["iron_condor"]

        result = await svc.run_fno_backtest_sample("iron_condor", allow_synthetic=False)
        assert "error" in result
        assert result["needs_synthetic_confirmation"] is True

    @pytest.mark.asyncio
    async def test_fno_paper_trade_sample_deny_synthetic(self):
        from src.api.service import TradingService
        with patch.object(TradingService, "__init__", lambda self, *a, **kw: None):
            svc = TradingService.__new__(TradingService)

        svc._resolve_sample_data = AsyncMock(
            return_value=(pd.DataFrame(), "none", "No data. Set allow_synthetic=true.")
        )
        svc._is_valid_fno_strategy = MagicMock(return_value=True)
        svc.FNO_STRATEGIES = ["iron_condor"]

        result = await svc.run_fno_paper_trade_sample("iron_condor", allow_synthetic=False)
        assert "error" in result
        assert result["needs_synthetic_confirmation"] is True


# ─── Interval handling ───────────────────────────────────────

class TestIntervalHandling:
    """Verify resolver handles different intervals correctly."""

    @pytest.mark.asyncio
    async def test_5minute_interval(self):
        svc = _build_service(authenticated=False, cached_candles=[])
        df, source, err = await svc._resolve_sample_data(
            bars=200, interval="5minute", allow_synthetic=True,
            base_price=100.0, style="equity",
        )
        assert err is None
        assert source == "synthetic"
        assert len(df) == 200

    @pytest.mark.asyncio
    async def test_day_interval(self):
        svc = _build_service(authenticated=False, cached_candles=[])
        df, source, err = await svc._resolve_sample_data(
            bars=200, interval="day", allow_synthetic=True,
            base_price=100.0, style="equity",
        )
        assert err is None
        assert source == "synthetic"
        assert len(df) == 200

    @pytest.mark.asyncio
    async def test_15minute_interval(self):
        svc = _build_service(authenticated=False, cached_candles=[])
        df, source, err = await svc._resolve_sample_data(
            bars=100, interval="15minute", allow_synthetic=True,
            base_price=100.0, style="equity",
        )
        assert err is None
        assert source == "synthetic"
        assert len(df) == 100


# ─── Edge cases ──────────────────────────────────────────────

class TestEdgeCases:
    """Edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_no_data_store(self):
        """If _data_store is None, skip tier 2 gracefully."""
        svc = _build_service(authenticated=False)
        svc._data_store = None

        df, source, err = await svc._resolve_sample_data(
            bars=100, allow_synthetic=True, base_price=100.0,
        )
        assert err is None
        assert source == "synthetic"

    @pytest.mark.asyncio
    async def test_db_cache_exception_handled(self):
        """If db_cache throws, falls to tier 3."""
        svc = _build_service(authenticated=False)
        svc._data_store.get_candles = MagicMock(side_effect=Exception("DB corrupted"))

        df, source, err = await svc._resolve_sample_data(
            bars=100, allow_synthetic=True, base_price=100.0,
        )
        assert err is None
        assert source == "synthetic"

    @pytest.mark.asyncio
    async def test_data_source_in_result_key(self):
        """Resolver always populates data_source string."""
        svc = _build_service(authenticated=False, cached_candles=[])

        _, source_deny, err = await svc._resolve_sample_data(bars=10, allow_synthetic=False)
        assert source_deny == "none"

        _, source_allow, err2 = await svc._resolve_sample_data(
            bars=10, allow_synthetic=True, base_price=100.0,
        )
        assert source_allow == "synthetic"

    @pytest.mark.asyncio
    async def test_default_allow_synthetic_is_false(self):
        """By default, allow_synthetic should be False (require confirmation)."""
        svc = _build_service(authenticated=False, cached_candles=[])

        # Call without allow_synthetic → should default to False → error
        df, source, err = await svc._resolve_sample_data(bars=100)
        assert err is not None
        assert "allow_synthetic" in err
