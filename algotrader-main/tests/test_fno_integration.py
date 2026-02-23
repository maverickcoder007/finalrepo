"""
F&O Integration Tests — End-to-end flows combining multiple engines.

Tests cover:
  1. Data → Regime → Chain → Strategy → Execution → Result (full pipeline)
  2. Backtest vs Paper Trading consistency
  3. Cost model integrated with backtest
  4. Greeks tracking through a complete backtest
  5. Expiry handling during backtest
  6. Multi-underlying testing
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta

from src.derivatives.fno_backtest import FnOBacktestEngine
from src.derivatives.fno_paper_trader import FnOPaperTradingEngine, FnOPaperTradeResult
from src.derivatives.chain_builder import HistoricalChainBuilder
from src.derivatives.regime_engine import RegimeEngine, MarketRegime
from src.derivatives.greeks_engine import GreeksEngine
from src.derivatives.fno_cost_model import IndianFnOCostModel, TransactionSide
from src.derivatives.margin_engine import MarginEngine
from src.derivatives.contracts import (
    DerivativeContract,
    InstrumentType,
    MultiLegPosition,
    OptionLeg,
    StructureType,
)


# ═══════════════════════════════════════════════════════════
# 1. DATA → REGIME → CHAIN PIPELINE
# ═══════════════════════════════════════════════════════════

class TestDataToChainPipeline:
    """Test that we can go from raw OHLCV → regime → option chain."""

    def test_classify_regime_from_data(self, nifty_daily_data: pd.DataFrame):
        engine = RegimeEngine()
        closes = nifty_daily_data["close"].values
        result = engine.classify(closes)
        assert result.regime != MarketRegime.UNKNOWN or len(closes) < 30
        assert result.hv_20 > 0

    def test_build_chain_from_data(self, nifty_daily_data: pd.DataFrame):
        builder = HistoricalChainBuilder("NIFTY")
        spot = float(nifty_daily_data["close"].iloc[-1])
        expiry = date.today() + timedelta(days=15)
        chain = builder.build_chain(
            spot=spot,
            timestamp=datetime.now(),
            expiry=expiry,
            hv=0.16,
        )
        assert chain is not None
        assert chain.spot_price == spot
        assert len(chain.strikes) > 10

    def test_regime_drives_strategy_recommendation(self, nifty_daily_data: pd.DataFrame):
        engine = RegimeEngine()
        closes = nifty_daily_data["close"].values
        result = engine.classify(closes)
        assert isinstance(result.recommended_structures, list)
        if result.regime != MarketRegime.UNKNOWN:
            assert len(result.recommended_structures) > 0

    def test_chain_has_valid_greeks(self, nifty_daily_data: pd.DataFrame):
        builder = HistoricalChainBuilder("NIFTY")
        spot = float(nifty_daily_data["close"].iloc[-1])
        expiry = date.today() + timedelta(days=20)
        chain = builder.build_chain(
            spot=spot,
            timestamp=datetime.now(),
            expiry=expiry,
            hv=0.16,
        )
        # Check ATM call has reasonable delta
        atm = chain.atm_strike
        if atm in chain.strikes and "CE" in chain.strikes[atm]:
            ce = chain.strikes[atm]["CE"]
            assert 0.3 < ce.delta < 0.7, f"ATM call delta={ce.delta}"
            assert ce.gamma > 0
            assert ce.theta < 0
            assert ce.vega > 0


# ═══════════════════════════════════════════════════════════
# 2. BACKTEST vs PAPER TRADE CONSISTENCY
# ═══════════════════════════════════════════════════════════

class TestBacktestVsPaperTradeConsistency:
    """Both engines on same data should produce comparable structures."""

    def test_both_engines_run_same_strategy(self, nifty_daily_data: pd.DataFrame):
        bt_engine = FnOBacktestEngine(
            strategy_name="iron_condor",
            initial_capital=500_000.0,
            slippage_model="zero",
            use_regime_filter=False,
        )
        pt_engine = FnOPaperTradingEngine(
            strategy_name="iron_condor",
            initial_capital=500_000.0,
            slippage_model="zero",
        )

        bt_result = bt_engine.run(nifty_daily_data)
        pt_result = pt_engine.run(nifty_daily_data)

        # Both should succeed
        assert "error" not in bt_result
        assert isinstance(pt_result, FnOPaperTradeResult)

        # Both should have equity curves
        assert len(bt_result.get("equity_curve", [])) > 1
        assert len(pt_result.equity_curve) > 1

    def test_both_engines_start_at_same_capital(self, nifty_daily_data: pd.DataFrame):
        bt = FnOBacktestEngine(
            strategy_name="iron_condor",
            initial_capital=750_000.0,
            slippage_model="zero",
            use_regime_filter=False,
        )
        pt = FnOPaperTradingEngine(
            strategy_name="iron_condor",
            initial_capital=750_000.0,
            slippage_model="zero",
        )

        bt_result = bt.run(nifty_daily_data)
        pt_result = pt.run(nifty_daily_data)

        assert bt_result.get("initial_capital") == 750_000.0
        assert pt_result.initial_capital == 750_000.0

    def test_multiple_strategies_both_engines(self, nifty_daily_data: pd.DataFrame):
        """Run 3 strategies through both engines and verify they complete."""
        for strategy in ["iron_condor", "bull_call_spread", "short_straddle"]:
            bt = FnOBacktestEngine(
                strategy_name=strategy,
                initial_capital=500_000.0,
                slippage_model="zero",
                use_regime_filter=False,
            )
            pt = FnOPaperTradingEngine(
                strategy_name=strategy,
                initial_capital=500_000.0,
                slippage_model="zero",
            )

            bt_result = bt.run(nifty_daily_data)
            pt_result = pt.run(nifty_daily_data)

            assert "error" not in bt_result, f"Backtest failed for {strategy}"
            assert isinstance(pt_result, FnOPaperTradeResult), f"Paper trade failed for {strategy}"


# ═══════════════════════════════════════════════════════════
# 3. COST MODEL INTEGRATION
# ═══════════════════════════════════════════════════════════

class TestCostModelIntegration:
    """Verify costs are properly deducted during backtest."""

    def test_backtest_with_costs(self, nifty_daily_data: pd.DataFrame):
        engine = FnOBacktestEngine(
            strategy_name="iron_condor",
            initial_capital=500_000.0,
            slippage_model="realistic",
            use_regime_filter=False,
        )
        result = engine.run(nifty_daily_data)
        assert "error" not in result
        # If positions were opened, costs should be > 0
        if result.get("positions_opened", 0) > 0:
            assert result.get("total_costs", 0) > 0, (
                "Positions opened but no costs recorded"
            )

    def test_cost_model_round_trip(self):
        """Verify cost model works for a typical option round-trip."""
        model = IndianFnOCostModel()
        contract = DerivativeContract(
            symbol="NIFTY",
            tradingsymbol="NIFTY26MAR24500CE",
            instrument_type=InstrumentType.CE,
            exchange="NFO",
            strike=24500,
            expiry=date(2026, 3, 3),
            lot_size=65,
        )
        entry = model.calculate(contract, TransactionSide.BUY, 350.0, 1)
        exit_ = model.calculate(contract, TransactionSide.SELL, 400.0, 1)
        total = entry.total + exit_.total
        # Typical round-trip cost for 1 lot NIFTY option: ₹50-200
        assert 10 < total < 500, f"Round-trip cost {total:.2f} outside expected range"


# ═══════════════════════════════════════════════════════════
# 4. GREEKS TRACKING THROUGH BACKTEST
# ═══════════════════════════════════════════════════════════

class TestGreeksTracking:
    """Verify Greeks are tracked throughout a backtest."""

    def test_greeks_history_recorded(self, nifty_daily_data: pd.DataFrame):
        engine = FnOBacktestEngine(
            strategy_name="iron_condor",
            initial_capital=500_000.0,
            slippage_model="zero",
            use_regime_filter=False,
        )
        result = engine.run(nifty_daily_data)
        gh = result.get("greeks_history", [])
        if result.get("positions_opened", 0) > 0:
            assert len(gh) > 0, "Greeks history empty despite open positions"

    def test_greeks_snapshots_have_required_fields(self, nifty_daily_data: pd.DataFrame):
        engine = FnOBacktestEngine(
            strategy_name="iron_condor",
            initial_capital=500_000.0,
            slippage_model="zero",
            use_regime_filter=False,
        )
        result = engine.run(nifty_daily_data)
        gh = result.get("greeks_history", [])
        if gh:
            snap = gh[0]
            for key in ("net_delta", "net_gamma", "net_theta", "net_vega"):
                assert key in snap, f"Missing key '{key}' in Greeks snapshot"

    def test_iron_condor_near_zero_delta(self, nifty_daily_data: pd.DataFrame):
        """Iron condor should have relatively small net delta at entry."""
        engine = FnOBacktestEngine(
            strategy_name="iron_condor",
            initial_capital=500_000.0,
            slippage_model="zero",
            use_regime_filter=False,
        )
        result = engine.run(nifty_daily_data)
        gh = result.get("greeks_history", [])
        if gh:
            first_delta = abs(gh[0].get("net_delta", 0))
            # Iron condor delta should be relatively small
            assert first_delta < 5.0, (
                f"Iron condor initial delta {first_delta} too large"
            )


# ═══════════════════════════════════════════════════════════
# 5. EXPIRY HANDLING
# ═══════════════════════════════════════════════════════════

class TestExpiryHandling:
    """Verify expiry events are processed in backtest."""

    def test_expiry_trades_in_result(self, nifty_daily_data: pd.DataFrame):
        engine = FnOBacktestEngine(
            strategy_name="iron_condor",
            initial_capital=500_000.0,
            slippage_model="zero",
            use_regime_filter=False,
            entry_dte_min=5,   # Allow entries close to expiry
            entry_dte_max=15,
        )
        result = engine.run(nifty_daily_data)
        trades = result.get("trades", [])
        # Should have at least some ENTRY and EXIT/EXPIRY trades
        trade_types = set(t.get("type") for t in trades)
        if result.get("positions_opened", 0) > 0:
            assert "ENTRY" in trade_types, "No ENTRY trades despite positions opened"

    def test_all_positions_closed_at_end(self, nifty_daily_data: pd.DataFrame):
        engine = FnOBacktestEngine(
            strategy_name="iron_condor",
            initial_capital=500_000.0,
            slippage_model="zero",
            use_regime_filter=False,
        )
        result = engine.run(nifty_daily_data)
        # All positions should be closed at the end (force close remaining)
        closed = result.get("positions_closed", 0)
        opened = result.get("positions_opened", 0)
        assert closed == opened, (
            f"Not all positions closed: opened={opened}, closed={closed}"
        )


# ═══════════════════════════════════════════════════════════
# 6. MULTI-UNDERLYING TESTING
# ═══════════════════════════════════════════════════════════

class TestMultiUnderlying:
    """Test backtest with different underlyings."""

    def test_nifty_backtest(self, nifty_daily_data: pd.DataFrame):
        engine = FnOBacktestEngine(
            strategy_name="iron_condor",
            underlying="NIFTY",
            initial_capital=500_000.0,
            slippage_model="zero",
            use_regime_filter=False,
        )
        result = engine.run(nifty_daily_data, tradingsymbol="NIFTY")
        assert "error" not in result
        assert result.get("underlying") == "NIFTY"

    def test_banknifty_backtest(self, banknifty_daily_data: pd.DataFrame):
        engine = FnOBacktestEngine(
            strategy_name="iron_condor",
            underlying="BANKNIFTY",
            initial_capital=1_000_000.0,
            slippage_model="zero",
            use_regime_filter=False,
        )
        result = engine.run(banknifty_daily_data, tradingsymbol="BANKNIFTY")
        assert "error" not in result
        assert result.get("underlying") == "BANKNIFTY"

    def test_sensex_backtest(self, sensex_daily_data: pd.DataFrame):
        engine = FnOBacktestEngine(
            strategy_name="iron_condor",
            underlying="SENSEX",
            initial_capital=500_000.0,
            slippage_model="zero",
            use_regime_filter=False,
        )
        result = engine.run(sensex_daily_data, tradingsymbol="SENSEX")
        assert "error" not in result


# ═══════════════════════════════════════════════════════════
# 7. MARGIN INTEGRATION
# ═══════════════════════════════════════════════════════════

class TestMarginIntegration:
    """Test margin calculations during backtest."""

    def test_margin_history_recorded(self, nifty_daily_data: pd.DataFrame):
        engine = FnOBacktestEngine(
            strategy_name="iron_condor",
            initial_capital=500_000.0,
            slippage_model="zero",
            use_regime_filter=False,
        )
        result = engine.run(nifty_daily_data)
        mh = result.get("margin_history", [])
        if result.get("positions_opened", 0) > 0:
            assert len(mh) > 0, "Margin history empty despite positions"

    def test_margin_within_capital(self, nifty_daily_data: pd.DataFrame):
        engine = FnOBacktestEngine(
            strategy_name="iron_condor",
            initial_capital=500_000.0,
            slippage_model="zero",
            use_regime_filter=False,
        )
        result = engine.run(nifty_daily_data)
        mh = result.get("margin_history", [])
        for m in mh:
            margin = m.get("margin", m.get("margin_required", 0))
            # Margin should not exceed 3× initial capital (accounting for MtM gains)
            assert margin < 500_000 * 3, f"Margin {margin} seems unreasonably high"


# ═══════════════════════════════════════════════════════════
# 8. FULL PIPELINE SMOKE TEST
# ═══════════════════════════════════════════════════════════

class TestFullPipelineSmokeTest:
    """End-to-end smoke test: generate data → run all strategies → validate."""

    SMOKE_STRATEGIES = [
        "iron_condor",
        "bull_call_spread",
        "bear_put_spread",
        "short_straddle",
        "strangle",
        "iron_butterfly",
    ]

    @pytest.fixture(params=SMOKE_STRATEGIES)
    def backtest_result(self, request, nifty_daily_data: pd.DataFrame):
        engine = FnOBacktestEngine(
            strategy_name=request.param,
            initial_capital=500_000.0,
            slippage_model="zero",
            use_regime_filter=False,
        )
        return engine.run(nifty_daily_data)

    def test_no_error(self, backtest_result):
        assert "error" not in backtest_result

    def test_capital_not_negative(self, backtest_result):
        assert backtest_result.get("final_capital", 0) > 0, "Capital went negative"

    def test_equity_curve_reasonable(self, backtest_result):
        curve = backtest_result.get("equity_curve", [])
        if len(curve) > 1:
            # Max drawdown should not exceed 80%
            peak = curve[0]
            max_dd = 0
            for v in curve:
                peak = max(peak, v)
                dd = (peak - v) / peak * 100
                max_dd = max(max_dd, dd)
            assert max_dd < 80, f"Max drawdown {max_dd:.1f}% exceeds 80%"
