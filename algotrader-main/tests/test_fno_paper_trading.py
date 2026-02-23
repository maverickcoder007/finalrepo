"""
F&O Paper Trading Engine Tests — Validate all strategies in paper-trade mode.

Covers the 10 strategies in FnOPaperTradingEngine.STRATEGY_MAP and validates:
  - Engine runs without errors, returns FnOPaperTradeResult
  - Result structure and metrics
  - Order audit trail populated
  - Reset functionality
  - Edge cases (empty data, high vol, different underlyings)
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from src.derivatives.fno_paper_trader import (
    FnOPaperTradingEngine,
    FnOPaperTradeResult,
)


# ─── All supported paper-trade strategies ───────────────────────
ALL_PT_STRATEGIES = list(FnOPaperTradingEngine.STRATEGY_MAP.keys())

REQUIRED_RESULT_FIELDS = {
    "strategy_name",
    "underlying",
    "structure_type",
    "initial_capital",
    "final_capital",
    "total_return_pct",
    "total_pnl",
    "total_trades",
    "equity_curve",
    "orders",
}


# ─── Parametrized: every strategy ──────────────────────────────

@pytest.mark.parametrize("strategy_name", ALL_PT_STRATEGIES)
class TestPaperTradeAllStrategies:
    """Run each strategy against synthetic NIFTY data in paper mode."""

    @pytest.fixture(autouse=True)
    def _run_engine(self, strategy_name: str, nifty_daily_data: pd.DataFrame):
        engine = FnOPaperTradingEngine(
            strategy_name=strategy_name,
            underlying="NIFTY",
            initial_capital=500_000.0,
            max_positions=2,
            profit_target_pct=50.0,
            stop_loss_pct=100.0,
            entry_dte_min=10,
            entry_dte_max=45,
            delta_target=0.16,
            slippage_model="zero",
        )
        self.result = engine.run(nifty_daily_data, tradingsymbol="NIFTY")
        self.engine = engine

    def test_returns_correct_type(self, strategy_name: str):
        assert isinstance(self.result, FnOPaperTradeResult), (
            f"[{strategy_name}] Expected FnOPaperTradeResult, got {type(self.result)}"
        )

    def test_required_fields_present(self, strategy_name: str):
        d = self.result.to_dict()
        for key in REQUIRED_RESULT_FIELDS:
            assert key in d, f"[{strategy_name}] Missing field '{key}' in result dict"

    def test_equity_curve_populated(self, strategy_name: str):
        assert len(self.result.equity_curve) >= 1, (
            f"[{strategy_name}] Empty equity curve"
        )

    def test_equity_curve_starts_at_capital(self, strategy_name: str):
        curve = self.result.equity_curve
        assert abs(curve[0] - 500_000.0) < 1.0

    def test_return_consistent(self, strategy_name: str):
        expected_pct = (self.result.final_capital - self.result.initial_capital) / self.result.initial_capital * 100
        assert abs(self.result.total_return_pct - expected_pct) < 0.1, (
            f"[{strategy_name}] Return mismatch: {self.result.total_return_pct} vs {expected_pct:.2f}"
        )

    def test_pnl_equals_capital_change(self, strategy_name: str):
        expected_pnl = self.result.final_capital - self.result.initial_capital
        assert abs(self.result.total_pnl - expected_pnl) < 1.0

    def test_strategy_name_preserved(self, strategy_name: str):
        assert self.result.strategy_name == strategy_name

    def test_win_rate_range(self, strategy_name: str):
        assert 0 <= self.result.win_rate <= 100

    def test_sharpe_ratio_finite(self, strategy_name: str):
        assert np.isfinite(self.result.sharpe_ratio)


# ─── FnOPaperTradeResult serialization ─────────────────────────

class TestPaperTradeResultSerialization:
    """Test result to_dict / to_dict_safe."""

    @pytest.fixture
    def result(self, nifty_daily_data: pd.DataFrame):
        engine = FnOPaperTradingEngine(
            strategy_name="iron_condor",
            initial_capital=500_000.0,
            slippage_model="zero",
        )
        return engine.run(nifty_daily_data)

    def test_to_dict_keys(self, result: FnOPaperTradeResult):
        d = result.to_dict()
        assert d["engine"] == "fno_paper_trade"
        assert isinstance(d["orders"], list)
        assert isinstance(d["equity_curve"], list)

    def test_to_dict_safe(self, result: FnOPaperTradeResult):
        d = result.to_dict_safe()
        # Should be same as to_dict but with truncated lists
        assert "orders" in d
        assert len(d["orders"]) <= 200
        assert len(d["positions"]) <= 100

    def test_greeks_history_in_dict(self, result: FnOPaperTradeResult):
        d = result.to_dict()
        assert "greeks_history" in d
        assert isinstance(d["greeks_history"], list)

    def test_margin_history_in_dict(self, result: FnOPaperTradeResult):
        d = result.to_dict()
        assert "margin_history" in d
        assert isinstance(d["margin_history"], list)


# ─── Reset / state isolation ───────────────────────────────────

class TestPaperTradeReset:
    """Test that reset() properly clears state between runs."""

    def test_reset_clears_state(self, nifty_daily_data: pd.DataFrame):
        engine = FnOPaperTradingEngine(
            strategy_name="iron_condor",
            initial_capital=500_000.0,
            slippage_model="zero",
        )
        # First run
        result1 = engine.run(nifty_daily_data)
        # Second run (engine should reset internally)
        result2 = engine.run(nifty_daily_data)

        # Both runs should produce the same result (deterministic with zero slippage)
        assert abs(result1.final_capital - result2.final_capital) < 1.0
        assert len(result1.equity_curve) == len(result2.equity_curve)

    def test_explicit_reset(self, nifty_daily_data: pd.DataFrame):
        engine = FnOPaperTradingEngine(
            strategy_name="iron_condor",
            initial_capital=500_000.0,
            slippage_model="zero",
        )
        engine.run(nifty_daily_data)

        engine.reset()
        assert len(engine.open_positions) == 0
        assert len(engine.closed_positions) == 0
        assert len(engine._orders) == 0
        assert len(engine._equity_curve) == 0


# ─── Order audit trail ─────────────────────────────────────────

class TestPaperTradeOrderAudit:
    """Validate the order audit trail."""

    @pytest.fixture
    def result(self, nifty_daily_data: pd.DataFrame):
        engine = FnOPaperTradingEngine(
            strategy_name="iron_condor",
            initial_capital=500_000.0,
            max_positions=3,
            slippage_model="zero",
        )
        return engine.run(nifty_daily_data)

    def test_orders_have_correct_fields(self, result: FnOPaperTradeResult):
        for order in result.orders[:20]:
            assert "order_id" in order
            assert "tradingsymbol" in order
            assert "side" in order
            assert "lots" in order
            assert "fill_price" in order

    def test_order_prices_positive(self, result: FnOPaperTradeResult):
        for order in result.orders[:20]:
            if order.get("status") in ("COMPLETE", "EXERCISED"):
                assert order.get("fill_price", 0) >= 0


# ─── Edge cases ─────────────────────────────────────────────────

class TestPaperTradeEdgeCases:
    """Test paper trading with edge case data."""

    def test_empty_dataframe(self):
        engine = FnOPaperTradingEngine()
        result = engine.run(pd.DataFrame())
        assert isinstance(result, FnOPaperTradeResult)
        assert result.total_trades == 0

    def test_missing_columns_raises(self):
        engine = FnOPaperTradingEngine()
        df = pd.DataFrame({"close": [100, 101, 102]})
        with pytest.raises(ValueError, match="Missing columns"):
            engine.run(df)

    def test_high_vol_data(self, nifty_high_vol_data: pd.DataFrame):
        engine = FnOPaperTradingEngine(
            strategy_name="iron_condor",
            initial_capital=500_000.0,
            slippage_model="zero",
        )
        result = engine.run(nifty_high_vol_data)
        assert isinstance(result, FnOPaperTradeResult)

    def test_banknifty_paper_trade(self, banknifty_daily_data: pd.DataFrame):
        engine = FnOPaperTradingEngine(
            strategy_name="iron_condor",
            underlying="BANKNIFTY",
            initial_capital=1_000_000.0,
            slippage_model="zero",
        )
        result = engine.run(banknifty_daily_data, tradingsymbol="BANKNIFTY")
        assert isinstance(result, FnOPaperTradeResult)
        assert result.underlying == "BANKNIFTY"

    def test_short_data_no_crash(self, nifty_daily_data: pd.DataFrame):
        """Very short data should not crash, just produce no trades."""
        engine = FnOPaperTradingEngine(
            strategy_name="iron_condor",
            slippage_model="zero",
        )
        short = nifty_daily_data.head(25)
        result = engine.run(short)
        assert isinstance(result, FnOPaperTradeResult)


# ─── Straddle/strangle paper trade ─────────────────────────────

class TestPaperTradeStraddleStrangle:
    """Test straddle/strangle variants in paper mode."""

    @pytest.fixture(params=["straddle", "short_straddle", "strangle", "short_strangle"])
    def result(self, request, nifty_daily_data: pd.DataFrame):
        engine = FnOPaperTradingEngine(
            strategy_name=request.param,
            initial_capital=500_000.0,
            slippage_model="zero",
        )
        return engine.run(nifty_daily_data)

    def test_runs_successfully(self, result: FnOPaperTradeResult):
        assert isinstance(result, FnOPaperTradeResult)
        assert len(result.equity_curve) > 0


# ─── Vertical spreads paper trade ──────────────────────────────

class TestPaperTradeVerticalSpreads:
    """Test directional spread strategies in paper mode."""

    @pytest.fixture(params=[
        "bull_call_spread", "bear_put_spread",
        "bull_put_spread", "bear_call_spread",
    ])
    def result(self, request, nifty_daily_data: pd.DataFrame):
        engine = FnOPaperTradingEngine(
            strategy_name=request.param,
            initial_capital=500_000.0,
            delta_target=0.30,
            slippage_model="zero",
        )
        return engine.run(nifty_daily_data)

    def test_runs_successfully(self, result: FnOPaperTradeResult):
        assert isinstance(result, FnOPaperTradeResult)

    def test_equity_curve_exists(self, result: FnOPaperTradeResult):
        assert len(result.equity_curve) > 1


# ─── Metrics validation ────────────────────────────────────────

class TestPaperTradeMetrics:
    """Validate computed metrics (win rate, profit factor, drawdown, Sharpe)."""

    @pytest.fixture
    def result(self, nifty_daily_data: pd.DataFrame):
        engine = FnOPaperTradingEngine(
            strategy_name="iron_condor",
            initial_capital=500_000.0,
            max_positions=3,
            slippage_model="zero",
        )
        return engine.run(nifty_daily_data)

    def test_win_loss_counts_consistent(self, result: FnOPaperTradeResult):
        assert result.winning_trades + result.losing_trades <= result.total_trades

    def test_profit_factor_non_negative(self, result: FnOPaperTradeResult):
        assert result.profit_factor >= 0

    def test_max_drawdown_non_negative(self, result: FnOPaperTradeResult):
        assert result.max_drawdown_pct >= 0

    def test_avg_win_non_negative(self, result: FnOPaperTradeResult):
        if result.winning_trades > 0:
            assert result.avg_win >= 0

    def test_avg_loss_non_negative(self, result: FnOPaperTradeResult):
        if result.losing_trades > 0:
            assert result.avg_loss >= 0
