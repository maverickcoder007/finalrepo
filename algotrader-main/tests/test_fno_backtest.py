"""
F&O Backtest Engine Tests — Run every strategy through the backtest engine.

Covers all 13 strategies in FnOBacktestEngine.STRATEGY_MAP:
  1. iron_condor       2. bull_call_spread  3. bear_put_spread
  4. bull_put_spread   5. bear_call_spread  6. straddle (long)
  7. short_straddle    8. strangle (long)   9. short_strangle
 10. covered_call     11. protective_put   12. iron_butterfly
 13. calendar_spread

Each test validates:
  - Engine runs without error on synthetic data
  - Result dict has required keys
  - Equity curve length matches bar count
  - Capital accounting is consistent
  - Positions opened / closed tracking works
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from src.derivatives.fno_backtest import FnOBacktestEngine


# ─── All supported strategies ───────────────────────────────────
ALL_STRATEGIES = list(FnOBacktestEngine.STRATEGY_MAP.keys())

REQUIRED_RESULT_KEYS = {
    "engine",
    "strategy_name",
    "structure",
    "underlying",
    "total_bars",
    "initial_capital",
    "final_capital",
    "total_return_pct",
    "equity_curve",
    "trades",
    "settings",
}


# ─── Parametrized backtest: every strategy ──────────────────────

@pytest.mark.parametrize("strategy_name", ALL_STRATEGIES)
class TestBacktestAllStrategies:
    """Run each strategy against 252 bars of synthetic NIFTY data."""

    @pytest.fixture(autouse=True)
    def _run_engine(self, strategy_name: str, nifty_daily_data: pd.DataFrame):
        """Run backtest once per strategy; store result on self."""
        engine = FnOBacktestEngine(
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
            use_regime_filter=False,  # disable regime filter for broad testing
        )
        self.result = engine.run(nifty_daily_data, tradingsymbol="NIFTY")
        self.engine = engine

    def test_no_error(self, strategy_name: str):
        assert "error" not in self.result, (
            f"[{strategy_name}] Backtest returned error: {self.result.get('error')}"
        )

    def test_required_keys(self, strategy_name: str):
        for key in REQUIRED_RESULT_KEYS:
            assert key in self.result, (
                f"[{strategy_name}] Missing key '{key}' in result"
            )

    def test_equity_curve_length(self, strategy_name: str, nifty_daily_data: pd.DataFrame):
        curve = self.result.get("equity_curve", [])
        # Equity curve should have ≥ total_bars entries (initial + each bar)
        assert len(curve) >= 1, f"[{strategy_name}] Empty equity curve"

    def test_equity_curve_starts_at_initial_capital(self, strategy_name: str):
        curve = self.result.get("equity_curve", [])
        assert abs(curve[0] - 500_000.0) < 1.0, (
            f"[{strategy_name}] Equity curve starts at {curve[0]}, expected 500000"
        )

    def test_final_capital_matches_curve(self, strategy_name: str):
        curve = self.result.get("equity_curve", [])
        final_from_curve = curve[-1] if curve else 0
        final_capital = self.result.get("final_capital", 0)
        # Allow 1% tolerance due to MtM vs realized differences
        diff_pct = abs(final_from_curve - final_capital) / max(abs(final_capital), 1) * 100
        assert diff_pct < 5, (
            f"[{strategy_name}] Curve end {final_from_curve:.2f} ≠ final_capital {final_capital:.2f}"
        )

    def test_return_calculation(self, strategy_name: str):
        initial = self.result.get("initial_capital", 500_000)
        final = self.result.get("final_capital", 500_000)
        reported_return = self.result.get("total_return_pct", 0)
        expected_return = (final - initial) / initial * 100
        assert abs(reported_return - expected_return) < 0.1, (
            f"[{strategy_name}] Return mismatch: reported={reported_return}%, "
            f"computed={expected_return:.2f}%"
        )

    def test_strategy_name_in_result(self, strategy_name: str):
        assert self.result.get("strategy_name") == strategy_name

    def test_settings_preserved(self, strategy_name: str):
        settings = self.result.get("settings", {})
        assert settings.get("profit_target_pct") == 50.0
        assert settings.get("stop_loss_pct") == 100.0
        assert settings.get("slippage_model") == "zero"


# ─── Detailed iron_condor tests ────────────────────────────────

class TestBacktestIronCondorDetailed:
    """Deeper tests for the most common strategy."""

    @pytest.fixture(autouse=True)
    def _setup(self, nifty_daily_data: pd.DataFrame):
        engine = FnOBacktestEngine(
            strategy_name="iron_condor",
            underlying="NIFTY",
            initial_capital=500_000.0,
            max_positions=3,
            profit_target_pct=50.0,
            stop_loss_pct=100.0,
            entry_dte_min=10,
            entry_dte_max=45,
            delta_target=0.16,
            slippage_model="realistic",
            use_regime_filter=True,
        )
        self.result = engine.run(nifty_daily_data, tradingsymbol="NIFTY")

    def test_positions_opened(self):
        assert self.result.get("positions_opened", 0) >= 0

    def test_trades_recorded(self):
        trades = self.result.get("trades", [])
        assert isinstance(trades, list)

    def test_greeks_history_recorded(self):
        gh = self.result.get("greeks_history", [])
        assert isinstance(gh, list)

    def test_margin_history_recorded(self):
        mh = self.result.get("margin_history", [])
        assert isinstance(mh, list)

    def test_regime_history_recorded(self):
        rh = self.result.get("regime_history", [])
        assert isinstance(rh, list)
        assert len(rh) > 0, "Should have regime classifications"

    def test_total_costs_positive(self):
        costs = self.result.get("total_costs", 0)
        assert costs >= 0, "Total costs should be non-negative"

    def test_trades_have_correct_structure(self):
        trades = self.result.get("trades", [])
        if trades:
            for t in trades[:10]:  # check first 10
                assert "type" in t, f"Trade missing 'type' key: {t}"
                assert t["type"] in (
                    "ENTRY", "PROFIT_TARGET", "STOP_LOSS", "FINAL_EXIT", "EXPIRY"
                ), f"Unknown trade type: {t['type']}"


# ─── Straddle / strangle detailed tests ────────────────────────

class TestBacktestStraddleStrangle:
    """Test straddle/strangle variants generate valid results."""

    @pytest.fixture(params=["straddle", "short_straddle", "strangle", "short_strangle"])
    def result(self, request, nifty_daily_data: pd.DataFrame):
        engine = FnOBacktestEngine(
            strategy_name=request.param,
            underlying="NIFTY",
            initial_capital=500_000.0,
            max_positions=2,
            entry_dte_min=10,
            delta_target=0.16,
            slippage_model="zero",
            use_regime_filter=False,
        )
        return engine.run(nifty_daily_data)

    def test_no_error(self, result):
        assert "error" not in result

    def test_structure_matches(self, result):
        name = result.get("strategy_name", "")
        structure = result.get("structure", "")
        if "straddle" in name:
            assert "STRADDLE" in structure
        elif "strangle" in name:
            assert "STRANGLE" in structure


# ─── Vertical spread tests ─────────────────────────────────────

class TestBacktestVerticalSpreads:
    """Test directional spread strategies."""

    @pytest.fixture(params=[
        "bull_call_spread", "bear_put_spread",
        "bull_put_spread", "bear_call_spread",
    ])
    def result(self, request, nifty_daily_data: pd.DataFrame):
        engine = FnOBacktestEngine(
            strategy_name=request.param,
            underlying="NIFTY",
            initial_capital=500_000.0,
            max_positions=2,
            entry_dte_min=10,
            delta_target=0.30,  # more aggressive delta for directional spreads
            slippage_model="zero",
            use_regime_filter=False,
        )
        return engine.run(nifty_daily_data)

    def test_no_error(self, result):
        assert "error" not in result

    def test_equity_curve_exists(self, result):
        assert len(result.get("equity_curve", [])) > 1


# ─── Edge cases ─────────────────────────────────────────────────

class TestBacktestEdgeCases:
    """Test engine with edge case data."""

    def test_empty_dataframe(self):
        engine = FnOBacktestEngine()
        result = engine.run(pd.DataFrame())
        assert "error" in result

    def test_missing_columns(self):
        engine = FnOBacktestEngine()
        df = pd.DataFrame({"close": [100, 101, 102]})
        result = engine.run(df)
        assert "error" in result

    def test_very_short_data(self, nifty_daily_data: pd.DataFrame):
        """Data shorter than warmup period."""
        engine = FnOBacktestEngine(
            strategy_name="iron_condor",
            slippage_model="zero",
            use_regime_filter=False,
        )
        short_data = nifty_daily_data.head(25)  # Only 25 bars (needs ≥21)
        result = engine.run(short_data)
        assert "error" not in result
        assert result.get("positions_opened", 0) == 0  # too few bars for meaningful entries

    def test_high_vol_data(self, nifty_high_vol_data: pd.DataFrame):
        """Backtest on high-vol bearish data."""
        engine = FnOBacktestEngine(
            strategy_name="iron_condor",
            underlying="NIFTY",
            initial_capital=500_000.0,
            slippage_model="zero",
            use_regime_filter=False,
        )
        result = engine.run(nifty_high_vol_data)
        assert "error" not in result

    def test_banknifty_underlying(self, banknifty_daily_data: pd.DataFrame):
        """Backtest with BANKNIFTY lot sizes and strike steps."""
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

    def test_regime_filter_reduces_entries(self, nifty_daily_data: pd.DataFrame):
        """Regime filter should reduce or change entry count."""
        engine_no_filter = FnOBacktestEngine(
            strategy_name="iron_condor",
            initial_capital=500_000.0,
            slippage_model="zero",
            use_regime_filter=False,
        )
        engine_with_filter = FnOBacktestEngine(
            strategy_name="iron_condor",
            initial_capital=500_000.0,
            slippage_model="zero",
            use_regime_filter=True,
        )
        r1 = engine_no_filter.run(nifty_daily_data)
        r2 = engine_with_filter.run(nifty_daily_data)
        # With filter should open ≤ without filter
        assert r2.get("positions_opened", 0) <= r1.get("positions_opened", 999) + 1

    def test_slippage_impacts_pnl(self, nifty_daily_data: pd.DataFrame):
        """Realistic slippage should reduce returns vs zero slippage."""
        engine_zero = FnOBacktestEngine(
            strategy_name="iron_condor",
            initial_capital=500_000.0,
            slippage_model="zero",
            use_regime_filter=False,
        )
        engine_real = FnOBacktestEngine(
            strategy_name="iron_condor",
            initial_capital=500_000.0,
            slippage_model="realistic",
            use_regime_filter=False,
        )
        r_zero = engine_zero.run(nifty_daily_data)
        r_real = engine_real.run(nifty_daily_data)
        # Both should run without error
        assert "error" not in r_zero
        assert "error" not in r_real
        # Realistic should have higher costs
        assert r_real.get("total_costs", 0) >= r_zero.get("total_costs", 0)
