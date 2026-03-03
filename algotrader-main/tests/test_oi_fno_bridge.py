"""
Tests for OI → FNO Bridge: signal mapping, backtest validation, execution gating.
"""

from __future__ import annotations

import asyncio
import pytest

from src.options.oi_fno_bridge import (
    OIFnOBridge,
    BacktestThresholds,
    FnOStrategyType,
    StrategyMapping,
    BacktestResult,
    SignalExecutionPlan,
    _SIGNAL_TYPE_STRATEGY_OVERRIDE,
    _ACTION_STRATEGY_MAP,
    _DIRECTION_STRATEGY_MAP,
)
from src.options.oi_strategy import OISignal, OISignalType, OIStrategyAction


# ─── Fixtures ────────────────────────────────────────────────

def _make_signal(
    signal_type: str = OISignalType.PCR_EXTREME_BEARISH.value,
    action: str = OIStrategyAction.BUY_PE.value,
    direction: str = "BEARISH",
    confidence: float = 75.0,
    underlying: str = "NIFTY",
    strike: float = 24500.0,
    option_type: str = "PE",
    entry_price: float = 150.0,
    signal_id: str = "",
) -> OISignal:
    return OISignal(
        id=signal_id or f"test_{signal_type}_{int(confidence)}",
        timestamp="2025-01-15T10:30:00",
        underlying=underlying,
        signal_type=signal_type,
        action=action,
        direction=direction,
        confidence=confidence,
        strike=strike,
        option_type=option_type,
        entry_price=entry_price,
        stop_loss=round(entry_price * 0.7, 2),
        target=round(entry_price * 1.5, 2),
        lot_size=65 if underlying == "NIFTY" else 20,
        lots=1,
        exchange="NFO" if underlying == "NIFTY" else "BFO",
        tradingsymbol=f"{underlying}25JAN15{int(strike)}{option_type}",
        spot_price=24400.0 if underlying == "NIFTY" else 79000.0,
        atm_strike=24400.0 if underlying == "NIFTY" else 79000.0,
        pcr_oi=0.42 if direction == "BEARISH" else 1.6,
        reasons=[f"Test signal: {signal_type}"],
    )


# ─── 1. Signal → Strategy Mapping Tests ─────────────────────


class TestSignalMapping:
    """Test that OI signals map to correct FNO strategies."""

    def test_bearish_pcr_maps_to_call_credit_spread(self):
        bridge = OIFnOBridge()
        sig = _make_signal(
            signal_type=OISignalType.PCR_EXTREME_BEARISH.value,
            direction="BEARISH",
        )
        mapping = bridge.map_signal_to_strategy(sig)
        assert mapping.fno_strategy == FnOStrategyType.CALL_CREDIT_SPREAD.value

    def test_bullish_pcr_maps_to_put_credit_spread(self):
        bridge = OIFnOBridge()
        sig = _make_signal(
            signal_type=OISignalType.PCR_EXTREME_BULLISH.value,
            action=OIStrategyAction.BUY_CE.value,
            direction="BULLISH",
        )
        mapping = bridge.map_signal_to_strategy(sig)
        assert mapping.fno_strategy == FnOStrategyType.PUT_CREDIT_SPREAD.value

    def test_call_writing_resistance_maps_to_call_credit_spread(self):
        bridge = OIFnOBridge()
        sig = _make_signal(
            signal_type=OISignalType.CALL_WRITING_RESISTANCE.value,
            action=OIStrategyAction.SELL_CE.value,
            direction="BEARISH",
        )
        mapping = bridge.map_signal_to_strategy(sig)
        assert mapping.fno_strategy == FnOStrategyType.CALL_CREDIT_SPREAD.value

    def test_put_writing_support_maps_to_put_credit_spread(self):
        bridge = OIFnOBridge()
        sig = _make_signal(
            signal_type=OISignalType.PUT_WRITING_SUPPORT.value,
            action=OIStrategyAction.SELL_PE.value,
            direction="BULLISH",
        )
        mapping = bridge.map_signal_to_strategy(sig)
        assert mapping.fno_strategy == FnOStrategyType.PUT_CREDIT_SPREAD.value

    def test_iv_skew_maps_to_iron_condor(self):
        bridge = OIFnOBridge()
        sig = _make_signal(
            signal_type=OISignalType.IV_SKEW_REVERSAL.value,
            action=OIStrategyAction.SELL_PE.value,
            direction="BULLISH",
        )
        mapping = bridge.map_signal_to_strategy(sig)
        assert mapping.fno_strategy == FnOStrategyType.IRON_CONDOR.value

    def test_breakout_up_maps_to_bull_call_spread(self):
        bridge = OIFnOBridge()
        sig = _make_signal(
            signal_type=OISignalType.OI_WALL_BREAKOUT_UP.value,
            action=OIStrategyAction.BUY_CE.value,
            direction="BULLISH",
        )
        mapping = bridge.map_signal_to_strategy(sig)
        assert mapping.fno_strategy == FnOStrategyType.BULL_CALL_SPREAD.value

    def test_breakout_down_maps_to_bear_put_spread(self):
        bridge = OIFnOBridge()
        sig = _make_signal(
            signal_type=OISignalType.OI_WALL_BREAKOUT_DOWN.value,
            action=OIStrategyAction.BUY_PE.value,
            direction="BEARISH",
        )
        mapping = bridge.map_signal_to_strategy(sig)
        assert mapping.fno_strategy == FnOStrategyType.BEAR_PUT_SPREAD.value

    def test_combined_bearish_maps_to_call_credit_spread(self):
        bridge = OIFnOBridge()
        sig = _make_signal(
            signal_type=OISignalType.COMBINED_BEARISH.value,
            direction="BEARISH",
        )
        mapping = bridge.map_signal_to_strategy(sig)
        assert mapping.fno_strategy == FnOStrategyType.CALL_CREDIT_SPREAD.value

    def test_combined_bullish_maps_to_put_credit_spread(self):
        bridge = OIFnOBridge()
        sig = _make_signal(
            signal_type=OISignalType.COMBINED_BULLISH.value,
            action=OIStrategyAction.BUY_CE.value,
            direction="BULLISH",
        )
        mapping = bridge.map_signal_to_strategy(sig)
        assert mapping.fno_strategy == FnOStrategyType.PUT_CREDIT_SPREAD.value

    def test_low_confidence_returns_none_strategy(self):
        bridge = OIFnOBridge(thresholds=BacktestThresholds(min_signal_confidence=70.0))
        sig = _make_signal(confidence=55.0)
        mapping = bridge.map_signal_to_strategy(sig)
        assert mapping.fno_strategy == FnOStrategyType.NONE.value
        assert "below minimum" in mapping.strategy_rationale

    def test_mapping_includes_alternatives(self):
        bridge = OIFnOBridge()
        sig = _make_signal(
            signal_type=OISignalType.PCR_EXTREME_BEARISH.value,
            direction="BEARISH",
        )
        mapping = bridge.map_signal_to_strategy(sig)
        assert len(mapping.alternative_strategies) > 0

    def test_mapping_derives_strategy_params(self):
        bridge = OIFnOBridge()
        sig = _make_signal(underlying="SENSEX", strike=79000.0)
        mapping = bridge.map_signal_to_strategy(sig)
        assert mapping.strategy_params.get("spread_width") == 300
        assert mapping.strategy_params.get("underlying") == "SENSEX"

    def test_all_signal_types_have_mapping(self):
        """Every OI signal type should have a strategy mapping."""
        bridge = OIFnOBridge()
        for sig_type in OISignalType:
            sig = _make_signal(
                signal_type=sig_type.value,
                direction="BEARISH" if "BEARISH" in sig_type.value else "BULLISH",
                confidence=80.0,
            )
            mapping = bridge.map_signal_to_strategy(sig)
            assert mapping.fno_strategy != FnOStrategyType.NONE.value, (
                f"Signal type {sig_type.value} has no strategy mapping"
            )


# ─── 2. Backtest Evaluation Tests ───────────────────────────


class TestBacktestEvaluation:
    """Test backtest result evaluation against thresholds."""

    def test_passing_backtest(self):
        bridge = OIFnOBridge()
        mapping = StrategyMapping(
            fno_strategy="call_credit_spread_runner",
            underlying="NIFTY",
        )
        raw = {
            "win_rate": 60.0,
            "sharpe_ratio": 1.2,
            "max_drawdown_pct": -10.0,
            "total_return_pct": 15.0,
            "positions_closed": 20,
            "profit_factor": 1.8,
            "avg_trade_pnl": 5000.0,
            "trades": [{"pnl": 5000}] * 15 + [{"pnl": -2000}] * 5,
            "initial_capital": 500000,
        }
        result = bridge._evaluate_backtest(raw, mapping)
        assert result.passed is True
        assert result.win_rate == 60.0
        assert result.sharpe_ratio == 1.2
        assert len(result.failure_reasons) == 0

    def test_failing_backtest_low_win_rate(self):
        bridge = OIFnOBridge()
        mapping = StrategyMapping(
            fno_strategy="call_credit_spread_runner",
            underlying="NIFTY",
        )
        raw = {
            "win_rate": 30.0,
            "sharpe_ratio": 0.2,
            "max_drawdown_pct": -35.0,
            "total_return_pct": -5.0,
            "positions_closed": 3,
            "profit_factor": 0.5,
            "avg_trade_pnl": -3000.0,
            "trades": [{"pnl": -5000}] * 3,
            "initial_capital": 500000,
        }
        result = bridge._evaluate_backtest(raw, mapping)
        assert result.passed is False
        assert len(result.failure_reasons) > 0

    def test_custom_thresholds_stricter(self):
        bridge = OIFnOBridge(thresholds=BacktestThresholds(
            min_win_rate=70.0,
            min_sharpe_ratio=2.0,
        ))
        mapping = StrategyMapping(
            fno_strategy="put_credit_spread_runner",
            underlying="NIFTY",
        )
        raw = {
            "win_rate": 60.0,  # Fails 70% threshold
            "sharpe_ratio": 1.5,  # Fails 2.0 threshold
            "max_drawdown_pct": -10.0,
            "total_return_pct": 15.0,
            "positions_closed": 20,
            "profit_factor": 1.8,
            "avg_trade_pnl": 5000.0,
            "trades": [{"pnl": 5000}] * 12 + [{"pnl": -2000}] * 8,
            "initial_capital": 500000,
        }
        result = bridge._evaluate_backtest(raw, mapping)
        assert result.passed is False
        # Should fail on win_rate and sharpe
        assert any("win_rate" in r for r in result.failure_reasons)
        assert any("sharpe_ratio" in r for r in result.failure_reasons)

    def test_insufficient_trades_fails(self):
        bridge = OIFnOBridge(thresholds=BacktestThresholds(min_trades=10))
        mapping = StrategyMapping(fno_strategy="iron_condor", underlying="NIFTY")
        raw = {
            "win_rate": 80.0,
            "sharpe_ratio": 2.0,
            "max_drawdown_pct": -5.0,
            "total_return_pct": 25.0,
            "positions_closed": 3,  # Too few
            "profit_factor": 2.5,
            "avg_trade_pnl": 10000.0,
            "trades": [{"pnl": 10000}] * 3,
            "initial_capital": 500000,
        }
        result = bridge._evaluate_backtest(raw, mapping)
        assert result.passed is False
        assert any("total_trades" in r for r in result.failure_reasons)


# ─── 3. Risk Warnings Tests ─────────────────────────────────


class TestRiskWarnings:
    """Test risk warning generation."""

    def test_low_confidence_warning(self):
        bridge = OIFnOBridge()
        sig = _make_signal(confidence=62.0)
        mapping = bridge.map_signal_to_strategy(sig)
        warnings = bridge._assess_risk_warnings(sig, mapping)
        assert any("moderate" in w.lower() or "confidence" in w.lower() for w in warnings)

    def test_iv_skew_warning(self):
        bridge = OIFnOBridge()
        sig = _make_signal(
            signal_type=OISignalType.IV_SKEW_REVERSAL.value,
            direction="BULLISH",
        )
        mapping = bridge.map_signal_to_strategy(sig)
        warnings = bridge._assess_risk_warnings(sig, mapping)
        assert any("IV" in w for w in warnings)

    def test_credit_spread_risk_warning(self):
        bridge = OIFnOBridge()
        sig = _make_signal(
            signal_type=OISignalType.CALL_WRITING_RESISTANCE.value,
            direction="BEARISH",
        )
        mapping = bridge.map_signal_to_strategy(sig)
        warnings = bridge._assess_risk_warnings(sig, mapping)
        assert any("credit spread" in w.lower() or "Credit spread" in w for w in warnings)


# ─── 4. Execution Plan Tests ────────────────────────────────


class TestExecutionPlan:
    """Test execution plan creation with backtest gating."""

    @pytest.mark.asyncio
    async def test_plan_no_backtest_skip(self):
        """Plan with backtest skipped should still approve if signal is valid."""
        bridge = OIFnOBridge(thresholds=BacktestThresholds(require_backtest=False))
        sig = _make_signal(confidence=80.0)
        plan = await bridge.create_execution_plan(sig)
        assert plan.execution_approved is True
        assert plan.mapping["fno_strategy"] != "none"

    @pytest.mark.asyncio
    async def test_plan_no_data_fetcher_fails_backtest(self):
        """If backtest is required but no data fetcher, plan should fail."""
        bridge = OIFnOBridge(thresholds=BacktestThresholds(require_backtest=True))
        sig = _make_signal(confidence=80.0)
        plan = await bridge.create_execution_plan(sig, market_data_fetcher=None)
        assert plan.execution_approved is False
        assert any("backtest" in r.lower() or "Backtest" in r for r in plan.rejection_reasons)

    @pytest.mark.asyncio
    async def test_plan_with_mock_passing_backtest(self):
        """Plan with mock passing backtest should approve execution."""
        import pandas as pd
        import numpy as np

        async def mock_data_fetcher(underlying, days):
            np.random.seed(42)
            n = max(days, 100)
            prices = np.cumsum(np.random.randn(n) * 50) + 24000
            df = pd.DataFrame({
                "open": prices + np.random.rand(n) * 10,
                "high": prices + np.abs(np.random.randn(n)) * 30,
                "low": prices - np.abs(np.random.randn(n)) * 30,
                "close": prices,
                "volume": np.random.randint(100000, 500000, n),
            })
            return df, 256265

        # Use very lenient thresholds so synthetic data can pass
        bridge = OIFnOBridge(thresholds=BacktestThresholds(
            require_backtest=True,
            min_win_rate=0.0,
            min_sharpe_ratio=-10.0,
            max_drawdown_pct=100.0,
            min_total_return_pct=-100.0,
            min_trades=0,
            min_profit_factor=0.0,
            max_avg_loss_pct=100.0,
            backtest_days=100,
        ))
        sig = _make_signal(confidence=80.0)
        plan = await bridge.create_execution_plan(
            sig, market_data_fetcher=mock_data_fetcher
        )
        # Should get as far as having a backtest (may pass/fail)
        assert plan.mapping["fno_strategy"] != "none"
        # With ultra-lenient thresholds, should approve
        assert plan.execution_approved is True


# ─── 5. Batch Processing Tests ──────────────────────────────


class TestBatchProcessing:
    """Test batch signal processing."""

    @pytest.mark.asyncio
    async def test_batch_filters_low_confidence(self):
        bridge = OIFnOBridge(thresholds=BacktestThresholds(
            min_signal_confidence=70.0,
            require_backtest=False,
        ))
        signals = [
            _make_signal(confidence=80.0, signal_id="sig_1"),
            _make_signal(confidence=50.0, signal_id="sig_2"),  # Below threshold
            _make_signal(confidence=75.0, signal_id="sig_3"),
        ]
        plans = await bridge.process_all_signals(signals)
        assert len(plans) == 2  # sig_2 filtered out

    @pytest.mark.asyncio
    async def test_batch_sorted_by_approval_then_confidence(self):
        bridge = OIFnOBridge(thresholds=BacktestThresholds(
            require_backtest=False,
            min_signal_confidence=60.0,
        ))
        signals = [
            _make_signal(confidence=65.0, signal_id="low_conf"),
            _make_signal(confidence=90.0, signal_id="high_conf"),
            _make_signal(confidence=75.0, signal_id="mid_conf"),
        ]
        plans = await bridge.process_all_signals(signals)
        assert len(plans) == 3
        # All approved, sorted by confidence descending
        confidences = [p.signal.get("confidence", 0) for p in plans]
        assert confidences == sorted(confidences, reverse=True)


# ─── 6. Threshold Management Tests ──────────────────────────


class TestThresholdManagement:
    def test_update_thresholds(self):
        bridge = OIFnOBridge()
        original_wr = bridge.thresholds.min_win_rate
        bridge.update_thresholds({"min_win_rate": 55.0})
        assert bridge.thresholds.min_win_rate == 55.0
        assert bridge.thresholds.min_win_rate != original_wr

    def test_update_thresholds_clears_cache(self):
        bridge = OIFnOBridge()
        # Simulate cached result
        bridge._backtest_cache["test:NIFTY"] = BacktestResult(passed=True)
        bridge.update_thresholds({"min_win_rate": 55.0})
        assert len(bridge._backtest_cache) == 0

    def test_get_thresholds_returns_all_fields(self):
        bridge = OIFnOBridge()
        t = bridge.get_thresholds()
        assert "min_win_rate" in t
        assert "min_sharpe_ratio" in t
        assert "max_drawdown_pct" in t
        assert "require_backtest" in t
        assert "backtest_days" in t


# ─── 7. Mapping Info Tests ──────────────────────────────────


class TestMappingInfo:
    def test_mapping_info_structure(self):
        bridge = OIFnOBridge()
        info = bridge.get_strategy_mapping_info()
        assert "direction_map" in info
        assert "action_map" in info
        assert "signal_type_overrides" in info
        assert "BEARISH" in info["direction_map"]
        assert "BULLISH" in info["direction_map"]

    def test_all_oi_actions_mapped(self):
        """Every OI strategy action should have a mapping."""
        for action in OIStrategyAction:
            assert action.value in _ACTION_STRATEGY_MAP, (
                f"Action {action.value} has no strategy mapping"
            )


# ─── 8. Complete Pipeline Integration Test ──────────────────


class TestFullPipeline:
    """Integration test: signal → map → evaluate → plan."""

    @pytest.mark.asyncio
    async def test_bearish_signal_full_pipeline(self):
        """BEARISH signal → call credit spread → plan created."""
        bridge = OIFnOBridge(thresholds=BacktestThresholds(require_backtest=False))

        # Create bearish signal
        sig = _make_signal(
            signal_type=OISignalType.CALL_WRITING_RESISTANCE.value,
            action=OIStrategyAction.SELL_CE.value,
            direction="BEARISH",
            confidence=78.0,
        )

        # Map
        mapping = bridge.map_signal_to_strategy(sig)
        assert mapping.fno_strategy == "call_credit_spread_runner"
        assert mapping.signal_direction == "BEARISH"

        # Create plan
        plan = await bridge.create_execution_plan(sig)
        assert plan.execution_approved is True
        assert plan.mapping["fno_strategy"] == "call_credit_spread_runner"
        assert len(plan.risk_warnings) > 0  # Should have credit spread warning

    @pytest.mark.asyncio
    async def test_bullish_signal_full_pipeline(self):
        """BULLISH signal → put credit spread → plan created."""
        bridge = OIFnOBridge(thresholds=BacktestThresholds(require_backtest=False))

        sig = _make_signal(
            signal_type=OISignalType.PUT_WRITING_SUPPORT.value,
            action=OIStrategyAction.SELL_PE.value,
            direction="BULLISH",
            confidence=72.0,
        )

        mapping = bridge.map_signal_to_strategy(sig)
        assert mapping.fno_strategy == "put_credit_spread_runner"

        plan = await bridge.create_execution_plan(sig)
        assert plan.execution_approved is True

    @pytest.mark.asyncio
    async def test_neutral_signal_maps_to_iron_condor(self):
        """NEUTRAL-ish signal (IV Skew / MaxPain) → iron condor."""
        bridge = OIFnOBridge(thresholds=BacktestThresholds(require_backtest=False))

        sig = _make_signal(
            signal_type=OISignalType.MAX_PAIN_MAGNET.value,
            action=OIStrategyAction.BUY_PE.value,
            direction="BEARISH",
            confidence=68.0,
        )

        mapping = bridge.map_signal_to_strategy(sig)
        assert mapping.fno_strategy == "iron_condor"
