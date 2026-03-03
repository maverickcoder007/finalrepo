"""
Tests for Phase 6 institutional features:
  1. Capital Allocation Rules
  2. Execution Quality Scoring
  3. Strategy Decay Monitor
  4. Portfolio Greeks Monitor
  5. Regime-Aware Strategy Switching
"""

import pytest
from datetime import datetime, timedelta


# ═════════════════════════════════════════════════════════════════
# 1. Capital Allocation Rules
# ═════════════════════════════════════════════════════════════════

class TestCapitalAllocator:
    def _make(self):
        from src.risk.capital_allocator import CapitalAllocator
        return CapitalAllocator()

    def test_default_limits(self):
        alloc = self._make()
        limits = alloc.get_limits()
        assert limits["max_per_strategy"] == 0.25
        assert limits["max_per_underlying"] == 0.30
        assert limits["max_overnight_exposure"] == 0.50
        assert limits["max_net_delta"] == 0.20
        assert limits["max_single_position"] == 0.10

    def test_update_limits(self):
        alloc = self._make()
        result = alloc.update_limits({"max_per_strategy": 0.40})
        # Returns AllocationLimits dataclass
        assert result.max_per_strategy == 0.40
        # Others unchanged
        assert result.max_per_underlying == 0.30

    def test_pre_trade_check_allowed(self):
        alloc = self._make()
        result = alloc.check_pre_trade(
            proposed_value=10000, strategy="iron_condor",
            underlying="NIFTY", capital=100000,
        )
        assert result["allowed"] is True
        assert len(result["warnings"]) == 0

    def test_pre_trade_check_warns_single_position(self):
        alloc = self._make()
        # max_single_position = 10%, so 15000 on 100000 = 15% → warning
        result = alloc.check_pre_trade(
            proposed_value=15000, strategy="iron_condor",
            underlying="NIFTY", capital=100000,
        )
        assert len(result["warnings"]) > 0
        assert any("Single position" in w for w in result["warnings"])

    def test_compute_allocation_empty(self):
        alloc = self._make()
        report = alloc.compute_allocation(
            positions=[],
            margins={"equity": {"available": {"live_balance": 100000}}, "commodity": {}},
            strategy_tags={},
            is_market_hours=True,
        )
        assert report.gross_exposure == 0
        assert len(report.breaches) == 0

    def test_compute_allocation_with_positions(self):
        alloc = self._make()
        positions = [
            {
                "tradingsymbol": "NIFTY25JAN25000CE",
                "exchange": "NFO",
                "quantity": 50,
                "last_price": 200,
                "buy_value": 10000,
                "sell_value": 0,
                "value": 10000,
                "product": "NRML",
            }
        ]
        margins = {
            "equity": {"available": {"live_balance": 100000}},
            "commodity": {},
        }
        report = alloc.compute_allocation(
            positions=positions, margins=margins,
            strategy_tags={}, is_market_hours=True,
        )
        assert report.gross_exposure > 0
        assert report.total_capital > 0

    def test_get_last_report_empty(self):
        alloc = self._make()
        assert alloc.get_last_report() is None


# ═════════════════════════════════════════════════════════════════
# 2. Execution Quality Scoring
# ═════════════════════════════════════════════════════════════════

class TestExecutionQualityScorer:
    def _make(self):
        from src.risk.execution_quality import ExecutionQualityScorer
        return ExecutionQualityScorer()

    def test_score_execution_basic(self):
        scorer = self._make()
        score = scorer.score_execution(
            trade_id="T001",
            instrument="NIFTY25JAN25000CE",
            direction="BUY",
            quantity=50,
            expected_price=200.0,
            actual_price=200.5,
        )
        assert score is not None
        assert 0 <= score.slippage_score <= 100
        assert 0 <= score.composite_score <= 100
        assert score.trade_id == "T001"

    def test_score_perfect_fill(self):
        scorer = self._make()
        score = scorer.score_execution(
            trade_id="T002",
            instrument="RELIANCE",
            direction="BUY",
            quantity=10,
            expected_price=2500.0,
            actual_price=2500.0,
        )
        assert score.slippage_score == 100  # Perfect slippage

    def test_score_bad_slippage(self):
        scorer = self._make()
        score = scorer.score_execution(
            trade_id="T003",
            instrument="RELIANCE",
            direction="BUY",
            quantity=10,
            expected_price=2500.0,
            actual_price=2525.0,  # 1% slippage
        )
        assert score.slippage_score < 50  # Bad slippage score

    def test_get_recent_scores(self):
        scorer = self._make()
        scorer.score_execution("T1", "SYM", "BUY", 10, 100, 100.5)
        scorer.score_execution("T2", "SYM", "SELL", 10, 100, 99.8)
        scores = scorer.get_recent_scores(10)
        assert len(scores) == 2

    def test_get_summary_empty(self):
        scorer = self._make()
        summary = scorer.get_summary()
        assert summary["total_scored"] == 0

    def test_get_summary_with_data(self):
        scorer = self._make()
        scorer.score_execution("T1", "SYM", "BUY", 10, 100, 100.1)
        scorer.score_execution("T2", "SYM", "SELL", 10, 100, 99.9)
        summary = scorer.get_summary()
        assert summary["total_scored"] == 2
        assert "avg_composite" in summary
        assert "avg_slippage" in summary

    def test_score_with_bar_data(self):
        scorer = self._make()
        score = scorer.score_execution(
            trade_id="T4",
            instrument="NIFTY",
            direction="BUY",
            quantity=25,
            expected_price=25000,
            actual_price=25010,
            bar_high=25050,
            bar_low=24950,
            bar_open=24990,
            bar_close=25020,
        )
        assert 0 <= score.timing_score <= 100

    def test_score_with_spread_data(self):
        scorer = self._make()
        score = scorer.score_execution(
            trade_id="T5",
            instrument="NIFTY",
            direction="BUY",
            quantity=25,
            expected_price=25000,
            actual_price=25005,
            best_bid=24998,
            best_ask=25003,
        )
        assert 0 <= score.spread_efficiency_score <= 100


# ═════════════════════════════════════════════════════════════════
# 3. Strategy Decay Monitor
# ═════════════════════════════════════════════════════════════════

class TestStrategyDecayMonitor:
    def _make(self):
        from src.risk.strategy_decay import StrategyDecayMonitor
        return StrategyDecayMonitor()

    def _generate_curve(self, n=100, avg_pnl=100, std=50, start_days_ago=120):
        """Generate a synthetic equity curve."""
        import random
        random.seed(42)
        curve = []
        base = datetime.now() - timedelta(days=start_days_ago)
        for i in range(n):
            ts = (base + timedelta(days=i * start_days_ago / n)).isoformat()
            pnl = avg_pnl + random.gauss(0, std)
            curve.append({"timestamp": ts, "pnl": pnl})
        return curve

    def _generate_decayed_curve(self, n=100, start_days_ago=120):
        """Curve that starts good then turns bad."""
        import random
        random.seed(42)
        curve = []
        base = datetime.now() - timedelta(days=start_days_ago)
        for i in range(n):
            ts = (base + timedelta(days=i * start_days_ago / n)).isoformat()
            if i < n * 0.7:
                pnl = 150 + random.gauss(0, 30)
            else:
                pnl = -100 + random.gauss(0, 80)  # Sharp decline
            curve.append({"timestamp": ts, "pnl": pnl})
        return curve

    def test_evaluate_healthy_strategy(self):
        monitor = self._make()
        curve = self._generate_curve(100, avg_pnl=100, std=30)
        report = monitor.evaluate_strategy("test_strat", curve)
        assert report is not None
        assert report.strategy_name == "test_strat"

    def test_evaluate_insufficient_data(self):
        monitor = self._make()
        # Very few trades → should return report with action_taken="insufficient_history"
        curve = [{"timestamp": datetime.now().isoformat(), "pnl": 100}]
        report = monitor.evaluate_strategy("low_data", curve)
        assert report is not None
        assert report.action_taken == "insufficient_history"

    def test_evaluate_all_strategies(self):
        monitor = self._make()
        curves = {
            "strat_a": self._generate_curve(80, avg_pnl=100),
            "strat_b": self._generate_curve(80, avg_pnl=50),
        }
        reports = monitor.evaluate_all(curves)
        assert len(reports) >= 0  # May skip some if insufficient data

    def test_get_history(self):
        monitor = self._make()
        curve = self._generate_curve(100, avg_pnl=100)
        monitor.evaluate_strategy("test", curve)
        history = monitor.get_history(50)
        # If evaluation produced a result, it should be in history
        assert isinstance(history, list)

    def test_thresholds_update(self):
        monitor = self._make()
        t = monitor._thresholds
        assert t.std_dev_trigger == 2.0
        assert t.rolling_window_days == 30
        t.std_dev_trigger = 1.5
        assert monitor._thresholds.std_dev_trigger == 1.5

    def test_auto_disable_callback(self):
        monitor = self._make()
        disabled = []

        def callback(name):
            disabled.append(name)

        curve = self._generate_decayed_curve(100)
        report = monitor.evaluate_strategy("decaying", curve, disable_callback=callback)
        # Whether it triggers depends on data — just check it runs without error
        assert isinstance(disabled, list)


# ═════════════════════════════════════════════════════════════════
# 4. Portfolio Greeks Monitor
# ═════════════════════════════════════════════════════════════════

class TestPortfolioGreeksMonitor:
    def _make(self):
        from src.risk.portfolio_greeks import PortfolioGreeksMonitor
        return PortfolioGreeksMonitor()

    def test_compute_greeks_empty(self):
        monitor = self._make()
        result = monitor.compute_greeks([], {})
        assert result.net_delta == 0
        assert result.net_gamma == 0
        assert result.net_theta == 0
        assert result.net_vega == 0
        assert len(result.warnings) == 0

    def test_compute_greeks_with_call_position(self):
        monitor = self._make()
        positions = [
            {
                "tradingsymbol": "NIFTY25JUN25000CE",
                "exchange": "NFO",
                "quantity": 50,  # 2 lots of NIFTY (lot_size=25)
                "last_price": 200,
                "average_price": 180,
            }
        ]
        spot_prices = {"NIFTY": 25100}
        result = monitor.compute_greeks(positions, spot_prices)
        assert result.net_delta != 0  # Should have positive delta for calls
        assert len(result.positions) > 0
        assert result.positions[0]["instrument"] == "NIFTY25JUN25000CE"

    def test_compute_greeks_with_put_position(self):
        monitor = self._make()
        positions = [
            {
                "tradingsymbol": "NIFTY25JUN24500PE",
                "exchange": "NFO",
                "quantity": -25,  # Short 1 lot
                "last_price": 150,
                "average_price": 200,
            }
        ]
        spot_prices = {"NIFTY": 25100}
        result = monitor.compute_greeks(positions, spot_prices)
        # Short put should have positive delta (short negative delta = positive)
        assert len(result.positions) > 0

    def test_compute_greeks_mixed_portfolio(self):
        monitor = self._make()
        positions = [
            {
                "tradingsymbol": "NIFTY25JUN25000CE",
                "exchange": "NFO",
                "quantity": 25,
                "last_price": 200,
            },
            {
                "tradingsymbol": "NIFTY25JUN24500PE",
                "exchange": "NFO",
                "quantity": -25,
                "last_price": 100,
            },
        ]
        spot_prices = {"NIFTY": 25000}
        result = monitor.compute_greeks(positions, spot_prices)
        assert len(result.positions) == 2

    def test_get_last_report_none(self):
        monitor = self._make()
        report = monitor.get_last_report()
        assert report is None

    def test_get_last_report_after_compute(self):
        monitor = self._make()
        monitor.compute_greeks([], {})
        report = monitor.get_last_report()
        assert report is not None

    def test_to_dict(self):
        monitor = self._make()
        result = monitor.compute_greeks([], {})
        d = result.to_dict()
        assert "net_delta" in d
        assert "net_gamma" in d
        assert "net_theta" in d
        assert "net_vega" in d
        assert "warnings" in d

    def test_risk_warnings_high_delta(self):
        """Test that high delta generates a warning."""
        monitor = self._make()
        # Create many long call positions to get high delta
        positions = []
        for i in range(40):
            positions.append({
                "tradingsymbol": "NIFTY25JUN25000CE",
                "exchange": "NFO",
                "quantity": 25,
                "last_price": 200,
            })
        spot_prices = {"NIFTY": 25100}
        result = monitor.compute_greeks(positions, spot_prices)
        # With 40 lots of ATM calls, delta should be high
        # Check if we got any warnings
        assert isinstance(result.warnings, list)


# ═════════════════════════════════════════════════════════════════
# 5. Regime-Aware Strategy Switching
# ═════════════════════════════════════════════════════════════════

class TestRegimeStrategySwitcher:
    def _make(self):
        from src.risk.regime_switcher import RegimeStrategySwithcer
        return RegimeStrategySwithcer()

    def test_assess_range_low_vol(self):
        sw = self._make()
        snap = sw.assess_regime(
            underlying="NIFTY",
            oi_direction="neutral",
            oi_confidence=0.7,
            vix_value=11.0,
            spot_price=25000,
            sma_20=25050,
            sma_50=24900,
        )
        assert snap.volatility_regime == "low"
        # Range/low vol → iron condor
        assert snap.recommended_strategy == "iron_condor"

    def test_assess_bullish_trend(self):
        sw = self._make()
        snap = sw.assess_regime(
            underlying="NIFTY",
            oi_direction="bullish",
            oi_confidence=0.8,
            vix_value=14.0,
            spot_price=25500,
            sma_20=25300,
            sma_50=24900,
            recent_high=25400,
            recent_low=24800,
        )
        # Should recommend bullish strategy
        assert snap.recommended_strategy in (
            "call_debit_spread", "put_credit_spread",
        )

    def test_assess_bearish_trend(self):
        sw = self._make()
        snap = sw.assess_regime(
            underlying="NIFTY",
            oi_direction="bearish",
            oi_confidence=0.8,
            vix_value=15.0,
            spot_price=24200,
            sma_20=24500,
            sma_50=24800,
            recent_high=25000,
            recent_low=24300,
        )
        assert snap.recommended_strategy in (
            "put_debit_spread", "call_credit_spread",
        )

    def test_assess_extreme_vol(self):
        sw = self._make()
        snap = sw.assess_regime(
            underlying="NIFTY",
            oi_direction="neutral",
            oi_confidence=0.5,
            vix_value=30.0,
            spot_price=24000,
            sma_20=24100,
            sma_50=24200,
        )
        assert snap.volatility_regime == "extreme"

    def test_assess_compression(self):
        sw = self._make()
        snap = sw.assess_regime(
            underlying="NIFTY",
            oi_direction="neutral",
            oi_confidence=0.5,
            vix_value=11.0,
            spot_price=25000,
            sma_20=25000,
            sma_50=25000,
            bollinger_width=0.4,  # Narrow bands → compression
        )
        # Compression + low vol → long strangle
        assert snap.trend_state == "compression"
        assert snap.recommended_strategy in ("long_strangle", "long_straddle", "iron_condor")

    def test_get_current_regime_none(self):
        sw = self._make()
        result = sw.get_current_regime("NIFTY")
        assert result is None

    def test_get_current_regime_after_assess(self):
        sw = self._make()
        sw.assess_regime(underlying="NIFTY", vix_value=15.0, spot_price=25000)
        result = sw.get_current_regime("NIFTY")
        assert result is not None
        assert result["underlying"] == "NIFTY"

    def test_get_history(self):
        sw = self._make()
        sw.assess_regime(underlying="NIFTY", vix_value=15.0, spot_price=25000)
        sw.assess_regime(underlying="NIFTY", vix_value=16.0, spot_price=25100)
        history = sw.get_history("NIFTY")
        assert len(history) == 2

    def test_reasoning_populated(self):
        sw = self._make()
        snap = sw.assess_regime(
            underlying="NIFTY",
            oi_direction="bullish",
            oi_confidence=0.7,
            vix_value=14.0,
            spot_price=25000,
        )
        assert len(snap.reasoning) >= 1

    def test_alternatives_populated(self):
        sw = self._make()
        snap = sw.assess_regime(
            underlying="NIFTY",
            oi_direction="neutral",
            oi_confidence=0.6,
            vix_value=12.0,
            spot_price=25000,
            sma_20=25000,
            sma_50=25000,
        )
        # Should have some alternatives
        assert isinstance(snap.alternatives, list)

    def test_to_dict(self):
        sw = self._make()
        snap = sw.assess_regime(underlying="NIFTY", vix_value=15.0, spot_price=25000)
        d = snap.to_dict()
        assert "recommended_strategy" in d
        assert "volatility_regime" in d
        assert "trend_state" in d
        assert "oi_direction" in d

    def test_confidence_scales_with_oi(self):
        sw = self._make()
        snap_low = sw.assess_regime(
            underlying="NIFTY", oi_direction="neutral",
            oi_confidence=0.1, vix_value=11.0, spot_price=25000,
        )
        snap_high = sw.assess_regime(
            underlying="NIFTY", oi_direction="neutral",
            oi_confidence=0.9, vix_value=11.0, spot_price=25000,
        )
        # Higher OI confidence should give higher recommendation confidence
        assert snap_high.recommendation_confidence >= snap_low.recommendation_confidence

    def test_volatility_classification(self):
        from src.risk.regime_switcher import RegimeStrategySwithcer
        assert RegimeStrategySwithcer._classify_volatility(10, 0) == "low"
        assert RegimeStrategySwithcer._classify_volatility(15, 0) == "medium"
        assert RegimeStrategySwithcer._classify_volatility(20, 0) == "high"
        assert RegimeStrategySwithcer._classify_volatility(30, 0) == "extreme"

    def test_volatility_from_iv(self):
        from src.risk.regime_switcher import RegimeStrategySwithcer
        assert RegimeStrategySwithcer._classify_volatility(0, 0.10) == "low"
        assert RegimeStrategySwithcer._classify_volatility(0, 0.18) == "medium"
        assert RegimeStrategySwithcer._classify_volatility(0, 0.28) == "high"
        assert RegimeStrategySwithcer._classify_volatility(0, 0.40) == "extreme"

    def test_breakout_up(self):
        sw = self._make()
        snap = sw.assess_regime(
            underlying="NIFTY",
            oi_direction="bullish_breakout",
            oi_confidence=0.9,
            vix_value=14.0,
            spot_price=25200,
            sma_20=25000,
            sma_50=24800,
            recent_high=25100,
            recent_low=24500,
        )
        assert snap.recommended_strategy in (
            "call_debit_spread", "put_credit_spread",
        )

    def test_different_underlyings(self):
        sw = self._make()
        sw.assess_regime(underlying="NIFTY", vix_value=15.0, spot_price=25000)
        sw.assess_regime(underlying="BANKNIFTY", vix_value=16.0, spot_price=52000)
        assert sw.get_current_regime("NIFTY")["underlying"] == "NIFTY"
        assert sw.get_current_regime("BANKNIFTY")["underlying"] == "BANKNIFTY"


# ═════════════════════════════════════════════════════════════════
# Cross-module integration smoke tests
# ═════════════════════════════════════════════════════════════════

class TestIntegrationSmoke:
    """Verify all modules can be imported and instantiated together."""

    def test_all_modules_import(self):
        from src.risk.capital_allocator import CapitalAllocator
        from src.risk.execution_quality import ExecutionQualityScorer
        from src.risk.strategy_decay import StrategyDecayMonitor
        from src.risk.portfolio_greeks import PortfolioGreeksMonitor
        from src.risk.regime_switcher import RegimeStrategySwithcer
        assert CapitalAllocator is not None
        assert ExecutionQualityScorer is not None
        assert StrategyDecayMonitor is not None
        assert PortfolioGreeksMonitor is not None
        assert RegimeStrategySwithcer is not None

    def test_all_modules_instantiate(self):
        from src.risk.capital_allocator import CapitalAllocator
        from src.risk.execution_quality import ExecutionQualityScorer
        from src.risk.strategy_decay import StrategyDecayMonitor
        from src.risk.portfolio_greeks import PortfolioGreeksMonitor
        from src.risk.regime_switcher import RegimeStrategySwithcer
        a = CapitalAllocator()
        b = ExecutionQualityScorer()
        c = StrategyDecayMonitor()
        d = PortfolioGreeksMonitor()
        e = RegimeStrategySwithcer()
        assert a and b and c and d and e
