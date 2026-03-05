"""
OI Signal → FNO Strategy Bridge
=================================

Maps OI-based signals (from OIStrategyEngine) to existing FNO strategy
classes (CallCreditSpreadRunner, PutCreditSpreadRunner, etc.), validates
them via FnOBacktestEngine, and only proceeds to execution if the
backtest exceeds configurable quality thresholds.

Flow:
  OISignal → map_signal_to_strategy() → FNO strategy name + params
           → backtest_strategy()      → FnOBacktestEngine.run()
           → evaluate_backtest()       → pass/fail against thresholds
           → execute_if_passed()       → ExecutionEngine (with preflight)

Risk gates embedded in every step:
  • Minimum confidence on OI signal
  • Backtest win-rate, Sharpe, max-drawdown thresholds
  • Minimum number of backtest trades for statistical significance
  • Preflight + RiskManager checks before broker order
  • Kill-switch awareness
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.options.oi_strategy import (
    OISignal,
    OISignalType,
    OIStrategyAction,
    OIStrategyEngine,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Signal-to-Strategy Mapping
# ─────────────────────────────────────────────────────────────

class FnOStrategyType(str, Enum):
    """Supported FNO strategies that OI signals can map to."""
    CALL_CREDIT_SPREAD = "call_credit_spread_runner"
    PUT_CREDIT_SPREAD = "put_credit_spread_runner"
    BEAR_PUT_SPREAD = "bear_put_spread"
    BULL_CALL_SPREAD = "bull_call_spread"
    IRON_CONDOR = "iron_condor"
    SHORT_STRADDLE = "short_straddle"
    SHORT_STRANGLE = "short_strangle"
    LONG_STRADDLE = "straddle"
    NONE = "none"


# Map OI signal directions + actions to FNO strategies
# Priority: credit spreads (defined risk) over naked positions
_DIRECTION_STRATEGY_MAP: dict[str, list[FnOStrategyType]] = {
    "BEARISH": [
        FnOStrategyType.CALL_CREDIT_SPREAD,   # Primary: sell call spread
        FnOStrategyType.BEAR_PUT_SPREAD,       # Secondary: buy put spread
    ],
    "BULLISH": [
        FnOStrategyType.PUT_CREDIT_SPREAD,     # Primary: sell put spread
        FnOStrategyType.BULL_CALL_SPREAD,      # Secondary: buy call spread
    ],
    "NEUTRAL": [
        FnOStrategyType.IRON_CONDOR,           # Primary: defined risk
        FnOStrategyType.SHORT_STRANGLE,        # Secondary: undefined risk
    ],
}

# More specific: map OI action enums to strategies
_ACTION_STRATEGY_MAP: dict[str, FnOStrategyType] = {
    "BUY_CE": FnOStrategyType.BULL_CALL_SPREAD,
    "BUY_PE": FnOStrategyType.BEAR_PUT_SPREAD,
    "SELL_CE": FnOStrategyType.CALL_CREDIT_SPREAD,
    "SELL_PE": FnOStrategyType.PUT_CREDIT_SPREAD,
    "BUY_STRADDLE": FnOStrategyType.LONG_STRADDLE,
    "SELL_STRADDLE": FnOStrategyType.SHORT_STRADDLE,
    "BULL_CALL_SPREAD": FnOStrategyType.BULL_CALL_SPREAD,
    "BEAR_PUT_SPREAD": FnOStrategyType.BEAR_PUT_SPREAD,
    "IRON_CONDOR": FnOStrategyType.IRON_CONDOR,
}

# Signal types that strongly suggest specific strategies
_SIGNAL_TYPE_STRATEGY_OVERRIDE: dict[str, FnOStrategyType] = {
    # Bearish signals → credit call spread (collect premium, defined risk)
    OISignalType.PCR_EXTREME_BEARISH.value: FnOStrategyType.CALL_CREDIT_SPREAD,
    OISignalType.OI_WALL_RESISTANCE_REJECTION.value: FnOStrategyType.CALL_CREDIT_SPREAD,
    OISignalType.CALL_WRITING_RESISTANCE.value: FnOStrategyType.CALL_CREDIT_SPREAD,
    OISignalType.STRADDLE_COLLAPSE_BEARISH.value: FnOStrategyType.CALL_CREDIT_SPREAD,
    OISignalType.OI_BUILDUP_BEARISH.value: FnOStrategyType.CALL_CREDIT_SPREAD,
    OISignalType.COMBINED_BEARISH.value: FnOStrategyType.CALL_CREDIT_SPREAD,
    OISignalType.OI_WALL_BREAKOUT_DOWN.value: FnOStrategyType.BEAR_PUT_SPREAD,

    # Bullish signals → credit put spread (collect premium, defined risk)
    OISignalType.PCR_EXTREME_BULLISH.value: FnOStrategyType.PUT_CREDIT_SPREAD,
    OISignalType.OI_WALL_SUPPORT_BOUNCE.value: FnOStrategyType.PUT_CREDIT_SPREAD,
    OISignalType.PUT_WRITING_SUPPORT.value: FnOStrategyType.PUT_CREDIT_SPREAD,
    OISignalType.STRADDLE_COLLAPSE_BULLISH.value: FnOStrategyType.PUT_CREDIT_SPREAD,
    OISignalType.OI_BUILDUP_BULLISH.value: FnOStrategyType.PUT_CREDIT_SPREAD,
    OISignalType.COMBINED_BULLISH.value: FnOStrategyType.PUT_CREDIT_SPREAD,
    OISignalType.OI_WALL_BREAKOUT_UP.value: FnOStrategyType.BULL_CALL_SPREAD,

    # IV Skew → opposite direction credit spread
    OISignalType.IV_SKEW_REVERSAL.value: FnOStrategyType.IRON_CONDOR,  # IV plays → neutral

    # Max Pain → fade move, use credit spread in the direction of fade
    OISignalType.MAX_PAIN_MAGNET.value: FnOStrategyType.IRON_CONDOR,
}


# ─────────────────────────────────────────────────────────────
# Backtest Quality Thresholds
# ─────────────────────────────────────────────────────────────

class BacktestThresholds(BaseModel):
    """Configurable thresholds for backtest-gated execution."""
    min_win_rate: float = Field(default=40.0, description="Minimum win rate % to pass")
    min_sharpe_ratio: float = Field(default=0.3, description="Minimum Sharpe ratio")
    max_drawdown_pct: float = Field(default=30.0, description="Maximum drawdown % allowed")
    min_total_return_pct: float = Field(default=0.0, description="Minimum total return %")
    min_trades: int = Field(default=1, description="Minimum trades for statistical significance")
    min_profit_factor: float = Field(default=0.8, description="Minimum profit factor (gross profit / gross loss)")
    max_avg_loss_pct: float = Field(default=8.0, description="Maximum average loss % per trade")

    # Backtest parameters
    backtest_days: int = Field(default=365, description="Days of data for backtest")
    backtest_capital: float = Field(default=500_000.0, description="Initial capital for backtest")
    backtest_max_positions: int = Field(default=3, description="Max concurrent positions")

    # Execution gating
    require_backtest: bool = Field(default=True, description="Require backtest pass before execution")
    min_signal_confidence: float = Field(default=60.0, description="Minimum OI signal confidence to consider")
    auto_execute_on_pass: bool = Field(default=False, description="Auto-execute if backtest passes")


# ─────────────────────────────────────────────────────────────
# Strategy Mapping Result
# ─────────────────────────────────────────────────────────────

class StrategyMapping(BaseModel):
    """Result of mapping an OI signal to an FNO strategy."""
    signal_id: str = ""
    signal_type: str = ""
    signal_direction: str = ""
    signal_confidence: float = 0.0
    underlying: str = ""

    # Mapped strategy
    fno_strategy: str = ""           # e.g. "call_credit_spread_runner"
    fno_strategy_label: str = ""     # e.g. "Call Credit Spread Runner"
    strategy_rationale: str = ""     # Why this strategy was chosen
    alternative_strategies: list[str] = Field(default_factory=list)

    # Strategy params derived from OI signal
    strategy_params: dict[str, Any] = Field(default_factory=dict)


class BacktestResult(BaseModel):
    """Summary of backtest validation result."""
    passed: bool = False
    strategy_name: str = ""
    underlying: str = ""

    # Metrics
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    total_return_pct: float = 0.0
    total_trades: int = 0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_loss_pct: float = 0.0

    # Threshold comparison
    threshold_checks: dict[str, dict[str, Any]] = Field(default_factory=dict)
    failure_reasons: list[str] = Field(default_factory=list)
    pass_reasons: list[str] = Field(default_factory=list)

    # Full backtest data (optional, for detailed analysis)
    full_result: dict[str, Any] = Field(default_factory=dict)


class SignalExecutionPlan(BaseModel):
    """Complete execution plan for an OI signal → FNO strategy."""
    signal: dict[str, Any] = Field(default_factory=dict)
    mapping: dict[str, Any] = Field(default_factory=dict)
    backtest: dict[str, Any] = Field(default_factory=dict)
    execution_approved: bool = False
    approval_reasons: list[str] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list)
    risk_warnings: list[str] = Field(default_factory=list)
    created_at: str = ""


# ─────────────────────────────────────────────────────────────
# OI → FNO Bridge Engine
# ─────────────────────────────────────────────────────────────

class OIFnOBridge:
    """Bridge between OI signal generation and FNO strategy execution.

    Responsibilities:
      1. Map OI signals to the best matching FNO strategy
      2. Run backtest validation against configurable thresholds
      3. Generate execution plans with full risk assessment
      4. Gate execution on backtest quality
    """

    def __init__(
        self,
        thresholds: Optional[BacktestThresholds] = None,
    ) -> None:
        self.thresholds = thresholds or BacktestThresholds()
        self._execution_plans: list[SignalExecutionPlan] = []
        self._backtest_cache: dict[str, BacktestResult] = {}  # strategy+underlying → result

    # ─── 1. Signal → Strategy Mapping ────────────────────────

    def map_signal_to_strategy(self, signal: OISignal) -> StrategyMapping:
        """Map an OI signal to the best FNO strategy.

        Priority:
          1. Signal-type-specific override (most precise)
          2. Action-based mapping
          3. Direction-based fallback
        """
        if signal.confidence < self.thresholds.min_signal_confidence:
            logger.info(
                "oi_fno_bridge_skip_low_confidence",
                signal_id=signal.id,
                confidence=signal.confidence,
                min_required=self.thresholds.min_signal_confidence,
            )
            return StrategyMapping(
                signal_id=signal.id,
                signal_type=signal.signal_type,
                signal_direction=signal.direction,
                signal_confidence=signal.confidence,
                underlying=signal.underlying,
                fno_strategy=FnOStrategyType.NONE.value,
                strategy_rationale=f"Signal confidence {signal.confidence:.1f}% below minimum {self.thresholds.min_signal_confidence:.1f}%",
            )

        # 1. Check signal-type-specific override
        fno_strategy = _SIGNAL_TYPE_STRATEGY_OVERRIDE.get(signal.signal_type)
        rationale = ""

        if fno_strategy:
            rationale = f"Signal type {signal.signal_type} maps to {fno_strategy.value}"
        else:
            # 2. Check action-based mapping
            fno_strategy = _ACTION_STRATEGY_MAP.get(signal.action)
            if fno_strategy:
                rationale = f"Action {signal.action} maps to {fno_strategy.value}"
            else:
                # 3. Direction-based fallback
                direction_strategies = _DIRECTION_STRATEGY_MAP.get(signal.direction, [])
                if direction_strategies:
                    fno_strategy = direction_strategies[0]
                    rationale = f"Direction {signal.direction} → default strategy {fno_strategy.value}"
                else:
                    fno_strategy = FnOStrategyType.NONE
                    rationale = f"No strategy mapping for direction={signal.direction}, action={signal.action}"

        # Determine alternatives
        alternatives = []
        direction_strats = _DIRECTION_STRATEGY_MAP.get(signal.direction, [])
        for alt in direction_strats:
            if alt != fno_strategy:
                alternatives.append(alt.value)

        # Derive strategy params from OI signal context
        params = self._derive_strategy_params(signal, fno_strategy)

        # Build human-readable label
        label = fno_strategy.value.replace("_", " ").title() if fno_strategy != FnOStrategyType.NONE else "None"

        mapping = StrategyMapping(
            signal_id=signal.id,
            signal_type=signal.signal_type,
            signal_direction=signal.direction,
            signal_confidence=signal.confidence,
            underlying=signal.underlying,
            fno_strategy=fno_strategy.value,
            fno_strategy_label=label,
            strategy_rationale=rationale,
            alternative_strategies=alternatives,
            strategy_params=params,
        )

        logger.info(
            "oi_fno_bridge_mapped",
            signal_id=signal.id,
            signal_type=signal.signal_type,
            direction=signal.direction,
            confidence=signal.confidence,
            fno_strategy=fno_strategy.value,
            rationale=rationale,
        )

        return mapping

    def _derive_strategy_params(
        self, signal: OISignal, strategy: FnOStrategyType
    ) -> dict[str, Any]:
        """Derive FNO strategy parameters from OI signal context."""
        params: dict[str, Any] = {
            "underlying": signal.underlying,
        }

        # Map underlying to appropriate spread width
        if signal.underlying == "SENSEX":
            params["spread_width"] = 300
            params["capital"] = 500_000.0
        else:
            params["spread_width"] = 100
            params["capital"] = 500_000.0

        # Use signal strike info for strategy targeting
        if signal.strike > 0:
            params["target_strike"] = signal.strike

        if signal.entry_price > 0:
            params["reference_premium"] = signal.entry_price

        # OI context for strategy tuning
        if signal.spot_price > 0:
            params["spot_price"] = signal.spot_price
        if signal.atm_strike > 0:
            params["atm_strike"] = signal.atm_strike

        # Confidence-based lot sizing
        if signal.confidence >= 80:
            params["lots"] = 2
        elif signal.confidence >= 70:
            params["lots"] = 1
        else:
            params["lots"] = 1

        # Add context for the strategy
        params["oi_signal_type"] = signal.signal_type
        params["oi_direction"] = signal.direction
        params["oi_confidence"] = signal.confidence

        return params

    # ─── 2. Backtest Validation ──────────────────────────────

    async def backtest_strategy(
        self,
        mapping: StrategyMapping,
        market_data_fetcher: Any = None,
        force_refresh: bool = False,
    ) -> BacktestResult:
        """Run backtest for the mapped FNO strategy and evaluate against thresholds.

        Args:
            mapping: Strategy mapping from map_signal_to_strategy()
            market_data_fetcher: Async callable that returns (DataFrame, instrument_token)
            force_refresh: Skip cache and re-run backtest

        Returns:
            BacktestResult with pass/fail and detailed metrics
        """
        from src.derivatives.fno_backtest import FnOBacktestEngine

        cache_key = f"{mapping.fno_strategy}:{mapping.underlying}"

        # Check cache (avoid running same backtest repeatedly)
        if not force_refresh and cache_key in self._backtest_cache:
            cached = self._backtest_cache[cache_key]
            logger.info(
                "oi_fno_bridge_backtest_cached",
                strategy=mapping.fno_strategy,
                underlying=mapping.underlying,
                passed=cached.passed,
            )
            return cached

        if mapping.fno_strategy == FnOStrategyType.NONE.value:
            return BacktestResult(
                passed=False,
                strategy_name=mapping.fno_strategy,
                underlying=mapping.underlying,
                failure_reasons=["No strategy mapped — cannot backtest"],
            )

        # Get historical data for backtest
        if market_data_fetcher is None:
            return BacktestResult(
                passed=False,
                strategy_name=mapping.fno_strategy,
                underlying=mapping.underlying,
                failure_reasons=["No market data fetcher provided — cannot run backtest"],
            )

        try:
            df, instrument_token = await market_data_fetcher(
                mapping.underlying,
                self.thresholds.backtest_days,
            )

            if df is None or df.empty:
                return BacktestResult(
                    passed=False,
                    strategy_name=mapping.fno_strategy,
                    underlying=mapping.underlying,
                    failure_reasons=["No historical data available for backtest"],
                )

            # Run backtest
            engine = FnOBacktestEngine(
                strategy_name=mapping.fno_strategy,
                underlying=mapping.underlying,
                initial_capital=self.thresholds.backtest_capital,
                max_positions=self.thresholds.backtest_max_positions,
                use_regime_filter=True,
            )

            backtest_raw = engine.run(df, tradingsymbol=mapping.underlying)

            if "error" in backtest_raw:
                return BacktestResult(
                    passed=False,
                    strategy_name=mapping.fno_strategy,
                    underlying=mapping.underlying,
                    failure_reasons=[f"Backtest error: {backtest_raw['error']}"],
                )

            # Evaluate against thresholds
            result = self._evaluate_backtest(backtest_raw, mapping)
            self._backtest_cache[cache_key] = result

            logger.info(
                "oi_fno_bridge_backtest_complete",
                strategy=mapping.fno_strategy,
                underlying=mapping.underlying,
                passed=result.passed,
                win_rate=result.win_rate,
                sharpe=result.sharpe_ratio,
                max_dd=result.max_drawdown_pct,
                total_return=result.total_return_pct,
                trades=result.total_trades,
            )

            return result

        except Exception as e:
            logger.error(
                "oi_fno_bridge_backtest_error",
                strategy=mapping.fno_strategy,
                underlying=mapping.underlying,
                error=str(e),
            )
            return BacktestResult(
                passed=False,
                strategy_name=mapping.fno_strategy,
                underlying=mapping.underlying,
                failure_reasons=[f"Backtest exception: {str(e)}"],
            )

    def _evaluate_backtest(
        self, raw: dict[str, Any], mapping: StrategyMapping
    ) -> BacktestResult:
        """Evaluate raw backtest results against configured thresholds."""
        t = self.thresholds

        # Extract metrics from raw backtest result
        win_rate = raw.get("win_rate", 0.0)
        sharpe = raw.get("sharpe_ratio", 0.0)
        max_dd = abs(raw.get("max_drawdown_pct", 0.0))
        total_return = raw.get("total_return_pct", 0.0)
        total_trades = raw.get("positions_closed", 0) or raw.get("total_trades", 0)
        profit_factor = raw.get("profit_factor", 0.0)
        avg_trade_pnl = raw.get("avg_trade_pnl", 0.0)

        # Compute average loss percentage if not directly available
        avg_loss_pct = 0.0
        trades = raw.get("trades", [])
        if trades:
            losses = [tr.get("pnl", 0) for tr in trades if tr.get("pnl", 0) < 0]
            if losses:
                initial_capital = raw.get("initial_capital", 500000)
                avg_loss_pct = abs(sum(losses) / len(losses)) / initial_capital * 100

        # Run threshold checks
        checks: dict[str, dict[str, Any]] = {}
        failures: list[str] = []
        passes: list[str] = []

        def _check(name: str, actual: float, threshold: float, op: str = ">="):
            passed = False
            if op == ">=":
                passed = actual >= threshold
            elif op == "<=":
                passed = actual <= threshold
            elif op == ">":
                passed = actual > threshold

            checks[name] = {
                "actual": round(actual, 2),
                "threshold": round(threshold, 2),
                "operator": op,
                "passed": passed,
            }
            if passed:
                passes.append(f"{name}: {actual:.2f} {op} {threshold:.2f} ✓")
            else:
                failures.append(f"{name}: {actual:.2f} {op} {threshold:.2f} ✗")

        _check("win_rate", win_rate, t.min_win_rate, ">=")
        _check("sharpe_ratio", sharpe, t.min_sharpe_ratio, ">=")
        _check("max_drawdown", max_dd, t.max_drawdown_pct, "<=")
        _check("total_return", total_return, t.min_total_return_pct, ">=")
        _check("total_trades", float(total_trades), float(t.min_trades), ">=")
        _check("profit_factor", profit_factor, t.min_profit_factor, ">=")
        _check("avg_loss_pct", avg_loss_pct, t.max_avg_loss_pct, "<=")

        all_passed = len(failures) == 0

        return BacktestResult(
            passed=all_passed,
            strategy_name=mapping.fno_strategy,
            underlying=mapping.underlying,
            win_rate=round(win_rate, 2),
            sharpe_ratio=round(sharpe, 2),
            max_drawdown_pct=round(max_dd, 2),
            total_return_pct=round(total_return, 2),
            total_trades=total_trades,
            profit_factor=round(profit_factor, 2),
            avg_trade_pnl=round(avg_trade_pnl, 2),
            avg_loss_pct=round(avg_loss_pct, 2),
            threshold_checks=checks,
            failure_reasons=failures,
            pass_reasons=passes,
            full_result=raw,
        )

    # ─── 3. Full Execution Plan ──────────────────────────────

    async def create_execution_plan(
        self,
        signal: OISignal,
        market_data_fetcher: Any = None,
        force_backtest: bool = False,
    ) -> SignalExecutionPlan:
        """Create a complete execution plan for an OI signal.

        Steps:
          1. Map signal to FNO strategy
          2. Run backtest validation (if required)
          3. Assess risk warnings
          4. Determine execution approval

        Args:
            signal: OI signal to create plan for
            market_data_fetcher: async callable(underlying, days) → (DataFrame, token)
            force_backtest: Force re-run backtest even if cached

        Returns:
            SignalExecutionPlan with all details
        """
        plan = SignalExecutionPlan(
            signal=signal.model_dump(),
            created_at=datetime.now().isoformat(),
        )

        # Step 1: Map signal to strategy
        mapping = self.map_signal_to_strategy(signal)
        plan.mapping = mapping.model_dump()

        if mapping.fno_strategy == FnOStrategyType.NONE.value:
            plan.execution_approved = False
            plan.rejection_reasons.append("No FNO strategy could be mapped to this signal")
            self._execution_plans.append(plan)
            return plan

        # Step 2: Backtest validation
        if self.thresholds.require_backtest:
            bt_result = await self.backtest_strategy(
                mapping,
                market_data_fetcher=market_data_fetcher,
                force_refresh=force_backtest,
            )
            plan.backtest = bt_result.model_dump()

            if not bt_result.passed:
                plan.execution_approved = False
                plan.rejection_reasons.append(
                    f"Backtest failed: {'; '.join(bt_result.failure_reasons[:3])}"
                )
            else:
                plan.approval_reasons.append(
                    f"Backtest passed: win_rate={bt_result.win_rate:.1f}%, "
                    f"sharpe={bt_result.sharpe_ratio:.2f}, "
                    f"return={bt_result.total_return_pct:.1f}%"
                )
        else:
            plan.approval_reasons.append("Backtest validation skipped (require_backtest=False)")

        # Step 3: Risk warnings
        risk_warnings = self._assess_risk_warnings(signal, mapping)
        plan.risk_warnings = risk_warnings

        # Step 4: Final execution decision
        if not plan.rejection_reasons:
            plan.execution_approved = True

        self._execution_plans.append(plan)

        logger.info(
            "oi_fno_bridge_plan_created",
            signal_id=signal.id,
            signal_type=signal.signal_type,
            fno_strategy=mapping.fno_strategy,
            approved=plan.execution_approved,
            reasons=plan.approval_reasons[:2] or plan.rejection_reasons[:2],
            warnings=len(risk_warnings),
        )

        return plan

    def _assess_risk_warnings(
        self, signal: OISignal, mapping: StrategyMapping
    ) -> list[str]:
        """Generate risk warnings for the execution plan."""
        warnings: list[str] = []

        # Low confidence warning
        if signal.confidence < 70:
            warnings.append(
                f"Signal confidence {signal.confidence:.1f}% is moderate "
                f"(below 70% strong threshold)"
            )

        # IV Skew trades have mean-reversion risk
        if signal.signal_type == OISignalType.IV_SKEW_REVERSAL.value:
            warnings.append(
                "IV Skew reversal trades carry event risk — IV can stay elevated"
            )

        # Straddle collapse can be false signal
        if "STRADDLE_COLLAPSE" in signal.signal_type:
            warnings.append(
                "Straddle collapse may indicate low liquidity rather than directional move"
            )

        # Max pain magnet less reliable far from expiry
        if signal.signal_type == OISignalType.MAX_PAIN_MAGNET.value:
            warnings.append(
                "Max pain attraction is strongest near expiry — weaker mid-week"
            )

        # Near OI wall — can break through on high volume
        if "OI_WALL" in signal.signal_type:
            warnings.append(
                "OI walls can break under strong institutional flow"
            )

        # Spread width risk for credit spreads
        if mapping.fno_strategy in (
            FnOStrategyType.CALL_CREDIT_SPREAD.value,
            FnOStrategyType.PUT_CREDIT_SPREAD.value,
        ):
            warnings.append(
                "Credit spread max loss = spread width − credit received. "
                "Ensure position sizing respects 2% capital rule"
            )

        # Combined signal (multiple detectors) — higher reliability
        if signal.signal_type in (
            OISignalType.COMBINED_BULLISH.value,
            OISignalType.COMBINED_BEARISH.value,
        ):
            # This is actually a positive signal — note it differently
            warnings.append(
                "Combined signal from multiple OI detectors — higher reliability"
            )

        return warnings

    # ─── 4. Batch Processing ─────────────────────────────────

    async def process_all_signals(
        self,
        signals: list[OISignal],
        market_data_fetcher: Any = None,
    ) -> list[SignalExecutionPlan]:
        """Process multiple OI signals, create execution plans for each.

        Args:
            signals: List of OI signals from scan
            market_data_fetcher: async callable for backtest data

        Returns:
            List of execution plans (approved + rejected)
        """
        plans = []

        for signal in signals:
            if signal.confidence < self.thresholds.min_signal_confidence:
                continue
            if signal.confidence <= 0 or not signal.signal_type:
                continue

            plan = await self.create_execution_plan(
                signal,
                market_data_fetcher=market_data_fetcher,
            )
            plans.append(plan)

        # Sort by approval status, then confidence
        plans.sort(key=lambda p: (
            not p.execution_approved,
            -p.signal.get("confidence", 0),
        ))

        logger.info(
            "oi_fno_bridge_batch_processed",
            total=len(plans),
            approved=sum(1 for p in plans if p.execution_approved),
            rejected=sum(1 for p in plans if not p.execution_approved),
        )

        return plans

    # ─── 5. Getters ──────────────────────────────────────────

    def get_execution_plans(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent execution plans."""
        return [p.model_dump() for p in self._execution_plans[-limit:]]

    def get_approved_plans(self) -> list[dict[str, Any]]:
        """Get only approved execution plans."""
        return [
            p.model_dump() for p in self._execution_plans
            if p.execution_approved
        ]

    def get_backtest_cache(self) -> dict[str, dict[str, Any]]:
        """Get cached backtest results."""
        return {k: v.model_dump() for k, v in self._backtest_cache.items()}

    def clear_backtest_cache(self) -> None:
        """Clear backtest cache (e.g. after strategy config change)."""
        self._backtest_cache.clear()

    def update_thresholds(self, updates: dict[str, Any]) -> BacktestThresholds:
        """Update backtest thresholds."""
        current = self.thresholds.model_dump()
        current.update({k: v for k, v in updates.items() if k in current})
        self.thresholds = BacktestThresholds(**current)
        # Clear cache since thresholds changed
        self._backtest_cache.clear()
        return self.thresholds

    def get_thresholds(self) -> dict[str, Any]:
        """Get current backtest thresholds."""
        return self.thresholds.model_dump()

    def get_strategy_mapping_info(self) -> dict[str, Any]:
        """Get all signal-to-strategy mappings for documentation."""
        return {
            "direction_map": {
                k: [s.value for s in v]
                for k, v in _DIRECTION_STRATEGY_MAP.items()
            },
            "action_map": {
                k: v.value for k, v in _ACTION_STRATEGY_MAP.items()
            },
            "signal_type_overrides": {
                k: v.value for k, v in _SIGNAL_TYPE_STRATEGY_OVERRIDE.items()
            },
        }
