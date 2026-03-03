"""
Strategy Decay Monitor
======================

Monitors rolling performance of every active strategy and automatically
disables strategies that have decayed significantly from their historical
baseline.

Metrics tracked (rolling 30-day window):
  • Sharpe ratio
  • Win rate
  • Profit factor

A strategy is flagged for decay when any metric deviates by more than
2 standard deviations from its historical mean (computed over the full
equity curve).  On detection the strategy is auto-disabled and a system
event is recorded.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Optional, Callable

logger = logging.getLogger("strategy_decay")


@dataclass
class DecayThresholds:
    """Configurable thresholds for decay detection."""
    std_dev_trigger: float = 2.0       # Number of std devs to trigger decay alert
    rolling_window_days: int = 30      # Rolling window for current metrics
    min_history_days: int = 60         # Minimum history needed to compute baseline
    min_trades_in_window: int = 5      # Minimum trades in window to evaluate
    auto_disable: bool = True          # Automatically disable decayed strategies


@dataclass
class StrategyDecayReport:
    """Decay analysis for a single strategy."""
    strategy_name: str = ""
    timestamp: str = ""

    # Current rolling metrics
    rolling_sharpe: float = 0.0
    rolling_win_rate: float = 0.0
    rolling_profit_factor: float = 0.0
    rolling_trades: int = 0

    # Historical baseline
    baseline_sharpe: float = 0.0
    baseline_win_rate: float = 0.0
    baseline_profit_factor: float = 0.0
    baseline_sharpe_std: float = 0.0
    baseline_win_rate_std: float = 0.0
    baseline_profit_factor_std: float = 0.0

    # Deviation (in std devs)
    sharpe_deviation: float = 0.0
    win_rate_deviation: float = 0.0
    profit_factor_deviation: float = 0.0

    # Flags
    is_decayed: bool = False
    decay_reasons: list[str] = field(default_factory=list)
    action_taken: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in d:
            if isinstance(d[k], float):
                d[k] = round(d[k], 4)
        return d


class StrategyDecayMonitor:
    """
    Monitors strategy performance decay using rolling vs. historical comparison.

    Usage:
        monitor = StrategyDecayMonitor()
        reports = monitor.evaluate_all(equity_curve_fn, active_strategies, disable_fn)
    """

    def __init__(self, thresholds: Optional[DecayThresholds] = None) -> None:
        self._thresholds = thresholds or DecayThresholds()
        self._history: list[StrategyDecayReport] = []

    @property
    def thresholds(self) -> DecayThresholds:
        return self._thresholds

    def update_thresholds(self, updates: dict[str, Any]) -> DecayThresholds:
        for k, v in updates.items():
            if hasattr(self._thresholds, k):
                setattr(self._thresholds, k, type(getattr(self._thresholds, k))(v))
        return self._thresholds

    def get_thresholds(self) -> dict[str, Any]:
        return asdict(self._thresholds)

    def evaluate_strategy(
        self,
        strategy_name: str,
        equity_curve: list[dict[str, Any]],
        disable_callback: Optional[Callable[[str], Any]] = None,
    ) -> StrategyDecayReport:
        """
        Evaluate a single strategy for decay.

        Args:
            strategy_name: Name of the strategy.
            equity_curve: List of dicts with keys: date, pnl, cumulative_pnl, trades_count.
                          Sorted ascending by date.
            disable_callback: Optional function called with strategy_name if decay is detected
                              and auto_disable=True.

        Returns:
            StrategyDecayReport.
        """
        report = StrategyDecayReport(
            strategy_name=strategy_name,
            timestamp=datetime.now().isoformat(),
        )

        if len(equity_curve) < self._thresholds.min_history_days:
            report.action_taken = "insufficient_history"
            return report

        # Split into rolling window and historical
        cutoff_date = (datetime.now() - timedelta(days=self._thresholds.rolling_window_days)).strftime("%Y-%m-%d")

        rolling_data = [d for d in equity_curve if d.get("date", "") >= cutoff_date]
        historical_data = [d for d in equity_curve if d.get("date", "") < cutoff_date]

        if not historical_data or len(historical_data) < self._thresholds.min_history_days:
            report.action_taken = "insufficient_baseline"
            return report

        rolling_trades = sum(d.get("trades_count", 0) for d in rolling_data)
        report.rolling_trades = rolling_trades

        if rolling_trades < self._thresholds.min_trades_in_window:
            report.action_taken = "insufficient_rolling_trades"
            return report

        # ── Compute rolling metrics ──────────────────────────
        rolling_pnls = [d.get("pnl", 0) for d in rolling_data]
        report.rolling_sharpe = self._compute_sharpe(rolling_pnls)
        report.rolling_win_rate = self._compute_win_rate(rolling_pnls)
        report.rolling_profit_factor = self._compute_profit_factor(rolling_pnls)

        # ── Compute baseline using sliding windows ───────────
        window = self._thresholds.rolling_window_days
        sharpes, win_rates, profit_factors = [], [], []

        for i in range(0, len(historical_data) - window + 1, max(1, window // 4)):
            chunk = historical_data[i:i + window]
            pnls = [d.get("pnl", 0) for d in chunk]
            if sum(d.get("trades_count", 0) for d in chunk) >= self._thresholds.min_trades_in_window:
                sharpes.append(self._compute_sharpe(pnls))
                win_rates.append(self._compute_win_rate(pnls))
                profit_factors.append(self._compute_profit_factor(pnls))

        if not sharpes:
            report.action_taken = "insufficient_baseline_windows"
            return report

        report.baseline_sharpe = self._mean(sharpes)
        report.baseline_win_rate = self._mean(win_rates)
        report.baseline_profit_factor = self._mean(profit_factors)
        report.baseline_sharpe_std = self._std(sharpes)
        report.baseline_win_rate_std = self._std(win_rates)
        report.baseline_profit_factor_std = self._std(profit_factors)

        # ── Compute deviations ───────────────────────────────
        report.sharpe_deviation = self._deviation(
            report.rolling_sharpe, report.baseline_sharpe, report.baseline_sharpe_std
        )
        report.win_rate_deviation = self._deviation(
            report.rolling_win_rate, report.baseline_win_rate, report.baseline_win_rate_std
        )
        report.profit_factor_deviation = self._deviation(
            report.rolling_profit_factor, report.baseline_profit_factor, report.baseline_profit_factor_std
        )

        # ── Check for decay ──────────────────────────────────
        threshold = self._thresholds.std_dev_trigger
        decay_reasons = []

        if report.sharpe_deviation < -threshold:
            decay_reasons.append(
                f"Sharpe decayed: rolling={report.rolling_sharpe:.2f} vs "
                f"baseline={report.baseline_sharpe:.2f} ({report.sharpe_deviation:.1f}σ)"
            )
        if report.win_rate_deviation < -threshold:
            decay_reasons.append(
                f"Win rate decayed: rolling={report.rolling_win_rate:.1%} vs "
                f"baseline={report.baseline_win_rate:.1%} ({report.win_rate_deviation:.1f}σ)"
            )
        if report.profit_factor_deviation < -threshold:
            decay_reasons.append(
                f"Profit factor decayed: rolling={report.rolling_profit_factor:.2f} vs "
                f"baseline={report.baseline_profit_factor:.2f} ({report.profit_factor_deviation:.1f}σ)"
            )

        report.decay_reasons = decay_reasons
        report.is_decayed = len(decay_reasons) > 0

        if report.is_decayed and self._thresholds.auto_disable and disable_callback:
            try:
                disable_callback(strategy_name)
                report.action_taken = "auto_disabled"
                logger.warning(
                    "strategy_auto_disabled_decay",
                    strategy=strategy_name,
                    reasons=decay_reasons,
                )
            except Exception as e:
                report.action_taken = f"disable_failed: {e}"
                logger.error("strategy_disable_failed", strategy=strategy_name, error=str(e))
        elif report.is_decayed:
            report.action_taken = "flagged_only"

        self._history.append(report)
        if len(self._history) > 500:
            self._history = self._history[-500:]

        return report

    def evaluate_all(
        self,
        strategy_equity_curves: dict[str, list[dict[str, Any]]],
        disable_callback: Optional[Callable[[str], Any]] = None,
    ) -> list[dict[str, Any]]:
        """
        Evaluate all strategies at once.

        Args:
            strategy_equity_curves: Map of strategy_name → equity curve data.
            disable_callback: Called with strategy_name if decay detected.

        Returns:
            List of decay report dicts.
        """
        results = []
        for name, curve in strategy_equity_curves.items():
            report = self.evaluate_strategy(name, curve, disable_callback)
            results.append(report.to_dict())
        return results

    def get_history(self, limit: int = 50) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self._history[-limit:]]

    # ── Math helpers ─────────────────────────────────────────

    @staticmethod
    def _compute_sharpe(pnls: list[float]) -> float:
        if len(pnls) < 2:
            return 0.0
        mean_ret = sum(pnls) / len(pnls)
        variance = sum((p - mean_ret) ** 2 for p in pnls) / (len(pnls) - 1)
        std = math.sqrt(variance) if variance > 0 else 1e-9
        # Annualise assuming daily data
        return (mean_ret / std) * math.sqrt(252)

    @staticmethod
    def _compute_win_rate(pnls: list[float]) -> float:
        if not pnls:
            return 0.0
        wins = sum(1 for p in pnls if p > 0)
        return wins / len(pnls)

    @staticmethod
    def _compute_profit_factor(pnls: list[float]) -> float:
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        if gross_loss == 0:
            return 10.0 if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @staticmethod
    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    @staticmethod
    def _std(vals: list[float]) -> float:
        if len(vals) < 2:
            return 1e-9
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
        return math.sqrt(var) if var > 0 else 1e-9

    @staticmethod
    def _deviation(current: float, baseline_mean: float, baseline_std: float) -> float:
        """Compute deviation in standard deviations (negative = worse than baseline)."""
        if baseline_std <= 1e-9:
            return 0.0
        return (current - baseline_mean) / baseline_std
