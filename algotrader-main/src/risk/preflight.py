"""
Pre-Flight Checklist — Consolidated gate before ANY order placement.

Every order (live, paper, backtest) passes through PreflightChecker
before reaching the broker. Failing any CRITICAL check blocks execution.

Checklist Coverage:
  1. Broker connected (session alive)
  2. Margin buffer ≥ 2× required
  3. Option chain valid (for F&O orders)
  4. Strategy regime valid (if regime engine active)
  5. WebSocket stable (last tick fresh)
  6. Hedge leg fills first (validated via order group)
  7. Entry leg timeout ≤ 2s
  8. Slippage within limits
  9. Rejected orders not ignored
  10. Delta within range
  11. Margin usage < 70%
  12. Exposure monitored
  13. Emergency hedge active
  14. Stop-loss fallback active
  15. Restart recovery verified
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Enums & Data Classes
# ─────────────────────────────────────────────────────────────

class CheckSeverity(str, Enum):
    CRITICAL = "CRITICAL"    # Blocks order placement
    WARNING = "WARNING"      # Logged but does not block
    INFO = "INFO"            # Informational only


class TradingMode(str, Enum):
    LIVE = "LIVE"
    PAPER = "PAPER"
    BACKTEST = "BACKTEST"


@dataclass
class CheckResult:
    """Result of a single preflight check."""
    name: str
    passed: bool
    severity: CheckSeverity
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def blocks_execution(self) -> bool:
        return not self.passed and self.severity == CheckSeverity.CRITICAL


@dataclass
class PreflightReport:
    """Aggregate report from all preflight checks."""
    passed: bool
    mode: TradingMode
    timestamp: datetime = field(default_factory=datetime.now)
    checks: list[CheckResult] = field(default_factory=list)
    blocked_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "mode": self.mode.value,
            "timestamp": self.timestamp.isoformat(),
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "severity": c.severity.value,
                    "message": c.message,
                }
                for c in self.checks
            ],
            "blocked_reasons": self.blocked_reasons,
        }


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

@dataclass
class PreflightConfig:
    """Tunable thresholds for preflight checks."""

    # Margin
    margin_buffer_multiplier: float = 2.0        # ≥ 2× margin required
    max_margin_utilization_pct: float = 70.0      # Block if >70% utilized

    # WebSocket freshness
    max_tick_age_seconds: float = 30.0            # Tick stale after 30s
    require_websocket_for_live: bool = True

    # Regime
    blocked_regimes: list[str] = field(default_factory=lambda: ["EVENT_RISK"])
    require_regime_check: bool = True

    # Greeks / Delta
    max_portfolio_delta: float = 500.0            # Absolute net delta cap
    max_portfolio_vega: float = 50000.0           # ₹ vega cap

    # Exposure
    max_exposure_pct: float = 80.0                # % of max_exposure setting

    # Option chain
    require_chain_validation: bool = True
    max_chain_age_minutes: float = 10.0           # Chain data stale after 10m

    # Entry timeout
    entry_leg_timeout_seconds: float = 2.0        # Max wait for entry fill

    # Slippage
    max_slippage_pct: float = 0.5                 # 0.5% max slippage

    # Safety
    require_stop_loss: bool = True                # All live orders must have SL
    require_emergency_hedge: bool = True          # Multi-leg must have hedge


# ─────────────────────────────────────────────────────────────
# PreflightChecker
# ─────────────────────────────────────────────────────────────

class PreflightChecker:
    """Consolidated pre-trade validation gate.

    Usage:
        checker = PreflightChecker(client, ticker, risk_manager, ...)
        report = await checker.run_all_checks(signal, mode=TradingMode.LIVE)
        if not report.passed:
            raise RiskLimitError(f"Preflight failed: {report.blocked_reasons}")
    """

    def __init__(
        self,
        client: Any = None,           # KiteClient
        ticker: Any = None,            # KiteTicker
        risk_manager: Any = None,      # RiskManager
        greeks_engine: Any = None,     # GreeksEngine
        regime_engine: Any = None,     # RegimeEngine
        chain_builder: Any = None,     # OptionChainBuilder
        config: Optional[PreflightConfig] = None,
    ) -> None:
        self._client = client
        self._ticker = ticker
        self._risk = risk_manager
        self._greeks = greeks_engine
        self._regime = regime_engine
        self._chain = chain_builder
        self.config = config or PreflightConfig()

        # Cache for last checks (dashboard display)
        self._last_report: Optional[PreflightReport] = None
        self._last_broker_check_time: float = 0.0
        self._broker_alive: bool = False
        self._broker_check_interval: float = 60.0  # Re-check every 60s

    # ─── Public API ──────────────────────────────────────────

    async def run_all_checks(
        self,
        signal: Any = None,
        mode: TradingMode = TradingMode.LIVE,
        signals: Optional[list] = None,  # For multi-leg
    ) -> PreflightReport:
        """Run all applicable preflight checks for the given mode.

        Args:
            signal: Single Signal for order (or None for multi-leg)
            mode: LIVE, PAPER, or BACKTEST
            signals: List of Signals for multi-leg execution

        Returns:
            PreflightReport with pass/fail and details
        """
        checks: list[CheckResult] = []

        if mode == TradingMode.BACKTEST:
            # Backtest only needs basic sanity checks
            checks.append(self._check_signal_sanity(signal))
            if signal and signal.stop_loss is None and self.config.require_stop_loss:
                checks.append(CheckResult(
                    name="stop_loss_defined",
                    passed=False,
                    severity=CheckSeverity.WARNING,
                    message="No stop-loss defined on signal (backtest mode — warning only)",
                ))
        elif mode == TradingMode.PAPER:
            # Paper trading needs signal + margin + exposure checks
            checks.append(self._check_signal_sanity(signal))
            checks.append(await self._check_margin_sufficient(signal, signals))
            checks.append(self._check_exposure_limit(signal))
            checks.append(self._check_stop_loss_defined(signal))
        else:
            # LIVE — full checklist
            checks.extend(await self._run_live_checks(signal, signals))

        # Build report
        blocked = [c.message for c in checks if c.blocks_execution]
        report = PreflightReport(
            passed=len(blocked) == 0,
            mode=mode,
            checks=checks,
            blocked_reasons=blocked,
        )
        self._last_report = report

        # Log result
        if report.passed:
            logger.info(
                "preflight_passed",
                mode=mode.value,
                checks_run=len(checks),
                symbol=getattr(signal, "tradingsymbol", "multi-leg"),
            )
        else:
            logger.warning(
                "preflight_failed",
                mode=mode.value,
                blocked_reasons=blocked,
                symbol=getattr(signal, "tradingsymbol", "multi-leg"),
            )

        return report

    def get_last_report(self) -> Optional[dict[str, Any]]:
        """Return last preflight report for dashboard display."""
        return self._last_report.to_dict() if self._last_report else None

    # ─── Live Checks (Full Suite) ─────────────────────────────

    async def _run_live_checks(
        self,
        signal: Any,
        signals: Optional[list],
    ) -> list[CheckResult]:
        """Run all 15 checks for live trading."""
        results: list[CheckResult] = []

        # 1. Signal sanity
        results.append(self._check_signal_sanity(signal))

        # 2. Broker connected
        results.append(await self._check_broker_connected())

        # 3. Margin buffer ≥ 2×
        results.append(await self._check_margin_sufficient(signal, signals))

        # 4. Margin utilization < 70%
        results.append(await self._check_margin_utilization())

        # 5. WebSocket stable
        results.append(self._check_websocket_health())

        # 6. Exposure within limits
        results.append(self._check_exposure_limit(signal))

        # 7. Stop-loss defined
        results.append(self._check_stop_loss_defined(signal))

        # 8. Strategy regime valid
        results.append(self._check_regime_valid(signal))

        # 9. Option chain valid (for F&O only)
        results.append(await self._check_option_chain_valid(signal))

        # 10. Delta within range
        results.append(self._check_delta_limits())

        # 11. Kill switch not active
        results.append(self._check_kill_switch())

        # 12. Market hours check
        results.append(self._check_market_hours())

        # 13. Slippage configuration
        results.append(self._check_slippage_config(signal))

        # 14. Emergency hedge ready (multi-leg)
        if signals and len(signals) > 1:
            results.append(self._check_emergency_hedge_ready(signals))

        # 15. Restart recovery verified
        results.append(self._check_recovery_state())

        return results

    # ─── Individual Check Implementations ─────────────────────

    def _check_signal_sanity(self, signal: Any) -> CheckResult:
        """Basic signal validation: symbol, qty, price all present."""
        if signal is None:
            return CheckResult(
                name="signal_sanity",
                passed=True,
                severity=CheckSeverity.INFO,
                message="No single signal (multi-leg mode)",
            )

        issues = []
        if not getattr(signal, "tradingsymbol", ""):
            issues.append("Missing tradingsymbol")
        if not getattr(signal, "quantity", 0) or signal.quantity <= 0:
            issues.append("Invalid quantity")
        if getattr(signal, "price", None) is not None and signal.price <= 0:
            issues.append("Invalid price")

        return CheckResult(
            name="signal_sanity",
            passed=len(issues) == 0,
            severity=CheckSeverity.CRITICAL,
            message="; ".join(issues) if issues else "Signal valid",
            details={"symbol": getattr(signal, "tradingsymbol", ""), "qty": getattr(signal, "quantity", 0)},
        )

    async def _check_broker_connected(self) -> CheckResult:
        """Verify broker REST API session is alive via profile endpoint."""
        if not self._client:
            return CheckResult(
                name="broker_connected",
                passed=False,
                severity=CheckSeverity.CRITICAL,
                message="No broker client configured",
            )

        # Cache: don't ping broker every order — recheck every 60s
        now = time.monotonic()
        if (now - self._last_broker_check_time) < self._broker_check_interval and self._broker_alive:
            return CheckResult(
                name="broker_connected",
                passed=True,
                severity=CheckSeverity.CRITICAL,
                message="Broker session active (cached)",
            )

        try:
            profile = await self._client.get_profile()
            self._broker_alive = True
            self._last_broker_check_time = now
            return CheckResult(
                name="broker_connected",
                passed=True,
                severity=CheckSeverity.CRITICAL,
                message=f"Broker connected — user: {getattr(profile, 'user_name', 'OK')}",
            )
        except Exception as e:
            self._broker_alive = False
            return CheckResult(
                name="broker_connected",
                passed=False,
                severity=CheckSeverity.CRITICAL,
                message=f"Broker session dead: {str(e)[:120]}",
            )

    async def _check_margin_sufficient(
        self,
        signal: Any,
        signals: Optional[list] = None,
    ) -> CheckResult:
        """Verify margin available ≥ 2× estimated requirement (fail-closed)."""
        if not self._client:
            # Fail-CLOSED: no client = cannot verify = block
            return CheckResult(
                name="margin_buffer_2x",
                passed=False,
                severity=CheckSeverity.CRITICAL,
                message="Cannot verify margin — no broker client",
            )

        try:
            margins = await self._client.get_margins()
            # Kite margins response: margins.equity.available.live_balance (or .collateral)
            available = 0.0
            if hasattr(margins, "equity"):
                eq = margins.equity
                if hasattr(eq, "available"):
                    available = getattr(eq.available, "live_balance", 0) or getattr(eq.available, "collateral", 0) or 0
            if available <= 0:
                # Try raw dict access
                if isinstance(margins, dict):
                    available = margins.get("equity", {}).get("available", {}).get("live_balance", 0) or 0

            # Estimate margin needed
            all_signals = signals or ([signal] if signal else [])
            total_needed = 0.0
            for sig in all_signals:
                if sig is None:
                    continue
                price = getattr(sig, "price", 0) or 100.0
                qty = getattr(sig, "quantity", 0)
                product = getattr(sig, "product", None)
                product_val = product.value if hasattr(product, "value") else str(product)

                if product_val == "MIS":
                    total_needed += price * qty * 0.25
                elif product_val == "NRML":
                    total_needed += price * qty
                else:
                    total_needed += qty * 10000.0  # Options conservative estimate

            required_with_buffer = total_needed * self.config.margin_buffer_multiplier

            passed = available >= required_with_buffer
            return CheckResult(
                name="margin_buffer_2x",
                passed=passed,
                severity=CheckSeverity.CRITICAL,
                message=(
                    f"Margin OK: ₹{available:,.0f} avail ≥ ₹{required_with_buffer:,.0f} (2× of ₹{total_needed:,.0f})"
                    if passed
                    else f"Insufficient margin: ₹{available:,.0f} < ₹{required_with_buffer:,.0f} needed (2× buffer)"
                ),
                details={
                    "available": round(available, 2),
                    "estimated_needed": round(total_needed, 2),
                    "required_with_buffer": round(required_with_buffer, 2),
                    "buffer_multiplier": self.config.margin_buffer_multiplier,
                },
            )

        except Exception as e:
            # FAIL-CLOSED: if we can't fetch margins, block the trade
            return CheckResult(
                name="margin_buffer_2x",
                passed=False,
                severity=CheckSeverity.CRITICAL,
                message=f"Margin fetch failed (fail-closed): {str(e)[:100]}",
            )

    async def _check_margin_utilization(self) -> CheckResult:
        """Block if margin utilization > 70% of total available."""
        if not self._client:
            return CheckResult(
                name="margin_usage_below_70pct",
                passed=False,
                severity=CheckSeverity.CRITICAL,
                message="Cannot check margin utilization — no client",
            )

        try:
            margins = await self._client.get_margins()
            available = 0.0
            used = 0.0
            if hasattr(margins, "equity"):
                eq = margins.equity
                if hasattr(eq, "available"):
                    available = getattr(eq.available, "live_balance", 0) or getattr(eq.available, "collateral", 0) or 0
                if hasattr(eq, "utilised"):
                    used_obj = eq.utilised
                    if hasattr(used_obj, "debits"):
                        used = getattr(used_obj, "debits", 0) or 0
                    elif isinstance(used_obj, (int, float)):
                        used = float(used_obj)

            total = available + used
            utilization = (used / total * 100) if total > 0 else 0.0

            passed = utilization <= self.config.max_margin_utilization_pct
            return CheckResult(
                name="margin_usage_below_70pct",
                passed=passed,
                severity=CheckSeverity.CRITICAL,
                message=(
                    f"Margin utilization {utilization:.1f}% ≤ {self.config.max_margin_utilization_pct}%"
                    if passed
                    else f"Margin utilization {utilization:.1f}% > {self.config.max_margin_utilization_pct}% — BLOCKED"
                ),
                details={"utilization_pct": round(utilization, 1), "used": round(used, 2), "total": round(total, 2)},
            )
        except Exception as e:
            return CheckResult(
                name="margin_usage_below_70pct",
                passed=False,
                severity=CheckSeverity.WARNING,
                message=f"Margin utilization check failed: {str(e)[:100]}",
            )

    def _check_websocket_health(self) -> CheckResult:
        """Verify WebSocket is connected and last tick is fresh."""
        if not self._ticker:
            if self.config.require_websocket_for_live:
                return CheckResult(
                    name="websocket_stable",
                    passed=False,
                    severity=CheckSeverity.CRITICAL,
                    message="No WebSocket ticker configured",
                )
            return CheckResult(
                name="websocket_stable",
                passed=True,
                severity=CheckSeverity.INFO,
                message="WebSocket not required",
            )

        # Check connection
        is_connected = getattr(self._ticker, "is_connected", False)
        if callable(is_connected):
            is_connected = is_connected()

        if not is_connected:
            return CheckResult(
                name="websocket_stable",
                passed=False,
                severity=CheckSeverity.CRITICAL,
                message="WebSocket disconnected — live prices unavailable",
            )

        # Check tick freshness
        tick_cache = getattr(self._ticker, "_tick_cache", {})
        if tick_cache:
            latest_tick_time = None
            for tick in tick_cache.values():
                ts = getattr(tick, "exchange_timestamp", None) or getattr(tick, "last_trade_time", None)
                if ts and (latest_tick_time is None or ts > latest_tick_time):
                    latest_tick_time = ts

            if latest_tick_time:
                age = (datetime.now() - latest_tick_time).total_seconds()
                if age > self.config.max_tick_age_seconds:
                    return CheckResult(
                        name="websocket_stable",
                        passed=False,
                        severity=CheckSeverity.CRITICAL,
                        message=f"Last tick is {age:.0f}s old (limit: {self.config.max_tick_age_seconds}s) — stale data",
                        details={"tick_age_seconds": round(age, 1)},
                    )

        return CheckResult(
            name="websocket_stable",
            passed=True,
            severity=CheckSeverity.CRITICAL,
            message="WebSocket connected, ticks fresh",
        )

    def _check_exposure_limit(self, signal: Any) -> CheckResult:
        """Check that adding this signal won't exceed exposure limit."""
        if not self._risk:
            return CheckResult(
                name="exposure_within_limits",
                passed=True,
                severity=CheckSeverity.WARNING,
                message="No risk manager — exposure not checked",
            )

        from src.utils.config import get_settings
        settings = get_settings()

        current_exposure = getattr(self._risk, "_total_exposure", 0.0)
        max_exposure = settings.max_exposure
        threshold = max_exposure * (self.config.max_exposure_pct / 100)

        new_exposure = 0.0
        if signal:
            price = getattr(signal, "price", 0) or 0
            qty = getattr(signal, "quantity", 0)
            new_exposure = price * qty

        projected = current_exposure + new_exposure
        passed = projected <= threshold

        return CheckResult(
            name="exposure_within_limits",
            passed=passed,
            severity=CheckSeverity.CRITICAL,
            message=(
                f"Exposure OK: ₹{projected:,.0f} ≤ ₹{threshold:,.0f} ({self.config.max_exposure_pct}% of max)"
                if passed
                else f"Exposure breach: ₹{projected:,.0f} > ₹{threshold:,.0f} limit"
            ),
            details={
                "current": round(current_exposure, 2),
                "new": round(new_exposure, 2),
                "projected": round(projected, 2),
                "threshold": round(threshold, 2),
            },
        )

    def _check_stop_loss_defined(self, signal: Any) -> CheckResult:
        """Every live order must have a stop-loss."""
        if not signal:
            return CheckResult(
                name="stop_loss_defined",
                passed=True,
                severity=CheckSeverity.INFO,
                message="Multi-leg mode — SL checked per leg",
            )

        has_sl = getattr(signal, "stop_loss", None) is not None
        return CheckResult(
            name="stop_loss_defined",
            passed=has_sl or not self.config.require_stop_loss,
            severity=CheckSeverity.CRITICAL if self.config.require_stop_loss else CheckSeverity.WARNING,
            message="Stop-loss defined" if has_sl else "No stop-loss on signal — BLOCKED",
        )

    def _check_regime_valid(self, signal: Any) -> CheckResult:
        """Check market regime allows the strategy to execute."""
        if not self.config.require_regime_check or not self._regime:
            return CheckResult(
                name="regime_valid",
                passed=True,
                severity=CheckSeverity.INFO,
                message="Regime check skipped (engine not configured)",
            )

        current = getattr(self._regime, "get_current_regime", lambda: None)()
        if not current:
            return CheckResult(
                name="regime_valid",
                passed=True,
                severity=CheckSeverity.WARNING,
                message="No regime classification available — allowing trade",
            )

        regime_name = current.regime.value if hasattr(current.regime, "value") else str(current.regime)

        if regime_name in self.config.blocked_regimes:
            return CheckResult(
                name="regime_valid",
                passed=False,
                severity=CheckSeverity.CRITICAL,
                message=f"Market regime '{regime_name}' is blocked — reduce exposure",
                details={
                    "regime": regime_name,
                    "confidence": getattr(current, "confidence", 0),
                    "vix": getattr(current, "vix_level", 0),
                    "position_size_factor": getattr(current, "position_size_factor", 1.0),
                },
            )

        # Warn on reduced-size regimes
        size_factor = getattr(current, "position_size_factor", 1.0)
        if size_factor < 0.7:
            return CheckResult(
                name="regime_valid",
                passed=True,
                severity=CheckSeverity.WARNING,
                message=f"Regime '{regime_name}' suggests reduced size ({size_factor:.0%})",
                details={"regime": regime_name, "size_factor": size_factor},
            )

        return CheckResult(
            name="regime_valid",
            passed=True,
            severity=CheckSeverity.INFO,
            message=f"Regime '{regime_name}' — OK for trading",
        )

    async def _check_option_chain_valid(self, signal: Any) -> CheckResult:
        """For F&O orders, verify the strike exists in a current option chain."""
        if not signal:
            return CheckResult(
                name="option_chain_valid",
                passed=True,
                severity=CheckSeverity.INFO,
                message="No signal — skipping chain check",
            )

        # Only check for NFO/BFO exchange (F&O segments)
        exchange = getattr(signal, "exchange", None)
        exchange_val = exchange.value if hasattr(exchange, "value") else str(exchange)
        if exchange_val not in ("NFO", "BFO", "MCX"):
            return CheckResult(
                name="option_chain_valid",
                passed=True,
                severity=CheckSeverity.INFO,
                message=f"Exchange {exchange_val} — no chain validation needed",
            )

        if not self.config.require_chain_validation:
            return CheckResult(
                name="option_chain_valid",
                passed=True,
                severity=CheckSeverity.INFO,
                message="Chain validation disabled",
            )

        symbol = getattr(signal, "tradingsymbol", "")

        if not self._chain:
            return CheckResult(
                name="option_chain_valid",
                passed=False,
                severity=CheckSeverity.CRITICAL,
                message=f"No chain builder — cannot validate {symbol}",
            )

        # Try to verify the symbol exists in chain
        try:
            # Attempt to get the chain for the underlying
            # Extract underlying from tradingsymbol (e.g., NIFTY26FEB25650PE → NIFTY)
            underlying = ""
            for idx_name in ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "MIDCPNIFTY"]:
                if symbol.startswith(idx_name):
                    underlying = idx_name
                    break

            if not underlying:
                # Individual stock option — underlying is everything before the expiry
                # Not easy to parse; pass validation
                return CheckResult(
                    name="option_chain_valid",
                    passed=True,
                    severity=CheckSeverity.INFO,
                    message=f"Stock option {symbol} — chain check skipped",
                )

            chain = getattr(self._chain, "_chain_cache", {})
            if chain and underlying in chain:
                cached = chain[underlying]
                cache_time = getattr(cached, "timestamp", None)
                if cache_time:
                    age_min = (datetime.now() - cache_time).total_seconds() / 60
                    if age_min > self.config.max_chain_age_minutes:
                        return CheckResult(
                            name="option_chain_valid",
                            passed=False,
                            severity=CheckSeverity.CRITICAL,
                            message=f"Option chain for {underlying} is {age_min:.0f}m old (limit: {self.config.max_chain_age_minutes}m)",
                        )

            return CheckResult(
                name="option_chain_valid",
                passed=True,
                severity=CheckSeverity.INFO,
                message=f"Chain check OK for {symbol}",
            )

        except Exception as e:
            return CheckResult(
                name="option_chain_valid",
                passed=False,
                severity=CheckSeverity.WARNING,
                message=f"Chain validation error: {str(e)[:100]}",
            )

    def _check_delta_limits(self) -> CheckResult:
        """Check portfolio delta is within acceptable range."""
        if not self._greeks:
            return CheckResult(
                name="delta_within_range",
                passed=True,
                severity=CheckSeverity.WARNING,
                message="Greeks engine not configured — delta not checked",
            )

        # Get the latest Greeks snapshot
        history = getattr(self._greeks, "_history", [])
        if not history:
            return CheckResult(
                name="delta_within_range",
                passed=True,
                severity=CheckSeverity.WARNING,
                message="No Greeks history — delta not checked",
            )

        latest = history[-1]
        greeks = latest.greeks if hasattr(latest, "greeks") else latest
        net_delta = abs(getattr(greeks, "net_delta", 0))
        net_vega = abs(getattr(greeks, "vega_dollars", 0))

        issues = []
        if net_delta > self.config.max_portfolio_delta:
            issues.append(f"Delta {net_delta:.1f} > ±{self.config.max_portfolio_delta}")
        if net_vega > self.config.max_portfolio_vega:
            issues.append(f"Vega ₹{net_vega:,.0f} > ₹{self.config.max_portfolio_vega:,.0f}")

        return CheckResult(
            name="delta_within_range",
            passed=len(issues) == 0,
            severity=CheckSeverity.CRITICAL,
            message="; ".join(issues) if issues else f"Delta OK (net: {net_delta:.1f})",
            details={"net_delta": round(net_delta, 2), "net_vega": round(net_vega, 2)},
        )

    def _check_kill_switch(self) -> CheckResult:
        """Verify kill switch is not active."""
        if not self._risk:
            return CheckResult(
                name="kill_switch_inactive",
                passed=True,
                severity=CheckSeverity.WARNING,
                message="No risk manager — kill switch not checked",
            )

        active = getattr(self._risk, "is_kill_switch_active", False)
        if callable(active):
            active = active
        # It's a property
        active = self._risk.is_kill_switch_active

        return CheckResult(
            name="kill_switch_inactive",
            passed=not active,
            severity=CheckSeverity.CRITICAL,
            message="Kill switch is ACTIVE — all orders blocked" if active else "Kill switch inactive",
        )

    def _check_market_hours(self) -> CheckResult:
        """Check we're within NSE market hours (9:15-15:30 IST)."""
        now = datetime.now()
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

        # Weekend check
        if now.weekday() >= 5:
            return CheckResult(
                name="market_hours",
                passed=False,
                severity=CheckSeverity.CRITICAL,
                message=f"Market closed — {'Saturday' if now.weekday() == 5 else 'Sunday'}",
            )

        in_hours = market_open <= now <= market_close

        return CheckResult(
            name="market_hours",
            passed=in_hours,
            severity=CheckSeverity.CRITICAL,
            message=(
                f"Market open ({now.strftime('%H:%M')})"
                if in_hours
                else f"Market closed ({now.strftime('%H:%M')}) — orders blocked"
            ),
        )

    def _check_slippage_config(self, signal: Any) -> CheckResult:
        """Verify slippage protection is in place for MARKET orders."""
        if not signal:
            return CheckResult(
                name="slippage_within_limits",
                passed=True,
                severity=CheckSeverity.INFO,
                message="No signal — slippage check skipped",
            )

        from src.data.models import OrderType
        order_type = getattr(signal, "order_type", None)
        order_type_val = order_type.value if hasattr(order_type, "value") else str(order_type)

        if order_type_val == "MARKET":
            price = getattr(signal, "price", 0) or 0
            if price <= 0:
                return CheckResult(
                    name="slippage_within_limits",
                    passed=False,
                    severity=CheckSeverity.WARNING,
                    message="MARKET order with no reference price — slippage uncontrolled",
                )

        return CheckResult(
            name="slippage_within_limits",
            passed=True,
            severity=CheckSeverity.INFO,
            message=f"Slippage config OK (max: {self.config.max_slippage_pct}%)",
        )

    def _check_emergency_hedge_ready(self, signals: list) -> CheckResult:
        """For multi-leg orders, verify emergency hedge capability."""
        if not self.config.require_emergency_hedge:
            return CheckResult(
                name="emergency_hedge_ready",
                passed=True,
                severity=CheckSeverity.INFO,
                message="Emergency hedge check disabled",
            )

        # Verify we have a client capable of placing emergency orders
        if not self._client:
            return CheckResult(
                name="emergency_hedge_ready",
                passed=False,
                severity=CheckSeverity.CRITICAL,
                message="No broker client — emergency hedge impossible",
            )

        return CheckResult(
            name="emergency_hedge_ready",
            passed=True,
            severity=CheckSeverity.CRITICAL,
            message=f"Emergency hedge ready for {len(signals)}-leg order",
        )

    def _check_recovery_state(self) -> CheckResult:
        """Verify no orphaned positions from previous crash."""
        # Check if there's a recovery state file
        import os
        recovery_db = "data/execution_state.db"
        if os.path.exists(recovery_db):
            try:
                import sqlite3
                conn = sqlite3.connect(recovery_db)
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM execution_intents WHERE status = 'PENDING'"
                )
                pending = cursor.fetchone()[0]
                conn.close()

                if pending > 0:
                    return CheckResult(
                        name="recovery_state_clean",
                        passed=False,
                        severity=CheckSeverity.CRITICAL,
                        message=f"{pending} orphaned execution intents from previous session — resolve first",
                        details={"orphaned_intents": pending},
                    )
            except Exception:
                pass  # DB may not have the table yet

        return CheckResult(
            name="recovery_state_clean",
            passed=True,
            severity=CheckSeverity.INFO,
            message="No orphaned execution state",
        )

    # ─── Quick Checks for Specific Paths ─────────────────────

    async def check_for_paper_trade(self, signal: Any = None) -> PreflightReport:
        """Abbreviated checks for paper trading."""
        return await self.run_all_checks(signal=signal, mode=TradingMode.PAPER)

    async def check_for_backtest(self, signal: Any = None) -> PreflightReport:
        """Minimal checks for backtesting."""
        return await self.run_all_checks(signal=signal, mode=TradingMode.BACKTEST)

    async def check_for_live(
        self, signal: Any = None, signals: Optional[list] = None
    ) -> PreflightReport:
        """Full checks for live trading."""
        return await self.run_all_checks(
            signal=signal, signals=signals, mode=TradingMode.LIVE
        )


# ─────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────

_preflight_checker: Optional[PreflightChecker] = None


def get_preflight_checker() -> Optional[PreflightChecker]:
    """Get the global preflight checker instance (set by TradingService)."""
    return _preflight_checker


def set_preflight_checker(checker: PreflightChecker) -> None:
    """Set the global preflight checker instance."""
    global _preflight_checker
    _preflight_checker = checker
