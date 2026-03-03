"""
Capital Allocator — Hard Capital Allocation Rules
===================================================

Enforces portfolio-level capital constraints:
  • Max % of capital per strategy
  • Max % of capital per underlying
  • Max overnight exposure
  • Net delta exposure tracking

Reads live positions, margins, and strategy metadata to produce a real-time
allocation snapshot and flag breaches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger("capital_allocator")


@dataclass
class AllocationLimits:
    """Configurable capital allocation limits (all as fractions, e.g. 0.25 = 25%)."""
    max_per_strategy: float = 0.25        # 25 % of capital for any single strategy
    max_per_underlying: float = 0.30      # 30 % of capital in any single underlying
    max_overnight_exposure: float = 0.50  # 50 % of capital held overnight
    max_net_delta: float = 0.20           # 20 % of capital as net delta exposure
    max_single_position: float = 0.10     # 10 % of capital in a single position


@dataclass
class AllocationBreachment:
    """A single allocation rule breachment."""
    rule: str
    limit: float
    actual: float
    severity: str  # "warning" | "critical"
    detail: str = ""


@dataclass
class CapitalAllocationReport:
    """Full capital allocation snapshot."""
    timestamp: str = ""
    total_capital: float = 0.0
    available_margin: float = 0.0
    used_margin: float = 0.0

    # Breakdowns
    by_strategy: dict[str, float] = field(default_factory=dict)
    by_underlying: dict[str, float] = field(default_factory=dict)

    # Exposure metrics
    overnight_exposure: float = 0.0
    net_delta_exposure: float = 0.0
    gross_exposure: float = 0.0

    # Utilisation percentages
    strategy_utilisation: dict[str, float] = field(default_factory=dict)
    underlying_utilisation: dict[str, float] = field(default_factory=dict)
    overnight_utilisation: float = 0.0
    delta_utilisation: float = 0.0

    # Limits & breaches
    limits: dict[str, float] = field(default_factory=dict)
    breaches: list[dict[str, Any]] = field(default_factory=list)
    is_healthy: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_capital": round(self.total_capital, 2),
            "available_margin": round(self.available_margin, 2),
            "used_margin": round(self.used_margin, 2),
            "by_strategy": {k: round(v, 2) for k, v in self.by_strategy.items()},
            "by_underlying": {k: round(v, 2) for k, v in self.by_underlying.items()},
            "overnight_exposure": round(self.overnight_exposure, 2),
            "net_delta_exposure": round(self.net_delta_exposure, 2),
            "gross_exposure": round(self.gross_exposure, 2),
            "strategy_utilisation_pct": {k: round(v * 100, 2) for k, v in self.strategy_utilisation.items()},
            "underlying_utilisation_pct": {k: round(v * 100, 2) for k, v in self.underlying_utilisation.items()},
            "overnight_utilisation_pct": round(self.overnight_utilisation * 100, 2),
            "delta_utilisation_pct": round(self.delta_utilisation * 100, 2),
            "limits": {k: round(v * 100, 2) for k, v in self.limits.items()},
            "breaches": self.breaches,
            "is_healthy": self.is_healthy,
        }


class CapitalAllocator:
    """
    Tracks and enforces hard capital allocation rules.

    Consumes positions, margins, and strategy tags to compute real-time
    allocation and detect limit breaches.
    """

    def __init__(self, limits: Optional[AllocationLimits] = None) -> None:
        self._limits = limits or AllocationLimits()
        self._last_report: Optional[CapitalAllocationReport] = None

    # ── Public API ──────────────────────────────────────────

    @property
    def limits(self) -> AllocationLimits:
        return self._limits

    def update_limits(self, updates: dict[str, float]) -> AllocationLimits:
        """Update allocation limits from a dict."""
        for key, val in updates.items():
            if hasattr(self._limits, key) and isinstance(val, (int, float)):
                setattr(self._limits, key, float(val))
        return self._limits

    def get_limits(self) -> dict[str, float]:
        return {
            "max_per_strategy": self._limits.max_per_strategy,
            "max_per_underlying": self._limits.max_per_underlying,
            "max_overnight_exposure": self._limits.max_overnight_exposure,
            "max_net_delta": self._limits.max_net_delta,
            "max_single_position": self._limits.max_single_position,
        }

    def compute_allocation(
        self,
        positions: list[dict[str, Any]],
        margins: dict[str, Any],
        strategy_tags: Optional[dict[str, str]] = None,
        is_market_hours: bool = True,
    ) -> CapitalAllocationReport:
        """
        Compute a full capital allocation report.

        Args:
            positions: List of position dicts (from Kite API — net positions).
                       Each should have: tradingsymbol, quantity, last_price,
                       product, exchange, value (optional).
            margins: Margins dict with 'equity' and 'commodity' sections,
                     each with 'available.live_balance' and 'utilised.debits'.
            strategy_tags: Optional map of tradingsymbol → strategy name.
            is_market_hours: Whether market is currently open.

        Returns:
            CapitalAllocationReport with breakdowns and breach flags.
        """
        report = CapitalAllocationReport(
            timestamp=datetime.now().isoformat(),
            limits=self.get_limits(),
        )

        # ── Parse margins ───────────────────────────────────
        equity = margins.get("equity", {})
        available = equity.get("available", {})
        utilised = equity.get("utilised", {})
        live_balance = available.get("live_balance", 0.0) if isinstance(available, dict) else 0.0
        total_debits = utilised.get("debits", 0.0) if isinstance(utilised, dict) else 0.0

        report.available_margin = float(live_balance)
        report.used_margin = float(total_debits)
        report.total_capital = report.available_margin + report.used_margin
        if report.total_capital <= 0:
            # Fallback: use available as total
            report.total_capital = max(report.available_margin, 1.0)

        # ── Aggregate positions ─────────────────────────────
        strategy_exposure: dict[str, float] = {}
        underlying_exposure: dict[str, float] = {}
        overnight_exposure = 0.0
        net_delta_value = 0.0
        gross_exposure = 0.0

        tags = strategy_tags or {}

        for pos in positions:
            qty = pos.get("quantity", 0) or 0
            if qty == 0:
                continue

            price = pos.get("last_price", 0) or 0
            symbol = pos.get("tradingsymbol", "")
            product = pos.get("product", "")
            value = abs(qty * price)
            gross_exposure += value

            # Net delta contribution (long → positive, short → negative)
            net_delta_value += qty * price

            # Strategy attribution
            strat = tags.get(symbol, "untagged")
            strategy_exposure[strat] = strategy_exposure.get(strat, 0) + value

            # Underlying extraction (strip expiry / option suffixes)
            underlying = self._extract_underlying(symbol)
            underlying_exposure[underlying] = underlying_exposure.get(underlying, 0) + value

            # Overnight exposure: only CNC / NRML carry overnight
            if product in ("CNC", "NRML"):
                overnight_exposure += value

        report.by_strategy = strategy_exposure
        report.by_underlying = underlying_exposure
        report.overnight_exposure = overnight_exposure
        report.net_delta_exposure = net_delta_value
        report.gross_exposure = gross_exposure

        # ── Compute utilisation ──────────────────────────────
        cap = report.total_capital
        for strat, val in strategy_exposure.items():
            report.strategy_utilisation[strat] = val / cap if cap > 0 else 0
        for und, val in underlying_exposure.items():
            report.underlying_utilisation[und] = val / cap if cap > 0 else 0
        report.overnight_utilisation = overnight_exposure / cap if cap > 0 else 0
        report.delta_utilisation = abs(net_delta_value) / cap if cap > 0 else 0

        # ── Check breaches ───────────────────────────────────
        breaches: list[AllocationBreachment] = []

        for strat, util in report.strategy_utilisation.items():
            if util > self._limits.max_per_strategy:
                breaches.append(AllocationBreachment(
                    rule="max_per_strategy",
                    limit=self._limits.max_per_strategy,
                    actual=util,
                    severity="critical" if util > self._limits.max_per_strategy * 1.2 else "warning",
                    detail=f"Strategy '{strat}' using {util*100:.1f}% (limit {self._limits.max_per_strategy*100:.0f}%)",
                ))

        for und, util in report.underlying_utilisation.items():
            if util > self._limits.max_per_underlying:
                breaches.append(AllocationBreachment(
                    rule="max_per_underlying",
                    limit=self._limits.max_per_underlying,
                    actual=util,
                    severity="critical" if util > self._limits.max_per_underlying * 1.2 else "warning",
                    detail=f"Underlying '{und}' using {util*100:.1f}% (limit {self._limits.max_per_underlying*100:.0f}%)",
                ))

        if not is_market_hours and report.overnight_utilisation > self._limits.max_overnight_exposure:
            breaches.append(AllocationBreachment(
                rule="max_overnight_exposure",
                limit=self._limits.max_overnight_exposure,
                actual=report.overnight_utilisation,
                severity="critical",
                detail=f"Overnight exposure {report.overnight_utilisation*100:.1f}% (limit {self._limits.max_overnight_exposure*100:.0f}%)",
            ))

        if report.delta_utilisation > self._limits.max_net_delta:
            breaches.append(AllocationBreachment(
                rule="max_net_delta",
                limit=self._limits.max_net_delta,
                actual=report.delta_utilisation,
                severity="critical" if report.delta_utilisation > self._limits.max_net_delta * 1.5 else "warning",
                detail=f"Net delta {report.delta_utilisation*100:.1f}% (limit {self._limits.max_net_delta*100:.0f}%)",
            ))

        report.breaches = [
            {"rule": b.rule, "limit": round(b.limit * 100, 2), "actual": round(b.actual * 100, 2),
             "severity": b.severity, "detail": b.detail}
            for b in breaches
        ]
        report.is_healthy = len(breaches) == 0

        self._last_report = report
        return report

    def get_last_report(self) -> Optional[dict[str, Any]]:
        """Return the last computed report."""
        return self._last_report.to_dict() if self._last_report else None

    def check_pre_trade(
        self,
        proposed_value: float,
        strategy: str,
        underlying: str,
        capital: float,
    ) -> dict[str, Any]:
        """
        Pre-trade check: would this trade breach allocation limits?

        Returns dict with 'allowed': bool and 'warnings': list.
        """
        warnings = []
        if capital <= 0:
            return {"allowed": True, "warnings": ["Capital unknown — skipping allocation check"]}

        # Strategy check
        current_strat = 0.0
        if self._last_report:
            current_strat = self._last_report.by_strategy.get(strategy, 0)
        new_strat_pct = (current_strat + proposed_value) / capital
        if new_strat_pct > self._limits.max_per_strategy:
            warnings.append(
                f"Strategy '{strategy}' would be {new_strat_pct*100:.1f}% "
                f"(limit {self._limits.max_per_strategy*100:.0f}%)"
            )

        # Underlying check
        current_und = 0.0
        if self._last_report:
            current_und = self._last_report.by_underlying.get(underlying, 0)
        new_und_pct = (current_und + proposed_value) / capital
        if new_und_pct > self._limits.max_per_underlying:
            warnings.append(
                f"Underlying '{underlying}' would be {new_und_pct*100:.1f}% "
                f"(limit {self._limits.max_per_underlying*100:.0f}%)"
            )

        # Single position check
        single_pct = proposed_value / capital
        if single_pct > self._limits.max_single_position:
            warnings.append(
                f"Single position would be {single_pct*100:.1f}% "
                f"(limit {self._limits.max_single_position*100:.0f}%)"
            )

        blocked = any("critical" in w.lower() for w in warnings)
        return {"allowed": not blocked, "warnings": warnings}

    # ── Helpers ──────────────────────────────────────────────

    @staticmethod
    def _extract_underlying(tradingsymbol: str) -> str:
        """Extract underlying name from tradingsymbol (e.g., NIFTY25MAR25000CE → NIFTY)."""
        known = ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "MIDCPNIFTY"]
        upper = tradingsymbol.upper()
        for k in known:
            if upper.startswith(k):
                return k
        # For equities, symbol IS the underlying
        return tradingsymbol.split()[0] if tradingsymbol else "UNKNOWN"
