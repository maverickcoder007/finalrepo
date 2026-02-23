"""
Portfolio Greeks Engine — Dynamic Greeks tracking for F&O positions.

Most profitable options systems earn from theta decay and volatility
regime changes — not direction. Without Greeks journaling you cannot
optimize.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.derivatives.contracts import MultiLegPosition, OptionLeg
from src.options.greeks import BlackScholes, compute_all_greeks


@dataclass
class PortfolioGreeks:
    """Aggregate Greeks for the entire portfolio."""
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_theta: float = 0.0         # daily theta (P&L from time decay)
    net_vega: float = 0.0
    delta_dollars: float = 0.0     # delta × spot × lot_size (directional exposure in ₹)
    gamma_dollars: float = 0.0
    theta_dollars: float = 0.0
    vega_dollars: float = 0.0
    timestamp: datetime | None = None

    # Risk metrics
    delta_pct_of_capital: float = 0.0
    theta_pct_of_capital: float = 0.0
    max_vega_exposure: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "net_delta": round(self.net_delta, 4),
            "net_gamma": round(self.net_gamma, 6),
            "net_theta": round(self.net_theta, 4),
            "net_vega": round(self.net_vega, 4),
            "delta_dollars": round(self.delta_dollars, 2),
            "gamma_dollars": round(self.gamma_dollars, 2),
            "theta_dollars": round(self.theta_dollars, 2),
            "vega_dollars": round(self.vega_dollars, 2),
            "delta_pct_of_capital": round(self.delta_pct_of_capital, 2),
            "theta_pct_of_capital": round(self.theta_pct_of_capital, 4),
            "timestamp": str(self.timestamp) if self.timestamp else None,
        }


@dataclass
class GreeksSnapshot:
    """Point-in-time Greeks snapshot for journaling."""
    timestamp: datetime
    spot_price: float
    greeks: PortfolioGreeks
    positions_count: int = 0
    legs_count: int = 0


class GreeksEngine:
    """Compute and track portfolio-level Greeks.

    Features:
    • Per-leg Greek computation via Black-Scholes
    • Portfolio aggregation (net delta, gamma, theta, vega)
    • Dollar-denominated risk exposure
    • Greeks history for journaling
    • Risk limit checking (max delta, max vega)
    """

    def __init__(
        self,
        risk_free_rate: float = 0.065,
        max_portfolio_delta: float = 500.0,    # max net delta exposure
        max_portfolio_vega: float = 50000.0,   # max vega exposure in ₹
        max_portfolio_theta: float = -50000.0,  # max negative theta (daily bleed)
    ) -> None:
        self.risk_free_rate = risk_free_rate
        self.max_portfolio_delta = max_portfolio_delta
        self.max_portfolio_vega = max_portfolio_vega
        self.max_portfolio_theta = max_portfolio_theta
        self._history: list[GreeksSnapshot] = []

    def update_leg_greeks(
        self,
        leg: OptionLeg,
        spot: float,
    ) -> None:
        """Recompute Greeks for a single leg."""
        contract = leg.contract
        if not contract.is_option or contract.strike is None:
            # Futures: delta = 1 (long) or -1 (short)
            if contract.is_future:
                leg.delta = 1.0 if leg.is_long else -1.0
                leg.gamma = 0.0
                leg.theta = 0.0
                leg.vega = 0.0
            return

        tte = max(contract.tte, 1 / 365)
        iv = leg.iv_current if leg.iv_current > 0 else (leg.iv_at_entry or 0.20)
        opt_type = "CE" if contract.is_call else "PE"

        greeks = compute_all_greeks(
            spot, contract.strike, self.risk_free_rate, tte, iv, opt_type
        )

        leg.delta = greeks.get("delta", 0)
        leg.gamma = greeks.get("gamma", 0)
        leg.theta = greeks.get("theta", 0)
        leg.vega = greeks.get("vega", 0)

        # Update current price
        if opt_type == "CE":
            leg.current_price = max(0.05, BlackScholes.call_price(
                spot, contract.strike, self.risk_free_rate, tte, iv
            ))
        else:
            leg.current_price = max(0.05, BlackScholes.put_price(
                spot, contract.strike, self.risk_free_rate, tte, iv
            ))

    def compute_portfolio_greeks(
        self,
        positions: list[MultiLegPosition],
        spot: float,
        capital: float = 0.0,
        timestamp: datetime | None = None,
    ) -> PortfolioGreeks:
        """Compute aggregate portfolio Greeks across all positions."""
        pg = PortfolioGreeks(timestamp=timestamp or datetime.now())

        for pos in positions:
            if pos.is_closed:
                continue
            for leg in pos.legs:
                if leg.is_closed:
                    continue

                # Recompute Greeks
                self.update_leg_greeks(leg, spot)

                lot_size = leg.contract.lot_size

                # Aggregate position-weighted Greeks
                pg.net_delta += leg.delta * leg.quantity
                pg.net_gamma += leg.gamma * abs(leg.quantity)
                pg.net_theta += leg.theta * leg.quantity
                pg.net_vega += leg.vega * abs(leg.quantity)

                # Dollar-denominated
                pg.delta_dollars += leg.delta * leg.quantity * spot * lot_size
                pg.gamma_dollars += leg.gamma * abs(leg.quantity) * spot * lot_size
                pg.theta_dollars += leg.theta * leg.quantity * lot_size
                pg.vega_dollars += leg.vega * abs(leg.quantity) * lot_size

        # Capital-relative metrics
        if capital > 0:
            pg.delta_pct_of_capital = (abs(pg.delta_dollars) / capital) * 100
            pg.theta_pct_of_capital = (pg.theta_dollars / capital) * 100

        pg.max_vega_exposure = abs(pg.vega_dollars)

        return pg

    def check_risk_limits(self, greeks: PortfolioGreeks) -> list[str]:
        """Check if portfolio Greeks exceed risk limits."""
        violations: list[str] = []

        if abs(greeks.net_delta) > self.max_portfolio_delta:
            violations.append(
                f"Delta limit breached: {greeks.net_delta:.2f} > ±{self.max_portfolio_delta}"
            )

        if abs(greeks.vega_dollars) > self.max_portfolio_vega:
            violations.append(
                f"Vega limit breached: ₹{greeks.vega_dollars:.0f} > ₹{self.max_portfolio_vega}"
            )

        if greeks.theta_dollars < self.max_portfolio_theta:
            violations.append(
                f"Theta limit breached: ₹{greeks.theta_dollars:.0f} < ₹{self.max_portfolio_theta}"
            )

        return violations

    def record_snapshot(
        self,
        positions: list[MultiLegPosition],
        spot: float,
        capital: float = 0.0,
        timestamp: datetime | None = None,
    ) -> GreeksSnapshot:
        """Record a Greeks snapshot for journaling."""
        greeks = self.compute_portfolio_greeks(positions, spot, capital, timestamp)
        open_positions = [p for p in positions if not p.is_closed]
        total_legs = sum(
            len([l for l in p.legs if not l.is_closed])
            for p in open_positions
        )
        snapshot = GreeksSnapshot(
            timestamp=timestamp or datetime.now(),
            spot_price=spot,
            greeks=greeks,
            positions_count=len(open_positions),
            legs_count=total_legs,
        )
        self._history.append(snapshot)

        # Keep last 2000 snapshots
        if len(self._history) > 2000:
            self._history = self._history[-2000:]

        return snapshot

    def get_greeks_history(self) -> list[dict[str, Any]]:
        """Get Greeks history for charting."""
        return [
            {
                "timestamp": str(s.timestamp),
                "spot": s.spot_price,
                **s.greeks.to_dict(),
                "positions": s.positions_count,
                "legs": s.legs_count,
            }
            for s in self._history
        ]
