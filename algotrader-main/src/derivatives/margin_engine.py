"""
SPAN-like Margin Engine — Portfolio-level margin calculation.

Simulates NSE SPAN + Exposure margin for F&O positions.
Critical for realistic backtesting: many option strategies look
profitable but fail because margin spikes → forced liquidation.

Uses scenario-based approach:
• ±3%, ±5%, ±7% underlying moves
• IV shock: +25%, +50%, -25%
• Spread benefit for hedged positions
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.derivatives.contracts import (
    DerivativeContract,
    InstrumentType,
    MultiLegPosition,
    OptionLeg,
    StructureType,
)
from src.options.greeks import BlackScholes


@dataclass
class MarginResult:
    """Result of margin calculation."""
    span_margin: float = 0.0
    exposure_margin: float = 0.0
    premium_margin: float = 0.0      # for option buyers
    total_margin: float = 0.0
    spread_benefit: float = 0.0      # reduction for hedged positions
    worst_scenario: str = ""
    scenarios: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_margin": round(self.span_margin, 2),
            "exposure_margin": round(self.exposure_margin, 2),
            "premium_margin": round(self.premium_margin, 2),
            "total_margin": round(self.total_margin, 2),
            "spread_benefit": round(self.spread_benefit, 2),
            "worst_scenario": self.worst_scenario,
        }


class MarginEngine:
    """SPAN-like margin calculator for Indian F&O.

    Scenario matrix:
    • Price shocks: ±1σ, ±2σ, ±3σ (mapped to ±3%, ±5%, ±7%)
    • IV shocks: +25%, +50%, -25%
    • For each scenario, compute portfolio loss → SPAN = worst case
    • Exposure margin = % of notional (additional buffer)
    """

    # Scenario price moves (as fraction)
    PRICE_SCENARIOS = [-0.07, -0.05, -0.03, -0.01, 0.0, 0.01, 0.03, 0.05, 0.07]
    # IV multipliers for each price scenario
    IV_SCENARIOS = [1.50, 1.25, 1.10, 1.00, 1.00, 1.00, 0.90, 0.75, 0.50]

    # Exposure margin rates
    EXPOSURE_RATE_INDEX = 0.03     # 3% for index
    EXPOSURE_RATE_STOCK = 0.05     # 5% for stock
    EXPOSURE_RATE_DEEP_OTM = 0.02  # 2% for deep OTM

    # Minimum margin per lot
    MIN_MARGIN_PER_LOT = 5000.0

    def calculate_margin(
        self,
        position: MultiLegPosition,
        spot: float,
        risk_free_rate: float = 0.065,
    ) -> MarginResult:
        """Calculate SPAN-like margin for a multi-leg position."""
        if not position.legs:
            return MarginResult()

        # Separate long options (premium only) and short/futures (SPAN)
        long_option_legs = [l for l in position.legs if l.is_long and l.contract.is_option]
        margin_legs = [l for l in position.legs if not (l.is_long and l.contract.is_option)]

        # Premium margin for long options
        premium_margin = sum(
            abs(l.entry_price * l.quantity) for l in long_option_legs
        )

        if not margin_legs and not long_option_legs:
            return MarginResult(premium_margin=premium_margin, total_margin=premium_margin)

        # Compute SPAN across scenarios
        scenarios: list[dict[str, Any]] = []
        worst_loss = 0.0
        worst_scenario = ""

        for i, price_move in enumerate(self.PRICE_SCENARIOS):
            iv_mult = self.IV_SCENARIOS[i]
            scenario_spot = spot * (1 + price_move)

            # Calculate portfolio value at this scenario
            scenario_pnl = 0.0
            for leg in position.legs:
                contract = leg.contract
                if contract.is_option and contract.strike is not None:
                    tte = max(contract.tte, 1 / 365)
                    iv = (leg.iv_current or leg.iv_at_entry or 0.20) * iv_mult

                    if contract.is_call:
                        new_price = BlackScholes.call_price(
                            scenario_spot, contract.strike, risk_free_rate, tte, iv
                        )
                    else:
                        new_price = BlackScholes.put_price(
                            scenario_spot, contract.strike, risk_free_rate, tte, iv
                        )

                    # PnL = (new − current) × quantity (signed)
                    leg_pnl = (new_price - leg.current_price) * leg.quantity
                    scenario_pnl += leg_pnl

                elif contract.is_future:
                    # Futures: linear PnL
                    leg_pnl = (scenario_spot - leg.current_price) * leg.quantity * contract.lot_size
                    scenario_pnl += leg_pnl

            scenario_loss = -scenario_pnl  # loss is negative PnL
            desc = f"{"+" if price_move >= 0 else ""}{price_move*100:.0f}% / IV×{iv_mult:.2f}"
            scenarios.append({"scenario": desc, "pnl": round(scenario_pnl, 2), "loss": round(scenario_loss, 2)})

            if scenario_loss > worst_loss:
                worst_loss = scenario_loss
                worst_scenario = desc

        span_margin = max(worst_loss, 0)

        # Spread benefit for hedged positions
        spread_benefit = self._calculate_spread_benefit(position, spot)
        span_margin = max(span_margin - spread_benefit, 0)

        # Exposure margin
        notional = self._calculate_notional(position, spot)
        is_index = position.underlying.upper() in ("NIFTY", "BANKNIFTY", "FINNIFTY", "SENSEX", "MIDCPNIFTY")
        exposure_rate = self.EXPOSURE_RATE_INDEX if is_index else self.EXPOSURE_RATE_STOCK
        exposure_margin = notional * exposure_rate

        # Minimum margin
        total_lots = position.total_lots
        min_margin = total_lots * self.MIN_MARGIN_PER_LOT
        total_margin = max(span_margin + exposure_margin + premium_margin, min_margin)

        return MarginResult(
            span_margin=span_margin,
            exposure_margin=exposure_margin,
            premium_margin=premium_margin,
            total_margin=total_margin,
            spread_benefit=spread_benefit,
            worst_scenario=worst_scenario,
            scenarios=scenarios,
        )

    def _calculate_spread_benefit(self, position: MultiLegPosition, spot: float) -> float:
        """Calculate spread margin benefit for hedged positions."""
        if position.structure in (
            StructureType.IRON_CONDOR, StructureType.IRON_BUTTERFLY,
            StructureType.BULL_CALL_SPREAD, StructureType.BEAR_PUT_SPREAD,
            StructureType.BULL_PUT_SPREAD, StructureType.BEAR_CALL_SPREAD,
        ):
            # For defined-risk spreads: benefit = notional × 40-60%
            max_loss = position.max_loss
            if max_loss != float("inf"):
                lot_size = max((l.contract.lot_size for l in position.legs), default=1)
                return max(0, max_loss * 0.5)

        return 0.0

    def _calculate_notional(self, position: MultiLegPosition, spot: float) -> float:
        """Calculate total notional exposure."""
        notional = 0.0
        for leg in position.legs:
            lot_size = leg.contract.lot_size
            if leg.contract.is_future:
                notional += spot * abs(leg.quantity) * lot_size
            elif leg.contract.is_option and leg.is_short:
                notional += spot * abs(leg.quantity) * lot_size
        return notional

    def check_margin_available(
        self,
        position: MultiLegPosition,
        spot: float,
        available_margin: float,
        risk_free_rate: float = 0.065,
    ) -> tuple[bool, MarginResult]:
        """Check if there's enough margin for this position."""
        result = self.calculate_margin(position, spot, risk_free_rate)
        return available_margin >= result.total_margin, result

    def simulate_margin_spike(
        self,
        position: MultiLegPosition,
        spot: float,
        volatility_shock: float = 1.5,
        risk_free_rate: float = 0.065,
    ) -> MarginResult:
        """Simulate margin after volatility spike (e.g., event risk)."""
        # Shock all leg IVs
        for leg in position.legs:
            leg.iv_current = (leg.iv_current or 0.20) * volatility_shock
        result = self.calculate_margin(position, spot, risk_free_rate)
        # Restore
        for leg in position.legs:
            leg.iv_current = (leg.iv_current or 0.20) / volatility_shock
        return result
