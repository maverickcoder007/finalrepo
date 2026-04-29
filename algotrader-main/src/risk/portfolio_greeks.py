"""
Portfolio Greeks Monitor
========================

Computes portfolio-level option Greeks for F&O positions:
  • Net Delta   — directional exposure
  • Net Gamma   — delta sensitivity to underlying moves
  • Net Theta   — time decay exposure (daily)
  • Net Vega    — volatility sensitivity

Uses Black-Scholes approximations when IV data is available from
the option chain, or falls back to estimates from moneyness + DTE.

Critical for the OI → FNO bridge system to monitor spread risk.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from typing import Any, Optional

logger = logging.getLogger("portfolio_greeks")

# ── Black-Scholes Helpers ────────────────────────────────────

_SQRT_2PI = math.sqrt(2 * math.pi)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def _bs_d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


def _bs_d2(d1: float, sigma: float, T: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    return d1 - sigma * math.sqrt(T)


@dataclass
class PositionGreeks:
    """Greeks for a single option position."""
    instrument: str = ""
    underlying: str = ""
    option_type: str = ""  # CE / PE
    strike: float = 0.0
    expiry: str = ""
    quantity: int = 0
    spot_price: float = 0.0
    iv: float = 0.0
    dte: int = 0

    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0

    # Position-level (delta * qty, etc.)
    position_delta: float = 0.0
    position_gamma: float = 0.0
    position_theta: float = 0.0
    position_vega: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in d:
            if isinstance(d[k], float):
                d[k] = round(d[k], 6)
        return d


@dataclass
class PortfolioGreeksReport:
    """Aggregate portfolio Greeks."""
    timestamp: str = ""

    # Net portfolio Greeks
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_theta: float = 0.0
    net_vega: float = 0.0

    # Absolute (gross) exposure
    gross_delta: float = 0.0
    gross_gamma: float = 0.0
    gross_theta: float = 0.0
    gross_vega: float = 0.0

    # By underlying
    delta_by_underlying: dict[str, float] = field(default_factory=dict)
    gamma_by_underlying: dict[str, float] = field(default_factory=dict)
    theta_by_underlying: dict[str, float] = field(default_factory=dict)
    vega_by_underlying: dict[str, float] = field(default_factory=dict)

    # Position details
    positions: list[dict[str, Any]] = field(default_factory=list)
    position_count: int = 0

    # Risk flags
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "net_delta": round(self.net_delta, 4),
            "net_gamma": round(self.net_gamma, 6),
            "net_theta": round(self.net_theta, 2),
            "net_vega": round(self.net_vega, 2),
            "gross_delta": round(self.gross_delta, 4),
            "gross_gamma": round(self.gross_gamma, 6),
            "gross_theta": round(self.gross_theta, 2),
            "gross_vega": round(self.gross_vega, 2),
            "delta_by_underlying": {k: round(v, 4) for k, v in self.delta_by_underlying.items()},
            "gamma_by_underlying": {k: round(v, 6) for k, v in self.gamma_by_underlying.items()},
            "theta_by_underlying": {k: round(v, 2) for k, v in self.theta_by_underlying.items()},
            "vega_by_underlying": {k: round(v, 2) for k, v in self.vega_by_underlying.items()},
            "positions": self.positions,
            "position_count": self.position_count,
            "warnings": self.warnings,
        }


# ── Known lot sizes ──────────────────────────────────────────

LOT_SIZES = {
    "NIFTY": 25, "BANKNIFTY": 15, "FINNIFTY": 25,
    "MIDCPNIFTY": 50, "SENSEX": 10,
}


class PortfolioGreeksMonitor:
    """
    Computes and tracks portfolio-level option Greeks.

    Parses F&O positions, identifies option legs, computes per-contract
    Greeks via Black-Scholes, and aggregates to portfolio level.
    """

    # Risk-free rate (India 10Y ~ 7%)
    RISK_FREE_RATE = 0.07

    def __init__(self) -> None:
        self._last_report: Optional[PortfolioGreeksReport] = None

    def compute_greeks(
        self,
        positions: list[dict[str, Any]],
        spot_prices: Optional[dict[str, float]] = None,
        iv_overrides: Optional[dict[str, float]] = None,
    ) -> PortfolioGreeksReport:
        """
        Compute portfolio Greeks from live positions.

        Args:
            positions: List of position dicts with keys:
                       tradingsymbol, quantity, exchange, instrument_type (optional),
                       last_price, strike (optional), expiry (optional).
            spot_prices: Map of underlying → spot price (e.g. {"NIFTY": 25000}).
            iv_overrides: Map of tradingsymbol → IV (decimal, e.g. 0.15 = 15%).
                         If not provided, IV is estimated from option price.

        Returns:
            PortfolioGreeksReport with net and gross Greeks.
        """
        report = PortfolioGreeksReport(timestamp=datetime.now().isoformat())
        spots = spot_prices or {}
        iv_map = iv_overrides or {}

        all_greeks: list[PositionGreeks] = []
        warnings: list[str] = []

        for pos in positions:
            qty = pos.get("quantity", 0) or 0
            if qty == 0:
                continue

            symbol = pos.get("tradingsymbol", "")
            exchange = pos.get("exchange", "")

            # Only process F&O instruments
            if exchange not in ("NFO", "BFO", "MCX") and not self._looks_like_option(symbol):
                continue

            parsed = self._parse_option_symbol(symbol)
            if not parsed:
                continue

            underlying = parsed["underlying"]
            option_type = parsed["option_type"]
            strike = parsed["strike"]
            expiry_str = parsed.get("expiry", "")

            spot = spots.get(underlying, 0)
            if spot <= 0:
                warnings.append(f"No spot price for {underlying}, skipping {symbol}")
                continue

            # DTE
            dte = self._compute_dte(expiry_str)
            if dte <= 0:
                dte = 1  # Expiry day

            T = dte / 365.0
            r = self.RISK_FREE_RATE

            # IV: use override, or estimate from option price
            iv = iv_map.get(symbol, 0)
            if iv <= 0:
                option_price = pos.get("last_price", 0) or 0
                iv = self._estimate_iv(spot, strike, T, r, option_type, option_price)
            if iv <= 0:
                iv = 0.20  # Default fallback 20%

            # Compute per-contract Greeks
            d1 = _bs_d1(spot, strike, T, r, iv)
            d2 = _bs_d2(d1, iv, T)

            if option_type == "CE":
                delta = _norm_cdf(d1)
            else:
                delta = _norm_cdf(d1) - 1

            sqrt_T = math.sqrt(T) if T > 0 else 1e-9
            gamma = _norm_pdf(d1) / (spot * iv * sqrt_T) if (spot * iv * sqrt_T) > 0 else 0
            theta_annual = (
                -(spot * _norm_pdf(d1) * iv) / (2 * sqrt_T)
                - r * strike * math.exp(-r * T) * (
                    _norm_cdf(d2) if option_type == "CE" else _norm_cdf(-d2)
                )
            )
            theta_daily = theta_annual / 365.0
            vega_annual = spot * _norm_pdf(d1) * sqrt_T
            vega_pct = vega_annual / 100.0  # Per 1% IV change

            # Lot size
            lot = LOT_SIZES.get(underlying, 1)
            lots = qty // lot if lot > 0 else qty

            pg = PositionGreeks(
                instrument=symbol,
                underlying=underlying,
                option_type=option_type,
                strike=strike,
                expiry=expiry_str,
                quantity=qty,
                spot_price=spot,
                iv=iv,
                dte=dte,
                delta=delta,
                gamma=gamma,
                theta=theta_daily,
                vega=vega_pct,
                position_delta=delta * qty,
                position_gamma=gamma * qty,
                position_theta=theta_daily * qty,
                position_vega=vega_pct * qty,
            )
            all_greeks.append(pg)

        # ── Aggregate ────────────────────────────────────────
        for pg in all_greeks:
            report.net_delta += pg.position_delta
            report.net_gamma += pg.position_gamma
            report.net_theta += pg.position_theta
            report.net_vega += pg.position_vega

            report.gross_delta += abs(pg.position_delta)
            report.gross_gamma += abs(pg.position_gamma)
            report.gross_theta += abs(pg.position_theta)
            report.gross_vega += abs(pg.position_vega)

            u = pg.underlying
            report.delta_by_underlying[u] = report.delta_by_underlying.get(u, 0) + pg.position_delta
            report.gamma_by_underlying[u] = report.gamma_by_underlying.get(u, 0) + pg.position_gamma
            report.theta_by_underlying[u] = report.theta_by_underlying.get(u, 0) + pg.position_theta
            report.vega_by_underlying[u] = report.vega_by_underlying.get(u, 0) + pg.position_vega

        report.positions = [pg.to_dict() for pg in all_greeks]
        report.position_count = len(all_greeks)
        report.warnings = warnings

        # ── Risk warnings ────────────────────────────────────
        if abs(report.net_delta) > 150:
            report.warnings.append(
                f"High net delta exposure: {report.net_delta:.2f} "
                f"(directional risk — approaches RiskManager.PORTFOLIO_DELTA_LIMIT=150)"
            )
        if abs(report.net_gamma) > 100:
            report.warnings.append(
                f"High gamma exposure: {report.net_gamma:.4f} "
                f"(convexity risk)"
            )
        if report.net_theta < -5000:
            report.warnings.append(
                f"Significant theta bleed: ₹{report.net_theta:.0f}/day"
            )
        if abs(report.net_vega) > 2000:
            report.warnings.append(
                f"High vega exposure: ₹{report.net_vega:.0f} per 1% IV move"
            )

        self._last_report = report
        return report

    def get_last_report(self) -> Optional[dict[str, Any]]:
        return self._last_report.to_dict() if self._last_report else None

    # ── Parsing helpers ──────────────────────────────────────

    @staticmethod
    def _looks_like_option(symbol: str) -> bool:
        upper = symbol.upper()
        return upper.endswith("CE") or upper.endswith("PE")

    @staticmethod
    def _parse_option_symbol(symbol: str) -> Optional[dict[str, Any]]:
        """
        Parse F&O tradingsymbol to extract underlying, strike, option_type, expiry.

        NSE format examples:
          NIFTY25MAR25000CE → underlying=NIFTY, strike=25000, CE
          BANKNIFTY25MAR45000PE → underlying=BANKNIFTY, strike=45000, PE
        """
        upper = symbol.upper().strip()
        if not (upper.endswith("CE") or upper.endswith("PE")):
            return None

        option_type = upper[-2:]
        rest = upper[:-2]

        # Known underlyings
        known = ["BANKNIFTY", "MIDCPNIFTY", "FINNIFTY", "NIFTY", "SENSEX"]
        underlying = ""
        for k in known:
            if rest.startswith(k):
                underlying = k
                rest = rest[len(k):]
                break

        if not underlying:
            # Unknown underlying — try to extract
            # Find where digits start for expiry
            for i, c in enumerate(rest):
                if c.isdigit():
                    underlying = rest[:i] if i > 0 else symbol[:3]
                    rest = rest[i:]
                    break
            if not underlying:
                return None

        # rest should be like "25MAR25000" or "2503025000"
        # Extract strike (last numeric part)
        strike_str = ""
        expiry_part = ""
        # Walk backwards to find strike
        i = len(rest) - 1
        while i >= 0 and rest[i].isdigit():
            i -= 1
        if i < len(rest) - 1:
            strike_str = rest[i + 1:]
            expiry_part = rest[:i + 1]

        strike = float(strike_str) if strike_str else 0.0

        return {
            "underlying": underlying,
            "option_type": option_type,
            "strike": strike,
            "expiry": expiry_part,
        }

    @staticmethod
    def _compute_dte(expiry_str: str) -> int:
        """Compute days to expiry from date string."""
        if not expiry_str:
            return 30  # Default
        try:
            # Try YYYY-MM-DD
            exp = datetime.strptime(expiry_str, "%Y-%m-%d").date()
            return max(0, (exp - date.today()).days)
        except ValueError:
            pass
        try:
            # Try DD-Mon-YYYY
            exp = datetime.strptime(expiry_str, "%d-%b-%Y").date()
            return max(0, (exp - date.today()).days)
        except ValueError:
            pass
        return 30  # Default

    @staticmethod
    def _estimate_iv(
        spot: float, strike: float, T: float, r: float,
        option_type: str, market_price: float,
    ) -> float:
        """
        Estimate IV from market price using bisection on BS formula.

        Returns IV as decimal (e.g. 0.15 for 15%).
        """
        if market_price <= 0 or spot <= 0 or strike <= 0 or T <= 0:
            return 0.0

        low, high = 0.01, 3.0  # IV range: 1% to 300%
        for _ in range(50):
            mid = (low + high) / 2
            d1 = _bs_d1(spot, strike, T, r, mid)
            d2 = _bs_d2(d1, mid, T)

            if option_type == "CE":
                price = spot * _norm_cdf(d1) - strike * math.exp(-r * T) * _norm_cdf(d2)
            else:
                price = strike * math.exp(-r * T) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)

            if abs(price - market_price) < 0.01:
                return mid
            if price > market_price:
                high = mid
            else:
                low = mid

        return (low + high) / 2
