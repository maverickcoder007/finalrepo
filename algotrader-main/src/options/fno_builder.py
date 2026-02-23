"""
F&O Custom Strategy Builder.

Allows users to create custom options strategies by defining legs
(BUY/SELL CE/PE at specific strikes or delta-based selection)
then backtest or paper trade them.

Each leg specifies:
  - action: buy / sell
  - option_type: CE / PE
  - strike_mode: absolute / offset / delta
  - strike_value: the strike price, offset from ATM, or delta target
  - lots: number of lots for this leg
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from src.options.chain import OptionChainData, OptionContract
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ────────────────────────────────────────────────────────────
# Data models
# ────────────────────────────────────────────────────────────

@dataclass
class FnOLeg:
    """A single leg of an F&O strategy."""
    action: str         # "buy" or "sell"
    option_type: str    # "CE" or "PE"
    strike_mode: str    # "absolute", "offset", "delta"
    strike_value: float  # strike price, offset from ATM, or delta target
    lots: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FnOLeg":
        return cls(
            action=d.get("action", "buy"),
            option_type=d.get("option_type", "CE"),
            strike_mode=d.get("strike_mode", "absolute"),
            strike_value=float(d.get("strike_value", 0)),
            lots=int(d.get("lots", 1)),
        )


@dataclass
class FnOStrategyConfig:
    """Complete configuration for a custom F&O strategy."""
    name: str
    description: str = ""
    underlying: str = "NIFTY"
    strategy_type: str = "custom"  # credit_spread, debit_spread, straddle, custom etc
    legs: list[FnOLeg] = field(default_factory=list)
    profit_target_pct: float = 50.0
    stop_loss_pct: float = 100.0
    max_positions: int = 1
    lot_size: int = 50  # auto-set from underlying
    entry_dte_min: int = 15
    entry_dte_max: int = 45

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "underlying": self.underlying,
            "strategy_type": self.strategy_type,
            "legs": [leg.to_dict() for leg in self.legs],
            "profit_target_pct": self.profit_target_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "max_positions": self.max_positions,
            "lot_size": self.lot_size,
            "entry_dte_min": self.entry_dte_min,
            "entry_dte_max": self.entry_dte_max,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FnOStrategyConfig":
        legs = [FnOLeg.from_dict(l) for l in d.get("legs", [])]
        return cls(
            name=d.get("name", "untitled"),
            description=d.get("description", ""),
            underlying=d.get("underlying", "NIFTY"),
            strategy_type=d.get("strategy_type", "custom"),
            legs=legs,
            profit_target_pct=float(d.get("profit_target_pct", 50)),
            stop_loss_pct=float(d.get("stop_loss_pct", 100)),
            max_positions=int(d.get("max_positions", 1)),
            lot_size=int(d.get("lot_size", 50)),
            entry_dte_min=int(d.get("entry_dte_min", 15)),
            entry_dte_max=int(d.get("entry_dte_max", 45)),
        )


# ────────────────────────────────────────────────────────────
# Pre-built F&O strategy templates
# ────────────────────────────────────────────────────────────

FNO_TEMPLATES: dict[str, dict[str, Any]] = {
    "bull_put_credit_spread": {
        "description": "Sell OTM Put + Buy further OTM Put (bullish, collect premium)",
        "strategy_type": "credit_spread",
        "legs": [
            {"action": "sell", "option_type": "PE", "strike_mode": "offset", "strike_value": -100, "lots": 1},
            {"action": "buy", "option_type": "PE", "strike_mode": "offset", "strike_value": -250, "lots": 1},
        ],
    },
    "bear_call_credit_spread": {
        "description": "Sell OTM Call + Buy further OTM Call (bearish, collect premium)",
        "strategy_type": "credit_spread",
        "legs": [
            {"action": "sell", "option_type": "CE", "strike_mode": "offset", "strike_value": 100, "lots": 1},
            {"action": "buy", "option_type": "CE", "strike_mode": "offset", "strike_value": 250, "lots": 1},
        ],
    },
    "iron_condor_template": {
        "description": "Sell OTM PE + Buy further OTM PE + Sell OTM CE + Buy further OTM CE",
        "strategy_type": "iron_condor",
        "legs": [
            {"action": "sell", "option_type": "PE", "strike_mode": "delta", "strike_value": 0.20, "lots": 1},
            {"action": "buy", "option_type": "PE", "strike_mode": "delta", "strike_value": 0.10, "lots": 1},
            {"action": "sell", "option_type": "CE", "strike_mode": "delta", "strike_value": 0.20, "lots": 1},
            {"action": "buy", "option_type": "CE", "strike_mode": "delta", "strike_value": 0.10, "lots": 1},
        ],
    },
    "long_straddle_template": {
        "description": "Buy ATM CE + Buy ATM PE (expecting big move)",
        "strategy_type": "straddle",
        "legs": [
            {"action": "buy", "option_type": "CE", "strike_mode": "offset", "strike_value": 0, "lots": 1},
            {"action": "buy", "option_type": "PE", "strike_mode": "offset", "strike_value": 0, "lots": 1},
        ],
    },
    "short_straddle_template": {
        "description": "Sell ATM CE + Sell ATM PE (expecting range-bound)",
        "strategy_type": "straddle",
        "legs": [
            {"action": "sell", "option_type": "CE", "strike_mode": "offset", "strike_value": 0, "lots": 1},
            {"action": "sell", "option_type": "PE", "strike_mode": "offset", "strike_value": 0, "lots": 1},
        ],
    },
    "long_strangle_template": {
        "description": "Buy OTM CE + Buy OTM PE (cheaper than straddle)",
        "strategy_type": "strangle",
        "legs": [
            {"action": "buy", "option_type": "CE", "strike_mode": "offset", "strike_value": 200, "lots": 1},
            {"action": "buy", "option_type": "PE", "strike_mode": "offset", "strike_value": -200, "lots": 1},
        ],
    },
    "ratio_spread_template": {
        "description": "Buy 1 ATM CE + Sell 2 OTM CE (ratio 1:2)",
        "strategy_type": "ratio_spread",
        "legs": [
            {"action": "buy", "option_type": "CE", "strike_mode": "offset", "strike_value": 0, "lots": 1},
            {"action": "sell", "option_type": "CE", "strike_mode": "offset", "strike_value": 250, "lots": 2},
        ],
    },
    "jade_lizard_template": {
        "description": "Short Put + Short Call Spread (no upside risk)",
        "strategy_type": "jade_lizard",
        "legs": [
            {"action": "sell", "option_type": "PE", "strike_mode": "offset", "strike_value": -200, "lots": 1},
            {"action": "sell", "option_type": "CE", "strike_mode": "offset", "strike_value": 100, "lots": 1},
            {"action": "buy", "option_type": "CE", "strike_mode": "offset", "strike_value": 250, "lots": 1},
        ],
    },
}


# ────────────────────────────────────────────────────────────
# Leg resolver — resolves abstract leg defs to concrete strikes
# ────────────────────────────────────────────────────────────

def resolve_legs(
    config: FnOStrategyConfig,
    spot_price: float,
    chain: Optional[OptionChainData] = None,
) -> list[dict[str, Any]]:
    """Resolve strategy legs to concrete strike prices.
    
    Returns list of dicts:
      {"action", "option_type", "strike", "lots", "contract": OptionContract|None}
    """
    resolved: list[dict[str, Any]] = []

    # Get available strikes from chain
    available_strikes = sorted([e.strike for e in chain.entries]) if chain else []
    atm_strike = _find_nearest_strike(spot_price, available_strikes) if available_strikes else round(spot_price / 50) * 50

    for leg in config.legs:
        strike: float
        contract: Optional[OptionContract] = None

        if leg.strike_mode == "absolute":
            strike = leg.strike_value
        elif leg.strike_mode == "offset":
            strike = atm_strike + leg.strike_value
        elif leg.strike_mode == "delta":
            # Find strike by delta from chain
            if chain:
                contract = _find_by_delta(chain, leg.option_type, leg.strike_value)
                strike = contract.strike if contract else atm_strike
            else:
                # Approximate: higher delta → closer to ATM
                offset = int((0.50 - abs(leg.strike_value)) * spot_price * 0.02)
                if leg.option_type == "CE":
                    strike = atm_strike + offset
                else:
                    strike = atm_strike - offset
        else:
            strike = atm_strike

        # Snap to nearest available strike
        if available_strikes:
            strike = _find_nearest_strike(strike, available_strikes)

        # Find contract from chain
        if chain and not contract:
            for entry in chain.entries:
                if entry.strike == strike:
                    contract = entry.ce if leg.option_type == "CE" else entry.pe
                    break

        resolved.append({
            "action": leg.action,
            "option_type": leg.option_type,
            "strike": strike,
            "lots": leg.lots,
            "contract": contract,
        })

    return resolved


def compute_strategy_payoff(
    resolved_legs: list[dict[str, Any]],
    spot_price: float,
    lot_size: int = 50,
    price_range_pct: float = 10.0,
    steps: int = 100,
) -> dict[str, Any]:
    """Compute payoff profile for the resolved strategy at expiry.
    
    Returns {
        "prices": [...],
        "payoff": [...],
        "max_profit": float,
        "max_loss": float,
        "breakeven": [float, ...],
        "net_premium": float,
    }
    """
    low = spot_price * (1 - price_range_pct / 100)
    high = spot_price * (1 + price_range_pct / 100)
    prices = [low + (high - low) * i / steps for i in range(steps + 1)]

    # Net premium received/paid
    net_premium = 0.0
    for leg in resolved_legs:
        premium = 0.0
        c = leg.get("contract")
        if c:
            premium = c.last_price
        # Sell → receive premium, Buy → pay premium
        direction = 1 if leg["action"] == "sell" else -1
        net_premium += direction * premium * leg["lots"] * lot_size

    payoffs: list[float] = []
    for price in prices:
        total = 0.0
        for leg in resolved_legs:
            strike = leg["strike"]
            opt_type = leg["option_type"]
            action = leg["action"]
            lots = leg["lots"]

            # Intrinsic value at expiry
            if opt_type == "CE":
                intrinsic = max(price - strike, 0)
            else:
                intrinsic = max(strike - price, 0)

            # For seller, P&L = premium - intrinsic; for buyer, P&L = intrinsic - premium
            c = leg.get("contract")
            premium = c.last_price if c else 0

            if action == "sell":
                leg_pnl = (premium - intrinsic) * lots * lot_size
            else:
                leg_pnl = (intrinsic - premium) * lots * lot_size

            total += leg_pnl
        payoffs.append(round(total, 2))

    # Find breakevens (where payoff crosses 0)
    breakevens: list[float] = []
    for i in range(len(payoffs) - 1):
        if (payoffs[i] <= 0 <= payoffs[i + 1]) or (payoffs[i] >= 0 >= payoffs[i + 1]):
            # Linear interpolation
            if payoffs[i + 1] != payoffs[i]:
                ratio = -payoffs[i] / (payoffs[i + 1] - payoffs[i])
                be = prices[i] + ratio * (prices[i + 1] - prices[i])
                breakevens.append(round(be, 2))

    return {
        "prices": [round(p, 2) for p in prices],
        "payoff": payoffs,
        "max_profit": max(payoffs) if payoffs else 0,
        "max_loss": min(payoffs) if payoffs else 0,
        "breakeven": breakevens,
        "net_premium": round(net_premium, 2),
    }


# ────────────────────────────────────────────────────────────
# Persistence — save/load custom F&O strategies
# ────────────────────────────────────────────────────────────

class FnOStrategyStore:
    """Persists custom F&O strategy configs to JSON."""

    def __init__(self, path: str = "data/fno_custom_strategies.json") -> None:
        self._path = path
        self._configs: dict[str, FnOStrategyConfig] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._path):
            try:
                with open(self._path, "r") as f:
                    data = json.load(f)
                for item in data:
                    cfg = FnOStrategyConfig.from_dict(item)
                    self._configs[cfg.name] = cfg
                logger.info("fno_strategy_store_loaded", count=len(self._configs))
            except Exception as e:
                logger.error("fno_strategy_store_load_error", error=str(e))

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        with open(self._path, "w") as f:
            json.dump([cfg.to_dict() for cfg in self._configs.values()], f, indent=2)

    def save(self, config: FnOStrategyConfig) -> dict[str, Any]:
        self._configs[config.name] = config
        self._save()
        return {"status": "saved", "name": config.name}

    def delete(self, name: str) -> dict[str, Any]:
        if name in self._configs:
            del self._configs[name]
            self._save()
            return {"status": "deleted", "name": name}
        return {"error": f"Strategy '{name}' not found"}

    def get(self, name: str) -> Optional[FnOStrategyConfig]:
        return self._configs.get(name)

    def list_all(self) -> list[dict[str, Any]]:
        return [cfg.to_dict() for cfg in self._configs.values()]

    def get_templates(self) -> list[dict[str, Any]]:
        """Return pre-built templates for the UI."""
        result = []
        for name, tmpl in FNO_TEMPLATES.items():
            result.append({
                "name": name,
                "description": tmpl["description"],
                "strategy_type": tmpl["strategy_type"],
                "legs": tmpl["legs"],
            })
        return result


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────

def _find_nearest_strike(target: float, strikes: list[float]) -> float:
    if not strikes:
        return target
    return min(strikes, key=lambda s: abs(s - target))


def _find_by_delta(
    chain: OptionChainData, option_type: str, target_delta: float
) -> Optional[OptionContract]:
    best: Optional[OptionContract] = None
    best_diff = float("inf")
    for entry in chain.entries:
        contract = entry.ce if option_type == "CE" else entry.pe
        if contract and contract.last_price > 0:
            diff = abs(abs(contract.delta) - abs(target_delta))
            if diff < best_diff:
                best_diff = diff
                best = contract
    return best


# ────────────────────────────────────────────────────────────
# Underlying lot sizes
# ────────────────────────────────────────────────────────────

UNDERLYING_LOT_SIZES: dict[str, int] = {
    "NIFTY": 50,
    "BANKNIFTY": 15,
    "FINNIFTY": 25,
    "MIDCPNIFTY": 50,
    "SENSEX": 10,
}

UNDERLYING_STRIKE_GAPS: dict[str, float] = {
    "NIFTY": 50,
    "BANKNIFTY": 100,
    "FINNIFTY": 50,
    "MIDCPNIFTY": 25,
    "SENSEX": 100,
}
