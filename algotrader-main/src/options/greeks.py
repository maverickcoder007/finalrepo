from __future__ import annotations

import math
from typing import Optional

from scipy.stats import norm

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BlackScholes:
    @staticmethod
    def d1(
        spot: float,
        strike: float,
        rate: float,
        time_to_expiry: float,
        volatility: float,
    ) -> float:
        if time_to_expiry <= 0 or volatility <= 0 or spot <= 0 or strike <= 0:
            return 0.0
        return (
            math.log(spot / strike) + (rate + 0.5 * volatility ** 2) * time_to_expiry
        ) / (volatility * math.sqrt(time_to_expiry))

    @staticmethod
    def d2(
        spot: float,
        strike: float,
        rate: float,
        time_to_expiry: float,
        volatility: float,
    ) -> float:
        if time_to_expiry <= 0 or volatility <= 0:
            return 0.0
        d1_val = BlackScholes.d1(spot, strike, rate, time_to_expiry, volatility)
        return d1_val - volatility * math.sqrt(time_to_expiry)

    @staticmethod
    def call_price(
        spot: float,
        strike: float,
        rate: float,
        time_to_expiry: float,
        volatility: float,
    ) -> float:
        if time_to_expiry <= 0:
            return max(spot - strike, 0.0)
        d1_val = BlackScholes.d1(spot, strike, rate, time_to_expiry, volatility)
        d2_val = BlackScholes.d2(spot, strike, rate, time_to_expiry, volatility)
        return spot * norm.cdf(d1_val) - strike * math.exp(-rate * time_to_expiry) * norm.cdf(d2_val)

    @staticmethod
    def put_price(
        spot: float,
        strike: float,
        rate: float,
        time_to_expiry: float,
        volatility: float,
    ) -> float:
        if time_to_expiry <= 0:
            return max(strike - spot, 0.0)
        d1_val = BlackScholes.d1(spot, strike, rate, time_to_expiry, volatility)
        d2_val = BlackScholes.d2(spot, strike, rate, time_to_expiry, volatility)
        return strike * math.exp(-rate * time_to_expiry) * norm.cdf(-d2_val) - spot * norm.cdf(-d1_val)

    @staticmethod
    def delta(
        spot: float,
        strike: float,
        rate: float,
        time_to_expiry: float,
        volatility: float,
        option_type: str = "CE",
    ) -> float:
        if time_to_expiry <= 0:
            if option_type == "CE":
                return 1.0 if spot > strike else 0.0
            return -1.0 if spot < strike else 0.0
        d1_val = BlackScholes.d1(spot, strike, rate, time_to_expiry, volatility)
        if option_type == "CE":
            return norm.cdf(d1_val)
        return norm.cdf(d1_val) - 1.0

    @staticmethod
    def gamma(
        spot: float,
        strike: float,
        rate: float,
        time_to_expiry: float,
        volatility: float,
    ) -> float:
        if time_to_expiry <= 0 or volatility <= 0 or spot <= 0:
            return 0.0
        d1_val = BlackScholes.d1(spot, strike, rate, time_to_expiry, volatility)
        return norm.pdf(d1_val) / (spot * volatility * math.sqrt(time_to_expiry))

    @staticmethod
    def theta(
        spot: float,
        strike: float,
        rate: float,
        time_to_expiry: float,
        volatility: float,
        option_type: str = "CE",
    ) -> float:
        if time_to_expiry <= 0 or volatility <= 0:
            return 0.0
        d1_val = BlackScholes.d1(spot, strike, rate, time_to_expiry, volatility)
        d2_val = BlackScholes.d2(spot, strike, rate, time_to_expiry, volatility)
        common = -(spot * norm.pdf(d1_val) * volatility) / (2 * math.sqrt(time_to_expiry))
        if option_type == "CE":
            return (common - rate * strike * math.exp(-rate * time_to_expiry) * norm.cdf(d2_val)) / 365
        return (common + rate * strike * math.exp(-rate * time_to_expiry) * norm.cdf(-d2_val)) / 365

    @staticmethod
    def vega(
        spot: float,
        strike: float,
        rate: float,
        time_to_expiry: float,
        volatility: float,
    ) -> float:
        if time_to_expiry <= 0 or volatility <= 0:
            return 0.0
        d1_val = BlackScholes.d1(spot, strike, rate, time_to_expiry, volatility)
        return spot * norm.pdf(d1_val) * math.sqrt(time_to_expiry) / 100

    @staticmethod
    def implied_volatility(
        market_price: float,
        spot: float,
        strike: float,
        rate: float,
        time_to_expiry: float,
        option_type: str = "CE",
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> float:
        if time_to_expiry <= 0 or market_price <= 0:
            return 0.0

        vol = 0.3
        for _ in range(max_iterations):
            if option_type == "CE":
                price = BlackScholes.call_price(spot, strike, rate, time_to_expiry, vol)
            else:
                price = BlackScholes.put_price(spot, strike, rate, time_to_expiry, vol)

            vega_val = BlackScholes.vega(spot, strike, rate, time_to_expiry, vol) * 100
            if abs(vega_val) < 1e-10:
                break

            diff = market_price - price
            if abs(diff) < tolerance:
                return vol

            vol += diff / vega_val
            vol = max(0.01, min(vol, 5.0))

        return vol


def compute_all_greeks(
    spot: float,
    strike: float,
    rate: float,
    time_to_expiry: float,
    volatility: float,
    option_type: str = "CE",
) -> dict[str, float]:
    return {
        "delta": BlackScholes.delta(spot, strike, rate, time_to_expiry, volatility, option_type),
        "gamma": BlackScholes.gamma(spot, strike, rate, time_to_expiry, volatility),
        "theta": BlackScholes.theta(spot, strike, rate, time_to_expiry, volatility, option_type),
        "vega": BlackScholes.vega(spot, strike, rate, time_to_expiry, volatility),
        "iv": volatility,
        "theoretical_price": (
            BlackScholes.call_price(spot, strike, rate, time_to_expiry, volatility)
            if option_type == "CE"
            else BlackScholes.put_price(spot, strike, rate, time_to_expiry, volatility)
        ),
    }
