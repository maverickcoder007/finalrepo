"""
Shared fixtures and synthetic data generators for F&O strategy testing.

Generates realistic NIFTY / BANKNIFTY underlying OHLCV data using
geometric Brownian motion (GBM) with regime-switching volatility,
intraday patterns, and gap opens.

All strategy × mode tests share these fixtures.
"""

from __future__ import annotations

import math
import random
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────
# Synthetic Underlying Price Generator
# ─────────────────────────────────────────────────────────

def generate_underlying_ohlcv(
    start_price: float = 24_500.0,
    bars: int = 252,
    annual_drift: float = 0.10,
    annual_vol: float = 0.16,
    seed: int = 42,
    symbol: str = "NIFTY",
    start_date: date | None = None,
) -> pd.DataFrame:
    """Generate realistic daily OHLCV data for an Indian index.

    Uses GBM with:
      - Regime-switching vol (normal → high-vol spikes)
      - Intraday high/low envelope from parkinson estimator
      - Gap opens (weekend / overnight moves)
      - Volume that correlates with absolute returns

    Args:
        start_price: initial spot price
        bars: number of trading bars
        annual_drift: annualized drift (μ)
        annual_vol: annualized base volatility (σ)
        seed: random seed for reproducibility
        symbol: underlying symbol (affects lot size defaults)
        start_date: first trading date (default: 252 bars back from today)

    Returns:
        DataFrame with columns: [timestamp, open, high, low, close, volume]
    """
    rng = np.random.default_rng(seed)

    if start_date is None:
        start_date = date.today() - timedelta(days=int(bars * 1.5))

    dt = 1 / 252  # daily time step
    mu = annual_drift
    sigma = annual_vol

    # Pre-allocate
    opens = np.zeros(bars)
    highs = np.zeros(bars)
    lows = np.zeros(bars)
    closes = np.zeros(bars)
    volumes = np.zeros(bars, dtype=int)

    price = start_price
    vol_regime = sigma  # current vol regime

    for i in range(bars):
        # Regime switching: 5% chance of switching between normal / high-vol
        if rng.random() < 0.05:
            vol_regime = sigma * rng.uniform(1.5, 2.5)  # spike
        elif rng.random() < 0.10:
            vol_regime = sigma  # revert

        daily_return = (mu - 0.5 * vol_regime ** 2) * dt + vol_regime * math.sqrt(dt) * rng.standard_normal()

        # Gap open (1.5% chance of ±0.5-1.5% gap)
        gap = 0.0
        if rng.random() < 0.015:
            gap = price * rng.uniform(-0.015, 0.015)

        open_price = price + gap
        close_price = open_price * math.exp(daily_return)

        # Intraday high/low: Parkinson-style envelope
        intraday_vol = abs(daily_return) + vol_regime * math.sqrt(dt) * 0.5
        intraday_range = open_price * intraday_vol
        high_price = max(open_price, close_price) + abs(rng.standard_normal()) * intraday_range * 0.5
        low_price = min(open_price, close_price) - abs(rng.standard_normal()) * intraday_range * 0.5
        low_price = max(low_price, close_price * 0.95)  # circuit breaker limit

        # Volume: higher on big moves, typical range 5M-20M for NIFTY
        base_volume = 12_000_000 if "NIFTY" in symbol else 8_000_000
        vol_multiplier = 1.0 + 5.0 * abs(daily_return)
        daily_volume = int(base_volume * vol_multiplier * rng.uniform(0.7, 1.3))

        opens[i] = round(open_price, 2)
        highs[i] = round(high_price, 2)
        lows[i] = round(low_price, 2)
        closes[i] = round(close_price, 2)
        volumes[i] = daily_volume

        price = close_price

    # Build date index (skip weekends)
    dates = []
    d = start_date
    while len(dates) < bars:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(dates),
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })
    return df


def generate_intraday_ohlcv(
    spot: float = 24_500.0,
    bars: int = 75,  # 75 × 5min = 6.25 hours (NSE session)
    interval_min: int = 5,
    daily_vol: float = 0.012,
    seed: int = 99,
) -> pd.DataFrame:
    """Generate intraday 5-minute OHLCV bars for a single session."""
    rng = np.random.default_rng(seed)
    bar_vol = daily_vol / math.sqrt(bars)

    base_time = datetime(2026, 2, 23, 9, 15)
    timestamps = [base_time + timedelta(minutes=interval_min * i) for i in range(bars)]

    opens = np.zeros(bars)
    highs = np.zeros(bars)
    lows = np.zeros(bars)
    closes = np.zeros(bars)
    volumes = np.zeros(bars, dtype=int)

    price = spot
    for i in range(bars):
        ret = rng.standard_normal() * bar_vol
        open_p = price
        close_p = price * (1 + ret)
        high_p = max(open_p, close_p) * (1 + abs(rng.standard_normal()) * bar_vol * 0.3)
        low_p = min(open_p, close_p) * (1 - abs(rng.standard_normal()) * bar_vol * 0.3)

        opens[i] = round(open_p, 2)
        highs[i] = round(high_p, 2)
        lows[i] = round(low_p, 2)
        closes[i] = round(close_p, 2)
        volumes[i] = int(rng.uniform(50_000, 500_000))

        price = close_p

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


# ─────────────────────────────────────────────────────────
# Pytest Fixtures
# ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def nifty_daily_data() -> pd.DataFrame:
    """252 bars of realistic NIFTY daily OHLCV."""
    return generate_underlying_ohlcv(
        start_price=24_500.0,
        bars=252,
        annual_drift=0.10,
        annual_vol=0.16,
        seed=42,
        symbol="NIFTY",
    )


@pytest.fixture(scope="session")
def banknifty_daily_data() -> pd.DataFrame:
    """252 bars of BANKNIFTY daily OHLCV."""
    return generate_underlying_ohlcv(
        start_price=51_000.0,
        bars=252,
        annual_drift=0.12,
        annual_vol=0.20,
        seed=123,
        symbol="BANKNIFTY",
    )


@pytest.fixture(scope="session")
def nifty_short_data() -> pd.DataFrame:
    """60 bars of NIFTY data (minimum for backtest)."""
    return generate_underlying_ohlcv(
        start_price=24_000.0,
        bars=60,
        annual_vol=0.14,
        seed=7,
        symbol="NIFTY",
    )


@pytest.fixture(scope="session")
def nifty_high_vol_data() -> pd.DataFrame:
    """100 bars of NIFTY data with high volatility (crash scenario)."""
    return generate_underlying_ohlcv(
        start_price=25_000.0,
        bars=100,
        annual_drift=-0.15,  # bearish drift
        annual_vol=0.35,     # high vol
        seed=666,
        symbol="NIFTY",
    )


@pytest.fixture(scope="session")
def nifty_intraday_data() -> pd.DataFrame:
    """75 × 5min bars for intraday testing."""
    return generate_intraday_ohlcv(spot=24_500.0, seed=99)


@pytest.fixture(scope="session")
def sensex_daily_data() -> pd.DataFrame:
    """120 bars of SENSEX data."""
    return generate_underlying_ohlcv(
        start_price=80_000.0,
        bars=120,
        annual_vol=0.15,
        seed=555,
        symbol="SENSEX",
    )
