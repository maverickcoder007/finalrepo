from __future__ import annotations

import abc
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from src.data.models import Signal, Tick
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseStrategy(abc.ABC):
    """Base class for all trading strategies.

    Supports timeframe-aware parameter scaling: when ``set_timeframe``
    is called the strategy records the operating interval and computes
    a *moderate* scaling multiplier so that lookback-period based params
    (EMA span, RSI period, etc.) are automatically adjusted for
    intraday vs daily data.

    Multiplier table (Indian markets, 375 min / trading day):
        day → 1.0, 60min → 1.0, 30min → 1.5, 15min → 2.0,
        10min → 2.5, 5min → 3.0, 3min → 4.0, minute → 5.0
    """

    # Moderate scaling factors – NOT linear bar-count ratio.
    # Designed so that a 9-bar EMA on daily stays 9, while on 5-min
    # it becomes 27 (≈ ~2.25 hours) which is a sensible intraday window.
    _TF_MULTIPLIERS: dict[str, float] = {
        "day": 1.0,
        "60minute": 1.0,
        "30minute": 1.5,
        "15minute": 2.0,
        "10minute": 2.5,
        "5minute": 3.0,
        "3minute": 4.0,
        "minute": 5.0,
    }

    def __init__(self, name: str, params: Optional[dict[str, Any]] = None) -> None:
        self.name = name
        self.params = params or {}
        self.is_active = True
        self._tick_buffer: dict[int, list[Tick]] = {}
        self._bar_data: dict[int, pd.DataFrame] = {}
        self._signals: list[Signal] = []
        self._last_signal_time: Optional[datetime] = None
        self._timeframe: str = "day"
        self._tf_multiplier: float = 1.0

    # ─── Timeframe scaling ────────────────────────────────────

    def set_timeframe(self, interval: str) -> None:
        """Set the operating timeframe so period parameters auto-scale."""
        self._timeframe = interval
        self._tf_multiplier = self._TF_MULTIPLIERS.get(interval, 1.0)
        logger.info(
            "strategy_timeframe_set",
            strategy=self.name,
            timeframe=interval,
            multiplier=self._tf_multiplier,
        )

    def _scale_period(self, base_period: int) -> int:
        """Scale a lookback period based on the current timeframe multiplier."""
        return max(base_period, int(base_period * self._tf_multiplier))

    @property
    def timeframe(self) -> str:
        return self._timeframe

    # ─── Abstract interface ───────────────────────────────────

    @abc.abstractmethod
    async def on_tick(self, ticks: list[Tick]) -> list[Signal]:
        pass

    @abc.abstractmethod
    async def on_bar(self, instrument_token: int, bar: pd.Series) -> list[Signal]:
        pass

    @abc.abstractmethod
    def generate_signal(self, data: pd.DataFrame, instrument_token: int) -> Optional[Signal]:
        pass

    # ─── Bar / tick helpers ───────────────────────────────────

    def update_bar_data(self, instrument_token: int, df: pd.DataFrame) -> None:
        self._bar_data[instrument_token] = df

    def get_bar_data(self, instrument_token: int) -> Optional[pd.DataFrame]:
        return self._bar_data.get(instrument_token)

    def add_tick_to_buffer(self, tick: Tick) -> None:
        token = tick.instrument_token
        if token not in self._tick_buffer:
            self._tick_buffer[token] = []
        self._tick_buffer[token].append(tick)

    def clear_tick_buffer(self, instrument_token: int) -> None:
        self._tick_buffer.pop(instrument_token, None)

    def activate(self) -> None:
        self.is_active = True
        logger.info("strategy_activated", strategy=self.name)

    def deactivate(self) -> None:
        self.is_active = False
        logger.info("strategy_deactivated", strategy=self.name)
