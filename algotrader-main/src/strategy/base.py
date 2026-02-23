from __future__ import annotations

import abc
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from src.data.models import Signal, Tick
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseStrategy(abc.ABC):
    def __init__(self, name: str, params: Optional[dict[str, Any]] = None) -> None:
        self.name = name
        self.params = params or {}
        self.is_active = True
        self._tick_buffer: dict[int, list[Tick]] = {}
        self._bar_data: dict[int, pd.DataFrame] = {}
        self._signals: list[Signal] = []
        self._last_signal_time: Optional[datetime] = None

    @abc.abstractmethod
    async def on_tick(self, ticks: list[Tick]) -> list[Signal]:
        pass

    @abc.abstractmethod
    async def on_bar(self, instrument_token: int, bar: pd.Series) -> list[Signal]:
        pass

    @abc.abstractmethod
    def generate_signal(self, data: pd.DataFrame, instrument_token: int) -> Optional[Signal]:
        pass

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
