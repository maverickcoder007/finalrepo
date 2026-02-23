from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from src.analysis.indicators import atr
from src.data.models import (
    Exchange,
    OrderType,
    ProductType,
    Signal,
    Tick,
    TransactionType,
)
from src.strategy.base import BaseStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RSIStrategy(BaseStrategy):
    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        defaults = {
            "rsi_period": 14,
            "overbought": 65.0,      # Relaxed from 70 for more signals
            "oversold": 35.0,        # Relaxed from 30 for more signals
            "quantity": 1,
            "exchange": "NSE",
            "product": "MIS",
            "use_atr_sl": True,
            "atr_sl_multiple": 1.5,
            "tradingsymbol_map": {},
        }
        merged = {**defaults, **(params or {})}
        super().__init__("rsi_strategy", merged)
        self._prev_signal: dict[int, str] = {}

    @staticmethod
    def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    async def on_tick(self, ticks: list[Tick]) -> list[Signal]:
        signals: list[Signal] = []
        for tick in ticks:
            self.add_tick_to_buffer(tick)
            df = self.get_bar_data(tick.instrument_token)
            if df is not None and len(df) >= self.params["rsi_period"] + 1:
                signal = self.generate_signal(df, tick.instrument_token)
                if signal:
                    signals.append(signal)
        return signals

    async def on_bar(self, instrument_token: int, bar: pd.Series) -> list[Signal]:
        df = self.get_bar_data(instrument_token)
        if df is None:
            return []
        signal = self.generate_signal(df, instrument_token)
        return [signal] if signal else []

    def generate_signal(self, data: pd.DataFrame, instrument_token: int) -> Optional[Signal]:
        period = self.params["rsi_period"]
        if len(data) < period + 1:
            return None

        rsi = self.compute_rsi(data["close"], period)
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]

        if pd.isna(current_rsi) or pd.isna(prev_rsi):
            return None

        # Calculate stop loss if using ATR
        stop_loss = None
        if self.params["use_atr_sl"] and len(data) >= 15:
            atr_val = atr(data, 14).iloc[-1]
            stop_loss = atr_val * self.params["atr_sl_multiple"]

        tradingsymbol = self.params.get("tradingsymbol_map", {}).get(
            instrument_token, f"TOKEN_{instrument_token}"
        )

        oversold = self.params["oversold"]
        overbought = self.params["overbought"]
        prev = self._prev_signal.get(instrument_token)

        if prev_rsi <= oversold and current_rsi > oversold and prev != "BUY":
            self._prev_signal[instrument_token] = "BUY"
            return Signal(
                tradingsymbol=tradingsymbol,
                exchange=Exchange(self.params["exchange"]),
                transaction_type=TransactionType.BUY,
                quantity=self.params["quantity"],
                order_type=OrderType.MARKET,
                stop_loss=stop_loss,
                product=ProductType(self.params["product"]),
                strategy_name=self.name,
                confidence=max(0, (50 - current_rsi) / 50 * 100),
                metadata={"rsi": current_rsi, "prev_rsi": prev_rsi, "atr_stop_loss": stop_loss},
            )

        elif prev_rsi >= overbought and current_rsi < overbought and prev != "SELL":
            self._prev_signal[instrument_token] = "SELL"
            return Signal(
                tradingsymbol=tradingsymbol,
                exchange=Exchange(self.params["exchange"]),
                transaction_type=TransactionType.SELL,
                quantity=self.params["quantity"],
                order_type=OrderType.MARKET,
                stop_loss=stop_loss,
                product=ProductType(self.params["product"]),
                strategy_name=self.name,
                confidence=max(0, (current_rsi - 50) / 50 * 100),
                metadata={"rsi": current_rsi, "prev_rsi": prev_rsi, "atr_stop_loss": stop_loss},
            )

        return None
