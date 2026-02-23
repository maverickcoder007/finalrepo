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


class MeanReversionStrategy(BaseStrategy):
    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        defaults = {
            "lookback_period": 20,
            "z_score_entry": 1.5,    # Relaxed from 2.0 for more signals
            "z_score_exit": 0.5,
            "quantity": 1,
            "exchange": "NSE",
            "product": "MIS",
            "use_atr_sl": True,
            "atr_sl_multiple": 1.5,
            "tradingsymbol_map": {},
        }
        merged = {**defaults, **(params or {})}
        super().__init__("mean_reversion", merged)
        self._prev_signal: dict[int, str] = {}

    async def on_tick(self, ticks: list[Tick]) -> list[Signal]:
        signals: list[Signal] = []
        for tick in ticks:
            self.add_tick_to_buffer(tick)
            df = self.get_bar_data(tick.instrument_token)
            if df is not None and len(df) >= self.params["lookback_period"]:
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
        lookback = self.params["lookback_period"]
        if len(data) < lookback:
            return None

        close = data["close"].iloc[-lookback:]
        mean = close.mean()
        std = close.std()

        if std == 0:
            return None

        current_price = close.iloc[-1]
        z_score = (current_price - mean) / std

        tradingsymbol = self.params.get("tradingsymbol_map", {}).get(
            instrument_token, f"TOKEN_{instrument_token}"
        )

        # Calculate stop loss if using ATR
        stop_loss = None
        if self.params["use_atr_sl"] and len(data) >= 15:
            atr_val = atr(data, 14).iloc[-1]
            stop_loss = atr_val * self.params["atr_sl_multiple"]

        z_entry = self.params["z_score_entry"]
        z_exit = self.params["z_score_exit"]
        prev = self._prev_signal.get(instrument_token)

        if z_score < -z_entry and prev != "BUY":
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
                confidence=min(abs(z_score) / z_entry * 50, 100),
                metadata={"z_score": z_score, "mean": mean, "std": std, "atr_stop_loss": stop_loss},
            )

        elif z_score > z_entry and prev != "SELL":
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
                confidence=min(abs(z_score) / z_entry * 50, 100),
                metadata={"z_score": z_score, "mean": mean, "std": std, "atr_stop_loss": stop_loss},
            )

        elif abs(z_score) < z_exit and prev is not None:
            del self._prev_signal[instrument_token]
            exit_type = TransactionType.SELL if prev == "BUY" else TransactionType.BUY
            return Signal(
                tradingsymbol=tradingsymbol,
                exchange=Exchange(self.params["exchange"]),
                transaction_type=exit_type,
                quantity=self.params["quantity"],
                order_type=OrderType.MARKET,
                product=ProductType(self.params["product"]),
                strategy_name=self.name,
                confidence=30.0,
                metadata={"z_score": z_score, "mean": mean, "action": "exit"},
            )

        return None
