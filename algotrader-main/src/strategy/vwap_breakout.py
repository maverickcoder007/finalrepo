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


class VWAPBreakoutStrategy(BaseStrategy):
    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        defaults = {
            "breakout_threshold": 0.3,
            "quantity": 1,
            "exchange": "NSE",
            "product": "MIS",
            "min_volume_ratio": 0.0,     # Volume filter (0=disabled)
            "use_atr_sl": True,
            "atr_sl_multiple": 1.5,
            "tradingsymbol_map": {},
            "vwap_period": 20,           # Rolling VWAP lookback
        }
        merged = {**defaults, **(params or {})}
        super().__init__("vwap_breakout", merged)
        self._prev_signal: dict[int, str] = {}

    async def on_tick(self, ticks: list[Tick]) -> list[Signal]:
        signals: list[Signal] = []
        for tick in ticks:
            self.add_tick_to_buffer(tick)
            df = self.get_bar_data(tick.instrument_token)
            if df is not None and len(df) >= 5:
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
        vwap_period = self._scale_period(self.params.get("vwap_period", 20))
        min_bars = max(5, vwap_period)
        if len(data) < min_bars:
            return None

        df = data.copy()

        # Use rolling VWAP (not cumulative from bar 0 which stabilises and
        # produces monotonically growing distance on trending data).
        period = min(vwap_period, len(df))
        recent = df.iloc[-period:]
        cum_vol = recent["volume"].sum()
        if cum_vol == 0:
            return None
        rolling_vwap = (recent["close"] * recent["volume"]).sum() / cum_vol

        current_price = df["close"].iloc[-1]
        avg_volume = df["volume"].rolling(min(20, len(df))).mean().iloc[-1]
        current_volume = df["volume"].iloc[-1]

        # Calculate stop loss if using ATR
        stop_loss = None
        if self.params["use_atr_sl"] and len(data) >= 15:
            atr_val = atr(data, 14).iloc[-1]
            stop_loss = atr_val * self.params["atr_sl_multiple"]

        threshold_pct = self.params["breakout_threshold"]
        distance_pct = ((current_price - rolling_vwap) / rolling_vwap) * 100
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

        tradingsymbol = self.params.get("tradingsymbol_map", {}).get(
            instrument_token, f"TOKEN_{instrument_token}"
        )

        # Volume filter (only if enabled)
        min_vol = self.params["min_volume_ratio"]
        if min_vol > 0 and volume_ratio < min_vol:
            return None

        if (
            distance_pct > threshold_pct
            and self._prev_signal.get(instrument_token) != "BUY"
        ):
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
                confidence=min(abs(distance_pct) * max(volume_ratio, 1), 100),
                metadata={"vwap": rolling_vwap, "distance_pct": distance_pct, "volume_ratio": volume_ratio, "atr_stop_loss": stop_loss},
            )

        elif (
            distance_pct < -threshold_pct
            and self._prev_signal.get(instrument_token) != "SELL"
        ):
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
                confidence=min(abs(distance_pct) * max(volume_ratio, 1), 100),
                metadata={"vwap": rolling_vwap, "distance_pct": distance_pct, "volume_ratio": volume_ratio, "atr_stop_loss": stop_loss},
            )

        return None
