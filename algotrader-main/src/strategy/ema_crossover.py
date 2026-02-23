from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from src.analysis.indicators import adx, atr
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


class EMACrossoverStrategy(BaseStrategy):
    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        defaults = {
            "fast_period": 9,
            "slow_period": 21,
            "quantity": 1,
            "exchange": "NSE",
            "product": "MIS",
            "tradingsymbol_map": {},
            # Filters (relaxed defaults for broad applicability)
            "min_adx": 0.0,            # ADX trend filter (0=disabled)
            "min_volume_ratio": 0.0,   # Volume filter (0=disabled)
            "use_atr_sl": True,        # Use ATR for stop loss
            "atr_sl_multiple": 1.5,    # Stop loss = Entry - 1.5*ATR
        }
        merged = {**defaults, **(params or {})}
        super().__init__("ema_crossover", merged)
        self._prev_signal: dict[int, str] = {}

    async def on_tick(self, ticks: list[Tick]) -> list[Signal]:
        signals: list[Signal] = []
        for tick in ticks:
            self.add_tick_to_buffer(tick)
            df = self.get_bar_data(tick.instrument_token)
            if df is not None and len(df) >= self.params["slow_period"]:
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
        if len(data) < self.params["slow_period"]:
            return None

        fast_ema = data["close"].ewm(span=self.params["fast_period"], adjust=False).mean()
        slow_ema = data["close"].ewm(span=self.params["slow_period"], adjust=False).mean()

        current_fast = fast_ema.iloc[-1]
        current_slow = slow_ema.iloc[-1]
        prev_fast = fast_ema.iloc[-2]
        prev_slow = slow_ema.iloc[-2]

        # Detect crossover first (cheap), then apply filters only on signal bars
        bullish_cross = prev_fast <= prev_slow and current_fast > current_slow
        bearish_cross = prev_fast >= prev_slow and current_fast < current_slow

        if not bullish_cross and not bearish_cross:
            return None

        # Filter 1: Check ADX for trend strength (only if enabled)
        if self.params["min_adx"] > 0 and len(data) >= 28:
            adx_val = adx(data, 14).iloc[-1]
            if adx_val < self.params["min_adx"]:
                logger.debug("ema_signal_rejected", reason="adx_too_low", adx=adx_val, threshold=self.params["min_adx"])
                return None

        # Filter 2: Check volume (only if enabled)
        if self.params["min_volume_ratio"] > 0 and "volume" in data.columns and len(data) >= 20:
            avg_volume = data["volume"].rolling(20).mean().iloc[-1]
            current_volume = data["volume"].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            if volume_ratio < self.params["min_volume_ratio"]:
                logger.debug("ema_signal_rejected", reason="volume_too_low", ratio=volume_ratio, threshold=self.params["min_volume_ratio"])
                return None

        tradingsymbol = self.params.get("tradingsymbol_map", {}).get(
            instrument_token, f"TOKEN_{instrument_token}"
        )

        # Calculate stop loss if using ATR
        stop_loss = None
        if self.params["use_atr_sl"] and len(data) >= 15:
            atr_val = atr(data, 14).iloc[-1]
            stop_loss = atr_val * self.params["atr_sl_multiple"]

        if bullish_cross:
            if self._prev_signal.get(instrument_token) != "BUY":
                self._prev_signal[instrument_token] = "BUY"
                signal = Signal(
                    tradingsymbol=tradingsymbol,
                    exchange=Exchange(self.params["exchange"]),
                    transaction_type=TransactionType.BUY,
                    quantity=self.params["quantity"],
                    order_type=OrderType.MARKET,
                    stop_loss=stop_loss,
                    product=ProductType(self.params["product"]),
                    strategy_name=self.name,
                    confidence=abs(current_fast - current_slow) / current_slow * 100,
                    metadata={
                        "fast_ema": current_fast,
                        "slow_ema": current_slow,
                        "signal_type": "entry",
                        "atr_stop_loss": stop_loss,
                    },
                )
                return signal

        elif bearish_cross:
            if self._prev_signal.get(instrument_token) != "SELL":
                self._prev_signal[instrument_token] = "SELL"
                signal = Signal(
                    tradingsymbol=tradingsymbol,
                    exchange=Exchange(self.params["exchange"]),
                    transaction_type=TransactionType.SELL,
                    quantity=self.params["quantity"],
                    order_type=OrderType.MARKET,
                    stop_loss=stop_loss,
                    product=ProductType(self.params["product"]),
                    strategy_name=self.name,
                    confidence=abs(current_fast - current_slow) / current_slow * 100,
                    metadata={
                        "fast_ema": current_fast,
                        "slow_ema": current_slow,
                        "signal_type": "entry",
                        "atr_stop_loss": stop_loss,
                    },
                )
                return signal

        return None
