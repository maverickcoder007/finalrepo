"""
Custom Strategy Builder — Create strategies from indicator conditions.

Allows users to compose trading strategies from available technical indicators
without writing code. Conditions are defined as JSON rules that get evaluated
against bar data at runtime.

Example custom strategy config:
{
    "name": "my_golden_cross",
    "description": "Buy when EMA9 > EMA21 and RSI < 70",
    "entry_rules": [
        {"indicator": "ema_crossover", "params": {"fast": 9, "slow": 21}, "condition": "cross_above"},
        {"indicator": "rsi", "params": {"period": 14}, "condition": "below", "value": 70}
    ],
    "exit_rules": [
        {"indicator": "ema_crossover", "params": {"fast": 9, "slow": 21}, "condition": "cross_below"},
        {"indicator": "rsi", "params": {"period": 14}, "condition": "above", "value": 80}
    ],
    "entry_logic": "all",    # "all" = AND, "any" = OR
    "exit_logic": "any",
    "quantity": 10,
    "exchange": "NSE",
    "product": "MIS",
    "stop_loss_pct": 2.0,
    "target_pct": 4.0,
    "use_atr_sl": true,
    "atr_sl_multiple": 1.5
}
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from src.analysis.indicators import (
    adx,
    atr,
    bollinger_bands,
    ema,
    macd,
    rsi,
    sma,
    volume_ratio,
)
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

CUSTOM_STRATEGIES_FILE = "data/custom_strategies.json"

# ────────────────────────────────────────────────────────────────
# Available indicator definitions for the builder UI
# ────────────────────────────────────────────────────────────────

AVAILABLE_INDICATORS = [
    {
        "id": "ema_crossover",
        "name": "EMA Crossover",
        "description": "Fast EMA crosses slow EMA",
        "params": [
            {"name": "fast", "type": "int", "default": 9, "min": 2, "max": 200, "label": "Fast Period"},
            {"name": "slow", "type": "int", "default": 21, "min": 5, "max": 200, "label": "Slow Period"},
        ],
        "conditions": ["cross_above", "cross_below", "above", "below"],
    },
    {
        "id": "sma_crossover",
        "name": "SMA Crossover",
        "description": "Fast SMA crosses slow SMA",
        "params": [
            {"name": "fast", "type": "int", "default": 20, "min": 2, "max": 200, "label": "Fast Period"},
            {"name": "slow", "type": "int", "default": 50, "min": 5, "max": 200, "label": "Slow Period"},
        ],
        "conditions": ["cross_above", "cross_below", "above", "below"],
    },
    {
        "id": "rsi",
        "name": "RSI",
        "description": "Relative Strength Index value check",
        "params": [
            {"name": "period", "type": "int", "default": 14, "min": 2, "max": 50, "label": "Period"},
        ],
        "conditions": ["above", "below", "cross_above", "cross_below"],
        "value_required": True,
        "value_default": 30,
        "value_label": "RSI Level",
        "value_min": 0,
        "value_max": 100,
    },
    {
        "id": "macd",
        "name": "MACD",
        "description": "MACD line vs signal line",
        "params": [
            {"name": "fast", "type": "int", "default": 12, "min": 2, "max": 50, "label": "Fast"},
            {"name": "slow", "type": "int", "default": 26, "min": 5, "max": 100, "label": "Slow"},
            {"name": "signal", "type": "int", "default": 9, "min": 2, "max": 50, "label": "Signal"},
        ],
        "conditions": ["cross_above", "cross_below", "histogram_positive", "histogram_negative"],
    },
    {
        "id": "adx",
        "name": "ADX",
        "description": "Average Directional Index — trend strength",
        "params": [
            {"name": "period", "type": "int", "default": 14, "min": 5, "max": 50, "label": "Period"},
        ],
        "conditions": ["above", "below"],
        "value_required": True,
        "value_default": 25,
        "value_label": "ADX Threshold",
        "value_min": 0,
        "value_max": 100,
    },
    {
        "id": "bollinger",
        "name": "Bollinger Bands",
        "description": "Price relative to Bollinger Bands",
        "params": [
            {"name": "period", "type": "int", "default": 20, "min": 5, "max": 100, "label": "Period"},
            {"name": "std_dev", "type": "float", "default": 2.0, "min": 0.5, "max": 4.0, "label": "Std Dev"},
        ],
        "conditions": ["above_upper", "below_lower", "cross_above_middle", "cross_below_middle"],
    },
    {
        "id": "volume_spike",
        "name": "Volume Spike",
        "description": "Current volume vs average",
        "params": [
            {"name": "period", "type": "int", "default": 20, "min": 5, "max": 100, "label": "Avg Period"},
        ],
        "conditions": ["above", "below"],
        "value_required": True,
        "value_default": 1.5,
        "value_label": "Volume Ratio",
        "value_min": 0.1,
        "value_max": 10.0,
    },
    {
        "id": "price_above_sma",
        "name": "Price vs SMA",
        "description": "Price above or below a moving average",
        "params": [
            {"name": "period", "type": "int", "default": 200, "min": 5, "max": 500, "label": "SMA Period"},
        ],
        "conditions": ["above", "below", "cross_above", "cross_below"],
    },
    {
        "id": "atr_breakout",
        "name": "ATR Breakout",
        "description": "Price moves more than N×ATR from previous close",
        "params": [
            {"name": "period", "type": "int", "default": 14, "min": 5, "max": 50, "label": "ATR Period"},
        ],
        "conditions": ["above", "below"],
        "value_required": True,
        "value_default": 1.5,
        "value_label": "ATR Multiple",
        "value_min": 0.5,
        "value_max": 5.0,
    },
    {
        "id": "price_change",
        "name": "Price Change %",
        "description": "Percentage change over N bars",
        "params": [
            {"name": "lookback", "type": "int", "default": 1, "min": 1, "max": 50, "label": "Lookback Bars"},
        ],
        "conditions": ["above", "below"],
        "value_required": True,
        "value_default": 1.0,
        "value_label": "Change %",
        "value_min": -20.0,
        "value_max": 20.0,
    },
]


# ────────────────────────────────────────────────────────────────
# Rule Evaluator — evaluates a single condition against bar data
# ────────────────────────────────────────────────────────────────

class RuleEvaluator:
    """Evaluate a single indicator rule against OHLCV data."""

    @staticmethod
    def evaluate(rule: dict[str, Any], data: pd.DataFrame) -> bool:
        """Evaluate one rule. Returns True if condition is met."""
        indicator = rule.get("indicator", "")
        condition = rule.get("condition", "")
        params = rule.get("params", {})
        threshold = rule.get("value", 0)

        try:
            if indicator == "ema_crossover":
                return RuleEvaluator._eval_ma_crossover(data, params, condition, ma_func=ema)
            elif indicator == "sma_crossover":
                return RuleEvaluator._eval_ma_crossover(data, params, condition, ma_func=sma)
            elif indicator == "rsi":
                return RuleEvaluator._eval_rsi(data, params, condition, threshold)
            elif indicator == "macd":
                return RuleEvaluator._eval_macd(data, params, condition)
            elif indicator == "adx":
                return RuleEvaluator._eval_adx(data, params, condition, threshold)
            elif indicator == "bollinger":
                return RuleEvaluator._eval_bollinger(data, params, condition)
            elif indicator == "volume_spike":
                return RuleEvaluator._eval_volume(data, params, condition, threshold)
            elif indicator == "price_above_sma":
                return RuleEvaluator._eval_price_vs_sma(data, params, condition)
            elif indicator == "atr_breakout":
                return RuleEvaluator._eval_atr_breakout(data, params, condition, threshold)
            elif indicator == "price_change":
                return RuleEvaluator._eval_price_change(data, params, condition, threshold)
            else:
                logger.warning("unknown_indicator", indicator=indicator)
                return False
        except Exception as e:
            logger.debug("rule_eval_error", indicator=indicator, error=str(e))
            return False

    @staticmethod
    def _eval_ma_crossover(data: pd.DataFrame, params: dict, condition: str, ma_func) -> bool:
        fast_p = params.get("fast", 9)
        slow_p = params.get("slow", 21)
        if len(data) < slow_p + 1:
            return False
        fast = ma_func(data["close"], fast_p)
        slow = ma_func(data["close"], slow_p)
        curr_fast, curr_slow = fast.iloc[-1], slow.iloc[-1]
        prev_fast, prev_slow = fast.iloc[-2], slow.iloc[-2]

        if condition == "cross_above":
            return prev_fast <= prev_slow and curr_fast > curr_slow
        elif condition == "cross_below":
            return prev_fast >= prev_slow and curr_fast < curr_slow
        elif condition == "above":
            return curr_fast > curr_slow
        elif condition == "below":
            return curr_fast < curr_slow
        return False

    @staticmethod
    def _eval_rsi(data: pd.DataFrame, params: dict, condition: str, threshold: float) -> bool:
        period = params.get("period", 14)
        if len(data) < period + 2:
            return False
        rsi_series = rsi(data["close"], period)
        curr = rsi_series.iloc[-1]
        prev = rsi_series.iloc[-2]
        if pd.isna(curr) or pd.isna(prev):
            return False

        if condition == "above":
            return curr > threshold
        elif condition == "below":
            return curr < threshold
        elif condition == "cross_above":
            return prev <= threshold and curr > threshold
        elif condition == "cross_below":
            return prev >= threshold and curr < threshold
        return False

    @staticmethod
    def _eval_macd(data: pd.DataFrame, params: dict, condition: str) -> bool:
        fast_p = params.get("fast", 12)
        slow_p = params.get("slow", 26)
        signal_p = params.get("signal", 9)
        if len(data) < slow_p + signal_p:
            return False
        m = macd(data["close"], fast_p, slow_p, signal_p)
        macd_line = m["macd"]
        signal_line = m["signal"]
        hist = m["histogram"]

        if condition == "cross_above":
            return macd_line.iloc[-2] <= signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]
        elif condition == "cross_below":
            return macd_line.iloc[-2] >= signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]
        elif condition == "histogram_positive":
            return hist.iloc[-1] > 0
        elif condition == "histogram_negative":
            return hist.iloc[-1] < 0
        return False

    @staticmethod
    def _eval_adx(data: pd.DataFrame, params: dict, condition: str, threshold: float) -> bool:
        period = params.get("period", 14)
        if len(data) < period * 2:
            return False
        adx_series = adx(data, period)
        curr = adx_series.iloc[-1]
        if pd.isna(curr):
            return False
        if condition == "above":
            return curr > threshold
        elif condition == "below":
            return curr < threshold
        return False

    @staticmethod
    def _eval_bollinger(data: pd.DataFrame, params: dict, condition: str) -> bool:
        period = params.get("period", 20)
        std_dev = params.get("std_dev", 2.0)
        if len(data) < period + 1:
            return False
        bb = bollinger_bands(data["close"], period, std_dev)
        price = data["close"].iloc[-1]
        prev_price = data["close"].iloc[-2]
        mid = bb["middle"].iloc[-1]
        prev_mid = bb["middle"].iloc[-2]

        if condition == "above_upper":
            return price > bb["upper"].iloc[-1]
        elif condition == "below_lower":
            return price < bb["lower"].iloc[-1]
        elif condition == "cross_above_middle":
            return prev_price <= prev_mid and price > mid
        elif condition == "cross_below_middle":
            return prev_price >= prev_mid and price < mid
        return False

    @staticmethod
    def _eval_volume(data: pd.DataFrame, params: dict, condition: str, threshold: float) -> bool:
        period = params.get("period", 20)
        if "volume" not in data.columns or len(data) < period:
            return False
        vr = volume_ratio(data["volume"], period)
        curr = vr.iloc[-1]
        if pd.isna(curr):
            return False
        if condition == "above":
            return curr > threshold
        elif condition == "below":
            return curr < threshold
        return False

    @staticmethod
    def _eval_price_vs_sma(data: pd.DataFrame, params: dict, condition: str) -> bool:
        period = params.get("period", 200)
        if len(data) < period + 1:
            return False
        sma_series = sma(data["close"], period)
        price = data["close"].iloc[-1]
        prev_price = data["close"].iloc[-2]
        curr_sma = sma_series.iloc[-1]
        prev_sma = sma_series.iloc[-2]

        if condition == "above":
            return price > curr_sma
        elif condition == "below":
            return price < curr_sma
        elif condition == "cross_above":
            return prev_price <= prev_sma and price > curr_sma
        elif condition == "cross_below":
            return prev_price >= prev_sma and price < curr_sma
        return False

    @staticmethod
    def _eval_atr_breakout(data: pd.DataFrame, params: dict, condition: str, threshold: float) -> bool:
        period = params.get("period", 14)
        if len(data) < period + 2:
            return False
        atr_series = atr(data, period)
        atr_val = atr_series.iloc[-1]
        if pd.isna(atr_val) or atr_val == 0:
            return False
        price_change = data["close"].iloc[-1] - data["close"].iloc[-2]
        normalized = abs(price_change) / atr_val

        if condition == "above":
            return normalized > threshold
        elif condition == "below":
            return normalized < threshold
        return False

    @staticmethod
    def _eval_price_change(data: pd.DataFrame, params: dict, condition: str, threshold: float) -> bool:
        lookback = params.get("lookback", 1)
        if len(data) < lookback + 1:
            return False
        pct = ((data["close"].iloc[-1] / data["close"].iloc[-1 - lookback]) - 1) * 100

        if condition == "above":
            return pct > threshold
        elif condition == "below":
            return pct < threshold
        return False


# ────────────────────────────────────────────────────────────────
# Custom Strategy — a BaseStrategy driven by JSON rules
# ────────────────────────────────────────────────────────────────

class CustomStrategy(BaseStrategy):
    """Strategy built from user-defined indicator conditions."""

    def __init__(self, config: dict[str, Any]) -> None:
        name = config.get("name", "custom_strategy")
        params = {
            "quantity": config.get("quantity", 1),
            "exchange": config.get("exchange", "NSE"),
            "product": config.get("product", "MIS"),
            "stop_loss_pct": config.get("stop_loss_pct", 0),
            "target_pct": config.get("target_pct", 0),
            "use_atr_sl": config.get("use_atr_sl", False),
            "atr_sl_multiple": config.get("atr_sl_multiple", 1.5),
            "tradingsymbol_map": config.get("tradingsymbol_map", {}),
        }
        super().__init__(name, params)
        self._config = config
        self._entry_rules: list[dict] = config.get("entry_rules", [])
        self._exit_rules: list[dict] = config.get("exit_rules", [])
        self._entry_logic: str = config.get("entry_logic", "all")  # "all" or "any"
        self._exit_logic: str = config.get("exit_logic", "any")
        self._prev_signal: dict[int, str] = {}
        self._evaluator = RuleEvaluator()

    @property
    def config(self) -> dict[str, Any]:
        return self._config

    async def on_tick(self, ticks: list[Tick]) -> list[Signal]:
        signals: list[Signal] = []
        for tick in ticks:
            self.add_tick_to_buffer(tick)
            df = self.get_bar_data(tick.instrument_token)
            if df is not None and len(df) >= 30:
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
        if len(data) < 30:
            return None

        tradingsymbol = self.params.get("tradingsymbol_map", {}).get(
            instrument_token, f"TOKEN_{instrument_token}"
        )

        # Check EXIT rules first (if we have an open position direction)
        prev_direction = self._prev_signal.get(instrument_token)
        if prev_direction and self._exit_rules:
            exit_results = [self._evaluator.evaluate(rule, data) for rule in self._exit_rules]
            should_exit = all(exit_results) if self._exit_logic == "all" else any(exit_results)

            if should_exit:
                exit_type = TransactionType.SELL if prev_direction == "BUY" else TransactionType.BUY
                del self._prev_signal[instrument_token]
                return self._make_signal(
                    tradingsymbol, exit_type, data, instrument_token,
                    metadata={"action": "exit", "rules_matched": sum(exit_results)},
                )

        # Check ENTRY rules
        entry_results = [self._evaluator.evaluate(rule, data) for rule in self._entry_rules]
        should_enter = all(entry_results) if self._entry_logic == "all" else any(entry_results)

        if not should_enter:
            return None

        # Determine direction from the dominant entry signal type
        direction = self._infer_direction()
        if self._prev_signal.get(instrument_token) == direction.value:
            return None  # Already in this direction

        self._prev_signal[instrument_token] = direction.value
        return self._make_signal(
            tradingsymbol, direction, data, instrument_token,
            metadata={"action": "entry", "rules_matched": sum(entry_results)},
        )

    def _infer_direction(self) -> TransactionType:
        """Infer BUY/SELL from the entry rule types."""
        buy_signals = {"cross_above", "above", "below_lower", "histogram_positive", "cross_above_middle"}
        sell_signals = {"cross_below", "below", "above_upper", "histogram_negative", "cross_below_middle"}

        buy_count = sum(1 for r in self._entry_rules if r.get("condition") in buy_signals)
        sell_count = sum(1 for r in self._entry_rules if r.get("condition") in sell_signals)
        return TransactionType.BUY if buy_count >= sell_count else TransactionType.SELL

    def _make_signal(
        self,
        tradingsymbol: str,
        transaction_type: TransactionType,
        data: pd.DataFrame,
        instrument_token: int,
        metadata: dict[str, Any] | None = None,
    ) -> Signal:
        stop_loss = None
        target = None

        if self.params.get("use_atr_sl") and len(data) >= 15:
            atr_val = atr(data, 14).iloc[-1]
            if not pd.isna(atr_val):
                stop_loss = atr_val * self.params.get("atr_sl_multiple", 1.5)
        elif self.params.get("stop_loss_pct", 0) > 0:
            price = data["close"].iloc[-1]
            stop_loss = price * self.params["stop_loss_pct"] / 100

        if self.params.get("target_pct", 0) > 0:
            price = data["close"].iloc[-1]
            target = price * self.params["target_pct"] / 100

        meta = metadata or {}
        meta["atr_stop_loss"] = stop_loss
        meta["target"] = target

        return Signal(
            tradingsymbol=tradingsymbol,
            exchange=Exchange(self.params.get("exchange", "NSE")),
            transaction_type=transaction_type,
            quantity=self.params.get("quantity", 1),
            order_type=OrderType.MARKET,
            stop_loss=stop_loss,
            target=target,
            product=ProductType(self.params.get("product", "MIS")),
            strategy_name=self.name,
            confidence=50.0,
            metadata=meta,
        )


# ────────────────────────────────────────────────────────────────
# Persistence — save/load custom strategies to/from JSON
# ────────────────────────────────────────────────────────────────

class CustomStrategyStore:
    """Persist custom strategy configurations to disk."""

    def __init__(self, filepath: str = CUSTOM_STRATEGIES_FILE) -> None:
        self._filepath = filepath
        self._strategies: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._filepath):
            try:
                with open(self._filepath, "r") as f:
                    self._strategies = json.load(f)
                logger.info("custom_strategies_loaded", count=len(self._strategies))
            except Exception as e:
                logger.error("custom_strategies_load_failed", error=str(e))

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._filepath) or ".", exist_ok=True)
        try:
            with open(self._filepath, "w") as f:
                json.dump(self._strategies, f, indent=2)
        except Exception as e:
            logger.error("custom_strategies_save_failed", error=str(e))

    def save_strategy(self, config: dict[str, Any]) -> dict[str, Any]:
        """Save or update a custom strategy config."""
        name = config.get("name", "")
        if not name:
            return {"error": "Strategy name is required"}
        if not config.get("entry_rules"):
            return {"error": "At least one entry rule is required"}

        config["created_at"] = config.get("created_at", datetime.now().isoformat())
        config["updated_at"] = datetime.now().isoformat()
        self._strategies[name] = config
        self._save()
        logger.info("custom_strategy_saved", name=name)
        return {"success": True, "name": name}

    def delete_strategy(self, name: str) -> dict[str, Any]:
        if name in self._strategies:
            del self._strategies[name]
            self._save()
            return {"success": True, "deleted": name}
        return {"error": f"Strategy '{name}' not found"}

    def get_strategy(self, name: str) -> Optional[dict[str, Any]]:
        return self._strategies.get(name)

    def list_strategies(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "description": cfg.get("description", ""),
                "entry_rules": len(cfg.get("entry_rules", [])),
                "exit_rules": len(cfg.get("exit_rules", [])),
                "created_at": cfg.get("created_at", ""),
                "updated_at": cfg.get("updated_at", ""),
            }
            for name, cfg in self._strategies.items()
        ]

    def build_strategy(self, name: str) -> Optional[CustomStrategy]:
        """Instantiate a CustomStrategy from a saved config."""
        config = self._strategies.get(name)
        if not config:
            return None
        return CustomStrategy(config)

    def get_all_configs(self) -> dict[str, dict[str, Any]]:
        return self._strategies.copy()


def get_available_indicators() -> list[dict[str, Any]]:
    """Return the indicator catalog for the strategy builder UI."""
    return AVAILABLE_INDICATORS
