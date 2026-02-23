"""
Individual Analysis-Based Strategies.

Each strategy focuses on ONE specific technical analysis trigger pattern
(as used in the full ScannerStrategy), making them individually selectable
in backtest, paper trade, and live trading dropdowns.

All share a common base that computes indicators and evaluates a single trigger.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from src.analysis.indicators import (
    atr,
    sma,
    ema,
    rsi as compute_rsi,
    macd as compute_macd,
    adx as compute_adx,
    stochastic,
    bollinger_bands,
    bollinger_bandwidth,
    mfi as compute_mfi,
    obv,
    ichimoku,
    detect_vcp,
    detect_stage,
    detect_pocket_pivot,
    volume_ratio as compute_volume_ratio,
    compute_all_indicators,
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


# ────────────────────────────────────────────────────────────
# Common base for single-trigger analysis strategies
# ────────────────────────────────────────────────────────────

class SingleTriggerStrategy(BaseStrategy):
    """Base class for strategies that use ONE specific technical trigger."""

    def __init__(self, name: str, params: Optional[dict[str, Any]] = None) -> None:
        defaults = {
            "quantity": 1,
            "exchange": "NSE",
            "product": "MIS",
            "tradingsymbol_map": {},
            "use_atr_sl": True,
            "atr_sl_multiple": 1.5,
            "min_bars": 50,
        }
        merged = {**defaults, **(params or {})}
        super().__init__(name, merged)
        self._prev_signal: dict[int, str] = {}
        self._benchmark_data: Optional[pd.DataFrame] = None

    def set_benchmark_data(self, benchmark_df: pd.DataFrame) -> None:
        self._benchmark_data = benchmark_df

    async def on_tick(self, ticks: list[Tick]) -> list[Signal]:
        signals: list[Signal] = []
        for tick in ticks:
            self.add_tick_to_buffer(tick)
            df = self.get_bar_data(tick.instrument_token)
            if df is not None and len(df) >= self.params.get("min_bars", 50):
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

    def _make_signal(
        self, instrument_token: int, ltp: float, direction: str,
        confidence: float, trigger_name: str, stop_loss: float = 0.0,
    ) -> Optional[Signal]:
        """Create a Signal object for BUY or SELL."""
        prev = self._prev_signal.get(instrument_token)
        if prev == direction:
            return None  # Don't repeat same signal
        self._prev_signal[instrument_token] = direction

        sym = self.params.get("tradingsymbol_map", {}).get(
            str(instrument_token), f"TOKEN_{instrument_token}"
        )
        return Signal(
            tradingsymbol=sym,
            exchange=Exchange(self.params.get("exchange", "NSE")),
            transaction_type=TransactionType.BUY if direction == "BUY" else TransactionType.SELL,
            quantity=self.params.get("quantity", 1),
            price=ltp,
            order_type=OrderType.MARKET,
            product=ProductType(self.params.get("product", "MIS")),
            strategy_name=self.name,
            confidence=confidence,
            metadata={
                "trigger": trigger_name,
                "stop_loss": round(stop_loss, 2) if stop_loss else 0,
            },
        )

    def _compute_atr_sl(self, data: pd.DataFrame, ltp: float, direction: str) -> float:
        if not self.params.get("use_atr_sl"):
            return 0.0
        atr_vals = atr(data["high"], data["low"], data["close"])
        if atr_vals is None or len(atr_vals) < 1:
            return 0.0
        last_atr = float(atr_vals.iloc[-1])
        mult = self.params.get("atr_sl_multiple", 1.5)
        if direction == "BUY":
            return ltp - last_atr * mult
        else:
            return ltp + last_atr * mult


# ────────────────────────────────────────────────────────────
# 1. MACD Crossover Strategy
# ────────────────────────────────────────────────────────────

class MACDCrossoverStrategy(SingleTriggerStrategy):
    """BUY on MACD histogram turning positive, SELL on negative."""

    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        super().__init__("macd_crossover", params)

    def generate_signal(self, data: pd.DataFrame, instrument_token: int = 0) -> Optional[Signal]:
        if len(data) < 50:
            return None
        close = data["close"]
        ltp = float(close.iloc[-1])

        macd_line, signal_line, histogram = compute_macd(close)
        if histogram is None or len(histogram) < 2:
            return None

        h_cur = float(histogram.iloc[-1])
        h_prev = float(histogram.iloc[-2])
        sl = self._compute_atr_sl(data, ltp, "BUY" if h_cur > 0 else "SELL")

        if h_prev <= 0 < h_cur:
            return self._make_signal(instrument_token, ltp, "BUY", 70.0, "macd_bullish_cross", sl)
        elif h_prev >= 0 > h_cur:
            return self._make_signal(instrument_token, ltp, "SELL", 70.0, "macd_bearish_cross", sl)
        return None


# ────────────────────────────────────────────────────────────
# 2. Golden / Death Cross Strategy
# ────────────────────────────────────────────────────────────

class GoldenDeathCrossStrategy(SingleTriggerStrategy):
    """BUY on Golden Cross (SMA50 > SMA200), SELL on Death Cross."""

    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        defaults = {"min_bars": 210}
        merged = {**defaults, **(params or {})}
        super().__init__("golden_death_cross", merged)

    def generate_signal(self, data: pd.DataFrame, instrument_token: int = 0) -> Optional[Signal]:
        if len(data) < 210:
            return None
        close = data["close"]
        ltp = float(close.iloc[-1])

        sma50 = sma(close, 50)
        sma200 = sma(close, 200)
        if sma50 is None or sma200 is None or len(sma50) < 2:
            return None

        s50_cur, s50_prev = float(sma50.iloc[-1]), float(sma50.iloc[-2])
        s200_cur, s200_prev = float(sma200.iloc[-1]), float(sma200.iloc[-2])
        sl = self._compute_atr_sl(data, ltp, "BUY")

        # Golden Cross
        if s50_prev <= s200_prev and s50_cur > s200_cur:
            return self._make_signal(instrument_token, ltp, "BUY", 80.0, "golden_cross", sl)
        # Death Cross
        if s50_prev >= s200_prev and s50_cur < s200_cur:
            sl = self._compute_atr_sl(data, ltp, "SELL")
            return self._make_signal(instrument_token, ltp, "SELL", 80.0, "death_cross", sl)
        return None


# ────────────────────────────────────────────────────────────
# 3. Volume Breakout Strategy (Analysis-based)
# ────────────────────────────────────────────────────────────

class VolumeBreakoutAnalysisStrategy(SingleTriggerStrategy):
    """BUY on high volume + price surge, SELL on high volume + price drop."""

    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        defaults = {"volume_ratio_threshold": 2.0, "price_change_pct": 2.0}
        merged = {**defaults, **(params or {})}
        super().__init__("volume_breakout_analysis", merged)

    def generate_signal(self, data: pd.DataFrame, instrument_token: int = 0) -> Optional[Signal]:
        if len(data) < 50:
            return None
        close = data["close"]
        ltp = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        pct_chg = ((ltp - prev_close) / prev_close) * 100

        vol_ratio = compute_volume_ratio(data["volume"])
        if vol_ratio is None:
            return None
        vr = float(vol_ratio.iloc[-1])
        threshold = self.params.get("volume_ratio_threshold", 2.0)
        pct_threshold = self.params.get("price_change_pct", 2.0)

        if vr >= threshold and pct_chg > pct_threshold:
            sl = self._compute_atr_sl(data, ltp, "BUY")
            return self._make_signal(instrument_token, ltp, "BUY", 75.0, "volume_breakout", sl)
        elif vr >= threshold and pct_chg < -pct_threshold:
            sl = self._compute_atr_sl(data, ltp, "SELL")
            return self._make_signal(instrument_token, ltp, "SELL", 75.0, "volume_breakdown", sl)
        return None


# ────────────────────────────────────────────────────────────
# 4. Bollinger Bands Strategy
# ────────────────────────────────────────────────────────────

class BollingerBandsStrategy(SingleTriggerStrategy):
    """BUY below lower band / on squeeze, SELL above upper band."""

    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        defaults = {"squeeze_threshold": 5.0}
        merged = {**defaults, **(params or {})}
        super().__init__("bollinger_bands", merged)

    def generate_signal(self, data: pd.DataFrame, instrument_token: int = 0) -> Optional[Signal]:
        if len(data) < 50:
            return None
        close = data["close"]
        ltp = float(close.iloc[-1])

        upper, middle, lower = bollinger_bands(close)
        bw = bollinger_bandwidth(close)
        if upper is None or lower is None:
            return None

        u = float(upper.iloc[-1])
        l_ = float(lower.iloc[-1])
        squeeze = self.params.get("squeeze_threshold", 5.0)

        # Below lower band → BUY
        if ltp <= l_:
            sl = self._compute_atr_sl(data, ltp, "BUY")
            return self._make_signal(instrument_token, ltp, "BUY", 65.0, "bollinger_lower_touch", sl)
        # Above upper band → SELL
        elif ltp >= u:
            sl = self._compute_atr_sl(data, ltp, "SELL")
            return self._make_signal(instrument_token, ltp, "SELL", 65.0, "bollinger_upper_touch", sl)
        # Squeeze (low bandwidth)
        elif bw is not None and float(bw.iloc[-1]) < squeeze:
            sl = self._compute_atr_sl(data, ltp, "BUY")
            return self._make_signal(instrument_token, ltp, "BUY", 60.0, "bollinger_squeeze", sl)
        return None


# ────────────────────────────────────────────────────────────
# 5. Stochastic Oscillator Strategy
# ────────────────────────────────────────────────────────────

class StochasticStrategy(SingleTriggerStrategy):
    """BUY on stochastic oversold, SELL on overbought."""

    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        defaults = {"stoch_oversold": 20, "stoch_overbought": 80}
        merged = {**defaults, **(params or {})}
        super().__init__("stochastic", merged)

    def generate_signal(self, data: pd.DataFrame, instrument_token: int = 0) -> Optional[Signal]:
        if len(data) < 50:
            return None
        close = data["close"]
        ltp = float(close.iloc[-1])

        k, d = stochastic(data["high"], data["low"], close)
        if k is None or d is None or len(k) < 1:
            return None

        k_val = float(k.iloc[-1])
        d_val = float(d.iloc[-1])
        oversold = self.params.get("stoch_oversold", 20)
        overbought = self.params.get("stoch_overbought", 80)

        if k_val < oversold and d_val < oversold:
            sl = self._compute_atr_sl(data, ltp, "BUY")
            return self._make_signal(instrument_token, ltp, "BUY", 65.0, "stoch_oversold", sl)
        elif k_val > overbought and d_val > overbought:
            sl = self._compute_atr_sl(data, ltp, "SELL")
            return self._make_signal(instrument_token, ltp, "SELL", 65.0, "stoch_overbought", sl)
        return None


# ────────────────────────────────────────────────────────────
# 6. Ichimoku Cloud Strategy
# ────────────────────────────────────────────────────────────

class IchimokuStrategy(SingleTriggerStrategy):
    """BUY when price above cloud, SELL when below cloud."""

    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        defaults = {"min_bars": 60}
        merged = {**defaults, **(params or {})}
        super().__init__("ichimoku_cloud", merged)

    def generate_signal(self, data: pd.DataFrame, instrument_token: int = 0) -> Optional[Signal]:
        if len(data) < 60:
            return None
        close = data["close"]
        ltp = float(close.iloc[-1])

        try:
            ichi = ichimoku(data["high"], data["low"], close)
            if ichi is None:
                return None
            span_a = float(ichi["senkou_a"].iloc[-1]) if "senkou_a" in ichi else None
            span_b = float(ichi["senkou_b"].iloc[-1]) if "senkou_b" in ichi else None
            if span_a is None or span_b is None or np.isnan(span_a) or np.isnan(span_b):
                return None

            cloud_top = max(span_a, span_b)
            cloud_bottom = min(span_a, span_b)

            if ltp > cloud_top:
                sl = self._compute_atr_sl(data, ltp, "BUY")
                return self._make_signal(instrument_token, ltp, "BUY", 70.0, "ichimoku_above_cloud", sl)
            elif ltp < cloud_bottom:
                sl = self._compute_atr_sl(data, ltp, "SELL")
                return self._make_signal(instrument_token, ltp, "SELL", 70.0, "ichimoku_below_cloud", sl)
        except Exception:
            pass
        return None


# ────────────────────────────────────────────────────────────
# 7. VCP (Volatility Contraction Pattern) Strategy
# ────────────────────────────────────────────────────────────

class VCPStrategy(SingleTriggerStrategy):
    """BUY on VCP pattern detection (Mark Minervini setup)."""

    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        defaults = {"min_bars": 60}
        merged = {**defaults, **(params or {})}
        super().__init__("vcp_pattern", merged)

    def generate_signal(self, data: pd.DataFrame, instrument_token: int = 0) -> Optional[Signal]:
        if len(data) < 60:
            return None
        close = data["close"]
        ltp = float(close.iloc[-1])

        try:
            vcp = detect_vcp(data, lookback=60)
            if vcp and vcp.get("detected"):
                sl = self._compute_atr_sl(data, ltp, "BUY")
                confidence = min(90.0, 50.0 + vcp.get("tightness", 0) * 10)
                return self._make_signal(instrument_token, ltp, "BUY", confidence, "vcp_pattern", sl)
        except Exception:
            pass
        return None


# ────────────────────────────────────────────────────────────
# 8. Pocket Pivot Strategy
# ────────────────────────────────────────────────────────────

class PocketPivotStrategy(SingleTriggerStrategy):
    """BUY on Pocket Pivot volume signal (institutional buying)."""

    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        super().__init__("pocket_pivot", params)

    def generate_signal(self, data: pd.DataFrame, instrument_token: int = 0) -> Optional[Signal]:
        if len(data) < 50:
            return None
        close = data["close"]
        ltp = float(close.iloc[-1])

        try:
            pp = detect_pocket_pivot(data)
            if pp and pp.get("detected"):
                sl = self._compute_atr_sl(data, ltp, "BUY")
                return self._make_signal(instrument_token, ltp, "BUY", 75.0, "pocket_pivot", sl)
        except Exception:
            pass
        return None


# ────────────────────────────────────────────────────────────
# 9. Weinstein Stage Strategy
# ────────────────────────────────────────────────────────────

class StageAnalysisStrategy(SingleTriggerStrategy):
    """BUY in Stage 2 uptrend, SELL in Stage 4 downtrend."""

    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        defaults = {"min_bars": 210, "min_confidence": 70}
        merged = {**defaults, **(params or {})}
        super().__init__("stage_analysis", merged)

    def generate_signal(self, data: pd.DataFrame, instrument_token: int = 0) -> Optional[Signal]:
        if len(data) < 210:
            return None
        close = data["close"]
        ltp = float(close.iloc[-1])

        try:
            stage = detect_stage(data)
            if stage is None:
                return None
            s = stage.get("stage", 0)
            conf = stage.get("confidence", 0)
            min_conf = self.params.get("min_confidence", 70)

            if s == 2 and conf >= min_conf:
                sl = self._compute_atr_sl(data, ltp, "BUY")
                return self._make_signal(instrument_token, ltp, "BUY", float(conf), "stage2_uptrend", sl)
            elif s == 4 and conf >= min_conf:
                sl = self._compute_atr_sl(data, ltp, "SELL")
                return self._make_signal(instrument_token, ltp, "SELL", float(conf), "stage4_downtrend", sl)
        except Exception:
            pass
        return None


# ────────────────────────────────────────────────────────────
# 10. CCI Extreme Strategy
# ────────────────────────────────────────────────────────────

class CCIExtremeStrategy(SingleTriggerStrategy):
    """BUY on extreme CCI oversold (< -200), SELL on extreme overbought (> 200)."""

    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        defaults = {"cci_extreme": 200, "cci_period": 20}
        merged = {**defaults, **(params or {})}
        super().__init__("cci_extreme", merged)

    def generate_signal(self, data: pd.DataFrame, instrument_token: int = 0) -> Optional[Signal]:
        if len(data) < 50:
            return None
        close = data["close"]
        ltp = float(close.iloc[-1])

        period = self.params.get("cci_period", 20)
        tp = (data["high"] + data["low"] + close) / 3
        tp_sma = tp.rolling(period).mean()
        tp_mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)

        cci = (tp - tp_sma) / (0.015 * tp_mad)
        if cci is None or len(cci) < 1 or np.isnan(float(cci.iloc[-1])):
            return None

        extreme = self.params.get("cci_extreme", 200)
        cci_val = float(cci.iloc[-1])

        if cci_val < -extreme:
            sl = self._compute_atr_sl(data, ltp, "BUY")
            return self._make_signal(instrument_token, ltp, "BUY", 75.0, "cci_extreme_oversold", sl)
        elif cci_val > extreme:
            sl = self._compute_atr_sl(data, ltp, "SELL")
            return self._make_signal(instrument_token, ltp, "SELL", 75.0, "cci_extreme_overbought", sl)
        return None


# ────────────────────────────────────────────────────────────
# 11. MFI (Money Flow Index) Strategy
# ────────────────────────────────────────────────────────────

class MFIStrategy(SingleTriggerStrategy):
    """BUY on MFI oversold (< 20), SELL on MFI overbought (> 80)."""

    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        defaults = {"mfi_oversold": 20, "mfi_overbought": 80}
        merged = {**defaults, **(params or {})}
        super().__init__("mfi_strategy", merged)

    def generate_signal(self, data: pd.DataFrame, instrument_token: int = 0) -> Optional[Signal]:
        if len(data) < 50:
            return None
        close = data["close"]
        ltp = float(close.iloc[-1])

        mfi_vals = compute_mfi(data["high"], data["low"], close, data["volume"])
        if mfi_vals is None or len(mfi_vals) < 1:
            return None

        mfi_val = float(mfi_vals.iloc[-1])
        if np.isnan(mfi_val):
            return None

        if mfi_val < self.params.get("mfi_oversold", 20):
            sl = self._compute_atr_sl(data, ltp, "BUY")
            return self._make_signal(instrument_token, ltp, "BUY", 65.0, "mfi_oversold", sl)
        elif mfi_val > self.params.get("mfi_overbought", 80):
            sl = self._compute_atr_sl(data, ltp, "SELL")
            return self._make_signal(instrument_token, ltp, "SELL", 65.0, "mfi_overbought", sl)
        return None


# ────────────────────────────────────────────────────────────
# 12. 52-Week High/Low Strategy
# ────────────────────────────────────────────────────────────

class FiftyTwoWeekStrategy(SingleTriggerStrategy):
    """BUY near 52-week high (momentum), SELL near 52-week low (breakdown)."""

    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        defaults = {"min_bars": 252, "high_pct": 0.98, "low_pct": 1.05}
        merged = {**defaults, **(params or {})}
        super().__init__("52_week_hl", merged)

    def generate_signal(self, data: pd.DataFrame, instrument_token: int = 0) -> Optional[Signal]:
        min_bars = self.params.get("min_bars", 252)
        if len(data) < min_bars:
            return None
        close = data["close"]
        ltp = float(close.iloc[-1])

        lookback = min(252, len(data))
        high_52w = float(data["high"].iloc[-lookback:].max())
        low_52w = float(data["low"].iloc[-lookback:].min())

        if ltp >= high_52w * self.params.get("high_pct", 0.98):
            sl = self._compute_atr_sl(data, ltp, "BUY")
            return self._make_signal(instrument_token, ltp, "BUY", 70.0, "near_52w_high", sl)
        elif ltp <= low_52w * self.params.get("low_pct", 1.05):
            sl = self._compute_atr_sl(data, ltp, "SELL")
            return self._make_signal(instrument_token, ltp, "SELL", 70.0, "near_52w_low", sl)
        return None


# ────────────────────────────────────────────────────────────
# Strategy registry for this module
# ────────────────────────────────────────────────────────────

ANALYSIS_STRATEGIES: dict[str, type[SingleTriggerStrategy]] = {
    "macd_crossover": MACDCrossoverStrategy,
    "golden_death_cross": GoldenDeathCrossStrategy,
    "volume_breakout_analysis": VolumeBreakoutAnalysisStrategy,
    "bollinger_bands": BollingerBandsStrategy,
    "stochastic": StochasticStrategy,
    "ichimoku_cloud": IchimokuStrategy,
    "vcp_pattern": VCPStrategy,
    "pocket_pivot": PocketPivotStrategy,
    "stage_analysis": StageAnalysisStrategy,
    "cci_extreme": CCIExtremeStrategy,
    "mfi_strategy": MFIStrategy,
    "52_week_hl": FiftyTwoWeekStrategy,
}

ANALYSIS_STRATEGY_LABELS: dict[str, str] = {
    "macd_crossover": "MACD Crossover",
    "golden_death_cross": "Golden / Death Cross",
    "volume_breakout_analysis": "Volume Breakout",
    "bollinger_bands": "Bollinger Bands",
    "stochastic": "Stochastic Oscillator",
    "ichimoku_cloud": "Ichimoku Cloud",
    "vcp_pattern": "VCP Pattern",
    "pocket_pivot": "Pocket Pivot",
    "stage_analysis": "Stage Analysis (Weinstein)",
    "cci_extreme": "CCI Extreme",
    "mfi_strategy": "MFI (Money Flow Index)",
    "52_week_hl": "52-Week High / Low",
}
