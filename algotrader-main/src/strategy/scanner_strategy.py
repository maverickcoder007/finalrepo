"""
ScannerStrategy — Converts scanner technical analysis triggers into
tradeable signals for live execution, backtesting, and paper trading.

Uses the same 15+ trigger detectors from StockScanner (MACD crossover,
RSI zones, Golden/Death Cross, Volume breakout, Bollinger, Stochastic,
MFI, CCI, Ichimoku, VCP, Pocket Pivot, Stage 2/4, RS Leader/Laggard)
plus trend scoring and super-performance criteria.

Generates BUY/SELL signals with ATR-based stop losses when triggers align.
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
# Trigger weights for composite scoring
# ────────────────────────────────────────────────────────────

TRIGGER_WEIGHTS: dict[str, float] = {
    # Strong signals (weight 3)
    "macd_crossover": 3.0,
    "golden_cross": 3.0,
    "death_cross": 3.0,
    "volume_breakout": 3.0,
    "volume_breakdown": 3.0,
    "vcp_setup": 3.0,
    "pocket_pivot": 3.0,
    "ichimoku_bullish": 3.0,
    "ichimoku_bearish": 3.0,
    "stage2_uptrend": 3.0,
    "stage4_downtrend": 3.0,
    "cci_extreme_oversold": 3.0,
    "cci_extreme_overbought": 3.0,
    "rs_leader": 3.0,
    "bollinger_squeeze": 2.5,
    # Moderate signals (weight 2)
    "rsi_oversold": 2.0,
    "rsi_overbought": 2.0,
    "near_52w_high": 2.0,
    "near_52w_low": 2.0,
    "bollinger_upper": 2.0,
    "bollinger_lower": 2.0,
    "stoch_oversold": 2.0,
    "stoch_overbought": 2.0,
    "mfi_oversold": 2.0,
    "mfi_overbought": 2.0,
    "rs_laggard": 2.0,
    # Additional EMA signals
    "ema_9_21_cross": 2.5,
    "ema_above_sma200": 2.0,
    "ema_below_sma200": 2.0,
    # ADX confirmation
    "strong_trend": 1.5,
    "tight_consolidation": 1.5,
}


class ScannerStrategy(BaseStrategy):
    """
    Composite technical scanner strategy that generates BUY/SELL signals
    based on multi-indicator trigger confluence.

    Works with live ticks, backtesting, and paper trading through
    the standard BaseStrategy interface.

    Configurable parameters:
    - min_bullish_score: Minimum weighted score to trigger BUY (default 5.0)
    - min_bearish_score: Minimum weighted score to trigger SELL (default 5.0)
    - min_triggers: Minimum number of aligned triggers (default 2)
    - use_trend_filter: Require trend_score > 40 for BUY (default True)
    - use_atr_sl: Use ATR-based stop loss (default True)
    - atr_sl_multiple: ATR multiplier for stop loss (default 2.0)
    - rsi_oversold: RSI level for oversold (default 30)
    - rsi_overbought: RSI level for overbought (default 70)
    - volume_breakout_ratio: Volume ratio threshold (default 2.0)
    - require_stage2: Only BUY in Stage 2 uptrend (default False)
    - require_superperf: Only BUY super performers (default False)
    """

    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        defaults = {
            "quantity": 1,
            "exchange": "NSE",
            "product": "MIS",
            "tradingsymbol_map": {},
            # Signal thresholds
            "min_bullish_score": 5.0,
            "min_bearish_score": 5.0,
            "min_triggers": 2,
            # Filters
            "use_trend_filter": True,
            "min_trend_score": 40,
            "use_atr_sl": True,
            "atr_sl_multiple": 2.0,
            "require_stage2": False,
            "require_superperf": False,
            # Indicator thresholds
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "volume_breakout_ratio": 2.0,
            "stoch_oversold": 20,
            "stoch_overbought": 80,
            "mfi_oversold": 20,
            "mfi_overbought": 80,
            "cci_extreme": 200,
            "bollinger_squeeze_threshold": 5.0,
            "rs_leader_threshold": 85,
            "rs_laggard_threshold": 15,
        }
        merged = {**defaults, **(params or {})}
        super().__init__("scanner_strategy", merged)
        self._prev_signal: dict[int, str] = {}
        self._benchmark_data: Optional[pd.DataFrame] = None

    def set_benchmark_data(self, benchmark_df: pd.DataFrame) -> None:
        """Set NIFTY 50 benchmark data for RS rating calculation."""
        self._benchmark_data = benchmark_df

    async def on_tick(self, ticks: list[Tick]) -> list[Signal]:
        signals: list[Signal] = []
        for tick in ticks:
            self.add_tick_to_buffer(tick)
            df = self.get_bar_data(tick.instrument_token)
            if df is not None and len(df) >= 50:
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

    def generate_signal(
        self, data: pd.DataFrame, instrument_token: int = 0
    ) -> Optional[Signal]:
        """
        Main signal generation: run all trigger detectors on the data,
        compute composite bullish/bearish scores, and emit BUY or SELL
        if thresholds are met.
        """
        if len(data) < 50:
            return None

        close = data["close"]
        ltp = float(close.iloc[-1])

        # Run all indicators
        indicators = compute_all_indicators(data, self._benchmark_data)

        # Detect all triggers
        triggers = self._detect_all_triggers(data, indicators, ltp)

        # Compute composite scores
        bullish_score = 0.0
        bearish_score = 0.0
        bullish_triggers: list[dict] = []
        bearish_triggers: list[dict] = []

        for t in triggers:
            weight = t.get("weight", 1.0)
            if t["type"] == "bullish":
                bullish_score += weight
                bullish_triggers.append(t)
            elif t["type"] == "bearish":
                bearish_score += weight
                bearish_triggers.append(t)

        # Compute trend score
        trend_score = self._compute_trend_score(indicators)

        # Apply filters
        if self.params["use_trend_filter"] and self.params["min_trend_score"] > 0:
            if bullish_score > 0 and trend_score < self.params["min_trend_score"]:
                bullish_score *= 0.5  # Discount bullish in weak trend

        stage = indicators.get("stage")
        if self.params["require_stage2"] and stage != 2:
            bullish_score = 0  # Only buy in Stage 2

        # Calculate stop loss
        stop_loss = None
        if self.params["use_atr_sl"] and len(data) >= 15:
            atr_val = atr(data, 14).iloc[-1]
            if not np.isnan(atr_val):
                stop_loss = atr_val * self.params["atr_sl_multiple"]

        tradingsymbol = self.params.get("tradingsymbol_map", {}).get(
            instrument_token, f"TOKEN_{instrument_token}"
        )

        # Check for BUY signal
        if (
            bullish_score >= self.params["min_bullish_score"]
            and len(bullish_triggers) >= self.params["min_triggers"]
            and bullish_score > bearish_score
        ):
            if self._prev_signal.get(instrument_token) != "BUY":
                self._prev_signal[instrument_token] = "BUY"
                confidence = min(bullish_score / 15.0 * 100, 99.0)  # Normalize to 0-99
                return Signal(
                    tradingsymbol=tradingsymbol,
                    exchange=Exchange(self.params["exchange"]),
                    transaction_type=TransactionType.BUY,
                    quantity=self.params["quantity"],
                    order_type=OrderType.MARKET,
                    stop_loss=stop_loss,
                    product=ProductType(self.params["product"]),
                    strategy_name=self.name,
                    confidence=confidence,
                    metadata={
                        "signal_type": "entry",
                        "direction": "BULLISH",
                        "bullish_score": round(bullish_score, 2),
                        "bearish_score": round(bearish_score, 2),
                        "trend_score": round(trend_score, 1),
                        "trigger_count": len(bullish_triggers),
                        "triggers": [t["signal"] for t in bullish_triggers],
                        "stage": stage,
                        "rs_rating": indicators.get("rs_rating"),
                        "rsi": indicators.get("rsi_14"),
                        "adx": indicators.get("adx_14"),
                        "atr_stop_loss": stop_loss,
                    },
                )

        # Check for SELL signal
        if (
            bearish_score >= self.params["min_bearish_score"]
            and len(bearish_triggers) >= self.params["min_triggers"]
            and bearish_score > bullish_score
        ):
            if self._prev_signal.get(instrument_token) != "SELL":
                self._prev_signal[instrument_token] = "SELL"
                confidence = min(bearish_score / 15.0 * 100, 99.0)
                return Signal(
                    tradingsymbol=tradingsymbol,
                    exchange=Exchange(self.params["exchange"]),
                    transaction_type=TransactionType.SELL,
                    quantity=self.params["quantity"],
                    order_type=OrderType.MARKET,
                    stop_loss=stop_loss,
                    product=ProductType(self.params["product"]),
                    strategy_name=self.name,
                    confidence=confidence,
                    metadata={
                        "signal_type": "entry",
                        "direction": "BEARISH",
                        "bullish_score": round(bullish_score, 2),
                        "bearish_score": round(bearish_score, 2),
                        "trend_score": round(trend_score, 1),
                        "trigger_count": len(bearish_triggers),
                        "triggers": [t["signal"] for t in bearish_triggers],
                        "stage": stage,
                        "rs_rating": indicators.get("rs_rating"),
                        "rsi": indicators.get("rsi_14"),
                        "adx": indicators.get("adx_14"),
                        "atr_stop_loss": stop_loss,
                    },
                )

        return None

    # ────────────────────────────────────────────────────────
    # Trigger Detectors (same logic as StockScanner._detect_triggers)
    # ────────────────────────────────────────────────────────

    def _detect_all_triggers(
        self, df: pd.DataFrame, ind: dict, ltp: float
    ) -> list[dict]:
        """Run all trigger detectors; returns list of {type, signal, weight}."""
        triggers: list[dict] = []
        close = df["close"]
        p = self.params

        # ── MACD crossover ──
        if len(close) >= 27:
            try:
                m = compute_macd(close)
                hist = m["histogram"]
                if len(hist) >= 2:
                    if hist.iloc[-2] < 0 and hist.iloc[-1] > 0:
                        triggers.append({"type": "bullish", "signal": "MACD Bullish Crossover", "weight": TRIGGER_WEIGHTS["macd_crossover"]})
                    elif hist.iloc[-2] > 0 and hist.iloc[-1] < 0:
                        triggers.append({"type": "bearish", "signal": "MACD Bearish Crossover", "weight": TRIGGER_WEIGHTS["macd_crossover"]})
            except Exception:
                pass

        # ── EMA 9/21 crossover ──
        if len(close) >= 22:
            try:
                fast = ema(close, 9)
                slow = ema(close, 21)
                if len(fast) >= 2 and len(slow) >= 2:
                    if fast.iloc[-2] <= slow.iloc[-2] and fast.iloc[-1] > slow.iloc[-1]:
                        triggers.append({"type": "bullish", "signal": "EMA 9/21 Bullish Cross", "weight": TRIGGER_WEIGHTS["ema_9_21_cross"]})
                    elif fast.iloc[-2] >= slow.iloc[-2] and fast.iloc[-1] < slow.iloc[-1]:
                        triggers.append({"type": "bearish", "signal": "EMA 9/21 Bearish Cross", "weight": TRIGGER_WEIGHTS["ema_9_21_cross"]})
            except Exception:
                pass

        # ── RSI zones ──
        rsi_val = ind.get("rsi_14")
        if rsi_val is not None:
            if rsi_val < p["rsi_oversold"]:
                triggers.append({"type": "bullish", "signal": f"RSI Oversold ({rsi_val:.1f})", "weight": TRIGGER_WEIGHTS["rsi_oversold"]})
            elif rsi_val > p["rsi_overbought"]:
                triggers.append({"type": "bearish", "signal": f"RSI Overbought ({rsi_val:.1f})", "weight": TRIGGER_WEIGHTS["rsi_overbought"]})

        # ── Golden / Death Cross ──
        if len(close) >= 201:
            try:
                sma50 = sma(close, 50)
                sma200 = sma(close, 200)
                if len(sma50) >= 2 and len(sma200) >= 2:
                    if sma50.iloc[-2] < sma200.iloc[-2] and sma50.iloc[-1] > sma200.iloc[-1]:
                        triggers.append({"type": "bullish", "signal": "Golden Cross (50 SMA > 200 SMA)", "weight": TRIGGER_WEIGHTS["golden_cross"]})
                    elif sma50.iloc[-2] > sma200.iloc[-2] and sma50.iloc[-1] < sma200.iloc[-1]:
                        triggers.append({"type": "bearish", "signal": "Death Cross (50 SMA < 200 SMA)", "weight": TRIGGER_WEIGHTS["death_cross"]})
            except Exception:
                pass

        # ── Price vs SMA 200 ──
        sma_200_val = ind.get("sma_200")
        if sma_200_val and ltp > 0:
            if ltp > sma_200_val:
                triggers.append({"type": "bullish", "signal": "Price above SMA 200", "weight": TRIGGER_WEIGHTS["ema_above_sma200"]})
            else:
                triggers.append({"type": "bearish", "signal": "Price below SMA 200", "weight": TRIGGER_WEIGHTS["ema_below_sma200"]})

        # ── Volume breakout / breakdown ──
        if "volume" in df.columns and len(df) >= 20:
            vol_ratio = ind.get("vol_ratio")
            change_1d = ind.get("change_1d", 0)
            if vol_ratio and vol_ratio > p["volume_breakout_ratio"]:
                if change_1d and change_1d > 2.0:
                    triggers.append({"type": "bullish", "signal": f"Volume Breakout ({vol_ratio:.1f}x, +{change_1d:.1f}%)", "weight": TRIGGER_WEIGHTS["volume_breakout"]})
                elif change_1d and change_1d < -2.0:
                    triggers.append({"type": "bearish", "signal": f"Volume Breakdown ({vol_ratio:.1f}x, {change_1d:.1f}%)", "weight": TRIGGER_WEIGHTS["volume_breakdown"]})

        # ── 52-week proximity ──
        high_52w = ind.get("high_52w", 0)
        if high_52w and ltp >= high_52w * 0.98:
            triggers.append({"type": "bullish", "signal": "Near 52-Week High", "weight": TRIGGER_WEIGHTS["near_52w_high"]})

        low_52w = ind.get("low_52w", float("inf"))
        if low_52w and low_52w != float("inf") and ltp <= low_52w * 1.05:
            triggers.append({"type": "bearish", "signal": "Near 52-Week Low", "weight": TRIGGER_WEIGHTS["near_52w_low"]})

        # ── Bollinger Bands ──
        bb_upper = ind.get("bb_upper")
        bb_lower = ind.get("bb_lower")
        if bb_upper and ltp > bb_upper:
            triggers.append({"type": "bearish", "signal": "Above Upper Bollinger Band", "weight": TRIGGER_WEIGHTS["bollinger_upper"]})
        elif bb_lower and ltp < bb_lower:
            triggers.append({"type": "bullish", "signal": "Below Lower Bollinger Band", "weight": TRIGGER_WEIGHTS["bollinger_lower"]})

        # ── Bollinger Squeeze ──
        bb_bw = ind.get("bb_bandwidth")
        if bb_bw is not None and bb_bw < p["bollinger_squeeze_threshold"]:
            triggers.append({"type": "bullish", "signal": f"Bollinger Squeeze (BW={bb_bw:.1f}%)", "weight": TRIGGER_WEIGHTS["bollinger_squeeze"]})

        # ── ADX strong trend ──
        adx_val = ind.get("adx_14")
        if adx_val and adx_val > 40:
            triggers.append({"type": "bullish", "signal": f"Very Strong Trend (ADX {adx_val:.1f})", "weight": TRIGGER_WEIGHTS["strong_trend"]})

        # ── Stochastic ──
        stoch_k = ind.get("stoch_k")
        stoch_d = ind.get("stoch_d")
        if stoch_k is not None and stoch_d is not None:
            if stoch_k < p["stoch_oversold"] and stoch_d < p["stoch_oversold"]:
                triggers.append({"type": "bullish", "signal": f"Stochastic Oversold (K={stoch_k:.0f})", "weight": TRIGGER_WEIGHTS["stoch_oversold"]})
            elif stoch_k > p["stoch_overbought"] and stoch_d > p["stoch_overbought"]:
                triggers.append({"type": "bearish", "signal": f"Stochastic Overbought (K={stoch_k:.0f})", "weight": TRIGGER_WEIGHTS["stoch_overbought"]})

        # ── MFI ──
        mfi_val = ind.get("mfi_14")
        if mfi_val is not None:
            if mfi_val < p["mfi_oversold"]:
                triggers.append({"type": "bullish", "signal": f"MFI Oversold ({mfi_val:.0f})", "weight": TRIGGER_WEIGHTS["mfi_oversold"]})
            elif mfi_val > p["mfi_overbought"]:
                triggers.append({"type": "bearish", "signal": f"MFI Overbought ({mfi_val:.0f})", "weight": TRIGGER_WEIGHTS["mfi_overbought"]})

        # ── CCI extremes ──
        cci_val = ind.get("cci_20")
        if cci_val is not None:
            if cci_val < -p["cci_extreme"]:
                triggers.append({"type": "bullish", "signal": f"CCI Extremely Oversold ({cci_val:.0f})", "weight": TRIGGER_WEIGHTS["cci_extreme_oversold"]})
            elif cci_val > p["cci_extreme"]:
                triggers.append({"type": "bearish", "signal": f"CCI Extremely Overbought ({cci_val:.0f})", "weight": TRIGGER_WEIGHTS["cci_extreme_overbought"]})

        # ── Ichimoku cloud ──
        ichi_signal = ind.get("ichimoku_signal")
        if ichi_signal == "bullish":
            triggers.append({"type": "bullish", "signal": "Price Above Ichimoku Cloud", "weight": TRIGGER_WEIGHTS["ichimoku_bullish"]})
        elif ichi_signal == "bearish":
            triggers.append({"type": "bearish", "signal": "Price Below Ichimoku Cloud", "weight": TRIGGER_WEIGHTS["ichimoku_bearish"]})

        # ── VCP setup ──
        if ind.get("vcp_detected"):
            triggers.append({"type": "bullish", "signal": f"VCP Setup (tightness={ind.get('vcp_tightness', 0):.2f})", "weight": TRIGGER_WEIGHTS["vcp_setup"]})

        # ── Pocket pivot ──
        if ind.get("pocket_pivot"):
            triggers.append({"type": "bullish", "signal": "Pocket Pivot Buy Signal", "weight": TRIGGER_WEIGHTS["pocket_pivot"]})

        # ── Stage analysis ──
        stage = ind.get("stage")
        stage_conf = ind.get("stage_confidence", 0)
        if stage == 2 and stage_conf > 70:
            triggers.append({"type": "bullish", "signal": f"Stage 2 Uptrend (conf {stage_conf}%)", "weight": TRIGGER_WEIGHTS["stage2_uptrend"]})
        elif stage == 4 and stage_conf > 70:
            triggers.append({"type": "bearish", "signal": f"Stage 4 Downtrend (conf {stage_conf}%)", "weight": TRIGGER_WEIGHTS["stage4_downtrend"]})

        # ── RS rating ──
        rs = ind.get("rs_rating")
        if rs is not None:
            if rs >= p["rs_leader_threshold"]:
                triggers.append({"type": "bullish", "signal": f"RS Leader ({rs:.0f}/99)", "weight": TRIGGER_WEIGHTS["rs_leader"]})
            elif rs <= p["rs_laggard_threshold"]:
                triggers.append({"type": "bearish", "signal": f"RS Laggard ({rs:.0f}/99)", "weight": TRIGGER_WEIGHTS["rs_laggard"]})

        # ── Tight consolidation ──
        if len(close) >= 5:
            recent_highs = df["high"].tail(5)
            recent_lows = df["low"].tail(5)
            range_pct = ((recent_highs.max() - recent_lows.min()) / ltp) * 100
            if range_pct < 3.0:
                triggers.append({"type": "bullish", "signal": f"Tight Consolidation ({range_pct:.1f}%)", "weight": TRIGGER_WEIGHTS["tight_consolidation"]})

        return triggers

    # ────────────────────────────────────────────────────────
    # Trend Score (mirrors StockScanner._compute_trend_score)
    # ────────────────────────────────────────────────────────

    def _compute_trend_score(self, ind: dict) -> float:
        """Compute 0-100 trend score from indicators."""
        score = 50.0  # Start neutral

        # RSI
        rsi_val = ind.get("rsi_14")
        if rsi_val:
            if 50 < rsi_val < 70:
                score += 10
            elif rsi_val >= 70:
                score += 5
            elif rsi_val < 30:
                score -= 10

        # MACD
        macd_h = ind.get("macd_histogram")
        if macd_h is not None:
            if macd_h > 0:
                score += 8
            else:
                score -= 6

        # ADX
        adx_val = ind.get("adx_14")
        if adx_val:
            if adx_val > 25:
                score += 8
            elif adx_val < 15:
                score -= 4

        # Stage
        stage = ind.get("stage")
        if stage == 2:
            score += 12
        elif stage == 1:
            score += 5
        elif stage == 3:
            score -= 5
        elif stage == 4:
            score -= 12

        # RS rating
        rs = ind.get("rs_rating")
        if rs is not None:
            if rs >= 80:
                score += 10
            elif rs >= 60:
                score += 5
            elif rs <= 20:
                score -= 8

        # Volume
        vol_r = ind.get("vol_ratio")
        if vol_r and vol_r > 1.5:
            score += 5

        return max(0, min(100, score))
