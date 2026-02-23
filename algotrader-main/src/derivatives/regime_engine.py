"""
Market Regime Engine — Classify market conditions for strategy selection.

Regimes determine which F&O strategy structures are optimal:
  LOW_VOL          → sell premium (iron condors, strangles)
  HIGH_VOL         → buy wings, reduce size
  TRENDING_UP/DOWN → directional spreads, covered positions
  MEAN_REVERTING   → short straddles, butterflies
  EVENT_RISK       → reduce exposure or buy protection
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np


class MarketRegime(str, Enum):
    LOW_VOL = "LOW_VOL"
    HIGH_VOL = "HIGH_VOL"
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    MEAN_REVERTING = "MEAN_REVERTING"
    EVENT_RISK = "EVENT_RISK"
    UNKNOWN = "UNKNOWN"


@dataclass
class RegimeClassification:
    """Result of regime classification."""
    regime: MarketRegime
    confidence: float = 0.0       # 0-1 confidence in classification
    hv_20: float = 0.0            # 20-day historical volatility (annualised)
    hv_percentile: float = 0.0    # HV percentile vs 252-day lookback
    iv_hv_ratio: float = 0.0      # implied vol / historical vol
    adx: float = 0.0              # Average Directional Index (trend strength)
    trend_direction: float = 0.0  # +1 up, -1 down, ~0 neutral
    mean_reversion_z: float = 0.0 # z-score from 20-day mean
    vix_level: float = 0.0        # India VIX or proxy
    timestamp: datetime | None = None

    # Strategy hints
    recommended_structures: list[str] = field(default_factory=list)
    position_size_factor: float = 1.0  # 0.5 = half size, 1.0 = normal

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime.value,
            "confidence": round(self.confidence, 2),
            "hv_20": round(self.hv_20, 4),
            "hv_percentile": round(self.hv_percentile, 1),
            "iv_hv_ratio": round(self.iv_hv_ratio, 2),
            "adx": round(self.adx, 2),
            "trend_direction": round(self.trend_direction, 2),
            "mean_reversion_z": round(self.mean_reversion_z, 2),
            "vix_level": round(self.vix_level, 2),
            "recommended_structures": self.recommended_structures,
            "position_size_factor": round(self.position_size_factor, 2),
        }


class RegimeEngine:
    """Classify market regime from price history and volatility data.

    Uses ensemble of indicators:
    1. HV percentile (252-day lookback)
    2. IV/HV ratio (if available)
    3. ADX for trend strength
    4. Z-score for mean reversion detection
    5. VIX level for event risk
    """

    # Regime thresholds
    LOW_VOL_HV_PCTILE = 30       # Below 30th percentile → low vol
    HIGH_VOL_HV_PCTILE = 70      # Above 70th percentile → high vol
    TREND_ADX_THRESHOLD = 25      # ADX > 25 → trending
    MEAN_REV_Z_THRESHOLD = 1.5    # |z-score| > 1.5 → mean reverting opportunity
    EVENT_VIX_THRESHOLD = 22      # VIX > 22 → event risk (India VIX reference)

    # Strategy mapping
    REGIME_STRATEGIES: dict[MarketRegime, list[str]] = {
        MarketRegime.LOW_VOL: [
            "IRON_CONDOR", "SHORT_STRANGLE", "SHORT_STRADDLE",
            "COVERED_CALL", "CALENDAR_SPREAD"
        ],
        MarketRegime.HIGH_VOL: [
            "LONG_STRADDLE", "LONG_STRANGLE", "PROTECTIVE_PUT",
            "BULL_PUT_SPREAD", "BEAR_CALL_SPREAD"
        ],
        MarketRegime.TRENDING_UP: [
            "BULL_CALL_SPREAD", "BULL_PUT_SPREAD", "COVERED_CALL",
            "NAKED_PUT"
        ],
        MarketRegime.TRENDING_DOWN: [
            "BEAR_PUT_SPREAD", "BEAR_CALL_SPREAD", "PROTECTIVE_PUT",
            "LONG_PUT"
        ],
        MarketRegime.MEAN_REVERTING: [
            "SHORT_STRADDLE", "IRON_BUTTERFLY", "SHORT_STRANGLE",
            "IRON_CONDOR"
        ],
        MarketRegime.EVENT_RISK: [
            "LONG_STRADDLE", "PROTECTIVE_PUT", "COLLAR",
        ],
    }

    REGIME_SIZE_FACTOR: dict[MarketRegime, float] = {
        MarketRegime.LOW_VOL: 1.0,
        MarketRegime.HIGH_VOL: 0.5,      # reduce size in high vol
        MarketRegime.TRENDING_UP: 0.8,
        MarketRegime.TRENDING_DOWN: 0.7,
        MarketRegime.MEAN_REVERTING: 1.0,
        MarketRegime.EVENT_RISK: 0.3,     # minimal size in event risk
        MarketRegime.UNKNOWN: 0.5,
    }

    def __init__(self) -> None:
        self._history: list[RegimeClassification] = []

    def classify(
        self,
        closes: np.ndarray,
        current_iv: float = 0.0,
        vix: float = 0.0,
        timestamp: datetime | None = None,
    ) -> RegimeClassification:
        """Classify current market regime from price history.

        Args:
            closes: array of closing prices (at least 252 bars recommended)
            current_iv: current at-the-money implied volatility
            vix: India VIX or proxy volatility index
            timestamp: current timestamp
        """
        if len(closes) < 30:
            return RegimeClassification(
                regime=MarketRegime.UNKNOWN,
                timestamp=timestamp,
            )

        # ── Historical Volatility ───
        returns = np.diff(np.log(closes))
        hv_20 = float(np.std(returns[-20:]) * np.sqrt(252)) if len(returns) >= 20 else 0
        hv_all = [
            float(np.std(returns[max(0, i - 20):i]) * np.sqrt(252))
            for i in range(21, len(returns) + 1)
        ] if len(returns) >= 21 else [hv_20]
        hv_percentile = float(np.percentile(
            hv_all, (np.searchsorted(np.sort(hv_all), hv_20) / max(len(hv_all), 1)) * 100
        )) if hv_all else 50

        # Better: rank-based percentile
        hv_arr = np.array(hv_all) if hv_all else np.array([hv_20])
        hv_percentile = float(np.sum(hv_arr <= hv_20) / max(len(hv_arr), 1) * 100)

        # ── IV/HV Ratio ───
        iv_hv_ratio = current_iv / max(hv_20, 0.01) if current_iv > 0 else 1.0

        # ── ADX (Average Directional Index) ───
        adx = self._compute_adx(closes, period=14)

        # ── Trend direction ───
        sma_20 = float(np.mean(closes[-20:])) if len(closes) >= 20 else float(closes[-1])
        sma_50 = float(np.mean(closes[-50:])) if len(closes) >= 50 else sma_20
        current_price = float(closes[-1])

        if current_price > sma_20 > sma_50:
            trend_direction = 1.0
        elif current_price < sma_20 < sma_50:
            trend_direction = -1.0
        else:
            trend_direction = 0.0

        # ── Mean reversion z-score ───
        mean_20 = float(np.mean(closes[-20:])) if len(closes) >= 20 else current_price
        std_20 = float(np.std(closes[-20:])) if len(closes) >= 20 else 1.0
        mean_rev_z = (current_price - mean_20) / max(std_20, 0.01)

        # ── Regime classification (priority-based) ───
        regime = MarketRegime.UNKNOWN
        confidence = 0.0

        # 1. Event risk (highest priority)
        if vix > self.EVENT_VIX_THRESHOLD or (hv_percentile > 90 and iv_hv_ratio > 1.5):
            regime = MarketRegime.EVENT_RISK
            confidence = min(1.0, vix / 30) if vix > 0 else 0.8

        # 2. High volatility
        elif hv_percentile > self.HIGH_VOL_HV_PCTILE:
            if adx > self.TREND_ADX_THRESHOLD:
                regime = (
                    MarketRegime.TRENDING_UP if trend_direction > 0
                    else MarketRegime.TRENDING_DOWN
                )
                confidence = min(1.0, adx / 40)
            else:
                regime = MarketRegime.HIGH_VOL
                confidence = hv_percentile / 100

        # 3. Low volatility
        elif hv_percentile < self.LOW_VOL_HV_PCTILE:
            if abs(mean_rev_z) > self.MEAN_REV_Z_THRESHOLD:
                regime = MarketRegime.MEAN_REVERTING
                confidence = min(1.0, abs(mean_rev_z) / 3)
            else:
                regime = MarketRegime.LOW_VOL
                confidence = 1 - hv_percentile / 100

        # 4. Trending
        elif adx > self.TREND_ADX_THRESHOLD:
            regime = (
                MarketRegime.TRENDING_UP if trend_direction > 0
                else MarketRegime.TRENDING_DOWN
            )
            confidence = min(1.0, adx / 40)

        # 5. Mean reverting
        elif abs(mean_rev_z) > self.MEAN_REV_Z_THRESHOLD:
            regime = MarketRegime.MEAN_REVERTING
            confidence = min(1.0, abs(mean_rev_z) / 3)

        # 6. Default: check IV/HV for direction
        else:
            if iv_hv_ratio > 1.3:
                regime = MarketRegime.HIGH_VOL
                confidence = 0.5
            elif iv_hv_ratio < 0.8:
                regime = MarketRegime.LOW_VOL
                confidence = 0.5
            else:
                regime = MarketRegime.UNKNOWN
                confidence = 0.3

        result = RegimeClassification(
            regime=regime,
            confidence=confidence,
            hv_20=hv_20,
            hv_percentile=hv_percentile,
            iv_hv_ratio=iv_hv_ratio,
            adx=adx,
            trend_direction=trend_direction,
            mean_reversion_z=mean_rev_z,
            vix_level=vix,
            timestamp=timestamp,
            recommended_structures=self.REGIME_STRATEGIES.get(regime, []),
            position_size_factor=self.REGIME_SIZE_FACTOR.get(regime, 0.5),
        )

        self._history.append(result)
        if len(self._history) > 1000:
            self._history = self._history[-1000:]

        return result

    def _compute_adx(self, closes: np.ndarray, period: int = 14) -> float:
        """Compute Average Directional Index."""
        if len(closes) < period * 2:
            return 0.0

        # Simplified ADX using close prices only (no high/low)
        # True range approximation: |close[i] - close[i-1]|
        diffs = np.diff(closes)
        abs_diffs = np.abs(diffs)

        # +DM and -DM approximation
        plus_dm = np.where(diffs > 0, diffs, 0)
        minus_dm = np.where(diffs < 0, -diffs, 0)

        # Smoothed averages
        tr_smooth = self._ema(abs_diffs, period)
        plus_smooth = self._ema(plus_dm, period)
        minus_smooth = self._ema(minus_dm, period)

        if len(tr_smooth) == 0:
            return 0.0

        # +DI and -DI
        plus_di = np.where(tr_smooth > 0, 100 * plus_smooth / tr_smooth, 0)
        minus_di = np.where(tr_smooth > 0, 100 * minus_smooth / tr_smooth, 0)

        # DX
        di_sum = plus_di + minus_di
        dx = np.where(di_sum > 0, 100 * np.abs(plus_di - minus_di) / di_sum, 0)

        # ADX = smoothed DX
        adx_vals = self._ema(dx, period)
        return float(adx_vals[-1]) if len(adx_vals) > 0 else 0.0

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        """Exponential moving average."""
        if len(data) < period:
            return data
        alpha = 2.0 / (period + 1)
        result = np.empty_like(data, dtype=float)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        return result

    def get_regime_history(self) -> list[dict[str, Any]]:
        """Get regime classification history."""
        return [r.to_dict() for r in self._history]

    def get_current_regime(self) -> RegimeClassification | None:
        """Get most recent regime classification."""
        return self._history[-1] if self._history else None
