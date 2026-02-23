from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ────────────────────────────────────────────────────────────────
# Moving Averages
# ────────────────────────────────────────────────────────────────

def sma(series: Any, period: int) -> Any:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: Any, period: int) -> Any:
    return series.ewm(span=period, adjust=False).mean()


# ────────────────────────────────────────────────────────────────
# Momentum Oscillators
# ────────────────────────────────────────────────────────────────

def rsi(series: Any, period: int = 14) -> Any:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> dict[str, Any]:
    """Stochastic Oscillator (%K and %D)."""
    low_min = df["low"].rolling(window=k_period, min_periods=k_period).min()
    high_max = df["high"].rolling(window=k_period, min_periods=k_period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    k = 100 * ((df["close"] - low_min) / denom)
    d = k.rolling(window=d_period).mean()
    return {"k": k, "d": d}


def williams_r(df: pd.DataFrame, period: int = 14) -> Any:
    """Williams %R — momentum indicator."""
    high_max = df["high"].rolling(window=period, min_periods=period).max()
    low_min = df["low"].rolling(window=period, min_periods=period).min()
    denom = (high_max - low_min).replace(0, np.nan)
    return -100 * ((high_max - df["close"]) / denom)


def cci(df: pd.DataFrame, period: int = 20) -> Any:
    """Commodity Channel Index."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))


def roc(series: Any, period: int = 12) -> Any:
    """Rate of Change — momentum as percentage."""
    shifted = series.shift(period)
    return ((series - shifted) / shifted.replace(0, np.nan)) * 100


# ────────────────────────────────────────────────────────────────
# Trend Indicators
# ────────────────────────────────────────────────────────────────

def macd(series: Any, fast: int = 12, slow: int = 26, signal: int = 9) -> dict[str, Any]:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return {"macd": macd_line, "signal": signal_line, "histogram": histogram}


def adx(df: pd.DataFrame, period: int = 14) -> Any:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_val = tr.ewm(alpha=1 / period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr_val)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr_val)

    di_sum = plus_di + minus_di
    di_sum = di_sum.replace(0, np.nan)
    dx = 100 * ((plus_di - minus_di).abs() / di_sum)
    adx_result = dx.ewm(alpha=1 / period, min_periods=period).mean()
    return adx_result


def adx_components(df: pd.DataFrame, period: int = 14) -> dict[str, Any]:
    """Return ADX with +DI and -DI components for directional analysis."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_val = tr.ewm(alpha=1 / period, min_periods=period).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr_val)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr_val)
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * ((plus_di - minus_di).abs() / di_sum)
    adx_val = dx.ewm(alpha=1 / period, min_periods=period).mean()

    return {"adx": adx_val, "plus_di": plus_di, "minus_di": minus_di}


def ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52) -> dict[str, Any]:
    """Ichimoku Cloud components."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    senkou_b_line = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)
    chikou = close.shift(-kijun)

    return {
        "tenkan": tenkan_sen,
        "kijun": kijun_sen,
        "senkou_a": senkou_a,
        "senkou_b": senkou_b_line,
        "chikou": chikou,
    }


def linear_regression_slope(series: Any, period: int = 20) -> Any:
    """Slope of linear regression line — positive = uptrend."""
    def _slope(arr):
        if len(arr) < period:
            return np.nan
        x = np.arange(len(arr))
        coeffs = np.polyfit(x, arr, 1)
        return coeffs[0]
    return series.rolling(window=period).apply(_slope, raw=True)


# ────────────────────────────────────────────────────────────────
# Volatility Indicators
# ────────────────────────────────────────────────────────────────

def atr(df: pd.DataFrame, period: int = 14) -> Any:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def bollinger_bands(series: Any, period: int = 20, std_dev: float = 2.0) -> dict[str, Any]:
    mid = sma(series, period)
    std = series.rolling(window=period).std()
    return {
        "upper": mid + std_dev * std,
        "middle": mid,
        "lower": mid - std_dev * std,
    }


def bollinger_bandwidth(series: Any, period: int = 20, std_dev: float = 2.0) -> Any:
    """Bollinger Bandwidth — measures volatility contraction/expansion."""
    bb = bollinger_bands(series, period, std_dev)
    mid = bb["middle"].replace(0, np.nan)
    return ((bb["upper"] - bb["lower"]) / mid) * 100


def historical_volatility(series: Any, period: int = 20) -> Any:
    """Annualized historical volatility (std of log returns)."""
    log_ret = np.log(series / series.shift(1))
    return log_ret.rolling(window=period).std() * np.sqrt(252) * 100


# ────────────────────────────────────────────────────────────────
# Volume Indicators
# ────────────────────────────────────────────────────────────────

def volume_ratio(volume: Any, period: int = 20) -> Any:
    avg_vol = volume.rolling(window=period, min_periods=period).mean()
    return volume / avg_vol.replace(0, np.nan)


def obv(df: pd.DataFrame) -> Any:
    """On-Balance Volume — cumulative volume flow."""
    close = df["close"]
    vol = df["volume"]
    direction = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
    return (vol * direction).cumsum()


def mfi(df: pd.DataFrame, period: int = 14) -> Any:
    """Money Flow Index — volume-weighted RSI."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]
    pos_mf = mf.where(tp > tp.shift(1), 0.0)
    neg_mf = mf.where(tp < tp.shift(1), 0.0)
    pos_sum = pos_mf.rolling(window=period).sum()
    neg_sum = neg_mf.rolling(window=period).sum().replace(0, np.nan)
    ratio = pos_sum / neg_sum
    return 100 - (100 / (1 + ratio))


def accumulation_distribution(df: pd.DataFrame) -> Any:
    """Accumulation/Distribution Line."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]
    hl_range = (high - low).replace(0, np.nan)
    clv = ((close - low) - (high - close)) / hl_range
    return (clv * vol).cumsum()


# ────────────────────────────────────────────────────────────────
# Relative Strength & Support/Resistance
# ────────────────────────────────────────────────────────────────

def relative_strength(close: Any, benchmark_close: Any) -> Any:
    stock_ret = close.pct_change(periods=63).fillna(0)
    bench_ret = benchmark_close.pct_change(periods=63).fillna(0)
    rs = stock_ret / bench_ret.replace(0, np.nan)
    return rs


def rs_rating(close: Any, benchmark_close: Any) -> float:
    """IBD-style Relative Strength Rating (0-99) combining multiple timeframes."""
    if len(close) < 252 or len(benchmark_close) < 252:
        # Fallback with available data
        n = min(len(close), len(benchmark_close))
        if n < 21:
            return 50.0
        stock_ret = float((close.iloc[-1] / close.iloc[0]) - 1)
        bench_ret = float((benchmark_close.iloc[-1] / benchmark_close.iloc[0]) - 1)
        raw = stock_ret - bench_ret
        return max(1.0, min(99.0, 50 + raw * 200))

    # Weighted: 40% Q1 + 20% Q2 + 20% Q3 + 20% Q4
    def _qtr_perf(s, start, end):
        if end >= len(s) or start >= len(s):
            return 0.0
        return float((s.iloc[-1 - start] / s.iloc[-1 - end]) - 1) if s.iloc[-1 - end] != 0 else 0.0

    stock_q1 = _qtr_perf(close, 0, 63)
    stock_q2 = _qtr_perf(close, 63, 126)
    stock_q3 = _qtr_perf(close, 126, 189)
    stock_q4 = _qtr_perf(close, 189, 252)

    bench_q1 = _qtr_perf(benchmark_close, 0, 63)
    bench_q2 = _qtr_perf(benchmark_close, 63, 126)
    bench_q3 = _qtr_perf(benchmark_close, 126, 189)
    bench_q4 = _qtr_perf(benchmark_close, 189, 252)

    stock_score = 0.4 * stock_q1 + 0.2 * stock_q2 + 0.2 * stock_q3 + 0.2 * stock_q4
    bench_score = 0.4 * bench_q1 + 0.2 * bench_q2 + 0.2 * bench_q3 + 0.2 * bench_q4

    # Normalize to 1-99 scale
    raw_rs = stock_score - bench_score
    rating = max(1.0, min(99.0, 50 + raw_rs * 300))
    return round(rating, 1)


def pivot_points(df: pd.DataFrame) -> dict[str, float]:
    """Classic pivot points from previous day's OHLC."""
    if len(df) < 2:
        return {}
    prev = df.iloc[-2]
    pp = (prev["high"] + prev["low"] + prev["close"]) / 3
    r1 = 2 * pp - prev["low"]
    s1 = 2 * pp - prev["high"]
    r2 = pp + (prev["high"] - prev["low"])
    s2 = pp - (prev["high"] - prev["low"])
    r3 = prev["high"] + 2 * (pp - prev["low"])
    s3 = prev["low"] - 2 * (prev["high"] - pp)
    return {"pp": pp, "r1": r1, "r2": r2, "r3": r3, "s1": s1, "s2": s2, "s3": s3}


def week52_high_low(close: Any) -> dict[str, float]:
    if len(close) < 5:
        return {"high_52w": float(close.max()), "low_52w": float(close.min())}
    lookback = min(len(close), 252)
    recent = close.tail(lookback)
    return {"high_52w": float(recent.max()), "low_52w": float(recent.min())}


def price_change_pct(close: Any, periods: int = 1) -> float:
    if len(close) < periods + 1:
        return 0.0
    return float(((close.iloc[-1] / close.iloc[-1 - periods]) - 1) * 100)


# ────────────────────────────────────────────────────────────────
# Pattern Detection Helpers
# ────────────────────────────────────────────────────────────────

def detect_vcp(df: pd.DataFrame, contractions: int = 3) -> dict[str, Any]:
    """
    Detect Volatility Contraction Pattern (Mark Minervini).
    Looks for successive tightening of price ranges indicating institutional accumulation.
    Uses 120-bar lookback with 15-bar windows for better detection coverage.
    Returns {'detected': bool, 'contractions': int, 'tightness': float, 'pivot': float}
    """
    if len(df) < 50:
        return {"detected": False}

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Use up to 120 bars, divided into windows of 15 bars each
    lookback = min(len(df), 120)
    window_size = 15
    num_windows = lookback // window_size

    if num_windows < 3:
        return {"detected": False}

    # Compute percentage range for each window (oldest → newest)
    ranges = []
    for i in range(num_windows):
        start_idx = -lookback + i * window_size
        end_idx = start_idx + window_size
        window = df.iloc[start_idx:] if end_idx >= 0 else df.iloc[start_idx:end_idx]
        if len(window) < 5:
            continue
        pct_range = ((window["high"].max() - window["low"].min()) / window["close"].mean()) * 100
        ranges.append(pct_range)

    if len(ranges) < 3:
        return {"detected": False}

    # Count contractions (window range smaller than or ≈ equal to previous)
    contraction_count = 0
    for i in range(1, len(ranges)):
        if ranges[i] < ranges[i - 1] * 1.05:  # 5% tolerance for noise
            contraction_count += 1

    # Overall contraction ratio: newest window vs oldest window
    contraction_ratio = ranges[-1] / ranges[0] if ranges[0] > 0 else 1.0

    # Current tightness (last 10 bars vs last 50 bars)
    tail_wide = min(50, len(df))
    recent_range = ((high.tail(10).max() - low.tail(10).min()) / close.tail(10).mean()) * 100
    wider_range = ((high.tail(tail_wide).max() - low.tail(tail_wide).min()) / close.tail(tail_wide).mean()) * 100
    tightness = recent_range / wider_range if wider_range > 0 else 1.0

    # Pivot = recent high as breakout level
    pivot = float(high.tail(10).max())

    # Detection criteria (multiple paths — any is sufficient):
    # Path 1: Enough contracting windows + tight recent range
    # Path 2: Strong overall contraction even with fewer counted windows
    min_contractions = max(1, contractions - 1)
    detected = (
        (contraction_count >= min_contractions and tightness < 0.70)
        or (contraction_ratio < 0.50 and contraction_count >= 2)
    )

    return {
        "detected": detected,
        "contractions": contraction_count,
        "tightness": round(tightness, 3),
        "pivot": round(pivot, 2),
        "ranges": [round(r, 2) for r in ranges],
    }


def detect_stage(df: pd.DataFrame) -> dict[str, Any]:
    """
    Stan Weinstein Stage Analysis:
    Stage 1 — Basing (price near flat 200 SMA, low volume)
    Stage 2 — Advancing (price above rising 200 SMA, volume expanding)
    Stage 3 — Topping (price starts failing at 200 SMA, volume high)
    Stage 4 — Declining (price below falling 200 SMA)
    """
    if len(df) < 220:
        return {"stage": 0, "stage_name": "Insufficient Data", "confidence": 0}

    close = df["close"]
    vol = df["volume"] if "volume" in df.columns else pd.Series([0] * len(df))
    ltp = float(close.iloc[-1])

    sma_200 = sma(close, 200)
    sma_50 = sma(close, 50)

    current_200 = float(sma_200.iloc[-1])
    prev_200_20 = float(sma_200.iloc[-20])
    prev_200_60 = float(sma_200.iloc[-60]) if len(sma_200) >= 60 else current_200
    current_50 = float(sma_50.iloc[-1])

    slope_200 = (current_200 - prev_200_20) / prev_200_20 * 100 if prev_200_20 else 0
    slope_200_long = (current_200 - prev_200_60) / prev_200_60 * 100 if prev_200_60 else 0

    price_vs_200 = ((ltp / current_200) - 1) * 100 if current_200 else 0
    price_vs_50 = ((ltp / current_50) - 1) * 100 if current_50 else 0

    # Average volume trend
    avg_vol_recent = float(vol.tail(20).mean()) if len(vol) >= 20 else 0
    avg_vol_older = float(vol.tail(60).head(40).mean()) if len(vol) >= 60 else avg_vol_recent
    vol_trend = (avg_vol_recent / avg_vol_older) if avg_vol_older > 0 else 1.0

    # Determine stage
    if abs(slope_200) < 0.3 and abs(price_vs_200) < 5:
        stage = 1
        stage_name = "Basing"
        confidence = min(90, int(100 - abs(slope_200) * 50 - abs(price_vs_200) * 5))
    elif price_vs_200 > 0 and slope_200 > 0.1 and price_vs_50 > -5:
        stage = 2
        stage_name = "Advancing"
        confidence = min(95, int(30 + price_vs_200 * 2 + slope_200 * 10))
    elif abs(price_vs_200) < 8 and slope_200 < 0.5 and slope_200_long > 0 and vol_trend > 1.1:
        stage = 3
        stage_name = "Topping"
        confidence = min(85, int(50 + vol_trend * 10))
    elif price_vs_200 < 0 and slope_200 < -0.1:
        stage = 4
        stage_name = "Declining"
        confidence = min(95, int(30 + abs(price_vs_200) * 2 + abs(slope_200) * 10))
    else:
        # Transitional
        if price_vs_200 > 0:
            stage = 2
            stage_name = "Advancing (Weak)"
            confidence = max(20, int(30 + price_vs_200))
        else:
            stage = 4
            stage_name = "Declining (Weak)"
            confidence = max(20, int(30 + abs(price_vs_200)))

    return {
        "stage": stage,
        "stage_name": stage_name,
        "confidence": min(99, max(10, confidence)),
        "price_vs_sma200_pct": round(price_vs_200, 2),
        "sma200_slope_pct": round(slope_200, 3),
        "volume_trend": round(vol_trend, 2),
    }


def detect_pocket_pivot(df: pd.DataFrame) -> bool:
    """
    Pocket Pivot (Gil Morales) — volume-based buying signal.
    Current up-day volume > max of last 10 down-day volumes.
    """
    if len(df) < 15:
        return False
    close = df["close"]
    vol = df["volume"]

    # current bar must be an up day
    if close.iloc[-1] <= close.iloc[-2]:
        return False

    curr_vol = vol.iloc[-1]

    # Get max volume of down days in last 10 bars
    last_10 = df.tail(11).head(10)  # exclude current bar
    down_days = last_10[last_10["close"] < last_10["close"].shift(1)]
    if len(down_days) == 0:
        return True  # No down days = strong

    max_down_vol = down_days["volume"].max()
    return bool(curr_vol > max_down_vol)


# ────────────────────────────────────────────────────────────────
# Composite Indicator Function
# ────────────────────────────────────────────────────────────────

def compute_all_indicators(df: pd.DataFrame, benchmark_df: pd.DataFrame | None = None) -> dict[str, Any]:
    close = df["close"]
    results: dict[str, Any] = {}

    # Moving averages
    results["sma_20"] = float(sma(close, 20).iloc[-1]) if len(close) >= 20 else None
    results["sma_50"] = float(sma(close, 50).iloc[-1]) if len(close) >= 50 else None
    results["sma_150"] = float(sma(close, 150).iloc[-1]) if len(close) >= 150 else None
    results["sma_200"] = float(sma(close, 200).iloc[-1]) if len(close) >= 200 else None
    results["ema_12"] = float(ema(close, 12).iloc[-1]) if len(close) >= 12 else None
    results["ema_26"] = float(ema(close, 26).iloc[-1]) if len(close) >= 26 else None
    results["ema_9"] = float(ema(close, 9).iloc[-1]) if len(close) >= 9 else None
    results["ema_21"] = float(ema(close, 21).iloc[-1]) if len(close) >= 21 else None

    # RSI
    results["rsi_14"] = float(rsi(close, 14).iloc[-1]) if len(close) >= 15 else None

    # Stochastic
    if len(df) >= 17:
        stoch = stochastic(df)
        results["stoch_k"] = float(stoch["k"].iloc[-1]) if not pd.isna(stoch["k"].iloc[-1]) else None
        results["stoch_d"] = float(stoch["d"].iloc[-1]) if not pd.isna(stoch["d"].iloc[-1]) else None
    else:
        results["stoch_k"] = results["stoch_d"] = None

    # Williams %R
    results["williams_r"] = float(williams_r(df).iloc[-1]) if len(df) >= 15 else None

    # CCI
    results["cci_20"] = float(cci(df, 20).iloc[-1]) if len(df) >= 21 else None

    # Rate of Change
    results["roc_12"] = float(roc(close, 12).iloc[-1]) if len(close) >= 13 else None

    # MACD
    if len(close) >= 26:
        m = macd(close)
        results["macd_line"] = float(m["macd"].iloc[-1])
        results["macd_signal"] = float(m["signal"].iloc[-1])
        results["macd_histogram"] = float(m["histogram"].iloc[-1])
    else:
        results["macd_line"] = results["macd_signal"] = results["macd_histogram"] = None

    # ADX with directional components
    if len(df) >= 28:
        adx_data = adx_components(df, 14)
        results["adx_14"] = float(adx_data["adx"].iloc[-1])
        results["plus_di"] = float(adx_data["plus_di"].iloc[-1])
        results["minus_di"] = float(adx_data["minus_di"].iloc[-1])
    else:
        results["adx_14"] = results["plus_di"] = results["minus_di"] = None

    # ATR
    results["atr_14"] = float(atr(df, 14).iloc[-1]) if len(df) >= 15 else None

    # Bollinger Bands + Bandwidth
    if len(close) >= 20:
        bb = bollinger_bands(close)
        results["bb_upper"] = float(bb["upper"].iloc[-1])
        results["bb_lower"] = float(bb["lower"].iloc[-1])
        results["bb_middle"] = float(bb["middle"].iloc[-1])
        bw = bollinger_bandwidth(close)
        results["bb_bandwidth"] = float(bw.iloc[-1]) if not pd.isna(bw.iloc[-1]) else None
    else:
        results["bb_upper"] = results["bb_lower"] = results["bb_middle"] = results["bb_bandwidth"] = None

    # Historical Volatility
    results["hist_volatility"] = float(historical_volatility(close, 20).iloc[-1]) if len(close) >= 21 else None

    # Volume indicators
    if "volume" in df.columns and len(df) >= 20:
        results["vol_ratio"] = float(volume_ratio(df["volume"]).iloc[-1])
        results["obv_trend"] = _obv_trend(df)
        results["mfi_14"] = float(mfi(df, 14).iloc[-1]) if len(df) >= 15 else None
    else:
        results["vol_ratio"] = None
        results["obv_trend"] = None
        results["mfi_14"] = None

    # Ichimoku summary
    if len(df) >= 52:
        ichi = ichimoku(df)
        ltp = float(close.iloc[-1])
        tenkan_val = float(ichi["tenkan"].iloc[-1]) if not pd.isna(ichi["tenkan"].iloc[-1]) else None
        kijun_val = float(ichi["kijun"].iloc[-1]) if not pd.isna(ichi["kijun"].iloc[-1]) else None
        # Cloud position
        sa = ichi["senkou_a"].iloc[-1] if not pd.isna(ichi["senkou_a"].iloc[-1]) else None
        sb = ichi["senkou_b"].iloc[-1] if not pd.isna(ichi["senkou_b"].iloc[-1]) else None
        if sa is not None and sb is not None:
            cloud_top = max(float(sa), float(sb))
            cloud_bot = min(float(sa), float(sb))
            if ltp > cloud_top:
                results["ichimoku_signal"] = "bullish"
            elif ltp < cloud_bot:
                results["ichimoku_signal"] = "bearish"
            else:
                results["ichimoku_signal"] = "neutral"
            results["ichimoku_cloud_top"] = round(cloud_top, 2)
            results["ichimoku_cloud_bot"] = round(cloud_bot, 2)
        else:
            results["ichimoku_signal"] = None
            results["ichimoku_cloud_top"] = results["ichimoku_cloud_bot"] = None
        results["ichimoku_tenkan"] = round(tenkan_val, 2) if tenkan_val else None
        results["ichimoku_kijun"] = round(kijun_val, 2) if kijun_val else None
    else:
        results["ichimoku_signal"] = results["ichimoku_tenkan"] = results["ichimoku_kijun"] = None
        results["ichimoku_cloud_top"] = results["ichimoku_cloud_bot"] = None

    # Linear regression slope
    results["lr_slope_20"] = float(linear_regression_slope(close, 20).iloc[-1]) if len(close) >= 20 else None

    # 52-week high/low
    hl = week52_high_low(close)
    results["high_52w"] = hl["high_52w"]
    results["low_52w"] = hl["low_52w"]

    # Price changes
    results["change_1d"] = price_change_pct(close, 1)
    results["change_5d"] = price_change_pct(close, 5) if len(close) >= 6 else 0.0
    results["change_20d"] = price_change_pct(close, 20) if len(close) >= 21 else 0.0
    results["change_60d"] = price_change_pct(close, 63) if len(close) >= 64 else 0.0
    results["change_120d"] = price_change_pct(close, 126) if len(close) >= 127 else 0.0
    results["change_250d"] = price_change_pct(close, 252) if len(close) >= 253 else 0.0

    ltp = float(close.iloc[-1])
    results["ltp"] = ltp
    results["dist_from_52w_high"] = ((ltp / hl["high_52w"]) - 1) * 100 if hl["high_52w"] > 0 else 0
    results["dist_from_52w_low"] = ((ltp / hl["low_52w"]) - 1) * 100 if hl["low_52w"] > 0 else 0

    # RS Rating (vs benchmark)
    if benchmark_df is not None and len(benchmark_df) > 20:
        results["rs_rating"] = rs_rating(close, benchmark_df["close"])
    else:
        results["rs_rating"] = None

    # Stage Analysis
    stage_info = detect_stage(df)
    results["stage"] = stage_info["stage"]
    results["stage_name"] = stage_info["stage_name"]
    results["stage_confidence"] = stage_info["confidence"]

    # VCP Detection
    vcp = detect_vcp(df)
    results["vcp_detected"] = vcp["detected"]
    results["vcp_tightness"] = vcp.get("tightness")
    results["vcp_pivot"] = vcp.get("pivot")

    # Pocket Pivot
    results["pocket_pivot"] = detect_pocket_pivot(df) if "volume" in df.columns and len(df) >= 15 else False

    # Pivot Points
    pp = pivot_points(df)
    results["pivot_pp"] = round(pp.get("pp", 0), 2) if pp else None
    results["pivot_r1"] = round(pp.get("r1", 0), 2) if pp else None
    results["pivot_s1"] = round(pp.get("s1", 0), 2) if pp else None

    return results


def _obv_trend(df: pd.DataFrame) -> str:
    """Determine OBV trend direction over last 20 bars."""
    if len(df) < 25:
        return "flat"
    obv_series = obv(df)
    obv_20 = obv_series.tail(20)
    if len(obv_20) < 20:
        return "flat"
    slope = np.polyfit(range(len(obv_20)), obv_20.values, 1)[0]
    avg_vol = df["volume"].tail(20).mean()
    if avg_vol == 0:
        return "flat"
    normalized = slope / avg_vol
    if normalized > 0.05:
        return "rising"
    elif normalized < -0.05:
        return "falling"
    return "flat"
