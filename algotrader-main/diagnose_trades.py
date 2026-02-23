#!/usr/bin/env python3
"""Diagnose: why so few trades in backtest & paper trade with sample data?"""
import numpy as np
import pandas as pd
import sys

# ── 1. Generate the SAME sample data that run_backtest_sample produces (NEW) ──
np.random.seed(42)
bars = 500
dates = pd.date_range(end=pd.Timestamp.now(), periods=bars, freq="D")
base = 100.0

# Realistic trending regime data
prices = np.zeros(bars)
prices[0] = base
regime_length = max(20, bars // 10)
trend = 0.0
volatility = 0.02
for i in range(1, bars):
    if i % regime_length == 0:
        trend = np.random.choice([-0.003, -0.001, 0.0, 0.001, 0.003])
        volatility = np.random.choice([0.015, 0.02, 0.03])
    noise = np.random.normal(trend, volatility)
    prices[i] = prices[i - 1] * (1 + noise)

# Volume with spikes
base_vol = np.random.randint(500_000, 2_000_000, bars).astype(float)
spike_mask = np.random.random(bars) < 0.20
base_vol[spike_mask] *= np.random.uniform(1.5, 3.0, spike_mask.sum())
volumes_bt = base_vol.astype(int)

df_bt = pd.DataFrame({
    "timestamp": dates,
    "open": prices * (1 + np.random.uniform(-0.008, 0.008, bars)),
    "high": prices * (1 + np.abs(np.random.normal(0, 0.012, bars))),
    "low": prices * (1 - np.abs(np.random.normal(0, 0.012, bars))),
    "close": prices,
    "volume": volumes_bt,
}, index=dates)

# ── 2. Generate same paper trade sample data (NEW) ──
np.random.seed(42)
dates2 = pd.date_range("2024-01-01", periods=bars, freq="5min")
price = 100.0
opens, highs, lows, closes, volumes = [], [], [], [], []
regime_length_pt = max(20, bars // 10)
trend_pt = 0.0
vol_scale = 0.5
for i in range(bars):
    if i % regime_length_pt == 0:
        trend_pt = np.random.choice([-0.15, -0.05, 0.0, 0.05, 0.15])
        vol_scale = np.random.choice([0.3, 0.5, 0.8])
    change = np.random.normal(trend_pt, vol_scale)
    o = price
    c = price + change
    h = max(o, c) + abs(np.random.normal(0, vol_scale * 0.4))
    l = min(o, c) - abs(np.random.normal(0, vol_scale * 0.4))
    v = int(np.random.uniform(20000, 80000))
    if np.random.random() < 0.15:
        v = int(v * np.random.uniform(2.0, 4.0))
    opens.append(o); highs.append(h); lows.append(l); closes.append(c); volumes.append(v)
    price = c
df_pt = pd.DataFrame({
    "timestamp": dates2, "open": opens, "high": highs,
    "low": lows, "close": closes, "volume": volumes,
})

print("=" * 60)
print("BACKTEST SAMPLE DATA (500 daily bars)")
print(f"  Price range: {df_bt['close'].min():.2f} — {df_bt['close'].max():.2f}")
print(f"  close[-1]: {df_bt['close'].iloc[-1]:.2f}")
print(f"  std(close): {df_bt['close'].std():.4f}")
print(f"  Volume range: {df_bt['volume'].min()} — {df_bt['volume'].max()}")
print()
print("PAPER TRADE SAMPLE DATA (500 5min bars)")
print(f"  Price range: {df_pt['close'].min():.2f} — {df_pt['close'].max():.2f}")
print(f"  close[-1]: {df_pt['close'].iloc[-1]:.2f}")
print(f"  std(close): {df_pt['close'].std():.4f}")
print(f"  Volume range: {df_pt['volume'].min()} — {df_pt['volume'].max()}")

# ── 3. Run each strategy and count signals ──
from src.strategy.ema_crossover import EMACrossoverStrategy
from src.strategy.rsi_strategy import RSIStrategy
from src.strategy.mean_reversion import MeanReversionStrategy
from src.strategy.vwap_breakout import VWAPBreakoutStrategy
from src.analysis.indicators import adx, atr

strategies = {
    "ema_crossover": EMACrossoverStrategy(),
    "rsi_strategy": RSIStrategy(),
    "mean_reversion": MeanReversionStrategy(),
    "vwap_breakout": VWAPBreakoutStrategy(),
}

for sname, strat in strategies.items():
    print("\n" + "=" * 60)
    print(f"STRATEGY: {sname}")
    print(f"  Params: {strat.params}")

    # Count signals on BACKTEST data
    signal_count = 0
    buy_count = 0
    sell_count = 0
    rejected_reasons = {}
    for i in range(1, len(df_bt)):
        slice_df = df_bt.iloc[:i+1].copy()
        strat.update_bar_data(0, slice_df)
        sig = strat.generate_signal(slice_df, 0)
        if sig:
            signal_count += 1
            if sig.transaction_type.value == "BUY":
                buy_count += 1
            else:
                sell_count += 1

    print(f"  [BT] Signals: {signal_count} (BUY: {buy_count}, SELL: {sell_count}) / {len(df_bt)} bars")

    # Show why signals might be blocked for the strategy
    if sname == "ema_crossover":
        # Check ADX values
        if len(df_bt) >= 28:
            adx_vals = adx(df_bt, 14)
            above_threshold = (adx_vals >= strat.params["min_adx"]).sum()
            print(f"  ADX >= {strat.params['min_adx']}: {above_threshold}/{len(adx_vals)} bars")
            print(f"  ADX range: {adx_vals.dropna().min():.2f} — {adx_vals.dropna().max():.2f}, median: {adx_vals.dropna().median():.2f}")
        # Check EMA crossovers
        fast_ema = df_bt["close"].ewm(span=strat.params["fast_period"], adjust=False).mean()
        slow_ema = df_bt["close"].ewm(span=strat.params["slow_period"], adjust=False).mean()
        crossovers = 0
        for j in range(1, len(fast_ema)):
            if (fast_ema.iloc[j-1] <= slow_ema.iloc[j-1] and fast_ema.iloc[j] > slow_ema.iloc[j]) or \
               (fast_ema.iloc[j-1] >= slow_ema.iloc[j-1] and fast_ema.iloc[j] < slow_ema.iloc[j]):
                crossovers += 1
        print(f"  Raw EMA crossovers (before filters): {crossovers}")
        # Check volume
        avg_vol = df_bt["volume"].rolling(20).mean()
        vol_ratio = df_bt["volume"] / avg_vol
        above_vol = (vol_ratio >= strat.params["min_volume_ratio"]).sum()
        print(f"  Volume ratio >= {strat.params['min_volume_ratio']}: {above_vol}/{len(vol_ratio)} bars")

    elif sname == "rsi_strategy":
        rsi = RSIStrategy.compute_rsi(df_bt["close"], strat.params["rsi_period"])
        rsi_ok = rsi.dropna()
        below_30 = (rsi_ok <= strat.params["oversold"]).sum()
        above_70 = (rsi_ok >= strat.params["overbought"]).sum()
        # Count crossings
        crossings = 0
        for j in range(1, len(rsi_ok)):
            if (rsi_ok.iloc[j-1] <= strat.params["oversold"] and rsi_ok.iloc[j] > strat.params["oversold"]) or \
               (rsi_ok.iloc[j-1] >= strat.params["overbought"] and rsi_ok.iloc[j] < strat.params["overbought"]):
                crossings += 1
        print(f"  RSI range: {rsi_ok.min():.2f} — {rsi_ok.max():.2f}, median: {rsi_ok.median():.2f}")
        print(f"  Bars RSI <= {strat.params['oversold']}: {below_30}")
        print(f"  Bars RSI >= {strat.params['overbought']}: {above_70}")
        print(f"  RSI crossings (entry signals): {crossings}")

    elif sname == "mean_reversion":
        lookback = strat.params["lookback_period"]
        z_scores = []
        for j in range(lookback, len(df_bt)):
            window = df_bt["close"].iloc[j-lookback:j+1]
            mean = window.mean()
            std = window.std()
            if std > 0:
                z_scores.append((window.iloc[-1] - mean) / std)
        z_arr = np.array(z_scores)
        z_entry = strat.params["z_score_entry"]
        extreme = ((z_arr > z_entry) | (z_arr < -z_entry)).sum()
        print(f"  Z-score range: {z_arr.min():.2f} — {z_arr.max():.2f}, std: {z_arr.std():.2f}")
        print(f"  |Z| >= {z_entry}: {extreme}/{len(z_arr)} bars")

    elif sname == "vwap_breakout":
        cum_vol = df_bt["volume"].cumsum()
        cum_vp = (df_bt["close"] * df_bt["volume"]).cumsum()
        vwap = cum_vp / cum_vol
        distance_pct = ((df_bt["close"] - vwap) / vwap) * 100
        threshold = strat.params["breakout_threshold"]
        above = (distance_pct.abs() > threshold).sum()
        avg_vol = df_bt["volume"].rolling(20).mean()
        vol_ratio = df_bt["volume"] / avg_vol
        vol_ok = (vol_ratio >= strat.params["min_volume_ratio"]).sum()
        print(f"  VWAP distance range: {distance_pct.min():.2f}% — {distance_pct.max():.2f}%")
        print(f"  |dist| > {threshold}%: {above} bars")
        print(f"  Volume ratio >= {strat.params['min_volume_ratio']}: {vol_ok} bars")

    # Reset strategy state
    strat._prev_signal.clear()

    # Count signals on PAPER TRADE data
    signal_count_pt = 0
    buy_count_pt = 0
    sell_count_pt = 0
    for i in range(1, len(df_pt)):
        slice_df = df_pt.iloc[:i+1].copy()
        strat.update_bar_data(0, slice_df)
        sig = strat.generate_signal(slice_df, 0)
        if sig:
            signal_count_pt += 1
            if sig.transaction_type.value == "BUY":
                buy_count_pt += 1
            else:
                sell_count_pt += 1
    print(f"  [PT] Signals: {signal_count_pt} (BUY: {buy_count_pt}, SELL: {sell_count_pt}) / {len(df_pt)} bars")

# ── 4. Run actual backtest engine and paper trade engine ──
print("\n" + "=" * 60)
print("ACTUAL BACKTEST ENGINE RUN")
from src.data.backtest import BacktestEngine
for sname in ["ema_crossover", "rsi_strategy", "mean_reversion", "vwap_breakout"]:
    strat = strategies[sname].__class__()  # fresh instance
    engine = BacktestEngine(strategy=strat, initial_capital=100000)
    result = engine.run(df_bt.copy(), tradingsymbol="SAMPLE")
    if isinstance(result, dict):
        trades = result.get("trades", [])
        entries = [t for t in trades if t.get("type", "").endswith("_ENTRY")]
        exits = [t for t in trades if "EXIT" in t.get("type", "")]
        print(f"  {sname}: {len(entries)} entries, {len(exits)} exits, total trades: {result.get('total_trades', 0)}")
    else:
        print(f"  {sname}: ERROR - {result}")

print("\n" + "=" * 60)
print("ACTUAL PAPER TRADE ENGINE RUN")
from src.data.paper_trader import PaperTradingEngine
for sname in ["ema_crossover", "rsi_strategy", "mean_reversion", "vwap_breakout"]:
    strat = strategies[sname].__class__()  # fresh instance
    engine = PaperTradingEngine(strategy=strat, initial_capital=100000)
    result = engine.run(df_pt.copy(), instrument_token=0, tradingsymbol="SAMPLE", timeframe="5min")
    d = result.to_dict_safe()
    print(f"  {sname}: {d['total_trades']} trades, {d['winning_trades']} wins, {d['losing_trades']} losses, pnl: {d['total_pnl']:.2f}")

print("\nDONE")
