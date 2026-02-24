"""Quick integration test for all engines + synthetic data + timestamp preservation."""
import traceback, sys

def test_equity_backtest():
    from src.data.backtest import BacktestEngine
    from src.data.synthetic import generate_synthetic_ohlcv
    from src.strategy.ema_crossover import EMACrossoverStrategy

    s = EMACrossoverStrategy()
    df = generate_synthetic_ohlcv(200, 100.0, "D", style="equity")
    print(f"  Synthetic df: cols={list(df.columns)}, index_type={type(df.index).__name__}")

    e = BacktestEngine(strategy=s, initial_capital=100000)
    result = e.run(df, tradingsymbol="TEST", timeframe="day")

    trades = result.get("trades", [])
    entries = [t for t in trades if "ENTRY" in t.get("type", "")]
    exits = [t for t in trades if "EXIT" in t.get("type", "")]

    print(f"  total_trades={result.get('total_trades', 0)}, timeframe={result.get('timeframe')}")
    if entries:
        t0 = entries[0]
        print(f"  1st entry_time={t0.get('entry_time')}, target={t0.get('target', 'N/A')}")
    if exits:
        t0 = exits[0]
        print(f"  1st exit_time={t0.get('exit_time')}, exit_type={t0.get('type')}")

    # Verify timestamps are not integers
    for t in trades:
        for key in ("entry_time", "exit_time"):
            val = t.get(key)
            if val is not None:
                val_str = str(val)
                if val_str.isdigit():
                    raise AssertionError(f"Trade {key}={val_str} looks like an integer, not a date!")
    print("  [OK] Timestamps are valid dates")

    # Check TP stats exist
    tp = result.get("tp_exits", 0)
    print(f"  tp_exits={tp}")
    return True


def test_equity_with_zerodha_like_index():
    """Simulate what happens when Zerodha data has DatetimeIndex (no 'timestamp' col)."""
    import pandas as pd
    from src.data.backtest import BacktestEngine
    from src.data.synthetic import generate_synthetic_ohlcv
    from src.strategy.ema_crossover import EMACrossoverStrategy

    s = EMACrossoverStrategy()
    df = generate_synthetic_ohlcv(200, 100.0, "D", style="equity")

    # Simulate Zerodha: set timestamp as index
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    print(f"  Zerodha-like df: cols={list(df.columns)}, index_type={type(df.index).__name__}")

    e = BacktestEngine(strategy=s, initial_capital=100000)
    result = e.run(df, tradingsymbol="TEST_ZERODHA", timeframe="day")

    trades = result.get("trades", [])
    for t in trades:
        for key in ("entry_time", "exit_time"):
            val = t.get(key)
            if val is not None:
                val_str = str(val)
                if val_str.isdigit():
                    raise AssertionError(f"Zerodha-like: Trade {key}={val_str} is integer!")
    print(f"  total_trades={result.get('total_trades', 0)}")
    print("  [OK] Timestamps preserved from DatetimeIndex")
    return True


def test_paper_trade():
    from src.data.paper_trader import PaperTradingEngine
    from src.data.synthetic import generate_synthetic_ohlcv
    from src.strategy.ema_crossover import EMACrossoverStrategy

    s = EMACrossoverStrategy()
    df = generate_synthetic_ohlcv(200, 100.0, "D", style="equity")

    e = PaperTradingEngine(strategy=s, initial_capital=100000)
    result = e.run(df, tradingsymbol="TEST_PAPER", timeframe="day")
    print(f"  engine={result.get('engine')}, timeframe={result.get('timeframe')}")
    print(f"  total_trades={result.get('total_trades', 0)}")
    return True


def test_fno_backtest():
    from src.derivatives.fno_backtest import FnOBacktestEngine
    from src.data.synthetic import generate_synthetic_ohlcv

    df = generate_synthetic_ohlcv(100, 22000.0, "D", style="index")
    print(f"  FnO df: cols={list(df.columns)}, index_type={type(df.index).__name__}")

    e = FnOBacktestEngine(initial_capital=500000, underlying="NIFTY")
    result = e.run(df)

    trades = result.get("trades", [])
    bar_dates = result.get("bar_dates", [])
    print(f"  trades={len(trades)}, bar_dates={len(bar_dates)}")
    if bar_dates:
        print(f"  bar_dates[0]={bar_dates[0]}, bar_dates[-1]={bar_dates[-1]}")
        # Verify bar_dates are not integers
        for bd in bar_dates[:3]:
            if str(bd).isdigit():
                raise AssertionError(f"bar_date={bd} looks like an integer!")
    print("  [OK] FnO backtest ran")
    return True


def test_fno_paper_trade():
    from src.derivatives.fno_paper_trader import FnOPaperTradingEngine
    from src.data.synthetic import generate_synthetic_ohlcv

    df = generate_synthetic_ohlcv(100, 22000.0, "D", style="index")

    e = FnOPaperTradingEngine(initial_capital=500000, underlying="NIFTY")
    result = e.run(df)

    # result is FnOPaperTradeResult dataclass, not a dict
    result_dict = result.to_dict() if hasattr(result, "to_dict") else result
    print(f"  start_date={result_dict.get('start_date')}, end_date={result_dict.get('end_date')}")
    trades = result_dict.get("orders", [])
    print(f"  orders={len(trades)}, total_trades={result_dict.get('total_trades', 0)}")
    print("  [OK] FnO paper trade ran")
    return True


def test_synthetic_module():
    from src.data.synthetic import generate_synthetic_ohlcv
    import pandas as pd

    for freq in ("D", "5min", "15min"):
        df = generate_synthetic_ohlcv(50, 100.0, freq)
        assert isinstance(df, pd.DataFrame), f"Not a DataFrame for freq={freq}"
        assert "timestamp" in df.columns, f"No timestamp column for freq={freq}"
        assert len(df) == 50, f"Expected 50 rows, got {len(df)} for freq={freq}"
    print("  [OK] All freq modes produce valid DataFrames with timestamp column")
    return True


if __name__ == "__main__":
    tests = [
        ("Synthetic module", test_synthetic_module),
        ("Equity backtest", test_equity_backtest),
        ("Equity backtest (Zerodha-like index)", test_equity_with_zerodha_like_index),
        ("Paper trade", test_paper_trade),
        ("FnO backtest", test_fno_backtest),
        ("FnO paper trade", test_fno_paper_trade),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n=== {name} ===")
        try:
            fn()
            passed += 1
        except Exception:
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
