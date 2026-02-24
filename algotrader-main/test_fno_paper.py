"""Quick diagnostic for FnO Paper Trading iron condor."""
import json
import sys
import traceback

try:
    from src.derivatives.fno_paper_trader import FnOPaperTradingEngine
    from src.data.synthetic import generate_synthetic_ohlcv

    print("=== Generating synthetic data (100 bars) ===")
    df = generate_synthetic_ohlcv(bars=100, base_price=20000.0, freq="D", style="index")
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Index type: {type(df.index)}")
    print(f"First few rows:\n{df.head(3)}")
    print()

    print("=== Creating FnO Paper Trading Engine ===")
    engine = FnOPaperTradingEngine(
        strategy_name="iron_condor",
        underlying="NIFTY",
        initial_capital=500_000.0,
        max_positions=3,
        profit_target_pct=50.0,
        stop_loss_pct=100.0,
        delta_target=0.16,
    )

    print("=== Running paper trade ===")
    result = engine.run(df, tradingsymbol="NIFTY", timeframe="day")
    rd = result.to_dict_safe()

    print(f"\n=== RESULTS ===")
    print(f"Engine: {rd.get('engine')}")
    print(f"Total Trades: {rd.get('total_trades')}")
    print(f"Positions: {len(rd.get('positions', []))}")
    print(f"Orders: {len(rd.get('orders', []))}")
    print(f"Equity curve length: {len(rd.get('equity_curve', []))}")
    print(f"Start: {rd.get('start_date')}")
    print(f"End: {rd.get('end_date')}")
    print(f"PnL: {rd.get('total_pnl')}")
    print(f"Return: {rd.get('total_return_pct')}")
    print(f"Margin History: {len(rd.get('margin_history', []))}")
    print(f"Greeks History: {len(rd.get('greeks_history', []))}")

    if rd.get('positions'):
        print(f"\n--- FIRST POSITION ---")
        print(json.dumps(rd['positions'][0], indent=2, default=str))
    else:
        print("\n!!! NO POSITIONS — engine did not open any trades!")

    if rd.get('orders'):
        print(f"\n--- FIRST ORDER ---")
        print(json.dumps(rd['orders'][0], indent=2, default=str))
    else:
        print("\n!!! NO ORDERS")

    if rd.get('margin_history'):
        print(f"\n--- FIRST MARGIN ---")
        print(json.dumps(rd['margin_history'][0], indent=2, default=str))

    # Check the trade log data that the dashboard would render
    print("\n=== TRADE LOG CHECK (what dashboard would show) ===")
    positions = rd.get('positions', [])
    for i, p in enumerate(positions):
        et = p.get('exit_time') or p.get('exit') or '--'
        er = p.get('exit_reason') or p.get('reason') or '--'
        bug = " *** BUG: no exit_time!" if str(et) in ('None', '--', 'unknown', '') else ""
        print(f"  Trade {i+1}: PnL={p.get('pnl','--')}  Entry={str(p.get('entry_time','--'))[:10]}  Exit={str(et)[:19]}  Reason={er}{bug}")

    if rd.get('_warnings'):
        print(f"\n!!! WARNINGS: {rd['_warnings']}")

    # Check if profit target / stop loss ever triggered
    reasons = [p.get('exit_reason') or p.get('reason') for p in positions]
    from collections import Counter
    print(f"\nExit reasons: {dict(Counter(reasons))}")

    # Check if positions only expire - test with higher max_positions
    print("\n=== RE-RUN with 500 bars for more trades ===")
    df2 = generate_synthetic_ohlcv(bars=500, base_price=20000.0, freq="D", style="index")
    engine2 = FnOPaperTradingEngine(
        strategy_name="iron_condor", underlying="NIFTY",
        initial_capital=500_000.0, max_positions=3,
        profit_target_pct=50.0, stop_loss_pct=100.0, delta_target=0.16,
    )
    result2 = engine2.run(df2, tradingsymbol="NIFTY", timeframe="day")
    rd2 = result2.to_dict_safe()
    reasons2 = [p.get('exit_reason') or p.get('reason') for p in rd2.get('positions', [])]
    print(f"Total Trades: {rd2.get('total_trades')}")
    print(f"Exit reasons: {dict(Counter(reasons2))}")
    print(f"PnL: {rd2.get('total_pnl')}")
    pos2 = rd2.get('positions', [])
    # Show a few positions
    for i, p in enumerate(pos2[:5]):
        et = p.get('exit_time') or '--'
        er = p.get('exit_reason') or p.get('reason') or '--'
        print(f"  Trade {i+1}: PnL={p.get('pnl')}  Entry={str(p.get('entry_time',''))[:10]}  Exit={str(et)[:19]}  Reason={er}")

    print("\n=== DONE ===")

except Exception as e:
    print(f"FATAL ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)
