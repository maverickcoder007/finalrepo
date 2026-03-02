"""Diagnose health report score for call credit spread runner."""
import asyncio
import pandas as pd
import numpy as np
from src.derivatives.fno_backtest import FnOBacktestEngine
from src.strategy.health_engine import compute_health_report

async def test():
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
    np.random.seed(42)
    prices = 22000 + np.cumsum(np.random.randn(50) * 50)
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices - 10,
        'high': prices + 30,
        'low': prices - 30,
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 50)
    })

    engine = FnOBacktestEngine(
        strategy_name='call_credit_spread_runner',
        underlying='NIFTY',
        initial_capital=500000,
    )
    result = engine.run(df, tradingsymbol='NIFTY')
    
    print('=== BACKTEST RESULT ===')
    print(f'Total Trades: {result.get("total_trades", 0)}')
    print(f'Win Rate: {result.get("win_rate", 0):.2f}%')
    print(f'Total PnL: {result.get("total_pnl", 0):.2f}')
    print(f'Max Drawdown: {result.get("max_drawdown", 0):.2f}%')
    print(f'Profit Factor: {result.get("profit_factor", 0):.2f}')
    print(f'Trades: {len(result.get("trades", []))}')
    
    # Show detailed trade list
    trades = result.get("trades", [])
    print(f'\n=== TRADE DETAILS ({len(trades)} trades) ===')
    for i, t in enumerate(trades[:10]):  # Show first 10
        print(f'  {i+1}. PnL={t.get("pnl", 0):.2f}, Exit={t.get("exit_reason", "?")}')
    
    # Compute health report
    health = compute_health_report(result, strategy_type='fno', strategy_name='call_credit_spread_runner')
    
    print('\n=== HEALTH REPORT ===')
    print(f'Overall Score: {health.overall_score:.1f}')
    print(f'Overall Verdict: {health.overall_verdict}')
    print(f'Execution Ready: {health.execution_ready}')
    print('\n--- PILLAR SCORES ---')
    for name, pillar in health.pillars.items():
        print(f'{name:20s}: {pillar.score:5.1f} ({pillar.verdict:4s})')
        for note in pillar.notes:
            print(f'    > {note}')
        for k, v in pillar.metrics.items():
            if isinstance(v, float):
                print(f'    - {k}: {v:.3f}')
            else:
                print(f'    - {k}: {v}')
    
    print('\n--- BLOCKERS ---')
    for b in health.blockers:
        print(f'  X {b}')
    print('\n--- WARNINGS ---')
    for w in health.warnings:
        print(f'  ! {w}')
    print('\n--- SUMMARY ---')
    for s in health.summary:
        print(f'  * {s}')

if __name__ == "__main__":
    asyncio.run(test())
