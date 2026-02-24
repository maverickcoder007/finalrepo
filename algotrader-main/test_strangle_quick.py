"""Quick focused diagnostic on short strangle mismatches."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd
from src.derivatives.fno_paper_trader import FnOPaperTradingEngine

np.random.seed(42)
days = 90; base = 25000
dates = pd.date_range("2025-01-01", periods=days, freq="B")
closes = [base]
for _ in range(days - 1):
    closes.append(closes[-1] * (1 + np.random.normal(0, 0.012)))
closes = np.array(closes)
df = pd.DataFrame({"open": closes*1.001, "high": closes*1.008, "low": closes*0.992, "close": closes, "volume": np.random.randint(100000, 500000, days)}, index=dates)

engine = FnOPaperTradingEngine(strategy_name="short_strangle", underlying="NIFTY", initial_capital=500000, max_positions=3, profit_target_pct=50.0, stop_loss_pct=100.0, entry_dte_min=7, entry_dte_max=45, delta_target=0.25)
result = engine.run(df, timeframe="day")

positions = result.positions or []
sum_pos_pnl = sum(p.get("pnl", 0) for p in positions)
capital_pnl = result.total_pnl
eq = result.equity_curve

print("=" * 60)
print("SHORT STRANGLE MISMATCH ANALYSIS")
print("=" * 60)
print(f"total_trades:                {result.total_trades}")
print(f"total_pnl (capital-based):   ₹{capital_pnl:.2f}")
print(f"sum of position PnLs:        ₹{sum_pos_pnl:.2f}")
print(f"total_costs:                 ₹{result.total_costs:.2f}")
print(f"Gap (pos_sum - capital_pnl): ₹{sum_pos_pnl - capital_pnl:.2f}")
print(f"Expected gap ≈ costs?        {abs(sum_pos_pnl - capital_pnl - result.total_costs) < 1}")
print()
print(f"Equity curve last value:     ₹{eq[-1]:.2f}")
print(f"Final capital:               ₹{result.final_capital:.2f}")
print(f"Equity ≠ Capital MISMATCH:   {abs(eq[-1] - result.final_capital) > 1}")
print()

print("=" * 60)
print("NET_PREMIUM vs PNL SCALE CHECK (first 5 positions)")
print("=" * 60)
print(f"{'#':>3} {'premium':>10} {'pnl':>12} {'ratio':>8}  note")
print("-" * 60)
for i, pos in enumerate(positions[:10]):
    np_val = pos.get("net_premium", 0)
    pnl_val = pos.get("pnl", 0)
    ratio = pnl_val / np_val if np_val else 0
    note = "SCALE MISMATCH (pnl ~65x premium)" if abs(ratio) > 20 else ""
    print(f"{i+1:>3} {np_val:>10.2f} {pnl_val:>12.2f} {ratio:>8.1f}x  {note}")

print()
print("=" * 60)
print("LOTS / MAX_LOSS / MAX_PROFIT CHECK (first 5 positions)")
print("=" * 60)
for i, pos in enumerate(engine.closed_positions[:5]):
    lots = [leg.lots for leg in pos.legs]
    print(f"Pos #{i+1}: legs_lots={lots}, total_lots={pos.total_lots}, "
          f"max_loss={pos.max_loss:.2f}, max_profit={pos.max_profit:.2f}")
    for j, leg in enumerate(pos.legs):
        print(f"  Leg {j+1}: qty={leg.quantity}, lot_size={leg.contract.lot_size}, lots_prop={leg.lots}")

print()
print("=" * 60)
print("PREMIUM IN RUPEES vs POINTS")
print("=" * 60)
for i, pos in enumerate(engine.closed_positions[:5]):
    lot_size = pos.legs[0].contract.lot_size
    premium_points = pos.net_premium
    premium_rupees = premium_points * lot_size
    pos_pnl = positions[i].get("pnl", 0)
    print(f"Pos #{i+1}: premium_points={premium_points:.2f}, "
          f"premium_₹={premium_rupees:.2f}, pnl_₹={pos_pnl:.2f}, "
          f"lot_size={lot_size}")
    print(f"  → Dashboard shows premium={premium_points:.2f} next to pnl={pos_pnl:.2f} — "
          f"{'UNITS MISMATCH!' if abs(pos_pnl) > premium_points * 5 else 'OK'}")
