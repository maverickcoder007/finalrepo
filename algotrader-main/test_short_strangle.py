"""Diagnostic: Short Strangle Paper Trade — trace every mismatch."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.derivatives.fno_paper_trader import FnOPaperTradingEngine
from src.derivatives.contracts import StructureType

# ── Build synthetic OHLCV data ──
np.random.seed(42)
days = 90
base = 25000
dates = pd.date_range("2025-01-01", periods=days, freq="B")
closes = [base]
for _ in range(days - 1):
    closes.append(closes[-1] * (1 + np.random.normal(0, 0.012)))
closes = np.array(closes)

df = pd.DataFrame({
    "open": closes * (1 + np.random.uniform(-0.003, 0.003, days)),
    "high": closes * (1 + np.random.uniform(0.002, 0.015, days)),
    "low": closes * (1 - np.random.uniform(0.002, 0.015, days)),
    "close": closes,
    "volume": np.random.randint(100_000, 500_000, days),
}, index=dates)

# ── Run Short Strangle Paper Trade ──
engine = FnOPaperTradingEngine(
    strategy_name="short_strangle",
    underlying="NIFTY",
    initial_capital=500_000,
    max_positions=3,
    profit_target_pct=50.0,
    stop_loss_pct=100.0,
    entry_dte_min=7,
    entry_dte_max=45,
    delta_target=0.25,
)

print("=" * 80)
print("SHORT STRANGLE PAPER TRADE DIAGNOSTIC")
print("=" * 80)
print(f"Strategy: {engine.strategy_name}")
print(f"Structure: {engine.structure_type}")
print(f"Initial Capital: {engine.initial_capital}")
print(f"Data: {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")
print()

result = engine.run(df, timeframe="day")

# ── Summary metrics ──
print("─" * 60)
print("SUMMARY METRICS (from result)")
print("─" * 60)
print(f"  total_trades:      {result.total_trades}")
print(f"  winning_trades:    {result.winning_trades}")
print(f"  losing_trades:     {result.losing_trades}")
print(f"  win_rate:          {result.win_rate:.1f}%")
print(f"  total_pnl:         ₹{result.total_pnl:.2f}")
print(f"  total_return_pct:  {result.total_return_pct:.2f}%")
print(f"  initial_capital:   ₹{result.initial_capital:.2f}")
print(f"  final_capital:     ₹{result.final_capital:.2f}")
print(f"  total_costs:       ₹{result.total_costs:.2f}")
print(f"  max_drawdown_pct:  {result.max_drawdown_pct:.2f}%")
print(f"  sharpe_ratio:      {result.sharpe_ratio:.4f}")
print()

# ── Capital-based vs position-based PnL ──
positions = result.positions or []
sum_pos_pnl = sum(p.get("pnl", 0) for p in positions)
capital_pnl = result.total_pnl  # = final_capital - initial_capital

print("─" * 60)
print("MISMATCH CHECK #1: Capital PnL vs Sum of Position PnLs")
print("─" * 60)
print(f"  Capital-based PnL (final - initial):  ₹{capital_pnl:.2f}")
print(f"  Sum of position PnLs:                 ₹{sum_pos_pnl:.2f}")
print(f"  Total costs:                          ₹{result.total_costs:.2f}")
print(f"  Expected diff (costs):                ₹{sum_pos_pnl - capital_pnl:.2f}")
print(f"  MATCH? {abs(sum_pos_pnl - capital_pnl - result.total_costs) < 1:.5f}")
print()

# ── Per-position details ──
print("─" * 60)
print("PER-POSITION DETAILS")
print("─" * 60)
for i, pos in enumerate(positions):
    print(f"\n  Position #{i+1}: {pos.get('id')} | {pos.get('structure')}")
    print(f"    legs:        {pos.get('legs')}")
    print(f"    net_premium: ₹{pos.get('net_premium', 0):.2f}")
    print(f"    pnl:         ₹{pos.get('pnl', 0):.2f}")
    print(f"    entry_time:  {pos.get('entry_time')}")
    print(f"    exit_time:   {pos.get('exit_time')}")
    print(f"    exit_reason: {pos.get('exit_reason')}")
    print(f"    qty:         {pos.get('qty')}")
    print(f"    regime:      {pos.get('regime')}")

    # Verify net_premium sign
    np_val = pos.get('net_premium', 0)
    if np_val > 0:
        print(f"    ✓ net_premium is POSITIVE (credit) — correct for short strangle")
    elif np_val < 0:
        print(f"    ✗ net_premium is NEGATIVE (debit) — WRONG for short strangle!")
    else:
        print(f"    ✗ net_premium is ZERO — unexpected")

print()

# ── Detailed leg inspection from closed positions ──
print("─" * 60)
print("DETAILED LEG INSPECTION (from engine.closed_positions)")
print("─" * 60)
for i, pos in enumerate(engine.closed_positions):
    print(f"\n  Position #{i+1}: {pos.position_id} | {pos.structure.value}")
    print(f"    is_closed:     {pos.is_closed}")
    print(f"    net_premium:   ₹{pos.net_premium:.2f} ({'credit' if pos.net_premium > 0 else 'debit'})")
    print(f"    max_profit:    ₹{pos.max_profit:.2f}")
    print(f"    max_loss:      ₹{pos.max_loss:.2f}")
    print(f"    exit_reason:   {pos.exit_reason}")
    print(f"    exit_time:     {pos.exit_time}")

    leg_pnl_sum_build_result = 0
    leg_pnl_sum_realised = 0

    for j, leg in enumerate(pos.legs):
        direction = "SHORT" if leg.quantity < 0 else "LONG"
        opt_type = "CE" if leg.contract.is_call else "PE"
        print(f"\n    Leg {j+1}: {direction} {opt_type} strike={leg.contract.strike}")
        print(f"      quantity:    {leg.quantity}")
        print(f"      entry_price: ₹{leg.entry_price:.2f}")
        print(f"      exit_price:  ₹{leg.exit_price:.2f}")
        print(f"      is_closed:   {leg.is_closed}")
        print(f"      lot_size:    {leg.contract.lot_size}")

        # Method 1: _build_result formula
        if leg.is_closed:
            m1 = (leg.exit_price - leg.entry_price) * leg.quantity * leg.contract.lot_size
        else:
            m1 = 0
        leg_pnl_sum_build_result += m1

        # Method 2: realised_pnl (no lot_size!)
        m2 = leg.realised_pnl  # (exit - entry) * |qty| for long, (entry - exit) * |qty| for short
        leg_pnl_sum_realised += m2

        # Method 3: manual calc matching _close_position
        if leg.is_closed:
            if leg.quantity > 0:
                m3 = (leg.exit_price - leg.entry_price) * abs(leg.quantity) * leg.contract.lot_size
            else:
                m3 = (leg.entry_price - leg.exit_price) * abs(leg.quantity) * leg.contract.lot_size
        else:
            m3 = 0

        print(f"      PnL (_build_result formula): ₹{m1:.2f}")
        print(f"      PnL (realised_pnl prop):     ₹{m2:.2f}  [NO lot_size!]")
        print(f"      PnL (_close_position calc):   ₹{m3:.2f}")

        if leg.quantity < 0:
            print(f"      ✓ This is a SHORT leg (correct for short strangle)")
        else:
            print(f"      ✗ This is a LONG leg — WRONG for short strangle!")

    print(f"\n    Σ PnL (_build_result):    ₹{leg_pnl_sum_build_result:.2f}")
    print(f"    Σ PnL (realised_pnl):     ₹{leg_pnl_sum_realised:.2f}  [NO lot_size]")
    print(f"    Position dict pnl:        ₹{positions[i].get('pnl', 0):.2f}")
    match_ok = abs(leg_pnl_sum_build_result - positions[i].get('pnl', 0)) < 0.01
    print(f"    MATCH? {match_ok}")

print()

# ── Check expiry-related orders ──
print("─" * 60)
print("EXPIRY ORDERS")
print("─" * 60)
expiry_orders = [o for o in result.orders if o.get("side") == "EXPIRY"]
print(f"  Total expiry orders: {len(expiry_orders)}")
for o in expiry_orders[:10]:
    print(f"    {o.get('tradingsymbol')} | status={o.get('status')} | "
          f"price={o.get('price')} | fill={o.get('fill_price')} | "
          f"pos={o.get('position_id')}")

print()

# ── Check if equity curve matches final capital ──
print("─" * 60)
print("EQUITY CURVE VALIDATION")
print("─" * 60)
eq = result.equity_curve
if eq:
    print(f"  First value:  ₹{eq[0]:.2f}")
    print(f"  Last value:   ₹{eq[-1]:.2f}")
    print(f"  Final capital: ₹{result.final_capital:.2f}")
    print(f"  MATCH? {abs(eq[-1] - result.final_capital) < 1}")
print()

# ── Check for win/loss count mismatch ──
print("─" * 60)
print("WIN/LOSS VERIFICATION")
print("─" * 60)
actual_wins = sum(1 for p in positions if p.get('pnl', 0) > 0)
actual_losses = sum(1 for p in positions if p.get('pnl', 0) <= 0)
print(f"  Result winning_trades: {result.winning_trades}")
print(f"  Actual wins from pos:  {actual_wins}")
print(f"  Result losing_trades:  {result.losing_trades}")
print(f"  Actual losses from pos: {actual_losses}")
print(f"  MATCH? wins={result.winning_trades == actual_wins}, losses={result.losing_trades == actual_losses}")
print()

# ── Check exit reason distribution ──
from collections import Counter
reasons = Counter(p.get('exit_reason', 'unknown') for p in positions)
print("─" * 60)
print("EXIT REASON DISTRIBUTION")
print("─" * 60)
for reason, count in reasons.most_common():
    pnl_for_reason = sum(p.get('pnl', 0) for p in positions if p.get('exit_reason') == reason)
    print(f"  {reason:20s}: {count:3d} trades, total PnL ₹{pnl_for_reason:.2f}")

print()
print("=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
