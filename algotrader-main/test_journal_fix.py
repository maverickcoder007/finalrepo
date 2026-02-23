#!/usr/bin/env python3
"""Quick test: verify journal recording works for backtest + paper trades."""
import ast
import os
import sys

# 1. Syntax check
files = [
    'src/api/service.py', 'src/data/paper_trader.py',
    'src/data/backtest.py', 'src/execution/engine.py',
]
for f in files:
    try:
        ast.parse(open(f).read())
        print(f'OK: {f}')
    except SyntaxError as e:
        print(f'SYNTAX ERROR: {f} line {e.lineno}: {e.msg}')
        sys.exit(1)

print()
print('--- Functional test ---')

from src.journal.journal_models import JournalEntry, ExecutionRecord, TradeDirection

# 2. Test backtest-style journal entry
exec_rec = ExecutionRecord(
    trade_id='BT-ema_crossover-0',
    instrument='RELIANCE',
    exchange='NSE',
    direction='LONG',
    quantity=10,
    actual_fill_price=2500.0,
    actual_exit_price=2550.0,
    expected_entry_price=2500.0,
    net_pnl=500.0,
    gross_pnl=502.5,
)
print('ExecutionRecord created OK')

entry = JournalEntry(
    trade_id='BT-ema_crossover-0',
    instrument='RELIANCE',
    tradingsymbol='RELIANCE',
    exchange='NSE',
    direction='LONG',
    is_closed=True,
    net_pnl=500.0,
    gross_pnl=502.5,
    total_costs=2.5,
    strategy_name='ema_crossover',
    source='backtest',
    entry_time='2025-01-15',
    exit_time='2025-01-16',
    mae=-1.5,
    mfe=3.2,
    execution=exec_rec.to_dict(),
    tags=['backtest', 'ema_crossover'],
    notes='Test backtest entry',
)
print('JournalEntry created OK')

d = entry.to_dict()
print('to_dict() OK')
print(f'  trade_id: {d["trade_id"]}')
print(f'  direction: {d["direction"]}')
print(f'  net_pnl: {d["net_pnl"]}')
print(f'  execution.actual_fill_price: {d["execution"]["actual_fill_price"]}')
print(f'  execution.quantity: {d["execution"]["quantity"]}')
print(f'  source: {d["source"]}')
print(f'  is_closed: {d["is_closed"]}')

# 3. Write to store and read back
from src.journal.journal_store import JournalStore

test_db = 'data/test_journal.db'
if os.path.exists(test_db):
    os.remove(test_db)
store = JournalStore(db_path=test_db)

eid = store.record_entry(entry)
print(f'  Stored! entry_id={eid}')

entries = store.query_entries(source='backtest')
print(f'  Read back: {len(entries)} entries')
if entries:
    e = entries[0]
    print(f'  trade_id={e.trade_id}, net_pnl={e.net_pnl}, source={e.source}')
    assert e.net_pnl == 500.0, f"PnL mismatch: {e.net_pnl}"
    assert e.source == 'backtest', f"Source mismatch: {e.source}"
    assert e.is_closed == True, f"is_closed mismatch: {e.is_closed}"
else:
    print("  ERROR: No entries read back!")
    sys.exit(1)

# 4. Test paper-trade-style entry
exec_rec2 = ExecutionRecord(
    trade_id='PT-rsi-0',
    instrument='INFY',
    exchange='NSE',
    direction='SHORT',
    quantity=5,
    actual_fill_price=1557.22,
    actual_exit_price=1559.06,
    expected_entry_price=1557.22,
    net_pnl=-9.20,
)
pt_entry = JournalEntry(
    trade_id='PT-rsi-0',
    instrument='INFY',
    tradingsymbol='INFY',
    exchange='NSE',
    direction='SHORT',
    is_closed=True,
    net_pnl=-9.20,
    strategy_name='rsi',
    source='paper_trade',
    entry_time='2026-02-22T10:30:00',
    exit_time='2026-02-22T11:15:00',
    execution=exec_rec2.to_dict(),
    tags=['paper_trade', 'rsi', 'stop_loss'],
    notes='Paper trade rsi on INFY | Exit: stop_loss',
)
eid2 = store.record_entry(pt_entry)
print(f'  Paper trade stored! entry_id={eid2}')

pt_entries = store.query_entries(source='paper_trade')
print(f'  Paper trades read back: {len(pt_entries)}')
assert len(pt_entries) == 1, f"Expected 1, got {len(pt_entries)}"

# Cleanup
os.remove(test_db)
print()
print('ALL TESTS PASSED')
