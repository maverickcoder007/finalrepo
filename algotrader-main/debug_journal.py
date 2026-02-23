#!/usr/bin/env python3
"""Debug: check stored journal entries and API round-trip."""
import sqlite3, json, sys

conn = sqlite3.connect('data/journal.db')
conn.row_factory = sqlite3.Row

# 1. Check raw stored data
cur = conn.cursor()
cur.execute('SELECT data FROM journal_entries LIMIT 1')
row = cur.fetchone()
if not row:
    print('No entries in DB!'); sys.exit(1)

d = json.loads(row['data'])
print('=== Stored entry keys ===')
print(sorted(d.keys()))
print(f"entry_time: {d.get('entry_time')}")
print(f"exit_time: {d.get('exit_time')}")
print(f"is_closed: {d.get('is_closed')}")
print(f"source: {d.get('source')}")
print(f"net_pnl: {d.get('net_pnl')}")
exec_data = d.get('execution', {})
print(f"execution type: {type(exec_data).__name__}")
if isinstance(exec_data, dict):
    print(f"  actual_fill_price: {exec_data.get('actual_fill_price')}")
    print(f"  actual_exit_price: {exec_data.get('actual_exit_price')}")
    print(f"  quantity: {exec_data.get('quantity')}")

# 2. Test from_dict round-trip
print('\n=== Round-trip test ===')
from src.journal.journal_models import JournalEntry
try:
    entry = JournalEntry.from_dict(d)
    print('from_dict: OK')
    d2 = entry.to_dict()
    print('to_dict: OK')
    print(f"  entry_time: {d2.get('entry_time')}")
    print(f"  net_pnl: {d2.get('net_pnl')}")
    print(f"  execution type: {type(d2.get('execution')).__name__}")
except Exception as e:
    print(f'ROUND-TRIP ERROR: {e}')
    import traceback; traceback.print_exc()

# 3. Test full query_entries flow
print('\n=== query_entries test ===')
from src.journal.journal_store import JournalStore
store = JournalStore(db_path='data/journal.db')
try:
    entries = store.query_entries(limit=5)
    print(f'query_entries returned: {len(entries)} entries')
    for e in entries:
        d = e.to_dict()
        print(f"  {d.get('trade_id')}: pnl={d.get('net_pnl')}, source={d.get('source')}, entry_time={d.get('entry_time')}")
except Exception as ex:
    print(f'QUERY ERROR: {ex}')
    import traceback; traceback.print_exc()

# 4. Test get_summary
print('\n=== get_summary test ===')
try:
    summary = store.get_summary()
    print(f"total_trades: {summary.get('total_trades')}")
    print(f"total_pnl: {summary.get('total_pnl')}")
    print(f"win_rate: {summary.get('win_rate')}")
except Exception as ex:
    print(f'SUMMARY ERROR: {ex}')
    import traceback; traceback.print_exc()

# 5. Simulate what the entries API returns
print('\n=== Simulated API response ===')
try:
    entries = store.query_entries(limit=3)
    api_response = {"entries": [e.to_dict() for e in entries], "total": len(entries)}
    print(f"entries count: {len(api_response['entries'])}")
    if api_response['entries']:
        e = api_response['entries'][0]
        print(f"  trade_id: {e.get('trade_id')}")
        print(f"  execution.actual_fill_price: {e.get('execution', {}).get('actual_fill_price')}")
except Exception as ex:
    print(f'API SIMULATION ERROR: {ex}')
    import traceback; traceback.print_exc()

conn.close()
print('\nDONE')
