# Visual Architecture Guide: Credit Spreads & Order Execution

---

## Order Execution Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRADING STRATEGY                                 │
│                (EMA, RSI, Iron Condor, Credit Spread, etc)               │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │  Strategy.generate_signal()    │
        │   → Signal Object              │
        │     • tradingsymbol            │
        │     • quantity                 │
        │     • transaction_type         │
        │     • stop_loss (optional)     │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────┐
        │  TradingService._process_      │
        │      signal()                  │
        │  • Log to WebSocket            │
        │  • Validate signal             │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────────────────────┐
        │  ExecutionEngine.execute_signal(signal)            │
        │  1. RiskManager.validate_signal()                  │
        │  2. _signal_to_order(signal)                       │
        │  3. Apply stop-loss ❌ (BUG: no execution price)   │
        │  4. client.place_order(order_request)              │
        │     → Returns: order_id                            │
        └────────────┬─────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────┐
        │  KiteClient.place_order()          │
        │  POST /orders/regular              │
        │  { tradingsymbol,                  │
        │    exchange,                       │
        │    transaction_type,               │
        │    order_type,                     │
        │    quantity,                       │
        │    price,                          │
        │    trigger_price }                 │
        └────────────┬────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────┐
        │  Zerodha Kite Broker API           │
        │                                    │
        │  ✓ Order ACK: order_id=ABC123     │
        │  • Queued in order book            │
        │  • Waiting for matching            │
        └────────────┬────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────┐
        │  TradeJournal.record_trade()       │
        │  • Store entry with status=PLACED  │
        │  • Save to disk                    │
        │  • Log to trades.json              │
        └────────────────────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────────────┐
        │  Broker Execution (Continuous)          │
        │  • Monitor for fill                     │
        │  • Update executed price                │
        │  • Return filled_quantity & avg_price   │
        └────────────┬──────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────────┐
        │  ExecutionEngine.reconcile_orders()    │
        │  (Every 30 seconds)                    │
        │  • Fetch order status from broker      │
        │  • Update pending_orders → filled_     │
        │    orders                              │
        │  • Update PositionTracker              │
        └────────────────────────────────────────┘
```

---

## Multi-Leg Strategy Execution (Current - PROBLEMATIC)

```
Iron Condor Entry Sequence:

Signal 1: SELL 20000 CE
├─ execute_signal() → place_order() ✓ FILLED 50  [T=0ms]
│
Signal 2: BUY 21000 CE  
├─ execute_signal() → place_order() ⚠️  PARTIAL 30/50  [T=10ms]
│
Signal 3: SELL 20000 PE
├─ execute_signal() → place_order() ✓ FILLED 50  [T=20ms]
│
Signal 4: BUY 19000 PE
├─ execute_signal() → place_order() ❌ REJECTED  [T=30ms]

═══════════════════════════════════════════════════════════════

Result: ORPHANED POSITION CREATED! ☠️

Broker Position:
┌──────────────────────────────────┐
│ SHORT 50 x 20000 CE (FULL)       │  ← Unhedged!
│ LONG 30 x 21000 CE (PARTIAL)     │
│ SHORT 50 x 20000 PE (FULL)       │  ← Unhedged!
│ BUY 0 x 19000 PE (FAILED)        │
└──────────────────────────────────┘

Exposure:
┌──────────────────────────────────┐
│ 20 SHORT calls (UNHEDGED)        │  Max Loss: 20 * 1000 = 20,000+
│ 50 SHORT puts (UNHEDGED)         │  Max Loss: 50 * 1000 = 50,000+
│ TOTAL RISK: 70,000+              │
│ EXPECTED: 500-1000               │
│ RATIO: 70-140x MORE RISK!        │
└──────────────────────────────────┘

Current System: ❌ DOES NOT DETECT THIS
```

---

## Multi-Leg Strategy Execution (Fixed)

```
Iron Condor Entry Sequence (WITH FIX #1):

CREATE OrderGroup(strategy="iron_condor", qty=50)
│
├─ Add Leg 1: SELL 20000 CE (required: 50)
├─ Add Leg 2: BUY 21000 CE (required: 50)
├─ Add Leg 3: SELL 20000 PE (required: 50)
└─ Add Leg 4: BUY 19000 PE (required: 50)

VALIDATE Pre-Execution
├─ RiskManager.validate_all_legs()
└─ Check margin for ALL legs (pre-validation - FIX #5)

EXECUTE with ROLLBACK
├─ Place order 1 ✓ FILLED 50
├─ Place order 2 ⚠️  FILLED 30/50
├─ Place order 3 ✓ FILLED 50
└─ Place order 4 ❌ FAILED
    │
    └─ VALIDATION FAILED ✗
       │
       ├─ Cancel order 1 ✓
       ├─ Cancel order 2 ✓
       ├─ Cancel order 3 ✓
       └─ Cancel order 4 (already failed)
    
RESULT: ✓ NO POSITION CREATED
        ✓ ALL FILLS ROLLED BACK
        ✓ GROUP STATUS = FAILED
        ✓ RETRY SIGNAL SENT
```

---

## P&L Calculation Architecture

```
┌──────────────────────────────────────────────────────┐
│            POSITION TRACKING P&L                     │
└──────────────────────────────────────────────────────┘

PositionTracker (Real-time)
│
├─ Position Data
│  ├─ entry_price: 100.0
│  ├─ current_price: 98.5 (updated every tick)
│  ├─ quantity: 50
│  └─ transaction_type: BUY
│
├─ Unrealized P&L Calculation
│  ├─ For BUY: (98.5 - 100.0) * 50 = -75.0
│  ├─ For SELL: (100.0 - 98.5) * 50 = 75.0
│  └─ Updated: Real-time (every tick)
│
├─ Multi-Leg Aggregation (NEW)
│  ├─ Iron Condor Group ID: IC_001
│  │  ├─ SHORT 20000 CE: + 150 (profit)
│  │  ├─ LONG 21000 CE: - 100 (loss)
│  │  ├─ SHORT 20000 PE: + 200 (profit)
│  │  └─ LONG 19000 PE: - 50 (loss)
│  │  └─ GROUP P&L: +200 (net)
│  │
│  └─ Credit Spread Group ID: CS_002
│     └─ GROUP P&L: +125
│
└─ Dashboard
   ├─ By Symbol: {NIFTY: +200, BANK: -50}
   ├─ By Strategy: {iron_condor: +200, ema: +1000}
   └─ Total: +1200

┌──────────────────────────────────────────────────────┐
│         TRADE JOURNAL P&L (Realized)                 │
└──────────────────────────────────────────────────────┘

TradeJournal
│
├─ Entry Record
│  └─ status: PLACED / FILLED
│
├─ Closure Record  
│  ├─ exit_price: 98.5
│  ├─ realized_pnl: -75.0
│  └─ status: CLOSED
│
├─ Daily P&L
│  └─ get_daily_pnl() → Sum of all closed_pnl for today
│
└─ Historical P&L
   ├─ get_pnl_by_strategy()
   ├─ get_pnl_by_instrument()
   └─ get_total_pnl()

ISSUE: ❌ Exercise not recorded
       ❌ Unrealized P&L not aggregated to journal
       ❌ No reconciliation with broker positions
```

---

## Stop-Loss Execution Price Flow

```
CURRENT (WRONG):
════════════════

Signal: BUY @ 100, stop_loss=95
     │
     ▼
ExecutionEngine:
  order.trigger_price = 95     ← TRIGGER price
  order.price = None           ← EXECUTION price ❌ MISSING!
  order.type = SL
     │
     ▼
Broker receives:
  "Execute at MARKET when price hits 95"
     │
     ▼
Scenario: Market gaps down
  Price: 95.5 → 95.0 → 90.0 → 85.0 (no fills in between)
     │
     ▼
Broker execution: MARKET order at 85.0
     │
     ▼
Actual loss: 100 - 85 = 15 (150% of expected 10!)

══════════════════════════════════════════════════════════

FIXED (CORRECT):
════════════════

Signal: BUY @ 100, stop_loss=95
     │
     ▼
ExecutionEngine (WITH FIX #2):
  order.trigger_price = 95       ← TRIGGER price
  order.price = 94.5             ← EXECUTION price ✓
  order.type = SL_LIMIT
     │
     ▼
Broker receives:
  "Execute at 94.5 (or better) when price hits 95"
     │
     ▼
Scenario: Market gaps down
  Price: 95.5 → 95.0 → 90.0
         ↑ TRIGGER fires
            │
            └─ Execute at 94.5 (or fill instantly at that level)
     │
     ▼
Actual loss: 100 - 94.5 = 5.5 (Expected!)

PROTECTION GAINED: 
  Gap floor established
  Slippage limited to 0.5 points
  Loss protection: 5.5 vs 15 = 63% improvement
```

---

## Position Validation Flow

```
Continuous Monitoring (Every 30 seconds):
═════════════════════════════════════════

PositionTracker
   │
   ├─ Get all open positions
   │  └─ [IC_001_leg1, IC_001_leg2, IC_001_leg3, IC_001_leg4]
   │
   ├─ Group by strategy
   │  └─ iron_condor: [4 legs]
   │
   ├─ Validate hedge ratios
   │  ├─ Call spread: SHORT 50 vs LONG 50? ✓ OK
   │  └─ Put spread: SHORT 50 vs LONG 50? ❌ MISMATCH (30 vs 50)
   │
   ├─ Calculate variance %
   │  └─ (50-30)/50 * 100 = 40% mismatch
   │
   └─ Alert if > 10% variance
      │
      ├─ Log: POSITION_VALIDATION_FAILED
      ├─ Severity: CRITICAL
      ├─ Symbol: NIFTY
      ├─ Issue: "Mismatched hedge: 50 short, 30 long"
      │
      └─ Notify Risk Team
         ├─ Email alert
         ├─ Slack notification
         └─ Dashboard red alert

════════════════════════════════════════════════════════

Auto-Recovery (Next Steps):
═══════════════════════════

Option 1: Auto-Close Unhedged Portion
  └─ Close 20 SHORT calls (unhedged part)

Option 2: Buy Back Missing Hedge
  └─ Buy 20 LONG calls @ market

Option 3: Manual Review
  └─ Alert trader, manual decision
```

---

## Exercise/Assignment Timeline

```
Bull Put Spread Position:
═════════════════════════

SETUP (Monday):
┌─────────────────────────────────┐
│ SELL 19500 PUT @ 100 (Delta 0.20)│  Premium: +5000
│ BUY 19000 PUT @ 50 (Delta 0.10) │  Cost: -2500
│ NET CREDIT: +2500               │  Max Loss: 25,000
│ Max Profit: 2500                │
└─────────────────────────────────┘

WEDNESDAY (Expiry Day):
│
├─ Market close NIFTY @ 18800 (ITM by 700!)
│
└─ Day End Processing
   │
   ├─ Exercise Handler.monitor_exercises()
   │
   ├─ Check if ITM: 18800 < 19500? YES
   │
   ├─ Exercise the PUT
   │  └─ Creates SHORT 50 NIFTY @ 19500
   │     Position value: 975,000
   │
   ├─ Record Exercise Event
   │  └─ "Auto-exercised 50 NIFTY short"
   │
   ├─ Update PositionTracker
   │  └─ Add new equity short position
   │
   └─ Calculate P&L
      ├─ Credit spread closes: +2500 (max profit)
      ├─ Long put exercised: 0 (OTM, worthless)
      └─ Short NIFTY created: Danger!

════════════════════════════════════════════════════════

THURSDAY (Next Day):
│
├─ Trader discovers SHORT NIFTY position (surprise!)
│
├─ Market opens NIFTY @ 18500 (DOWN 300 more)
│
├─ P&L Impact
│  └─ SHORT NIFTY unhedged: 300 points loss = -150,000
│
└─ Forced liquidation
   └─ Account margin insufficient
      System forced to buy back at market

════════════════════════════════════════════════════════

PREVENTION with Fix #4:
═══════════════════════

ExerciseHandler monitors continuously
   │
   ├─ 1 day before expiry: Alert trader
   │
   ├─ Send close signal automatically
   │  └─ Close both legs (atomic, FIX #1)
   │
   └─ Result: ✓ No surprise exercise
              ✓ Position closed cleanly
              ✓ Profit realized safely
```

---

## Credit Spread Risk Profile

```
Bull Call Credit Spread:
╔═════════════════════════════════════════════════════════╗
║ Setup: SELL 20000 CE @ 100, BUY 21000 CE @ 50          ║
║ Net Credit: 50 points = +2500 (50 qty)                  ║
║ Max Profit: 2500 (credit collected)                     ║
║ Max Loss: (21000-20000)*50 - 2500 = 47500              ║
║ Risk/Reward: 47500/2500 = 19:1 (bad ratio!)            ║
╚═════════════════════════════════════════════════════════╝

Zone Analysis:
══════════════

NIFTY @ 19800          VERY PROFITABLE ZONE ✓
  ├─ Both calls OTM
  ├─ Full credit = profit
  └─ Theta decay = more profit

NIFTY @ 20000          PROFITABLE ZONE ✓
  ├─ SHORT call @ strike
  ├─ LONG call protection active
  └─ Max profit zone

NIFTY @ 20500          DANGER ZONE ⚠️
  ├─ SHORT call ITM (losing)
  ├─ LONG call ITM (winning less)
  ├─ Adjustment needed (reduce position or roll)
  └─ → Close if profit target hit (FIX #6)

NIFTY @ 21000          MAXIMUM LOSS ZONE ☠️
  ├─ SHORT call 1000 ITM
  ├─ LONG call right at strike
  ├─ Max loss realized
  └─ → MUST HAVE CLOSED BEFORE HERE!

Bear Put Credit Spread:
╔═════════════════════════════════════════════════════════╗
║ Setup: SELL 19500 PE @ 100, BUY 19000 PE @ 50           ║
║ Net Credit: 50 points = +2500                            ║
║ Max Profit: 2500                                         ║
║ Max Loss: (19500-19000)*50 - 2500 = 47500              ║
║ Risk/Reward: 19:1 (bad, needs tighter adjustment)      ║
╚═════════════════════════════════════════════════════════╝

Zone Analysis:
══════════════

NIFTY @ 20000          VERY PROFITABLE ZONE ✓
  ├─ Both puts OTM
  └─ Full credit = profit

NIFTY @ 19500          PROFITABLE ZONE ✓
  ├─ SHORT put @ strike
  └─ Max profit zone

NIFTY @ 19000          DANGER ZONE ⚠️
  ├─ SHORT put 500 ITM
  ├─ LONG put protection kicking in
  └─ Close for profit if available (FIX #6)

NIFTY @ 18500          MAXIMUM LOSS ZONE ☠️
  ├─ Both puts ITM
  ├─ Max loss = 47500
  └─ AVOID AT ALL COSTS
```

---

## Key Metrics Dashboard

```
┌─────────────────────────────────────────────────────┐
│        REAL-TIME TRADING DASHBOARD LAYOUT            │
└─────────────────────────────────────────────────────┘

┌──────────────────┐  ┌──────────────────┐
│     POSITIONS    │  │   STRATEGIES     │
├──────────────────┤  ├──────────────────┤
│ NIFTY: +2500 P&L │  │ Iron Condor: 1   │
│ BANK: -500 P&L   │  │ Bull Call CS: 2  │
│ INFY: +1200 P&L  │  │ Bear Put CS: 1   │
└──────────────────┘  └──────────────────┘

┌──────────────────────────────────────────────────────┐
│               P&L BY STRATEGY                        │
├──────────────────────────────────────────────────────┤
│ Iron Condor:        +  1000 (2500 max profit)       │
│ Bull Call CR:       +  2300 (on track)              │
│ Bear Put CR:        -   800 (adjustment needed)     │
│ EMA Crossover:      +  5200 (strong)                │
│ RSI Strategy:       +   800                         │
│ ─────────────────────────────                       │
│ TOTAL:              + 10100 (daily)                 │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│            RISK METRICS                              │
├──────────────────────────────────────────────────────┤
│ Max Daily Loss:     - 50000 (Set)                   │
│ Current Loss:       +    0 (Green)                  │
│ Margin Used:        50% (Healthy)                   │
│ Position Validation: ✓ PASS                         │
│ Multi-Leg Hedge:    ✓ ALL MATCHED                   │
│ Orphan Positions:   0                               │
│ ITM at Expiry:      0                               │
└──────────────────────────────────────────────────────┘
```

---

## Summary

This visual guide shows:
1. **Order flow**: Signal → Order → Broker → Execution
2. **Multi-leg problem**: Partial fills create orphaned positions  
3. **Multi-leg solution**: Order groups with rollback (Fix #1)
4. **P&L tracking**: Real-time unrealized + realized journal
5. **Stop-loss issue**: Missing execution price
6. **Stop-loss fix**: Add execution price with slippage (Fix #2)
7. **Position validation**: Continuous hedge ratio monitoring (Fix #3)
8. **Exercise handling**: Auto-detection of ITM options (Fix #4)
9. **Credit spreads**: Risk profiles for bull/bear versions

