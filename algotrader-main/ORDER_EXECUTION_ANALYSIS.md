# Order Execution Flow & P&L Tracking: Critical Analysis

**Date**: February 21, 2026  
**Focus**: How orders are placed, executed, traced, and P&L is calculated; Critical scenarios for multi-leg strategies

---

## 1. Order Execution Flow

### 1.1 Signal to Order Pipeline

```
Strategy generates Signal
    ↓
TradingService._process_signal()
    ├─ Logs signal to _signal_log
    ├─ Broadcasts via WebSocket
    ├─ Executes ExecutionEngine.execute_signal(signal)
    │   ├─ RiskManager.validate_signal()
    │   ├─ ExecutionEngine._signal_to_order() → OrderRequest
    │   ├─ Apply stop-loss if present (convert to SL or SL-M order type)
    │   └─ KiteClient.place_order() → Returns order_id
    │
    └─ TradeJournal.record_trade()
        └─ Records entry with status="PLACED"
```

**Files Involved**:
- [src/api/service.py](src/api/service.py#L410-L450) - Signal processing
- [src/execution/engine.py](src/execution/engine.py#L35-L75) - Order execution
- [src/api/client.py](src/api/client.py#L206-L240) - Kite API order placement
- [src/data/journal.py](src/data/journal.py#L20-L60) - Trade recording

### 1.2 Order Request Structure

```python
OrderRequest:
  tradingsymbol: str          # e.g., "NIFTY23DEC20000CE"
  exchange: Exchange          # NFO for options
  transaction_type: BUY|SELL # Direction
  order_type: MARKET|LIMIT|SL|SL_M
  quantity: int              # Lot size * lots
  product: NRML|MIS|CNC      # Position type
  price: float (optional)    # For LIMIT orders
  trigger_price: float (optional) # For SL orders (stop-loss activation)
  variety: REGULAR|AMO|CO|ICEBERG
```

### 1.3 Stop-Loss Implementation

**Current Implementation** - [src/execution/engine.py](src/execution/engine.py#L48-L60):

```python
if signal.stop_loss:
    order_request.trigger_price = signal.stop_loss
    if order_request.order_type in (OrderType.MARKET, OrderType.LIMIT):
        order_request.order_type = OrderType.SL  # Stop Loss
        if order_request.price is None:
            order_request.order_type = OrderType.SL_M  # Stop Loss Market
```

**Issue**: The trigger_price is passed as the STOP price, but the execution price is not specified!

```python
# INCORRECT INTERPRETATION:
Signal(stop_loss=100.0)
  → OrderRequest(trigger_price=100.0, order_type=SL)
  → Broker waits for price ≤ 100.0, then executes at... what price? (MARKET?)
```

**This is CRITICAL**: For SL orders, you need BOTH:
- `trigger_price`: When to activate (e.g., 100.0)
- `price`: Execution price when triggered (e.g., 99.8 for LIMIT, or MARKET for SL_M)

Current code doesn't specify execution price, relies on broker default (likely market).

---

## 2. Multi-Leg Order Execution Problem

### 2.1 How Multi-Leg Strategies Execute (Current Implementation)

**Iron Condor Example**:

```python
signals = [
    Signal(tradingsymbol="NIFTY23DEC20000CE", transaction_type=SELL, ...),  # Sell call
    Signal(tradingsymbol="NIFTY23DEC21000CE", transaction_type=BUY, ...),   # Buy call
    Signal(tradingsymbol="NIFTY23DEC20000PE", transaction_type=SELL, ...),  # Sell put
    Signal(tradingsymbol="NIFTY23DEC19000PE", transaction_type=BUY, ...),   # Buy put
]

for signal in signals:
    order_id = await execution_engine.execute_signal(signal)  # 4 SEPARATE API CALLS
    journal.record_trade(order_id=order_id, status="PLACED")
```

### 2.2 Critical Problem: Non-Atomic Execution

**Scenario 1: Partial Fill Risk**

```
Time 0:  Sell call order executed (filled 50 contracts)
Time 1:  Buy call order rejected due to lack of balance
Time 2:  Sell put order executed (filled 50 contracts)
Time 3:  Buy put order rejected due to lack of balance

Result: UNHEDGED position = SHORT 100 calls + SHORT 100 puts
        Maximum loss = UNLIMITED (unlimited call loss + 100 * spread)
        Account blows up
```

**Scenario 2: Order Cancellation Race**

```
Time 0: Place SELL call order (order_id = "123")
Time 1: Place BUY call order (order_id = "124")
Time 2: System detects order "124" is being rejected
Time 3: System tries to cancel order "123" but it already filled
Time 4: Position created with orphaned short leg
```

**Scenario 3: Partial Fill on Multi-Leg**

```
Iron Condor: Target 50 lots (4 legs = 200 contracts individual orders)

Execution:
  ✓ SELL call: 50 lots filled
  ✓ BUY call:  40 lots filled (10 partial)
  ✓ SELL put:  50 lots filled  
  ✗ BUY put:   35 lots filled (15 partial, 50 rejected)

Result:
  - Position: 50 short calls, 40 long calls, 50 short puts, 35 long puts
  - Net: 10 unhedged short calls, 15 unhedged short puts
  - This creates SIGNIFICANT naked short exposure!
```

### 2.3 Why This Happens

1. **Zerodha Kite API doesn't support atomic multi-leg orders** at the API level
2. **Orders placed sequentially** (one at a time) - no grouping/pairing
3. **No transaction rollback** if one leg fails
4. **No position pairing validation** after execution
5. **PositionTracker is unaware** of multi-leg grouping requirement

---

## 3. P&L Calculation & Tracking

### 3.1 Current P&L Architecture

```
Manual Entry:
  Entry Price: When traders execute order
  ↓
Position tracking:
  Current Price: Updated from market ticks
  ↓
Unrealized P&L:
  BUY: (current_price - entry_price) * quantity
  SELL: (entry_price - current_price) * quantity
  ↓
Realized P&L:
  When position closed
  ↓
TradeJournal.record_trade(pnl=)
  └─ Stored in _entries list
  └─ Saved to disk (date-based)
```

**Files**:
- [src/data/position_tracker.py](src/data/position_tracker.py) - Unrealized P&L calculation
- [src/data/journal.py](src/data/journal.py) - Realized P&L recording

### 3.2 P&L Calculation Issues

**Issue 1: Multi-Leg P&L Aggregation**

```python
# Current code treats each leg independently
position_1 = Position(tradingsymbol="20000CE", side=SELL, price=100, qty=50)
position_2 = Position(tradingsymbol="21000CE", side=BUY, price=50, qty=50)

# When calculating P&L for "bull_call_credit_spread":
current_prices = {
  "20000CE": 95,
  "21000CE": 48
}

P&L_leg1 = (100 - 95) * 50 = +250  (sell credit, price went down = profit)
P&L_leg2 = (48 - 50) * 50 = -100   (buy debit, price went down = loss)
Total_P&L = +250 - 100 = +150

# But PositionTracker calculates per position:
tracker.get_positions("BULL_CALL_SPREAD") only groups by strategy
NOT by spread group_id
```

**Issue 2: Assignment & Exercise Not Tracked**

Options can be exercised/assigned. Current system doesn't handle:
- Automatic exercise (if ITM at expiry)
- Early assignment (seller's choice)
- Physical delivery of underlying

```python
# No mechanism for:
position.is_exercised = True
position.exercise_price = strike
position.created_underlying_position = True
```

**Issue 3: Margin Requirement Not Linked to Positions**

```python
# System fetches margin separately:
margins = await client.get_margins()

# But doesn't validate:
if required_margin > available_margin:
    raise InsufficientMarginError()

# This can cause surprise rejections during order execution
```

### 3.3 P&L by Strategy

**Current Implementation** - [src/data/journal.py](src/data/journal.py#L79-L85):

```python
def get_pnl_by_strategy(self) -> dict[str, float]:
    result: dict[str, float] = {}
    for entry in self._entries:
        result[entry.strategy] = result.get(entry.strategy, 0.0) + entry.pnl
    return result
```

**Problem**: Only works if PNL is recorded in journal at trade closure. For intraday tracking:

```python
# During the day:
entry = Signal(strategy="iron_condor", entry_price=100)
# executed...
# Later, prices move:
current_price = 98
# Current code: No way to calculate unrealized P&L by strategy!
# PositionTracker can do it, but results not integrated with journal
```

---

## 4. Critical Scenarios & Failure Modes

### Scenario A: Assignment on Short Call

**Setup**: Sold call spread, short call ITM at expiry

```
Position:
  - SHORT 1 call @ 20000 strike (premium collected = 100)
  - LONG  1 call @ 21000 strike (premium paid = 50)
  - Net credit = 50

Market: NIFTY at 20100 (call assigned)

What happens:
  1. Broker auto-exercises short call @ 20000 → creates short NIFTY position
  2. System doesn't know about this (no API notification)
  3. Position now shows:
     Closed_call_leg: P&L = -50 (loss on spread closure since exercised below max profit)
     New: SHORT 50 NIFTY contracts
  4. Trader thinks spread closed, but now has naked short equity exposure!
```

**Current System Issues**:
- No code to handle exercise/assignment
- No automatic buy back of long call to cover assignment
- Treasury desk not notified of underlying position

### Scenario B: Partial fill leads to unhedged position

**Setup**: Bull call credit spread, place 50-lot orders

```
Order 1: SELL 50 x 20000 CE
  → Filled 50 @ 100.0 = +5000 credit

Order 2: BUY 50 x 21000 CE (protection)
  → Filled 30 @ 50.0 = -1500 (only 30 filled!)
  → Remaining 20 stay unfilled in order queue

Order 3: Risk manager sees unmatched sell, tries to cancel order 1
  → Too late, already filled

Result:
  - Trading account: +3500 net credit (sold 50 @ 100, bought 30 @ 50)
  - Actual position: 50 SHORT calls (unhedged up to 20 contracts)
  - Max loss: Should be 500, actually could be 5000+ (20 * 500 strike width)
```

**Current System Issues**:
- No validation that multi-leg fills are synchronized
- No alerting on "mismatched_legs"
- Position tracker doesn't validate hedge ratios

### Scenario C: PositionTracker not linked to execution

```python
# In execution engine
order_id = await client.place_order(order_request)  # Order placed
# But tracker doesn't know!

# In position tracker
position = tracker.add_position(...)  # Added AFTER execution
# If execution fails here, order is orphaned

# Later, when broker reports fill:
broker_fills = await client.get_trades()
# System doesn't reconcile with PositionTracker
# Trade might exist in broker but not in tracker
```

### Scenario D: Stop Loss Order Not Filled

```python
# Signal has stop_loss = 100
signal = Signal(stop_loss=100.0)
  → Converted to SL order with trigger_price=100

# What happens:
- Market drops to 100.5 → Order not triggered (> trigger)
- Market drops to 100.1 → Order not triggered (> trigger)
- Market gaps down to 99.5 → Order triggered!
  → Executes at market (SL_M) → Could fill at 95, 90, etc.
  
# Current code doesn't specify execution price, uses default broker behavior
# Result: Stop loss could be much worse than expected!
```

### Scenario E: Options Expiry without Position Closure

```
1. Buy NIFTY 20000 PUT expiring today
2. Position held until market close (no exit signal generated)
3. Expiry happens → position auto-settled
4. System doesn't record exercise/settlement
5. Trader's P&L is wrong (settlement not reflected)
```

---

## 5. Order Execution Trace Mechanism

### 5.1 Current Tracing

**What's tracked**:
- Signal log (in-memory + WebSocket broadcast)
- Order placed event (logged via KiteClient.place_order)
- Journal entry (recorded with status="PLACED")
- Reconciliation (ExecutionEngine.reconcile_orders)

**What's NOT tracked**:
- Order lifetime (creation → fill → settlement)
- Partial fills that don't match legs
- Broker-side events (rejection, cancellation)
- Assignment/exercise

### 5.2 Missing Trace Points

```
IDEAL TRACE:
Signal(id=123, side=SELL, symbol=20000CE)
  ├─ Created [2:15:23.123]
  ├─ Validated by RiskManager [2:15:23.234]
  ├─ Sent to Broker [2:15:23.345]
  ├─ Broker ACK: OrderID=456 [2:15:23.456]
  ├─ Partial fill: 25 contracts @ 99.5 [2:15:24.567]
  ├─ Remaining: 25 contracts in queue [2:15:24.600]
  ├─ Partial fill: 25 contracts @ 99.3 [2:15:25.678]
  ├─ Order COMPLETE [2:15:25.789]
  ├─ Position created: 50 SHORT @ avg 99.4 [2:15:25.800]
  ├─ Linked to PositionTracker [2:15:25.811] ← MISSING!
  ├─ Exit signal generated [2:15:35.000]
  └─ Order closure reconciled [2:15:35.100]

CURRENT SYSTEM:
Signal → Order placed → Trade journal entry → [BLANK] → No further trace
```

---

## 6. Critical Recommendations

### 6.1 URGENT FIXES (Security/Correctness)

```python
# 1. Multi-Leg Order Grouping
class MultiLegOrderGroup:
    group_id: str  # UUID to track related orders
    legs: list[str]  # order_ids
    required_fills: dict[str, int]  # {order_id: required_quantity}
    status: "PENDING_ALL" | "PARTIAL" | "COMPLETE" | "FAILED"
    
    def validate_fills(self):
        """Ensure all legs are filled proportionally"""
        for order_id, required_qty in self.required_fills.items():
            actual = self.get_filled_qty(order_id)
            if actual < required_qty:
                raise PhantomLegError(f"Leg {order_id} not fully filled")

# 2. Stop Loss Fix
def execute_signal_with_stop_loss(signal: Signal):
    if signal.stop_loss:
        # BEFORE: order.trigger_price = signal.stop_loss  (WRONG!)
        # AFTER:
        order.trigger_price = signal.stop_loss
        order.price = calculate_sl_limit_price(side, signal.stop_loss)
        # e.g., for SHORT: stop_loss=100 → price=99.8 (1% margin)

# 3. Position Closure Validation
def create_close_signals(positions: list[Position], reason: str):
    """Ensure ALL legs close atomically"""
    if len(positions) > 1:
        # Create multi-leg close order group
        group = MultiLegOrderGroup(
            group_id=str(uuid4()),
            legs=[p.tradingsymbol for p in positions],
            required_fills={p.tradingsymbol: p.quantity for p in positions}
        )
        # Execute group, not individual orders
        return execute_order_group(group, positions)
    else:
        # Single leg, normal execution
        return [create_close_signal(positions[0])]
```

### 6.2 Medium Priority (Completeness)

```python
# Track assignment/exercise
@dataclass
class Position:
    # ... existing fields ...
    exercise_info: Optional[ExerciseInfo] = None  # NEW
    assigned_at: Optional[datetime] = None
    underlying_position_created: bool = False

# Margin pre-validation
async def validate_margin_before_execution(orders: list[OrderRequest]):
    required_margin = calculate_margin_requirement(orders)
    available_margin = self.margin_tracker.get_available()
    
    if required_margin > available_margin:
        raise InsufficientMarginError(
            f"Need {required_margin}, have {available_margin}"
        )

# P&L reconciliation
async def reconcile_with_broker():
    broker_positions = await client.get_positions()
    tracker_positions = self.tracker.get_positions()
    
    for symbol, broker_qty in broker_positions.items():
        tracker_qty = sum(p.quantity for p in tracker_positions 
                         if p.tradingsymbol == symbol)
        if abs(broker_qty - tracker_qty) > 0:
            logger.error("position_mismatch", symbol=symbol, 
                        broker=broker_qty, tracker=tracker_qty)
```

### 6.3 Operational Best Practices

1. **Never** execute multi-leg orders without atomic grouping
2. **Always** validate stop-loss prices before placing orders
3. **Daily reconciliation** of positions vs broker
4. **Pre-market checks**: Ensure no orphaned positions from previous day
5. **Assignment handling**: Monitor ITM positions, auto-close or handle exercise

---

## 7. Credit Spread Risk Profile

### BullCallCreditSpread (Newly Added)

```
Position: SELL call @ 20000 (delta 0.20), BUY call @ 21000 (delta 0.10)

Max Profit: Net credit collected (typically 20-50 points)
Max Loss: Strike difference - Credit (typically 900-980 points)
Profit Zone: Price below short strike @ expiry

Risk Scenarios:
  1. ✓ Price stays flat → Theta decay profits (50-75% of max profit typical)
  2. ✗ Price jumps up → Position breaches short strike
     - Adjustment triggered (close or roll)
     - If not closed: Assignment on short leg at expiry
  3. ✓ Price drops → Both legs have reduced cost, close early for profit
  4. ⚠️  Gap up at market open → Can't exit, forced to assignment
```

### BearPutCreditSpread (Newly Added)

```
Position: SELL put @ 20000 (delta 0.20), BUY put @ 19000 (delta 0.10)

Max Profit: Net credit collected  
Max Loss: Strike difference - Credit
Profit Zone: Price above short strike @ expiry

Risk Scenarios:
  1. ✓ Price stays flat → Theta decay profits
  2. ✗ Price jumps down → Position breaches short strike
     - Adjustment triggered
  3. ✓ Price rises → Both puts OTM, close early for full profit
  4. ⚠️  Gap down at market open → Can't exit, forced to assignment
```

### Critical Difference from Debit Spreads

| Aspect | Debit Spread (BullCallSpread) | Credit Spread (BullCallCreditSpread) |
|--------|------|------|
| Premium | Pay upfront (debit) | Collect upfront (credit) | 
| P&L on entry | Negative (debit collected) | Positive (credit in account) |
| P&L at expiry | Limited to max profit | Limited to premium collected |
| Assignment risk | LOW (long leg) | HIGH (short leg) |
| Margin required | Lower (spread debit) | Higher (short exposure) |
| Best case | Full profit @ zero cost | Full profit + early close bonus |
| Worst case | Full loss of debit | Full loss (strike width - credit) |

**CRITICAL**: Credit spreads require **2-3x more margin** than debit spreads for same quantity!

---

## Summary: What Works vs What Needs Fixing

✅ **What Works Well**:
- Signal generation per strategy
- Individual order placement
- Trade journal recording
- Basic P&L calculation
- Position tracking infrastructure

❌ **Critical Issues**:
- Multi-leg orders placed sequentially (non-atomic)
- Stop-loss orders lack execution price specification
- No position pairing validation
- No assignment/exercise handling
- P&L not reconciled with broker positions

⚠️ **Use With Caution**:
- Credit spreads (due to multi-leg execution risk)
- Options assignment (no mechanism exists)
- Gap risk (can't exit at open, forced assignment)
- Partial fills (no validation that all legs matched)

