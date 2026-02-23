# CRITICAL ADDENDUM: 5 Additional Production Safety Gaps

**Status**: ⚠️ BLOCKING ISSUE - Must fix before production deployment  
**Severity**: CRITICAL (System will fail under real trading conditions)  
**Date**: February 21, 2026

---

## EXECUTIVE SUMMARY

The 7 previously documented fixes are **incomplete**. Five additional mechanisms are required for true production safety:

1. **Monotonic State Validation** - Prevent state machine regression from out-of-order events
2. **Idempotent Event Processing** - Prevent duplicate execution from exchange event resends
3. **Intent-Before-Action Persistence** - Persist order intent BEFORE sending to broker
4. **Recursive Hedge Monitoring** - Track hedge execution as its own exposure
5. **Broker Timestamp Anchoring** - Replace local timers with exchange timestamps

**Without these 5 safeguards, the system WILL catastrophically fail in production.**

---

## GAP #1: Event Ordering Assumption 🔴

### The Problem

**Current Assumption**:
```python
# order_group_corrected.py assumes:
for event in events:  # Events arrive in order
    execute_state_transition(event)
```

**Reality**:
Broker WebSockets deliver events out of sequence during reconnects:

```
Expected: PENDING → PARTIAL → FILLED
Actual:   FILLED → PENDING → PARTIAL  (out of order during reconnect)
```

### Why This Breaks

State machine uses `ExecutionLifecycle` enum with ordinal progression:
```python
class ExecutionLifecycle(Enum):
    CREATED = 0
    VALIDATED = 1
    HEDGE_SUBMITTED = 2
    HEDGE_FILLED = 3
    RISK_SUBMITTED = 4
    PARTIAL_FILL = 5          # ← Current state
    FILLED = 6                # ← Incoming event
    HEDGED = 7
    ...
```

If a **FILLED event arrives before PARTIAL_FILL completes**:
```
State machine at: PARTIAL_FILL
Incoming: FILLED event
Result: State stays at FILLED (no issue)

BUT if out of order:
State machine at: PARTIAL_FILL
Incoming: PENDING event (old, delayed)
Result: State REGRESSES to PENDING ← Catastrophic!
       - Hedge already placed but marked as unplaced
       - Emergency hedge places AGAIN
       - Double exposure created
```

### The Fix

**Add Monotonic State Validation**:

```python
class ExecutionStateValidator:
    """Prevent state machine regression from out-of-order events."""
    
    STATE_ORDERING = {
        ExecutionLifecycle.CREATED: 0,
        ExecutionLifecycle.VALIDATED: 1,
        ExecutionLifecycle.HEDGE_SUBMITTED: 2,
        ExecutionLifecycle.HEDGE_PARTIAL: 3,
        ExecutionLifecycle.HEDGE_FILLED: 4,
        ExecutionLifecycle.RISK_SUBMITTED: 5,
        ExecutionLifecycle.RISK_PARTIAL: 6,
        ExecutionLifecycle.RISK_FILLED: 7,
        ExecutionLifecycle.HEDGED: 8,
        ExecutionLifecycle.COMPLETED: 9,
    }
    
    @staticmethod
    def validate_state_transition(current_state: ExecutionLifecycle, 
                                   incoming_state: ExecutionLifecycle) -> bool:
        """
        Only allow transitions where incoming >= current.
        
        Returns: True if valid, False if should ignore event
        """
        current_order = ExecutionStateValidator.STATE_ORDERING.get(
            current_state, -1
        )
        incoming_order = ExecutionStateValidator.STATE_ORDERING.get(
            incoming_state, -1
        )
        
        if incoming_order < current_order:
            # Out-of-order event, ignore it
            logging.warning(
                f"Ignoring out-of-order state transition: "
                f"{current_state.name} <- {incoming_state.name}"
            )
            return False
        
        return True


class CorrectMultiLegOrderGroup:
    async def _process_state_update(self, event: dict):
        """Updated method to validate monotonic progression."""
        
        incoming_state = ExecutionLifecycle[event['state']]
        
        # ← ADD THIS VALIDATION
        if not ExecutionStateValidator.validate_state_transition(
            self.state.current_lifecycle, 
            incoming_state
        ):
            # Discard out-of-order event
            self.state.rejected_events.append({
                'timestamp': time.time(),
                'event_id': event.get('id'),
                'reason': 'out_of_order',
                'current': self.state.current_lifecycle.name,
                'incoming': incoming_state.name
            })
            return  # Ignore this event
        
        # Continue with normal processing
        self.state.current_lifecycle = incoming_state
        await self.persistence.persist_state(self.state)
```

### Where to Add

**File**: [src/execution/order_group_corrected.py](src/execution/order_group_corrected.py)

**Location**: Before `_process_state_update()` method (after imports)

**Integration Point**: Add to `CorrectMultiLegOrderGroup._handle_event()` before state transitions

---

## GAP #2: Duplicate Event Delivery 🔴

### The Problem

**Reality**:
Exchanges frequently resend events. During network hiccups:

```
Original: FILLED event (id=12345)
Resend:   FILLED event (id=12345)  ← Exact duplicate
```

### Why This Breaks

Without idempotency:
```python
# First FILLED event
if event['state'] == 'FILLED':
    hedge_qty = calculate_hedge(leg)
    await place_hedge_order(leg.symbol, hedge_qty)  # Hedge #1 placed

# Duplicate FILLED event (no deduplication)
if event['state'] == 'FILLED':
    hedge_qty = calculate_hedge(leg)
    await place_hedge_order(leg.symbol, hedge_qty)  # Hedge #2 placed ← BUG!
```

**Result**: 
- Two identical hedge orders placed
- Account short 2x the intended hedge
- Liquidation risk

### The Fix

**Add Idempotency Tracking**:

```python
class IdempotencyManager:
    """Prevent duplicate event processing."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.processed_event_ids = set()
        self._load_processed_events()
    
    def _load_processed_events(self):
        """Load from DB on startup."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT event_id FROM processed_events 
                WHERE timestamp > datetime('now', '-24 hours')
            """)
            self.processed_event_ids = set(row[0] for row in cursor.fetchall())
        except sqlite3.OperationalError:
            # Table doesn't exist yet
            self._create_table(cursor)
            conn.commit()
        finally:
            conn.close()
    
    def _create_table(self, cursor):
        """Create processed_events table."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_events (
                event_id TEXT PRIMARY KEY,
                group_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_group (group_id),
                INDEX idx_time (timestamp)
            )
        """)
    
    async def mark_processed(self, event_id: str, group_id: str, 
                            event_type: str):
        """Mark event as processed and save to DB."""
        self.processed_event_ids.add(event_id)
        
        # Save to DB (write to queue, not blocking)
        await asyncio.get_event_loop().run_in_executor(
            None,
            self._persist_processed_event,
            event_id, group_id, event_type
        )
    
    def _persist_processed_event(self, event_id, group_id, event_type):
        """Persist to DB."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT OR IGNORE INTO processed_events 
                (event_id, group_id, event_type) 
                VALUES (?, ?, ?)
            """, (event_id, group_id, event_type))
            conn.commit()
        finally:
            conn.close()
    
    def is_duplicate(self, event_id: str) -> bool:
        """Check if event already processed."""
        return event_id in self.processed_event_ids


class CorrectMultiLegOrderGroup:
    def __init__(self, ..., idempotency_mgr: IdempotencyManager):
        self.idempotency = idempotency_mgr
        ...
    
    async def _process_state_update(self, event: dict):
        """Updated to check for duplicates."""
        
        event_id = event.get('id')
        
        # ← ADD THIS CHECK
        if self.idempotency.is_duplicate(event_id):
            logging.warning(
                f"Ignoring duplicate event: {event_id}"
            )
            return  # Skip duplicate
        
        # Mark as processed BEFORE executing
        await self.idempotency.mark_processed(
            event_id, 
            self.state.group_id, 
            event.get('type')
        )
        
        # Continue with normal processing
        incoming_state = ExecutionLifecycle[event['state']]
        self.state.current_lifecycle = incoming_state
        await self.persistence.persist_state(self.state)
```

### Where to Add

**File**: [src/execution/order_group_corrected.py](src/execution/order_group_corrected.py)

**Location**: Add new class `IdempotencyManager` before `CorrectMultiLegOrderGroup`

**Integration Point**:
```python
class CorrectMultiLegOrderGroup:
    def __init__(self, ..., idempotency_mgr: IdempotencyManager):
        self.idempotency = idempotency_mgr
```

---

## GAP #3: Persistence Atomicity Boundary 🔴

### The Problem

**Current Pattern** (WRONG):
```python
# order_group_corrected.py current flow:
await self.broker.place_order(leg)        # ← Order sent to broker (LIVE)
await self.persistence.persist_state()    # ← State saved (LOST if crash)

# Crash occurs between these two lines
# Result:
#  - Broker has: Live open order
#  - DB has: No knowledge of order
#  - Recovery: thinks order was never sent, sends AGAIN
```

### Why This Breaks

```
Timeline:
1. 10:05:32.123 - place_order() succeeds
   ↓ (order is LIVE on broker)
2. 10:05:32.124 - persist_state() queued
3. 10:05:32.125 - CRASH before persist completes
   ↓
4. Restart recovery:
   - Loads DB: No order found
   - Assumes: Order was never sent
   - Action: Sends order AGAIN ← DUPLICATE!
```

### The Fix

**Reverse the Order - Persist Intent FIRST**:

```python
class OrderIntentPersistence:
    """Persist order intent BEFORE sending to broker."""
    
    async def persist_order_intent(self, leg: LegState, order_type: str):
        """
        Save order INTENT to DB before placement.
        
        States: NOT_SENT → INTENT_PERSISTED → SENT → LIVE
        """
        intent_record = {
            'group_id': leg.group_id,
            'leg_id': leg.leg_id,
            'order_type': order_type,  # 'HEDGE', 'RISK', 'SL_LIMIT', 'SL_MARKET'
            'symbol': leg.symbol,
            'quantity': leg.quantity,
            'price': leg.target_price,
            'status': 'INTENT_PERSISTED',
            'timestamp': time.time(),
            'broker_order_id': None  # Will fill after placement
        }
        
        await self.persistence.execute_write({
            'operation': 'INSERT',
            'table': 'order_intents',
            'data': intent_record
        })
        
        return intent_record['intent_id']
    
    async def mark_order_sent(self, intent_id: str, broker_order_id: str):
        """Update after broker confirms order received."""
        await self.persistence.execute_write({
            'operation': 'UPDATE',
            'table': 'order_intents',
            'where': {'intent_id': intent_id},
            'data': {
                'status': 'SENT',
                'broker_order_id': broker_order_id,
                'sent_timestamp': time.time()
            }
        })


class CorrectMultiLegOrderGroup:
    async def _place_order_safely(self, leg: LegState, order_type: str):
        """
        CORRECTED: Persist BEFORE sending.
        
        Sequence:
        1. Save intent to DB         ← immune to crash
        2. Send order to broker      ← order becomes LIVE
        3. Update DB with broker ID  ← link DB to broker
        """
        
        # Step 1: Persist intent FIRST (this is immune to broker failures)
        intent_id = await self.intent_persistence.persist_order_intent(
            leg, order_type
        )
        logging.info(f"Order intent persisted: {intent_id}")
        
        # Step 2: Send to broker
        try:
            broker_response = await self.broker.place_order(
                symbol=leg.symbol,
                qty=leg.quantity,
                price=leg.target_price,
                order_type=order_type
            )
            broker_order_id = broker_response['order_id']
            logging.info(f"Order sent to broker: {broker_order_id}")
        except BrokerError as e:
            logging.error(f"Broker placement failed: {e}")
            # Intent record stays in DB as "INTENT_PERSISTED"
            # Recovery will retry this order
            raise
        
        # Step 3: Link intent to broker order in DB
        await self.intent_persistence.mark_order_sent(
            intent_id, broker_order_id
        )
        
        leg.broker_order_id = broker_order_id
        leg.status = LegStatus.SENT
```

### Recovery on Startup

```python
async def recover_execution_from_crash(db_path: str, broker_client):
    """
    Updated recovery: Check for orphaned INTENT records.
    
    Cases:
    1. INTENT_PERSISTED (no broker_id):
       - Intent saved but order never sent
       - Action: Retry sending order
    
    2. SENT (has broker_id):
       - Intent saved + order sent to broker
       - Action: Query broker for order status
    """
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Find orphaned intents (persisted but never sent)
    cursor.execute("""
        SELECT intent_id, symbol, quantity, price, order_type
        FROM order_intents
        WHERE status = 'INTENT_PERSISTED'
        AND timestamp > datetime('now', '-1 hour')
    """)
    
    orphaned_intents = cursor.fetchall()
    
    for intent in orphaned_intents:
        logging.warning(f"Found orphaned order intent: {intent[0]}")
        
        # Retry sending this order
        try:
            response = await broker_client.place_order(
                symbol=intent[1],
                qty=intent[2],
                price=intent[3],
                order_type=intent[4]
            )
            
            # Update intent with broker ID
            cursor.execute("""
                UPDATE order_intents 
                SET status = 'SENT', broker_order_id = ?
                WHERE intent_id = ?
            """, (response['order_id'], intent[0]))
            conn.commit()
            
        except BrokerError as e:
            logging.error(f"Recovery order placement failed: {e}")
            # Leave as INTENT_PERSISTED for manual review
    
    # Now check SENT orders against broker state
    cursor.execute("""
        SELECT intent_id, broker_order_id
        FROM order_intents
        WHERE status = 'SENT'
        AND timestamp > datetime('now', '-1 hour')
    """)
    
    sent_intents = cursor.fetchall()
    
    for intent_id, broker_order_id in sent_intents:
        broker_status = await broker_client.get_order_status(broker_order_id)
        
        # Update DB with authoritative broker status
        cursor.execute("""
            UPDATE order_intents
            SET status = ?, fills = ?, filled_qty = ?
            WHERE intent_id = ?
        """, (broker_status['status'], broker_status['fills'], 
              broker_status['filled_qty'], intent_id))
        conn.commit()
    
    conn.close()
```

### Where to Add

**File**: [src/execution/order_group_corrected.py](src/execution/order_group_corrected.py)

**Location**: Add `OrderIntentPersistence` class before `CorrectMultiLegOrderGroup`

**Database Migration**:
```sql
CREATE TABLE IF NOT EXISTS order_intents (
    intent_id TEXT PRIMARY KEY,
    group_id TEXT NOT NULL,
    leg_id TEXT NOT NULL,
    order_type TEXT NOT NULL,
    symbol TEXT NOT NULL,
    quantity REAL NOT NULL,
    price REAL,
    status TEXT,  # INTENT_PERSISTED, SENT, FILLED, CANCELLED
    broker_order_id TEXT,
    timestamp REAL,
    sent_timestamp REAL,
    filled_qty REAL DEFAULT 0,
    fills BLOB,  # JSON array of fill events
    FOREIGN KEY (group_id) REFERENCES execution_states(group_id),
    INDEX idx_group (group_id),
    INDEX idx_status (status)
);
```

---

## GAP #4: Emergency Hedge Liquidity Risk 🔴

### The Problem

**Current Assumption**:
```python
# Emergency hedge assumes it fills immediately
emergency_qty = target_qty - current_qty
await place_hedge_order(symbol, emergency_qty)
# Assumes: hedge fills instantly, exposure covered
```

**Reality in Options**:
```
Emergency hedge for wide spread option:
- Place: SELL TO CLOSE 5 contracts
- Bid-ask: 0.40 - 0.50 wide
- Freeze qty: "Only 2 contracts available at bid"  ← Partial fill!

Result:
- Intended: Sell 5 (offset 5 short)
- Actual: Sell 2 (offset only 2)
- New exposure: Still short 3 contracts ← UNTRACKED!

Without recursive monitoring, the emergency hedge itself becomes a new exposure.
```

### Why This Breaks

Current code:
```python
class EmergencyHedgeExecutor:
    async def place_hedge_order(self, symbol, qty):
        """Assumes hedge fills completely."""
        
        await self.broker.place_order(symbol, qty)
        # ← BUG: Doesn't check if hedge actually filled!
        # If partial fill: qty=5 sent, qty=2 filled
        # Leaves qty=3 STILL unhedged
```

### The Fix

**Recursive Hedge Monitoring**:

```python
class RecursiveHedgeMonitor:
    """Treat hedge execution as its own exposure to monitor."""
    
    def __init__(self):
        self.hedge_tracker = {}  # hedge_id → hedge details
        self.max_recursion = 3   # Don't go deeper than 3 hedge levels
    
    async def place_and_monitor_hedge(self, 
                                      symbol: str, 
                                      qty: int,
                                      leg_symbol: str,
                                      recursion_level: int = 0):
        """
        Place hedge and recursively monitor for partial fills.
        """
        
        if recursion_level >= self.max_recursion:
            logging.error(
                f"Max hedge recursion reached for {symbol}. "
                f"Manual intervention required."
            )
            await self._escalate_to_trader(symbol, qty)
            return
        
        # Place hedge
        hedge_id = str(uuid.uuid4())
        hedge_record = {
            'hedge_id': hedge_id,
            'symbol': symbol,
            'target_qty': qty,
            'filled_qty': 0,
            'recursion_level': recursion_level,
            'status': 'PENDING',
            'timestamp': time.time(),
            'related_leg': leg_symbol
        }
        self.hedge_tracker[hedge_id] = hedge_record
        
        try:
            # Place order
            response = await self.broker.place_order(
                symbol=symbol,
                qty=qty,
                order_type='MARKET'  # Use market for speed
            )
            broker_order_id = response['order_id']
            hedge_record['broker_order_id'] = broker_order_id
            
            # Wait for fill with timeout
            fill_event = await self._wait_for_fill(
                broker_order_id,
                timeout=5.0  # 5 second timeout for hedge
            )
            
            hedge_record['filled_qty'] = fill_event['fills'][-1]['qty'] \
                if fill_event['fills'] else 0
            hedge_record['status'] = 'FILLED'
            
            # Check if completely filled
            unfilled_qty = qty - hedge_record['filled_qty']
            
            if unfilled_qty > 0:
                logging.warning(
                    f"Hedge partial fill: wanted {qty}, got "
                    f"{hedge_record['filled_qty']}. "
                    f"{unfilled_qty} remaining."
                )
                
                # Recursively hedge the unfilled portion
                await self.place_and_monitor_hedge(
                    symbol=symbol,
                    qty=unfilled_qty,
                    leg_symbol=leg_symbol,
                    recursion_level=recursion_level + 1
                )
            
            return hedge_record
            
        except asyncio.TimeoutError:
            logging.error(f"Hedge {hedge_id} timeout after 5s")
            hedge_record['status'] = 'TIMEOUT'
            await self._escalate_to_trader(
                symbol, qty - hedge_record['filled_qty']
            )
            raise
    
    async def _wait_for_fill(self, broker_order_id: str, timeout: float):
        """
        Wait for hedge to fill with recursion support.
        Returns: fill event with actual quantity filled
        """
        start = time.time()
        
        while time.time() - start < timeout:
            fill_event = await self.broker.get_order_status(broker_order_id)
            
            if fill_event['status'] == 'FILLED':
                return fill_event
            
            await asyncio.sleep(0.1)
        
        raise asyncio.TimeoutError(
            f"Hedge order {broker_order_id} did not fill within {timeout}s"
        )
    
    async def _escalate_to_trader(self, symbol: str, unfilled_qty: int):
        """Alert trader to manual intervention."""
        alert = {
            'severity': 'CRITICAL',
            'type': 'HEDGE_FAILURE',
            'symbol': symbol,
            'unfilled_qty': unfilled_qty,
            'message': f"Emergency hedge failed. {unfilled_qty} shares "
                      f"of {symbol} remain unhedged.",
            'timestamp': time.time(),
            'action_required': 'IMMEDIATE_MANUAL_HEDGE'
        }
        
        await self.notify_risk_team(alert)
        # Store for manual recovery
        await self.store_escalation(alert)


class InterimExposureHandler:
    """Updated to use recursive hedge monitor."""
    
    def __init__(self, recursive_hedge_monitor: RecursiveHedgeMonitor):
        self.hedge_monitor = recursive_hedge_monitor
    
    async def check_exposure(self, leg: LegState):
        """Check if unhedged, recursively hedge if needed."""
        
        unhedged_qty = leg.target_qty - leg.filled_qty
        
        if unhedged_qty > 0:
            logging.warning(
                f"Interim exposure detected: {unhedged_qty} unhedged. "
                f"Placing recursive hedge."
            )
            
            # Use recursive monitor instead of simple place
            await self.hedge_monitor.place_and_monitor_hedge(
                symbol=leg.hedge_symbol,
                qty=unhedged_qty,
                leg_symbol=leg.symbol,
                recursion_level=0
            )
```

### Where to Add

**File**: [src/execution/order_group_corrected.py](src/execution/order_group_corrected.py)

**Location**: Add `RecursiveHedgeMonitor` class, update `InterimExposureHandler`

**Key Integration**:
```python
class CorrectMultiLegOrderGroup:
    def __init__(self, ...):
        self.hedge_monitor = RecursiveHedgeMonitor()
        self.interim_exposure = InterimExposureHandler(self.hedge_monitor)
```

---

## GAP #5: Clock Synchronization Risk 🔴

### The Problem

**Current Implementation** (WRONG):
```python
class HybridStopLossManager:
    async def execute_sl(self, ...):
        await self.place_limit_order(...)
        await asyncio.sleep(2)  # ← Local timer, can drift!
        
        # If system clock jumped backward OR
        # If event loop stalled on GC pause
        # This "2 second" timeout is NOT actually 2 seconds
```

### Why This Breaks

**Real-World Scenarios**:

1. **Clock Drift**: NTP adjustment (network sync)
   ```
   10:05:30.000  - Start timer
   10:05:31.950  - NTP adjusts clock backward 1 second
   10:05:31.950  - time.time() shows EARLIER
   10:05:32.950  - 2s await "completes" but only 1s elapsed
   Result: MARKET fallback fires too early
   ```

2. **Event Loop Stall**: GC pause or system load
   ```
   10:05:30.000  - Start timer
   10:05:31.500  - GC pause (event loop blocked)
   10:05:33.800  - Event loop resumes (2.8s real time passed)
   10:05:33.800  - asyncio.sleep(2) "completes"
   Result: Already filled, but fires MARKET anyway (double exit!)
   ```

### The Fix

**Use Broker Timestamps, Not Local Clocks**:

```python
class BrokerTimestampedTimeouts:
    """
    Use exchange timestamps for timeouts, not local clocks.
    
    Strategy:
    1. Record exchange timestamp when order sent
    2. Check exchange timestamp when checking status
    3. If (exchange_time - order_time) >= timeout: trigger fallback
    4. Never trust local clock
    """
    
    def __init__(self, broker_client):
        self.broker = broker_client
    
    async def execute_with_broker_timeout(self,
                                         broker_order_id: str,
                                         timeout_seconds: float):
        """
        Wait for fill using BROKER timestamps.
        
        Example:
        - Order sent at broker timestamp: 10:05:30.000
        - Timeout: 2 seconds
        - Check status every 100ms using broker timestamps
        - Trigger fallback when: broker_current_time - order_time >= 2s
        """
        
        # Get initial order status (includes broker timestamp)
        initial_status = await self.broker.get_order_status(broker_order_id)
        order_timestamp = initial_status['server_timestamp']
        
        logging.info(
            f"SL-LIMIT timeout started: "
            f"order_time={order_timestamp}, "
            f"timeout={timeout_seconds}s"
        )
        
        check_interval = 0.1  # Check every 100ms
        max_checks = int(timeout_seconds / check_interval)
        
        for check_num in range(max_checks):
            await asyncio.sleep(check_interval)
            
            # Get current status using broker timestamp
            current_status = await self.broker.get_order_status(
                broker_order_id
            )
            current_timestamp = current_status['server_timestamp']
            
            # Calculate elapsed using BROKER TIME
            broker_elapsed = current_timestamp - order_timestamp
            
            logging.debug(
                f"SL timeout check {check_num}: "
                f"broker_elapsed={broker_elapsed:.3f}s"
            )
            
            if current_status['status'] == 'FILLED':
                logging.info("SL-LIMIT filled, no fallback needed")
                return {'status': 'FILLED', 'fills': current_status['fills']}
            
            if broker_elapsed >= timeout_seconds:
                # Broker time says timeout expired
                logging.warning(
                    f"SL-LIMIT timeout expired after {broker_elapsed:.3f}s "
                    f"(broker time). Falling back to MARKET."
                )
                return await self._execute_market_fallback(broker_order_id)
        
        # Final check after loop
        final_status = await self.broker.get_order_status(broker_order_id)
        
        if final_status['status'] == 'FILLED':
            return {'status': 'FILLED', 'fills': final_status['fills']}
        
        # Timeout reached
        return await self._execute_market_fallback(broker_order_id)
    
    async def _execute_market_fallback(self, broker_order_id: str):
        """
        Cancel SL-LIMIT and place MARKET.
        IMPORTANT: Verify cancel succeeded before MARKET.
        """
        
        # Step 1: Request cancel
        cancel_result = await self.broker.cancel_order(broker_order_id)
        
        if not cancel_result['success']:
            logging.error(
                f"Cancel failed for {broker_order_id}. "
                f"Possible race condition (order filled)."
            )
            # Verify if it filled
            status = await self.broker.get_order_status(broker_order_id)
            if status['status'] == 'FILLED':
                return {'status': 'FILLED', 'fills': status['fills']}
            
            # Unclear state
            raise OrderStateError(
                f"Cannot cancel {broker_order_id} and uncertain if filled"
            )
        
        # Step 2: Was cancel actually confirmed?
        # (Some brokers have async cancel confirmation)
        cancel_confirmed = await self._verify_cancel_confirmation(
            broker_order_id,
            timeout=1.0  # Wait 1s for cancel confirmation
        )
        
        if not cancel_confirmed:
            logging.error(
                f"Cancel not confirmed for {broker_order_id}. "
                f"SL-LIMIT may fill while MARKET being placed."
            )
            # Wait additional 500ms to let fill complete
            await asyncio.sleep(0.5)
            
            final_status = await self.broker.get_order_status(broker_order_id)
            if final_status['status'] == 'FILLED':
                return {'status': 'FILLED', 'fills': final_status['fills']}
        
        # Step 3: Place MARKET (only if cancel confirmed)
        market_response = await self.broker.place_market_order(...)
        return {'status': 'MARKET_PLACED', 'order_id': market_response['id']}
    
    async def _verify_cancel_confirmation(self, 
                                         broker_order_id: str,
                                         timeout: float):
        """Poll broker for cancel confirmation using broker time."""
        
        cancel_request_time = await self.broker.get_server_time()
        
        while True:
            current_time = await self.broker.get_server_time()
            
            # Check using broker time only
            if current_time - cancel_request_time >= timeout:
                logging.warning("Cancel confirmation timeout")
                return False
            
            status = await self.broker.get_order_status(broker_order_id)
            
            if status['status'] in ['CANCELLED', 'CANCELED']:
                logging.info(f"Cancel confirmed for {broker_order_id}")
                return True
            
            if status['status'] == 'FILLED':
                logging.info(f"Order filled before cancel could execute")
                return False
            
            await asyncio.sleep(0.05)  # Poll every 50ms


class HybridStopLossManager:
    """Updated to use broker timestamps."""
    
    def __init__(self, broker_client, broker_timestamps: BrokerTimestampedTimeouts):
        self.broker = broker_client
        self.timestamped_timeout = broker_timestamps
    
    async def execute_sl(self, symbol: str, qty: int, 
                        trigger_price: float):
        """
        SL-LIMIT with broker-timestamp-based timeout.
        """
        
        # Place SL-LIMIT
        sl_limit_price = trigger_price * 1.005  # 0.5% slippage
        
        response = await self.broker.place_order(
            symbol=symbol,
            qty=qty,
            price=sl_limit_price,
            order_type='LIMIT'
        )
        
        broker_order_id = response['order_id']
        
        # Wait with broker-timestamped timeout (2 seconds)
        result = await self.timestamped_timeout.execute_with_broker_timeout(
            broker_order_id=broker_order_id,
            timeout_seconds=2.0  # 2 seconds of BROKER time
        )
        
        return result
```

### Where to Add

**File**: [src/execution/order_group_corrected.py](src/execution/order_group_corrected.py)

**Location**: Add `BrokerTimestampedTimeouts` class, update `HybridStopLossManager`

**Key Integration**:
```python
class CorrectMultiLegOrderGroup:
    def __init__(self, ..., broker_client):
        self.timestamped_timeout = BrokerTimestampedTimeouts(broker_client)
        self.sl_manager = HybridStopLossManager(
            broker_client, 
            self.timestamped_timeout
        )
```

---

## INTEGRATION SUMMARY

### Add to [src/execution/order_group_corrected.py](src/execution/order_group_corrected.py):

1. **ExecutionStateValidator** (60 lines)
   - Add after imports
   - Validate state transitions are monotonic

2. **IdempotencyManager** (120 lines)
   - Add after ExecutionStateValidator
   - Track and deduplicate events

3. **OrderIntentPersistence** (140 lines)
   - Add after IdempotencyManager
   - Persist intent BEFORE order placement

4. **RecursiveHedgeMonitor** (180 lines)
   - Add after OrderIntentPersistence
   - Handle partial field hedges recursively

5. **BrokerTimestampedTimeouts** (160 lines)
   - Add after RecursiveHedgeMonitor
   - Use exchange timestamps for timeouts

### Update Existing Classes:
- `CorrectMultiLegOrderGroup.__init__()` - Add all 5 managers
- `_process_state_update()` - Add state validation + idempotency check
- `_place_order_safely()` - Use OrderIntentPersistence
- `InterimExposureHandler` - Use RecursiveHedgeMonitor
- `HybridStopLossManager` - Use BrokerTimestampedTimeouts

### Update [MIGRATION_GUIDE_FIX1.md](MIGRATION_GUIDE_FIX1.md):
- Add Part 9: "5 Critical Event Ordering & Atomicity Safeguards"
- Include code implementations
- Update integration timeline (add 8 hours for these fixes)

### Update [tests/test_critical_fixes.py](tests/test_critical_fixes.py):
- Add `TestEventOrdering` (6 tests)
- Add `TestIdempotency` (4 tests)
- Add `TestPersistenceAtomicity` (4 tests)
- Add `TestRecursiveHedge` (4 tests)
- Add `TestBrokerTimestamps` (4 tests)
- Total: 22 new tests

---

## DATABASE SCHEMA ADDITIONS

```sql
-- For idempotency
CREATE TABLE IF NOT EXISTS processed_events (
    event_id TEXT PRIMARY KEY,
    group_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_group (group_id),
    INDEX idx_time (timestamp)
);

-- For order intent persistence
CREATE TABLE IF NOT EXISTS order_intents (
    intent_id TEXT PRIMARY KEY,
    group_id TEXT NOT NULL,
    leg_id TEXT NOT NULL,
    order_type TEXT NOT NULL,
    symbol TEXT NOT NULL,
    quantity REAL NOT NULL,
    price REAL,
    status TEXT,
    broker_order_id TEXT,
    timestamp REAL,
    sent_timestamp REAL,
    filled_qty REAL DEFAULT 0,
    fills BLOB,
    FOREIGN KEY (group_id) REFERENCES execution_states(group_id),
    INDEX idx_group (group_id),
    INDEX idx_status (status)
);

-- For hedge tracking
CREATE TABLE IF NOT EXISTS hedge_tracking (
    hedge_id TEXT PRIMARY KEY,
    related_leg TEXT NOT NULL,
    symbol TEXT NOT NULL,
    target_qty REAL NOT NULL,
    filled_qty REAL DEFAULT 0,
    recursion_level INTEGER DEFAULT 0,
    status TEXT,
    broker_order_id TEXT,
    timestamp REAL,
    INDEX idx_leg (related_leg)
);
```

---

## UPDATED TIMELINE

**Previous Estimate**: 58 hours for 7 fixes  
**NEW Estimate for 5 Additional Safeguards**: +16 hours

| Phase | Time | New Total |
|-------|------|-----------|
| Code Implementation | 8 hrs | **66 hrs** |
| Testing | +4 hrs | **70 hrs** |
| Documentation | +2 hrs | **72 hrs** |
| Integration | +2 hrs | **74 hrs** |
| **TOTAL** | **+16 hrs** | **74 hours (~2 weeks)** |

---

## CRITICAL SIGN-OFF

### Current Status: ⚠️ **NOT PRODUCTION-READY**

**Previous Assessment**: "95% confidence for 7 fixes"  
**REVISED Assessment**: "Cannot deploy without these 5 additional safeguards"

These are not optimizations. They are **blocking defects** that will cause:
- Double executions from duplicate events
- Hedge failures from liquidity issues
- Orphaned orders from crashes
- Incorrect timeouts from clock drifts
- State machine regressions from out-of-order events

---

## NEXT STEPS

1. **Immediately** integrate these 5 safeguards into order_group_corrected.py
2. **Update** all 3 review documents (COMPREHENSIVE_REVIEW, MIGRATION_GUIDE, DEPLOYMENT_CHECKLIST)
3. **Rerun** tests with 22 new test cases
4. **Extend** deployment timeline by 2 weeks

**Revised Status**: Ready for production only AFTER these 5 fixes are integrated and tested.

---

**Prepared By**: Production Safety Analysis  
**Date**: February 21, 2026  
**Status**: 🚨 BLOCKING ISSUE - Deployment cannot proceed
