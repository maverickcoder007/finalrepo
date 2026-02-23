# CRITICAL REVIEW: Fix #1 Rollback Model is BROKEN

**Date**: February 21, 2026  
**Status**: ⚠️ PRODUCTION BLOCKER IDENTIFIED  
**Severity**: CRITICAL - Current implementation WILL fail in production

---

## Executive Summary

The original Fix #1 (Multi-Leg Order Groups) has **fundamental architectural flaws** that make it dangerous in production:

1. **Rollback via Cancellation is IMPOSSIBLE** - Filled orders cannot be cancelled
2. **Partial Fill Timing** - 8-second delays create unhedged exposure windows
3. **Stop-Loss Fix is Incomplete** - Gap-through scenarios not handled
4. **Margin Shock Risk** - Pre-execution margin checks don't predict hedge margin spikes
5. **Missing State Machine** - No way to recover after crashes
6. **Polling is Too Slow** - 30-second intervals = 30-second unhedged holding periods

**This document provides the CORRECTED models.**

---

## 1. The Rollback Myth: Cancellation vs Reversal

### ❌ WRONG (What I Provided)

```
Execution sequence:
  Leg 1: SELL 50 NIFTY calls → FILLED ✓
  Leg 2: BUY 50 NIFTY calls  → FAILED ✗

Rollback action:
  Cancel Leg 1
  
Reality:
  ***CANNOT CANCEL A FILLED ORDER***
  Exchange has final state.
  Order cannot be reversed by cancellation.
```

### ✅ CORRECT: Emergency Hedge Executor

```
Execution sequence:
  Leg 1: SELL 50 NIFTY calls @ 100 → FILLED ✓ 
         (Position: SHORT 50 calls, naked, unlimited loss)
  Leg 2: BUY 50 NIFTY calls @ 101   → REJECTED ✗

Rollback action:
  Emergency Hedge Executor triggers
  └─ IMMEDIATELY place: BUY 50 NIFTY calls @ MARKET
     └─ Speed necessary to close within ms, not seconds
  
Result:
  SHORT 50 + BUY 50 = Neutral (0 exposure)
  Loss = 50 * (100 - market_fill_price)
  
  Example: Filled at ₹99 = 50 * (100 - 99) = ₹50 profit
  Example: Filled at ₹102 = 50 * (100 - 102) = -₹100 loss
```

### Correct Implementation Pattern

```python
class MultiLegOrderGroup:
    """Real atomic model: rollback via REVERSE TRADES."""
    
    async def execute_with_hedge_recovery(self, execution_engine):
        """
        Execute all legs with hedge-first strategy.
        
        If any leg fails:
        1. Do NOT cancel filled legs (impossible)
        2. DO immediately hedge all filled legs with reverse trades
        3. Log emergency hedge execution
        4. Return FAILED status with hedge activity
        """
        placed_orders = []
        
        try:
            # CRUCIAL: Place protection (hedge) legs FIRST
            # This is the professional order:
            protection_legs = [leg for leg in self.legs if leg.is_protection]
            risk_legs = [leg for leg in self.legs if not leg.is_protection]
            
            # Step 1: Place all protection legs first
            for leg in protection_legs:
                order_id = await execution_engine.execute_signal(leg.signal)
                placed_orders.append(("protection", leg, order_id))
            
            # Step 2: Verify protection filled
            await asyncio.sleep(0.5)
            protection_filled = all(
                await self._check_leg_filled(leg) 
                for _, leg, _ in placed_orders
            )
            
            if not protection_filled:
                # Protection failed - don't proceed with risk
                await self._cancel_all_protection(placed_orders, execution_engine)
                return False
            
            # Step 3: Now place risk legs (SHORT calls/puts)
            for leg in risk_legs:
                order_id = await execution_engine.execute_signal(leg.signal)
                placed_orders.append(("risk", leg, order_id))
            
            # Step 4: Verify all risk legs filled
            await asyncio.sleep(0.5)
            all_filled = all(
                await self._check_leg_filled(leg)
                for _, leg, _ in placed_orders
            )
            
            if all_filled:
                self.status = OrderGroupStatus.COMPLETE
                return True
            
            # PARTIAL FILL: Emergency hedge execution
            return await self._emergency_hedge_recover(placed_orders, execution_engine)
        
        except Exception as e:
            logger.error("execution_error", error=str(e))
            return await self._emergency_hedge_recover(placed_orders, execution_engine)
    
    async def _emergency_hedge_recover(self, placed_orders, execution_engine):
        """
        When partial fills occur, immediately reverse filled legs.
        This creates hedged flat positions, locking in losses.
        """
        logger.critical("emergency_hedge_triggered", group_id=self.group_id)
        
        for leg_type, leg, order_id in placed_orders:
            # Check if this leg was filled
            filled_qty = await self._check_filled_quantity(order_id, execution_engine)
            
            if filled_qty > 0:
                # Create reverse signal
                reverse_signal = self._create_reverse_signal(leg.signal, filled_qty)
                
                try:
                    # Place reverse trade at MARKET (speed critical)
                    reverse_order_id = await execution_engine.execute_signal_immediate(
                        reverse_signal,
                        order_type=OrderType.MARKET
                    )
                    
                    logger.critical("emergency_hedge_placed",
                                   original_order_id=order_id,
                                   reverse_order_id=reverse_order_id,
                                   quantity=filled_qty)
                
                except Exception as e:
                    logger.critical("emergency_hedge_failed",
                                   original_order_id=order_id,
                                   quantity=filled_qty,
                                   error=str(e))
        
        self.status = OrderGroupStatus.FAILED
        return False
    
    def _create_reverse_signal(self, original_signal: Signal, quantity: int) -> Signal:
        """Create opposite signal to hedge filled position."""
        return Signal(
            tradingsymbol=original_signal.tradingsymbol,
            exchange=original_signal.exchange,
            transaction_type=(
                TransactionType.BUY if original_signal.transaction_type == TransactionType.SELL
                else TransactionType.SELL
            ),
            quantity=quantity,
            order_type=OrderType.MARKET,  # Speed over price
            strategy_name="emergency_hedge",
            metadata={"hedge_for": original_signal.strategy_name}
        )
```

---

## 2. Partial Fill Timing Risk: The 8-Second Window

### The Reality

```
T+0ms:   Place Leg 1: SELL 50 NIFTY calls
         └─ Network latency: ~5ms
T+5ms:   Arrives at OMS
         └─ OMS batching: 10ms
T+15ms:  Arrives at exchange
         └─ Matching engine: 50ms
T+65ms:  FILLED ✓

T+100ms: Place Leg 2: BUY 50 NIFTY calls
T+500ms: STILL PENDING (exchange queueing, volatility, liquidity)
T+2000ms: STILL PENDING
T+8000ms: Finally filled OR rejected

Between T+65ms - T+8000ms:
  Market condition: Spot moved +500 points
  Short call exposure: NAKED for 8 seconds
  P&L swing: ±₹25,000 on 50 qty
  
Your system:
  ✓ Sees Leg 1 filled
  ✗ Doesn't know Leg 2 status
  ✗ Doesn't know it's unhedged
  ✗ Can't react to market moves
```

### ✅ SOLUTION: Interim Exposure Tracking

```python
@dataclass
class ExecutionState:
    """Fine-grained tracking of multi-leg execution."""
    
    group_id: str
    status: str  # CREATED, SUBMITTED, PARTIALLY_EXPOSED, HEDGED, COMPLETE, FAILED
    
    legs: dict[str, "LegState"] = field(default_factory=dict)
    
    # CRITICAL: Track interim exposure
    interim_exposure: dict[str, int] = field(default_factory=dict)
    # Example: {"NIFTY": -50}  (short 50 NIFTY calls, no protection yet)
    
    exposed_duration_ms: float = 0.0
    hedge_placed_at: Optional[datetime] = None
    
    def get_interim_delta(self) -> float:
        """Calculate net delta while partially filled."""
        total_delta = 0.0
        for symbol, quantity in self.interim_exposure.items():
            # SHORT call: delta = -1 * quantity
            # LONG call: delta = +1 * quantity
            # etc.
            total_delta += quantity * self._get_delta(symbol)
        return total_delta


class PartialFillHandler:
    """Handles unhedged windows between fills."""
    
    async def monitor_interim_exposure(self, group_state: ExecutionState):
        """
        While group is PARTIALLY_EXPOSED, immediately place hedging.
        
        Example:
          SHORT 50 calls (unhedged)
          └─ Need BUY protection immediately
          └─ Even if other leg hasn't arrived yet
        """
        while group_state.status == "PARTIALLY_EXPOSED":
            interim_delta = group_state.get_interim_delta()
            
            if abs(interim_delta) > self.delta_limit:
                # Unhedged exposure too large
                protect_signal = self._create_delta_hedge(interim_delta)
                
                logger.warning("interim_exposure_hedge",
                              group_id=group_state.group_id,
                              delta=interim_delta,
                              action="place_hedge")
                
                await self.execution_engine.execute_signal(protect_signal)
            
            await asyncio.sleep(0.1)  # Check every 100ms, not 30s
```

---

## 3. Stop-Loss Fix is Incomplete: Gap-Through Risk

### ❌ PROBLEM: SL-LIMIT Order Never Fills on Gap

```
Setting:
  BUY @ 100
  SL @ 95 with limit @ 94.5

Market sequence:
  T=100ms: Price 100.5 (normal)
  T=200ms: Price 98.0  (moving down)
  T=250ms: Price 95.5  (trigger level)
  T=300ms: Price 90.0  (GAP DOWN - circuit break, news)
  
What happens:
  SL triggered at 95.5
  Limit order placed: "Fill at 94.5 or better"
  
  But price is now 90.0
  Order sits unfilled because:
    - Limit 94.5 is WORSE than market 90.0?
    - NO, limit 94.5 is BETTER (for SELL)
    - Order SHOULD fill at 90.0
  
  BUT in reality:
    - Broker blocks the order (circuit limit low)
    - Or liquidity vanishes (circuit break)
    - Order NEVER fills
  
  Position still OPEN:
    - Still holding BUY @ 100
    - No fill at 94.5
    - Now trading at 80
    - Loss: ₹20 instead of expected ₹5

This is WORSE than no stop-loss.
```

### ✅ SOLUTION: Hybrid Stop Logic

```python
class HybridStopLossManager:
    """SL-LIMIT primary + TIMEOUT fallback."""
    
    async def execute_with_hybrid_stop(self, signal: Signal):
        """
        Place SL-LIMIT primary.
        If gap-through occurs, convert to MARKET fallback.
        """
        # Primary: SL_LIMIT
        sl_limit_order = OrderRequest(
            traditionsymbol=signal.tradingsymbol,
            order_type=OrderType.SL_LIMIT,
            trigger_price=signal.stop_loss,
            price=signal.stop_loss * (1 - self.slippage / 100),  # Limit protection
        )
        
        order_id = await self.client.place_order(sl_limit_order)
        
        # Track when it was placed
        order_placed_at = datetime.now()
        
        # Monitor for fill
        while True:
            order = await self.client.get_order(order_id)
            
            if order.status == "COMPLETE":
                # Filled at limit price
                logger.info("sl_limit_filled", avg_price=order.average_price)
                return order
            
            if order.status == "REJECTED":
                # Limit price too far from market - convert to MARKET
                logger.warning("sl_limit_rejected_attempting_market")
                return await self._fallback_to_market(signal)
            
            # Check if gap-through occurred
            # (price moved beyond SL without fill)
            current_price = await self.client.get_ltp(signal.tradingsymbol)
            
            # For BUY position with SL at 95:
            # If current_price < 94 and order still OPEN = gap-through
            if self._is_gap_through(signal, order, current_price):
                logger.critical("gap_through_detected",
                               stop_loss=signal.stop_loss,
                               current_price=current_price)
                
                # Cancel SL-LIMIT and execute MARKET immediately
                await self.client.cancel_order(order_id)
                return await self._fallback_to_market(signal)
            
            # Check timeout (2 seconds is reasonable for SL-LIMIT)
            elapsed_ms = (datetime.now() - order_placed_at).total_seconds() * 1000
            if elapsed_ms > 2000:  # 2 second timeout
                logger.warning("sl_limit_timeout_converting_market",
                              elapsed_ms=elapsed_ms)
                await self.client.cancel_order(order_id)
                return await self._fallback_to_market(signal)
            
            await asyncio.sleep(0.01)  # Poll every 10ms
    
    async def _fallback_to_market(self, signal: Signal):
        """Execute MARKET order if SL-LIMIT fails."""
        market_order = OrderRequest(
            traditionsymbol=signal.tradingsymbol,
            order_type=OrderType.MARKET,
            quantity=signal.quantity,
        )
        return await self.client.place_order(market_order)
    
    def _is_gap_through(self, signal: Signal, order, current_price: float) -> bool:
        """Detect if price gapped past stop-loss without triggering."""
        if signal.transaction_type == TransactionType.BUY:
            # BUY position SL at 95
            # Gap-through if current < 94 and order still open
            return current_price < signal.stop_loss - 1 and order.status == "OPEN"
        else:
            # SELL position SL at 105
            # Gap-through if current > 106 and order still open
            return current_price > signal.stop_loss + 1 and order.status == "OPEN"
```

---

## 4. Margin Shock Risk: Projected Worst-Case Simulator

### ❌ PROBLEM: Pre-Execution Check Inadequate

```
Pre-execution state:
  Available margin: ₹100,000
  
Margin calculator checks:
  "SELL 50 NIFTY calls @ 100"
  Estimated margin: ₹50,000
  ✓ Available 100,000 > Required 50,000
  
Order placed - but partial fill happens:

After Leg 1 FILLED (SHORT 50 calls):
  Position: SHORT 50 calls @ 100
  Margin required NOW: ₹50,000 (initial margin)
  ✓ Still safe
  
Leg 2 PENDING (haven't bought protection yet):
  Market moves: NIFTY +300 points
  
Call value now: ₹400 (was ₹100)
  
Margin required SPIKED:
  Original position margin: ₹50,000
  Unrealised loss: ₹150,000 (50 * 300)
  Add-on margin: ₹75,000 (50% of span)
  Total needed: ₹225,000
  
Available margin: ₹100,000
  
Broker auto square-off:
  ✗ Force close SHORT 50 calls @ ₹400
  ✗ Loss: -₹150,000
  ✗ Reversed because protection never arrived
```

### ✅ SOLUTION: Worst-Case Margin Simulator

```python
class MarginShockPredictor:
    """Simulate worst-case margin requirement scenario."""
    
    async def validate_multi_leg_margin(self, legs: list[Signal]):
        """
        Before placing ANY order, simulate worst-case scenario
        where first leg fills but others are pending for N seconds.
        """
        # Get current available margin
        margins = await self.client.get_margins()
        initial_available = margins.equity.available.collateral
        
        # For each possible single-leg fill state:
        for i, leg in enumerate(legs):
            # Simulate: This leg gets filled, others are pending
            await self._simulate_worst_case_fill(
                filled_leg=leg,
                other_legs=legs[:i] + legs[i+1:],
                initial_available=initial_available
            )
    
    async def _simulate_worst_case_fill(self, filled_leg, other_legs, initial_available):
        """
        Simulate: 
        - Filled leg is now in-the-money
        - Market moved dramatically against us
        - Protection legs still pending
        """
        # 1. Calculate margin for filled leg
        leg_margin = self._estimate_margin(filled_leg)
        
        # 2. Simulate worst market move (e.g., +3% for SHORT calls)
        price_move = self._get_worst_case_move(filled_leg)
        unrealized_loss = filled_leg.quantity * price_move * self.price_shock_pct
        
        # 3. Calculate add-on margin (broker requirement on losses)
        addon_margin = abs(unrealized_loss) * 0.5  # 50% of loss
        
        # 4. Calculate total required
        total_required = leg_margin + addon_margin
        
        # 5. Check if we still have margin
        if total_required > initial_available:
            raise RiskLimitError(
                f"Margin shock risk: {filled_leg.tradingsymbol} fills,"
                f"market moves {price_move}%, "
                f"margin required {total_required}, "
                f"available {initial_available}"
            )
        
        logger.info("margin_shock_check_passed",
                   leg=filled_leg.tradingsymbol,
                   worst_case_margin=total_required,
                   available=initial_available)
```

---

## 5. Exchange Freeze Quantities: Child Order Tracking

### Problem: Large Orders Split Automatically

```
Order: SELL 100 NIFTY calls

Exchange freeze quantity: 20 (max per order)

Exchange splits into:
  Order A: 20 calls
  Order B: 20 calls
  Order C: 20 calls
  Order D: 20 calls
  Order E: 20 calls

If each fills separately:
  T+100ms: Order A filled (20)
  T+200ms: Order B filled (20)
  T+300ms: Order C PENDING
  T+400ms: Order D filled (20)
  T+500ms: Order E REJECTED (circuit limit)

Your system sees:
  Total filled: 80
  Total rejected: 20

But the 80 filled = UNHEDGED for seconds while waiting for protection orders.
```

### Solution: Child Order Tracking

```python
class ChildOrderTracker:
    """Track exchange-split sub-orders."""
    
    async def execute_with_child_tracking(self, signal: Signal):
        """
        Place order, detect if exchange splits it,
        track each child order separately.
        """
        # Get exchange freeze quantity
        freeze_qty = self._get_freeze_quantity(signal.traditionsymbol)
        
        # Calculate child orders needed
        child_orders = self._split_into_children(signal, freeze_qty)
        
        placed_children = []
        for i, child_signal in enumerate(child_orders):
            order_id = await self.client.place_order(child_signal)
            
            placed_children.append({
                "order_id": order_id,
                "quantity": child_signal.quantity,
                "status": "SUBMITTED",
                "filled": 0,
                "child_index": i,
            })
        
        # Track each child separately
        while True:
            total_filled = 0
            pending_children = 0
            
            for child in placed_children:
                order = await self.client.get_order(child["order_id"])
                child["filled"] = order.filled_quantity
                
                if order.status == "COMPLETE":
                    child["status"] = "FILLED"
                    total_filled += child["filled"]
                elif order.status == "OPEN" or order.status == "PENDING":
                    pending_children += 1
                elif order.status == "REJECTED":
                    child["status"] = "REJECTED"
            
            logger.debug("child_order_progress",
                        total_filled=total_filled,
                        pending=pending_children,
                        total_expected=signal.quantity)
            
            if pending_children == 0:
                break
            
            await asyncio.sleep(0.05)  # Check every 50ms
        
        return total_filled
```

---

## 6. Missing State Machine: Execution Lifecycle

### ✅ Correct State Model

```python
class ExecutionLifecycle(str, Enum):
    """All possible states during execution."""
    
    CREATED = "CREATED"                 # Initial state
    VALIDATED = "VALIDATED"             # Passed risk checks
    SUBMITTED = "SUBMITTED"             # Orders placed at broker
    PARTIALLY_FILLED = "PARTIALLY_FILLED"  # Some legs filled, others pending
    PARTIALLY_EXPOSED = "PARTIALLY_EXPOSED"  # Interim unhedged window
    FILLED = "FILLED"                   # All legs filled
    HEDGED = "HEDGED"                   # Protection orders confirmed
    CLOSED = "CLOSED"                   # Position closed cleanly
    FAILED = "FAILED"                   # Execution failed
    EMERGENCY_HEDGED = "EMERGENCY_HEDGED"  # Reversed positions due to failure


async def execute_with_state_machine(group: MultiLegOrderGroup):
    """
    Execution with persistent state tracking.
    Can restart/recover from any state.
    """
    group.state = ExecutionLifecycle.CREATED
    
    try:
        # VALIDATE
        group.state = ExecutionLifecycle.VALIDATED
        for leg in group.legs:
            if not risk_manager.validate_signal(leg.signal):
                raise RiskLimitError("Validation failed")
        
        # SUBMIT
        group.state = ExecutionLifecycle.SUBMITTED
        for leg in group.legs:
            leg.order_id = await execution_engine.execute_signal(leg.signal)
        
        # TRACK PARTIAL FILLS
        await asyncio.sleep(0.5)
        filled_count = sum(1 for leg in group.legs if await _is_filled(leg))
        
        if 0 < filled_count < len(group.legs):
            group.state = ExecutionLifecycle.PARTIALLY_FILLED
            group.state = ExecutionLifecycle.PARTIALLY_EXPOSED
            
            # Start interim exposure hedge
            await partial_fill_handler.hedge_interim(group)
        
        # WAIT FOR FULL FILL
        while True:
            all_filled = all(await _is_filled(leg) for leg in group.legs)
            if all_filled:
                group.state = ExecutionLifecycle.FILLED
                break
            
            if await _is_failed(group):
                group.state = ExecutionLifecycle.FAILED
                await _emergency_hedge_recover(group)
                group.state = ExecutionLifecycle.EMERGENCY_HEDGED
                return False
            
            await asyncio.sleep(0.1)
        
        # VERIFY HEDGE
        group.state = ExecutionLifecycle.HEDGED
        
        # CLOSE
        group.state = ExecutionLifecycle.CLOSED
        
        # PERSIST state to DB
        await persistence_layer.save_execution_state(group)
        
        return True
    
    except Exception as e:
        group.state = ExecutionLifecycle.FAILED
        await _emergency_hedge_recover(group)
        group.state = ExecutionLifecycle.EMERGENCY_HEDGED
        await persistence_layer.save_execution_state(group)
        return False
```

---

## 7. Event-Driven Architecture (Not Polling)

### ❌ CURRENT: 30-Second Polling

```
While _running:
  sleep(30 seconds)
  get_positions()
  update_pnl()
  check_kill_switch()

Problem:
  - 30-second delay before detecting problems
  - In that time: market moves ±500 points
  - Position exposure completely changes
  - Kill switch triggers too late
```

### ✅ CORRECT: Event-Driven

```python
class EventDrivenExecution:
    """React instantly to broker events."""
    
    def __init__(self, broker_ws):
        self.broker_ws = broker_ws
        self.event_bus = asyncio.Queue()
        self.subscribers = {}
    
    async def start_event_loop(self):
        """Start event dispatcher."""
        # Listen to broker WebSocket events
        async for event in self.broker_ws.listen():
            # Publish to event bus
            await self.event_bus.put(event)
            
            # Route to appropriate handler
            await self._dispatch_event(event)
    
    async def _dispatch_event(self, event):
        """Route events instantly to handlers."""
        event_type = event["type"]
        
        if event_type == "order_update":
            # React to order fills INSTANTLY
            await self._on_order_update(event)
            
            # Check if fills changed execution state
            await self._check_execution_state()
            
            # Check if we're now unhedged
            await self._check_interim_exposure()
            
            # Update risk manager instantly
            await self._update_risk_manager()
        
        elif event_type == "position_update":
            # Broker sent position change
            await self._on_position_update(event)
            
            # Check kill switch threshold
            if self._check_kill_switch():
                await self._execute_kill_switch()
        
        elif event_type == "margin_update":
            # Margin changed
            if event["margin"] < self.min_margin_threshold:
                logger.critical("margin_critical")
                await self._liquidate_positions()
    
    async def subscribe(self, event_type: str, handler):
        """Subscribe handler to event type."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
```

**Timing Comparison:**

```
Event-Driven:
  T+0: Event received at broker WS
  T+5ms: Dispatch to handler
  T+10ms: Execution state updated
  T+15ms: Risk check complete
  T+20ms: If needed, hedge placed
  
  TOTAL: 20ms reaction time

Polling (30 seconds):
  T+0: Event at broker
  T+30,000ms: Next poll check
  T+30,050ms: Handler runs
  
  DELAY: 30 seconds = catastrophic
```

---

## 8. Strategy ≠ Sizing: Separation of Concerns

### ❌ WRONG: Strategy Controls Quantity

```python
class EMAStrategy(BaseStrategy):
    async def on_tick(self, ticks):
        if ema50 > ema200:
            signal = Signal(
                traditionsymbol="NIFTY",
                quantity=50,  # ❌ Strategy shouldn't decide this
                price=ltp,
            )
            return [signal]
```

**Problem:**
- Strategy doesn't know available margin
- Strategy doesn't know portfolio exposure
- Strategy doesn't know volatility regime
- Strategy might size 50 when risk allows only 2

### ✅ CORRECT: RiskManager Controls Sizing

```python
class EMAStrategy(BaseStrategy):
    async def on_tick(self, ticks):
        if ema50 > ema200:
            signal_intent = Signal(
                traditionsymbol="NIFTY",
                quantity=1,  # Placeholder, will be sized
                price=ltp,
                metadata={"intent": "BUY", "confidence": 0.85}
            )
            return [signal_intent]


class RiskManager:
    async def size_signal(self, signal_intent: Signal) -> Signal:
        """Calculate actual quantity based on risk parameters."""
        
        # Get portfolio state
        margin_available = await self.client.get_margins()
        portfolio_delta = self.position_tracker.get_portfolio_delta()
        portfolio_vega = self.position_tracker.get_portfolio_vega()
        
        # Sizing based on:
        # 1. Available margin
        max_qty_by_margin = self._calculate_max_qty_by_margin(
            signal_intent.traditionsymbol,
            margin_available.equity.available.collateral
        )
        
        # 2. Portfolio delta limit (e.g., ±200)
        max_qty_by_delta = self._calculate_max_qty_by_delta_limit(
            signal_intent,
            target_delta=200,
            current_delta=portfolio_delta
        )
        
        # 3. Daily loss limit (avoid sizing if losing)
        if self.daily_loss < -self.daily_loss_limit / 2:
            max_qty_by_loss = int(max_qty_by_margin * 0.25)  # Reduce 75%
        else:
            max_qty_by_loss = max_qty_by_margin
        
        # 4. Volatility regime (size down in high volatility)
        vol_multiplier = self._get_volatility_multiplier()
        
        # Calculate final size
        final_qty = min(
            max_qty_by_margin,
            max_qty_by_delta,
            max_qty_by_loss,
        ) * vol_multiplier
        
        signal = signal_intent.copy()
        signal.quantity = int(final_qty)
        
        return signal
```

---

## 9. Architectural Correction: Execution Coordinator

### New Layer: Between Strategy & Execution

```
Strategy Engine
    ↓ (direction only: BUY/SELL)
    
Signal Bus
    ↓
    
Risk Engine (portfolio-level) ⭐
    ├─ Validate signal scope
    ├─ Size position
    └─ Check margin
    ↓
    
Execution Coordinator (NEW) ⭐⭐⭐
    ├─ Group orchestration (multi-leg)
    ├─ Interim exposure tracking
    ├─ Partial fill handling
    ├─ Emergency hedge logic
    ├─ State machine persistence
    └─ Retry logic
    ↓
    
Execution Engine
    ├─ Place orders
    ├─ Reconcile fills
    └─ Track order status
    ↓
    
Broker Adapter
    └─ REST + WebSocket
    ↓
    
Event Stream (Instant)
    └─ Order updates, Position updates
    ↓
    
State Machine Updater
    └─ ExecutionLifecycle tracking
    ↓
    
Position Engine
    ├─ Track P&L
    ├─ Track delta
    └─ Monitor hedges
    ↓
    
Persistence Layer (SQLite)
    ├─ orders table
    ├─ order_groups table
    ├─ fills table
    ├─ positions table
    └─ execution_events table
```

---

## 10. Production-Ready Implementation Plan

### Phase 1: CRITICAL (Week 1) - Must Fix Before Live

- [ ] Replace cancellation with hedge-first model
- [ ] Implement Emergency Hedge Executor
- [ ] Add SL limit + timeout fallback (hybrid)
- [ ] Add Partial Fill Handler for interim exposure
- [ ] Implement Execution State Machine

### Phase 2: IMPORTANT (Week 2) - Before Multi-Leg Trading

- [ ] Add Margin Shock Predictor
- [ ] Implement Event-Driven Architecture
- [ ] Separate strategy from sizing (RiskManager)
- [ ] Add Execution Coordinator layer
- [ ] Switch from JSON to SQLite persistence

### Phase 3: NICE-TO-HAVE (Week 3)

- [ ] Add child order tracking for freeze quantities
- [ ] Portfolio delta/vega limits
- [ ] Advanced kill switch (close + cancel)
- [ ] Volatility-based sizing multiplier

### Phase 4: PROFESSIONAL (Week 4+)

- [ ] Hedge-first execution order (protection before risk)
- [ ] Greeks-based hedging (auto-delta hedge)
- [ ] Circuit breaker detection
- [ ] Slippage attribution analysis

---

## 11. Summary: What Was Wrong & What's Fixed

| Issue | Original Fix #1 | Corrected Model |
|-------|-----------------|-----------------|
| **Rollback** | Cancel filled orders ❌ | Reverse trade hedge ✅ |
| **Partial Fills** | Ignored interim exposure ❌ | Track & hedge immediately ✅ |
| **Stop-Loss** | SL-LIMIT only ❌ | Hybrid SL + timeout ✅ |
| **Margin Shock** | Pre-check only ❌ | Worst-case simulator ✅ |
| **State Tracking** | No recovery ❌ | State machine persistence ✅ |
| **Speed** | 30s polling ❌ | Event-driven instant ✅ |
| **Child Orders** | Not tracked ❌ | Child tracking enabled ✅ |
| **Strategy Sizing** | Built into strategy ❌ | RiskManager controls ✅ |

---

**Status**: ⚠️ CRITICAL - Original Fix #1 requires comprehensive redesign before production use

**Next Step**: Implement Phase 1 fixes before trading any multi-leg strategies
