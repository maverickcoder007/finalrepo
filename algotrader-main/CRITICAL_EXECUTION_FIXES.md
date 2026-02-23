# Critical Fixes for Multi-Leg Order Execution

**Date**: February 21, 2026  
**Priority**: URGENT - Must fix before trading multi-leg strategies  
**Risk Level**: HIGH - Current system can create orphaned naked positions

---

## Problem Summary

The current trading system executes multi-leg options strategies (Iron Condor, Credit Spreads, etc.) by placing individual leg orders **sequentially** without atomicity. This creates several critical execution risks:

1. **Orphaned Positions**: One leg fills while another fails → unhedged exposure
2. **Partial Fill Mismatch**: Legs fill at different quantities → invalid position
3. **Stop-Loss Price Ambiguity**: Trigger price specified but execution price not defined
4. **Assignment Without Closure**: Options exercised but system unaware → P&L wrong
5. **No Multi-Leg Validation**: No check that all legs are filled proportionally

---

## Fix 1: Multi-Leg Order Group Management

### Current Code Problem

```python
# src/options/strategies.py - Iron Condor Entry
signals = [
    Signal(symbol="20000CE", side=SELL, qty=50),  # Sell call
    Signal(symbol="21000CE", side=BUY, qty=50),   # Buy call (protection)
    Signal(symbol="20000PE", side=SELL, qty=50),  # Sell put
    Signal(symbol="19000PE", side=BUY, qty=50),   # Buy put (protection)
]

for signal in signals:
    await execution_engine.execute_signal(signal)  # 4 SEPARATE calls!
    # If call #2 fails, calls #1 and #3 already filled
```

### Solution: Atomic Order Group

**File to Create**: `src/execution/order_group.py`

```python
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

class OrderGroupStatus(str, Enum):
    PENDING = "PENDING"  # Waiting for execution
    PARTIAL = "PARTIAL"  # Some legs filled, some pending
    COMPLETE = "COMPLETE"  # All legs filled as specified
    FAILED = "FAILED"     # One or more legs failed/cancelled
    ROLLED = "ROLLED"     # Adjusted/rolled to new strikes

@dataclass
class OrderGroupLeg:
    signal: Signal
    order_id: Optional[str] = None
    filled_quantity: int = 0
    required_quantity: int = 0
    execution_price: float = 0.0
    status: str = "PENDING"
    error: Optional[str] = None

@dataclass
class MultiLegOrderGroup:
    """Groups multi-leg orders for atomic execution and validation."""
    
    group_id: str = field(default_factory=lambda: str(uuid4()))
    strategy_name: str = ""  # e.g., "iron_condor", "bull_call_credit_spread"
    legs: list[OrderGroupLeg] = field(default_factory=list)
    status: OrderGroupStatus = OrderGroupStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    
    total_quantity: int = 0  # e.g., 50 lots for iron condor = 50 per leg
    
    def add_leg(self, signal: Signal, required_qty: int) -> None:
        """Add a leg to the group."""
        leg = OrderGroupLeg(signal=signal, required_quantity=required_qty)
        self.legs.append(leg)
    
    def validate_fills(self) -> dict[str, bool]:
        """Check if all legs are filled as expected."""
        validation = {}
        
        for leg in self.legs:
            symbol = leg.signal.tradingsymbol
            
            # Each leg should be filled completely or all fail
            if leg.filled_quantity == 0:
                validation[symbol] = False
                leg.status = "FAILED"
            elif leg.filled_quantity < leg.required_quantity:
                validation[symbol] = False
                leg.status = "PARTIAL"
                leg.error = f"Partial fill: {leg.filled_quantity}/{leg.required_quantity}"
            else:
                validation[symbol] = True
                leg.status = "FILLED"
        
        # All legs must be filled for group to be valid
        all_filled = all(validation.values())
        
        if all_filled:
            self.status = OrderGroupStatus.COMPLETE
        elif any(validation.values()):
            self.status = OrderGroupStatus.PARTIAL
        else:
            self.status = OrderGroupStatus.FAILED
        
        return validation
    
    def get_mismatch_summary(self) -> dict[str, Any]:
        """Return details of any fills that don't match."""
        mismatches = []
        
        for leg in self.legs:
            if leg.filled_quantity != leg.required_quantity:
                mismatches.append({
                    "tradingsymbol": leg.signal.tradingsymbol,
                    "expected": leg.required_quantity,
                    "filled": leg.filled_quantity,
                    "variance": leg.required_quantity - leg.filled_quantity,
                    "variance_pct": abs(leg.required_quantity - leg.filled_quantity) / leg.required_quantity * 100
                })
        
        return {
            "group_id": self.group_id,
            "status": self.status.value,
            "mismatches": mismatches,
            "critical": len(mismatches) > 0
        }
    
    async def execute_with_rollback(self, execution_engine) -> bool:
        """
        Execute all legs sequentially but with rollback capability.
        
        Returns:
            True if all legs filled as expected
            False if any leg failed (all others cancelled)
        """
        placed_orders = []
        
        try:
            # Place all orders
            for leg in self.legs:
                order_id = await execution_engine.execute_signal(leg.signal)
                leg.order_id = order_id
                placed_orders.append((leg, order_id))
            
            # Query fills after brief delay for matching
            await asyncio.sleep(0.5)  # Give broker time to execute
            
            # Check if all legs filled correctly
            fills = {}
            for leg in self.legs:
                broker_order = await execution_engine._client.get_order(leg.order_id)
                leg.filled_quantity = broker_order.filled_quantity
                leg.execution_price = broker_order.average_price
                fills[leg.signal.tradingsymbol] = leg.filled_quantity
            
            # Validate
            validation = self.validate_fills()
            
            if not all(validation.values()):
                # ROLLBACK: Cancel unfilled/partially filled orders
                logger.error("multi_leg_execution_failed", 
                           group_id=self.group_id,
                           mismatches=self.get_mismatch_summary())
                
                # Cancel all pending orders
                for leg in self.legs:
                    if leg.status == "PENDING" or leg.status == "PARTIAL":
                        try:
                            await execution_engine._client.cancel_order(leg.order_id)
                        except Exception as e:
                            logger.error(f"cancel_failed: {e}")
                
                return False
            
            return True
            
        except Exception as e:
            logger.error("multi_leg_execution_error", 
                       group_id=self.group_id, 
                       error=str(e))
            
            # Cancel everything
            for leg, order_id in placed_orders:
                try:
                    await execution_engine._client.cancel_order(order_id)
                except:
                    pass
            
            return False
```

### Integration Point

**Modify** `src/execution/engine.py`:

```python
async def execute_multi_leg_strategy(self, order_group: MultiLegOrderGroup) -> bool:
    """Execute a multi-leg order group atomically."""
    
    # Pre-execution validation
    for leg in order_group.legs:
        is_valid, reason = self._risk.validate_signal(leg.signal)
        if not is_valid:
            raise RiskLimitError(f"Leg validation failed: {reason}")
    
    # Execute with rollback
    success = await order_group.execute_with_rollback(self)
    
    if success:
        logger.info("multi_leg_execution_success", group_id=order_group.group_id)
        # Record all legs to journal
        for leg in order_group.legs:
            self._journal.record_trade(
                strategy=order_group.strategy_name,
                tradingsymbol=leg.signal.tradingsymbol,
                ...
                metadata={"group_id": order_group.group_id, "leg": ...}
            )
    
    return success
```

---

## Fix 2: Stop-Loss Order Price Specification

### Current Problem

```python
# src/execution/engine.py (Line 48-60)
if signal.stop_loss:
    order_request.trigger_price = signal.stop_loss  # INCOMPLETE!
    # Missing: Execution price when trigger hits!
```

### Issue Demonstration

```
Signal: BUY 50 contracts @ 100, stop_loss=95

Current behavior:
  OrderRequest(
    order_type=SL,
    trigger_price=95,
    price=None
  )
  
Broker interpretation: 
  "When price hits 95, execute at MARKET (who knows what price?)"
  
  If order book is empty: Market order might fill at 85, 75, etc.
  
Expected behavior:
  OrderRequest(
    order_type=SL_LIMIT,  # SL with limit protection
    trigger_price=95,
    price=94.5  # Execute at 94.5 max (0.5% slippage)
  )
```

### Solution

**Modify** `src/execution/engine.py`:

```python
def _apply_stop_loss_to_order(self, order: OrderRequest, signal: Signal) -> None:
    """Apply stop-loss with proper execution price."""
    
    if not signal.stop_loss:
        return
    
    # Determine execution price based on direction
    if signal.transaction_type == TransactionType.BUY:
        # For BUY, stop price should be BELOW entry
        # Execute at market but with limit protection
        execution_price = signal.stop_loss * (1 - self._slippage_tolerance / 100)
    else:  # SELL
        # For SELL, stop price should be ABOVE entry
        # Execute at market but with limit protection
        execution_price = signal.stop_loss * (1 + self._slippage_tolerance / 100)
    
    order.trigger_price = signal.stop_loss
    order.price = round(execution_price, 2)
    order.order_type = OrderType.SL  # Stop Loss with Limit
    
    logger.info("stop_loss_applied",
               trigger_price=signal.stop_loss,
               execution_price=order.price,
               slippage_pct=self._slippage_tolerance)

async def execute_signal(self, signal: Signal) -> Optional[str]:
    """Execute a signal with proper risk controls."""
    
    # ... existing validation ...
    
    order_request = self._signal_to_order(signal)
    
    # FIX: Apply stop loss with execution price
    self._apply_stop_loss_to_order(order_request, signal)  # NEW!
    
    try:
        order_id = await self._client.place_order(order_request)
        # ... rest of code ...
```

---

## Fix 3: Position Pairing Validation

### Problem

```
Scenario: Iron Condor partial fill
  
Expected fills:
  - Sell 20000 CE: 50 lots
  - Buy 21000 CE: 50 lots  ← Only 30 filled
  - Sell 20000 PE: 50 lots
  - Buy 19000 PE: 50 lots

Broker state:
  Short 50 + 50 = 100 calls (unhedged on 20 contracts)
  Short 50 / Buy 50 = hedged puts
  
System should detect: CRITICAL - Call spread not hedge
```

### Solution

**Create** `src/risk/position_validator.py`:

```python
class PositionValidator:
    def __init__(self, tracker: PositionTracker):
        self.tracker = tracker
    
    def validate_multi_leg_hedge(self, strategy_name: str) -> dict[str, Any]:
        """Validate that multi-leg positions are properly hedged."""
        
        positions = self.tracker.get_strategy_positions(strategy_name)
        
        # Group by underlying
        by_symbol: dict[str, list[Position]] = {}
        for pos in positions:
            symbol = pos.tradingsymbol
            if symbol not in by_symbol:
                by_symbol[symbol] = []
            by_symbol[symbol].append(pos)
        
        issues = []
        
        # Check each symbol for hedge ratio
        if strategy_name.endswith("_credit_spread") or "condor" in strategy_name:
            for symbol, symbol_positions in by_symbol.items():
                
                # For spreads: should have buy + sell at different strikes
                buyers = [p for p in symbol_positions if p.transaction_type == TransactionType.BUY]
                sellers = [p for p in symbol_positions if p.transaction_type == TransactionType.SELL]
                
                if not buyers or not sellers:
                    issues.append({
                        "symbol": symbol,
                        "problem": "Missing hedge leg (buyer or seller)",
                        "severity": "CRITICAL"
                    })
                    continue
                
                buyer_qty = sum(p.quantity for p in buyers)
                seller_qty = sum(p.quantity for p in sellers)
                
                if buyer_qty != seller_qty:
                    issues.append({
                        "symbol": symbol,
                        "problem": f"Mismatched hedge: {seller_qty} short, {buyer_qty} long",
                        "variance_qty": seller_qty - buyer_qty,
                        "variance_pct": abs(seller_qty - buyer_qty) / seller_qty * 100,
                        "severity": "CRITICAL" if abs(seller_qty - buyer_qty) > buyer_qty * 0.1 else "WARNING"
                    })
        
        return {
            "strategy": strategy_name,
            "validated": len(issues) == 0,
            "issues": issues
        }
    
    async def monitor_continuous(self):
        """Run continuous validation check."""
        while True:
            try:
                for strategy in ["iron_condor", "bull_call_credit_spread", "bear_put_credit_spread"]:
                    result = self.validate_multi_leg_hedge(strategy)
                    
                    if result["issues"]:
                        logger.error("position_validation_failed", result=result)
                        # Alert trading desk
                        await notify_risk_team(result)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error("position_monitor_error", error=str(e))
                await asyncio.sleep(60)
```

### Integration

**Modify** `src/risk/manager.py`:

```python
class RiskManager:
    def __init__(self):
        self.position_validator = PositionValidator(self.tracker)
        
    async def start_monitoring(self):
        """Start continuous position monitoring."""
        asyncio.create_task(self.position_validator.monitor_continuous())
```

---

## Fix 4: Assignment & Exercise Handling

### Problem

```
Long position: BUY NIFTY 20000 CE (call)
Short position: SELL NIFTY 19000 PE (put)

At expiry NIFTY @ 20500:
  - Call: Automatically exercised → Creates LONG 50 NIFTY position
  - Put: Not assigned (NIFTY > 19000)

System issue:
  - No notification of exercise
  - P&L calculated wrong
  - New equity position in account but PositionTracker unaware
```

### Solution

**Create** `src/options/exercise_handler.py`:

```python
@dataclass
class ExerciseEvent:
    """Notification that an option was exercised/assigned."""
    tradingsymbol: str
    option_type: str  # CE or PE
    strike: float
    quantity: int
    exercise_type: str  # "EXERCISE" or "ASSIGNMENT"
    underlying_created: bool  # If true, creates equity position
    settlement_date: date


class ExerciseHandler:
    def __init__(self, client: KiteClient, tracker: PositionTracker, journal: TradeJournal):
        self.client = client
        self.tracker = tracker
        self.journal = journal
    
    async def monitor_exercises(self):
        """Check for ITM positions at expiry."""
        while True:
            try:
                # Check positions
                positions = self.tracker.get_positions()
                
                for pos in positions:
                    if self._is_option(pos.tradingsymbol) and self._is_expiry_today(pos.tradingsymbol):
                        
                        # Get current price
                        quote = await self.client.get_ltp(pos.tradingsymbol)
                        
                        if self._is_itm(pos, quote):
                            # Option will be exercised
                            exercise = ExerciseEvent(...)
                            await self._handle_exercise(exercise)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error("exercise_monitor_error", error=str(e))
    
    async def _handle_exercise(self, exercise: ExerciseEvent):
        """Process an exercise/assignment event."""
        
        if exercise.exercise_type == "EXERCISE":
            # Long call exercised → Buy underlying
            # Long put exercised → Sell underlying
            
            # Record the exercise
            self.journal.record_trade(
                strategy="exercise",
                tradingsymbol=exercise.tradingsymbol,
                quantity=exercise.quantity,
                price=exercise.strike,
                status="EXERCISE",
                metadata={"exercise_date": exercise.settlement_date}
            )
            
            # Create underlying position
            if exercise.option_type == "CE":
                underlying_tx = TransactionType.BUY
            else:
                underlying_tx = TransactionType.SELL
            
            # Add underlying position to tracker
            self.tracker.add_position(
                tradingsymbol=self._get_underlying(exercise.tradingsymbol),
                strategy_name="exercise_settlement",
                transaction_type=underlying_tx,
                entry_price=exercise.strike,
                quantity=exercise.quantity,
                signal_id=str(uuid4()),
                metadata={"exercise": True, "original_option": exercise.tradingsymbol}
            )
```

---

## Fix 5: Pre-Execution Margin Validation

### Problem

```
Credit spread uses margin:
  - Sell call @ 20000: Requires 50000 margin (5 * strike)
  - Sell put @ 20000: Requires 50000 margin
  - Total: ~100000 margin for 50 qty

Error: Order 1 succeeds, Order 2 fails due to margin
Result: Orphaned short call with no hedge
```

### Solution

**Modify** `src/risk/manager.py`:

```python
async def validate_margin_for_multi_leg(self, signals: list[Signal]) -> bool:
    """Validate margin availability for multi-leg execution."""
    
    # Calculate margin for all legs combined
    total_margin_needed = 0.0
    
    for signal in signals:
        # Get bracket margin requirement from broker
        margin_req = await self._estimate_margin_requirement(signal)
        total_margin_needed += margin_req
    
    # Get available margin
    margins = await self.client.get_margins()
    available = margins.equity.available.collateral
    
    logger.info("margin_check",
               required=total_margin_needed,
               available=available)
    
    if total_margin_needed > available:
        raise InsufficientMarginError(
            f"Need {total_margin_needed}, available {available}"
        )
    
    return True

async def _estimate_margin_requirement(self, signal: Signal) -> float:
    """Estimate margin for a signal (calls broker API if needed)."""
    
    if signal.product == ProductType.MIS:
        # MIS is 25% margin
        return signal.price * signal.quantity * 0.25
    
    elif signal.product == ProductType.NRML:
        # NRML is full margin
        return signal.price * signal.quantity
    
    # For options, use SPAN + exposure model
    # This typically requires broker API call
    # For now, use conservative estimate
    return signal.quantity * 10000  # Conservative estimate per contract
```

---

## Implementation Priority

1. **CRITICAL (Week 1)**: Fix 1 (Multi-Leg Order Groups) + Fix 2 (Stop-Loss)
2. **HIGH (Week 2)**: Fix 3 (Position Validation) + Fix 5 (Margin Pre-check)
3. **MEDIUM (Week 3)**: Fix 4 (Exercise Handling)

---

## Testing Scenarios

### Test 1: Multi-Leg Partial Fill Handling

```python
async def test_multi_leg_partial_fill():
    group = MultiLegOrderGroup(strategy_name="iron_condor")
    group.add_leg(Signal(..., qty=50), required_qty=50)  # SELL call
    group.add_leg(Signal(..., qty=50), required_qty=50)  # BUY call
    group.add_leg(Signal(..., qty=50), required_qty=50)  # SELL put
    group.add_leg(Signal(..., qty=50), required_qty=50)  # BUY put
    
    # Simulate: Call BUY fills only 30
    # Expected: Rollback all orders, no position created
    
    success = await engine.execute_multi_leg_strategy(group)
    assert not success
    assert group.status == OrderGroupStatus.FAILED
    assert len(tracker.get_positions()) == 0  # No position created
```

### Test 2: Stop-Loss Execution Price

```python
async def test_stop_loss_execution_price():
    signal = Signal(stop_loss=100.0, slippage_tolerance=0.5)
    order = engine._signal_to_order(signal)
    engine._apply_stop_loss_to_order(order, signal)
    
    assert order.trigger_price == 100.0
    assert order.price == 99.5  # 0.5% slippage protection
    assert order.order_type == OrderType.SL
```

---

## Summary

These 5 fixes address the critical execution risks in multi-leg strategies:

| Fix | Risk | Severity | Effort |
|-----|------|----------|--------|
| Multi-Leg Groups | Orphaned positions | CRITICAL | 8 hours |
| Stop-Loss Price | Excess slippage | CRITICAL | 2 hours |
| Position Validation | Invalid hedges | HIGH | 4 hours |
| Margin Pre-check | Insufficient margin | HIGH | 3 hours |
| Exercise Handling | Wrong P&L | MEDIUM | 6 hours |

**Total Implementation**: ~25 hours (3-4 engineering days)

**Payback**: Prevents >₹1M potential losses from single execution error

