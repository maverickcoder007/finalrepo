Implementation Guide: Position Tracking for Trading Strategies
===============================================================

## Overview

The position tracking infrastructure provides centralized management of all open positions across all trading strategies. It enables:

- Real-time P&L monitoring
- Automated profit-taking and stop-loss execution
- Position lifecycle management (entry → update → exit)
- Multi-strategy portfolio visibility

## Architecture

### Position Class (`src/data/position_tracker.py`)

Represents a single trade position with:

```python
Position(
    tradingsymbol="NIFTY23DECFUT",      # Trading symbol
    strategy_name="ema_crossover",       # Strategy that created it
    transaction_type=TransactionType.BUY,# BUY or SELL
    entry_price=20000.0,                 # Entry execution price
    quantity=1,                          # Contract/share quantity
    entry_time=datetime.now(),           # When position opened
    current_price=20050.0,               # Latest mark price
    current_pnl=50.0,                    # Unrealized P&L
    current_pnl_pct=0.25,                # P&L percentage
)
```

**Key Methods:**
- `update_price(price)`: Update current mark price and recalculate P&L
- `close(exit_price, reason)`: Close position and finalize P&L

### PositionTracker Class

Singleton manager for all positions:

```python
tracker = get_position_tracker()

# Add a new position
position = tracker.add_position(
    tradingsymbol="TCS",
    strategy_name="mean_reversion",
    transaction_type=TransactionType.BUY,
    entry_price=3450.50,
    quantity=1,
    signal_id="signal_123",
    metadata={"z_score": 2.1}
)

# Update prices from market ticks
tracker.update_prices("TCS", current_price=3455.20)

# Check P&L
pnl_stats = tracker.get_pnl(symbol="TCS")
# {'total_pnl': 4.70, 'avg_pnl_pct': 0.14, 'open_positions': 1}

# Get symbol exposure
exposure = tracker.get_symbol_exposure("TCS")
# {'symbol': 'TCS', 'long_qty': 2, 'short_qty': 0, 'net_exposure': 2, ...}

# Close position
tracker.close_position(position, exit_price=3460.0, reason="profit_target_hit")
```

## Integration with Strategies

### Equity Strategies (EMA, RSI, Mean Reversion, VWAP)

All 4 equity strategies now generate stop-loss prices. Integrate with PositionTracker:

```python
# In strategy execution (e.g., TradingService)
from src.data import get_position_tracker

tracker = get_position_tracker()

# When signal is executed
signal = await strategy.generate_signal(...)
if signal:
    position = tracker.add_position(
        tradingsymbol=signal.tradingsymbol,
        strategy_name=signal.strategy_name,
        transaction_type=signal.transaction_type,
        entry_price=execution_price,  # From actual order
        quantity=signal.quantity,
        signal_id=signal.id,
        metadata=signal.metadata,
    )
    
    # Check exit conditions on each tick
    tracker.update_prices(signal.tradingsymbol, current_price)
    
    # Auto-exit on stop-loss
    if tracker.should_exit_by_loss(position, signal.stop_loss):
        close_signal = Signal(
            ...,
            transaction_type=TransactionType.SELL,
            metadata={...,"reason": "stop_loss_triggered"}
        )
```

### Options Strategies (Iron Condor, Straddle, Bull/Bear Spreads)

Multi-leg strategies now track multiple positions per strategy:

```python
# IronCondor creates 4 legs (sell_ce, buy_ce, sell_pe, buy_pe)
for leg_signal in entry_signals:  # 4 signals
    position = tracker.add_position(
        traditionsymbol=leg_signal.tradingsymbol,
        strategy_name="iron_condor",
        transaction_type=leg_signal.transaction_type,
        entry_price=execution_price,
        quantity=leg_signal.quantity,
        signal_id=leg_signal.id,
        metadata={
            "leg": "sell_ce",
            "strategy_type": "iron_condor",
            "net_premium": 100.0,
            "max_loss_per_lot": 5000.0,
        }
    )

# Later: Calculate net P&L across all legs
all_legs = tracker.get_strategy_positions("iron_condor")
net_pnl = sum(p.current_pnl for p in all_legs)

# Exit when strategy triggers closes (via _evaluate_exit)
if net_pnl >= max_profit_target:
    for leg in all_legs:
        tracker.close_position(leg, exit_price, "profit_target_hit")
```

## Usage Patterns

### Pattern 1: Stop-Loss Management

```python
# Equity strategies already have stop_loss in Signal.stop_loss
# Execute:
position = tracker.add_position(...)
tracker.update_prices(symbol, tick.price)

if position.stop_loss and tracker.should_exit_by_loss(position, position.stop_loss):
    # Execute SELL order at market
    # Then close position:
    tracker.close_position(position, exit_price, "stop_loss_hit")
```

### Pattern 2: Profit-Taking

```python
# Check if position hit profit target (50% of max profit)
profit_target = signal.metadata.get("profit_target", 0)

if tracker.should_exit_by_profit(position, profit_target):
    # Execute exit
    tracker.close_position(position, exit_price, "profit_target_hit")
```

### Pattern 3: Time-Based Exit

```python
import datetime

# Exit positions held for >N hours
open_positions = tracker.get_positions()
now = datetime.datetime.now()

for pos in open_positions:
    hours_held = (now - pos.entry_time).total_seconds() / 3600
    if hours_held > 4:  # Exit after 4 hours
        tracker.close_position(pos, current_price, "time_limit_reached")
```

### Pattern 4: Portfolio Monitoring

```python
# Real-time P&L dashboard
all_pnl = tracker.get_pnl()
print(f"Total P&L: ₹{all_pnl['total_pnl']}")
print(f"Avg Return: {all_pnl['avg_pnl_pct']:.2f}%")
print(f"Open Positions: {all_pnl['open_positions']}")

# By strategy
strategy_pnl = tracker.get_pnl(strategy="iron_condor")

# By symbol
symbol_pnl = tracker.get_pnl(symbol="NIFTY")

# Symbol exposure
exposure = tracker.get_symbol_exposure("INFY")
if exposure["net_exposure"] > 10:
    logger.warning("High exposure on INFY", qty=exposure["net_exposure"])
```

## Implementation Checklist

**Phase 1: Core Integration (1-2 days)**
- [ ] Integrate PositionTracker with TradingService
- [ ] Hook position creation on signal execution
- [ ] Implement price updates on every tick
- [ ] Add stop-loss exit logic for equity strategies

**Phase 2: Profit-Taking (2-3 days)**
- [ ] Implement profit target checks for all strategies
- [ ] Add time-based exits for options
- [ ] Create dashboard for position monitoring

**Phase 3: Advanced Features (3-5 days)**
- [ ] Add max daily loss limit enforcement
- [ ] Implement position-size limits by symbol
- [ ] Add correlation-based hedging
- [ ] Create end-of-day settlement process

## Key Metrics Tracked

| Metric | Type | Purpose |
|--------|------|---------|
| `current_pnl` | float | Unrealized profit/loss |
| `current_pnl_pct` | float | Return percentage |
| `entry_price` | float | Position cost basis |
| `current_price` | float | Latest mark price |
| `hours_held` | float | Position duration tracking |
| `is_closed` | bool | Lifecycle state |

## Error Handling

```python
# Always validate before closing
try:
    position = tracker.add_position(...)
except ValueError as e:
    logger.error("Invalid position parameters", error=e)

# Gracefully handle price updates
try:
    tracker.update_prices(symbol, price)
except KeyError:
    logger.debug(f"No positions for {symbol}")

# Safe exit
if position and not position.is_closed:
    tracker.close_position(position, exit_price, "cleanup")
```

## Real-World Example

```python
async def trading_loop():
    tracker = get_position_tracker()
    
    for tick in market_ticks:
        # Update all positions with latest price
        tracker.update_prices(tick.symbol, tick.price)
        
        # Check exits
        for strategy in active_strategies:
            for position in tracker.get_strategy_positions(strategy.name):
                
                # Profit-taking
                if position.current_pnl >= strategy.profit_target:
                    exit_signal = create_exit_signal(position, "profit_target")
                    await execute_order(exit_signal)
                    tracker.close_position(position, tick.price, "profit_target_hit")
                
                # Stop-loss
                elif position.current_pnl <= -strategy.stop_loss:
                    exit_signal = create_exit_signal(position, "stop_loss")
                    await execute_order(exit_signal)
                    tracker.close_position(position, tick.price, "stop_loss_hit")
        
        # Generate entries
        entry_signals = await strategy.on_tick([tick])
        for signal in entry_signals:
            order = await execute_order(signal)
            tracker.add_position(
                signal.tradingsymbol,
                signal.strategy_name,
                signal.transaction_type,
                order.executed_price,
                signal.quantity,
                signal.id,
                signal.metadata,
            )
```

## See Also

- [Strategy Analysis Review](STRATEGY_ANALYSIS_REVIEW.md) - Strategy quality assessment
- [Security Audit Report](SECURITY_AUDIT_REPORT.md) - Security improvements implemented
