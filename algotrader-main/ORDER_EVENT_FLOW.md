# Order Event Flow — Complete Process Documentation

> **Version**: 2.0 · **Updated**: 21 Feb 2026  
> **Scope**: End-to-end lifecycle of every order — from strategy signal through broker execution, WebSocket events, reconciliation, and crash recovery.  
> **v2.0 Changes**: Added OrderCoordinator layer (freeze splitting, market state guard, liquidity checker, execution queue, timestamp resolver), split reconciliation, margin safety factor.

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Component Inventory](#2-component-inventory)
3. [Single-Leg Order Flow](#3-single-leg-order-flow)
4. [Multi-Leg Order Flow (Hedge Recovery)](#4-multi-leg-order-flow-hedge-recovery)
5. [Execution State Machine](#5-execution-state-machine)
6. [WebSocket Tick & Order Event Pipeline](#6-websocket-tick--order-event-pipeline)
7. [Five Production Safeguards](#7-five-production-safeguards)
8. [Risk Validation Pipeline](#8-risk-validation-pipeline)
9. [Stop-Loss Lifecycle (Hybrid SL)](#9-stop-loss-lifecycle-hybrid-sl)
10. [Reconciliation Loop](#10-reconciliation-loop)
11. [Crash Recovery Flow](#11-crash-recovery-flow)
12. [Kite Connect API Method Reference](#12-kite-connect-api-method-reference)
13. [Data Models Quick Reference](#13-data-models-quick-reference)
14. [Error Handling & Escalation Matrix](#14-error-handling--escalation-matrix)
15. [Broker Microstructure Layer (OrderCoordinator)](#15-broker-microstructure-layer-ordercoordinator)

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ALGO TRADING SYSTEM                              │
│                                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────┐                │
│  │ Strategy │───▶│  Execution   │───▶│  Order         │                │
│  │  Layer   │    │   Engine     │    │  Coordinator   │                │
│  └──────────┘    └──────────────┘    │  (NEW v2.0)    │                │
│       │                │             └───────┬────────┘                │
│       │          ┌─────┴──────┐              │                         │
│       │          │ Risk       │      ┌───────▼────────┐  ┌───────────┐│
│       │          │ Manager    │      │  KiteClient    │─▶│  Zerodha  ││
│       │          └────────────┘      │  (REST API)    │  │  Kite API ││
│       │                │             └────────────────┘  └───────────┘│
│       │                │                                       │      │
│       ▼                ▼                                       ▼      │
│  ┌──────────┐    ┌──────────────┐                     ┌─────────────┐│
│  │ Option   │    │  SQLite      │                     │ WebSocket   ││
│  │ Chain    │    │  Persistence │                     │ (Ticker)    ││
│  │ Builder  │    │  (WAL mode)  │                     └─────────────┘│
│  └──────────┘    └──────────────┘                                    │
│                                                                       │
│  ┌─── OrderCoordinator Pipeline ──────────────────────────────────┐  │
│  │ MarketStateGuard → FreezeQtySplit → LiquidityCheck → ExecQueue │  │
│  │ (auction/circuit)  (NSE limits)    (spread check)   (8 ord/s)  │  │
│  └────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

| Layer | File | Role |
|-------|------|------|
| **Strategy** | `src/strategy/*.py`, `src/options/strategies.py` | Generate `Signal` objects from tick/bar data |
| **Service** | `src/api/service.py` | Orchestrate strategies, manage lifecycle, broadcast to WebSocket clients |
| **Execution Engine** | `src/execution/engine.py` | Single-leg execution, slippage protection, dual reconciliation loops |
| **Order Coordinator** | `src/execution/order_coordinator.py` | **NEW v2.0** — Broker microstructure layer: freeze splitting, market state guard, liquidity checks, rate pacing, timestamp precedence |
| **Multi-Leg Group** | `src/execution/order_group_corrected.py` | Multi-leg hedge-first execution with state machine + 5 safeguards |
| **Risk Manager** | `src/risk/manager.py` | Validate signals, margin checks, kill switch, position sizing |
| **KiteClient** | `src/api/client.py` | Async REST wrapper for Kite Connect v3 with rate limiting + retries |
| **KiteTicker** | `src/api/ticker.py` | WebSocket binary tick parser + order update events |
| **Persistence** | `src/execution/order_group_corrected.py` → `PersistenceLayer` | SQLite WAL for crash recovery |
| **Trade Journal** | `src/data/journal.py` | Record all trades for audit trail |

---

## 2. Component Inventory

### Key Classes and Their Roles

| Class | Module | Purpose |
|-------|--------|---------|
| `TradingService` | `service.py` | Top-level orchestrator. Manages strategies, ticker, execution engine. |
| `ExecutionEngine` | `engine.py` | Single-leg: `Signal` → `OrderRequest` → `place_order()`. Reconciliation. |
| `CorrectMultiLegOrderGroup` | `order_group_corrected.py` | Multi-leg: hedge-first, state machine, emergency recovery. |
| `ExecutionStateValidator` | `order_group_corrected.py` | Safeguard #1: Prevent out-of-order state transitions. |
| `IdempotencyManager` | `order_group_corrected.py` | Safeguard #2: Deduplicate broker events. |
| `OrderIntentPersistence` | `order_group_corrected.py` | Safeguard #3: Persist intent BEFORE broker call. |
| `RecursiveHedgeMonitor` | `order_group_corrected.py` | Safeguard #4: Recursively hedge partial fills. |
| `BrokerTimestampedTimeouts` | `order_group_corrected.py` | Safeguard #5: Use broker clock, not local clock. |
| `PersistenceLayer` | `order_group_corrected.py` | SQLite storage for `ExecutionState` + `LegState`. |
| `InterimExposureHandler` | `order_group_corrected.py` | Monitor delta exposure between partial fills. |
| `EmergencyHedgeExecutor` | `order_group_corrected.py` | Reverse all filled legs on catastrophic failure. |
| `HybridStopLossManager` | `order_group_corrected.py` | SL-LIMIT with MARKET fallback on timeout/gap-through. |
| `WorstCaseMarginSimulator` | `order_group_corrected.py` | Pre-flight margin check for worst-case partial fills. |
| `ChildOrderManager` | `order_group_corrected.py` | Split orders exceeding exchange freeze quantity. |
| `FreezeQuantityManager` | `order_coordinator.py` | **NEW v2.0** — NSE freeze limit table; splits orders BEFORE margin simulation. |
| `MarketStateGuard` | `order_coordinator.py` | **NEW v2.0** — Blocks orders during pre-open/auction/circuit breaker. |
| `ExecutionQueue` | `order_coordinator.py` | **NEW v2.0** — Priority FIFO with rate pacing at 8 orders/sec. |
| `LiquidityChecker` | `order_coordinator.py` | **NEW v2.0** — Checks bid-ask spread; converts MARKET to aggressive LIMIT if wide. |
| `TimestampResolver` | `order_coordinator.py` | **NEW v2.0** — Prevents REST from overwriting newer WebSocket status. |
| `OrderCoordinator` | `order_coordinator.py` | **NEW v2.0** — Main orchestrator: market state → freeze split → liquidity → queue. |
| `RiskManager` | `manager.py` | Kill switch, daily PnL limits, position sizing, margin validation. |
| `KiteClient` | `client.py` | Async HTTP client for all Kite Connect REST endpoints. |
| `KiteTicker` | `ticker.py` | WebSocket client for real-time ticks + order updates. |

---

## 3. Single-Leg Order Flow

This is the primary path for equity strategy signals (EMA Crossover, Mean Reversion, RSI, VWAP Breakout).

### Step-by-Step Flow

```
Strategy.on_tick(ticks)
    │
    ▼
Signal(tradingsymbol, exchange, transaction_type, quantity, order_type, ...)
    │
    ▼
TradingService._process_signal(signal)
    │
    ├──▶ Log signal + broadcast to WS clients
    │
    ▼
ExecutionEngine.execute_signal(signal)
    │
    ├──▶ [1] Kill Switch Check ─── active? ──▶ raise KillSwitchError
    │
    ├──▶ [2] RiskManager.validate_signal(signal)
    │         ├── Daily PnL limit check
    │         ├── Position size limit check
    │         └── Total exposure limit check
    │         └── Failed? ──▶ raise RiskLimitError
    │
    ├──▶ [3] MarketStateGuard.is_safe_to_execute(symbol, exchange)   ← NEW v2.0
    │         ├── Time-based: 9:00-9:15 pre-open → BLOCK
    │         ├── Time-based: after 15:30 → BLOCK
    │         ├── Quote-based: circuit breaker detected → BLOCK
    │         └── Unsafe? ──▶ raise OrderError
    │
    ├──▶ [4] FreezeQuantityManager.split_signal(signal)              ← NEW v2.0
    │         ├── Check qty vs NSE freeze limit for underlying
    │         ├── If qty ≤ freeze_limit → [signal] (no split)
    │         └── If qty > freeze_limit → [child_1, child_2, ...] (chunked)
    │         │   e.g. NIFTY 3600 lots → [1800, 1800]
    │
    ├──▶ [5] For each child signal:
    │         _execute_single_signal(child)
    │           │
    │           ├── _signal_to_order(signal) → OrderRequest
    │           │
    │           ├── _apply_stop_loss_to_order(order, signal)
    │           │     If signal.stop_loss set:
    │           │       order.trigger_price = stop_loss
    │           │       order.price = stop_loss ± slippage
    │           │       order.order_type = SL
    │           │
    │           ├── OrderCoordinator.place_order(OrderRequest)       ← NEW v2.0
    │           │     │
    │           │     ├── MarketStateGuard check (redundant safety)
    │           │     ├── FreezeQuantityManager.split_order()
    │           │     ├── LiquidityChecker.check_liquidity()
    │           │     │     ├── Spread < 5% → MARKET OK
    │           │     │     ├── Spread ≥ 5% → convert to aggressive LIMIT
    │           │     │     └── No liquidity → ABORT with fallback
    │           │     └── ExecutionQueue.enqueue() → rate-paced at 8/sec
    │           │           └── KiteClient.place_order() → order_id
    │           │
    │           ├── Track in _pending_orders[order_id] = OrderRequest
    │           │
    │           └── Return order_id to caller
    │
    └──▶ [6] Return first order_id to caller
              │
              ▼
         TradingService._process_signal logs in TradeJournal
```

### With Slippage Protection Variant

```
ExecutionEngine.execute_with_slippage_protection(signal, max_slippage_pct)
    │
    ├── If signal is MARKET with a price hint:
    │     Convert to LIMIT at price ± slippage%
    │     Then execute_signal(modified_signal)
    │
    └── Otherwise: regular execute_signal(signal)
```

### Signal → OrderRequest Mapping

| Signal Field | OrderRequest Field | Notes |
|---|---|---|
| `tradingsymbol` | `tradingsymbol` | e.g. "NIFTY24FEB22000CE" |
| `exchange` | `exchange` | NSE, NFO, BSE, etc. |
| `transaction_type` | `transaction_type` | BUY or SELL |
| `order_type` | `order_type` | MARKET, LIMIT, SL, SL-M |
| `quantity` | `quantity` | Integer units |
| `product` | `product` | MIS, NRML, CNC |
| `price` | `price` | Required for LIMIT/SL |
| `trigger_price` | `trigger_price` | Required for SL/SL-M |
| `stop_loss` | Applied via `_apply_stop_loss_to_order()` | Converted to SL order type |

---

## 4. Multi-Leg Order Flow (Hedge Recovery)

This is the critical path for options strategies (Iron Condor, Straddle/Strangle, Bull Call, Bear Put). Multi-leg execution ensures protection legs fill BEFORE risk legs.

### Step-by-Step Flow

```
CorrectMultiLegOrderGroup.execute(legs)
    │
    ▼
[1] VALIDATE ── lifecycle → VALIDATED
    │
    ▼
[1.5] FREEZE SPLIT (NEW v2.0) ── Split BEFORE margin check
    │   FreezeQuantityManager.split_signal(leg) for each leg
    │   e.g. 4 legs (1 with qty 3600 NIFTY) → 5 split_legs (1800+1800)
    │   This MUST happen first: margin sim on unsplit legs is invalid
    │   because NSE rejects the oversized order anyway
    │
    ▼
[2] WorstCaseMarginSimulator.validate_margin(split_legs)
    │   Simulate each leg filling individually under 3% adverse move
    │   ★ Uses 1.15x MARGIN SAFETY FACTOR (NEW v2.0)
    │   ★ buffered_margin = worst_case_margin × 1.15
    │   ★ Prevents margin drift between fills from causing rejection
    │   If margin insufficient → raise RuntimeError (abort before any orders)
    │
    ▼
[3] SORT LEGS by type:
    │   protection_legs = [l for l in split_legs if l.leg_type == PROTECTION]  ← FIRST
    │   risk_legs       = [l for l in split_legs if l.leg_type == RISK]        ← SECOND
    │   ordered_legs    = protection_legs + risk_legs
    │
    ▼
[4] SUBMIT ── lifecycle → HEDGE_SUBMITTED
    │   For each leg (protection first):
    │     ├── Create LegState(leg_id, tradingsymbol, quantity, leg_type)
    │     ├── ExecutionEngine.execute_signal(leg) → order_id
    │     ├── LegState.order_id = order_id
    │     └── If PROTECTION leg REJECTED → return False (abort)
    │
    ▼
[5] TRACK FILLS ── wait 500ms
    │   check filled_count vs total legs
    │
    ├── IF partial fills detected:
    │     │
    │     ▼
    │   InterimExposureHandler.monitor_interim_exposure(state)
    │     │  lifecycle → PARTIALLY_EXPOSED
    │     │  Loop every 100ms:
    │     │    ├── Calculate interim delta (unhedged legs)
    │     │    ├── If |delta| > max_interim_delta (50):
    │     │    │     Place interim hedge order
    │     │    │     Track in state.interim_hedge_orders
    │     │    └── Check if all legs filled → break
    │     └── PersistenceLayer.save_state(state)
    │
    ▼
[6] MONITOR LOOP ── poll every 100ms
    │   For each leg:
    │     order = _get_latest_order_status(client, order_id)
    │     Update: leg.filled_quantity, leg.status
    │
    ├── All filled? → lifecycle → HEDGED → break
    │
    ├── Any REJECTED?
    │     │
    │     ▼
    │   EmergencyHedgeExecutor.execute_recovery(state)
    │     │  For each filled leg:
    │     │    Create reverse Signal (BUY↔SELL)
    │     │    Execute at MARKET (speed critical)
    │     │    Track in state.emergency_hedge_orders
    │     │  lifecycle → EMERGENCY_HEDGED
    │     └── return success/failure
    │
    ▼
[7] SUCCESS ── lifecycle → CLOSED
    │   PersistenceLayer.save_state(state)
    │   Log "multi_leg_execution_complete"
    │   return True
    │
    ▼
[EXCEPTION PATH]
    │   Any unhandled exception:
    │     EmergencyHedgeExecutor.execute_recovery(state)
    │     PersistenceLayer.save_state(state)
    │     return False
```

### Leg Execution Order Example (Iron Condor)

| Execution Order | Leg | Type | Transaction | Purpose |
|:---:|------|------|-------------|---------|
| 1 | Buy CE (OTM) | PROTECTION | BUY | Covers short call risk |
| 2 | Buy PE (OTM) | PROTECTION | BUY | Covers short put risk |
| 3 | Sell CE (ATM) | RISK | SELL | Premium income (call side) |
| 4 | Sell PE (ATM) | RISK | SELL | Premium income (put side) |

**Protection legs always execute first.** If a protection leg is rejected, the entire group aborts — no naked short exposure is ever created.

---

## 5. Execution State Machine

### Lifecycle States (11 total)

```
                                ┌──────────────────┐
                                │     CREATED       │  (Order 0)
                                │   Group initialized│
                                └────────┬─────────┘
                                         │
                                ┌────────▼─────────┐
                                │    VALIDATED      │  (Order 1)
                                │  Margin + risk OK │
                                └────────┬─────────┘
                                         │
                                ┌────────▼─────────┐
                                │ HEDGE_SUBMITTED   │  (Order 2)
                                │  Protection sent  │
                                └────────┬─────────┘
                                         │
                                ┌────────▼─────────┐
                                │  HEDGE_FILLED     │  (Order 3)
                                │  Protection done  │
                                └────────┬─────────┘
                                         │
                                ┌────────▼─────────┐
                                │ RISK_SUBMITTED    │  (Order 4)
                                │  Risk legs sent   │
                                └────────┬─────────┘
                                        ╱╲
                                       ╱  ╲
                          ┌───────────╱    ╲───────────┐
                          ▼                            ▼
                ┌─────────────────┐          ┌──────────────────┐
                │ PARTIALLY_FILLED│          │     FILLED       │  (Order 7)
                │   (Order 5)     │          │   All legs done  │
                └────────┬────────┘          └────────┬─────────┘
                         │                            │
                ┌────────▼────────┐          ┌────────▼─────────┐
                │PARTIALLY_EXPOSED│          │     HEDGED       │  (Order 8)
                │   (Order 6)     │          │   Position net   │
                │ Interim hedging │          └────────┬─────────┘
                └────────┬────────┘                   │
                         │                   ┌────────▼─────────┐
                         │                   │     CLOSED       │  (Order 9)
                         │                   │   Complete OK    │
                         │                   └──────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
    ┌──────────────────┐   ┌──────────────┐
    │EMERGENCY_HEDGED  │   │   FAILED     │  (Order 11)
    │   (Order 10)     │   │  Cannot fix  │
    │ Reverse trades   │   └──────────────┘
    │ placed           │
    └──────────────────┘
```

### Monotonic State Ordering (Safeguard #1)

States have a strict numerical ordering. The `ExecutionStateValidator` **rejects** any transition where `incoming_order < current_order`:

| State | Order | Can transition TO |
|-------|:-----:|-------------------|
| CREATED | 0 | → VALIDATED |
| VALIDATED | 1 | → HEDGE_SUBMITTED |
| HEDGE_SUBMITTED | 2 | → HEDGE_FILLED |
| HEDGE_FILLED | 3 | → RISK_SUBMITTED |
| RISK_SUBMITTED | 4 | → PARTIALLY_FILLED, FILLED |
| PARTIALLY_FILLED | 5 | → PARTIALLY_EXPOSED, FILLED |
| PARTIALLY_EXPOSED | 6 | → FILLED, EMERGENCY_HEDGED, FAILED |
| FILLED | 7 | → HEDGED |
| HEDGED | 8 | → CLOSED |
| CLOSED | 9 | (terminal) |
| EMERGENCY_HEDGED | 10 | (terminal) |
| FAILED | 11 | (terminal) |

---

## 6. WebSocket Tick & Order Event Pipeline

### Connection & Subscription

```
TradingService.start_live(tokens, mode)
    │
    ├── Set callbacks: on_ticks, on_connect, on_close, on_error, on_order_update
    ├── _seed_bar_data(tokens)  ← Fetch 5 days of 5min candles for each token
    ├── ticker.subscribe(tokens, mode)
    │
    ├── asyncio.create_task(ticker.connect())         ← Background
    ├── asyncio.create_task(reconciliation_loop())    ← Background
    └── asyncio.create_task(position_update_loop())   ← Background
                                                        (All with exception handlers)
```

### WebSocket URL

```
wss://ws.kite.trade?api_key={api_key}&access_token={access_token}
```

### Tick Processing Pipeline

```
KiteTicker._listen()
    │
    ├── Binary message (ticks)
    │     │
    │     ▼
    │   _parse_binary(data: bytes) → list[Tick]
    │     │  Header: 2 bytes = num_packets (big-endian unsigned short)
    │     │  Per packet: 2 bytes length + payload
    │     │
    │     │  Packet sizes:
    │     │    8 bytes  → LTP mode  (token + last_price)
    │     │   28 bytes  → Quote mode (+ qty, avg, volume, buy/sell qty)
    │     │   44 bytes  → Quote + OHLC
    │     │  184 bytes  → Full mode (+ change, timestamps, OI, depth)
    │     │
    │     │  CDS segment (segment & 0xFF == 3): divisor = 10,000,000
    │     │  All other segments: divisor = 100
    │     │
    │     ▼
    │   for each Tick:
    │     ├── Cache in _tick_cache[instrument_token]
    │     ├── Update _live_ticks for dashboard
    │     │
    │     ├── Update OI tracker if NIFTY/SENSEX token
    │     │     oi_tracker.set_spot_price()
    │     │     oi_tracker.update_from_tick() (if oi data present)
    │     │
    │     ├── Append tick to strategy bar data (rolling 500 bars)
    │     │
    │     └── For each active strategy:
    │           strategy.on_tick(ticks) → list[Signal]
    │             For each signal:
    │               _process_signal(signal)
    │                 ├── Log + broadcast to WS clients
    │                 ├── ExecutionEngine.execute_signal(signal)
    │                 └── TradeJournal.record_trade()
    │
    ├── String message (order updates)
    │     │
    │     ▼
    │   JSON parse → {"type": "order", "data": {...}}
    │     │
    │     ▼
    │   on_order_update(data)
    │     ├── TradeJournal.record_trade(strategy="live", ...)
    │     └── Broadcast {"type": "order_update", "data": data} to WS clients
    │
    └── Connection lost
          │
          ▼
        Reconnect with exponential backoff
          delay = min(5 * attempt, 60) seconds
          max 50 attempts
          Re-subscribe all tokens on reconnect
```

### Order Update Event (from Kite WebSocket)

When Kite Connect pushes an order status change over WebSocket:

```json
{
  "type": "order",
  "data": {
    "order_id": "220221000000001",
    "exchange_order_id": "1300000000000001",
    "tradingsymbol": "NIFTY24FEB22000CE",
    "status": "COMPLETE",
    "transaction_type": "BUY",
    "exchange": "NFO",
    "order_type": "MARKET",
    "quantity": 50,
    "filled_quantity": 50,
    "pending_quantity": 0,
    "average_price": 150.25,
    "variety": "regular"
  }
}
```

**Important**: Order updates arrive over WebSocket but the system also uses REST polling (`get_order_history`) for reliability. The WebSocket is "optimistic" — the reconciliation loop provides the source of truth.

---

## 7. Five Production Safeguards

### Safeguard #1: ExecutionStateValidator

**Problem**: Broker WebSocket can deliver events out-of-order during reconnects.  
**Solution**: Monotonic state ordering — reject any transition where `incoming < current`.

```
Broker Event: HEDGE_SUBMITTED (order=2) arrives
Current State: RISK_SUBMITTED (order=4)

  ExecutionStateValidator: 2 < 4 → REJECTED ✋
  "Out-of-order state transition rejected: RISK_SUBMITTED ← HEDGE_SUBMITTED"
```

### Safeguard #2: IdempotencyManager

**Problem**: Exchanges resend events (duplicates). Without dedup, a duplicate FILLED event could trigger duplicate hedge placement.  
**Solution**: Track event IDs in SQLite `processed_events` table. Ignore already-processed events.

```
Event arrives: {id: "EVT_123", type: "FILLED"}

  IdempotencyManager.mark_processed("EVT_123", group_id, "FILLED")
    ├── Check in-memory set → NOT found → process it
    ├── Add to in-memory set
    └── Persist to SQLite (non-blocking, via executor)

Same event arrives again:
    ├── Check in-memory set → FOUND → return False (skip)
    └── Log "Duplicate event detected: EVT_123"
```

### Safeguard #3: OrderIntentPersistence

**Problem**: If the app crashes AFTER `place_order()` but BEFORE `save_state()`, the system loses knowledge of a live order on the exchange.  
**Solution**: Persist the order intent to SQLite BEFORE calling `place_order()`.

```
place_order_with_intent_persistence(leg)
    │
    ├── [1] persist_order_intent() → SQLite (status: INTENT_PERSISTED)
    │       ↑ If crash here: recovery finds orphaned intent, retries placement
    │
    ├── [2] client.place_order(OrderRequest) → broker_order_id
    │       ↑ If crash here: intent shows INTENT_PERSISTED, recovery retries
    │
    └── [3] mark_order_sent(intent_id, broker_order_id)
            ↑ Intent updated to SENT with broker ID
```

### Safeguard #4: RecursiveHedgeMonitor

**Problem**: Emergency hedge itself may partially fill, creating a NEW unhedged exposure. Without recursive monitoring, you have a cascading risk.  
**Solution**: Treat every hedge as its own exposure. If hedge partially fills, recursively hedge the unfilled portion (max depth = 3).

```
place_and_monitor_hedge(symbol, qty=100, recursion=0)
    │
    ├── place MARKET order → filled 80/100
    │
    ├── unfilled = 20
    │     │
    │     ▼
    │   place_and_monitor_hedge(symbol, qty=20, recursion=1)  ← RECURSIVE
    │     │
    │     ├── place MARKET order → filled 20/20 ✓
    │     └── total_filled = 20
    │
    └── total_filled = 80 + 20 = 100 ✓

If recursion reaches depth 3:
    └── CRITICAL alert → _escalate_to_trader()
        "Max hedge recursion reached. Manual intervention required."
```

### Safeguard #5: BrokerTimestampedTimeouts

**Problem**: Local clock can drift (NTP adjustments) or event loop can stall (GC pauses), making `asyncio.sleep()` unreliable for time-critical trading.  
**Solution**: Use the `exchange_update_timestamp` from Kite order history for all timeout calculations.

```
execute_with_broker_timeout(order_id, timeout_seconds=2.0)
    │
    ├── Get initial order → extract server_timestamp (T₀)
    │
    └── Poll loop every 100ms:
          │
          ├── Get order status → extract current server_timestamp (T₁)
          ├── broker_elapsed = T₁ - T₀  (BROKER TIME, not local)
          │
          ├── status == FILLED → return success
          └── broker_elapsed >= timeout → _execute_market_fallback()
```

---

## 8. Risk Validation Pipeline

Every signal passes through this pipeline before any order is placed:

```
Signal arrives from strategy
    │
    ▼
[Gate 1] Kill Switch Check
    │  RiskManager._kill_switch_active?
    │  Triggers when: daily_pnl ≤ −kill_switch_loss
    │  If active → KillSwitchError (all strategies deactivated)
    │
    ▼
[Gate 2] Daily PnL Limit
    │  daily_pnl ≤ −max_daily_loss?
    │  If exceeded → RiskLimitError
    │
    ▼
[Gate 3] Position Size Limit
    │  signal.quantity > max_position_size?
    │  If exceeded → RiskLimitError
    │
    ▼
[Gate 4] Total Exposure Limit
    │  current_exposure + (qty × price) > max_exposure?
    │  If exceeded → RiskLimitError
    │
    ▼
[Gate 5] Multi-Leg Margin Check (options only)
    │  validate_margin_for_multi_leg(signals)
    │  Fetches live margins from broker
    │  Estimates per-leg margin (MIS=25%, NRML=100%, options=₹10K/contract)
    │  If total > available → RiskLimitError
    │
    ▼
[Gate 6] Worst-Case Margin Simulation (multi-leg only)
    │  WorstCaseMarginSimulator.validate_margin(legs)
    │  Simulates 3% adverse move on each leg filling individually
    │  If worst_case_margin > available → RuntimeError (abort)
    │
    ▼
Signal approved → proceed to order placement
```

---

## 9. Stop-Loss Lifecycle (Hybrid SL)

The `HybridStopLossManager` implements a two-phase stop-loss that prevents infinite slippage:

### Phase 1: SL-LIMIT Order

```
execute_with_hybrid_stop(signal)
    │
    ├── Create SL-LIMIT order:
    │     trigger_price = signal.stop_loss
    │     price = stop_loss × (1 − 0.5%)   ← slippage buffer
    │     order_type = SL
    │
    ├── place_order(sl_order) → order_id
    │
    └── Monitor loop (every 10ms):
          │
          ├── Check order status via _get_latest_order_status()
          │
          ├── COMPLETE → return (SL filled normally) ✓
          │
          ├── Elapsed > 2000ms → TIMEOUT → Phase 2
          │
          └── Check LTP for gap-through:
                get_ltp([exchange:symbol]) → current_price
                If BUY SL at 95 and current_price < 94 → GAP THROUGH → Phase 2
```

### Phase 2: MARKET Fallback (with Liquidity Check — NEW v2.0)

```
_place_market_fallback(signal)
    │
    ├── [1] Create MARKET OrderRequest for remaining pending_quantity
    │
    ├── [2] LiquidityChecker.check_liquidity(symbol, exchange)     ← NEW v2.0
    │         │
    │         ├── Fetch quote depth (top 5 bids/asks)
    │         ├── Calculate spread = best_ask − best_bid
    │         ├── spread_pct = spread / best_bid × 100
    │         │
    │         ├── spread < 5% → "MARKET" recommendation → proceed as MARKET
    │         │
    │         ├── spread ≥ 5% → "LIMIT" recommendation:
    │         │     convert_to_aggressive_limit()
    │         │     For BUY:  price = best_ask + (best_ask × 0.5%)
    │         │     For SELL: price = best_bid − (best_bid × 0.5%)
    │         │     Log sl_fallback_using_aggressive_limit
    │         │
    │         └── No depth / zero bids → "ABORT" recommendation:
    │               Log sl_fallback_no_liquidity (CRITICAL)
    │               Still place MARKET as last resort (better than unhedged)
    │
    ├── [3] Place order (MARKET or aggressive LIMIT)
    │         KiteClient.place_order() → order_id
    │
    └── [4] Return order_id
```

### Gap-Through Detection

```
BUY position with SL at ₹95:
  Gap-through if current_price < ₹94 (1 rupee past trigger)

SELL position with SL at ₹105:
  Gap-through if current_price > ₹106 (1 rupee past trigger)
```

---

## 10. Reconciliation Loop (Dual-Speed — NEW v2.0)

The reconciliation system now runs **two loops** concurrently to minimize the exposure window from 30s down to ~2s for active orders:

### Fast Reconciliation (2 seconds) — Active Orders

```
ExecutionEngine._fast_reconciliation_loop()
    │
    └── While running (every 2s):
          │
          ├── If _pending_orders is empty → sleep and continue
          │
          └── For each pending order_id:
                │
                ├── GET /orders → broker_orders
                │
                ├── status == COMPLETE → move to _filled_orders
                │     Log fast_recon_fill_confirmed
                │
                ├── status == REJECTED → move + log warning
                │     Log fast_recon_rejection_detected
                │     ★ Rejection caught within 2s (was 30s previously)
                │     ★ This enables immediate re-hedge or emergency recovery
                │
                ├── status == CANCELLED → move to _filled_orders
                │
                └── Still OPEN/PENDING → leave (check again in 2s)
```

**Why 2 seconds?** A rejected protection leg leaves the portfolio unhedged. Previously, it took up to 30s to discover this rejection. With fast reconciliation, the exposure window is capped at ~2 seconds — the time for one round-trip to Kite API.

### Slow Reconciliation (30 seconds) — Full Cleanup

```
ExecutionEngine._slow_reconciliation_loop()
    │
    └── While running (every 30s):
          │
          └── reconcile_orders():
                │
                ├── GET /orders → broker_orders (all today's orders)
                │
                ├── Build map: {order_id → Order}
                │
                └── For each pending order_id:
                      │
                      ├── Found in broker?
                      │     ├── status == COMPLETE → move to _filled_orders ✓
                      │     ├── status == CANCELLED → move to _filled_orders
                      │     ├── status == REJECTED → move + log warning
                      │     └── Still OPEN/PENDING → leave in _pending_orders
                      │
                      └── NOT found on broker?
                            └── Log as mismatch error
```

### Startup

```
ExecutionEngine.start_reconciliation_loop()
    │
    └── asyncio.gather(
          _fast_reconciliation_loop(),   ← 2s interval, active orders only
          _slow_reconciliation_loop(),   ← 30s interval, full reconciliation
        )
```

### Position Update Loop

Runs every 30 seconds in parallel:

```
_position_update_loop()
    │
    └── While running (every 30s):
          ├── GET /portfolio/positions → Positions
          ├── RiskManager.update_positions(positions.net)
          ├── Calculate daily_pnl = sum(p.pnl for p in positions.net)
          ├── RiskManager.update_daily_pnl(daily_pnl)
          │     └── If pnl ≤ −kill_switch_loss → activate kill switch
          └── Broadcast to WS clients: {type: "positions", data, pnl}
```

---

## 11. Crash Recovery Flow

### On System Restart

```
recover_execution_from_crash(group_id, persistence, engine, client, intent_persistence)
    │
    ├── [1] Load state from SQLite:
    │         persistence.load_state(group_id) → ExecutionState
    │         If no state → return (nothing to recover)
    │
    ├── [2] Handle orphaned order intents (Safeguard #3):
    │         intent_persistence.load_orphaned_intents()
    │         Query: status = 'INTENT_PERSISTED' AND timestamp > 1 hour ago
    │
    │         For each orphaned intent:
    │           ├── Retry: client.place_order(OrderRequest)
    │           ├── Update: intent_persistence.mark_order_sent(intent_id, order_id)
    │           └── On failure: log CRITICAL (manual intervention needed)
    │
    ├── [3] Handle mid-execution crash:
    │         If state.lifecycle == PARTIALLY_EXPOSED:
    │           ├── CRITICAL: "crash_during_interim_exposure_hedging"
    │           └── EmergencyHedgeExecutor.execute_recovery(state)
    │                 Reverse all filled legs at MARKET
    │
    └── [4] Save recovered state:
              persistence.save_state(state)
```

### SQLite Schema (Recovery Data)

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `execution_states` | Group lifecycle | `group_id`, `lifecycle`, `state_json`, `emergency_hedge_executed` |
| `leg_states` | Per-leg tracking | `leg_id`, `group_id`, `order_id`, `filled_quantity`, `status` |
| `emergency_hedges` | Hedge records | `hedge_id`, `group_id`, `original_order_id`, `hedge_order_id` |
| `processed_events` | Dedup (Safeguard #2) | `event_id`, `group_id`, `event_type`, `timestamp` |
| `order_intents` | Intent persistence (Safeguard #3) | `intent_id`, `group_id`, `symbol`, `quantity`, `status`, `broker_order_id` |
| `hedge_tracking` | Recursive hedges (Safeguard #4) | Tracked via `RecursiveHedgeMonitor.hedge_tracker` in-memory |

---

## 12. Kite Connect API Method Reference

All broker interactions go through `KiteClient` (async, rate-limited, with retries).

### Order Methods

| Method | HTTP | Endpoint | Input | Output | Rate Limiter |
|--------|------|----------|-------|--------|:---:|
| `place_order(OrderRequest)` | POST | `/orders/{variety}` | OrderRequest (Pydantic) | `str` (order_id) | order |
| `modify_order(OrderModifyRequest)` | PUT | `/orders/{variety}/{order_id}` | OrderModifyRequest | `str` (order_id) | order |
| `cancel_order(order_id, variety)` | DELETE | `/orders/{variety}/{order_id}` | str, OrderVariety | `str` (order_id) | order |
| `get_orders()` | GET | `/orders` | — | `list[Order]` | default |
| `get_order_history(order_id)` | GET | `/orders/{order_id}` | str | `list[Order]` | default |
| `get_trades()` | GET | `/trades` | — | `list[Trade]` | default |
| `get_order_trades(order_id)` | GET | `/orders/{order_id}/trades` | str | `list[Trade]` | default |

### Quote Methods

| Method | HTTP | Endpoint | Input | Output | Rate Limiter |
|--------|------|----------|-------|--------|:---:|
| `get_quote(instruments)` | GET | `/quote` | `list[str]` | `dict[str, Quote]` | quote |
| `get_ohlc(instruments)` | GET | `/quote/ohlc` | `list[str]` | `dict[str, OHLCQuote]` | quote |
| `get_ltp(instruments)` | GET | `/quote/ltp` | `list[str]` | `dict[str, LTPQuote]` | quote |

### Portfolio Methods

| Method | HTTP | Endpoint | Input | Output | Rate Limiter |
|--------|------|----------|-------|--------|:---:|
| `get_positions()` | GET | `/portfolio/positions` | — | `Positions` | default |
| `get_holdings()` | GET | `/portfolio/holdings` | — | `list[Holding]` | default |
| `get_margins(segment?)` | GET | `/user/margins[/{segment}]` | Optional str | `Margins` | default |

### Important API Notes

1. **No `get_order(id)` method exists.** Use `get_order_history(id)` which returns a list of state transitions. Take the **last** element for the current status. This is wrapped in the helper `_get_latest_order_status()`.

2. **`place_order()` accepts only `OrderRequest`** (Pydantic model), never a raw dict.

3. **`cancel_order()` requires `variety`** parameter (default: `OrderVariety.REGULAR`).

4. **`get_ltp()` keys** are in format `"EXCHANGE:SYMBOL"` (e.g., `"NFO:NIFTY24FEB22000CE"`).

5. **Instruments CSV** is returned as plain text from `/instruments/{exchange}`, parsed with `csv.DictReader`.

---

## 13. Data Models Quick Reference

### OrderRequest (sent to broker)

```python
class OrderRequest(BaseModel):
    tradingsymbol: str                    # "NIFTY24FEB22000CE"
    exchange: Exchange                    # NFO, NSE, BSE, CDS, MCX
    transaction_type: TransactionType     # BUY, SELL
    order_type: OrderType                 # MARKET, LIMIT, SL, SL-M
    quantity: int                         # Number of units
    product: ProductType                  # MIS, NRML, CNC
    price: Optional[float]               # Required for LIMIT, SL
    trigger_price: Optional[float]        # Required for SL, SL-M
    variety: OrderVariety = REGULAR       # regular, amo, co, iceberg
    validity: OrderValidity = DAY         # DAY, IOC, TTL
    tag: Optional[str]                    # Custom tag (max 20 chars)
```

### Order (received from broker)

```python
class Order(BaseModel):
    order_id: str                         # "220221000000001"
    status: str                           # COMPLETE, REJECTED, OPEN, etc.
    tradingsymbol: str
    exchange: str
    transaction_type: str
    order_type: str
    quantity: int
    filled_quantity: int                  # How much has been filled
    pending_quantity: int                 # How much is still open
    average_price: float                  # Avg fill price
    exchange_timestamp: Optional[str]     # Exchange server time
    exchange_update_timestamp: Optional[str]  # Last update time (broker)
    status_message: Optional[str]         # Rejection reason if REJECTED
```

### Signal (strategy output)

```python
class Signal(BaseModel):
    tradingsymbol: str
    exchange: Exchange
    transaction_type: TransactionType
    quantity: int
    price: Optional[float]
    order_type: OrderType = MARKET
    product: ProductType = MIS
    stop_loss: Optional[float]           # Converted to SL order type
    target: Optional[float]
    strategy_name: str
    confidence: float                     # 0-100
    metadata: dict[str, Any]
```

### ExecutionState (multi-leg tracking)

```python
@dataclass
class ExecutionState:
    group_id: str
    lifecycle: ExecutionLifecycle          # Current state machine state
    legs: Dict[str, LegState]             # All legs in this group
    interim_exposure: Dict[str, int]      # Unhedged deltas
    emergency_hedge_executed: bool
    emergency_hedge_orders: List[str]     # Order IDs of emergency hedges
```

### LegState (per-leg tracking)

```python
@dataclass
class LegState:
    leg_id: str
    leg_type: LegType                     # PROTECTION or RISK
    tradingsymbol: str
    quantity: int
    order_id: Optional[str]
    filled_quantity: int
    fill_price: Optional[float]
    status: str                           # CREATED, SUBMITTED, PENDING, FILLED, REJECTED
```

---

## 14. Error Handling & Escalation Matrix

### Exception Hierarchy

| Exception | Raised When | Handler Action |
|-----------|-------------|----------------|
| `KillSwitchError` | Daily loss exceeds threshold | Deactivate ALL strategies immediately |
| `RiskLimitError` | Signal fails risk validation | Reject signal, log warning |
| `OrderError` | Broker rejects order (4xx) | Log error, propagate to caller |
| `AuthenticationError` | Token expired/invalid (401/403) | Recreate HTTP session, re-auth |
| `NetworkError` | Connection timeout/failure | Retry with exponential backoff (max 3) |
| `RateLimitError` | HTTP 429 | Wait `retry_delay × 2^attempt` seconds |
| `RecursionError` | Hedge recursion exceeds depth 3 | Escalate to trader for manual intervention |

### Escalation Levels

```
Level 1: WARNING — logged, operation continues
  Examples: Partial fill detected, OI broadcast error, bar data seed failed

Level 2: ERROR — logged, operation fails gracefully
  Examples: Order rejected, margin fetch failed, cancel failed

Level 3: CRITICAL — logged, emergency action taken
  Examples: Kill switch triggered, emergency hedge, crash during exposure,
            gap-through detected, hedge recursion maxed, hedge placement failed
```

### Background Task Exception Handling

All `asyncio.create_task()` calls in `TradingService` use `_task_exception_handler` callback:

```python
@staticmethod
def _task_exception_handler(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("background_task_failed", task=task.get_name(), error=str(exc))
```

This ensures background task failures (ticker, reconciliation, position updates, OI broadcast) are **always logged** rather than silently swallowed.

---

## Appendix: Complete Event Timeline (Iron Condor Example)

```
T+0.000s  IronCondorStrategy._evaluate_entry() → 4 Signals
T+0.001s  TradingService._process_signal(sell_ce) → log + broadcast

T+0.002s  ExecutionEngine.execute_multi_leg_strategy(group)
T+0.003s  FreezeQuantityManager.split_signal() for each leg → 4 legs (no split)
T+0.004s  WorstCaseMarginSimulator.validate_margin() × 1.15 → OK
T+0.005s  Sort legs: [buy_ce, buy_pe, sell_ce, sell_pe]

T+0.006s  MarketStateGuard.is_safe_to_execute(buy_ce) → OK
T+0.007s  OrderCoordinator.place_order(buy_ce) → LiquidityChecker → Queue
T+0.132s  ExecutionQueue fires → KiteClient.place_order(buy_ce) → order_id_1 ✓

T+0.257s  OrderCoordinator.place_order(buy_pe) → order_id_2 ✓
T+0.382s  OrderCoordinator.place_order(sell_ce) → order_id_3 ✓
T+0.507s  OrderCoordinator.place_order(sell_pe) → order_id_4 ✓

T+0.650s  Check fills: buy_ce=COMPLETE, buy_pe=COMPLETE, sell_ce=OPEN, sell_pe=OPEN
T+0.750s  Poll: sell_ce → filled_quantity=50 → COMPLETE
T+0.850s  Poll: sell_pe → filled_quantity=50 → COMPLETE

T+0.860s  All 4 legs FILLED → lifecycle = HEDGED → CLOSED
T+0.861s  PersistenceLayer.save_state()
T+0.862s  Log "multi_leg_execution_complete" ✓

T+2.000s  Fast reconciliation confirms all 4 orders COMPLETE ✓
T+30.00s  Slow reconciliation: full order book consistency check ✓
T+30.01s  Position update: RiskManager.update_positions() ✓
```

### Emergency Scenario Timeline

```
T+0.000s  Iron Condor submitted (4 legs)
T+0.003s  FreezeQuantityManager.split_signal() → no split needed
T+0.004s  WorstCaseMarginSimulator.validate_margin() × 1.15 → OK (marginal)
T+0.130s  buy_ce FILLED ✓ (via OrderCoordinator → ExecutionQueue)
T+0.260s  buy_pe FILLED ✓
T+0.390s  sell_ce REJECTED ✗ (margin drifted 2% — but 1.15x caught most cases)

T+0.391s  DETECTED: rejected_count > 0
T+0.392s  EmergencyHedgeExecutor.execute_recovery()
T+0.393s    Reverse buy_ce: SELL 50 at MARKET
T+0.394s    LiquidityChecker: spread 1.2% → MARKET OK
T+0.395s    ExecutionQueue.enqueue(priority=PROTECTION) → order_id_5
T+0.396s    Reverse buy_pe: SELL 50 at MARKET → order_id_6
T+0.450s  lifecycle → EMERGENCY_HEDGED
T+0.451s  PersistenceLayer.save_state()
T+0.452s  Return success=True (hedged, no open exposure)

T+2.000s  Fast reconciliation: confirms order_id_5, _6 COMPLETE ✓
```

---

*End of document. For architecture diagrams, see `VISUAL_ARCHITECTURE_GUIDE.md`. For security details, see `SECURITY_AUDIT_REPORT.md`.*

---

## 15. Broker Microstructure Layer (OrderCoordinator)

> **Module**: `src/execution/order_coordinator.py` · **Added**: v2.0

The OrderCoordinator is a new orchestration layer between `ExecutionEngine` and `KiteClient` that handles 6 broker-specific concerns that were previously missing. Without this layer, orders were sent directly to Kite Connect without awareness of NSE freeze limits, market states, illiquidity, or rate limits.

### 15.1 Architecture

```
ExecutionEngine.execute_signal(signal)
    │
    ├── RiskManager validation
    ├── MarketStateGuard check          ← engine-level check
    ├── FreezeQuantityManager.split_signal()  ← engine-level split
    │
    └── For each child signal:
          _execute_single_signal(child)
            │
            └── OrderCoordinator.place_order(OrderRequest)
                  │
                  ├── [1] MarketStateGuard.is_safe_to_execute()  ← redundant safety
                  ├── [2] FreezeQuantityManager.split_order()
                  ├── [3] LiquidityChecker.check_liquidity()
                  │         └── MARKET → aggressive LIMIT if spread ≥ 5%
                  ├── [4] ExecutionQueue.enqueue()
                  │         └── Rate-paced at 8 orders/sec
                  └── [5] KiteClient.place_order() → order_id
```

### 15.2 FreezeQuantityManager

Prevents NSE freeze quantity rejections by splitting oversized orders into compliant child orders.

**NSE Freeze Limits** (lots, not shares):

| Underlying | Freeze Limit |
|-----------|:------------:|
| NIFTY | 1,800 |
| BANKNIFTY | 900 |
| FINNIFTY | 1,800 |
| MIDCPNIFTY | 4,200 |
| SENSEX (BSE) | 1,000 |
| BANKEX (BSE) | 1,500 |
| Default (equity) | 10,000 |

**Split Logic**:
```
split_signal(signal: Signal) → list[Signal]:
    freeze_limit = FREEZE_LIMITS.get(underlying, 10000)
    if signal.quantity <= freeze_limit:
        return [signal]  # No split needed
    
    chunks = []
    remaining = signal.quantity
    while remaining > 0:
        chunk_size = min(remaining, freeze_limit)
        child = signal.copy(quantity=chunk_size)
        chunks.append(child)
        remaining -= chunk_size
    return chunks
```

**Critical Design Decision**: Freeze split happens BEFORE margin simulation in `CorrectMultiLegOrderGroup.execute()`. If you simulate margin on qty=3600 but NSE rejects it at 1800 limit, the margin check was meaningless.

### 15.3 MarketStateGuard

Prevents orders from hanging indefinitely during exchange states that don't process new orders.

**Detection Methods**:

| Method | Checks | Blocks When |
|--------|--------|-------------|
| Time-based | System clock | 00:00–09:15 (pre-open), after 15:30 (close) |
| Quote-based | `get_quote()` depth | Circuit breaker: LTP at upper/lower limit with freeze |
| Circuit detection | Best bid/ask = 0 or frozen | No counterparty to fill |

**Caching**: Market state is cached for 5 seconds (`_cache_ttl = 5.0`) to avoid excessive API calls during burst signal periods.

**Circuit Breaker Detection**:
```python
# Check if stock is at circuit limit
if quote.lower_circuit_limit and quote.last_price <= quote.lower_circuit_limit:
    return MarketState.CIRCUIT_HALT
if quote.upper_circuit_limit and quote.last_price >= quote.upper_circuit_limit:
    return MarketState.CIRCUIT_HALT
```

### 15.4 ExecutionQueue

Priority FIFO queue with rate pacing to prevent burst flooding of the Kite API.

**Priority Levels**:

| Priority | Value | Use Case | Example |
|----------|:-----:|----------|---------|
| EMERGENCY | 0 | Emergency hedges, kill switch | Reverse naked short |
| PROTECTION | 1 | Stop-loss, protective legs | SL MARKET fallback |
| RISK_REDUCTION | 2 | Reduce exposure | Close winning leg |
| NORMAL | 3 | Strategy signals | New entry signal |

**Rate Pacing**:
- Target: **8 orders/second** (conservative vs Zerodha's ~10/sec limit)
- Minimum interval: **125ms** between orders
- Background `asyncio.Task` pulls from queue and enforces pacing
- Emergency orders bypass queue via `enqueue_immediate()`

**Backpressure**: When 5 strategies fire simultaneously at market open, the queue serializes them at 8/sec instead of sending 15+ orders in 100ms (which would trigger 429 Too Many Requests).

### 15.5 LiquidityChecker

Prevents catastrophic slippage when MARKET orders hit illiquid order books.

**Problem**: Options far OTM may show Bid:100 but next bid at 65. A MARKET sell fills at 65 — instant ₹35/lot slippage that wipes out the spread credit.

**Decision Matrix**:

| Spread % | Action | Reason |
|----------|--------|--------|
| < 5% | Send MARKET | Sufficient liquidity |
| ≥ 5% | Convert to aggressive LIMIT | Bid ± 0.5% to prevent cliff |
| No depth / 0 bids | ABORT (or last-resort MARKET) | Truly no liquidity |

**Aggressive LIMIT Calculation**:
```python
# For BUY: just above best ask
price = best_ask * 1.005

# For SELL: just below best bid  
price = best_bid * 0.995
```

Used in two places:
1. `OrderCoordinator.place_order()` — all normal orders
2. `HybridStopLossManager._place_market_fallback()` — SL failover (critical: cannot be unhedged)

### 15.6 TimestampResolver

Prevents WebSocket/REST race condition where a stale REST response overwrites a newer WebSocket update.

**Problem**: WebSocket delivers `COMPLETE` at T+1.2s. REST poll at T+1.5s returns cached `OPEN` from T+0.8s. Without timestamp comparison, the system reverts from COMPLETE → OPEN.

**Resolution Logic**:
```python
should_update(order_id, status, timestamp) → bool:
    │
    ├── First update for this order? → accept (return True)
    │
    ├── Terminal status already recorded (COMPLETE/REJECTED/CANCELLED)?
    │     → reject non-terminal overwrite (return False)
    │
    └── Compare exchange_update_timestamp:
          incoming > stored   → accept (return True)
          incoming ≤ stored   → reject (return False, log stale)
```

**Integration Point**: `CorrectMultiLegOrderGroup._process_state_update()` calls `TimestampResolver.should_update()` before applying any state change. This is Safeguard #6, layered on top of the existing 5 safeguards.

### 15.7 Margin Safety Factor (1.15x)

`WorstCaseMarginSimulator` now multiplies worst-case margin by **1.15** before comparing to available balance.

**Why 15%?** Between filling a protection leg (BUY PE) and a risk leg (SELL PE), the market can move 1-3%. This increases the margin requirement for the risk leg. Without buffer, the risk leg gets rejected → emergency hedge fires → expensive unwind.

```python
# Before (v1.0):
if worst_case_margin > available:   # Race condition: market moves 2% → rejection

# After (v2.0):
buffered_margin = worst_case_margin * 1.15   # 15% buffer absorbs drift
if buffered_margin > available:               # Rejects proactively, not reactively
```

### 15.8 Complete Pipeline Timeline (v2.0)

```
T+0.000s  IronCondorStrategy._evaluate_entry() → 4 Signals
T+0.001s  TradingService._process_signal(sell_ce) → log + broadcast

T+0.002s  ExecutionEngine.execute_multi_leg_strategy(group)

T+0.003s  CorrectMultiLegOrderGroup.execute(4 legs):
T+0.004s    FreezeQuantityManager.split_signal() ── 4 legs → 4 legs (no split needed)
T+0.005s    WorstCaseMarginSimulator.validate_margin() × 1.15 → OK
T+0.006s    Sort: [buy_ce, buy_pe, sell_ce, sell_pe]

T+0.007s    ExecutionEngine.execute_signal(buy_ce PROTECTION):
T+0.008s      MarketStateGuard.is_safe_to_execute() → OK
T+0.009s      FreezeQuantityManager.split_signal() → [buy_ce] (no split)
T+0.010s      OrderCoordinator.place_order(buy_ce):
T+0.011s        LiquidityChecker.check_liquidity() → spread 0.8% → MARKET OK
T+0.012s        ExecutionQueue.enqueue(priority=NORMAL)
T+0.137s        Queue fires (125ms pacing) → KiteClient.place_order() → order_id_1

T+0.262s  buy_pe:  OrderCoordinator → LiquidityChecker → Queue → order_id_2
T+0.387s  sell_ce: OrderCoordinator → LiquidityChecker → Queue → order_id_3
T+0.512s  sell_pe: OrderCoordinator → LiquidityChecker → Queue → order_id_4

T+0.650s  Fast recon check (2s interval): all OPEN → skip
T+1.012s  Fill: buy_ce=COMPLETE, buy_pe=COMPLETE
T+1.200s  WebSocket: order_id_1 COMPLETE (exchange_ts: T+0.980s)
T+1.500s  REST poll returns order_id_1 OPEN (exchange_ts: T+0.800s)  ← STALE
T+1.501s  TimestampResolver: 0.800 < 0.980 → REJECT stale update ✓

T+2.650s  Fast recon: sell_ce=COMPLETE, sell_pe=COMPLETE
T+2.660s  All 4 legs FILLED → lifecycle = HEDGED → CLOSED
T+2.661s  PersistenceLayer.save_state()

T+30.00s  Slow recon confirms all 4 orders COMPLETE ✓
T+30.01s  Position update: RiskManager.update_positions() ✓
```

### 15.9 Configuration Reference

| Parameter | Value | Location | Tuning |
|-----------|-------|----------|--------|
| Margin safety factor | 1.15 | `WorstCaseMarginSimulator.MARGIN_SAFETY_FACTOR` | Increase for volatile markets |
| Fast recon interval | 2.0s | `ExecutionEngine._fast_recon_interval` | Lower for faster rejection detection |
| Slow recon interval | 30.0s | `ExecutionEngine._slow_recon_interval` | Standard cleanup cycle |
| Rate limit | 8 orders/sec | `ExecutionQueue.MAX_ORDERS_PER_SECOND` | ≤ broker limit (~10/s) |
| Min order interval | 125ms | `ExecutionQueue.MIN_INTERVAL` | = 1000 / MAX_ORDERS_PER_SECOND |
| Spread threshold | 5.0% | `LiquidityChecker.MAX_SPREAD_PCT` | Wider for volatile instruments |
| Absolute spread cap | ₹5.0 | `LiquidityChecker.MAX_SPREAD_ABS` | Min spread for rejection |
| Aggressive limit buffer | 0.5% | `LiquidityChecker.AGGRESSIVE_LIMIT_BUFFER` | How far past best to place |
| Market state cache TTL | 5.0s | `MarketStateGuard._cache_ttl` | Lower for more reactive checks |
| Pre-open end time | 09:15 | `MarketStateGuard` | NSE official pre-open close |
| Market close time | 15:30 | `MarketStateGuard` | NSE official close |

---

*End of document. For architecture diagrams, see `VISUAL_ARCHITECTURE_GUIDE.md`. For security details, see `SECURITY_AUDIT_REPORT.md`.*
