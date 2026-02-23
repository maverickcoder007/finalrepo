# Complete Trading Flow: Login to Execution End

**Date**: February 21, 2026  
**Purpose**: Document the complete user journey from authentication through trade execution and settlement

---

## 1. LOGIN & AUTHENTICATION FLOW

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          LOGIN TO DASHBOARD                                 │
└─────────────────────────────────────────────────────────────────────────────┘

Step 1: User Access
┌──────────────────────┐
│  User opens webapp   │
│  http://localhost    │
└──────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│  src/api/webapp.py - Auth Middleware (Line 57)           │
│  ├─ Check if path is public (/login, /auth/*)            │
│  ├─ Check if user authenticated (SESSION_COOKIE)         │
│  ├─ If not authenticated → Redirect to /login            │
│  └─ If authenticated → Continue                          │
└──────────────────────────────────────────────────────────┘
           │
           ▼
Step 2: Login Page
┌──────────────────────────────────────────────────────────┐
│  @app.get("/login") - Line 61                            │
│  ├─ Render login.html template                           │
│  ├─ User enters: Zerodha ID, Email                       │
│  └─ Form posts to /auth/send-otp                         │
└──────────────────────────────────────────────────────────┘
           │
           ▼
Step 3: Send OTP
┌──────────────────────────────────────────────────────────┐
│  @app.post("/auth/send-otp") - Line 70                   │
│  ├─ Extract email, zerodha_id from request body          │
│  ├─ Validate email matches user config                   │
│  ├─ Call otp_auth.generate_otp(zerodha_id)               │
│  │  └─ Generates 6-digit OTP (6 min validity)            │
│  ├─ Call otp_auth.send_otp_email(otp, zerodha_id)        │
│  │  └─ Send via email | Dev: return in response          │
│  └─ Return {success: true, otp: "123456"}                │
└──────────────────────────────────────────────────────────┘
           │
           ▼
Step 4: Verify OTP
┌──────────────────────────────────────────────────────────┐
│  @app.post("/auth/verify-otp") - Line 104                │
│  ├─ Extract otp, zerodha_id from request body            │
│  ├─ Call otp_auth.verify_otp(otp, zerodha_id)            │
│  │  ├─ Check OTP validity (timestamp + 6 min)            │
│  │  ├─ If valid → Generate session_token                 │
│  │  └─ If invalid → Return error                         │
│  ├─ Set cookie: kta_session = session_token              │
│  │  └─ HTTPOnly, 7-day expiry, SameSite=lax              │
│  └─ Return {success: true, zerodha_id: "ABC123"}         │
└──────────────────────────────────────────────────────────┘
           │
           ▼
Step 5: Access Dashboard
┌──────────────────────────────────────────────────────────┐
│  @app.get("/") - Line 137                                │
│  ├─ Middleware verifies session_token valid              │
│  ├─ Call get_user_service(request)                       │
│  │  ├─ Extract zerodha_id from session token             │
│  │  ├─ Load user config (api_key, api_secret)            │
│  │  ├─ Create TradingService instance                    │
│  │  │  └─ Initialize KiteClient, RiskManager, etc        │
│  │  └─ Return service                                    │
│  ├─ Render dashboard.html with:                          │
│  │  ├─ is_authenticated = service.is_authenticated       │
│  │  ├─ is_running = service.is_running                   │
│  │  ├─ zerodha_id = extracted from token                 │
│  │  └─ static assets (CSS, JS)                           │
│  └─ User sees dashboard UI                               │
└──────────────────────────────────────────────────────────┘

═════════════════════════════════════════════════════════════════════════════

SESSION PERSISTENCE:
  ✓ Login creates session_token
  ✓ Token stored in HTTPOnly cookie (secure, not accessible to JS)
  ✓ Every request validates cookie
  ✓ WebSocket connections also validated via cookie

═════════════════════════════════════════════════════════════════════════════
```

---

## 2. SERVICE INITIALIZATION FLOW

```
┌──────────────────────────────────────────────────────────────────────┐
│           TradingService Initialization & Configuration              │
└──────────────────────────────────────────────────────────────────────┘

When user accesses dashboard, TradingService is created:

TradingService.__init__() - src/api/service.py Line 48
│
├─ 1. Authentication Setup
│  ├─ KiteAuthenticator(api_key, api_secret, zerodha_id)
│  │  └─ Prepares OAuth flow with Zerodha
│  └─ KiteClient(authenticator)
│     └─ HTTP async client for Zerodha API calls
│
├─ 2. Market Data Setup
│  ├─ KiteTicker(authenticator)
│  │  └─ WebSocket connection for live ticks
│  └─ Initializes tick buffers & callbacks
│
├─ 3. Trading Infrastructure Setup
│  ├─ RiskManager()
│  │  └─ Risk limits, kill switch, position tracking
│  ├─ ExecutionEngine(client, risk_manager)
│  │  └─ Place orders, check fills, reconcile
│  ├─ TradeJournal()
│  │  └─ Record all trades to disk
│  ├─ OptionChainBuilder()
│  │  └─ Build option chains for analysis
│  ├─ OITracker()
│  │  └─ Track open interest for hedging
│  └─ StockScanner()
│     └─ Fundamental analysis of stocks
│
├─ 4. Strategy Infrastructure
│  ├─ strategies: list[BaseStrategy] = []
│  └─ Available strategies loaded on-demand:
│     ├─ EMA Crossover
│     ├─ VWAP Breakout
│     ├─ Mean Reversion
│     ├─ RSI
│     ├─ Iron Condor
│     ├─ Straddle/Strangle
│     ├─ Bull Call Spread
│     └─ Bear Put Spread
│
├─ 5. Live Data Infrastructure
│  ├─ subscribed_tokens: list[int] = []
│  │  └─ Tokens to stream live
│  ├─ live_ticks: dict[token, tick_data]
│  │  └─ Current LTP, volume, OHLC for all symbols
│  └─ ws_clients: list[WebSocket]
│     └─ Connected dashboard clients for broadcasting
│
├─ 6. Trade Recording Setup
│  ├─ signal_log: list[signal_data]
│  │  └─ All generated signals logged here
│  └─ execution state
│     └─ Pending orders, fills, journal entries
│
└─ Service ready for trading operations
   Status: is_authenticated = (API key valid)
   Status: is_running = False (not streaming yet)

═══════════════════════════════════════════════════════════════════════════

SERVICE MANAGER (Singleton):
  └─ TradingServiceManager maintains dict[zerodha_id → TradingService]
     └─ Supports multiple users, each with own service instance

═══════════════════════════════════════════════════════════════════════════
```

---

## 3. STRATEGY SETUP & LIVE DATA FLOW

```
┌──────────────────────────────────────────────────────────────────────┐
│         Add Strategy & Start Live Trading                            │
└──────────────────────────────────────────────────────────────────────┘

Step 1: Add Strategy (Dashboard → API)
┌────────────────────────────────────────────────────────────┐
│  User clicks "Add Strategy" → Selects "EMA Crossover"      │
│  Sends: POST /api/strategies/add                           │
│  Body: {strategy: "ema_crossover", params: {...}}          │
└────────────────────────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────────────┐
│  src/api/service.py - add_strategy() - Line 240            │
│  ├─ Look up strategy class from strategy_map               │
│  │  └─ "ema_crossover" → EMACrossoverStrategy              │
│  ├─ Instantiate: strategy = cls(params)                    │
│  ├─ Add to self._strategies list                           │
│  └─ Return {success: true, name: "ema_crossover"}          │
└────────────────────────────────────────────────────────────┘

Step 2: Select Symbols & Start Live
┌────────────────────────────────────────────────────────────┐
│  User selects symbols: [NIFTY, BANKNIFTY, etc]             │
│  Sends: POST /api/live/start                               │
│  Body: {tokens: [256265029, 260105729, ...]}               │
└────────────────────────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────────────┐
│  src/api/service.py - start_live() - Line 284              │
│  ├─ Check not already running                              │
│  ├─ Set self._running = True                               │
│  ├─ Set self._subscribed_tokens = tokens                   │
│  ├─ Register ticker callbacks                              │
│  │  ├─ on_ticks → _on_ticks (process market data)          │
│  │  ├─ on_connect → _on_connect (status update)            │
│  │  ├─ on_close → _on_close (connection closed)            │
│  │  ├─ on_error → _on_error (error handling)               │
│  │  └─ on_order_update → _on_order_update (fills)          │
│  ├─ Seed historical data for indicators                    │
│  │  └─ _seed_bar_data() fetches 5 days of 5-min candles    │
│  ├─ Subscribe to ticks on broker                           │
│  │  └─ KiteTicker.subscribe(tokens, mode="quote")          │
│  ├─ Start WebSocket connection                             │
│  │  └─ asyncio.create_task(ticker.connect())               │
│  ├─ Start order reconciliation loop                         │
│  │  └─ asyncio.create_task(execution.start_reconciliation()) │
│  └─ Start position update loop                             │
│     └─ asyncio.create_task(_position_update_loop())        │
└────────────────────────────────────────────────────────────┘

Step 3: Connect WebSocket for Dashboard Updates
┌────────────────────────────────────────────────────────────┐
│  Dashboard client establishes WS connection:                │
│  WebSocket /ws/live                                        │
└────────────────────────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────────────┐
│  src/api/webapp.py - websocket_endpoint() - Line 442        │
│  ├─ Validate session cookie (zerodha_id)                   │
│  ├─ Load user config (API credentials)                     │
│  ├─ Get TradingService for this user                      │
│  ├─ Call service.register_ws_client(websocket)             │
│  │  └─ Add to service._ws_clients list                     │
│  ├─ Listen for incoming messages (ping/pong)               │
│  └─ On disconnect: unregister_ws_client()                  │
└────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════

State at this point:
  ✓ Strategy: EMA Crossover active
  ✓ Subscribed to: NIFTY, BANKNIFTY ticks
  ✓ Historical data loaded for indicators
  ✓ Live WebSocket connected for real-time updates
  ✓ Order reconciliation running (30-sec check)
  ✓ Position monitor running (30-sec check)

═══════════════════════════════════════════════════════════════════════════
```

---

## 4. LIVE DATA & SIGNAL GENERATION FLOW

```
┌──────────────────────────────────────────────────────────────────────┐
│        Tick Reception → Signal Generation → Broadcasting             │
└──────────────────────────────────────────────────────────────────────┘

Continuous process every tick (typically 100ms - 1sec intervals):

Step 1: Receive Tick from Broker WebSocket
┌────────────────────────────────────────────────────────────┐
│  Zerodha WebSocket                                         │
│  ├─ Send: {token: 256265029, ltp: 24500, volume: 1000} │
│  └─ Received by KiteTicker._on_message()                  │
└────────────────────────────────────────────────────────────┘
           │
           ▼
Step 2: Process Tick Data
┌────────────────────────────────────────────────────────────┐
│  src/api/service.py - _on_ticks() - Line 322               │
│  ├─ Iterate through all ticks in batch                     │
│  │                                                        │
│  ├─ For each tick:                                         │
│  │  ├─ 1. Update live_ticks cache                          │
│  │  │    └─ self._live_ticks[token] = tick_data            │
│  │  │                                                     │
│  │  ├─ 2. Update OI tracker (if option)                    │
│  │  │    ├─ Check NIFTY_TOKEN or SENSEX_TOKEN (spot)       │
│  │  │    ├─ Update spot price in OI tracker                │
│  │  │    └─ Update OI data (Open Interest)                 │
│  │  │                                                     │
│  │  ├─ 3. Update bar data for strategies                   │
│  │  │    ├─ Get strategy.bar_data (5-min candles)          │
│  │  │    ├─ Append new tick as row to DataFrame            │
│  │  │    ├─ Keep last 500 candles (memory limit)           │
│  │  │    └─ Update strategy's internal data                │
│  │  │                                                     │
│  │  └─ 4. Feed to strategies for signal generation         │
│  │     (see Step 3 below)                                  │
│  │                                                        │
│  └─ Broadcast tick update to all connected dashboards      │
│     └─ await _broadcast_ws({type: "ticks", data: ...})     │
└────────────────────────────────────────────────────────────┘
           │
           ▼
Step 3: Strategy Signal Generation
┌────────────────────────────────────────────────────────────┐
│  For each active strategy:                                 │
│  ├─ Call: signals = await strategy.on_tick(ticks)          │
│  │                                                        │
│  ├─ Strategy processes based on indicator logic:           │
│  │  ├─ EMA Crossover: Check if EMA50 crosses EMA200        │
│  │  ├─ RSI: Check if RSI(14) < 30 (oversold) or > 70 (overbought)
│  │  ├─ VWAP: Check if price breaks above/below VWAP        │
│  │  ├─ Mean Reversion: Check if price deviates from mean   │
│  │  ├─ Iron Condor: Check if spot in profitable zone       │
│  │  ├─ Bull Call Spread: Check entry conditions (delta, IV) │
│  │  └─ Bear Put Spread: Check entry conditions              │
│  │                                                        │
│  ├─ Returns: list[Signal] (may be empty if no signal)      │
│  │  └─ Each Signal contains:                               │
│  │     ├─ tradingsymbol: "NIFTY" or "20000CE"              │
│  │     ├─ exchange: NFO or NSE                             │
│  │     ├─ transaction_type: BUY or SELL                    │
│  │     ├─ quantity: 1, 50, etc                             │
│  │     ├─ price: entry price (may be None for MARKET)      │
│  │     ├─ stop_loss: stop loss price (optional)            │
│  │     ├─ target: profit target (optional)                 │
│  │     ├─ strategy_name: "ema_crossover"                   │
│  │     ├─ confidence: 0.0-1.0 confidence score             │
│  │     └─ metadata: {...} strategy-specific data           │
│  │                                                        │
│  └─ If signals generated: for sig in signals:              │
│     └─ Call _process_signal(sig) (see Step 4)              │
└────────────────────────────────────────────────────────────┘
           │
           ▼
Step 4: Process Signal (Log & Execute)
┌────────────────────────────────────────────────────────────┐
│  src/api/service.py - _process_signal() - Line 447          │
│  ├─ 1. Log signal to signal_log                            │
│  │    └─ service._signal_log.append(signal_data)            │
│  │                                                        │
│  ├─ 2. Broadcast to dashboard via WebSocket                │
│  │    └─ await _broadcast_ws({                             │
│  │        "type": "signal",                                │
│  │        "data": signal_data                              │
│  │       })                                                │
│  │                                                        │
│  ├─ 3. Execute order (see Step 5)                          │
│  │    └─ order_id = await execution.execute_signal(signal) │
│  │                                                        │
│  ├─ 4. Record to journal if execution successful           │
│  │    └─ journal.record_trade(...)                         │
│  │       └─ Saved to trades.json on disk                   │
│  │                                                        │
│  ├─ Error handling:                                        │
│  │  ├─ KillSwitchError: Deactivate all strategies          │
│  │  ├─ RiskLimitError: Log warning, don't execute          │
│  │  └─ Other exceptions: Log error, continue               │
│  │                                                        │
│  └─ Dashboard receives signal: User sees in log            │
└────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════

Timing:
  • Tick received → ~0ms
  • Bar updated → ~10ms
  • Strategy.on_tick() → ~20-50ms (indicator calculations)
  • Signal logged → ~60ms
  • Order placed → ~100-500ms (network latency)

Throughput:
  • NIFTY: ~10-20 ticks/sec = 10-20 signal evaluations/sec
  • Multiple symbols: Can handle 100+ ticks/sec efficiently

═══════════════════════════════════════════════════════════════════════════
```

---

## 5. ORDER EXECUTION FLOW

```
┌──────────────────────────────────────────────────────────────────────┐
│      Signal → Order Placement → Execution & Reconciliation           │
└──────────────────────────────────────────────────────────────────────┘

Step 1: Execute Signal
┌────────────────────────────────────────────────────────────┐
│  src/execution/engine.py - execute_signal() - Line 42      │
│  ├─ Check kill switch (not active)                         │
│  │  └─ If active: raise KillSwitchError                    │
│  │                                                        │
│  ├─ Validate signal with RiskManager                       │
│  │  └─ is_valid, reason = risk.validate_signal(signal)     │
│  │     ├─ Check daily loss limit                           │
│  │     ├─ Check position size limit                        │
│  │     ├─ Check total exposure limit                       │
│  │     └─ If invalid: raise RiskLimitError                 │
│  │                                                        │
│  ├─ Convert signal to OrderRequest                         │
│  │  └─ Copy: symbol, exchange, qty, price, order_type      │
│  │                                                        │
│  ├─ Apply stop-loss (NEW FIX #2)                           │
│  │  └─ _apply_stop_loss_to_order(order, signal)            │
│  │     ├─ If signal.stop_loss specified:                   │
│  │     ├─  Set trigger_price = stop_loss                   │
│  │     ├─  Set execution_price = stop_loss ± slippage      │
│  │     └─  Set order_type = SL (Stop Loss with Limit)      │
│  │                                                        │
│  ├─ Place order on broker                                  │
│  │  └─ order_id = await client.place_order(order_request)  │
│  │     └─ HTTP POST to Zerodha /orders/regular             │
│  │                                                        │
│  ├─ Track pending order                                    │
│  │  └─ self._pending_orders[order_id] = order_request      │
│  │                                                        │
│  └─ Log execution                                          │
│     └─ logger.info("signal_executed", ...)                │
└────────────────────────────────────────────────────────────┘
           │
           ▼
Step 2: Broker Processes Order
┌────────────────────────────────────────────────────────────┐
│  Zerodha Broker                                            │
│  ├─ ACK: Returns order_id (immediate)                      │
│  │  └─ Order enqueued for matching                         │
│  │                                                        │
│  ├─ Order Status: OPEN / PENDING                           │
│  │  └─ Waiting for market liquidity                        │
│  │                                                        │
│  ├─ Order Status: COMPLETE (when filled)                   │
│  │  ├─ filled_quantity = requested quantity                │
│  │  ├─ average_price = execution price                     │
│  │  └─ Callback: on_order_update(data)                     │
│  │                                                        │
│  └─ Order Status: REJECTED / CANCELLED                     │
│     └─ Reason: Margin, circuit limit, instrument closed    │
└────────────────────────────────────────────────────────────┘
           │
           ▼
Step 3: Order Update Webhook
┌────────────────────────────────────────────────────────────┐
│  Zerodha sends order update via WebSocket                  │
│  ├─ Payload: order_id, status, filled_qty, avg_price, etc. │
│  └─ Received by KiteTicker.on_order_update()               │
└────────────────────────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────────────┐
│  src/api/service.py - _on_order_update() - Line 411        │
│  ├─ Extract order data from webhook                        │
│  │  └─ order_id, status, quantity, average_price, etc.     │
│  │                                                        │
│  ├─ Record trade to journal                                │
│  │  └─ journal.record_trade(                               │
│  │       strategy="live",                                  │
│  │       tradingsymbol=...,                                │
│  │       quantity=...,                                    │
│  │       price=...,                                       │
│  │       order_id=...,                                    │
│  │       status=...                                       │
│  │     )                                                   │
│  │     └─ Saved to trades.json + memory                    │
│  │                                                        │
│  └─ Broadcast update to dashboard                          │
│     └─ _broadcast_ws({type: "order_update", ...})          │
└────────────────────────────────────────────────────────────┘

Step 4: Continuous Reconciliation (Every 30 seconds)
┌────────────────────────────────────────────────────────────┐
│  src/execution/engine.py - reconcile_orders() - Line 159    │
│  ├─ Running in background loop                             │
│  │  └─ started by: execution.start_reconciliation_loop()   │
│  │                                                        │
│  ├─ Fetch all pending orders from broker                   │
│  │  └─ broker_orders = await client.get_orders()           │
│  │                                                        │
│  ├─ Compare with local pending orders                      │
│  │  └─ For each local pending order:                       │
│  │     ├─ Check broker status                              │
│  │     ├─ If COMPLETE/REJECTED/CANCELLED:                  │
│  │     │  ├─ Move to _filled_orders dict                   │
│  │     │  └─ Remove from _pending_orders                   │
│  │     └─ Update filled_quantity, average_price            │
│  │                                                        │
│  └─ Result summary                                         │
│     └─ {matched: 5, mismatched: 0, errors: []}             │
└────────────────────────────────────────────────────────────┘

Step 5: Position Updates (Every 30 seconds)
┌────────────────────────────────────────────────────────────┐
│  src/api/service.py - _position_update_loop() - Line 493   │
│  ├─ Running in background loop while self._running         │
│  │                                                        │
│  ├─ Fetch positions from broker                            │
│  │  └─ positions = await client.get_positions()            │
│  │     └─ Returns: {net: [Position], day: [Position]}      │
│  │                                                        │
│  ├─ Update RiskManager with positions                      │
│  │  ├─ risk.update_positions(positions.net)                │
│  │  └─ Calculates total exposure for next risk check       │
│  │                                                        │
│  ├─ Calculate P&L                                          │
│  │  ├─ daily_pnl = sum(p.pnl for p in positions.net)       │
│  │  ├─ risk.update_daily_pnl(daily_pnl)                    │
│  │  │  └─ Checks kill switch threshold                     │
│  │  └─ Check if daily loss limit hit                       │
│  │     └─ Activate kill switch if exceeded                 │
│  │                                                        │
│  └─ Broadcast to dashboard                                 │
│     └─ _broadcast_ws({type: "positions", pnl: ...})        │
└────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════

Error Handling in Execution:

KillSwitchError
  └─ Occurs when daily loss exceeds threshold (e.g., -₹50,000)
  ├─ Trigger: risk.update_daily_pnl(daily_pnl)
  ├─ Action: Stop all strategies, log CRITICAL
  └─ Recovery: Manual reset by admin

RiskLimitError
  └─ Occurs when signal fails risk checks
  ├─ Reasons: 
  │  ├─ Position size exceeds limit (e.g., max 100 qty)
  │  ├─ Total exposure exceeds limit (e.g., max ₹100M)
  │  └─ Daily loss already hit
  ├─ Action: Log warning, skip this trade
  └─ Recovery: Automatic (try next signal)

OrderError
  └─ Occurs when broker rejects order
  ├─ Reasons:
  │  ├─ Insufficient margin
  │  ├─ Circuit limit hit
  │  ├─ Instrument not available
  │  └─ Bad order parameters
  ├─ Action: Log error, don't record to journal
  └─ Recovery: Manual intervention or automatic retry

═══════════════════════════════════════════════════════════════════════════
```

---

## 6. END OF EXECUTION: STOP & SETTLEMENT

```
┌──────────────────────────────────────────────────────────────────────┐
│          Stop Live Trading & Settlement                              │
└──────────────────────────────────────────────────────────────────────┘

Step 1: Stop Live Trading (User clicks Stop)
┌────────────────────────────────────────────────────────────┐
│  Dashboard sends: POST /api/live/stop                       │
└────────────────────────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────────────┐
│  src/api/service.py - stop_live() - Line 315               │
│  ├─ Set self._running = False                              │
│  │  └─ Stops position_update_loop & other async tasks      │
│  │                                                        │
│  ├─ Stop order reconciliation                              │
│  │  └─ await execution.stop_reconciliation_loop()          │
│  │     └─ Stops reconcile_orders() background task         │
│  │                                                        │
│  ├─ Disconnect from broker WebSocket                       │
│  │  └─ await ticker.disconnect()                           │
│  │     ├─ Unsubscribe from all symbol tokens               │
│  │     └─ Close WebSocket connection                       │
│  │                                                        │
│  └─ Return {success: true}                                 │
└────────────────────────────────────────────────────────────┘
           │
           ▼
Step 2: Final Reconciliation
┌────────────────────────────────────────────────────────────┐
│  Before stopping, final reconciliation happens:             │
│  ├─ Check all pending orders one last time                 │
│  ├─ Move any COMPLETE orders to filled                     │
│  ├─ Cancel remaining OPEN orders (optional)                │
│  └─ Final count of executed orders                         │
└────────────────────────────────────────────────────────────┘

Step 3: Final Position Report
┌────────────────────────────────────────────────────────────┐
│  Dashboard fetches final position state:                    │
│  ├─ GET /api/positions                                     │
│  ├─ GET /api/holdings                                      │
│  ├─ GET /api/journal                                       │
│  └─ GET /api/risk/summary                                  │
└────────────────────────────────────────────────────────────┘
           │
           ▼
Step 4: Trade Journal Summary
┌────────────────────────────────────────────────────────────┐
│  src/api/service.py - get_journal_summary()                │
│  └─ Returns:                                               │
│     ├─ total_trades: 25                                    │
│     ├─ winning_trades: 16                                  │
│     ├─ losing_trades: 9                                    │
│     ├─ win_rate: 64%                                       │
│     ├─ total_pnl: ₹15,000                                  │
│     ├─ avg_win: ₹2,000                                     │
│     ├─ avg_loss: ₹1,000                                    │
│     ├─ profit_factor: 1.5                                  │
│     └─ trades: [                                           │
│        {timestamp, strategy, symbol, type, qty, price, pnl},
│        ...                                                 │
│      ]                                                     │
└────────────────────────────────────────────────────────────┘

Step 5: Risk Summary
┌────────────────────────────────────────────────────────────┐
│  src/api/service.py - get_risk_summary()                   │
│  └─ Returns:                                               │
│     ├─ daily_pnl: ₹15,000                                  │
│     ├─ total_exposure: ₹5,000,000                          │
│     ├─ max_daily_loss: -₹50,000                            │
│     ├─ max_exposure: ₹10,000,000                           │
│     ├─ kill_switch_active: false                           │
│     ├─ position_count: 2                                   │
│     ├─ trade_count: 25                                     │
│     ├─ utilization_pct: 50%                                │
│     └─ session_start: "2024-02-21T09:00:00"                │
└────────────────────────────────────────────────────────────┘

Step 6: Execution Summary
┌────────────────────────────────────────────────────────────┐
│  src/api/service.py - get_execution_summary()              │
│  └─ Returns:                                               │
│     ├─ pending_orders: 0                                   │
│     ├─ filled_orders: 25                                   │
│     └─ pending_order_ids: []                               │
└────────────────────────────────────────────────────────────┘

Step 7: T+1 Settlement (Next Day)
┌────────────────────────────────────────────────────────────┐
│  Zerodha settlement process (automatic):                    │
│  ├─ All trades from Friday → Settlement on Monday          │
│  ├─ Delivery transfers (T+2 for securities)                │
│  ├─ Margin released (if intraday trade closed)             │
│  └─ Dividend/Interest adjustments                          │
│                                                           │
│  User can see next day:                                    │
│  ├─ Final balances                                         │
│  ├─ Holdings updated                                       │
│  ├─ P&L finalized                                          │
│  └─ Available margin updated                               │
└────────────────────────────────────────────────────────────┘

Step 8: End of Day Cleanup (Optional)
┌────────────────────────────────────────────────────────────┐
│  System can optionally:                                    │
│  ├─ Archive journal entries to S3/DB                       │
│  ├─ Reset daily counters for next trading day              │
│  ├─ Generate performance reports                           │
│  ├─ Send email summary to user                             │
│  └─ Clear tick cache to free memory                        │
└────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════

Logout Process:
┌────────────────────────────────────────────────────────────┐
│  User clicks Logout                                        │
│  └─ POST /auth/logout                                      │
│     ├─ Stop live trading (if running)                      │
│     ├─ Close WebSocket connections                         │
│     ├─ Call otp_auth.logout(session_token)                 │
│     │  └─ Invalidate session in server memory              │
│     └─ Delete kta_session cookie                           │
│     └─ Redirect to /login                                  │
└────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════
```

---

## 7. COMPLETE FLOW DIAGRAM

```
LOGIN ────┐
          │
          ▼
   EMAIL + ZERODHA_ID
          │
          ▼
   SEND OTP ─────► VERIFY OTP
          │          │
          ▼          ▼
    INVALID      VALID
          │          │
          ▼          ▼
    RETRY         SESSION TOKEN
                      │
                      ▼
                   DASHBOARD
                      │
          ┌───────────┼───────────┐
          │           │           │
          ▼           ▼           ▼
       STATUS      PROFILE     MARGINS
       CHECK       FETCH       FETCH
          │           │           │
          ▼           ▼           ▼
       ADD STRATEGY ──────────────┐
          │                       │
          ▼                       ▼
       SUBSCRIBE SYMBOLS ────── START LIVE
          │
          ├─ Websocket ticker.connect()
          ├─ execution.start_reconciliation_loop()
          ├─ _position_update_loop()
          └─ Dashboard WS connect (/ws/live)
          │
          ▼
       RECEIVE TICKS ◄────────────────────┐
          │                               │
          │ (Every 100-1000ms)            │
          │                               │
          ├─► Update bar data             │
          ├─► Strategy.on_tick()          │
          ├─► SignalGenerate?             │
          │   │                           │
          │   YES ──► _process_signal()   │
          │   │       │                   │
          │   │       ├─► Risk check      │
          │   │       ├─► Validate        │
          │   │       ├─► execute_signal()
          │   │       ├─► place_order()   │
          │   │       ├─► Journal record  │
          │   │       └─► Broadcast WS    │
          │   │                           │
          │   NO ──► Continue             │
          │                               │
          └──────────────────────────────┘
          
       RECONCILIATION (Every 30s)
          │
          ├─► get_orders()
          ├─► check fills
          ├─► update status
          └─► record completion
          
       POSITION UPDATE (Every 30s)
          │
          ├─► get_positions()
          ├─► update risk manager
          ├─► check kill switch
          └─► broadcast to dashboard
          
       USER CLICKS STOP
          │
          ▼
       stop_live()
          │
          ├─► Set _running = False
          ├─► stop_reconciliation_loop()
          ├─► ticker.disconnect()
          └─► Wait for final settlement
          │
          ▼
       FINAL REPORT
          │
          ├─► get_journal_summary()
          ├─► get_risk_summary()
          ├─► get_execution_summary()
          └─► render to dashboard
          │
          ▼
       LOGOUT
          │
          ├─► otp_auth.logout()
          ├─► Delete cookie
          └─► Redirect to /login

═══════════════════════════════════════════════════════════════════════════
```

---

## 8. DATA FLOW ARCHITECTURE

```
┌────────────────────────────────────────────────────────────────────┐
│                    DATA FLOW ARCHITECTURE                          │
└────────────────────────────────────────────────────────────────────┘

EXTERNAL SYSTEMS:
┌─────────────┐
│ Zerodha API │ ◄─► Authenticate, Place Orders, Get Positions, etc.
├─────────────┤
│ - REST API  │  (/orders, /positions, /holdings, /margins, etc.)
│ - WebSocket │  (Live ticks, order updates, market data)
└─────────────┘
      ▲
      │ WebSocket ticks
      │ Order updates
      │
      ▼
┌──────────────────────────────────────────────────────────────────┐
│                    KiteTicker                                    │
│  ├─ WebSocket connection to Zerodha                              │
│  ├─ Subscribe/unsubscribe to symbols                             │
│  ├─ Callback on_ticks → service._on_ticks()                      │
│  ├─ Callback on_order_update → service._on_order_update()        │
│  └─ Handles reconnection, heartbeat, rate limiting               │
└──────────────────────────────────────────────────────────────────┘
      │
      │ Tick objects (Tick model)
      │ Order updates (dict)
      │
      ▼
┌──────────────────────────────────────────────────────────────────┐
│                   TradingService                                 │
│  ├─ _on_ticks(ticks: list[Tick])                                 │
│  │  ├─ Update live_ticks dict                                    │
│  │  ├─ Feed to strategies for signal generation                  │
│  │  ├─ Broadcast tick update to dashboard                        │
│  │  └─ Feed OI tracker                                           │
│  │                                                              │
│  ├─ Strategy.on_tick() → generates signals                       │
│  │                                                              │
│  ├─ _process_signal(signal)                                      │
│  │  ├─ Log signal                                               │
│  │  ├─ Execute via ExecutionEngine                              │
│  │  ├─ Record to journal                                        │
│  │  └─ Broadcast signal to dashboard                            │
│  │                                                              │
│  ├─ _on_order_update(order_data)                                 │
│  │  ├─ Record to journal                                        │
│  │  └─ Broadcast to dashboard                                   │
│  │                                                              │
│  ├─ _position_update_loop()                                      │
│  │  ├─ Get positions every 30s                                  │
│  │  ├─ Update risk manager                                      │
│  │  └─ Broadcast P&L to dashboard                               │
│  │                                                              │
│  └─ _broadcast_ws(data) → sends to all dashboard clients         │
└──────────────────────────────────────────────────────────────────┘
      │
      ├──────────────────────────────────────────┐
      │                                          │
      ▼                                          ▼
┌─────────────────────────┐       ┌──────────────────────────┐
│  KiteClient (HTTP)      │       │  Dashboard WebSocket     │
│  - place_order()        │       │  (/ws/live)              │
│  - get_orders()         │       │  ├─ Receives ticks       │
│  - get_positions()      │       │  ├─ Receives signals     │
│  - get_margins()        │       │  ├─ Receives P&L         │
│  - get_holdings()       │       │  └─ Receives order updates
│  └─ get_profile()       │       └──────────────────────────┘
└─────────────────────────┘              │
      │                                   │ JSON messages
      │ REST API calls                    │
      │                                   ▼
      └──────────────► Zerodha ◄─────────── Dashboard HTML/JS
      
INTERNAL DATA STRUCTURES:
┌──────────────────────────────────────────────────────────────────┐
│ TradeJournal                                                    │
│  ├─ _entries: list[TradeJournalEntry]  (memory)                 │
│  └─ File: data/trades.json              (disk persistence)      │
│     └─ Loaded on startup, appended on each trade                │
│                                                                │
│ PositionTracker                                                │
│  ├─ _positions: dict[symbol → [Position]]                      │
│  ├─ _by_strategy: dict[strategy → [Position]]                  │
│  ├─ Track entry/exit prices                                    │
│  └─ Calculate P&L real-time                                    │
│                                                                │
│ RiskManager                                                    │
│  ├─ _daily_pnl: float (updated every 30s)                       │
│  ├─ _total_exposure: float (updated every 30s)                  │
│  ├─ Checks kill switch threshold                               │
│  └─ Validates each signal before execution                     │
└──────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════
```

---

## 9. KEY FILES & COMPONENTS

| File | Purpose | Key method |
|------|---------|-----------|
| [main.py](main.py) | FastAPI app startup | `main()` |
| [src/api/webapp.py](src/api/webapp.py) | HTTP endpoints, WebSocket | `app`, auth middleware |
| [src/api/service.py](src/api/service.py) | Core trading logic | `TradingService`, `_on_ticks()`, `_process_signal()` |
| [src/api/client.py](src/api/client.py) | Zerodha API client | `place_order()`, `get_positions()` |
| [src/api/ticker.py](src/api/ticker.py) | WebSocket tick stream | `subscribe()`, `connect()` |
| [src/execution/engine.py](src/execution/engine.py) | Order execution | `execute_signal()`, `reconcile_orders()` |
| [src/execution/order_group.py](src/execution/order_group.py) | Multi-leg atomicity (NEW) | `execute_with_rollback()` |
| [src/risk/manager.py](src/risk/manager.py) | Risk limits, validation | `validate_signal()`, `update_daily_pnl()` |
| [src/risk/position_validator.py](src/risk/position_validator.py) | Hedge ratio validation (NEW) | `validate_multi_leg_hedge()` |
| [src/data/journal.py](src/data/journal.py) | Trade recording | `record_trade()`, `get_summary()` |
| [src/data/position_tracker.py](src/data/position_tracker.py) | Position tracking | `add_position()`, `get_pnl()` |
| [src/options/exercise_handler.py](src/options/exercise_handler.py) | Exercise/assignment (NEW) | `monitor_exercises()`, `_handle_exercise()` |
| [src/auth/authenticator.py](src/auth/authenticator.py) | OAuth with Zerodha | `generate_session()` |
| [src/auth/otp_auth.py](src/auth/otp_auth.py) | OTP login | `verify_otp()`, `generate_otp()` |

---

## 10. SUMMARY TIMELINE

```
T+0:    User enters credentials
T+1:    OTP sent to email
T+5:    User verifies OTP
T+10:   Session created, dashboard loaded
T+15:   User adds strategies
T+20:   User subscribes to symbols, clicks Start
T+25:   Live data streaming begins
T+100:  First tick received
T+120:  First signal generated
T+300:  Order placed
T+400:  Order filled
T+1000: Position updated
T+5000: User clicks Stop
T+5100: Final reconciliation
T+5200: Dashboard shows final P&L
        User logs out
```

---

## 11. ERROR HANDLING & RECOVERY

| Error Type | Trigger | Action | Recovery |
|------------|---------|--------|----------|
| **Invalid OTP** | Wrong code entered | Reject, ask retry | User re-enters |
| **Kill Switch** | Daily loss > limit | Deactivate all strategies | Manual admin reset |
| **Risk Limit** | Position > max size | Skip trade | Auto-retry next signal |
| **Margin Insufficient** | Order placement | Reject order | Wait for margin release |
| **Partial Fill** | Only 30/50 qty filled | Rollback all legs (NEW) | Retry strategy logic |
| **WebSocket Disconnect** | Connection lost | Attempt reconnect (5x) | Manual restart |
| **Order Reject** | Bad parameters | Log error | User modifies params |
| **Hedge Mismatch** | Call/put imbalance | Alert (NEW) | Auto-rebalance or manual |

These 11 sections document the **complete end-to-end flow** from user login through trade execution, P&L tracking, and settlement.

