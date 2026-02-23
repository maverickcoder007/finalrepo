# Kite Trading Agent

## Overview
Production-grade algorithmic trading agent with interactive web dashboard, built in Python. Fully integrated with Zerodha Kite Connect API v3. Supports live trading, options strategies, backtesting, and real-time risk management.

## Architecture
```
src/
  auth/
    authenticator.py - Per-user Kite OAuth with isolated token files
    otp_auth.py      - OTP login with session-to-zerodha_id binding
    user_config.py   - UserConfigService: Zerodha ID -> API key/secret mapping (data/users.json)
  api/
    client.py     - Async REST client wrapping all Kite Connect v3 endpoints
    ticker.py     - WebSocket client for real-time market data
    service.py    - TradingService (per-user) + TradingServiceManager (instance cache)
    webapp.py     - FastAPI web application with REST + WebSocket endpoints (user-scoped)
  analysis/
    indicators.py - Technical indicators (SMA, EMA, RSI, MACD, ADX, ATR, BB, volume ratio)
    scanner.py    - Stock scanner with super performance criteria, trigger detection
  strategy/       - Pluggable strategy framework (EMA, VWAP, MeanRev, RSI)
  options/
    greeks.py     - Black-Scholes Greeks/IV engine
    chain.py      - Option chain builder with max pain, PCR
    strategies.py - Iron Condor, Straddle/Strangle, Bull/Bear Spreads
  execution/      - Smart order routing, slippage protection, reconciliation
  risk/           - Risk management, position sizing, kill switch
  data/           - Pydantic models, backtest engine, trade journal
  utils/          - Config, structured logging, exceptions
templates/        - Jinja2 HTML templates (dashboard)
static/           - CSS + JS for frontend
main.py           - FastAPI server entry point (port 5000)
```

## Multi-User Architecture
- **UserConfigService**: Maps Zerodha IDs to API key/secret pairs, stored in `data/users.json`
- **Per-user KiteAuthenticator**: Each user gets own authenticator with own API credentials and token file (`.kite_session_{zerodha_id}.json`)
- **TradingServiceManager**: Caches per-user TradingService instances keyed by Zerodha ID
- **Session binding**: OTP sessions store `zerodha_id`, all API routes resolve user from session cookie
- **Data isolation**: Each user gets their own TradingService with own KiteClient, strategies, risk manager, etc.
- **Login flow**: Requires Zerodha ID + email, validates against UserConfigService before issuing OTP

## Key Components
- **Web Dashboard**: Interactive UI with live market data, positions, orders, signals, strategy management, risk controls, stock analysis
- **KiteClient**: Async REST client with rate limiting, retry logic, all Kite v3 endpoints
- **KiteTicker**: WebSocket client with binary tick parsing, auto-reconnection, heartbeat keep-alive
- **Options Module**: Black-Scholes Greeks, IV calculation, option chain builder, max pain
- **Analysis Module**: Technical indicators (SMA, EMA, RSI, MACD, ADX, ATR, Bollinger Bands), super performance stock scanner (Minervini Trend Template), daily/weekly/monthly gainers & losers, trigger detection
- **8 Strategies**: EMA Crossover, VWAP Breakout, Mean Reversion, RSI, Iron Condor, Straddle/Strangle, Bull Call Spread, Bear Put Spread
- **RiskManager**: Daily loss limits, position sizing, exposure control, kill switch
- **ExecutionEngine**: Order routing, slippage protection, partial fill handling, reconciliation loop
- **BacktestEngine**: Vectorized backtesting with Sharpe, CAGR, drawdown, win rate
- **TradeJournal**: PnL tracking by instrument and strategy

## Tech Stack
- Python 3.11, FastAPI, uvicorn
- asyncio, aiohttp, websockets
- Pydantic v2, pydantic-settings
- pandas, numpy, scipy (Black-Scholes)
- structlog, aiolimiter
- Jinja2, vanilla JS frontend

## API Endpoints
- `GET /` - Dashboard
- `GET /api/status` - Full system status
- `GET /api/profile`, `/api/margins`, `/api/orders`, `/api/positions`, `/api/holdings`
- `POST /api/strategies/add`, `/api/strategies/remove`, `/api/strategies/toggle`
- `POST /api/live/start`, `/api/live/stop`
- `GET /api/signals`, `/api/ticks`, `/api/risk`, `/api/journal`
- `GET /api/options/chain`
- `GET /api/oi/summary?underlying=NIFTY` - OI tracker summary for NIFTY/SENSEX
- `GET /api/oi/all` - Combined OI data for all tracked underlyings
- `POST /api/oi/start` - Start OI tracking with instrument registration
- `GET /api/analysis/status` - Scanner status (scanning, last_scan_time)
- `POST /api/analysis/scan` - Run full NIFTY 50 stock scan
- `GET /api/analysis/results` - Get cached analysis results
- `POST /api/risk/kill-switch`
- `WS /ws/live` - Live WebSocket stream with heartbeat keep-alive (ticks, signals, positions)
- `WS /ws/oi` - Dedicated OI WebSocket stream (continuous OI updates every 2s)

## Configuration
Environment variables via `.env` file (see `.env.template`).

## Recent Changes
- 2026-02-21: Multi-user support - Zerodha ID login, per-user API credentials (data/users.json), isolated TradingService instances, session-to-user binding
- 2026-02-21: Added stock analysis module with technical indicators, super performance scanner, daily/weekly analysis, trigger detection, and Analysis tab in dashboard
- 2026-02-21: Enhanced WebSocket with heartbeat ping/pong keep-alive and exponential backoff reconnection
- 2026-02-21: Added interactive web dashboard, options trading module, FastAPI backend
- 2026-02-21: Initial build - complete trading agent with all modules
