# AlgoTrader — Agent Tool Reference

> **Canonical reference for AI agent traders.**
> Every REST endpoint, MCP tool, MCP resource, and WebSocket channel exposed by the AlgoTrader platform is documented here with signatures, parameters, and usage notes so an autonomous agent can discover and invoke any capability.

---

## 1  System Overview

| Property | Value |
|----------|-------|
| **Base URL** | `http://localhost:5000` (configurable via `PORT` env var) |
| **Framework** | FastAPI (Python 3.13) with Uvicorn |
| **Broker** | Zerodha Kite Connect |
| **Data Stores** | SQLite — `data/journal.db`, `data/market_data.db`, `data/fno_data.db`, `data/execution_state.db` |
| **MCP Server** | `StockResearch` via `mcp.server.fastmcp.FastMCP` |
| **Authentication** | OTP-based email auth → session cookie (`session_id`) **or** Zerodha OAuth callback |

### Authentication Flow

1. **Send OTP** — `POST /auth/send-otp` with `{ email, zerodha_id }`
2. **Verify OTP** — `POST /auth/verify-otp` with `{ otp, zerodha_id }` → sets `session_id` cookie
3. **Zerodha Login** — Redirect user to the URL from `GET /api/login-url`, Zerodha calls back to `GET /api/auth/callback`
4. **Manual Token** — `POST /api/authenticate` with `{ request_token }` if you already have the Kite request token

All `/api/*` endpoints require valid auth unless otherwise noted.

---

## 2  REST API Endpoints

### 2.1  Auth & Session

| Method | Path | Parameters / Body | Description |
|--------|------|-------------------|-------------|
| GET | `/login` | — | HTML login page |
| POST | `/auth/send-otp` | `{ email, zerodha_id }` | Send OTP |
| POST | `/auth/verify-otp` | `{ otp, zerodha_id }` | Verify OTP → session cookie |
| POST | `/auth/logout` | — | Clear session |
| GET | `/api/auth/callback` | `?request_token`, `?status` | Zerodha OAuth callback |
| GET | `/api/auth/postback` | — | Zerodha postback (GET) |
| POST | `/api/auth/postback` | *(any)* | Zerodha postback (POST) |
| POST | `/api/authenticate` | `{ request_token }` | Manual Kite token auth |

---

### 2.2  Dashboard & Account

| Method | Path | Parameters / Body | Description |
|--------|------|-------------------|-------------|
| GET | `/` | — | HTML dashboard |
| GET | `/api/status` | `?full=false` | System status (auth, strategies, risk, journal, ticks, signals; optionally margins/positions/holdings/profile) |
| GET | `/api/login-url` | — | Zerodha login URL |
| GET | `/api/profile` | — | User profile |
| GET | `/api/margins` | — | Margin info |
| GET | `/api/orders` | — | All orders |
| GET | `/api/positions` | — | All positions |
| GET | `/api/holdings` | — | All holdings |
| GET | `/api/risk` | — | Risk summary |

---

### 2.3  Preflight & Risk

| Method | Path | Parameters / Body | Description |
|--------|------|-------------------|-------------|
| GET | `/api/preflight` | `?mode="live"` | Full pre-flight checklist |
| GET | `/api/preflight/last` | — | Last cached preflight report |
| POST | `/api/risk/kill-switch` | `{ activate: bool }` | Activate / deactivate kill switch — **halts all trading** |

---

### 2.4  Journal & Signals

| Method | Path | Parameters / Body | Description |
|--------|------|-------------------|-------------|
| GET | `/api/journal` | — | Basic journal summary |
| GET | `/api/signals` | `?limit=50` | Recent trading signals |

---

### 2.5  Strategy Management

| Method | Path | Parameters / Body | Description |
|--------|------|-------------------|-------------|
| GET | `/api/strategies` | — | List active strategies |
| POST | `/api/strategies/add` | `{ name, params, timeframe="5minute", require_tested=true }` | Add strategy to live trading |
| GET | `/api/strategies/tested` | — | Strategies eligible for live |
| POST | `/api/strategies/remove` | `{ name }` | Remove strategy |
| POST | `/api/strategies/toggle` | `{ name }` | Enable / disable strategy |
| GET | `/api/analysis-strategies` | — | All analysis strategies (for dropdowns) |

---

### 2.6  Market Data & Quotes

| Method | Path | Parameters / Body | Description |
|--------|------|-------------------|-------------|
| GET | `/api/market/quote` | `?instruments=NSE:RELIANCE,NSE:TCS` | Full market quote |
| GET | `/api/market/ltp` | `?instruments=NSE:RELIANCE` | Last traded price |
| GET | `/api/market/ohlc` | `?instruments=NSE:RELIANCE` | OHLC data |
| GET | `/api/market/overview` | `?symbols=RELIANCE,TCS` | Market overview (indices + stocks) |
| GET | `/api/market/ticks` | — | Current tick cache from WebSocket |
| GET | `/api/market/resolve-token` | `?symbol` (required), `?exchange="NSE"` | Resolve tradingsymbol → instrument_token |

---

### 2.7  Historical Charts

| Method | Path | Parameters / Body | Description |
|--------|------|-------------------|-------------|
| GET | `/api/chart/historical` | `?token` (int, required), `?interval="day"`, `?days=365`, `?from_date`, `?to_date`, `?include_indicators="false"` | OHLCV by instrument token |
| GET | `/api/chart/historical/{symbol}` | `?exchange="NSE"`, `?interval="day"`, `?days=365`, `?from_date`, `?to_date`, `?include_indicators="false"` | OHLCV by symbol (auto-resolves token) |

---

### 2.8  Instrument Search

| Method | Path | Parameters / Body | Description |
|--------|------|-------------------|-------------|
| GET | `/api/instruments/search` | `?q=""`, `?exchange="NSE"` | Search equity instruments |
| GET | `/api/instruments/fno/search` | `?q`, `?exchange="NFO"`, `?underlying`, `?expiry`, `?instrument_type`, `?option_type`, `?strike_range="atm"`, `?spot_price=0` | Search F&O instruments |
| GET | `/api/instruments/fno/expiries` | `?exchange="NFO"`, `?underlying="NIFTY"`, `?instrument_type="FUT"` | F&O expiry dates |

---

### 2.9  Live Ticks & WebSocket

| Method | Path | Parameters / Body | Description |
|--------|------|-------------------|-------------|
| POST | `/api/live/start` | `{ tokens: [int], mode: "quote", names: {token: name} }` | Start live tick streaming |
| POST | `/api/live/start-default` | — | Start with default watchlist |
| POST | `/api/live/stop` | — | Stop tick streaming |
| GET | `/api/ticks` | — | Snapshot of current tick cache |
| **WS** | `/ws/live` | — | WebSocket — real-time tick updates |

---

### 2.10  Options & Open Interest

| Method | Path | Parameters / Body | Description |
|--------|------|-------------------|-------------|
| GET | `/api/options/chain` | `?underlying="NIFTY"`, `?expiry` | Option chain |
| GET | `/api/oi/summary` | `?underlying="NIFTY"` | OI summary |
| GET | `/api/oi/all` | — | All OI data |
| POST | `/api/oi/start` | `{ nifty_spot, sensex_spot, instruments }` | Start OI tracking |
| **WS** | `/ws/oi` | — | WebSocket — real-time OI updates |

---

### 2.11  OI Deep Analysis

| Method | Path | Parameters / Body | Description |
|--------|------|-------------------|-------------|
| POST | `/api/oi/futures/scan` | — | Run futures OI analysis |
| GET | `/api/oi/futures/report` | — | Last futures OI report |
| POST | `/api/oi/options/scan` | `{ underlying="NIFTY", expiry? }` | Near-ATM options OI analysis |
| GET | `/api/oi/options/report` | `?underlying="NIFTY"` | Last options OI report |
| POST | `/api/oi/options/compare` | `{ underlying="NIFTY" }` | Current vs next expiry OI comparison |
| GET | `/api/oi/pcr-history` | `?underlying="NIFTY"` | PCR history |
| GET | `/api/oi/straddle-history` | `?underlying="NIFTY"` | Straddle premium history |

---

### 2.12  OI Intraday Snapshots & Flow

| Method | Path | Parameters / Body | Description |
|--------|------|-------------------|-------------|
| POST | `/api/oi/snapshots/start` | — | Start 15-min OI snapshot scheduler |
| POST | `/api/oi/snapshots/stop` | — | Stop scheduler |
| GET | `/api/oi/snapshots/status` | — | Scheduler status |
| POST | `/api/oi/snapshots/trigger` | `{ force: false }` | Manual snapshot trigger |
| GET | `/api/oi/snapshots/today` | `?underlying="NIFTY"` | Today's snapshot timestamps |
| GET | `/api/oi/intraday/analysis` | `?underlying="NIFTY"`, `?date=""` | Full intraday OI flow analysis (PCR trend, IV skew, OI walls, smart money signals, direction + confidence) |
| GET | `/api/oi/intraday/direction` | `?underlying="NIFTY"` | Market direction from latest OI flow |

---

### 2.13  OI Strategy (Signal → Trade)

| Method | Path | Parameters / Body | Description |
|--------|------|-------------------|-------------|
| POST | `/api/oi/strategy/scan` | `{ underlying="NIFTY" }` | Scan OI for trading signals |
| POST | `/api/oi/strategy/scan-both` | — | Scan NIFTY + SENSEX |
| POST | `/api/oi/strategy/execute` | `{ signal_id }` | Execute an OI signal |
| POST | `/api/oi/strategy/close` | `{ position_id, exit_price=0.0 }` | Close OI strategy position |
| GET | `/api/oi/strategy/positions` | `?underlying=""` | Active positions |
| GET | `/api/oi/strategy/signals` | `?underlying=""`, `?limit=50` | Recent signals |
| GET | `/api/oi/strategy/summary` | — | Strategy performance summary |
| GET | `/api/oi/strategy/config` | — | Get current OI strategy config |
| POST | `/api/oi/strategy/config` | *(config dict)* | Update OI strategy config |

---

### 2.14  OI → FNO Bridge (Backtest-Gated Execution)

> Maps OI signals to structured F&O spread strategies, requires backtest qualification before execution.

**Signal Mapping:**
| OI Signal | FNO Strategy |
|-----------|-------------|
| Bearish | Call Credit Spread |
| Bullish | Put Credit Spread |
| Neutral / Range | Iron Condor |
| Bullish Breakout | Call Debit Spread |
| Bearish Breakout | Put Debit Spread |

**Default Backtest Thresholds:** `min_win_rate=45%`, `min_sharpe=0.5`, `max_drawdown=25%`, `min_trades=5`, `min_profit_factor=1.1`

| Method | Path | Parameters / Body | Description |
|--------|------|-------------------|-------------|
| POST | `/api/oi/bridge/scan` | `{ underlying="NIFTY" }` | Scan OI signals → map to FNO strategies |
| POST | `/api/oi/bridge/backtest` | `{ signal_id, force_refresh=false }` | Backtest the mapped strategy for a signal |
| POST | `/api/oi/bridge/execute` | `{ signal_id, skip_backtest=false }` | Execute signal via mapped FNO strategy |
| POST | `/api/oi/bridge/process-all` | `{ underlying="NIFTY" }` | End-to-end pipeline: scan → map → backtest → approve/reject |
| GET | `/api/oi/bridge/plans` | `?limit=50` | Recent execution plans |
| GET | `/api/oi/bridge/thresholds` | — | Current backtest thresholds |
| POST | `/api/oi/bridge/thresholds` | *(thresholds dict)* | Update thresholds |
| GET | `/api/oi/bridge/mapping-info` | — | Signal → strategy mapping reference |
| POST | `/api/oi/bridge/clear-cache` | — | Clear backtest cache |

**Typical agent workflow:**
```
1. POST /api/oi/bridge/process-all  {"underlying":"NIFTY"}
   → Returns plans with status APPROVED / REJECTED for each signal
2. Review approved plans
3. POST /api/oi/bridge/execute  {"signal_id": "<id>"}
```

---

### 2.15  Capital Allocation Rules

> Hard per-strategy, per-underlying, overnight, and delta exposure limits.
> Pre-trade check blocks orders that would breach allocation rules.

| Method | Path | Parameters / Body | Description |
|--------|------|-------------------|-------------|
| GET | `/api/capital/allocation` | — | Compute current capital allocation report (by strategy, by underlying, overnight, delta) |
| GET | `/api/capital/limits` | — | Get current allocation limits |
| POST | `/api/capital/limits` | `{ max_per_strategy?, max_per_underlying?, max_overnight_exposure?, max_net_delta?, max_single_position? }` | Update allocation limits |
| POST | `/api/capital/pre-trade-check` | `{ proposed_value, strategy, underlying, capital }` | Pre-trade capital check — returns allowed/blocked with breach details |

**Default Limits:** `max_per_strategy=25%`, `max_per_underlying=30%`, `max_overnight_exposure=50%`, `max_net_delta=20%`, `max_single_position=10%`

---

### 2.16  Execution Quality Scoring

> Auto-scores every trade execution on 4 dimensions. Hooked into the execution fill callback — every fill is scored automatically.

| Method | Path | Parameters / Body | Description |
|--------|------|-------------------|-------------|
| GET | `/api/execution-quality/scores` | `?limit=50` | Recent execution quality scores |
| GET | `/api/execution-quality/summary` | — | Summary statistics (avg composite, per-dimension averages, worst scores) |

**Scoring Dimensions (0-100):**
- **Slippage** (35% weight) — actual vs expected price
- **Timing** (25% weight) — fill position within bar range
- **Spread Efficiency** (20% weight) — actual vs bid-ask spread
- **Market Impact** (20% weight) — price movement after fill

---

### 2.17  Strategy Decay Monitor

> Rolling 30-day Sharpe / win rate / profit factor vs historical baseline.
> Auto-disables strategy if deviation > 2 standard deviations.

| Method | Path | Parameters / Body | Description |
|--------|------|-------------------|-------------|
| POST | `/api/strategy/decay/evaluate` | `{ strategy_name="" }` | Evaluate decay for one or all strategies |
| GET | `/api/strategy/decay/history` | `?limit=50` | Decay evaluation history |
| GET | `/api/strategy/decay/thresholds` | — | Current decay detection thresholds |
| POST | `/api/strategy/decay/thresholds` | `{ std_dev_trigger?, rolling_window_days?, min_history_days?, min_trades_in_window?, auto_disable? }` | Update thresholds |

**Default Thresholds:** `std_dev_trigger=2.0`, `rolling_window_days=30`, `min_history_days=60`, `min_trades_in_window=5`, `auto_disable=true`

---

### 2.18  Portfolio Greeks (F&O)

> Net delta, gamma, theta, vega across all F&O positions.
> Auto-flags high-risk exposures.

| Method | Path | Parameters / Body | Description |
|--------|------|-------------------|-------------|
| GET | `/api/portfolio/greeks` | — | Compute portfolio-level Greeks for all F&O positions |
| GET | `/api/portfolio/greeks/last` | — | Get last computed Greeks report |

**Risk Warnings Auto-Flagged:**
- Net delta > ±500
- Net gamma > 100
- Theta bleed > 5,000/day
- Vega exposure > 2,000

---

### 2.19  Regime-Aware Strategy Switching

> Combines OI flow + volatility regime + index trend → auto-recommends optimal F&O strategy type.

| Method | Path | Parameters / Body | Description |
|--------|------|-------------------|-------------|
| POST | `/api/regime/evaluate` | `{ underlying="NIFTY" }` | Evaluate current regime and recommend strategy |
| GET | `/api/regime/current` | `?underlying="NIFTY"` | Most recent regime assessment |
| GET | `/api/regime/history` | `?underlying=""`, `?limit=50` | Regime assessment history |

**Regime → Strategy Mapping:**
| Market Condition | Recommended Strategy |
|-----------------|---------------------|
| Range + Low Vol | Iron Condor |
| Range + High Vol + Bullish Bias | Put Credit Spread |
| Range + High Vol + Bearish Bias | Call Credit Spread |
| Trending Up + Bullish OI | Call Debit Spread |
| Trending Down + Bearish OI | Put Debit Spread |
| Compression + Low Vol | Long Strangle |
| Extreme Vol + Range | Short Straddle |

---

### 2.20  Trade Data Capture & Daily Analysis

> Automatically stores OHLCV candle data for every traded instrument from trade entry to exit.
> Provides full trade replay (MFE/MAE, bar-by-bar PnL, volatility) and daily aggregated analysis
> broken down by instrument, tradingsymbol, and strategy.

| Method | Path | Parameters / Body | Description |
|--------|------|-------------------|-------------|
| GET | `/api/trade-data/replay/{trade_id}` | — | Full trade replay: OHLCV bars + MFE/MAE + bar PnL + edge ratio |
| GET | `/api/trade-data/daily` | `?date=""` | Daily analysis — instrument-wise, strategy-wise, symbol-wise |
| GET | `/api/trade-data/daily/range` | `?from_date`, `?to_date`, `?days=7` | Daily analysis for a date range |
| GET | `/api/trade-data/trades` | `?date`, `?instrument`, `?limit=50` | Get tracked trades (filter by date or instrument) |
| GET | `/api/trade-data/open` | — | Currently open tracked trades |
| GET | `/api/trade-data/stats` | — | Trade data store statistics |
| POST | `/api/trade-data/fetch-candles` | — | Trigger OHLCV fetch for closed trades missing candle data |
| GET | `/api/trade-data/instrument/{symbol}` | `?limit=50` | All trades for a specific instrument |

**Trade Replay Analysis Fields:**
- **MFE** (Max Favorable Excursion) — best unrealized P&L during trade
- **MAE** (Max Adverse Excursion) — worst unrealized drawdown during trade
- **Edge Ratio** — MFE / MAE (higher = better risk/reward exploitation)
- **Bar PnL** — bar-by-bar unrealized P&L from entry to exit
- **Volatility** — standard deviation of bar returns during trade

**Daily Analysis Breakdown:**
- `by_instrument` — grouped by underlying (NIFTY, BANKNIFTY, RELIANCE, etc.)
- `by_tradingsymbol` — grouped by exact trading symbol
- `by_strategy` — grouped by strategy name
- `pnl_curve` — cumulative intraday P&L time series

---

### 2.21  Stock Analysis / Scanner

| Method | Path | Parameters / Body | Description |
|--------|------|-------------------|-------------|
| GET | `/api/analysis/status` | — | Scanner status |
| POST | `/api/analysis/scan` | `{ symbols?, universe="nifty50" }` | Run stock analysis scan |
| GET | `/api/analysis/results` | — | Cached scan results |
| GET | `/api/analysis/deep/{symbol}` | — | Deep analysis profile for one stock |
| GET | `/api/analysis/sectors` | — | Sector-level analysis |

---

### 2.22  Custom Strategy Builder (Equity)

| Method | Path | Parameters / Body | Description |
|--------|------|-------------------|-------------|
| GET | `/api/strategy-builder/indicators` | — | Available technical indicators |
| GET | `/api/strategy-builder/list` | — | List saved custom strategies |
| GET | `/api/strategy-builder/get/{name}` | — | Get one strategy |
| POST | `/api/strategy-builder/save` | *(strategy config dict)* | Save strategy |
| DELETE | `/api/strategy-builder/{name}` | — | Delete strategy |
| POST | `/api/strategy-builder/test` | `{ config, bars=500, capital=100000 }` | Save + quick paper-trade |

---

### 2.23  F&O Strategy Builder

| Method | Path | Parameters / Body | Description |
|--------|------|-------------------|-------------|
| GET | `/api/fno-builder/templates` | — | Pre-built F&O templates |
| GET | `/api/fno-builder/list` | — | List custom F&O strategies |
| GET | `/api/fno-builder/get/{name}` | — | Get one F&O strategy |
| POST | `/api/fno-builder/save` | *(F&O strategy config dict)* | Save F&O strategy |
| DELETE | `/api/fno-builder/{name}` | — | Delete F&O strategy |
| POST | `/api/fno-builder/payoff` | `{ config, spot_price=25000 }` | Compute payoff diagram |
| POST | `/api/fno-builder/test` | `{ name, legs?, underlying="NIFTY", bars=500, capital=500000 }` | Save + quick backtest |

---

### 2.24  Backtesting (Equity)

| Method | Path | Parameters / Body | Description |
|--------|------|-------------------|-------------|
| POST | `/api/backtest/run` | `{ tradingsymbol, exchange="NSE", strategy="ema_crossover", interval="day", days=365, from_date?, to_date?, capital=100000, position_sizing="fixed", risk_per_trade=0.02, capital_fraction=0.10, slippage_pct=0.05, use_indian_costs=true, is_intraday=true, strategy_params? }` | Backtest on live Kite data |
| POST | `/api/backtest/sample` | `{ strategy="ema_crossover", bars=500, capital=100000, position_sizing="fixed", slippage_pct=0.05, use_indian_costs=true, is_intraday=false, risk_per_trade=0.02, capital_fraction=0.10, interval="day", strategy_params?, allow_synthetic=false }` | Backtest on synthetic data |
| GET | `/api/backtest/results` | — | Cached backtest results |

---

### 2.25  F&O Backtesting

| Method | Path | Parameters / Body | Description |
|--------|------|-------------------|-------------|
| POST | `/api/fno-backtest/run` | `{ strategy="iron_condor", underlying="NIFTY", instrument_token=0, interval="day", days=365, from_date?, to_date?, capital=500000, max_positions=3, profit_target_pct=50, stop_loss_pct=100, entry_dte_min=15, entry_dte_max=45, delta_target=0.16, slippage_model="realistic", use_regime_filter=true }` | F&O backtest on real data |
| POST | `/api/fno-backtest/sample` | `{ strategy="iron_condor", underlying="NIFTY", bars=500, capital=500000, max_positions=3, profit_target_pct=50, stop_loss_pct=100, delta_target=0.16, allow_synthetic=false }` | F&O backtest on synthetic data |

---

### 2.26  F&O Strategies & Listing

| Method | Path | Parameters / Body | Description |
|--------|------|-------------------|-------------|
| GET | `/api/fno/strategies` | — | Available F&O option strategies |

---

### 2.27  Paper Trading (Equity)

| Method | Path | Parameters / Body | Description |
|--------|------|-------------------|-------------|
| POST | `/api/paper-trade/run` | `{ strategy, tradingsymbol, exchange="NSE", days=60, interval="5minute", capital=100000, commission_pct=0.03, slippage_pct=0.05, strategy_params? }` | Paper trade on historical data |
| POST | `/api/paper-trade/sample` | `{ strategy, tradingsymbol="SAMPLE", bars=500, capital=100000, interval="5minute", strategy_params?, allow_synthetic=false }` | Paper trade on synthetic data |
| GET | `/api/paper-trade/results` | — | Cached paper trade results |

---

### 2.28  F&O Paper Trading

| Method | Path | Parameters / Body | Description |
|--------|------|-------------------|-------------|
| POST | `/api/fno-paper-trade/run` | `{ strategy="iron_condor", underlying="NIFTY", instrument_token=0, interval="day", days=60, capital=500000, max_positions=3, profit_target_pct=50, stop_loss_pct=100, entry_dte_min=15, entry_dte_max=45, delta_target=0.16, slippage_model="realistic" }` | F&O paper trading on real underlying data |
| POST | `/api/fno-paper-trade/sample` | `{ strategy="iron_condor", underlying="NIFTY", bars=500, capital=500000, max_positions=3, profit_target_pct=50, stop_loss_pct=100, delta_target=0.16, allow_synthetic=false }` | F&O paper trade on synthetic data |

---

### 2.29  Strategy Health Reports

| Method | Path | Parameters / Body | Description |
|--------|------|-------------------|-------------|
| GET | `/api/strategy/health-report` | — | Last computed health report |
| POST | `/api/strategy/health-report` | `{ strategy_type="", strategy_name="" }` | Compute health report (equity / fno / intraday) |

---

### 2.30  Portfolio Reports

| Method | Path | Parameters / Body | Description |
|--------|------|-------------------|-------------|
| GET | `/api/reports/holdings` | — | Holdings report with P&L |
| GET | `/api/reports/positions` | — | Positions with realized/unrealized P&L |
| GET | `/api/reports/trades` | — | Today's trade execution report |
| GET | `/api/reports/orders` | — | Orders with status breakdown |
| GET | `/api/reports/pnl` | — | Combined P&L (positions + holdings) |
| GET | `/api/reports/margins` | — | Margin utilization |

---

### 2.31  Production Journal (Pro Journal)

| Method | Path | Parameters / Body | Description |
|--------|------|-------------------|-------------|
| GET | `/api/pro-journal/summary` | `?source`, `?days`, `?strategy`, `?instrument`, `?direction`, `?trade_type` | Filtered journal summary |
| GET | `/api/pro-journal/analytics` | `?strategy`, `?source`, `?days`, `?instrument`, `?direction`, `?trade_type` | Journal analytics |
| GET | `/api/pro-journal/fno-analytics` | `?strategy`, `?days`, `?instrument`, `?direction`, `?trade_type`, `?source` | F&O journal analytics |
| GET | `/api/pro-journal/entries` | `?strategy`, `?instrument`, `?trade_type`, `?source`, `?direction`, `?review_status`, `?is_closed`, `?from_date`, `?to_date`, `?days`, `?page=1`, `?page_size=100` | Paginated journal entries |
| GET | `/api/pro-journal/entry/{entry_id}` | — | Single journal entry |
| POST | `/api/pro-journal/entry` | *(journal entry dict)* | Create entry |
| PUT | `/api/pro-journal/entry/{entry_id}` | *(update fields)* | Update entry |
| GET | `/api/pro-journal/regime-matrix` | `?strategy`, `?instrument`, `?source`, `?direction`, `?trade_type`, `?days` | Regime performance matrix |
| GET | `/api/pro-journal/slippage-drift` | `?days=30`, `?strategy`, `?instrument`, `?source`, `?direction`, `?trade_type` | Slippage & execution drift |
| GET | `/api/pro-journal/daily-pnl` | `?days=30`, `?strategy`, `?instrument`, `?source`, `?direction`, `?trade_type` | Daily P&L history |
| GET | `/api/pro-journal/strategy-breakdown` | `?source`, `?days`, `?instrument`, `?direction`, `?trade_type`, `?strategy` | Strategy-level breakdown |
| GET | `/api/pro-journal/portfolio-health` | — | Portfolio health metrics |
| GET | `/api/pro-journal/portfolio-health/history` | `?days=30` | Health history |
| GET | `/api/pro-journal/equity-curve` | `?days=365` | Equity curve data |
| GET | `/api/pro-journal/system-events` | `?event_type`, `?limit=100` | System events log |
| GET | `/api/pro-journal/db-stats` | — | Journal database stats |
| GET | `/api/pro-journal/filters` | — | Available filter values |
| GET | `/api/pro-journal/export` | — | Export all entries |

---

### 2.32  AI Trade Execution Analysis

| Method | Path | Parameters / Body | Description |
|--------|------|-------------------|-------------|
| POST | `/api/trade-analysis/analyze` | `{ trade_ids?, strategy?, instrument?, from_date?, to_date?, limit=20, use_analysis_context=false, filter_by_analysis=false }` | AI-powered trade execution analysis (via OpenRouter) |
| POST | `/api/trade-analysis/analyze-order` | `{ order_id?, order_data?, underlying_candles? }` | Single order execution quality |
| GET | `/api/trade-analysis/status` | — | Check if AI analysis is configured |

---

### 2.33  F&O Data Store

| Method | Path | Parameters / Body | Description |
|--------|------|-------------------|-------------|
| GET | `/api/fno-data/coverage` | `?underlying` | Data coverage summary |
| GET | `/api/fno-data/stats` | — | Data store stats |
| GET | `/api/fno-data/sync-history` | `?underlying`, `?limit=50` | Sync history |
| GET | `/api/fno-data/expiries/{underlying}` | — | Available expiries in store |
| POST | `/api/fno-data/fetch` | `{ underlying, from_date, to_date, interval="minute", include_options=true, expiry? }` | Fetch & store F&O data from Kite |
| POST | `/api/fno-data/snapshot` | `{ underlying, expiry, spot_price=0 }` | Snapshot current option chain |

---

### 2.34  Stock Research (MCP via REST)

| Method | Path | Parameters / Body | Description |
|--------|------|-------------------|-------------|
| GET | `/api/research/mcp/status` | — | MCP server connection status |
| GET | `/api/research/mcp/tools` | — | List available MCP tools |
| POST | `/api/research/mcp/call` | `{ tool, args={} }` | Call any MCP tool by name |
| GET | `/api/research/news/{symbol}` | `?limit=15` | News for a symbol |
| GET | `/api/research/annual-reports/{symbol}` | `?limit=10` | Annual reports for a symbol |
| GET | `/api/research/best-stocks` | `?top_n=5`, `?enrich=true` | Top analytical picks |
| GET | `/api/research/full/{symbol}` | — | Full research package |
| GET | `/api/research/search-news` | `?topic` (required), `?limit=10` | Search market news by topic |

---

### 2.35  Cache Management

| Method | Path | Parameters / Body | Description |
|--------|------|-------------------|-------------|
| GET | `/api/cache/stats` | — | Cache hit/miss statistics |
| POST | `/api/cache/purge` | `{ days_to_keep=730 }` | Delete old candle data |
| GET | `/api/cache/candle-range` | `?token=0`, `?interval="day"` | Cached date range for token + interval |

---

## 3  MCP Tools (Model Context Protocol)

The MCP server (`StockResearch`) can be invoked directly via MCP protocol or through the REST bridge at `POST /api/research/mcp/call` with `{ tool: "<name>", args: {...} }`.

### 3.1  `fetch_stock_news`

```
symbol: str          — NSE trading symbol (e.g., RELIANCE, TCS)
limit: int = 15      — Max articles
```
Aggregates news from Google News RSS, MoneyControl, Economic Times, NSE announcements.
Returns JSON with articles (title, source, published, url).

### 3.2  `fetch_stock_annual_reports`

```
symbol: str          — NSE trading symbol
limit: int = 10      — Max report links
```
Searches BSE India, NSE India, Google, TrendLyne, Screener.in for annual report PDFs.
Returns JSON with report links, financial highlights (PE, market cap, ROE), pros/cons.

### 3.3  `get_best_stocks`

```
top_n: int = 5       — Number of top picks
enrich: bool = True  — Add news + reports to each pick
```
Ranks stocks using the algotrader's scanner (Minervini super-performance, trend scoring, VCP detection, institutional accumulation). Falls back to Screener.in data.
Returns ranked picks with composite score, triggers, news, reports.

### 3.4  `get_stock_research`

```
symbol: str          — NSE trading symbol
```
Comprehensive single-stock research combining news, annual reports, financial highlights, and scanner analysis into one report.

### 3.5  `analyze_intraday_oi_flow`

```
underlying: str = "NIFTY"    — Index name (NIFTY / SENSEX)
analysis_date: str = ""      — YYYY-MM-DD (default: today)
```
Analyzes 15-minute OI snapshots for intraday flow:
- Per-interval flow changes and signals
- PCR trend, straddle trend, IV skew evolution
- OI wall (support/resistance) movement
- Smart money detection
- Overall market direction with confidence score
- Trend reinforcement (consecutive confirmations)

### 3.6  `search_stock_news_by_topic`

```
topic: str           — Search topic (e.g., "pharma sector Q3 results")
limit: int = 10      — Max articles
```
Broad Google News search for Indian stock market topics.

---

## 4  MCP Resources

| URI | Description |
|-----|-------------|
| `stock://scanner/top-picks` | Top 10 ranked picks from the last scanner run (if available) |
| `stock://scanner/summary` | Summary of the last scan — total stocks scanned + symbols list |

---

## 5  WebSocket Channels

### 5.1  Live Ticks — `ws://localhost:5000/ws/live`

Real-time market tick updates from Kite Connect WebSocket.
- **Start streaming first** via `POST /api/live/start` with token list
- Receives JSON messages with tick data (LTP, volume, OHLC, depth)

### 5.2  OI Updates — `ws://localhost:5000/ws/oi`

Real-time open interest updates.
- **Start tracking first** via `POST /api/oi/start`
- Receives JSON messages with OI changes, PCR shifts

---

## 6  Agent Workflow Recipes

### 6.1  Full Research → Backtest → Execute Pipeline

```
Step 1: Research
  GET /api/research/best-stocks?top_n=10&enrich=true
  GET /api/research/full/{symbol}

Step 2: Scan & Analyze
  POST /api/analysis/scan  {"universe":"nifty50"}
  GET  /api/analysis/results
  GET  /api/analysis/deep/{symbol}

Step 3: Backtest Strategy
  POST /api/backtest/run  {
    "tradingsymbol": "RELIANCE",
    "strategy": "ema_crossover",
    "days": 365,
    "capital": 100000
  }

Step 4: Paper Trade (validate)
  POST /api/paper-trade/run  {
    "strategy": "ema_crossover",
    "tradingsymbol": "RELIANCE",
    "days": 60
  }

Step 5: Add to Live (if results meet threshold)
  POST /api/strategies/add  {
    "name": "ema_crossover",
    "params": {...},
    "require_tested": true
  }
```

### 6.2  OI-Driven F&O Execution Pipeline

```
Step 1: Start OI Monitoring
  POST /api/oi/snapshots/start

Step 2: Wait for snapshots to accumulate, then analyze
  GET /api/oi/intraday/direction?underlying=NIFTY

Step 3: Run full OI → FNO bridge pipeline
  POST /api/oi/bridge/process-all  {"underlying":"NIFTY"}
  → Returns plans with APPROVED / REJECTED status

Step 4: Review & execute approved plans
  GET  /api/oi/bridge/plans
  POST /api/oi/bridge/execute  {"signal_id":"<id>"}

Step 5: Monitor positions
  GET /api/oi/strategy/positions
  GET /api/reports/positions
```

### 6.3  Custom F&O Strategy Build → Test → Deploy

```
Step 1: Check available templates
  GET /api/fno-builder/templates

Step 2: Save custom strategy
  POST /api/fno-builder/save  {
    "name": "my_iron_condor",
    "legs": [...],
    "underlying": "NIFTY",
    ...
  }

Step 3: Compute payoff
  POST /api/fno-builder/payoff  {
    "config": {...},
    "spot_price": 25000
  }

Step 4: Backtest
  POST /api/fno-backtest/run  {
    "strategy": "my_iron_condor",
    "underlying": "NIFTY",
    "days": 365
  }

Step 5: Paper trade
  POST /api/fno-paper-trade/run  {
    "strategy": "my_iron_condor",
    "underlying": "NIFTY",
    "days": 60
  }
```

### 6.4  Risk Monitoring Loop

```
GET /api/risk              → Current risk summary
GET /api/preflight         → Full preflight checklist
GET /api/reports/pnl       → Combined P&L
GET /api/reports/margins   → Margin utilization

If risk limits breached:
  POST /api/risk/kill-switch  {"activate": true}
```

### 6.5  Journal Analytics & Improvement

```
GET /api/pro-journal/analytics
GET /api/pro-journal/regime-matrix
GET /api/pro-journal/slippage-drift?days=30
GET /api/pro-journal/strategy-breakdown
GET /api/pro-journal/portfolio-health
GET /api/pro-journal/equity-curve?days=365
```

### 6.6  Post-Trade Analysis Pipeline

```
Step 1: Check trade data store health
  GET /api/trade-data/stats

Step 2: Review today's trading performance
  GET /api/trade-data/daily
  → by_instrument, by_strategy, by_tradingsymbol breakdowns + pnl_curve

Step 3: Drill into multi-day trends
  GET /api/trade-data/daily/range?days=7

Step 4: Replay individual trades (MFE/MAE/edge ratio)
  GET /api/trade-data/replay/{trade_id}

Step 5: Backfill candles for any trades missing OHLCV data
  POST /api/trade-data/fetch-candles

Step 6: Inspect all trades for a specific instrument
  GET /api/trade-data/instrument/{symbol}?limit=50
```

---

## 7  Common Patterns

### Error Handling

All endpoints return JSON. On error:
```json
{ "error": "Error message describing what went wrong" }
```
HTTP status codes: `200` success, `400` bad request, `401` unauthorized, `404` not found, `500` internal error.

### Instrument Format

Instruments are specified as `EXCHANGE:SYMBOL` for quote endpoints:
- `NSE:RELIANCE` — Equity
- `NFO:NIFTY24JAN25000CE` — F&O option
- Comma-separated for multiple: `NSE:RELIANCE,NSE:TCS,NSE:INFY`

### Date Formats

- Dates: `YYYY-MM-DD` (e.g., `2025-01-15`)
- Intervals: `minute`, `3minute`, `5minute`, `10minute`, `15minute`, `30minute`, `60minute`, `day`

### Authentication

Include the `session_id` cookie with every request. Obtain it via OTP flow or Zerodha OAuth callback.

---

## 8  Endpoint Count Summary

| Category | Count |
|----------|-------|
| Auth & Session | 8 |
| Dashboard & Account | 9 |
| Preflight & Risk | 3 |
| Journal & Signals | 2 |
| Strategy Management | 6 |
| Market Data & Quotes | 6 |
| Historical Charts | 2 |
| Instrument Search | 3 |
| Live Ticks | 4 + 1 WS |
| Options & OI | 4 + 1 WS |
| OI Deep Analysis | 7 |
| OI Intraday Snapshots | 7 |
| OI Strategy | 9 |
| OI → FNO Bridge | 9 |
| Capital Allocation | 4 |
| Execution Quality | 2 |
| Strategy Decay | 4 |
| Portfolio Greeks | 2 |
| Regime Switching | 3 |
| Trade Data Capture | 8 |
| Analysis / Scanner | 5 |
| Strategy Builder (Equity) | 6 |
| F&O Builder | 7 |
| Backtesting (Equity) | 3 |
| F&O Backtesting | 2 |
| F&O Strategies | 1 |
| Paper Trading (Equity) | 3 |
| F&O Paper Trading | 2 |
| Strategy Health | 2 |
| Portfolio Reports | 6 |
| Pro Journal | 19 |
| AI Trade Analysis | 3 |
| F&O Data Store | 6 |
| Research / MCP | 8 |
| Cache Management | 3 |
| **Total** | **~178 HTTP + 2 WS** |
| MCP Tools | 6 |
| MCP Resources | 2 |
