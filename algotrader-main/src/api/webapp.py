from __future__ import annotations

import os
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.api.service import TradingService, TradingServiceManager
from src.auth.otp_auth import otp_auth
from src.auth.user_config import UserConfigService
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Kite Trading Agent", version="2.0")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

SESSION_COOKIE = "kta_session"

PUBLIC_PATHS = {"/login", "/auth/send-otp", "/auth/verify-otp", "/api/auth/callback", "/api/auth/postback"}


def get_user_service(request: Request) -> TradingService:
    token = request.cookies.get(SESSION_COOKIE, "")
    zerodha_id = otp_auth.get_session_user(token)
    if not zerodha_id:
        raise HTTPException(401, "Not authenticated")
    user_cfg = UserConfigService.get_instance().get_user(zerodha_id)
    if not user_cfg:
        raise HTTPException(403, "User not configured")
    mgr = TradingServiceManager.get_instance()
    return mgr.get_service(zerodha_id, api_key=user_cfg.api_key, api_secret=user_cfg.api_secret)


def get_session_zerodha_id(request: Request) -> str:
    token = request.cookies.get(SESSION_COOKIE, "")
    return otp_auth.get_session_user(token) or ""


def is_logged_in(request: Request) -> bool:
    token = request.cookies.get(SESSION_COOKIE, "")
    return otp_auth.validate_session(token)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path
    if path.startswith("/static/") or path in PUBLIC_PATHS:
        return await call_next(request)
    if path.startswith("/ws/"):
        token = request.cookies.get(SESSION_COOKIE, "")
        if not otp_auth.validate_session(token):
            return JSONResponse({"error": "Not authenticated"}, status_code=401)
        return await call_next(request)
    if not is_logged_in(request):
        if path.startswith("/api/"):
            return JSONResponse({"error": "Not authenticated"}, status_code=401)
        if path != "/login":
            return RedirectResponse("/login", status_code=302)
    return await call_next(request)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> HTMLResponse:
    if is_logged_in(request):
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/auth/send-otp")
async def send_otp(request: Request) -> dict[str, Any]:
    body = await request.json()
    email = body.get("email", "").strip().lower()
    zerodha_id = body.get("zerodha_id", "").strip().upper()
    if not email:
        return {"success": False, "message": "Email is required"}
    if not zerodha_id:
        return {"success": False, "message": "Zerodha ID is required"}

    user_cfg_svc = UserConfigService.get_instance()
    user = user_cfg_svc.get_user(zerodha_id)
    if not user:
        return {"success": False, "message": "Zerodha ID not registered"}
    if user.email and user.email.lower() != email:
        return {"success": False, "message": "Email does not match this Zerodha ID"}

    try:
        otp = otp_auth.generate_otp(zerodha_id)
    except Exception as e:
        return {"success": False, "message": str(e)}
    
    sent = otp_auth.send_otp_email(otp, zerodha_id)
    if sent:
        return {"success": True, "message": "OTP sent to your email", "mode": "email"}
    else:
        return {"success": True, "message": "OTP generated", "mode": "direct", "otp": otp}


@app.post("/auth/verify-otp")
async def verify_otp(request: Request) -> Any:
    body = await request.json()
    otp = body.get("otp", "").strip()
    zerodha_id = body.get("zerodha_id", "").strip().upper()
    if not otp:
        return {"success": False, "message": "OTP is required"}
    if not zerodha_id:
        return {"success": False, "message": "Zerodha ID is required"}

    session_token = otp_auth.verify_otp(otp, zerodha_id)
    if session_token:
        response = JSONResponse({"success": True, "message": "Login successful", "zerodha_id": zerodha_id})
        response.set_cookie(
            SESSION_COOKIE,
            session_token,
            max_age=86400 * 7,
            httponly=True,
            samesite="lax",
        )
        return response
    else:
        return {"success": False, "message": "Invalid or expired OTP"}


@app.post("/auth/logout")
async def logout(request: Request) -> Any:
    token = request.cookies.get(SESSION_COOKIE, "")
    otp_auth.logout(token)
    response = JSONResponse({"success": True})
    response.delete_cookie(SESSION_COOKIE)
    return response


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    if not is_logged_in(request):
        return RedirectResponse("/login", status_code=302)
    svc = get_user_service(request)
    zerodha_id = get_session_zerodha_id(request)
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "is_authenticated": svc.is_authenticated,
        "is_running": svc.is_running,
        "zerodha_id": zerodha_id,
    })


@app.get("/api/status")
async def api_status(request: Request, full: bool = False) -> dict[str, Any]:
    svc = get_user_service(request)
    result = {
        "authenticated": svc.is_authenticated,
        "running": svc.is_running,
        "zerodha_id": svc.zerodha_id,
        "strategies": svc.get_strategies_info(),
        "risk": svc.get_risk_summary(),
        "journal": svc.get_journal_summary(),
        "execution": svc.get_execution_summary(),
        "ticks": svc.get_live_ticks(),
        "signals": svc.get_recent_signals(20),
    }
    if svc.is_authenticated and full:
        try:
            margins = await svc.get_margins()
            result["margins"] = margins
        except Exception:
            result["margins"] = None
        try:
            positions = await svc.get_positions()
            result["positions"] = positions
        except Exception:
            result["positions"] = None
        try:
            holdings = await svc.get_holdings()
            result["holdings"] = holdings
        except Exception:
            result["holdings"] = None
        try:
            profile = await svc.get_profile()
            result["profile"] = profile
        except Exception:
            result["profile"] = None
    return result


@app.get("/api/login-url")
async def login_url(request: Request) -> dict[str, str]:
    svc = get_user_service(request)
    return {"url": svc.get_login_url()}


@app.get("/api/auth/callback")
async def auth_callback(request: Request):
    request_token = request.query_params.get("request_token", "")
    status = request.query_params.get("status", "")
    if status != "success" or not request_token:
        return RedirectResponse("/?auth_error=Login+failed+or+was+cancelled")
    try:
        svc = get_user_service(request)
        await svc.authenticate(request_token)
        return RedirectResponse("/?auth=success")
    except HTTPException:
        return RedirectResponse("/?auth_error=Please+login+first")
    except Exception as e:
        logger.error("auth_callback_failed", error=str(e))
        return RedirectResponse(f"/?auth_error={str(e)[:100]}")


@app.get("/api/auth/postback")
async def auth_postback(request: Request):
    return {"status": "ok"}


@app.post("/api/auth/postback")
async def auth_postback_post(request: Request):
    body = await request.json()
    logger.info("postback_received", data=body)
    return {"status": "ok"}


@app.post("/api/authenticate")
async def authenticate(request: Request) -> dict[str, Any]:
    body = await request.json()
    token = body.get("request_token", "")
    if not token:
        raise HTTPException(400, "request_token is required")
    svc = get_user_service(request)
    result = await svc.authenticate(token)
    return result


@app.get("/api/profile")
async def profile(request: Request) -> dict[str, Any]:
    svc = get_user_service(request)
    try:
        return await svc.get_profile()
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/margins")
async def margins(request: Request) -> dict[str, Any]:
    svc = get_user_service(request)
    try:
        return await svc.get_margins()
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/orders")
async def orders(request: Request) -> list[dict[str, Any]]:
    svc = get_user_service(request)
    try:
        return await svc.get_orders()
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/positions")
async def positions(request: Request) -> dict[str, Any]:
    svc = get_user_service(request)
    try:
        return await svc.get_positions()
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/holdings")
async def holdings(request: Request) -> list[dict[str, Any]]:
    svc = get_user_service(request)
    try:
        return await svc.get_holdings()
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/risk")
async def risk(request: Request) -> dict[str, Any]:
    return get_user_service(request).get_risk_summary()


@app.get("/api/preflight")
async def preflight_check(request: Request, mode: str = "live") -> dict[str, Any]:
    """Run full pre-flight checklist and return pass/fail report."""
    return await get_user_service(request).run_preflight_check(mode)


@app.get("/api/preflight/last")
async def preflight_last(request: Request) -> dict[str, Any]:
    """Get the most recent preflight report (cached)."""
    report = get_user_service(request).get_last_preflight_report()
    return report or {"passed": None, "message": "No preflight check has been run yet"}


@app.get("/api/journal")
async def journal(request: Request) -> dict[str, Any]:
    return get_user_service(request).get_journal_summary()


@app.get("/api/signals")
async def signals(request: Request, limit: int = 50) -> list[dict[str, Any]]:
    return get_user_service(request).get_recent_signals(limit)


@app.get("/api/strategies")
async def strategies(request: Request) -> list[dict[str, Any]]:
    return get_user_service(request).get_strategies_info()


@app.post("/api/strategies/add")
async def add_strategy(request: Request) -> dict[str, Any]:
    body = await request.json()
    name = body.get("name", "")
    params = body.get("params")
    return get_user_service(request).add_strategy(name, params)


@app.post("/api/strategies/remove")
async def remove_strategy(request: Request) -> dict[str, Any]:
    body = await request.json()
    return get_user_service(request).remove_strategy(body.get("name", ""))


@app.post("/api/strategies/toggle")
async def toggle_strategy(request: Request) -> dict[str, Any]:
    body = await request.json()
    return get_user_service(request).toggle_strategy(body.get("name", ""))


@app.get("/api/instruments/search")
async def search_instruments(request: Request, q: str = "", exchange: str = "NSE") -> list[dict[str, Any]]:
    svc = get_user_service(request)
    try:
        instruments = await svc.search_instruments(q, exchange)
        return instruments
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/live/start")
async def start_live(request: Request) -> dict[str, Any]:
    body = await request.json()
    tokens = body.get("tokens", [])
    mode = body.get("mode", "quote")
    if not tokens:
        raise HTTPException(400, "tokens list is required")
    svc = get_user_service(request)
    return await svc.start_live(tokens, mode)


@app.post("/api/live/start-default")
async def start_live_default(request: Request) -> dict[str, Any]:
    """Start live data with default watchlist symbols (NIFTY 50, NIFTY BANK, etc)."""
    svc = get_user_service(request)
    return await svc.start_live_with_defaults("full")


@app.post("/api/live/stop")
async def stop_live(request: Request) -> dict[str, Any]:
    return await get_user_service(request).stop_live()


@app.get("/api/ticks")
async def live_ticks(request: Request) -> dict[int, dict[str, Any]]:
    return get_user_service(request).get_live_ticks()


@app.get("/api/options/chain")
async def option_chain(request: Request, underlying: str = "NIFTY", expiry: Optional[str] = None) -> dict[str, Any]:
    return await get_user_service(request).get_option_chain(underlying, expiry)


@app.get("/api/oi/summary")
async def oi_summary(request: Request, underlying: str = "NIFTY") -> dict[str, Any]:
    return get_user_service(request).get_oi_summary(underlying)


@app.get("/api/oi/all")
async def oi_all(request: Request) -> dict[str, Any]:
    return get_user_service(request).get_oi_all()


@app.post("/api/oi/start")
async def start_oi_tracking(request: Request) -> dict[str, Any]:
    body = await request.json()
    nifty_spot = body.get("nifty_spot", 0.0)
    sensex_spot = body.get("sensex_spot", 0.0)
    instruments = body.get("instruments")
    svc = get_user_service(request)
    return await svc.start_oi_tracking(nifty_spot, sensex_spot, instruments)


@app.websocket("/ws/oi")
async def oi_websocket(websocket: WebSocket) -> None:
    token = websocket.cookies.get(SESSION_COOKIE, "")
    zerodha_id = otp_auth.get_session_user(token)
    if not zerodha_id:
        await websocket.close(code=4001)
        return
    user_cfg = UserConfigService.get_instance().get_user(zerodha_id)
    if not user_cfg:
        await websocket.close(code=4001)
        return

    await websocket.accept()
    mgr = TradingServiceManager.get_instance()
    svc = mgr.get_service(zerodha_id, api_key=user_cfg.api_key, api_secret=user_cfg.api_secret)
    svc.oi_tracker.register_oi_ws_client(websocket)
    try:
        data = svc.get_oi_all()
        await websocket.send_json(data)
        while True:
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        svc.oi_tracker.unregister_oi_ws_client(websocket)
    except Exception:
        svc.oi_tracker.unregister_oi_ws_client(websocket)


# ─── OI Deep Analysis ─────────────────────────────────────────

@app.post("/api/oi/futures/scan")
async def futures_oi_scan(request: Request) -> dict[str, Any]:
    """Run full futures OI analysis (index + stock futures)."""
    svc = get_user_service(request)
    return await svc.get_futures_oi_report()


@app.get("/api/oi/futures/report")
async def futures_oi_report(request: Request) -> dict[str, Any]:
    """Get last computed futures OI report."""
    return get_user_service(request).get_futures_oi_cached()


@app.post("/api/oi/options/scan")
async def options_oi_scan(request: Request) -> dict[str, Any]:
    """Run near-ATM options OI analysis for an index."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    underlying = body.get("underlying", "NIFTY") if body else "NIFTY"
    expiry = body.get("expiry") if body else None
    svc = get_user_service(request)
    return await svc.get_options_oi_report(underlying, expiry)


@app.get("/api/oi/options/report")
async def options_oi_report(request: Request, underlying: str = "NIFTY") -> dict[str, Any]:
    """Get last computed options OI report."""
    return get_user_service(request).get_options_oi_cached(underlying)


@app.post("/api/oi/options/compare")
async def options_oi_compare(request: Request) -> dict[str, Any]:
    """Compare current vs next expiry options OI."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    underlying = body.get("underlying", "NIFTY") if body else "NIFTY"
    svc = get_user_service(request)
    return await svc.get_options_oi_comparison(underlying)


@app.get("/api/oi/pcr-history")
async def pcr_history(request: Request, underlying: str = "NIFTY") -> list[dict[str, Any]]:
    return get_user_service(request).get_pcr_history(underlying)


@app.get("/api/oi/straddle-history")
async def straddle_history(request: Request, underlying: str = "NIFTY") -> list[dict[str, Any]]:
    return get_user_service(request).get_straddle_history(underlying)


# ── OI Strategy Routes ─────────────────────────────────────────

@app.post("/api/oi/strategy/scan")
async def oi_strategy_scan(request: Request) -> dict[str, Any]:
    """Scan NIFTY or SENSEX options OI and detect trading signals."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    underlying = body.get("underlying", "NIFTY") if body else "NIFTY"
    return await get_user_service(request).oi_strategy_scan(underlying)


@app.post("/api/oi/strategy/scan-both")
async def oi_strategy_scan_both(request: Request) -> dict[str, Any]:
    """Scan both NIFTY and SENSEX for OI signals."""
    return await get_user_service(request).oi_strategy_scan_both()


@app.post("/api/oi/strategy/execute")
async def oi_strategy_execute(request: Request) -> dict[str, Any]:
    """Execute a specific OI signal by creating a position."""
    try:
        body = await request.json()
    except Exception:
        return {"error": "Invalid request body"}
    signal_id = body.get("signal_id", "")
    if not signal_id:
        return {"error": "signal_id required"}
    return get_user_service(request).oi_strategy_execute_signal(signal_id)


@app.post("/api/oi/strategy/close")
async def oi_strategy_close(request: Request) -> dict[str, Any]:
    """Close an OI strategy position."""
    try:
        body = await request.json()
    except Exception:
        return {"error": "Invalid request body"}
    position_id = body.get("position_id", "")
    exit_price = body.get("exit_price", 0.0)
    if not position_id:
        return {"error": "position_id required"}
    return get_user_service(request).oi_strategy_close_position(position_id, exit_price)


@app.get("/api/oi/strategy/positions")
async def oi_strategy_positions(request: Request, underlying: str = "") -> dict[str, Any]:
    return get_user_service(request).oi_strategy_get_positions(underlying or None)


@app.get("/api/oi/strategy/signals")
async def oi_strategy_signals(request: Request, underlying: str = "", limit: int = 50) -> list[dict[str, Any]]:
    return get_user_service(request).oi_strategy_get_signals(underlying or None, limit)


@app.get("/api/oi/strategy/summary")
async def oi_strategy_summary(request: Request) -> dict[str, Any]:
    return get_user_service(request).oi_strategy_get_summary()


@app.get("/api/oi/strategy/config")
async def oi_strategy_config(request: Request) -> dict[str, Any]:
    return get_user_service(request).oi_strategy_get_config()


@app.post("/api/oi/strategy/config")
async def oi_strategy_update_config(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except Exception:
        return {"error": "Invalid request body"}
    return get_user_service(request).oi_strategy_update_config(body)


@app.get("/api/analysis/status")
async def analysis_status(request: Request) -> dict[str, Any]:
    return get_user_service(request).get_scanner_status()


@app.post("/api/analysis/scan")
async def run_analysis_scan(request: Request) -> dict[str, Any]:
    svc = get_user_service(request)
    try:
        body = await request.json()
    except Exception:
        body = {}
    symbols = body.get("symbols") if body else None
    universe = (body.get("universe", "nifty50") if body else "nifty50")
    try:
        return await svc.run_analysis_scan(symbols, universe=universe)
    except Exception as e:
        logger.error("analysis_scan_error", error=str(e))
        raise HTTPException(500, str(e))


@app.get("/api/analysis/results")
async def analysis_results(request: Request) -> dict[str, Any]:
    return get_user_service(request).get_analysis_results()


@app.get("/api/analysis/deep/{symbol}")
async def analysis_deep_profile(request: Request, symbol: str) -> dict[str, Any]:
    """Get deep analysis profile for a single stock (on-demand if not cached)."""
    svc = get_user_service(request)
    sym = symbol.upper().strip()
    # Try cached first
    result = svc.get_deep_profile(sym)
    if not result.get("error"):
        return result
    # Fall back to on-demand analysis
    return await svc.analyze_stock_on_demand(sym)


@app.get("/api/analysis/sectors")
async def analysis_sectors(request: Request) -> dict[str, Any]:
    """Get sector-level analysis from cached scan data."""
    return get_user_service(request).get_sector_analysis()


# ─── Paper Trading ─────────────────────────────────────────────

@app.post("/api/paper-trade/run")
async def run_paper_trade(request: Request) -> dict[str, Any]:
    """Run paper trade on historical data with a strategy.

    Symbol↔token binding: If tradingsymbol is provided, the server
    resolves the instrument_token from Zerodha's instrument list.
    Any client-supplied token is IGNORED to prevent mismatches.
    """
    body = await request.json()
    strategy_name = body.get("strategy", "")
    if not strategy_name:
        raise HTTPException(400, "strategy is required")

    tradingsymbol = (body.get("tradingsymbol") or "").strip().upper()
    exchange = body.get("exchange", "NSE")
    if not tradingsymbol:
        raise HTTPException(400, "tradingsymbol is required")

    svc = get_user_service(request)

    # ── Authoritative server-side token resolution ──────────────
    instrument_token = await svc.resolve_instrument_token(tradingsymbol, exchange)
    if not instrument_token:
        raise HTTPException(
            404,
            f"Instrument not found: {exchange}:{tradingsymbol}. "
            "Check symbol spelling and exchange.",
        )

    try:
        result = await svc.run_paper_trade(
            strategy_name=strategy_name,
            instrument_token=instrument_token,
            tradingsymbol=tradingsymbol,
            exchange=exchange,
            days=body.get("days", 60),
            interval=body.get("interval", "5minute"),
            capital=body.get("capital", 100000),
            commission_pct=body.get("commission_pct", 0.03),
            slippage_pct=body.get("slippage_pct", 0.05),
            strategy_params=body.get("strategy_params"),
        )
        return result
    except Exception as e:
        logger.error("paper_trade_error", error=str(e))
        raise HTTPException(500, str(e))


@app.post("/api/paper-trade/sample")
async def run_paper_trade_sample(request: Request) -> dict[str, Any]:
    """Run paper trade with synthetic sample data (no auth to broker needed)."""
    body = await request.json()
    strategy_name = body.get("strategy", "")
    if not strategy_name:
        raise HTTPException(400, "strategy is required")
    svc = get_user_service(request)
    try:
        return await svc.run_paper_trade_with_sample_data(
            strategy_name=strategy_name,
            tradingsymbol=body.get("tradingsymbol", "SAMPLE"),
            bars=body.get("bars", 500),
            capital=body.get("capital", 100000),
            strategy_params=body.get("strategy_params"),
        )
    except Exception as e:
        logger.error("paper_trade_sample_error", error=str(e))
        raise HTTPException(500, str(e))


@app.get("/api/paper-trade/results")
async def paper_trade_results(request: Request) -> dict[str, Any]:
    return get_user_service(request).get_paper_results()


# ─── Custom Strategy Builder ──────────────────────────────────

@app.get("/api/strategy-builder/indicators")
async def builder_indicators(request: Request) -> list[dict[str, Any]]:
    return get_user_service(request).get_builder_indicators()


@app.get("/api/strategy-builder/list")
async def list_custom_strategies(request: Request) -> list[dict[str, Any]]:
    return get_user_service(request).list_custom_strategies()


@app.get("/api/strategy-builder/get/{name}")
async def get_custom_strategy(request: Request, name: str) -> dict[str, Any]:
    result = get_user_service(request).get_custom_strategy(name)
    if result is None:
        raise HTTPException(404, f"Strategy '{name}' not found")
    return result


@app.post("/api/strategy-builder/save")
async def save_custom_strategy(request: Request) -> dict[str, Any]:
    body = await request.json()
    return get_user_service(request).save_custom_strategy(body)


@app.delete("/api/strategy-builder/{name}")
async def delete_custom_strategy(request: Request, name: str) -> dict[str, Any]:
    return get_user_service(request).delete_custom_strategy(name)


@app.post("/api/strategy-builder/test")
async def test_custom_strategy(request: Request) -> dict[str, Any]:
    """Save a custom strategy and immediately paper-trade it with sample data."""
    body = await request.json()
    svc = get_user_service(request)
    config = body.get("config", body)
    name = config.get("name", "")
    if not name:
        raise HTTPException(400, "Strategy name is required")

    # Save it first
    save_result = svc.save_custom_strategy(config)
    if "error" in save_result:
        return save_result

    # Paper trade with sample data
    try:
        result = await svc.run_paper_trade_with_sample_data(
            strategy_name=name,
            bars=body.get("bars", 500),
            capital=body.get("capital", 100000),
        )
        result["strategy_saved"] = True
        return result
    except Exception as e:
        logger.error("strategy_test_error", error=str(e))
        raise HTTPException(500, str(e))


# ────────────────────────────────────────────────────────────────
# F&O Custom Strategy Builder APIs
# ────────────────────────────────────────────────────────────────

@app.get("/api/fno-builder/templates")
async def get_fno_builder_templates(request: Request) -> dict[str, Any]:
    """Get pre-built F&O strategy templates."""
    svc = get_user_service(request)
    return {"templates": svc.get_fno_builder_templates()}


@app.get("/api/fno-builder/list")
async def list_fno_custom_strategies(request: Request) -> list[dict[str, Any]]:
    """List saved custom F&O strategies."""
    svc = get_user_service(request)
    return svc.list_fno_custom_strategies()


@app.get("/api/fno-builder/get/{name}")
async def get_fno_custom_strategy(request: Request, name: str) -> dict[str, Any]:
    """Get a specific custom F&O strategy."""
    svc = get_user_service(request)
    result = svc.get_fno_custom_strategy(name)
    if not result:
        raise HTTPException(404, f"F&O strategy '{name}' not found")
    return result


@app.post("/api/fno-builder/save")
async def save_fno_custom_strategy(request: Request) -> dict[str, Any]:
    """Save a custom F&O strategy."""
    body = await request.json()
    svc = get_user_service(request)
    return svc.save_fno_custom_strategy(body)


@app.delete("/api/fno-builder/{name}")
async def delete_fno_custom_strategy(request: Request, name: str) -> dict[str, Any]:
    """Delete a custom F&O strategy."""
    svc = get_user_service(request)
    return svc.delete_fno_custom_strategy(name)


@app.post("/api/fno-builder/payoff")
async def compute_fno_payoff(request: Request) -> dict[str, Any]:
    """Compute payoff diagram for a custom F&O strategy config."""
    body = await request.json()
    svc = get_user_service(request)
    spot_price = float(body.get("spot_price", 25000))
    return svc.compute_fno_payoff(body.get("config", body), spot_price)


@app.post("/api/fno-builder/test")
async def test_fno_custom_strategy(request: Request) -> dict[str, Any]:
    """Save and run a quick backtest on a custom F&O strategy."""
    body = await request.json()
    svc = get_user_service(request)
    name = body.get("name", "custom_fno")

    # If legs are present, this is a full save+test from the builder
    if body.get("legs"):
        save_result = svc.save_fno_custom_strategy(body)
        if "error" in save_result:
            return save_result
    else:
        # Just a test of an already-saved strategy (from backtest/paper trade dropdowns)
        existing = svc.get_fno_custom_strategy(name)
        if not existing:
            return {"error": f"Custom F&O strategy '{name}' not found. Build it in the Strategy Builder first."}

    try:
        result = await svc.run_fno_backtest_sample(
            strategy_name=name,
            underlying=body.get("underlying", "NIFTY"),
            bars=int(body.get("bars", 500)),
            initial_capital=float(body.get("capital", 500000)),
        )
        result["strategy_saved"] = True
        return result
    except Exception as e:
        logger.error("fno_strategy_test_error", error=str(e))
        return {"error": str(e), "strategy_saved": True}


@app.get("/api/analysis-strategies")
async def get_analysis_strategies(request: Request) -> dict[str, Any]:
    """Get list of individual analysis strategies for dropdowns."""
    svc = get_user_service(request)
    return {"strategies": svc.get_analysis_strategy_list()}


# ────────────────────────────────────────────────────────────────
# Strategy Health Report APIs
# ────────────────────────────────────────────────────────────────

@app.get("/api/strategy/health-report")
async def get_health_report(request: Request) -> dict[str, Any]:
    """Get the last computed strategy health report."""
    svc = get_user_service(request)
    return svc.get_last_health_report()


@app.post("/api/strategy/health-report")
async def compute_health_report_api(request: Request) -> dict[str, Any]:
    """Compute health report for specific backtest/paper-trade results.
    
    Body (optional):
        strategy_type: "equity" / "fno" / "intraday"
        strategy_name: override strategy name
    """
    svc = get_user_service(request)
    try:
        body = await request.json()
    except Exception:
        body = {}
    return svc.compute_strategy_health(
        strategy_type=body.get("strategy_type", ""),
        strategy_name=body.get("strategy_name", ""),
    )


# ────────────────────────────────────────────────────────────────
# Historical Chart Data APIs
# ────────────────────────────────────────────────────────────────

@app.get("/api/chart/historical")
async def get_historical_chart(request: Request) -> dict[str, Any]:
    """Fetch historical OHLCV data for charting."""
    svc = get_user_service(request)
    params = request.query_params
    token = int(params.get("token", "0"))
    interval = params.get("interval", "day")
    days = int(params.get("days", "365"))
    from_date = params.get("from_date")
    to_date = params.get("to_date")
    indicators = params.get("include_indicators", params.get("indicators", "false")).lower() == "true"
    if token == 0:
        raise HTTPException(400, "instrument token required")
    return await svc.get_historical_chart(token, interval, days, from_date, to_date, indicators)


@app.get("/api/chart/historical/{symbol}")
async def get_historical_chart_by_symbol(symbol: str, request: Request) -> dict[str, Any]:
    """Fetch chart data by symbol name (auto-resolves token)."""
    svc = get_user_service(request)
    params = request.query_params
    exchange = params.get("exchange", "NSE")
    interval = params.get("interval", "day")
    days = int(params.get("days", "365"))
    from_date = params.get("from_date")
    to_date = params.get("to_date")
    indicators = params.get("include_indicators", params.get("indicators", "false")).lower() == "true"
    token = await svc.resolve_instrument_token(symbol, exchange)
    if not token:
        raise HTTPException(404, f"Instrument not found: {exchange}:{symbol}")
    return await svc.get_historical_chart(token, interval, days, from_date, to_date, indicators)


# ────────────────────────────────────────────────────────────────
# Live Market Quotes & Data APIs
# ────────────────────────────────────────────────────────────────

@app.get("/api/market/quote")
async def get_market_quote(request: Request) -> dict[str, Any]:
    """Full market quote. ?instruments=NSE:RELIANCE,NSE:TCS"""
    svc = get_user_service(request)
    raw = request.query_params.get("instruments", "")
    instruments = [i.strip() for i in raw.split(",") if i.strip()]
    if not instruments:
        raise HTTPException(400, "instruments parameter required (comma-separated)")
    return await svc.get_market_quote(instruments)


@app.get("/api/market/ltp")
async def get_market_ltp(request: Request) -> dict[str, Any]:
    """Last traded price. ?instruments=NSE:RELIANCE,NSE:TCS"""
    svc = get_user_service(request)
    raw = request.query_params.get("instruments", "")
    instruments = [i.strip() for i in raw.split(",") if i.strip()]
    if not instruments:
        raise HTTPException(400, "instruments parameter required")
    return await svc.get_market_ltp(instruments)


@app.get("/api/market/ohlc")
async def get_market_ohlc(request: Request) -> dict[str, Any]:
    """OHLC data. ?instruments=NSE:RELIANCE,NSE:TCS"""
    svc = get_user_service(request)
    raw = request.query_params.get("instruments", "")
    instruments = [i.strip() for i in raw.split(",") if i.strip()]
    if not instruments:
        raise HTTPException(400, "instruments parameter required")
    return await svc.get_market_ohlc(instruments)


@app.get("/api/market/overview")
async def get_market_overview(request: Request) -> dict[str, Any]:
    """Market overview for key indices/stocks."""
    svc = get_user_service(request)
    raw = request.query_params.get("symbols", "")
    symbols = [s.strip() for s in raw.split(",") if s.strip()] or None
    return await svc.get_market_overview(symbols)


@app.get("/api/market/ticks")
async def get_live_ticks_snapshot(request: Request) -> dict[str, Any]:
    """Get current tick cache from live WebSocket."""
    svc = get_user_service(request)
    return {"ticks": svc.get_live_ticks()}


@app.get("/api/market/resolve-token")
async def resolve_instrument_token(request: Request) -> dict[str, Any]:
    """Resolve a tradingsymbol to its instrument_token.

    Usage: /api/market/resolve-token?symbol=RELIANCE&exchange=NSE
    Returns: {"symbol": "RELIANCE", "exchange": "NSE", "instrument_token": 738561}
    """
    symbol = (request.query_params.get("symbol") or "").strip().upper()
    exchange = request.query_params.get("exchange", "NSE").strip().upper()
    if not symbol:
        raise HTTPException(400, "symbol query param is required")
    svc = get_user_service(request)
    token = await svc.resolve_instrument_token(symbol, exchange)
    if not token:
        raise HTTPException(404, f"Instrument not found: {exchange}:{symbol}")
    return {"symbol": symbol, "exchange": exchange, "instrument_token": token}


# ────────────────────────────────────────────────────────────────
# Portfolio Reports & Investing APIs
# ────────────────────────────────────────────────────────────────

@app.get("/api/reports/holdings")
async def get_holdings_report(request: Request) -> dict[str, Any]:
    """Comprehensive holdings report with P&L analysis."""
    svc = get_user_service(request)
    try:
        return await svc.get_holdings_report()
    except Exception as e:
        logger.error("holdings_report_route_error", error=str(e))
        return {"holdings": [], "summary": {}, "error": str(e)}


@app.get("/api/reports/positions")
async def get_positions_report(request: Request) -> dict[str, Any]:
    """Positions report with realized/unrealized P&L."""
    svc = get_user_service(request)
    try:
        return await svc.get_positions_report()
    except Exception as e:
        logger.error("positions_report_route_error", error=str(e))
        return {"positions": [], "summary": {}, "error": str(e)}


@app.get("/api/reports/trades")
async def get_trades_report(request: Request) -> dict[str, Any]:
    """Today's trade execution report."""
    svc = get_user_service(request)
    try:
        return await svc.get_trades_report()
    except Exception as e:
        logger.error("trades_report_route_error", error=str(e))
        return {"trades": [], "summary": {}, "error": str(e)}


@app.get("/api/reports/orders")
async def get_orders_report(request: Request) -> dict[str, Any]:
    """Orders report with status breakdown."""
    svc = get_user_service(request)
    try:
        return await svc.get_orders_report()
    except Exception as e:
        logger.error("orders_report_route_error", error=str(e))
        return {"orders": [], "summary": {}, "error": str(e)}


@app.get("/api/reports/pnl")
async def get_pnl_report(request: Request) -> dict[str, Any]:
    """Combined P&L report (positions + holdings)."""
    svc = get_user_service(request)
    try:
        return await svc.get_pnl_report()
    except Exception as e:
        logger.error("pnl_report_route_error", error=str(e))
        return {"error": str(e)}


@app.get("/api/reports/margins")
async def get_margins_report(request: Request) -> dict[str, Any]:
    """Detailed margin utilization report."""
    svc = get_user_service(request)
    try:
        return await svc.get_margins_report()
    except Exception as e:
        logger.error("margins_report_route_error", error=str(e))
        return {"margins": {}, "error": str(e)}


# ────────────────────────────────────────────────────────────────
# Advanced Backtesting APIs
# ────────────────────────────────────────────────────────────────

@app.post("/api/backtest/run")
async def run_backtest(request: Request) -> dict[str, Any]:
    """Run advanced backtest on live Zerodha historical data.

    Symbol↔token binding: tradingsymbol is resolved server-side
    to the correct instrument_token. Client-sent token is IGNORED.
    """
    body = await request.json()
    svc = get_user_service(request)

    tradingsymbol = (body.get("tradingsymbol") or "").strip().upper()
    exchange = body.get("exchange", "NSE")
    if not tradingsymbol:
        raise HTTPException(400, "tradingsymbol is required for live backtest")

    # ── Authoritative server-side token resolution ──────────────
    instrument_token = await svc.resolve_instrument_token(tradingsymbol, exchange)
    if not instrument_token:
        raise HTTPException(
            404,
            f"Instrument not found: {exchange}:{tradingsymbol}. "
            "Check symbol spelling and exchange.",
        )

    try:
        result = await svc.run_backtest(
            strategy_name=body.get("strategy", "ema_crossover"),
            instrument_token=instrument_token,
            tradingsymbol=tradingsymbol,
            interval=body.get("interval", "day"),
            days=body.get("days", 365),
            from_date=body.get("from_date"),
            to_date=body.get("to_date"),
            initial_capital=body.get("capital", 100000),
            position_sizing=body.get("position_sizing", "fixed"),
            risk_per_trade=body.get("risk_per_trade", 0.02),
            capital_fraction=body.get("capital_fraction", 0.10),
            slippage_pct=body.get("slippage_pct", 0.05),
            use_indian_costs=body.get("use_indian_costs", True),
            is_intraday=body.get("is_intraday", True),
            strategy_params=body.get("strategy_params"),
        )
        return result
    except Exception as e:
        logger.error("backtest_error", error=str(e))
        raise HTTPException(500, str(e))


@app.post("/api/backtest/sample")
async def run_backtest_sample(request: Request) -> dict[str, Any]:
    """Run backtest on synthetic sample data."""
    body = await request.json()
    svc = get_user_service(request)

    try:
        result = await svc.run_backtest_sample(
            strategy_name=body.get("strategy", "ema_crossover"),
            bars=body.get("bars", 500),
            initial_capital=body.get("capital", 100000),
            position_sizing=body.get("position_sizing", "fixed"),
            slippage_pct=body.get("slippage_pct", 0.05),
            use_indian_costs=body.get("use_indian_costs", True),
            is_intraday=body.get("is_intraday", False),
            risk_per_trade=body.get("risk_per_trade", 0.02),
            capital_fraction=body.get("capital_fraction", 0.10),
            strategy_params=body.get("strategy_params"),
        )
        return result
    except Exception as e:
        logger.error("backtest_sample_error", error=str(e))
        raise HTTPException(500, str(e))


@app.get("/api/backtest/results")
async def get_backtest_results(request: Request) -> dict[str, Any]:
    """Get cached backtest results."""
    svc = get_user_service(request)
    return svc.get_backtest_results()


# ────────────────────────────────────────────────────────
# F&O Derivatives Backtest & Paper Trade
# ────────────────────────────────────────────────────────

@app.get("/api/fno/strategies")
async def get_fno_strategies(request: Request) -> dict[str, Any]:
    """Get available F&O option strategies."""
    svc = get_user_service(request)
    return {"strategies": svc.get_fno_strategies()}


@app.post("/api/fno-backtest/run")
async def run_fno_backtest(request: Request) -> dict[str, Any]:
    """Run F&O derivatives backtest on real underlying data."""
    body = await request.json()
    svc = get_user_service(request)
    try:
        return await svc.run_fno_backtest(
            strategy_name=body.get("strategy", "iron_condor"),
            underlying=body.get("underlying", "NIFTY"),
            instrument_token=int(body.get("instrument_token", 0)),
            interval=body.get("interval", "day"),
            days=int(body.get("days", 365)),
            from_date=body.get("from_date"),
            to_date=body.get("to_date"),
            initial_capital=float(body.get("capital", 500000)),
            max_positions=int(body.get("max_positions", 3)),
            profit_target_pct=float(body.get("profit_target_pct", 50)),
            stop_loss_pct=float(body.get("stop_loss_pct", 100)),
            entry_dte_min=int(body.get("entry_dte_min", 15)),
            entry_dte_max=int(body.get("entry_dte_max", 45)),
            delta_target=float(body.get("delta_target", 0.16)),
            slippage_model=body.get("slippage_model", "realistic"),
            use_regime_filter=body.get("use_regime_filter", True),
        )
    except Exception as e:
        logger.error("fno_backtest_error", error=str(e), strategy=body.get("strategy"))
        return {"error": f"Backtest failed: {str(e)}"}


@app.post("/api/fno-backtest/sample")
async def run_fno_backtest_sample(request: Request) -> dict[str, Any]:
    """Run F&O backtest on synthetic data (no auth required for data)."""
    body = await request.json()
    svc = get_user_service(request)
    try:
        return await svc.run_fno_backtest_sample(
            strategy_name=body.get("strategy", "iron_condor"),
            underlying=body.get("underlying", "NIFTY"),
            bars=int(body.get("bars", 500)),
            initial_capital=float(body.get("capital", 500000)),
            max_positions=int(body.get("max_positions", 3)),
            profit_target_pct=float(body.get("profit_target_pct", 50)),
            stop_loss_pct=float(body.get("stop_loss_pct", 100)),
            delta_target=float(body.get("delta_target", 0.16)),
        )
    except Exception as e:
        logger.error("fno_backtest_sample_error", error=str(e), strategy=body.get("strategy"))
        return {"error": f"Sample backtest failed: {str(e)}"}


@app.post("/api/fno-paper-trade/run")
async def run_fno_paper_trade(request: Request) -> dict[str, Any]:
    """Run F&O paper trading simulation on real underlying data."""
    body = await request.json()
    svc = get_user_service(request)
    try:
        return await svc.run_fno_paper_trade(
            strategy_name=body.get("strategy", "iron_condor"),
            underlying=body.get("underlying", "NIFTY"),
            instrument_token=int(body.get("instrument_token", 0)),
            interval=body.get("interval", "day"),
            days=int(body.get("days", 60)),
            initial_capital=float(body.get("capital", 500000)),
            max_positions=int(body.get("max_positions", 3)),
            profit_target_pct=float(body.get("profit_target_pct", 50)),
            stop_loss_pct=float(body.get("stop_loss_pct", 100)),
            entry_dte_min=int(body.get("entry_dte_min", 15)),
            entry_dte_max=int(body.get("entry_dte_max", 45)),
            delta_target=float(body.get("delta_target", 0.16)),
            slippage_model=body.get("slippage_model", "realistic"),
        )
    except Exception as e:
        logger.error("fno_paper_trade_error", error=str(e), strategy=body.get("strategy"))
        return {"error": f"Paper trade failed: {str(e)}"}


@app.post("/api/fno-paper-trade/sample")
async def run_fno_paper_trade_sample(request: Request) -> dict[str, Any]:
    """Run F&O paper trade on synthetic data (no auth required for data)."""
    body = await request.json()
    svc = get_user_service(request)
    try:
        return await svc.run_fno_paper_trade_sample(
            strategy_name=body.get("strategy", "iron_condor"),
            underlying=body.get("underlying", "NIFTY"),
            bars=int(body.get("bars", 500)),
            initial_capital=float(body.get("capital", 500000)),
            max_positions=int(body.get("max_positions", 3)),
            profit_target_pct=float(body.get("profit_target_pct", 50)),
            stop_loss_pct=float(body.get("stop_loss_pct", 100)),
            delta_target=float(body.get("delta_target", 0.16)),
        )
    except Exception as e:
        logger.error("fno_paper_trade_sample_error", error=str(e), strategy=body.get("strategy"))
        return {"error": f"Sample paper trade failed: {str(e)}"}


@app.post("/api/risk/kill-switch")
async def toggle_kill_switch(request: Request) -> dict[str, Any]:
    body = await request.json()
    svc = get_user_service(request)
    if body.get("activate"):
        svc._risk.activate_kill_switch("Manual activation via dashboard")
    else:
        svc._risk.deactivate_kill_switch()
    return svc.get_risk_summary()


# ────────────────────────────────────────────────────────────
# 3-Layer Production Journal API
# ────────────────────────────────────────────────────────────

@app.get("/api/pro-journal/summary")
async def get_pro_journal_summary(request: Request) -> dict[str, Any]:
    try:
        svc = get_user_service(request)
        p = request.query_params
        return svc.get_pro_journal_summary(
            source=p.get("source", ""), days=int(p.get("days", 0)),
            strategy=p.get("strategy", ""), instrument=p.get("instrument", ""),
            direction=p.get("direction", ""), trade_type=p.get("trade_type", ""))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("pro_journal_summary_error", error=str(e))
        return {"error": str(e), "total_trades": 0}


@app.get("/api/pro-journal/analytics")
async def get_journal_analytics(request: Request) -> dict[str, Any]:
    try:
        svc = get_user_service(request)
        p = request.query_params
        return svc.get_journal_analytics(
            strategy=p.get("strategy", ""), source=p.get("source", ""),
            days=int(p.get("days", 0)), instrument=p.get("instrument", ""),
            direction=p.get("direction", ""), trade_type=p.get("trade_type", ""))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("pro_journal_analytics_error", error=str(e))
        return {"error": str(e), "total_trades": 0}


@app.get("/api/pro-journal/fno-analytics")
async def get_fno_journal_analytics(request: Request) -> dict[str, Any]:
    try:
        svc = get_user_service(request)
        p = request.query_params
        return svc.get_fno_journal_analytics(
            strategy=p.get("strategy", ""), days=int(p.get("days", 0)),
            instrument=p.get("instrument", ""), direction=p.get("direction", ""),
            trade_type=p.get("trade_type", ""), source=p.get("source", ""))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("pro_journal_fno_analytics_error", error=str(e))
        return {"error": str(e)}


@app.get("/api/pro-journal/entries")
async def get_journal_entries(request: Request) -> dict[str, Any]:
    try:
        svc = get_user_service(request)
        params = request.query_params
        is_closed = None
        if "is_closed" in params:
            is_closed = params["is_closed"].lower() == "true"
        from_date = params.get("from_date", "")
        to_date = params.get("to_date", "")
        # Convert 'days' filter to from_date if not already specified
        days = int(params.get("days", 0))
        if days and not from_date:
            from datetime import datetime, timedelta
            from_date = (datetime.now() - timedelta(days=days)).isoformat()
        result = svc.get_journal_entries(
            strategy=params.get("strategy", ""),
            instrument=params.get("instrument", ""),
            trade_type=params.get("trade_type", ""),
            source=params.get("source", ""),
            direction=params.get("direction", ""),
            review_status=params.get("review_status", ""),
            is_closed=is_closed,
            from_date=from_date,
            to_date=to_date,
            limit=int(params.get("page_size", params.get("limit", 100))),
            offset=(int(params.get("page", 1)) - 1) * int(params.get("page_size", params.get("limit", 100))),
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("pro_journal_entries_error", error=str(e))
        return {"entries": [], "total": 0, "error": str(e)}


@app.get("/api/pro-journal/entry/{entry_id}")
async def get_journal_entry(request: Request, entry_id: str) -> dict[str, Any]:
    try:
        svc = get_user_service(request)
        entry = svc.get_journal_entry(entry_id)
        if not entry:
            return {"error": "Entry not found"}
        return entry
    except HTTPException:
        raise
    except Exception as e:
        logger.error("pro_journal_entry_error", error=str(e))
        return {"error": str(e)}


@app.post("/api/pro-journal/entry")
async def create_journal_entry(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
        svc = get_user_service(request)
        entry_id = svc.record_journal_entry(body)
        return {"entry_id": entry_id, "status": "created"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("pro_journal_create_error", error=str(e))
        return {"error": str(e)}


@app.put("/api/pro-journal/entry/{entry_id}")
async def update_journal_entry(request: Request, entry_id: str) -> dict[str, Any]:
    try:
        body = await request.json()
        svc = get_user_service(request)
        ok = svc.update_journal_entry(entry_id, body)
        return {"updated": ok}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("pro_journal_update_error", error=str(e))
        return {"error": str(e)}


@app.get("/api/pro-journal/regime-matrix")
async def get_regime_matrix(request: Request) -> dict[str, Any]:
    try:
        svc = get_user_service(request)
        p = request.query_params
        return {"matrix": svc.get_regime_matrix(
            strategy=p.get("strategy", ""), instrument=p.get("instrument", ""),
            source=p.get("source", ""), direction=p.get("direction", ""),
            trade_type=p.get("trade_type", ""), days=int(p.get("days", 0)))}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("pro_journal_regime_error", error=str(e))
        return {"matrix": []}


@app.get("/api/pro-journal/slippage-drift")
async def get_slippage_drift(request: Request) -> dict[str, Any]:
    try:
        svc = get_user_service(request)
        p = request.query_params
        return svc.get_slippage_drift(
            days=int(p.get("days", 30)), strategy=p.get("strategy", ""),
            instrument=p.get("instrument", ""), source=p.get("source", ""),
            direction=p.get("direction", ""), trade_type=p.get("trade_type", ""))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("pro_journal_slippage_error", error=str(e))
        return {"error": str(e)}


@app.get("/api/pro-journal/daily-pnl")
async def get_daily_pnl_history(request: Request) -> dict[str, Any]:
    try:
        svc = get_user_service(request)
        p = request.query_params
        return {"daily_pnl": svc.get_daily_pnl_history(
            days=int(p.get("days", 30)), strategy=p.get("strategy", ""),
            instrument=p.get("instrument", ""), source=p.get("source", ""),
            direction=p.get("direction", ""), trade_type=p.get("trade_type", ""))}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("pro_journal_daily_pnl_error", error=str(e))
        return {"daily_pnl": []}


@app.get("/api/pro-journal/strategy-breakdown")
async def get_strategy_breakdown(request: Request) -> dict[str, Any]:
    try:
        svc = get_user_service(request)
        p = request.query_params
        return {"strategies": svc.get_strategy_breakdown(
            source=p.get("source", ""), days=int(p.get("days", 0)),
            instrument=p.get("instrument", ""), direction=p.get("direction", ""),
            trade_type=p.get("trade_type", ""), strategy=p.get("strategy", ""))}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("pro_journal_strategy_error", error=str(e))
        return {"strategies": []}


@app.get("/api/pro-journal/portfolio-health")
async def get_portfolio_health(request: Request) -> dict[str, Any]:
    try:
        svc = get_user_service(request)
        return svc.get_portfolio_health()
    except HTTPException:
        raise
    except Exception as e:
        logger.error("pro_journal_portfolio_error", error=str(e))
        return {"error": str(e)}


@app.get("/api/pro-journal/portfolio-health/history")
async def get_portfolio_health_history(request: Request) -> dict[str, Any]:
    try:
        svc = get_user_service(request)
        days = int(request.query_params.get("days", 30))
        return svc.get_portfolio_health_history(days=days)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("pro_journal_health_history_error", error=str(e))
        return {"error": str(e)}


@app.get("/api/pro-journal/equity-curve")
async def get_equity_curve(request: Request) -> dict[str, Any]:
    try:
        svc = get_user_service(request)
        days = int(request.query_params.get("days", 365))
        return {"curve": svc.get_equity_curve(days=days)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("pro_journal_equity_error", error=str(e))
        return {"curve": []}


@app.get("/api/pro-journal/system-events")
async def get_system_events(request: Request) -> dict[str, Any]:
    try:
        svc = get_user_service(request)
        event_type = request.query_params.get("event_type", "")
        limit = int(request.query_params.get("limit", 100))
        return {"events": svc.get_system_events(event_type=event_type, limit=limit)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("pro_journal_events_error", error=str(e))
        return {"events": []}


@app.get("/api/pro-journal/db-stats")
async def get_journal_db_stats(request: Request) -> dict[str, Any]:
    try:
        svc = get_user_service(request)
        return svc.get_journal_db_stats()
    except HTTPException:
        raise
    except Exception as e:
        logger.error("pro_journal_stats_error", error=str(e))
        return {"error": str(e)}


@app.get("/api/pro-journal/filters")
async def get_journal_filters(request: Request) -> dict[str, Any]:
    try:
        svc = get_user_service(request)
        return {
            "strategies": svc.get_journal_strategies(),
            "instruments": svc.get_journal_instruments(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("pro_journal_filters_error", error=str(e))
        return {"strategies": [], "instruments": []}


@app.get("/api/pro-journal/export")
async def export_journal(request: Request) -> dict[str, Any]:
    try:
        svc = get_user_service(request)
        return {"entries": svc.export_journal()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("pro_journal_export_error", error=str(e))
        return {"entries": [], "error": str(e)}


@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket) -> None:
    token = websocket.cookies.get(SESSION_COOKIE, "")
    zerodha_id = otp_auth.get_session_user(token)
    if not zerodha_id:
        await websocket.close(code=4001)
        return
    user_cfg = UserConfigService.get_instance().get_user(zerodha_id)
    if not user_cfg:
        await websocket.close(code=4001)
        return

    await websocket.accept()
    mgr = TradingServiceManager.get_instance()
    svc = mgr.get_service(zerodha_id, api_key=user_cfg.api_key, api_secret=user_cfg.api_secret)
    svc.register_ws_client(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        svc.unregister_ws_client(websocket)
    except Exception:
        svc.unregister_ws_client(websocket)
