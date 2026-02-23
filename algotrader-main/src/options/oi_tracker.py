from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.utils.logger import get_logger

logger = get_logger(__name__)


NIFTY_TOKEN = 256265
SENSEX_TOKEN = 265

NIFTY_LOT_SIZE = 65
SENSEX_LOT_SIZE = 20

NIFTY_STRIKE_GAP = 50
SENSEX_STRIKE_GAP = 100

NEAR_ATM_RANGE = 5  # ±5 strikes from ATM for nearest-strike focus


class OISnapshot(BaseModel):
    timestamp: str = ""
    instrument_token: int = 0
    tradingsymbol: str = ""
    underlying: str = ""
    strike: float = 0.0
    option_type: str = ""
    expiry: str = ""
    oi: int = 0
    oi_day_high: int = 0
    oi_day_low: int = 0
    volume: int = 0
    last_price: float = 0.0
    change: float = 0.0


class OIChangeEntry(BaseModel):
    instrument_token: int = 0
    tradingsymbol: str = ""
    underlying: str = ""
    strike: float = 0.0
    option_type: str = ""
    expiry: str = ""
    current_oi: int = 0
    previous_oi: int = 0
    oi_change_abs: int = 0
    oi_change_pct: float = 0.0
    oi_day_high: int = 0
    oi_day_low: int = 0
    volume: int = 0
    last_price: float = 0.0
    ltp_change: float = 0.0
    is_near_atm: bool = False
    atm_distance: int = 0
    activity_score: float = 0.0


class OITrackerSummary(BaseModel):
    underlying: str = ""
    spot_price: float = 0.0
    atm_strike: float = 0.0
    timestamp: str = ""
    total_ce_oi: int = 0
    total_pe_oi: int = 0
    pcr_oi: float = 0.0
    total_ce_volume: int = 0
    total_pe_volume: int = 0
    pcr_volume: float = 0.0
    most_active_ce: list[dict[str, Any]] = Field(default_factory=list)
    most_active_pe: list[dict[str, Any]] = Field(default_factory=list)
    oi_gainers_ce: list[dict[str, Any]] = Field(default_factory=list)
    oi_gainers_pe: list[dict[str, Any]] = Field(default_factory=list)
    oi_losers_ce: list[dict[str, Any]] = Field(default_factory=list)
    oi_losers_pe: list[dict[str, Any]] = Field(default_factory=list)
    near_atm_data: list[dict[str, Any]] = Field(default_factory=list)
    oi_buildup_signals: list[dict[str, Any]] = Field(default_factory=list)


class OITracker:
    def __init__(self) -> None:
        self._snapshots: dict[str, dict[int, list[OISnapshot]]] = {
            "NIFTY": {},
            "SENSEX": {},
        }
        self._current: dict[str, dict[int, OISnapshot]] = {
            "NIFTY": {},
            "SENSEX": {},
        }
        self._instrument_map: dict[int, dict[str, Any]] = {}
        self._spot_prices: dict[str, float] = {"NIFTY": 0.0, "SENSEX": 0.0}
        self._max_snapshots = 100
        self._ws_clients: list[Any] = []
        self._streaming = False
        self._lock = asyncio.Lock()
        # Cache for nearest expiry per underlying (refreshed on register)
        self._nearest_expiry: dict[str, str] = {"NIFTY": "", "SENSEX": ""}

    def _get_nearest_expiry(self, underlying: str) -> str:
        """Find the nearest (soonest) expiry date for a given underlying from registered instruments."""
        cached = self._nearest_expiry.get(underlying, "")
        if cached:
            return cached
        expiries: set[str] = set()
        for inst in self._instrument_map.values():
            if inst.get("_underlying") != underlying:
                continue
            if inst.get("instrument_type") not in ("CE", "PE"):
                continue
            exp = inst.get("expiry", "")
            if exp:
                expiries.add(str(exp))
        if not expiries:
            return ""
        nearest = sorted(expiries)[0]
        self._nearest_expiry[underlying] = nearest
        return nearest

    def set_spot_price(self, underlying: str, price: float) -> None:
        if underlying.upper() in self._spot_prices:
            self._spot_prices[underlying.upper()] = price

    def get_spot_price(self, underlying: str) -> float:
        return self._spot_prices.get(underlying.upper(), 0.0)

    def register_instruments(self, instruments: list[dict[str, Any]]) -> None:
        # Exact-match sets for each underlying index
        nifty_names = {"NIFTY", "NIFTY 50"}
        sensex_names = {"SENSEX"}
        for inst in instruments:
            token = inst.get("instrument_token", 0)
            if token:
                # Normalize expiry to plain string to avoid type-mismatch issues
                raw_exp = inst.get("expiry")
                if raw_exp is not None:
                    inst["expiry"] = str(raw_exp).strip()
                self._instrument_map[token] = inst
                underlying = ""
                name = inst.get("name", "").upper()
                if name in nifty_names:
                    underlying = "NIFTY"
                elif name in sensex_names:
                    underlying = "SENSEX"
                if underlying:
                    inst["_underlying"] = underlying
        # Refresh nearest-expiry cache
        self._nearest_expiry = {"NIFTY": "", "SENSEX": ""}
        for u in ("NIFTY", "SENSEX"):
            self._get_nearest_expiry(u)

    def get_tracked_tokens(self) -> list[int]:
        """Return only near-ATM, nearest-expiry tokens for tracked underlyings.
        Falls back to all nearest-expiry tokens if no spot price is available yet."""
        tokens: list[int] = []
        for underlying in ("NIFTY", "SENSEX"):
            nearest_exp = self._get_nearest_expiry(underlying)
            spot = self._spot_prices.get(underlying, 0.0)
            if spot > 0:
                gap = NIFTY_STRIKE_GAP if underlying == "NIFTY" else SENSEX_STRIKE_GAP
                tokens += self._get_near_atm_tokens(underlying, spot, gap, nearest_exp or None)
            else:
                # No spot yet — include all nearest-expiry for this underlying
                tokens += [
                    t for t, inst in self._instrument_map.items()
                    if inst.get("instrument_type") in ("CE", "PE")
                    and inst.get("_underlying") == underlying
                    and (not nearest_exp or str(inst.get("expiry", "")) == nearest_exp)
                ]
        return tokens

    def get_nifty_option_tokens(self, spot_price: float, expiry: Optional[str] = None) -> list[int]:
        return self._get_near_atm_tokens("NIFTY", spot_price, NIFTY_STRIKE_GAP, expiry)

    def get_sensex_option_tokens(self, spot_price: float, expiry: Optional[str] = None) -> list[int]:
        return self._get_near_atm_tokens("SENSEX", spot_price, SENSEX_STRIKE_GAP, expiry)

    def _get_near_atm_tokens(
        self, underlying: str, spot: float, gap: int, expiry: Optional[str] = None
    ) -> list[int]:
        atm = round(spot / gap) * gap
        range_low = atm - gap * NEAR_ATM_RANGE
        range_high = atm + gap * NEAR_ATM_RANGE

        # Auto-pick nearest expiry if none specified
        if not expiry:
            expiry = self._get_nearest_expiry(underlying)

        tokens = []
        for token, inst in self._instrument_map.items():
            if inst.get("_underlying") != underlying:
                continue
            if inst.get("instrument_type") not in ("CE", "PE"):
                continue
            # Filter by nearest expiry
            if expiry and str(inst.get("expiry", "")) != expiry:
                continue
            strike = float(inst.get("strike", 0))
            if range_low <= strike <= range_high:
                tokens.append(token)
        return tokens

    def update_from_tick(self, tick_data: dict[str, Any]) -> Optional[str]:
        token = tick_data.get("instrument_token", 0)
        inst = self._instrument_map.get(token)
        if not inst:
            return None

        underlying = inst.get("_underlying", "")
        if underlying not in ("NIFTY", "SENSEX"):
            return None

        oi = tick_data.get("oi", 0)
        if oi is None:
            oi = 0

        snapshot = OISnapshot(
            timestamp=datetime.now().isoformat(),
            instrument_token=token,
            tradingsymbol=inst.get("tradingsymbol", ""),
            underlying=underlying,
            strike=float(inst.get("strike", 0)),
            option_type=inst.get("instrument_type", ""),
            expiry=str(inst.get("expiry", "") or ""),
            oi=oi,
            oi_day_high=tick_data.get("oi_day_high", 0) or 0,
            oi_day_low=tick_data.get("oi_day_low", 0) or 0,
            volume=tick_data.get("volume_traded", 0) or tick_data.get("volume", 0) or 0,
            last_price=tick_data.get("last_price", 0) or 0,
            change=tick_data.get("change", 0) or 0,
        )

        prev = self._current[underlying].get(token)
        self._current[underlying][token] = snapshot

        if token not in self._snapshots[underlying]:
            self._snapshots[underlying][token] = []
        self._snapshots[underlying][token].append(snapshot)
        if len(self._snapshots[underlying][token]) > self._max_snapshots:
            self._snapshots[underlying][token] = self._snapshots[underlying][token][-self._max_snapshots:]

        return underlying

    def get_oi_changes(self, underlying: str, lookback: int = 1) -> list[OIChangeEntry]:
        underlying = underlying.upper()
        if underlying not in self._current:
            return []

        spot = self._spot_prices.get(underlying, 0.0)
        strike_gap = NIFTY_STRIKE_GAP if underlying == "NIFTY" else SENSEX_STRIKE_GAP
        atm = round(spot / strike_gap) * strike_gap if spot > 0 else 0

        # Only process near-ATM strikes (±NEAR_ATM_RANGE from ATM)
        range_low = atm - strike_gap * NEAR_ATM_RANGE if atm > 0 else 0
        range_high = atm + strike_gap * NEAR_ATM_RANGE if atm > 0 else float('inf')

        # Only process nearest expiry instruments
        nearest_exp = self._get_nearest_expiry(underlying)

        changes = []
        skipped_expiry = 0
        for token, current in self._current[underlying].items():
            # Skip strikes outside near-ATM range
            if atm > 0 and not (range_low <= current.strike <= range_high):
                continue
            # Skip non-nearest expiry
            if nearest_exp and current.expiry and str(current.expiry) != nearest_exp:
                skipped_expiry += 1
                continue
            history = self._snapshots[underlying].get(token, [])

            # Determine previous OI:
            # 1) If enough history, use lookback snapshot
            # 2) Otherwise, use oi_day_low as proxy for day-open OI
            # 3) Last resort: use current OI (oi_change = 0) to avoid inflated numbers
            prev_oi = current.oi  # safe default: no change
            if len(history) > lookback:
                prev_oi = history[-(lookback + 1)].oi
            elif current.oi_day_low > 0:
                prev_oi = current.oi_day_low
            # If prev_oi is still 0 somehow, avoid div-by-zero
            if prev_oi <= 0:
                prev_oi = current.oi

            oi_change_abs = current.oi - prev_oi
            oi_change_pct = (oi_change_abs / prev_oi * 100) if prev_oi > 0 else 0.0

            atm_dist = abs(current.strike - atm) / strike_gap if strike_gap > 0 else 999
            is_near = atm_dist <= NEAR_ATM_RANGE

            activity = current.volume + abs(oi_change_abs)

            entry = OIChangeEntry(
                instrument_token=token,
                tradingsymbol=current.tradingsymbol,
                underlying=underlying,
                strike=current.strike,
                option_type=current.option_type,
                expiry=current.expiry,
                current_oi=current.oi,
                previous_oi=prev_oi,
                oi_change_abs=oi_change_abs,
                oi_change_pct=round(oi_change_pct, 2),
                oi_day_high=current.oi_day_high,
                oi_day_low=current.oi_day_low,
                volume=current.volume,
                last_price=current.last_price,
                ltp_change=current.change,
                is_near_atm=is_near,
                atm_distance=int(atm_dist),
                activity_score=float(activity),
            )
            changes.append(entry)

        if skipped_expiry > 0:
            logger.debug("oi_expiry_filter", underlying=underlying,
                        nearest_exp=nearest_exp, skipped=skipped_expiry,
                        kept=len(changes), total=len(self._current[underlying]))

        return changes

    def get_summary(self, underlying: str, top_n: int = 10) -> OITrackerSummary:
        underlying = underlying.upper()
        changes = self.get_oi_changes(underlying)
        if not changes:
            return OITrackerSummary(
                underlying=underlying,
                timestamp=datetime.now().isoformat(),
            )

        spot = self._spot_prices.get(underlying, 0.0)
        strike_gap = NIFTY_STRIKE_GAP if underlying == "NIFTY" else SENSEX_STRIKE_GAP
        atm = round(spot / strike_gap) * strike_gap if spot > 0 else 0

        ce_entries = [c for c in changes if c.option_type == "CE"]
        pe_entries = [c for c in changes if c.option_type == "PE"]

        total_ce_oi = sum(c.current_oi for c in ce_entries)
        total_pe_oi = sum(c.current_oi for c in pe_entries)
        total_ce_vol = sum(c.volume for c in ce_entries)
        total_pe_vol = sum(c.volume for c in pe_entries)

        def entry_dict(e: OIChangeEntry) -> dict[str, Any]:
            return {
                "token": e.instrument_token,
                "symbol": e.tradingsymbol,
                "strike": e.strike,
                "type": e.option_type,
                "expiry": e.expiry,
                "oi": e.current_oi,
                "prev_oi": e.previous_oi,
                "oi_change": e.oi_change_abs,
                "oi_change_pct": e.oi_change_pct,
                "oi_high": e.oi_day_high,
                "oi_low": e.oi_day_low,
                "volume": e.volume,
                "ltp": e.last_price,
                "ltp_change": e.ltp_change,
                "near_atm": e.is_near_atm,
                "atm_dist": e.atm_distance,
                "activity": e.activity_score,
            }

        most_active_ce = sorted(ce_entries, key=lambda x: x.activity_score, reverse=True)[:top_n]
        most_active_pe = sorted(pe_entries, key=lambda x: x.activity_score, reverse=True)[:top_n]

        oi_gainers_ce = sorted(ce_entries, key=lambda x: x.oi_change_abs, reverse=True)[:top_n]
        oi_gainers_pe = sorted(pe_entries, key=lambda x: x.oi_change_abs, reverse=True)[:top_n]

        oi_losers_ce = sorted(ce_entries, key=lambda x: x.oi_change_abs)[:top_n]
        oi_losers_pe = sorted(pe_entries, key=lambda x: x.oi_change_abs)[:top_n]

        near_atm = sorted(
            [c for c in changes if c.is_near_atm],
            key=lambda x: (x.strike, x.option_type),
        )

        buildup_signals = self._detect_buildup_signals(changes, spot)

        return OITrackerSummary(
            underlying=underlying,
            spot_price=spot,
            atm_strike=atm,
            timestamp=datetime.now().isoformat(),
            total_ce_oi=total_ce_oi,
            total_pe_oi=total_pe_oi,
            pcr_oi=round(total_pe_oi / total_ce_oi, 3) if total_ce_oi > 0 else 0.0,
            total_ce_volume=total_ce_vol,
            total_pe_volume=total_pe_vol,
            pcr_volume=round(total_pe_vol / total_ce_vol, 3) if total_ce_vol > 0 else 0.0,
            most_active_ce=[entry_dict(e) for e in most_active_ce],
            most_active_pe=[entry_dict(e) for e in most_active_pe],
            oi_gainers_ce=[entry_dict(e) for e in oi_gainers_ce],
            oi_gainers_pe=[entry_dict(e) for e in oi_gainers_pe],
            oi_losers_ce=[entry_dict(e) for e in oi_losers_ce],
            oi_losers_pe=[entry_dict(e) for e in oi_losers_pe],
            near_atm_data=[entry_dict(e) for e in near_atm],
            oi_buildup_signals=[s for s in buildup_signals],
        )

    def _detect_buildup_signals(
        self, changes: list[OIChangeEntry], spot: float
    ) -> list[dict[str, Any]]:
        signals = []
        for c in changes:
            if not c.is_near_atm or c.current_oi == 0:
                continue

            if c.oi_change_abs > 0 and c.ltp_change > 0:
                signal_type = "LONG_BUILDUP" if c.option_type == "CE" else "SHORT_COVERING"
            elif c.oi_change_abs > 0 and c.ltp_change < 0:
                signal_type = "SHORT_BUILDUP" if c.option_type == "CE" else "LONG_BUILDUP"
            elif c.oi_change_abs < 0 and c.ltp_change < 0:
                signal_type = "LONG_UNWINDING" if c.option_type == "CE" else "SHORT_BUILDUP"
            elif c.oi_change_abs < 0 and c.ltp_change > 0:
                signal_type = "SHORT_COVERING" if c.option_type == "CE" else "LONG_UNWINDING"
            else:
                continue

            if abs(c.oi_change_pct) < 1.0:
                continue

            bullish_types = ("LONG_BUILDUP", "SHORT_COVERING")
            sentiment = "BULLISH" if signal_type in bullish_types else "BEARISH"

            signals.append({
                "symbol": c.tradingsymbol,
                "strike": c.strike,
                "type": c.option_type,
                "signal": signal_type,
                "sentiment": sentiment,
                "oi_change": c.oi_change_abs,
                "oi_change_pct": c.oi_change_pct,
                "ltp": c.last_price,
                "ltp_change": c.ltp_change,
                "volume": c.volume,
                "atm_dist": c.atm_distance,
            })

        signals.sort(key=lambda x: abs(x["oi_change_pct"]), reverse=True)
        return signals[:20]

    def register_oi_ws_client(self, ws: Any) -> None:
        self._ws_clients.append(ws)

    def unregister_oi_ws_client(self, ws: Any) -> None:
        if ws in self._ws_clients:
            self._ws_clients.remove(ws)

    async def broadcast_oi_update(self, data: dict[str, Any]) -> None:
        dead: list[Any] = []
        for ws in self._ws_clients:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._ws_clients.remove(ws)

    def get_all_tracked_data(self) -> dict[str, Any]:
        nifty_summary = self.get_summary("NIFTY")
        sensex_summary = self.get_summary("SENSEX")
        return {
            "type": "oi_update",
            "timestamp": datetime.now().isoformat(),
            "nifty": nifty_summary.model_dump(),
            "sensex": sensex_summary.model_dump(),
        }
