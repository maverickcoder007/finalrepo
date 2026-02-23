"""
State-of-the-art Open Interest Analysis Engine.

Two independent analyzers:
  1. FuturesOIAnalyzer  – NIFTY, SENSEX & major F&O stock futures
     • OI + price correlation  (long/short buildup, unwinding, covering)
     • Multi-expiry rollover detection
     • Price-OI divergence signals
     • Sector-level futures heat-map aggregation
  2. OptionsOIAnalyzer  – Near-ATM options for NIFTY & SENSEX
     • Strike-level OI walls (support / resistance)
     • PCR trend (OI & volume, with 5-snapshot moving average)
     • Max-pain computation
     • Straddle premium tracking  (ATM CE + PE combined)
     • IV skew  (OTM put IV vs OTM call IV)
     • OI change concentration  (where is smart money positioning?)
     • Multi-expiry comparison  (current vs next)
"""

from __future__ import annotations

import asyncio
import math
from collections import defaultdict
from datetime import datetime, date, timedelta
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.options.greeks import BlackScholes
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

NIFTY_STRIKE_GAP = 50
SENSEX_STRIKE_GAP = 100
NEAR_ATM_STRIKES = 5  # ±5 strikes from ATM — nearest strike focus

# Major F&O stocks (tradingsymbol base names used in NFO segment)
FNO_STOCKS: list[str] = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "BAJFINANCE", "MARUTI", "HCLTECH",
    "SUNPHARMA", "TITAN", "TATAMOTORS", "NTPC", "POWERGRID",
    "ULTRACEMCO", "TATASTEEL", "WIPRO", "ADANIENT", "JSWSTEEL",
    "TECHM", "INDUSINDBK", "ONGC", "COALINDIA", "BAJAJFINSV",
    "M&M", "DRREDDY", "NESTLEIND", "CIPLA", "GRASIM",
    "APOLLOHOSP", "HEROMOTOCO", "DIVISLAB", "BPCL", "BRITANNIA",
    "EICHERMOT", "TATACONSUM", "HINDALCO", "VEDL", "BANKBARODA",
    "PNB", "DLF", "LTIM", "ZOMATO", "TRENT",
]

SECTOR_FUTURES: dict[str, str] = {
    "RELIANCE": "Energy", "ONGC": "Energy", "BPCL": "Energy",
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT", "TECHM": "IT", "LTIM": "IT",
    "HDFCBANK": "Banking", "ICICIBANK": "Banking", "SBIN": "Banking",
    "KOTAKBANK": "Banking", "AXISBANK": "Banking", "INDUSINDBK": "Banking",
    "BANKBARODA": "Banking", "PNB": "Banking",
    "BAJFINANCE": "NBFC", "BAJAJFINSV": "NBFC",
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG", "TATACONSUM": "FMCG",
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma", "DIVISLAB": "Pharma",
    "APOLLOHOSP": "Healthcare",
    "BHARTIARTL": "Telecom",
    "MARUTI": "Auto", "TATAMOTORS": "Auto", "M&M": "Auto",
    "HEROMOTOCO": "Auto", "EICHERMOT": "Auto",
    "TATASTEEL": "Metals", "JSWSTEEL": "Metals", "HINDALCO": "Metals", "VEDL": "Metals",
    "ULTRACEMCO": "Cement", "GRASIM": "Cement",
    "LT": "Infra", "DLF": "Infra", "POWERGRID": "Power", "NTPC": "Power",
    "TITAN": "Consumer", "TRENT": "Consumer", "ZOMATO": "Consumer",
    "COALINDIA": "Mining", "ADANIENT": "Conglomerate",
}


# ─────────────────────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────────────────────

class FuturesOIEntry(BaseModel):
    """Single futures contract OI snapshot."""
    symbol: str = ""
    expiry: str = ""
    ltp: float = 0.0
    prev_close: float = 0.0
    change_pct: float = 0.0
    oi: int = 0
    prev_oi: int = 0
    oi_change: int = 0
    oi_change_pct: float = 0.0
    volume: int = 0
    lot_size: int = 0
    value_lakhs: float = 0.0
    buildup: str = ""          # LONG_BUILDUP | SHORT_BUILDUP | LONG_UNWINDING | SHORT_COVERING
    sentiment: str = ""        # BULLISH | BEARISH
    sector: str = ""


class FuturesRollover(BaseModel):
    """Rollover analysis between current & next month."""
    symbol: str = ""
    current_expiry: str = ""
    next_expiry: str = ""
    current_oi: int = 0
    next_oi: int = 0
    rollover_pct: float = 0.0
    current_ltp: float = 0.0
    next_ltp: float = 0.0
    basis_pct: float = 0.0    # (next - current) / current * 100
    signal: str = ""           # POSITIVE_ROLLOVER | NEGATIVE_ROLLOVER | NEUTRAL


class FuturesOIReport(BaseModel):
    """Complete futures OI report."""
    timestamp: str = ""
    index_futures: list[FuturesOIEntry] = Field(default_factory=list)
    stock_futures: list[FuturesOIEntry] = Field(default_factory=list)
    rollover_data: list[FuturesRollover] = Field(default_factory=list)
    top_long_buildup: list[FuturesOIEntry] = Field(default_factory=list)
    top_short_buildup: list[FuturesOIEntry] = Field(default_factory=list)
    top_long_unwinding: list[FuturesOIEntry] = Field(default_factory=list)
    top_short_covering: list[FuturesOIEntry] = Field(default_factory=list)
    sector_summary: list[dict[str, Any]] = Field(default_factory=list)
    market_sentiment: dict[str, Any] = Field(default_factory=dict)


class OptionsOIStrike(BaseModel):
    """OI data for a single strike in the options chain."""
    strike: float = 0.0
    ce_oi: int = 0
    ce_oi_change: int = 0
    ce_volume: int = 0
    ce_ltp: float = 0.0
    ce_iv: float = 0.0
    pe_oi: int = 0
    pe_oi_change: int = 0
    pe_volume: int = 0
    pe_ltp: float = 0.0
    pe_iv: float = 0.0
    pcr_strike: float = 0.0   # PE OI / CE OI at this strike
    net_oi_change: int = 0     # PE OI change - CE OI change  (>0 = bullish)
    is_atm: bool = False
    distance: int = 0          # strikes from ATM


class OptionsOIReport(BaseModel):
    """Complete near-ATM options OI report for an index."""
    underlying: str = ""
    spot_price: float = 0.0
    atm_strike: float = 0.0
    expiry: str = ""
    timestamp: str = ""
    # Aggregates
    total_ce_oi: int = 0
    total_pe_oi: int = 0
    pcr_oi: float = 0.0
    total_ce_volume: int = 0
    total_pe_volume: int = 0
    pcr_volume: float = 0.0
    # Strike data
    strikes: list[OptionsOIStrike] = Field(default_factory=list)
    # Walls
    max_ce_oi_strike: float = 0.0   # strongest CE OI wall (resistance)
    max_pe_oi_strike: float = 0.0   # strongest PE OI wall (support)
    max_ce_oi: int = 0
    max_pe_oi: int = 0
    # Max pain
    max_pain: float = 0.0
    # Straddle premium
    atm_straddle_premium: float = 0.0
    # IV skew (avg put IV - avg call IV near ATM)
    iv_skew: float = 0.0
    avg_ce_iv: float = 0.0
    avg_pe_iv: float = 0.0
    # OI change concentration
    top_ce_oi_additions: list[dict[str, Any]] = Field(default_factory=list)
    top_pe_oi_additions: list[dict[str, Any]] = Field(default_factory=list)
    # Buildup signals
    buildup_signals: list[dict[str, Any]] = Field(default_factory=list)
    # Verdict
    bias: str = ""              # BULLISH | BEARISH | NEUTRAL
    bias_reasons: list[str] = Field(default_factory=list)


class OptionsOIComparison(BaseModel):
    """Compare OI across two expiries."""
    underlying: str = ""
    current_expiry: str = ""
    next_expiry: str = ""
    current_report: Optional[OptionsOIReport] = None
    next_report: Optional[OptionsOIReport] = None
    pcr_shift: float = 0.0         # next PCR - current PCR
    max_pain_shift: float = 0.0    # next max pain - current max pain
    premium_shift: float = 0.0     # straddle premium shift


# ─────────────────────────────────────────────────────────────
# Futures OI Analyzer
# ─────────────────────────────────────────────────────────────

class FuturesOIAnalyzer:
    """
    Analyzes open-interest data for index & stock futures.

    Data source: Kite API via
      • get_instruments('NFO')  → to find FUT instrument tokens
      • get_quote([...])        → to get LTP, OI, volume for each
      • get_historical_data(... oi=True) → for previous-day close/OI
    """

    def __init__(self) -> None:
        self._instruments: dict[str, list[dict[str, Any]]] = {}   # symbol → list of FUT instruments (per expiry)
        self._last_report: Optional[FuturesOIReport] = None
        self._historical_oi: dict[str, list[dict[str, Any]]] = {}  # symbol → time-series

    # ── Public API ────────────────────────────────────────────

    async def analyze(self, client: Any) -> FuturesOIReport:
        """
        Full futures OI scan.  Fetches instruments, quotes, and builds the report.
        `client` is a KiteClient instance.
        """
        # 1 – Discover FUT instruments
        instruments = await self._load_instruments(client)
        if not instruments:
            return FuturesOIReport(timestamp=datetime.now().isoformat())

        # 2 – Batch quote fetch (Kite allows 500 instruments per call)
        quotes = await self._fetch_quotes(client, instruments)

        # 3 – Build entries per symbol
        entries: list[FuturesOIEntry] = []
        rollover_map: dict[str, list[FuturesOIEntry]] = defaultdict(list)

        for inst in instruments:
            sym = inst["name"]
            token = inst["instrument_token"]
            expiry = inst.get("expiry", "")
            lot = inst.get("lot_size", 1)
            key = f"{inst.get('exchange', 'NFO')}:{inst.get('tradingsymbol', '')}"
            q = quotes.get(key, {})

            ltp = q.get("last_price", 0.0)
            volume = q.get("volume", 0)
            oi = q.get("oi", 0)
            oi_high = q.get("oi_day_high", 0)
            oi_low = q.get("oi_day_low", 0)
            ohlc = q.get("ohlc", {})
            prev_close = ohlc.get("close", ltp) if isinstance(ohlc, dict) else ltp

            # Previous day OI from historical cache or approximate from oi_day_low
            prev_oi = self._get_prev_oi(sym, expiry, oi)
            oi_change = oi - prev_oi
            oi_change_pct = (oi_change / prev_oi * 100) if prev_oi > 0 else 0.0
            price_change_pct = ((ltp - prev_close) / prev_close * 100) if prev_close > 0 else 0.0

            buildup, sentiment = self._classify_buildup(price_change_pct, oi_change)

            entry = FuturesOIEntry(
                symbol=sym,
                expiry=expiry,
                ltp=round(ltp, 2),
                prev_close=round(prev_close, 2),
                change_pct=round(price_change_pct, 2),
                oi=oi,
                prev_oi=prev_oi,
                oi_change=oi_change,
                oi_change_pct=round(oi_change_pct, 2),
                volume=volume,
                lot_size=lot,
                value_lakhs=round((oi * lot * ltp) / 100000, 2) if ltp > 0 else 0.0,
                buildup=buildup,
                sentiment=sentiment,
                sector=SECTOR_FUTURES.get(sym, "Other"),
            )
            entries.append(entry)
            rollover_map[sym].append(entry)

            # Cache OI for next run
            self._store_oi(sym, expiry, oi, ltp)

        # 4 – Separate index vs stock
        index_syms = {"NIFTY", "BANKNIFTY", "NIFTY 50", "SENSEX", "FINNIFTY", "MIDCPNIFTY"}
        index_futures = [e for e in entries if e.symbol.upper() in index_syms]
        stock_futures = [e for e in entries if e.symbol.upper() not in index_syms]

        # Keep only nearest expiry for per-stock list
        nearest_stock = self._nearest_expiry_only(stock_futures)
        nearest_index = self._nearest_expiry_only(index_futures)

        # 5 – Rollover analysis
        rollovers = self._compute_rollovers(rollover_map)

        # 6 – Top buildups
        long_buildup = sorted(
            [e for e in nearest_stock if e.buildup == "LONG_BUILDUP"],
            key=lambda x: x.oi_change_pct, reverse=True
        )[:15]
        short_buildup = sorted(
            [e for e in nearest_stock if e.buildup == "SHORT_BUILDUP"],
            key=lambda x: x.oi_change_pct, reverse=True
        )[:15]
        long_unwinding = sorted(
            [e for e in nearest_stock if e.buildup == "LONG_UNWINDING"],
            key=lambda x: abs(x.oi_change_pct), reverse=True
        )[:15]
        short_covering = sorted(
            [e for e in nearest_stock if e.buildup == "SHORT_COVERING"],
            key=lambda x: abs(x.oi_change_pct), reverse=True
        )[:15]

        # 7 – Sector summary
        sector_summ = self._compute_sector_summary(nearest_stock)

        # 8 – Overall sentiment
        sentiment = self._compute_market_sentiment(nearest_stock, nearest_index)

        report = FuturesOIReport(
            timestamp=datetime.now().isoformat(),
            index_futures=nearest_index,
            stock_futures=nearest_stock,
            rollover_data=rollovers,
            top_long_buildup=long_buildup,
            top_short_buildup=short_buildup,
            top_long_unwinding=long_unwinding,
            top_short_covering=short_covering,
            sector_summary=sector_summ,
            market_sentiment=sentiment,
        )
        self._last_report = report
        return report

    def get_last_report(self) -> Optional[FuturesOIReport]:
        return self._last_report

    # ── Internals ─────────────────────────────────────────────

    async def _load_instruments(self, client: Any) -> list[dict[str, Any]]:
        """Fetch NFO instruments and filter FUT contracts for our target list."""
        try:
            all_nfo = await client.get_instruments("NFO")
        except Exception as e:
            logger.error("futures_oi_instruments_error", error=str(e))
            return []

        target_syms = set(s.upper() for s in FNO_STOCKS)
        target_syms.update({"NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "MIDCPNIFTY", "NIFTY 50"})

        result: list[dict[str, Any]] = []
        for inst in all_nfo:
            inst_dict = inst.model_dump() if hasattr(inst, "model_dump") else inst
            if inst_dict.get("instrument_type") != "FUT":
                continue
            name = inst_dict.get("name", "").upper()
            if name not in target_syms:
                continue
            result.append(inst_dict)

        self._instruments = defaultdict(list)
        for inst in result:
            self._instruments[inst["name"].upper()].append(inst)

        logger.info("futures_instruments_loaded", count=len(result))
        return result

    async def _fetch_quotes(self, client: Any, instruments: list[dict[str, Any]]) -> dict[str, Any]:
        """Fetch quotes in batches of 500."""
        keys = [f"{inst.get('exchange', 'NFO')}:{inst.get('tradingsymbol', '')}" for inst in instruments]
        quotes: dict[str, Any] = {}
        for i in range(0, len(keys), 500):
            batch = keys[i:i + 500]
            try:
                batch_quotes = await client.get_quote(batch)
                for k, v in batch_quotes.items():
                    quotes[k] = v.model_dump() if hasattr(v, "model_dump") else v
            except Exception as e:
                logger.error("futures_quote_error", batch_start=i, error=str(e))
        return quotes

    def _get_prev_oi(self, symbol: str, expiry: str, current_oi: int) -> int:
        """Get previous OI from cache. Returns current_oi if unavailable (no false signals on first scan)."""
        key = f"{symbol}:{expiry}"
        history = self._historical_oi.get(key, [])
        if len(history) >= 2:
            return history[-2].get("oi", 0)
        # First scan: return current_oi so oi_change=0, preventing false buildup signals
        return current_oi

    def _store_oi(self, symbol: str, expiry: str, oi: int, ltp: float) -> None:
        key = f"{symbol}:{expiry}"
        if key not in self._historical_oi:
            self._historical_oi[key] = []
        self._historical_oi[key].append({
            "timestamp": datetime.now().isoformat(),
            "oi": oi, "ltp": ltp,
        })
        # Keep last 50 snapshots
        if len(self._historical_oi[key]) > 50:
            self._historical_oi[key] = self._historical_oi[key][-50:]

    @staticmethod
    def _classify_buildup(price_change_pct: float, oi_change: int) -> tuple[str, str]:
        """
        Classify OI + Price movement into one of 4 buildup types.
        Price ↑ + OI ↑ = Long Buildup     (BULLISH)
        Price ↓ + OI ↑ = Short Buildup    (BEARISH)
        Price ↓ + OI ↓ = Long Unwinding   (BEARISH)
        Price ↑ + OI ↓ = Short Covering   (BULLISH)
        """
        if oi_change > 0 and price_change_pct > 0:
            return "LONG_BUILDUP", "BULLISH"
        elif oi_change > 0 and price_change_pct < 0:
            return "SHORT_BUILDUP", "BEARISH"
        elif oi_change < 0 and price_change_pct < 0:
            return "LONG_UNWINDING", "BEARISH"
        elif oi_change < 0 and price_change_pct > 0:
            return "SHORT_COVERING", "BULLISH"
        return "NEUTRAL", "NEUTRAL"

    @staticmethod
    def _nearest_expiry_only(entries: list[FuturesOIEntry]) -> list[FuturesOIEntry]:
        """Keep only the nearest expiry per symbol."""
        best: dict[str, FuturesOIEntry] = {}
        for e in entries:
            if e.symbol not in best or e.expiry < best[e.symbol].expiry:
                best[e.symbol] = e
        return sorted(best.values(), key=lambda x: abs(x.oi_change_pct), reverse=True)

    @staticmethod
    def _compute_rollovers(rollover_map: dict[str, list[FuturesOIEntry]]) -> list[FuturesRollover]:
        """Compute rollover % from current→next expiry."""
        rollovers: list[FuturesRollover] = []
        for sym, entries in rollover_map.items():
            if len(entries) < 2:
                continue
            sorted_entries = sorted(entries, key=lambda x: x.expiry)
            current = sorted_entries[0]
            nxt = sorted_entries[1]
            total_oi = current.oi + nxt.oi
            rollover_pct = (nxt.oi / total_oi * 100) if total_oi > 0 else 0.0
            basis_pct = ((nxt.ltp - current.ltp) / current.ltp * 100) if current.ltp > 0 else 0.0

            signal = "NEUTRAL"
            if rollover_pct > 30 and basis_pct > 0.1:
                signal = "POSITIVE_ROLLOVER"
            elif rollover_pct > 30 and basis_pct < -0.1:
                signal = "NEGATIVE_ROLLOVER"

            rollovers.append(FuturesRollover(
                symbol=sym,
                current_expiry=current.expiry,
                next_expiry=nxt.expiry,
                current_oi=current.oi,
                next_oi=nxt.oi,
                rollover_pct=round(rollover_pct, 2),
                current_ltp=current.ltp,
                next_ltp=nxt.ltp,
                basis_pct=round(basis_pct, 2),
                signal=signal,
            ))
        rollovers.sort(key=lambda x: x.rollover_pct, reverse=True)
        return rollovers

    @staticmethod
    def _compute_sector_summary(stock_entries: list[FuturesOIEntry]) -> list[dict[str, Any]]:
        """Aggregate futures OI by sector."""
        sectors: dict[str, dict[str, Any]] = defaultdict(lambda: {
            "long_buildup": 0, "short_buildup": 0, "long_unwinding": 0,
            "short_covering": 0, "total_oi_change": 0, "total_volume": 0,
            "avg_change_pct": [], "stocks": [],
        })
        for e in stock_entries:
            sec = e.sector or "Other"
            d = sectors[sec]
            d[e.buildup.lower()] = d.get(e.buildup.lower(), 0) + 1
            d["total_oi_change"] += e.oi_change
            d["total_volume"] += e.volume
            d["avg_change_pct"].append(e.change_pct)
            d["stocks"].append({"symbol": e.symbol, "buildup": e.buildup, "change_pct": e.change_pct})

        result: list[dict[str, Any]] = []
        for sec_name, d in sectors.items():
            avg_chg = sum(d["avg_change_pct"]) / len(d["avg_change_pct"]) if d["avg_change_pct"] else 0.0
            bullish = d.get("long_buildup", 0) + d.get("short_covering", 0)
            bearish = d.get("short_buildup", 0) + d.get("long_unwinding", 0)
            total = bullish + bearish
            bias = "BULLISH" if bullish > bearish else "BEARISH" if bearish > bullish else "NEUTRAL"
            result.append({
                "sector": sec_name,
                "avg_change_pct": round(avg_chg, 2),
                "long_buildup": d.get("long_buildup", 0),
                "short_buildup": d.get("short_buildup", 0),
                "long_unwinding": d.get("long_unwinding", 0),
                "short_covering": d.get("short_covering", 0),
                "total_oi_change": d["total_oi_change"],
                "total_volume": d["total_volume"],
                "bias": bias,
                "bullish_pct": round(bullish / total * 100, 1) if total > 0 else 50.0,
                "stock_count": len(d["avg_change_pct"]),
                "stocks": d["stocks"],
            })
        result.sort(key=lambda x: x["avg_change_pct"], reverse=True)
        return result

    @staticmethod
    def _compute_market_sentiment(
        stocks: list[FuturesOIEntry],
        indices: list[FuturesOIEntry],
    ) -> dict[str, Any]:
        """Overall market sentiment from futures OI data."""
        if not stocks:
            return {"bias": "NEUTRAL", "confidence": 50, "reasons": []}

        long_b = sum(1 for e in stocks if e.buildup == "LONG_BUILDUP")
        short_b = sum(1 for e in stocks if e.buildup == "SHORT_BUILDUP")
        long_u = sum(1 for e in stocks if e.buildup == "LONG_UNWINDING")
        short_c = sum(1 for e in stocks if e.buildup == "SHORT_COVERING")
        total = len(stocks)

        bullish_count = long_b + short_c
        bearish_count = short_b + long_u
        bullish_pct = (bullish_count / total * 100) if total > 0 else 50

        reasons: list[str] = []
        if long_b > total * 0.3:
            reasons.append(f"Strong long buildup in {long_b} stocks")
        if short_b > total * 0.3:
            reasons.append(f"Heavy short buildup in {short_b} stocks")
        if short_c > total * 0.2:
            reasons.append(f"Short covering in {short_c} stocks")
        if long_u > total * 0.2:
            reasons.append(f"Long unwinding in {long_u} stocks")

        # Index signal
        for idx in indices:
            if idx.buildup in ("LONG_BUILDUP", "SHORT_COVERING"):
                reasons.append(f"{idx.symbol} futures: {idx.buildup.replace('_', ' ').title()}")
            elif idx.buildup in ("SHORT_BUILDUP", "LONG_UNWINDING"):
                reasons.append(f"{idx.symbol} futures: {idx.buildup.replace('_', ' ').title()}")

        bias = "BULLISH" if bullish_pct >= 55 else "BEARISH" if bullish_pct <= 45 else "NEUTRAL"
        confidence = int(abs(bullish_pct - 50) * 2)

        return {
            "bias": bias,
            "confidence": min(confidence, 95),
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "long_buildup": long_b,
            "short_buildup": short_b,
            "long_unwinding": long_u,
            "short_covering": short_c,
            "bullish_pct": round(bullish_pct, 1),
            "reasons": reasons,
        }


# ─────────────────────────────────────────────────────────────
# Options OI Analyzer
# ─────────────────────────────────────────────────────────────

class OptionsOIAnalyzer:
    """
    Near-ATM options OI analysis for NIFTY & SENSEX.

    Provides:
      • PCR (OI + Volume)
      • OI walls  (max CE/PE OI strikes → resistance/support)
      • Max-pain computation
      • ATM straddle premium
      • IV skew (put IV vs call IV)
      • Buildup signals at each strike
      • Overall directional bias
    """

    def __init__(self) -> None:
        self._pcr_history: dict[str, list[dict[str, Any]]] = {"NIFTY": [], "SENSEX": []}
        self._straddle_history: dict[str, list[dict[str, Any]]] = {"NIFTY": [], "SENSEX": []}
        self._last_reports: dict[str, OptionsOIReport] = {}
        # Track per-instrument OI from previous scan for accurate OI change
        # Key: instrument tradingsymbol → previous scan OI value
        self._prev_scan_oi: dict[str, int] = {}
        self._risk_free_rate = 0.065  # ~6.5% Indian risk-free rate for IV calc

    # ── Public API ────────────────────────────────────────────

    async def analyze(
        self,
        underlying: str,
        client: Any,
        spot_price: float = 0.0,
        expiry_filter: Optional[str] = None,
    ) -> OptionsOIReport:
        """
        Full near-ATM options OI analysis for an index.
        """
        underlying = underlying.upper()
        strike_gap = NIFTY_STRIKE_GAP if underlying in ("NIFTY", "NIFTY 50") else SENSEX_STRIKE_GAP

        # 1 – Discover option instruments
        instruments = await self._load_option_instruments(client, underlying, expiry_filter)
        if not instruments:
            return OptionsOIReport(underlying=underlying, timestamp=datetime.now().isoformat())

        # 2 – Get spot if not provided
        if spot_price <= 0:
            spot_price = await self._get_spot_price(client, underlying)
        atm_strike = round(spot_price / strike_gap) * strike_gap

        # 3 – Filter to near-ATM range
        low_bound = atm_strike - strike_gap * NEAR_ATM_STRIKES
        high_bound = atm_strike + strike_gap * NEAR_ATM_STRIKES
        near_atm_inst = [i for i in instruments if low_bound <= i.get("strike", 0) <= high_bound]

        # 4 – Determine expiry
        expiry = expiry_filter
        if not expiry and near_atm_inst:
            expiries = sorted(set(str(i.get("expiry", "")) for i in near_atm_inst if i.get("expiry")))
            expiry = expiries[0] if expiries else ""
        if expiry:
            near_atm_inst = [i for i in near_atm_inst if str(i.get("expiry", "")) == str(expiry)]

        # 5 – Fetch quotes
        quotes = await self._fetch_quotes(client, near_atm_inst)

        # Calculate time-to-expiry for IV computation
        tte = 0.0
        if expiry:
            try:
                # Expiry can be "YYYY-MM-DD" or date object
                exp_date = datetime.strptime(str(expiry), "%Y-%m-%d").date() if isinstance(expiry, str) else expiry
                days_to_exp = (exp_date - date.today()).days
                tte = max(days_to_exp, 1) / 365.0  # At least 1 day to avoid div-by-zero
            except Exception:
                tte = 7 / 365.0  # Fallback: assume ~1 week

        # 6 – Build strike map
        strike_data: dict[float, dict[str, Any]] = defaultdict(lambda: {
            "ce_oi": 0, "ce_oi_change": 0, "ce_volume": 0, "ce_ltp": 0.0, "ce_iv": 0.0,
            "pe_oi": 0, "pe_oi_change": 0, "pe_volume": 0, "pe_ltp": 0.0, "pe_iv": 0.0,
        })

        new_oi_snapshot: dict[str, int] = {}  # symbol → current OI (to store after loop)

        for inst in near_atm_inst:
            strike = float(inst.get("strike", 0))
            opt_type = inst.get("instrument_type", "")
            tsym = inst.get("tradingsymbol", "")
            key = f"{inst.get('exchange', 'NFO')}:{tsym}"
            q = quotes.get(key, {})

            oi = q.get("oi", 0)
            volume = q.get("volume", 0)
            ltp = q.get("last_price", 0.0)
            oi_high = q.get("oi_day_high", 0)
            oi_low = q.get("oi_day_low", 0)

            # ── OI Change: use prev-scan tracking when available ──
            # Priority: (1) previous scan OI, (2) oi_day_low approximation for first scan
            prev_oi = self._prev_scan_oi.get(tsym)
            if prev_oi is not None:
                oi_change = oi - prev_oi
            else:
                # First scan: oi_day_low gives the day's opening OI (closest approx)
                # But only if it's sensible (>0 and < current OI × 2)
                if oi_low > 0 and oi_low <= oi * 2:
                    oi_change = oi - oi_low
                else:
                    oi_change = 0  # No data → show 0 instead of inflated number
            new_oi_snapshot[tsym] = oi

            # ── IV: compute from Black-Scholes ──
            iv = 0.0
            if ltp > 0 and spot_price > 0 and strike > 0 and tte > 0:
                try:
                    iv = BlackScholes.implied_volatility(
                        market_price=ltp,
                        spot=spot_price,
                        strike=strike,
                        rate=self._risk_free_rate,
                        time_to_expiry=tte,
                        option_type=opt_type,
                    )
                    iv = iv * 100  # Convert to percentage
                except Exception:
                    iv = 0.0

            prefix = "ce_" if opt_type == "CE" else "pe_"
            d = strike_data[strike]
            d[f"{prefix}oi"] = oi
            d[f"{prefix}oi_change"] = oi_change
            d[f"{prefix}volume"] = volume
            d[f"{prefix}ltp"] = ltp
            d[f"{prefix}iv"] = iv

        # Update prev-scan OI for next call
        self._prev_scan_oi.update(new_oi_snapshot)

        # 7 – Build strike entries
        strikes_list: list[OptionsOIStrike] = []
        total_ce_oi = total_pe_oi = total_ce_vol = total_pe_vol = 0
        max_ce_oi = max_pe_oi = 0
        max_ce_strike = max_pe_strike = atm_strike
        ce_ivs: list[float] = []
        pe_ivs: list[float] = []

        for strike_val in sorted(strike_data.keys()):
            d = strike_data[strike_val]
            distance = int((strike_val - atm_strike) / strike_gap)
            pcr_s = d["pe_oi"] / d["ce_oi"] if d["ce_oi"] > 0 else 0.0
            net_oi_change = d["pe_oi_change"] - d["ce_oi_change"]

            entry = OptionsOIStrike(
                strike=strike_val,
                ce_oi=d["ce_oi"], ce_oi_change=d["ce_oi_change"],
                ce_volume=d["ce_volume"], ce_ltp=round(d["ce_ltp"], 2),
                ce_iv=round(d["ce_iv"], 2),
                pe_oi=d["pe_oi"], pe_oi_change=d["pe_oi_change"],
                pe_volume=d["pe_volume"], pe_ltp=round(d["pe_ltp"], 2),
                pe_iv=round(d["pe_iv"], 2),
                pcr_strike=round(pcr_s, 2),
                net_oi_change=net_oi_change,
                is_atm=(strike_val == atm_strike),
                distance=distance,
            )
            strikes_list.append(entry)

            total_ce_oi += d["ce_oi"]
            total_pe_oi += d["pe_oi"]
            total_ce_vol += d["ce_volume"]
            total_pe_vol += d["pe_volume"]

            if d["ce_oi"] > max_ce_oi:
                max_ce_oi = d["ce_oi"]
                max_ce_strike = strike_val
            if d["pe_oi"] > max_pe_oi:
                max_pe_oi = d["pe_oi"]
                max_pe_strike = strike_val

            if d["ce_iv"] > 0:
                ce_ivs.append(d["ce_iv"])
            if d["pe_iv"] > 0:
                pe_ivs.append(d["pe_iv"])

        pcr_oi = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0.0
        pcr_vol = total_pe_vol / total_ce_vol if total_ce_vol > 0 else 0.0

        # ATM straddle premium
        straddle_premium = 0.0
        atm_data = strike_data.get(atm_strike)
        if atm_data:
            straddle_premium = atm_data["ce_ltp"] + atm_data["pe_ltp"]

        # IV skew
        avg_ce_iv = sum(ce_ivs) / len(ce_ivs) if ce_ivs else 0.0
        avg_pe_iv = sum(pe_ivs) / len(pe_ivs) if pe_ivs else 0.0
        iv_skew = avg_pe_iv - avg_ce_iv

        # Max pain
        max_pain = self._compute_max_pain(strikes_list)

        # Top OI additions
        top_ce_adds = sorted(
            [{"strike": s.strike, "oi": s.ce_oi, "oi_change": s.ce_oi_change, "ltp": s.ce_ltp}
             for s in strikes_list if s.ce_oi_change > 0],
            key=lambda x: x["oi_change"], reverse=True
        )[:5]
        top_pe_adds = sorted(
            [{"strike": s.strike, "oi": s.pe_oi, "oi_change": s.pe_oi_change, "ltp": s.pe_ltp}
             for s in strikes_list if s.pe_oi_change > 0],
            key=lambda x: x["oi_change"], reverse=True
        )[:5]

        # Buildup signals
        buildup_signals = self._detect_buildup_signals(strikes_list, atm_strike)

        # Bias
        bias, bias_reasons = self._compute_bias(
            pcr_oi, max_pain, spot_price, max_ce_strike, max_pe_strike,
            iv_skew, buildup_signals, straddle_premium, strikes_list
        )

        report = OptionsOIReport(
            underlying=underlying,
            spot_price=round(spot_price, 2),
            atm_strike=atm_strike,
            expiry=expiry or "",
            timestamp=datetime.now().isoformat(),
            total_ce_oi=total_ce_oi,
            total_pe_oi=total_pe_oi,
            pcr_oi=round(pcr_oi, 3),
            total_ce_volume=total_ce_vol,
            total_pe_volume=total_pe_vol,
            pcr_volume=round(pcr_vol, 3),
            strikes=strikes_list,
            max_ce_oi_strike=max_ce_strike,
            max_pe_oi_strike=max_pe_strike,
            max_ce_oi=max_ce_oi,
            max_pe_oi=max_pe_oi,
            max_pain=max_pain,
            atm_straddle_premium=round(straddle_premium, 2),
            iv_skew=round(iv_skew, 2),
            avg_ce_iv=round(avg_ce_iv, 2),
            avg_pe_iv=round(avg_pe_iv, 2),
            top_ce_oi_additions=top_ce_adds,
            top_pe_oi_additions=top_pe_adds,
            buildup_signals=buildup_signals,
            bias=bias,
            bias_reasons=bias_reasons,
        )

        # Track PCR and straddle history
        self._track_history(underlying, pcr_oi, straddle_premium)
        self._last_reports[underlying] = report
        return report

    async def analyze_comparison(
        self, underlying: str, client: Any, spot_price: float = 0.0,
    ) -> OptionsOIComparison:
        """Compare current vs next expiry for an index."""
        underlying = underlying.upper()

        instruments = await self._load_option_instruments(client, underlying)
        expiries = sorted(set(str(i.get("expiry", "")) for i in instruments if i.get("expiry")))
        if len(expiries) < 2:
            current_report = await self.analyze(underlying, client, spot_price, expiries[0] if expiries else None)
            return OptionsOIComparison(
                underlying=underlying,
                current_expiry=expiries[0] if expiries else "",
                current_report=current_report,
            )

        current_report = await self.analyze(underlying, client, spot_price, expiries[0])
        next_report = await self.analyze(underlying, client, spot_price, expiries[1])

        return OptionsOIComparison(
            underlying=underlying,
            current_expiry=expiries[0],
            next_expiry=expiries[1],
            current_report=current_report,
            next_report=next_report,
            pcr_shift=round(next_report.pcr_oi - current_report.pcr_oi, 3),
            max_pain_shift=next_report.max_pain - current_report.max_pain,
            premium_shift=round(next_report.atm_straddle_premium - current_report.atm_straddle_premium, 2),
        )

    def get_pcr_history(self, underlying: str) -> list[dict[str, Any]]:
        return self._pcr_history.get(underlying.upper(), [])

    def get_straddle_history(self, underlying: str) -> list[dict[str, Any]]:
        return self._straddle_history.get(underlying.upper(), [])

    def get_last_report(self, underlying: str) -> Optional[OptionsOIReport]:
        return self._last_reports.get(underlying.upper())

    # ── Internals ─────────────────────────────────────────────

    async def _load_option_instruments(
        self, client: Any, underlying: str, expiry_filter: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Fetch NFO instruments and filter options for the given underlying."""
        try:
            exchange = "BFO" if underlying == "SENSEX" else "NFO"
            all_inst = await client.get_instruments(exchange)
        except Exception as e:
            logger.error("options_oi_instruments_error", error=str(e))
            return []

        result: list[dict[str, Any]] = []
        underlying_upper = underlying.upper()
        # Canonical name mapping: some instruments use different name variants
        canonical_names = {underlying_upper}
        if underlying_upper == "NIFTY":
            canonical_names.update({"NIFTY", "NIFTY 50"})
        elif underlying_upper == "NIFTY 50":
            canonical_names.update({"NIFTY", "NIFTY 50"})
        elif underlying_upper == "BANKNIFTY":
            canonical_names.update({"BANKNIFTY", "NIFTY BANK"})

        for inst in all_inst:
            d = inst.model_dump() if hasattr(inst, "model_dump") else inst
            if d.get("instrument_type") not in ("CE", "PE"):
                continue
            # Normalize expiry to string for consistent comparisons
            raw_exp = d.get("expiry")
            if raw_exp is not None:
                d["expiry"] = str(raw_exp).strip()
            name = d.get("name", "").upper()
            # Exact match only — prevents NIFTY matching BANKNIFTY/FINNIFTY/MIDCPNIFTY
            if name not in canonical_names:
                continue
            if expiry_filter and d.get("expiry") != str(expiry_filter):
                continue
            result.append(d)
        return result

    async def _get_spot_price(self, client: Any, underlying: str) -> float:
        """Fetch spot price for an index."""
        try:
            idx_map = {
                "NIFTY": "NSE:NIFTY 50",
                "NIFTY 50": "NSE:NIFTY 50",
                "SENSEX": "BSE:SENSEX",
                "BANKNIFTY": "NSE:NIFTY BANK",
            }
            key = idx_map.get(underlying.upper(), f"NSE:{underlying}")
            ltp_data = await client.get_ltp([key])
            for k, v in ltp_data.items():
                price = v.last_price if hasattr(v, "last_price") else v.get("last_price", 0)
                if price > 0:
                    return price
        except Exception as e:
            logger.error("spot_price_fetch_error", underlying=underlying, error=str(e))
        return 0.0

    async def _fetch_quotes(self, client: Any, instruments: list[dict[str, Any]]) -> dict[str, Any]:
        """Fetch quotes in batches of 500."""
        keys = [f"{inst.get('exchange', 'NFO')}:{inst.get('tradingsymbol', '')}" for inst in instruments]
        quotes: dict[str, Any] = {}
        for i in range(0, len(keys), 500):
            batch = keys[i:i + 500]
            try:
                batch_quotes = await client.get_quote(batch)
                for k, v in batch_quotes.items():
                    quotes[k] = v.model_dump() if hasattr(v, "model_dump") else v
            except Exception as e:
                logger.error("options_quote_error", batch_start=i, error=str(e))
        return quotes

    @staticmethod
    def _compute_max_pain(strikes: list[OptionsOIStrike]) -> float:
        """Calculate max pain strike from OI data."""
        if not strikes:
            return 0.0
        min_pain = float("inf")
        max_pain_strike = strikes[0].strike

        for test in strikes:
            total_pain = 0.0
            for s in strikes:
                # CE writers pain
                if s.ce_oi > 0:
                    intrinsic = max(test.strike - s.strike, 0)
                    total_pain += intrinsic * s.ce_oi
                # PE writers pain
                if s.pe_oi > 0:
                    intrinsic = max(s.strike - test.strike, 0)
                    total_pain += intrinsic * s.pe_oi
            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = test.strike
        return max_pain_strike

    @staticmethod
    def _detect_buildup_signals(
        strikes: list[OptionsOIStrike], atm_strike: float,
    ) -> list[dict[str, Any]]:
        """Detect buildup patterns at each strike."""
        signals: list[dict[str, Any]] = []
        for s in strikes:
            # CE buildup analysis
            if s.ce_oi_change > 0 and s.ce_ltp > 0:
                if s.strike > atm_strike:
                    # OTM CE OI increase = Resistance writing (BEARISH)
                    signals.append({
                        "strike": s.strike, "type": "CE", "pattern": "CE_WRITING",
                        "sentiment": "BEARISH", "oi_change": s.ce_oi_change,
                        "description": f"CE writers adding at {s.strike} (resistance)",
                    })
                else:
                    # ITM CE OI increase = Smart money long
                    signals.append({
                        "strike": s.strike, "type": "CE", "pattern": "CE_BUYING",
                        "sentiment": "BULLISH", "oi_change": s.ce_oi_change,
                        "description": f"CE buying at {s.strike} (bullish)",
                    })

            # PE buildup analysis
            if s.pe_oi_change > 0 and s.pe_ltp > 0:
                if s.strike < atm_strike:
                    # OTM PE OI increase = Support writing (BULLISH)
                    signals.append({
                        "strike": s.strike, "type": "PE", "pattern": "PE_WRITING",
                        "sentiment": "BULLISH", "oi_change": s.pe_oi_change,
                        "description": f"PE writers adding at {s.strike} (support)",
                    })
                else:
                    # ITM PE OI increase = Smart money short
                    signals.append({
                        "strike": s.strike, "type": "PE", "pattern": "PE_BUYING",
                        "sentiment": "BEARISH", "oi_change": s.pe_oi_change,
                        "description": f"PE buying at {s.strike} (bearish)",
                    })

        signals.sort(key=lambda x: abs(x["oi_change"]), reverse=True)
        return signals[:15]

    @staticmethod
    def _compute_bias(
        pcr_oi: float, max_pain: float, spot: float,
        max_ce_strike: float, max_pe_strike: float,
        iv_skew: float, signals: list[dict[str, Any]],
        straddle_premium: float, strikes: list[OptionsOIStrike],
    ) -> tuple[str, list[str]]:
        """Compute overall directional bias from all OI dimensions."""
        bull_points = 0
        bear_points = 0
        reasons: list[str] = []

        # PCR
        if pcr_oi > 1.2:
            bull_points += 2
            reasons.append(f"High PCR ({pcr_oi:.2f}) = oversold/bullish")
        elif pcr_oi < 0.7:
            bear_points += 2
            reasons.append(f"Low PCR ({pcr_oi:.2f}) = overbought/bearish")
        elif pcr_oi > 1.0:
            bull_points += 1
            reasons.append(f"PCR above 1 ({pcr_oi:.2f})")
        elif pcr_oi < 0.9:
            bear_points += 1

        # Max pain vs spot
        if max_pain > 0 and spot > 0:
            diff_pct = ((max_pain - spot) / spot) * 100
            if diff_pct > 0.5:
                bull_points += 1
                reasons.append(f"Max pain above spot ({max_pain:.0f} vs {spot:.0f})")
            elif diff_pct < -0.5:
                bear_points += 1
                reasons.append(f"Max pain below spot ({max_pain:.0f} vs {spot:.0f})")

        # OI walls
        if max_pe_strike > 0 and spot > 0:
            pe_wall_dist = ((spot - max_pe_strike) / spot) * 100
            if pe_wall_dist < 2:
                bull_points += 1
                reasons.append(f"Strong PE OI wall (support) at {max_pe_strike:.0f}")
        if max_ce_strike > 0 and spot > 0:
            ce_wall_dist = ((max_ce_strike - spot) / spot) * 100
            if ce_wall_dist < 2:
                bear_points += 1
                reasons.append(f"Strong CE OI wall (resistance) at {max_ce_strike:.0f}")

        # IV skew
        if iv_skew > 3:
            bear_points += 1
            reasons.append(f"Put IV premium (+{iv_skew:.1f}%) = bearish skew")
        elif iv_skew < -3:
            bull_points += 1
            reasons.append(f"Call IV premium ({iv_skew:.1f}%) = bullish skew")

        # Signal count
        bull_signals = sum(1 for s in signals if s["sentiment"] == "BULLISH")
        bear_signals = sum(1 for s in signals if s["sentiment"] == "BEARISH")
        if bull_signals > bear_signals + 2:
            bull_points += 1
            reasons.append(f"More bullish OI buildup ({bull_signals} vs {bear_signals})")
        elif bear_signals > bull_signals + 2:
            bear_points += 1
            reasons.append(f"More bearish OI buildup ({bear_signals} vs {bull_signals})")

        total = bull_points + bear_points
        if total == 0:
            return "NEUTRAL", ["All OI indicators balanced — no directional bias"]

        if bull_points >= bear_points + 3:
            return "BULLISH", reasons
        elif bear_points >= bull_points + 3:
            return "BEARISH", reasons
        elif bull_points > bear_points:
            return "MILDLY BULLISH", reasons
        elif bear_points > bull_points:
            return "MILDLY BEARISH", reasons
        return "NEUTRAL", reasons

    def _track_history(self, underlying: str, pcr: float, straddle_premium: float) -> None:
        """Store PCR and straddle premium for trend tracking."""
        ts = datetime.now().isoformat()
        key = underlying.upper()

        if key not in self._pcr_history:
            self._pcr_history[key] = []
        self._pcr_history[key].append({"timestamp": ts, "pcr": round(pcr, 3)})
        if len(self._pcr_history[key]) > 100:
            self._pcr_history[key] = self._pcr_history[key][-100:]

        if key not in self._straddle_history:
            self._straddle_history[key] = []
        self._straddle_history[key].append({"timestamp": ts, "premium": round(straddle_premium, 2)})
        if len(self._straddle_history[key]) > 100:
            self._straddle_history[key] = self._straddle_history[key][-100:]
