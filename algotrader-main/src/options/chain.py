from __future__ import annotations

from datetime import datetime, date
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.options.greeks import BlackScholes, compute_all_greeks
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OptionContract(BaseModel):
    instrument_token: int = 0
    tradingsymbol: str = ""
    exchange: str = "NFO"
    underlying: str = ""
    expiry: str = ""
    strike: float = 0.0
    option_type: str = ""
    lot_size: int = 0
    last_price: float = 0.0
    volume: int = 0
    oi: int = 0
    bid_price: float = 0.0
    ask_price: float = 0.0
    iv: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    theoretical_price: float = 0.0


class OptionChainEntry(BaseModel):
    strike: float = 0.0
    ce: Optional[OptionContract] = None
    pe: Optional[OptionContract] = None


class OptionChainData(BaseModel):
    underlying: str = ""
    spot_price: float = 0.0
    expiry: str = ""
    atm_strike: float = 0.0
    atm_iv: float = 0.0
    total_ce_oi: int = 0
    total_pe_oi: int = 0
    pcr: float = 0.0
    max_pain: float = 0.0
    entries: list[OptionChainEntry] = Field(default_factory=list)
    updated_at: str = ""


class OptionChainBuilder:
    def __init__(self, risk_free_rate: float = 0.07) -> None:
        self._risk_free_rate = risk_free_rate
        self._chains: dict[str, OptionChainData] = {}

    def build_chain(
        self,
        underlying: str,
        spot_price: float,
        instruments: list[dict[str, Any]],
        quotes: dict[str, dict[str, Any]],
        expiry_filter: Optional[str] = None,
    ) -> OptionChainData:
        option_instruments = [
            i for i in instruments
            if i.get("instrument_type") in ("CE", "PE")
            and i.get("name", "").upper() == underlying.upper()
            and (not expiry_filter or i.get("expiry") == expiry_filter)
        ]

        if not option_instruments and instruments:
            option_instruments = [
                i for i in instruments
                if i.get("instrument_type") in ("CE", "PE")
                and underlying.upper() in i.get("tradingsymbol", "").upper()
                and (not expiry_filter or i.get("expiry") == expiry_filter)
            ]

        strike_map: dict[float, dict[str, Any]] = {}
        for inst in option_instruments:
            strike = float(inst.get("strike", 0))
            if strike not in strike_map:
                strike_map[strike] = {}
            opt_type = inst.get("instrument_type", "")
            strike_map[strike][opt_type] = inst

        entries: list[OptionChainEntry] = []
        total_ce_oi = 0
        total_pe_oi = 0

        expiry_str = expiry_filter or ""
        if not expiry_str and option_instruments:
            expiries = sorted(set(i.get("expiry", "") for i in option_instruments if i.get("expiry")))
            expiry_str = expiries[0] if expiries else ""

        time_to_expiry = self._calculate_tte(expiry_str)

        for strike in sorted(strike_map.keys()):
            entry = OptionChainEntry(strike=strike)

            for opt_type_key, attr_name in [("CE", "ce"), ("PE", "pe")]:
                inst = strike_map[strike].get(opt_type_key)
                if inst:
                    symbol = f"{inst.get('exchange', 'NFO')}:{inst.get('tradingsymbol', '')}"
                    quote_data = quotes.get(symbol, {})
                    ltp = quote_data.get("last_price", inst.get("last_price", 0))
                    volume = quote_data.get("volume", 0)
                    oi = quote_data.get("oi", 0)

                    iv = 0.0
                    greeks = {}
                    if ltp > 0 and spot_price > 0 and time_to_expiry > 0:
                        iv = BlackScholes.implied_volatility(
                            ltp, spot_price, strike,
                            self._risk_free_rate, time_to_expiry, opt_type_key,
                        )
                        greeks = compute_all_greeks(
                            spot_price, strike,
                            self._risk_free_rate, time_to_expiry, iv, opt_type_key,
                        )

                    contract = OptionContract(
                        instrument_token=inst.get("instrument_token", 0),
                        tradingsymbol=inst.get("tradingsymbol", ""),
                        exchange=inst.get("exchange", "NFO"),
                        underlying=underlying,
                        expiry=expiry_str,
                        strike=strike,
                        option_type=opt_type_key,
                        lot_size=inst.get("lot_size", 0),
                        last_price=ltp,
                        volume=volume,
                        oi=oi,
                        iv=round(iv * 100, 2),
                        delta=round(greeks.get("delta", 0), 4),
                        gamma=round(greeks.get("gamma", 0), 6),
                        theta=round(greeks.get("theta", 0), 4),
                        vega=round(greeks.get("vega", 0), 4),
                        theoretical_price=round(greeks.get("theoretical_price", 0), 2),
                    )
                    setattr(entry, attr_name, contract)

                    if opt_type_key == "CE":
                        total_ce_oi += oi
                    else:
                        total_pe_oi += oi

            entries.append(entry)

        atm_strike = self._find_atm_strike(spot_price, [e.strike for e in entries])
        atm_iv = 0.0
        for e in entries:
            if e.strike == atm_strike:
                if e.ce:
                    atm_iv = e.ce.iv
                break

        pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0.0
        max_pain = self._calculate_max_pain(entries, spot_price)

        chain = OptionChainData(
            underlying=underlying,
            spot_price=spot_price,
            expiry=expiry_str,
            atm_strike=atm_strike,
            atm_iv=atm_iv,
            total_ce_oi=total_ce_oi,
            total_pe_oi=total_pe_oi,
            pcr=round(pcr, 2),
            max_pain=max_pain,
            entries=entries,
            updated_at=datetime.now().isoformat(),
        )

        cache_key = f"{underlying}:{expiry_str}"
        self._chains[cache_key] = chain
        return chain

    def get_cached_chain(self, underlying: str, expiry: str = "") -> Optional[OptionChainData]:
        key = f"{underlying}:{expiry}"
        return self._chains.get(key)

    def _find_atm_strike(self, spot: float, strikes: list[float]) -> float:
        if not strikes:
            return 0.0
        return min(strikes, key=lambda s: abs(s - spot))

    def _calculate_tte(self, expiry_str: str) -> float:
        if not expiry_str:
            return 0.0
        try:
            for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d%b%Y"):
                try:
                    exp_date = datetime.strptime(expiry_str, fmt).date()
                    days = (exp_date - date.today()).days
                    return max(days / 365.0, 1 / 365.0)
                except ValueError:
                    continue
            return 7 / 365.0
        except Exception:
            return 7 / 365.0

    def _calculate_max_pain(self, entries: list[OptionChainEntry], spot: float) -> float:
        if not entries:
            return 0.0

        strikes = [e.strike for e in entries]
        min_pain = float("inf")
        max_pain_strike = strikes[0] if strikes else 0.0

        for test_strike in strikes:
            total_pain = 0.0
            for entry in entries:
                if entry.ce and entry.ce.oi > 0:
                    intrinsic = max(test_strike - entry.strike, 0)
                    total_pain += intrinsic * entry.ce.oi
                if entry.pe and entry.pe.oi > 0:
                    intrinsic = max(entry.strike - test_strike, 0)
                    total_pain += intrinsic * entry.pe.oi
            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = test_strike

        return max_pain_strike
