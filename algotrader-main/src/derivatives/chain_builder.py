"""
Historical Option Chain Builder — Reconstructs option chains for backtesting.

When historical IV data is unavailable, uses:
• Black-Scholes IV approximation from underlying HV
• Historical volatility proxy with term-structure model
• Skew model for OTM options
• Bid-ask spread simulation based on moneyness + DTE
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.derivatives.contracts import DerivativeContract, InstrumentType, SettlementType
from src.options.greeks import BlackScholes, compute_all_greeks

# ────────────────────────────────────────────────────────────
# NSE F&O Constants
# ────────────────────────────────────────────────────────────

NIFTY_LOT = 65
BANKNIFTY_LOT = 15
FINNIFTY_LOT = 25
SENSEX_LOT = 20

INDEX_CONFIGS: dict[str, dict[str, Any]] = {
    "NIFTY": {"lot_size": NIFTY_LOT, "strike_step": 50, "freeze_qty": 1800, "settlement": "CASH", "exchange": "NFO", "expiry_day": 1},
    "BANKNIFTY": {"lot_size": BANKNIFTY_LOT, "strike_step": 100, "freeze_qty": 900, "settlement": "CASH", "exchange": "NFO", "expiry_day": 1},
    "FINNIFTY": {"lot_size": FINNIFTY_LOT, "strike_step": 50, "freeze_qty": 1800, "settlement": "CASH", "exchange": "NFO", "expiry_day": 1},
    "SENSEX": {"lot_size": SENSEX_LOT, "strike_step": 100, "freeze_qty": 1000, "settlement": "CASH", "exchange": "BFO", "expiry_day": 3},
    "MIDCPNIFTY": {"lot_size": 50, "strike_step": 25, "freeze_qty": 2800, "settlement": "CASH", "exchange": "NFO", "expiry_day": 1},
}

STOCK_LOT_SIZES: dict[str, int] = {
    "RELIANCE": 250, "TCS": 150, "HDFCBANK": 550, "INFY": 300,
    "ICICIBANK": 700, "SBIN": 750, "BHARTIARTL": 475, "ITC": 1600,
    "KOTAKBANK": 400, "LT": 150, "AXISBANK": 600, "HINDUNILVR": 300,
    "TATAMOTORS": 700, "WIPRO": 1500, "MARUTI": 100,
    "BAJFINANCE": 125, "TATASTEEL": 550, "SUNPHARMA": 350,
    "HCLTECH": 350, "ADANIENT": 250, "M&M": 350, "NTPC": 2875,
    "POWERGRID": 2700, "ULTRACEMCO": 100, "TITAN": 375,
    "ASIANPAINT": 300, "NESTLEIND": 50, "JSWSTEEL": 675,
    "ONGC": 3850, "COALINDIA": 2100, "TECHM": 600,
}

RISK_FREE_RATE = 0.065  # India ~6.5% (10Y Gsec yield)


@dataclass
class SyntheticOptionQuote:
    """A synthetic option quote for historical backtesting."""
    strike: float
    option_type: str          # "CE" or "PE"
    expiry: date
    price: float              # theoretical mid price
    bid: float
    ask: float
    iv: float
    delta: float
    gamma: float
    theta: float
    vega: float
    oi: int = 0               # simulated OI
    volume: int = 0           # simulated volume
    intrinsic: float = 0.0
    time_value: float = 0.0
    moneyness: str = ""       # ITM / ATM / OTM

    def to_dict(self) -> dict[str, Any]:
        return {
            "strike": self.strike,
            "option_type": self.option_type,
            "expiry": str(self.expiry),
            "price": round(self.price, 2),
            "bid": round(self.bid, 2),
            "ask": round(self.ask, 2),
            "iv": round(self.iv, 4),
            "delta": round(self.delta, 4),
            "gamma": round(self.gamma, 6),
            "theta": round(self.theta, 4),
            "vega": round(self.vega, 4),
            "oi": self.oi,
            "volume": self.volume,
            "intrinsic": round(self.intrinsic, 2),
            "time_value": round(self.time_value, 2),
            "moneyness": self.moneyness,
        }


@dataclass
class SyntheticChain:
    """A complete synthetic option chain at a point in time."""
    underlying: str
    spot_price: float
    timestamp: datetime
    expiry: date
    dte: int
    atm_strike: float
    atm_iv: float
    risk_free_rate: float
    hv_20: float                 # 20-day historical volatility
    strikes: dict[float, dict[str, SyntheticOptionQuote]] = field(default_factory=dict)
    # strikes[strike] = {"CE": quote, "PE": quote}

    @property
    def total_ce_oi(self) -> int:
        return sum(
            s["CE"].oi for s in self.strikes.values() if "CE" in s
        )

    @property
    def total_pe_oi(self) -> int:
        return sum(
            s["PE"].oi for s in self.strikes.values() if "PE" in s
        )

    @property
    def pcr_oi(self) -> float:
        ce = self.total_ce_oi
        return self.total_pe_oi / ce if ce > 0 else 0.0


class HistoricalChainBuilder:
    """Rebuild historical option chains from underlying OHLCV data.

    This is CRITICAL for F&O backtesting — without it, option strategy
    backtests are meaningless.

    Uses:
    • Historical volatility (HV) as base IV
    • Skew model: OTM puts get higher IV, OTM calls get lower
    • Term structure: longer DTE → slightly higher IV
    • Smile: far OTM options get higher IV
    • Bid-ask spread: wider for OTM, tighter for ATM
    """

    def __init__(
        self,
        underlying: str = "NIFTY",
        risk_free_rate: float = RISK_FREE_RATE,
        num_strikes: int = 20,  # strikes above + below ATM
    ) -> None:
        self.underlying = underlying
        self.risk_free_rate = risk_free_rate
        self.num_strikes = num_strikes

        config = INDEX_CONFIGS.get(underlying, {})
        self.lot_size = config.get("lot_size", STOCK_LOT_SIZES.get(underlying, 1))
        self.strike_step = config.get("strike_step", self._auto_strike_step(underlying))
        self.freeze_qty = config.get("freeze_qty", 900)
        self.settlement = SettlementType.CASH if config.get("settlement") == "CASH" else SettlementType.PHYSICAL
        self.exchange = config.get("exchange", "NFO")
        # expiry_day: 1=Tuesday (NSE indices), 3=Thursday (BSE SENSEX)
        self.expiry_day = config.get("expiry_day", 1)

        self._hv_cache: dict[str, float] = {}

    @staticmethod
    def _auto_strike_step(symbol: str) -> float:
        """Guess strike step for stock F&O."""
        # Rough heuristic: step ≈ 2.5% of typical price
        return 5.0  # default for stocks

    # ────────────────────────────────────────────────────────
    # Expiry Calendar
    # ────────────────────────────────────────────────────────

    @staticmethod
    def get_expiry_dates(
        from_date: date, to_date: date, weekly: bool = True,
        expiry_weekday: int = 1,
    ) -> list[date]:
        """Generate expiry dates.

        Args:
            expiry_weekday: 1=Tuesday (NSE), 3=Thursday (BSE SENSEX).
        """
        expiries: list[date] = []
        current = from_date
        while current <= to_date:
            days_to_exp = (expiry_weekday - current.weekday()) % 7
            exp_day = current + timedelta(days=days_to_exp)
            if exp_day >= from_date and exp_day <= to_date and exp_day not in expiries:
                expiries.append(exp_day)
            current += timedelta(days=7) if weekly else timedelta(days=28)
        return sorted(set(expiries))

    @staticmethod
    def nearest_expiry(ref_date: date, expiries: list[date]) -> date | None:
        """Find nearest future expiry."""
        future = [e for e in expiries if e >= ref_date]
        return min(future) if future else None

    # ────────────────────────────────────────────────────────
    # Historical Volatility
    # ────────────────────────────────────────────────────────

    @staticmethod
    def compute_hv(closes: np.ndarray, window: int = 20) -> np.ndarray:
        """Rolling historical volatility (annualized)."""
        if len(closes) < window + 1:
            return np.full(len(closes), 0.20)  # fallback 20%
        log_returns = np.diff(np.log(closes))
        hv = pd.Series(log_returns).rolling(window).std() * np.sqrt(252)
        # Pad with first valid value
        result = hv.fillna(method="bfill").fillna(0.20).values
        return np.concatenate([[result[0]], result])

    # ────────────────────────────────────────────────────────
    # IV Model (Skew + Smile + Term Structure)
    # ────────────────────────────────────────────────────────

    @staticmethod
    def model_iv(
        atm_iv: float,
        strike: float,
        spot: float,
        dte: int,
        option_type: str,
    ) -> float:
        """Model IV with skew, smile, and term structure.

        Professional desks calibrate these from market data.
        This is a realistic approximation for backtesting.
        """
        moneyness = (strike - spot) / spot  # positive = OTM call / ITM put

        # 1. Skew: OTM puts get higher IV (put skew / fear premium)
        skew = 0.0
        if option_type == "PE":
            skew = -moneyness * 0.15  # puts: lower strike → higher IV
        else:
            skew = moneyness * 0.05   # calls: higher strike → slightly higher IV

        # 2. Smile: far OTM options get higher IV (volatility smile)
        smile = abs(moneyness) ** 2 * 0.3

        # 3. Term structure: longer DTE → slightly higher IV
        term = max(0, (dte - 7) * 0.0005)

        iv = atm_iv + skew + smile + term
        return max(0.03, min(iv, 2.0))  # clamp

    # ────────────────────────────────────────────────────────
    # Bid-Ask Spread Model
    # ────────────────────────────────────────────────────────

    @staticmethod
    def model_spread(
        mid_price: float,
        moneyness_abs: float,
        dte: int,
    ) -> tuple[float, float]:
        """Model bid-ask spread based on liquidity factors.

        ATM → tight spread (~0.5-1%)
        OTM → wider spread (~2-5%)
        Near expiry → tighter for ATM, wider for OTM
        """
        base_spread_pct = 0.01  # 1% base

        # Moneyness factor: OTM wider
        moneyness_factor = 1.0 + moneyness_abs * 5.0

        # DTE factor: near expiry, OTM spreads widen
        dte_factor = 1.0 + max(0, (7 - dte)) * 0.1 * moneyness_abs

        spread_pct = base_spread_pct * moneyness_factor * dte_factor
        half_spread = mid_price * spread_pct / 2

        bid = max(0.05, mid_price - half_spread)
        ask = mid_price + half_spread
        return round(bid, 2), round(ask, 2)

    # ────────────────────────────────────────────────────────
    # OI Simulation
    # ────────────────────────────────────────────────────────

    @staticmethod
    def simulate_oi(
        spot: float, strike: float, option_type: str, dte: int, base_oi: int = 50000,
    ) -> tuple[int, int]:
        """Simulate OI and volume. ATM gets highest OI."""
        distance = abs(strike - spot) / spot
        oi_factor = max(0.05, 1.0 - distance * 8.0)
        dte_factor = max(0.3, min(1.0, dte / 30.0))
        oi = int(base_oi * oi_factor * dte_factor)
        volume = int(oi * 0.3 * max(0.1, 1.0 - distance * 4.0))
        return max(100, oi), max(10, volume)

    # ────────────────────────────────────────────────────────
    # Build Chain at a Point in Time
    # ────────────────────────────────────────────────────────

    def build_chain(
        self,
        spot: float,
        timestamp: datetime,
        expiry: date,
        hv: float = 0.20,
    ) -> SyntheticChain:
        """Build a complete synthetic option chain at a specific timestamp.

        Args:
            spot: Current underlying price
            timestamp: Current bar timestamp
            expiry: Target expiry date
            hv: Historical volatility (annualized) at this point
        """
        dte = max(1, (expiry - timestamp.date() if isinstance(timestamp, datetime) else expiry - timestamp).days)
        tte = max(dte / 365.0, 1 / 365.0)

        # ATM strike
        atm_strike = round(spot / self.strike_step) * self.strike_step

        # ATM IV ≈ HV with a premium
        atm_iv = max(0.08, hv * 1.05 + 0.02)  # slight premium over HV

        # Generate strikes
        strikes_range = range(
            int(atm_strike - self.num_strikes * self.strike_step),
            int(atm_strike + (self.num_strikes + 1) * self.strike_step),
            int(self.strike_step),
        )

        chain_strikes: dict[float, dict[str, SyntheticOptionQuote]] = {}

        for strike in strikes_range:
            if strike <= 0:
                continue

            for opt_type in ("CE", "PE"):
                iv = self.model_iv(atm_iv, strike, spot, dte, opt_type)

                # Price via Black-Scholes
                if opt_type == "CE":
                    price = BlackScholes.call_price(spot, strike, self.risk_free_rate, tte, iv)
                else:
                    price = BlackScholes.put_price(spot, strike, self.risk_free_rate, tte, iv)

                price = max(0.05, price)

                # Greeks
                greeks = compute_all_greeks(spot, strike, self.risk_free_rate, tte, iv, opt_type)

                # Bid-ask
                moneyness_abs = abs(strike - spot) / spot
                bid, ask = self.model_spread(price, moneyness_abs, dte)

                # OI
                oi, volume = self.simulate_oi(spot, strike, opt_type, dte)

                # Intrinsic + Time value
                if opt_type == "CE":
                    intrinsic = max(0, spot - strike)
                else:
                    intrinsic = max(0, strike - spot)
                time_val = max(0, price - intrinsic)

                # Moneyness
                if opt_type == "CE":
                    m = "ITM" if spot > strike * 1.005 else ("OTM" if spot < strike * 0.995 else "ATM")
                else:
                    m = "ITM" if spot < strike * 0.995 else ("OTM" if spot > strike * 1.005 else "ATM")

                quote = SyntheticOptionQuote(
                    strike=float(strike),
                    option_type=opt_type,
                    expiry=expiry,
                    price=round(price, 2),
                    bid=bid,
                    ask=ask,
                    iv=iv,
                    delta=greeks.get("delta", 0),
                    gamma=greeks.get("gamma", 0),
                    theta=greeks.get("theta", 0),
                    vega=greeks.get("vega", 0),
                    oi=oi,
                    volume=volume,
                    intrinsic=round(intrinsic, 2),
                    time_value=round(time_val, 2),
                    moneyness=m,
                )

                if float(strike) not in chain_strikes:
                    chain_strikes[float(strike)] = {}
                chain_strikes[float(strike)][opt_type] = quote

        return SyntheticChain(
            underlying=self.underlying,
            spot_price=spot,
            timestamp=timestamp,
            expiry=expiry,
            dte=dte,
            atm_strike=float(atm_strike),
            atm_iv=round(atm_iv, 4),
            risk_free_rate=self.risk_free_rate,
            hv_20=round(hv, 4),
            strikes=chain_strikes,
        )

    def build_chains_from_df(
        self,
        df: pd.DataFrame,
        expiry: date | None = None,
    ) -> list[SyntheticChain]:
        """Build chains for every bar in a DataFrame.

        If expiry not specified, auto-selects nearest weekly expiry.
        """
        if df.empty:
            return []

        closes = df["close"].values
        hvs = self.compute_hv(closes, window=20)

        # Generate expiry calendar
        start_date = df.index[0] if isinstance(df.index[0], (date, datetime)) else date.today() - timedelta(days=len(df))
        end_date = df.index[-1] if isinstance(df.index[-1], (date, datetime)) else date.today()
        if isinstance(start_date, datetime):
            start_date = start_date.date()
        if isinstance(end_date, datetime):
            end_date = end_date.date()

        expiries = self.get_expiry_dates(start_date, end_date + timedelta(days=60))

        chains: list[SyntheticChain] = []
        for i in range(len(df)):
            spot = closes[i]
            hv = hvs[i] if i < len(hvs) else 0.20
            ts = df.index[i] if isinstance(df.index[i], datetime) else datetime.now()
            ts_date = ts.date() if isinstance(ts, datetime) else ts

            if expiry:
                exp = expiry
            else:
                exp = self.nearest_expiry(ts_date, expiries)
                if not exp:
                    exp = ts_date + timedelta(days=7)

            chain = self.build_chain(spot, ts, exp, hv)
            chains.append(chain)

        return chains

    def find_strike_by_delta(
        self,
        chain: SyntheticChain,
        target_delta: float,
        option_type: str,
    ) -> SyntheticOptionQuote | None:
        """Find strike nearest to target delta."""
        best: SyntheticOptionQuote | None = None
        best_diff = float("inf")

        for strike_data in chain.strikes.values():
            q = strike_data.get(option_type)
            if not q:
                continue
            diff = abs(abs(q.delta) - abs(target_delta))
            if diff < best_diff:
                best_diff = diff
                best = q

        return best

    def find_atm_strike(self, chain: SyntheticChain) -> float:
        return chain.atm_strike

    def make_contract(
        self, quote: SyntheticOptionQuote, chain: SyntheticChain
    ) -> DerivativeContract:
        """Create a DerivativeContract from a synthetic quote."""
        inst_type = InstrumentType.CE if quote.option_type == "CE" else InstrumentType.PE
        # NSE format: NIFTY24FEB26 25600 CE
        tsym = f"{chain.underlying}{chain.expiry.strftime('%d%b%y').upper()}{int(quote.strike)}{quote.option_type}"
        return DerivativeContract(
            symbol=chain.underlying,
            tradingsymbol=tsym,
            instrument_type=inst_type,
            exchange=self.exchange,
            strike=quote.strike,
            expiry=chain.expiry,
            lot_size=self.lot_size,
            tick_size=0.05,
            underlying=chain.underlying,
            freeze_quantity=self.freeze_qty,
            settlement=self.settlement,
        )
