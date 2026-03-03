"""
Intraday OI Flow Analyzer — Market Direction from 15-min Option Chain Snapshots
=================================================================================

Analyses how NIFTY/SENSEX option chain data changes across intraday snapshots
captured every 15 minutes.  Focuses on ATM and near-ATM strikes to derive:

  • PCR trend & velocity (is PCR rising or falling?)
  • OI wall shifts (are support/resistance levels moving?)
  • Straddle premium decay / expansion
  • Smart-money positioning (where is OI concentrating?)
  • IV skew evolution
  • Max-pain drift
  • Aggressive writing vs unwinding patterns

Combines all signals into a single market direction verdict with confidence
and "trend reinforcement" scoring — multiple consecutive snapshots confirming
the same direction indicate a stronger intraday trend.

Integration point:  added to OptionsOIAnalyzer and exposed via the OI API endpoints.
"""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from datetime import datetime, date, timedelta, timezone, time
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.data.fno_data_store import FnODataStore, get_fno_data_store

logger = logging.getLogger("oi_intraday_analyzer")

IST = timezone(timedelta(hours=5, minutes=30))

NIFTY_STRIKE_GAP = 50
SENSEX_STRIKE_GAP = 100
NEAR_ATM_STRIKES = 5  # ±5 strikes from ATM


# ─────────────────────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────────────────────

class SnapshotSummary(BaseModel):
    """Summary of a single 15-min snapshot."""
    timestamp: str = ""
    spot_price: float = 0.0
    atm_strike: float = 0.0
    total_ce_oi: int = 0
    total_pe_oi: int = 0
    pcr_oi: float = 0.0
    total_ce_volume: int = 0
    total_pe_volume: int = 0
    pcr_volume: float = 0.0
    max_ce_oi_strike: float = 0.0
    max_pe_oi_strike: float = 0.0
    max_pain: float = 0.0
    atm_straddle_premium: float = 0.0
    avg_ce_iv: float = 0.0
    avg_pe_iv: float = 0.0
    iv_skew: float = 0.0


class OIFlowChange(BaseModel):
    """Change between two consecutive snapshots."""
    from_ts: str = ""
    to_ts: str = ""
    duration_minutes: int = 0
    spot_change: float = 0.0
    spot_change_pct: float = 0.0
    pcr_change: float = 0.0
    straddle_premium_change: float = 0.0
    straddle_premium_change_pct: float = 0.0
    iv_skew_change: float = 0.0
    max_pain_shift: float = 0.0
    ce_wall_shift: float = 0.0
    pe_wall_shift: float = 0.0
    net_ce_oi_change: int = 0
    net_pe_oi_change: int = 0
    # Per-strike significant changes
    top_ce_oi_additions: list[dict[str, Any]] = Field(default_factory=list)
    top_pe_oi_additions: list[dict[str, Any]] = Field(default_factory=list)
    top_ce_oi_exits: list[dict[str, Any]] = Field(default_factory=list)
    top_pe_oi_exits: list[dict[str, Any]] = Field(default_factory=list)
    # Interval signal
    signal: str = ""  # BULLISH | BEARISH | NEUTRAL
    signal_reasons: list[str] = Field(default_factory=list)


class TrendReinforcement(BaseModel):
    """Measures how consistently signals confirm the same direction."""
    consecutive_bullish: int = 0
    consecutive_bearish: int = 0
    consecutive_neutral: int = 0
    trend_strength: str = ""  # STRONG_BULLISH | MODERATE_BULLISH | WEAK | MODERATE_BEARISH | STRONG_BEARISH
    trend_score: float = 0.0  # -100 (max bearish) to +100 (max bullish)
    reinforcement_count: int = 0  # How many consecutive intervals confirm the trend


class IntradayOIAnalysis(BaseModel):
    """Complete intraday OI flow analysis report."""
    underlying: str = ""
    analysis_date: str = ""
    analysis_timestamp: str = ""
    total_snapshots: int = 0
    snapshot_timestamps: list[str] = Field(default_factory=list)
    # Snapshot summaries
    snapshots: list[SnapshotSummary] = Field(default_factory=list)
    # Change analysis
    flow_changes: list[OIFlowChange] = Field(default_factory=list)
    # Trend metrics
    pcr_trend: dict[str, Any] = Field(default_factory=dict)
    straddle_trend: dict[str, Any] = Field(default_factory=dict)
    iv_skew_trend: dict[str, Any] = Field(default_factory=dict)
    max_pain_drift: dict[str, Any] = Field(default_factory=dict)
    oi_wall_analysis: dict[str, Any] = Field(default_factory=dict)
    # Smart money activity
    smart_money_signals: list[dict[str, Any]] = Field(default_factory=list)
    # Overall verdict
    market_direction: str = ""  # BULLISH | BEARISH | NEUTRAL | MILDLY_BULLISH | MILDLY_BEARISH
    direction_confidence: int = 0  # 0-100
    direction_reasons: list[str] = Field(default_factory=list)
    # Trend reinforcement
    trend_reinforcement: Optional[TrendReinforcement] = None
    # Actionable summary
    summary: str = ""


# ─────────────────────────────────────────────────────────────
# Intraday OI Flow Analyzer
# ─────────────────────────────────────────────────────────────

class IntradayOIFlowAnalyzer:
    """
    Analyses intraday option chain snapshots (captured every 15 minutes)
    to detect market direction and trading signals.

    Reads from FnODataStore (SQLite) and processes changes between
    consecutive snapshots for ATM and near-ATM strikes.
    """

    def __init__(self, store: Optional[FnODataStore] = None) -> None:
        self._store = store or get_fno_data_store()

    # ─── Public API ─────────────────────────────────────────────

    def analyze(
        self,
        underlying: str = "NIFTY",
        analysis_date: Optional[str] = None,
    ) -> IntradayOIAnalysis:
        """
        Run full intraday OI flow analysis for a given underlying and date.
        Defaults to today if no date is provided.
        """
        underlying = underlying.upper()
        target_date = analysis_date or date.today().isoformat()

        # 1 — Get all distinct timestamps for the day
        timestamps = self._store.get_distinct_timestamps(underlying, target_date, target_date)
        if not timestamps:
            return IntradayOIAnalysis(
                underlying=underlying,
                analysis_date=target_date,
                analysis_timestamp=datetime.now(IST).isoformat(),
                summary="No snapshot data available for this date.",
            )

        # 2 — Load snapshot data for each timestamp
        strike_gap = NIFTY_STRIKE_GAP if underlying in ("NIFTY", "NIFTY 50") else SENSEX_STRIKE_GAP
        summaries: list[SnapshotSummary] = []

        for ts in timestamps:
            raw = self._store.get_option_chain_at_timestamp(underlying, ts)
            if not raw:
                continue
            summary = self._build_snapshot_summary(raw, strike_gap)
            summaries.append(summary)

        if not summaries:
            return IntradayOIAnalysis(
                underlying=underlying,
                analysis_date=target_date,
                analysis_timestamp=datetime.now(IST).isoformat(),
                summary="No valid snapshot data found.",
            )

        # 3 — Compute flow changes between consecutive snapshots
        flow_changes: list[OIFlowChange] = []
        for i in range(1, len(summaries)):
            prev_ts = timestamps[i - 1]
            curr_ts = timestamps[i]
            prev_raw = self._store.get_option_chain_at_timestamp(underlying, prev_ts)
            curr_raw = self._store.get_option_chain_at_timestamp(underlying, curr_ts)
            change = self._compute_flow_change(
                summaries[i - 1], summaries[i],
                prev_raw, curr_raw, strike_gap,
            )
            flow_changes.append(change)

        # 4 — Trend analysis
        pcr_trend = self._analyze_pcr_trend(summaries)
        straddle_trend = self._analyze_straddle_trend(summaries)
        iv_skew_trend = self._analyze_iv_skew_trend(summaries)
        max_pain_drift = self._analyze_max_pain_drift(summaries)
        oi_wall_analysis = self._analyze_oi_walls(summaries)

        # 5 — Smart money signals
        smart_money = self._detect_smart_money_signals(summaries, flow_changes)

        # 6 — Overall direction
        direction, confidence, reasons = self._compute_market_direction(
            pcr_trend, straddle_trend, iv_skew_trend,
            max_pain_drift, oi_wall_analysis, smart_money,
            flow_changes, summaries,
        )

        # 7 — Trend reinforcement
        reinforcement = self._compute_trend_reinforcement(flow_changes)

        # Boost confidence if reinforcement is strong
        if reinforcement.reinforcement_count >= 3:
            confidence = min(confidence + 10, 95)

        # 8 — Generate summary
        summary_text = self._generate_summary(
            underlying, summaries, direction, confidence,
            reasons, reinforcement, pcr_trend, straddle_trend,
        )

        return IntradayOIAnalysis(
            underlying=underlying,
            analysis_date=target_date,
            analysis_timestamp=datetime.now(IST).isoformat(),
            total_snapshots=len(summaries),
            snapshot_timestamps=timestamps,
            snapshots=summaries,
            flow_changes=flow_changes,
            pcr_trend=pcr_trend,
            straddle_trend=straddle_trend,
            iv_skew_trend=iv_skew_trend,
            max_pain_drift=max_pain_drift,
            oi_wall_analysis=oi_wall_analysis,
            smart_money_signals=smart_money,
            market_direction=direction,
            direction_confidence=confidence,
            direction_reasons=reasons,
            trend_reinforcement=reinforcement,
            summary=summary_text,
        )

    # ─── Snapshot Summary Builder ───────────────────────────────

    def _build_snapshot_summary(
        self,
        raw_data: list[dict[str, Any]],
        strike_gap: float,
    ) -> SnapshotSummary:
        """Build a summary from raw option chain rows for one timestamp."""
        if not raw_data:
            return SnapshotSummary()

        ts = raw_data[0].get("ts", "")
        spot = raw_data[0].get("spot_price", 0.0)
        atm_strike = round(spot / strike_gap) * strike_gap
        low_bound = atm_strike - strike_gap * NEAR_ATM_STRIKES
        high_bound = atm_strike + strike_gap * NEAR_ATM_STRIKES

        # Filter near-ATM
        near_atm = [r for r in raw_data if low_bound <= r.get("strike", 0) <= high_bound]

        total_ce_oi = total_pe_oi = 0
        total_ce_vol = total_pe_vol = 0
        max_ce_oi = max_pe_oi = 0
        max_ce_strike = max_pe_strike = atm_strike
        ce_ivs: list[float] = []
        pe_ivs: list[float] = []
        atm_ce_ltp = atm_pe_ltp = 0.0

        # Build per-strike map
        strike_map: dict[float, dict[str, Any]] = defaultdict(lambda: {
            "ce_oi": 0, "pe_oi": 0, "ce_ltp": 0.0, "pe_ltp": 0.0,
            "ce_vol": 0, "pe_vol": 0, "ce_iv": 0.0, "pe_iv": 0.0,
        })

        for row in near_atm:
            strike = row.get("strike", 0)
            otype = row.get("option_type", "")
            oi = row.get("oi", 0)
            vol = row.get("volume", 0)
            ltp = row.get("last_price", 0.0)
            iv = row.get("iv", 0.0)

            prefix = "ce" if otype == "CE" else "pe"
            d = strike_map[strike]
            d[f"{prefix}_oi"] = oi
            d[f"{prefix}_vol"] = vol
            d[f"{prefix}_ltp"] = ltp
            d[f"{prefix}_iv"] = iv

        for strike_val, d in strike_map.items():
            total_ce_oi += d["ce_oi"]
            total_pe_oi += d["pe_oi"]
            total_ce_vol += d["ce_vol"]
            total_pe_vol += d["pe_vol"]

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

            if strike_val == atm_strike:
                atm_ce_ltp = d["ce_ltp"]
                atm_pe_ltp = d["pe_ltp"]

        pcr_oi = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0.0
        pcr_vol = total_pe_vol / total_ce_vol if total_ce_vol > 0 else 0.0
        avg_ce_iv = statistics.mean(ce_ivs) if ce_ivs else 0.0
        avg_pe_iv = statistics.mean(pe_ivs) if pe_ivs else 0.0

        # Max pain
        max_pain = self._compute_max_pain_from_map(strike_map)

        return SnapshotSummary(
            timestamp=ts,
            spot_price=round(spot, 2),
            atm_strike=atm_strike,
            total_ce_oi=total_ce_oi,
            total_pe_oi=total_pe_oi,
            pcr_oi=round(pcr_oi, 3),
            total_ce_volume=total_ce_vol,
            total_pe_volume=total_pe_vol,
            pcr_volume=round(pcr_vol, 3),
            max_ce_oi_strike=max_ce_strike,
            max_pe_oi_strike=max_pe_strike,
            max_pain=max_pain,
            atm_straddle_premium=round(atm_ce_ltp + atm_pe_ltp, 2),
            avg_ce_iv=round(avg_ce_iv, 2),
            avg_pe_iv=round(avg_pe_iv, 2),
            iv_skew=round(avg_pe_iv - avg_ce_iv, 2),
        )

    # ─── Flow Change Computation ────────────────────────────────

    def _compute_flow_change(
        self,
        prev_sum: SnapshotSummary,
        curr_sum: SnapshotSummary,
        prev_raw: list[dict[str, Any]],
        curr_raw: list[dict[str, Any]],
        strike_gap: float,
    ) -> OIFlowChange:
        """Compute changes between two consecutive snapshots."""
        # Parse timestamps to compute duration
        try:
            t1 = datetime.fromisoformat(prev_sum.timestamp)
            t2 = datetime.fromisoformat(curr_sum.timestamp)
            duration = int((t2 - t1).total_seconds() / 60)
        except Exception:
            duration = 15

        spot_change = curr_sum.spot_price - prev_sum.spot_price
        spot_change_pct = (
            (spot_change / prev_sum.spot_price * 100) if prev_sum.spot_price > 0 else 0.0
        )

        # Build per-strike OI maps for both snapshots
        prev_oi_map = self._build_oi_map(prev_raw, strike_gap, curr_sum.atm_strike)
        curr_oi_map = self._build_oi_map(curr_raw, strike_gap, curr_sum.atm_strike)

        # Per-strike OI changes
        all_strikes = set(prev_oi_map.keys()) | set(curr_oi_map.keys())
        ce_additions: list[dict[str, Any]] = []
        pe_additions: list[dict[str, Any]] = []
        ce_exits: list[dict[str, Any]] = []
        pe_exits: list[dict[str, Any]] = []
        net_ce_oi_change = 0
        net_pe_oi_change = 0

        for strike in all_strikes:
            prev_d = prev_oi_map.get(strike, {"ce_oi": 0, "pe_oi": 0})
            curr_d = curr_oi_map.get(strike, {"ce_oi": 0, "pe_oi": 0})

            ce_diff = curr_d["ce_oi"] - prev_d["ce_oi"]
            pe_diff = curr_d["pe_oi"] - prev_d["pe_oi"]
            net_ce_oi_change += ce_diff
            net_pe_oi_change += pe_diff

            if ce_diff > 0:
                ce_additions.append({"strike": strike, "oi_change": ce_diff, "new_oi": curr_d["ce_oi"]})
            elif ce_diff < 0:
                ce_exits.append({"strike": strike, "oi_change": ce_diff, "remaining_oi": curr_d["ce_oi"]})

            if pe_diff > 0:
                pe_additions.append({"strike": strike, "oi_change": pe_diff, "new_oi": curr_d["pe_oi"]})
            elif pe_diff < 0:
                pe_exits.append({"strike": strike, "oi_change": pe_diff, "remaining_oi": curr_d["pe_oi"]})

        # Sort by magnitude
        ce_additions.sort(key=lambda x: x["oi_change"], reverse=True)
        pe_additions.sort(key=lambda x: x["oi_change"], reverse=True)
        ce_exits.sort(key=lambda x: x["oi_change"])
        pe_exits.sort(key=lambda x: x["oi_change"])

        # Interval signal
        signal, signal_reasons = self._compute_interval_signal(
            prev_sum, curr_sum, spot_change_pct,
            net_ce_oi_change, net_pe_oi_change,
            ce_additions, pe_additions, ce_exits, pe_exits,
        )

        return OIFlowChange(
            from_ts=prev_sum.timestamp,
            to_ts=curr_sum.timestamp,
            duration_minutes=duration,
            spot_change=round(spot_change, 2),
            spot_change_pct=round(spot_change_pct, 3),
            pcr_change=round(curr_sum.pcr_oi - prev_sum.pcr_oi, 3),
            straddle_premium_change=round(
                curr_sum.atm_straddle_premium - prev_sum.atm_straddle_premium, 2
            ),
            straddle_premium_change_pct=round(
                ((curr_sum.atm_straddle_premium - prev_sum.atm_straddle_premium) /
                 prev_sum.atm_straddle_premium * 100)
                if prev_sum.atm_straddle_premium > 0 else 0.0, 2
            ),
            iv_skew_change=round(curr_sum.iv_skew - prev_sum.iv_skew, 2),
            max_pain_shift=curr_sum.max_pain - prev_sum.max_pain,
            ce_wall_shift=curr_sum.max_ce_oi_strike - prev_sum.max_ce_oi_strike,
            pe_wall_shift=curr_sum.max_pe_oi_strike - prev_sum.max_pe_oi_strike,
            net_ce_oi_change=net_ce_oi_change,
            net_pe_oi_change=net_pe_oi_change,
            top_ce_oi_additions=ce_additions[:5],
            top_pe_oi_additions=pe_additions[:5],
            top_ce_oi_exits=ce_exits[:5],
            top_pe_oi_exits=pe_exits[:5],
            signal=signal,
            signal_reasons=signal_reasons,
        )

    # ─── OI Map Builder ────────────────────────────────────────

    @staticmethod
    def _build_oi_map(
        raw: list[dict[str, Any]],
        strike_gap: float,
        atm_strike: float,
    ) -> dict[float, dict[str, int]]:
        """Build per-strike OI map from raw data, filtered to near-ATM."""
        low = atm_strike - strike_gap * NEAR_ATM_STRIKES
        high = atm_strike + strike_gap * NEAR_ATM_STRIKES
        result: dict[float, dict[str, int]] = {}

        for row in raw:
            strike = row.get("strike", 0)
            if strike < low or strike > high:
                continue
            otype = row.get("option_type", "")
            oi = row.get("oi", 0)
            if strike not in result:
                result[strike] = {"ce_oi": 0, "pe_oi": 0}
            if otype == "CE":
                result[strike]["ce_oi"] = oi
            elif otype == "PE":
                result[strike]["pe_oi"] = oi

        return result

    # ─── Interval Signal ────────────────────────────────────────

    @staticmethod
    def _compute_interval_signal(
        prev: SnapshotSummary,
        curr: SnapshotSummary,
        spot_change_pct: float,
        net_ce_oi_change: int,
        net_pe_oi_change: int,
        ce_adds: list[dict[str, Any]],
        pe_adds: list[dict[str, Any]],
        ce_exits: list[dict[str, Any]],
        pe_exits: list[dict[str, Any]],
    ) -> tuple[str, list[str]]:
        """Determine signal for a single 15-min interval."""
        bull = 0
        bear = 0
        reasons: list[str] = []

        # PCR direction
        pcr_change = curr.pcr_oi - prev.pcr_oi
        if pcr_change > 0.03:
            bull += 1
            reasons.append(f"PCR rising (+{pcr_change:.3f}) → more PE writing (bullish)")
        elif pcr_change < -0.03:
            bear += 1
            reasons.append(f"PCR falling ({pcr_change:.3f}) → more CE writing (bearish)")

        # Spot vs OI correlation
        if spot_change_pct > 0.1 and net_pe_oi_change > net_ce_oi_change:
            bull += 1
            reasons.append("Spot up + PE OI addition → support building (bullish)")
        elif spot_change_pct < -0.1 and net_ce_oi_change > net_pe_oi_change:
            bear += 1
            reasons.append("Spot down + CE OI addition → resistance building (bearish)")

        # CE unwinding (shorts covering) = bullish
        total_ce_exits = sum(abs(e["oi_change"]) for e in ce_exits)
        total_pe_exits = sum(abs(e["oi_change"]) for e in pe_exits)
        if total_ce_exits > total_pe_exits * 1.5 and total_ce_exits > 0:
            bull += 1
            reasons.append("CE OI unwinding > PE OI unwinding → resistance weakening")
        elif total_pe_exits > total_ce_exits * 1.5 and total_pe_exits > 0:
            bear += 1
            reasons.append("PE OI unwinding > CE OI unwinding → support weakening")

        # Heavy PE writing at lower strikes = bullish
        pe_add_below_atm = sum(
            a["oi_change"] for a in pe_adds if a["strike"] < curr.atm_strike
        )
        ce_add_above_atm = sum(
            a["oi_change"] for a in ce_adds if a["strike"] > curr.atm_strike
        )
        if pe_add_below_atm > ce_add_above_atm * 1.3 and pe_add_below_atm > 0:
            bull += 1
            reasons.append("Heavy PE writing below ATM → support building")
        elif ce_add_above_atm > pe_add_below_atm * 1.3 and ce_add_above_atm > 0:
            bear += 1
            reasons.append("Heavy CE writing above ATM → resistance building")

        # Straddle premium change
        straddle_change_pct = (
            (curr.atm_straddle_premium - prev.atm_straddle_premium) /
            prev.atm_straddle_premium * 100
        ) if prev.atm_straddle_premium > 0 else 0.0

        if straddle_change_pct < -3:
            reasons.append(f"Straddle premium compressing ({straddle_change_pct:.1f}%) → trend forming")
        elif straddle_change_pct > 3:
            reasons.append(f"Straddle premium expanding (+{straddle_change_pct:.1f}%) → volatility spike")

        # IV skew shift
        iv_skew_change = curr.iv_skew - prev.iv_skew
        if iv_skew_change > 2:
            bear += 1
            reasons.append(f"Put IV rising faster than Call IV (+{iv_skew_change:.1f}%) → bearish skew")
        elif iv_skew_change < -2:
            bull += 1
            reasons.append(f"Call IV rising faster than Put IV ({iv_skew_change:.1f}%) → bullish skew")

        # Determine signal
        if bull > bear + 1:
            return "BULLISH", reasons
        elif bear > bull + 1:
            return "BEARISH", reasons
        elif bull > bear:
            return "MILDLY_BULLISH", reasons
        elif bear > bull:
            return "MILDLY_BEARISH", reasons
        return "NEUTRAL", reasons

    # ─── Trend Analysis Methods ─────────────────────────────────

    @staticmethod
    def _analyze_pcr_trend(summaries: list[SnapshotSummary]) -> dict[str, Any]:
        """Analyze PCR trend across snapshots."""
        if len(summaries) < 2:
            return {"trend": "INSUFFICIENT_DATA", "values": []}

        pcr_values = [s.pcr_oi for s in summaries]
        first_pcr = pcr_values[0]
        last_pcr = pcr_values[-1]
        change = last_pcr - first_pcr

        # Linear trend direction
        trend = "RISING" if change > 0.05 else "FALLING" if change < -0.05 else "STABLE"

        # Velocity: rate of change
        velocity = change / len(pcr_values) if pcr_values else 0.0

        # Consistency: how often does it move in the same direction?
        ups = sum(1 for i in range(1, len(pcr_values)) if pcr_values[i] > pcr_values[i - 1])
        downs = len(pcr_values) - 1 - ups
        consistency = max(ups, downs) / (len(pcr_values) - 1) if len(pcr_values) > 1 else 0.0

        return {
            "trend": trend,
            "first": round(first_pcr, 3),
            "last": round(last_pcr, 3),
            "change": round(change, 3),
            "velocity": round(velocity, 4),
            "consistency": round(consistency, 2),
            "values": [round(v, 3) for v in pcr_values],
            "interpretation": (
                "PCR rising → more PE writing/less CE writing → BULLISH support building"
                if trend == "RISING"
                else "PCR falling → more CE writing/less PE writing → BEARISH resistance building"
                if trend == "FALLING"
                else "PCR stable → no clear directional bias from put-call ratio"
            ),
        }

    @staticmethod
    def _analyze_straddle_trend(summaries: list[SnapshotSummary]) -> dict[str, Any]:
        """Analyze ATM straddle premium trend."""
        if len(summaries) < 2:
            return {"trend": "INSUFFICIENT_DATA", "values": []}

        premiums = [s.atm_straddle_premium for s in summaries]
        first = premiums[0]
        last = premiums[-1]
        change_pct = ((last - first) / first * 100) if first > 0 else 0.0

        trend = "EXPANDING" if change_pct > 3 else "COMPRESSING" if change_pct < -3 else "STABLE"

        return {
            "trend": trend,
            "first": round(first, 2),
            "last": round(last, 2),
            "change_pct": round(change_pct, 2),
            "values": [round(v, 2) for v in premiums],
            "interpretation": (
                "Straddle expanding → volatility increasing, expect sharp move"
                if trend == "EXPANDING"
                else "Straddle compressing → sellers confident, range-bound or trend developing"
                if trend == "COMPRESSING"
                else "Straddle stable → no significant volatility shift"
            ),
        }

    @staticmethod
    def _analyze_iv_skew_trend(summaries: list[SnapshotSummary]) -> dict[str, Any]:
        """Analyze IV skew (put IV - call IV) trend."""
        if len(summaries) < 2:
            return {"trend": "INSUFFICIENT_DATA", "values": []}

        skews = [s.iv_skew for s in summaries]
        first = skews[0]
        last = skews[-1]
        change = last - first

        trend = "PUT_SKEW_RISING" if change > 2 else "CALL_SKEW_RISING" if change < -2 else "STABLE"

        return {
            "trend": trend,
            "first": round(first, 2),
            "last": round(last, 2),
            "change": round(change, 2),
            "values": [round(v, 2) for v in skews],
            "interpretation": (
                "Put IV rising faster → fear/hedging increasing → bearish undertone"
                if trend == "PUT_SKEW_RISING"
                else "Call IV rising faster → upside demand growing → bullish undertone"
                if trend == "CALL_SKEW_RISING"
                else "IV skew stable → no significant hedging demand shift"
            ),
        }

    @staticmethod
    def _analyze_max_pain_drift(summaries: list[SnapshotSummary]) -> dict[str, Any]:
        """Analyze how max pain is moving through the day."""
        if len(summaries) < 2:
            return {"trend": "INSUFFICIENT_DATA", "values": []}

        pains = [s.max_pain for s in summaries]
        spots = [s.spot_price for s in summaries]
        first = pains[0]
        last = pains[-1]
        drift = last - first
        last_spot = spots[-1]

        trend = "DRIFTING_UP" if drift > 0 else "DRIFTING_DOWN" if drift < 0 else "ANCHORED"
        gap = last_spot - last if last > 0 else 0
        gap_pct = (gap / last * 100) if last > 0 else 0

        return {
            "trend": trend,
            "first": first,
            "last": last,
            "drift": drift,
            "spot_vs_max_pain": round(gap, 2),
            "spot_vs_max_pain_pct": round(gap_pct, 2),
            "values": pains,
            "interpretation": (
                f"Max pain drifting up → market gravitating higher (spot {'above' if gap > 0 else 'below'} max pain by {abs(gap_pct):.1f}%)"
                if trend == "DRIFTING_UP"
                else f"Max pain drifting down → market gravitating lower (spot {'above' if gap > 0 else 'below'} max pain by {abs(gap_pct):.1f}%)"
                if trend == "DRIFTING_DOWN"
                else f"Max pain anchored at {last:.0f} (spot {'above' if gap > 0 else 'below'} by {abs(gap_pct):.1f}%)"
            ),
        }

    @staticmethod
    def _analyze_oi_walls(summaries: list[SnapshotSummary]) -> dict[str, Any]:
        """Track how support (PE wall) and resistance (CE wall) move."""
        if len(summaries) < 2:
            return {"ce_wall_movement": "INSUFFICIENT_DATA", "pe_wall_movement": "INSUFFICIENT_DATA"}

        ce_walls = [s.max_ce_oi_strike for s in summaries]
        pe_walls = [s.max_pe_oi_strike for s in summaries]

        ce_drift = ce_walls[-1] - ce_walls[0]
        pe_drift = pe_walls[-1] - pe_walls[0]

        return {
            "ce_wall_first": ce_walls[0],
            "ce_wall_last": ce_walls[-1],
            "ce_wall_drift": ce_drift,
            "ce_wall_movement": "RISING" if ce_drift > 0 else "FALLING" if ce_drift < 0 else "STABLE",
            "pe_wall_first": pe_walls[0],
            "pe_wall_last": pe_walls[-1],
            "pe_wall_drift": pe_drift,
            "pe_wall_movement": "RISING" if pe_drift > 0 else "FALLING" if pe_drift < 0 else "STABLE",
            "ce_wall_values": ce_walls,
            "pe_wall_values": pe_walls,
            "interpretation": IntradayOIFlowAnalyzer._interpret_wall_movement(ce_drift, pe_drift),
        }

    @staticmethod
    def _interpret_wall_movement(ce_drift: float, pe_drift: float) -> str:
        if ce_drift > 0 and pe_drift > 0:
            return "Both walls shifting up → range shifting higher (BULLISH)"
        elif ce_drift < 0 and pe_drift < 0:
            return "Both walls shifting down → range shifting lower (BEARISH)"
        elif ce_drift > 0 and pe_drift < 0:
            return "Range widening → volatility expanding"
        elif ce_drift < 0 and pe_drift > 0:
            return "Range tightening → squeeze forming"
        elif ce_drift > 0:
            return "Resistance moving up → upside room expanding (MILDLY BULLISH)"
        elif pe_drift > 0:
            return "Support moving up → floor rising (BULLISH)"
        elif ce_drift < 0:
            return "Resistance moving down → capping upside (BEARISH)"
        elif pe_drift < 0:
            return "Support moving down → floor weakening (BEARISH)"
        return "Walls stable → range-bound"

    # ─── Smart Money Detection ──────────────────────────────────

    @staticmethod
    def _detect_smart_money_signals(
        summaries: list[SnapshotSummary],
        changes: list[OIFlowChange],
    ) -> list[dict[str, Any]]:
        """Detect patterns that suggest institutional/smart money positioning."""
        signals: list[dict[str, Any]] = []

        if len(summaries) < 3:
            return signals

        # 1 — Aggressive PE writing with spot rising = strong support (smart money selling puts)
        for chg in changes:
            if chg.spot_change_pct > 0.15 and chg.net_pe_oi_change > chg.net_ce_oi_change * 2:
                signals.append({
                    "type": "AGGRESSIVE_PE_WRITING",
                    "timestamp": chg.to_ts,
                    "sentiment": "BULLISH",
                    "description": f"Smart money writing PEs aggressively while spot rising (+{chg.spot_change_pct:.2f}%)",
                    "pe_oi_added": chg.net_pe_oi_change,
                    "ce_oi_added": chg.net_ce_oi_change,
                })

        # 2 — Aggressive CE writing with spot falling = strong resistance
        for chg in changes:
            if chg.spot_change_pct < -0.15 and chg.net_ce_oi_change > chg.net_pe_oi_change * 2:
                signals.append({
                    "type": "AGGRESSIVE_CE_WRITING",
                    "timestamp": chg.to_ts,
                    "sentiment": "BEARISH",
                    "description": f"Smart money writing CEs aggressively while spot falling ({chg.spot_change_pct:.2f}%)",
                    "ce_oi_added": chg.net_ce_oi_change,
                    "pe_oi_added": chg.net_pe_oi_change,
                })

        # 3 — Sudden PE unwinding = support gone, bearish
        for chg in changes:
            if chg.net_pe_oi_change < 0 and abs(chg.net_pe_oi_change) > 100000:
                signals.append({
                    "type": "PE_UNWINDING",
                    "timestamp": chg.to_ts,
                    "sentiment": "BEARISH",
                    "description": f"Massive PE unwinding ({chg.net_pe_oi_change:,}) → support collapsing",
                    "pe_oi_change": chg.net_pe_oi_change,
                })

        # 4 — Sudden CE unwinding = resistance gone, bullish
        for chg in changes:
            if chg.net_ce_oi_change < 0 and abs(chg.net_ce_oi_change) > 100000:
                signals.append({
                    "type": "CE_UNWINDING",
                    "timestamp": chg.to_ts,
                    "sentiment": "BULLISH",
                    "description": f"Massive CE unwinding ({chg.net_ce_oi_change:,}) → resistance collapsing",
                    "ce_oi_change": chg.net_ce_oi_change,
                })

        # 5 — Max pain tracking the spot closely = expiry pinning
        last = summaries[-1]
        if last.max_pain > 0 and last.spot_price > 0:
            gap_pct = abs(last.spot_price - last.max_pain) / last.spot_price * 100
            if gap_pct < 0.3:
                signals.append({
                    "type": "MAX_PAIN_PINNING",
                    "timestamp": last.timestamp,
                    "sentiment": "NEUTRAL",
                    "description": f"Spot pinning near max pain ({last.max_pain:.0f}) — expiry day gravitational pull",
                    "spot": last.spot_price,
                    "max_pain": last.max_pain,
                    "gap_pct": round(gap_pct, 2),
                })

        # 6 — PCR extreme readings
        if summaries[-1].pcr_oi > 1.5:
            signals.append({
                "type": "PCR_EXTREME_HIGH",
                "timestamp": summaries[-1].timestamp,
                "sentiment": "BULLISH",
                "description": f"PCR extremely high ({summaries[-1].pcr_oi:.2f}) — oversold, potential bull reversal",
                "pcr": summaries[-1].pcr_oi,
            })
        elif summaries[-1].pcr_oi < 0.5:
            signals.append({
                "type": "PCR_EXTREME_LOW",
                "timestamp": summaries[-1].timestamp,
                "sentiment": "BEARISH",
                "description": f"PCR extremely low ({summaries[-1].pcr_oi:.2f}) — overbought, potential bear reversal",
                "pcr": summaries[-1].pcr_oi,
            })

        # 7 — Wall shift with spot (leading indicator)
        if len(summaries) >= 3:
            pe_wall_up = summaries[-1].max_pe_oi_strike > summaries[0].max_pe_oi_strike
            spot_up = summaries[-1].spot_price > summaries[0].spot_price
            if pe_wall_up and spot_up:
                signals.append({
                    "type": "SUPPORT_RISING_WITH_SPOT",
                    "timestamp": summaries[-1].timestamp,
                    "sentiment": "BULLISH",
                    "description": "PE OI wall (support) rising alongside spot → strong bullish floor",
                    "pe_wall_from": summaries[0].max_pe_oi_strike,
                    "pe_wall_to": summaries[-1].max_pe_oi_strike,
                })

            ce_wall_down = summaries[-1].max_ce_oi_strike < summaries[0].max_ce_oi_strike
            spot_down = summaries[-1].spot_price < summaries[0].spot_price
            if ce_wall_down and spot_down:
                signals.append({
                    "type": "RESISTANCE_FALLING_WITH_SPOT",
                    "timestamp": summaries[-1].timestamp,
                    "sentiment": "BEARISH",
                    "description": "CE OI wall (resistance) falling alongside spot → strong bearish ceiling",
                    "ce_wall_from": summaries[0].max_ce_oi_strike,
                    "ce_wall_to": summaries[-1].max_ce_oi_strike,
                })

        return signals

    # ─── Market Direction Engine ────────────────────────────────

    @staticmethod
    def _compute_market_direction(
        pcr_trend: dict[str, Any],
        straddle_trend: dict[str, Any],
        iv_skew_trend: dict[str, Any],
        max_pain_drift: dict[str, Any],
        oi_wall_analysis: dict[str, Any],
        smart_money: list[dict[str, Any]],
        flow_changes: list[OIFlowChange],
        summaries: list[SnapshotSummary],
    ) -> tuple[str, int, list[str]]:
        """
        Combine all trend dimensions into a single market direction verdict.
        Returns: (direction, confidence_0_to_100, reasons)
        """
        bull = 0
        bear = 0
        reasons: list[str] = []

        # ── PCR Trend (weight: 2) ──
        pcr_t = pcr_trend.get("trend", "")
        if pcr_t == "RISING":
            bull += 2
            reasons.append(f"PCR trend RISING ({pcr_trend.get('first', 0):.2f}→{pcr_trend.get('last', 0):.2f}) = bullish support building")
        elif pcr_t == "FALLING":
            bear += 2
            reasons.append(f"PCR trend FALLING ({pcr_trend.get('first', 0):.2f}→{pcr_trend.get('last', 0):.2f}) = bearish resistance building")

        # ── OI Walls (weight: 2) ──
        ce_wall = oi_wall_analysis.get("ce_wall_movement", "")
        pe_wall = oi_wall_analysis.get("pe_wall_movement", "")
        if pe_wall == "RISING":
            bull += 2
            reasons.append(f"PE support wall rising ({oi_wall_analysis.get('pe_wall_first', 0):.0f}→{oi_wall_analysis.get('pe_wall_last', 0):.0f})")
        elif pe_wall == "FALLING":
            bear += 1
            reasons.append(f"PE support wall falling ({oi_wall_analysis.get('pe_wall_first', 0):.0f}→{oi_wall_analysis.get('pe_wall_last', 0):.0f})")
        if ce_wall == "RISING":
            bull += 1
            reasons.append(f"CE resistance wall rising ({oi_wall_analysis.get('ce_wall_first', 0):.0f}→{oi_wall_analysis.get('ce_wall_last', 0):.0f})")
        elif ce_wall == "FALLING":
            bear += 2
            reasons.append(f"CE resistance wall falling ({oi_wall_analysis.get('ce_wall_first', 0):.0f}→{oi_wall_analysis.get('ce_wall_last', 0):.0f})")

        # ── Max Pain Drift (weight: 1) ──
        mp_t = max_pain_drift.get("trend", "")
        if mp_t == "DRIFTING_UP":
            bull += 1
            reasons.append(f"Max pain drifting up ({max_pain_drift.get('first', 0):.0f}→{max_pain_drift.get('last', 0):.0f})")
        elif mp_t == "DRIFTING_DOWN":
            bear += 1
            reasons.append(f"Max pain drifting down ({max_pain_drift.get('first', 0):.0f}→{max_pain_drift.get('last', 0):.0f})")

        # ── IV Skew (weight: 1) ──
        iv_t = iv_skew_trend.get("trend", "")
        if iv_t == "CALL_SKEW_RISING":
            bull += 1
            reasons.append("Call IV demand rising → bullish undertone")
        elif iv_t == "PUT_SKEW_RISING":
            bear += 1
            reasons.append("Put IV demand rising → bearish undertone")

        # ── Straddle Trend (weight: 1) ──
        str_t = straddle_trend.get("trend", "")
        if str_t == "COMPRESSING" and summaries:
            # Compression + spot direction = trend confirmation
            spot_change = summaries[-1].spot_price - summaries[0].spot_price
            if spot_change > 0:
                bull += 1
                reasons.append("Straddle compressing + spot up → bullish trend forming")
            elif spot_change < 0:
                bear += 1
                reasons.append("Straddle compressing + spot down → bearish trend forming")

        # ── Smart Money Signals (weight: 1 each) ──
        sm_bull = sum(1 for s in smart_money if s.get("sentiment") == "BULLISH")
        sm_bear = sum(1 for s in smart_money if s.get("sentiment") == "BEARISH")
        if sm_bull > sm_bear:
            bull += min(sm_bull - sm_bear, 2)
            reasons.append(f"Smart money signals: {sm_bull} bullish vs {sm_bear} bearish")
        elif sm_bear > sm_bull:
            bear += min(sm_bear - sm_bull, 2)
            reasons.append(f"Smart money signals: {sm_bear} bearish vs {sm_bull} bullish")

        # ── Flow Change Consensus (weight: 2) ──
        if flow_changes:
            bull_intervals = sum(1 for c in flow_changes if c.signal in ("BULLISH", "MILDLY_BULLISH"))
            bear_intervals = sum(1 for c in flow_changes if c.signal in ("BEARISH", "MILDLY_BEARISH"))
            total_intervals = len(flow_changes)
            if bull_intervals > bear_intervals + 1:
                bull += 2
                reasons.append(f"{bull_intervals}/{total_intervals} intervals bullish")
            elif bear_intervals > bull_intervals + 1:
                bear += 2
                reasons.append(f"{bear_intervals}/{total_intervals} intervals bearish")

        # ── Spot price trend (weight: 1) ──
        if len(summaries) >= 2:
            spot_first = summaries[0].spot_price
            spot_last = summaries[-1].spot_price
            spot_change_pct = ((spot_last - spot_first) / spot_first * 100) if spot_first > 0 else 0
            if spot_change_pct > 0.3:
                bull += 1
                reasons.append(f"Spot up +{spot_change_pct:.2f}% intraday")
            elif spot_change_pct < -0.3:
                bear += 1
                reasons.append(f"Spot down {spot_change_pct:.2f}% intraday")

        # ── Compute Final Verdict ──
        total = bull + bear
        if total == 0:
            return "NEUTRAL", 50, ["All indicators balanced — no clear directional bias"]

        net = bull - bear
        confidence = int(min(abs(net) / max(total, 1) * 100, 95))
        confidence = max(confidence, 20)  # floor at 20

        if net >= 5:
            return "BULLISH", confidence, reasons
        elif net <= -5:
            return "BEARISH", confidence, reasons
        elif net >= 3:
            return "MILDLY_BULLISH", confidence, reasons
        elif net <= -3:
            return "MILDLY_BEARISH", confidence, reasons
        elif net > 0:
            return "MILDLY_BULLISH", confidence, reasons
        elif net < 0:
            return "MILDLY_BEARISH", confidence, reasons
        return "NEUTRAL", 50, reasons

    # ─── Trend Reinforcement ────────────────────────────────────

    @staticmethod
    def _compute_trend_reinforcement(changes: list[OIFlowChange]) -> TrendReinforcement:
        """
        Measure how many consecutive intervals confirm the same direction.
        More consecutive same-direction signals → stronger trend conviction.
        """
        if not changes:
            return TrendReinforcement(trend_strength="NO_DATA", trend_score=0)

        # Count consecutive streaks
        consecutive_bull = 0
        consecutive_bear = 0
        consecutive_neutral = 0
        max_bull_streak = 0
        max_bear_streak = 0

        # Compute running trend score: +1 for bullish, -1 for bearish, 0 for neutral
        score = 0.0
        for chg in changes:
            if chg.signal in ("BULLISH", "MILDLY_BULLISH"):
                weight = 1.0 if chg.signal == "BULLISH" else 0.5
                score += weight
                consecutive_bull += 1
                consecutive_bear = 0
                consecutive_neutral = 0
                max_bull_streak = max(max_bull_streak, consecutive_bull)
            elif chg.signal in ("BEARISH", "MILDLY_BEARISH"):
                weight = 1.0 if chg.signal == "BEARISH" else 0.5
                score -= weight
                consecutive_bear += 1
                consecutive_bull = 0
                consecutive_neutral = 0
                max_bear_streak = max(max_bear_streak, consecutive_bear)
            else:
                consecutive_neutral += 1
                consecutive_bull = 0
                consecutive_bear = 0

        # Normalize score to -100..+100 range
        max_possible = len(changes)
        normalized_score = (score / max_possible * 100) if max_possible > 0 else 0.0

        # Determine strength
        reinforcement = max(max_bull_streak, max_bear_streak)
        if normalized_score >= 60:
            strength = "STRONG_BULLISH"
        elif normalized_score >= 30:
            strength = "MODERATE_BULLISH"
        elif normalized_score <= -60:
            strength = "STRONG_BEARISH"
        elif normalized_score <= -30:
            strength = "MODERATE_BEARISH"
        else:
            strength = "WEAK"

        return TrendReinforcement(
            consecutive_bullish=max_bull_streak,
            consecutive_bearish=max_bear_streak,
            consecutive_neutral=consecutive_neutral,
            trend_strength=strength,
            trend_score=round(normalized_score, 1),
            reinforcement_count=reinforcement,
        )

    # ─── Max Pain Helper ────────────────────────────────────────

    @staticmethod
    def _compute_max_pain_from_map(strike_map: dict[float, dict[str, Any]]) -> float:
        """Max pain computation from strike_map."""
        strikes = list(strike_map.keys())
        if not strikes:
            return 0.0

        min_pain = float("inf")
        max_pain_strike = strikes[0]

        for test_strike in strikes:
            total_pain = 0.0
            for s, d in strike_map.items():
                ce_oi = d.get("ce_oi", 0)
                pe_oi = d.get("pe_oi", 0)
                if ce_oi > 0:
                    intrinsic = max(test_strike - s, 0)
                    total_pain += intrinsic * ce_oi
                if pe_oi > 0:
                    intrinsic = max(s - test_strike, 0)
                    total_pain += intrinsic * pe_oi
            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = test_strike

        return max_pain_strike

    # ─── Summary Generator ──────────────────────────────────────

    @staticmethod
    def _generate_summary(
        underlying: str,
        summaries: list[SnapshotSummary],
        direction: str,
        confidence: int,
        reasons: list[str],
        reinforcement: TrendReinforcement,
        pcr_trend: dict[str, Any],
        straddle_trend: dict[str, Any],
    ) -> str:
        """Generate a human-readable summary of the analysis."""
        if not summaries:
            return "No data available for analysis."

        first = summaries[0]
        last = summaries[-1]
        spot_change = last.spot_price - first.spot_price
        spot_change_pct = (spot_change / first.spot_price * 100) if first.spot_price > 0 else 0

        lines = [
            f"📊 {underlying} Intraday OI Flow Analysis ({len(summaries)} snapshots)",
            f"",
            f"Spot: {first.spot_price:.2f} → {last.spot_price:.2f} ({spot_change_pct:+.2f}%)",
            f"PCR: {first.pcr_oi:.3f} → {last.pcr_oi:.3f} (trend: {pcr_trend.get('trend', 'N/A')})",
            f"Straddle: {first.atm_straddle_premium:.2f} → {last.atm_straddle_premium:.2f} ({straddle_trend.get('trend', 'N/A')})",
            f"Support (PE Wall): {first.max_pe_oi_strike:.0f} → {last.max_pe_oi_strike:.0f}",
            f"Resistance (CE Wall): {first.max_ce_oi_strike:.0f} → {last.max_ce_oi_strike:.0f}",
            f"",
            f"🎯 DIRECTION: {direction} (confidence: {confidence}%)",
            f"📈 Trend Strength: {reinforcement.trend_strength} (score: {reinforcement.trend_score:+.1f})",
            f"🔄 Max Reinforcement: {reinforcement.reinforcement_count} consecutive intervals",
            f"",
            f"Key Reasons:",
        ]

        for i, reason in enumerate(reasons[:8], 1):
            lines.append(f"  {i}. {reason}")

        return "\n".join(lines)


# ─── Singleton ──────────────────────────────────────────────────

_analyzer_instance: Optional[IntradayOIFlowAnalyzer] = None


def get_intraday_oi_analyzer() -> IntradayOIFlowAnalyzer:
    """Get or create singleton analyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = IntradayOIFlowAnalyzer()
    return _analyzer_instance
