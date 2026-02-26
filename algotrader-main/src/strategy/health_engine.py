"""
Strategy Health Engine — Pre-execution strategy validation & scoring.

Computes a comprehensive health report across 8 pillars:
  1. Core Profitability  (Expectancy, Profit Factor, Risk-Adjusted Return)
  2. Drawdown Control     (Max Drawdown, Recovery Factor, Equity Curve R²)
  3. Trade Quality        (Win/Loss Structure, Avg Winner vs Loser, Payoff Ratio)
  4. Robustness           (Parameter Stability, Regime Performance, Trade Distribution)
  5. Execution Reality    (Slippage Impact, Cost Drag, Fill Quality)
  6. Capital Efficiency   (Return on Margin, Capital Utilisation)
  7. Risk Architecture    (Position Sizing, Exposure Control, Kill Switch)
  8. Psychological        (Longest Losing Streak, Time Underwater, Trades/Day)

Each pillar produces a score 0-100, a verdict (PASS / WARN / FAIL), and detail metrics.
An overall "Health Score" is the weighted average.

Usage:
    report = compute_health_report(backtest_result, strategy_type="fno")
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import numpy as np


# ────────────────────────────────────────────────────────────
# Data structures
# ────────────────────────────────────────────────────────────

@dataclass
class PillarResult:
    """Score & detail for one pillar."""
    name: str
    score: float = 0.0          # 0-100
    verdict: str = "FAIL"       # PASS / WARN / FAIL
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HealthReport:
    """Complete strategy health report."""
    strategy_name: str = ""
    strategy_type: str = ""          # "equity" / "fno" / "oi_strategy"
    overall_score: float = 0.0       # 0-100
    overall_verdict: str = "FAIL"    # PASS / WARN / FAIL
    execution_ready: bool = False    # True only if overall_score >= 60 and no FAIL pillars
    pillars: dict[str, PillarResult] = field(default_factory=dict)
    summary: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    raw_input: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "strategy_name": self.strategy_name,
            "strategy_type": self.strategy_type,
            "overall_score": round(self.overall_score, 1),
            "overall_verdict": self.overall_verdict,
            "execution_ready": self.execution_ready,
            "pillars": {k: v.to_dict() for k, v in self.pillars.items()},
            "summary": self.summary,
            "warnings": self.warnings,
            "blockers": self.blockers,
        }
        return d


# ────────────────────────────────────────────────────────────
# Pillar weights (sum = 1.0)
# ────────────────────────────────────────────────────────────

PILLAR_WEIGHTS = {
    "profitability": 0.25,
    "drawdown": 0.20,
    "trade_quality": 0.15,
    "robustness": 0.10,
    "execution": 0.10,
    "capital_efficiency": 0.08,
    "risk_architecture": 0.07,
    "psychological": 0.05,
}


# ────────────────────────────────────────────────────────────
# Thresholds per strategy type
# ────────────────────────────────────────────────────────────

THRESHOLDS = {
    "equity": {
        "expectancy_good": 0.2,
        "expectancy_ok": 0.05,
        "profit_factor_good": 1.5,
        "profit_factor_ok": 1.2,
        "sharpe_good": 1.5,
        "sharpe_ok": 1.0,
        "sortino_good": 2.0,
        "sortino_ok": 1.2,
        "max_dd_good": 10.0,
        "max_dd_ok": 15.0,
        "recovery_good": 5.0,
        "recovery_ok": 3.0,
        "payoff_good": 1.5,
        "payoff_ok": 1.0,
        "avg_win_ratio_good": 1.3,
        "avg_win_ratio_ok": 1.0,
        "rom_good": 30.0,
        "rom_ok": 15.0,
    },
    "fno": {
        "expectancy_good": 0.3,
        "expectancy_ok": 0.1,
        "profit_factor_good": 1.5,
        "profit_factor_ok": 1.2,
        "sharpe_good": 1.5,
        "sharpe_ok": 1.0,
        "sortino_good": 2.0,
        "sortino_ok": 1.2,
        "max_dd_good": 15.0,
        "max_dd_ok": 25.0,
        "recovery_good": 5.0,
        "recovery_ok": 3.0,
        "payoff_good": 2.0,
        "payoff_ok": 1.3,
        "avg_win_ratio_good": 1.8,
        "avg_win_ratio_ok": 1.3,
        "rom_good": 40.0,
        "rom_ok": 20.0,
    },
    "intraday": {
        "expectancy_good": 0.1,
        "expectancy_ok": 0.03,
        "profit_factor_good": 1.5,
        "profit_factor_ok": 1.2,
        "sharpe_good": 1.5,
        "sharpe_ok": 1.0,
        "sortino_good": 2.0,
        "sortino_ok": 1.2,
        "max_dd_good": 7.0,
        "max_dd_ok": 10.0,
        "recovery_good": 5.0,
        "recovery_ok": 3.0,
        "payoff_good": 1.3,
        "payoff_ok": 1.0,
        "avg_win_ratio_good": 1.3,
        "avg_win_ratio_ok": 1.0,
        "rom_good": 30.0,
        "rom_ok": 15.0,
    },
    # ── Credit / income F&O strategies ──────────────────────
    # Iron butterfly, iron condor, short straddle, short strangle,
    # bull put spread, bear call spread: high win-rate, small wins,
    # occasional large (but capped) losses.  Traditional payoff /
    # avg-win-ratio thresholds are inappropriate for these.
    "fno_credit": {
        "expectancy_good": 0.15,      # smaller edge per R is normal
        "expectancy_ok": 0.05,
        "profit_factor_good": 1.3,    # 1.3 PF on credit spreads is solid
        "profit_factor_ok": 1.05,
        "sharpe_good": 1.2,
        "sharpe_ok": 0.7,
        "sortino_good": 1.5,
        "sortino_ok": 0.8,
        "max_dd_good": 20.0,          # credit structures can spike
        "max_dd_ok": 35.0,
        "recovery_good": 3.0,
        "recovery_ok": 1.5,
        "payoff_good": 0.6,           # avg_win < avg_loss by design
        "payoff_ok": 0.3,
        "avg_win_ratio_good": 0.7,    # not expected to exceed 1.0
        "avg_win_ratio_ok": 0.3,
        "rom_good": 25.0,
        "rom_ok": 12.0,
    },
}

# Structures that are naturally "credit / income" strategies
CREDIT_STRUCTURES: set[str] = {
    "iron_butterfly", "iron_condor", "short_straddle", "short_strangle",
    "bull_put_spread", "bear_call_spread",
    # StructureType enum values (lowercase)
    "IRON_BUTTERFLY", "IRON_CONDOR", "SHORT_STRADDLE", "SHORT_STRANGLE",
    "BULL_PUT_SPREAD", "BEAR_CALL_SPREAD",
}


def _is_credit_strategy(result: dict) -> bool:
    """Detect whether the result belongs to a credit / income F&O structure."""
    struct = result.get("structure", result.get("structure_type", ""))
    sname = result.get("strategy_name", "")
    return (
        struct.lower().replace(" ", "_") in {s.lower() for s in CREDIT_STRUCTURES}
        or sname.lower().replace(" ", "_") in {s.lower() for s in CREDIT_STRUCTURES}
    )


def _get_thresholds(strategy_type: str, result: dict | None = None) -> dict[str, float]:
    """Return threshold dict appropriate for the strategy type and structure."""
    # Credit / income F&O strategies get specialised thresholds
    if result and _is_credit_strategy(result):
        return THRESHOLDS["fno_credit"]

    stype = strategy_type.lower()
    if stype in THRESHOLDS:
        return THRESHOLDS[stype]
    if "intraday" in stype:
        return THRESHOLDS["intraday"]
    if "fno" in stype or "option" in stype or "oi" in stype:
        return THRESHOLDS["fno"]
    return THRESHOLDS["equity"]


# ────────────────────────────────────────────────────────────
# Streak helper  (used by _normalize_result and Pillar 4/8)
# ────────────────────────────────────────────────────────────

def _compute_streaks(pnls: list[float]) -> tuple[int, int]:
    """Compute max winning and losing streaks."""
    max_win = max_lose = cur_win = cur_lose = 0
    for pnl in pnls:
        if pnl > 0:
            cur_win += 1
            cur_lose = 0
            max_win = max(max_win, cur_win)
        else:
            cur_lose += 1
            cur_win = 0
            max_lose = max(max_lose, cur_lose)
    return max_win, max_lose


# ────────────────────────────────────────────────────────────
# Result-dict normalisation
# ────────────────────────────────────────────────────────────

# Trade types that represent an actual exit (with PnL)
_EXIT_TYPES: set[str] = {
    "PROFIT_TARGET", "STOP_LOSS", "FINAL_EXIT", "EXPIRY",
    "SL_LONG_EXIT", "SL_SHORT_EXIT", "TP_LONG_EXIT", "TP_SHORT_EXIT",
    "EXIT", "CLOSE",
}


def _extract_exit_pnls(result: dict) -> list[float]:
    """Extract PnL values from exit trades only, handling all result shapes.

    Equity BacktestEngine:   trades have "pnl" on every entry.
    FnO BacktestEngine:      trades have mixed types; only exits have "pnl".
    FnO PaperTradingEngine:  "orders" list or "positions" list.
    """
    pnls: list[float] = []

    # 1. Try trades list (equity + FnO backtest)
    trades = result.get("trades", [])
    if trades:
        for t in trades:
            ttype = t.get("type", "")
            pnl = t.get("pnl")
            if pnl is None:
                continue
            # If type is present and it's an ENTRY, skip it
            if ttype and ttype.upper() == "ENTRY":
                continue
            # If type is present and it's a known exit, include
            if ttype and ttype.upper() in _EXIT_TYPES:
                pnls.append(float(pnl))
            # If no type tag (equity backtest) just include all with pnl
            elif not ttype:
                pnls.append(float(pnl))
        if pnls:
            return pnls

    # 2. Try positions list (FnO paper trade)
    positions = result.get("positions", [])
    if positions:
        for pos in positions:
            pnl = pos.get("pnl")
            if pnl is not None:
                pnls.append(float(pnl))
        if pnls:
            return pnls

    # 3. Try orders list (FnO paper trade)
    orders = result.get("orders", [])
    if orders:
        for o in orders:
            pnl = o.get("pnl")
            if pnl is not None:
                pnls.append(float(pnl))

    return pnls


def _synthesize_monthly_pnl(result: dict) -> dict[str, float]:
    """Compute monthly PnL from trades when trade_analytics.monthly_pnl is absent."""
    monthly: dict[str, float] = {}

    # Collect (date_str, pnl) pairs
    items: list[tuple[str, float]] = []
    for t in result.get("trades", []):
        ttype = t.get("type", "")
        if ttype and ttype.upper() == "ENTRY":
            continue
        pnl = t.get("pnl")
        dt = t.get("date", t.get("exit_time", t.get("entry_time", "")))
        if pnl is not None and dt:
            items.append((str(dt), float(pnl)))

    if not items:
        for pos in result.get("positions", []):
            pnl = pos.get("pnl")
            dt = pos.get("exit", pos.get("date", ""))
            if pnl is not None and dt:
                items.append((str(dt), float(pnl)))

    for dt_str, pnl in items:
        try:
            month_key = dt_str[:7]  # "YYYY-MM"
            if len(month_key) == 7 and month_key[4] == "-":
                monthly[month_key] = monthly.get(month_key, 0) + pnl
        except (IndexError, ValueError):
            pass

    return monthly


def _normalize_result(result: dict) -> None:
    """Fill commonly-missing keys so pillar functions see a uniform shape.

    Mutates ``result`` in-place.  Must be called before pillar computation.
    """
    engine = result.get("engine", "")
    is_fno = "fno" in engine.lower()

    # ── total_pnl ──
    if "total_pnl" not in result:
        ic = result.get("initial_capital", 0)
        fc = result.get("final_capital", 0)
        if ic and fc:
            result["total_pnl"] = fc - ic

    # ── payoff_ratio ──
    if "payoff_ratio" not in result:
        aw = result.get("avg_win", 0)
        al = abs(result.get("avg_loss", 0))
        result["payoff_ratio"] = round(aw / al, 4) if al > 0 else 0

    # ── sortino_ratio from equity_curve ──
    if "sortino_ratio" not in result or result.get("sortino_ratio", 0) == 0:
        eq = result.get("equity_curve", [])
        if len(eq) > 2:
            arr = np.array(eq, dtype=float)
            rets = np.diff(arr) / np.where(arr[:-1] != 0, arr[:-1], 1)
            neg = rets[rets < 0]
            if len(neg) > 0 and np.std(neg) > 0:
                result["sortino_ratio"] = round(
                    float(np.mean(rets) / np.std(neg) * np.sqrt(252)), 4
                )

    # ── trade_analytics (streaks, best/worst) from exit PnLs ──
    analytics = result.setdefault("trade_analytics", {})
    exit_pnls = _extract_exit_pnls(result)
    if exit_pnls:
        if not analytics.get("max_win_streak") and not analytics.get("max_lose_streak"):
            ws, ls = _compute_streaks(exit_pnls)
            analytics.setdefault("max_win_streak", ws)
            analytics.setdefault("max_lose_streak", ls)
        analytics.setdefault("best_trade", round(max(exit_pnls), 2))
        analytics.setdefault("worst_trade", round(min(exit_pnls), 2))

        # monthly_pnl synthesis
        if not analytics.get("monthly_pnl"):
            mpnl = _synthesize_monthly_pnl(result)
            if mpnl:
                analytics["monthly_pnl"] = mpnl

    # ── Equity settings fallback (from stop_loss_stats + result keys) ──
    if "settings" not in result:
        sl_stats = result.get("stop_loss_stats", {})
        result["settings"] = {
            "stop_loss_pct": 1 if sl_stats.get("sl_exits", 0) > 0 else 0,
            "profit_target_pct": 1 if sl_stats.get("tp_exits", 0) > 0 else 0,
            "max_positions": result.get("max_positions", 0),
            "position_sizing": result.get("position_sizing", "fixed"),
        }

    # ── Normalize margin_history keys ──
    # FnO backtest uses "margin", FnO paper trade uses "margin_required"
    margin_hist = result.get("margin_history", [])
    for m in margin_hist:
        if "margin" not in m and "margin_required" in m:
            m["margin"] = m["margin_required"]

    # ── expectancy ──
    if "expectancy" not in result:
        wr = result.get("win_rate", 0) / 100
        aw = result.get("avg_win", 0)
        al = abs(result.get("avg_loss", 0))
        result["expectancy"] = round(wr * aw - (1 - wr) * al, 2)


def _score_metric(value: float, good: float, ok: float, higher_is_better: bool = True) -> tuple[float, str]:
    """Score a single metric, return (score 0-100, verdict)."""
    if higher_is_better:
        if value >= good:
            return min(100, 70 + 30 * min((value - good) / max(good, 0.01), 1.0)), "PASS"
        elif value >= ok:
            return 40 + 30 * (value - ok) / max(good - ok, 0.01), "WARN"
        else:
            if ok > 0:
                return max(0, 40 * value / ok), "FAIL"
            return 0, "FAIL"
    else:
        # Lower is better (e.g. drawdown)
        if value <= good:
            return min(100, 70 + 30 * max(0, 1 - value / max(good, 0.01))), "PASS"
        elif value <= ok:
            return 40 + 30 * (1 - (value - good) / max(ok - good, 0.01)), "WARN"
        else:
            return max(0, 40 * (1 - (value - ok) / max(ok, 0.01))), "FAIL"


# ────────────────────────────────────────────────────────────
# Pillar 1: Core Profitability
# ────────────────────────────────────────────────────────────

def _compute_profitability(result: dict, th: dict) -> PillarResult:
    p = PillarResult(name="Core Profitability")

    win_rate = result.get("win_rate", 0) / 100  # convert to 0-1
    avg_win = result.get("avg_win", 0)
    avg_loss = abs(result.get("avg_loss", 0))

    # Expectancy
    expectancy = result.get("expectancy")
    if expectancy is None:
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    p.metrics["expectancy"] = round(expectancy, 2)

    # Normalise expectancy to R (risk per trade = avg_loss or initial capital fraction)
    risk_per_trade = avg_loss if avg_loss > 0 else 1
    expectancy_r = expectancy / risk_per_trade if risk_per_trade > 0 else 0
    p.metrics["expectancy_r"] = round(expectancy_r, 3)

    exp_score, exp_v = _score_metric(expectancy_r, th["expectancy_good"], th["expectancy_ok"])

    # Profit factor
    pf = result.get("profit_factor", 0)
    p.metrics["profit_factor"] = round(pf, 2)
    pf_score, pf_v = _score_metric(pf, th["profit_factor_good"], th["profit_factor_ok"])

    # Overfit warning
    if pf > 3.0:
        p.notes.append("Profit factor >3 may indicate overfitting")
        pf_score = min(pf_score, 75)

    # Synthetic backtest sanity flags (from fno_backtest engine)
    bt_warnings = result.get("backtest_warnings", [])
    if "win_rate_suspiciously_high" in bt_warnings:
        p.notes.append("Win rate >85% flagged — verify with realistic data")
        exp_score = min(exp_score, 60)
        pf_score = min(pf_score, 60)
    if "low_trade_count" in bt_warnings:
        p.notes.append("Low trade count (<15) — results statistically unreliable")
        exp_score = min(exp_score, 50)
    if "profit_factor_suspiciously_high" in bt_warnings:
        p.notes.append("Profit factor >5 flagged — likely synthetic data artifact")
        pf_score = min(pf_score, 50)

    # Sharpe
    sharpe = result.get("sharpe_ratio", 0)
    p.metrics["sharpe_ratio"] = round(sharpe, 4)
    sh_score, sh_v = _score_metric(sharpe, th["sharpe_good"], th["sharpe_ok"])

    # Sortino
    sortino = result.get("sortino_ratio", 0)
    p.metrics["sortino_ratio"] = round(sortino, 4)
    so_score, so_v = _score_metric(sortino, th["sortino_good"], th["sortino_ok"])

    # ── Credit strategy win-rate bonus ──
    # Credit / income strategies derive edge from high win-rate × many small wins.
    # Even with payoff < 1.0, the math works if win_rate is high enough.
    is_credit = result.get("_is_credit_strategy", False)
    if is_credit and win_rate >= 0.55:
        wr_bonus = min(30, (win_rate - 0.55) * 200)  # up to +30 for 70%+ WR
        p.metrics["credit_win_rate_bonus"] = round(wr_bonus, 1)
        p.notes.append(f"Credit strategy: win-rate {win_rate:.0%} adds {wr_bonus:.0f} pts")
    else:
        wr_bonus = 0

    # Weighted pillar score (credit strategies de-emphasise payoff-dependent metrics)
    if is_credit:
        # PF and expectancy matter most; Sharpe less so (short gamma spikes it)
        p.score = 0.30 * exp_score + 0.35 * pf_score + 0.15 * sh_score + 0.10 * so_score + wr_bonus * 0.10
    else:
        p.score = 0.35 * exp_score + 0.25 * pf_score + 0.20 * sh_score + 0.20 * so_score
    p.score = min(100, p.score)

    # Verdict
    if expectancy_r < 0 or pf < 1.0:
        p.verdict = "FAIL"
        p.notes.append("Negative expectancy or profit factor <1 — strategy loses money")
    elif exp_v == "PASS" and pf_v in ("PASS", "WARN"):
        p.verdict = "PASS"
    elif is_credit and pf >= 1.0 and expectancy_r > 0:
        # Credit strategies with positive expectancy and PF≥1 should not FAIL
        p.verdict = "WARN" if exp_v == "FAIL" or pf_v == "FAIL" else "PASS"
    elif any(v == "FAIL" for v in [exp_v, pf_v]):
        p.verdict = "FAIL"
    else:
        p.verdict = "WARN"

    return p


# ────────────────────────────────────────────────────────────
# Pillar 2: Drawdown Control
# ────────────────────────────────────────────────────────────

def _compute_drawdown(result: dict, th: dict) -> PillarResult:
    p = PillarResult(name="Drawdown Control")

    max_dd = result.get("max_drawdown_pct", 0)
    p.metrics["max_drawdown_pct"] = round(max_dd, 2)
    dd_score, dd_v = _score_metric(max_dd, th["max_dd_good"], th["max_dd_ok"], higher_is_better=False)

    # Recovery Factor
    recovery = result.get("recovery_factor")
    if recovery is None:
        equity_curve = result.get("equity_curve", [])
        if len(equity_curve) >= 2:
            net_profit = equity_curve[-1] - equity_curve[0]
            ic = equity_curve[0]
            max_dd_abs = ic * max_dd / 100 if ic > 0 else 0
            recovery = abs(net_profit) / max_dd_abs if max_dd_abs > 0 else 0
        else:
            recovery = 0
    p.metrics["recovery_factor"] = round(recovery, 2)
    rf_score, rf_v = _score_metric(recovery, th["recovery_good"], th["recovery_ok"])

    # Equity Curve R² (smoothness)
    equity_curve = result.get("equity_curve", [])
    r_squared = 0.0
    if len(equity_curve) > 10:
        y = np.array(equity_curve, dtype=float)
        x = np.arange(len(y), dtype=float)
        # Linear regression R²
        if np.std(y) > 0:
            correlation = np.corrcoef(x, y)[0, 1]
            r_squared = correlation ** 2 if not np.isnan(correlation) else 0
    p.metrics["equity_curve_r2"] = round(r_squared, 4)
    r2_score, r2_v = _score_metric(r_squared, 0.85, 0.65)

    # Ulcer Index (RMS of drawdowns)
    ulcer_index = 0.0
    if len(equity_curve) > 1:
        eq = np.array(equity_curve, dtype=float)
        peak = np.maximum.accumulate(eq)
        dd_pct = np.where(peak > 0, (peak - eq) / peak * 100, 0)
        ulcer_index = float(np.sqrt(np.mean(dd_pct ** 2)))
    p.metrics["ulcer_index"] = round(ulcer_index, 2)

    p.score = 0.40 * dd_score + 0.30 * rf_score + 0.30 * r2_score
    if dd_v == "FAIL":
        p.verdict = "FAIL"
        p.notes.append(f"Max drawdown {max_dd:.1f}% exceeds threshold")
    elif dd_v == "PASS" and rf_v in ("PASS", "WARN"):
        p.verdict = "PASS"
    else:
        p.verdict = "WARN"

    return p


# ────────────────────────────────────────────────────────────
# Pillar 3: Trade Quality
# ────────────────────────────────────────────────────────────

def _compute_trade_quality(result: dict, th: dict) -> PillarResult:
    p = PillarResult(name="Trade Quality")

    win_rate = result.get("win_rate", 0)
    avg_win = result.get("avg_win", 0)
    avg_loss = abs(result.get("avg_loss", 0))
    total_trades = result.get("total_trades", 0)

    p.metrics["win_rate"] = round(win_rate, 1)
    p.metrics["avg_win"] = round(avg_win, 2)
    p.metrics["avg_loss"] = round(avg_loss, 2)
    p.metrics["total_trades"] = total_trades

    # Payoff ratio
    payoff = result.get("payoff_ratio")
    if payoff is None:
        payoff = avg_win / avg_loss if avg_loss > 0 else 0
    p.metrics["payoff_ratio"] = round(payoff, 2)
    pay_score, pay_v = _score_metric(payoff, th["payoff_good"], th["payoff_ok"])

    # Avg Win / Avg Loss ratio
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
    p.metrics["win_loss_ratio"] = round(win_loss_ratio, 2)
    wl_score, wl_v = _score_metric(win_loss_ratio, th["avg_win_ratio_good"], th["avg_win_ratio_ok"])

    # Win/Loss structure classification
    is_credit = result.get("_is_credit_strategy", False)
    if is_credit:
        # ── Credit / income profile ──
        # High win-rate + low payoff is *by design*.  The real quality metric
        # is whether (win_rate × avg_win) > ((1-win_rate) × avg_loss).
        structure = "credit_income"
        edge = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss)
        p.metrics["credit_edge_per_trade"] = round(edge, 2)
        p.notes.append(f"Credit/income profile: {win_rate:.0f}% WR, edge ₹{edge:,.0f}/trade")

        # Re-score payoff: for credit strategies payoff < 1.0 is expected
        if edge > 0:
            # Positive edge → payoff score floor based on win-rate strength
            credit_pay_floor = min(80, 40 + (win_rate - 50) * 1.5) if win_rate > 50 else 30
            pay_score = max(pay_score, credit_pay_floor)
            wl_score = max(wl_score, credit_pay_floor)
        elif payoff >= 0.3:
            pay_score = max(pay_score, 40)
            wl_score = max(wl_score, 40)
    elif win_rate >= 60:
        structure = "mean_reversion"
        p.notes.append(f"Mean reversion profile: {win_rate:.0f}% win rate")
        # For mean reversion, smaller payoff is acceptable
        if win_rate >= 60 and payoff >= 0.7:
            pay_score = max(pay_score, 65)
    elif win_rate >= 35:
        structure = "trend_following"
        p.notes.append(f"Trend following profile: {win_rate:.0f}% win rate")
    else:
        structure = "low_win_rate"
        p.notes.append(f"Low win rate: {win_rate:.0f}% — needs high payoff ratio")
    p.metrics["structure_type"] = structure

    # Danger zone: 50-55% win rate with bad RR (skip for credit strategies)
    if not is_credit and 50 <= win_rate <= 55 and payoff < 1.0:
        p.notes.append("Danger zone: 50-55% win rate with payoff <1 — edge is razor thin")
        pay_score = min(pay_score, 30)

    # Streak analysis from trades (uses normalised exit PnLs)
    analytics = result.get("trade_analytics", {})
    max_win_streak = analytics.get("max_win_streak", 0)
    max_lose_streak = analytics.get("max_lose_streak", 0)

    if max_win_streak == 0 and max_lose_streak == 0:
        exit_pnls = _extract_exit_pnls(result)
        if exit_pnls:
            max_win_streak, max_lose_streak = _compute_streaks(exit_pnls)

    p.metrics["max_win_streak"] = max_win_streak
    p.metrics["max_lose_streak"] = max_lose_streak

    # Trade count validation
    trade_score = 100 if total_trades >= 30 else (total_trades / 30) * 100
    if total_trades < 10:
        p.notes.append(f"Only {total_trades} trades — insufficient for reliable statistics")
        trade_score = min(trade_score, 20)
    p.metrics["statistical_significance"] = total_trades >= 30

    p.score = 0.35 * pay_score + 0.30 * wl_score + 0.20 * trade_score + 0.15 * min(100, max(0, 100 - max_lose_streak * 8))

    if total_trades < 10:
        p.verdict = "FAIL"
    elif pay_v == "FAIL" and wl_v == "FAIL":
        p.verdict = "FAIL"
    elif pay_v in ("PASS", "WARN") and wl_v in ("PASS", "WARN"):
        p.verdict = "PASS" if pay_score > 60 else "WARN"
    else:
        p.verdict = "WARN"

    return p


# ────────────────────────────────────────────────────────────
# Pillar 4: Robustness
# ────────────────────────────────────────────────────────────

def _compute_robustness(result: dict, th: dict) -> PillarResult:
    p = PillarResult(name="Robustness")

    # Monthly PnL distribution
    analytics = result.get("trade_analytics", {})
    monthly_pnl = analytics.get("monthly_pnl", {})
    monthly_scores = []

    if monthly_pnl:
        values = [v for v in monthly_pnl.values() if isinstance(v, (int, float))]
        if values:
            positive_months = sum(1 for v in values if v > 0)
            total_months = len(values)
            monthly_win_rate = (positive_months / total_months) * 100 if total_months > 0 else 0
            p.metrics["monthly_win_rate"] = round(monthly_win_rate, 1)
            p.metrics["positive_months"] = positive_months
            p.metrics["total_months"] = total_months
            monthly_scores.append(min(100, monthly_win_rate * 1.2))

            # Check if profit concentrated in few months
            if values:
                total_profit = sum(v for v in values if v > 0)
                sorted_profits = sorted([v for v in values if v > 0], reverse=True)
                if total_profit > 0 and len(sorted_profits) >= 3:
                    top3_share = sum(sorted_profits[:3]) / total_profit
                    p.metrics["top3_month_concentration"] = round(top3_share * 100, 1)
                    if top3_share > 0.7:
                        p.notes.append("Profit concentrated in few months — potential overfit")
                        monthly_scores.append(30)
                    else:
                        monthly_scores.append(80)

    # Regime performance (from backtest regime data)
    regime_hist = result.get("regime_history", [])
    if regime_hist:
        regime_pnls: dict[str, list[float]] = {}
        for r in regime_hist:
            regime = r.get("regime", "unknown")
            pnl = r.get("pnl", 0)
            if pnl != 0:
                regime_pnls.setdefault(regime, []).append(pnl)

        regime_profitability = {}
        for regime, pnls in regime_pnls.items():
            avg = sum(pnls) / len(pnls) if pnls else 0
            regime_profitability[regime] = avg

        p.metrics["regime_performance"] = {k: round(v, 2) for k, v in regime_profitability.items()}

        if regime_profitability:
            all_positive = all(v > 0 for v in regime_profitability.values())
            if all_positive:
                monthly_scores.append(90)
                p.notes.append("Profitable across all regimes")
            else:
                negative_regimes = [k for k, v in regime_profitability.items() if v <= 0]
                monthly_scores.append(50)
                p.notes.append(f"Lost money in regimes: {', '.join(negative_regimes)}")

    # Trade distribution stability (are trades spread across the period?)
    exit_pnls = _extract_exit_pnls(result)
    if exit_pnls and len(exit_pnls) > 5:
        p.metrics["trade_count_for_robustness"] = len(exit_pnls)
        # Simple: enough trades = more robust
        count_score = min(100, len(exit_pnls) * 2)
        monthly_scores.append(count_score)

    if monthly_scores:
        p.score = sum(monthly_scores) / len(monthly_scores)
    else:
        p.score = 50  # insufficient data — neutral
        p.notes.append("Insufficient data for robustness analysis")

    if p.score >= 65:
        p.verdict = "PASS"
    elif p.score >= 40:
        p.verdict = "WARN"
    else:
        p.verdict = "FAIL"

    return p


# ────────────────────────────────────────────────────────────
# Pillar 5: Execution Reality
# ────────────────────────────────────────────────────────────

def _compute_execution(result: dict, th: dict) -> PillarResult:
    p = PillarResult(name="Execution Reality")

    # Cost drag
    total_pnl = result.get("total_pnl", result.get("total_return_abs", 0))
    total_costs = result.get("total_costs", 0)
    if isinstance(total_costs, dict):
        total_cost_val = total_costs.get("total", 0)
    else:
        total_cost_val = float(total_costs)

    p.metrics["total_costs"] = round(total_cost_val, 2)
    cost_drag = 0
    if total_pnl != 0:
        cost_drag = (total_cost_val / abs(total_pnl)) * 100 if total_pnl != 0 else 0
    p.metrics["cost_drag_pct"] = round(cost_drag, 2)

    cost_score = 100 if cost_drag < 10 else (80 if cost_drag < 25 else (50 if cost_drag < 50 else 20))

    # Slippage impact
    slippage = 0
    if isinstance(total_costs, dict):
        slippage = total_costs.get("slippage", 0)
    elif total_cost_val > 0:
        # FnO engines return total_costs as a plain float without slippage breakdown.
        # Estimate slippage as ~20% of total costs (industry convention for multi-leg fills).
        slippage = total_cost_val * 0.20
    p.metrics["slippage_cost"] = round(slippage, 2)
    slippage_pct = (slippage / abs(total_pnl)) * 100 if total_pnl != 0 else 0
    p.metrics["slippage_pct"] = round(slippage_pct, 2)

    slip_score = 100 if slippage_pct < 5 else (70 if slippage_pct < 15 else (40 if slippage_pct < 30 else 10))

    # Fill quality (for options — orders may not fill at expected price)
    strategy_type = result.get("engine", "")
    is_credit = result.get("_is_credit_strategy", False)
    if "fno" in strategy_type.lower():
        if is_credit:
            # Defined-risk credit structures trade near ATM strikes (more liquid)
            # and have built-in max-loss.  Smaller penalty.
            p.notes.append("Credit spread: trades near-ATM strikes with defined risk")
            slip_score = min(slip_score, 90)
        else:
            p.notes.append("F&O strategies have significantly higher slippage in illiquid strikes")
            slip_score = min(slip_score, 80)  # inherent penalty

    p.score = 0.50 * cost_score + 0.50 * slip_score

    if p.score >= 65:
        p.verdict = "PASS"
    elif p.score >= 40:
        p.verdict = "WARN"
    else:
        p.verdict = "FAIL"
        p.notes.append("Execution costs eating too much of the edge")

    return p


# ────────────────────────────────────────────────────────────
# Pillar 6: Capital Efficiency
# ────────────────────────────────────────────────────────────

def _compute_capital_efficiency(result: dict, th: dict) -> PillarResult:
    p = PillarResult(name="Capital Efficiency")

    initial = result.get("initial_capital", 0)
    final = result.get("final_capital", 0)
    total_return_pct = result.get("total_return_pct", 0)

    p.metrics["initial_capital"] = round(initial, 2)
    p.metrics["final_capital"] = round(final, 2)
    p.metrics["total_return_pct"] = round(total_return_pct, 2)

    # CAGR
    cagr = result.get("cagr", 0)
    if cagr == 0 and initial > 0 and final > 0:
        # Estimate from equity curve length
        eq = result.get("equity_curve", [])
        bars = len(eq)
        years = bars / 252 if bars > 0 else 1
        if years > 0 and final > initial * 0.1:
            cagr = ((final / initial) ** (1 / years) - 1) * 100
    p.metrics["cagr"] = round(cagr, 2)

    # Return on Margin (for F&O)
    margin_hist = result.get("margin_history", [])
    if margin_hist:
        avg_margin = sum(m.get("margin", 0) for m in margin_hist) / len(margin_hist)
        total_pnl = result.get("total_pnl", result.get("total_return_abs", 0))
        rom = (total_pnl / avg_margin * 100) if avg_margin > 0 else 0
        p.metrics["return_on_margin"] = round(rom, 2)
        p.metrics["avg_margin_used"] = round(avg_margin, 2)
        rom_score, _ = _score_metric(rom, th["rom_good"], th["rom_ok"])
    else:
        rom_score = 50  # neutral for equity
        p.metrics["return_on_margin"] = None

    # Capital utilisation
    eq = result.get("equity_curve", [])
    if len(eq) > 10:
        eq_arr = np.array(eq, dtype=float)
        peak = np.max(eq_arr)
        avg_eq = np.mean(eq_arr)
        util = (avg_eq / peak * 100) if peak > 0 else 0
        p.metrics["capital_utilisation_pct"] = round(util, 1)
    else:
        util = 50

    cagr_score = min(100, max(0, cagr * 3))  # 33% CAGR → 100
    util_score = min(100, util)

    p.score = 0.40 * rom_score + 0.30 * cagr_score + 0.30 * util_score

    if p.score >= 65:
        p.verdict = "PASS"
    elif p.score >= 40:
        p.verdict = "WARN"
    else:
        p.verdict = "FAIL"

    return p


# ────────────────────────────────────────────────────────────
# Pillar 7: Risk Architecture
# ────────────────────────────────────────────────────────────

def _compute_risk_architecture(result: dict, th: dict) -> PillarResult:
    p = PillarResult(name="Risk Architecture")

    settings = result.get("settings", {})
    engine = result.get("engine", "")
    is_fno = "fno" in engine.lower()

    # Position sizing
    position_sizing = result.get("position_sizing", settings.get("position_sizing", "fixed"))
    p.metrics["position_sizing"] = position_sizing
    if position_sizing in ("volatility", "risk_pct", "kelly"):
        sizing_score = 90
        p.notes.append(f"Dynamic position sizing: {position_sizing}")
    else:
        sizing_score = 50
        p.notes.append("Fixed position sizing — consider volatility-based")

    # Exposure control (max positions)
    max_positions = settings.get("max_positions", 0)
    # FnO structures are inherently limited to 1 position at a time
    if max_positions == 0 and is_fno:
        max_positions = 1
    p.metrics["max_positions"] = max_positions
    if 1 <= max_positions <= 5:
        exposure_score = 85
    elif max_positions <= 10:
        exposure_score = 65
    elif max_positions == 0:
        # No max_positions specified — neutral rather than penalty
        exposure_score = 55
    else:
        exposure_score = 40
        p.notes.append("High max positions — consider tightening")

    # Stop loss presence
    has_sl = settings.get("stop_loss_pct", 0) > 0 or result.get("stop_loss_stats", {}).get("sl_exits", 0) > 0
    is_credit = result.get("_is_credit_strategy", False)
    p.metrics["has_stop_loss"] = has_sl
    if is_credit:
        # Credit spreads have built-in defined max loss (spread width)
        sl_score = 90
        p.notes.append("Defined-risk structure — max loss capped by spread width")
        p.metrics["defined_risk"] = True
    elif has_sl:
        sl_score = 85
    else:
        sl_score = 30
        p.notes.append("No stop loss detected — critical risk gap")

    # Profit target
    has_pt = settings.get("profit_target_pct", 0) > 0
    p.metrics["has_profit_target"] = has_pt
    if is_credit and has_pt:
        # Credit strategies with profit target (e.g., 50% of max profit) show discipline
        pt_score = 90
        p.notes.append(f"Profit target at {settings.get('profit_target_pct', 0):.0f}% of max profit")
    elif has_pt:
        pt_score = 80
    else:
        pt_score = 45

    p.score = 0.30 * sizing_score + 0.25 * exposure_score + 0.25 * sl_score + 0.20 * pt_score

    if p.score >= 65:
        p.verdict = "PASS"
    elif p.score >= 40:
        p.verdict = "WARN"
    else:
        p.verdict = "FAIL"
        p.notes.append("Risk controls insufficient for live trading")

    return p


# ────────────────────────────────────────────────────────────
# Pillar 8: Psychological Stability
# ────────────────────────────────────────────────────────────

def _compute_psychological(result: dict, th: dict) -> PillarResult:
    p = PillarResult(name="Psychological Stability")

    # Longest losing streak
    analytics = result.get("trade_analytics", {})
    max_lose = analytics.get("max_lose_streak", 0)

    if max_lose == 0:
        exit_pnls = _extract_exit_pnls(result)
        if exit_pnls:
            _, max_lose = _compute_streaks(exit_pnls)

    p.metrics["max_losing_streak"] = max_lose
    streak_score = 100 if max_lose <= 3 else (80 if max_lose <= 5 else (50 if max_lose <= 8 else (20 if max_lose <= 12 else 5)))

    # Time underwater
    eq = result.get("equity_curve", [])
    max_underwater_bars = 0
    if len(eq) > 1:
        eq_arr = np.array(eq, dtype=float)
        peak = np.maximum.accumulate(eq_arr)
        underwater = eq_arr < peak
        cur_uw = 0
        for uw in underwater:
            if uw:
                cur_uw += 1
                max_underwater_bars = max(max_underwater_bars, cur_uw)
            else:
                cur_uw = 0

    total_bars = len(eq)
    uw_pct = (max_underwater_bars / total_bars * 100) if total_bars > 0 else 0
    p.metrics["max_underwater_bars"] = max_underwater_bars
    p.metrics["underwater_pct"] = round(uw_pct, 1)
    uw_score = 100 if uw_pct < 10 else (70 if uw_pct < 25 else (40 if uw_pct < 50 else 15))

    # Trades per day (for intraday)
    total_trades = result.get("total_trades", 0)
    total_days = total_bars  # rough approximation
    trades_per_day = total_trades / total_days if total_days > 0 else 0
    p.metrics["trades_per_day"] = round(trades_per_day, 2)

    # Worst single trade
    worst_trade = analytics.get("worst_trade", 0)
    best_trade = analytics.get("best_trade", 0)
    if worst_trade == 0:
        exit_pnls = _extract_exit_pnls(result)
        if exit_pnls:
            worst_trade = min(exit_pnls)
            best_trade = max(exit_pnls)
    p.metrics["worst_trade"] = round(worst_trade, 2)
    p.metrics["best_trade"] = round(best_trade, 2)

    # Dependence on single large win
    initial = result.get("initial_capital", 0)
    if initial > 0 and best_trade > 0:
        total_pnl = result.get("total_pnl", result.get("total_return_abs", 0))
        if total_pnl > 0 and best_trade / total_pnl > 0.5:
            p.notes.append("Over 50% of profit comes from a single trade — fragile edge")

    p.score = 0.40 * streak_score + 0.35 * uw_score + 0.25 * 70  # base tradability
    if max_lose > 10:
        p.notes.append(f"Longest losing streak: {max_lose} trades — psychologically challenging")
    if uw_pct > 40:
        p.notes.append(f"Underwater {uw_pct:.0f}% of the time — long recovery periods")

    if p.score >= 65:
        p.verdict = "PASS"
    elif p.score >= 40:
        p.verdict = "WARN"
    else:
        p.verdict = "FAIL"
        p.notes.append("Strategy likely to cause emotional override during live trading")

    return p


# ────────────────────────────────────────────────────────────
# Main Entry Point
# ────────────────────────────────────────────────────────────

def compute_health_report(
    result: dict[str, Any],
    strategy_type: str = "equity",
    strategy_name: str = "",
) -> HealthReport:
    """
    Compute a full health report from backtest / paper trade results.

    Args:
        result: Backtest or paper trade result dict (from BacktestEngine or FnOBacktestEngine)
        strategy_type: "equity", "fno", "intraday", "oi_strategy"
        strategy_name: Name of the strategy

    Returns:
        HealthReport with overall score, per-pillar grades, and execution readiness
    """
    if not result or "error" in result:
        return HealthReport(
            strategy_name=strategy_name,
            strategy_type=strategy_type,
            overall_verdict="FAIL",
            blockers=["No valid backtest result to evaluate"],
        )

    th = _get_thresholds(strategy_type, result)
    name = strategy_name or result.get("strategy_name", "unknown")

    # Tag credit strategy flag so pillar functions can use it
    credit = _is_credit_strategy(result)
    result["_is_credit_strategy"] = credit

    # ── Normalise result dict (fill missing keys across all engine types) ──
    _normalize_result(result)

    # Compute all 8 pillars
    pillars = {
        "profitability": _compute_profitability(result, th),
        "drawdown": _compute_drawdown(result, th),
        "trade_quality": _compute_trade_quality(result, th),
        "robustness": _compute_robustness(result, th),
        "execution": _compute_execution(result, th),
        "capital_efficiency": _compute_capital_efficiency(result, th),
        "risk_architecture": _compute_risk_architecture(result, th),
        "psychological": _compute_psychological(result, th),
    }

    # Pillar weights: credit strategies shift weight towards PF and risk arch,
    # away from trade_quality payoff ratio which penalises them unfairly.
    if credit:
        weights = {
            "profitability": 0.25,
            "drawdown": 0.18,
            "trade_quality": 0.10,      # reduced — payoff ratio is misleading
            "robustness": 0.10,
            "execution": 0.10,
            "capital_efficiency": 0.10,  # margin efficiency matters more
            "risk_architecture": 0.10,  # defined risk is a strength
            "psychological": 0.07,
        }
    else:
        weights = PILLAR_WEIGHTS

    # Overall score = weighted average
    overall = sum(
        pillars[k].score * weights[k]
        for k in weights
    )

    # Execution readiness
    fail_pillars = [k for k, v in pillars.items() if v.verdict == "FAIL"]
    warn_pillars = [k for k, v in pillars.items() if v.verdict == "WARN"]

    blockers = []
    warnings = []
    summary = []

    for k in fail_pillars:
        blockers.append(f"{pillars[k].name}: {', '.join(pillars[k].notes) or 'Below minimum threshold'}")
    for k in warn_pillars:
        warnings.append(f"{pillars[k].name}: {', '.join(pillars[k].notes) or 'Needs improvement'}")

    # Critical blockers override
    critical_fails = {"profitability", "drawdown"} & set(fail_pillars)
    execution_ready = overall >= 60 and len(critical_fails) == 0

    if overall >= 75:
        overall_verdict = "PASS"
        summary.append("Strategy meets professional quality standards")
    elif overall >= 50:
        overall_verdict = "WARN"
        summary.append("Strategy has potential but needs improvement in some areas")
    else:
        overall_verdict = "FAIL"
        summary.append("Strategy does not meet minimum thresholds for live execution")

    if not execution_ready:
        if critical_fails:
            summary.append(f"Critical failures in: {', '.join(pillars[k].name for k in critical_fails)}")
        summary.append("Resolve blockers before deploying to live trading")

    if credit:
        summary.insert(0, f"Credit/income strategy detected — using structure-aware scoring")

    # Clean up internal flag from result dict
    result.pop("_is_credit_strategy", None)

    return HealthReport(
        strategy_name=name,
        strategy_type=strategy_type,
        overall_score=overall,
        overall_verdict=overall_verdict,
        execution_ready=execution_ready,
        pillars=pillars,
        summary=summary,
        warnings=warnings,
        blockers=blockers,
    )


def get_health_grade(score: float) -> str:
    """Convert score to letter grade."""
    if score >= 90:
        return "A+"
    elif score >= 80:
        return "A"
    elif score >= 70:
        return "B+"
    elif score >= 60:
        return "B"
    elif score >= 50:
        return "C"
    elif score >= 40:
        return "D"
    else:
        return "F"
