"""
Journal Analytics Engine — Rolling metrics, edge decay, regime matrix
=====================================================================

Computes ALL the long-run profitability metrics from the journal store:
  - Expectancy (THE most important)
  - Profit Factor
  - Sharpe / Sortino / Calmar ratios
  - Rolling performance (monthly, weekly)
  - Edge stability & decay tracking
  - Regime performance matrix
  - Execution quality analysis
  - Drawdown recovery analysis
  - Slippage drift detection
"""

from __future__ import annotations
import math
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from collections import defaultdict

from src.journal.journal_store import JournalStore
from src.journal.journal_models import JournalEntry

logger = logging.getLogger("journal_analytics")


class JournalAnalytics:
    """
    Computes all advanced analytics from the journal store.
    Designed for both API consumption (dashboard) and programmatic use.
    """

    def __init__(self, store: JournalStore):
        self._store = store

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CORE PROFITABILITY METRICS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def compute_full_analytics(self, strategy: str = "", source: str = "",
                                days: int = 0, instrument: str = "",
                                direction: str = "", trade_type: str = "") -> Dict[str, Any]:
        """
        Comprehensive analytics report — the main endpoint.
        Returns all three layers of metrics.
        """
        entries = self._store.query_entries(
            strategy=strategy, source=source, is_closed=True,
            instrument=instrument, direction=direction, trade_type=trade_type,
            from_date=(datetime.now() - timedelta(days=days)).isoformat() if days else "",
            limit=10000,
        )
        if not entries:
            return {"error": "No closed trades found", "total_trades": 0}

        pnls = [e.net_pnl for e in entries]
        returns = [e.return_pct for e in entries if e.return_pct is not None and e.return_pct != 0]

        # Shared filter kwargs for sub-store calls
        _filt = dict(strategy=strategy, instrument=instrument, source=source,
                     direction=direction, trade_type=trade_type)

        result = {}
        result["total_trades"] = len(entries)
        result["core_metrics"] = self._core_metrics(entries, pnls)
        result["risk_adjusted"] = self._risk_adjusted_metrics(pnls, returns)
        result["edge_stability"] = self._edge_stability(entries)
        result["execution_quality"] = self._execution_quality(entries)
        result["drawdown_analysis"] = self._drawdown_analysis(pnls)
        result["regime_matrix"] = self._store.get_regime_performance_matrix(**_filt, days=days)
        result["slippage_drift"] = self._store.get_slippage_drift(window_days=days or 30, **_filt)
        result["strategy_breakdown"] = self._store.get_pnl_by_strategy(**_filt, days=days)
        result["daily_pnl"] = self._store.get_pnl_by_date(days=days or 365, **_filt)
        result["trade_distribution"] = self._trade_distribution(entries)
        result["cost_analysis"] = self._cost_analysis(entries)
        result["mae_mfe_analysis"] = self._mae_mfe_analysis(entries)
        result["consistency_score"] = self._consistency_score(entries)
        result["edge_decay"] = self._edge_decay(entries)

        return result

    def _core_metrics(self, entries: List[JournalEntry], pnls: List[float]) -> Dict:
        """Core P&L metrics — expectancy, profit factor, win rate."""
        total = len(pnls)
        if total == 0:
            return {}

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_count = len(wins)
        loss_count = len(losses)

        total_profit = sum(wins) if wins else 0
        total_loss = sum(abs(l) for l in losses) if losses else 0

        win_rate = win_count / total if total else 0
        loss_rate = 1 - win_rate
        avg_win = total_profit / win_count if win_count else 0
        avg_loss = total_loss / loss_count if loss_count else 0

        # ── Expectancy (THE MOST IMPORTANT METRIC) ──
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

        # ── Profit Factor ──
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

        # ── Streaks ──
        max_win_streak, max_loss_streak = 0, 0
        current_streak, is_winning = 0, True
        for p in pnls:
            if p > 0:
                if is_winning:
                    current_streak += 1
                else:
                    current_streak = 1
                    is_winning = True
                max_win_streak = max(max_win_streak, current_streak)
            else:
                if not is_winning:
                    current_streak += 1
                else:
                    current_streak = 1
                    is_winning = False
                max_loss_streak = max(max_loss_streak, current_streak)

        # ── Consecutive losses probability ──
        # P(N consecutive losses) = (1 - win_rate)^N
        prob_5_losses = loss_rate ** 5 if loss_rate else 0
        prob_10_losses = loss_rate ** 10 if loss_rate else 0

        return {
            "total_trades": total,
            "winning_trades": win_count,
            "losing_trades": loss_count,
            "win_rate": round(win_rate, 4),
            "loss_rate": round(loss_rate, 4),
            "total_pnl": round(sum(pnls), 2),
            "total_profit": round(total_profit, 2),
            "total_loss": round(total_loss, 2),
            "avg_pnl": round(sum(pnls) / total, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "median_pnl": round(sorted(pnls)[total // 2], 2),
            "best_trade": round(max(pnls), 2),
            "worst_trade": round(min(pnls), 2),
            "expectancy": round(expectancy, 2),
            "profit_factor": round(profit_factor, 4),
            "payoff_ratio": round(avg_win / avg_loss, 4) if avg_loss else 0,
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
            "prob_5_consecutive_losses": round(prob_5_losses * 100, 4),
            "prob_10_consecutive_losses": round(prob_10_losses * 100, 6),
        }

    def _risk_adjusted_metrics(self, pnls: List[float], returns: List[float]) -> Dict:
        """Sharpe, Sortino, Calmar ratios + drawdown metrics."""
        if len(pnls) < 2:
            return {}

        mean_return = sum(returns) / len(returns) if returns else 0
        std_return = self._std(returns) if returns else 0

        # ── Sharpe Ratio (annualized, assuming 252 trading days) ──
        sharpe = (mean_return / std_return * math.sqrt(252)
                  if std_return > 0 else 0)

        # ── Sortino Ratio (downside deviation only) ──
        downside = [r for r in returns if r < 0]
        downside_std = self._std(downside) if len(downside) > 1 else 0
        sortino = (mean_return / downside_std * math.sqrt(252)
                   if downside_std > 0 else 0)

        # ── Max Drawdown ──
        cumulative = []
        running = 0
        for p in pnls:
            running += p
            cumulative.append(running)

        peak = cumulative[0]
        max_dd = 0
        for c in cumulative:
            peak = max(peak, c)
            dd = peak - c
            max_dd = max(max_dd, dd)

        total_pnl = sum(pnls)

        # ── Calmar Ratio (Return / Max DD) — best for options ──
        calmar = abs(total_pnl / max_dd) if max_dd > 0 else 0

        # ── Recovery Factor ──
        recovery_factor = total_pnl / max_dd if max_dd > 0 else 0

        return {
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "calmar_ratio": round(calmar, 4),
            "max_drawdown": round(max_dd, 2),
            "max_drawdown_pct": round((max_dd / max(abs(total_pnl), max_dd, 1)) * 100, 2),
            "recovery_factor": round(recovery_factor, 4),
            "mean_return_pct": round(mean_return, 4),
            "return_std_pct": round(std_return, 4),
            "downside_deviation": round(downside_std, 4),
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # EDGE STABILITY & DECAY TRACKING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _edge_stability(self, entries: List[JournalEntry]) -> Dict:
        """
        Edge stability analysis — monthly expectancy, rolling Sharpe.
        Stable edge > high returns. Unstable edge = danger.
        """
        monthly: Dict[str, List[float]] = defaultdict(list)
        weekly: Dict[str, List[float]] = defaultdict(list)

        for e in entries:
            if e.exit_time:
                try:
                    dt = datetime.fromisoformat(e.exit_time.replace("Z", "+00:00"))
                    month_key = dt.strftime("%Y-%m")
                    week_key = dt.strftime("%Y-W%W")
                    monthly[month_key].append(e.net_pnl)
                    weekly[week_key].append(e.net_pnl)
                except (ValueError, TypeError):
                    pass

        monthly_stats = []
        for month, pnls in sorted(monthly.items()):
            total = sum(pnls)
            wins = sum(1 for p in pnls if p > 0)
            profit = sum(p for p in pnls if p > 0)
            loss = sum(abs(p) for p in pnls if p < 0)
            monthly_stats.append({
                "month": month,
                "trades": len(pnls),
                "pnl": round(total, 2),
                "win_rate": round(wins / len(pnls), 4) if pnls else 0,
                "expectancy": round(total / len(pnls), 2) if pnls else 0,
                "profit_factor": round(profit / loss, 4) if loss > 0 else 0,
            })

        weekly_stats = []
        for week, pnls in sorted(weekly.items()):
            weekly_stats.append({
                "week": week,
                "trades": len(pnls),
                "pnl": round(sum(pnls), 2),
            })

        # Rolling Sharpe (20-trade window)
        all_pnls = [e.net_pnl for e in entries]
        rolling_sharpe = []
        window = 20
        for i in range(window, len(all_pnls)):
            chunk = all_pnls[i - window:i]
            mean = sum(chunk) / len(chunk)
            std = self._std(chunk)
            rs = mean / std if std > 0 else 0
            rolling_sharpe.append(round(rs, 4))

        # Monthly expectancy stability (coefficient of variation)
        m_expects = [s["expectancy"] for s in monthly_stats if s["trades"] >= 3]
        expect_mean = sum(m_expects) / len(m_expects) if m_expects else 0
        expect_std = self._std(m_expects) if len(m_expects) > 1 else 0
        stability_score = 1 - (expect_std / abs(expect_mean)) if expect_mean else 0

        return {
            "monthly_performance": monthly_stats,
            "weekly_performance": weekly_stats[-12:],  # last 12 weeks
            "rolling_sharpe_20": rolling_sharpe[-50:],  # last 50 data points
            "rolling_sharpe": rolling_sharpe[-50:],  # alias for JS
            "monthly_expectancy_stability": round(stability_score, 4),
            "expectancy_stability_score": round(stability_score * 100, 2),  # as %
            "months_profitable": sum(1 for s in monthly_stats if s["pnl"] > 0),
            "months_total": len(monthly_stats),
        }

    def _edge_decay(self, entries: List[JournalEntry]) -> Dict:
        """
        Detect when market adapts to strategy.
        Compare backtest expected vs live realized P&L.
        Also track recent vs historical performance.
        """
        with_backtest = [e for e in entries if e.backtest_expected_pnl != 0]
        decay_data = []
        for e in with_backtest:
            decay_data.append({
                "entry_time": e.entry_time,
                "backtest_pnl": e.backtest_expected_pnl,
                "live_pnl": e.net_pnl,
                "diff": round(e.net_pnl - e.backtest_expected_pnl, 2),
            })

        # Compare first half vs second half performance
        n = len(entries)
        if n >= 10:
            first_half = entries[:n // 2]
            second_half = entries[n // 2:]
            first_expectancy = sum(e.net_pnl for e in first_half) / len(first_half)
            second_expectancy = sum(e.net_pnl for e in second_half) / len(second_half)
            decay_pct = ((second_expectancy - first_expectancy) / abs(first_expectancy) * 100
                         if first_expectancy else 0)
        else:
            first_expectancy = 0
            second_expectancy = 0
            decay_pct = 0

        return {
            "backtest_vs_live": decay_data[-20:],
            "first_half_expectancy": round(first_expectancy, 2),
            "second_half_expectancy": round(second_expectancy, 2),
            "decay_pct": round(decay_pct, 2),
            "expectancy_decay_pct": round(abs(decay_pct), 2),  # alias for JS
            "is_decaying": decay_pct < -20,
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # EXECUTION QUALITY ANALYSIS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _execution_quality(self, entries: List[JournalEntry]) -> Dict:
        """Analyze execution quality — slippage, alpha, latency."""
        slippages = []
        alphas = []
        latencies = []
        fill_rates = []
        cost_ratios = []

        for e in entries:
            ex = e.execution if isinstance(e.execution, dict) else {}
            slip = ex.get("entry_slippage_pct", 0)
            if slip:
                slippages.append(slip)
            alpha = ex.get("execution_alpha", 0)
            if alpha:
                alphas.append(alpha)
            lat = ex.get("signal_to_fill_ms", 0)
            if lat:
                latencies.append(lat)
            partials = ex.get("partial_fill_count", 0)
            if partials:
                fill_rates.append(partials)
            # Cost as % of trade value
            if e.total_costs and e.gross_pnl:
                denom = abs(e.gross_pnl + e.total_costs)
                if denom > 0:
                    cost_ratios.append(abs(e.total_costs / denom) * 100)

        def safe_avg(lst):
            return round(sum(lst) / len(lst), 4) if lst else 0

        return {
            "avg_slippage_pct": safe_avg(slippages),
            "max_slippage_pct": round(max(slippages), 4) if slippages else 0,
            "total_slippage_cost": round(sum(slippages), 4),
            "avg_execution_alpha": safe_avg(alphas),
            "alpha_positive_pct": round(
                sum(1 for a in alphas if a > 0) / len(alphas) * 100, 1
            ) if alphas else 0,
            "avg_latency_ms": safe_avg(latencies),
            "max_latency_ms": round(max(latencies), 1) if latencies else 0,
            "avg_partial_fills": safe_avg(fill_rates),
            "avg_cost_pct_of_pnl": safe_avg(cost_ratios),
            "execution_trades_analyzed": len(slippages),
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # DRAWDOWN & RECOVERY ANALYSIS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _drawdown_analysis(self, pnls: List[float]) -> Dict:
        """
        Drawdown recovery analysis.
        How long until equity makes new high? → Long recovery = risk.
        """
        if not pnls:
            return {}

        cumulative = []
        running = 0
        for p in pnls:
            running += p
            cumulative.append(running)

        peak = cumulative[0]
        drawdowns = []
        current_dd_start = None
        current_dd_depth = 0

        for i, c in enumerate(cumulative):
            if c >= peak:
                if current_dd_start is not None:
                    drawdowns.append({
                        "start_trade": current_dd_start,
                        "end_trade": i,
                        "recovery_trades": i - current_dd_start,
                        "max_depth": round(current_dd_depth, 2),
                    })
                    current_dd_start = None
                    current_dd_depth = 0
                peak = c
            else:
                dd = peak - c
                if current_dd_start is None:
                    current_dd_start = i
                current_dd_depth = max(current_dd_depth, dd)

        # Still in drawdown at end
        if current_dd_start is not None:
            drawdowns.append({
                "start_trade": current_dd_start,
                "end_trade": len(cumulative) - 1,
                "recovery_trades": None,  # still in drawdown
                "max_depth": round(current_dd_depth, 2),
                "still_active": True,
            })

        avg_recovery = (
            sum(d["recovery_trades"] for d in drawdowns if d.get("recovery_trades"))
            / sum(1 for d in drawdowns if d.get("recovery_trades"))
            if any(d.get("recovery_trades") for d in drawdowns) else 0
        )

        return {
            "total_drawdown_periods": len(drawdowns),
            "avg_recovery_trades": round(avg_recovery, 1),
            "longest_recovery": max(
                (d["recovery_trades"] for d in drawdowns if d.get("recovery_trades")),
                default=0
            ),
            "deepest_drawdown": round(max(
                (d["max_depth"] for d in drawdowns), default=0
            ), 2),
            "currently_in_drawdown": bool(drawdowns and drawdowns[-1].get("still_active")),
            "drawdown_periods": drawdowns[-10:],  # last 10
            "equity_curve": [round(c, 2) for c in cumulative],
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TRADE DISTRIBUTION & COST ANALYSIS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _trade_distribution(self, entries: List[JournalEntry]) -> Dict:
        """Distribution analysis — by day, session, direction, trade type."""
        by_day = defaultdict(lambda: {"trades": 0, "pnl": 0, "wins": 0})
        by_session = defaultdict(lambda: {"trades": 0, "pnl": 0, "wins": 0})
        by_direction = defaultdict(lambda: {"trades": 0, "pnl": 0, "wins": 0})
        by_type = defaultdict(lambda: {"trades": 0, "pnl": 0, "wins": 0})
        by_confidence = defaultdict(lambda: {"trades": 0, "pnl": 0, "wins": 0})

        days_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}

        for e in entries:
            ctx = e.strategy_context if isinstance(e.strategy_context, dict) else {}

            # By day of week
            dow = ctx.get("day_of_week")
            if dow is not None:
                day_name = days_map.get(dow, str(dow))
                by_day[day_name]["trades"] += 1
                by_day[day_name]["pnl"] += e.net_pnl
                if e.net_pnl > 0: by_day[day_name]["wins"] += 1

            # By session
            sess = ctx.get("session", "unknown")
            by_session[sess]["trades"] += 1
            by_session[sess]["pnl"] += e.net_pnl
            if e.net_pnl > 0: by_session[sess]["wins"] += 1

            # By direction
            by_direction[e.direction or "unknown"]["trades"] += 1
            by_direction[e.direction or "unknown"]["pnl"] += e.net_pnl
            if e.net_pnl > 0: by_direction[e.direction or "unknown"]["wins"] += 1

            # By trade type
            by_type[e.trade_type or "equity"]["trades"] += 1
            by_type[e.trade_type or "equity"]["pnl"] += e.net_pnl
            if e.net_pnl > 0: by_type[e.trade_type or "equity"]["wins"] += 1

            # By confidence bucket
            conf = ctx.get("signal_confidence", 0)
            if conf:
                bucket = "high" if conf >= 0.7 else ("medium" if conf >= 0.4 else "low")
                by_confidence[bucket]["trades"] += 1
                by_confidence[bucket]["pnl"] += e.net_pnl
                if e.net_pnl > 0: by_confidence[bucket]["wins"] += 1

        def finalize(d):
            result = {}
            for k, v in d.items():
                v["win_rate"] = round(v["wins"] / v["trades"], 4) if v["trades"] else 0
                v["pnl"] = round(v["pnl"], 2)
                v["total_pnl"] = v["pnl"]  # alias for JS
                v["avg_pnl"] = round(v["pnl"] / v["trades"], 2) if v["trades"] else 0
                result[k] = v
            return result

        return {
            "by_day_of_week": finalize(by_day),
            "by_session": finalize(by_session),
            "by_direction": finalize(by_direction),
            "by_trade_type": finalize(by_type),
            "by_confidence": finalize(by_confidence),
        }

    def _cost_analysis(self, entries: List[JournalEntry]) -> Dict:
        """Analyze impact of transaction costs on edge."""
        total_gross = sum(e.gross_pnl for e in entries)
        total_net = sum(e.net_pnl for e in entries)
        total_costs = sum(e.total_costs for e in entries)

        # Cost breakdown aggregation
        brokerage = 0
        stt = 0
        exchange_txn = 0
        gst = 0
        stamp = 0
        slippage = 0

        for e in entries:
            ex = e.execution if isinstance(e.execution, dict) else {}
            costs = ex.get("costs", {})
            if isinstance(costs, dict):
                brokerage += costs.get("brokerage", 0)
                stt += costs.get("stt", 0)
                exchange_txn += costs.get("exchange_txn_charge", 0)
                gst += costs.get("gst", 0)
                stamp += costs.get("stamp_duty", 0)
                slippage += costs.get("slippage_cost", 0)

        # Edge erosion: what % of gross profit is eaten by costs?
        edge_erosion_pct = (total_costs / total_gross * 100) if total_gross > 0 else 0

        return {
            "total_gross_pnl": round(total_gross, 2),
            "total_net_pnl": round(total_net, 2),
            "total_costs": round(total_costs, 2),
            "edge_erosion_pct": round(edge_erosion_pct, 2),
            "avg_cost_per_trade": round(total_costs / len(entries), 2) if entries else 0,
            "breakdown": {
                "brokerage": round(brokerage, 2),
                "stt": round(stt, 2),
                "exchange_txn": round(exchange_txn, 2),
                "gst": round(gst, 2),
                "stamp_duty": round(stamp, 2),
                "slippage": round(slippage, 2),
            }
        }

    def _mae_mfe_analysis(self, entries: List[JournalEntry]) -> Dict:
        """
        MAE/MFE analysis — the most powerful trade diagnostic.
        Shows if you're leaving money on the table or holding losers too long.
        """
        maes = [e.mae for e in entries if e.mae]
        mfes = [e.mfe for e in entries if e.mfe]
        edge_ratios = [e.edge_ratio for e in entries if e.edge_ratio]

        winners = [e for e in entries if e.net_pnl > 0]
        losers = [e for e in entries if e.net_pnl <= 0]

        def safe_avg(lst):
            return round(sum(lst) / len(lst), 2) if lst else 0

        # Winners: how much of MFE did we capture?
        win_mfe_capture = []
        for w in winners:
            if w.mfe and w.mfe > 0:
                capture = abs(w.net_pnl / w.mfe) * 100
                win_mfe_capture.append(capture)

        # Losers: how bad was MAE vs realized loss?
        loss_mae_ratio = []
        for l in losers:
            if l.mae and l.mae > 0:
                ratio = abs(l.net_pnl / l.mae) * 100
                loss_mae_ratio.append(ratio)

        return {
            "avg_mae": safe_avg(maes),
            "avg_mfe": safe_avg(mfes),
            "avg_edge_ratio": safe_avg(edge_ratios),
            "winners_mfe_capture_pct": safe_avg(win_mfe_capture),
            "losers_mae_to_loss_pct": safe_avg(loss_mae_ratio),
            "trades_with_excursion": len(maes),
            "interpretation": (
                "Good MFE capture" if safe_avg(win_mfe_capture) > 60
                else "Leaving money on table — consider trailing stops"
            ) if win_mfe_capture else "No excursion data"
        }

    def _consistency_score(self, entries: List[JournalEntry]) -> Dict:
        """
        Overall system consistency score (0-100).
        Combines edge stability, cost efficiency, execution quality.
        """
        if len(entries) < 10:
            return {"score": 0, "message": "Need 10+ trades for consistency score"}

        pnls = [e.net_pnl for e in entries]

        # 1. Win rate score (0-25)
        win_rate = sum(1 for p in pnls if p > 0) / len(pnls)
        win_score = min(25, win_rate * 50)  # 50% win rate = 25 pts

        # 2. Profit factor score (0-25)
        wins_total = sum(p for p in pnls if p > 0)
        loss_total = sum(abs(p) for p in pnls if p < 0)
        pf = wins_total / loss_total if loss_total else 2.0
        pf_score = min(25, (pf / 2) * 25)  # PF 2.0 = 25 pts

        # 3. Expectancy stability (0-25)
        if len(pnls) >= 20:
            half = len(pnls) // 2
            first_exp = sum(pnls[:half]) / half
            second_exp = sum(pnls[half:]) / (len(pnls) - half)
            if first_exp > 0 and second_exp > 0:
                stability = min(second_exp, first_exp) / max(second_exp, first_exp)
                stab_score = stability * 25
            elif first_exp > 0 or second_exp > 0:
                stab_score = 5
            else:
                stab_score = 0
        else:
            stab_score = 12.5  # neutral

        # 4. Recovery capability (0-25)
        max_dd = 0
        peak = 0
        running = 0
        for p in pnls:
            running += p
            peak = max(peak, running)
            max_dd = max(max_dd, peak - running)
        total_return = sum(pnls)
        recovery_ratio = total_return / max_dd if max_dd else 2.0
        rec_score = min(25, recovery_ratio * 12.5)  # recovery 2x = 25 pts

        total_score = round(win_score + pf_score + stab_score + rec_score, 1)

        rating = (
            "Excellent" if total_score >= 80 else
            "Good" if total_score >= 60 else
            "Fair" if total_score >= 40 else
            "Poor" if total_score >= 20 else
            "Critical"
        )

        return {
            "score": total_score,
            "rating": rating,
            "components": {
                "win_rate_score": round(win_score, 1),
                "profit_factor_score": round(pf_score, 1),
                "stability_score": round(stab_score, 1),
                "recovery_score": round(rec_score, 1),
            },
            "trades_analyzed": len(entries),
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # IV / GREEKS ANALYSIS (F&O specific)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def compute_fno_analytics(self, strategy: str = "", days: int = 0) -> Dict:
        """F&O-specific analytics — IV analysis, Greeks P&L attribution."""
        entries = self._store.query_entries(
            strategy=strategy, trade_type="fno", is_closed=True,
            from_date=(datetime.now() - timedelta(days=days)).isoformat() if days else "",
            limit=5000,
        )
        if not entries:
            return {"error": "No F&O trades found"}

        # IV bucket analysis
        iv_buckets = defaultdict(lambda: {"trades": 0, "pnl": 0, "wins": 0})
        dte_buckets = defaultdict(lambda: {"trades": 0, "pnl": 0, "wins": 0})
        greeks_pnl = {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}

        for e in entries:
            ctx = e.strategy_context if isinstance(e.strategy_context, dict) else {}
            ps = e.position_structure if isinstance(e.position_structure, dict) else {}

            # IV percentile buckets
            iv_pct = ctx.get("iv_percentile", 0)
            if iv_pct:
                bucket = (
                    "0-20" if iv_pct < 20 else
                    "20-40" if iv_pct < 40 else
                    "40-60" if iv_pct < 60 else
                    "60-80" if iv_pct < 80 else
                    "80-100"
                )
                iv_buckets[bucket]["trades"] += 1
                iv_buckets[bucket]["pnl"] += e.net_pnl
                if e.net_pnl > 0: iv_buckets[bucket]["wins"] += 1

            # DTE buckets
            dte = ctx.get("dte", 0)
            if dte:
                dte_bucket = (
                    "0-7" if dte <= 7 else
                    "8-15" if dte <= 15 else
                    "16-30" if dte <= 30 else
                    "30-45" if dte <= 45 else
                    "45+"
                )
                dte_buckets[dte_bucket]["trades"] += 1
                dte_buckets[dte_bucket]["pnl"] += e.net_pnl
                if e.net_pnl > 0: dte_buckets[dte_bucket]["wins"] += 1

            # Greeks contribution (approximate)
            if ps.get("entry_theta"):
                greeks_pnl["theta"] += ps["entry_theta"]

        def finalize(d):
            for v in d.values():
                v["win_rate"] = round(v["wins"] / v["trades"], 4) if v["trades"] else 0
                v["pnl"] = round(v["pnl"], 2)
                v["total_pnl"] = v["pnl"]  # alias for JS
                v["avg_pnl"] = round(v["pnl"] / v["trades"], 2) if v["trades"] else 0
            return dict(d)

        # Convert dict-of-dicts to list-of-dicts with bucket key
        iv_list = [{"bucket": k, **v} for k, v in sorted(finalize(iv_buckets).items())]
        dte_list = [{"bucket": k, **v} for k, v in sorted(finalize(dte_buckets).items())]
        return {
            "total_fno_trades": len(entries),
            "iv_percentile_analysis": iv_list,
            "dte_analysis": dte_list,
            "greeks_pnl_attribution": greeks_pnl,
            "greeks_attribution": greeks_pnl,
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # HELPERS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _std(values: List[float]) -> float:
        if len(values) < 2:
            return 0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return math.sqrt(variance) if variance > 0 else 0
