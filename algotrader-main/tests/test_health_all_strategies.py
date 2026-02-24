"""
Test health scoring across ALL strategy profiles.

Profiles:
  1. Equity trend-following  (EMA crossover — 38% WR, high payoff)
  2. Equity mean-reversion   (65% WR, low payoff)
  3. Intraday equity         (many trades, tight stops)
  4. FnO directional         (bull_call_spread from FnOBacktestEngine)
  5. FnO credit              (iron_butterfly from FnOBacktestEngine)
  6. FnO paper trade dir.    (bear_put_spread from FnOPaperTradingEngine)
  7. FnO paper trade credit  (iron_condor from FnOPaperTradingEngine)
  8. Equity paper trade      (same shape as equity backtest)

A **profitable** strategy must NEVER score FAIL overall.
"""
from __future__ import annotations
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.strategy.health_engine import compute_health_report, _extract_exit_pnls

# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────

def _equity_curve(initial: float, trades_pnl: list[float]) -> list[float]:
    """Build equity curve from trade PnLs."""
    curve = [initial]
    for pnl in trades_pnl:
        curve.append(curve[-1] + pnl)
    return curve

def _trade_list(pnls: list[float]) -> list[dict]:
    """Simple equity backtest trade list."""
    return [{"pnl": p, "date": f"2024-{1 + i % 12:02d}-15"} for i, p in enumerate(pnls)]

def _fno_backtest_trades(pnls: list[float]) -> list[dict]:
    """FnO backtest trades: mixed ENTRY + EXIT types."""
    trades = []
    for i, p in enumerate(pnls):
        trades.append({"type": "ENTRY", "date": f"2024-{1 + i % 12:02d}-01"})
        exit_type = "PROFIT_TARGET" if p > 0 else "STOP_LOSS"
        trades.append({"type": exit_type, "pnl": p, "date": f"2024-{1 + i % 12:02d}-15"})
    return trades

def _fno_paper_positions(pnls: list[float]) -> list[dict]:
    """FnO paper trade positions list."""
    return [{"pnl": p, "exit": f"2024-{1 + i % 12:02d}-15"} for i, p in enumerate(pnls)]

def _margin_history_backtest(n: int, margin: float) -> list[dict]:
    """Margin history for FnO backtest (uses 'margin' key)."""
    return [{"margin": margin, "date": f"2024-{1 + i % 12:02d}-01"} for i in range(n)]

def _margin_history_paper(n: int, margin: float) -> list[dict]:
    """Margin history for FnO paper trade (uses 'margin_required' key)."""
    return [{"margin_required": margin, "date": f"2024-{1 + i % 12:02d}-01"} for i in range(n)]


# ────────────────────────────────────────────────────────────
# 1. Equity trend-following
# ────────────────────────────────────────────────────────────

def _make_equity_trend() -> dict:
    """EMA crossover: 38% win rate, payoff ~2.5."""
    pnls = []
    for i in range(50):
        if i % 5 in (0, 3):  # ~40% wins
            pnls.append(2500)  # big win
        else:
            pnls.append(-1000)  # small loss
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    total_pnl = sum(pnls)
    curve = _equity_curve(1_000_000, pnls)
    return {
        "engine": "backtest",
        "strategy_name": "ema_crossover",
        "initial_capital": 1_000_000,
        "final_capital": 1_000_000 + total_pnl,
        "total_pnl": total_pnl,
        "total_return_pct": total_pnl / 1_000_000 * 100,
        "total_trades": 50,
        "win_rate": len(wins) / 50 * 100,
        "avg_win": sum(wins) / len(wins),
        "avg_loss": sum(losses) / len(losses),
        "profit_factor": sum(wins) / abs(sum(losses)),
        "sharpe_ratio": 1.1,
        "sortino_ratio": 1.5,
        "max_drawdown_pct": 5.0,
        "equity_curve": curve,
        "trades": _trade_list(pnls),
        "trade_analytics": {
            "monthly_pnl": {"2024-01": 1500, "2024-02": -500, "2024-03": 2000, "2024-04": 1000},
            "max_win_streak": 3,
            "max_lose_streak": 4,
            "best_trade": max(pnls),
            "worst_trade": min(pnls),
        },
        "stop_loss_stats": {"sl_exits": 20, "tp_exits": 15},
        "total_costs": {"total": 5000, "brokerage": 2500, "slippage": 1500, "stt": 1000},
        "position_sizing": "risk_pct",
    }


# ────────────────────────────────────────────────────────────
# 2. Equity mean-reversion
# ────────────────────────────────────────────────────────────

def _make_equity_mean_rev() -> dict:
    """Mean reversion: 65% win rate, payoff ~0.8."""
    pnls = []
    for i in range(60):
        if i % 10 < 7:  # ~70% wins
            pnls.append(800)
        else:
            pnls.append(-1000)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    total_pnl = sum(pnls)
    curve = _equity_curve(500_000, pnls)
    return {
        "engine": "backtest",
        "strategy_name": "mean_reversion",
        "initial_capital": 500_000,
        "final_capital": 500_000 + total_pnl,
        "total_pnl": total_pnl,
        "total_return_pct": total_pnl / 500_000 * 100,
        "total_trades": 60,
        "win_rate": len(wins) / 60 * 100,
        "avg_win": sum(wins) / len(wins),
        "avg_loss": sum(losses) / len(losses),
        "profit_factor": sum(wins) / abs(sum(losses)),
        "sharpe_ratio": 0.9,
        "sortino_ratio": 1.2,
        "max_drawdown_pct": 4.0,
        "equity_curve": curve,
        "trades": _trade_list(pnls),
        "total_costs": {"total": 3000, "brokerage": 1500, "slippage": 1000, "stt": 500},
        "stop_loss_stats": {"sl_exits": 10, "tp_exits": 30},
    }


# ────────────────────────────────────────────────────────────
# 3. Intraday equity
# ────────────────────────────────────────────────────────────

def _make_intraday() -> dict:
    """Intraday: 55% win rate, payoff ~1.1, many trades."""
    pnls = []
    for i in range(120):
        if i % 20 < 11:  # ~55%
            pnls.append(500)
        else:
            pnls.append(-450)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    total_pnl = sum(pnls)
    curve = _equity_curve(200_000, pnls)
    return {
        "engine": "backtest",
        "strategy_name": "vwap_breakout",
        "initial_capital": 200_000,
        "final_capital": 200_000 + total_pnl,
        "total_pnl": total_pnl,
        "total_return_pct": total_pnl / 200_000 * 100,
        "total_trades": 120,
        "win_rate": len(wins) / 120 * 100,
        "avg_win": sum(wins) / len(wins),
        "avg_loss": sum(losses) / len(losses),
        "profit_factor": sum(wins) / abs(sum(losses)),
        "sharpe_ratio": 0.8,
        "sortino_ratio": 1.0,
        "max_drawdown_pct": 3.0,
        "equity_curve": curve,
        "trades": _trade_list(pnls),
        "total_costs": {"total": 8000, "brokerage": 4000, "slippage": 3000, "stt": 1000},
        "stop_loss_stats": {"sl_exits": 40, "tp_exits": 50},
    }


# ────────────────────────────────────────────────────────────
# 4. FnO directional (FnO Backtest Engine shape)
# ────────────────────────────────────────────────────────────

def _make_fno_directional() -> dict:
    """Bull call spread from FnOBacktestEngine: mixed ENTRY/EXIT trades, margin key."""
    pnls = []
    for i in range(40):
        if i % 5 < 2:  # ~40% wins
            pnls.append(4000)
        else:
            pnls.append(-1500)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    total_pnl = sum(pnls)
    curve = _equity_curve(500_000, pnls)
    return {
        "engine": "fno_backtest",
        "strategy_name": "bull_call_spread",
        "structure": "bull_call_spread",
        "initial_capital": 500_000,
        "final_capital": 500_000 + total_pnl,
        "total_pnl": total_pnl,
        "total_return_pct": total_pnl / 500_000 * 100,
        "total_trades": 40,
        "win_rate": len(wins) / 40 * 100,
        "avg_win": sum(wins) / len(wins),
        "avg_loss": sum(losses) / len(losses),
        "profit_factor": sum(wins) / abs(sum(losses)),
        "sharpe_ratio": 0.95,
        # NO sortino_ratio — should be computed by normalizer
        "max_drawdown_pct": 6.0,
        "equity_curve": curve,
        "trades": _fno_backtest_trades(pnls),  # mixed ENTRY + EXIT
        # NO trade_analytics — should be synthesized
        "total_costs": 6000.0,  # plain float, not dict
        "margin_history": _margin_history_backtest(40, 80_000),
        "settings": {"stop_loss_pct": 2.0, "profit_target_pct": 3.0, "max_positions": 2},
    }


# ────────────────────────────────────────────────────────────
# 5. FnO credit (FnO Backtest Engine shape)
# ────────────────────────────────────────────────────────────

def _make_fno_credit() -> dict:
    """Iron butterfly from FnOBacktestEngine."""
    pnls = []
    for i in range(50):
        if i % 10 < 7:  # 70% wins
            pnls.append(3000)
        else:
            pnls.append(-5000)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    total_pnl = sum(pnls)
    curve = _equity_curve(500_000, pnls)
    return {
        "engine": "fno_backtest",
        "strategy_name": "iron_butterfly",
        "structure": "iron_butterfly",
        "initial_capital": 500_000,
        "final_capital": 500_000 + total_pnl,
        "total_pnl": total_pnl,
        "total_return_pct": total_pnl / 500_000 * 100,
        "total_trades": 50,
        "win_rate": len(wins) / 50 * 100,
        "avg_win": sum(wins) / len(wins),
        "avg_loss": sum(losses) / len(losses),
        "profit_factor": sum(wins) / abs(sum(losses)),
        "sharpe_ratio": 0.7,
        "max_drawdown_pct": 8.0,
        "equity_curve": curve,
        "trades": _fno_backtest_trades(pnls),
        "total_costs": 4500.0,
        "margin_history": _margin_history_backtest(50, 120_000),
        "settings": {"stop_loss_pct": 0, "profit_target_pct": 50, "max_positions": 1},
    }


# ────────────────────────────────────────────────────────────
# 6. FnO paper trade directional (FnO Paper Trading Engine shape)
# ────────────────────────────────────────────────────────────

def _make_fno_paper_directional() -> dict:
    """Bear put spread from FnOPaperTradingEngine: NO trades, uses positions + orders."""
    pnls = []
    for i in range(30):
        if i % 5 < 2:
            pnls.append(3500)
        else:
            pnls.append(-1200)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    total_pnl = sum(pnls)
    curve = _equity_curve(300_000, pnls)
    return {
        "engine": "fno_paper_trade",
        "strategy_name": "bear_put_spread",
        "structure_type": "bear_put_spread",  # NOT "structure" — paper trade key
        # NO "settings" dict
        # NO "trades" key
        # NO "sortino_ratio"
        "initial_capital": 300_000,
        "final_capital": 300_000 + total_pnl,
        "total_pnl": total_pnl,
        "total_return_pct": total_pnl / 300_000 * 100,
        "total_trades": 30,
        "win_rate": len(wins) / 30 * 100,
        "avg_win": sum(wins) / len(wins),
        "avg_loss": sum(losses) / len(losses),
        "profit_factor": sum(wins) / abs(sum(losses)),
        "sharpe_ratio": 0.85,
        "max_drawdown_pct": 5.0,
        "equity_curve": curve,
        "positions": _fno_paper_positions(pnls),  # positions, not trades
        "orders": [],
        "total_costs": 3000.0,
        "margin_history": _margin_history_paper(30, 60_000),  # uses "margin_required" key
    }


# ────────────────────────────────────────────────────────────
# 7. FnO paper trade credit (FnO Paper Trading Engine shape)
# ────────────────────────────────────────────────────────────

def _make_fno_paper_credit() -> dict:
    """Iron condor from FnOPaperTradingEngine."""
    pnls = []
    for i in range(45):
        if i % 10 < 7:
            pnls.append(2500)
        else:
            pnls.append(-4000)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    total_pnl = sum(pnls)
    curve = _equity_curve(500_000, pnls)
    return {
        "engine": "fno_paper_trade",
        "strategy_name": "iron_condor",
        "structure_type": "iron_condor",  # paper trade key
        # NO "settings", NO "trades", NO "sortino_ratio"
        "initial_capital": 500_000,
        "final_capital": 500_000 + total_pnl,
        "total_pnl": total_pnl,
        "total_return_pct": total_pnl / 500_000 * 100,
        "total_trades": 45,
        "win_rate": len(wins) / 45 * 100,
        "avg_win": sum(wins) / len(wins),
        "avg_loss": sum(losses) / len(losses),
        "profit_factor": sum(wins) / abs(sum(losses)),
        "sharpe_ratio": 0.6,
        "max_drawdown_pct": 7.0,
        "equity_curve": curve,
        "positions": _fno_paper_positions(pnls),
        "orders": [],
        "total_costs": 5000.0,
        "margin_history": _margin_history_paper(45, 100_000),
    }


# ────────────────────────────────────────────────────────────
# 8. Equity paper trade (same shape as equity backtest)
# ────────────────────────────────────────────────────────────

def _make_equity_paper() -> dict:
    """Equity paper trade — identical to equity backtest + engine tag."""
    result = _make_equity_trend()
    result["engine"] = "paper_trade"
    return result


# ════════════════════════════════════════════════════════════
# TEST FUNCTIONS
# ════════════════════════════════════════════════════════════

def _report_summary(report) -> str:
    lines = [
        f"  Overall: {report.overall_score:.1f}/100 ({report.overall_verdict})",
        f"  Execution-ready: {report.execution_ready}",
    ]
    for key, pillar in report.pillars.items():
        notes = "; ".join(pillar.notes) if pillar.notes else ""
        lines.append(f"    {pillar.name:25s}  {pillar.score:5.1f}  {pillar.verdict:4s}  {notes}")
    if report.blockers:
        lines.append(f"  BLOCKERS: {report.blockers}")
    if report.warnings:
        lines.append(f"  WARNINGS: {report.warnings}")
    return "\n".join(lines)


def test_equity_trend():
    result = _make_equity_trend()
    report = compute_health_report(result, strategy_type="equity", strategy_name="ema_crossover")
    print(f"\n{'='*60}\n1. EQUITY TREND (ema_crossover)")
    print(_report_summary(report))
    assert report.overall_verdict != "FAIL", f"Profitable equity trend should not FAIL: {report.overall_score:.1f}"
    assert report.overall_score >= 60, f"Score too low: {report.overall_score:.1f}"


def test_equity_mean_reversion():
    result = _make_equity_mean_rev()
    report = compute_health_report(result, strategy_type="equity", strategy_name="mean_reversion")
    print(f"\n{'='*60}\n2. EQUITY MEAN-REVERSION")
    print(_report_summary(report))
    assert report.overall_verdict != "FAIL", f"Profitable mean-reversion should not FAIL: {report.overall_score:.1f}"
    assert report.overall_score >= 55, f"Score too low: {report.overall_score:.1f}"


def test_intraday():
    result = _make_intraday()
    report = compute_health_report(result, strategy_type="intraday", strategy_name="vwap_breakout")
    print(f"\n{'='*60}\n3. INTRADAY EQUITY")
    print(_report_summary(report))
    assert report.overall_verdict != "FAIL", f"Profitable intraday should not FAIL: {report.overall_score:.1f}"
    assert report.overall_score >= 50, f"Score too low: {report.overall_score:.1f}"


def test_fno_directional():
    result = _make_fno_directional()
    report = compute_health_report(result, strategy_type="fno", strategy_name="bull_call_spread")
    print(f"\n{'='*60}\n4. FnO DIRECTIONAL (bull_call_spread) — FnO Backtest Engine")
    print(_report_summary(report))

    # Verify _extract_exit_pnls filters ENTRY trades
    exit_pnls = _extract_exit_pnls(result)
    entry_count = sum(1 for t in result["trades"] if t.get("type") == "ENTRY")
    assert len(exit_pnls) == 40, f"Should have 40 exit PnLs, got {len(exit_pnls)} (entries: {entry_count})"

    assert report.overall_verdict != "FAIL", f"Profitable FnO directional should not FAIL: {report.overall_score:.1f}"
    assert report.overall_score >= 50, f"Score too low: {report.overall_score:.1f}"


def test_fno_credit():
    result = _make_fno_credit()
    report = compute_health_report(result, strategy_type="fno", strategy_name="iron_butterfly")
    print(f"\n{'='*60}\n5. FnO CREDIT (iron_butterfly) — FnO Backtest Engine")
    print(_report_summary(report))
    assert report.overall_verdict != "FAIL", f"Profitable iron_butterfly should not FAIL: {report.overall_score:.1f}"
    assert report.overall_score >= 60, f"Score too low: {report.overall_score:.1f}"


def test_fno_paper_directional():
    result = _make_fno_paper_directional()
    report = compute_health_report(result, strategy_type="fno", strategy_name="bear_put_spread")
    print(f"\n{'='*60}\n6. FnO PAPER TRADE DIRECTIONAL (bear_put_spread)")
    print(_report_summary(report))

    # Key verifications for paper trade shape
    # 1. Exit PnLs should come from positions
    exit_pnls = _extract_exit_pnls(result)
    assert len(exit_pnls) == 30, f"Should extract 30 PnLs from positions, got {len(exit_pnls)}"

    # 2. Margin should be normalised ("margin_required" → "margin")
    for m in result.get("margin_history", []):
        assert "margin" in m, "margin_history should be normalised to have 'margin' key"

    # 3. Sortino should be computed from equity curve
    assert result.get("sortino_ratio", 0) != 0, "sortino_ratio should be computed by normalizer"

    assert report.overall_verdict != "FAIL", f"Profitable FnO paper dir should not FAIL: {report.overall_score:.1f}"
    assert report.overall_score >= 50, f"Score too low: {report.overall_score:.1f}"


def test_fno_paper_credit():
    result = _make_fno_paper_credit()
    report = compute_health_report(result, strategy_type="fno", strategy_name="iron_condor")
    print(f"\n{'='*60}\n7. FnO PAPER TRADE CREDIT (iron_condor)")
    print(_report_summary(report))
    assert report.overall_verdict != "FAIL", f"Profitable iron_condor paper should not FAIL: {report.overall_score:.1f}"
    assert report.overall_score >= 55, f"Score too low: {report.overall_score:.1f}"


def test_equity_paper():
    result = _make_equity_paper()
    report = compute_health_report(result, strategy_type="equity", strategy_name="ema_crossover")
    print(f"\n{'='*60}\n8. EQUITY PAPER TRADE")
    print(_report_summary(report))
    assert report.overall_verdict != "FAIL", f"Equity paper trade should not FAIL: {report.overall_score:.1f}"
    assert report.overall_score >= 60, f"Score too low: {report.overall_score:.1f}"


# ────────────────────────────────────────────────────────────
# Edge-case tests
# ────────────────────────────────────────────────────────────

def test_extract_exit_pnls_fno_backtest():
    """ENTRY trades should not pollute PnL list."""
    trades = [
        {"type": "ENTRY"},
        {"type": "PROFIT_TARGET", "pnl": 500},
        {"type": "ENTRY"},
        {"type": "STOP_LOSS", "pnl": -200},
    ]
    pnls = _extract_exit_pnls({"trades": trades})
    assert pnls == [500, -200], f"Expected [500, -200], got {pnls}"


def test_extract_exit_pnls_paper_trade():
    """Paper trade: PnLs from positions, not trades."""
    result = {
        "positions": [{"pnl": 100}, {"pnl": -50}],
        "orders": [],
    }
    pnls = _extract_exit_pnls(result)
    assert pnls == [100, -50], f"Expected [100, -50], got {pnls}"


def test_margin_key_normalisation():
    """margin_required → margin normalisation."""
    from src.strategy.health_engine import _normalize_result
    result = {
        "engine": "fno_paper_trade",
        "equity_curve": [100_000, 110_000],
        "margin_history": [{"margin_required": 50000}],
    }
    _normalize_result(result)
    assert result["margin_history"][0].get("margin") == 50000


if __name__ == "__main__":
    tests = [
        test_equity_trend,
        test_equity_mean_reversion,
        test_intraday,
        test_fno_directional,
        test_fno_credit,
        test_fno_paper_directional,
        test_fno_paper_credit,
        test_equity_paper,
        test_extract_exit_pnls_fno_backtest,
        test_extract_exit_pnls_paper_trade,
        test_margin_key_normalisation,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  ✅ {t.__name__} PASSED")
        except AssertionError as e:
            failed += 1
            print(f"  ❌ {t.__name__} FAILED: {e}")
        except Exception as e:
            failed += 1
            print(f"  ❌ {t.__name__} ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed == 0:
        print("🎉 ALL TESTS PASSED")
    else:
        print("⚠️  SOME TESTS FAILED")
