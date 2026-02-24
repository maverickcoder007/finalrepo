"""Test Strategy Health Report scoring for credit/income F&O strategies.

Verifies that iron_butterfly, iron_condor, etc. are scored with
structure-aware thresholds instead of generic directional thresholds.
"""
import traceback
import sys


def test_iron_butterfly_scoring():
    """Iron butterfly with good results should score well, not poorly."""
    from src.strategy.health_engine import compute_health_report, get_health_grade, _is_credit_strategy

    # Simulate a *good* iron_butterfly result — high win rate, positive PnL
    result = {
        "engine": "fno_backtest",
        "strategy_name": "iron_butterfly",
        "structure": "IRON_BUTTERFLY",
        "underlying": "NIFTY",
        "initial_capital": 500000,
        "final_capital": 545000,
        "total_return_pct": 9.0,
        "total_pnl": 45000,
        "total_trades": 40,
        "winning_trades": 28,
        "losing_trades": 12,
        "win_rate": 70.0,
        "avg_win": 3500,     # small premium collected
        "avg_loss": -4500,   # larger but capped loss
        "profit_factor": 1.82,
        "max_drawdown_pct": 8.5,
        "sharpe_ratio": 1.1,
        "sortino_ratio": 1.4,
        "total_costs": 2800,
        "equity_curve": [500000 + i * 1125 for i in range(40)],
        "trades": [
            {"type": "PROFIT_TARGET", "pnl": 3500, "date": f"2025-{(i % 12)+1:02d}-15"}
            for i in range(28)
        ] + [
            {"type": "STOP_LOSS", "pnl": -4500, "date": f"2025-{(i % 12)+1:02d}-20"}
            for i in range(12)
        ],
        "settings": {
            "max_positions": 3,
            "profit_target_pct": 50.0,
            "stop_loss_pct": 100.0,
            "entry_dte_min": 15,
            "entry_dte_max": 45,
            "delta_target": 0.16,
            "slippage_model": "realistic",
            "use_regime_filter": True,
        },
        "margin_history": [{"margin": 120000} for _ in range(40)],
        "greeks_history": [],
        "regime_history": [],
    }

    # Verify credit detection
    assert _is_credit_strategy(result), "iron_butterfly should be detected as credit strategy"

    report = compute_health_report(result, strategy_type="fno", strategy_name="iron_butterfly")
    rd = report.to_dict()

    print(f"  Overall Score: {rd['overall_score']:.1f}/100")
    print(f"  Grade: {get_health_grade(rd['overall_score'])}")
    print(f"  Verdict: {rd['overall_verdict']}")
    print(f"  Execution Ready: {rd['execution_ready']}")

    for pname, pdata in rd["pillars"].items():
        v = pdata["verdict"]
        s = pdata["score"]
        notes = pdata.get("notes", [])
        emoji = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(v, "?")
        print(f"  {emoji} {pname:25s} {s:5.1f}/100  {v}")
        for n in notes[:2]:
            print(f"     → {n}")

    # Core assertions: a good iron_butterfly should NOT fail
    assert rd["overall_score"] >= 50, f"Score {rd['overall_score']:.1f} too low for a profitable iron_butterfly"
    assert rd["overall_verdict"] != "FAIL", f"Verdict should not be FAIL for a profitable iron_butterfly"

    # Trade Quality should not FAIL just because payoff < 1.0
    tq = rd["pillars"]["trade_quality"]
    assert tq["verdict"] != "FAIL", f"Trade Quality FAIL is wrong — credit spread with 70% WR and PF 1.82"
    assert tq["metrics"].get("structure_type") == "credit_income", f"Should detect credit_income structure"

    # Profitability should not FAIL with PF > 1.0 and positive expectancy
    prof = rd["pillars"]["profitability"]
    assert prof["verdict"] != "FAIL", f"Profitability FAIL is wrong — PF=1.82, positive PnL"

    # Risk Architecture should recognize defined risk
    risk = rd["pillars"]["risk_architecture"]
    assert risk["metrics"].get("defined_risk") is True, "Should detect defined-risk structure"

    # Summary should mention credit strategy
    assert any("credit" in s.lower() or "income" in s.lower() for s in rd["summary"]), \
        "Summary should mention credit/income strategy detection"

    # Cleanup flag should not leak
    assert "_is_credit_strategy" not in result, "Internal flag should be cleaned up"

    print("  [OK] Iron butterfly scored fairly")
    return True


def test_iron_condor_vs_generic_fno():
    """Compare: iron_condor (credit) should score differently from bull_call_spread (directional)."""
    from src.strategy.health_engine import compute_health_report, _is_credit_strategy

    base = {
        "engine": "fno_backtest",
        "underlying": "NIFTY",
        "initial_capital": 500000,
        "final_capital": 530000,
        "total_return_pct": 6.0,
        "total_pnl": 30000,
        "total_trades": 30,
        "winning_trades": 20,
        "losing_trades": 10,
        "win_rate": 66.7,
        "avg_win": 2800,
        "avg_loss": -4100,
        "profit_factor": 1.37,
        "max_drawdown_pct": 10.0,
        "sharpe_ratio": 0.9,
        "sortino_ratio": 1.1,
        "total_costs": 1500,
        "equity_curve": [500000 + i * 1000 for i in range(30)],
        "trades": [{"type": "EXIT", "pnl": 2800}] * 20 + [{"type": "EXIT", "pnl": -4100}] * 10,
        "settings": {"max_positions": 3, "profit_target_pct": 50.0, "stop_loss_pct": 100.0},
        "margin_history": [],
        "greeks_history": [],
        "regime_history": [],
    }

    # Iron condor (credit)
    ic_result = {**base, "strategy_name": "iron_condor", "structure": "IRON_CONDOR"}
    assert _is_credit_strategy(ic_result), "iron_condor should be credit"

    # Bull call spread (directional)
    bcs_result = {**base, "strategy_name": "bull_call_spread", "structure": "BULL_CALL_SPREAD"}
    assert not _is_credit_strategy(bcs_result), "bull_call_spread should NOT be credit"

    ic_report = compute_health_report(ic_result, strategy_type="fno", strategy_name="iron_condor")
    bcs_report = compute_health_report(bcs_result, strategy_type="fno", strategy_name="bull_call_spread")

    ic_score = ic_report.overall_score
    bcs_score = bcs_report.overall_score

    print(f"  iron_condor  score: {ic_score:.1f}/100  ({ic_report.overall_verdict})")
    print(f"  bull_call_sp score: {bcs_score:.1f}/100  ({bcs_report.overall_verdict})")

    # Iron condor should score HIGHER (or at least equal) since it gets credit-appropriate thresholds
    assert ic_score >= bcs_score - 5, \
        f"Iron condor ({ic_score:.1f}) should not score much lower than bull_call_spread ({bcs_score:.1f})"

    print("  [OK] Credit strategy detection works for comparison")
    return True


def test_short_straddle_detected():
    """Short straddle should be detected as credit strategy."""
    from src.strategy.health_engine import _is_credit_strategy

    for sname in ["iron_butterfly", "iron_condor", "short_straddle", "short_strangle",
                   "bull_put_spread", "bear_call_spread"]:
        r = {"strategy_name": sname, "structure": sname.upper()}
        assert _is_credit_strategy(r), f"{sname} should be detected as credit"

    for sname in ["bull_call_spread", "bear_put_spread", "long_straddle",
                   "long_strangle", "protective_put", "covered_call"]:
        r = {"strategy_name": sname, "structure": sname.upper()}
        assert not _is_credit_strategy(r), f"{sname} should NOT be credit"

    print("  [OK] Credit/non-credit detection correct for all structures")
    return True


def test_paper_trade_fno_health():
    """Simulate FnO paper trade result (to_dict format) health scoring."""
    from src.strategy.health_engine import compute_health_report

    result_dict = {
        "engine": "fno_paper_trade",
        "strategy_name": "iron_butterfly",
        "structure_type": "IRON_BUTTERFLY",
        "underlying": "NIFTY",
        "timeframe": "day",
        "initial_capital": 500000,
        "final_capital": 525000,
        "total_return_pct": 5.0,
        "total_pnl": 25000,
        "total_trades": 25,
        "winning_trades": 17,
        "losing_trades": 8,
        "win_rate": 68.0,
        "avg_win": 3000,
        "avg_loss": -3700,
        "profit_factor": 1.72,
        "max_drawdown_pct": 7.0,
        "sharpe_ratio": 0.8,
        "sortino_ratio": 1.0,
        "total_costs": 1800,
        "equity_curve": [500000 + i * 1000 for i in range(25)],
        "orders": [],
        "positions": [],
        "greeks_history": [],
        "margin_history": [],
    }

    report = compute_health_report(result_dict, strategy_type="fno", strategy_name="iron_butterfly")
    rd = report.to_dict()

    print(f"  Paper trade score: {rd['overall_score']:.1f}/100  ({rd['overall_verdict']})")
    assert rd["overall_score"] >= 45, f"Paper trade score {rd['overall_score']:.1f} too low"
    assert rd["overall_verdict"] != "FAIL", "Paper trade with positive PnL should not FAIL"

    print("  [OK] Paper trade F&O health scored fairly")
    return True


if __name__ == "__main__":
    tests = [
        ("Credit strategy detection", test_short_straddle_detected),
        ("Iron butterfly scoring", test_iron_butterfly_scoring),
        ("Iron condor vs directional", test_iron_condor_vs_generic_fno),
        ("Paper trade FnO health", test_paper_trade_fno_health),
    ]

    passed = failed = 0
    for name, fn in tests:
        print(f"\n=== {name} ===")
        try:
            fn()
            passed += 1
        except Exception:
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
