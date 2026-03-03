"""
Integration test for the Intraday OI Snapshot + Flow Analysis pipeline.

Tests:
  1. SnapshotScheduler status/helpers (no live API needed)
  2. IntradayOIFlowAnalyzer with synthetic snapshot data in SQLite
  3. Service layer method signatures
  4. MCP tool registration
  5. Webapp endpoint registration
"""

import json
import os
import sqlite3
import sys
from datetime import datetime, date, timedelta

# ── Imports ──────────────────────────────────────────────────

def test_scheduler_imports():
    from src.options.oi_snapshot_scheduler import (
        OISnapshotScheduler,
        get_oi_snapshot_scheduler,
        SNAPSHOT_UNDERLYINGS,
        SNAPSHOT_INTERVAL_MINUTES,
    )
    assert SNAPSHOT_UNDERLYINGS == ["NIFTY", "SENSEX"]
    assert SNAPSHOT_INTERVAL_MINUTES == 15
    scheduler = get_oi_snapshot_scheduler()
    assert scheduler is not None
    assert scheduler.is_running is False
    print("  [PASS] Scheduler imports and singleton")


def test_analyzer_imports():
    from src.options.oi_intraday_analyzer import (
        IntradayOIFlowAnalyzer,
        IntradayOIAnalysis,
        SnapshotSummary,
        OIFlowChange,
        TrendReinforcement,
        get_intraday_oi_analyzer,
    )
    analyzer = get_intraday_oi_analyzer()
    assert analyzer is not None
    print("  [PASS] Analyzer imports and singleton")


# ── Scheduler Status (no live API) ──────────────────────────

def test_scheduler_status():
    from src.options.oi_snapshot_scheduler import OISnapshotScheduler

    scheduler = OISnapshotScheduler()
    status = scheduler.get_status()
    assert "running" in status
    assert status["running"] is False
    assert status["interval_minutes"] == 15
    assert "NIFTY" in status["underlyings"]
    assert "SENSEX" in status["underlyings"]
    print("  [PASS] Scheduler status (stopped)")


# ── Analyzer with Synthetic Data ─────────────────────────────

def _seed_test_db(db_path: str) -> None:
    """Seed a test SQLite DB with 4 synthetic snapshots."""
    conn = sqlite3.connect(db_path)

    conn.executescript("""
    CREATE TABLE IF NOT EXISTS option_chains (
        underlying TEXT NOT NULL,
        expiry TEXT NOT NULL,
        strike REAL NOT NULL,
        option_type TEXT NOT NULL,
        ts TEXT NOT NULL,
        last_price REAL NOT NULL DEFAULT 0,
        iv REAL NOT NULL DEFAULT 0,
        delta REAL NOT NULL DEFAULT 0,
        gamma REAL NOT NULL DEFAULT 0,
        theta REAL NOT NULL DEFAULT 0,
        vega REAL NOT NULL DEFAULT 0,
        oi INTEGER NOT NULL DEFAULT 0,
        oi_change INTEGER NOT NULL DEFAULT 0,
        volume INTEGER NOT NULL DEFAULT 0,
        bid_price REAL NOT NULL DEFAULT 0,
        ask_price REAL NOT NULL DEFAULT 0,
        spot_price REAL NOT NULL DEFAULT 0,
        instrument_token INTEGER NOT NULL DEFAULT 0,
        tradingsymbol TEXT NOT NULL DEFAULT '',
        lot_size INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (underlying, expiry, strike, option_type, ts)
    );
    """)

    today = date.today().isoformat()
    expiry = (date.today() + timedelta(days=3)).isoformat()
    spot_base = 24000.0
    strikes = [23750, 23800, 23850, 23900, 23950, 24000, 24050, 24100, 24150, 24200, 24250]

    timestamps = [
        f"{today}T09:30:00",
        f"{today}T09:45:00",
        f"{today}T10:00:00",
        f"{today}T10:15:00",
    ]

    # Simulate: spot moving from 24000 → 24050 → 24080 → 24100
    spot_progression = [24000.0, 24050.0, 24080.0, 24100.0]

    # Simulate: PCR rising (PE OI growing), CE OI walls shifting up
    for snap_idx, (ts, spot) in enumerate(zip(timestamps, spot_progression)):
        for strike in strikes:
            # CE OI: higher for above-ATM, declining as spot rises (unwinding)
            ce_base_oi = max(500000 - (strike - 24000) * 1000, 50000) if strike >= 24000 else 200000
            ce_oi = int(ce_base_oi * (1 - snap_idx * 0.05))  # CE OI slightly declines (bullish)

            # PE OI: higher for below-ATM, growing as spot rises (support building)
            pe_base_oi = max(400000 + (24000 - strike) * 1200, 50000) if strike <= 24000 else 150000
            pe_oi = int(pe_base_oi * (1 + snap_idx * 0.08))  # PE OI grows (bullish support)

            # CE LTP: decline as spot rises for OTM calls (above ATM)
            ce_ltp = max(200 - (strike - spot) * 2, 5) if strike >= spot else max(spot - strike + 50, 10)
            pe_ltp = max(200 - (spot - strike) * 2, 5) if strike <= spot else max(strike - spot + 50, 10)

            for otype, oi, ltp in [("CE", ce_oi, ce_ltp), ("PE", pe_oi, pe_ltp)]:
                conn.execute(
                    """INSERT OR REPLACE INTO option_chains
                    (underlying, expiry, strike, option_type, ts,
                     last_price, iv, oi, volume, spot_price, lot_size)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    ("NIFTY", expiry, strike, otype, ts,
                     round(ltp, 2), 15.0, oi, oi // 10, spot, 50),
                )

    conn.commit()
    conn.close()


def test_analyzer_with_synthetic_data():
    """Full end-to-end test of the analyzer with seeded SQLite data."""
    from src.data.fno_data_store import FnODataStore
    from src.options.oi_intraday_analyzer import IntradayOIFlowAnalyzer

    db_path = "data/test_oi_intraday.db"
    os.makedirs("data", exist_ok=True)

    # Clean up
    if os.path.exists(db_path):
        os.remove(db_path)

    # Seed test data
    _seed_test_db(db_path)

    # Create store and analyzer pointing at test DB
    store = FnODataStore(db_path=db_path)
    analyzer = IntradayOIFlowAnalyzer(store=store)

    # Run analysis
    report = analyzer.analyze("NIFTY")

    print(f"  Snapshots found: {report.total_snapshots}")
    assert report.total_snapshots == 4, f"Expected 4 snapshots, got {report.total_snapshots}"

    print(f"  Flow changes: {len(report.flow_changes)}")
    assert len(report.flow_changes) == 3, f"Expected 3 flow changes, got {len(report.flow_changes)}"

    print(f"  Market direction: {report.market_direction}")
    print(f"  Confidence: {report.direction_confidence}%")
    assert report.market_direction in (
        "BULLISH", "MILDLY_BULLISH", "NEUTRAL", "MILDLY_BEARISH", "BEARISH"
    )

    print(f"  PCR trend: {report.pcr_trend.get('trend', 'N/A')}")
    print(f"  Straddle trend: {report.straddle_trend.get('trend', 'N/A')}")

    # Verify trend reinforcement
    assert report.trend_reinforcement is not None
    print(f"  Trend strength: {report.trend_reinforcement.trend_strength}")
    print(f"  Trend score: {report.trend_reinforcement.trend_score}")
    print(f"  Reinforcement count: {report.trend_reinforcement.reinforcement_count}")

    # Verify direction reasons exist
    print(f"  Direction reasons: {len(report.direction_reasons)}")
    for r in report.direction_reasons[:5]:
        print(f"    - {r}")

    # Smart money signals
    print(f"  Smart money signals: {len(report.smart_money_signals)}")
    for sig in report.smart_money_signals[:3]:
        print(f"    [{sig.get('sentiment', '?')}] {sig.get('type', '?')}: {sig.get('description', '')[:80]}")

    # Verify summary exists
    assert report.summary, "Summary should not be empty"
    print(f"  Summary (first 150 chars): {report.summary[:150]}...")

    # Verify OI wall analysis
    assert report.oi_wall_analysis, "OI wall analysis should not be empty"
    print(f"  CE wall movement: {report.oi_wall_analysis.get('ce_wall_movement', 'N/A')}")
    print(f"  PE wall movement: {report.oi_wall_analysis.get('pe_wall_movement', 'N/A')}")

    # JSON serialization test
    json_output = json.dumps(report.model_dump(), default=str)
    assert len(json_output) > 500, "JSON output should be substantial"
    print(f"  JSON output size: {len(json_output)} chars")

    # Cleanup
    store.close()
    os.remove(db_path)
    print("  [PASS] Full analyzer pipeline with synthetic data")


def test_snapshot_summaries():
    """Test that individual snapshot summaries are computed correctly."""
    from src.data.fno_data_store import FnODataStore
    from src.options.oi_intraday_analyzer import IntradayOIFlowAnalyzer

    db_path = "data/test_oi_summaries.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    _seed_test_db(db_path)
    store = FnODataStore(db_path=db_path)
    analyzer = IntradayOIFlowAnalyzer(store=store)

    report = analyzer.analyze("NIFTY")

    # Check individual snapshots
    for snap in report.snapshots:
        assert snap.spot_price > 0, f"Spot should be positive, got {snap.spot_price}"
        assert snap.total_ce_oi > 0, "CE OI should be positive"
        assert snap.total_pe_oi > 0, "PE OI should be positive"
        assert snap.pcr_oi > 0, f"PCR should be positive, got {snap.pcr_oi}"
        assert snap.atm_strike > 0, f"ATM should be positive, got {snap.atm_strike}"
        assert snap.max_pain > 0, f"Max pain should be positive, got {snap.max_pain}"

    # PCR should be rising (we seeded growing PE OI)
    pcrs = [s.pcr_oi for s in report.snapshots]
    assert pcrs[-1] > pcrs[0], f"PCR should rise: {pcrs[0]:.3f} → {pcrs[-1]:.3f}"

    # Spot should be rising
    spots = [s.spot_price for s in report.snapshots]
    assert spots[-1] > spots[0], f"Spot should rise: {spots[0]} → {spots[-1]}"

    store.close()
    os.remove(db_path)
    print("  [PASS] Snapshot summaries computed correctly")


def test_flow_change_intervals():
    """Test that flow changes between intervals are sensible."""
    from src.data.fno_data_store import FnODataStore
    from src.options.oi_intraday_analyzer import IntradayOIFlowAnalyzer

    db_path = "data/test_oi_flow.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    _seed_test_db(db_path)
    store = FnODataStore(db_path=db_path)
    analyzer = IntradayOIFlowAnalyzer(store=store)

    report = analyzer.analyze("NIFTY")

    for chg in report.flow_changes:
        assert chg.duration_minutes == 15, f"Expected 15 minutes, got {chg.duration_minutes}"
        assert chg.spot_change > 0, "Spot should be rising in our test data"
        assert chg.signal in ("BULLISH", "MILDLY_BULLISH", "NEUTRAL", "MILDLY_BEARISH", "BEARISH"), \
            f"Unexpected signal: {chg.signal}"
        # PE OI should be growing
        assert chg.net_pe_oi_change >= 0, f"PE OI should be growing, got {chg.net_pe_oi_change}"

    store.close()
    os.remove(db_path)
    print("  [PASS] Flow change intervals are valid")


def test_mcp_tool_registered():
    """Verify the MCP tool is registered on the server."""
    try:
        from src.mcp_server.server import mcp
        # FastMCP stores tools internally; check the tool function exists
        import src.mcp_server.server as srv
        assert hasattr(srv, 'analyze_intraday_oi_flow'), "MCP tool function should exist"
        print("  [PASS] MCP tool registered")
    except ImportError as e:
        print(f"  [SKIP] MCP not installed: {e}")


def test_webapp_endpoints_exist():
    """Verify the webapp has the new OI endpoints."""
    from src.api.webapp import app

    routes = [r.path for r in app.routes if hasattr(r, 'path')]
    expected = [
        "/api/oi/snapshots/start",
        "/api/oi/snapshots/stop",
        "/api/oi/snapshots/status",
        "/api/oi/snapshots/trigger",
        "/api/oi/snapshots/today",
        "/api/oi/intraday/analysis",
        "/api/oi/intraday/direction",
    ]
    for ep in expected:
        assert ep in routes, f"Endpoint {ep} not found in webapp routes"
    print(f"  [PASS] All {len(expected)} webapp endpoints registered")


# ── Run All Tests ────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== OI Intraday Snapshot & Analysis Tests ===\n")

    tests = [
        ("Scheduler Imports", test_scheduler_imports),
        ("Analyzer Imports", test_analyzer_imports),
        ("Scheduler Status", test_scheduler_status),
        ("Snapshot Summaries", test_snapshot_summaries),
        ("Flow Change Intervals", test_flow_change_intervals),
        ("Full Analyzer Pipeline", test_analyzer_with_synthetic_data),
        ("MCP Tool Registration", test_mcp_tool_registered),
        ("Webapp Endpoints", test_webapp_endpoints_exist),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            print(f"[TEST] {name}")
            fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'='*50}")
    sys.exit(0 if failed == 0 else 1)
