"""
Comprehensive Test Suite for OI Analysis Enhancement.
Tests: oi_analysis.py, service.py integration, webapp.py routes, JS/backend field parity.

  Run with: python3 tests/test_oi_analysis.py
"""
import asyncio
import json
import sys
import traceback
from datetime import datetime, timedelta
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, '.')

from src.options.oi_analysis import (
    FuturesOIAnalyzer, OptionsOIAnalyzer,
    FuturesOIEntry, FuturesRollover, FuturesOIReport,
    OptionsOIStrike, OptionsOIReport, OptionsOIComparison,
    FNO_STOCKS, SECTOR_FUTURES, NEAR_ATM_STRIKES,
    NIFTY_STRIKE_GAP, SENSEX_STRIKE_GAP,
)

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def make_futures_entry(sym, exp='2026-02-26', ltp=100, prev=95, oi=1_000_000,
                       prev_oi=900_000, volume=50_000, lot=100, buildup='LONG_BUILDUP',
                       sentiment='BULLISH', sector='IT'):
    chg_pct = ((ltp - prev) / prev * 100) if prev else 0
    oi_chg = oi - prev_oi
    oi_chg_pct = (oi_chg / prev_oi * 100) if prev_oi else 0
    return FuturesOIEntry(
        symbol=sym, expiry=exp, ltp=ltp, prev_close=prev,
        change_pct=round(chg_pct, 2), oi=oi, prev_oi=prev_oi,
        oi_change=oi_chg, oi_change_pct=round(oi_chg_pct, 2),
        volume=volume, lot_size=lot,
        value_lakhs=round(oi * lot * ltp / 100000, 2),
        buildup=buildup, sentiment=sentiment, sector=sector,
    )


def make_strike(strike, ce_oi=100_000, pe_oi=100_000,
                ce_chg=10_000, pe_chg=10_000,
                ce_vol=50_000, pe_vol=50_000,
                ce_ltp=50, pe_ltp=50, ce_iv=12, pe_iv=13,
                is_atm=False, dist=0):
    return OptionsOIStrike(
        strike=strike, ce_oi=ce_oi, pe_oi=pe_oi,
        ce_oi_change=ce_chg, pe_oi_change=pe_chg,
        ce_volume=ce_vol, pe_volume=pe_vol,
        ce_ltp=ce_ltp, pe_ltp=pe_ltp,
        ce_iv=ce_iv, pe_iv=pe_iv,
        pcr_strike=round(pe_oi / ce_oi, 2) if ce_oi else 0,
        net_oi_change=pe_chg - ce_chg,
        is_atm=is_atm, distance=dist,
    )


def build_options_chain(atm=24100, gap=50, n=10):
    """Build a realistic +/-n-strike options chain around ATM."""
    strikes = []
    for i in range(-n, n + 1):
        s = atm + i * gap
        dist_from_atm = abs(i)
        ce_oi = 500_000 - dist_from_atm * 20_000 if i >= 0 else 300_000 - dist_from_atm * 15_000
        pe_oi = 300_000 - dist_from_atm * 15_000 if i >= 0 else 500_000 - dist_from_atm * 20_000
        ce_oi = max(ce_oi, 50_000)
        pe_oi = max(pe_oi, 50_000)
        activity = max(1, n - dist_from_atm)
        ce_chg = activity * 5000 if i >= 0 else activity * 3000
        pe_chg = activity * 3000 if i >= 0 else activity * 5000
        ce_ltp = max(5, 200 - (i * 20)) if i >= 0 else max(5, 200 + (-i * 10))
        pe_ltp = max(5, 200 + (i * 20)) if i <= 0 else max(5, 200 - (i * 10))
        ce_iv = 12 + dist_from_atm * 0.3
        pe_iv = 13 + dist_from_atm * 0.35

        strikes.append(make_strike(
            strike=float(s), ce_oi=ce_oi, pe_oi=pe_oi,
            ce_chg=ce_chg, pe_chg=pe_chg,
            ce_vol=ce_oi // 5, pe_vol=pe_oi // 5,
            ce_ltp=ce_ltp, pe_ltp=pe_ltp,
            ce_iv=round(ce_iv, 1), pe_iv=round(pe_iv, 1),
            is_atm=(i == 0), dist=i,
        ))
    return strikes


# ─────────────────────────────────────────────────────────────
#   TEST 1: Constants integrity
# ─────────────────────────────────────────────────────────────

def test_constants():
    assert len(FNO_STOCKS) == 50, f"Expected 50 FNO stocks, got {len(FNO_STOCKS)}"
    assert len(set(FNO_STOCKS)) == 50, "Duplicate in FNO_STOCKS!"
    sectors = set(SECTOR_FUTURES.values())
    assert len(sectors) >= 10, f"Expected 10+ sectors, got {len(sectors)}"
    for s in FNO_STOCKS:
        assert s in SECTOR_FUTURES, f"{s} missing from SECTOR_FUTURES"
    assert NEAR_ATM_STRIKES == 10
    assert NIFTY_STRIKE_GAP == 50
    assert SENSEX_STRIKE_GAP == 100
    print(f"  Constants: {len(FNO_STOCKS)} stocks, {len(sectors)} sectors -- OK")


# ─────────────────────────────────────────────────────────────
#   TEST 2: Buildup classification (exhaustive)
# ─────────────────────────────────────────────────────────────

def test_buildup_classification_exhaustive():
    cb = FuturesOIAnalyzer._classify_buildup
    assert cb(2.0, 1000) == ('LONG_BUILDUP', 'BULLISH')
    assert cb(-2.0, 1000) == ('SHORT_BUILDUP', 'BEARISH')
    assert cb(-1.0, -500) == ('LONG_UNWINDING', 'BEARISH')
    assert cb(1.0, -500) == ('SHORT_COVERING', 'BULLISH')
    assert cb(0.0, 0) == ('NEUTRAL', 'NEUTRAL')
    assert cb(0.0, 1000) == ('NEUTRAL', 'NEUTRAL')
    assert cb(0.0, -1000) == ('NEUTRAL', 'NEUTRAL')
    assert cb(5.0, 0) == ('NEUTRAL', 'NEUTRAL')
    assert cb(-5.0, 0) == ('NEUTRAL', 'NEUTRAL')
    assert cb(0.01, 1) == ('LONG_BUILDUP', 'BULLISH')
    assert cb(-0.01, -1) == ('LONG_UNWINDING', 'BEARISH')
    assert cb(15.0, 100_000_000) == ('LONG_BUILDUP', 'BULLISH')
    assert cb(-15.0, 100_000_000) == ('SHORT_BUILDUP', 'BEARISH')
    print("  Buildup classification: 13/13 cases -- OK")


# ─────────────────────────────────────────────────────────────
#   TEST 3: Nearest expiry filter
# ─────────────────────────────────────────────────────────────

def test_nearest_expiry_only():
    fa = FuturesOIAnalyzer()
    entries = [
        make_futures_entry('RELIANCE', exp='2026-02-26', oi=5_000_000),
        make_futures_entry('RELIANCE', exp='2026-03-26', oi=1_000_000),
        make_futures_entry('TCS', exp='2026-02-26', oi=3_000_000),
        make_futures_entry('TCS', exp='2026-03-26', oi=500_000),
    ]
    entries[0].oi_change_pct = 15.0
    entries[2].oi_change_pct = 8.0
    result = fa._nearest_expiry_only(entries)
    assert len(result) == 2, f"Expected 2 symbols, got {len(result)}"
    syms = [e.symbol for e in result]
    assert 'RELIANCE' in syms
    assert 'TCS' in syms
    for e in result:
        assert e.expiry == '2026-02-26', f"{e.symbol} got {e.expiry}"
    assert result[0].symbol == 'RELIANCE', "Higher OI chg % should be first"
    print("  Nearest expiry filter: OK")


# ─────────────────────────────────────────────────────────────
#   TEST 4: Rollover computation (comprehensive)
# ─────────────────────────────────────────────────────────────

def test_rollover_computation():
    cb = FuturesOIAnalyzer._compute_rollovers
    def _e(sym, exp, oi, ltp):
        return make_futures_entry(sym, exp=exp, oi=oi, ltp=ltp,
                                  prev_oi=oi, buildup='NEUTRAL', sentiment='NEUTRAL')

    # Normal rollover: positive basis -> POSITIVE_ROLLOVER
    rm = {
        'RELIANCE': [
            _e('RELIANCE', '2026-02-26', 7_000_000, 2500),
            _e('RELIANCE', '2026-03-26', 3_500_000, 2520),
        ],
    }
    r = cb(rm)
    assert len(r) == 1
    assert r[0].symbol == 'RELIANCE'
    assert r[0].rollover_pct > 30
    assert r[0].basis_pct > 0
    assert r[0].signal == 'POSITIVE_ROLLOVER'

    # Negative rollover: negative basis -> NEGATIVE_ROLLOVER
    rm2 = {
        'TCS': [
            _e('TCS', '2026-02-26', 6_000_000, 3800),
            _e('TCS', '2026-03-26', 4_000_000, 3780),
        ],
    }
    r2 = cb(rm2)
    assert r2[0].signal == 'NEGATIVE_ROLLOVER'

    # Low rollover -> NEUTRAL
    rm3 = {
        'INFY': [
            _e('INFY', '2026-02-26', 9_000_000, 1500),
            _e('INFY', '2026-03-26', 500_000, 1505),
        ],
    }
    r3 = cb(rm3)
    assert r3[0].signal == 'NEUTRAL'

    # Single expiry -> skipped
    rm4 = {'HDFCBANK': [_e('HDFCBANK', '2026-02-26', 8_000_000, 1650)]}
    assert len(cb(rm4)) == 0

    # Zero OI edge case
    rm5 = {'ACC': [_e('ACC', '2026-02-26', 0, 100), _e('ACC', '2026-03-26', 0, 100)]}
    r5 = cb(rm5)
    assert r5[0].rollover_pct == 0.0

    print("  Rollover computation: 5/5 cases -- OK")


# ─────────────────────────────────────────────────────────────
#   TEST 5: Sector summary
# ─────────────────────────────────────────────────────────────

def test_sector_summary_comprehensive():
    cb = FuturesOIAnalyzer._compute_sector_summary

    stocks = [
        make_futures_entry('RELIANCE', ltp=2600, prev=2500, buildup='LONG_BUILDUP', sector='Energy'),
        make_futures_entry('ONGC', ltp=260, prev=250, buildup='LONG_BUILDUP', sector='Energy'),
        make_futures_entry('BPCL', ltp=400, prev=410, buildup='SHORT_BUILDUP', sector='Energy'),
        make_futures_entry('TCS', ltp=3700, prev=3800, buildup='SHORT_BUILDUP', sector='IT'),
        make_futures_entry('INFY', ltp=1400, prev=1450, buildup='LONG_UNWINDING', sector='IT'),
        make_futures_entry('HDFCBANK', ltp=1700, prev=1650, buildup='SHORT_COVERING', sector='Banking'),
    ]

    result = cb(stocks)
    assert isinstance(result, list)
    sectors_found = {s['sector'] for s in result}
    assert 'Energy' in sectors_found
    assert 'IT' in sectors_found
    assert 'Banking' in sectors_found

    energy = [s for s in result if s['sector'] == 'Energy'][0]
    assert energy['long_buildup'] == 2
    assert energy['short_buildup'] == 1
    assert energy['bias'] == 'BULLISH'
    assert energy['stock_count'] == 3

    it = [s for s in result if s['sector'] == 'IT'][0]
    assert it['short_buildup'] == 1
    assert it['long_unwinding'] == 1
    assert it['bias'] == 'BEARISH'

    banking = [s for s in result if s['sector'] == 'Banking'][0]
    assert banking['short_covering'] == 1
    assert banking['bias'] == 'BULLISH'

    assert cb([]) == []

    required_keys = {'sector', 'avg_change_pct', 'long_buildup', 'short_buildup',
                     'long_unwinding', 'short_covering', 'bias', 'stock_count', 'bullish_pct'}
    for s in result:
        for k in required_keys:
            assert k in s, f"Missing key '{k}' in sector summary"

    print(f"  Sector summary: {len(result)} sectors, all assertions -- OK")


# ─────────────────────────────────────────────────────────────
#   TEST 6: Market sentiment
# ─────────────────────────────────────────────────────────────

def test_market_sentiment_comprehensive():
    ms = FuturesOIAnalyzer._compute_market_sentiment

    # All long buildup -> BULLISH
    bulls = [make_futures_entry(f'S{i}', buildup='LONG_BUILDUP') for i in range(10)]
    r = ms(bulls, [])
    assert r['bias'] == 'BULLISH'
    assert r['confidence'] > 50
    assert r['long_buildup'] == 10

    # All short buildup -> BEARISH
    bears = [make_futures_entry(f'S{i}', buildup='SHORT_BUILDUP') for i in range(10)]
    r2 = ms(bears, [])
    assert r2['bias'] == 'BEARISH'
    assert r2['short_buildup'] == 10

    # Mixed -> NEUTRAL
    mixed = (
        [make_futures_entry(f'L{i}', buildup='LONG_BUILDUP') for i in range(5)] +
        [make_futures_entry(f'S{i}', buildup='SHORT_BUILDUP') for i in range(5)]
    )
    r3 = ms(mixed, [])
    assert r3['bias'] == 'NEUTRAL'
    assert r3['confidence'] == 0

    # Empty -> NEUTRAL
    r4 = ms([], [])
    assert r4['bias'] == 'NEUTRAL'

    # With index signals
    idx = [make_futures_entry('NIFTY', buildup='LONG_BUILDUP')]
    r5 = ms(bulls, idx)
    assert any('NIFTY' in reason for reason in r5['reasons'])

    # Verify all required keys
    keys = {'bias', 'confidence', 'bullish_count', 'bearish_count',
            'long_buildup', 'short_buildup', 'long_unwinding', 'short_covering',
            'bullish_pct', 'reasons'}
    for k in keys:
        assert k in r, f"Missing key '{k}' in sentiment"

    # Confidence capped at 95
    all_bull = [make_futures_entry(f'X{i}', buildup='LONG_BUILDUP') for i in range(100)]
    r6 = ms(all_bull, [])
    assert r6['confidence'] <= 95, f"Confidence should cap at 95, got {r6['confidence']}"

    print("  Market sentiment: 7 scenarios -- OK")


# ─────────────────────────────────────────────────────────────
#   TEST 7: Max pain computation
# ─────────────────────────────────────────────────────────────

def test_max_pain_computation():
    mp = OptionsOIAnalyzer._compute_max_pain

    strikes = [
        make_strike(24000, ce_oi=0, pe_oi=0),
        make_strike(24050, ce_oi=1_000_000, pe_oi=1_000_000),
        make_strike(24100, ce_oi=0, pe_oi=0),
    ]
    assert mp(strikes) == 24050, "Max pain should be at the heavy OI strike"

    chain = build_options_chain(24100, 50, 10)
    pain = mp(chain)
    assert 23600 <= pain <= 24600, f"Max pain {pain} unreasonable for ATM=24100"

    assert mp([]) == 0.0

    ss = [make_strike(24000, ce_oi=500_000, pe_oi=500_000)]
    assert mp(ss) == 24000

    print(f"  Max pain: realistic={pain}, simple=24050 -- OK")


# ─────────────────────────────────────────────────────────────
#   TEST 8: Buildup signal detection
# ─────────────────────────────────────────────────────────────

def test_buildup_signal_detection():
    ds = OptionsOIAnalyzer._detect_buildup_signals
    chain = build_options_chain(24100, 50, 10)
    signals = ds(chain, 24100)

    assert isinstance(signals, list)
    assert len(signals) <= 15, "Should cap at 15 signals"

    for s in signals:
        assert 'strike' in s
        assert 'type' in s and s['type'] in ('CE', 'PE')
        assert 'pattern' in s and s['pattern'] in ('CE_WRITING', 'CE_BUYING', 'PE_WRITING', 'PE_BUYING')
        assert 'sentiment' in s and s['sentiment'] in ('BULLISH', 'BEARISH')
        assert 'oi_change' in s
        assert 'description' in s

    for s in signals:
        if s['pattern'] == 'CE_WRITING':
            assert s['strike'] > 24100, f"CE_WRITING at {s['strike']} should be > ATM"
            assert s['sentiment'] == 'BEARISH'
        elif s['pattern'] == 'CE_BUYING':
            assert s['strike'] <= 24100, f"CE_BUYING at {s['strike']} should be <= ATM"
            assert s['sentiment'] == 'BULLISH'
        elif s['pattern'] == 'PE_WRITING':
            assert s['strike'] < 24100, f"PE_WRITING at {s['strike']} should be < ATM"
            assert s['sentiment'] == 'BULLISH'
        elif s['pattern'] == 'PE_BUYING':
            assert s['strike'] >= 24100, f"PE_BUYING at {s['strike']} should be >= ATM"
            assert s['sentiment'] == 'BEARISH'

    for i in range(len(signals) - 1):
        assert abs(signals[i]['oi_change']) >= abs(signals[i + 1]['oi_change'])

    flat_chain = [make_strike(24000 + i * 50, ce_chg=0, pe_chg=0) for i in range(5)]
    assert len(ds(flat_chain, 24100)) == 0

    zero_ltp = [make_strike(24200, ce_chg=50000, pe_chg=50000, ce_ltp=0, pe_ltp=0)]
    assert len(ds(zero_ltp, 24100)) == 0

    print(f"  Buildup signals: {len(signals)} signals, all validated -- OK")


# ─────────────────────────────────────────────────────────────
#   TEST 9: Bias computation
# ─────────────────────────────────────────────────────────────

def test_bias_computation():
    cb = OptionsOIAnalyzer._compute_bias
    chain = build_options_chain(24100, 50, 10)

    # High PCR -> Bullish
    bias, reasons = cb(
        pcr_oi=1.5, max_pain=24200, spot=24100,
        max_ce_strike=24300, max_pe_strike=24000,
        iv_skew=-4.0, signals=[{"sentiment": "BULLISH"}] * 5 + [{"sentiment": "BEARISH"}] * 1,
        straddle_premium=300, strikes=chain,
    )
    assert bias in ('BULLISH', 'MILDLY BULLISH'), f"Expected bullish bias, got {bias}"
    assert len(reasons) > 0

    # Low PCR -> Bearish
    bias2, _ = cb(
        pcr_oi=0.5, max_pain=23900, spot=24100,
        max_ce_strike=24150, max_pe_strike=23600,
        iv_skew=5.0, signals=[{"sentiment": "BEARISH"}] * 5 + [{"sentiment": "BULLISH"}] * 1,
        straddle_premium=300, strikes=chain,
    )
    assert bias2 in ('BEARISH', 'MILDLY BEARISH'), f"Expected bearish bias, got {bias2}"

    # Balanced -> Neutral (zero all inputs to get true zero scoring)
    bias3, reasons3 = cb(
        pcr_oi=1.0, max_pain=24100, spot=24100,
        max_ce_strike=0, max_pe_strike=0,
        iv_skew=0, signals=[], straddle_premium=300, strikes=[],
    )
    assert bias3 == 'NEUTRAL'
    assert any('balanced' in r.lower() for r in reasons3), f"Expected balanced reason, got {reasons3}"

    # Balanced via equal bull/bear points -> still NEUTRAL
    bias3b, reasons3b = cb(
        pcr_oi=1.0, max_pain=24100, spot=24100,
        max_ce_strike=24500, max_pe_strike=23700,
        iv_skew=0, signals=[], straddle_premium=300, strikes=[],
    )
    assert bias3b == 'NEUTRAL'
    assert len(reasons3b) >= 2  # OI wall reasons present

    # Edge: zero spot
    bias4, _ = cb(0.9, 0, 0, 0, 0, 0, [], 0, [])
    assert bias4 == 'NEUTRAL'

    print("  Bias computation: 4 scenarios -- OK")


# ─────────────────────────────────────────────────────────────
#   TEST 10: PCR and straddle history tracking
# ─────────────────────────────────────────────────────────────

def test_history_tracking():
    oa = OptionsOIAnalyzer()

    for i in range(5):
        oa._track_history("NIFTY", 0.8 + i * 0.1, 200 + i * 10)

    pcr_hist = oa.get_pcr_history("NIFTY")
    assert len(pcr_hist) == 5
    assert pcr_hist[0]['pcr'] == 0.8
    assert pcr_hist[4]['pcr'] == 1.2

    straddle_hist = oa.get_straddle_history("NIFTY")
    assert len(straddle_hist) == 5
    assert straddle_hist[0]['premium'] == 200
    assert straddle_hist[4]['premium'] == 240

    for entry in pcr_hist:
        assert 'timestamp' in entry

    # Max 100 entries
    for i in range(120):
        oa._track_history("SENSEX", 1.0, 500)
    assert len(oa.get_pcr_history("SENSEX")) == 100
    assert len(oa.get_straddle_history("SENSEX")) == 100

    # Unknown -> empty
    assert oa.get_pcr_history("BANKNIFTY") == []
    assert oa.get_straddle_history("BANKNIFTY") == []

    print("  History tracking: OK")


# ─────────────────────────────────────────────────────────────
#   TEST 11: _get_prev_oi (first-scan fix)
# ─────────────────────────────────────────────────────────────

def test_prev_oi_first_scan():
    fa = FuturesOIAnalyzer()

    # First call: no history -> returns current_oi (not 0)
    prev = fa._get_prev_oi("TEST", "2026-02-26", 1_000_000)
    assert prev == 1_000_000, f"First scan should return current_oi to avoid false signals, got {prev}"

    # Store one snapshot
    fa._store_oi("TEST", "2026-02-26", 1_000_000, 100.0)
    # Still only 1 entry -> returns current_oi
    prev2 = fa._get_prev_oi("TEST", "2026-02-26", 1_050_000)
    assert prev2 == 1_050_000, f"Needs >=2 snapshots, got {prev2}"

    # Store second
    fa._store_oi("TEST", "2026-02-26", 1_050_000, 102.0)
    # Now 2 entries -> returns 2nd-to-last
    prev3 = fa._get_prev_oi("TEST", "2026-02-26", 1_100_000)
    assert prev3 == 1_000_000, f"Should return first snapshot OI, got {prev3}"

    # Verify max 50 entries
    for i in range(60):
        fa._store_oi("HEAVY", "2026-02-26", i * 1000, 100.0)
    key = "HEAVY:2026-02-26"
    assert len(fa._historical_oi[key]) == 50

    print("  _get_prev_oi (first-scan fix): OK")


# ─────────────────────────────────────────────────────────────
#   TEST 12: Pydantic model serialization roundtrip
# ─────────────────────────────────────────────────────────────

def test_model_serialization():
    entry = make_futures_entry('RELIANCE', ltp=2500, prev=2450)
    d = entry.model_dump()
    entry2 = FuturesOIEntry(**d)
    assert entry2.symbol == 'RELIANCE'
    assert entry2.ltp == 2500

    roll = FuturesRollover(
        symbol='TCS', current_expiry='2026-02-26', next_expiry='2026-03-26',
        current_oi=5_000_000, next_oi=2_000_000, rollover_pct=28.57,
        current_ltp=3800, next_ltp=3810, basis_pct=0.26, signal='NEUTRAL',
    )
    rd = roll.model_dump()
    assert rd['rollover_pct'] == 28.57

    report = FuturesOIReport(
        timestamp=datetime.now().isoformat(),
        index_futures=[entry],
        stock_futures=[entry],
        rollover_data=[roll],
    )
    rep_d = report.model_dump()
    assert len(rep_d['index_futures']) == 1
    rep_json = json.dumps(rep_d)
    assert '"RELIANCE"' in rep_json

    strike = make_strike(24100, is_atm=True)
    sd = strike.model_dump()
    assert sd['is_atm'] is True

    chain = build_options_chain(24100, 50, 5)
    opt_report = OptionsOIReport(
        underlying='NIFTY', spot_price=24100, atm_strike=24100,
        expiry='2026-02-26', timestamp=datetime.now().isoformat(),
        total_ce_oi=5_000_000, total_pe_oi=5_000_000,
        pcr_oi=1.0, total_ce_volume=1_000_000, total_pe_volume=1_000_000,
        pcr_volume=1.0, strikes=chain,
        max_ce_oi_strike=24300, max_pe_oi_strike=23900,
        max_ce_oi=500_000, max_pe_oi=500_000,
        max_pain=24100, atm_straddle_premium=350,
        iv_skew=1.5, avg_ce_iv=12.0, avg_pe_iv=13.5,
        top_ce_oi_additions=[{"strike": 24200, "oi_change": 50000}],
        top_pe_oi_additions=[{"strike": 24000, "oi_change": 60000}],
        buildup_signals=[{"strike": 24200, "pattern": "CE_WRITING", "type": "CE"}],
        bias='MILDLY BULLISH',
        bias_reasons=['PCR above 1', 'PE writers adding support'],
    )
    opt_d = opt_report.model_dump()
    opt_json = json.dumps(opt_d)
    assert '"NIFTY"' in opt_json
    assert len(opt_d['strikes']) == 11

    comp = OptionsOIComparison(
        underlying='NIFTY', current_expiry='2026-02-26', next_expiry='2026-03-26',
        current_report=opt_report, next_report=opt_report,
        pcr_shift=0.05, max_pain_shift=50, premium_shift=-10,
    )
    cd = comp.model_dump()
    json.dumps(cd)

    print("  Model serialization: all models roundtrip OK")


# ─────────────────────────────────────────────────────────────
#   TEST 13: Futures OI analyze() with mock client
# ─────────────────────────────────────────────────────────────

def test_futures_analyze_mock():
    fa = FuturesOIAnalyzer()

    # Pre-store prev OI baselines
    fa._store_oi("NIFTY", "2026-02-26", 9_000_000, 24000)
    fa._store_oi("NIFTY", "2026-02-26", 9_500_000, 24050)
    fa._store_oi("RELIANCE", "2026-02-26", 4_000_000, 2450)
    fa._store_oi("RELIANCE", "2026-02-26", 4_500_000, 2480)

    mock_instruments = [
        {"name": "NIFTY", "instrument_token": 1, "instrument_type": "FUT",
         "expiry": "2026-02-26", "lot_size": 25, "exchange": "NFO",
         "tradingsymbol": "NIFTY26FEBFUT"},
        {"name": "RELIANCE", "instrument_token": 2, "instrument_type": "FUT",
         "expiry": "2026-02-26", "lot_size": 250, "exchange": "NFO",
         "tradingsymbol": "RELIANCE26FEBFUT"},
        {"name": "RELIANCE", "instrument_token": 3, "instrument_type": "FUT",
         "expiry": "2026-03-26", "lot_size": 250, "exchange": "NFO",
         "tradingsymbol": "RELIANCE26MARFUT"},
    ]

    mock_quotes = {
        "NFO:NIFTY26FEBFUT": {
            "last_price": 24200, "volume": 500_000, "oi": 10_000_000,
            "oi_day_high": 10_500_000, "oi_day_low": 9_800_000,
            "ohlc": {"close": 24100},
        },
        "NFO:RELIANCE26FEBFUT": {
            "last_price": 2520, "volume": 100_000, "oi": 5_000_000,
            "oi_day_high": 5_200_000, "oi_day_low": 4_800_000,
            "ohlc": {"close": 2480},
        },
        "NFO:RELIANCE26MARFUT": {
            "last_price": 2530, "volume": 20_000, "oi": 1_500_000,
            "oi_day_high": 1_600_000, "oi_day_low": 1_400_000,
            "ohlc": {"close": 2490},
        },
    }

    client = AsyncMock()
    client.get_instruments = AsyncMock(return_value=mock_instruments)
    client.get_quote = AsyncMock(return_value=mock_quotes)

    report = asyncio.get_event_loop().run_until_complete(fa.analyze(client))

    assert isinstance(report, FuturesOIReport)
    assert report.timestamp != ""
    assert len(report.index_futures) >= 1, "Should have NIFTY index future"
    rel_entries = [e for e in report.stock_futures if e.symbol == 'RELIANCE']
    assert len(rel_entries) >= 1, "Should have RELIANCE stock future"

    rel_roll = [r for r in report.rollover_data if r.symbol == 'RELIANCE']
    assert len(rel_roll) == 1, "Should have RELIANCE rollover"
    assert rel_roll[0].rollover_pct > 0

    assert 'bias' in report.market_sentiment
    assert 'reasons' in report.market_sentiment

    assert fa.get_last_report() is not None
    assert fa.get_last_report().timestamp == report.timestamp

    print(f"  Futures analyze (mock): idx={len(report.index_futures)}, stocks={len(report.stock_futures)}, "
          f"rollovers={len(report.rollover_data)}, sectors={len(report.sector_summary)} -- OK")


# ─────────────────────────────────────────────────────────────
#   TEST 14: Options OI analyze() with mock client
# ─────────────────────────────────────────────────────────────

def test_options_analyze_mock():
    oa = OptionsOIAnalyzer()

    gap = NIFTY_STRIKE_GAP
    atm = 24100
    expiry = "2026-02-26"

    mock_instruments = []
    for i in range(-NEAR_ATM_STRIKES, NEAR_ATM_STRIKES + 1):
        strike = atm + i * gap
        for opt_type in ("CE", "PE"):
            mock_instruments.append({
                "name": "NIFTY", "strike": strike, "instrument_type": opt_type,
                "expiry": expiry, "exchange": "NFO",
                "tradingsymbol": f"NIFTY26FEB{strike}{opt_type}",
                "instrument_token": 1000 + i * 2 + (0 if opt_type == "CE" else 1),
                "lot_size": 25,
            })

    mock_quotes = {}
    for inst in mock_instruments:
        key = f"NFO:{inst['tradingsymbol']}"
        strike = inst['strike']
        dist = abs(strike - atm) // gap
        base_oi = max(100_000, 500_000 - dist * 30_000)
        mock_quotes[key] = {
            "last_price": max(5, 200 - dist * 20),
            "volume": base_oi // 5,
            "oi": base_oi,
            "oi_day_high": base_oi + 20_000,
            "oi_day_low": base_oi - 15_000,
            "iv": 12 + dist * 0.5,
        }

    mock_ltp = {"NSE:NIFTY 50": MagicMock(last_price=24100)}

    client = AsyncMock()
    client.get_instruments = AsyncMock(return_value=mock_instruments)
    client.get_quote = AsyncMock(return_value=mock_quotes)
    client.get_ltp = AsyncMock(return_value=mock_ltp)

    report = asyncio.get_event_loop().run_until_complete(
        oa.analyze("NIFTY", client)
    )

    assert isinstance(report, OptionsOIReport)
    assert report.underlying == "NIFTY"
    assert report.spot_price == 24100
    assert report.atm_strike == 24100
    assert report.expiry == expiry

    assert len(report.strikes) == 2 * NEAR_ATM_STRIKES + 1

    atm_strikes = [s for s in report.strikes if s.is_atm]
    assert len(atm_strikes) == 1
    assert atm_strikes[0].strike == 24100

    assert report.total_ce_oi > 0
    assert report.total_pe_oi > 0
    assert report.pcr_oi > 0

    assert report.max_ce_oi_strike > 0
    assert report.max_pe_oi_strike > 0
    assert report.max_ce_oi > 0
    assert report.max_pe_oi > 0

    assert 23600 <= report.max_pain <= 24600

    assert report.atm_straddle_premium > 0

    assert report.bias != ""
    assert len(report.bias_reasons) > 0

    assert isinstance(report.buildup_signals, list)
    for sig in report.buildup_signals:
        assert sig['pattern'] in ('CE_WRITING', 'CE_BUYING', 'PE_WRITING', 'PE_BUYING')

    assert oa.get_last_report("NIFTY") is not None

    assert len(oa.get_pcr_history("NIFTY")) == 1
    assert len(oa.get_straddle_history("NIFTY")) == 1

    d = report.model_dump()
    json.dumps(d)

    print(f"  Options analyze (mock): strikes={len(report.strikes)}, "
          f"pcr={report.pcr_oi:.3f}, maxpain={report.max_pain}, bias={report.bias} -- OK")


# ─────────────────────────────────────────────────────────────
#   TEST 15: Options comparison (mock)
# ─────────────────────────────────────────────────────────────

def test_options_comparison_mock():
    oa = OptionsOIAnalyzer()

    exp1, exp2 = "2026-02-26", "2026-03-05"
    atm, gap = 24100, 50

    mock_instruments = []
    for exp in (exp1, exp2):
        for i in range(-5, 6):
            strike = atm + i * gap
            for opt_type in ("CE", "PE"):
                mock_instruments.append({
                    "name": "NIFTY", "strike": strike, "instrument_type": opt_type,
                    "expiry": exp, "exchange": "NFO",
                    "tradingsymbol": f"NIFTY{exp[:10].replace('-','')}{strike}{opt_type}",
                    "instrument_token": hash(f"{exp}{strike}{opt_type}") & 0xFFFF,
                    "lot_size": 25,
                })

    mock_quotes = {}
    for inst in mock_instruments:
        key = f"NFO:{inst['tradingsymbol']}"
        mock_quotes[key] = {
            "last_price": 100, "volume": 50_000, "oi": 200_000,
            "oi_day_high": 210_000, "oi_day_low": 190_000, "iv": 12,
        }

    mock_ltp = {"NSE:NIFTY 50": MagicMock(last_price=24100)}

    client = AsyncMock()
    client.get_instruments = AsyncMock(return_value=mock_instruments)
    client.get_quote = AsyncMock(return_value=mock_quotes)
    client.get_ltp = AsyncMock(return_value=mock_ltp)

    comp = asyncio.get_event_loop().run_until_complete(
        oa.analyze_comparison("NIFTY", client)
    )

    assert isinstance(comp, OptionsOIComparison)
    assert comp.underlying == "NIFTY"
    assert comp.current_expiry == exp1
    assert comp.next_expiry == exp2
    assert comp.current_report is not None
    assert comp.next_report is not None
    assert isinstance(comp.pcr_shift, float)
    assert isinstance(comp.max_pain_shift, float)
    assert isinstance(comp.premium_shift, float)

    json.dumps(comp.model_dump())

    print("  Options comparison (mock): OK")


# ─────────────────────────────────────────────────────────────
#   TEST 16: Service layer integration
# ─────────────────────────────────────────────────────────────

def test_service_integration():
    from src.api.service import TradingService

    methods = [
        'get_futures_oi_report', 'get_futures_oi_cached',
        'get_options_oi_report', 'get_options_oi_cached',
        'get_options_oi_comparison', 'get_pcr_history', 'get_straddle_history',
    ]
    for m in methods:
        assert hasattr(TradingService, m), f"TradingService missing {m}"
        assert callable(getattr(TradingService, m)), f"{m} not callable"

    import inspect
    sig_opt = inspect.signature(TradingService.get_options_oi_report)
    assert 'underlying' in sig_opt.parameters
    assert 'expiry' in sig_opt.parameters

    print(f"  Service integration: {len(methods)} methods verified -- OK")


# ─────────────────────────────────────────────────────────────
#   TEST 17: Webapp route definitions
# ─────────────────────────────────────────────────────────────

def test_webapp_routes():
    from src.api.webapp import app

    from starlette.routing import Route
    routes = {r.path: r.methods for r in app.routes if isinstance(r, Route)}

    expected_routes = {
        "/api/oi/futures/scan": {"POST"},
        "/api/oi/futures/report": {"GET"},
        "/api/oi/options/scan": {"POST"},
        "/api/oi/options/report": {"GET"},
        "/api/oi/options/compare": {"POST"},
        "/api/oi/pcr-history": {"GET"},
        "/api/oi/straddle-history": {"GET"},
    }

    for path, methods in expected_routes.items():
        assert path in routes, f"Route {path} not found"
        for m in methods:
            assert m in routes[path], f"{path} missing method {m}"

    print(f"  Webapp routes: {len(expected_routes)} routes present -- OK")


# ─────────────────────────────────────────────────────────────
#   TEST 18: Backend <-> JS field parity
# ─────────────────────────────────────────────────────────────

def test_backend_js_field_parity():
    chain = build_options_chain(24100, 50, 5)
    report = OptionsOIReport(
        underlying='NIFTY', spot_price=24100, atm_strike=24100,
        expiry='2026-02-26', timestamp=datetime.now().isoformat(),
        total_ce_oi=5_000_000, total_pe_oi=5_000_000,
        pcr_oi=1.1, total_ce_volume=1_000_000, total_pe_volume=1_100_000,
        pcr_volume=1.1, strikes=chain,
        max_ce_oi_strike=24300, max_pe_oi_strike=23900,
        max_ce_oi=500_000, max_pe_oi=500_000,
        max_pain=24100, atm_straddle_premium=350,
        iv_skew=1.5, avg_ce_iv=12.0, avg_pe_iv=13.5,
        top_ce_oi_additions=[{"strike": 24200, "oi_change": 50000}],
        top_pe_oi_additions=[{"strike": 24000, "oi_change": 60000}],
        buildup_signals=[{"strike": 24200, "pattern": "CE_WRITING", "type": "CE",
                          "sentiment": "BEARISH", "oi_change": 50000}],
        bias='MILDLY BULLISH',
        bias_reasons=['Test reason'],
    )
    d = report.model_dump()

    js_fields = [
        'spot_price', 'atm_strike', 'expiry', 'pcr_oi', 'pcr_volume',
        'max_pain', 'atm_straddle_premium', 'bias', 'bias_reasons',
        'max_pe_oi_strike', 'max_ce_oi_strike', 'max_ce_oi', 'max_pe_oi',
        'iv_skew', 'avg_ce_iv', 'avg_pe_iv',
        'total_ce_oi', 'total_pe_oi', 'total_ce_volume', 'total_pe_volume',
        'strikes', 'top_ce_oi_additions', 'top_pe_oi_additions', 'buildup_signals',
    ]
    for f in js_fields:
        assert f in d, f"Options report missing field '{f}' used by JS"

    strike_d = d['strikes'][0]
    strike_js_fields = [
        'strike', 'ce_oi', 'ce_oi_change', 'ce_volume', 'ce_ltp', 'ce_iv',
        'pe_oi', 'pe_oi_change', 'pe_volume', 'pe_ltp', 'pe_iv', 'is_atm',
    ]
    for f in strike_js_fields:
        assert f in strike_d, f"Strike missing field '{f}' used by JS"

    entry = make_futures_entry('RELIANCE')
    roll = FuturesRollover(
        symbol='RELIANCE', current_expiry='2026-02-26', next_expiry='2026-03-26',
        current_oi=5_000_000, next_oi=2_000_000, rollover_pct=28.57,
        current_ltp=2500, next_ltp=2510, basis_pct=0.4, signal='NEUTRAL',
    )
    fut_report = FuturesOIReport(
        timestamp=datetime.now().isoformat(),
        index_futures=[entry], stock_futures=[entry],
        rollover_data=[roll],
        top_long_buildup=[entry], top_short_buildup=[entry],
        top_long_unwinding=[entry], top_short_covering=[entry],
        sector_summary=[{"sector": "IT", "avg_change_pct": 1.5, "long_buildup": 3,
                         "short_buildup": 1, "long_unwinding": 0, "short_covering": 1,
                         "bias": "BULLISH", "stock_count": 5}],
        market_sentiment={"bias": "BULLISH", "confidence": 65, "long_buildup": 8,
                          "short_buildup": 3, "long_unwinding": 2, "short_covering": 4,
                          "bullish_pct": 60, "reasons": ["test"]},
    )
    fd = fut_report.model_dump()

    fut_js_fields = [
        'market_sentiment', 'index_futures', 'top_long_buildup', 'top_short_buildup',
        'top_long_unwinding', 'top_short_covering', 'stock_futures',
        'rollover_data', 'sector_summary',
    ]
    for f in fut_js_fields:
        assert f in fd, f"Futures report missing field '{f}' used by JS"

    ms = fd['market_sentiment']
    ms_js_fields = ['bias', 'confidence', 'long_buildup', 'short_buildup',
                    'long_unwinding', 'short_covering', 'reasons']
    for f in ms_js_fields:
        assert f in ms, f"market_sentiment missing field '{f}' used by JS"

    sec = fd['sector_summary'][0]
    sec_js_fields = ['sector', 'avg_change_pct', 'long_buildup', 'short_buildup',
                     'long_unwinding', 'short_covering', 'bias', 'stock_count']
    for f in sec_js_fields:
        assert f in sec, f"sector_summary missing field '{f}' used by JS"

    roll_d = fd['rollover_data'][0]
    roll_js_fields = ['symbol', 'current_expiry', 'next_expiry', 'current_oi',
                      'next_oi', 'rollover_pct', 'basis_pct', 'signal']
    for f in roll_js_fields:
        assert f in roll_d, f"rollover_data missing field '{f}' used by JS"

    sig = d['buildup_signals'][0]
    assert 'pattern' in sig, "'pattern' key must exist for signalBadge()"

    fe_d = fd['stock_futures'][0]
    fe_js_fields = ['symbol', 'sector', 'expiry', 'ltp', 'change_pct', 'oi',
                    'oi_change', 'oi_change_pct', 'volume', 'value_lakhs', 'buildup']
    for f in fe_js_fields:
        assert f in fe_d, f"stock_futures entry missing field '{f}' used by JS"

    print("  Backend <-> JS field parity: ALL fields verified -- OK")


# ─────────────────────────────────────────────────────────────
#   TEST 19: Edge cases / error handling
# ─────────────────────────────────────────────────────────────

def test_edge_cases():
    fa = FuturesOIAnalyzer()
    oa = OptionsOIAnalyzer()

    # Empty instruments -> empty report
    client = AsyncMock()
    client.get_instruments = AsyncMock(return_value=[])
    report = asyncio.get_event_loop().run_until_complete(fa.analyze(client))
    assert report.timestamp != ""
    assert len(report.stock_futures) == 0
    assert len(report.index_futures) == 0

    # Empty instruments -> empty options report
    client2 = AsyncMock()
    client2.get_instruments = AsyncMock(return_value=[])
    opt_report = asyncio.get_event_loop().run_until_complete(
        oa.analyze("NIFTY", client2)
    )
    assert opt_report.underlying == "NIFTY"
    assert len(opt_report.strikes) == 0

    # Instruments error -> empty report
    client3 = AsyncMock()
    client3.get_instruments = AsyncMock(side_effect=Exception("Network error"))
    report3 = asyncio.get_event_loop().run_until_complete(fa.analyze(client3))
    assert len(report3.stock_futures) == 0

    opt_report3 = asyncio.get_event_loop().run_until_complete(
        oa.analyze("NIFTY", client3)
    )
    assert len(opt_report3.strikes) == 0

    # Max pain with single strike
    assert OptionsOIAnalyzer._compute_max_pain([make_strike(24000)]) == 24000

    # Detect signals with empty chain
    assert OptionsOIAnalyzer._detect_buildup_signals([], 24100) == []

    # Sector summary with NEUTRAL buildup
    neutral = [make_futures_entry('X', buildup='NEUTRAL', sector='Test')]
    ss = FuturesOIAnalyzer._compute_sector_summary(neutral)
    assert len(ss) == 1
    assert ss[0]['sector'] == 'Test'

    print("  Edge cases: all handled gracefully -- OK")


# ─────────────────────────────────────────────────────────────
#   TEST 20: Full report JSON size sanity
# ─────────────────────────────────────────────────────────────

def test_report_json_size():
    chain = build_options_chain(24100, 50, 10)
    report = OptionsOIReport(
        underlying='NIFTY', spot_price=24100, atm_strike=24100,
        expiry='2026-02-26', timestamp=datetime.now().isoformat(),
        strikes=chain,
        buildup_signals=[{"s": i} for i in range(15)],
        bias='NEUTRAL', bias_reasons=['test'],
    )
    d = report.model_dump()
    size = len(json.dumps(d))
    assert size < 50_000, f"Options report JSON too large: {size} bytes"

    stocks = [make_futures_entry(f'STOCK{i}') for i in range(50)]
    fut_report = FuturesOIReport(
        timestamp=datetime.now().isoformat(),
        stock_futures=stocks,
        sector_summary=[{"sector": f"S{i}", "bias": "NEUTRAL"} for i in range(16)],
        market_sentiment={"bias": "NEUTRAL"},
    )
    fsize = len(json.dumps(fut_report.model_dump()))
    assert fsize < 100_000, f"Futures report JSON too large: {fsize} bytes"

    print(f"  Report JSON size: opt={size}B, fut={fsize}B -- OK")


# ─────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    tests = [
        ("TEST  1: Constants integrity", test_constants),
        ("TEST  2: Buildup classification (exhaustive)", test_buildup_classification_exhaustive),
        ("TEST  3: Nearest expiry filter", test_nearest_expiry_only),
        ("TEST  4: Rollover computation", test_rollover_computation),
        ("TEST  5: Sector summary", test_sector_summary_comprehensive),
        ("TEST  6: Market sentiment", test_market_sentiment_comprehensive),
        ("TEST  7: Max pain computation", test_max_pain_computation),
        ("TEST  8: Buildup signal detection", test_buildup_signal_detection),
        ("TEST  9: Bias computation", test_bias_computation),
        ("TEST 10: History tracking", test_history_tracking),
        ("TEST 11: _get_prev_oi (first-scan fix)", test_prev_oi_first_scan),
        ("TEST 12: Model serialization", test_model_serialization),
        ("TEST 13: Futures analyze (mock client)", test_futures_analyze_mock),
        ("TEST 14: Options analyze (mock client)", test_options_analyze_mock),
        ("TEST 15: Options comparison (mock)", test_options_comparison_mock),
        ("TEST 16: Service layer integration", test_service_integration),
        ("TEST 17: Webapp route definitions", test_webapp_routes),
        ("TEST 18: Backend <-> JS field parity", test_backend_js_field_parity),
        ("TEST 19: Edge cases / error handling", test_edge_cases),
        ("TEST 20: Report JSON size sanity", test_report_json_size),
    ]

    passed = 0
    failed = 0
    errors = []

    for name, fn in tests:
        try:
            print(f"\n[{name}]")
            fn()
            passed += 1
            print(f"  PASS")
        except Exception as e:
            failed += 1
            tb = traceback.format_exc()
            errors.append((name, tb))
            print(f"  FAIL: {e}")

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {passed}/{passed + failed} PASSED, {failed} FAILED")

    if errors:
        print(f"\n{'_' * 60}")
        print("FAILURE DETAILS:")
        for name, tb in errors:
            print(f"\n  [{name}]")
            for line in tb.strip().split('\n'):
                print(f"    {line}")

    if failed == 0:
        print("\nALL 20 TESTS PASSED!")

    sys.exit(1 if failed else 0)
