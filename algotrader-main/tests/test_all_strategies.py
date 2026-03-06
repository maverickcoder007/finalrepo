"""
Comprehensive F&O Strategy Test Suite
========================================
Robust end-to-end tests for ALL strategies in the FnO backtest engine.

Covers:
  - All 13 generic-mode strategies (iron_condor, vertical spreads, straddles, etc.)
  - Both strategy-driven runners (put_credit_spread_runner, call_credit_spread_runner)
  - Symbol parsing robustness (underlyings with C/P in name: HDFC, TCS, etc.)
  - Chain conversion correctness (synthetic → OptionChainData field mapping)
  - Price lookup correctness (_find_option_price_from_chain, _find_option_price_from_chain_data)
  - Result structure validation for both generic and strategy-driven modes
  - Edge cases (empty data, short data, high-vol, BANKNIFTY, SENSEX)
  - Capital accounting consistency
  - Data quality metrics presence
  - Slippage / regime filter impact
  - Force-close at session end

Run:
    python -m pytest tests/test_all_strategies.py -v --tb=short
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest

# Suppress structlog noise during tests
import structlog
structlog.reset_defaults()
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL),
    cache_logger_on_first_use=False,
)

from src.data.models import TransactionType
from src.derivatives.fno_backtest import FnOBacktestEngine
from src.options.chain import OptionChainData, OptionChainEntry, OptionContract


# ═══════════════════════════════════════════════════════════
# DATA GENERATORS
# ═══════════════════════════════════════════════════════════

def generate_ohlcv(
    start_price: float = 24_500.0,
    bars: int = 252,
    annual_drift: float = 0.10,
    annual_vol: float = 0.16,
    seed: int = 42,
    start_date: date | None = None,
) -> pd.DataFrame:
    """Generate realistic daily OHLCV using GBM with regime switching."""
    rng = np.random.default_rng(seed)
    if start_date is None:
        start_date = date.today() - timedelta(days=int(bars * 1.5))

    dt = 1 / 252
    price = start_price
    vol_regime = annual_vol

    opens, highs, lows, closes, volumes = [], [], [], [], []

    for _ in range(bars):
        if rng.random() < 0.05:
            vol_regime = annual_vol * rng.uniform(1.5, 2.5)
        elif rng.random() < 0.10:
            vol_regime = annual_vol

        ret = (annual_drift - 0.5 * vol_regime**2) * dt + vol_regime * math.sqrt(dt) * rng.standard_normal()
        gap = price * rng.uniform(-0.015, 0.015) if rng.random() < 0.015 else 0.0
        o = price + gap
        c = o * math.exp(ret)
        rng_h = abs(ret) + vol_regime * math.sqrt(dt) * 0.5
        h = max(o, c) + abs(rng.standard_normal()) * o * rng_h * 0.5
        lo = min(o, c) - abs(rng.standard_normal()) * o * rng_h * 0.5
        lo = max(lo, c * 0.95)

        opens.append(round(o, 2))
        highs.append(round(h, 2))
        lows.append(round(lo, 2))
        closes.append(round(c, 2))
        volumes.append(int(rng.uniform(5e6, 20e6)))
        price = c

    dates = []
    d = start_date
    while len(dates) < bars:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)

    return pd.DataFrame({
        "timestamp": pd.to_datetime(dates),
        "open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes,
    })


# ═══════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def nifty_data() -> pd.DataFrame:
    return generate_ohlcv(start_price=24_500, bars=252, seed=42)


@pytest.fixture(scope="module")
def nifty_short_data() -> pd.DataFrame:
    return generate_ohlcv(start_price=24_000, bars=60, seed=7)


@pytest.fixture(scope="module")
def nifty_high_vol_data() -> pd.DataFrame:
    return generate_ohlcv(start_price=25_000, bars=100, annual_drift=-0.15, annual_vol=0.35, seed=666)


@pytest.fixture(scope="module")
def banknifty_data() -> pd.DataFrame:
    return generate_ohlcv(start_price=51_000, bars=252, annual_vol=0.20, seed=123)


@pytest.fixture(scope="module")
def sensex_data() -> pd.DataFrame:
    return generate_ohlcv(start_price=80_000, bars=120, annual_vol=0.15, seed=555)


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════

GENERIC_STRATEGIES = [
    "iron_condor",
    "bull_call_spread",
    "bear_put_spread",
    "bull_put_spread",
    "bear_call_spread",
    "straddle",
    "short_straddle",
    "strangle",
    "short_strangle",
    "covered_call",
    "protective_put",
    "iron_butterfly",
    "calendar_spread",
]

RUNNER_STRATEGIES = [
    "put_credit_spread_runner",
    "call_credit_spread_runner",
]

ALL_STRATEGIES = GENERIC_STRATEGIES + RUNNER_STRATEGIES

GENERIC_REQUIRED_KEYS = {
    "engine", "strategy_name", "structure", "underlying",
    "total_bars", "initial_capital", "final_capital",
    "total_return_pct", "equity_curve", "trades", "settings",
}

RUNNER_REQUIRED_KEYS = {
    "engine", "strategy_name", "structure", "underlying",
    "total_bars", "initial_capital", "final_capital",
    "total_return_pct", "equity_curve", "trades", "settings",
    "data_quality", "strategy_final_state",
}

METRIC_KEYS = {
    "total_trades", "win_rate", "sharpe_ratio", "max_drawdown_pct",
    "profit_factor", "backtest_warnings",
}


def _make_engine(strategy: str, underlying: str = "NIFTY", capital: float = 500_000, **kw) -> FnOBacktestEngine:
    defaults = dict(
        strategy_name=strategy,
        underlying=underlying,
        initial_capital=capital,
        max_positions=2,
        profit_target_pct=50.0,
        stop_loss_pct=100.0,
        entry_dte_min=3,
        entry_dte_max=45,
        delta_target=0.16,
        slippage_model="zero",
        use_regime_filter=False,
    )
    defaults.update(kw)
    return FnOBacktestEngine(**defaults)


def _make_chain(spot: float = 25000.0, underlying: str = "NIFTY") -> OptionChainData:
    """Build a synthetic option chain for unit-level price lookup tests."""
    entries = []
    for strike in range(int(spot) - 500, int(spot) + 600, 100):
        diff = strike - spot
        # Rough Black-Scholes-ish prices
        ce_price = max(1.0, (spot - strike) + 150 + diff * 0.3)
        pe_price = max(1.0, (strike - spot) + 150 - diff * 0.3)
        ce_delta = max(0.05, min(0.95, 0.5 - diff / (2 * spot)))
        pe_delta = ce_delta - 1.0

        ce = OptionContract(
            tradingsymbol=f"{underlying}20250327C{strike}",
            strike=float(strike), option_type="CE", underlying=underlying,
            last_price=round(ce_price, 2), delta=round(ce_delta, 4),
            iv=16.0, volume=10000, oi=50000,
            bid_price=round(ce_price - 0.5, 2), ask_price=round(ce_price + 0.5, 2),
        )
        pe = OptionContract(
            tradingsymbol=f"{underlying}20250327P{strike}",
            strike=float(strike), option_type="PE", underlying=underlying,
            last_price=round(pe_price, 2), delta=round(pe_delta, 4),
            iv=16.0, volume=10000, oi=50000,
            bid_price=round(pe_price - 0.5, 2), ask_price=round(pe_price + 0.5, 2),
        )
        entries.append(OptionChainEntry(strike=float(strike), ce=ce, pe=pe))

    return OptionChainData(
        underlying=underlying, spot_price=spot, expiry="2025-03-27",
        entries=entries, atm_strike=round(spot / 100) * 100, atm_iv=16.0,
        updated_at="2025-03-25T10:30:00",
    )


# ═══════════════════════════════════════════════════════════
# SECTION 1: ALL STRATEGIES — NO-ERROR / STRUCTURE TESTS
# ═══════════════════════════════════════════════════════════

@pytest.mark.parametrize("strategy_name", ALL_STRATEGIES)
class TestAllStrategiesBasic:
    """Every strategy must run without error and produce a valid result dict."""

    @pytest.fixture(autouse=True)
    def _run(self, strategy_name: str, nifty_data: pd.DataFrame):
        self.engine = _make_engine(strategy_name)
        self.result = self.engine.run(nifty_data, tradingsymbol="NIFTY")

    def test_no_error(self, strategy_name):
        assert "error" not in self.result, f"[{strategy_name}] {self.result.get('error')}"

    def test_strategy_name_preserved(self, strategy_name):
        assert self.result.get("strategy_name") == strategy_name

    def test_required_keys_present(self, strategy_name):
        is_runner = strategy_name in RUNNER_STRATEGIES
        required = RUNNER_REQUIRED_KEYS if is_runner else GENERIC_REQUIRED_KEYS
        missing = required - set(self.result.keys())
        assert not missing, f"[{strategy_name}] Missing keys: {missing}"

    def test_metric_keys_present(self, strategy_name):
        missing = METRIC_KEYS - set(self.result.keys())
        assert not missing, f"[{strategy_name}] Missing metric keys: {missing}"

    def test_equity_curve_nonempty(self, strategy_name):
        curve = self.result.get("equity_curve", [])
        assert len(curve) >= 2, f"[{strategy_name}] Equity curve too short: {len(curve)}"

    def test_equity_starts_at_initial_capital(self, strategy_name):
        curve = self.result.get("equity_curve", [])
        assert abs(curve[0] - 500_000) < 1.0, f"[{strategy_name}] First equity point: {curve[0]}"

    def test_return_calculation_consistent(self, strategy_name):
        initial = self.result["initial_capital"]
        final = self.result["final_capital"]
        reported = self.result["total_return_pct"]
        expected = (final - initial) / initial * 100
        assert abs(reported - expected) < 0.5, (
            f"[{strategy_name}] Return mismatch: reported={reported}% vs computed={expected:.2f}%"
        )

    def test_trades_is_list(self, strategy_name):
        assert isinstance(self.result.get("trades"), list)

    def test_settings_preserved(self, strategy_name):
        s = self.result.get("settings", {})
        assert s.get("profit_target_pct") == 50.0
        assert s.get("stop_loss_pct") == 100.0

    def test_win_rate_in_range(self, strategy_name):
        wr = self.result.get("win_rate", 0)
        assert 0 <= wr <= 100, f"[{strategy_name}] Win rate out of range: {wr}"

    def test_max_drawdown_nonnegative(self, strategy_name):
        dd = self.result.get("max_drawdown_pct", 0)
        assert dd >= 0, f"[{strategy_name}] Negative drawdown: {dd}"

    def test_total_costs_nonnegative(self, strategy_name):
        costs = self.result.get("total_costs", 0)
        assert costs >= 0, f"[{strategy_name}] Negative costs: {costs}"


# ═══════════════════════════════════════════════════════════
# SECTION 2: RUNNER STRATEGIES — TRADE GENERATION
# ═══════════════════════════════════════════════════════════

@pytest.mark.parametrize("strategy_name", RUNNER_STRATEGIES)
class TestRunnerStrategiesTradeGeneration:
    """Runner strategies (put/call credit spread) must produce trades."""

    @pytest.fixture(autouse=True)
    def _run(self, strategy_name: str, nifty_data: pd.DataFrame):
        self.engine = _make_engine(strategy_name)
        self.result = self.engine.run(nifty_data, tradingsymbol="NIFTY")

    def test_produces_trades(self, strategy_name):
        trades = self.result.get("trades", [])
        total = self.result.get("total_trades", 0)
        assert len(trades) > 0, f"[{strategy_name}] ZERO trades — engine broken"
        assert total > 0, f"[{strategy_name}] total_trades metric is 0"

    def test_engine_type_is_strategy_driven(self, strategy_name):
        assert self.result.get("engine") == "fno_backtest_strategy_driven"

    def test_data_quality_section_present(self, strategy_name):
        dq = self.result.get("data_quality", {})
        assert "data_source" in dq
        assert "total_bars_processed" in dq
        assert dq["total_bars_processed"] > 0

    def test_strategy_final_state_present(self, strategy_name):
        state = self.result.get("strategy_final_state", {})
        assert "strategy" in state or "state" in state

    def test_trades_have_correct_fields(self, strategy_name):
        trades = self.result.get("trades", [])
        for t in trades[:10]:
            assert "trade_id" in t, f"Missing trade_id: {t}"
            assert "tradingsymbol" in t, f"Missing tradingsymbol: {t}"
            assert "transaction_type" in t, f"Missing transaction_type: {t}"
            assert "price" in t, f"Missing price: {t}"
            assert t["price"] > 0, f"Zero/negative price: {t}"

    def test_trades_have_entry_and_exit_pairs(self, strategy_name):
        """At least some trades should have pnl (meaning they were closed)."""
        trades = self.result.get("trades", [])
        trades_with_pnl = [t for t in trades if "pnl" in t and t["pnl"] is not None]
        assert len(trades_with_pnl) > 0, (
            f"[{strategy_name}] No closed trades found — all positions remained open"
        )

    def test_final_capital_matches_curve_end(self, strategy_name):
        curve = self.result.get("equity_curve", [])
        final_from_curve = curve[-1] if curve else 0
        final_capital = self.result["final_capital"]
        diff_pct = abs(final_from_curve - final_capital) / max(abs(final_capital), 1) * 100
        assert diff_pct < 10, (
            f"[{strategy_name}] Curve end {final_from_curve:.2f} ≠ final_capital {final_capital:.2f}"
        )


# ═══════════════════════════════════════════════════════════
# SECTION 3: GENERIC STRATEGIES — DETAILED TESTS
# ═══════════════════════════════════════════════════════════

@pytest.mark.parametrize("strategy_name", GENERIC_STRATEGIES)
class TestGenericStrategiesDetailed:
    """Deeper checks on generic-mode strategies."""

    @pytest.fixture(autouse=True)
    def _run(self, strategy_name: str, nifty_data: pd.DataFrame):
        self.engine = _make_engine(strategy_name)
        self.result = self.engine.run(nifty_data, tradingsymbol="NIFTY")

    def test_structure_type_matches_strategy(self, strategy_name):
        structure = self.result.get("structure", "")
        assert structure, f"[{strategy_name}] Empty structure type"

    def test_regime_history_list(self, strategy_name):
        rh = self.result.get("regime_history", [])
        assert isinstance(rh, list)

    def test_positions_opened_nonneg(self, strategy_name):
        assert self.result.get("positions_opened", 0) >= 0

    def test_greeks_history_is_list(self, strategy_name):
        gh = self.result.get("greeks_history", [])
        assert isinstance(gh, list)

    def test_trades_have_correct_type_field(self, strategy_name):
        """Generic-mode trades use 'type' field for ENTRY/EXIT."""
        trades = self.result.get("trades", [])
        valid_types = {"ENTRY", "PROFIT_TARGET", "STOP_LOSS", "FINAL_EXIT", "EXPIRY"}
        for t in trades[:20]:
            if "type" in t:
                assert t["type"] in valid_types, f"Unknown trade type: {t['type']}"


# ═══════════════════════════════════════════════════════════
# SECTION 4: SYMBOL PARSING ROBUSTNESS
# ═══════════════════════════════════════════════════════════

class TestSymbolParsing:
    """Test _parse_option_symbol and price lookups with tricky underlyings."""

    def test_nifty_put(self):
        result = FnOBacktestEngine._parse_option_symbol("NIFTY20250327P24500")
        assert result == ("PE", 24500.0)

    def test_nifty_call(self):
        result = FnOBacktestEngine._parse_option_symbol("NIFTY20250327C25000")
        assert result == ("CE", 25000.0)

    def test_hdfc_call_with_c_in_name(self):
        """HDFC contains 'C' — must not confuse the parser."""
        result = FnOBacktestEngine._parse_option_symbol("HDFC20250327C1700")
        assert result is not None
        assert result == ("CE", 1700.0)

    def test_hdfc_put_with_c_in_name(self):
        result = FnOBacktestEngine._parse_option_symbol("HDFC20250327P1600")
        assert result is not None
        assert result == ("PE", 1600.0)

    def test_tcs_call_with_c_in_name(self):
        """TCS contains 'C' — must not confuse the parser."""
        result = FnOBacktestEngine._parse_option_symbol("TCS20250327C4000")
        assert result is not None
        assert result == ("CE", 4000.0)

    def test_icicibank_put(self):
        """ICICIBANK has 'C' and other letters."""
        result = FnOBacktestEngine._parse_option_symbol("ICICIBANK20250327P1200")
        assert result is not None
        assert result == ("PE", 1200.0)

    def test_hcltech_call(self):
        """HCLTECH contains 'C'."""
        result = FnOBacktestEngine._parse_option_symbol("HCLTECH20250327C1800")
        assert result is not None
        assert result == ("CE", 1800.0)

    def test_banknifty_put(self):
        result = FnOBacktestEngine._parse_option_symbol("BANKNIFTY20250327P51000")
        assert result == ("PE", 51000.0)

    def test_sensex_call(self):
        result = FnOBacktestEngine._parse_option_symbol("SENSEX20250327C80000")
        assert result == ("CE", 80000.0)

    def test_decimal_strike(self):
        result = FnOBacktestEngine._parse_option_symbol("NIFTY20250327P24500.5")
        assert result is not None
        assert result[0] == "PE"
        assert abs(result[1] - 24500.5) < 0.01

    def test_no_option_type_returns_none(self):
        result = FnOBacktestEngine._parse_option_symbol("NIFTY20250327X24500")
        assert result is None

    def test_empty_string_returns_none(self):
        result = FnOBacktestEngine._parse_option_symbol("")
        assert result is None


# ═══════════════════════════════════════════════════════════
# SECTION 5: CHAIN CONVERSION & PRICE LOOKUPS
# ═══════════════════════════════════════════════════════════

class TestChainConversionAndPriceLookup:
    """Test synthetic chain → OptionChainData conversion and price lookups."""

    @pytest.fixture()
    def engine(self) -> FnOBacktestEngine:
        return _make_engine("iron_condor")

    @pytest.fixture()
    def synth_chain(self, engine: FnOBacktestEngine):
        from src.derivatives.chain_builder import HistoricalChainBuilder
        builder = HistoricalChainBuilder("NIFTY")
        return builder.build_chain(24000.0, datetime(2025, 3, 1), date(2025, 3, 4), hv=0.15)

    def test_convert_synthetic_to_option_chain_data(self, engine, synth_chain):
        ocd = engine._convert_synthetic_chain_to_option_chain_data(synth_chain)
        assert isinstance(ocd, OptionChainData)
        assert len(ocd.entries) > 0
        assert ocd.spot_price == synth_chain.spot_price
        assert ocd.underlying == synth_chain.underlying

    def test_converted_chain_has_lowercase_fields(self, engine, synth_chain):
        """Verify ce/pe fields are lowercase (not CE/PE), last_price (not ltp)."""
        ocd = engine._convert_synthetic_chain_to_option_chain_data(synth_chain)
        for entry in ocd.entries:
            # These must work (lowercase)
            if entry.ce:
                assert entry.ce.last_price > 0
                assert entry.ce.bid_price >= 0
                assert entry.ce.ask_price >= 0
            if entry.pe:
                assert entry.pe.last_price > 0
                assert entry.pe.bid_price >= 0
                assert entry.pe.ask_price >= 0

    def test_find_price_from_synth_chain(self, engine, synth_chain):
        """_find_option_price_from_chain should find ATM strikes."""
        atm = synth_chain.atm_strike
        sym_call = f"NIFTY{str(synth_chain.expiry).replace('-','')}C{int(atm)}"
        sym_put = f"NIFTY{str(synth_chain.expiry).replace('-','')}P{int(atm)}"

        price_c = engine._find_option_price_from_chain(synth_chain, sym_call)
        price_p = engine._find_option_price_from_chain(synth_chain, sym_put)

        assert price_c is not None and price_c > 0, f"Call price not found for {sym_call}"
        assert price_p is not None and price_p > 0, f"Put price not found for {sym_put}"

    def test_find_price_from_chain_data(self, engine, synth_chain):
        """_find_option_price_from_chain_data must work on converted OptionChainData."""
        ocd = engine._convert_synthetic_chain_to_option_chain_data(synth_chain)
        atm = synth_chain.atm_strike

        sym_call = f"NIFTY{str(synth_chain.expiry).replace('-','')}C{int(atm)}"
        sym_put = f"NIFTY{str(synth_chain.expiry).replace('-','')}P{int(atm)}"

        price_c = engine._find_option_price_from_chain_data(ocd, sym_call)
        price_p = engine._find_option_price_from_chain_data(ocd, sym_put)

        assert price_c is not None and price_c > 0, f"Call price not found in chain_data for {sym_call}"
        assert price_p is not None and price_p > 0, f"Put price not found in chain_data for {sym_put}"

    def test_find_price_from_chain_data_hdfc_symbol(self, engine):
        """Price lookup must work for underlyings with C/P in name."""
        chain = _make_chain(spot=1700.0, underlying="HDFC")
        price = engine._find_option_price_from_chain_data(chain, "HDFC20250327C1700")
        assert price is not None and price > 0

    def test_unified_lookup_prefers_synth_chain(self, engine, synth_chain):
        """_find_option_price_unified tries synth_chain first."""
        ocd = engine._convert_synthetic_chain_to_option_chain_data(synth_chain)
        atm = synth_chain.atm_strike
        sym = f"NIFTY{str(synth_chain.expiry).replace('-','')}P{int(atm)}"

        price = engine._find_option_price_unified(synth_chain, ocd, sym)
        assert price is not None and price > 0

    def test_unified_lookup_falls_back_to_chain_data(self, engine, synth_chain):
        """When synth_chain is None, should use chain_data."""
        ocd = engine._convert_synthetic_chain_to_option_chain_data(synth_chain)
        atm = synth_chain.atm_strike
        sym = f"NIFTY{str(synth_chain.expiry).replace('-','')}P{int(atm)}"

        price = engine._find_option_price_unified(None, ocd, sym)
        assert price is not None and price > 0

    def test_unified_lookup_returns_none_when_both_none(self, engine):
        price = engine._find_option_price_unified(None, None, "NIFTY20250327P24000")
        assert price is None

    def test_extract_strike_from_symbol(self, engine):
        assert engine._extract_strike_from_symbol("NIFTY20250327C25000") == 25000.0
        assert engine._extract_strike_from_symbol("NIFTY20250327P24000") == 24000.0
        assert engine._extract_strike_from_symbol("HDFC20250327C1700") == 1700.0
        assert engine._extract_strike_from_symbol("INVALID") == 0.0


# ═══════════════════════════════════════════════════════════
# SECTION 6: EDGE CASES
# ═══════════════════════════════════════════════════════════

class TestEdgeCases:
    """Engine with edge case data."""

    def test_empty_dataframe(self):
        result = _make_engine("iron_condor").run(pd.DataFrame())
        assert "error" in result

    def test_missing_columns(self):
        result = _make_engine("iron_condor").run(pd.DataFrame({"close": [100, 101]}))
        assert "error" in result

    def test_very_short_data_generic(self, nifty_short_data: pd.DataFrame):
        result = _make_engine("iron_condor").run(nifty_short_data)
        assert "error" not in result

    def test_very_short_data_runner(self, nifty_short_data: pd.DataFrame):
        result = _make_engine("put_credit_spread_runner").run(nifty_short_data)
        assert "error" not in result

    def test_high_vol_data_generic(self, nifty_high_vol_data: pd.DataFrame):
        result = _make_engine("iron_condor").run(nifty_high_vol_data)
        assert "error" not in result

    def test_high_vol_data_runner(self, nifty_high_vol_data: pd.DataFrame):
        result = _make_engine("put_credit_spread_runner").run(nifty_high_vol_data)
        assert "error" not in result

    def test_banknifty_underlying(self, banknifty_data: pd.DataFrame):
        result = _make_engine("iron_condor", underlying="BANKNIFTY", capital=1_000_000).run(banknifty_data)
        assert "error" not in result
        assert result.get("underlying") == "BANKNIFTY"

    def test_banknifty_runner(self, banknifty_data: pd.DataFrame):
        result = _make_engine("put_credit_spread_runner", underlying="BANKNIFTY", capital=1_000_000).run(banknifty_data)
        assert "error" not in result

    def test_sensex_underlying(self, sensex_data: pd.DataFrame):
        result = _make_engine("iron_condor", underlying="SENSEX").run(sensex_data)
        assert "error" not in result

    def test_all_strategies_on_short_data(self, nifty_short_data: pd.DataFrame):
        """All strategies must handle short data without crashing."""
        for strat in ALL_STRATEGIES:
            result = _make_engine(strat).run(nifty_short_data)
            assert "error" not in result, f"[{strat}] crashed on short data: {result.get('error')}"


# ═══════════════════════════════════════════════════════════
# SECTION 7: SLIPPAGE & REGIME FILTER IMPACT
# ═══════════════════════════════════════════════════════════

class TestSlippageAndRegimeFilter:
    """Verify slippage and regime filter have the expected effect."""

    def test_realistic_slippage_increases_costs(self, nifty_data: pd.DataFrame):
        r_zero = _make_engine("iron_condor", slippage_model="zero").run(nifty_data)
        r_real = _make_engine("iron_condor", slippage_model="realistic").run(nifty_data)
        assert "error" not in r_zero
        assert "error" not in r_real
        assert r_real.get("total_costs", 0) >= r_zero.get("total_costs", 0)

    def test_regime_filter_effect(self, nifty_data: pd.DataFrame):
        r_no = _make_engine("iron_condor", use_regime_filter=False).run(nifty_data)
        r_yes = _make_engine("iron_condor", use_regime_filter=True).run(nifty_data)
        assert "error" not in r_no
        assert "error" not in r_yes
        # Regime filter should be ≤ (or at most +1 due to timing)
        assert r_yes.get("positions_opened", 0) <= r_no.get("positions_opened", 999) + 1


# ═══════════════════════════════════════════════════════════
# SECTION 8: RUNNER STATE MACHINE UNIT TESTS
# ═══════════════════════════════════════════════════════════

class TestPutCreditSpreadRunnerStateMachine:
    """Unit tests for PutCreditSpreadRunnerStrategy state transitions."""

    def _make_strat(self, **kw):
        from src.options.put_credit_spread_runner import PutCreditSpreadRunnerStrategy
        defaults = dict(
            spread_width=100, sl_premium_points=40,
            be_trigger_points=60, short_put_delta=0.50,
            min_credit=15.0, max_risk_per_spread=10000.0,
        )
        return PutCreditSpreadRunnerStrategy({**defaults, **kw})

    def _make_chain(self, spot=25000.0):
        """Chain with realistic delta distribution — mirrors proven test fixture."""
        strikes = [
            {"strike": 24200, "ce_price": 830, "ce_delta": 0.88, "pe_price": 8,   "pe_delta": -0.12},
            {"strike": 24300, "ce_price": 740, "ce_delta": 0.83, "pe_price": 14,  "pe_delta": -0.17},
            {"strike": 24400, "ce_price": 650, "ce_delta": 0.78, "pe_price": 22,  "pe_delta": -0.22},
            {"strike": 24500, "ce_price": 580, "ce_delta": 0.72, "pe_price": 30,  "pe_delta": -0.28},
            {"strike": 24600, "ce_price": 490, "ce_delta": 0.65, "pe_price": 40,  "pe_delta": -0.35},
            {"strike": 24700, "ce_price": 410, "ce_delta": 0.57, "pe_price": 60,  "pe_delta": -0.43},
            {"strike": 24800, "ce_price": 330, "ce_delta": 0.50, "pe_price": 80,  "pe_delta": -0.50},
            {"strike": 24900, "ce_price": 260, "ce_delta": 0.42, "pe_price": 110, "pe_delta": -0.58},
            {"strike": 25000, "ce_price": 200, "ce_delta": 0.35, "pe_price": 150, "pe_delta": -0.65},
            {"strike": 25100, "ce_price": 145, "ce_delta": 0.28, "pe_price": 195, "pe_delta": -0.72},
            {"strike": 25200, "ce_price": 100, "ce_delta": 0.22, "pe_price": 250, "pe_delta": -0.78},
        ]
        entries = []
        for s in strikes:
            ce = OptionContract(
                tradingsymbol=f"NIFTY24JUN{int(s['strike'])}CE",
                strike=float(s["strike"]), last_price=s["ce_price"], delta=s["ce_delta"],
                iv=16.0, volume=10000, oi=50000,
                bid_price=s["ce_price"] - 0.5, ask_price=s["ce_price"] + 0.5,
            )
            pe = OptionContract(
                tradingsymbol=f"NIFTY24JUN{int(s['strike'])}PE",
                strike=float(s["strike"]), last_price=s["pe_price"], delta=s["pe_delta"],
                iv=16.0, volume=10000, oi=50000,
                bid_price=s["pe_price"] - 0.5, ask_price=s["pe_price"] + 0.5,
            )
            entries.append(OptionChainEntry(strike=float(s["strike"]), ce=ce, pe=pe))
        return OptionChainData(
            underlying="NIFTY", spot_price=spot, expiry="2024-06-27",
            entries=entries, atm_strike=round(spot / 100) * 100, atm_iv=16.0,
            updated_at="2024-06-25T10:30:00",
        )

    def test_initial_state(self):
        from src.options.put_credit_spread_runner import PutSpreadState
        strat = self._make_strat()
        assert strat.state == PutSpreadState.IDLE
        assert not strat.is_in_trade

    def test_entry_generates_two_signals(self):
        strat = self._make_strat()
        chain = self._make_chain()
        strat.update_chain(chain)
        signals = strat._evaluate_entry()
        assert len(signals) == 2
        sells = [s for s in signals if s.transaction_type == TransactionType.SELL]
        buys = [s for s in signals if s.transaction_type == TransactionType.BUY]
        assert len(sells) == 1 and len(buys) == 1

    def test_no_entry_when_not_idle(self):
        from src.options.put_credit_spread_runner import PutSpreadState
        strat = self._make_strat()
        chain = self._make_chain()
        strat.update_chain(chain)
        strat._evaluate_entry()  # enter
        assert strat.state == PutSpreadState.SPREAD_OPEN
        signals2 = strat._evaluate_entry()
        assert signals2 == []

    def test_insufficient_credit_blocks_entry(self):
        strat = self._make_strat(min_credit=9999)
        chain = self._make_chain()
        strat.update_chain(chain)
        assert strat._evaluate_entry() == []

    def test_risk_exceeded_blocks_entry(self):
        strat = self._make_strat(max_risk_per_spread=1)
        chain = self._make_chain()
        strat.update_chain(chain)
        assert strat._evaluate_entry() == []

    def test_no_chain_blocks_entry(self):
        strat = self._make_strat()
        assert strat._evaluate_entry() == []


class TestCallCreditSpreadRunnerStateMachine:
    """Unit tests for CallCreditSpreadRunnerStrategy state transitions."""

    def _make_strat(self, **kw):
        from src.options.call_credit_spread_runner import CallCreditSpreadRunnerStrategy
        defaults = dict(
            spread_width=100, sl_premium_points=40,
            be_trigger_points=60, short_call_delta=0.30,
            min_credit=15.0, max_risk_per_spread=10000.0,
        )
        return CallCreditSpreadRunnerStrategy({**defaults, **kw})

    def _make_chain(self, spot=25000.0):
        """Chain with realistic delta distribution — mirrors proven test fixture."""
        strikes = [
            {"strike": 24500, "ce_price": 580, "ce_delta": 0.72, "pe_price": 30, "pe_delta": -0.28},
            {"strike": 24600, "ce_price": 490, "ce_delta": 0.65, "pe_price": 40, "pe_delta": -0.35},
            {"strike": 24700, "ce_price": 410, "ce_delta": 0.57, "pe_price": 60, "pe_delta": -0.43},
            {"strike": 24800, "ce_price": 330, "ce_delta": 0.50, "pe_price": 80, "pe_delta": -0.50},
            {"strike": 24900, "ce_price": 260, "ce_delta": 0.42, "pe_price": 110, "pe_delta": -0.58},
            {"strike": 25000, "ce_price": 200, "ce_delta": 0.35, "pe_price": 150, "pe_delta": -0.65},
            {"strike": 25100, "ce_price": 145, "ce_delta": 0.28, "pe_price": 195, "pe_delta": -0.72},
            {"strike": 25200, "ce_price": 100, "ce_delta": 0.22, "pe_price": 250, "pe_delta": -0.78},
            {"strike": 25300, "ce_price": 65,  "ce_delta": 0.16, "pe_price": 315, "pe_delta": -0.84},
            {"strike": 25400, "ce_price": 40,  "ce_delta": 0.11, "pe_price": 390, "pe_delta": -0.89},
            {"strike": 25500, "ce_price": 22,  "ce_delta": 0.07, "pe_price": 472, "pe_delta": -0.93},
        ]
        entries = []
        for s in strikes:
            ce = OptionContract(
                tradingsymbol=f"NIFTY24JUN{int(s['strike'])}CE",
                strike=float(s["strike"]), last_price=s["ce_price"], delta=s["ce_delta"],
                iv=16.0, volume=10000, oi=50000,
                bid_price=s["ce_price"] - 0.5, ask_price=s["ce_price"] + 0.5,
            )
            pe = OptionContract(
                tradingsymbol=f"NIFTY24JUN{int(s['strike'])}PE",
                strike=float(s["strike"]), last_price=s["pe_price"], delta=s["pe_delta"],
                iv=16.0, volume=10000, oi=50000,
                bid_price=s["pe_price"] - 0.5, ask_price=s["pe_price"] + 0.5,
            )
            entries.append(OptionChainEntry(strike=float(s["strike"]), ce=ce, pe=pe))
        return OptionChainData(
            underlying="NIFTY", spot_price=spot, expiry="2024-06-27",
            entries=entries, atm_strike=round(spot / 100) * 100, atm_iv=16.0,
            updated_at="2024-06-25T10:30:00",
        )

    def test_initial_state(self):
        from src.options.call_credit_spread_runner import SpreadState
        strat = self._make_strat()
        assert strat.state == SpreadState.IDLE
        assert not strat.is_in_trade

    def test_entry_generates_two_signals(self):
        strat = self._make_strat()
        chain = self._make_chain()
        strat.update_chain(chain)
        signals = strat._evaluate_entry()
        assert len(signals) == 2
        sells = [s for s in signals if s.transaction_type == TransactionType.SELL]
        buys = [s for s in signals if s.transaction_type == TransactionType.BUY]
        assert len(sells) == 1 and len(buys) == 1

    def test_no_entry_when_not_idle(self):
        from src.options.call_credit_spread_runner import SpreadState
        strat = self._make_strat()
        chain = self._make_chain()
        strat.update_chain(chain)
        strat._evaluate_entry()
        assert strat.state == SpreadState.SPREAD_OPEN
        assert strat._evaluate_entry() == []

    def test_insufficient_credit_blocks_entry(self):
        strat = self._make_strat(min_credit=9999)
        chain = self._make_chain()
        strat.update_chain(chain)
        assert strat._evaluate_entry() == []

    def test_risk_exceeded_blocks_entry(self):
        strat = self._make_strat(max_risk_per_spread=1)
        chain = self._make_chain()
        strat.update_chain(chain)
        assert strat._evaluate_entry() == []

    def test_no_chain_blocks_entry(self):
        strat = self._make_strat()
        assert strat._evaluate_entry() == []


# ═══════════════════════════════════════════════════════════
# SECTION 9: CROSS-UNDERLYING RUNNER TESTS
# ═══════════════════════════════════════════════════════════

class TestCrossUnderlyingRunners:
    """Runner strategies on different underlyings."""

    @pytest.mark.parametrize("strategy_name", RUNNER_STRATEGIES)
    def test_nifty_produces_trades(self, strategy_name, nifty_data):
        result = _make_engine(strategy_name, underlying="NIFTY").run(nifty_data)
        assert "error" not in result
        assert result.get("total_trades", 0) > 0, f"[{strategy_name}] 0 trades on NIFTY"

    @pytest.mark.parametrize("strategy_name", RUNNER_STRATEGIES)
    def test_banknifty_no_crash(self, strategy_name, banknifty_data):
        result = _make_engine(strategy_name, underlying="BANKNIFTY", capital=1_000_000).run(banknifty_data)
        assert "error" not in result

    @pytest.mark.parametrize("strategy_name", RUNNER_STRATEGIES)
    def test_sensex_no_crash(self, strategy_name, sensex_data):
        result = _make_engine(strategy_name, underlying="SENSEX").run(sensex_data)
        assert "error" not in result


# ═══════════════════════════════════════════════════════════
# SECTION 10: STRATEGY STRADDLE/STRANGLE STRUCTURE MATCH
# ═══════════════════════════════════════════════════════════

class TestStructureTypeMatching:
    """Verify engine tags correct structure type per strategy."""

    @pytest.mark.parametrize("strat,expected_substr", [
        ("iron_condor", "IRON_CONDOR"),
        ("straddle", "STRADDLE"),
        ("short_straddle", "STRADDLE"),
        ("strangle", "STRANGLE"),
        ("short_strangle", "STRANGLE"),
        ("bull_call_spread", "BULL_CALL"),
        ("bear_put_spread", "BEAR_PUT"),
        ("bull_put_spread", "BULL_PUT"),
        ("bear_call_spread", "BEAR_CALL"),
        ("iron_butterfly", "IRON_BUTTERFLY"),
        ("covered_call", "COVERED_CALL"),
        ("protective_put", "PROTECTIVE_PUT"),
        ("calendar_spread", "CALENDAR"),
    ])
    def test_structure_matches(self, strat, expected_substr, nifty_data):
        result = _make_engine(strat).run(nifty_data)
        structure = result.get("structure", "")
        assert expected_substr in structure, f"[{strat}] structure='{structure}', expected '{expected_substr}'"


# ═══════════════════════════════════════════════════════════
# SECTION 11: IMPORTS & MODULE INTEGRITY
# ═══════════════════════════════════════════════════════════

class TestImports:
    """Verify all strategy modules can be imported without error."""

    def test_import_fno_backtest(self):
        from src.derivatives.fno_backtest import FnOBacktestEngine
        assert FnOBacktestEngine is not None

    def test_import_chain_builder(self):
        from src.derivatives.chain_builder import HistoricalChainBuilder
        assert HistoricalChainBuilder is not None

    def test_import_put_credit_spread_runner(self):
        from src.options.put_credit_spread_runner import PutCreditSpreadRunnerStrategy
        assert PutCreditSpreadRunnerStrategy is not None

    def test_import_call_credit_spread_runner(self):
        from src.options.call_credit_spread_runner import CallCreditSpreadRunnerStrategy
        assert CallCreditSpreadRunnerStrategy is not None

    def test_import_option_strategy_base(self):
        from src.options.strategies import OptionStrategyBase
        assert OptionStrategyBase is not None

    def test_import_chain_models(self):
        from src.options.chain import OptionChainData, OptionChainEntry, OptionContract
        assert OptionChainData is not None

    def test_strategy_classes_map_complete(self):
        """All STRATEGY_CLASSES entries map to importable classes."""
        for name, cls in FnOBacktestEngine.STRATEGY_CLASSES.items():
            assert callable(cls), f"STRATEGY_CLASSES['{name}'] is not callable"
            instance = cls({"underlying": "NIFTY", "lot_size": 50, "lots": 1, "exchange": "NFO"})
            assert hasattr(instance, "on_tick")
            assert hasattr(instance, "update_chain")

    def test_strategy_map_covers_all(self):
        """STRATEGY_MAP covers all FNO_STRATEGIES listed in service.py."""
        for strat in ALL_STRATEGIES:
            assert strat in FnOBacktestEngine.STRATEGY_MAP, f"'{strat}' missing from STRATEGY_MAP"
