"""
Unit tests for PutCreditSpreadRunnerStrategy
===============================================
Covers:
  - Entry signal generation (delta-based, credit validation)
  - State transitions: IDLE → SPREAD_OPEN → RUNNER_LONG_ONLY → IDLE
  - State transitions: IDLE → SPREAD_OPEN → SPREAD_BE → IDLE
  - Profit target exit
  - Edge cases (insufficient credit, risk exceeded, missing chain)
"""

from __future__ import annotations

import pytest

from src.options.put_credit_spread_runner import (
    PutCreditSpreadRunnerStrategy,
    PutSpreadState,
)
from src.options.chain import OptionChainData, OptionChainEntry, OptionContract
from src.data.models import TransactionType


# ═══════════════════════════════════════════════════════════
# Helpers — build a synthetic option chain
# ═══════════════════════════════════════════════════════════

def _make_contract(
    strike: float,
    opt_type: str,
    last_price: float,
    delta: float,
    tradingsymbol: str | None = None,
) -> OptionContract:
    sym = tradingsymbol or f"NIFTY24JUN{int(strike)}{opt_type}"
    return OptionContract(
        tradingsymbol=sym,
        strike=strike,
        last_price=last_price,
        delta=delta,
        iv=16.0,
        volume=10000,
        oi=50000,
        bid_price=last_price - 0.5,
        ask_price=last_price + 0.5,
    )


def _build_chain(
    spot: float = 25000.0,
    strikes: list[dict] | None = None,
) -> OptionChainData:
    if strikes is None:
        # Chain needs strikes below 24500 so spread_width=100 can find a long put
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
            {"strike": 25300, "ce_price": 65,  "ce_delta": 0.16, "pe_price": 315, "pe_delta": -0.84},
            {"strike": 25400, "ce_price": 40,  "ce_delta": 0.11, "pe_price": 390, "pe_delta": -0.89},
            {"strike": 25500, "ce_price": 22,  "ce_delta": 0.07, "pe_price": 472, "pe_delta": -0.93},
        ]

    entries = []
    for s in strikes:
        ce = _make_contract(s["strike"], "CE", s["ce_price"], s["ce_delta"])
        pe = _make_contract(s["strike"], "PE", s["pe_price"], s["pe_delta"])
        entries.append(OptionChainEntry(strike=s["strike"], ce=ce, pe=pe))

    return OptionChainData(
        underlying="NIFTY",
        spot_price=spot,
        expiry="2024-06-27",
        entries=entries,
        atm_strike=round(spot / 100) * 100,
        atm_iv=16.0,
        updated_at="2024-06-25T10:30:00",
    )


def _update_chain_prices(
    chain: OptionChainData,
    updates: dict[float, dict],
    new_spot: float | None = None,
) -> OptionChainData:
    new_entries = []
    for entry in chain.entries:
        upd = updates.get(entry.strike, {})
        ce = entry.ce
        pe = entry.pe
        if ce and "ce_price" in upd:
            ce = _make_contract(ce.strike, "CE", upd["ce_price"], ce.delta, ce.tradingsymbol)
        if pe and "pe_price" in upd:
            pe = _make_contract(pe.strike, "PE", upd["pe_price"], pe.delta, pe.tradingsymbol)
        new_entries.append(OptionChainEntry(strike=entry.strike, ce=ce, pe=pe))

    return OptionChainData(
        underlying=chain.underlying,
        spot_price=new_spot if new_spot is not None else chain.spot_price,
        expiry=chain.expiry,
        entries=new_entries,
        atm_strike=chain.atm_strike,
        atm_iv=chain.atm_iv,
        updated_at=chain.updated_at,
    )


# ═══════════════════════════════════════════════════════════
# Test class
# ═══════════════════════════════════════════════════════════

class TestPutCreditSpreadRunner:

    def _make_strategy(self, **overrides) -> PutCreditSpreadRunnerStrategy:
        params = {
            "spread_width": 100,
            "sl_premium_points": 40,
            "be_trigger_points": 60,
            "profit_target_pct": 80.0,
            "short_put_delta": 0.50,   # ATM-ish gives enough credit
            "max_risk_per_spread": 10000.0,
            "min_credit": 15.0,
            **overrides,
        }
        return PutCreditSpreadRunnerStrategy(params)

    # ──────────────────────────────────────────────────
    # Construction / state
    # ──────────────────────────────────────────────────

    def test_initial_state_is_idle(self):
        strat = self._make_strategy()
        assert strat.state == PutSpreadState.IDLE
        assert not strat.is_in_trade

    def test_default_params(self):
        strat = PutCreditSpreadRunnerStrategy()
        assert strat.params["spread_width"] == 100
        assert strat.params["sl_premium_points"] == 40
        assert strat.params["short_put_delta"] == 0.30
        assert strat.params["lot_size"] == 50

    def test_param_override(self):
        strat = self._make_strategy(spread_width=200, sl_premium_points=50)
        assert strat.params["spread_width"] == 200
        assert strat.params["sl_premium_points"] == 50

    # ──────────────────────────────────────────────────
    # Entry
    # ──────────────────────────────────────────────────

    def test_entry_generates_two_signals(self):
        strat = self._make_strategy()
        chain = _build_chain(spot=25000.0)
        strat.update_chain(chain)

        signals = strat._evaluate_entry()
        assert len(signals) == 2

        short_sig = [s for s in signals if s.transaction_type == TransactionType.SELL]
        long_sig = [s for s in signals if s.transaction_type == TransactionType.BUY]
        assert len(short_sig) == 1
        assert len(long_sig) == 1

        assert strat.state == PutSpreadState.SPREAD_OPEN
        assert strat.is_in_trade

    def test_entry_uses_put_options(self):
        """Both legs should be PE contracts."""
        strat = self._make_strategy()
        chain = _build_chain(spot=25000.0)
        strat.update_chain(chain)

        signals = strat._evaluate_entry()
        assert len(signals) == 2
        for sig in signals:
            assert "PE" in sig.tradingsymbol, f"Expected PE in {sig.tradingsymbol}"

    def test_entry_short_put_near_target_delta(self):
        """Short put should be near target abs delta."""
        strat = self._make_strategy(short_put_delta=0.50)
        chain = _build_chain(spot=25000.0)
        strat.update_chain(chain)

        signals = strat._evaluate_entry()
        assert len(signals) == 2
        # In our chain, 24800PE has delta -0.50 (abs 0.50)
        short_sig = [s for s in signals if s.transaction_type == TransactionType.SELL][0]
        assert "24800" in short_sig.tradingsymbol

    def test_entry_long_put_is_lower_strike(self):
        """Long put strike = short put strike - spread_width."""
        strat = self._make_strategy(spread_width=100)
        chain = _build_chain(spot=25000.0)
        strat.update_chain(chain)

        signals = strat._evaluate_entry()
        assert len(signals) == 2

        short_meta = [s for s in signals if s.transaction_type == TransactionType.SELL][0].metadata
        long_meta = [s for s in signals if s.transaction_type == TransactionType.BUY][0].metadata

        # Long put should be BELOW short put
        assert long_meta["long_put_strike"] < short_meta["short_put_strike"]

    def test_entry_records_credit_and_sl(self):
        strat = self._make_strategy()
        chain = _build_chain(spot=25000.0)
        strat.update_chain(chain)

        signals = strat._evaluate_entry()
        assert len(signals) == 2

        assert strat._entry_credit > 0
        assert strat._entry_spot == 25000.0
        assert strat._short_leg is not None
        assert strat._long_leg is not None
        assert strat._short_leg["sl_premium"] == strat._short_entry_premium + 40

    def test_entry_no_chain_returns_empty(self):
        strat = self._make_strategy()
        assert strat._evaluate_entry() == []

    def test_entry_insufficient_credit_returns_empty(self):
        strat = self._make_strategy(min_credit=500)
        chain = _build_chain(spot=25000.0)
        strat.update_chain(chain)
        assert strat._evaluate_entry() == []

    def test_entry_risk_exceeded_returns_empty(self):
        strat = self._make_strategy(max_risk_per_spread=50)
        chain = _build_chain(spot=25000.0)
        strat.update_chain(chain)
        assert strat._evaluate_entry() == []

    def test_no_double_entry(self):
        strat = self._make_strategy()
        chain = _build_chain(spot=25000.0)
        strat.update_chain(chain)

        signals1 = strat._evaluate_entry()
        assert len(signals1) == 2
        signals2 = strat._evaluate_entry()
        assert signals2 == []

    # ──────────────────────────────────────────────────
    # on_tick integration
    # ──────────────────────────────────────────────────

    async def test_on_tick_generates_entry(self):
        strat = self._make_strategy()
        chain = _build_chain(spot=25000.0)
        strat.update_chain(chain)

        signals = await strat.on_tick([])
        assert len(signals) == 2
        assert strat.state == PutSpreadState.SPREAD_OPEN

    async def test_on_tick_no_entry_without_chain(self):
        strat = self._make_strategy()
        signals = await strat.on_tick([])
        assert signals == []

    # ──────────────────────────────────────────────────
    # Scenario 1: Short SL hit → Runner
    # ──────────────────────────────────────────────────

    async def test_short_sl_hit_transitions_to_runner(self):
        """When short put premium rises above SL trigger → close short, keep long."""
        strat = self._make_strategy()
        chain = _build_chain(spot=25000.0)
        strat.update_chain(chain)

        await strat.on_tick([])
        assert strat.state == PutSpreadState.SPREAD_OPEN
        short_strike = strat._short_leg["strike"]
        sl_premium = strat._short_leg["sl_premium"]

        # Simulate short put premium rises above SL
        updated = _update_chain_prices(chain, {short_strike: {"pe_price": sl_premium + 5}})
        strat.update_chain(updated)

        exit_signals = await strat.on_tick([])
        assert len(exit_signals) == 1
        assert exit_signals[0].transaction_type == TransactionType.BUY
        assert exit_signals[0].metadata["leg"] == "short_put_exit"
        assert exit_signals[0].metadata["reason"] == "sl_hit"
        assert strat.state == PutSpreadState.RUNNER_LONG_ONLY

    async def test_runner_be_exit(self):
        """In RUNNER state, if long put drops to entry premium → exit."""
        strat = self._make_strategy()
        chain = _build_chain(spot=25000.0)
        strat.update_chain(chain)

        await strat.on_tick([])
        short_strike = strat._short_leg["strike"]
        long_strike = strat._long_leg["strike"]
        sl_premium = strat._short_leg["sl_premium"]
        long_entry = strat._long_entry_premium

        # Trigger short SL
        updated = _update_chain_prices(chain, {short_strike: {"pe_price": sl_premium + 5}})
        strat.update_chain(updated)
        await strat.on_tick([])
        assert strat.state == PutSpreadState.RUNNER_LONG_ONLY

        # Long put drops to entry premium → BE exit
        updated2 = _update_chain_prices(chain, {long_strike: {"pe_price": long_entry}})
        strat.update_chain(updated2)
        exit_signals = await strat.on_tick([])

        assert len(exit_signals) == 1
        assert exit_signals[0].transaction_type == TransactionType.SELL
        assert exit_signals[0].metadata["reason"] == "runner_be_exit"
        assert strat.state == PutSpreadState.IDLE

    # ──────────────────────────────────────────────────
    # Scenario 2: Market moves in favour → Spread BE
    # ──────────────────────────────────────────────────

    async def test_be_trigger_transitions_to_spread_be(self):
        """When spot rises 60+ pts (favourable for put credit spread) → SPREAD_BE."""
        strat = self._make_strategy(be_trigger_points=60)
        chain = _build_chain(spot=25000.0)
        strat.update_chain(chain)

        await strat.on_tick([])
        assert strat.state == PutSpreadState.SPREAD_OPEN

        # Spot rises 70 pts (bullish = good for put credit spread)
        updated = _build_chain(spot=25070.0)
        strat.update_chain(updated)

        exit_signals = await strat.on_tick([])
        assert exit_signals == []
        assert strat.state == PutSpreadState.SPREAD_BE

    async def test_spread_be_exit_when_credit_lost(self):
        """In SPREAD_BE state, if spread cost >= entry credit → close."""
        strat = self._make_strategy(be_trigger_points=60)
        chain = _build_chain(spot=25000.0)
        strat.update_chain(chain)

        await strat.on_tick([])
        entry_credit = strat._entry_credit
        short_strike = strat._short_leg["strike"]
        long_strike = strat._long_leg["strike"]

        # Trigger BE (spot rises 70)
        updated = _build_chain(spot=25070.0)
        strat.update_chain(updated)
        await strat.on_tick([])
        assert strat.state == PutSpreadState.SPREAD_BE

        # Spread cost rises to entry credit
        updated2 = _update_chain_prices(
            updated,
            {short_strike: {"pe_price": entry_credit + 10}, long_strike: {"pe_price": 5}},
        )
        strat.update_chain(updated2)

        exit_signals = await strat.on_tick([])
        assert len(exit_signals) == 2
        assert strat.state == PutSpreadState.IDLE

    # ──────────────────────────────────────────────────
    # Profit target
    # ──────────────────────────────────────────────────

    async def test_profit_target_closes_spread(self):
        strat = self._make_strategy(profit_target_pct=80.0)
        chain = _build_chain(spot=25000.0)
        strat.update_chain(chain)

        await strat.on_tick([])
        entry_credit = strat._entry_credit
        short_strike = strat._short_leg["strike"]
        long_strike = strat._long_leg["strike"]

        small_cost = entry_credit * 0.1
        updated = _update_chain_prices(
            chain,
            {short_strike: {"pe_price": small_cost + 2}, long_strike: {"pe_price": 2}},
        )
        strat.update_chain(updated)

        exit_signals = await strat.on_tick([])
        assert len(exit_signals) == 2
        assert any(s.metadata.get("reason") == "profit_target_hit" for s in exit_signals)
        assert strat.state == PutSpreadState.IDLE

    # ──────────────────────────────────────────────────
    # get_status
    # ──────────────────────────────────────────────────

    def test_get_status_idle(self):
        strat = self._make_strategy()
        status = strat.get_status()
        assert status["state"] == "idle"
        assert status["in_trade"] is False
        assert status["strategy"] == "put_credit_spread_runner"

    def test_get_status_in_trade(self):
        strat = self._make_strategy()
        chain = _build_chain(spot=25000.0)
        strat.update_chain(chain)
        strat._evaluate_entry()

        status = strat.get_status()
        assert status["state"] == "spread_open"
        assert status["in_trade"] is True
        assert status["entry_credit"] > 0

    # ──────────────────────────────────────────────────
    # Signal metadata quality
    # ──────────────────────────────────────────────────

    def test_entry_signals_have_correct_metadata(self):
        strat = self._make_strategy()
        chain = _build_chain(spot=25000.0)
        strat.update_chain(chain)

        signals = strat._evaluate_entry()
        for sig in signals:
            assert sig.strategy_name == "put_credit_spread_runner"
            assert sig.metadata["strategy_type"] == "put_credit_spread_runner"
            assert sig.metadata["net_credit"] > 0
            assert sig.metadata["spread_width"] >= 100
            assert sig.metadata["spot_at_entry"] == 25000.0

    def test_exit_signals_have_exit_metadata(self):
        strat = self._make_strategy()
        chain = _build_chain(spot=25000.0)
        strat.update_chain(chain)

        strat._evaluate_entry()
        short_strike = strat._short_leg["strike"]
        sl_premium = strat._short_leg["sl_premium"]

        updated = _update_chain_prices(chain, {short_strike: {"pe_price": sl_premium + 5}})
        strat.update_chain(updated)

        exit_signals = strat._evaluate_exit([])
        assert len(exit_signals) == 1
        sig = exit_signals[0]
        assert sig.metadata["signal_type"] == "exit"
        assert sig.metadata["reason"] == "sl_hit"
        assert sig.metadata["strategy_type"] == "put_credit_spread_runner"

    # ──────────────────────────────────────────────────
    # State enum
    # ──────────────────────────────────────────────────

    def test_put_spread_state_values(self):
        assert PutSpreadState.IDLE.value == "idle"
        assert PutSpreadState.SPREAD_OPEN.value == "spread_open"
        assert PutSpreadState.RUNNER_LONG_ONLY.value == "runner_long"
        assert PutSpreadState.SPREAD_BE.value == "spread_be"
        assert PutSpreadState.CLOSED.value == "closed"

    # ──────────────────────────────────────────────────
    # Re-entry
    # ──────────────────────────────────────────────────

    async def test_can_reenter_after_exit(self):
        strat = self._make_strategy()
        chain = _build_chain(spot=25000.0)
        strat.update_chain(chain)

        await strat.on_tick([])
        assert strat.state == PutSpreadState.SPREAD_OPEN
        short_strike = strat._short_leg["strike"]
        long_strike = strat._long_leg["strike"]
        entry_credit = strat._entry_credit

        # Close via profit target
        updated = _update_chain_prices(
            chain,
            {short_strike: {"pe_price": entry_credit * 0.1 + 2}, long_strike: {"pe_price": 2}},
        )
        strat.update_chain(updated)
        await strat.on_tick([])
        assert strat.state == PutSpreadState.IDLE

        # Re-enter
        strat.update_chain(chain)
        re_entry = await strat.on_tick([])
        assert len(re_entry) == 2
        assert strat.state == PutSpreadState.SPREAD_OPEN
