"""
Unit tests for CallCreditSpreadRunnerStrategy
===============================================
Covers:
  - Entry signal generation (delta-based, credit validation)
  - State transitions: IDLE → SPREAD_OPEN → RUNNER_LONG_ONLY → IDLE
  - State transitions: IDLE → SPREAD_OPEN → SPREAD_BE → IDLE
  - Profit target exit
  - Edge cases (insufficient credit, risk exceeded, missing chain)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

import pytest

from src.options.call_credit_spread_runner import (
    CallCreditSpreadRunnerStrategy,
    SpreadState,
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
    tradingsymbol: Optional[str] = None,
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
    strikes: Optional[list[dict]] = None,
) -> OptionChainData:
    """Build a synthetic option chain for testing.
    
    Each strike dict: {strike, ce_price, pe_price, ce_delta, pe_delta}
    """
    if strikes is None:
        # Default chain: 250-point interval, covering 24500-25500
        strikes = [
            {"strike": 24500, "ce_price": 580, "ce_delta": 0.72, "pe_price": 30, "pe_delta": -0.28},
            {"strike": 24600, "ce_price": 490, "ce_delta": 0.65, "pe_price": 40, "pe_delta": -0.35},
            {"strike": 24700, "ce_price": 410, "ce_delta": 0.57, "pe_price": 60, "pe_delta": -0.43},
            {"strike": 24800, "ce_price": 330, "ce_delta": 0.50, "pe_price": 80, "pe_delta": -0.50},
            {"strike": 24900, "ce_price": 260, "ce_delta": 0.42, "pe_price": 110, "pe_delta": -0.58},
            {"strike": 25000, "ce_price": 200, "ce_delta": 0.35, "pe_price": 150, "pe_delta": -0.65},
            {"strike": 25100, "ce_price": 145, "ce_delta": 0.28, "pe_price": 195, "pe_delta": -0.72},
            {"strike": 25200, "ce_price": 100, "ce_delta": 0.22, "pe_price": 250, "pe_delta": -0.78},
            {"strike": 25300, "ce_price": 65, "ce_delta": 0.16, "pe_price": 315, "pe_delta": -0.84},
            {"strike": 25400, "ce_price": 40, "ce_delta": 0.11, "pe_price": 390, "pe_delta": -0.89},
            {"strike": 25500, "ce_price": 22, "ce_delta": 0.07, "pe_price": 472, "pe_delta": -0.93},
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
) -> OptionChainData:
    """Return a new chain with updated prices for specific strikes.
    
    updates: {strike: {"ce_price": ..., "pe_price": ...}}
    """
    new_entries = []
    for entry in chain.entries:
        upd = updates.get(entry.strike, {})
        ce = entry.ce
        pe = entry.pe
        if ce and "ce_price" in upd:
            ce = _make_contract(
                ce.strike, "CE", upd["ce_price"],
                ce.delta, ce.tradingsymbol,
            )
        if pe and "pe_price" in upd:
            pe = _make_contract(
                pe.strike, "PE", upd["pe_price"],
                pe.delta, pe.tradingsymbol,
            )
        new_entries.append(OptionChainEntry(strike=entry.strike, ce=ce, pe=pe))

    return OptionChainData(
        underlying=chain.underlying,
        spot_price=updates.get("spot", chain.spot_price),
        expiry=chain.expiry,
        entries=new_entries,
        atm_strike=chain.atm_strike,
        atm_iv=chain.atm_iv,
        updated_at=chain.updated_at,
    )


# ═══════════════════════════════════════════════════════════
# Test class
# ═══════════════════════════════════════════════════════════

class TestCallCreditSpreadRunner:
    """Tests for CallCreditSpreadRunnerStrategy."""

    def _make_strategy(self, **overrides) -> CallCreditSpreadRunnerStrategy:
        params = {
            "spread_width": 100,
            "sl_premium_points": 40,
            "be_trigger_points": 60,
            "profit_target_pct": 80.0,
            "short_call_delta": 0.30,
            "max_risk_per_spread": 10000.0,
            "min_credit": 15.0,
            **overrides,
        }
        return CallCreditSpreadRunnerStrategy(params)

    # ──────────────────────────────────────────────────
    # Construction / state
    # ──────────────────────────────────────────────────

    def test_initial_state_is_idle(self):
        strat = self._make_strategy()
        assert strat.state == SpreadState.IDLE
        assert not strat.is_in_trade

    def test_default_params(self):
        strat = CallCreditSpreadRunnerStrategy()
        assert strat.params["spread_width"] == 100
        assert strat.params["sl_premium_points"] == 40
        assert strat.params["be_trigger_points"] == 60
        assert strat.params["short_call_delta"] == 0.30
        assert strat.params["lot_size"] == 50  # from OptionStrategyBase

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

        assert len(signals) == 2, f"Expected 2 signals, got {len(signals)}"

        # Short call signal
        short_sig = [s for s in signals if s.transaction_type == TransactionType.SELL]
        long_sig = [s for s in signals if s.transaction_type == TransactionType.BUY]
        assert len(short_sig) == 1, "Must have exactly 1 SELL signal"
        assert len(long_sig) == 1, "Must have exactly 1 BUY signal"

        # State changed
        assert strat.state == SpreadState.SPREAD_OPEN
        assert strat.is_in_trade

    def test_entry_short_call_is_near_target_delta(self):
        """The short call should be close to the 0.30 delta target."""
        strat = self._make_strategy(short_call_delta=0.30)
        chain = _build_chain(spot=25000.0)
        strat.update_chain(chain)

        signals = strat._evaluate_entry()
        assert len(signals) == 2

        short_sig = [s for s in signals if s.transaction_type == TransactionType.SELL][0]
        # In our chain, 25100CE has delta 0.28 — closest to 0.30
        assert "25100" in short_sig.tradingsymbol or "25000" in short_sig.tradingsymbol

    def test_entry_long_call_is_spread_width_higher(self):
        """Long call strike = short call strike + spread_width."""
        strat = self._make_strategy(spread_width=100)
        chain = _build_chain(spot=25000.0)
        strat.update_chain(chain)

        signals = strat._evaluate_entry()
        assert len(signals) == 2

        short_meta = [s for s in signals if s.transaction_type == TransactionType.SELL][0].metadata
        long_meta = [s for s in signals if s.transaction_type == TransactionType.BUY][0].metadata

        short_strike = short_meta["short_call_strike"]
        long_strike = long_meta["long_call_strike"]
        assert long_strike >= short_strike + 100

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
        # SL premium = entry premium + 40
        assert strat._short_leg["sl_premium"] == strat._short_entry_premium + 40

    def test_entry_no_chain_returns_empty(self):
        strat = self._make_strategy()
        signals = strat._evaluate_entry()
        assert signals == []

    def test_entry_insufficient_credit_returns_empty(self):
        """If credit < min_credit, reject."""
        strat = self._make_strategy(min_credit=500)  # very high threshold
        chain = _build_chain(spot=25000.0)
        strat.update_chain(chain)

        signals = strat._evaluate_entry()
        assert signals == []

    def test_entry_risk_exceeded_returns_empty(self):
        """If max_loss_per_lot > max_risk_per_spread, reject."""
        strat = self._make_strategy(max_risk_per_spread=50)  # very low
        chain = _build_chain(spot=25000.0)
        strat.update_chain(chain)

        signals = strat._evaluate_entry()
        assert signals == []

    def test_no_double_entry(self):
        """If already in a trade (SPREAD_OPEN), no new entry signals."""
        strat = self._make_strategy()
        chain = _build_chain(spot=25000.0)
        strat.update_chain(chain)

        signals1 = strat._evaluate_entry()
        assert len(signals1) == 2

        # Try again while still SPREAD_OPEN
        signals2 = strat._evaluate_entry()
        assert signals2 == []

    # ──────────────────────────────────────────────────
    # on_tick integration
    # ──────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_on_tick_generates_entry(self):
        strat = self._make_strategy()
        chain = _build_chain(spot=25000.0)
        strat.update_chain(chain)

        signals = await strat.on_tick([])
        assert len(signals) == 2
        assert strat.state == SpreadState.SPREAD_OPEN

    @pytest.mark.asyncio
    async def test_on_tick_no_entry_without_chain(self):
        strat = self._make_strategy()
        signals = await strat.on_tick([])
        assert signals == []

    # ──────────────────────────────────────────────────
    # Scenario 1: Short SL hit → Runner
    # ──────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_short_sl_hit_transitions_to_runner(self):
        """When short call premium rises above SL trigger → close short, keep long."""
        strat = self._make_strategy()
        chain = _build_chain(spot=25000.0)
        strat.update_chain(chain)

        # Enter
        await strat.on_tick([])
        assert strat.state == SpreadState.SPREAD_OPEN
        short_strike = strat._short_leg["strike"]
        sl_premium = strat._short_leg["sl_premium"]

        # Simulate: short call premium rises above SL
        # sl_premium = entry + 40, so set price to sl_premium + 5
        new_price = sl_premium + 5
        updated = _update_chain_prices(chain, {short_strike: {"ce_price": new_price}})
        strat.update_chain(updated)

        exit_signals = await strat.on_tick([])

        # Should get 1 signal: buy back the short call
        assert len(exit_signals) == 1
        assert exit_signals[0].transaction_type == TransactionType.BUY
        assert exit_signals[0].metadata["leg"] == "short_call_exit"
        assert exit_signals[0].metadata["reason"] == "sl_hit"

        # State should be RUNNER_LONG_ONLY
        assert strat.state == SpreadState.RUNNER_LONG_ONLY
        assert strat._long_leg is not None
        assert strat._sl_triggered_at is not None

    @pytest.mark.asyncio
    async def test_runner_be_exit(self):
        """In RUNNER state, if long call drops to entry premium → exit."""
        strat = self._make_strategy()
        chain = _build_chain(spot=25000.0)
        strat.update_chain(chain)

        # Enter
        await strat.on_tick([])
        short_strike = strat._short_leg["strike"]
        long_strike = strat._long_leg["strike"]
        sl_premium = strat._short_leg["sl_premium"]
        long_entry = strat._long_entry_premium

        # Trigger short SL
        updated = _update_chain_prices(chain, {short_strike: {"ce_price": sl_premium + 5}})
        strat.update_chain(updated)
        await strat.on_tick([])
        assert strat.state == SpreadState.RUNNER_LONG_ONLY

        # Now simulate long call dropping to entry premium (BE exit)
        updated2 = _update_chain_prices(chain, {long_strike: {"ce_price": long_entry}})
        strat.update_chain(updated2)

        exit_signals = await strat.on_tick([])
        assert len(exit_signals) == 1
        assert exit_signals[0].transaction_type == TransactionType.SELL
        assert exit_signals[0].metadata["reason"] == "runner_be_exit"

        # Should be back to IDLE
        assert strat.state == SpreadState.IDLE
        assert not strat.is_in_trade

    # ──────────────────────────────────────────────────
    # Scenario 2: Market moves in favour → Spread BE
    # ──────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_be_trigger_transitions_to_spread_be(self):
        """When spot drops 60+ pts (favourable for call credit spread) → SPREAD_BE."""
        strat = self._make_strategy(be_trigger_points=60)
        chain = _build_chain(spot=25000.0)
        strat.update_chain(chain)

        # Enter
        await strat.on_tick([])
        assert strat.state == SpreadState.SPREAD_OPEN

        # Simulate spot drop of 70 pts (favourable for bear position)
        updated = _build_chain(spot=24930.0)  # 25000 - 70 = 24930
        strat.update_chain(updated)

        exit_signals = await strat.on_tick([])
        # BE trigger doesn't generate exit signals, just state change
        assert exit_signals == []
        assert strat.state == SpreadState.SPREAD_BE

    @pytest.mark.asyncio
    async def test_spread_be_exit_when_credit_lost(self):
        """In SPREAD_BE state, if spread cost >= entry credit → close spread."""
        strat = self._make_strategy(be_trigger_points=60)
        chain = _build_chain(spot=25000.0)
        strat.update_chain(chain)

        # Enter
        await strat.on_tick([])
        entry_credit = strat._entry_credit
        short_strike = strat._short_leg["strike"]
        long_strike = strat._long_leg["strike"]

        # Trigger BE state (spot drops 70)
        updated = _build_chain(spot=24930.0)
        strat.update_chain(updated)
        await strat.on_tick([])
        assert strat.state == SpreadState.SPREAD_BE

        # Now simulate spread cost rising to entry credit
        # spread_cost = short_premium - long_premium >= entry_credit
        # Make short premium high and long premium low
        updated2 = _update_chain_prices(
            updated,
            {
                short_strike: {"ce_price": entry_credit + 10},
                long_strike: {"ce_price": 5},
            },
        )
        strat.update_chain(updated2)

        exit_signals = await strat.on_tick([])
        assert len(exit_signals) == 2  # close both legs
        assert strat.state == SpreadState.IDLE

    # ──────────────────────────────────────────────────
    # Profit target exits
    # ──────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_profit_target_closes_spread(self):
        """If profit captured >= 80% of max profit → close spread."""
        strat = self._make_strategy(profit_target_pct=80.0)
        chain = _build_chain(spot=25000.0)
        strat.update_chain(chain)

        # Enter
        await strat.on_tick([])
        assert strat.state == SpreadState.SPREAD_OPEN
        entry_credit = strat._entry_credit
        short_strike = strat._short_leg["strike"]
        long_strike = strat._long_leg["strike"]

        # Simulate spread narrowing significantly (both premiums drop)
        # profit = entry_credit - current_spread_cost
        # We need profit >= 0.8 * entry_credit
        # → current_spread_cost <= 0.2 * entry_credit
        small_cost = entry_credit * 0.1  # 10% of credit remaining
        # short_premium - long_premium = small_cost
        updated = _update_chain_prices(
            chain,
            {
                short_strike: {"ce_price": small_cost + 2},
                long_strike: {"ce_price": 2},
            },
        )
        strat.update_chain(updated)

        exit_signals = await strat.on_tick([])
        assert len(exit_signals) == 2
        # Both legs closed
        sell_sigs = [s for s in exit_signals if s.transaction_type == TransactionType.SELL]
        buy_sigs = [s for s in exit_signals if s.transaction_type == TransactionType.BUY]
        assert len(sell_sigs) == 1  # sell long call
        assert len(buy_sigs) == 1  # buy back short call
        assert any(s.metadata.get("reason") == "profit_target_hit" for s in exit_signals)
        assert strat.state == SpreadState.IDLE

    # ──────────────────────────────────────────────────
    # get_status
    # ──────────────────────────────────────────────────

    def test_get_status_idle(self):
        strat = self._make_strategy()
        status = strat.get_status()
        assert status["state"] == "idle"
        assert status["in_trade"] is False
        assert status["strategy"] == "call_credit_spread_runner"

    def test_get_status_in_trade(self):
        strat = self._make_strategy()
        chain = _build_chain(spot=25000.0)
        strat.update_chain(chain)
        strat._evaluate_entry()

        status = strat.get_status()
        assert status["state"] == "spread_open"
        assert status["in_trade"] is True
        assert status["entry_credit"] > 0
        assert status["short_leg"] is not None
        assert status["long_leg"] is not None

    # ──────────────────────────────────────────────────
    # Signal metadata quality
    # ──────────────────────────────────────────────────

    def test_entry_signals_have_correct_metadata(self):
        strat = self._make_strategy()
        chain = _build_chain(spot=25000.0)
        strat.update_chain(chain)

        signals = strat._evaluate_entry()
        assert len(signals) == 2

        for sig in signals:
            assert sig.strategy_name == "call_credit_spread_runner"
            assert sig.metadata["strategy_type"] == "call_credit_spread_runner"
            assert sig.metadata["net_credit"] > 0
            assert sig.metadata["spread_width"] >= 100
            assert sig.metadata["spot_at_entry"] == 25000.0
            assert "leg" in sig.metadata

    def test_exit_signals_have_exit_metadata(self):
        """Exit signals should have signal_type='exit' and a reason."""
        strat = self._make_strategy()
        chain = _build_chain(spot=25000.0)
        strat.update_chain(chain)

        # Enter
        strat._evaluate_entry()
        short_strike = strat._short_leg["strike"]
        sl_premium = strat._short_leg["sl_premium"]

        # Trigger SL
        updated = _update_chain_prices(chain, {short_strike: {"ce_price": sl_premium + 5}})
        strat.update_chain(updated)

        exit_signals = strat._evaluate_exit([])
        assert len(exit_signals) == 1
        sig = exit_signals[0]
        assert sig.metadata["signal_type"] == "exit"
        assert sig.metadata["reason"] == "sl_hit"
        assert sig.metadata["strategy_type"] == "call_credit_spread_runner"

    # ──────────────────────────────────────────────────
    # State enum
    # ──────────────────────────────────────────────────

    def test_spread_state_values(self):
        assert SpreadState.IDLE.value == "idle"
        assert SpreadState.SPREAD_OPEN.value == "spread_open"
        assert SpreadState.RUNNER_LONG_ONLY.value == "runner_long"
        assert SpreadState.SPREAD_BE.value == "spread_be"
        assert SpreadState.CLOSED.value == "closed"

    # ──────────────────────────────────────────────────
    # Reset / re-entry
    # ──────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_can_reenter_after_exit(self):
        """After a full exit, state returns to IDLE and can re-enter."""
        strat = self._make_strategy()
        chain = _build_chain(spot=25000.0)
        strat.update_chain(chain)

        # Enter
        await strat.on_tick([])
        assert strat.state == SpreadState.SPREAD_OPEN
        short_strike = strat._short_leg["strike"]
        long_strike = strat._long_leg["strike"]
        entry_credit = strat._entry_credit

        # Close via profit target
        updated = _update_chain_prices(
            chain,
            {
                short_strike: {"ce_price": entry_credit * 0.1 + 2},
                long_strike: {"ce_price": 2},
            },
        )
        strat.update_chain(updated)
        exit_signals = await strat.on_tick([])
        assert strat.state == SpreadState.IDLE

        # Re-enter
        strat.update_chain(chain)  # fresh chain
        re_entry = await strat.on_tick([])
        assert len(re_entry) == 2
        assert strat.state == SpreadState.SPREAD_OPEN
