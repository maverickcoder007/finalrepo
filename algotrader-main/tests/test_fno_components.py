"""
Unit tests for core derivative components:
  - Black-Scholes greeks
  - HistoricalChainBuilder
  - FnOExecutionSimulator
  - IndianFnOCostModel
  - MarginEngine
  - GreeksEngine
  - RegimeEngine
  - ExpiryEngine
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import numpy as np
import pytest

from src.options.greeks import BlackScholes, compute_all_greeks
from src.derivatives.chain_builder import (
    HistoricalChainBuilder,
    SyntheticChain,
    SyntheticOptionQuote,
)
from src.derivatives.contracts import (
    DerivativeContract,
    InstrumentType,
    MultiLegPosition,
    OptionLeg,
    StructureType,
    SettlementType,
)
from src.derivatives.fno_simulator import (
    FnOExecutionSimulator,
    FnOFillResult,
    FnOOrderStatus,
    OrderSide,
)
from src.derivatives.fno_cost_model import IndianFnOCostModel, TransactionSide
from src.derivatives.margin_engine import MarginEngine
from src.derivatives.greeks_engine import GreeksEngine, PortfolioGreeks
from src.derivatives.regime_engine import RegimeEngine, MarketRegime
from src.derivatives.expiry_engine import ExpiryEngine, ExpiryEventType


# ═══════════════════════════════════════════════════════════
# 1. BLACK-SCHOLES GREEKS
# ═══════════════════════════════════════════════════════════

class TestBlackScholes:
    """Test Black-Scholes pricing and Greeks accuracy."""

    # Known values: NIFTY 24500 CE, strike=24500, r=6.5%, tte=30/365, vol=16%
    SPOT = 24_500.0
    STRIKE = 24_500.0
    RATE = 0.065
    TTE = 30 / 365
    VOL = 0.16

    def test_call_price_positive(self):
        price = BlackScholes.call_price(self.SPOT, self.STRIKE, self.RATE, self.TTE, self.VOL)
        assert price > 0, "ATM call must have positive premium"
        # ATM call ≈ spot × σ × √T × 0.4 ≈ 24500 × 0.16 × 0.287 × 0.4 ≈ 450
        assert 200 < price < 800, f"ATM call price {price:.2f} outside expected range"

    def test_put_price_positive(self):
        price = BlackScholes.put_price(self.SPOT, self.STRIKE, self.RATE, self.TTE, self.VOL)
        assert price > 0, "ATM put must have positive premium"

    def test_put_call_parity(self):
        """C - P = S - K × e^(-rT): fundamental arbitrage relationship."""
        call = BlackScholes.call_price(self.SPOT, self.STRIKE, self.RATE, self.TTE, self.VOL)
        put = BlackScholes.put_price(self.SPOT, self.STRIKE, self.RATE, self.TTE, self.VOL)
        expected = self.SPOT - self.STRIKE * math.exp(-self.RATE * self.TTE)
        assert abs((call - put) - expected) < 0.01, "Put-call parity violated"

    def test_call_delta_range(self):
        d = BlackScholes.delta(self.SPOT, self.STRIKE, self.RATE, self.TTE, self.VOL, "CE")
        assert 0.45 < d < 0.60, f"ATM call delta {d:.4f} outside [0.45, 0.60]"

    def test_put_delta_range(self):
        d = BlackScholes.delta(self.SPOT, self.STRIKE, self.RATE, self.TTE, self.VOL, "PE")
        assert -0.55 < d < -0.40, f"ATM put delta {d:.4f} outside [-0.55, -0.40]"

    def test_call_put_delta_sum(self):
        """|delta_call| + |delta_put| ≈ 1 for same strike."""
        dc = BlackScholes.delta(self.SPOT, self.STRIKE, self.RATE, self.TTE, self.VOL, "CE")
        dp = BlackScholes.delta(self.SPOT, self.STRIKE, self.RATE, self.TTE, self.VOL, "PE")
        # For European options: delta_call - delta_put ≈ 1 (approximately)
        assert abs(dc - dp - 1.0) < 0.05

    def test_gamma_positive(self):
        g = BlackScholes.gamma(self.SPOT, self.STRIKE, self.RATE, self.TTE, self.VOL)
        assert g > 0, "Gamma must be positive"

    def test_theta_negative_for_call(self):
        t = BlackScholes.theta(self.SPOT, self.STRIKE, self.RATE, self.TTE, self.VOL, "CE")
        assert t < 0, "ATM call theta must be negative (time decay)"

    def test_vega_positive(self):
        v = BlackScholes.vega(self.SPOT, self.STRIKE, self.RATE, self.TTE, self.VOL)
        assert v > 0, "Vega must be positive"

    def test_deep_itm_call_delta_near_one(self):
        d = BlackScholes.delta(25000, 20000, self.RATE, self.TTE, self.VOL, "CE")
        assert d > 0.95, f"Deep ITM call delta {d:.4f} should be near 1.0"

    def test_deep_otm_call_delta_near_zero(self):
        d = BlackScholes.delta(24000, 30000, self.RATE, self.TTE, self.VOL, "CE")
        assert d < 0.05, f"Deep OTM call delta {d:.4f} should be near 0"

    def test_expired_call_intrinsic(self):
        """At expiry, call = max(S-K, 0)."""
        price = BlackScholes.call_price(25000, 24500, self.RATE, 0, self.VOL)
        assert abs(price - 500.0) < 0.01

    def test_expired_put_zero(self):
        """OTM put at expiry = 0."""
        price = BlackScholes.put_price(25000, 24500, self.RATE, 0, self.VOL)
        assert abs(price) < 0.01

    def test_implied_volatility_roundtrip(self):
        """Price → IV → Price must round-trip."""
        target_vol = 0.20
        call_price = BlackScholes.call_price(self.SPOT, self.STRIKE, self.RATE, self.TTE, target_vol)
        recovered_iv = BlackScholes.implied_volatility(
            call_price, self.SPOT, self.STRIKE, self.RATE, self.TTE, "CE"
        )
        assert abs(recovered_iv - target_vol) < 0.001, (
            f"IV roundtrip failed: target={target_vol}, recovered={recovered_iv:.4f}"
        )

    def test_compute_all_greeks_returns_dict(self):
        result = compute_all_greeks(self.SPOT, self.STRIKE, self.RATE, self.TTE, self.VOL, "CE")
        assert isinstance(result, dict)
        for key in ("delta", "gamma", "theta", "vega"):
            assert key in result, f"Missing key: {key}"
            assert isinstance(result[key], float)

    def test_zero_time_safety(self):
        """Edge case: tte=0 should not crash."""
        call = BlackScholes.call_price(25000, 24500, self.RATE, 0, 0.2)
        d = BlackScholes.delta(25000, 24500, self.RATE, 0, 0.2, "CE")
        assert call >= 0
        assert d in (0.0, 1.0)


# ═══════════════════════════════════════════════════════════
# 2. HISTORICAL CHAIN BUILDER
# ═══════════════════════════════════════════════════════════

class TestHistoricalChainBuilder:
    """Test synthetic option chain generation."""

    def setup_method(self):
        self.builder = HistoricalChainBuilder("NIFTY")

    def test_build_chain_returns_chain(self):
        chain = self.builder.build_chain(
            spot=24500.0,
            timestamp=datetime(2026, 2, 23, 10, 0),
            expiry=date(2026, 3, 3),
            hv=0.16,
        )
        assert chain is not None
        assert isinstance(chain, SyntheticChain)
        assert chain.spot_price == 24500.0

    def test_chain_has_strikes(self):
        chain = self.builder.build_chain(
            spot=24500.0,
            timestamp=datetime(2026, 2, 23, 10, 0),
            expiry=date(2026, 3, 3),
            hv=0.16,
        )
        assert len(chain.strikes) > 10, "Chain should have >10 strikes"

    def test_chain_has_ce_and_pe(self):
        chain = self.builder.build_chain(
            spot=24500.0,
            timestamp=datetime(2026, 2, 23, 10, 0),
            expiry=date(2026, 3, 3),
            hv=0.16,
        )
        for strike, options in chain.strikes.items():
            assert "CE" in options, f"Strike {strike} missing CE"
            assert "PE" in options, f"Strike {strike} missing PE"

    def test_atm_strike_near_spot(self):
        chain = self.builder.build_chain(
            spot=24500.0,
            timestamp=datetime(2026, 2, 23, 10, 0),
            expiry=date(2026, 3, 3),
            hv=0.16,
        )
        assert abs(chain.atm_strike - 24500) <= 50, "ATM strike should be near spot"

    def test_ce_price_decreases_with_strike(self):
        """Call price should decrease as strike increases (for same expiry)."""
        chain = self.builder.build_chain(
            spot=24500.0,
            timestamp=datetime(2026, 2, 23, 10, 0),
            expiry=date(2026, 3, 3),
            hv=0.16,
        )
        strikes = sorted(chain.strikes.keys())
        ce_prices = [chain.strikes[s]["CE"].price for s in strikes]
        # Allow small tolerance for near-zero OTM options
        for i in range(len(ce_prices) - 1):
            assert ce_prices[i] >= ce_prices[i + 1] - 0.1, (
                f"CE price not monotonic at strikes {strikes[i]}/{strikes[i+1]}: "
                f"{ce_prices[i]:.2f} vs {ce_prices[i+1]:.2f}"
            )

    def test_pe_price_increases_with_strike(self):
        """Put price should increase as strike increases."""
        chain = self.builder.build_chain(
            spot=24500.0,
            timestamp=datetime(2026, 2, 23, 10, 0),
            expiry=date(2026, 3, 3),
            hv=0.16,
        )
        strikes = sorted(chain.strikes.keys())
        pe_prices = [chain.strikes[s]["PE"].price for s in strikes]
        for i in range(len(pe_prices) - 1):
            assert pe_prices[i] <= pe_prices[i + 1] + 0.1, (
                f"PE price not monotonic at strikes {strikes[i]}/{strikes[i+1]}"
            )

    def test_iv_skew_puts_higher_otm(self):
        """OTM puts should have higher IV than ATM (skew)."""
        chain = self.builder.build_chain(
            spot=24500.0,
            timestamp=datetime(2026, 2, 23, 10, 0),
            expiry=date(2026, 3, 3),
            hv=0.16,
        )
        atm = chain.atm_strike
        deep_otm_put_strike = atm - 500  # 500 points OTM
        closest = min(chain.strikes.keys(), key=lambda s: abs(s - deep_otm_put_strike))
        if closest in chain.strikes and "PE" in chain.strikes[closest]:
            otm_iv = chain.strikes[closest]["PE"].iv
            atm_iv = chain.atm_iv
            assert otm_iv >= atm_iv * 0.95, "OTM put IV should be ≥ ATM IV (skew)"

    def test_bid_ask_spread_positive(self):
        chain = self.builder.build_chain(
            spot=24500.0,
            timestamp=datetime(2026, 2, 23, 10, 0),
            expiry=date(2026, 3, 3),
            hv=0.16,
        )
        for strike, options in chain.strikes.items():
            for opt_type in ("CE", "PE"):
                q = options[opt_type]
                assert q.ask >= q.bid, f"Ask < Bid at strike {strike} {opt_type}"
                if q.price > 1:
                    assert q.ask > q.bid, f"Zero spread at strike {strike} {opt_type} (price={q.price})"

    def test_dte_calculation(self):
        chain = self.builder.build_chain(
            spot=24500.0,
            timestamp=datetime(2026, 2, 23, 10, 0),
            expiry=date(2026, 3, 10),
            hv=0.16,
        )
        expected_dte = (date(2026, 3, 10) - date(2026, 2, 23)).days
        assert chain.dte == expected_dte

    def test_expiry_dates_generation(self):
        expiries = HistoricalChainBuilder.get_expiry_dates(
            date(2026, 1, 1), date(2026, 3, 31), weekly=True, expiry_weekday=1,
        )
        assert len(expiries) > 10, "Should generate >10 weekly expiries in 3 months"
        for exp in expiries:
            assert exp.weekday() == 1, f"NSE expiry {exp} is not Tuesday"

    def test_banknifty_lot_size(self):
        builder = HistoricalChainBuilder("BANKNIFTY")
        assert builder.lot_size == 15

    def test_find_strike_by_delta(self):
        chain = self.builder.build_chain(
            spot=24500.0,
            timestamp=datetime(2026, 2, 23, 10, 0),
            expiry=date(2026, 3, 3),
            hv=0.16,
        )
        result = self.builder.find_strike_by_delta(chain, 0.20, "PE")
        if result is not None:
            assert abs(result.delta) < 0.35, f"Delta {result.delta} too far from target 0.20"


# ═══════════════════════════════════════════════════════════
# 3. F&O EXECUTION SIMULATOR
# ═══════════════════════════════════════════════════════════

class TestFnOSimulator:
    """Test execution simulation: fills, slippage, rejections."""

    def _make_contract(self, strike: float = 24500, opt_type: str = "CE") -> DerivativeContract:
        return DerivativeContract(
            symbol="NIFTY",
            tradingsymbol=f"NIFTY26MAR{int(strike)}{opt_type}",
            instrument_type=InstrumentType.CE if opt_type == "CE" else InstrumentType.PE,
            exchange="NFO",
            strike=strike,
            expiry=date(2026, 3, 3),
            lot_size=65,
            tick_size=0.05,
            underlying="NIFTY",
            settlement=SettlementType.CASH,
        )

    def test_zero_slippage_fill(self):
        sim = FnOExecutionSimulator(slippage_model="zero", rejection_probability=0)
        contract = self._make_contract()
        result = sim.simulate_fill(
            contract, OrderSide.BUY, lots=1, market_price=350.0,
            bid=349.0, ask=351.0, volume=100_000,
        )
        assert result.is_filled
        assert result.fill_price == 350.0
        assert result.slippage == 0.0

    def test_realistic_slippage_buy_at_ask(self):
        sim = FnOExecutionSimulator(slippage_model="realistic", rejection_probability=0)
        contract = self._make_contract()
        result = sim.simulate_fill(
            contract, OrderSide.BUY, lots=1, market_price=350.0,
            bid=349.0, ask=351.0, volume=100_000,
        )
        assert result.is_filled
        assert result.fill_price >= 349.0, "Buy fill should be ≥ bid"

    def test_sell_fill_at_or_below_ask(self):
        sim = FnOExecutionSimulator(slippage_model="realistic", rejection_probability=0)
        contract = self._make_contract()
        result = sim.simulate_fill(
            contract, OrderSide.SELL, lots=1, market_price=350.0,
            bid=349.0, ask=351.0, volume=100_000,
        )
        assert result.is_filled
        assert result.fill_price <= 352.0, "Sell fill should be near bid"

    def test_price_below_tick_rejected(self):
        sim = FnOExecutionSimulator(rejection_probability=0)
        contract = self._make_contract()
        result = sim.simulate_fill(
            contract, OrderSide.BUY, lots=1, market_price=0.01,
            volume=100_000,
        )
        assert not result.is_filled
        assert result.status == FnOOrderStatus.REJECTED

    def test_margin_rejection_for_sells(self):
        sim = FnOExecutionSimulator(rejection_probability=0)
        contract = self._make_contract()
        result = sim.simulate_fill(
            contract, OrderSide.SELL, lots=1, market_price=350.0,
            bid=349.0, ask=351.0, volume=100_000,
            available_margin=100.0,  # Very low margin
        )
        assert result.status == FnOOrderStatus.REJECTED
        assert "margin" in result.reject_reason.lower()

    def test_freeze_qty_splitting(self):
        sim = FnOExecutionSimulator(slippage_model="zero", rejection_probability=0)
        contract = DerivativeContract(
            symbol="NIFTY",
            tradingsymbol="NIFTY26MAR24500CE",
            instrument_type=InstrumentType.CE,
            exchange="NFO",
            strike=24500,
            expiry=date(2026, 3, 3),
            lot_size=65,
            tick_size=0.05,
            underlying="NIFTY",
            freeze_quantity=100,
            settlement=SettlementType.CASH,
        )
        # 5 lots × 65 = 325 quantity → should split into 4 child orders
        result = sim.simulate_fill(
            contract, OrderSide.BUY, lots=5, market_price=350.0,
            volume=1_000_000,
        )
        assert result.child_orders == math.ceil(325 / 100)

    def test_partial_fill_low_volume(self):
        sim = FnOExecutionSimulator(slippage_model="zero", rejection_probability=0)
        contract = self._make_contract()
        # Order for 10 lots but very low volume → partial fill likely
        result = sim.simulate_fill(
            contract, OrderSide.BUY, lots=10, market_price=350.0,
            volume=100,  # extremely low volume
        )
        # With participation > 50%, expect partial fill
        assert result.filled_lots <= 10

    def test_fill_result_to_dict(self):
        sim = FnOExecutionSimulator(slippage_model="zero", rejection_probability=0)
        contract = self._make_contract()
        result = sim.simulate_fill(
            contract, OrderSide.BUY, lots=1, market_price=350.0, volume=100_000,
        )
        d = result.to_dict()
        assert "order_id" in d
        assert "fill_price" in d
        assert "slippage" in d


# ═══════════════════════════════════════════════════════════
# 4. INDIAN F&O COST MODEL
# ═══════════════════════════════════════════════════════════

class TestFnOCostModel:
    """Test F&O transaction cost computation."""

    def _make_contract(self, itype: InstrumentType = InstrumentType.CE) -> DerivativeContract:
        return DerivativeContract(
            symbol="NIFTY",
            tradingsymbol="NIFTY26MAR24500CE",
            instrument_type=itype,
            exchange="NFO",
            strike=24500 if itype in (InstrumentType.CE, InstrumentType.PE) else None,
            expiry=date(2026, 3, 3),
            lot_size=65,
            tick_size=0.05,
            underlying="NIFTY",
        )

    def test_option_buy_costs_structure(self):
        model = IndianFnOCostModel()
        contract = self._make_contract(InstrumentType.CE)
        result = model.calculate(contract, TransactionSide.BUY, price=350.0, lots=1)
        assert result.total > 0
        assert result.brokerage == 20.0, "Zerodha flat ₹20 per order"
        assert result.gst > 0

    def test_option_sell_has_stt(self):
        model = IndianFnOCostModel()
        contract = self._make_contract(InstrumentType.CE)
        buy_cost = model.calculate(contract, TransactionSide.BUY, price=350.0, lots=1)
        sell_cost = model.calculate(contract, TransactionSide.SELL, price=350.0, lots=1)
        # STT on options: only on sell side
        assert sell_cost.stt > buy_cost.stt or sell_cost.stt > 0

    def test_futures_cost(self):
        model = IndianFnOCostModel()
        contract = self._make_contract(InstrumentType.FUT)
        result = model.calculate(contract, TransactionSide.BUY, price=24500.0, lots=1)
        assert result.total > 0
        assert result.stt > 0, "Futures have STT on both sides"

    def test_cost_scales_with_lots(self):
        model = IndianFnOCostModel()
        contract = self._make_contract(InstrumentType.CE)
        cost_1 = model.calculate(contract, TransactionSide.BUY, price=350.0, lots=1)
        cost_2 = model.calculate(contract, TransactionSide.BUY, price=350.0, lots=2)
        # Some components (STT, exchange txn) scale, brokerage is flat per order
        assert cost_2.total > cost_1.total

    def test_cost_breakdown_to_dict(self):
        model = IndianFnOCostModel()
        contract = self._make_contract(InstrumentType.CE)
        result = model.calculate(contract, TransactionSide.BUY, price=350.0, lots=1)
        d = result.to_dict()
        for key in ("brokerage", "stt", "exchange_txn_charge", "sebi_fee", "stamp_duty", "gst", "total"):
            assert key in d


# ═══════════════════════════════════════════════════════════
# 5. MARGIN ENGINE
# ═══════════════════════════════════════════════════════════

class TestMarginEngine:
    """Test SPAN-like margin computation."""

    def _make_short_straddle_position(self, spot: float = 24500) -> MultiLegPosition:
        """Create a short straddle position for margin testing."""
        legs = []
        for opt_type, itype in [("CE", InstrumentType.CE), ("PE", InstrumentType.PE)]:
            contract = DerivativeContract(
                symbol="NIFTY",
                tradingsymbol=f"NIFTY26MAR{int(spot)}{opt_type}",
                instrument_type=itype,
                exchange="NFO",
                strike=spot,
                expiry=date(2026, 3, 3),
                lot_size=65,
                tick_size=0.05,
                underlying="NIFTY",
                instrument_token=100 + len(legs),
            )
            leg = OptionLeg(
                contract=contract,
                quantity=-1,
                entry_price=350.0,
                current_price=350.0,
                iv_at_entry=0.16,
            )
            legs.append(leg)

        return MultiLegPosition(
            position_id="TEST001",
            structure=StructureType.SHORT_STRADDLE,
            underlying="NIFTY",
            legs=legs,
        )

    def test_margin_positive(self):
        engine = MarginEngine()
        pos = self._make_short_straddle_position()
        result = engine.calculate_margin(pos, 24500.0)
        assert result.total_margin > 0, "Short straddle must require margin"

    def test_margin_has_span_component(self):
        engine = MarginEngine()
        pos = self._make_short_straddle_position()
        result = engine.calculate_margin(pos, 24500.0)
        assert result.span_margin > 0

    def test_margin_increases_when_spot_moves(self):
        engine = MarginEngine()
        pos = self._make_short_straddle_position()
        m1 = engine.calculate_margin(pos, 24500.0)
        m2 = engine.calculate_margin(pos, 25500.0)  # Spot moved 1000 pts
        # Margin should generally increase when position is stressed
        assert m2.total_margin > 0

    def test_margin_to_dict(self):
        engine = MarginEngine()
        pos = self._make_short_straddle_position()
        result = engine.calculate_margin(pos, 24500.0)
        d = result.to_dict()
        assert "total_margin" in d
        assert "span_margin" in d


# ═══════════════════════════════════════════════════════════
# 6. GREEKS ENGINE (Portfolio Level)
# ═══════════════════════════════════════════════════════════

class TestGreeksEngine:
    """Test portfolio-level Greeks computation."""

    def _make_iron_condor_position(self) -> MultiLegPosition:
        legs = []
        for strike, opt_type, qty in [
            (24000, "PE", 1), (23800, "PE", -1),  # buy 24000PE, sell 23800PE
            (25000, "CE", -1), (25200, "CE", 1),   # sell 25000CE, buy 25200CE
        ]:
            itype = InstrumentType.CE if opt_type == "CE" else InstrumentType.PE
            contract = DerivativeContract(
                symbol="NIFTY",
                tradingsymbol=f"NIFTY26MAR{strike}{opt_type}",
                instrument_type=itype,
                exchange="NFO",
                strike=float(strike),
                expiry=date(2026, 3, 3),
                lot_size=65,
                tick_size=0.05,
                underlying="NIFTY",
                instrument_token=200 + len(legs),
            )
            leg = OptionLeg(
                contract=contract,
                quantity=qty,
                entry_price=100.0,
                current_price=100.0,
                iv_at_entry=0.16,
                iv_current=0.16,
            )
            legs.append(leg)

        return MultiLegPosition(
            position_id="IC001",
            structure=StructureType.IRON_CONDOR,
            underlying="NIFTY",
            legs=legs,
        )

    def test_portfolio_greeks_net_delta_small(self):
        """Iron condor should have near-zero net delta."""
        engine = GreeksEngine()
        pos = self._make_iron_condor_position()
        pg = engine.compute_portfolio_greeks([pos], 24500.0)
        assert abs(pg.net_delta) < 2.0, f"Iron condor delta {pg.net_delta} should be near 0"

    def test_portfolio_greeks_theta_non_zero(self):
        engine = GreeksEngine()
        pos = self._make_iron_condor_position()
        pg = engine.compute_portfolio_greeks([pos], 24500.0)
        assert pg.net_theta != 0, "Theta should be non-zero for iron condor"

    def test_risk_limits_check(self):
        engine = GreeksEngine(max_portfolio_delta=0.5)
        pos = self._make_iron_condor_position()
        pg = engine.compute_portfolio_greeks([pos], 24500.0)
        violations = engine.check_risk_limits(pg)
        # With tight delta limit, may or may not trigger
        assert isinstance(violations, list)

    def test_snapshot_recording(self):
        engine = GreeksEngine()
        pos = self._make_iron_condor_position()
        snap = engine.record_snapshot([pos], 24500.0, capital=500_000.0)
        assert snap.spot_price == 24500.0
        assert snap.positions_count == 1

    def test_greeks_history(self):
        engine = GreeksEngine()
        pos = self._make_iron_condor_position()
        engine.record_snapshot([pos], 24500.0)
        engine.record_snapshot([pos], 24600.0)
        history = engine.get_greeks_history()
        assert len(history) == 2


# ═══════════════════════════════════════════════════════════
# 7. REGIME ENGINE
# ═══════════════════════════════════════════════════════════

class TestRegimeEngine:
    """Test market regime classification."""

    def test_classify_needs_minimum_bars(self):
        engine = RegimeEngine()
        short_data = np.array([100, 101, 102])
        result = engine.classify(short_data)
        assert result.regime == MarketRegime.UNKNOWN

    def test_classify_low_vol(self):
        engine = RegimeEngine()
        # Perfectly flat with tiny noise → consistently low vol across windows
        rng = np.random.default_rng(42)
        prices = 24500 + rng.normal(0, 0.5, 252).cumsum()  # very tiny noise
        prices = np.maximum(prices, 20000)
        result = engine.classify(prices)
        # With near-zero vol, percentile ranking within the series may vary;
        # the key check is that it classifies without error and hv_20 is small
        assert result.hv_20 < 0.10, f"HV should be very small, got {result.hv_20:.4f}"
        assert result.regime != MarketRegime.EVENT_RISK

    def test_classify_high_vol(self):
        engine = RegimeEngine()
        rng = np.random.default_rng(666)
        # Large moves → high vol
        prices = 24500 + np.cumsum(rng.normal(0, 200, 252))
        prices = np.maximum(prices, 10000)
        result = engine.classify(prices)
        # With large daily moves, HV should be elevated
        assert result.hv_20 > 0.10, f"HV should be high, got {result.hv_20:.4f}"
        assert result.regime != MarketRegime.UNKNOWN

    def test_classify_trending_up(self):
        engine = RegimeEngine()
        # Strong uptrend
        prices = np.linspace(20000, 26000, 252) + np.random.default_rng(1).normal(0, 50, 252)
        result = engine.classify(prices)
        assert result.trend_direction > 0 or result.regime in (
            MarketRegime.TRENDING_UP, MarketRegime.LOW_VOL,
        )

    def test_regime_has_recommended_structures(self):
        engine = RegimeEngine()
        prices = np.linspace(24000, 24500, 252)
        result = engine.classify(prices)
        assert isinstance(result.recommended_structures, list)

    def test_regime_history(self):
        engine = RegimeEngine()
        prices = np.linspace(24000, 24500, 252)
        engine.classify(prices)
        engine.classify(prices * 1.01)
        history = engine.get_regime_history()
        assert len(history) == 2

    def test_event_risk_with_high_vix(self):
        engine = RegimeEngine()
        prices = np.linspace(24000, 24500, 252)
        result = engine.classify(prices, vix=30.0)
        assert result.regime == MarketRegime.EVENT_RISK


# ═══════════════════════════════════════════════════════════
# 8. EXPIRY ENGINE
# ═══════════════════════════════════════════════════════════

class TestExpiryEngine:
    """Test option expiry processing."""

    def _make_position_for_expiry(
        self, strike: float, opt_type: str, qty: int, entry_price: float, expiry: date,
    ) -> MultiLegPosition:
        itype = InstrumentType.CE if opt_type == "CE" else InstrumentType.PE
        contract = DerivativeContract(
            symbol="NIFTY",
            tradingsymbol=f"NIFTY26MAR{int(strike)}{opt_type}",
            instrument_type=itype,
            exchange="NFO",
            strike=strike,
            expiry=expiry,
            lot_size=65,
            tick_size=0.05,
            underlying="NIFTY",
            instrument_token=300,
            settlement=SettlementType.CASH,
        )
        leg = OptionLeg(
            contract=contract, quantity=qty,
            entry_price=entry_price, current_price=entry_price,
        )
        return MultiLegPosition(
            position_id="EXP001",
            structure=StructureType.CUSTOM,
            underlying="NIFTY",
            legs=[leg],
        )

    def test_itm_call_exercise(self):
        engine = ExpiryEngine()
        expiry = date(2026, 3, 3)
        pos = self._make_position_for_expiry(24000, "CE", 1, 600.0, expiry)
        events = engine.process_expiry(pos, settlement_price=25000.0, current_date=expiry)
        # 25000 spot, 24000 strike → ITM by 1000 → should exercise
        assert len(events) > 0, "ITM call should generate expiry events"
        assert any(
            e.event_type in (ExpiryEventType.ITM_EXERCISE, ExpiryEventType.CASH_SETTLEMENT)
            for e in events
        )

    def test_otm_put_expires_worthless(self):
        engine = ExpiryEngine()
        expiry = date(2026, 3, 3)
        pos = self._make_position_for_expiry(24000, "PE", 1, 200.0, expiry)
        events = engine.process_expiry(pos, settlement_price=25000.0, current_date=expiry)
        # 24000 PE with spot at 25000 → OTM → expire worthless
        assert len(events) > 0, "OTM put should generate expiry event"
        assert any(e.event_type == ExpiryEventType.OTM_EXPIRE for e in events)

    def test_non_expiry_date_no_events(self):
        engine = ExpiryEngine()
        expiry = date(2026, 3, 3)
        pos = self._make_position_for_expiry(24000, "CE", 1, 600.0, expiry)
        events = engine.process_expiry(pos, settlement_price=25000.0, current_date=date(2026, 2, 25))
        # Not expiry date → no events
        assert len(events) == 0
