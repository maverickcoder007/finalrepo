"""
F&O Backtesting Engine — Event-driven multi-leg derivatives backtester.

Event flow:
  MarketEvent → OptionChainUpdate → StrategySignal → RiskSizing →
  OrderSubmission → FillEvent → MarginRecalc → GreeksUpdate → ExpiryEvent

Unlike the equity BacktestEngine, this engine:
  • Builds synthetic option chains per bar (via HistoricalChainBuilder)
  • Supports multi-leg structures (spreads, condors, straddles)
  • Tracks portfolio Greeks over time
  • Runs SPAN-like margin calculations per bar
  • Handles option expiry (ITM exercise, OTM expire, assignment)
  • Uses F&O-specific cost model (different STT, exchange charges)
  • Records Greeks journal for analysis
"""
from __future__ import annotations

import math
from datetime import datetime, date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.derivatives.chain_builder import HistoricalChainBuilder, SyntheticChain
from src.derivatives.contracts import (
    DerivativeContract,
    InstrumentType,
    MultiLegPosition,
    OptionIntent,
    OptionLeg,
    StructureType,
)
from src.derivatives.expiry_engine import ExpiryEngine
from src.derivatives.fno_cost_model import IndianFnOCostModel, TransactionSide
from src.derivatives.fno_simulator import FnOExecutionSimulator, OrderSide
from src.derivatives.greeks_engine import GreeksEngine
from src.derivatives.margin_engine import MarginEngine
from src.derivatives.regime_engine import RegimeEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FnOBacktestEngine:
    """Event-driven F&O backtesting engine.

    Supports:
      • Pre-built strategies: iron_condor, bull_call_spread, bear_put_spread,
        straddle, strangle, covered_call, protective_put, custom
      • Synthetic option chain generation from underlying OHLCV
      • Multi-leg entry and exit with realistic costs
      • Portfolio Greeks tracking (delta, gamma, theta, vega)
      • SPAN margin computation per bar
      • Expiry handling (exercise / expire worthless / assignment)
      • Regime-aware strategy selection
      • Full journal with Greeks snapshots
    """

    STRATEGY_MAP: dict[str, StructureType] = {
        "iron_condor": StructureType.IRON_CONDOR,
        "bull_call_spread": StructureType.BULL_CALL_SPREAD,
        "bear_put_spread": StructureType.BEAR_PUT_SPREAD,
        "bull_put_spread": StructureType.BULL_PUT_SPREAD,
        "bear_call_spread": StructureType.BEAR_CALL_SPREAD,
        "straddle": StructureType.LONG_STRADDLE,
        "short_straddle": StructureType.SHORT_STRADDLE,
        "strangle": StructureType.LONG_STRANGLE,
        "short_strangle": StructureType.SHORT_STRANGLE,
        "covered_call": StructureType.COVERED_CALL,
        "protective_put": StructureType.PROTECTIVE_PUT,
        "iron_butterfly": StructureType.IRON_BUTTERFLY,
        "calendar_spread": StructureType.CALENDAR_SPREAD,
    }

    def __init__(
        self,
        strategy_name: str = "iron_condor",
        underlying: str = "NIFTY",
        initial_capital: float = 500_000.0,
        max_positions: int = 3,
        profit_target_pct: float = 50.0,   # % of max profit to exit
        stop_loss_pct: float = 100.0,      # % of max loss to exit
        entry_dte_min: int = 15,
        entry_dte_max: int = 45,
        delta_target: float = 0.16,        # for strike selection
        slippage_model: str = "realistic",
        use_regime_filter: bool = True,
    ) -> None:
        self.strategy_name = strategy_name
        self.underlying = underlying.upper()
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        self.entry_dte_min = entry_dte_min
        self.entry_dte_max = entry_dte_max
        self.delta_target = delta_target
        self.slippage_model = slippage_model
        self.use_regime_filter = use_regime_filter

        self.structure_type = self.STRATEGY_MAP.get(
            strategy_name.lower(), StructureType.IRON_CONDOR
        )

        # Sub-engines
        self._chain_builder = HistoricalChainBuilder(self.underlying)
        self._simulator = FnOExecutionSimulator(slippage_model=slippage_model)
        self._cost_model = IndianFnOCostModel()
        self._margin_engine = MarginEngine()
        self._greeks_engine = GreeksEngine()
        self._expiry_engine = ExpiryEngine()
        self._regime_engine = RegimeEngine()

    def run(
        self,
        data: pd.DataFrame,
        tradingsymbol: str = "NIFTY",
    ) -> dict[str, Any]:
        """Run F&O backtest on underlying OHLCV data.

        Args:
            data: DataFrame with columns [open, high, low, close, volume]
                  representing the underlying instrument's price history
            tradingsymbol: underlying symbol name

        Returns:
            Comprehensive backtest result dict
        """
        if data.empty:
            return {"error": "Empty dataset"}

        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(data.columns)
        if missing:
            return {"error": f"Missing columns: {missing}"}

        df = data.copy()
        # Preserve DatetimeIndex as 'timestamp' column if not already present
        if "timestamp" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df["timestamp"] = df.index
        df = df.reset_index(drop=True)
        closes = df["close"].values

        _has_ts = "timestamp" in df.columns
        def _bar_ts(idx: int) -> str:
            return str(df["timestamp"].iloc[idx]) if _has_ts else str(idx)

        # State tracking
        capital = self.initial_capital
        positions: list[MultiLegPosition] = []
        closed_positions: list[MultiLegPosition] = []
        equity_curve: list[float] = [capital]
        bar_dates: list[str] = [_bar_ts(0)]
        greeks_history: list[dict] = []
        margin_history: list[dict] = []
        regime_history: list[dict] = []
        trades: list[dict] = []
        total_costs = 0.0
        position_counter = 0

        # Pre-compute expiry dates
        start_date = self._parse_bar_date(df, 0)
        end_date = self._parse_bar_date(df, len(df) - 1)
        expiries = self._chain_builder.get_expiry_dates(
            start_date, end_date + timedelta(days=60),
            expiry_weekday=self._chain_builder.expiry_day,
        )

        logger.info(
            "fno_backtest_start",
            strategy=self.strategy_name,
            underlying=self.underlying,
            bars=len(df),
            capital=capital,
        )

        for i in range(max(5, min(21, len(df) // 10)), len(df)):
            bar = df.iloc[i]
            spot = float(bar["close"])
            bar_date = self._parse_bar_date(df, i)
            bar_dates.append(_bar_ts(i))

            # ── 1. Regime classification ───
            price_history = closes[:i + 1]
            regime = self._regime_engine.classify(price_history, timestamp=datetime.now())

            # ── 2. Build option chain ───
            # Pick expiry with enough DTE for new entries; fall back to nearest for MtM
            target_exp = None
            for exp in sorted(expiries):
                if (exp - bar_date).days >= self.entry_dte_min:
                    target_exp = exp
                    break
            if target_exp is None:
                target_exp = self._chain_builder.nearest_expiry(bar_date, expiries)
            if target_exp is None:
                equity_curve.append(equity_curve[-1])
                continue
            chain = self._chain_builder.build_chain(
                spot=spot,
                timestamp=datetime.combine(bar_date, datetime.min.time()),
                expiry=target_exp,
                hv=regime.hv_20 if regime.hv_20 > 0 else 0.20,
            )

            if chain is None:
                equity_curve.append(equity_curve[-1])
                continue

            # ── 3. Check expiry for open positions ───
            for pos in positions:
                if pos.is_closed:
                    continue
                events = self._expiry_engine.process_expiry(
                    pos, spot, bar_date,
                    timestamp=datetime.combine(bar_date, datetime.min.time()),
                )
                for evt in events:
                    trades.append({
                        "type": "EXPIRY",
                        "structure": pos.structure.value,
                        "exit_reason": "expiry",
                        "event": evt.event_type.value,
                        "bar": i,
                        "date": str(bar_date),
                        "entry_time": str(pos.entry_time) if pos.entry_time else str(bar_date),
                        "exit_time": str(bar_date),
                        "legs": len(pos.legs),
                        "net_premium": round(pos.net_premium * max((l.contract.lot_size for l in pos.legs), default=1), 2),
                        "qty": sum(abs(l.quantity) for l in pos.legs),
                        "spot": round(spot, 2),
                        "settlement_price": round(evt.settlement_price, 2),
                        "intrinsic": round(evt.intrinsic_value, 2),
                        "pnl": round(evt.pnl, 2),
                        "position_id": pos.position_id,
                    })
                    capital += evt.pnl
                # If expiry closed all legs, set position-level exit metadata
                if pos.is_closed and pos.exit_time is None:
                    pos.exit_time = datetime.combine(bar_date, datetime.min.time())
                    pos.exit_reason = "expiry"

            # ── 4. Update Greeks for open positions ───
            open_positions = [p for p in positions if not p.is_closed]
            if open_positions:
                pg = self._greeks_engine.compute_portfolio_greeks(
                    open_positions, spot, capital,
                )
                greeks_history.append({
                    "bar": i, "date": str(bar_date), **pg.to_dict()
                })

            # ── 5. Check profit/loss targets on open positions ───
            for pos in open_positions:
                if pos.is_closed:
                    continue
                current_pnl = sum(
                    l.unrealised_pnl * l.contract.lot_size
                    for l in pos.legs if not l.is_closed
                )
                # Robust max_profit/max_loss computation (Issue #8)
                # Use position's tracked values; fall back to net_premium × lot_size
                # with a minimum floor to prevent erratic triggers.
                _lot_size = pos.legs[0].contract.lot_size if pos.legs else 50
                _premium_based = abs(pos.net_premium) * _lot_size
                max_profit = pos.max_profit if (pos.max_profit and pos.max_profit > 0) else max(_premium_based, 500.0)
                import math
                max_loss = pos.max_loss if (pos.max_loss and math.isfinite(pos.max_loss) and pos.max_loss > 0) else max(max_profit * 2, 1000.0)

                # Profit target
                if max_profit > 0 and current_pnl >= max_profit * (self.profit_target_pct / 100):
                    exit_pnl, exit_costs = self._close_position(pos, chain, spot, i, bar_date)
                    capital += exit_pnl - exit_costs
                    total_costs += exit_costs
                    pos.exit_reason = "profit_target"
                    trades.append({
                        "type": "PROFIT_TARGET",
                        "structure": pos.structure.value,
                        "exit_reason": "profit_target",
                        "bar": i, "date": str(bar_date),
                        "entry_time": str(pos.entry_time) if pos.entry_time else str(bar_date),
                        "exit_time": str(pos.exit_time) if pos.exit_time else str(bar_date),
                        "legs": len(pos.legs),
                        "net_premium": round(pos.net_premium * max((l.contract.lot_size for l in pos.legs), default=1), 2),
                        "qty": sum(abs(l.quantity) for l in pos.legs),
                        "pnl": round(exit_pnl, 2),
                        "costs": round(exit_costs, 2),
                        "position_id": pos.position_id,
                    })

                # Stop loss
                elif max_loss > 0 and current_pnl <= -max_loss * (self.stop_loss_pct / 100):
                    exit_pnl, exit_costs = self._close_position(pos, chain, spot, i, bar_date)
                    capital += exit_pnl - exit_costs
                    total_costs += exit_costs
                    pos.exit_reason = "stop_loss"
                    trades.append({
                        "type": "STOP_LOSS",
                        "structure": pos.structure.value,
                        "exit_reason": "stop_loss",
                        "bar": i, "date": str(bar_date),
                        "entry_time": str(pos.entry_time) if pos.entry_time else str(bar_date),
                        "exit_time": str(pos.exit_time) if pos.exit_time else str(bar_date),
                        "legs": len(pos.legs),
                        "net_premium": round(pos.net_premium * max((l.contract.lot_size for l in pos.legs), default=1), 2),
                        "qty": sum(abs(l.quantity) for l in pos.legs),
                        "pnl": round(exit_pnl, 2),
                        "costs": round(exit_costs, 2),
                        "position_id": pos.position_id,
                    })

            # Move closed positions
            newly_closed = [p for p in positions if p.is_closed and p not in closed_positions]
            closed_positions.extend(newly_closed)

            # ── 6. Entry logic — open new position if conditions met ───
            open_count = len([p for p in positions if not p.is_closed])
            if open_count < self.max_positions and chain.dte >= self.entry_dte_min:
                # Regime filter
                should_enter = True
                if self.use_regime_filter:
                    should_enter = self._regime_allows_entry(regime, self.structure_type)

                if should_enter:
                    new_pos, entry_costs = self._open_position(
                        chain, spot, capital, i, bar_date, position_counter
                    )
                    if new_pos is not None:
                        positions.append(new_pos)
                        position_counter += 1
                        capital -= entry_costs
                        total_costs += entry_costs

                        # Margin check
                        margin_result = self._margin_engine.calculate_margin(
                            new_pos, spot
                        )
                        new_pos.margin_required = margin_result.total_margin
                        margin_history.append({
                            "bar": i, "date": str(bar_date),
                            "margin": round(margin_result.total_margin, 2),
                            "margin_used": round(margin_result.total_margin, 2),
                            "total_margin": round(margin_result.total_margin, 2),
                            "span": round(margin_result.span_margin, 2),
                        })

                        trades.append({
                            "type": "ENTRY",
                            "structure": self.structure_type.value,
                            "bar": i, "date": str(bar_date),
                            "spot": round(spot, 2),
                            "legs": len(new_pos.legs),
                            "net_premium": round(new_pos.net_premium * max((l.contract.lot_size for l in new_pos.legs), default=1), 2),
                            "margin": round(margin_result.total_margin, 2),
                            "costs": round(entry_costs, 2),
                            "position_id": new_pos.position_id,
                        })

            # ── 7. Mark to market ───
            mtm = capital
            for pos in positions:
                if pos.is_closed:
                    continue
                for leg in pos.legs:
                    if leg.is_closed:
                        continue
                    # Update current price from chain
                    self._update_leg_from_chain(leg, chain, spot)
                    mtm += leg.unrealised_pnl * leg.contract.lot_size

            equity_curve.append(mtm)

        # ── Force close remaining positions ───
        final_spot = float(df.iloc[-1]["close"])
        final_date = self._parse_bar_date(df, len(df) - 1)
        for pos in positions:
            if pos.is_closed:
                continue
            exit_pnl, exit_costs = self._close_position(
                pos, None, final_spot, len(df) - 1, final_date
            )
            capital += exit_pnl - exit_costs
            total_costs += exit_costs
            closed_positions.append(pos)
            pos.exit_reason = "final_exit"
            trades.append({
                "type": "FINAL_EXIT",
                "structure": pos.structure.value,
                "exit_reason": "final_exit",
                "bar": len(df) - 1,
                "date": str(final_date),
                "entry_time": str(pos.entry_time) if pos.entry_time else str(final_date),
                "exit_time": str(final_date),
                "legs": len(pos.legs),
                "net_premium": round(pos.net_premium * max((l.contract.lot_size for l in pos.legs), default=1), 2),
                "qty": sum(abs(l.quantity) for l in pos.legs),
                "pnl": round(exit_pnl, 2),
                "costs": round(exit_costs, 2),
                "position_id": pos.position_id,
            })

        # Append final capital so equity curve matches final_capital exactly
        equity_curve.append(capital)

        # ── Build results ───
        equity = np.array(equity_curve)
        metrics = self._compute_metrics(equity, closed_positions, trades)

        return self._sanitize({
            "engine": "fno_backtest",
            "strategy_name": self.strategy_name,
            "structure": self.structure_type.value,
            "underlying": self.underlying,
            "tradingsymbol": tradingsymbol,
            "total_bars": len(df),
            "start_date": str(self._parse_bar_date(df, 0)),
            "end_date": str(self._parse_bar_date(df, len(df) - 1)),
            "initial_capital": self.initial_capital,
            "final_capital": round(capital, 2),
            "total_return_pct": round(
                (capital - self.initial_capital) / self.initial_capital * 100, 2
            ),
            "total_return_abs": round(capital - self.initial_capital, 2),
            "total_costs": round(total_costs, 2),
            **metrics,
            "positions_opened": position_counter,
            "positions_closed": len(closed_positions),
            "trades": trades[-500:],
            "equity_curve": [round(e, 2) for e in equity_curve],
            "bar_dates": bar_dates[:len(equity_curve)],
            "greeks_history": greeks_history[-200:],
            "margin_history": margin_history[-200:],
            "regime_history": self._regime_engine.get_regime_history()[-200:],
            "settings": {
                "max_positions": self.max_positions,
                "profit_target_pct": self.profit_target_pct,
                "stop_loss_pct": self.stop_loss_pct,
                "entry_dte_min": self.entry_dte_min,
                "entry_dte_max": self.entry_dte_max,
                "delta_target": self.delta_target,
                "slippage_model": self.slippage_model,
                "use_regime_filter": self.use_regime_filter,
            },
        })

    # ──────────────────────────────────────────────
    # Position Open / Close
    # ──────────────────────────────────────────────

    def _open_position(
        self,
        chain: SyntheticChain,
        spot: float,
        capital: float,
        bar_idx: int,
        bar_date: date,
        counter: int,
    ) -> tuple[MultiLegPosition | None, float]:
        """Open a new multi-leg position based on strategy type."""
        legs: list[OptionLeg] = []
        total_cost = 0.0

        structure = self.structure_type

        if structure == StructureType.IRON_CONDOR:
            legs, total_cost = self._build_iron_condor(chain, spot, capital)
        elif structure in (StructureType.BULL_CALL_SPREAD, StructureType.BEAR_CALL_SPREAD):
            legs, total_cost = self._build_vertical_spread(chain, spot, capital, "CALL", structure)
        elif structure in (StructureType.BULL_PUT_SPREAD, StructureType.BEAR_PUT_SPREAD):
            legs, total_cost = self._build_vertical_spread(chain, spot, capital, "PUT", structure)
        elif structure in (StructureType.LONG_STRADDLE, StructureType.SHORT_STRADDLE):
            legs, total_cost = self._build_straddle(chain, spot, capital, structure)
        elif structure in (StructureType.LONG_STRANGLE, StructureType.SHORT_STRANGLE):
            legs, total_cost = self._build_strangle(chain, spot, capital, structure)
        elif structure == StructureType.IRON_BUTTERFLY:
            legs, total_cost = self._build_iron_butterfly(chain, spot, capital)
        else:
            # Default to iron condor
            legs, total_cost = self._build_iron_condor(chain, spot, capital)

        if not legs:
            return None, 0.0

        regime = self._regime_engine.get_current_regime()
        pos = MultiLegPosition(
            position_id=f"FNO_{counter:04d}",
            structure=structure,
            underlying=self.underlying,
            legs=legs,
            entry_time=datetime.combine(bar_date, datetime.min.time()),
            iv_percentile_at_entry=regime.hv_percentile if regime else 50,
            regime_at_entry=regime.regime.value if regime else "UNKNOWN",
        )
        return pos, total_cost

    def _build_iron_condor(
        self, chain: SyntheticChain, spot: float, capital: float,
    ) -> tuple[list[OptionLeg], float]:
        """Build iron condor: sell OTM put + OTM call, buy further OTM wings."""
        legs = []
        total_cost = 0.0

        # Find strikes by delta
        sell_put = self._chain_builder.find_strike_by_delta(chain, self.delta_target, "PE")
        sell_call = self._chain_builder.find_strike_by_delta(chain, self.delta_target, "CE")

        if sell_put is None or sell_call is None:
            return [], 0.0

        # Wings: 2 strikes further OTM
        wing_distance = (sell_call.strike - sell_put.strike) * 0.3
        buy_put_strike = sell_put.strike - wing_distance
        buy_call_strike = sell_call.strike + wing_distance

        # Find closest strikes in chain
        buy_put = self._find_closest_strike(chain, buy_put_strike, "PE")
        buy_call = self._find_closest_strike(chain, buy_call_strike, "CE")

        if buy_put is None or buy_call is None:
            return [], 0.0

        # Build legs with fills
        for quote, side_sign in [
            (sell_put, -1), (buy_put, 1),
            (sell_call, -1), (buy_call, 1),
        ]:
            contract = self._chain_builder.make_contract(quote, chain)
            fill = self._simulator.simulate_fill(
                contract,
                OrderSide.SELL if side_sign < 0 else OrderSide.BUY,
                lots=1,
                market_price=quote.price,
                bid=quote.bid,
                ask=quote.ask,
                volume=quote.volume,
                available_margin=capital,
            )
            if not fill.is_filled:
                return [], 0.0

            leg = self._simulator.build_option_leg(fill, iv=quote.iv)
            if leg:
                legs.append(leg)
                cost = self._cost_model.calculate(
                    contract,
                    TransactionSide.SELL if side_sign < 0 else TransactionSide.BUY,
                    fill.fill_price, 1,
                    child_orders=fill.child_orders,
                )
                total_cost += cost.total

        return legs, total_cost

    def _build_vertical_spread(
        self, chain: SyntheticChain, spot: float, capital: float,
        option_type: str, structure: StructureType,
    ) -> tuple[list[OptionLeg], float]:
        """Build bull/bear vertical spread."""
        legs = []
        total_cost = 0.0

        is_bull = structure in (StructureType.BULL_CALL_SPREAD, StructureType.BULL_PUT_SPREAD)
        opt = "CE" if option_type == "CALL" else "PE"

        # ATM and OTM strikes
        atm = self._find_closest_strike(chain, spot, opt)
        otm_distance = spot * 0.03  # 3% OTM
        otm_strike = (spot + otm_distance) if opt == "CE" else (spot - otm_distance)
        otm = self._find_closest_strike(chain, otm_strike, opt)

        if atm is None or otm is None:
            return [], 0.0

        if is_bull:
            # Bull: buy lower, sell higher (debit)
            buy_quote = atm if opt == "CE" else otm
            sell_quote = otm if opt == "CE" else atm
        else:
            # Bear: sell lower, buy higher (credit)
            sell_quote = atm if opt == "CE" else otm
            buy_quote = otm if opt == "CE" else atm

        for quote, side_sign in [(buy_quote, 1), (sell_quote, -1)]:
            contract = self._chain_builder.make_contract(quote, chain)
            fill = self._simulator.simulate_fill(
                contract,
                OrderSide.BUY if side_sign > 0 else OrderSide.SELL,
                lots=1,
                market_price=quote.price,
                bid=quote.bid,
                ask=quote.ask,
                volume=quote.volume,
                available_margin=capital,
            )
            if not fill.is_filled:
                return [], 0.0
            leg = self._simulator.build_option_leg(fill, iv=quote.iv)
            if leg:
                legs.append(leg)
                cost = self._cost_model.calculate(
                    contract,
                    TransactionSide.BUY if side_sign > 0 else TransactionSide.SELL,
                    fill.fill_price, 1,
                )
                total_cost += cost.total

        return legs, total_cost

    def _build_straddle(
        self, chain: SyntheticChain, spot: float, capital: float,
        structure: StructureType,
    ) -> tuple[list[OptionLeg], float]:
        """Build long or short straddle (ATM call + ATM put)."""
        legs = []
        total_cost = 0.0
        is_short = structure == StructureType.SHORT_STRADDLE

        for opt in ["CE", "PE"]:
            quote = self._find_closest_strike(chain, spot, opt)
            if quote is None:
                return [], 0.0
            contract = self._chain_builder.make_contract(quote, chain)
            side = OrderSide.SELL if is_short else OrderSide.BUY
            fill = self._simulator.simulate_fill(
                contract, side, lots=1,
                market_price=quote.price, bid=quote.bid, ask=quote.ask,
                volume=quote.volume, available_margin=capital,
            )
            if not fill.is_filled:
                return [], 0.0
            leg = self._simulator.build_option_leg(fill, iv=quote.iv)
            if leg:
                legs.append(leg)
                txn_side = TransactionSide.SELL if is_short else TransactionSide.BUY
                cost = self._cost_model.calculate(contract, txn_side, fill.fill_price, 1)
                total_cost += cost.total

        return legs, total_cost

    def _build_strangle(
        self, chain: SyntheticChain, spot: float, capital: float,
        structure: StructureType,
    ) -> tuple[list[OptionLeg], float]:
        """Build long or short strangle (OTM call + OTM put)."""
        legs = []
        total_cost = 0.0
        is_short = structure == StructureType.SHORT_STRANGLE

        otm_call = self._chain_builder.find_strike_by_delta(chain, self.delta_target, "CE")
        otm_put = self._chain_builder.find_strike_by_delta(chain, self.delta_target, "PE")

        if otm_call is None or otm_put is None:
            return [], 0.0

        for quote in [otm_put, otm_call]:
            contract = self._chain_builder.make_contract(quote, chain)
            side = OrderSide.SELL if is_short else OrderSide.BUY
            fill = self._simulator.simulate_fill(
                contract, side, lots=1,
                market_price=quote.price, bid=quote.bid, ask=quote.ask,
                volume=quote.volume, available_margin=capital,
            )
            if not fill.is_filled:
                return [], 0.0
            leg = self._simulator.build_option_leg(fill, iv=quote.iv)
            if leg:
                legs.append(leg)
                txn_side = TransactionSide.SELL if is_short else TransactionSide.BUY
                cost = self._cost_model.calculate(contract, txn_side, fill.fill_price, 1)
                total_cost += cost.total

        return legs, total_cost

    def _build_iron_butterfly(
        self, chain: SyntheticChain, spot: float, capital: float,
    ) -> tuple[list[OptionLeg], float]:
        """Build iron butterfly: sell ATM straddle + buy OTM wings."""
        legs = []
        total_cost = 0.0

        # Sell ATM
        atm_call = self._find_closest_strike(chain, spot, "CE")
        atm_put = self._find_closest_strike(chain, spot, "PE")

        # Buy wings
        wing_dist = spot * 0.05
        buy_call = self._find_closest_strike(chain, spot + wing_dist, "CE")
        buy_put = self._find_closest_strike(chain, spot - wing_dist, "PE")

        if not all([atm_call, atm_put, buy_call, buy_put]):
            return [], 0.0

        for quote, side_sign in [
            (atm_put, -1), (buy_put, 1),
            (atm_call, -1), (buy_call, 1),
        ]:
            contract = self._chain_builder.make_contract(quote, chain)
            fill = self._simulator.simulate_fill(
                contract,
                OrderSide.SELL if side_sign < 0 else OrderSide.BUY,
                lots=1,
                market_price=quote.price, bid=quote.bid, ask=quote.ask,
                volume=quote.volume, available_margin=capital,
            )
            if not fill.is_filled:
                return [], 0.0
            leg = self._simulator.build_option_leg(fill, iv=quote.iv)
            if leg:
                legs.append(leg)
                txn_side = TransactionSide.SELL if side_sign < 0 else TransactionSide.BUY
                cost = self._cost_model.calculate(contract, txn_side, fill.fill_price, 1)
                total_cost += cost.total

        return legs, total_cost

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    def _close_position(
        self,
        pos: MultiLegPosition,
        chain: SyntheticChain | None,
        spot: float,
        bar_idx: int,
        bar_date: date,
    ) -> tuple[float, float]:
        """Close all legs of a position, return (pnl, costs)."""
        total_pnl = 0.0
        total_cost = 0.0

        for leg in pos.legs:
            if leg.is_closed:
                continue
            # Get current price
            exit_price = leg.current_price
            if exit_price <= 0:
                exit_price = max(0.05, leg.entry_price * 0.1)

            # Calculate P&L
            if leg.quantity > 0:
                pnl = (exit_price - leg.entry_price) * abs(leg.quantity) * leg.contract.lot_size
            else:
                pnl = (leg.entry_price - exit_price) * abs(leg.quantity) * leg.contract.lot_size

            total_pnl += pnl

            # Transaction cost
            side = TransactionSide.SELL if leg.quantity > 0 else TransactionSide.BUY
            cost = self._cost_model.calculate(
                leg.contract, side, exit_price, abs(leg.quantity),
            )
            total_cost += cost.total

            leg.close(exit_price)

        pos.exit_time = datetime.combine(bar_date, datetime.min.time())
        return total_pnl, total_cost

    def _update_leg_from_chain(
        self, leg: OptionLeg, chain: SyntheticChain, spot: float,
    ) -> None:
        """Update leg's current price and IV from chain."""
        contract = leg.contract
        if contract.strike is None:
            return

        opt_type = "CE" if contract.is_call else "PE"
        strike_data = chain.strikes.get(contract.strike)

        if strike_data is not None and opt_type in strike_data:
            quote = strike_data[opt_type]
            leg.current_price = quote.price
            leg.iv_current = quote.iv
        else:
            # Fallback: compute from Black-Scholes
            from src.options.greeks import BlackScholes
            tte = max(contract.tte, 1 / 365)
            iv = leg.iv_current if leg.iv_current > 0 else 0.20
            if contract.is_call:
                leg.current_price = max(0.05, BlackScholes.call_price(
                    spot, contract.strike, 0.065, tte, iv
                ))
            else:
                leg.current_price = max(0.05, BlackScholes.put_price(
                    spot, contract.strike, 0.065, tte, iv
                ))

    def _find_closest_strike(
        self, chain: SyntheticChain, target_strike: float, opt_type: str,
    ):
        """Find closest quote in chain to target strike."""
        best = None
        best_dist = float("inf")
        for strike, strike_data in chain.strikes.items():
            quote = strike_data.get(opt_type)
            if quote is None:
                continue
            dist = abs(strike - target_strike)
            if dist < best_dist:
                best_dist = dist
                best = quote
        return best

    def _regime_allows_entry(self, regime, structure: StructureType) -> bool:
        """Check if current regime recommends the strategy structure."""
        if regime is None:
            return True
        recommended = regime.recommended_structures
        if not recommended:
            return True
        struct_name = structure.value
        return struct_name in recommended or len(recommended) == 0

    def _parse_bar_date(self, df: pd.DataFrame, idx: int) -> date:
        """Parse date from DataFrame 'timestamp' column or index."""
        # Prefer explicit 'timestamp' column (preserved from DatetimeIndex)
        if "timestamp" in df.columns:
            val = df["timestamp"].iloc[idx]
        else:
            val = df.index[idx]
        if isinstance(val, (datetime, pd.Timestamp)):
            return val.date()
        try:
            return datetime.fromisoformat(str(val)).date()
        except (ValueError, TypeError):
            return date.today()

    def _compute_metrics(
        self,
        equity: np.ndarray,
        closed_positions: list[MultiLegPosition],
        trades: list[dict],
    ) -> dict[str, Any]:
        """Compute performance metrics."""
        peak = np.maximum.accumulate(equity)
        drawdown = np.where(peak > 0, (peak - equity) / peak, 0)
        max_dd_pct = float(np.max(drawdown)) * 100 if len(drawdown) > 0 else 0

        returns = np.diff(equity) / np.where(equity[:-1] != 0, equity[:-1], 1)
        sharpe = 0.0
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))

        # Trade-level stats
        entry_trades = [t for t in trades if t.get("type") == "ENTRY"]
        exit_trades = [t for t in trades if t.get("type") in ("PROFIT_TARGET", "STOP_LOSS", "FINAL_EXIT", "EXPIRY")]
        pnls = [t.get("pnl", 0) for t in exit_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_trades = len(pnls)
        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        total_wins = sum(wins)
        total_losses = abs(sum(losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Sortino ratio
        neg_returns = returns[returns < 0]
        sortino = 0.0
        if len(neg_returns) > 0 and np.std(neg_returns) > 0:
            sortino = float(np.mean(returns) / np.std(neg_returns) * np.sqrt(252))

        total_pnl = round(total_wins - total_losses, 2)

        return {
            "total_trades": total_trades,
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(win_rate, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "total_pnl": total_pnl,
            "max_drawdown_pct": round(max_dd_pct, 2),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
        }

    @staticmethod
    def _sanitize(obj: Any) -> Any:
        """Convert numpy types to native Python for JSON."""
        if isinstance(obj, dict):
            return {k: FnOBacktestEngine._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [FnOBacktestEngine._sanitize(v) for v in obj]
        if isinstance(obj, (np.bool_, np.generic)):
            val = obj.item()
            if isinstance(val, float) and val != val:
                return None
            return val
        if isinstance(obj, float) and obj != obj:
            return None
        return obj
