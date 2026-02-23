"""
Derivatives Layer — State-of-the-art F&O infrastructure.

Provides:
• DerivativeContract model with full contract lifecycle
• SPAN-like Margin Engine with portfolio-level risk
• Greeks Engine for portfolio delta/gamma/theta/vega tracking
• Option Chain reconstruction for historical backtesting
• F&O Execution Simulator (realistic fills, liquidity, freeze qty)
• Expiry & Assignment Engine
• Market Regime Engine
• F&O-aware Cost Model (options/futures STT, exchange charges)
"""

from src.derivatives.contracts import (
    DerivativeContract,
    OptionLeg,
    OptionIntent,
    MultiLegPosition,
    InstrumentType,
    SettlementType,
)
from src.derivatives.margin_engine import MarginEngine, MarginResult
from src.derivatives.greeks_engine import GreeksEngine, PortfolioGreeks
from src.derivatives.chain_builder import HistoricalChainBuilder, SyntheticOptionQuote
from src.derivatives.fno_simulator import FnOExecutionSimulator, FnOFillResult
from src.derivatives.expiry_engine import ExpiryEngine, ExpiryEvent
from src.derivatives.regime_engine import RegimeEngine, MarketRegime
from src.derivatives.fno_cost_model import IndianFnOCostModel
from src.derivatives.fno_backtest import FnOBacktestEngine
from src.derivatives.fno_paper_trader import FnOPaperTradingEngine

__all__ = [
    "DerivativeContract", "OptionLeg", "OptionIntent", "MultiLegPosition",
    "InstrumentType", "SettlementType",
    "MarginEngine", "MarginResult",
    "GreeksEngine", "PortfolioGreeks",
    "HistoricalChainBuilder", "SyntheticOptionQuote",
    "FnOExecutionSimulator", "FnOFillResult",
    "ExpiryEngine", "ExpiryEvent",
    "RegimeEngine", "MarketRegime",
    "IndianFnOCostModel",
    "FnOBacktestEngine",
    "FnOPaperTradingEngine",
]
