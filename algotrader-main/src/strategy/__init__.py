"""Strategy package — all trading strategies for live, backtest & paper."""

from src.strategy.base import BaseStrategy
from src.strategy.ema_crossover import EMACrossoverStrategy
from src.strategy.mean_reversion import MeanReversionStrategy
from src.strategy.rsi_strategy import RSIStrategy
from src.strategy.vwap_breakout import VWAPBreakoutStrategy
from src.strategy.scanner_strategy import ScannerStrategy
from src.strategy.analysis_strategies import (
    ANALYSIS_STRATEGIES,
    ANALYSIS_STRATEGY_LABELS,
    MACDCrossoverStrategy,
    GoldenDeathCrossStrategy,
    VolumeBreakoutAnalysisStrategy,
    BollingerBandsStrategy,
    StochasticStrategy,
    IchimokuStrategy,
    VCPStrategy,
    PocketPivotStrategy,
    StageAnalysisStrategy,
    CCIExtremeStrategy,
    MFIStrategy,
    FiftyTwoWeekStrategy,
)

__all__ = [
    "BaseStrategy",
    "EMACrossoverStrategy",
    "MeanReversionStrategy",
    "RSIStrategy",
    "VWAPBreakoutStrategy",
    "ScannerStrategy",
    "ANALYSIS_STRATEGIES",
    "ANALYSIS_STRATEGY_LABELS",
    "MACDCrossoverStrategy",
    "GoldenDeathCrossStrategy",
    "VolumeBreakoutAnalysisStrategy",
    "BollingerBandsStrategy",
    "StochasticStrategy",
    "IchimokuStrategy",
    "VCPStrategy",
    "PocketPivotStrategy",
    "StageAnalysisStrategy",
    "CCIExtremeStrategy",
    "MFIStrategy",
    "FiftyTwoWeekStrategy",
]
