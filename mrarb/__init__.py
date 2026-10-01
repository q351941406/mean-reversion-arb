"""mrarb: mean-reversion (pairs/stat-arb) toolkit on synthetic data.

Pipeline: synthetic universe -> data-quality report -> pair selection
(cointegration + half-life) -> OU / Kalman spread model -> signals ->
cost-aware backtest -> in-sample vs out-of-sample report.

References: see README.md.
"""

from .config import TRADING_DAYS, BacktestConfig, StratParams, UniverseConfig
from .synth import Universe, generate_universe

__all__ = [
    "TRADING_DAYS",
    "BacktestConfig",
    "StratParams",
    "Universe",
    "UniverseConfig",
    "generate_universe",
]
