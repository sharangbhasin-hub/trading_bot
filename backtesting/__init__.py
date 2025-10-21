"""
Backtesting Module
Complete backtesting framework for trading strategies

This module provides a comprehensive backtesting system that:
- Loads historical data
- Replays market conditions
- Records signals and trades
- Analyzes performance
- Generates reports and visualizations
"""

__version__ = '1.0.0'
__author__ = 'Trading Bot Team'

# Core components
from .config import BacktestConfig
from .data_loader import DataLoader
from .replay_engine import ReplayEngine
from .signal_recorder import SignalRecorder
from .trade_simulator import TradeSimulator, Trade

# Analyzers
from .performance_analyzer import PerformanceAnalyzer
from .market_classifier import MarketClassifier
from .exit_analyzer import ExitAnalyzer
from .failure_analyzer import FailureAnalyzer
from .parameter_optimizer import ParameterOptimizer

# Output
from .visualization import Visualizer
from .report_generator import ReportGenerator

# Main orchestrator
from .backtest_runner import BacktestRunner

__all__ = [
    # Core
    'BacktestConfig',
    'DataLoader',
    'ReplayEngine',
    'SignalRecorder',
    'TradeSimulator',
    'Trade',
    
    # Analyzers
    'PerformanceAnalyzer',
    'MarketClassifier',
    'ExitAnalyzer',
    'FailureAnalyzer',
    'ParameterOptimizer',
    
    # Output
    'Visualizer',
    'ReportGenerator',
    
    # Main
    'BacktestRunner',
]
