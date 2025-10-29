"""
Paper Trading Module
====================

Real-time paper trading system for Cryptocurrency and Forex markets.
Supports scalping (1H→1min) and intraday (1D→1H) trading modes.

Author: Trading System
Version: 1.0.0
Date: October 29, 2025
"""

__version__ = "1.0.0"
__author__ = "Trading System"

# Module exports
from .config import PAPER_TRADING_CONFIG
from .pnl_calculator import PnLCalculator
from .trade_database import TradeDatabase

__all__ = [
    'PAPER_TRADING_CONFIG',
    'PnLCalculator',
    'TradeDatabase',
]
