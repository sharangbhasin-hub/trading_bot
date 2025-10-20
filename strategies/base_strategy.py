"""
Base Strategy Class - All strategies inherit from this
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, List
import pandas as pd

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.min_confidence = 70  # Minimum confidence to show signal
        self.retest_required = True  # All strategies need retest
    
    @abstractmethod
    def analyze(self, 
                df_5min: pd.DataFrame,
                df_15min: pd.DataFrame,
                df_1h: pd.DataFrame,
                df_4h: pd.DataFrame,
                spot_price: float,
                support: float,
                resistance: float,
                overall_trend: str) -> Dict:
        """
        Analyze market and return signal
        
        Returns dict with:
        {
            'signal': 'CALL' | 'PUT' | 'NO_TRADE',
            'confidence': 0-100,
            'entry_price': float,
            'stop_loss': float,
            'target': float,
            'reasoning': List[str],
            'setup_detected': bool,
            'retest_confirmed': bool,
            'candlestick_pattern': str | None
        }
        """
        pass
    
    def is_tradeable(self, result: Dict) -> bool:
        """Check if signal meets trading criteria"""
        if result['signal'] == 'NO_TRADE':
            return False
        if result['confidence'] < self.min_confidence:
            return False
        if self.retest_required and not result['retest_confirmed']:
            return False
        return True
    
    def _format_price(self, price: float) -> float:
        """Round price to 2 decimals"""
        return round(price, 2)
