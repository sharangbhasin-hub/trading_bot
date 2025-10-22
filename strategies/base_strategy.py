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
        self.min_confidence = 55  # Minimum confidence to show signal
        self.retest_required = False  # All strategies need retest
        self.min_risk_reward = 0.8  # Minimum R:R ratio

    def is_valid_trading_time(self, timestamp) -> bool:
        """
        Check if current time is suitable for trading
        Universal filter - works for any index
        
        Evidence from backtests:
        - 29-31% WR in closing hour (15:15-15:30) across ALL 4 tests
        - Lower liquidity and wider spreads at market close
        
        Args:
            timestamp: Can be datetime object, pandas Timestamp, or string
            
        Returns:
            bool: True if time is valid for trading
        """
        # Handle None case
        if timestamp is None:
            return True  # Allow if no timestamp provided
            
        # Convert string to datetime if needed
        if isinstance(timestamp, str):
            from datetime import datetime
            try:
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                return True  # Allow if can't parse
        
        # Extract hour and minute
        try:
            hour = timestamp.hour
            minute = timestamp.minute
        except AttributeError:
            return True  # Allow if no hour/minute attributes
        
        # FILTER: Block closing hour (15:15-15:30)
        # Evidence: 29-31% WR = guaranteed losses
        if hour == 15 and minute >= 15:
            return False
        
        return True
    
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
    
    def is_tradeable(self, result: Dict, timestamp=None) -> bool:
        """
        Check if signal meets trading criteria
        
        Args:
            result: Strategy result dict
            timestamp: Current timestamp to validate trading time
            
        Returns:
            bool: True if signal is tradeable
        """
        if result['signal'] == 'NO_TRADE':
            return False
        
        if result['confidence'] < self.min_confidence:
            return False
        
        if self.retest_required and not result['retest_confirmed']:
            return False
        
        # ✅ NEW: Check trading time if provided
        if current_time is not None:
            is_valid_time, time_reason = self.validate_trading_time(current_time)
            if not is_valid_time:
                # Add rejection reason to result
                if 'reasoning' in result:
                    result['reasoning'].append(f"⏰ {time_reason}")
                return False
        
        return True
    
    def calculate_risk_reward(self, entry: float, stop_loss: float, target: float) -> float:
        """
        Calculate risk:reward ratio
        
        Args:
            entry: Entry price
            stop_loss: Stop loss price
            target: Target price
            
        Returns:
            Risk:Reward ratio (e.g., 2.0 means 1:2 ratio)
        """
        risk = abs(entry - stop_loss)
        reward = abs(target - entry)
        
        if risk == 0:
            return 0.0
        
        return round(reward / risk, 2)
    
    def validate_risk_reward(self, result: Dict) -> Dict:
        """
        Validate and update result with R:R ratio
        
        Args:
            result: Strategy result dict
            
        Returns:
            Updated result dict with R:R validation
        """
        if result['signal'] == 'NO_TRADE':
            return result
        
        # Calculate R:R ratio
        rr_ratio = self.calculate_risk_reward(
            result['entry_price'],
            result['stop_loss'],
            result['target']
        )
        
        # Add R:R to result
        result['risk_reward_ratio'] = rr_ratio
        
        # Validate minimum R:R
        if rr_ratio < self.min_risk_reward:
            result['signal'] = 'NO_TRADE'
            result['confidence'] = 0
            result['reasoning'].append(
                f'❌ Risk:Reward too low: {rr_ratio:.2f}:1 (need minimum {self.min_risk_reward}:1)'
            )
        else:
            result['reasoning'].append(
                f'✅ Risk:Reward ratio: {rr_ratio:.2f}:1'
            )
        
        return result
    
    def calculate_dynamic_stop_loss(self, 
                                   zone_low: float, 
                                   zone_high: float, 
                                   direction: str,
                                   spot_price: float) -> float:
        """
        Calculate dynamic stop loss based on index type
        
        Args:
            zone_low: Bottom of the setup zone
            zone_high: Top of the setup zone
            direction: 'BULLISH' or 'BEARISH'
            spot_price: Current spot price
            
        Returns:
            Stop loss price
        """
        # Determine if BANKNIFTY or NIFTY based on price
        if spot_price > 40000:  # BANKNIFTY
            stop_distance_pct = 0.005  # 0.5%
        else:  # NIFTY
            stop_distance_pct = 0.003  # 0.3%
        
        if direction == 'BULLISH':
            # Stop below zone
            stop_loss = zone_low * (1 - stop_distance_pct)
        else:  # BEARISH
            # Stop above zone
            stop_loss = zone_high * (1 + stop_distance_pct)
        
        return self._format_price(stop_loss)
    
    def _format_price(self, price: float) -> float:
        """Round price to 2 decimals"""
        return round(price, 2)

    def validate_trading_time(self, current_time) -> tuple[bool, str]:
        """
        Check if current time is within trading hours
        
        Args:
            current_time: datetime object or pandas Timestamp
            
        Returns:
            (is_valid, reason)
        """
        from datetime import time
        
        # Convert to time object if needed
        if hasattr(current_time, 'time'):
            check_time = current_time.time()
        else:
            check_time = current_time
        
        # Market hours: 9:15 AM to 3:30 PM
        market_open = time(9, 15)
        market_close = time(15, 30)
        
        # ❌ BLOCK TRADES AFTER 3:00 PM (closing hour volatility)
        closing_hour_cutoff = time(15, 0)
        
        # Check if market is open
        if check_time < market_open:
            return False, "Market not open yet (opens 9:15 AM)"
        
        if check_time >= market_close:
            return False, "Market closed (closes 3:30 PM)"
        
        # ✅ NEW: Block closing hour trades
        if check_time >= closing_hour_cutoff:
            return False, "Closing hour (3:00-3:30 PM) - avoid low liquidity period"
        
        # ✅ Valid trading time
        return True, "Valid trading hours"

