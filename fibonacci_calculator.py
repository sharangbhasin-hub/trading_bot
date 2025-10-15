"""
Fibonacci Retracement & Extension Calculator
"""
import pandas as pd
from typing import Dict, Optional, List

class FibonacciCalculator:
    """Calculate Fibonacci levels for trading"""
    
    def __init__(self):
        self.retracement_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        self.extension_levels = [1.272, 1.414, 1.618, 2.0, 2.618]
    
    def calculate_levels(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Calculate Fibonacci levels based on swing high/low
        Args:
            df: OHLC DataFrame
        Returns:
            Dict with Fibonacci levels and targets
        """
        if df is None or df.empty or len(df) < 10:
            return None
        
        try:
            # Find swing high and low
            high_price = df['high'].max()
            low_price = df['low'].min()
            
            # Determine trend
            recent_close = df['close'].iloc[-1]
            trend = 'uptrend' if recent_close > (high_price + low_price) / 2 else 'downtrend'
            
            # Calculate retracement levels
            price_range = high_price - low_price
            fib_levels = {}
            
            if trend == 'uptrend':
                for level in self.retracement_levels:
                    fib_levels[f'{level*100:.1f}%'] = high_price - (price_range * level)
                
                # Extension targets (above high)
                targets = []
                for ext in self.extension_levels:
                    target_price = low_price + (price_range * ext)
                    distance = target_price - recent_close
                    targets.append({
                        'level': f'{ext*100:.1f}%',
                        'price': round(target_price, 2),
                        'distance': round(distance, 2)
                    })
            
            else:  # downtrend
                for level in self.retracement_levels:
                    fib_levels[f'{level*100:.1f}%'] = low_price + (price_range * level)
                
                # Extension targets (below low)
                targets = []
                for ext in self.extension_levels:
                    target_price = high_price - (price_range * ext)
                    distance = recent_close - target_price
                    targets.append({
                        'level': f'{ext*100:.1f}%',
                        'price': round(target_price, 2),
                        'distance': round(distance, 2)
                    })
            
            return {
                'trend': trend,
                'swing_high': round(high_price, 2),
                'swing_low': round(low_price, 2),
                'fib_levels': fib_levels,
                'targets': targets[:3]  # Top 3 targets
            }
        
        except Exception as e:
            print(f"Fibonacci calculation error: {e}")
            return None
