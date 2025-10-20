"""
Order Block Detector
Identifies the last opposite candle before a strong move
"""
import pandas as pd
from typing import Dict, Optional, List

class OrderBlockDetector:
    """Detects order blocks in price data"""
    
    def __init__(self):
        self.min_move_percent = 0.8  # Minimum 0.8% move to qualify
        self.lookback_candles = 50
    
    def detect(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect order blocks in dataframe
        
        Returns list of order blocks:
        [{
            'type': 'BULLISH' | 'BEARISH',
            'high': float,
            'low': float,
            'candle_index': int,
            'timestamp': datetime,
            'strength': 0-100
        }]
        """
        order_blocks = []
        
        # Need at least 20 candles
        if len(df) < 20:
            return order_blocks
        
        df_recent = df.tail(self.lookback_candles).reset_index(drop=True)
        
        # Find strong moves (>0.8% in 1-3 candles)
        for i in range(3, len(df_recent) - 1):
            # Check for bullish move
            move_start = df_recent['low'].iloc[i-3:i].min()
            move_end = df_recent['high'].iloc[i:i+3].max()
            move_pct = ((move_end - move_start) / move_start) * 100
            
            if move_pct >= self.min_move_percent:
                # Find last bearish candle before move
                for j in range(i-1, max(0, i-5), -1):
                    if df_recent['close'].iloc[j] < df_recent['open'].iloc[j]:
                        # This is order block
                        ob_high = df_recent['high'].iloc[j]
                        ob_low = df_recent['low'].iloc[j]
                        
                        # Calculate strength based on move size
                        strength = min(100, int(move_pct * 10))
                        
                        order_blocks.append({
                            'type': 'BULLISH',
                            'high': ob_high,
                            'low': ob_low,
                            'candle_index': j,
                            'timestamp': df_recent['timestamp'].iloc[j],
                            'strength': strength
                        })
                        break
            
            # Check for bearish move
            move_start = df_recent['high'].iloc[i-3:i].max()
            move_end = df_recent['low'].iloc[i:i+3].min()
            move_pct = ((move_start - move_end) / move_start) * 100
            
            if move_pct >= self.min_move_percent:
                # Find last bullish candle before move
                for j in range(i-1, max(0, i-5), -1):
                    if df_recent['close'].iloc[j] > df_recent['open'].iloc[j]:
                        # This is order block
                        ob_high = df_recent['high'].iloc[j]
                        ob_low = df_recent['low'].iloc[j]
                        
                        strength = min(100, int(move_pct * 10))
                        
                        order_blocks.append({
                            'type': 'BEARISH',
                            'high': ob_high,
                            'low': ob_low,
                            'candle_index': j,
                            'timestamp': df_recent['timestamp'].iloc[j],
                            'strength': strength
                        })
                        break
        
        # Return most recent order blocks (max 5)
        return order_blocks[-5:] if order_blocks else []
