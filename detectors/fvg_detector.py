"""
Fair Value Gap (FVG) Detector
Detects 3-candle gaps where middle candle doesn't overlap
"""
import pandas as pd
from typing import Dict, List

class FVGDetector:
    """Detects Fair Value Gaps"""
    
    def __init__(self):
        self.lookback_candles = 50
    
    def detect(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect FVGs in dataframe
        
        Returns list of FVGs:
        [{
            'type': 'BULLISH' | 'BEARISH',
            'top': float,
            'bottom': float,
            'candle_index': int,
            'timestamp': datetime,
            'filled': bool
        }]
        """
        fvgs = []
        
        if len(df) < 3:
            return fvgs
        
        df_recent = df.tail(self.lookback_candles).reset_index(drop=True)
        current_price = df['close'].iloc[-1]
        
        for i in range(1, len(df_recent) - 1):
            candle_before = df_recent.iloc[i-1]
            candle_middle = df_recent.iloc[i]
            candle_after = df_recent.iloc[i+1]
            
            # Bullish FVG: Gap between candle_before.high and candle_after.low
            if candle_after['low'] > candle_before['high']:
                fvg_bottom = candle_before['high']
                fvg_top = candle_after['low']
                
                # Check if already filled
                filled = current_price < fvg_bottom
                
                fvgs.append({
                    'type': 'BULLISH',
                    'top': fvg_top,
                    'bottom': fvg_bottom,
                    'candle_index': i,
                    'timestamp': df_recent['timestamp'].iloc[i],
                    'filled': filled
                })
            
            # Bearish FVG: Gap between candle_after.high and candle_before.low
            elif candle_after['high'] < candle_before['low']:
                fvg_top = candle_before['low']
                fvg_bottom = candle_after['high']
                
                # Check if already filled
                filled = current_price > fvg_top
                
                fvgs.append({
                    'type': 'BEARISH',
                    'top': fvg_top,
                    'bottom': fvg_bottom,
                    'candle_index': i,
                    'timestamp': df_recent['timestamp'].iloc[i],
                    'filled': filled
                })
        
        # Return unfilled FVGs only, most recent first
        unfilled = [fvg for fvg in fvgs if not fvg['filled']]
        return unfilled[-5:] if unfilled else []
