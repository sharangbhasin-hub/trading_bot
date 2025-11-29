"""
Order Block Detector
Identifies the last opposite candle before a strong move
"""
import pandas as pd
from typing import Dict, Optional, List

import logging

class OrderBlockDetector:
    """Detects order blocks in price data"""
    
    def __init__(self):
        self.min_move_percent = 0.5  # INCREASED from 0.8% to 1.2% for stronger signals
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

        logger = logging.getLogger(self.__class__.__name__)
        logger.info(f"🔍 OB Detection: Analyzing {len(df)} candles (lookback={self.lookback_candles}, min_move={self.min_move_percent}%)")
        
        move_count = 0
        
        # Need at least 20 candles
        if len(df) < 20:
            return order_blocks
        
        df_recent = df.tail(self.lookback_candles).reset_index(drop=True)
        
        # Find strong moves (>1.2% in 1-3 candles)
        for i in range(3, len(df_recent) - 1):
            # Check for bullish move
            move_start = df_recent['low'].iloc[i-3:i].min()
            move_end = df_recent['high'].iloc[i:i+3].max()
            move_pct = ((move_end - move_start) / move_start) * 100
            
            if move_pct >= self.min_move_percent:
                move_count += 1
                # Find the last bearish candle before the move (this is the bullish OB)
                for j in range(i-1, max(0, i-5), -1):
                    candle = df_recent.iloc[j]
                    if candle['close'] < candle['open']:  # Bearish candle
                        # Calculate strength based on move size
                        strength = min(100, int(move_pct * 8))
                        
                        order_blocks.append({
                            'type': 'BULLISH',
                            'high': candle['high'],
                            'low': candle['low'],
                            'candle_index': j,
                            'timestamp': candle.get('timestamp', None),
                            'strength': strength
                        })
                        break
            
            # Check for bearish move
            move_start = df_recent['high'].iloc[i-3:i].max()
            move_end = df_recent['low'].iloc[i:i+3].min()
            move_pct = ((move_start - move_end) / move_start) * 100
            
            if move_pct >= self.min_move_percent:
                # Find the last bullish candle before the move (this is the bearish OB)
                for j in range(i-1, max(0, i-5), -1):
                    candle = df_recent.iloc[j]
                    if candle['close'] > candle['open']:  # Bullish candle
                        # Calculate strength based on move size
                        strength = min(100, int(move_pct * 8))
                        
                        order_blocks.append({
                            'type': 'BEARISH',
                            'high': candle['high'],
                            'low': candle['low'],
                            'candle_index': j,
                            'timestamp': candle.get('timestamp', None),
                            'strength': strength
                        })
                        break
        
        # Remove duplicates (keep strongest)
        unique_obs = {}
        for ob in order_blocks:
            key = (ob['type'], round(ob['low'], 2), round(ob['high'], 2))
            if key not in unique_obs or ob['strength'] > unique_obs[key]['strength']:
                unique_obs[key] = ob

        logger.info(f"🔍 OB Results: Checked {len(df_recent)} candles, found {move_count} strong moves, detected {len(unique_obs)} OBs")
        
        return list(unique_obs.values())
