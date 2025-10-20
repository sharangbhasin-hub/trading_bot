"""
Retest Detector
Universal retest logic for all strategies
"""
import pandas as pd
from typing import Dict, Optional

class RetestDetector:
    """Detects retest patterns"""
    
    def __init__(self):
        self.min_move_away_pct = 0.3  # Must move 0.3% away from zone
        self.zone_tolerance_pct = 0.15  # 0.15% tolerance for "touching" zone
    
    def check_retest(self, 
                     df: pd.DataFrame, 
                     zone_high: float, 
                     zone_low: float,
                     expected_direction: str) -> Dict:
        """
        Check if retest pattern exists
        
        Args:
            df: Price dataframe (must have timestamp, open, high, low, close)
            zone_high: Top of the zone
            zone_low: Bottom of the zone
            expected_direction: 'BULLISH' or 'BEARISH'
        
        Returns:
            {
                'retest_confirmed': bool,
                'first_touch_index': int | None,
                'retest_index': int | None,
                'reasoning': str
            }
        """
        if len(df) < 5:
            return {
                'retest_confirmed': False,
                'first_touch_index': None,
                'retest_index': None,
                'reasoning': 'Not enough candles'
            }
        
        zone_mid = (zone_high + zone_low) / 2
        zone_range = zone_high - zone_low
        touch_tolerance = zone_range * (self.zone_tolerance_pct / 100)
        
        df_recent = df.tail(30).reset_index(drop=True)
        
        first_touch_idx = None
        moved_away = False
        retest_idx = None
        
        # Phase 1: Find first touch
        for i in range(len(df_recent)):
            candle_low = df_recent['low'].iloc[i]
            candle_high = df_recent['high'].iloc[i]
            
            # Check if candle touches zone
            if expected_direction == 'BULLISH':
                if candle_low <= zone_high + touch_tolerance and candle_high >= zone_low - touch_tolerance:
                    first_touch_idx = i
                    break
            else:  # BEARISH
                if candle_high >= zone_low - touch_tolerance and candle_low <= zone_high + touch_tolerance:
                    first_touch_idx = i
                    break
        
        if first_touch_idx is None:
            return {
                'retest_confirmed': False,
                'first_touch_index': None,
                'retest_index': None,
                'reasoning': 'Zone not touched yet'
            }
        
        # Phase 2: Check if price moved away
        for i in range(first_touch_idx + 1, len(df_recent)):
            if expected_direction == 'BULLISH':
                # Price should move up away from zone
                if df_recent['low'].iloc[i] > zone_high * (1 + self.min_move_away_pct / 100):
                    moved_away = True
                    break
            else:  # BEARISH
                # Price should move down away from zone
                if df_recent['high'].iloc[i] < zone_low * (1 - self.min_move_away_pct / 100):
                    moved_away = True
                    break
        
        if not moved_away:
            return {
                'retest_confirmed': False,
                'first_touch_index': first_touch_idx,
                'retest_index': None,
                'reasoning': 'Price has not moved away from zone yet'
            }
        
        # Phase 3: Check for retest (return to zone)
        for i in range(first_touch_idx + 2, len(df_recent)):
            candle_low = df_recent['low'].iloc[i]
            candle_high = df_recent['high'].iloc[i]
            
            # Check if candle touches zone again
            if expected_direction == 'BULLISH':
                if candle_low <= zone_high + touch_tolerance and candle_high >= zone_low - touch_tolerance:
                    retest_idx = i
                    break
            else:  # BEARISH
                if candle_high >= zone_low - touch_tolerance and candle_low <= zone_high + touch_tolerance:
                    retest_idx = i
                    break
        
        if retest_idx is not None:
            return {
                'retest_confirmed': True,
                'first_touch_index': first_touch_idx,
                'retest_index': retest_idx,
                'reasoning': f'Retest confirmed: First touch at candle {first_touch_idx}, retest at candle {retest_idx}'
            }
        else:
            return {
                'retest_confirmed': False,
                'first_touch_index': first_touch_idx,
                'retest_index': None,
                'reasoning': 'Price moved away but has not retested yet'
            }
