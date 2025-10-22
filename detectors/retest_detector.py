"""
Retest Detector
Universal retest logic for all strategies
"""
import pandas as pd
from typing import Dict, Optional

class RetestDetector:
    """Detects retest patterns"""
    
    def __init__(self):
        # ✅ RELAXED: 0.15% instead of 0.3%
        # NIFTY 23,000: 0.15% = 35 pts (realistic pullback ~1 ATR)
        # BANKNIFTY 50,000: 0.15% = 75 pts (realistic pullback ~1 ATR)
        self.min_move_away_pct = 0.15  # Reduced from 0.3%
        self.zone_tolerance_pct = 0.15  # Keep same
    
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
        touch_tolerance = zone_range * 0.15  # 15% of zone range
        
        # PHASE 1: Find first touch of zone
        first_touch_idx = None
        for i in range(len(df) - 4):  # Leave room for move away and retest
            candle = df.iloc[i]
            
            # Check if candle touched the zone
            if expected_direction == 'BULLISH':
                # For bullish, look for touch from above (wick down into zone)
                if candle['low'] <= zone_high and candle['close'] >= zone_low:
                    first_touch_idx = i
                    break
            else:  # BEARISH
                # For bearish, look for touch from below (wick up into zone)
                if candle['high'] >= zone_low and candle['close'] <= zone_high:
                    first_touch_idx = i
                    break
        
        if first_touch_idx is None:
            return {
                'retest_confirmed': False,
                'first_touch_index': None,
                'retest_index': None,
                'reasoning': 'No initial touch of zone detected'
            }
        
        # PHASE 2: Check if price moved away (at least 0.15%)
        # ✅ RELAXED: Only need 1 candle away (faster confirmation)
        moved_away = False
        consecutive_away_count = 0
        required_consecutive = 1  # REDUCED from 2 to 1
        move_away_start_idx = None
        
        for i in range(first_touch_idx + 1, len(df) - 2):
            candle = df.iloc[i]
            
            if expected_direction == 'BULLISH':
                # Price should move down (away from zone)
                distance_from_zone = ((zone_low - candle['close']) / zone_low) * 100
                if distance_from_zone >= self.min_move_away_pct:
                    consecutive_away_count += 1
                    if consecutive_away_count == 1:
                        move_away_start_idx = i
                    if consecutive_away_count >= required_consecutive:
                        moved_away = True
                        break
                else:
                    # Reset counter if price comes back too early
                    consecutive_away_count = 0
                    move_away_start_idx = None
            else:  # BEARISH
                # Price should move up (away from zone)
                distance_from_zone = ((candle['close'] - zone_high) / zone_high) * 100
                if distance_from_zone >= self.min_move_away_pct:
                    consecutive_away_count += 1
                    if consecutive_away_count == 1:
                        move_away_start_idx = i
                    if consecutive_away_count >= required_consecutive:
                        moved_away = True
                        break
                else:
                    # Reset counter if price comes back too early
                    consecutive_away_count = 0
                    move_away_start_idx = None
        
        if not moved_away:
            return {
                'retest_confirmed': False,
                'first_touch_index': first_touch_idx,
                'retest_index': None,
                'reasoning': f'Price did not move away {self.min_move_away_pct}% from zone (or not for {required_consecutive} consecutive candles)'
            }
        
        # PHASE 3: Check if price returned to zone (retest)
        retest_idx = None
        for i in range(move_away_start_idx + required_consecutive, len(df)):
            candle = df.iloc[i]
            
            # Check if candle touched zone again
            if expected_direction == 'BULLISH':
                # Bullish retest: Price comes back up to zone and bounces
                if candle['high'] >= zone_low - touch_tolerance and candle['low'] <= zone_high + touch_tolerance:
                    # Check for bounce (close should be in or above zone)
                    if candle['close'] >= zone_low:
                        retest_idx = i
                        break
            else:  # BEARISH
                # Bearish retest: Price comes back down to zone and bounces
                if candle['low'] <= zone_high + touch_tolerance and candle['high'] >= zone_low - touch_tolerance:
                    # Check for bounce (close should be in or below zone)
                    if candle['close'] <= zone_high:
                        retest_idx = i
                        break
        
        if retest_idx is None:
            return {
                'retest_confirmed': False,
                'first_touch_index': first_touch_idx,
                'retest_index': None,
                'reasoning': 'Price moved away but did not return to retest zone'
            }
        
        # SUCCESS: All 3 phases confirmed
        return {
            'retest_confirmed': True,
            'first_touch_index': first_touch_idx,
            'retest_index': retest_idx,
            'reasoning': f'Retest confirmed: Touch at {first_touch_idx} → Moved away for {required_consecutive}+ candles → Retested at {retest_idx}'
        }
