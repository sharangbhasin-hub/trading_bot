"""
Fair Value Gap (FVG) Detector
Detects 3-candle gaps where middle candle doesn't overlap
"""
import pandas as pd
from typing import Dict, List

import logging

class FVGDetector:
    """Detects Fair Value Gaps"""
    
    def __init__(self):
        self.lookback_candles = 30
    
    def detect(self, df: pd.DataFrame, require_volume_spike: bool = False) -> List[Dict]:
        """
        Detect FVGs in dataframe
        
        Returns list of FVGs:
        [{
            'type': 'BULLISH' | 'BEARISH',
            'top': float,
            'bottom': float,
            'candle_index': int,
            'timestamp': datetime,
            'filled': bool,
            'fill_percentage': float  # NEW: 0-100, how much of gap is filled
        }]
        """
        logger = logging.getLogger(self.__class__.__name__)
        logger.info(f"🔍 FVG Detection: Analyzing {len(df)} candles (lookback={self.lookback_candles})")
        
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
                
                # ✅ FIX #1: SIZE FILTER - Only accept meaningful gaps
                gap_size = fvg_top - fvg_bottom
                gap_size_pct = (gap_size / current_price) * 100
                
                # Skip tiny gaps (noise) and huge gaps (anomalies)
                if gap_size_pct < 0.15 or gap_size_pct > 5.0:
                    continue  # Skip this FVG

                # ✅ FIX #4: VOLUME FILTER (optional)
                if require_volume_spike and 'volume' in df_recent.columns:
                    avg_volume = df_recent['volume'].tail(20).mean()
                    middle_volume = candle_middle['volume']
                    
                    if middle_volume < avg_volume * 1.5:
                        continue
                
                # Check fill status and percentage
                fill_info = self._check_fill_status(
                    fvg_bottom, 
                    fvg_top, 
                    current_price, 
                    df_recent.iloc[i+1:],
                    'BULLISH'
                )
                
                fvgs.append({
                    'type': 'BULLISH',
                    'top': fvg_top,
                    'bottom': fvg_bottom,
                    'candle_index': i,
                    'timestamp': candle_middle.get('timestamp', None),
                    'filled': fill_info['filled'],
                    'fill_percentage': fill_info['fill_percentage'],
                    'age_candles': len(df_recent) - i - 1
                })
            
            # Bearish FVG: Gap between candle_after.high and candle_before.low
            if candle_after['high'] < candle_before['low']:
                fvg_bottom = candle_after['high']
                fvg_top = candle_before['low']
                
                # ✅ FIX #1: SIZE FILTER - Only accept meaningful gaps
                gap_size = fvg_top - fvg_bottom
                gap_size_pct = (gap_size / current_price) * 100
                
                # Skip tiny gaps (noise) and huge gaps (anomalies)
                if gap_size_pct < 0.15 or gap_size_pct > 5.0:
                    continue  # Skip this FVG

                # ✅ FIX #4: VOLUME FILTER (optional but powerful)
                if require_volume_spike and 'volume' in df_recent.columns:
                    # Check if middle candle had above-average volume
                    avg_volume = df_recent['volume'].tail(20).mean()
                    middle_volume = candle_middle['volume']
                    
                    # Require at least 1.5x average volume for institutional footprint
                    if middle_volume < avg_volume * 1.5:
                        continue
                
                # Check fill status and percentage
                fill_info = self._check_fill_status(
                    fvg_bottom, 
                    fvg_top, 
                    current_price, 
                    df_recent.iloc[i+1:],
                    'BEARISH'
                )
                
                fvgs.append({
                    'type': 'BEARISH',
                    'top': fvg_top,
                    'bottom': fvg_bottom,
                    'candle_index': i,
                    'timestamp': candle_middle.get('timestamp', None),
                    'filled': fill_info['filled'],
                    'fill_percentage': fill_info['fill_percentage'],
                    'age_candles': len(df_recent) - i - 1
                })
        
        # ✅ FIX #3: Only accept FRESH FVGs (< 30% filled)
        # ✅ PROFESSIONAL SMC: Accept FVGs based on fill status
        # For double bottom/top strategies, PARTIALLY FILLED FVGs are THE SETUP!
        active_fvgs = []
        
        for fvg in fvgs:
            # ✅ ONLY reject 100% filled FVGs (fully exhausted zones)
            if fvg['fill_percentage'] >= 100:
                continue
            
            # ✅ ACCEPT partially filled FVGs (20-80% filled)
            # These show price tested the zone = support/resistance confirmation!
            # This is EXACTLY what creates double bottom/top patterns!
            
            # ✅ Optional: Reject too-old FVGs (stale zones)
            if fvg.get('age_candles', 0) > 20:  # Increased from 10 to 20
                continue
            
            # Mark fill status for strategy to use
            if fvg['fill_percentage'] >= 20 and fvg['fill_percentage'] <= 80:
                fvg['quality'] = 'TESTED'  # High value - zone was tested!
            elif fvg['fill_percentage'] < 20:
                fvg['quality'] = 'FRESH'  # Untested zone
            else:  # 80-99%
                fvg['quality'] = 'WEAK'  # Almost exhausted
            
            active_fvgs.append(fvg)
        
        # ✅ FIX #6: Only return FVGs near current price (within 3%)
        nearby_fvgs = []
        
        for fvg in active_fvgs:
            fvg_mid = (fvg['top'] + fvg['bottom']) / 2
            distance_pct = abs((current_price - fvg_mid) / current_price) * 100
            
            # ✅ Accept FVGs within reasonable distance (8% for 15min timeframe)
            # For intraday trading: 5-8% is normal price swing range
            max_distance = 8.0  # More realistic for pattern formation
            
            if distance_pct <= max_distance:
                fvg['distance_pct'] = round(distance_pct, 2)  # Store for later use
                nearby_fvgs.append(fvg)
        
        logger.info(f"🔍 FVG Results: Total found={len(fvgs)}, After fill filter={len(active_fvgs)}, Final (nearby)={len(nearby_fvgs)}")

        # ✅ ADD THIS: Detailed breakdown for debugging
        if len(fvgs) > 0 and len(nearby_fvgs) == 0:
            logger.warning(f"⚠️ FVG FILTERING ISSUE:")
            logger.warning(f"  - Found {len(fvgs)} raw FVGs")
            logger.warning(f"  - {len(fvgs) - len(active_fvgs)} rejected by fill filter (100% filled or age >20)") 
            logger.warning(f"  - {len(active_fvgs) - len(nearby_fvgs)} rejected by distance filter (>{8.0}%)")
            logger.warning(f"  - Suggestion: Increase fill_percentage threshold or distance filter")
        
        return nearby_fvgs
    
    def _check_fill_status(self, fvg_bottom: float, fvg_top: float, 
                          current_price: float, subsequent_candles: pd.DataFrame,
                          fvg_type: str) -> Dict:
        """
        Check if FVG is filled and by how much
        
        Returns:
        {
            'filled': bool,  # True if 100% filled
            'fill_percentage': float  # 0-100
        }
        """
        gap_size = fvg_top - fvg_bottom
        
        # Check highest price that entered the gap
        if len(subsequent_candles) > 0:
            if fvg_type == 'BULLISH':
                # For bullish FVG, check how far down price came into the gap
                lowest_in_gap = subsequent_candles['low'].min()
                
                if lowest_in_gap <= fvg_bottom:
                    # Fully filled (price went below gap)
                    return {'filled': True, 'fill_percentage': 100.0}
                elif lowest_in_gap < fvg_top:
                    # Partially filled
                    fill_amount = fvg_top - lowest_in_gap
                    fill_pct = (fill_amount / gap_size) * 100
                    return {'filled': False, 'fill_percentage': round(fill_pct, 1)}
            
            else:  # BEARISH FVG
                # For bearish FVG, check how far up price came into the gap
                highest_in_gap = subsequent_candles['high'].max()
                
                if highest_in_gap >= fvg_top:
                    # Fully filled (price went above gap)
                    return {'filled': True, 'fill_percentage': 100.0}
                elif highest_in_gap > fvg_bottom:
                    # Partially filled
                    fill_amount = highest_in_gap - fvg_bottom
                    fill_pct = (fill_amount / gap_size) * 100
                    return {'filled': False, 'fill_percentage': round(fill_pct, 1)}
        
        # Not filled at all
        return {'filled': False, 'fill_percentage': 0.0}
