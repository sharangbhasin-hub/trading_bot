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
            'candle_index': int,  # ✅ FIXED: Real index in original df
            'timestamp': datetime,
            'filled': bool,
            'fill_percentage': float,
            'quality': str,  # 'TESTED', 'FRESH', 'WEAK'
            'age_candles': int,
            'distance_pct': float,  # Distance from current price
            'price_inside': bool  # ✅ NEW: Is price inside FVG right now?
        }]
        """
        logger = logging.getLogger(self.__class__.__name__)
        logger.info(f"🔍 FVG Detection: Analyzing {len(df)} candles (lookback={self.lookback_candles})")
        
        fvgs = []
        
        if len(df) < 3:
            return fvgs
        
        df_recent = df.tail(self.lookback_candles).reset_index(drop=True)
        current_price = df['close'].iloc[-1]
        
        # ✅ FIX #1: Calculate offset for correct candle_index
        offset = len(df) - len(df_recent)
        
        for i in range(1, len(df_recent) - 1):
            candle_before = df_recent.iloc[i-1]
            candle_middle = df_recent.iloc[i]
            candle_after = df_recent.iloc[i+1]
            
            # Bullish FVG: Gap between candle_before.high and candle_after.low
            if candle_after['low'] > candle_before['high']:
                fvg_bottom = candle_before['high']
                fvg_top = candle_after['low']
                
                # Size filter
                gap_size = fvg_top - fvg_bottom
                gap_size_pct = (gap_size / current_price) * 100
                
                if gap_size_pct < 0.15 or gap_size_pct > 5.0:
                    continue
    
                # Volume filter (optional)
                if require_volume_spike and 'volume' in df_recent.columns:
                    avg_volume = df_recent['volume'].tail(20).mean()
                    middle_volume = candle_middle['volume']
                    
                    if middle_volume < avg_volume * 1.5:
                        continue
                
                # Check fill status
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
                    'candle_index': offset + i,  # ✅ FIXED
                    'timestamp': candle_middle.get('timestamp', None),
                    'filled': fill_info['filled'],
                    'fill_percentage': fill_info['fill_percentage'],
                    'age_candles': len(df_recent) - i - 1
                })
            
            # Bearish FVG
            if candle_after['high'] < candle_before['low']:
                fvg_bottom = candle_after['high']
                fvg_top = candle_before['low']
                
                gap_size = fvg_top - fvg_bottom
                gap_size_pct = (gap_size / current_price) * 100
                
                if gap_size_pct < 0.15 or gap_size_pct > 5.0:
                    continue
    
                if require_volume_spike and 'volume' in df_recent.columns:
                    avg_volume = df_recent['volume'].tail(20).mean()
                    middle_volume = candle_middle['volume']
                    
                    if middle_volume < avg_volume * 1.5:
                        continue
                
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
                    'candle_index': offset + i,  # ✅ FIXED
                    'timestamp': candle_middle.get('timestamp', None),
                    'filled': fill_info['filled'],
                    'fill_percentage': fill_info['fill_percentage'],
                    'age_candles': len(df_recent) - i - 1
                })
        
        # ✅ FIX #3: Filter by age (3-15 candles for retest strategy)
        active_fvgs = []
        
        for fvg in fvgs:
            # Reject 100% filled FVGs
            if fvg['fill_percentage'] >= 95:
                continue
            
            # ✅ ADDED: Age filter for retest strategy
            age = fvg.get('age_candles', 0)
            
            if age < 3:
                continue  # Too fresh
            
            if age > 20:  # Changed from 20
                continue  # Too stale
            
            # Mark quality
            if fvg['fill_percentage'] >= 20 and fvg['fill_percentage'] <= 80:
                fvg['quality'] = 'TESTED'
            elif fvg['fill_percentage'] < 20:
                fvg['quality'] = 'FRESH'
            else:
                fvg['quality'] = 'WEAK'
            
            active_fvgs.append(fvg)

        # ✅ Distance filter: Keep FVGs within reasonable range
        nearby_fvgs = []
        
        for fvg in active_fvgs:
            fvg_mid = (fvg['top'] + fvg['bottom']) / 2
            distance_pct = abs((current_price - fvg_mid) / current_price) * 100
            
            # ✅ Check if price is INSIDE the FVG
            price_inside_fvg = (fvg['bottom'] <= current_price <= fvg['top'])
            
            # ✅ CHANGED: Increase max distance from 1.5% to 4.0%
            max_distance = 4.0  # Changed from 1.5% (was eliminating all FVGs)
            
            if price_inside_fvg or distance_pct <= max_distance:
                fvg['distance_pct'] = round(distance_pct, 2)
                fvg['price_inside'] = price_inside_fvg  # ✅ NEW FIELD
                nearby_fvgs.append(fvg)
        
        logger.info(f"🔍 FVG Results: Total found={len(fvgs)}, After filters={len(active_fvgs)}, Final (nearby)={len(nearby_fvgs)}")
        
        if len(fvgs) > 0 and len(nearby_fvgs) == 0:
            logger.warning(f"⚠️ FVG FILTERING BREAKDOWN:")
            logger.warning(f"  - Raw FVGs: {len(fvgs)}")
            logger.warning(f"  - After fill/age filter: {len(active_fvgs)}")
            logger.warning(f"  - After distance filter (4.0%): {len(nearby_fvgs)}")  # Updated percentage
            
            # ✅ NEW: Log why FVGs were rejected
            if len(active_fvgs) > 0:
                logger.warning(f"  - Rejected FVG distances:")
                for fvg in active_fvgs:
                    fvg_mid = (fvg['top'] + fvg['bottom']) / 2
                    distance_pct = abs((current_price - fvg_mid) / current_price) * 100
                    logger.warning(f"    • {fvg['type']} FVG: {distance_pct:.2f}% away (max: 4.0%)")
        
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
