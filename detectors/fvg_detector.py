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
            'filled': bool,
            'fill_percentage': float  # NEW: 0-100, how much of gap is filled
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
                    'fill_percentage': fill_info['fill_percentage']
                })
            
            # Bearish FVG: Gap between candle_after.high and candle_before.low
            if candle_after['high'] < candle_before['low']:
                fvg_bottom = candle_after['high']
                fvg_top = candle_before['low']
                
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
                    'fill_percentage': fill_info['fill_percentage']
                })
        
        # Filter out fully filled FVGs (keep partially filled ones)
        active_fvgs = [fvg for fvg in fvgs if fvg['fill_percentage'] < 100]
        
        return active_fvgs
    
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
