"""
Key Level Detector for CRT-TBS Strategy
========================================

Detects five types of key levels on HTF:
1. OHP (Old High Purged)
2. OLP (Old Low Purged)
3. FVG (Fair Value Gap)
4. OB (Order Block)
5. RB (Rejection Block)

Based on CRT-TBS trading strategy documentation.

Author: Trading System
Date: October 23, 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

# Import existing detectors (reuse what's already built)
from detectors.fvg_detector import FVGDetector
# from detectors.order_block_detector import detect_order_blocks
from detectors.liquidity_detector import find_liquidity_zones


class KeyLevelDetector:
    """
    Detects all five key levels required for CRT-TBS strategy.
    
    Key Levels (per documentation):
    - OHP: Old High Purged (price sweeps above previous high)
    - OLP: Old Low Purged (price sweeps below previous low)
    - FVG: Fair Value Gap (three-candle gap pattern)
    - OB: Order Block (last opposing candle before strong move)
    - RB: Rejection Block (candle with long wick showing rejection)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Key Level Detector.
        
        Args:
            config: Configuration dictionary with:
                   - ohp_olp_lookback: Lookback period for swing detection
                   - fvg_min_gap_percent: Minimum gap size for FVG (default 0.05%)
                   - ob_min_consecutive: Minimum consecutive candles for OB (default 5)
                   - rb_min_wick_ratio: Minimum wick ratio for RB (default 0.40)
        """
        self.config = config or {}
        
        # OHP/OLP configuration
        self.ohp_olp_lookback = self.config.get('ohp_olp_lookback', 20)
        self.swing_left = self.config.get('swing_left', 2)
        self.swing_right = self.config.get('swing_right', 2)
        
        # FVG configuration
        self.fvg_min_gap_percent = self.config.get('fvg_min_gap_percent', 0.05)
        
        # Order Block configuration
        self.ob_min_consecutive = self.config.get('ob_min_consecutive', 5)
        
        # Rejection Block configuration
        self.rb_min_wick_ratio = self.config.get('rb_min_wick_ratio', 0.40)
    
    def detect_swing_points(
        self, 
        df: pd.DataFrame
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Detect swing highs and lows for OHP/OLP identification.
        
        Uses standard technical analysis definition:
        - Swing High: Peak with lower highs on both sides
        - Swing Low: Trough with higher lows on both sides
        
        Args:
            df: OHLC dataframe
        
        Returns:
            Tuple of (swing_highs_list, swing_lows_list)
        """
        swing_highs = []
        swing_lows = []
        
        left = self.swing_left
        right = self.swing_right
        
        for i in range(left, len(df) - right):
            # Swing High Detection
            is_swing_high = True
            current_high = df.iloc[i]['high']
            
            # Check left side
            for j in range(i - left, i):
                if df.iloc[j]['high'] >= current_high:
                    is_swing_high = False
                    break
            
            # Check right side
            if is_swing_high:
                for j in range(i + 1, i + right + 1):
                    if df.iloc[j]['high'] >= current_high:
                        is_swing_high = False
                        break
            
            if is_swing_high:
                swing_highs.append({
                    'index': i,
                    'price': current_high,
                    'timestamp': df.index[i]
                })
            
            # Swing Low Detection
            is_swing_low = True
            current_low = df.iloc[i]['low']
            
            # Check left side
            for j in range(i - left, i):
                if df.iloc[j]['low'] <= current_low:
                    is_swing_low = False
                    break
            
            # Check right side
            if is_swing_low:
                for j in range(i + 1, i + right + 1):
                    if df.iloc[j]['low'] <= current_low:
                        is_swing_low = False
                        break
            
            if is_swing_low:
                swing_lows.append({
                    'index': i,
                    'price': current_low,
                    'timestamp': df.index[i]
                })
        
        return swing_highs, swing_lows
    
    def detect_ohp(
        self, 
        df: pd.DataFrame, 
        current_candle_idx: int
    ) -> Optional[Dict]:
        """
        Detect Old High Purged (OHP).
        
        OHP occurs when:
        1. Price has a swing high in recent history
        2. Current candle's high breaks above that swing high
        3. Current candle closes back below the swing high (manipulation)
        
        Args:
            df: OHLC dataframe
            current_candle_idx: Index of candle to check for OHP
        
        Returns:
            Dict with OHP info or None
        """
        # Get lookback window
        lookback_start = max(0, current_candle_idx - self.ohp_olp_lookback)
        df_lookback = df.iloc[lookback_start:current_candle_idx]
        
        if len(df_lookback) < 5:
            return None
        
        # Find swing highs in lookback period
        swing_highs, _ = self.detect_swing_points(df_lookback)
        
        if not swing_highs:
            return None
        
        # Get most recent swing high
        old_high = max(swing_highs, key=lambda x: x['price'])
        
        # Check if current candle purges this high
        current_candle = df.iloc[current_candle_idx]
        
        if (current_candle['high'] > old_high['price'] and 
            current_candle['close'] < old_high['price']):
            return {
                'type': 'OHP',
                'level': old_high['price'],
                'purge_candle_idx': current_candle_idx,
                'purge_high': current_candle['high'],
                'close': current_candle['close'],
                'timestamp': current_candle.name if hasattr(current_candle, 'name') else None
            }
        
        return None
    
    def detect_olp(
        self, 
        df: pd.DataFrame, 
        current_candle_idx: int
    ) -> Optional[Dict]:
        """
        Detect Old Low Purged (OLP).
        
        OLP occurs when:
        1. Price has a swing low in recent history
        2. Current candle's low breaks below that swing low
        3. Current candle closes back above the swing low (manipulation)
        
        Args:
            df: OHLC dataframe
            current_candle_idx: Index of candle to check for OLP
        
        Returns:
            Dict with OLP info or None
        """
        # Get lookback window
        lookback_start = max(0, current_candle_idx - self.ohp_olp_lookback)
        df_lookback = df.iloc[lookback_start:current_candle_idx]
        
        if len(df_lookback) < 5:
            return None
        
        # Find swing lows in lookback period
        _, swing_lows = self.detect_swing_points(df_lookback)
        
        if not swing_lows:
            return None
        
        # Get most recent swing low
        old_low = min(swing_lows, key=lambda x: x['price'])
        
        # Check if current candle purges this low
        current_candle = df.iloc[current_candle_idx]
        
        if (current_candle['low'] < old_low['price'] and 
            current_candle['close'] > old_low['price']):
            return {
                'type': 'OLP',
                'level': old_low['price'],
                'purge_candle_idx': current_candle_idx,
                'purge_low': current_candle['low'],
                'close': current_candle['close'],
                'timestamp': current_candle.name if hasattr(current_candle, 'name') else None
            }
        
        return None
    
    def detect_fvg_levels(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Fair Value Gaps (FVG) using existing FVGDetector.
        
        Args:
            df: OHLC dataframe
        
        Returns:
            List of FVG dictionaries
        """
        try:
            # ✅ Use the FVGDetector class (not a function)
            from detectors.fvg_detector import FVGDetector
            
            fvg_detector = FVGDetector()
            fvg_list = fvg_detector.detect(df)
            
            # Convert to CRT-TBS expected format
            fvgs = []
            for fvg in fvg_list:
                # Your FVGDetector returns:
                # {
                #     'type': 'BULLISH' or 'BEARISH',
                #     'top': float,
                #     'bottom': float,
                #     'candle_index': int,
                #     'timestamp': datetime,
                #     'filled': bool,
                #     'fill_percentage': float
                # }
                
                gap_size = fvg['top'] - fvg['bottom']
                gap_percent = (gap_size / fvg['bottom']) * 100 if fvg['bottom'] > 0 else 0
                
                # Filter by minimum gap size
                if gap_percent >= self.fvg_min_gap_percent:
                    fvgs.append({
                        'type': 'FVG',
                        'fvg_type': fvg['type'].lower(),  # 'BULLISH' -> 'bullish'
                        'top': fvg['top'],
                        'bottom': fvg['bottom'],
                        'gap_size': gap_size,
                        'gap_percent': gap_percent,
                        'index': fvg['candle_index'],
                        'timestamp': fvg['timestamp'],
                        'filled': fvg['filled'],
                        'fill_percentage': fvg['fill_percentage']
                    })
            
            return fvgs
            
        except Exception as e:
            # Fallback: manual FVG detection
            import logging
            logging.warning(f"FVGDetector failed, using manual detection: {e}")
            return self._detect_fvg_manual(df)
    
    def _detect_fvg_manual(self, df: pd.DataFrame) -> List[Dict]:
        """
        Manual FVG detection if existing detector fails.
        """
        fvgs = []
        
        for i in range(2, len(df)):
            candle_0 = df.iloc[i-2]
            candle_2 = df.iloc[i]
            
            # Bullish FVG: gap between candle_0.high and candle_2.low
            if candle_2['low'] > candle_0['high']:
                gap_size = candle_2['low'] - candle_0['high']
                gap_percent = (gap_size / candle_0['high']) * 100
                
                if gap_percent >= self.fvg_min_gap_percent:
                    fvgs.append({
                        'type': 'FVG',
                        'fvg_type': 'bullish',
                        'top': candle_2['low'],
                        'bottom': candle_0['high'],
                        'gap_size': gap_size,
                        'gap_percent': gap_percent,
                        'index': i,
                        'timestamp': df.index[i]
                    })
            
            # Bearish FVG: gap between candle_0.low and candle_2.high
            elif candle_2['high'] < candle_0['low']:
                gap_size = candle_0['low'] - candle_2['high']
                gap_percent = (gap_size / candle_0['low']) * 100
                
                if gap_percent >= self.fvg_min_gap_percent:
                    fvgs.append({
                        'type': 'FVG',
                        'fvg_type': 'bearish',
                        'top': candle_0['low'],
                        'bottom': candle_2['high'],
                        'gap_size': gap_size,
                        'gap_percent': gap_percent,
                        'index': i,
                        'timestamp': df.index[i]
                    })
        
        return fvgs
    
    def detect_order_block_levels(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Order Blocks using existing detector.
        
        Args:
            df: OHLC dataframe
        
        Returns:
            List of OB dictionaries
        """
        try:
            # Use existing order block detector
            ob_df = detect_order_blocks(df, lookback=self.ohp_olp_lookback)
            
            order_blocks = []
            for idx, row in ob_df.iterrows():
                if row.get('is_bullish_ob') or row.get('is_bearish_ob'):
                    ob_type = 'bullish' if row.get('is_bullish_ob') else 'bearish'
                    
                    order_blocks.append({
                        'type': 'OB',
                        'ob_type': ob_type,
                        'high': row['high'],
                        'low': row['low'],
                        'index': idx,
                        'timestamp': idx
                    })
            
            return order_blocks
        except Exception:
            return []
    
    def detect_rejection_block(
        self, 
        df: pd.DataFrame, 
        current_candle_idx: int
    ) -> Optional[Dict]:
        """
        Detect Rejection Block (RB).
        
        RB is identified by:
        1. Long wick (> 40% of total range)
        2. Shows price rejection from a level
        
        Args:
            df: OHLC dataframe
            current_candle_idx: Index of candle to check
        
        Returns:
            Dict with RB info or None
        """
        candle = df.iloc[current_candle_idx]
        
        body_size = abs(candle['close'] - candle['open'])
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        total_range = candle['high'] - candle['low']
        
        if total_range == 0:
            return None
        
        upper_wick_ratio = upper_wick / total_range
        lower_wick_ratio = lower_wick / total_range
        
        # Bearish rejection (long upper wick)
        if upper_wick_ratio >= self.rb_min_wick_ratio:
            return {
                'type': 'RB',
                'rb_type': 'bearish',
                'zone_top': candle['high'],
                'zone_bottom': candle['high'] - upper_wick,
                'wick_ratio': upper_wick_ratio,
                'index': current_candle_idx,
                'timestamp': candle.name if hasattr(candle, 'name') else None
            }
        
        # Bullish rejection (long lower wick)
        elif lower_wick_ratio >= self.rb_min_wick_ratio:
            return {
                'type': 'RB',
                'rb_type': 'bullish',
                'zone_top': candle['low'] + lower_wick,
                'zone_bottom': candle['low'],
                'wick_ratio': lower_wick_ratio,
                'index': current_candle_idx,
                'timestamp': candle.name if hasattr(candle, 'name') else None
            }
        
        return None
    
    def detect_all_keylevels(
        self, 
        df: pd.DataFrame, 
        crt_candle_idx: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Detect all five key levels for a given CRT candle or current position.
        
        Args:
            df: OHLC dataframe
            crt_candle_idx: Optional index of CRT candle (None = use last candle)
        
        Returns:
            Dictionary with all detected key levels:
            {
                'ohp': Dict or None,
                'olp': Dict or None,
                'fvg': List of Dicts,
                'ob': List of Dicts,
                'rb': Dict or None,
                'has_any_keylevel': Boolean
            }
        """
        if crt_candle_idx is None:
            crt_candle_idx = len(df) - 1
        
        # Detect each key level type
        ohp = self.detect_ohp(df, crt_candle_idx)
        olp = self.detect_olp(df, crt_candle_idx)
        fvg_list = self.detect_fvg_levels(df.iloc[:crt_candle_idx+1])
        ob_list = self.detect_order_block_levels(df.iloc[:crt_candle_idx+1])
        rb = self.detect_rejection_block(df, crt_candle_idx)
        
        # Check if any key level exists
        has_any = any([
            ohp is not None,
            olp is not None,
            len(fvg_list) > 0,
            len(ob_list) > 0,
            rb is not None
        ])
        
        return {
            'ohp': ohp,
            'olp': olp,
            'fvg': fvg_list,
            'ob': ob_list,
            'rb': rb,
            'has_any_keylevel': has_any,
            'crt_candle_idx': crt_candle_idx
        }
    
    def get_primary_keylevel(
        self, 
        keylevels: Dict
    ) -> Optional[Dict]:
        """
        Get the most relevant key level from detected levels.
        Priority: OHP/OLP > FVG > RB > OB
        
        Args:
            keylevels: Output from detect_all_keylevels()
        
        Returns:
            Single key level dictionary or None
        """
        # Priority 1: OHP/OLP (most common and reliable)
        if keylevels['ohp']:
            return keylevels['ohp']
        if keylevels['olp']:
            return keylevels['olp']
        
        # Priority 2: FVG (common and effective)
        if keylevels['fvg']:
            return keylevels['fvg'][-1]  # Most recent FVG
        
        # Priority 3: RB (rejection zones)
        if keylevels['rb']:
            return keylevels['rb']
        
        # Priority 4: OB (order blocks)
        if keylevels['ob']:
            return keylevels['ob'][-1]  # Most recent OB
        
        return None


# Example usage
if __name__ == "__main__":
    import pandas as pd
    
    # Test with sample data
    sample_data = pd.DataFrame({
        'open': [100, 105, 102, 108, 107, 110, 105, 112],
        'high': [104, 110, 106, 112, 109, 115, 108, 116],
        'low': [99, 104, 100, 107, 105, 109, 103, 110],
        'close': [103, 106, 105, 109, 106, 111, 107, 113]
    })
    
    detector = KeyLevelDetector(config={'ohp_olp_lookback': 5})
    keylevels = detector.detect_all_keylevels(sample_data, crt_candle_idx=7)
    
    print("Detected Key Levels:")
    print(f"OHP: {keylevels['ohp']}")
    print(f"OLP: {keylevels['olp']}")
    print(f"FVG Count: {len(keylevels['fvg'])}")
    print(f"OB Count: {len(keylevels['ob'])}")
    print(f"RB: {keylevels['rb']}")
    print(f"Has Any Key Level: {keylevels['has_any_keylevel']}")
