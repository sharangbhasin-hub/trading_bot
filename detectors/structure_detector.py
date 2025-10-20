"""
Structure Detector
Detects market structure: swing highs/lows, BOS, CHOCH
"""
import pandas as pd
from typing import Dict, List, Optional

class StructureDetector:
    """Detects market structure and breaks"""
    
    def __init__(self):
        self.swing_lookback = 5  # Candles on each side for swing detection
    
    def detect_swings(self, df: pd.DataFrame) -> Dict:
        """
        Detect swing highs and swing lows
        
        Returns:
        {
            'swing_highs': [(price, index, timestamp), ...],
            'swing_lows': [(price, index, timestamp), ...]
        }
        """
        swing_highs = []
        swing_lows = []
        
        if len(df) < self.swing_lookback * 2 + 1:
            return {'swing_highs': [], 'swing_lows': []}
        
        df_reset = df.reset_index(drop=True)
        
        for i in range(self.swing_lookback, len(df_reset) - self.swing_lookback):
            # Check for swing high
            high = df_reset['high'].iloc[i]
            if (high > df_reset['high'].iloc[i-self.swing_lookback:i].max() and
                high > df_reset['high'].iloc[i+1:i+self.swing_lookback+1].max()):
                swing_highs.append({
                    'price': high,
                    'index': i,
                    'timestamp': df_reset['timestamp'].iloc[i]
                })
            
            # Check for swing low
            low = df_reset['low'].iloc[i]
            if (low < df_reset['low'].iloc[i-self.swing_lookback:i].min() and
                low < df_reset['low'].iloc[i+1:i+self.swing_lookback+1].min()):
                swing_lows.append({
                    'price': low,
                    'index': i,
                    'timestamp': df_reset['timestamp'].iloc[i]
                })
        
        return {
            'swing_highs': swing_highs,
            'swing_lows': swing_lows
        }
    
    def detect_trend(self, df: pd.DataFrame) -> Dict:
        """
        Determine current trend based on structure
        
        Returns:
        {
            'trend': 'UPTREND' | 'DOWNTREND' | 'RANGING',
            'structure_type': 'HH_HL' | 'LH_LL' | 'CHOPPY',
            'reasoning': str
        }
        """
        swings = self.detect_swings(df)
        
        if len(swings['swing_highs']) < 2 or len(swings['swing_lows']) < 2:
            return {
                'trend': 'RANGING',
                'structure_type': 'CHOPPY',
                'reasoning': 'Not enough swings to determine trend'
            }
        
        # Get last 3 swing highs and lows
        recent_highs = swings['swing_highs'][-3:]
        recent_lows = swings['swing_lows'][-3:]
        
        # Check for Higher Highs and Higher Lows (uptrend)
        higher_highs = all(recent_highs[i]['price'] > recent_highs[i-1]['price'] 
                          for i in range(1, len(recent_highs)))
        higher_lows = all(recent_lows[i]['price'] > recent_lows[i-1]['price'] 
                         for i in range(1, len(recent_lows)))
        
        if higher_highs and higher_lows:
            return {
                'trend': 'UPTREND',
                'structure_type': 'HH_HL',
                'reasoning': 'Making Higher Highs and Higher Lows'
            }
        
        # Check for Lower Highs and Lower Lows (downtrend)
        lower_highs = all(recent_highs[i]['price'] < recent_highs[i-1]['price'] 
                         for i in range(1, len(recent_highs)))
        lower_lows = all(recent_lows[i]['price'] < recent_lows[i-1]['price'] 
                        for i in range(1, len(recent_lows)))
        
        if lower_highs and lower_lows:
            return {
                'trend': 'DOWNTREND',
                'structure_type': 'LH_LL',
                'reasoning': 'Making Lower Highs and Lower Lows'
            }
        
        return {
            'trend': 'RANGING',
            'structure_type': 'CHOPPY',
            'reasoning': 'Mixed structure - no clear trend'
        }
    
    def detect_bos(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Detect Break of Structure (BOS)
        
        Returns dict if BOS detected, None otherwise:
        {
            'type': 'BULLISH' | 'BEARISH',
            'broken_level': float,
            'break_candle_index': int,
            'breaker_block': {'high': float, 'low': float}
        }
        """
        trend_info = self.detect_trend(df)
        swings = self.detect_swings(df)
        
        if len(df) < 10:
            return None
        
        current_price = df['close'].iloc[-1]
        
        # Check for bullish BOS (uptrend continues)
        if trend_info['trend'] == 'UPTREND' and swings['swing_highs']:
            # Get previous swing high
            prev_high = swings['swing_highs'][-1]['price']
            
            # Check if current price broke above
            if current_price > prev_high:
                return {
                    'type': 'BULLISH',
                    'broken_level': prev_high,
                    'break_candle_index': len(df) - 1,
                    'breaker_block': {
                        'high': prev_high,
                        'low': swings['swing_lows'][-1]['price'] if swings['swing_lows'] else prev_high * 0.99
                    }
                }
        
        # Check for bearish BOS (downtrend continues)
        elif trend_info['trend'] == 'DOWNTREND' and swings['swing_lows']:
            # Get previous swing low
            prev_low = swings['swing_lows'][-1]['price']
            
            # Check if current price broke below
            if current_price < prev_low:
                return {
                    'type': 'BEARISH',
                    'broken_level': prev_low,
                    'break_candle_index': len(df) - 1,
                    'breaker_block': {
                        'low': prev_low,
                        'high': swings['swing_highs'][-1]['price'] if swings['swing_highs'] else prev_low * 1.01
                    }
                }
        
        return None
    
    def detect_choch(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Detect Change of Character (CHOCH)
        
        Returns dict if CHOCH detected, None otherwise:
        {
            'type': 'BULLISH' | 'BEARISH',
            'broken_level': float,
            'previous_trend': str,
            'breaker_block': {'high': float, 'low': float}
        }
        """
        trend_info = self.detect_trend(df)
        swings = self.detect_swings(df)
        
        if len(df) < 10:
            return None
        
        current_price = df['close'].iloc[-1]
        
        # CHOCH = trend reversal (break AGAINST trend)
        
        # Check for bullish CHOCH (was downtrend, now breaking up)
        if trend_info['trend'] == 'DOWNTREND' and swings['swing_highs']:
            # Get previous swing high (resistance in downtrend)
            prev_high = swings['swing_highs'][-1]['price']
            
            # Check if price broke above (reversal signal)
            if current_price > prev_high:
                return {
                    'type': 'BULLISH',
                    'broken_level': prev_high,
                    'previous_trend': 'DOWNTREND',
                    'breaker_block': {
                        'high': prev_high,
                        'low': swings['swing_lows'][-1]['price'] if swings['swing_lows'] else prev_high * 0.98
                    }
                }
        
        # Check for bearish CHOCH (was uptrend, now breaking down)
        elif trend_info['trend'] == 'UPTREND' and swings['swing_lows']:
            # Get previous swing low (support in uptrend)
            prev_low = swings['swing_lows'][-1]['price']
            
            # Check if price broke below (reversal signal)
            if current_price < prev_low:
                return {
                    'type': 'BEARISH',
                    'broken_level': prev_low,
                    'previous_trend': 'UPTREND',
                    'breaker_block': {
                        'low': prev_low,
                        'high': swings['swing_highs'][-1]['price'] if swings['swing_highs'] else prev_low * 1.02
                    }
                }
        
        return None

    def check_bos_choch_conflict(self, df: pd.DataFrame) -> Dict:
        """
        Check if both BOS and CHOCH are detected (conflict)
        
        Returns:
        {
            'conflict': bool,
            'bos_detected': bool,
            'choch_detected': bool,
            'priority': 'BOS' | 'CHOCH' | None,
            'reasoning': str
        }
        """
        bos = self.detect_bos(df)
        choch = self.detect_choch(df)
        
        if bos and choch:
            # Conflict detected - prioritize based on recency
            bos_recent = bos.get('candle_index', 0)
            choch_recent = choch.get('candle_index', 0)
            
            if choch_recent > bos_recent:
                # CHOCH is more recent - it's a reversal, ignore BOS
                return {
                    'conflict': True,
                    'bos_detected': True,
                    'choch_detected': True,
                    'priority': 'CHOCH',
                    'reasoning': 'CHOCH is more recent - reversal signal takes priority'
                }
            else:
                # BOS is more recent - continuation signal
                return {
                    'conflict': True,
                    'bos_detected': True,
                    'choch_detected': True,
                    'priority': 'BOS',
                    'reasoning': 'BOS is more recent - continuation signal takes priority'
                }
        
        return {
            'conflict': False,
            'bos_detected': bos is not None,
            'choch_detected': choch is not None,
            'priority': None,
            'reasoning': 'No conflict detected'
        }
