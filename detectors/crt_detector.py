"""
CRT (Candle Range Theory) Detector
==================================

Detects thick-bodied CRT candles on Higher Timeframes (HTF) for directional bias.
Based on CRT-TBS trading strategy documentation.

Author: Trading System
Date: October 23, 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

# Setup logger
logger = logging.getLogger(__name__)


class CRTDetector:
    """
    Detects CRT (Candle Range Theory) candles on HTF.
    
    CRT Candle Definition (per documentation):
    - "Thick candle" where body is larger than wicks
    - Body > (upper_wick + lower_wick)
    - Represents high liquidity and momentum
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize CRT Detector.
        
        Args:
            config: Optional configuration dictionary
                   - method: 'body_vs_wicks' or 'ratio'
                   - min_body_ratio: If using ratio method (default 0.50)
        """
        self.config = config or {}
        self.method = self.config.get('method', 'body_vs_wicks')
        self.min_body_ratio = self.config.get('min_body_ratio', 0.50)
    
    def detect_crt_candles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect CRT candles in dataframe.
        
        Args:
            df: OHLC dataframe with columns ['open', 'high', 'low', 'close']
        
        Returns:
            DataFrame with additional columns:
            - is_crt: Boolean flag for CRT candle
            - body_size: Size of candle body
            - body_ratio: Body as percentage of total range
            - crt_direction: 'bullish' or 'bearish'
        """
        df = df.copy()
        
        # Calculate candle components
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['total_range'] = df['high'] - df['low']
        
        # Avoid division by zero
        df['body_ratio'] = np.where(
            df['total_range'] > 0,
            df['body_size'] / df['total_range'],
            0
        )
        
        # CRT detection based on method
        if self.method == 'body_vs_wicks':
            # Document-specified: body > wicks
            df['is_crt'] = df['body_size'] > (df['upper_wick'] + df['lower_wick'])
        else:
            # Alternative: ratio method
            df['is_crt'] = df['body_ratio'] >= self.min_body_ratio
        
        # Determine direction
        df['crt_direction'] = np.where(
            df['close'] > df['open'],
            'bullish',
            'bearish'
        )
        
        return df

    def detect_inside_bar_crt(
        self,
        df: pd.DataFrame,
        max_inside_candles: int = 4
    ) -> List[Dict]:
        """
        Detect Inside Bar CRT Pattern.
        
        Pattern:
        - Candle 1: Thick CRT candle
        - Candles 2-N: Stay inside Candle 1's range
        - Final candle: Breaks out and purges
        
        Args:
            df: OHLC DataFrame
            max_inside_candles: Maximum candles allowed inside (default 4)
        
        Returns:
            List of inside bar CRT patterns
        """
        inside_bar_crts = []
        
        i = 0
        while i < len(df) - 2:  # Need at least 3 candles
            # Check if candle i is a thick CRT candle
            candle_i = df.iloc[i]
            
            # Check if it's a thick candle
            body_size = abs(candle_i['close'] - candle_i['open'])
            upper_wick = candle_i['high'] - max(candle_i['open'], candle_i['close'])
            lower_wick = min(candle_i['open'], candle_i['close']) - candle_i['low']
            
            is_thick = body_size > (upper_wick + lower_wick)
            
            if not is_thick:
                i += 1
                continue
            
            # Now check if next candles stay inside
            crt_high = candle_i['high']
            crt_low = candle_i['low']
            crt_close = candle_i['close']
            
            # Count how many consecutive candles stay inside
            inside_count = 0
            j = i + 1
            
            while j < len(df) and inside_count < max_inside_candles:
                inside_candle = df.iloc[j]
                
                # Check if candle stays inside
                if inside_candle['high'] <= crt_high and inside_candle['low'] >= crt_low:
                    inside_count += 1
                    j += 1
                else:
                    # Candle broke out - this is the purge candle
                    break
            
            # If we have at least 1 inside candle AND a breakout candle
            if inside_count >= 1 and j < len(df):
                breakout_candle = df.iloc[j]
                
                # Check if breakout is valid (purged above or below)
                bullish_purge = (crt_close > crt_high * 0.99 and  # CRT closed near high
                                breakout_candle['high'] > crt_high and  # Breakout went above
                                breakout_candle['close'] < crt_high)    # Then closed back inside
                
                bearish_purge = (crt_close < crt_low * 1.01 and  # CRT closed near low
                                breakout_candle['low'] < crt_low and  # Breakout went below
                                breakout_candle['close'] > crt_low)    # Then closed back inside
                
                if bullish_purge or bearish_purge:
                    direction = 'SELL' if bullish_purge else 'BUY'
                    
                    inside_bar_crts.append({
                        'type': 'INSIDE_BAR_CRT',
                        'direction': direction,
                        'crt_candle_index': i,
                        'inside_candles': inside_count,
                        'purge_candle_index': j,
                        'crt_high': crt_high,
                        'crt_low': crt_low,
                        'crt_range': crt_high - crt_low,
                        'timestamp': df.index[j],
                        'pattern_quality': 'A+' if 2 <= inside_count <= 6 else 'B'
                    })
                    
                    # Skip past this pattern
                    i = j + 1
                    continue
            
            i += 1
        
        return inside_bar_crts

    def detect(self, df: pd.DataFrame, detect_inside_bar: bool = True, **kwargs) -> List[Dict]:
        """
        Main unified CRT detection method.
        Detects both standard CRT candles and inside bar patterns.
        
        Args:
            df: OHLC DataFrame
            detect_inside_bar: Whether to also detect inside bar patterns (default True)
        
        Returns:
            List of CRT pattern dictionaries with keys:
            - type: 'STANDARD_CRT' or 'INSIDE_BAR_CRT'
            - direction: 'BUY' or 'SELL'
            - crt_high: High of CRT range
            - crt_low: Low of CRT range
            - crt_range: Size of CRT range
            - timestamp: Datetime of pattern completion
            - crt_candle_index: Index of first candle in pattern
            - pattern_quality: 'A+' or 'B' (for inside bar patterns)
        """
        all_crts = []
        
        # 1. Detect standard CRT candles
        df_with_crt = self.detect_crt_candles(df)
        
        for idx, row in df_with_crt[df_with_crt['is_crt']].iterrows():
            # Convert standard CRT to dictionary format
            direction = 'SELL' if row['crt_direction'] == 'bearish' else 'BUY'
            
            all_crts.append({
                'type': 'STANDARD_CRT',
                'direction': direction,
                'crt_candle_index': df.index.get_loc(idx),
                'crt_high': row['high'],
                'crt_low': row['low'],
                'crt_range': row['total_range'],
                'timestamp': idx,
                'body_ratio': row['body_ratio'],
                'pattern_quality': 'A+' if row['body_ratio'] >= 0.60 else 'B'
            })
        
        # 2. Optionally detect inside bar CRTs
        if detect_inside_bar:
            inside_bar_crts = self.detect_inside_bar_crt(df)
            all_crts.extend(inside_bar_crts)
        
        # 3. Sort by timestamp
        all_crts = sorted(all_crts, key=lambda x: x['timestamp'])
        
        return all_crts
    
    def get_crt_candles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get only CRT candles from dataframe.
        
        Args:
            df: OHLC dataframe
        
        Returns:
            Filtered dataframe containing only CRT candles
        """
        df_with_crt = self.detect_crt_candles(df)
        return df_with_crt[df_with_crt['is_crt']].copy()
    
    def calculate_50_percent_level(
        self, 
        crt_high: float, 
        crt_low: float
    ) -> float:
        """
        Calculate 50% retracement level of CRT candle range.
        This is TP-1 target per documentation.
        
        Args:
            crt_high: High of CRT candle
            crt_low: Low of CRT candle
        
        Returns:
            Exact 50% level (midpoint)
        """
        return crt_low + ((crt_high - crt_low) / 2)
    
    def get_crt_levels(
        self, 
        crt_candle: pd.Series
    ) -> Dict[str, float]:
        """
        Extract all relevant levels from a CRT candle.
        
        Args:
            crt_candle: Series representing single CRT candle row
        
        Returns:
            Dictionary with:
            - crt_high: High of candle
            - crt_low: Low of candle
            - tp1_level: 50% retracement
            - tp2_sell: CRT low (for sell trades)
            - tp2_buy: CRT high (for buy trades)
        """
        high = crt_candle['high']
        low = crt_candle['low']
        tp1 = self.calculate_50_percent_level(high, low)
        
        return {
            'crt_high': high,
            'crt_low': low,
            'tp1_level': tp1,
            'tp2_sell': low,  # 100% target for sell
            'tp2_buy': high,  # 100% target for buy
            'direction': crt_candle['crt_direction'],
            'body_ratio': crt_candle['body_ratio'],
            'timestamp': crt_candle.name if hasattr(crt_candle, 'name') else None
        }
    
    def is_valid_crt_candle(
        self, 
        candle: pd.Series, 
        min_range_pips: Optional[float] = None
    ) -> bool:
        """
        Additional validation for CRT candle quality.
        
        Args:
            candle: Single candle series
            min_range_pips: Optional minimum range requirement
        
        Returns:
            Boolean indicating if candle meets quality criteria
        """
        # Must be CRT candle
        if not candle.get('is_crt', False):
            return False
        
        # Optional: minimum range filter
        if min_range_pips is not None:
            total_range = candle['high'] - candle['low']
            if total_range < min_range_pips:
                return False
        
        # Must have substantial body (not doji)
        if candle['body_size'] == 0:
            return False
        
        return True
    
    def find_most_recent_crt(
        self, 
        df: pd.DataFrame, 
        lookback: Optional[int] = None
    ) -> Optional[Tuple[int, pd.Series]]:
        """
        Find most recent valid CRT candle.
        
        Args:
            df: OHLC dataframe with CRT detection already applied
            lookback: Optional lookback period (None = search all)
        
        Returns:
            Tuple of (index, candle_series) or None if not found
        """
        df_search = df.tail(lookback) if lookback else df
        
        crt_candles = df_search[df_search['is_crt']]
        
        if crt_candles.empty:
            return None
        
        # Return most recent
        most_recent_idx = crt_candles.index[-1]
        return (most_recent_idx, crt_candles.iloc[-1])
    
    def get_crt_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate statistics about CRT candles in dataset.
        Useful for strategy optimization and validation.
        
        Args:
            df: Dataframe with CRT detection applied
        
        Returns:
            Dictionary with CRT statistics
        """
        df_with_crt = self.detect_crt_candles(df)
        crt_candles = df_with_crt[df_with_crt['is_crt']]
        
        if crt_candles.empty:
            return {
                'total_candles': len(df),
                'crt_count': 0,
                'crt_percentage': 0.0,
                'avg_body_ratio': 0.0,
                'bullish_count': 0,
                'bearish_count': 0
            }
        
        return {
            'total_candles': len(df),
            'crt_count': len(crt_candles),
            'crt_percentage': (len(crt_candles) / len(df)) * 100,
            'avg_body_ratio': crt_candles['body_ratio'].mean(),
            'max_body_ratio': crt_candles['body_ratio'].max(),
            'min_body_ratio': crt_candles['body_ratio'].min(),
            'bullish_count': len(crt_candles[crt_candles['crt_direction'] == 'bullish']),
            'bearish_count': len(crt_candles[crt_candles['crt_direction'] == 'bearish']),
            'avg_range': crt_candles['total_range'].mean()
        }


# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    import pandas as pd
    
    # Sample OHLC data
    sample_data = pd.DataFrame({
        'open': [100, 105, 102, 108, 107],
        'high': [104, 110, 106, 112, 109],
        'low': [99, 104, 100, 107, 105],
        'close': [103, 106, 105, 109, 106]
    })
    
    detector = CRTDetector()
    result = detector.detect_crt_candles(sample_data)
    print("CRT Detection Results:")
    print(result[['open', 'high', 'low', 'close', 'is_crt', 'body_ratio', 'crt_direction']])
    
    stats = detector.get_crt_statistics(sample_data)
    print("\nCRT Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
