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
