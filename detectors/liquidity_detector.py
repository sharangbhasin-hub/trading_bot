"""
Liquidity Detector
Detects liquidity sweeps and stop hunts
"""
import pandas as pd
from typing import Dict, List, Optional

class LiquidityDetector:
    """Detects liquidity sweeps and stop hunts"""
    
    def __init__(self):
        self.sweep_tolerance_pct = 0.15  # Price must exceed level by 0.15%
        self.wick_ratio = 0.5  # Rejection wick must be 50%+ of candle range
        self.lookback_candles = 50
    
    def detect_sweep(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Detect liquidity sweep on recent candles
        
        Returns dict if sweep detected:
        {
            'type': 'HIGH_SWEEP' | 'LOW_SWEEP',
            'swept_level': float,
            'sweep_candle_index': int,
            'rejection_confirmed': bool,
            'wick_size_pct': float
        }
        """
        if len(df) < 20:
            return None
        
        # Get recent data
        df_recent = df.tail(self.lookback_candles).reset_index(drop=True)  # NOW uses 50 candles
        
        # Find swing high (last 20 candles)
        swing_high_idx = df_recent['high'].iloc[-20:-1].idxmax()
        swing_high = df_recent['high'].iloc[swing_high_idx]
        
        # Find swing low (last 20 candles)
        swing_low_idx = df_recent['low'].iloc[-20:-1].idxmin()
        swing_low = df_recent['low'].iloc[swing_low_idx]
        
        # Check last 3 candles for sweep
        for i in range(len(df_recent) - 3, len(df_recent)):
            candle = df_recent.iloc[i]
            body_size = abs(candle['close'] - candle['open'])
            total_range = candle['high'] - candle['low']
            
            # Check for high sweep
            if candle['high'] > swing_high * (1 + self.sweep_tolerance_pct / 100):
                # Check for rejection (wick)
                upper_wick = candle['high'] - max(candle['open'], candle['close'])
                rejection_confirmed = (upper_wick > total_range * self.wick_ratio and
                                     candle['close'] < swing_high)
                
                return {
                    'type': 'HIGH_SWEEP',
                    'swept_level': swing_high,
                    'sweep_candle_index': i,
                    'rejection_confirmed': rejection_confirmed,
                    'wick_size_pct': (upper_wick / total_range * 100) if total_range > 0 else 0
                }
            
            # Check for low sweep
            if candle['low'] < swing_low * (1 - self.sweep_tolerance_pct / 100):
                # Check for rejection (wick)
                lower_wick = min(candle['open'], candle['close']) - candle['low']
                rejection_confirmed = (lower_wick > total_range * self.wick_ratio and
                                     candle['close'] > swing_low)
                
                return {
                    'type': 'LOW_SWEEP',
                    'swept_level': swing_low,
                    'sweep_candle_index': i,
                    'rejection_confirmed': rejection_confirmed,
                    'wick_size_pct': (lower_wick / total_range * 100) if total_range > 0 else 0
                }
        
        return None
    
    def find_liquidity_pools(self, df: pd.DataFrame) -> Dict:
        """
        Find major liquidity pools (equal highs/lows)
        
        Returns:
        {
            'high_pools': [float, ...],  # Levels where stops likely above
            'low_pools': [float, ...]    # Levels where stops likely below
        }
        """
        if len(df) < 20:
            return {'high_pools': [], 'low_pools': []}
        
        df_recent = df.tail(50).reset_index(drop=True)
        
        tolerance_pct = 0.1  # Consider levels within 0.1% as "equal"
        
        high_pools = []
        low_pools = []
        
        # Find equal highs (liquidity pools above market)
        highs = df_recent['high'].values
        for i in range(len(highs) - 1):
            for j in range(i + 1, len(highs)):
                if abs(highs[i] - highs[j]) / highs[i] * 100 < tolerance_pct:
                    # Equal highs found
                    level = (highs[i] + highs[j]) / 2
                    if level not in high_pools:
                        high_pools.append(level)
        
        # Find equal lows (liquidity pools below market)
        lows = df_recent['low'].values
        for i in range(len(lows) - 1):
            for j in range(i + 1, len(lows)):
                if abs(lows[i] - lows[j]) / lows[i] * 100 < tolerance_pct:
                    # Equal lows found
                    level = (lows[i] + lows[j]) / 2
                    if level not in low_pools:
                        low_pools.append(level)
        
        return {
            'high_pools': sorted(high_pools, reverse=True)[:5],  # Top 5
            'low_pools': sorted(low_pools)[:5]  # Bottom 5
        }
