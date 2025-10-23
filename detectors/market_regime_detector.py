"""
Market Regime Detector
Identifies market state: TRENDING, RANGING, VOLATILE
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple

class MarketRegimeDetector:
    """
    Detects market regime using multiple indicators:
    - ADX for trend strength
    - ATR for volatility
    - Price action patterns
    """
    
    def __init__(self, 
                 adx_period: int = 14,
                 atr_period: int = 14,
                 trending_adx_threshold: float = 25.0,
                 ranging_adx_threshold: float = 20.0,
                 high_volatility_threshold: float = 1.5):
        """
        Initialize market regime detector
        
        Args:
            adx_period: Period for ADX calculation
            atr_period: Period for ATR calculation
            trending_adx_threshold: ADX above this = trending
            ranging_adx_threshold: ADX below this = ranging
            high_volatility_threshold: ATR multiplier for high volatility
        """
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.trending_adx_threshold = trending_adx_threshold
        self.ranging_adx_threshold = ranging_adx_threshold
        self.high_volatility_threshold = high_volatility_threshold
    
    def calculate_adx(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Average Directional Index (ADX)
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Series with ADX values
        """
        # Calculate True Range
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        high_diff = df['high'] - df['high'].shift(1)
        low_diff = df['low'].shift(1) - df['low']
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        # Smooth the values
        atr = tr.rolling(window=self.adx_period).mean()
        plus_di = 100 * (plus_dm.rolling(window=self.adx_period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=self.adx_period).mean() / atr)
        
        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=self.adx_period).mean()
        
        return adx
    
    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Average True Range (ATR)
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Series with ATR values
        """
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()
        
        return atr
    
    def detect_regime(self, df: pd.DataFrame, current_idx: int) -> Dict:
        """
        Detect current market regime
        
        Args:
            df: DataFrame with OHLC data
            current_idx: Current bar index
            
        Returns:
            Dict with regime info:
            {
                'regime': 'TRENDING' | 'RANGING' | 'VOLATILE',
                'adx': float,
                'atr': float,
                'atr_normalized': float,
                'trend_strength': 'STRONG' | 'MODERATE' | 'WEAK',
                'volatility': 'HIGH' | 'NORMAL' | 'LOW'
            }
        """
        # Need enough data for calculation
        lookback = max(self.adx_period, self.atr_period) * 2
        if current_idx < lookback:
            return {
                'regime': 'UNKNOWN',
                'adx': 0.0,
                'atr': 0.0,
                'atr_normalized': 0.0,
                'trend_strength': 'UNKNOWN',
                'volatility': 'UNKNOWN'
            }
        
        # Get data slice
        data_slice = df.iloc[:current_idx+1].tail(lookback * 2)
        
        # Calculate indicators
        adx = self.calculate_adx(data_slice).iloc[-1]
        atr = self.calculate_atr(data_slice).iloc[-1]
        
        # Normalize ATR as percentage of price
        current_price = df.iloc[current_idx]['close']
        atr_normalized = (atr / current_price) * 100
        
        # Calculate historical ATR average for comparison
        atr_series = self.calculate_atr(data_slice)
        atr_avg = atr_series.mean()
        atr_ratio = atr / atr_avg if atr_avg > 0 else 1.0
        
        # Determine regime
        if pd.isna(adx) or pd.isna(atr):
            regime = 'UNKNOWN'
            trend_strength = 'UNKNOWN'
            volatility = 'UNKNOWN'
        else:
            # Trend strength classification
            if adx >= self.trending_adx_threshold:
                trend_strength = 'STRONG'
            elif adx >= self.ranging_adx_threshold:
                trend_strength = 'MODERATE'
            else:
                trend_strength = 'WEAK'
            
            # Volatility classification
            if atr_ratio >= self.high_volatility_threshold:
                volatility = 'HIGH'
            elif atr_ratio >= 1.0:
                volatility = 'NORMAL'
            else:
                volatility = 'LOW'
            
            # Primary regime determination
            if volatility == 'HIGH':
                regime = 'VOLATILE'
            elif adx >= self.trending_adx_threshold:
                regime = 'TRENDING'
            elif adx <= self.ranging_adx_threshold:
                regime = 'RANGING'
            else:
                # Transitional state - default to trending
                regime = 'TRENDING'
        
        return {
            'regime': regime,
            'adx': float(adx) if not pd.isna(adx) else 0.0,
            'atr': float(atr) if not pd.isna(atr) else 0.0,
            'atr_normalized': float(atr_normalized) if not pd.isna(atr_normalized) else 0.0,
            'trend_strength': trend_strength,
            'volatility': volatility,
            'atr_ratio': float(atr_ratio)
        }
    
    def should_trade_in_regime(self, regime_info: Dict, strategy_type: str) -> Tuple[bool, str]:
        """
        Determine if strategy should trade in current regime
        
        Args:
            regime_info: Output from detect_regime()
            strategy_type: Type of strategy (TREND_FOLLOWING, REVERSAL, etc.)
            
        Returns:
            Tuple of (should_trade: bool, reason: str)
        """
        regime = regime_info['regime']
        
        # Trend following strategies
        if strategy_type in ['TREND_FOLLOWING', 'BOS_RETEST', 'CHOCH_OB']:
            if regime == 'TRENDING':
                return (True, f"Trending market (ADX={regime_info['adx']:.1f})")
            elif regime == 'RANGING':
                return (False, f"Ranging market not suitable for trend strategies (ADX={regime_info['adx']:.1f})")
            elif regime == 'VOLATILE':
                return (False, f"Too volatile (ATR ratio={regime_info['atr_ratio']:.2f})")
            else:
                return (False, "Unknown market regime")
        
        # Reversal/range strategies
        elif strategy_type in ['REVERSAL', 'LIQUIDITY_SWEEP', 'FAKE_BREAKOUT']:
            if regime == 'RANGING':
                return (True, f"Ranging market suitable for reversals (ADX={regime_info['adx']:.1f})")
            elif regime == 'TRENDING':
                # Can trade reversals in strong trends at key levels
                if regime_info['trend_strength'] == 'STRONG':
                    return (True, f"Strong trend - counter-trend at key levels (ADX={regime_info['adx']:.1f})")
                else:
                    return (False, f"Moderate trend - wait for clearer range (ADX={regime_info['adx']:.1f})")
            elif regime == 'VOLATILE':
                return (False, f"Too volatile for reversals (ATR ratio={regime_info['atr_ratio']:.2f})")
            else:
                return (False, "Unknown market regime")
        
        # Breakout strategies
        elif strategy_type in ['BREAKOUT', 'FVG_DOUBLE']:
            if regime == 'RANGING':
                return (True, f"Ranging market - potential breakout setup (ADX={regime_info['adx']:.1f})")
            elif regime == 'TRENDING' and regime_info['trend_strength'] == 'STRONG':
                return (True, f"Strong trend - continuation breakouts (ADX={regime_info['adx']:.1f})")
            elif regime == 'VOLATILE':
                return (False, f"Too volatile - false breakouts likely (ATR ratio={regime_info['atr_ratio']:.2f})")
            else:
                return (False, "Market regime not suitable for breakouts")
        
        # Default: allow trading but log warning
        return (True, f"No regime filter for strategy type: {strategy_type}")
