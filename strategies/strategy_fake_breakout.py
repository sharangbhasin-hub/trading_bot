"""
Strategy: Discounted Zone + Fake Breakout
Turtle soup variant - false breakdown that reverses
"""
import pandas as pd
from typing import Dict, Optional
from strategies.base_strategy import BaseStrategy
from detectors.retest_detector import RetestDetector

class FakeBreakoutStrategy(BaseStrategy):
    """Discounted Zone + Fake Breakout Strategy"""
    
    def __init__(self):
        super().__init__(name="Fake Breakout")
        self.retest_detector = RetestDetector()
        self.min_confidence = 70
    
    def analyze(self, 
                df_5min: pd.DataFrame,
                df_15min: pd.DataFrame,
                df_1h: pd.DataFrame,
                df_4h: pd.DataFrame,
                spot_price: float,
                support: float,
                resistance: float,
                overall_trend: str) -> Dict:
        """Analyze for fake breakout setup"""
        
        result = {
            'signal': 'NO_TRADE',
            'confidence': 0,
            'entry_price': spot_price,
            'stop_loss': 0,
            'target': 0,
            'reasoning': [],
            'setup_detected': False,
            'retest_confirmed': False,
            'candlestick_pattern': None
        }
        
        # Step 1: Identify support zone (discounted zone)
        support_zone = self._find_support_zone(df_15min)
        
        if not support_zone:
            result['reasoning'].append("No clear support zone identified")
            return result
        
        result['reasoning'].append(
            f"Support zone at {support_zone['low']:.2f}-{support_zone['high']:.2f}"
        )
        
        # Step 2: Detect fake breakdown on 5min chart
        fake_breakdown = self._detect_fake_breakdown(df_5min, support_zone)
        
        if not fake_breakdown:
            result['reasoning'].append("No fake breakdown detected")
            return result
        
        result['setup_detected'] = True
        result['reasoning'].append(
            f"Fake breakdown at {fake_breakdown['low_wick']:.2f}, "
            f"closed back above support"
        )
        
        # Step 3: Check for retest of support from above
        retest_result = self.retest_detector.check_retest(
            df=df_5min,
            zone_high=support_zone['high'],
            zone_low=support_zone['low'],
            expected_direction='BULLISH'  # Expecting bounce
        )
        
        result['retest_confirmed'] = retest_result['retest_confirmed']
        result['reasoning'].append(retest_result['reasoning'])
        
        if not retest_result['retest_confirmed']:
            return result
        
        # Step 4: Check candlestick confirmation
        candlestick_boost = self._check_candlestick(df_5min, 'BULLISH')
        if candlestick_boost['pattern']:
            result['candlestick_pattern'] = candlestick_boost['pattern']
            result['reasoning'].append(f"Candlestick: {candlestick_boost['pattern']}")
        
        # Step 5: Calculate confidence
        base_confidence = 72
        
        # Boost for strong wick rejection
        wick_strength = fake_breakdown['wick_pct']
        base_confidence += min(10, wick_strength // 10)
        
        # Boost for candlestick
        base_confidence += candlestick_boost['confidence_boost']
        
        # Alignment with trend
        if overall_trend == 'Bullish':
            base_confidence += 10
            result['reasoning'].append("Aligned with bullish trend")
        
        result['confidence'] = min(100, base_confidence)
        
        # Step 6: Set signal, stop, target
        result['signal'] = 'CALL'
        result['stop_loss'] = support_zone['low'] * 0.997  # Below support
        result['target'] = resistance  # Previous high
        
        return result
    
    def _find_support_zone(self, df: pd.DataFrame) -> Optional[Dict]:
        """Find support zone using swing lows"""
        if len(df) < 20:
            return None
        
        recent = df.tail(30)
        
        # Find swing lows
        swing_lows = []
        for i in range(5, len(recent) - 5):
            low = recent['low'].iloc[i]
            if (low < recent['low'].iloc[i-5:i].min() and
                low < recent['low'].iloc[i+1:i+6].min()):
                swing_lows.append(low)
        
        if not swing_lows:
            return None
        
        # Get most recent swing low
        support_level = swing_lows[-1]
        
        return {
            'low': support_level * 0.998,
            'high': support_level * 1.002
        }
    
    def _detect_fake_breakdown(self, df: pd.DataFrame, support_zone) -> Optional[Dict]:
        """Detect fake breakdown - wick below support, close above"""
        if len(df) < 5:
            return None
        
        recent = df.tail(10)
        
        for i in range(len(recent) - 3, len(recent)):
            candle = recent.iloc[i]
            
            # Check if wick went below support
            if candle['low'] < support_zone['low']:
                # Check if close is back above support
                if candle['close'] > support_zone['high']:
                    # This is fake breakdown
                    lower_wick = min(candle['open'], candle['close']) - candle['low']
                    total_range = candle['high'] - candle['low']
                    
                    wick_pct = (lower_wick / total_range * 100) if total_range > 0 else 0
                    
                    if wick_pct > 40:  # At least 40% wick
                        return {
                            'low_wick': candle['low'],
                            'close': candle['close'],
                            'wick_pct': wick_pct
                        }
        
        return None
    
    def _check_candlestick(self, df, direction):
        """Check for candlestick patterns"""
        if len(df) < 3:
            return {'pattern': None, 'confidence_boost': 0}
        
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]
        
        body = abs(last_candle['close'] - last_candle['open'])
        total_range = last_candle['high'] - last_candle['low']
        
        if total_range == 0:
            return {'pattern': None, 'confidence_boost': 0}
        
        lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
        
        if lower_wick > body * 2:
            return {'pattern': 'Hammer', 'confidence_boost': 15}
        
        if (last_candle['close'] > last_candle['open'] and
            last_candle['close'] > prev_candle['open']):
            return {'pattern': 'Bullish Engulfing', 'confidence_boost': 15}
        
        if lower_wick > total_range * 0.5:
            return {'pattern': 'Bullish Rejection', 'confidence_boost': 10}
        
        return {'pattern': None, 'confidence_boost': 0}
