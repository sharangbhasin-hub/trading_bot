"""
Strategy 8: FVG + Double Bottom/Top + Breakout
Classic pattern meets modern SMC
"""
import pandas as pd
from typing import Dict, Optional
from strategies.base_strategy import BaseStrategy
from detectors.fvg_detector import FVGDetector
from detectors.retest_detector import RetestDetector

class FVGDoubleBottomTopStrategy(BaseStrategy):
    """FVG + Double Bottom/Top + Breakout Strategy"""
    
    def __init__(self):
        super().__init__(name="FVG + Double Bottom/Top")
        self.fvg_detector = FVGDetector()
        self.retest_detector = RetestDetector()
    
    def analyze(self, 
                df_5min: pd.DataFrame,
                df_15min: pd.DataFrame,
                df_1h: pd.DataFrame,
                df_4h: pd.DataFrame,
                spot_price: float,
                support: float,
                resistance: float,
                overall_trend: str) -> Dict:
        """Analyze for FVG + Double Bottom/Top setup"""
        
        result = {
            'signal': 'NO_TRADE',
            'confidence': 0,
            'entry_price': spot_price,
            'stop_loss': 0
			'target': 0,
            'reasoning': [],
            'setup_detected': False,
            'retest_confirmed': False,
            'candlestick_pattern': None
        }
        
        # Step 1: Detect FVGs on 15min chart
        fvgs = self.fvg_detector.detect(df_15min)
        
        if not fvgs:
            result['reasoning'].append("No FVGs detected")
            return result
        
        # Step 2: Detect double bottom or double top on 15min chart
        pattern = self._detect_double_pattern(df_15min)
        
        if not pattern:
            result['reasoning'].append("No double bottom/top pattern detected")
            return result
        
        result['reasoning'].append(
            f"{pattern['type']} detected at {pattern['level']:.2f}"
        )
        
        # Step 3: Check if FVG overlaps with double bottom/top level
        fvg_at_pattern = self._find_fvg_at_level(fvgs, pattern['level'])
        
        if not fvg_at_pattern:
            result['reasoning'].append("No FVG at double bottom/top level")
            return result
        
        result['setup_detected'] = True
        result['reasoning'].append(
            f"FVG at {fvg_at_pattern['bottom']:.2f}-{fvg_at_pattern['top']:.2f} "
            f"overlaps with pattern"
        )
        
        # Step 4: Check for breakout
        breakout = self._check_breakout(df_15min, pattern)
        
        if not breakout:
            result['reasoning'].append("Pattern formed but no breakout yet")
            return result
        
        result['reasoning'].append(f"Breakout confirmed above {breakout['neckline']:.2f}")
        
        # Step 5: Check for retest of breakout level
        # After breakout, price should retest the neckline
        retest_result = self.retest_detector.check_retest(
            df=df_5min,
            zone_high=breakout['neckline'] * 1.002,
            zone_low=breakout['neckline'] * 0.998,
            expected_direction=pattern['expected_direction']
        )
        
        result['retest_confirmed'] = retest_result['retest_confirmed']
        result['reasoning'].append(retest_result['reasoning'])
        
        if not retest_result['retest_confirmed']:
            return result
        
        # Step 6: Check candlestick confirmation
        candlestick_boost = self._check_candlestick(
            df_5min, 
            pattern['expected_direction']
        )
        if candlestick_boost['pattern']:
            result['candlestick_pattern'] = candlestick_boost['pattern']
            result['reasoning'].append(f"Candlestick: {candlestick_boost['pattern']}")
        
        # Step 7: Calculate confidence
        base_confidence = 70
        
        # Boost for pattern reliability
        base_confidence += pattern['reliability_score']
        
        # Boost for FVG confluence
        base_confidence += 8
        result['reasoning'].append("FVG adds confluence to pattern")
        
        # Boost for candlestick
        base_confidence += candlestick_boost['confidence_boost']
        
        # Alignment with trend
        if ((pattern['expected_direction'] == 'BULLISH' and overall_trend == 'Bullish') or
            (pattern['expected_direction'] == 'BEARISH' and overall_trend == 'Bearish')):
            base_confidence += 10
            result['reasoning'].append("Aligned with overall trend")
        
        result['confidence'] = min(100, base_confidence)
        
        # Step 8: Set signal, stop, target
        if pattern['expected_direction'] == 'BULLISH':
            result['signal'] = 'CALL'
            result['stop_loss'] = pattern['level'] * 0.997  # Below double bottom
            # Target: Measure move (pattern height projected from breakout)
            pattern_height = breakout['neckline'] - pattern['level']
            result['target'] = breakout['neckline'] + pattern_height
        else:
            result['signal'] = 'PUT'
            result['stop_loss'] = pattern['level'] * 1.003  # Above double top
            # Target: Measure move
            pattern_height = pattern['level'] - breakout['neckline']
            result['target'] = breakout['neckline'] - pattern_height
        
        return result
    
    def _detect_double_pattern(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect double bottom or double top"""
        if len(df) < 20:
            return None
        
        recent = df.tail(30).reset_index(drop=True)
        
        # Find swing lows (potential double bottom)
        swing_lows = []
        for i in range(5, len(recent) - 5):
            low = recent['low'].iloc[i]
            if (low < recent['low'].iloc[i-5:i].min() and
                low < recent['low'].iloc[i+1:i+6].min()):
                swing_lows.append({
                    'price': low,
                    'index': i
                })
        
        # Check for double bottom (two lows within 0.3% of each other)
        for i in range(len(swing_lows) - 1):
            for j in range(i + 1, len(swing_lows)):
                low1 = swing_lows[i]['price']
                low2 = swing_lows[j]['price']
                
                if abs(low1 - low2) / low1 * 100 < 0.3:
                    # Double bottom found
                    # Calculate neckline (high between the two lows)
                    start_idx = swing_lows[i]['index']
                    end_idx = swing_lows[j]['index']
                    neckline = recent['high'].iloc[start_idx:end_idx].max()
                    
                    # Calculate reliability based on spacing
                    spacing = end_idx - start_idx
                    reliability = min(12, spacing)  # More spacing = more reliable
                    
                    return {
                        'type': 'DOUBLE_BOTTOM',
                        'level': (low1 + low2) / 2,
                        'neckline': neckline,
                        'expected_direction': 'BULLISH',
                        'reliability_score': reliability
                    }
        
        # Find swing highs (potential double top)
        swing_highs = []
        for i in range(5, len(recent) - 5):
            high = recent['high'].iloc[i]
            if (high > recent['high'].iloc[i-5:i].max() and
                high > recent['high'].iloc[i+1:i+6].max()):
                swing_highs.append({
                    'price': high,
                    'index': i
                })
        
        # Check for double top (two highs within 0.3% of each other)
        for i in range(len(swing_highs) - 1):
            for j in range(i + 1, len(swing_highs)):
                high1 = swing_highs[i]['price']
                high2 = swing_highs[j]['price']
                
                if abs(high1 - high2) / high1 * 100 < 0.3:
                    # Double top found
                    # Calculate neckline (low between the two highs)
                    start_idx = swing_highs[i]['index']
                    end_idx = swing_highs[j]['index']
                    neckline = recent['low'].iloc[start_idx:end_idx].min()
                    
                    spacing = end_idx - start_idx
                    reliability = min(12, spacing)
                    
                    return {
                        'type': 'DOUBLE_TOP',
                        'level': (high1 + high2) / 2,
                        'neckline': neckline,
                        'expected_direction': 'BEARISH',
                        'reliability_score': reliability
                    }
        
        return None
    
    def _find_fvg_at_level(self, fvgs, level):
        """Find FVG that overlaps with pattern level"""
        for fvg in fvgs:
            if fvg['bottom'] <= level <= fvg['top']:
                return fvg
        return None
    
    def _check_breakout(self, df, pattern):
        """Check if breakout occurred"""
        if len(df) < 5:
            return None
        
        recent = df.tail(10)
        current_price = df['close'].iloc[-1]
        
        if pattern['expected_direction'] == 'BULLISH':
            # Need close above neckline
            if current_price > pattern['neckline']:
                return {
                    'confirmed': True,
                    'neckline': pattern['neckline']
                }
        else:  # BEARISH
            # Need close below neckline
            if current_price < pattern['neckline']:
                return {
                    'confirmed': True,
                    'neckline': pattern['neckline']
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
        upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
        
        if direction == 'BULLISH':
            if lower_wick > body * 2:
                return {'pattern': 'Hammer', 'confidence_boost': 15}
            if (last_candle['close'] > last_candle['open'] and
                last_candle['close'] > prev_candle['open']):
                return {'pattern': 'Bullish Engulfing', 'confidence_boost': 15}
            if lower_wick > total_range * 0.5:
                return {'pattern': 'Bullish Rejection', 'confidence_boost': 10}
        else:
            if upper_wick > body * 2:
                return {'pattern': 'Shooting Star', 'confidence_boost': 15}
            if (last_candle['close'] < last_candle['open'] and
                last_candle['close'] < prev_candle['open']):
                return {'pattern': 'Bearish Engulfing', 'confidence_boost': 15}
            if upper_wick > total_range * 0.5:
                return {'pattern': 'Bearish Rejection', 'confidence_boost': 10}
        
        return {'pattern': None, 'confidence_boost': 0}
