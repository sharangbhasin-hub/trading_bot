"""
Strategy 3: BOS (Break of Structure) + Retest + Candlestick
"""
import pandas as pd
from typing import Dict
from strategies.base_strategy import BaseStrategy
from detectors.structure_detector import StructureDetector
from detectors.retest_detector import RetestDetector

class BOSRetestStrategy(BaseStrategy):
    """BOS + Retest Strategy"""
    
    def __init__(self):
        super().__init__(name="BOS + Retest")
        self.structure_detector = StructureDetector()
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
        """Analyze for BOS + Retest setup"""
        
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
        
        # Step 1: Detect trend on 15min
        trend_info = self.structure_detector.detect_trend(df_15min)
        
        if trend_info['trend'] == 'RANGING':
            result['reasoning'].append("No clear trend - ranging market")
            return result
        
        result['reasoning'].append(f"Trend: {trend_info['trend']} ({trend_info['structure_type']})")
        
        # Step 2: Detect BOS (Break of Structure)
        bos = self.structure_detector.detect_bos(df_15min)
        
        if not bos:
            result['reasoning'].append("No BOS detected")
            return result
        
        result['setup_detected'] = True
        result['reasoning'].append(f"{bos['type']} BOS at {bos['broken_level']:.2f}")
        
        # Step 3: Check for retest of breaker block
        breaker_block = bos['breaker_block']
        
        retest_result = self.retest_detector.check_retest(
            df=df_5min,
            zone_high=breaker_block['high'],
            zone_low=breaker_block['low'],
            expected_direction=bos['type']
        )
        
        result['retest_confirmed'] = retest_result['retest_confirmed']
        result['reasoning'].append(retest_result['reasoning'])
        
        if not retest_result['retest_confirmed']:
            return result
        
        # Step 4: Check candlestick confirmation
        candlestick_boost = self._check_candlestick(df_5min, bos['type'])
        if candlestick_boost['pattern']:
            result['candlestick_pattern'] = candlestick_boost['pattern']
            result['reasoning'].append(f"Candlestick: {candlestick_boost['pattern']}")
        
        # Step 5: Calculate confidence
        base_confidence = 72  # BOS is strong signal
        base_confidence += candlestick_boost['confidence_boost']
        
        # Alignment with overall trend
        if ((bos['type'] == 'BULLISH' and overall_trend == 'Bullish') or
            (bos['type'] == 'BEARISH' and overall_trend == 'Bearish')):
            base_confidence += 10
            result['reasoning'].append("Aligned with overall trend")
        
        result['confidence'] = min(100, base_confidence)
        
        # Step 6: Set signal, stop, target
        if bos['type'] == 'BULLISH':
            result['signal'] = 'CALL'
            result['stop_loss'] = breaker_block['low'] * 0.998
            result['target'] = resistance
        else:
            result['signal'] = 'PUT'
            result['stop_loss'] = breaker_block['high'] * 1.002
            result['target'] = support
        
        return result
    
    def _check_candlestick(self, df, direction):
        """Check for candlestick patterns"""
        if len(df) < 3:
            return {'pattern': None, 'confidence_boost': 0}
        
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]
        
        body = abs(last_candle['close'] - last_candle['open'])
        total_range = last_candle['high'] - last_candle['low']
        lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
        upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
        
        if direction == 'BULLISH':
            if lower_wick > body * 2 and upper_wick < body * 0.3:
                return {'pattern': 'Hammer', 'confidence_boost': 15}
            
            if (last_candle['close'] > last_candle['open'] and
                prev_candle['close'] < prev_candle['open'] and
                last_candle['close'] > prev_candle['open']):
                return {'pattern': 'Bullish Engulfing', 'confidence_boost': 15}
            
            if lower_wick > total_range * 0.5:
                return {'pattern': 'Bullish Rejection', 'confidence_boost': 10}
        
        else:  # BEARISH
            if upper_wick > body * 2 and lower_wick < body * 0.3:
                return {'pattern': 'Shooting Star', 'confidence_boost': 15}
            
            if (last_candle['close'] < last_candle['open'] and
                prev_candle['close'] > prev_candle['open'] and
                last_candle['close'] < prev_candle['open']):
                return {'pattern': 'Bearish Engulfing', 'confidence_boost': 15}
            
            if upper_wick > total_range * 0.5:
                return {'pattern': 'Bearish Rejection', 'confidence_boost': 10}
        return {'pattern': None, 'confidence_boost': 0}
