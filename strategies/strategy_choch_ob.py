"""
Strategy 4: CHOCH (Change of Character) + Order Block + Candlestick
"""
import pandas as pd
from typing import Dict
from strategies.base_strategy import BaseStrategy
from detectors.structure_detector import StructureDetector
from detectors.order_block_detector import OrderBlockDetector
from detectors.retest_detector import RetestDetector

class CHOCHOrderBlockStrategy(BaseStrategy):
    """CHOCH + Order Block Strategy"""
    
    def __init__(self):
        super().__init__(name="CHOCH + Order Block")
        self.structure_detector = StructureDetector()
        self.ob_detector = OrderBlockDetector()
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
        """Analyze for CHOCH + OB setup"""
        
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
        
        # Step 1: Detect current trend
        trend_info = self.structure_detector.detect_trend(df_15min)
        
        if trend_info['trend'] == 'RANGING':
            result['reasoning'].append("No clear trend for CHOCH detection")
            return result
        
        result['reasoning'].append(f"Previous trend: {trend_info['trend']}")
        
        # Step 2: Detect CHOCH (Change of Character)
        choch = self.structure_detector.detect_choch(df_15min)
        
        if not choch:
            result['reasoning'].append("No CHOCH detected")
            return result
        
        result['setup_detected'] = True
        result['reasoning'].append(
            f"{choch['type']} CHOCH - Reversing from {choch['previous_trend']}"
        )
        
        # Step 3: Find Order Block near CHOCH level
        order_blocks = self.ob_detector.detect(df_15min)
        
        # Filter OBs near CHOCH level
        choch_level = choch['broken_level']
        relevant_obs = [
            ob for ob in order_blocks
            if (abs(ob['high'] - choch_level) / choch_level < 0.01 or
                abs(ob['low'] - choch_level) / choch_level < 0.01)
        ]
        
        if not relevant_obs:
            result['reasoning'].append("No Order Block near CHOCH level")
            return result
        
        # Take the strongest OB
        best_ob = max(relevant_obs, key=lambda x: x['strength'])
        result['reasoning'].append(f"Order Block at {best_ob['low']:.2f}-{best_ob['high']:.2f}")
        
        # Step 4: Check for retest
        breaker_block = choch['breaker_block']
        
        retest_result = self.retest_detector.check_retest(
            df=df_5min,
            zone_high=max(breaker_block['high'], best_ob['high']),
            zone_low=min(breaker_block['low'], best_ob['low']),
            expected_direction=choch['type']
        )
        
        result['retest_confirmed'] = retest_result['retest_confirmed']
        result['reasoning'].append(retest_result['reasoning'])
        
        if not retest_result['retest_confirmed']:
            return result
        
        # Step 5: Check candlestick confirmation
        candlestick_boost = self._check_candlestick(df_5min, choch['type'])
        if candlestick_boost['pattern']:
            result['candlestick_pattern'] = candlestick_boost['pattern']
            result['reasoning'].append(f"Candlestick: {candlestick_boost['pattern']}")
        
        # Step 6: Calculate confidence
        base_confidence = 70  # CHOCH is reversal signal
        base_confidence += best_ob['strength'] // 10
        base_confidence += candlestick_boost['confidence_boost']
        
        # CHOCH goes AGAINST overall trend (reversal), so don't penalize
        # But boost if overall trend is already showing reversal signs
        
        result['confidence'] = min(100, base_confidence)
        
        # Step 7: Set signal, stop, target
        if choch['type'] == 'BULLISH':
            result['signal'] = 'CALL'
            result['stop_loss'] = min(breaker_block['low'], best_ob['low']) * 0.998
            result['target'] = resistance
        else:
            result['signal'] = 'PUT'
            result['stop_loss'] = max(breaker_block['high'], best_ob['high']) * 1.002
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
