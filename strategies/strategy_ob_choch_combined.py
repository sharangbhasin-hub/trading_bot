"""
Strategy 7: OB + CHOCH Combined
When CHOCH occurs, old OBs flip polarity
"""
import pandas as pd
from typing import Dict, List
from strategies.base_strategy import BaseStrategy
from detectors.structure_detector import StructureDetector
from detectors.order_block_detector import OrderBlockDetector
from detectors.retest_detector import RetestDetector

class OBCHOCHCombinedStrategy(BaseStrategy):
    """Order Block + CHOCH Combined Strategy"""
    
    def __init__(self):
        super().__init__(name="OB + CHOCH Combined")
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
        """Analyze for OB + CHOCH combined setup"""
        
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
        
        # Step 1: Detect CHOCH on 15min (reversal signal)
        choch = self.structure_detector.detect_choch(df_15min)
        
        if not choch:
            result['reasoning'].append("No CHOCH detected")
            return result
        
        result['reasoning'].append(
            f"{choch['type']} CHOCH detected - potential reversal"
        )
        
        # Step 2: Find old Order Blocks that will flip polarity
        order_blocks = self.ob_detector.detect(df_15min)
        
        if not order_blocks:
            result['reasoning'].append("No Order Blocks found")
            return result
        
        # Find OBs from BEFORE the CHOCH that are now flipped
        flipped_ob = self._find_flipped_ob(order_blocks, choch)
        
        if not flipped_ob:
            result['reasoning'].append("No flipped Order Block found after CHOCH")
            return result
        
        result['setup_detected'] = True
        result['reasoning'].append(
            f"Flipped OB found at {flipped_ob['low']:.2f} - {flipped_ob['high']:.2f} "
            f"(was {flipped_ob['original_type']}, now {flipped_ob['new_type']})"
        )
        
        # Step 3: Check for retest on 5min
        retest_result = self.retest_detector.check_retest(
            df=df_5min,
            zone_high=flipped_ob['high'],
            zone_low=flipped_ob['low'],
            expected_direction=flipped_ob['new_type']
        )
        
        result['retest_confirmed'] = retest_result['retest_confirmed']
        result['reasoning'].append(retest_result['reasoning'])
        
        if not retest_result['retest_confirmed']:
            return result
        
        # Step 4: Candlestick confirmation
        candlestick_boost = self._check_candlestick(df_5min, flipped_ob['new_type'])
        
        if candlestick_boost['pattern']:
            result['candlestick_pattern'] = candlestick_boost['pattern']
            result['reasoning'].append(f"Candlestick: {candlestick_boost['pattern']}")
        
        # Step 5: Calculate confidence
        base_confidence = 70
        base_confidence += flipped_ob['strength'] // 10
        base_confidence += candlestick_boost['confidence_boost']
        
        result['confidence'] = min(100, base_confidence)
        
        # Step 6: Set signal, dynamic stop loss, target
        if flipped_ob['new_type'] == 'BULLISH':
            result['signal'] = 'CALL'
            result['stop_loss'] = self.calculate_dynamic_stop_loss(
                zone_low=flipped_ob['low'],
                zone_high=flipped_ob['high'],
                direction='BULLISH',
                spot_price=spot_price
            )
            result['target'] = resistance
        else:
            result['signal'] = 'PUT'
            result['stop_loss'] = self.calculate_dynamic_stop_loss(
                zone_low=flipped_ob['low'],
                zone_high=flipped_ob['high'],
                direction='BEARISH',
                spot_price=spot_price
            )
            result['target'] = support
        
        # Step 7: Validate Risk:Reward Ratio
        result = self.validate_risk_reward(result)
        
        return result
    
    def _find_flipped_ob(self, order_blocks, choch):
        """Find OB that flipped polarity after CHOCH"""
        # After CHOCH, old bullish OBs become bearish resistance
        # and old bearish OBs become bullish support
        
        if choch['type'] == 'BULLISH':
            # Look for old BEARISH OBs that now act as BULLISH support
            for ob in order_blocks:
                if ob['type'] == 'BEARISH':
                    return {
                        'high': ob['high'],
                        'low': ob['low'],
                        'original_type': 'BEARISH',
                        'new_type': 'BULLISH',
                        'strength': ob['strength']
                    }
        
        else:  # BEARISH CHOCH
            # Look for old BULLISH OBs that now act as BEARISH resistance
            for ob in order_blocks:
                if ob['type'] == 'BULLISH':
                    return {
                        'high': ob['high'],
                        'low': ob['low'],
                        'original_type': 'BULLISH',
                        'new_type': 'BEARISH',
                        'strength': ob['strength']
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
        lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
        upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
        
        if direction == 'BULLISH':
            if lower_wick > body * 2 and upper_wick < body * 0.3:
                return {'pattern': 'Hammer', 'confidence_boost': 15}
            
            if (last_candle['close'] > last_candle['open'] and
                prev_candle['close'] < prev_candle['open'] and
                last_candle['open'] < prev_candle['close'] and
                last_candle['close'] > prev_candle['open']):
                return {'pattern': 'Bullish Engulfing', 'confidence_boost': 15}
            
            if lower_wick > total_range * 0.5:
                return {'pattern': 'Bullish Rejection', 'confidence_boost': 10}
        
        else:  # BEARISH
            if upper_wick > body * 2 and lower_wick < body * 0.3:
                return {'pattern': 'Shooting Star', 'confidence_boost': 15}
            
            if (last_candle['close'] < last_candle['open'] and
                prev_candle['close'] > prev_candle['open'] and
                last_candle['open'] > prev_candle['close'] and
                last_candle['close'] < prev_candle['open']):
                return {'pattern': 'Bearish Engulfing', 'confidence_boost': 15}
            
            if upper_wick > total_range * 0.5:
                return {'pattern': 'Bearish Rejection', 'confidence_boost': 10}
        
        return {'pattern': None, 'confidence_boost': 0}
