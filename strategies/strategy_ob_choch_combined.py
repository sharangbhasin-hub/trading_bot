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
        
        # Step 1: Detect CHOCH on 15min
        choch = self.structure_detector.detect_choch(df_15min)
        
        if not choch:
            result['reasoning'].append("No CHOCH detected")
            return result
        
        result['reasoning'].append(
            f"{choch['type']} CHOCH - Reversing from {choch['previous_trend']}"
        )
        
        # Step 2: Find Order Blocks that formed BEFORE the CHOCH
        all_obs = self.ob_detector.detect(df_15min)
        
        if not all_obs:
            result['reasoning'].append("No order blocks detected")
            return result
        
        # Step 3: Filter OBs from the OLD trend (before CHOCH)
        # These OBs will now flip polarity
        if choch['previous_trend'] == 'UPTREND':
            # Was uptrend, now bearish CHOCH
            # Old bullish OBs become resistance
            old_trend_obs = [ob for ob in all_obs if ob['type'] == 'BULLISH']
            flipped_type = 'BEARISH'  # OB now acts bearish (resistance)
        else:
            # Was downtrend, now bullish CHOCH
            # Old bearish OBs become support
            old_trend_obs = [ob for ob in all_obs if ob['type'] == 'BEARISH']
            flipped_type = 'BULLISH'  # OB now acts bullish (support)
        
        if not old_trend_obs:
            result['reasoning'].append("No old trend order blocks found")
            return result
        
        # Find OB closest to current price
        nearest_ob = min(
            old_trend_obs,
            key=lambda ob: min(
                abs(ob['high'] - spot_price),
                abs(ob['low'] - spot_price)
            )
        )
        
        # Check if price is approaching this OB
        distance_pct = min(
            abs(nearest_ob['high'] - spot_price) / spot_price * 100,
            abs(nearest_ob['low'] - spot_price) / spot_price * 100
        )
        
        if distance_pct > 2.0:
            result['reasoning'].append(
                f"Nearest flipped OB is {distance_pct:.2f}% away - too far"
            )
            return result
        
        result['setup_detected'] = True
        result['reasoning'].append(
            f"Old {nearest_ob['type']} OB now flipped to {flipped_type} "
            f"at {nearest_ob['low']:.2f}-{nearest_ob['high']:.2f}"
        )
        
        # Step 4: Check for retest of flipped OB
        retest_result = self.retest_detector.check_retest(
            df=df_5min,
            zone_high=nearest_ob['high'],
            zone_low=nearest_ob['low'],
            expected_direction=choch['type']  # CHOCH direction, not old OB type
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
        base_confidence = 72
        
        # Boost for OB strength
        base_confidence += nearest_ob['strength'] // 10
        
        # Boost for candlestick
        base_confidence += candlestick_boost['confidence_boost']
        
        # Polarity flip is powerful concept
        base_confidence += 8
        result['reasoning'].append("OB polarity flip confirmed by CHOCH")
        
        result['confidence'] = min(100, base_confidence)
        
        # Step 7: Set signal, stop, target
        if choch['type'] == 'BULLISH':
            result['signal'] = 'CALL'
            result['stop_loss'] = nearest_ob['low'] * 0.998
            result['target'] = resistance
        else:
            result['signal'] = 'PUT'
            result['stop_loss'] = nearest_ob['high'] * 1.002
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
