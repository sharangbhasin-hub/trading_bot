"""
Strategy 1: Order Block + FVG + Candlestick Confirmation
"""
import pandas as pd
from typing import Dict
from strategies.base_strategy import BaseStrategy
from detectors.order_block_detector import OrderBlockDetector
from detectors.fvg_detector import FVGDetector
from detectors.retest_detector import RetestDetector

class OrderBlockFVGStrategy(BaseStrategy):
    """Order Block + FVG Strategy with retest logic"""
    
    def __init__(self):
        super().__init__(name="Order Block + FVG")
        self.ob_detector = OrderBlockDetector()
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
        """Analyze for OB + FVG setup"""
        
        # Initialize result
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
        
        # Step 1: Detect Order Blocks on 15min chart
        order_blocks = self.ob_detector.detect(df_15min)
        if not order_blocks:
            result['reasoning'].append("No order blocks detected")
            return result
        
        # Step 2: Detect FVGs on 15min chart
        fvgs = self.fvg_detector.detect(df_15min)
        if not fvgs:
            result['reasoning'].append("No FVGs detected")
            return result
        
        # Step 3: Find overlap between OB and FVG
        confluence_zones = self._find_confluence(order_blocks, fvgs, spot_price)
        if not confluence_zones:
            result['reasoning'].append("No OB + FVG confluence found")
            return result
        
        # Take the nearest/strongest confluence zone
        best_zone = confluence_zones[0]
        result['setup_detected'] = True
        result['reasoning'].append(f"{best_zone['direction']} OB + FVG confluence detected at {best_zone['zone_low']:.2f} - {best_zone['zone_high']:.2f}")
        
        # Step 4: Check for retest using 5min data for precision
        retest_result = self.retest_detector.check_retest(
            df=df_5min,
            zone_high=best_zone['zone_high'],
            zone_low=best_zone['zone_low'],
            expected_direction=best_zone['direction']
        )
        
        result['retest_confirmed'] = retest_result['retest_confirmed']
        result['reasoning'].append(retest_result['reasoning'])
        
        if not retest_result['retest_confirmed']:
            return result
        
        # Step 5: Check candlestick pattern at retest
        candlestick_boost = self._check_candlestick_confirmation(
            df_5min, 
            best_zone['direction']
        )
        
        if candlestick_boost['pattern']:
            result['candlestick_pattern'] = candlestick_boost['pattern']
            result['reasoning'].append(f"Candlestick confirmation: {candlestick_boost['pattern']}")
        
        # Step 6: Calculate confidence
        base_confidence = 65
        
        # Boost for OB strength
        base_confidence += best_zone['ob_strength'] // 10
        
        # Boost for candlestick
        base_confidence += candlestick_boost['confidence_boost']
        
        # Boost if aligned with overall trend
        if ((best_zone['direction'] == 'BULLISH' and overall_trend == 'Bullish') or
            (best_zone['direction'] == 'BEARISH' and overall_trend == 'Bearish')):
            base_confidence += 10
            result['reasoning'].append("Aligned with overall market trend")
        
        result['confidence'] = min(100, base_confidence)
        
        # Step 7: Set signal, stop loss, target
        if expected_direction == 'BULLISH':
            result['signal'] = 'CALL'
            result['stop_loss'] = self.calculate_dynamic_stop_loss(
                zone_low=zone_low,
                zone_high=zone_high,
                direction='BULLISH',
                spot_price=spot_price
            )
            result['target'] = resistance
        else:
            result['signal'] = 'PUT'
            result['stop_loss'] = self.calculate_dynamic_stop_loss(
                zone_low=zone_low,
                zone_high=zone_high,
                direction='BEARISH',
                spot_price=spot_price
            )
            result['target'] = support
        
        # Validate Risk:Reward ratio
        result = self.validate_risk_reward(result)
    
    def _find_confluence(self, order_blocks, fvgs, current_price):
        """Find zones where OB and FVG overlap"""
        confluences = []
        
        for ob in order_blocks:
            for fvg in fvgs:
                # Check if same direction
                if ob['type'] != fvg['type']:
                    continue
                
                # Check if they overlap
                ob_range = (ob['low'], ob['high'])
                fvg_range = (fvg['bottom'], fvg['top'])
                
                overlap = self._ranges_overlap(ob_range, fvg_range)
                if overlap:
                    # Calculate confluence zone
                    zone_low = max(ob['low'], fvg['bottom'])
                    zone_high = min(ob['high'], fvg['top'])
                    
                    # Check if current price is near zone (within 1%)
                    zone_mid = (zone_low + zone_high) / 2
                    distance_pct = abs((current_price - zone_mid) / zone_mid) * 100
                    
                    if distance_pct < 2.0:  # Within 2% of zone
                        confluences.append({
                            'direction': ob['type'],
                            'zone_low': zone_low,
                            'zone_high': zone_high,
                            'ob_strength': ob['strength'],
                            'distance_pct': distance_pct
                        })
        
        # Sort by distance (nearest first) and strength
        confluences.sort(key=lambda x: (x['distance_pct'], -x['ob_strength']))
        return confluences
    
    def _ranges_overlap(self, range1, range2):
        """Check if two ranges overlap"""
        return range1[0] <= range2[1] and range2[0] <= range1[1]
    
    def _check_candlestick_confirmation(self, df, direction):
        """Check for candlestick patterns on last 3 candles"""
        if len(df) < 3:
            return {'pattern': None, 'confidence_boost': 0}
        
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]
        
        body = abs(last_candle['close'] - last_candle['open'])
        total_range = last_candle['high'] - last_candle['low']
        lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
        upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
        
        if direction == 'BULLISH':
            # Check for hammer
            if (lower_wick > body * 2 and upper_wick < body * 0.3 and
                last_candle['close'] > last_candle['open']):
                return {'pattern': 'Hammer', 'confidence_boost': 15}
            
            # Check for bullish engulfing
            if (last_candle['close'] > last_candle['open'] and
                prev_candle['close'] < prev_candle['open'] and
                last_candle['open'] < prev_candle['close'] and
                last_candle['close'] > prev_candle['open']):
                return {'pattern': 'Bullish Engulfing', 'confidence_boost': 15}
            
            # Check for strong rejection wick
            if lower_wick > total_range * 0.5:
                return {'pattern': 'Bullish Rejection', 'confidence_boost': 10}
        
        else:  # BEARISH
            # Check for shooting star
            if (upper_wick > body * 2 and lower_wick < body * 0.3 and
                last_candle['close'] < last_candle['open']):
                return {'pattern': 'Shooting Star', 'confidence_boost': 15}
            
            # Check for bearish engulfing
            if (last_candle['close'] < last_candle['open'] and
                prev_candle['close'] > prev_candle['open'] and
                last_candle['open'] > prev_candle['close'] and
                last_candle['close'] < prev_candle['open']):
                return {'pattern': 'Bearish Engulfing', 'confidence_boost': 15}
            
            # Check for strong rejection wick
            if upper_wick > total_range * 0.5:
                return {'pattern': 'Bearish Rejection', 'confidence_boost': 10}
        
        return {'pattern': None, 'confidence_boost': 0}
