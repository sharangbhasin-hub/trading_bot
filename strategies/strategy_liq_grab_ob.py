"""
Strategy 5: Liquidity Grab + Order Block
Combines liquidity sweep with order block for institutional entry
"""
import pandas as pd
from typing import Dict, Optional
from strategies.base_strategy import BaseStrategy
from detectors.liquidity_detector import LiquidityDetector
from detectors.order_block_detector import OrderBlockDetector
from detectors.retest_detector import RetestDetector

class LiquidityGrabOrderBlockStrategy(BaseStrategy):
    """Liquidity Grab + Order Block Strategy"""
    
    def __init__(self):
        super().__init__(name="Liquidity Grab + Order Block")
        self.liq_detector = LiquidityDetector()
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
        """Analyze for Liquidity Grab + OB setup"""
        
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
        
        # Step 1: Detect liquidity sweep on 5min chart
        sweep = self.liq_detector.detect_sweep(df_5min)
        
        if not sweep:
            result['reasoning'].append("No liquidity sweep detected")
            return result
        
        if not sweep['rejection_confirmed']:
            result['reasoning'].append("Liquidity sweep detected but no rejection")
            return result
        
        result['reasoning'].append(
            f"{sweep['type']} with {sweep['wick_size_pct']:.1f}% rejection"
        )
        
        # Step 2: Find Order Block on 15min chart
        order_blocks = self.ob_detector.detect(df_15min)
        
        if not order_blocks:
            result['reasoning'].append("No order blocks detected")
            return result
        
        # Step 3: Find OB that aligns with sweep direction
        if sweep['type'] == 'LOW_SWEEP':
            # Looking for bullish reversal - need bullish OB
            relevant_obs = [ob for ob in order_blocks if ob['type'] == 'BULLISH']
            direction = 'BULLISH'
        else:  # HIGH_SWEEP
            # Looking for bearish reversal - need bearish OB
            relevant_obs = [ob for ob in order_blocks if ob['type'] == 'BEARISH']
            direction = 'BEARISH'
        
        if not relevant_obs:
            result['reasoning'].append(f"No {direction} order blocks found")
            return result
        
        # Find OB closest to sweep level
        sweep_level = sweep['swept_level']
        best_ob = min(
            relevant_obs,
            key=lambda ob: min(
                abs(ob['high'] - sweep_level),
                abs(ob['low'] - sweep_level)
            )
        )
        
        # Check if OB is reasonably close to sweep (within 1%)
        distance_pct = min(
            abs(best_ob['high'] - sweep_level) / sweep_level * 100,
            abs(best_ob['low'] - sweep_level) / sweep_level * 100
        )
        
        if distance_pct > 1.5:
            result['reasoning'].append(
                f"Order Block too far from sweep level ({distance_pct:.2f}%)"
            )
            return result
        
        result['setup_detected'] = True
        result['reasoning'].append(
            f"Order Block at {best_ob['low']:.2f}-{best_ob['high']:.2f} "
            f"near sweep level"
        )
        
        # Step 4: Check for retest of OB zone
        retest_result = self.retest_detector.check_retest(
            df=df_5min,
            zone_high=best_ob['high'],
            zone_low=best_ob['low'],
            expected_direction=direction
        )
        
        result['retest_confirmed'] = retest_result['retest_confirmed']
        result['reasoning'].append(retest_result['reasoning'])
        
        if not retest_result['retest_confirmed']:
            return result
        
        # Step 5: Check candlestick confirmation
        candlestick_boost = self._check_candlestick(df_5min, direction)
        if candlestick_boost['pattern']:
            result['candlestick_pattern'] = candlestick_boost['pattern']
            result['reasoning'].append(f"Candlestick: {candlestick_boost['pattern']}")
        
        # Step 6: Calculate confidence
        base_confidence = 73  # High confidence setup
        
        # Boost for strong OB
        base_confidence += best_ob['strength'] // 10
        
        # Boost for strong sweep rejection
        base_confidence += min(12, sweep['wick_size_pct'] // 5)
        
        # Boost for candlestick
        base_confidence += candlestick_boost['confidence_boost']
        
        # Boost if close proximity between sweep and OB
        if distance_pct < 0.5:
            base_confidence += 5
            result['reasoning'].append("Tight confluence between sweep and OB")
        
        # Alignment with trend
        if ((direction == 'BULLISH' and overall_trend == 'Bullish') or
            (direction == 'BEARISH' and overall_trend == 'Bearish')):
            base_confidence += 8
            result['reasoning'].append("Aligned with overall trend")
        
        result['confidence'] = min(100, base_confidence)
        
        # Step 7: Set signal, stop, target
        if direction == 'BULLISH':
            result['signal'] = 'CALL'
            result['stop_loss'] = best_ob['low'] * 0.997
            result['target'] = resistance
        else:
            result['signal'] = 'PUT'
            result['stop_loss'] = best_ob['high'] * 1.003
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
        
        if total_range == 0:
            return {'pattern': None, 'confidence_boost': 0}
        
        if direction == 'BULLISH':
            if lower_wick > body * 2 and upper_wick < body * 0.3:
                return {'pattern': 'Hammer', 'confidence_boost': 15}
            
            if (last_candle['close'] > last_candle['open'] and
                prev_candle['close'] < prev_candle['open'] and
                last_candle['open'] < prev_candle['close'] and
                last_candle['close'] > prev_candle['open']):
                return {'pattern': 'Bullish Engulfing', 'confidence_boost': 15}
            
            if lower_wick > total_range * 0.5:
                return {'pattern': 'Bullish Pin Bar', 'confidence_boost': 10}
        
        else:  # BEARISH
            if upper_wick > body * 2 and lower_wick < body * 0.3:
                return {'pattern': 'Shooting Star', 'confidence_boost': 15}
            
            if (last_candle['close'] < last_candle['open'] and
                prev_candle['close'] > prev_candle['open'] and
                last_candle['open'] > prev_candle['close'] and
                last_candle['close'] < prev_candle['open']):
                return {'pattern': 'Bearish Engulfing', 'confidence_boost': 15}
            
            if upper_wick > total_range * 0.5:
                return {'pattern': 'Bearish Pin Bar', 'confidence_boost': 10}
        
        return {'pattern': None, 'confidence_boost': 0}
