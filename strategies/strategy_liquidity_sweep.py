"""
Strategy 2: Liquidity Sweep + Reversal + Candlestick
"""
import pandas as pd
from typing import Dict
from strategies.base_strategy import BaseStrategy
from detectors.liquidity_detector import LiquidityDetector
from detectors.retest_detector import RetestDetector

class LiquiditySweepStrategy(BaseStrategy):
    """Liquidity Sweep + Reversal Strategy"""
    
    def __init__(self):
        super().__init__(name="Liquidity Sweep + Reversal")
        self.liq_detector = LiquidityDetector()
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
        """Analyze for liquidity sweep setup"""
        
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
            result['reasoning'].append(f"{sweep['type']} detected but no rejection wick")
            return result
        
        result['setup_detected'] = True
        result['reasoning'].append(
            f"{sweep['type']} with {sweep['wick_size_pct']:.1f}% rejection wick"
        )
        
        # Step 2: Determine direction
        if sweep['type'] == 'LOW_SWEEP':
            direction = 'BULLISH'
            zone_high = sweep['swept_level'] * 1.001
            zone_low = sweep['swept_level'] * 0.999
        else:  # HIGH_SWEEP
            direction = 'BEARISH'
            zone_high = sweep['swept_level'] * 1.001
            zone_low = sweep['swept_level'] * 0.999
        
        # Step 3: Check for retest
        retest_result = self.retest_detector.check_retest(
            df=df_5min,
            zone_high=zone_high,
            zone_low=zone_low,
            expected_direction=direction
        )
        
        result['retest_confirmed'] = retest_result['retest_confirmed']
        result['reasoning'].append(retest_result['reasoning'])
        
        if not retest_result['retest_confirmed']:
            return result
        
        # Step 4: Check candlestick confirmation
        candlestick_boost = self._check_candlestick(df_5min, direction)
        if candlestick_boost['pattern']:
            result['candlestick_pattern'] = candlestick_boost['pattern']
            result['reasoning'].append(f"Candlestick: {candlestick_boost['pattern']}")
        
        # Step 5: Calculate confidence
        base_confidence = 70
        base_confidence += min(15, sweep['wick_size_pct'] // 5)  # Bigger wick = higher confidence
        base_confidence += candlestick_boost['confidence_boost']
        
        # Alignment with trend
        if ((direction == 'BULLISH' and overall_trend == 'Bullish') or
            (direction == 'BEARISH' and overall_trend == 'Bearish')):
            base_confidence += 10
            result['reasoning'].append("Aligned with overall trend")
        
        result['confidence'] = min(100, base_confidence)
        
        # Step 6: Set signal, stop, target
        if direction == 'BULLISH':
            result['signal'] = 'CALL'
            result['stop_loss'] = zone_low * 0.997
            result['target'] = resistance
        else:
            result['signal'] = 'PUT'
            result['stop_loss'] = zone_high * 1.003
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
            # Hammer or Pin Bar
            if lower_
