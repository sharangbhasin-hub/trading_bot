"""
Strategy: Discounted Zone + Fake Breakout
Turtle soup variant - false breakdown that reverses
"""
import pandas as pd
from typing import Dict, Optional
from strategies.base_strategy import BaseStrategy
from detectors.retest_detector import RetestDetector
from utils.dataframe_validator import DataFrameValidator

class FakeBreakoutStrategy(BaseStrategy):
    """Discounted Zone + Fake Breakout Strategy"""
    
    def __init__(self):
        super().__init__(name="Fake Breakout")
        self.retest_detector = RetestDetector()
        self.min_confidence = 70

    def detect(self, df: pd.DataFrame, current_idx: int) -> dict:
        """Detect Fake Breakout setup"""
        
        # MARKET REGIME FILTER & VALIDATION - Note: REVERSAL type
        should_trade, regime_reason = self.check_market_regime(df, current_idx, 'REVERSAL')
        if not should_trade:
            return {'signal_type': 'NO_TRADE', 'confidence': 0, 'setup_detected': False,
                    'retest_confirmed': False, 'reasoning': f"Market regime: {regime_reason}"}
        
        is_valid, errors = self.df_validator.validate_ohlc(df, strict=False, min_rows=20)
        if not is_valid:
            self.logger.error(f"Invalid data: {errors}")
            return {'signal_type': 'NO_TRADE', 'confidence': 0, 'setup_detected': False,
                    'retest_confirmed': False, 'reasoning': f"Data error: {errors[0] if errors else 'Unknown'}"}

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
        
        # Step 1: Identify support zone (discounted zone) on 15min
        support_zone = self._identify_support_zone(df_15min)
        
        if not support_zone:
            result['reasoning'].append("No clear support zone identified")
            return result
        
        result['reasoning'].append(
            f"Support zone identified: {support_zone['low']:.2f} - {support_zone['high']:.2f}"
        )
        
        # Step 2: Detect fake breakout on 5min
        fake_break = self._detect_fake_breakout(df_5min, support_zone)
        
        if not fake_break:
            result['reasoning'].append("No fake breakout detected")
            return result
        
        result['setup_detected'] = True
        result['reasoning'].append(
            f"Fake breakout detected: Price broke below {support_zone['low']:.2f} "
            f"and closed back above"
        )
        
        # Step 3: Check for retest
        retest_result = self.retest_detector.check_retest(
            df=df_5min,
            zone_high=support_zone['high'],
            zone_low=support_zone['low'],
            expected_direction='BULLISH'  # Fake breakout is bullish reversal
        )
        
        result['retest_confirmed'] = retest_result['retest_confirmed']
        result['reasoning'].append(retest_result['reasoning'])
        
        if not retest_result['retest_confirmed']:
            return result
        
        # Step 4: Candlestick confirmation
        candlestick_boost = self._check_candlestick(df_5min)
        
        if candlestick_boost['pattern']:
            result['candlestick_pattern'] = candlestick_boost['pattern']
            result['reasoning'].append(f"Candlestick: {candlestick_boost['pattern']}")
        
        # Step 5: Calculate confidence
        base_confidence = 70
        base_confidence += fake_break['wick_strength'] * 5  # Bigger wick = more confidence
        base_confidence += candlestick_boost['confidence_boost']
        
        if overall_trend == 'Bullish':
            base_confidence += 10
            result['reasoning'].append("Aligned with bullish trend")
        
        result['confidence'] = min(100, base_confidence)
        
        # Step 6: Set signal, dynamic stop loss, target
        result['signal'] = 'CALL'
        result['stop_loss'] = self.calculate_dynamic_stop_loss(
            zone_low=support_zone['low'],
            zone_high=support_zone['high'],
            direction='BULLISH',
            spot_price=spot_price
        )
        result['target'] = resistance
        
        # Step 7: Validate Risk:Reward Ratio
        result = self.validate_risk_reward(result)
        
        return result
    
    def _identify_support_zone(self, df):
        """Identify support/demand zone"""
        if len(df) < 20:
            return None
        
        recent = df.tail(40)
        
        # Find recent swing lows
        swing_lows = []
        for i in range(5, len(recent) - 5):
            if (recent['low'].iloc[i] < recent['low'].iloc[i-5:i].min() and
                recent['low'].iloc[i] < recent['low'].iloc[i+1:i+6].min()):
                swing_lows.append(recent['low'].iloc[i])
        
        if not swing_lows:
            return None
        
        # Use the lowest swing low as support zone base
        support_low = min(swing_lows)
        support_high = support_low * 1.002  # 0.2% zone
        
        return {
            'low': support_low,
            'high': support_high
        }
    
    def _detect_fake_breakout(self, df, support_zone):
        """Detect fake breakdown that reversed"""
        if len(df) < 10:
            return None
        
        recent = df.tail(15)
        
        for i in range(len(recent) - 2):
            candle = recent.iloc[i]
            
            # Check if candle broke below support
            if candle['low'] < support_zone['low']:
                # But closed back above support
                if candle['close'] > support_zone['low']:
                    # Calculate wick strength
                    wick_size = candle['close'] - candle['low']
                    body_size = abs(candle['close'] - candle['open'])
                    
                    if wick_size > body_size * 0.5:  # Wick is significant
                        wick_strength = min(3, int(wick_size / body_size))
                        
                        return {
                            'detected': True,
                            'candle_index': i,
                            'wick_strength': wick_strength
                        }
        
        return None
    
    def _check_candlestick(self, df):
        """Check for bullish candlestick patterns"""
        if len(df) < 3:
            return {'pattern': None, 'confidence_boost': 0}
        
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]
        
        body = abs(last_candle['close'] - last_candle['open'])
        total_range = last_candle['high'] - last_candle['low']
        lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
        upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
        
        # Hammer
        if lower_wick > body * 2 and upper_wick < body * 0.3:
            return {'pattern': 'Hammer', 'confidence_boost': 15}
        
        # Bullish Engulfing
        if (last_candle['close'] > last_candle['open'] and
            prev_candle['close'] < prev_candle['open'] and
            last_candle['open'] < prev_candle['close'] and
            last_candle['close'] > prev_candle['open']):
            return {'pattern': 'Bullish Engulfing', 'confidence_boost': 15}
        
        # Strong rejection wick
        if lower_wick > total_range * 0.5:
            return {'pattern': 'Bullish Pin Bar', 'confidence_boost': 10}
        
        return {'pattern': None, 'confidence_boost': 0}
