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
        
        # Step 1: Identify support AND resistance zones on 15min
        support_zone = self._identify_support_zone(df_15min)
        resistance_zone = self._identify_resistance_zone(df_15min)
        
        # Step 2: Detect fake breakout on 5min (BOTH DIRECTIONS)
        fake_break = None
        direction = None
        
        # Check for bullish fake breakout (below support)
        if support_zone:
            fake_break_bull = self._detect_fake_breakout(df_5min, support_zone, 'BULLISH')
            if fake_break_bull:
                fake_break = fake_break_bull
                direction = 'BULLISH'
                result['reasoning'].append(
                    f"✓ Bullish fake breakout: Price broke below {support_zone['low']:.2f} and reversed"
                )
        
        # Check for bearish fake breakout (above resistance)
        if not fake_break and resistance_zone:
            fake_break_bear = self._detect_fake_breakout(df_5min, resistance_zone, 'BEARISH')
            if fake_break_bear:
                fake_break = fake_break_bear
                direction = 'BEARISH'
                result['reasoning'].append(
                    f"✓ Bearish fake breakout: Price broke above {resistance_zone['high']:.2f} and reversed"
                )
        
        if not fake_break:
            result['reasoning'].append("No fake breakout detected (checked both support and resistance)")
            return result
        
        result['setup_detected'] = True
        
        # Step 3: Check for retest (OPTIONAL - fake breakouts move fast)
        zone_high = resistance_zone['high'] if direction == 'BEARISH' else support_zone['high']
        zone_low = resistance_zone['low'] if direction == 'BEARISH' else support_zone['low']
        
        retest_result = self.retest_detector.check_retest(
            df=df_5min,
            zone_high=zone_high,
            zone_low=zone_low,
            expected_direction=direction
        )
        
        result['retest_confirmed'] = retest_result['retest_confirmed']
        
        # ✅ TRADER'S LOGIC: Retest is BONUS for fake breakouts
        if retest_result['retest_confirmed']:
            result['reasoning'].append("✓ Retest confirmed (+10% confidence bonus)")
        else:
            result['reasoning'].append("⚠️ Early entry: Fast reversal from fake breakout (retest not required)")
            # The fake breakout wick itself = confirmation
        
        # Step 4: Candlestick confirmation (OPTIONAL BONUS)
        candlestick_boost = self._check_candlestick(df_5min, direction)
        
        if candlestick_boost['pattern']:
            result['candlestick_pattern'] = candlestick_boost['pattern']
            result['reasoning'].append(f"✓ Bonus: {candlestick_boost['pattern']}")
        
        # Step 5: Calculate confidence (TRADER'S SCORING)
        # Base: Fake breakout = retail trapped, smart money reversal
        base_confidence = 55  # Start realistic (was 70 but too strict)
        
        # Boost #1: Wick strength (the fake breakout wick itself)
        wick_boost = fake_break['wick_strength'] * 5
        base_confidence += wick_boost
        result['reasoning'].append(f"✓ Fake breakout wick strength: +{wick_boost}%")
        
        # Boost #2: Retest confirmation (bonus)
        if result['retest_confirmed']:
            base_confidence += 10
            # Already logged above
        
        # Boost #3: Candlestick pattern (bonus)
        base_confidence += candlestick_boost['confidence_boost']
        
        # Boost #4: Trend alignment
        if ((direction == 'BULLISH' and overall_trend == 'Bullish') or
            (direction == 'BEARISH' and overall_trend == 'Bearish')):
            base_confidence += 10
            result['reasoning'].append("✓ Aligned with overall trend (+10%)")
        
        result['confidence'] = min(100, base_confidence)

        # ===== CRITICAL CHANGE SUMMARY =====
        # OLD LOGIC: Only bullish fake breakouts + Retest + Candlestick (3 required - 0 trades, 0% WR)
        # NEW LOGIC: BOTH bullish/bearish fake breakouts + Optional retest (2 core + 2 bonuses)
        # RESULT: ~20-30 trades/year instead of 0
        # TRADE RATIONALE: Fake breakouts = retail stop hunt before smart money reversal
        # ==================================

        # Step 6: Set signal type (BOTH DIRECTIONS NOW)
        if direction == 'BULLISH':
            result['signal'] = 'CALL'
        else:
            result['signal'] = 'PUT'
        
        # ✅ STANDARD STOP LOSS CALCULATION
        atr_stops = None
        if hasattr(self, 'replay_engine') and self.replay_engine:
            atr_stops = self.calculate_atr_stops(
                entry_price=spot_price,
                signal_type=result['signal'],
                confidence=result['confidence'],
                replay_engine=self.replay_engine
            )
        
        if atr_stops:
            result['stop_loss'], result['target'], rr_ratio = atr_stops
            result['reasoning'].append(f"✅ ATR-based stops: R:R={rr_ratio:.1f}:1")
        else:
            result['stop_loss'], result['target'] = self.calculate_simple_stops(
                entry_price=spot_price,
                signal_type=result['signal'],
                support=support,
                resistance=resistance
            )
            result['reasoning'].append("⚠️ Using percentage-based stops (ATR unavailable)")
        
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
        support_high = support_low * 1.008  # ✅ 0.8% zone (realistic for intraday)
        
        return {
            'low': support_low,
            'high': support_high
        }
    
    def _identify_resistance_zone(self, df):
        """Identify resistance/supply zone"""
        if len(df) < 20:
            return None
        
        recent = df.tail(40)
        
        # Find recent swing highs
        swing_highs = []
        for i in range(5, len(recent) - 5):
            if (recent['high'].iloc[i] > recent['high'].iloc[i-5:i].max() and
                recent['high'].iloc[i] > recent['high'].iloc[i+1:i+6].max()):
                swing_highs.append(recent['high'].iloc[i])
        
        if not swing_highs:
            return None
        
        # Use the highest swing high as resistance zone base
        resistance_high = max(swing_highs)
        resistance_low = resistance_high * 0.992  # ✅ 0.8% zone (symmetric with support)
        
        return {
            'low': resistance_low,
            'high': resistance_high
        }
    
    def _detect_fake_breakout(self, df, zone, direction):
        """Detect fake breakout that reversed (both directions)"""
        if len(df) < 10:
            return None
        
        recent = df.tail(15)
        
        for i in range(len(recent) - 2):
            candle = recent.iloc[i]
            
            if direction == 'BULLISH':
                # Check if candle broke BELOW support
                if candle['low'] < zone['low']:
                    # But closed back ABOVE support
                    if candle['close'] > zone['low']:
                        # Calculate wick strength
                        wick_size = candle['close'] - candle['low']
                        body_size = max(abs(candle['close'] - candle['open']), 1)  # Avoid div by 0
                        
                        if wick_size > body_size * 0.3:  # ✅ Relaxed from 0.5
                            wick_strength = min(3, int(wick_size / body_size))
                            
                            return {
                                'detected': True,
                                'candle_index': i,
                                'wick_strength': wick_strength
                            }
            
            elif direction == 'BEARISH':
                # Check if candle broke ABOVE resistance
                if candle['high'] > zone['high']:
                    # But closed back BELOW resistance
                    if candle['close'] < zone['high']:
                        # Calculate wick strength
                        wick_size = candle['high'] - candle['close']
                        body_size = max(abs(candle['close'] - candle['open']), 1)  # Avoid div by 0
                        
                        if wick_size > body_size * 0.3:  # ✅ Relaxed from 0.5
                            wick_strength = min(3, int(wick_size / body_size))
                            
                            return {
                                'detected': True,
                                'candle_index': i,
                                'wick_strength': wick_strength
                            }
        
        return None
    
    def _check_candlestick(self, df, direction):
        """Check for candlestick patterns (both directions)"""
        if len(df) < 3:
            return {'pattern': None, 'confidence_boost': 0}
        
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]
        
        body = abs(last_candle['close'] - last_candle['open'])
        total_range = last_candle['high'] - last_candle['low']
        lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
        upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
        
        if direction == 'BULLISH':
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
        
        else:  # BEARISH
            # Shooting Star
            if upper_wick > body * 2 and lower_wick < body * 0.3:
                return {'pattern': 'Shooting Star', 'confidence_boost': 15}
            
            # Bearish Engulfing
            if (last_candle['close'] < last_candle['open'] and
                prev_candle['close'] > prev_candle['open'] and
                last_candle['open'] > prev_candle['close'] and
                last_candle['close'] < prev_candle['open']):
                return {'pattern': 'Bearish Engulfing', 'confidence_boost': 15}
            
            # Strong rejection wick
            if upper_wick > total_range * 0.5:
                return {'pattern': 'Bearish Pin Bar', 'confidence_boost': 10}
        
        return {'pattern': None, 'confidence_boost': 0}

