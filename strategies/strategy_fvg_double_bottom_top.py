"""
Strategy 8: FVG + Double Bottom/Top + Breakout
Classic pattern meets modern SMC
"""
import pandas as pd
from typing import Dict, Optional
from strategies.base_strategy import BaseStrategy
from detectors.fvg_detector import FVGDetector
from detectors.retest_detector import RetestDetector
from utils.dataframe_validator import DataFrameValidator

class FVGDoubleBottomTopStrategy(BaseStrategy):
    """FVG + Double Bottom/Top + Breakout Strategy"""
    
    def __init__(self):
        super().__init__(name="FVG + Double Bottom/Top")
        self.fvg_detector = FVGDetector()
        self.retest_detector = RetestDetector()

    def detect(self, df: pd.DataFrame, current_idx: int) -> dict:
        """Detect BOS + Retest setup"""
        
        # MARKET REGIME FILTER & VALIDATION
        should_trade, regime_reason = self.check_market_regime(df, current_idx, 'TREND_FOLLOWING')
        if not should_trade:
            return {'signal_type': 'NO_TRADE', 'confidence': 0, 'setup_detected': False, 
                    'retest_confirmed': False, 'reasoning': f"Market regime: {regime_reason}"}
        
        is_valid, errors = self.df_validator.validate_ohlc(df, strict=False, min_rows=20)
        if not is_valid:
            self.logger.error(f"Invalid data: {errors}")
            return {'signal_type': 'NO_TRADE', 'confidence': 0, 'setup_detected': False,
                    'retest_confirmed': False, 'reasoning': f"Data error: {errors[0] if errors else 'Unknown'}"}
        
        # ✅ ADD THIS: Return success if validation passes
        return {
            'signal_type': 'SETUP_READY',
            'confidence': 50,
            'setup_detected': True,
            'retest_confirmed': False,
            'reasoning': 'Market regime and data validation passed'
        }    
    
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
            'stop_loss': 0,
            'target': 0,
            'reasoning': [],
            'setup_detected': False,
            'retest_confirmed': False,
            'candlestick_pattern': None
        }
        
        # Step 1: Detect FVGs on 15min
        fvgs = self.fvg_detector.detect(df_15min)

        # ✅ ADD THIS: Debug logging
        self.logger.info(f"FVG Detection: Found {len(fvgs)} FVGs on 15min")
        if fvgs:
            for idx, fvg in enumerate(fvgs):
                self.logger.info(
                    f"  FVG #{idx+1}: Type={fvg['type']}, "
                    f"Range={fvg['bottom']:.2f}-{fvg['top']:.2f}"
                )
                    
        if not fvgs:
            result['reasoning'].append("No FVGs detected")
            return result

        # ✅ ADD THIS
        self.logger.info(f"🔥🔥🔥 ABOUT TO CALL _detect_double_bottom_top with df length={len(df_15min)}, FVGs={len(fvgs)}")
        
        # Step 2: Detect double bottom or double top
        pattern = self._detect_double_bottom_top(df_15min, fvgs)
        
        # ✅ ADD THIS
        self.logger.info(f"🔥🔥🔥 RETURNED FROM _detect_double_bottom_top, pattern={pattern}")

        # ✅ ADD THIS: Debug logging
        if not pattern:
            self.logger.info("Pattern Detection: No double bottom/top found")
            self.logger.info(f"  Checked {len(fvgs)} FVGs against swing points")
            result['reasoning'].append("No double bottom/top pattern found")
            return result
        else:
            self.logger.info(
                f"Pattern Detection: {pattern['type']} found at "
                f"{pattern['level_1']:.2f} / {pattern['level_2']:.2f}"
            )

        result['setup_detected'] = True
        result['reasoning'].append(
            f"{pattern['type']} pattern detected at {pattern['level_1']:.2f} / {pattern['level_2']:.2f}"
        )
        result['reasoning'].append(
            f"FVG zone: {pattern['fvg']['bottom']:.2f} - {pattern['fvg']['top']:.2f}"
        )
        
        # Step 3: Check for breakout above/below neckline
        breakout = self._check_breakout(df_15min, pattern)
        
        if not breakout:
            result['reasoning'].append("No breakout detected yet")
            return result
        
        result['reasoning'].append(
            f"Breakout confirmed above/below neckline at {breakout['neckline']:.2f}"
        )
        
        # Step 4: Check for retest on 5min
        zone_low = breakout['neckline'] * 0.998
        zone_high = breakout['neckline'] * 1.002
        
        retest_result = self.retest_detector.check_retest(
            df=df_5min,
            zone_high=zone_high,
            zone_low=zone_low,
            expected_direction=pattern['expected_direction']
        )
        
        result['retest_confirmed'] = retest_result['retest_confirmed']
        result['reasoning'].append(retest_result['reasoning'])
        
        # ✅ TRADER'S FIX: Retest is OPTIONAL (bonus confidence)
        if not result['retest_confirmed']:
            result['reasoning'].append("⚠️ Early entry - retest not confirmed yet")
            # Don't return - allow trade without retest
        
        # Step 5: Candlestick confirmation
        candlestick_boost = self._check_candlestick(df_5min, pattern['expected_direction'])
        
        if candlestick_boost['pattern']:
            result['candlestick_pattern'] = candlestick_boost['pattern']
            result['reasoning'].append(f"Candlestick: {candlestick_boost['pattern']}")
        
        # Step 6: Calculate confidence (TRADER'S SCORING)
        base_confidence = 50  # Pattern + Breakout = moderate setup
        
        # Boost #1: FVG confirmation (+10%)
        base_confidence += 10
        result['reasoning'].append("✓ FVG confirmed at pattern level (+10%)")
        
        # Boost #2: Retest confirmed (+12%)
        if result['retest_confirmed']:
            base_confidence += 12
            result['reasoning'].append("✓ Breakout retest confirmed (+12%)")
        
        # Boost #3: Candlestick pattern (max +10%)
        candlestick_score = min(10, candlestick_boost['confidence_boost'])
        base_confidence += candlestick_score
        if candlestick_boost['pattern']:
            result['reasoning'].append(f"✓ {candlestick_boost['pattern']} (+{candlestick_score}%)")
        
        # Boost #4: Trend alignment (+8%)
        trend_aligned = False
        if ((pattern['expected_direction'] == 'BULLISH' and overall_trend == 'Bullish') or
            (pattern['expected_direction'] == 'BEARISH' and overall_trend == 'Bearish')):
            base_confidence += 8
            trend_aligned = True
            result['reasoning'].append("✓ Aligned with overall trend (+8%)")
        
        # Cap confidence at 70%
        result['confidence'] = max(45, min(70, base_confidence))
        
        # Final confidence log
        result['reasoning'].append(
            f"Final Confidence: {result['confidence']}% "
            f"(pattern=50, FVG=10, retest={'+12' if result['retest_confirmed'] else '0'}, "
            f"candle={candlestick_score}, trend={'+8' if trend_aligned else '0'})"
        )
        
        # Step 7: Set signal type
        if pattern['expected_direction'] == 'BULLISH':
            result['signal'] = 'CALL'
        else:
            result['signal'] = 'PUT'
        
        # ========== ✅ FIX: CALCULATE STRATEGY-SPECIFIC TARGET ==========
        # For Double Bottom/Top patterns, target is calculated using:
        # Pattern Height Projection (classic technical analysis)
        
        if pattern['expected_direction'] == 'BULLISH':
            # Bullish: Pattern height projected above neckline
            # Pattern height = neckline - pattern_low
            pattern_low = min(pattern['level_1'], pattern['level_2'])
            pattern_height = pattern['neckline'] - pattern_low
            
            # Target = neckline + pattern_height (1:1 projection)
            strategy_target = pattern['neckline'] + pattern_height
            strategy_support = pattern_low  # Pattern low as support
            
            result['reasoning'].append(
                f"✅ Double Bottom target: {strategy_target:.2f} "
                f"(neckline {pattern['neckline']:.2f} + pattern height {pattern_height:.2f})"
            )
        else:  # BEARISH
            # Bearish: Pattern height projected below neckline
            # Pattern height = pattern_high - neckline
            pattern_high = max(pattern['level_1'], pattern['level_2'])
            pattern_height = pattern_high - pattern['neckline']
            
            # Target = neckline - pattern_height (1:1 projection)
            strategy_target = pattern['neckline'] - pattern_height
            strategy_support = pattern_high  # Pattern high as resistance
            
            result['reasoning'].append(
                f"✅ Double Top target: {strategy_target:.2f} "
                f"(neckline {pattern['neckline']:.2f} - pattern height {pattern_height:.2f})"
            )
        # ================================================================
        
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
            # ✅ FIX: Use strategy-specific target instead of global S/R
            result['stop_loss'], result['target'] = self.calculate_simple_stops(
                entry_price=spot_price,
                signal_type=result['signal'],
                support=strategy_support,  # ✅ Strategy-specific support
                resistance=strategy_target,  # ✅ Strategy-specific target (pattern projection)
                atr=None,  # ATR not available
                confidence=result['confidence']
            )
            result['reasoning'].append("⚠️ Using percentage-based stops (ATR unavailable)")
        
        # Step 8: Validate Risk:Reward Ratio
        result = self.validate_risk_reward(result)
        
        return result
    
    def _detect_double_bottom_top(self, df, fvgs):
        """Detect double bottom or double top pattern"""
        # ✅ ADD: Entry debug log
        self.logger.info(f"🔥 PATTERN DETECTION STARTED: df length={len(df)}, FVGs={len(fvgs)}")
        
        # ✅ CHANGED: Lowered from 30 to 20 candles
        if len(df) < 20:
            self.logger.info(f"❌ EARLY EXIT: Not enough data (df length={len(df)} < 20)")
            return None
        
        # ✅ ADD: Success log after passing threshold
        self.logger.info(f"✅ Passed length check, processing...")
        
        recent = df.tail(50).reset_index(drop=True)
        # ✅ ADD: Log how many candles we're analyzing
        self.logger.info(f"   Analyzing last {len(recent)} candles for swing detection")
        
        # Find swing lows for double bottom
        swing_lows = []
        lookback = 3  # ✅ REDUCED: More sensitive to swing points
        for i in range(lookback, len(recent) - lookback):
            current_low = recent['low'].iloc[i]
            
            # Check if this is a local minimum
            is_swing_low = True
            
            # Check left side (previous 5 candles)
            for j in range(i - lookback, i):
                if recent['low'].iloc[j] <= current_low:
                    is_swing_low = False
                    break
            
            # Check right side (next 5 candles)
            if is_swing_low:
                for j in range(i + 1, min(i + lookback + 1, len(recent))):
                    if recent['low'].iloc[j] <= current_low:
                        is_swing_low = False
                        break
            
            if is_swing_low:
                swing_lows.append({
                    'index': i,
                    'price': current_low
                })
        
        # ✅ ADD DEBUG LOGGING
        self.logger.info(f"📍 Swing Low Detection: Found {len(swing_lows)} swing lows")
        if swing_lows:
            prices = [f"{sl['price']:.2f}" for sl in swing_lows]
            self.logger.info(f"   Prices: {', '.join(prices)}")

        # Check for double bottom (two lows within 0.2% of each other)
        for i in range(len(swing_lows) - 1):
            for j in range(i + 1, len(swing_lows)):
                level_1 = swing_lows[i]['price']
                level_2 = swing_lows[j]['price']
                
                diff_pct = abs((level_2 - level_1) / level_1) * 100

                if diff_pct < 2.5:
                    # ✅ ADD DEBUG LOGGING
                    self.logger.info(
                        f"🔍 Double bottom candidate: "
                        f"Low1={level_1:.2f} at idx {swing_lows[i]['index']}, "
                        f"Low2={level_2:.2f} at idx {swing_lows[j]['index']}, "
                        f"Difference={diff_pct:.2f}%"
                    )
                                    
                    # Check if FVG exists NEAR this level (within 1%)
                    for fvg in fvgs:
                        fvg_mid = (fvg['top'] + fvg['bottom']) / 2
                        distance_pct = abs((level_1 - fvg_mid) / level_1) * 100

                        # ✅ ADD DEBUG LOGGING BEFORE THE CHECK
                        self.logger.info(
                            f"   Checking FVG: Type={fvg['type']}, "
                            f"Mid={fvg_mid:.2f}, Distance={distance_pct:.2f}%"
                        )
                        
                        if (fvg['type'] == 'BULLISH' and distance_pct < 5.0):
                            self.logger.info(
                                f"✅ MATCH FOUND: Double bottom at {level_1:.2f}/{level_2:.2f} "
                                f"with bullish FVG at {fvg_mid:.2f}"
                            )
                                                    
                            # Calculate neckline (highest high between the two bottoms)
                            start_idx = swing_lows[i]['index']
                            end_idx = swing_lows[j]['index']

                            neckline = recent['high'].iloc[start_idx:end_idx+1].max()
                            
                            return {
                                'type': 'DOUBLE_BOTTOM',
                                'level_1': level_1,
                                'level_2': level_2,
                                'neckline': neckline,
                                'fvg': fvg,
                                'expected_direction': 'BULLISH'
                            }
        
        # Find swing highs for double top
        swing_highs = []
        lookback = 3  # ✅ REDUCED: More sensitive to swing points
        for i in range(lookback, len(recent) - lookback):
            current_high = recent['high'].iloc[i]
            
            # Check if this is a local maximum
            is_swing_high = True
            
            # Check left side (previous 5 candles)
            for j in range(i - lookback, i):
                if recent['high'].iloc[j] >= current_high:
                    is_swing_high = False
                    break
            
            # Check right side (next 5 candles)
            if is_swing_high:
                for j in range(i + 1, min(i + lookback + 1, len(recent))):
                    if recent['high'].iloc[j] >= current_high:
                        is_swing_high = False
                        break
            
            if is_swing_high:
                swing_highs.append({
                    'index': i,
                    'price': current_high
                })
        
        # ✅ ADD DEBUG LOGGING
        self.logger.info(f"📍 Swing High Detection: Found {len(swing_highs)} swing highs")
        if swing_highs:
            prices = [f"{sh['price']:.2f}" for sh in swing_highs]
            self.logger.info(f"   Prices: {', '.join(prices)}")
        
        # Check for double top
        for i in range(len(swing_highs) - 1):
            for j in range(i + 1, len(swing_highs)):
                level_1 = swing_highs[i]['price']
                level_2 = swing_highs[j]['price']
                
                diff_pct = abs((level_2 - level_1) / level_1) * 100
                
                if diff_pct < 2.5:
                    # ✅ ADD DEBUG LOGGING
                    self.logger.info(
                        f"🔍 Double top candidate: "
                        f"High1={level_1:.2f} at idx {swing_highs[i]['index']}, "
                        f"High2={level_2:.2f} at idx {swing_highs[j]['index']}, "
                        f"Difference={diff_pct:.2f}%"
                    )
                    
                    # Check if FVG exists NEAR this level (within 1%)
                    for fvg in fvgs:
                        fvg_mid = (fvg['top'] + fvg['bottom']) / 2
                        distance_pct = abs((level_1 - fvg_mid) / level_1) * 100
                        
                        # ✅ ADD: Debug log for FVG checking
                        self.logger.info(
                            f"   Checking FVG: Type={fvg['type']}, "
                            f"Mid={fvg_mid:.2f}, Distance={distance_pct:.2f}%"
                        )
                        
                        if (fvg['type'] == 'BEARISH' and distance_pct < 5.0):
                            # ✅ ADD: Match found log
                            self.logger.info(
                                f"✅ MATCH FOUND: Double top at {level_1:.2f}/{level_2:.2f} "
                                f"with bearish FVG at {fvg_mid:.2f}"
                            )
                                                    
                            # Calculate neckline (lowest low between the two tops)
                            start_idx = swing_highs[i]['index']
                            end_idx = swing_highs[j]['index']
                            neckline = recent['low'].iloc[start_idx:end_idx+1].min()
                            
                            return {
                                'type': 'DOUBLE_TOP',
                                'level_1': level_1,
                                'level_2': level_2,
                                'neckline': neckline,
                                'fvg': fvg,
                                'expected_direction': 'BEARISH'
                            }
        
        return None
    
    def _check_breakout(self, df, pattern):
        """Check if breakout occurred with minimum distance confirmation"""
        if len(df) < 5:
            return None
        
        recent = df.tail(10)
        current_price = df['close'].iloc[-1]
        
        # ✅ Require minimum 0.3% move beyond neckline (avoid false breakouts)
        min_breakout_distance = 0.003  # 0.3%
        
        if pattern['expected_direction'] == 'BULLISH':
            # Calculate breakout percentage
            breakout_pct = (current_price - pattern['neckline']) / pattern['neckline']
            
            # Need CLEAR close above neckline (at least 0.3%)
            if breakout_pct > min_breakout_distance:
                
                # ✅ Optional: Check volume if available
                volume_confirmed = True
                if 'volume' in df.columns and len(df) >= 10:
                    recent_avg_vol = recent['volume'].iloc[:-1].mean()
                    current_vol = df['volume'].iloc[-1]
                    volume_confirmed = current_vol > recent_avg_vol * 1.1  # 10% above average
                
                return {
                    'confirmed': True,
                    'neckline': pattern['neckline'],
                    'breakout_pct': breakout_pct * 100,  # For logging
                    'volume_confirmed': volume_confirmed
                }
        else:
            # Bearish breakout
            breakout_pct = (pattern['neckline'] - current_price) / pattern['neckline']
            
            # Need CLEAR close below neckline
            if breakout_pct > min_breakout_distance:
                
                volume_confirmed = True
                if 'volume' in df.columns and len(df) >= 10:
                    recent_avg_vol = recent['volume'].iloc[:-1].mean()
                    current_vol = df['volume'].iloc[-1]
                    volume_confirmed = current_vol > recent_avg_vol * 1.1
                
                return {
                    'confirmed': True,
                    'neckline': pattern['neckline'],
                    'breakout_pct': breakout_pct * 100,
                    'volume_confirmed': volume_confirmed
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
        
        else:
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
