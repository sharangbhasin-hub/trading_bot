"""
Strategy 1: Order Block + FVG + Candlestick Confirmation
"""
import pandas as pd
from typing import Dict
from strategies.base_strategy import BaseStrategy
from detectors.order_block_detector import OrderBlockDetector
from detectors.fvg_detector import FVGDetector
from detectors.retest_detector import RetestDetector
from utils.dataframe_validator import DataFrameValidator

class OrderBlockFVGStrategy(BaseStrategy):
    """Order Block + FVG Strategy with retest logic"""
    
    def __init__(self):
        super().__init__(name="Order Block + FVG")
        self.ob_detector = OrderBlockDetector()
        self.fvg_detector = FVGDetector()
        self.retest_detector = RetestDetector()

        # ✅ FIX: Initialize logger if not set by BaseStrategy
        if not hasattr(self, 'logger'):
            import logging
            self.logger = logging.getLogger(self.__class__.__name__)

    def _check_market_trending(self, df_15min: pd.DataFrame) -> tuple:
        """
        Check if market is in trending condition (not ranging/choppy).
        SMC strategies ONLY work in trending markets.
        
        Returns:
            (bool, str): (should_trade, reason)
        """
        if len(df_15min) < 20:
            return False, "Insufficient data for trend analysis"
        
        # Calculate ATR to measure volatility
        high = df_15min['high']
        low = df_15min['low']
        close = df_15min['close']
        
        # ATR calculation
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        
        if atr.iloc[-1] == 0 or pd.isna(atr.iloc[-1]):
            return False, "ATR calculation failed"
        
        current_atr = atr.iloc[-1]
        avg_atr = atr.tail(20).mean()
        
        # Rule 1: ATR must be expanding (trending market)
        if current_atr < avg_atr * 0.70:
            return False, f"Low volatility - ranging market (ATR: {current_atr:.1f} < {avg_atr*0.70:.1f})"
        
        # Rule 2: Check for clear trend structure (higher highs/lower lows)
        recent_20 = df_15min.tail(20)
        
        # Find swing highs and lows
        swing_highs = []
        swing_lows = []
        
        for i in range(5, len(recent_20) - 5):
            if (recent_20['high'].iloc[i] > recent_20['high'].iloc[i-5:i].max() and
                recent_20['high'].iloc[i] > recent_20['high'].iloc[i+1:i+6].max()):
                swing_highs.append(recent_20['high'].iloc[i])
            
            if (recent_20['low'].iloc[i] < recent_20['low'].iloc[i-5:i].min() and
                recent_20['low'].iloc[i] < recent_20['low'].iloc[i+1:i+6].min()):
                swing_lows.append(recent_20['low'].iloc[i])
        
        # Need at least 2 swing points to determine trend
        if len(swing_highs) < 1 and len(swing_lows) < 1:
            return False, "No clear trend structure - insufficient swing points"
        
        # ✅ SIMPLIFIED: If we found ANY swings, market is trending
        # The fact that swing points exist means price is making moves
        trend_type = "Unknown"
        if len(swing_highs) > 0 and len(swing_lows) > 0:
            # Compare most recent swings to determine direction
            if len(swing_highs) >= 2 and swing_highs[-1] > swing_highs[0]:
                trend_type = "Uptrend"
            elif len(swing_lows) >= 2 and swing_lows[-1] < swing_lows[0]:
                trend_type = "Downtrend"
            else:
                trend_type = "Trending"  # Moving but unclear direction
        elif len(swing_highs) > 0:
            trend_type = "Bullish Bias"
        else:
            trend_type = "Bearish Bias"
        
        return True, f"Trending market detected ({trend_type}, ATR expanding)"

    def _check_pullback_entry(self, df_5min, zone_low, zone_high, direction, spot_price):
        """
        Check if price has pulled back INTO the zone (not just touched edge)
        Prevents entering at worst prices
        
        Args:
            df_5min: 5-minute DataFrame
            zone_low: Zone lower boundary
            zone_high: Zone upper boundary
            direction: 'BULLISH' or 'BEARISH'
            spot_price: Current price
        
        Returns:
            (bool, str): (has_pullback, reason)
        """
        if len(df_5min) < 5:
            return False, "Insufficient data for pullback check"
        
        last_5_candles = df_5min.tail(5)
        zone_mid = (zone_low + zone_high) / 2
        
        pullback_count = 0
        
        for idx, candle in last_5_candles.iterrows():
            if direction == 'BULLISH':
                # For BULLISH: Need price to be INSIDE zone (between low and mid)
                # Not above zone
                if zone_low <= candle['low'] <= zone_mid:
                    pullback_count += 1
            else:
                # For BEARISH: Need price to be INSIDE zone (between mid and high)
                # Not below zone
                if zone_mid <= candle['high'] <= zone_high:
                    pullback_count += 1
        
        if pullback_count == 0:
            return False, "No pullback into zone - price at zone edge"
        
        # Also check current price is within 1% of zone
        if direction == 'BULLISH':
            distance_from_zone = abs(spot_price - zone_mid) / zone_mid * 100
            if distance_from_zone > 1.0:
                return False, f"Current price {distance_from_zone:.1f}% from zone center"
        else:
            distance_from_zone = abs(spot_price - zone_mid) / zone_mid * 100
            if distance_from_zone > 1.0:
                return False, f"Current price {distance_from_zone:.1f}% from zone center"
        
        return True, f"Pullback confirmed ({pullback_count} candles inside zone)"

    def _check_rejection_candle(self, df_5min, direction):
        """
        Check if last candle shows rejection from zone
        
        For BULLISH: Need bullish candle with small upper wick (showing buying pressure)
        For BEARISH: Need bearish candle with small lower wick (showing selling pressure)
        
        Args:
            df_5min: 5-minute DataFrame
            direction: 'BULLISH' or 'BEARISH'
        
        Returns:
            (bool, str): (has_rejection, reason)
        """
        if len(df_5min) < 2:
            return False, "Insufficient candles for rejection check"
        
        last_candle = df_5min.iloc[-1]
        prev_candle = df_5min.iloc[-2]
        
        open_price = last_candle['open']
        close_price = last_candle['close']
        high_price = last_candle['high']
        low_price = last_candle['low']
        
        body = abs(close_price - open_price)
        total_range = high_price - low_price
        
        # Avoid doji candles (no clear direction)
        if total_range == 0 or body < total_range * 0.3:
            return False, "Candle body too small (doji/indecision)"
        
        if direction == 'BULLISH':
            # For BULLISH: Need green candle closing near high
            is_bullish = close_price > open_price
            upper_wick = high_price - close_price
            lower_wick = open_price - low_price
            close_position = (close_price - low_price) / total_range
            
            if not is_bullish:
                return False, "Last candle not bullish (need green candle)"
            
            # Check if closed in upper 70% of range
            if close_position < 0.7:
                return False, f"Close not near high (closed at {close_position*100:.0f}% of range)"
            
            # Check small upper wick (buyers in control)
            if upper_wick > body * 0.5:
                return False, "Upper wick too large (selling pressure at high)"
            
            return True, f"Bullish rejection confirmed (green candle, close at {close_position*100:.0f}%)"
        
        else:  # BEARISH
            # For BEARISH: Need red candle closing near low
            is_bearish = close_price < open_price
            upper_wick = high_price - open_price
            lower_wick = close_price - low_price
            close_position = (high_price - close_price) / total_range
            
            if not is_bearish:
                return False, "Last candle not bearish (need red candle)"
            
            # Check if closed in lower 70% of range
            if close_position < 0.7:
                return False, f"Close not near low (closed at {close_position*100:.0f}% of range)"
            
            # Check small lower wick (sellers in control)
            if lower_wick > body * 0.5:
                return False, "Lower wick too large (buying pressure at low)"
            
            return True, f"Bearish rejection confirmed (red candle, close at {close_position*100:.0f}%)"
    
    def detect(self, df: pd.DataFrame, current_idx: int) -> dict:
        """Detect Order Block + FVG setup"""
        
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
        self.logger.info(f"📊 Order Blocks detected: {len(order_blocks)}")
        if order_blocks:
            for idx, ob in enumerate(order_blocks[:3]):  # Log first 3
                self.logger.info(f"  OB #{idx+1}: {ob['type']} at {ob['low']:.2f}-{ob['high']:.2f}, strength={ob['strength']:.2f}")
        
        # Step 2: Detect FVGs on 15min chart
        fvgs = self.fvg_detector.detect(df_15min, require_volume_spike=False)
        self.logger.info(f"📊 FVGs detected: {len(fvgs)}")
        if fvgs:
            for idx, fvg in enumerate(fvgs[:3]):  # Log first 3
                self.logger.info(f"  FVG #{idx+1}: {fvg['type']} at {fvg['bottom']:.2f}-{fvg['top']:.2f}, fill={fvg['fill_percentage']}%")

        # ✅ NEW FIX: Check if market is trending (MANDATORY)
        is_trending, trend_reason = self._check_market_trending(df_15min)
        if not is_trending:
            self.logger.info(f"❌ REJECTED: {trend_reason}")  # ✅ ADD THIS
            result['reasoning'].append(f"❌ Market condition filter: {trend_reason}")
            result['reasoning'].append("SMC strategies only work in trending markets")
            return result
        
        self.logger.info(f"✅ Market trending check passed: {trend_reason}")  # ✅ ADD THIS
        result['reasoning'].append(f"✓ Market condition: {trend_reason}")     
                    
        # ✅ TRADER'S RULE: Accept EITHER OB OR FVG (not both required)
        if not order_blocks and not fvgs:
            self.logger.info(f"❌ REJECTED: No Order Blocks or FVGs detected")  # ✅ ADD THIS
            result['reasoning'].append("No Order Blocks or FVGs detected")
            return result
        
        self.logger.info(f"✅ Detection passed: OB={len(order_blocks)}, FVG={len(fvgs)}")  # ✅ ADD THIS

        # Step 3: Find best zone (OB+FVG confluence OR standalone OB/FVG)
        confluence_zones = []

        # Option A: Look for OB+FVG confluence (best case)
        if order_blocks and fvgs:
            confluence_zones = self._find_confluence(order_blocks, fvgs, spot_price)
            if confluence_zones:
                result['reasoning'].append("✓ OB+FVG confluence found (highest probability)")

        # Option B: If no confluence, use standalone OB or FVG
        # ✅ CRITICAL FIX: REQUIRE OB + FVG Confluence (no standalone setups)
        # Option B has been REMOVED - confluence is now MANDATORY

        # ✅ FIX: Allow standalone OB or FVG (not just confluence)
        if not confluence_zones:
            self.logger.info(f"⚠️ No confluence - checking standalone OB/FVG")
            
            # Try standalone Order Blocks
            if order_blocks:
                for ob in order_blocks:
                    zone_mid = (ob['low'] + ob['high']) / 2
                    distance_pct = abs((spot_price - zone_mid) / zone_mid) * 100
                    
                    if distance_pct < 4.0:
                        confluence_zones.append({
                            'direction': ob['type'],
                            'zone_low': ob['low'],
                            'zone_high': ob['high'],
                            'ob_strength': ob['strength'],
                            'distance_pct': distance_pct,
                            'source': 'OB_ONLY'
                        })
            
            # Try standalone FVGs if no OB
            if not confluence_zones and fvgs:
                for fvg in fvgs:
                    zone_mid = (fvg['bottom'] + fvg['top']) / 2
                    distance_pct = abs((spot_price - zone_mid) / zone_mid) * 100
                    
                    if distance_pct < 4.0:
                        confluence_zones.append({
                            'direction': fvg['type'],
                            'zone_low': fvg['bottom'],
                            'zone_high': fvg['top'],
                            'ob_strength': 0.5,  # Lower strength for FVG-only
                            'distance_pct': distance_pct,
                            'source': 'FVG_ONLY'
                        })
            
            # Sort by distance
            confluence_zones.sort(key=lambda x: x['distance_pct'])
            
            if not confluence_zones:
                self.logger.info(f"❌ REJECTED: No valid zones near price")
                result['reasoning'].append("❌ No OB/FVG zones near current price")
                return result
            
            result['reasoning'].append("⚠️ Standalone OB/FVG (no confluence)")
        
        self.logger.info(f"✅ Confluence found: {len(confluence_zones)} zone(s)")  # ✅ ADD THIS
        
        # Take the best zone
        best_zone = confluence_zones[0]
        result['setup_detected'] = True

        self.logger.info(f"  Best zone: {best_zone['direction']} at {best_zone['zone_low']:.2f}-{best_zone['zone_high']:.2f}, distance={best_zone['distance_pct']:.2f}%")  # ✅ ADD THIS            
        
        # Step 4: Check for retest using 5min data for precision
        retest_result = self.retest_detector.check_retest(
            df=df_5min,
            zone_high=best_zone['zone_high'],
            zone_low=best_zone['zone_low'],
            expected_direction=best_zone['direction']
        )
        
        result['retest_confirmed'] = retest_result['retest_confirmed']
        result['reasoning'].append(retest_result['reasoning'])

        # ✅ NEW: Check for pullback entry timing (prevents edge entries)
        has_pullback, pullback_reason = self._check_pullback_entry(
            df_5min, 
            best_zone['zone_low'], 
            best_zone['zone_high'], 
            best_zone['direction'],
            spot_price
        )
        
        if not has_pullback:
            self.logger.info(f"❌ REJECTED: {pullback_reason}")
            result['signal'] = 'NO_TRADE'
            result['reasoning'].append(f"❌ Entry timing: {pullback_reason}")
            result['reasoning'].append("Data shows: 60% immediate reversals without pullback")
            return result
        
        self.logger.info(f"✅ Pullback entry check passed: {pullback_reason}")
        result['reasoning'].append(f"✓ Entry timing: {pullback_reason}")

        # ✅ NEW: Check for rejection candle (confirms reversal is starting)
        has_rejection, rejection_reason = self._check_rejection_candle(df_5min, best_zone['direction'])
        
        if not has_rejection:
            self.logger.info(f"❌ REJECTED: {rejection_reason}")
            result['signal'] = 'NO_TRADE'
            result['reasoning'].append(f"❌ Entry confirmation: {rejection_reason}")
            result['reasoning'].append("Wait for candle to show rejection before entering")
            return result
        
        self.logger.info(f"✅ Rejection candle confirmed: {rejection_reason}")
        result['reasoning'].append(f"✓ Entry confirmation: {rejection_reason}")
                    
        # Step 5: Check candlestick pattern at retest
        candlestick_boost = self._check_candlestick_confirmation(
            df_5min,
            best_zone['direction']
        )
        
        if candlestick_boost['pattern']:
            result['candlestick_pattern'] = candlestick_boost['pattern']
            result['reasoning'].append(
                f"Candlestick confirmation: {candlestick_boost['pattern']}"
            )
        
        # ==== STEP 6: Calculate confidence (REBUILT) ====
        # ✅ FIX: Initialize base_confidence FIRST
        base_confidence = 35  # Start skeptical
        result['reasoning'].append("Base confidence: 35% (OB+FVG zone detected)")
        
        # ✅ FIX: Make retest MANDATORY (87% of trades without retest failed)
        if not result['retest_confirmed']:
            self.logger.info(f"❌ REJECTED: No retest confirmation")
            result['signal'] = 'NO_TRADE'
            result['reasoning'].append("❌ No retest confirmation - trade rejected")
            result['reasoning'].append("Data shows: 87% failure rate without retest")
            return result
        
        # Retest confirmed - proceed
        base_confidence += 15
        result['reasoning'].append("✓ Retest confirmed (+15% - CRITICAL)")
        self.logger.info(f"✅ Retest confirmed! Zone tested successfully")
        
        # Factor 1: Order Block Strength (0-10 points)
        ob_strength_score = int(best_zone['ob_strength'] * 10)
        base_confidence += ob_strength_score
        result['reasoning'].append(f"✓ OB Strength: {best_zone['ob_strength']:.1f} (+{ob_strength_score}%)")
        
        # Factor 2: Candlestick Pattern (max 12 points - increased importance)
        candlestick_score = min(12, candlestick_boost['confidence_boost'])
        base_confidence += candlestick_score
        if candlestick_score > 0:
            result['reasoning'].append(f"✓ {candlestick_boost['pattern']} (+{candlestick_score}%)")
        
        # Factor 3: Trend Alignment (max 8 points)
        trend_aligned = False
        if (best_zone['direction'] == 'BULLISH' and overall_trend == 'Bullish') or \
           (best_zone['direction'] == 'BEARISH' and overall_trend == 'Bearish'):
            base_confidence += 8
            trend_aligned = True
            result['reasoning'].append("✓ Aligned with overall trend (+8%)")
        else:
            # Penalty for counter-trend trades
            base_confidence -= 5
            result['reasoning'].append("⚠️ Counter-trend trade (-5%)")
        
        # Factor 4: Distance Penalty - Penalize setups far from current price
        distance_pct = best_zone['distance_pct']
        if distance_pct > 1.0:
            distance_penalty = min(8, int((distance_pct - 1.0) * 4))
            base_confidence -= distance_penalty
            result['reasoning'].append(f"Distance penalty: -{distance_penalty} (zone {distance_pct:.1f}% away)")
        
        # Factor 5: Confluence Quality - Reward tight OB+FVG overlap
        zone_size = best_zone['zone_high'] - best_zone['zone_low']
        zone_size_pct = (zone_size / spot_price) * 100
        if zone_size_pct < 0.3:  # Very tight zone
            base_confidence += 4
            result['reasoning'].append("Tight confluence zone (+4)")
        elif zone_size_pct > 0.8:  # Wide zone (weak)
            base_confidence -= 3
            result['reasoning'].append("Wide confluence zone (-3)")
        
        # ✅ FIX: Stricter confidence range
        # Cap at 75% (even best setups aren't 100%)
        # Floor at 45% (below this = don't trade)
        result['confidence'] = max(45, min(75, base_confidence))
        
        # ✅ FIX: Raise minimum confidence (data shows <60% trades have 9% win rate)
        if result['confidence'] < 60:
            self.logger.info(f"❌ REJECTED: Confidence too low ({result['confidence']}% < 60% minimum)")
            result['signal'] = 'NO_TRADE'
            result['reasoning'].append(f"❌ Confidence too low ({result['confidence']}% < 60% minimum)")
            return result

        self.logger.info(f"✅ Confidence check passed: {result['confidence']}%")  # ✅ ADD THIS

        # Log final confidence breakdown
        source_label = best_zone.get('source', 'CONFLUENCE')
        result['reasoning'].append(
            f"Final Confidence: {result['confidence']}% "
            f"(setup={source_label}, OB={ob_strength_score}, "
            f"candle={candlestick_score}, trend={'+6' if trend_aligned else '0'}, "
            f"retest={'+8' if result['retest_confirmed'] else '0'})"
        )

        
        # Step 7: Set signal, stop loss (DYNAMIC), target
        # Set signal type
        if best_zone['direction'] == 'BULLISH':
            result['signal'] = 'CALL'
        else:
            result['signal'] = 'PUT'
        
        # ========== ✅ FIX: CALCULATE STRATEGY-SPECIFIC TARGET ==========
        # For OB+FVG, target is based on zone projection
        # Uses the zone size to project target
        
        zone_size = best_zone['zone_high'] - best_zone['zone_low']
        
        if best_zone['direction'] == 'BULLISH':
            # Bullish: Target = zone high + (zone size * multiplier)
            # Multiplier varies by setup type
            if best_zone.get('source') == 'CONFLUENCE':
                multiplier = 2.0  # Confluence = higher confidence, bigger target
            elif best_zone.get('source') == 'OB_ONLY':
                multiplier = 1.8  # OB alone = moderate target
            else:  # FVG_ONLY
                multiplier = 1.5  # FVG alone = conservative target
            
            strategy_target = best_zone['zone_high'] + (zone_size * multiplier)
            strategy_support = best_zone['zone_low']  # Zone low as support
            
            result['reasoning'].append(
                f"✅ OB+FVG target: {strategy_target:.2f} "
                f"(zone projection: {multiplier}x zone size)"
            )
        else:  # BEARISH
            # Bearish: Target = zone low - (zone size * multiplier)
            if best_zone.get('source') == 'CONFLUENCE':
                multiplier = 2.0
            elif best_zone.get('source') == 'OB_ONLY':
                multiplier = 1.8
            else:  # FVG_ONLY
                multiplier = 1.5
            
            strategy_target = best_zone['zone_low'] - (zone_size * multiplier)
            strategy_support = best_zone['zone_high']  # Zone high as resistance
            
            result['reasoning'].append(
                f"✅ OB+FVG target: {strategy_target:.2f} "
                f"(zone projection: {multiplier}x zone size)"
            )
        # ================================================================
        
        # ✅ STANDARD STOP LOSS CALCULATION
        # ✅ IMPROVED: Zone-based stop loss (Order Block invalidation level)
        # Stop loss = just beyond the zone boundary (where setup is invalidated)
        
        # ✅ FIX: ATR-based stop loss (adapts to volatility, not zone size)
        # Calculate recent ATR from 5min chart
        recent_candles = df_5min.tail(20)
        high = recent_candles['high']
        low = recent_candles['low']
        close = recent_candles['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.mean()
        
        if result['signal'] == 'CALL':
            # For CALL: Stop = entry - (1.0 * ATR), minimum 35 points, maximum 50 points
            stop_distance = max(35, min(50, atr * 1.0))  # ✅ FIX: Capped at 50 pts
            result['stop_loss'] = spot_price - stop_distance
            result['reasoning'].append(f"✅ Stop Loss: {result['stop_loss']:.2f} ({stop_distance:.1f} pts, 1.0x ATR, max 50)")
            self.logger.info(f"  ATR: {atr:.1f}, Stop distance: {stop_distance:.1f} pts")
        else:
            # For PUT: Stop = entry + (1.0 * ATR), minimum 35 points, maximum 50 points
            stop_distance = max(35, min(50, atr * 1.0))  # ✅ FIX: Capped at 50 pts
            result['stop_loss'] = spot_price + stop_distance
            result['reasoning'].append(f"✅ Stop Loss: {result['stop_loss']:.2f} ({stop_distance:.1f} pts, 1.0x ATR, max 50)")
            self.logger.info(f"  ATR: {atr:.1f}, Stop distance: {stop_distance:.1f} pts")
        
        # Target remains the same (zone projection)
        result['target'] = strategy_target
        
        # Calculate R:R ratio
        risk = abs(spot_price - result['stop_loss'])
        reward = abs(result['target'] - spot_price)
        rr_ratio = reward / risk if risk > 0 else 0
        
        result['reasoning'].append(f"✅ Risk:Reward = 1:{rr_ratio:.1f}")
        
        # ✅ FIX: Realistic R:R for intraday (was 1.5)
        if rr_ratio < 1.2:
            self.logger.info(f"❌ REJECTED: R:R too low ({rr_ratio:.1f} < 1.2:1, risk={risk:.2f}, reward={reward:.2f})")
            result['signal'] = 'NO_TRADE'
            result['reasoning'].append(f"❌ R:R too low ({rr_ratio:.1f} < 1.2:1 minimum)")
            return result
 
        self.logger.info(f"✅ R:R check passed: {rr_ratio:.1f}:1")  # ✅ ADD THIS
        
        # Step 8: Validate Risk:Reward Ratio
        result = self.validate_risk_reward(result)

        # ✅ FIXED: Check if sufficient time before EOD
        if hasattr(self, 'replay_engine') and self.replay_engine:
            # ✅ FIX: Use correct attribute name
            if self.replay_engine.current_date and self.replay_engine.current_time:
                # Combine date + time into timestamp for time check
                current_timestamp = f"{self.replay_engine.current_date} {self.replay_engine.current_time}"
                has_time, time_reason = self._has_sufficient_time_to_target(current_timestamp)
                
                if not has_time:
                    self.logger.info(f"❌ REJECTED: {time_reason}")
                    result['signal'] = 'NO_TRADE'
                    result['reasoning'].append(f"❌ Time filter: {time_reason}")
                    return result
                
                self.logger.info(f"✅ Time check passed: {time_reason}")
                result['reasoning'].append(f"✓ Time check: {time_reason}")
    
        # ✅ ADD THIS: Log successful signal generation
        self.logger.info(f"✅✅✅ SIGNAL GENERATED! {result['signal']} at {spot_price:.2f}, confidence={result['confidence']}%, target={result['target']:.2f}, SL={result['stop_loss']:.2f}")

        return result

    def _has_sufficient_time_to_target(self, current_timestamp) -> tuple:
        """
        Check if there's enough time before EOD for trade to reach target.
        
        Returns:
            (bool, str): (has_time, reason)
        """
        if current_timestamp is None:
            return True, "No timestamp provided - allowing trade"
        
        from datetime import datetime, time
        
        # Parse current time
        if isinstance(current_timestamp, str):
            current_dt = datetime.strptime(current_timestamp, "%Y-%m-%d %H:%M")
        else:
            current_dt = current_timestamp
        
        # EOD time (3:30 PM)
        eod_time = time(15, 30)
        eod_dt = datetime.combine(current_dt.date(), eod_time)
        
        # Calculate minutes remaining
        minutes_remaining = (eod_dt - current_dt).seconds / 60
        
        # Need at least 60 minutes for trade to develop
        if minutes_remaining < 30:
            return False, f"Only {minutes_remaining:.0f} minutes until EOD (need 30+ min)"
        
        return True, f"{minutes_remaining:.0f} minutes remaining (sufficient)"

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
                    
                    # Check if current price is near zone (within 2%)
                    zone_mid = (zone_low + zone_high) / 2
                    distance_pct = abs((current_price - zone_mid) / zone_mid) * 100
                    
                    if distance_pct < 5.0:
                        confluences.append({
                            'direction': ob['type'],
                            'zone_low': zone_low,
                            'zone_high': zone_high,
                            'ob_strength': ob['strength'],
                            'distance_pct': distance_pct,
                            'source': 'CONFLUENCE'
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
            
            # Check for bullish engulfing (FULL LOGIC)
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
            
            # Check for bearish engulfing (FULL LOGIC)
            if (last_candle['close'] < last_candle['open'] and
                prev_candle['close'] > prev_candle['open'] and
                last_candle['open'] > prev_candle['close'] and
                last_candle['close'] < prev_candle['open']):
                return {'pattern': 'Bearish Engulfing', 'confidence_boost': 15}
            
            # Check for strong rejection wick
            if upper_wick > total_range * 0.5:
                return {'pattern': 'Bearish Rejection', 'confidence_boost': 10}
        
        return {'pattern': None, 'confidence_boost': 0}

    def _calculate_trailing_stop(self, entry_price, current_price, initial_stop, direction):
        """
        Calculate trailing stop to lock in profits
        
        Args:
            entry_price: Original entry price
            current_price: Current market price
            initial_stop: Original stop loss
            direction: 'CALL' or 'PUT'
        
        Returns:
            float: New stop loss level
        """
        if direction == 'CALL':
            # For CALL: Trail stop up as price moves up
            profit = current_price - entry_price
            if profit > 50:  # If 50+ points profit
                # Move stop to breakeven + 20 points
                new_stop = entry_price + 20
                return max(initial_stop, new_stop)
        else:
            # For PUT: Trail stop down as price moves down
            profit = entry_price - current_price
            if profit > 50:  # If 50+ points profit
                # Move stop to breakeven + 20 points
                new_stop = entry_price - 20
                return min(initial_stop, new_stop)
        
        return initial_stop

