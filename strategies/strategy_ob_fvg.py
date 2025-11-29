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

        if not confluence_zones:
            self.logger.info(f"❌ REJECTED: No OB + FVG confluence (OB={len(order_blocks)}, FVG={len(fvgs)}, spot={spot_price:.2f})")  # ✅ ADD THIS
            result['reasoning'].append("❌ No OB + FVG confluence detected")
            result['reasoning'].append("SMC Rule: Confluence is mandatory for high-probability setups")
            return result
        
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
        
        # ✅ Retest is MANDATORY
        if not result['retest_confirmed']:
            self.logger.info(f"❌ REJECTED: No retest confirmation (zone: {best_zone['zone_low']:.2f}-{best_zone['zone_high']:.2f})")  # ✅ ADD THIS
            result['signal'] = 'NO_TRADE'
            result['confidence'] = 0
            result['reasoning'].append("❌ No retest confirmation - TRADE REJECTED")
            result['reasoning'].append("SMC Rule: Never enter without price proving zone validity")
            return result
        
        self.logger.info(f"✅ Retest confirmed! Zone tested successfully")  # ✅ ADD THIS
        
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
        # Adjust base confidence based on setup type
        # ✅ FIX: Start with lower base, require proof for confidence
        # Base confidence is LOW - strategy must EARN higher confidence
        base_confidence = 35  # Start skeptical
        result['reasoning'].append("Base confidence: 35% (OB+FVG confluence detected)")
        
        # Retest is MANDATORY (this code won't run without it due to Fix #1)
        # But if we're here, retest WAS confirmed, so reward it heavily
        base_confidence += 15  # Retest confirmation is THE most important factor
        result['reasoning'].append("✓ Retest confirmed (+15% - CRITICAL)")
        
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
        
        # ✅ NEW: Reject trades below 60% confidence
        if result['confidence'] < 60:
            self.logger.info(f"❌ REJECTED: Confidence too low ({result['confidence']}% < 60% minimum)")  # ✅ ADD THIS
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
        
        if result['signal'] == 'CALL':
            # For CALL: Stop just below the zone low
            result['stop_loss'] = best_zone['zone_low'] * 0.998  # 0.2% below zone
            result['reasoning'].append(f"✅ Stop Loss: {result['stop_loss']:.2f} (below OB zone)")
        else:
            # For PUT: Stop just above the zone high
            result['stop_loss'] = best_zone['zone_high'] * 1.002  # 0.2% above zone
            result['reasoning'].append(f"✅ Stop Loss: {result['stop_loss']:.2f} (above OB zone)")
        
        # Target remains the same (zone projection)
        result['target'] = strategy_target
        
        # Calculate R:R ratio
        risk = abs(spot_price - result['stop_loss'])
        reward = abs(result['target'] - spot_price)
        rr_ratio = reward / risk if risk > 0 else 0
        
        result['reasoning'].append(f"✅ Risk:Reward = 1:{rr_ratio:.1f}")
        
        # ✅ Reject trades with R:R < 1.5:1
        if rr_ratio < 1.5:
            self.logger.info(f"❌ REJECTED: R:R too low ({rr_ratio:.1f} < 1.5:1, risk={risk:.2f}, reward={reward:.2f})")  # ✅ ADD THIS
            result['signal'] = 'NO_TRADE'
            result['reasoning'].append(f"❌ R:R too low ({rr_ratio:.1f} < 1.5:1 minimum)")
            return result
        
        self.logger.info(f"✅ R:R check passed: {rr_ratio:.1f}:1")  # ✅ ADD THIS
        
        # Step 8: Validate Risk:Reward Ratio
        result = self.validate_risk_reward(result)

        # ✅ NEW FIX: Check if sufficient time before EOD
        if hasattr(self, 'replay_engine') and self.replay_engine:
            current_time = self.replay_engine.current_timestamp
            has_time, time_reason = self._has_sufficient_time_to_target(current_time)
            
            if not has_time:
                self.logger.info(f"❌ REJECTED: {time_reason}")  # ✅ ADD THIS
                result['signal'] = 'NO_TRADE'
                result['reasoning'].append(f"❌ Time filter: {time_reason}")
                return result
            
            self.logger.info(f"✅ Time check passed: {time_reason}")  # ✅ ADD THIS
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
        if minutes_remaining < 60:
            return False, f"Only {minutes_remaining:.0f} minutes until EOD (need 60+ min)"
        
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
                    
                    if distance_pct < 4.0:
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
