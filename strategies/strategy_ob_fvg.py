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
        
        # Step 2: Detect FVGs on 15min chart
        fvgs = self.fvg_detector.detect(df_15min)

        # ✅ TRADER'S RULE: Accept EITHER OB OR FVG (not both required)
        if not order_blocks and not fvgs:
            result['reasoning'].append("No Order Blocks or FVGs detected")
            return result

        # Step 3: Find best zone (OB+FVG confluence OR standalone OB/FVG)
        confluence_zones = []

        # Option A: Look for OB+FVG confluence (best case)
        if order_blocks and fvgs:
            confluence_zones = self._find_confluence(order_blocks, fvgs, spot_price)
            if confluence_zones:
                result['reasoning'].append("✓ OB+FVG confluence found (highest probability)")

        # Option B: If no confluence, use standalone OB or FVG
        if not confluence_zones:
            if order_blocks:
                # Use strongest order block
                best_ob = max(order_blocks, key=lambda x: x['strength'])
                # Check if near price (within 4%)
                distance_pct = abs((spot_price - (best_ob['high'] + best_ob['low'])/2) / spot_price) * 100
                if distance_pct < 4.0:
                    confluence_zones.append({
                        'direction': best_ob['type'],
                        'zone_low': best_ob['low'],
                        'zone_high': best_ob['high'],
                        'ob_strength': best_ob['strength'],
                        'distance_pct': distance_pct,
                        'source': 'OB_ONLY'
                    })
                    result['reasoning'].append("✓ Standalone Order Block detected (no FVG)")
            
            if not confluence_zones and fvgs:
                # Use nearest FVG
                best_fvg = min(fvgs, key=lambda x: abs(spot_price - (x['top'] + x['bottom'])/2))
                distance_pct = abs((spot_price - (best_fvg['top'] + best_fvg['bottom'])/2) / spot_price) * 100
                if distance_pct < 4.0:
                    confluence_zones.append({
                        'direction': best_fvg['type'],
                        'zone_low': best_fvg['bottom'],
                        'zone_high': best_fvg['top'],
                        'ob_strength': 0.5,  # Default strength for FVG-only
                        'distance_pct': distance_pct,
                        'source': 'FVG_ONLY'
                    })
                    result['reasoning'].append("✓ Standalone FVG detected (no OB)")

        if not confluence_zones:
            result['reasoning'].append("No OB or FVG near current price (within 4%)")
            return result

        # Take the best zone
        best_zone = confluence_zones[0]
        result['setup_detected'] = True
        
        # Step 4: Check for retest using 5min data for precision
        retest_result = self.retest_detector.check_retest(
            df=df_5min,
            zone_high=best_zone['zone_high'],
            zone_low=best_zone['zone_low'],
            expected_direction=best_zone['direction']
        )
        
        result['retest_confirmed'] = retest_result['retest_confirmed']
        result['reasoning'].append(retest_result['reasoning'])
        
        # ✅ TRADER'S FIX: Retest is OPTIONAL (bonus confidence)
        if not result['retest_confirmed']:
            result['reasoning'].append("⚠️ Early entry - retest not confirmed yet")
            # Don't return - allow trade without retest
        
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
        if best_zone.get('source') == 'OB_ONLY' or best_zone.get('source') == 'FVG_ONLY':
            base_confidence = 45  # Lower for standalone setups
            result['reasoning'].append("Base confidence: 45% (standalone setup)")
        else:
            base_confidence = 52  # Higher for confluence
            result['reasoning'].append("Base confidence: 52% (OB+FVG confluence)")
        
        # Add retest bonus if confirmed
        if result['retest_confirmed']:
            base_confidence += 8
            result['reasoning'].append("✓ Retest confirmed (+8%)")
        
        # Factor 1: Order Block Strength (0-8 points, not 0-10)
        ob_strength_score = int(best_zone['ob_strength'] * 8)
        base_confidence += ob_strength_score
        
        # Factor 2: Candlestick Pattern (reduced from max 15 to max 10)
        candlestick_score = min(10, candlestick_boost['confidence_boost'])
        base_confidence += candlestick_score
        
        # Factor 3: Trend Alignment (reduced from 10 to 6)
        trend_aligned = False
        if (best_zone['direction'] == 'BULLISH' and overall_trend == 'Bullish') or \
           (best_zone['direction'] == 'BEARISH' and overall_trend == 'Bearish'):
            base_confidence += 6
            trend_aligned = True
            result['reasoning'].append("Aligned with overall market trend (+6)")
        
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
        
        # Cap confidence at 70 (not 100)
        result['confidence'] = max(35, min(70, base_confidence))

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
                support=strategy_support,  # ✅ Strategy-specific support (zone boundary)
                resistance=strategy_target,  # ✅ Strategy-specific target (zone projection)
                atr=None,  # ATR not available
                confidence=result['confidence']
            )
            result['reasoning'].append("⚠️ Using percentage-based stops (ATR unavailable)")
        
        # Step 8: Validate Risk:Reward Ratio
        result = self.validate_risk_reward(result)
        
        return result
    
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
