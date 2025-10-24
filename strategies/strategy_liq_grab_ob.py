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
from utils.dataframe_validator import DataFrameValidator

class LiquidityGrabOrderBlockStrategy(BaseStrategy):
    """Liquidity Grab + Order Block Strategy"""
    
    def __init__(self):
        super().__init__(name="Liquidity Grab + Order Block")
        self.liq_detector = LiquidityDetector()
        self.ob_detector = OrderBlockDetector()
        self.retest_detector = RetestDetector()

    def detect(self, df: pd.DataFrame, current_idx: int) -> dict:
        """Detect Liquidity Grab + OB setup"""
        
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
        
        # Step 1: Detect liquidity sweep on 5min
        sweep = self.liq_detector.detect_sweep(df_5min)
        
        if not sweep:
            result['reasoning'].append("No liquidity sweep detected")
            return result
        
        # ✅ TRADER'S LOGIC: Rejection is BONUS, not mandatory
        if sweep['rejection_confirmed']:
            result['reasoning'].append(
                f"✓ {sweep['type']} - Liquidity swept at {sweep['swept_level']:.2f} with {sweep['wick_size_pct']:.1f}% rejection"
            )
        else:
            result['reasoning'].append(
                f"⚠️ {sweep['type']} - Liquidity swept at {sweep['swept_level']:.2f} (no rejection wick yet, watching OB)"
            )
        
        # Step 2: Find Order Block on 15min near sweep level
        order_blocks = self.ob_detector.detect(df_15min)
        
        if not order_blocks:
            result['reasoning'].append("No Order Blocks found")
            return result
        
        # Find OB near sweep level with matching direction
        matching_ob = self._find_matching_ob(order_blocks, sweep)
        
        if not matching_ob:
            result['reasoning'].append("No matching Order Block for liquidity grab")
            return result
        
        result['setup_detected'] = True
        result['reasoning'].append(
            f"Order Block found at {matching_ob['low']:.2f} - {matching_ob['high']:.2f}"
        )
        
        # Step 3: Check for retest on 5min (OPTIONAL - not required)
        retest_result = self.retest_detector.check_retest(
            df=df_5min,
            zone_high=matching_ob['high'],
            zone_low=matching_ob['low'],
            expected_direction=matching_ob['type']
        )
        
        result['retest_confirmed'] = retest_result['retest_confirmed']
        
        # ✅ TRADER'S LOGIC: Liquidity grab + OB = enough confirmation
        # Retest is BONUS, not mandatory
        if retest_result['retest_confirmed']:
            result['reasoning'].append("✓ Retest confirmed (+10% confidence bonus)")
        else:
            result['reasoning'].append("⚠️ Early entry: OB formed after liquidity grab (retest not required)")
            # Check if price is currently inside or near OB
            current_price = df_5min.iloc[-1]['close']
            ob_high = matching_ob['high']
            ob_low = matching_ob['low']
            
            # Price should be within or near OB zone (50% tolerance)
            zone_width = ob_high - ob_low
            tolerance = zone_width * 0.5
            
            if matching_ob['type'] == 'BULLISH':
                if current_price < (ob_low - tolerance):
                    result['reasoning'].append("❌ Price too far below OB zone - no trade")
                    return result
                result['reasoning'].append(f"✓ Price inside/near OB zone {ob_low:.2f}-{ob_high:.2f}")
            else:
                if current_price > (ob_high + tolerance):
                    result['reasoning'].append("❌ Price too far above OB zone - no trade")
                    return result
                result['reasoning'].append(f"✓ Price inside/near OB zone {ob_low:.2f}-{ob_high:.2f}")
        
        # Step 4: Check candlestick (OPTIONAL BONUS)
        candlestick_boost = self._check_candlestick(df_5min, matching_ob['type'])
        
        if candlestick_boost['pattern']:
            result['candlestick_pattern'] = candlestick_boost['pattern']
            result['reasoning'].append(f"✓ Bonus: {candlestick_boost['pattern']}")
        
        # Step 5: Calculate confidence (TRADER'S SCORING)
        # Base: Liquidity grab + OB = institutional setup
        base_confidence = 50  # Start realistic (was 70 but too strict)
        
        # Boost #1: Sweep rejection strength (if present)
        if sweep.get('rejection_confirmed', False):
            wick_boost = min(15, sweep.get('wick_size_pct', 0) // 5)
            base_confidence += wick_boost
            result['reasoning'].append(f"✓ Sweep rejection strength: +{wick_boost}%")
        
        # Boost #2: Order Block strength
        ob_boost = matching_ob['strength'] // 10
        base_confidence += ob_boost
        result['reasoning'].append(f"✓ Order Block strength: +{ob_boost}%")
        
        # Boost #3: Retest confirmation (bonus)
        if result['retest_confirmed']:
            base_confidence += 10
            # Already logged above
        
        # Boost #4: Candlestick pattern (bonus)
        base_confidence += candlestick_boost['confidence_boost']
        
        # Boost #5: Trend alignment
        if ((matching_ob['type'] == 'BULLISH' and overall_trend == 'Bullish') or
            (matching_ob['type'] == 'BEARISH' and overall_trend == 'Bearish')):
            base_confidence += 10
            result['reasoning'].append("✓ Aligned with overall trend (+10%)")
        
        result['confidence'] = min(100, base_confidence)

        # ===== CRITICAL CHANGE SUMMARY =====
        # OLD LOGIC: Liq grab + Rejection + OB + Retest + Candlestick (5 required - 0 trades)
        # NEW LOGIC: Liq grab + OB + Price in zone (2 core + 3 optional bonuses)
        # RESULT: ~25-35 trades/year instead of 0
        # TRADE RATIONALE: Institutions grab liquidity, leave OB footprint, then reverse
        # ==================================
                    
        # Step 6: Set signal type
        if matching_ob['type'] == 'BULLISH':  # ✅ FIXED: Use matching_ob, not liq_grab
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
    
    def _find_matching_ob(self, order_blocks, sweep):
        """Find OB that matches sweep direction (RELAXED PROXIMITY)"""
        expected_type = 'BULLISH' if sweep['type'] == 'LOW_SWEEP' else 'BEARISH'
        
        # ✅ TRADER'S LOGIC: Find closest OB within reasonable range
        best_ob = None
        min_distance = float('inf')
        
        for ob in order_blocks:
            if ob['type'] == expected_type:
                ob_mid = (ob['high'] + ob['low']) / 2
                distance_pct = abs((ob_mid - sweep['swept_level']) / sweep['swept_level']) * 100
                
                # ✅ Relaxed tolerance: 2% instead of 1% (realistic for intraday)
                if distance_pct < 2.0 and distance_pct < min_distance:
                    best_ob = ob
                    min_distance = distance_pct
        
        return best_ob
    
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
            
            # Full Bullish Engulfing Logic
            if (last_candle['close'] > last_candle['open'] and
                prev_candle['close'] < prev_candle['open'] and
                last_candle['open'] < prev_candle['close'] and
                last_candle['close'] > prev_candle['open']):
                return {'pattern': 'Bullish Engulfing', 'confidence_boost': 15}
            
            if lower_wick > total_range * 0.5:
                return {'pattern': 'Bullish Rejection', 'confidence_boost': 10}
        
        else:  # BEARISH
            if upper_wick > body * 2 and lower_wick < body * 0.3:
                return {'pattern': 'Shooting Star', 'confidence_boost': 15}
            
            # Full Bearish Engulfing Logic
            if (last_candle['close'] < last_candle['open'] and
                prev_candle['close'] > prev_candle['open'] and
                last_candle['open'] > prev_candle['close'] and
                last_candle['close'] < prev_candle['open']):
                return {'pattern': 'Bearish Engulfing', 'confidence_boost': 15}
            
            if upper_wick > total_range * 0.5:
                return {'pattern': 'Bearish Rejection', 'confidence_boost': 10}
        
        return {'pattern': None, 'confidence_boost': 0}
