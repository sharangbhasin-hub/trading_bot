"""
Strategy 3: BOS (Break of Structure) + Retest + Candlestick
"""
import pandas as pd
from typing import Dict
from strategies.base_strategy import BaseStrategy
from detectors.structure_detector import StructureDetector
from detectors.retest_detector import RetestDetector

class BOSRetestStrategy(BaseStrategy):
    """BOS + Retest Strategy"""
    
    def __init__(self):
        super().__init__(name="BOS + Retest")
        self.structure_detector = StructureDetector()
        self.retest_detector = RetestDetector()
        self.retest_required = False  # ← ADD THIS LINE

    
    def analyze(self,
                df_5min: pd.DataFrame,
                df_15min: pd.DataFrame,
                df_1h: pd.DataFrame,
                df_4h: pd.DataFrame,
                spot_price: float,
                support: float,
                resistance: float,
                overall_trend: str) -> Dict:
        """Analyze for BOS + Retest setup"""
        
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
        
        # Step 1: Detect trend on 15min
        trend_info = self.structure_detector.detect_trend(df_15min)
        
        # ✅ FIX: Only skip if completely ranging
        # if trend_info.get('trend') == 'RANGING':
        #    result['reasoning'].append("Market is ranging, need clear trend for BOS")
        #    return result
        
        # Step 2: Detect BOS on 15min
        bos = self.structure_detector.detect_bos(df_15min)
        
        if not bos:
            result['reasoning'].append("No Break of Structure detected")
            return result
        
        result['setup_detected'] = True
        result['reasoning'].append(
            f"{bos['type']} BOS detected at {bos['broken_level']:.2f}"
        )
        
        # Step 3: Check for retest using 5min data
        zone_width = bos['broken_level'] * 0.002  # 0.2% zone
        zone_low = bos['broken_level'] - zone_width if bos['type'] == 'BULLISH' else bos['broken_level'] - zone_width
        zone_high = bos['broken_level'] + zone_width
        
        retest_result = self.retest_detector.check_retest(
            df=df_5min,
            zone_high=zone_high,
            zone_low=zone_low,
            expected_direction=bos['type']
        )
        
        # ✅ CRITICAL FIX: Make retest OPTIONAL, not required
        result['retest_confirmed'] = retest_result.get('retest_confirmed', False)
        result['reasoning'].append(retest_result.get('reasoning', 'Retest check performed'))
        
        # ✅ REMOVED: No longer return early if retest not confirmed
        # This allows BOS-only signals (early entries)
        
        # Step 4: Check candlestick confirmation
        candlestick_boost = self._check_candlestick(df_5min, bos['type'])
        
        if candlestick_boost['pattern']:
            result['candlestick_pattern'] = candlestick_boost['pattern']
            result['reasoning'].append(f"Candlestick: {candlestick_boost['pattern']}")
        
        # ==== STEP 5: Calculate confidence (SIMPLIFIED) ====
        if result['retest_confirmed']:
            base_confidence = 55  # Lowered from 72
            result['reasoning'].append("✓ Retest confirmed (base=52)")
        else:
            base_confidence = 48  # Lowered from 62
            result['reasoning'].append("⚠ Early BOS entry - retest pending (base=42)")
        
        # Factor 1: Candlestick Pattern (capped at 10)
        candlestick_score = min(10, candlestick_boost['confidence_boost'])
        base_confidence += candlestick_score
        
        # Factor 2: Trend Alignment (reduced to 6)
        trend_aligned = False
        if (bos['type'] == 'BULLISH' and overall_trend == 'Bullish') or \
           (bos['type'] == 'BEARISH' and overall_trend == 'Bearish'):
            base_confidence += 6
            trend_aligned = True
            result['reasoning'].append("Aligned with overall trend (+6)")
        
        # Cap confidence at 70
        result['confidence'] = max(30, min(70, base_confidence))
        
        # Log final confidence
        result['reasoning'].append(
            f"Confidence: {result['confidence']}% "
            f"(retest={'+10' if result['retest_confirmed'] else '0'}, "
            f"candle={candlestick_score}, trend={'+6' if trend_aligned else '0'})"
        )
        
        # Step 6: Set signal, dynamic stop loss, target
        if bos['type'] == 'BULLISH':
            result['signal'] = 'CALL'
        else:
            result['signal'] = 'PUT'
            
        # ✅ FIX 5: Try ATR-based stops first, fallback to zone-based
        atr_stops = None
        if hasattr(self, 'replay_engine') and self.replay_engine:
            atr_stops = self.calculate_atr_stops(
                entry_price=spot_price,
                signal_type=result['signal'],
                confidence=result['confidence'],
                replay_engine=self.replay_engine
            )
        
        if atr_stops:
            # Use ATR-based stops
            result['stop_loss'], result['target'], rr_ratio = atr_stops
            result['reasoning'].append(f"✅ ATR-based stops: R:R={rr_ratio:.1f}:1")
        else:
            # Fallback to original zone-based stops
            result['stop_loss'] = self.calculate_dynamic_stop_loss(
                zone_low=zone_low,
                zone_high=zone_high,
                direction='BULLISH' if result['signal'] == 'CALL' else 'BEARISH',
                spot_price=spot_price
            )
            
            # Set target based on support/resistance
            if result['signal'] == 'CALL':
                result['target'] = resistance
            else:
                result['target'] = support
            
            result['reasoning'].append("⚠️ Using zone-based stops (ATR unavailable)")

        
        # Step 7: Validate Risk:Reward Ratio
        result = self.validate_risk_reward(result)
        
        return result
    
    def _check_candlestick(self, df, direction):
        """Check for candlestick patterns"""
        if len(df) < 3:
            return {'pattern': None, 'confidence_boost': 0}
        
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]
        
        body = abs(last_candle['close'] - last_candle['open'])
        total_range = last_candle['high'] - last_candle['low']
        
        # ✅ FIX: Prevent division by zero
        if total_range == 0:
            return {'pattern': None, 'confidence_boost': 0}
        
        lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
        upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
        
        if direction == 'BULLISH':
            # Hammer
            if body > 0 and lower_wick > body * 2 and upper_wick < body * 0.3:
                return {'pattern': 'Hammer', 'confidence_boost': 15}
            
            # Bullish Engulfing (FULL LOGIC)
            if (last_candle['close'] > last_candle['open'] and
                prev_candle['close'] < prev_candle['open'] and
                last_candle['open'] < prev_candle['close'] and
                last_candle['close'] > prev_candle['open']):
                return {'pattern': 'Bullish Engulfing', 'confidence_boost': 15}
            
            # Rejection wick
            if lower_wick > total_range * 0.5:
                return {'pattern': 'Bullish Rejection', 'confidence_boost': 10}
        
        else:  # BEARISH
            # Shooting Star
            if body > 0 and upper_wick > body * 2 and lower_wick < body * 0.3:
                return {'pattern': 'Shooting Star', 'confidence_boost': 15}
            
            # Bearish Engulfing (FULL LOGIC)
            if (last_candle['close'] < last_candle['open'] and
                prev_candle['close'] > prev_candle['open'] and
                last_candle['open'] > prev_candle['close'] and
                last_candle['close'] < prev_candle['open']):
                return {'pattern': 'Bearish Engulfing', 'confidence_boost': 15}
            
            # Rejection wick
            if upper_wick > total_range * 0.5:
                return {'pattern': 'Bearish Rejection', 'confidence_boost': 10}
        
        return {'pattern': None, 'confidence_boost': 0}
