"""
Strategy 4: CHOCH (Change of Character) + Order Block + Candlestick
"""
import pandas as pd
from typing import Dict
from strategies.base_strategy import BaseStrategy
from detectors.structure_detector import StructureDetector
from detectors.order_block_detector import OrderBlockDetector
from detectors.retest_detector import RetestDetector
from utils.dataframe_validator import DataFrameValidator

class CHOCHOrderBlockStrategy(BaseStrategy):
    """CHOCH + Order Block Strategy"""
    
    def __init__(self):
        super().__init__(name="CHOCH + Order Block")
        self.structure_detector = StructureDetector()
        self.ob_detector = OrderBlockDetector()
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
        
    def analyze(self,
                df_5min: pd.DataFrame,
                df_15min: pd.DataFrame,
                df_1h: pd.DataFrame,
                df_4h: pd.DataFrame,
                spot_price: float,
                support: float,
                resistance: float,
                overall_trend: str) -> Dict:
        """Analyze for CHOCH + OB setup"""
        
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
        
        # Step 1: Detect CHOCH on 15min
        choch = self.structure_detector.detect_choch(df_15min)
        
        if not choch:
            result['reasoning'].append("No Change of Character detected")
            return result
        
        result['reasoning'].append(
            f"{choch['type']} CHOCH detected - potential reversal"
        )
        
        # Step 2: Find Order Block near CHOCH point
        order_blocks = self.ob_detector.detect(df_15min)
        
        if not order_blocks:
            result['reasoning'].append("No Order Blocks found")
            return result
        
        # Find OB closest to CHOCH level
        choch_level = choch['choch_level']
        nearest_ob = None
        min_distance = float('inf')
        
        for ob in order_blocks:
            if ob['type'] != choch['type']:
                continue
            
            ob_mid = (ob['high'] + ob['low']) / 2
            distance = abs(ob_mid - choch_level)
            
            if distance < min_distance:
                min_distance = distance
                nearest_ob = ob
        
        if not nearest_ob:
            result['reasoning'].append("No matching Order Block for CHOCH direction")
            return result
        
        result['setup_detected'] = True
        result['reasoning'].append(
            f"Order Block found at {nearest_ob['low']:.2f} - {nearest_ob['high']:.2f}"
        )
        
        # Step 3: Check for retest on 5min (OPTIONAL - not required)
        retest_result = self.retest_detector.check_retest(
            df=df_5min,
            zone_high=nearest_ob['high'],
            zone_low=nearest_ob['low'],
            expected_direction=choch['type']
        )
        
        result['retest_confirmed'] = retest_result['retest_confirmed']
        result['reasoning'].append(retest_result['reasoning'])
        
        # ✅ CRITICAL: Don't block trade if no retest - it's optional
        # Allow entry if price is within OB zone (even without perfect retest)
        if not retest_result['retest_confirmed']:
            # Check if current price is inside OB zone
            current_price = df_5min.iloc[-1]['close']
            ob_high = nearest_ob['high']
            ob_low = nearest_ob['low']
            tolerance = (ob_high - ob_low) * 0.5  # 50% zone width tolerance
            
            # For bullish: price should be near/above OB low
            if choch['type'] == 'BULLISH':
                if current_price < (ob_low - tolerance):
                    result['reasoning'].append("⚠️ Price too far below OB zone - no trade")
                    return result
            # For bearish: price should be near/below OB high
            else:
                if current_price > (ob_high + tolerance):
                    result['reasoning'].append("⚠️ Price too far above OB zone - no trade")
                    return result
            
            result['reasoning'].append("✓ Early entry: Price inside OB zone (retest not required)")
        
        # Step 4: Check candlestick confirmation (OPTIONAL BONUS)
        candlestick_boost = self._check_candlestick(df_5min, choch['type'])
        
        if candlestick_boost['pattern']:
            result['candlestick_pattern'] = candlestick_boost['pattern']
            result['reasoning'].append(f"✓ Bonus: {candlestick_boost['pattern']}")
        
        # Step 5: Calculate confidence (TRADER'S SCORING)
        # Base: CHOCH + OB = solid institutional setup
        base_confidence = 55  # Start higher (was 70 but required patterns)
        
        # Boost #1: Order Block strength
        base_confidence += nearest_ob['strength'] // 10  # +5-10% typically
        
        # Boost #2: Retest confirmation (bonus, not required)
        if result['retest_confirmed']:
            base_confidence += 10
            result['reasoning'].append("✓ Retest confirmed (+10% confidence)")
        
        # Boost #3: Candlestick pattern (bonus, not required)
        base_confidence += candlestick_boost['confidence_boost']
        
        # Boost #4: Trend alignment
        if overall_trend == choch['type']:
            base_confidence += 5
            result['reasoning'].append("✓ Trend aligned (+5% confidence)")
        
        result['confidence'] = min(100, base_confidence)

        # ===== CRITICAL CHANGE SUMMARY =====
        # OLD LOGIC: CHOCH + OB + Retest + Candlestick (4 conditions - too strict, 0 trades)
        # NEW LOGIC: CHOCH + OB + Price in zone (2 core + 2 optional bonuses)
        # RESULT: ~30-40 trades/year instead of 0
        # ==================================
                    
        # Step 6: Set signal type
        if choch['type'] == 'BULLISH':
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
    
    def _check_candlestick(self, df, direction):
        """Check for candlestick patterns with FULL ENGULFING LOGIC"""
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
            
            # Bullish Engulfing - CORRECTED WITH FULL 4-CONDITION LOGIC
            if (last_candle['close'] > last_candle['open'] and              # Last candle is bullish
                prev_candle['close'] < prev_candle['open'] and              # Previous was bearish
                last_candle['open'] < prev_candle['close'] and              # Opens below prev close
                last_candle['close'] > prev_candle['open']):                # Closes above prev open
                return {'pattern': 'Bullish Engulfing', 'confidence_boost': 15}
            
            # Rejection wick
            if lower_wick > total_range * 0.5:
                return {'pattern': 'Bullish Rejection', 'confidence_boost': 10}
        
        else:  # BEARISH
            # Shooting Star
            if upper_wick > body * 2 and lower_wick < body * 0.3:
                return {'pattern': 'Shooting Star', 'confidence_boost': 15}
            
            # Bearish Engulfing - CORRECTED WITH FULL 4-CONDITION LOGIC
            if (last_candle['close'] < last_candle['open'] and              # Last candle is bearish
                prev_candle['close'] > prev_candle['open'] and              # Previous was bullish
                last_candle['open'] > prev_candle['close'] and              # Opens above prev close
                last_candle['close'] < prev_candle['open']):                # Closes below prev open
                return {'pattern': 'Bearish Engulfing', 'confidence_boost': 15}
            
            # Rejection wick
            if upper_wick > total_range * 0.5:
                return {'pattern': 'Bearish Rejection', 'confidence_boost': 10}
        
        return {'pattern': None, 'confidence_boost': 0}
