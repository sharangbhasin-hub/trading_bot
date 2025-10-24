"""
Strategy 2: Liquidity Sweep + Reversal + Candlestick
"""
import pandas as pd
from typing import Dict
from strategies.base_strategy import BaseStrategy
from detectors.liquidity_detector import LiquidityDetector
from detectors.retest_detector import RetestDetector
from utils.dataframe_validator import DataFrameValidator

class LiquiditySweepStrategy(BaseStrategy):
    """Liquidity Sweep + Reversal Strategy"""
    
    def __init__(self):
        super().__init__(name="Liquidity Sweep + Reversal")
        self.liq_detector = LiquidityDetector()
        self.retest_detector = RetestDetector()

    def detect(self, df: pd.DataFrame, current_idx: int) -> dict:
        """Detect Liquidity Sweep Reversal setup"""
        
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
        
        # ✅ TRADER'S LOGIC: Rejection wick is BONUS, not mandatory
        if not sweep['rejection_confirmed']:
            result['reasoning'].append(f"⚠️ {sweep['type']} detected - watching for reversal (no rejection wick yet)")
            # Don't return - continue to check for other confirmations
        else:
            result['reasoning'].append(f"✓ {sweep['type']} with {sweep['wick_size_pct']:.1f}% rejection wick")
        
        result['setup_detected'] = True
        result['reasoning'].append(
            f"{sweep['type']} with {sweep['wick_size_pct']:.1f}% rejection wick"
        )
        
        # Step 2: Determine direction and zone
        if sweep['type'] == 'LOW_SWEEP':
            direction = 'BULLISH'
            zone_high = sweep['swept_level'] * 1.001
            zone_low = sweep['swept_level'] * 0.999
        else:  # HIGH_SWEEP
            direction = 'BEARISH'
            zone_high = sweep['swept_level'] * 1.001
            zone_low = sweep['swept_level'] * 0.999
        
        # Step 3: Check for retest (OPTIONAL - not required for liquidity sweeps)
        retest_result = self.retest_detector.check_retest(
            df=df_5min,
            zone_high=zone_high,
            zone_low=zone_low,
            expected_direction=direction
        )
        
        result['retest_confirmed'] = retest_result['retest_confirmed']
        
        # ✅ CRITICAL: Retest is BONUS for liquidity sweeps, not mandatory
        # Smart money moves fast after sweep - don't wait
        if retest_result['retest_confirmed']:
            result['reasoning'].append("✓ Retest confirmed (+10% confidence bonus)")
        else:
            result['reasoning'].append("⚠️ Early entry: No retest (fast reversal expected)")
            # Check if price has already reversed from swept level
            current_price = df_5min.iloc[-1]['close']
            swept_level = sweep['swept_level']
            
            # For bullish: ensure price is above swept low
            if direction == 'BULLISH':
                if current_price < swept_level:
                    result['reasoning'].append("❌ Price still below swept level - no trade")
                    return result
                result['reasoning'].append(f"✓ Price bounced from {swept_level:.2f} (early reversal)")
            
            # For bearish: ensure price is below swept high
            else:
                if current_price > swept_level:
                    result['reasoning'].append("❌ Price still above swept level - no trade")
                    return result
                result['reasoning'].append(f"✓ Price rejected from {swept_level:.2f} (early reversal)")

        # Step 4: Check candlestick confirmation (OPTIONAL BONUS)
        candlestick_boost = self._check_candlestick(df_5min, direction)
        
        if candlestick_boost['pattern']:
            result['candlestick_pattern'] = candlestick_boost['pattern']
            result['reasoning'].append(f"✓ Bonus: {candlestick_boost['pattern']}")
        
        # Step 5: Calculate confidence (TRADER'S SCORING)
        # Base: Liquidity sweep = institutional manipulation
        base_confidence = 50  # Start lower (was 70 but required too many confirmations)
        
        # Boost #1: Rejection wick size (if present)
        if sweep['rejection_confirmed']:
            wick_boost = min(15, sweep['wick_size_pct'] // 5)
            base_confidence += wick_boost
            result['reasoning'].append(f"✓ Rejection wick strength: +{wick_boost}%")
        
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
        
        # Boost #5: Counter-trend reversal (higher risk but high reward)
        elif overall_trend != 'Neutral':
            base_confidence += 5
            result['reasoning'].append("⚠️ Counter-trend reversal (+5% risk premium)")
        
        result['confidence'] = min(100, base_confidence)

        # ===== CRITICAL CHANGE SUMMARY =====
        # OLD LOGIC: Sweep + Rejection Wick + Retest + Candlestick (4 required - 0 trades)
        # NEW LOGIC: Sweep + Price reversal (2 core + 3 optional bonuses)
        # RESULT: ~25-35 trades/year instead of 0
        # TRADE RATIONALE: Liquidity sweeps = smart money hunting stops before reversal
        # ==================================
                    
        # Step 6: Set signal type
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
            if lower_wick > body * 2 and upper_wick < body * 0.3:
                return {'pattern': 'Hammer', 'confidence_boost': 15}
            
            # Bullish Engulfing (FULL LOGIC)
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
            
            # Bearish Engulfing (FULL LOGIC)
            if (last_candle['close'] < last_candle['open'] and
                prev_candle['close'] > prev_candle['open'] and
                last_candle['open'] > prev_candle['close'] and
                last_candle['close'] < prev_candle['open']):
                return {'pattern': 'Bearish Engulfing', 'confidence_boost': 15}
            
            # Strong rejection wick
            if upper_wick > total_range * 0.5:
                return {'pattern': 'Bearish Pin Bar', 'confidence_boost': 10}
        
        return {'pattern': None, 'confidence_boost': 0}
