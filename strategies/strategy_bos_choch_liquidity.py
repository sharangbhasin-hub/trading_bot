"""
Strategy 6: BOS + CHOCH + Internal/External Liquidity
Complete ICT/SMC strategy with full liquidity analysis
"""
import pandas as pd
from typing import Dict, Optional, List
from strategies.base_strategy import BaseStrategy
from detectors.structure_detector import StructureDetector
from detectors.liquidity_detector import LiquidityDetector
from detectors.retest_detector import RetestDetector
from utils.dataframe_validator import DataFrameValidator

class BOSCHOCHLiquidityStrategy(BaseStrategy):
    """BOS + CHOCH + Internal/External Liquidity Strategy"""
    
    def __init__(self):
        super().__init__(name="BOS + CHOCH + Liquidity")
        self.structure_detector = StructureDetector()
        self.liq_detector = LiquidityDetector()
        self.retest_detector = RetestDetector()
        self.min_confidence = 75  # Higher threshold for this complex setup

    def detect(self, df: pd.DataFrame, current_idx: int) -> dict:
        """Detect BOS + CHOCH + Liquidity setup"""
        
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
        """Analyze for complete BOS + CHOCH + Liquidity setup"""
        
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
        
        # Step 1: Identify consolidation range on 1H chart
        range_info = self._detect_consolidation(df_1h)
        
        if not range_info:
            result['reasoning'].append("No consolidation range detected")
            return result
        
        result['reasoning'].append(
            f"Range detected: {range_info['low']:.2f} - {range_info['high']:.2f}"
        )
        
        # Step 2: Identify internal and external liquidity
        liquidity_pools = self.liq_detector.find_liquidity_pools(df_1h)
        
        internal_liquidity = self._filter_internal_liquidity(
            liquidity_pools, 
            range_info
        )
        external_liquidity = self._filter_external_liquidity(
            liquidity_pools,
            range_info
        )
        
        if not internal_liquidity and not external_liquidity:
            result['reasoning'].append("No clear liquidity pools identified")
            return result
        
        result['reasoning'].append(
            f"Internal liquidity: {len(internal_liquidity)} levels, "
            f"External liquidity: {len(external_liquidity)} levels"
        )
        
        # Step 3: Check if internal liquidity was swept
        internal_swept = self._check_liquidity_swept(
            df_15min, 
            internal_liquidity
        )
        
        if not internal_swept:
            result['reasoning'].append("Internal liquidity not yet swept")
            return result
        
        result['reasoning'].append("Internal liquidity swept")
        
        # Step 4: Detect BOS toward external liquidity
        bos = self.structure_detector.detect_bos(df_15min)
        
        if not bos:
            result['reasoning'].append("No BOS detected after internal sweep")
            return result
        
        # Check if BOS is moving toward external liquidity
        if not self._bos_toward_external(bos, external_liquidity, range_info):
            result['reasoning'].append("BOS not targeting external liquidity")
            return result
        
        result['setup_detected'] = True
        result['reasoning'].append(f"{bos['type']} BOS toward external liquidity")
        
        # Step 5: Look for CHOCH after external liquidity grab
        choch = self.structure_detector.detect_choch(df_15min)
        
        if not choch:
            result['reasoning'].append("Waiting for CHOCH confirmation")
            return result
        
        result['reasoning'].append(f"{choch['type']} CHOCH detected")
        
        # Step 6: Check for retest of range boundary
        if choch['type'] == 'BULLISH':
            # Price should retest range low (now support)
            zone_high = range_info['low'] * 1.002
            zone_low = range_info['low'] * 0.998
        else:  # BEARISH
            # Price should retest range high (now resistance)
            zone_high = range_info['high'] * 1.002
            zone_low = range_info['high'] * 0.998
        
        retest_result = self.retest_detector.check_retest(
            df=df_5min,
            zone_high=zone_high,
            zone_low=zone_low,
            expected_direction=choch['type']
        )
        
        result['retest_confirmed'] = retest_result['retest_confirmed']
        result['reasoning'].append(retest_result['reasoning'])
        
        if not retest_result['retest_confirmed']:
            return result
        
        # Step 7: Check candlestick confirmation
        candlestick_boost = self._check_candlestick(df_5min, choch['type'])
        if candlestick_boost['pattern']:
            result['candlestick_pattern'] = candlestick_boost['pattern']
            result['reasoning'].append(f"Candlestick: {candlestick_boost['pattern']}")
        
        # Step 8: Calculate confidence (high for complete setup)
        base_confidence = 78  # Very strong setup when complete
        
        # Boost for candlestick
        base_confidence += candlestick_boost['confidence_boost']
        
        # Boost if aligned with overall trend
        if ((choch['type'] == 'BULLISH' and overall_trend == 'Bullish') or
            (choch['type'] == 'BEARISH' and overall_trend == 'Bearish')):
            base_confidence += 10
            result['reasoning'].append("Aligned with overall trend")
        
        result['confidence'] = min(100, base_confidence)
        
        # Step 9: Set signal type
        if choch['type'] == 'BULLISH':
            result['signal'] = 'CALL'
        else:
            result['signal'] = 'PUT'
        
        # ✅ STANDARD STOP LOSS CALCULATION
        # Try ATR-based stops first (most accurate)
        atr_stops = None
        if hasattr(self, 'replay_engine') and self.replay_engine:
            atr_stops = self.calculate_atr_stops(
                entry_price=spot_price,
                signal_type=result['signal'],
                confidence=result['confidence'],
                replay_engine=self.replay_engine
            )
        
        if atr_stops:
            # ✅ Path 1: ATR-based (preferred)
            result['stop_loss'], result['target'], rr_ratio = atr_stops
            result['reasoning'].append(f"✅ ATR-based stops: R:R={rr_ratio:.1f}:1")
        else:
            # ✅ Path 2: Simple percentage-based fallback
            result['stop_loss'], result['target'] = self.calculate_simple_stops(
                entry_price=spot_price,
                signal_type=result['signal'],
                support=support,
                resistance=resistance
            )
            result['reasoning'].append("⚠️ Using percentage-based stops (ATR unavailable)")
        
        # Step 10: Validate Risk:Reward Ratio
        result = self.validate_risk_reward(result)
        
        # Step 9: Validate Risk:Reward Ratio
        result = self.validate_risk_reward(result)
        
        return result
    
    def _detect_consolidation(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect consolidation range"""
        if len(df) < 20:
            return None
        
        # Look at last 20 candles
        recent = df.tail(20)
        
        # Calculate range
        range_high = recent['high'].max()
        range_low = recent['low'].min()
        range_size = range_high - range_low
        
        # Check if price is contained (not trending)
        # Calculate ATR for comparison
        atr = (recent['high'] - recent['low']).mean()
        
        # If range is small relative to ATR, it's consolidation
        if range_size < atr * 3:
            # Further check: most candles should be within range
            candles_in_range = sum(
                (recent['low'] >= range_low * 0.995) &
                (recent['high'] <= range_high * 1.005)
            )
            
            if candles_in_range >= len(recent) * 0.7:  # 70% of candles
                return {
                    'high': range_high,
                    'low': range_low,
                    'mid': (range_high + range_low) / 2
                }
        
        return None
    
    def _filter_internal_liquidity(self, liquidity_pools, range_info):
        """Filter liquidity pools inside the range"""
        internal = {
            'high_pools': [
                p for p in liquidity_pools['high_pools']
                if range_info['low'] < p < range_info['high']
            ],
            'low_pools': [
                p for p in liquidity_pools['low_pools']
                if range_info['low'] < p < range_info['high']
            ]
        }
        return internal
    
    def _filter_external_liquidity(self, liquidity_pools, range_info):
        """Filter liquidity pools outside the range"""
        external = {
            'high_pools': [
                p for p in liquidity_pools['high_pools']
                if p > range_info['high']
            ],
            'low_pools': [
                p for p in liquidity_pools['low_pools']
                if p < range_info['low']
            ]
        }
        return external
    
    def _check_liquidity_swept(self, df, internal_liquidity):
        """Check if internal liquidity was swept"""
        if not internal_liquidity['high_pools'] and not internal_liquidity['low_pools']:
            return False
        
        recent_high = df['high'].tail(10).max()
        recent_low = df['low'].tail(10).min()
        
        # Check if any internal liquidity was hit
        high_swept = any(
            recent_high > pool
            for pool in internal_liquidity['high_pools']
        )
        low_swept = any(
            recent_low < pool
            for pool in internal_liquidity['low_pools']
        )
        
        return high_swept or low_swept
    
    def _bos_toward_external(self, bos, external_liquidity, range_info):
        """Check if BOS is moving toward external liquidity"""
        if bos['type'] == 'BULLISH':
            # Should be breaking above range toward high external liquidity
            return (bos['broken_level'] > range_info['mid'] and
                    external_liquidity['high_pools'])
        else:  # BEARISH
            # Should be breaking below range toward low external liquidity
            return (bos['broken_level'] < range_info['mid'] and
                    external_liquidity['low_pools'])
    
    def _check_candlestick(self, df, direction):
        """Check for candlestick patterns"""
        if len(df) < 3:
            return {'pattern': None, 'confidence_boost': 0}
        
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]
        
        body = abs(last_candle['close'] - last_candle['open'])
        total_range = last_candle['high'] - last_candle['low']
        
        if total_range == 0:
            return {'pattern': None, 'confidence_boost': 0}
        
        lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
        upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
        
        if direction == 'BULLISH':
            if lower_wick > body * 2:
                return {'pattern': 'Hammer', 'confidence_boost': 12}
            if last_candle['close'] > prev_candle['open']:
                return {'pattern': 'Bullish Engulfing', 'confidence_boost': 12}
            if lower_wick > total_range * 0.5:
                return {'pattern': 'Bullish Rejection', 'confidence_boost': 8}
        else:
            if upper_wick > body * 2:
                return {'pattern': 'Shooting Star', 'confidence_boost': 12}
            if last_candle['close'] < prev_candle['open']:
                return {'pattern': 'Bearish Engulfing', 'confidence_boost': 12}
            if upper_wick > total_range * 0.5:
                return {'pattern': 'Bearish Rejection', 'confidence_boost': 8}
        return {'pattern': None, 'confidence_boost': 0}
