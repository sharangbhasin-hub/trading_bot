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

class FVGRetestStrategy(BaseStrategy):
    """FVG Retest + Rejection Candle Strategy - Pure Price Action"""
    
    def __init__(self):
        super().__init__(name="FVG Retest")
        self.fvg_detector = FVGDetector()

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
        """Analyze for FVG Retest setup"""
        
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
        
        # ========== STEP 1: TIME FILTER (CRITICAL FOR LIVE TRADING) ==========
        if len(df_15min) > 0 and hasattr(df_15min.index[-1], 'time'):
            current_time = df_15min.index[-1].time()
            
            # Only trade 9:30 AM - 1:30 PM (avoid late-day fake moves)
            if current_time < pd.Timestamp("09:30").time():
                result['reasoning'].append("⛔ Market not open yet (before 9:30 AM)")
                return result
            
            if current_time >= pd.Timestamp("13:30").time():
                result['reasoning'].append("⛔ Late trading hour - avoiding low-quality setups")
                return result
        
        # ========== STEP 2: DETECT FVGs ON 15MIN ==========
        fvgs = self.fvg_detector.detect(df_15min)
        
        self.logger.info(f"FVG Detection: Found {len(fvgs)} FVGs on 15min")
        
        if not fvgs:
            result['reasoning'].append("No FVGs detected on 15min timeframe")
            return result
        
        # ========== STEP 3: FILTER FVGs BY AGE (Must be recent but not too fresh) ==========
        valid_fvgs = []
        for fvg in fvgs:
            # FVG must have 'candle_index' field - if not, calculate it
            if 'candle_index' not in fvg:
                # Assuming FVG detector marks the index, skip if missing
                continue
            
            candles_ago = len(df_15min) - fvg['candle_index']
            
            # FVG should be 3-8 candles old (45min to 2 hours)
            if 3 <= candles_ago <= 8:
                valid_fvgs.append(fvg)
                self.logger.info(
                    f"✅ Valid FVG: Type={fvg['type']}, "
                    f"Age={candles_ago} candles, "
                    f"Range={fvg['bottom']:.2f}-{fvg['top']:.2f}"
                )
        
        if not valid_fvgs:
            result['reasoning'].append("No valid FVGs (must be 3-8 candles old)")
            return result
        
        # ========== STEP 4: CHECK IF PRICE IS INSIDE FVG ZONE ==========
        for fvg in valid_fvgs:
            fvg_top = fvg['top']
            fvg_bottom = fvg['bottom']
            fvg_mid = (fvg_top + fvg_bottom) / 2
            
            # Is current price inside the FVG?
            if not (fvg_bottom <= spot_price <= fvg_top):
                continue  # Skip this FVG, price not retesting it
            
            self.logger.info(
                f"🔥 Price {spot_price:.2f} is INSIDE FVG zone "
                f"{fvg_bottom:.2f}-{fvg_top:.2f}"
            )
            
            result['setup_detected'] = True
            result['reasoning'].append(
                f"FVG Retest: Price {spot_price:.2f} inside zone "
                f"{fvg_bottom:.2f}-{fvg_top:.2f}"
            )
            
            # ========== STEP 5: CHECK FOR REJECTION CANDLE ON 5MIN ==========
            if len(df_5min) < 2:
                continue
            
            last_candle = df_5min.iloc[-1]
            
            body = abs(last_candle['close'] - last_candle['open'])
            total_range = last_candle['high'] - last_candle['low']
            lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
            upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
            
            # Prevent division by zero
            if body == 0:
                body = 0.01
            
            # ========== BULLISH FVG RETEST ==========
            if fvg['type'] == 'BULLISH':
                # Need: Lower wick rejection (buyers defending FVG)
                # Criteria: Lower wick > 2x body AND bullish close
                if lower_wick > body * 2 and last_candle['close'] > last_candle['open']:
                    
                    # ========== STEP 6: TREND ALIGNMENT CHECK ==========
                    if overall_trend != 'Bullish':
                        result['reasoning'].append(
                            "⚠️ Bullish FVG but trend is not Bullish - skipping"
                        )
                        continue  # Only trade WITH the trend
                    
                    # ========== STEP 7: VOLUME CONFIRMATION (Optional but recommended) ==========
                    volume_boost = 0
                    if 'volume' in df_5min.columns and len(df_5min) >= 10:
                        recent_avg_vol = df_5min['volume'].iloc[-10:-1].mean()
                        current_vol = last_candle['volume']
                        
                        if current_vol >= recent_avg_vol * 1.2:
                            volume_boost = 5
                            result['reasoning'].append(
                                f"✓ Volume: {current_vol/recent_avg_vol:.1f}x average (+5%)"
                            )
                        elif current_vol < recent_avg_vol * 0.8:
                            result['reasoning'].append(
                                "⚠️ Low volume - reducing confidence"
                            )
                            volume_boost = -10
                    
                    # ========== STEP 8: GENERATE SIGNAL ==========
                    result['signal'] = 'CALL'
                    result['retest_confirmed'] = True
                    result['candlestick_pattern'] = 'Bullish Rejection'
                    
                    # Calculate confidence
                    base_confidence = 60  # Base for FVG retest
                    base_confidence += 5   # Rejection candle confirmed
                    base_confidence += volume_boost
                    
                    result['confidence'] = max(50, min(70, base_confidence))
                    
                    # ========== STEP 9: CALCULATE STOPS & TARGETS ==========
                    # Stop loss: Just below FVG bottom (0.2% buffer)
                    result['stop_loss'] = fvg_bottom * 0.998
                    
                    # Target: 1:2 Risk-Reward ratio
                    risk = abs(spot_price - result['stop_loss'])
                    result['target'] = spot_price + (risk * 2)
                    
                    result['reasoning'].append(
                        f"✅ CALL Signal: Bullish FVG retest at {fvg_mid:.2f}"
                    )
                    result['reasoning'].append(
                        f"✅ Rejection: Lower wick {lower_wick:.1f} pts "
                        f"({lower_wick/body:.1f}x body)"
                    )
                    result['reasoning'].append("✅ Aligned with bullish trend")
                    result['reasoning'].append(
                        f"✅ Stop: {result['stop_loss']:.2f}, "
                        f"Target: {result['target']:.2f} (1:2 R:R)"
                    )
                    
                    # Validate risk-reward
                    result = self.validate_risk_reward(result)
                    
                    return result  # Exit after first valid setup
            
            # ========== BEARISH FVG RETEST ==========
            elif fvg['type'] == 'BEARISH':
                # Need: Upper wick rejection (sellers defending FVG)
                # Criteria: Upper wick > 2x body AND bearish close
                if upper_wick > body * 2 and last_candle['close'] < last_candle['open']:
                    
                    # Trend alignment
                    if overall_trend != 'Bearish':
                        result['reasoning'].append(
                            "⚠️ Bearish FVG but trend is not Bearish - skipping"
                        )
                        continue
                    
                    # Volume check
                    volume_boost = 0
                    if 'volume' in df_5min.columns and len(df_5min) >= 10:
                        recent_avg_vol = df_5min['volume'].iloc[-10:-1].mean()
                        current_vol = last_candle['volume']
                        
                        if current_vol >= recent_avg_vol * 1.2:
                            volume_boost = 5
                            result['reasoning'].append(
                                f"✓ Volume: {current_vol/recent_avg_vol:.1f}x average (+5%)"
                            )
                        elif current_vol < recent_avg_vol * 0.8:
                            result['reasoning'].append(
                                "⚠️ Low volume - reducing confidence"
                            )
                            volume_boost = -10
                    
                    # Generate signal
                    result['signal'] = 'PUT'
                    result['retest_confirmed'] = True
                    result['candlestick_pattern'] = 'Bearish Rejection'
                    
                    base_confidence = 60
                    base_confidence += 5
                    base_confidence += volume_boost
                    
                    result['confidence'] = max(50, min(70, base_confidence))
                    
                    # Calculate stops & targets
                    result['stop_loss'] = fvg_top * 1.002  # 0.2% above FVG top
                    
                    risk = abs(result['stop_loss'] - spot_price)
                    result['target'] = spot_price - (risk * 2)
                    
                    result['reasoning'].append(
                        f"✅ PUT Signal: Bearish FVG retest at {fvg_mid:.2f}"
                    )
                    result['reasoning'].append(
                        f"✅ Rejection: Upper wick {upper_wick:.1f} pts "
                        f"({upper_wick/body:.1f}x body)"
                    )
                    result['reasoning'].append("✅ Aligned with bearish trend")
                    result['reasoning'].append(
                        f"✅ Stop: {result['stop_loss']:.2f}, "
                        f"Target: {result['target']:.2f} (1:2 R:R)"
                    )
                    
                    result = self.validate_risk_reward(result)
                    
                    return result
        
        # If we reach here, no valid setup found
        if result['setup_detected']:
            result['reasoning'].append(
                "❌ FVG detected but no rejection candle confirmed"
            )
        else:
            result['reasoning'].append(
                "❌ No price retest of valid FVG zones"
            )
        
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
