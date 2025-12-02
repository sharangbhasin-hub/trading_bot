"""
Strategy 8: FVG + Rejection Candle + Breakout
Pure price action with institutional footprints
"""
import pandas as pd
from typing import Dict, Optional
from datetime import datetime
from strategies.base_strategy import BaseStrategy
from detectors.fvg_detector import FVGDetector
from utils.dataframe_validator import DataFrameValidator

class FVGDoubleBottomTopStrategy(BaseStrategy):
    """FVG Retest + Rejection Candle Strategy - Pure Price Action"""
    
    def __init__(self):
        super().__init__(name="FVG Retest")
        self.fvg_detector = FVGDetector()
        self.df_validator = DataFrameValidator()
    
    def analyze(self, 
                df_5min: pd.DataFrame,
                df_15min: pd.DataFrame,
                df_1h: pd.DataFrame,
                df_4h: pd.DataFrame,
                spot_price: float,
                support: float,
                resistance: float,
                overall_trend: str) -> Dict:
        """
        ✅ FIXED: Old signature compatible with strategy manager
        
        Args:
            df_5min: 5-minute timeframe data
            df_15min: 15-minute timeframe data (primary for FVG detection)
            df_1h: 1-hour timeframe data
            df_4h: 4-hour/daily timeframe data
            spot_price: Current market price
            support: Support level
            resistance: Resistance level
            overall_trend: Overall market trend
        
        Returns:
            Signal dict
        """
        
        # ========== INITIALIZATION ==========
        # Get current time from dataframe
        # Handle empty dataframe edge case
        if df_15min is not None and not df_15min.empty and hasattr(df_15min.index[-1], 'to_pydatetime'):
            current_time = df_15min.index[-1].to_pydatetime()
        else:
            current_time = datetime.now()
        
        self.logger.warning("=" * 80)
        self.logger.warning(f"🔍 FVG RETEST STRATEGY: Starting analysis at {current_time}")
        self.logger.warning("=" * 80)
        
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
        
        # ========== STEP 1: VALIDATE DATA ==========
        self.logger.warning(f"📊 Data Check:")
        self.logger.warning(f"  - 5min: {len(df_5min) if df_5min is not None else 0} candles")
        self.logger.warning(f"  - 15min: {len(df_15min) if df_15min is not None else 0} candles")
        self.logger.warning(f"  - 1h: {len(df_1h) if df_1h is not None and not df_1h.empty else 0} candles")
        self.logger.warning(f"  - 4h/daily: {len(df_4h) if df_4h is not None and not df_4h.empty else 0} candles")
        self.logger.warning(f"  - Spot Price: {spot_price:.2f}")
        self.logger.warning(f"  - Trend: {overall_trend}")
        
        if df_15min is None or len(df_15min) < 20:
            self.logger.warning(f"⚠️ Insufficient 15min data: {len(df_15min) if df_15min is not None else 0}")
            result['reasoning'].append("Insufficient 15min data (need 20+ candles)")
            return result

        if df_5min is None or len(df_5min) < 10:
            self.logger.warning(f"⚠️ Insufficient 5min data: {len(df_5min) if df_5min is not None else 0}")
            result['reasoning'].append("Insufficient 5min data (need 10+ candles)")
            return result
        
        # ========== STEP 2: TIME FILTER ==========
        if hasattr(df_15min.index[-1], 'time'):
            current_time_only = df_15min.index[-1].time()
            
            self.logger.warning(f"⏰ Current Time: {current_time_only}")
            
            # Only trade 9:30 AM - 1:30 PM
            if current_time_only < pd.Timestamp("09:30").time():
                result['reasoning'].append("⛔ Before market hours (9:30 AM)")
                self.logger.warning("⛔ TIME FILTER: Before 9:30 AM - SKIPPING")
                return result
            
            if current_time_only >= pd.Timestamp("13:30").time():
                result['reasoning'].append("⛔ After 1:30 PM - avoiding late-day noise")
                self.logger.warning("⛔ TIME FILTER: After 1:30 PM - SKIPPING")
                return result
            
            self.logger.warning("✅ TIME FILTER: Inside trading window (9:30 AM - 1:30 PM)")
        
        # ========== STEP 3: DETECT FVGs ==========
        self.logger.warning(f"🔍 Calling FVG Detector on 15min data...")
        
        fvgs = self.fvg_detector.detect(df_15min)
        
        self.logger.warning(f"🔍 FVG Detector returned: {len(fvgs)} FVGs")
        
        if len(fvgs) == 0:
            result['reasoning'].append("No FVGs detected on 15min")
            self.logger.warning("⚠️ NO FVGs FOUND - Strategy cannot proceed")
            return result
        
        # ========== STEP 4: LOG EACH FVG IN DETAIL ==========
        self.logger.warning(f"📋 FVG Details:")
        for i, fvg in enumerate(fvgs):
            self.logger.warning(f"  FVG #{i+1}:")
            self.logger.warning(f"    Type: {fvg.get('type', 'N/A')}")
            self.logger.warning(f"    Top: {fvg.get('top', 0):.2f}")
            self.logger.warning(f"    Bottom: {fvg.get('bottom', 0):.2f}")
            self.logger.warning(f"    Candle Index: {fvg.get('candle_index', 'N/A')}")
            self.logger.warning(f"    Age (candles): {fvg.get('age_candles', 'N/A')}")
            self.logger.warning(f"    Fill %: {fvg.get('fill_percentage', 'N/A')}%")
            self.logger.warning(f"    Distance %: {fvg.get('distance_pct', 'N/A')}%")
            self.logger.warning(f"    Quality: {fvg.get('quality', 'N/A')}")
            self.logger.warning(f"    Price Inside: {fvg.get('price_inside', False)}")
        
        # ========== STEP 5: FILTER FVGs BY AGE ==========
        # Note: FVG detector already does age filtering (3-15 candles)
        # But we'll recalculate here for verification and logging
        valid_fvgs = []
        for fvg in fvgs:
            if 'candle_index' not in fvg:
                self.logger.warning(f"⚠️ FVG missing 'candle_index' - skipping")
                continue
            
            # Calculate age (already should be in fvg from detector)
            candles_ago = len(df_15min) - fvg['candle_index'] - 1
            
            self.logger.warning(f"🔍 FVG Age Check: {candles_ago} candles old")
            
            # FVG detector already filtered to 3-15, but double-check
            if 3 <= candles_ago <= 15:
                valid_fvgs.append(fvg)
                self.logger.warning(f"✅ FVG #{len(valid_fvgs)} passed age filter")
            else:
                self.logger.warning(f"❌ FVG rejected: Age {candles_ago} (need 3-15)")
        
        if not valid_fvgs:
            result['reasoning'].append("No valid FVGs (age must be 3-15 candles)")
            self.logger.warning("⚠️ NO VALID FVGs after age filter")
            return result
        
        self.logger.warning(f"✅ {len(valid_fvgs)} FVGs passed age filter")
        
        # ========== STEP 6: CHECK IF PRICE IS INSIDE FVG ==========
        for idx, fvg in enumerate(valid_fvgs):
            fvg_top = fvg['top']
            fvg_bottom = fvg['bottom']
            fvg_mid = (fvg_top + fvg_bottom) / 2
            
            self.logger.warning(f"🔍 Checking FVG #{idx+1}: {fvg_bottom:.2f} - {fvg_top:.2f}")
            self.logger.warning(f"   Current price: {spot_price:.2f}")
            
            # Check if price is inside FVG
            price_inside = (fvg_bottom <= spot_price <= fvg_top)
            
            if not price_inside:
                distance = min(abs(spot_price - fvg_bottom), abs(spot_price - fvg_top))
                self.logger.warning(f"   ❌ Price NOT inside FVG (distance: {distance:.2f} pts)")
                continue
            
            self.logger.warning(f"   🔥 PRICE INSIDE FVG! Testing zone...")
            
            result['setup_detected'] = True
            result['reasoning'].append(
                f"FVG Retest: Price {spot_price:.2f} inside "
                f"{fvg_bottom:.2f}-{fvg_top:.2f}"
            )
            
            # ========== STEP 7: CHECK FOR REJECTION CANDLE ON 5MIN ==========
            self.logger.warning(f"🔍 Checking 5min for rejection candle...")
            
            if len(df_5min) < 2:
                self.logger.warning(f"⚠️ Not enough 5min candles ({len(df_5min)})")
                continue
            
            last_candle = df_5min.iloc[-1]
            
            body = abs(last_candle['close'] - last_candle['open'])
            total_range = last_candle['high'] - last_candle['low']
            lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
            upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
            
            if body == 0:
                body = 0.01
            
            self.logger.warning(f"   Candle Analysis:")
            self.logger.warning(f"     Body: {body:.2f}, Range: {total_range:.2f}")
            self.logger.warning(f"     Lower Wick: {lower_wick:.2f}, Upper Wick: {upper_wick:.2f}")
            self.logger.warning(f"     Lower/Body: {lower_wick/body:.2f}x, Upper/Body: {upper_wick/body:.2f}x")
            
            # ========== BULLISH FVG RETEST ==========
            if fvg['type'] == 'BULLISH':
                self.logger.warning(f"   Checking for BULLISH rejection...")
                
                # Need: Lower wick > 2x body AND bullish close
                has_rejection = (lower_wick > body * 2 and 
                                last_candle['close'] > last_candle['open'])
                
                if has_rejection:
                    self.logger.warning(f"   ✅ BULLISH REJECTION CONFIRMED!")
                    
                    # Generate CALL signal
                    result['signal'] = 'CALL'
                    result['retest_confirmed'] = True
                    result['candlestick_pattern'] = 'Bullish Rejection'
                    result['confidence'] = 65
                    
                    # Calculate stops & targets
                    result['stop_loss'] = fvg_bottom * 0.998  # 0.2% below
                    risk = abs(spot_price - result['stop_loss'])
                    result['target'] = spot_price + (risk * 2)
                    
                    result['reasoning'].append(f"✅ CALL: Bullish FVG retest at {fvg_mid:.2f}")
                    result['reasoning'].append(f"✅ Rejection: Lower wick {lower_wick:.1f} pts")
                    result['reasoning'].append(f"✅ Stop: {result['stop_loss']:.2f}, Target: {result['target']:.2f}")
                    
                    self.logger.warning(f"🎯 SIGNAL GENERATED: CALL at {spot_price:.2f}")
                    self.logger.warning(f"   Stop: {result['stop_loss']:.2f}, Target: {result['target']:.2f}")
                    
                    return result
                else:
                    self.logger.warning(f"   ❌ No bullish rejection (need lower_wick > 2x body)")
            
            # ========== BEARISH FVG RETEST ==========
            elif fvg['type'] == 'BEARISH':
                self.logger.warning(f"   Checking for BEARISH rejection...")
                
                # Need: Upper wick > 2x body AND bearish close
                has_rejection = (upper_wick > body * 2 and 
                                last_candle['close'] < last_candle['open'])
                
                if has_rejection:
                    self.logger.warning(f"   ✅ BEARISH REJECTION CONFIRMED!")
                    
                    # Generate PUT signal
                    result['signal'] = 'PUT'
                    result['retest_confirmed'] = True
                    result['candlestick_pattern'] = 'Bearish Rejection'
                    result['confidence'] = 65
                    
                    # Calculate stops & targets
                    result['stop_loss'] = fvg_top * 1.002  # 0.2% above
                    risk = abs(result['stop_loss'] - spot_price)
                    result['target'] = spot_price - (risk * 2)
                    
                    result['reasoning'].append(f"✅ PUT: Bearish FVG retest at {fvg_mid:.2f}")
                    result['reasoning'].append(f"✅ Rejection: Upper wick {upper_wick:.1f} pts")
                    result['reasoning'].append(f"✅ Stop: {result['stop_loss']:.2f}, Target: {result['target']:.2f}")
                    
                    self.logger.warning(f"🎯 SIGNAL GENERATED: PUT at {spot_price:.2f}")
                    self.logger.warning(f"   Stop: {result['stop_loss']:.2f}, Target: {result['target']:.2f}")
                    
                    return result
                else:
                    self.logger.warning(f"   ❌ No bearish rejection (need upper_wick > 2x body)")
        
        # ========== NO VALID SETUP ==========
        if result['setup_detected']:
            result['reasoning'].append("❌ FVG detected but no rejection candle")
            self.logger.warning("❌ FVG retest detected but no rejection candle confirmed")
        else:
            result['reasoning'].append("❌ No price retest of valid FVG zones")
            self.logger.warning("❌ No price inside any valid FVG zone")
        
        self.logger.warning("=" * 80)
        self.logger.warning("🔍 FVG RETEST STRATEGY: No trade signal generated")
        self.logger.warning("=" * 80)
        
        return result
