"""
CRT-TBS (Candle Range Theory + Turtle Body Soup) Strategy
===========================================================

Multi-timeframe trading strategy based on institutional price action concepts.
Combines HTF directional bias (CRT) with LTF precision entry (TBS).

Strategy Flow:
HTF_SCANNING → HTF_SETUP_COMPLETE → LTF_MONITORING → 
TBS_CONFIRMED → MODEL1_CONFIRMED → ENTRY_TRIGGERED → POSITION_ACTIVE

Based on CRT-TBS trading strategy documentation.
Target Win Rate: 80-95% at TP-1

Author: Trading System
Date: October 23, 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

# Import base strategy class
from strategies.base_strategy import BaseStrategy

# Import detectors
from detectors.crt_detector import CRTDetector
from detectors.keylevel_detector import KeyLevelDetector
from detectors.tbs_detector import TBSDetector
from config_crt_tbs import get_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyCRTTBS(BaseStrategy):
    """
    CRT-TBS Strategy Implementation.
    
    Extends BaseStrategy to integrate with existing backtesting infrastructure.
    Implements state machine for setup detection and trade execution.
    """
    
    def __init__(self, config: Optional[Dict] = None, market_type: str = None, config_name: str = None):
        """
        Initialize CRT-TBS Strategy with automatic config selection.
        
        Priority: config_name > config dict > market_type > default
        
        Args:
            config: Strategy configuration dictionary (optional, for manual override)
            market_type: Market type for auto-config selection:
                       'Cryptocurrency', 'Forex', 'Commodities', 'Stock' (optional)
            config_name: Explicit config name:
                       'scalping', 'intraday', 'short_term', 'crypto', 'forex', 'commodities' (optional)
        
        Examples:
            # Automatic based on trading mode (best for UI)
            strategy = StrategyCRTTBS(market_type='Forex', config_name='scalping')  # 1H → 1min
            
            # Automatic based on market only
            strategy = StrategyCRTTBS(market_type='Cryptocurrency')  # 4H → 15min
            
            # Manual config (backward compatible)
            strategy = StrategyCRTTBS(config={'htf': '1D', 'ltf': '1H', ...})
        """
        super().__init__(config)
        
        # ✅ PRIORITY 1: Explicit config_name (from UI trading mode selection)
        if config_name:
            self.config = get_config(config_name=config_name)
            logger.info(
                f"✅ Using explicit config: {config_name.upper()} | "
                f"HTF={self.config.get('htf')}, LTF={self.config.get('ltf')}"
            )
        
        # ✅ PRIORITY 2: Manual config dict (backward compatibility)
        elif config is not None:
            self.config = config
            logger.info(
                f"✅ Using custom config dict | "
                f"HTF={self.config.get('htf')}, LTF={self.config.get('ltf')}"
            )
        
        # ✅ PRIORITY 3: Auto-select based on market_type
        elif market_type:
            self.config = get_config(market_type=market_type)
            logger.info(
                f"✅ Auto-selected config for {market_type} | "
                f"HTF={self.config.get('htf')}, LTF={self.config.get('ltf')}"
            )
        
        # ✅ PRIORITY 4: Fallback to default intraday
        else:
            self.config = get_config('intraday')
            logger.warning(
                f"⚠️ No config specified - defaulting to INTRADAY | "
                f"HTF={self.config.get('htf')}, LTF={self.config.get('ltf')}"
            )
        
        self.name = "CRT_TBS"
        self.market_type = market_type  # Store for reference
        self.config_name = config_name  # Store for reference
        
        # Initialize detectors with configuration
        self.crt_detector = CRTDetector(self._get_crt_config())
        self.keylevel_detector = KeyLevelDetector(self._get_keylevel_config())
        self.tbs_detector = TBSDetector(self._get_tbs_config())
        
        # Strategy state machine
        self.state = 'HTF_SCANNING'
        self.htf_setup = None
        self.ltf_setup = None
        
        # Risk management parameters
        self.min_rr_ratio = self.config.get('min_rr_ratio', 1.0)
        self.stop_buffer = self.config.get('stop_buffer', 0.0)
        self.risk_per_trade = self.config.get('risk_per_trade', 1.0)
        
        # Performance tracking
        self.setup_count = 0
        self.signal_count = 0
        
        logger.info(f"Initialized {self.name} strategy with config: {self.config}")
    
    def _get_crt_config(self) -> Dict:
        """Extract CRT detector configuration."""
        return {
            'method': self.config.get('crt_method', 'body_vs_wicks'),
            'min_body_ratio': self.config.get('crt_min_body_ratio', 0.50)
        }
    
    def _get_keylevel_config(self) -> Dict:
        """Extract key level detector configuration."""
        htf = self.config.get('htf', '1D')
        
        # Get timeframe-specific lookback periods
        lookback_map = {
            '1H': 20,
            '4H': 15,
            '1D': 10,
            '1W': 8
        }
        
        return {
            'ohp_olp_lookback': self.config.get('ohp_olp_lookback', lookback_map.get(htf, 20)),
            'swing_left': 2,
            'swing_right': 2,
            'fvg_min_gap_percent': self.config.get('fvg_min_gap_percent', 0.05),
            'ob_min_consecutive': self.config.get('ob_min_consecutive', 5),
            'rb_min_wick_ratio': self.config.get('rb_min_wick_ratio', 0.40)
        }
    
    def _get_tbs_config(self) -> Dict:
        """Extract TBS detector configuration."""
        ltf = self.config.get('ltf', '1H')
        
        # Get timeframe-specific max wait candles
        wait_map = {
            '1m': 10,
            '5m': 10,
            '1H': 8
        }
        
        return {
            'allow_multi_candle': self.config.get('allow_multi_candle', True),
            'valid_candle_range': self.config.get('valid_candle_range', [2, 6]),
            'model1_method': self.config.get('model1_method', 'body_vs_wicks'),
            'model1_min_body_ratio': self.config.get('model1_min_body_ratio', 0.60),
            'max_wait_candles': self.config.get('max_wait_candles', wait_map.get(ltf, 10))
        }
    
    def generate_signals(
        self, 
        df_htf: pd.DataFrame, 
        df_ltf: pd.DataFrame, 
        **kwargs
    ) -> Optional[Dict]:
        """
        Main strategy signal generation method.
        
        Implements state machine logic:
        1. HTF_SCANNING: Look for CRT + Key Level
        2. LTF_MONITORING: Wait for TBS pattern
        3. TBS_CONFIRMED: Wait for Model #1
        4. MODEL1_CONFIRMED: Check for entry trigger
        
        Args:
            df_htf: Higher timeframe OHLC data
            df_ltf: Lower timeframe OHLC data
            **kwargs: Additional parameters
        
        Returns:
            Signal dictionary or None:
            {
                'action': 'BUY' or 'SELL',
                'entry_price': float,
                'stop_loss': float,
                'take_profit_1': float,
                'take_profit_2': float,
                'position_scaling': list,
                'confidence': float,
                'setup_info': dict
            }
        """
        try:
            if self.state == 'HTF_SCANNING':
                return self._scan_htf(df_htf, df_ltf)
            
            elif self.state in ['LTF_MONITORING', 'TBS_CONFIRMED', 'MODEL1_CONFIRMED']:
                return self._monitor_ltf(df_htf, df_ltf)
            
            else:
                logger.warning(f"Unknown state: {self.state}. Resetting to HTF_SCANNING.")
                self.state = 'HTF_SCANNING'
                self.htf_setup = None
                return None
                
        except Exception as e:
            logger.error(f"Error in generate_signals: {str(e)}", exc_info=True)
            self._reset_state()
            return None

    def analyze(self, df: pd.DataFrame, **kwargs) -> Optional[Dict]:
        """
        Implement abstract method from BaseStrategy.
        
        For CRT-TBS, this redirects to generate_signals() with multi-timeframe data.
        
        Args:
            df: Single timeframe dataframe (not used for CRT-TBS)
            **kwargs: Additional parameters
        
        Returns:
            Signal dictionary or None
        """
        # CRT-TBS requires both HTF and LTF data
        # This method is a placeholder to satisfy BaseStrategy interface
        # Actual signal generation happens through generate_signals(df_htf, df_ltf)
        
        logger.warning("analyze() called on CRT-TBS strategy. Use generate_signals(df_htf, df_ltf) instead.")
        return None    
    
    def _scan_htf(
        self, 
        df_htf: pd.DataFrame, 
        df_ltf: pd.DataFrame
    ) -> None:
        """
        Scan HTF for CRT candle with key level.
        
        Steps:
        1. Detect CRT candles (both standard and inside bar patterns)
        2. Check for key levels at each CRT candle
        3. If setup complete, transition to LTF monitoring
        
        Args:
            df_htf: HTF dataframe
            df_ltf: LTF dataframe
        
        Returns:
            None (updates internal state)
        """
        if len(df_htf) < 10:
            return None
        
        # ✅ CHANGE #1: Use new unified detect() method
        # This returns List[Dict] with both standard and inside bar CRTs
        crt_patterns = self.crt_detector.detect(df_htf, detect_inside_bar=True)
        
        if not crt_patterns:
            return None
        
        # ✅ CHANGE #2: Get most recent CRT patterns (last 5)
        recent_crts = crt_patterns[-5:] if len(crt_patterns) > 5 else crt_patterns
        
        logger.info(f"Found {len(recent_crts)} recent CRT patterns to check")
        
        # ✅ CHANGE #3: Loop through CRT pattern dictionaries (not DataFrame rows)
        for crt_pattern in recent_crts:
            # Extract pattern info
            pattern_type = crt_pattern['type']  # 'STANDARD_CRT' or 'INSIDE_BAR_CRT'
            crt_idx = crt_pattern['crt_candle_index']
            crt_high = crt_pattern['crt_high']
            crt_low = crt_pattern['crt_low']
            crt_range = crt_pattern['crt_range']
            timestamp = crt_pattern['timestamp']
            
            logger.info(f"Checking {pattern_type} at {timestamp}")
            
            # Detect all key levels at this CRT candle
            keylevels = self.keylevel_detector.detect_all_keylevels(
                df_htf, 
                crt_candle_idx=crt_idx
            )
            
            if not keylevels['has_any_keylevel']:
                logger.debug(f"No key level found at {pattern_type}")
                continue
            
            # Get primary key level
            primary_keylevel = self.keylevel_detector.get_primary_keylevel(keylevels)
            
            if primary_keylevel is None:
                logger.debug(f"No primary key level at {pattern_type}")
                continue
            
            # ✅ CHANGE #4: Build CRT levels from pattern dictionary
            crt_levels = {
                'crt_high': crt_high,
                'crt_low': crt_low,
                'tp1_level': crt_low + ((crt_high - crt_low) / 2),  # 50% level
                'tp2_sell': crt_low,  # 100% target for sell
                'tp2_buy': crt_high,  # 100% target for buy
                'direction': crt_pattern['direction'],
                'body_ratio': crt_pattern.get('body_ratio', 0.60),
                'timestamp': timestamp
            }
            
            # Determine trade direction based on key level type
            direction = crt_pattern['direction'].lower()  # Already in pattern
            
            # Store HTF setup
            self.htf_setup = {
                'crt_pattern': crt_pattern,  # Store full pattern dict
                'crt_levels': crt_levels,
                'keylevel': primary_keylevel,
                'direction': direction,
                'htf_timestamp': timestamp,
                'setup_timestamp': datetime.now()
            }
            
            self.setup_count += 1
            self.state = 'LTF_MONITORING'
            
            logger.info(f"✅ HTF Setup #{self.setup_count} detected: {pattern_type} → {direction.upper()}")
            logger.info(f"   Key Level: {primary_keylevel['type']}")
            logger.info(f"   CRT Range: {crt_low:.2f} - {crt_high:.2f}")
            logger.info(f"   Pattern Quality: {crt_pattern.get('pattern_quality', 'A+')}")
            
            return None  # Continue to next iteration for LTF monitoring
        
        return None
    
    def _determine_direction(
        self, 
        keylevel: Dict, 
        crt_candle: pd.Series
    ) -> str:
        """
        Determine trade direction based on key level type and CRT candle.
        
        Rules:
        - OHP (Old High Purged) → SELL
        - OLP (Old Low Purged) → BUY
        - FVG above → SELL, FVG below → BUY
        - RB bearish → SELL, RB bullish → BUY
        - OB bearish → SELL, OB bullish → BUY
        
        Args:
            keylevel: Key level dictionary
            crt_candle: CRT candle series
        
        Returns:
            'sell' or 'buy'
        """
        keylevel_type = keylevel['type']
        
        if keylevel_type == 'OHP':
            return 'sell'
        elif keylevel_type == 'OLP':
            return 'buy'
        elif keylevel_type == 'FVG':
            # FVG above price → sell, below → buy
            fvg_type = keylevel.get('fvg_type', 'bullish')
            return 'sell' if fvg_type == 'bearish' else 'buy'
        elif keylevel_type == 'RB':
            # Bearish RB → sell, Bullish RB → buy
            rb_type = keylevel.get('rb_type', 'bearish')
            return 'sell' if rb_type == 'bearish' else 'buy'
        elif keylevel_type == 'OB':
            # Bearish OB → sell, Bullish OB → buy
            ob_type = keylevel.get('ob_type', 'bearish')
            return 'sell' if ob_type == 'bearish' else 'buy'
        else:
            # Fallback: use CRT candle direction (opposite)
            crt_direction = crt_candle.get('crt_direction', 'bullish')
            return 'sell' if crt_direction == 'bullish' else 'buy'
    
    def _monitor_ltf(
        self, 
        df_htf: pd.DataFrame, 
        df_ltf: pd.DataFrame
    ) -> Optional[Dict]:
        """
        Monitor LTF for TBS → Model #1 → Entry trigger.
        
        State transitions:
        LTF_MONITORING → TBS_CONFIRMED → MODEL1_CONFIRMED → Generate Signal
        
        Args:
            df_htf: HTF dataframe
            df_ltf: LTF dataframe
        
        Returns:
            Signal dictionary or None
        """
        if self.htf_setup is None:
            logger.warning("LTF monitoring called but no HTF setup. Resetting state.")
            self._reset_state()
            return None
        
        crt_pattern = self.htf_setup['crt_pattern']
        crt_timestamp = crt_pattern.get('timestamp')
        direction = self.htf_setup['direction']
        crt_levels = self.htf_setup['crt_levels']
        crt_high = crt_levels['crt_high']
        crt_low = crt_levels['crt_low']
        
        current_ltf_timestamp = df_ltf.iloc[-1]['timestamp'] if 'timestamp' in df_ltf.columns else datetime.now()
        
        # Create date-based key (YYYY-MM-DD) to prevent multiple trades on same day
        if isinstance(current_ltf_timestamp, str):
            current_date = current_ltf_timestamp[:10]  # Extract YYYY-MM-DD
        else:
            current_date = current_ltf_timestamp.strftime('%Y-%m-%d')
        
        # Enhanced setup key includes date to prevent same-day duplicates
        setup_key = f"{crt_timestamp}_{direction}_{current_date}_{crt_levels['crt_high']:.0f}_{crt_levels['crt_low']:.0f}"
        
        # Initialize setup tracker if not exists
        if not hasattr(self, '_traded_setups'):
            self._traded_setups = set()
        
        # Check if already traded THIS CRT on THIS DATE
        if setup_key in self._traded_setups:
            logger.info(
                f"Trade REJECTED (FIX #3 - DUPLICATE): Already traded CRT on {current_date}. "
                f"CRT: {crt_timestamp}, Direction: {direction.upper()}, Date: {current_date}"
            )
            self._reset_state()
            return None
        
        logger.debug(f"✅ Duplicate check passed: New setup for {current_date}")
        
        # Determine reference level for TBS detection
        if direction == 'sell':
            reference_level = crt_levels['crt_high']
        else:  # buy
            reference_level = crt_levels['crt_low']
        
        if self.state == 'LTF_MONITORING':
            # Step 1: Detect TBS pattern
            tbs = self.tbs_detector.detect_tbs_pattern(
                df_ltf,
                reference_level=reference_level,
                direction=direction
            )
            
            if tbs is None:
                return None
            
            # ✅ FIX #32: HTF Trend Filter
            # ✅ FIX #2: CORRECTED Counter-Trend Logic
            # CRT-TBS enters OPPOSITE to manipulation (reversal after liquidity grab)
            # Example: If CRT is BEARISH (high > low):
            #   - Manipulation sweeps ABOVE crt_high (fake breakout up)
            #   - Price reverses DOWN (SELL entry)
            #
            # Example: If CRT is BULLISH (low < high):
            #   - Manipulation sweeps BELOW crt_low (fake breakout down)
            #   - Price reverses UP (BUY entry)
            
            trade_direction = tbs['direction']  # Trade we're taking: 'sell' or 'buy'
            crt_direction = crt_pattern['direction']  # CRT candle direction: 'bullish' or 'bearish'
            
            # SELL Trade Validation (Reversal from manipulation UP)
            if trade_direction == 'sell':
                # For SELL: Manipulation must have swept ABOVE CRT (fake bullish breakout)
                # This means TBS reference level should be >= CRT high
                if reference_level < crt_high:
                    logger.info(
                        f"Trade REJECTED (FIX #2): SELL trade requires manipulation ABOVE CRT high. "
                        f"TBS ref: {reference_level:.2f}, CRT high: {crt_high:.2f}"
                    )
                    self._reset_state()
                    return None
                
                logger.debug(
                    f"✅ SELL REVERSAL validated: Manipulation {reference_level:.2f} >= CRT high {crt_high:.2f}"
                )
            
            # BUY Trade Validation (Reversal from manipulation DOWN)
            elif trade_direction == 'buy':
                # For BUY: Manipulation must have swept BELOW CRT (fake bearish breakout)
                # This means TBS reference level should be <= CRT low
                if reference_level > crt_low:
                    logger.info(
                        f"Trade REJECTED (FIX #2): BUY trade requires manipulation BELOW CRT low. "
                        f"TBS ref: {reference_level:.2f}, CRT low: {crt_low:.2f}"
                    )
                    self._reset_state()
                    return None
                
                logger.debug(
                    f"✅ BUY REVERSAL validated: Manipulation {reference_level:.2f} <= CRT low {crt_low:.2f}"
                )       
            
            # ✅ FIX #33: Manipulation Level Check
            # Verify TBS swept beyond CRT candle range
            if trade_direction == 'sell':
                # SELL: TBS must have swept ABOVE CRT high
                tbs_sweep_high = tbs.get('tbs_high', reference_level)
                
                if tbs_sweep_high <= crt_high:
                    logger.info(
                        f"Trade REJECTED (#33): SELL TBS sweep ({tbs_sweep_high:.2f}) "
                        f"did NOT break above CRT high ({crt_high:.2f})"
                    )
                    self._reset_state()
                    return None
                
                sweep_distance = tbs_sweep_high - crt_high
                logger.debug(f"✅ SELL manipulation check passed: Swept {sweep_distance:.2f} pts above CRT high")
            
            elif trade_direction == 'buy':
                # BUY: TBS must have swept BELOW CRT low
                tbs_sweep_low = tbs.get('tbs_low', reference_level)
                
                if tbs_sweep_low >= crt_low:
                    logger.info(
                        f"Trade REJECTED (#33): BUY TBS sweep ({tbs_sweep_low:.2f}) "
                        f"did NOT break below CRT low ({crt_low:.2f})"
                    )
                    self._reset_state()
                    return None
                
                sweep_distance = crt_low - tbs_sweep_low
                logger.debug(f"✅ BUY manipulation check passed: Swept {sweep_distance:.2f} pts below CRT low")
            
            self.ltf_setup = {'tbs': tbs}
            self.state = 'TBS_CONFIRMED'
            
            logger.info(f"TBS detected at index {tbs['tbs_index']}")
            logger.info(f"Is A+ TBS: {tbs['is_a_plus_tbs']}")
        
        if self.state == 'TBS_CONFIRMED':
            # Step 2: Detect Model #1
            tbs = self.ltf_setup['tbs']
            
            model1 = self.tbs_detector.detect_model1(
                df_ltf,
                tbs_info=tbs,
                direction=direction
            )
            
            if model1 is None:
                # Check if we've exceeded max wait time
                tbs_idx = tbs['tbs_index']
                current_idx = len(df_ltf) - 1
                candles_waited = current_idx - tbs_idx
                
                if candles_waited > self.tbs_detector.max_wait_candles:
                    logger.info(f"Model #1 timeout. Waited {candles_waited} candles. Resetting.")
                    self._reset_state()
                
                return None
            
            # ✅ FIX #34: Breakout Confirmation
            # ✅ FIX #2 (CRITICAL): Model #1 Sweep Validation WITH TOLERANCE
            # Allow sweep within 0.3% of manipulation level (institutional tolerance)
            reference_level = tbs['reference_level']
            model1_high = model1['model1_high']
            model1_low = model1['model1_low']
            
            # Calculate 0.3% tolerance (institutional standard for sweep validation)
            # Calculate 0.8% tolerance (institutional standard for Indian markets)
            # ✅ FIX #4: INCREASED SWEEP TOLERANCE FOR INDIAN INDICES
            # Indian markets have wider spreads and volatility than Forex
            # Increase tolerance from 0.8% to 1.5% (institutional standard for India)
            sweep_tolerance = abs(reference_level) * 0.015  # 1.5% buffer for Indian markets
            
            logger.debug(
                f"Model #1 sweep tolerance: ±{sweep_tolerance:.2f} points "
                f"(1.5% of manipulation level {reference_level:.2f})"
            )         
            
            if direction == 'sell':
                # ✅ FIX #4: SELL validation with 1.5% tolerance
                # Model #1 high must sweep within 1.5% of manipulation level
                min_acceptable_high = reference_level - sweep_tolerance
                
                if model1_high < min_acceptable_high:
                    logger.info(
                        f"Trade REJECTED (FIX #4): SELL Model #1 high ({model1_high:.2f}) too far from "
                        f"manipulation level ({reference_level:.2f}). "
                        f"Min acceptable: {min_acceptable_high:.2f} (tolerance: {sweep_tolerance:.2f})"
                    )
                    self._reset_state()
                    return None
                
                sweep_distance = model1_high - reference_level
                logger.debug(
                    f"✅ SELL sweep VALID: Model #1 high {model1_high:.2f} within tolerance "
                    f"(distance: {sweep_distance:+.2f}, tolerance: ±{sweep_tolerance:.2f})"
                )
            
            elif direction == 'buy':
                # ✅ FIX #4: BUY validation with 1.5% tolerance
                # Model #1 low must sweep within 1.5% of manipulation level
                max_acceptable_low = reference_level + sweep_tolerance
                
                if model1_low > max_acceptable_low:
                    logger.info(
                        f"Trade REJECTED (FIX #4): BUY Model #1 low ({model1_low:.2f}) too far from "
                        f"manipulation level ({reference_level:.2f}). "
                        f"Max acceptable: {max_acceptable_low:.2f} (tolerance: {sweep_tolerance:.2f})"
                    )
                    self._reset_state()
                    return None
                
                sweep_distance = reference_level - model1_low
                logger.debug(
                    f"✅ BUY sweep VALID: Model #1 low {model1_low:.2f} within tolerance "
                    f"(distance: {sweep_distance:+.2f}, tolerance: ±{sweep_tolerance:.2f})"
                )
            
            # ✅ FIXED: Removed duplicate elif direction == 'buy' block (lines 590-603)
            
            self.ltf_setup['model1'] = model1
            self.state = 'MODEL1_CONFIRMED'
            
            logger.info(f"Model #1 detected at index {model1['model1_index']}")
            logger.info(f"Body Ratio: {model1['body_ratio']:.2%}")
        
        if self.state == 'MODEL1_CONFIRMED':
            # Step 3: Check entry trigger
            tbs = self.ltf_setup['tbs']
            model1 = self.ltf_setup['model1']
            
            entry = self.tbs_detector.check_entry_trigger(
                df_ltf,
                model1_info=model1,
                direction=direction
            )
            
            if entry is None:
                return None
            
            # Entry trigger met! Calculate stop loss and targets
            # ✅ FIX #2: Institutional Target Calculation
            entry_price = entry['entry_price']
            crt_range = crt_high - crt_low
            
            # Calculate stop loss (we'll improve this in FIX #3)
            stop_loss = self.tbs_detector.calculate_stop_loss(
                tbs,
                model1,
                direction,
                buffer=self.stop_buffer
            )
            
            if direction == 'sell':
                # SELL: TP1 = 50% of CRT range, TP2 = CRT low + 20% extension
                tp1 = entry_price - (crt_range * 0.50)
                tp2 = crt_low - (crt_range * 0.20)
                
                # Use key level if more aggressive
                keylevel_price = self.htf_setup['keylevel'].get('price', tp2)
                if keylevel_price < (crt_high + crt_low) / 2:  # Below midpoint
                    tp2 = min(tp2, keylevel_price)  # More aggressive target
                
                logger.debug(f"SELL targets: TP1={tp1:.2f} (50% range), TP2={tp2:.2f} (low + extension)")
            
            else:  # direction == 'buy'
                # BUY: TP1 = 50% of CRT range, TP2 = CRT high + 20% extension
                tp1 = entry_price + (crt_range * 0.50)
                tp2 = crt_high + (crt_range * 0.20)
                
                # Use key level if more aggressive
                keylevel_price = self.htf_setup['keylevel'].get('price', tp2)
                if keylevel_price > (crt_high + crt_low) / 2:  # Above midpoint
                    tp2 = max(tp2, keylevel_price)  # More aggressive target
                
                logger.debug(f"BUY targets: TP1={tp1:.2f} (50% range), TP2={tp2:.2f} (high + extension)")

            
            # Calculate risk-reward ratio
            entry_price = entry['entry_price']
            risk = abs(entry_price - stop_loss)
            reward_tp1 = abs(entry_price - tp1)
            reward_tp2 = abs(entry_price - tp2)
            
            # Weighted average RR (50% at TP1, 50% at TP2)
            avg_reward = (reward_tp1 * 0.5) + (reward_tp2 * 0.5)
            rr_ratio = avg_reward / risk if risk > 0 else 0
            
            # Apply RR filter
            if rr_ratio < self.min_rr_ratio:
                logger.info(f"RR ratio {rr_ratio:.2f} below minimum {self.min_rr_ratio}. Skipping.")
                self._reset_state()
                return None
            
            # Valid signal! Generate trade signal
            signal = self._create_signal(
                entry=entry,
                stop_loss=stop_loss,
                tp1=tp1,
                tp2=tp2,
                direction=direction,
                rr_ratio=rr_ratio
            )
            
            self.signal_count += 1
            logger.info(f"Signal #{self.signal_count} generated: {direction.upper()}")
            logger.info(f"Entry: {entry_price:.2f} | SL: {stop_loss:.2f} | TP1: {tp1:.2f} | TP2: {tp2:.2f}")
            logger.info(f"RR Ratio: {rr_ratio:.2f}")
            
            # Reset state for next setup
            self._reset_state()
            
            return signal
        
        return None
    
    def _create_signal(
        self,
        entry: Dict,
        stop_loss: float,
        tp1: float,
        tp2: float,
        direction: str,
        rr_ratio: float
    ) -> Dict:
        """
        Create trade signal dictionary compatible with backtesting infrastructure.
        
        Args:
            entry: Entry trigger dictionary
            stop_loss: Stop loss price
            tp1: First take profit (50%)
            tp2: Second take profit (100%)
            direction: 'sell' or 'buy'
            rr_ratio: Risk-reward ratio
        
        Returns:
            Signal dictionary for backtesting engine
        """
        # ✅ FIXED: Create signal FIRST (don't return immediately)
        signal = {
            'action': 'SELL' if direction == 'sell' else 'BUY',
            'entry_price': entry['entry_price'],
            'stop_loss': stop_loss,
            'take_profit_1': tp1,
            'take_profit_2': tp2,
            'position_scaling': [0.5, 0.5],
            'move_to_breakeven': 'after_tp1',
            'rr_ratio': rr_ratio,
            'confidence': self._calculate_confidence(),
            'timestamp': entry.get('timestamp', datetime.now()),
            'setup_info': {
                'htf_setup': self.htf_setup,
                'ltf_setup': self.ltf_setup,
                'strategy': self.name
            }
        }
        
        # ✅ FIX #6: Mark this setup as traded to prevent duplicates
        if hasattr(self, '_traded_setups'):
            crt_pattern = self.htf_setup['crt_pattern']
            crt_timestamp = crt_pattern.get('timestamp')
            crt_levels = self.htf_setup['crt_levels']
            setup_key = f"{crt_timestamp}_{direction}_{crt_levels['crt_high']:.2f}_{crt_levels['crt_low']:.2f}"
            self._traded_setups.add(setup_key)
            logger.debug(f"Marked setup as traded: {setup_key}")
        
        return signal  # ✅ Return at the END
    
    def _calculate_confidence(self) -> float:
        """
        Calculate signal confidence score (0.0 to 1.0).
        
        Factors:
        - A+ TBS pattern: +0.2
        - Strong body ratios: +0.1
        - OHP/OLP key level: +0.1
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.5  # Base confidence
        
        if self.ltf_setup and 'tbs' in self.ltf_setup:
            # A+ TBS bonus
            if self.ltf_setup['tbs'].get('is_a_plus_tbs', False):
                confidence += 0.2
            
            # Strong Model #1 bonus
            if 'model1' in self.ltf_setup:
                body_ratio = self.ltf_setup['model1'].get('body_ratio', 0)
                if body_ratio > 0.70:
                    confidence += 0.1
        
        if self.htf_setup and 'keylevel' in self.htf_setup:
            # OHP/OLP bonus (most reliable)
            keylevel_type = self.htf_setup['keylevel'].get('type')
            if keylevel_type in ['OHP', 'OLP']:
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _reset_state(self):
        """Reset strategy state to start scanning for new setups."""
        self.state = 'HTF_SCANNING'
        self.htf_setup = None
        self.ltf_setup = None
    
    def get_strategy_info(self) -> Dict:
        """
        Get current strategy information and statistics.
        
        Returns:
            Dictionary with strategy info
        """
        return {
            'name': self.name,
            'current_state': self.state,
            'htf': self.config.get('htf', '1D'),
            'ltf': self.config.get('ltf', '1H'),
            'setup_count': self.setup_count,
            'signal_count': self.signal_count,
            'min_rr_ratio': self.min_rr_ratio,
            'has_active_setup': self.htf_setup is not None
        }


if __name__ == "__main__":
    # Test 1: Auto-config for crypto
    print("=== Test 1: Cryptocurrency ===")
    strategy_crypto = StrategyCRTTBS(market_type='Cryptocurrency')
    print(strategy_crypto.get_strategy_info())
    
    # Test 2: Auto-config for forex
    print("\n=== Test 2: Forex ===")
    strategy_forex = StrategyCRTTBS(market_type='Forex')
    print(strategy_forex.get_strategy_info())
    
    # Test 3: Manual config (backward compatible)
    print("\n=== Test 3: Manual Config ===")
    test_config = {
        'htf': '1D',
        'ltf': '1H',
        'crt_method': 'body_vs_wicks',
        'ohp_olp_lookback': 10,
        'min_rr_ratio': 1.5,
        'risk_per_trade': 1.0
    }
    strategy_manual = StrategyCRTTBS(config=test_config)
    
    print("Strategy Info:")
    info = strategy.get_strategy_info()
    for key, value in info.items():
        print(f"{key}: {value}")
