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
# from detectors.keylevel_detector import KeyLevelDetector
# from detectors.tbs_detector import TBSDetector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyCRTTBS(BaseStrategy):
    """
    CRT-TBS Strategy Implementation.
    
    Extends BaseStrategy to integrate with existing backtesting infrastructure.
    Implements state machine for setup detection and trade execution.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize CRT-TBS Strategy.
        
        Args:
            config: Strategy configuration dictionary with:
                   
                   Timeframes:
                   - htf: Higher timeframe ('1H', '4H', '1D')
                   - ltf: Lower timeframe ('1m', '5m', '1H')
                   
                   CRT Detection:
                   - crt_method: 'body_vs_wicks' or 'ratio'
                   - crt_min_body_ratio: If using ratio (default 0.50)
                   
                   Key Level Detection:
                   - ohp_olp_lookback: Lookback period (default 20)
                   - fvg_min_gap_percent: Minimum FVG gap (default 0.05)
                   - ob_min_consecutive: Min consecutive candles for OB (default 5)
                   - rb_min_wick_ratio: Min wick ratio for RB (default 0.40)
                   
                   TBS Detection:
                   - allow_multi_candle: Allow 2-6 candle patterns (default True)
                   - valid_candle_range: [2, 6] for A+ TBS
                   
                   Model #1 Detection:
                   - model1_method: 'body_vs_wicks' or 'ratio'
                   - model1_min_body_ratio: If using ratio (default 0.60)
                   - max_wait_candles: Max wait for Model #1 (default 10)
                   
                   Risk Management:
                   - min_rr_ratio: Minimum risk-reward ratio (default 1.5)
                   - stop_buffer: Buffer for stop loss in price units (default 0.0)
                   - risk_per_trade: Risk percentage per trade (default 1.0)
        """
        super().__init__(config)
        
        self.config = config or {}
        self.name = "CRT_TBS"
        
        # Initialize detectors with configuration
        self.crt_detector = CRTDetector(self._get_crt_config())
        self.keylevel_detector = KeyLevelDetector(self._get_keylevel_config())
        self.tbs_detector = TBSDetector(self._get_tbs_config())
        
        # Strategy state machine
        self.state = 'HTF_SCANNING'
        self.htf_setup = None
        self.ltf_setup = None
        
        # Risk management parameters
        self.min_rr_ratio = self.config.get('min_rr_ratio', 1.5)
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
        1. Detect CRT candles
        2. Check for key levels at CRT candle
        3. If setup complete, transition to LTF monitoring
        
        Args:
            df_htf: HTF dataframe
            df_ltf: LTF dataframe
        
        Returns:
            None (updates internal state)
        """
        if len(df_htf) < 10:
            return None
        
        # Step 1: Detect CRT candles
        df_htf_with_crt = self.crt_detector.detect_crt_candles(df_htf)
        
        # Get most recent CRT candles (last 5)
        recent_crts = df_htf_with_crt[df_htf_with_crt['is_crt']].tail(5)
        
        if recent_crts.empty:
            return None
        
        # Step 2: Check each CRT candle for key levels
        for idx, crt_candle in recent_crts.iterrows():
            crt_idx = df_htf.index.get_loc(idx)
            
            # Detect all key levels at this CRT candle
            keylevels = self.keylevel_detector.detect_all_keylevels(
                df_htf, 
                crt_candle_idx=crt_idx
            )
            
            if not keylevels['has_any_keylevel']:
                continue
            
            # Get primary key level
            primary_keylevel = self.keylevel_detector.get_primary_keylevel(keylevels)
            
            if primary_keylevel is None:
                continue
            
            # Step 3: Valid HTF setup found!
            crt_levels = self.crt_detector.get_crt_levels(crt_candle)
            
            # Determine trade direction based on key level type
            direction = self._determine_direction(primary_keylevel, crt_candle)
            
            # Store HTF setup
            self.htf_setup = {
                'crt_candle': crt_candle,
                'crt_levels': crt_levels,
                'keylevel': primary_keylevel,
                'direction': direction,
                'htf_timestamp': idx,
                'setup_timestamp': datetime.now()
            }
            
            self.setup_count += 1
            self.state = 'LTF_MONITORING'
            
            logger.info(f"HTF Setup #{self.setup_count} detected: {direction} @ {idx}")
            logger.info(f"Key Level: {primary_keylevel['type']}")
            logger.info(f"CRT Range: {crt_levels['crt_low']:.2f} - {crt_levels['crt_high']:.2f}")
            
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
        
        direction = self.htf_setup['direction']
        crt_levels = self.htf_setup['crt_levels']
        
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
            stop_loss = self.tbs_detector.calculate_stop_loss(
                tbs,
                model1,
                direction,
                buffer=self.stop_buffer
            )
            
            tp1 = crt_levels['tp1_level']
            tp2 = crt_levels['tp2_sell'] if direction == 'sell' else crt_levels['tp2_buy']
            
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
        return {
            'action': 'SELL' if direction == 'sell' else 'BUY',
            'entry_price': entry['entry_price'],
            'stop_loss': stop_loss,
            'take_profit_1': tp1,
            'take_profit_2': tp2,
            'position_scaling': [0.5, 0.5],  # Close 50% at each TP
            'move_to_breakeven': 'after_tp1',  # Critical: Only after TP1
            'rr_ratio': rr_ratio,
            'confidence': self._calculate_confidence(),
            'timestamp': entry.get('timestamp', datetime.now()),
            'setup_info': {
                'htf_setup': self.htf_setup,
                'ltf_setup': self.ltf_setup,
                'strategy': self.name
            }
        }
    
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


# Example usage
if __name__ == "__main__":
    # Test configuration
    test_config = {
        'htf': '1D',
        'ltf': '1H',
        'crt_method': 'body_vs_wicks',
        'ohp_olp_lookback': 10,
        'min_rr_ratio': 1.5,
        'risk_per_trade': 1.0
    }
    
    strategy = StrategyCRTTBS(config=test_config)
    
    print("Strategy Info:")
    info = strategy.get_strategy_info()
    for key, value in info.items():
        print(f"{key}: {value}")
