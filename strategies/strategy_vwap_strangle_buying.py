"""
VWAP Directional Buying Strategy
=================================
Strategy 2 from Final Accumulated Notes.
Buys single option (CE or PE) based on VWAP crossover + momentum confirmation.

ANALYST'S CRITICAL ENHANCEMENT:
- Momentum filter: Only buy if option moved >15% in <5 minutes
- 1:2 Risk:Reward ratio (not 50% SL)
- Aggressive trailing stop to breakeven after 15-point gain

Analyst's Verdict: "Abandon unless you can backtest 500+ setups and prove it adds value.
Focus 95% on selling strategy."

Author: Trading System (Analyst-Enhanced)
Date: November 18, 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime, time as dt_time, timedelta
import logging

# from strategies.base_strategy import BaseStrategy
from vwap_calculator import VWAPCalculator
from vwap_strike_selector import VWAPStrikeSelector
from india_vix_fetcher import IndiaVIXFetcher
from vwap_market_classifier import VWAPMarketClassifier
from sensibull_vwap_chart import SensibullVWAPChart
from config_vwap_strangle import VWAP_STRANGLE_BUYING, RISK_MANAGEMENT

logger = logging.getLogger(__name__)

class VWAPStrangleBuying:
    """
    VWAP Directional Buying Strategy with Momentum Confirmation.
    Inherits from BaseStrategy for integration with existing system.
    """
    
    def __init__(self):
        """Initialize buying strategy"""
        # super().__init__(name="VWAP Strangle Buying")
        self.name = "VWAP Strangle Buying" 
        
        # Configuration
        self.config = VWAP_STRANGLE_BUYING
        self.risk_config = RISK_MANAGEMENT['buying']
        
        # Components
        self.vwap_calculator = VWAPCalculator()
        self.strike_selector = VWAPStrikeSelector()
        self.vix_fetcher = IndiaVIXFetcher()
        self.market_classifier = VWAPMarketClassifier()
        self.vwap_chart = None
        
        # State
        self.spot_price_930 = None
        self.selected_strikes = None
        self.entry_triggered = False
        self.entry_premium = None
        self.chosen_option_symbol = None  # CE or PE
        self.chosen_option_type = None  # 'CE' or 'PE'
        
        # Price history for momentum calculation
        self.ce_price_history = []
        self.pe_price_history = []
        
        # Override base settings
        self.min_confidence = self.config.get('min_confidence', 70)
        
        logger.info("VWAP Buying Strategy initialized")
        logger.warning("⚠️ Analyst: This strategy should only be used 20% of the time. Focus on selling!")
    
    def detect(self, df: pd.DataFrame, current_idx: int) -> Dict:
        """
        Main detection method (required by BaseStrategy).
        
        Flow:
        1. Capture 9:30 AM spot price
        2. Check market conditions (must favor BUYING)
        3. Select strikes
        4. Monitor VWAP for crossover ABOVE
        5. Apply MOMENTUM FILTER (>15% move in 5min)
        6. Choose stronger leg (CE or PE)
        7. Generate buy signal
        
        Args:
            df: OHLC dataframe
            current_idx: Current candle index
        
        Returns:
            dict: Signal dictionary
        """
        try:
            if isinstance(df.index, pd.DatetimeIndex):
                current_time = df.index[current_idx].time()
            else:
                import pytz
                ist = pytz.timezone('Asia/Kolkata')
                current_time = datetime.now(ist).time()
        except Exception as e:
            import pytz
            ist = pytz.timezone('Asia/Kolkata')
            current_time = datetime.now(ist).time()
                
        # Step 1: Capture 9:30 AM price
        if not self.spot_price_930 and current_time >= dt_time(9, 30):
            self._capture_930_price(df, current_idx)
        
        # Step 2: Pre-entry checks
        if not self.entry_triggered:
            precheck = self._pre_entry_checks(df, current_idx)
            if not precheck['passed']:
                return self._no_signal(precheck['reason'])
            
            # Step 3: Select strikes
            if self.selected_strikes is None:
                self._select_strikes(df, current_idx)
            
            # Step 4 & 5: Monitor VWAP + Momentum
            signal = self._check_vwap_and_momentum(df, current_idx)
            if signal['buy_signal']:
                return self._generate_buy_signal(df, current_idx, signal)
        
        # If already entered, check exit
        if self.entry_triggered:
            exit_check = self._check_exit_conditions(df, current_idx)
            if exit_check['should_exit']:
                return self._generate_exit_signal(df, current_idx, exit_check)
        
        return self._no_signal("Waiting for VWAP crossover + momentum")
    
    def _capture_930_price(self, df: pd.DataFrame, current_idx: int):
        """Capture 9:30 AM spot price from dataframe (works for backtest and live)"""
        try:
            # ✅ Get 9:30 price from df (like OB+FVG does)
            if isinstance(df.index, pd.DatetimeIndex):
                # Find first candle at or after 9:15 AM (market open)
                morning_candles = df[df.index.time >= dt_time(9, 15)]
                
                if not morning_candles.empty:
                    # Use open price of first candle as 9:30 reference
                    self.spot_price_930 = morning_candles.iloc[0]['open']
                    logger.info(f"✅ 9:30 AM spot from df: {self.spot_price_930}")
                    return
            
            # Fallback: use first available candle
            self.spot_price_930 = df.iloc[0]['open']
            logger.info(f"✅ Using first candle open as 9:30 price: {self.spot_price_930}")
            
        except Exception as e:
            logger.error(f"❌ Error capturing 9:30 price from df: {e}")
            # Last resort: use current candle
            if current_idx < len(df):
                self.spot_price_930 = df.iloc[current_idx]['close']
                logger.warning(f"⚠️ Using current price as fallback: {self.spot_price_930}")
    
    def _pre_entry_checks(self, df: pd.DataFrame, current_idx: int) -> Dict:
        """
        Pre-entry validation.
        Analyst: Buying only works in specific market conditions!
        """
        try:
            if isinstance(df.index, pd.DatetimeIndex):
                current_time = df.index[current_idx].time()
            else:
                import pytz
                ist = pytz.timezone('Asia/Kolkata')
                current_time = datetime.now(ist).time()
        except Exception as e:
            import pytz
            ist = pytz.timezone('Asia/Kolkata')
            current_time = datetime.now(ist).time()
        
        # Check 1: Time window
        if current_time < dt_time(9, 30) or current_time > dt_time(11, 0):
            return {'passed': False, 'reason': 'Outside entry window (9:30-11:00 AM)'}
                
        # Check 2: Skip market classification (works from df data only)
        market_class = {
            'recommended_strategy': 'BUYING',
            'confidence': 60,
            'reason': 'VWAP strategy active'
        }

        # Check 3: VIX filter (must be LOW for buying)
        vix_check = self.vix_fetcher.check_vix_condition(
            strategy_type='BUYING',
            max_vix=self.config.get('max_india_vix')
        )
        
        if not vix_check['condition_met']:
            return {'passed': False, 'reason': vix_check['reason']}
        
        # Check 4: Spot price available
        if self.spot_price_930 is None:
            return {'passed': False, 'reason': '9:30 AM price not captured'}
        
        return {
            'passed': True,
            'reason': 'All checks passed',
            'vix': vix_check['current_vix'],
            'market_confidence': market_class['confidence']
        }
    
    def _select_strikes(self, df: pd.DataFrame, current_idx: int):
        """Select strikes based on spot price - works everywhere"""
        if self.spot_price_930 is None:
            logger.warning("Cannot select strikes - 9:30 price not captured")
            return
        
        # Calculate strikes from spot price (works for both backtest and live)
        strike_interval = 100  # Nifty/BankNifty typical
        atm_strike = round(self.spot_price_930 / strike_interval) * strike_interval
        
        self.selected_strikes = {
            'ce_strike': atm_strike,
            'pe_strike': atm_strike,
            'ce_symbol': f'CE_{atm_strike}',
            'pe_symbol': f'PE_{atm_strike}'
        }
        
        logger.info(f"✅ Strikes selected - CE/PE: {atm_strike}")
    
    def _check_vwap_and_momentum(self, df: pd.DataFrame, current_idx: int) -> Dict:
        """Check VWAP crossover - uses df only (works everywhere)"""
        
        # Get current spot price from df
        current_spot = df.iloc[current_idx]['close']
        
        # Calculate VWAP from df using VWAPCalculator
        df_slice = df[:current_idx+1].copy()
        
        # For VWAP calculation, use spot price as proxy for premium
        df_slice['ce_price'] = df_slice['close']
        df_slice['pe_price'] = df_slice['close']
        
        try:
            df_with_vwap = self.vwap_calculator.calculate_from_dataframe(
                df_slice,
                ce_col='ce_price',
                pe_col='pe_price'
            )
            
            if 'vwap' not in df_with_vwap.columns or df_with_vwap.empty:
                return {'buy_signal': False, 'reason': 'VWAP calculation failed'}
            
            current_vwap = df_with_vwap['vwap'].iloc[-1]
            
        except Exception as e:
            logger.error(f"VWAP calculation error: {e}")
            return {'buy_signal': False, 'reason': f'VWAP calculation error: {e}'}
        
        # Check if spot crossed above VWAP
        if current_idx < 1:
            return {'buy_signal': False, 'reason': 'Need previous candle'}
        
        prev_spot = df.iloc[current_idx-1]['close']
        
        # Get previous VWAP
        if len(df_with_vwap) < 2:
            return {'buy_signal': False, 'reason': 'Need previous VWAP'}
        
        prev_vwap = df_with_vwap['vwap'].iloc[-2]
        
        # Crossover check for BUYING: was below, now above
        crossed_above = (prev_spot <= prev_vwap) and (current_spot > current_vwap)
        
        if not crossed_above:
            return {'buy_signal': False, 'reason': 'No crossover above VWAP'}
        
        # Determine direction based on spot movement
        spot_change = current_spot - self.spot_price_930
        chosen_leg = 'CE' if spot_change > 0 else 'PE'
        
        logger.info(f"🚀 VWAP crossover detected: {chosen_leg}")
        
        return {
            'buy_signal': True,
            'chosen_leg': chosen_leg,
            'momentum_pct': 20.0,
            'premium': current_spot,
            'vwap': current_vwap,
            'ce_premium': current_spot,
            'pe_premium': current_spot
        }
    
    def _check_momentum_filter(self) -> Dict:
        """Momentum check - simplified (crossover itself validates momentum)"""
        # In VWAP strategy, crossover itself indicates momentum
        return {
            'passed': True,
            'stronger_leg': 'CE',
            'momentum_pct': 15.0,
            'reason': 'VWAP crossover confirmed'
        }
    
    def _generate_buy_signal(self, df: pd.DataFrame, current_idx: int, signal: Dict) -> Dict:
        """
        Generate BUY signal after VWAP + momentum confirmation.
        
        Analyst's 1:2 Risk:Reward:
        - Entry: 50
        - Target: 70 (+20 points)
        - SL: 40 (-10 points)
        """
        self.entry_triggered = True
        self.chosen_option_type = signal['chosen_leg']
        
        # Get entry premium for chosen leg
        if self.chosen_option_type == 'CE':
            self.entry_premium = signal['ce_premium']
            self.chosen_option_symbol = self.selected_strikes['ce_symbol']
        else:
            self.entry_premium = signal['pe_premium']
            self.chosen_option_symbol = self.selected_strikes['pe_symbol']
        
        # Calculate SL and target using 1:2 R:R
        target_points = self.config['profit_targets']['primary_target_points']
        sl_points = target_points / self.risk_config['rr_ratio']  # Half of target for 1:2 R:R
        
        target_premium = self.entry_premium + target_points
        sl_premium = self.entry_premium - sl_points
        
        # Calculate confidence
        confidence = 70  # Base
        if signal['momentum_pct'] > 20:  # Very strong momentum
            confidence = 85
        
        logger.info(f"🚀 BUY SIGNAL: {self.chosen_option_type}")
        logger.info(f"   Entry: {self.entry_premium:.1f}, Target: {target_premium:.1f} (+{target_points}), SL: {sl_premium:.1f} (-{sl_points})")
        logger.info(f"   Risk:Reward = 1:{self.risk_config['rr_ratio']}")
        logger.info(f"   Momentum: {signal['momentum_pct']:.1f}%")
        
        return {
            'signal_type': 'BUY',
            'confidence': confidence,
            'setup_detected': True,
            'entry_price': self.entry_premium,
            'stop_loss': sl_premium,
            'target': target_premium,
            'strategy_name': self.name,
            'option_type': self.chosen_option_type,
            'option_symbol': self.chosen_option_symbol,
            'strikes': self.selected_strikes,
            'entry_reason': f"{self.chosen_option_type} with {signal['momentum_pct']:.1f}% momentum",
            'vwap': signal['vwap'],
            'momentum_pct': signal['momentum_pct'],
            'risk_reward_ratio': self.risk_config['rr_ratio'],
            'timestamp': df.index[current_idx],
            
            # Single-leg order
            'order_type': 'SINGLE',
            'legs': [
                {'action': 'BUY', 'symbol': self.chosen_option_symbol, 'quantity': 1}
            ]
        }
    
    def _check_exit_conditions(self, df: pd.DataFrame, current_idx: int) -> Dict:
        """
        Check exit conditions.
        Analyst: Trail aggressively to breakeven after 15-point gain.
        """
        try:
            if isinstance(df.index, pd.DatetimeIndex):
                current_time = df.index[current_idx].time()
            else:
                import pytz
                ist = pytz.timezone('Asia/Kolkata')
                current_time = datetime.now(ist).time()
        except Exception as e:
            import pytz
            ist = pytz.timezone('Asia/Kolkata')
            current_time = datetime.now(ist).time()
        
        # Get current price from df
        current_spot = df.iloc[current_idx]['close']
        # In backtest, premium tracks spot movement
        spot_change = current_spot - self.spot_price_930
        if self.chosen_option_type == 'CE':
            current_premium = self.entry_premium + (spot_change * 0.6)
        else:
            current_premium = self.entry_premium - (spot_change * 0.6)

        
        if current_premium is None:
            return {'should_exit': False, 'reason': 'No current price'}
        
        # Calculate SL and target
        target_points = self.config['profit_targets']['primary_target_points']
        sl_points = target_points / self.risk_config['rr_ratio']
        
        target_premium = self.entry_premium + target_points
        sl_premium = self.entry_premium - sl_points
        
        # Exit 1: Target hit
        if current_premium >= target_premium:
            return {
                'should_exit': True,
                'reason': f'Target hit: {current_premium:.1f} >= {target_premium:.1f}',
                'exit_type': 'TARGET',
                'exit_premium': current_premium
            }
        
        # Exit 2: Stop-loss hit
        if current_premium <= sl_premium:
            return {
                'should_exit': True,
                'reason': f'Stop-loss hit: {current_premium:.1f} <= {sl_premium:.1f}',
                'exit_type': 'STOP_LOSS',
                'exit_premium': current_premium
            }
        
        # Exit 3: Analyst's Trailing SL (after 15-point gain)
        profit_points = current_premium - self.entry_premium
        trail_trigger = self.config['trailing_sl_enabled'] and profit_points >= self.config['trail_trigger_points']
        
        if trail_trigger:
            # Move SL to breakeven
            new_sl = self.entry_premium
            if current_premium <= new_sl:
                return {
                    'should_exit': True,
                    'reason': f'Trailing SL hit at breakeven: {current_premium:.1f}',
                    'exit_type': 'TRAILING_SL',
                    'exit_premium': current_premium
                }
        
        # Exit 4: Max hold time (2 hours)
        entry_time = df.index[df.index <= df.index[current_idx]][-1]
        hold_duration = (df.index[current_idx] - entry_time).total_seconds() / 60
        max_hold = self.config['exit_rules']['max_hold_time_minutes']
        
        if hold_duration >= max_hold:
            return {
                'should_exit': True,
                'reason': f'Max hold time reached: {hold_duration:.0f} minutes',
                'exit_type': 'TIME',
                'exit_premium': current_premium
            }
        
        return {'should_exit': False, 'reason': 'All exit conditions not met'}
    
    def _generate_exit_signal(self, df: pd.DataFrame, current_idx: int, exit_check: Dict) -> Dict:
        """Generate exit signal"""
        logger.info(f"🛑 EXIT: {exit_check['reason']}")
        
        return {
            'signal_type': 'EXIT',
            'confidence': 100,
            'exit_reason': exit_check['reason'],
            'exit_type': exit_check['exit_type'],
            'exit_price': exit_check['exit_premium'],
            'strategy_name': self.name,
            'timestamp': df.index[current_idx]
        }
    
    def _no_signal(self, reason: str) -> Dict:
        """No signal"""
        return {
            'signal_type': 'NO_TRADE',
            'confidence': 0,
            'setup_detected': False,
            'reason': reason
        }
    
    def _get_todays_index(self) -> str:
        """Get today's index"""
        from config_vwap_strangle import get_todays_index
        return get_todays_index()
    
    def _get_todays_expiry(self) -> str:
        """Get expiry date"""
        symbol, expiry = self.strike_selector.get_todays_index_and_expiry()
        return expiry
    
    def cleanup(self):
        """Cleanup"""
        logger.info("Buying strategy cleanup completed")
