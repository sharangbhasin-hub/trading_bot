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
        super().__init__(name="VWAP Strangle Buying")
        
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
        current_time = df.index[current_idx].time()
        
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
        """Capture 9:30 AM spot price"""
        symbol = self._get_todays_index()
        self.spot_price_930 = self.strike_selector.capture_930_spot_price(symbol)
        
        if self.spot_price_930:
            logger.info(f"✅ 9:30 AM spot: {self.spot_price_930}")
    
    def _pre_entry_checks(self, df: pd.DataFrame, current_idx: int) -> Dict:
        """
        Pre-entry validation.
        Analyst: Buying only works in specific market conditions!
        """
        current_time = df.index[current_idx].time()
        
        # Check 1: Time window
        if current_time < dt_time(9, 30) or current_time > dt_time(11, 0):
            return {'passed': False, 'reason': 'Outside entry window (9:30-11:00 AM)'}
        
        # Check 2: Market must favor BUYING
        symbol = self._get_todays_index()
        market_class = self.market_classifier.classify_market(symbol)
        
        if market_class['recommended_strategy'] != 'BUYING':
            return {
                'passed': False,
                'reason': f"Market favors {market_class['recommended_strategy']}: {market_class['reason']}"
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
        """Select strikes for watchlist"""
        symbol = self._get_todays_index()
        expiry = self._get_todays_expiry()
        
        self.selected_strikes = self.strike_selector.select_strikes_for_buying(
            symbol=symbol,
            expiry_date=expiry,
            spot_price=self.spot_price_930
        )
        
        if self.selected_strikes:
            logger.info(f"✅ Strikes selected - CE: {self.selected_strikes['ce_strike']}, PE: {self.selected_strikes['pe_strike']}")
            self._initialize_vwap_chart()
    
    def _initialize_vwap_chart(self):
        """Initialize VWAP chart monitoring"""
        if not self.selected_strikes:
            return
        
        # Same chart as selling (for consistency)
        self.vwap_chart = SensibullVWAPChart(
            ce_symbol=self.selected_strikes['ce_symbol'],
            pe_symbol=self.selected_strikes['pe_symbol']
        )
        
        self.vwap_chart.start()
        logger.info("📊 VWAP chart monitoring started")
    
    def _check_vwap_and_momentum(self, df: pd.DataFrame, current_idx: int) -> Dict:
        """
        ANALYST'S CRITICAL ENHANCEMENT: Check VWAP crossover + momentum filter.
        
        Two-stage process:
        1. Check if combined premium crossed ABOVE VWAP
        2. If yes, check which leg has >15% momentum in 5 minutes
        
        Returns:
            dict: {
                'buy_signal': bool,
                'chosen_leg': 'CE' or 'PE',
                'momentum_pct': float,
                'premium': float,
                'vwap': float
            }
        """
        if not self.vwap_chart or not self.vwap_chart.is_running:
            return {'buy_signal': False, 'reason': 'Chart not running'}
        
        state = self.vwap_chart.get_current_state()
        
        if state['combined_premium'] is None or state['vwap'] is None:
            return {'buy_signal': False, 'reason': 'Waiting for data'}
        
        # STAGE 1: Check VWAP crossover ABOVE
        crossover = self.vwap_calculator.check_crossover(
            current_premium=state['combined_premium'],
            direction='above'  # For BUYING
        )
        
        if not crossover['crossed']:
            return {'buy_signal': False, 'reason': 'No crossover above VWAP'}
        
        # STAGE 2: ANALYST'S MOMENTUM FILTER
        # Must move >15% in 5 minutes
        momentum_check = self._check_momentum_filter()
        
        if not momentum_check['passed']:
            logger.warning(f"⚠️ VWAP crossed but momentum insufficient: {momentum_check['reason']}")
            return {
                'buy_signal': False,
                'reason': f"Momentum filter failed: {momentum_check['reason']}"
            }
        
        # Both conditions met!
        logger.info(f"🚀 VWAP + MOMENTUM confirmed: {momentum_check['stronger_leg']} with {momentum_check['momentum_pct']:.1f}% move")
        
        return {
            'buy_signal': True,
            'chosen_leg': momentum_check['stronger_leg'],
            'momentum_pct': momentum_check['momentum_pct'],
            'premium': crossover['premium'],
            'vwap': crossover['vwap'],
            'ce_premium': state['ce_price'],
            'pe_premium': state['pe_price']
        }
    
    def _check_momentum_filter(self) -> Dict:
        """
        ANALYST'S CRITICAL ADDITION: Momentum filter.
        Only buy if chosen option moved >15% in <5 minutes.
        
        Returns:
            dict: {
                'passed': bool,
                'stronger_leg': 'CE' or 'PE',
                'momentum_pct': float,
                'reason': str
            }
        """
        # Get 5-minute price history
        ce_history = self._get_recent_prices('CE', minutes=5)
        pe_history = self._get_recent_prices('PE', minutes=5)
        
        if len(ce_history) < 2 or len(pe_history) < 2:
            return {
                'passed': False,
                'reason': 'Insufficient price history (need 5 minutes)'
            }
        
        # Calculate % change over last 5 minutes
        ce_change_pct = ((ce_history[-1] - ce_history[0]) / ce_history[0]) * 100 if ce_history[0] > 0 else 0
        pe_change_pct = ((pe_history[-1] - pe_history[0]) / pe_history[0]) * 100 if pe_history[0] > 0 else 0
        
        # Analyst's threshold: >15% move
        min_move_pct = self.config['entry_signal']['momentum_filter']['min_move_pct']
        
        # Check which leg has stronger momentum
        if abs(ce_change_pct) >= min_move_pct:
            return {
                'passed': True,
                'stronger_leg': 'CE',
                'momentum_pct': ce_change_pct,
                'reason': f'CE momentum {ce_change_pct:.1f}% >= {min_move_pct}%'
            }
        elif abs(pe_change_pct) >= min_move_pct:
            return {
                'passed': True,
                'stronger_leg': 'PE',
                'momentum_pct': pe_change_pct,
                'reason': f'PE momentum {pe_change_pct:.1f}% >= {min_move_pct}%'
            }
        else:
            return {
                'passed': False,
                'reason': f'Momentum too weak - CE: {ce_change_pct:.1f}%, PE: {pe_change_pct:.1f}% (need >{min_move_pct}%)'
            }
    
    def _get_recent_prices(self, option_type: str, minutes: int = 5) -> List[float]:
        """
        Get recent price history for momentum calculation.
        
        Args:
            option_type: 'CE' or 'PE'
            minutes: Lookback period in minutes
        
        Returns:
            list: Price history
        """
        try:
            # Get from chart data
            chart_df = self.vwap_chart.get_chart_dataframe()
            
            if chart_df.empty:
                logger.debug(f"Chart data empty - cannot get {option_type} prices")
                return []
            
            # Validate column exists
            price_col = 'ce_price' if option_type == 'CE' else 'pe_price'
            if price_col not in chart_df.columns:
                logger.warning(f"Column '{price_col}' not found in chart data")
                return []
            
            # Filter last N minutes (handle timezone-aware index)
            if isinstance(chart_df.index, pd.DatetimeIndex):
                # Use pandas timedelta for better compatibility
                cutoff_time = pd.Timestamp.now() - pd.Timedelta(minutes=minutes)
                recent = chart_df[chart_df.index >= cutoff_time]
            else:
                # Fallback: take last N rows (assuming 1-minute candles)
                recent = chart_df.tail(minutes)
            
            # Get prices as list
            prices = recent[price_col].dropna().tolist()
            
            logger.debug(f"Got {len(prices)} price points for {option_type} (last {minutes} minutes)")
            return prices
            
        except Exception as e:
            logger.error(f"Error getting recent prices for {option_type}: {e}")
            return []
    
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
        current_time = df.index[current_idx].time()
        
        # Get current price
        state = self.vwap_chart.get_current_state()
        if self.chosen_option_type == 'CE':
            current_premium = state.get('ce_price')
        else:
            current_premium = state.get('pe_price')
        
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
        if self.vwap_chart:
            self.vwap_chart.stop()
        logger.info("Buying strategy cleanup completed")
