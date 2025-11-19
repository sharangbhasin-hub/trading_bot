"""
VWAP Short Strangle Strategy (SELLING)
=======================================
Strategy 1 from Final Accumulated Notes.
Profits from theta decay using VWAP crossover as entry signal.

Setup: 4-leg iron strangle
Entry: Premium crosses BELOW VWAP (after being above)
Exit: 1:30 PM or target/SL hit
Win Rate Target: 80%+

Follows BaseStrategy pattern for integration with existing system.

Author: Trading System
Date: October 25, 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime, time as dt_time
import logging
# from strategies.base_strategy import BaseStrategy
from vwap_calculator import VWAPCalculator
from vwap_strike_selector import VWAPStrikeSelector
from india_vix_fetcher import IndiaVIXFetcher
from vwap_market_classifier import VWAPMarketClassifier
from sensibull_vwap_chart import SensibullVWAPChart
from config_vwap_strangle import VWAP_STRANGLE_SELLING, RISK_MANAGEMENT

logger = logging.getLogger(__name__)

class VWAPStrangleSelling:
    """
    VWAP Short Strangle Strategy (Selling)
    Inherits from BaseStrategy to integrate with existing strategy manager.
    """
    
    def __init__(self):
        """Initialize strategy"""
        # super().__init__(name="VWAP Strangle Selling")
        
        # Configuration
        self.config = VWAP_STRANGLE_SELLING
        self.risk_config = RISK_MANAGEMENT['selling']
        
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
        
        # Override base settings
        self.min_confidence = self.config.get('min_confidence', 65)
    
    def detect(self, df: pd.DataFrame, current_idx: int) -> Dict:
        """
        Main detection method (required by BaseStrategy).
        Called by StrategyManager for each candle.
        
        For this strategy, we override the flow:
        1. Check if it's 9:30 AM - capture spot price
        2. Check market conditions - should we trade SELLING today?
        3. Select strikes
        4. Monitor VWAP chart for crossover signal
        5. Generate entry signal when crossover detected
        
        Args:
            df: OHLC dataframe
            current_idx: Current candle index
        
        Returns:
            dict: Signal dictionary per BaseStrategy format
        """
        current_time = df.index[current_idx].time()
        
        # Step 1: Capture 9:30 AM spot price (one-time)
        if not self.spot_price_930 and current_time >= dt_time(9, 30):
            self._capture_930_price(df, current_idx)
        
        # Step 2: Pre-market checks (before entry)
        if not self.entry_triggered:
            # Check if we should trade SELLING today
            precheck = self._pre_entry_checks(df, current_idx)
            if not precheck['passed']:
                return self._no_signal(precheck['reason'])
            
            # Step 3: Select strikes (one-time)
            if self.selected_strikes is None:
                self._select_strikes(df, current_idx)
            
            # Step 4: Monitor for VWAP crossover
            crossover = self._check_vwap_crossover(df, current_idx)
            if crossover['crossed']:
                # Generate SELL signal!
                return self._generate_sell_signal(df, current_idx, crossover)
        
        # If already entered, check for exit
        if self.entry_triggered:
            exit_check = self._check_exit_conditions(df, current_idx)
            if exit_check['should_exit']:
                return self._generate_exit_signal(df, current_idx, exit_check)
        
        # No signal
        return self._no_signal("Waiting for VWAP crossover")
    
    def _capture_930_price(self, df: pd.DataFrame, current_idx: int):
        """Capture spot price at 9:30 AM"""
        symbol = self._get_todays_index()
        self.spot_price_930 = self.strike_selector.capture_930_spot_price(symbol)
        
        if self.spot_price_930:
            logger.info(f"✅ 9:30 AM spot price captured: {self.spot_price_930}")
        else:
            logger.error("❌ Failed to capture 9:30 AM price")
    
    def _pre_entry_checks(self, df: pd.DataFrame, current_idx: int) -> Dict:
        """
        Pre-entry validation checks.
        Implements Professional Enhancements #5 and #6.
        """
        # Check 1: Time validation (must be after 9:30 AM)
        current_time = df.index[current_idx].time()
        if current_time < dt_time(9, 30):
            return {'passed': False, 'reason': 'Before 9:30 AM - waiting'}
        
        # Check 2: Market condition filter (Professional Enhancement #6)
        symbol = self._get_todays_index()
        market_class = self.market_classifier.classify_market(symbol)
        
        if market_class['recommended_strategy'] != 'SELLING':
            return {
                'passed': False,
                'reason': f"Market conditions favor {market_class['recommended_strategy']}: {market_class['reason']}"
            }
        
        # Check 3: India VIX filter (Professional Enhancement #5)
        vix_check = self.vix_fetcher.check_vix_condition(
            strategy_type='SELLING',
            min_vix=self.config.get('min_india_vix')
        )
        
        if not vix_check['condition_met']:
            return {
                'passed': False,
                'reason': vix_check['reason']
            }
        
        # Check 4: Spot price available
        if self.spot_price_930 is None:
            return {'passed': False, 'reason': '9:30 AM price not captured'}
        
        # All checks passed
        return {
            'passed': True,
            'reason': 'All pre-entry checks passed',
            'vix': vix_check['current_vix'],
            'market_confidence': market_class['confidence']
        }
    
    def _select_strikes(self, df: pd.DataFrame, current_idx: int):
        """Select option strikes for 4-leg strategy"""
        symbol = self._get_todays_index()
        expiry = self._get_todays_expiry()
        
        self.selected_strikes = self.strike_selector.select_strikes_for_selling(
            symbol=symbol,
            expiry_date=expiry,
            spot_price=self.spot_price_930
        )
        
        if self.selected_strikes:
            logger.info(f"✅ Strikes selected:")
            logger.info(f"   Sell CE: {self.selected_strikes['sell_ce_strike']}")
            logger.info(f"   Sell PE: {self.selected_strikes['sell_pe_strike']}")
            logger.info(f"   Buy CE: {self.selected_strikes['buy_ce_strike']}")
            logger.info(f"   Buy PE: {self.selected_strikes['buy_pe_strike']}")
            logger.info(f"   Estimated premium: {self.selected_strikes['combined_premium_estimate']}")
            
            # Initialize VWAP chart monitoring
            self._initialize_vwap_chart()
        else:
            logger.error("❌ Failed to select strikes")
    
    def _initialize_vwap_chart(self):
        """Initialize VWAP chart for monitoring"""
        if not self.selected_strikes:
            return
        
        # Create chart for sold options (CE + PE)
        self.vwap_chart = SensibullVWAPChart(
            ce_symbol=self.selected_strikes['sell_ce_symbol'],
            pe_symbol=self.selected_strikes['sell_pe_symbol']
        )
        
        # Start monitoring (non-blocking)
        self.vwap_chart.start()
        logger.info("📊 VWAP chart monitoring started")
    
    def _check_vwap_crossover(self, df: pd.DataFrame, current_idx: int) -> Dict:
        """
        Check if premium crossed below VWAP (entry signal).
        From Final Notes: "Wait for premium to go ABOVE VWAP, then cross back BELOW"
        """
        if not self.vwap_chart or not self.vwap_chart.is_running:
            return {'crossed': False, 'reason': 'Chart not initialized'}
        
        # Get current state
        state = self.vwap_chart.get_current_state()
        
        if state['combined_premium'] is None or state['vwap'] is None:
            return {'crossed': False, 'reason': 'Waiting for data'}
        
        # Check crossover using calculator
        crossover = self.vwap_calculator.check_crossover(
            current_premium=state['combined_premium'],
            direction='below'  # For SELLING
        )
        
        return crossover
    
    def _generate_sell_signal(self, df: pd.DataFrame, current_idx: int, crossover: Dict) -> Dict:
        """
        Generate SELL signal when VWAP crossover detected.
        Format matches BaseStrategy signal structure.
        """
        self.entry_triggered = True
        self.entry_premium = crossover['premium']
        
        # Calculate stop-loss (Professional Enhancement #2)
        # 30% above entry premium (not 70%)
        sl_premium = self.entry_premium * (1 + self.risk_config['initial_sl_pct'])
        
        # Calculate target (Professional Enhancement #4)
        # 45% decay of premium
        target_premium = self.entry_premium * (1 - self.config['profit_target_percent'])
        
        # Calculate confidence
        confidence = 70  # Base confidence
        if crossover.get('was_above') and crossover.get('is_below'):
            confidence = 80  # Clean crossover
        
        logger.info(f"🚀 SELL SIGNAL GENERATED!")
        logger.info(f"   Entry Premium: {self.entry_premium}")
        logger.info(f"   Stop-Loss: {sl_premium} (+{self.risk_config['initial_sl_pct']*100}%)")
        logger.info(f"   Target: {target_premium} (-{self.config['profit_target_percent']*100}%)")
        
        # Return signal in BaseStrategy format
        return {
            'signal_type': 'SELL',
            'confidence': confidence,
            'setup_detected': True,
            'entry_price': self.entry_premium,
            'stop_loss': sl_premium,
            'target': target_premium,
            'strategy_name': self.name,
            'strikes': self.selected_strikes,
            'entry_reason': f"Premium crossed below VWAP at {crossover['premium']}",
            'vwap': crossover['vwap'],
            'timestamp': df.index[current_idx],
            
            # Multi-leg order details
            'order_type': 'BASKET',
            'legs': [
                {'action': 'SELL', 'symbol': self.selected_strikes['sell_ce_symbol'], 'quantity': 1},
                {'action': 'SELL', 'symbol': self.selected_strikes['sell_pe_symbol'], 'quantity': 1},
                {'action': 'BUY', 'symbol': self.selected_strikes['buy_ce_symbol'], 'quantity': 1},
                {'action': 'BUY', 'symbol': self.selected_strikes['buy_pe_symbol'], 'quantity': 1}
            ]
        }
    
    def _check_exit_conditions(self, df: pd.DataFrame, current_idx: int) -> Dict:
        """
        UPDATED: Analyst's conditional exit logic instead of blind 1:30 PM exit
        """
        current_time = df.index[current_idx].time()
        state = self.vwap_chart.get_current_state()
        current_premium = state.get('combined_premium')
        
        if current_premium is None:
            return {'should_exit': False, 'reason': 'No current price'}
        
        entry_premium = self.entry_premium
        sl_premium = entry_premium * (1 + self.risk_config['initial_sl_pct'])
        target_premium = entry_premium * (1 - self.config['profit_target_percent'])
        
        # Exit Condition 1: Stop-loss hit (IMMEDIATE)
        if current_premium >= sl_premium:
            return {
                'should_exit': True,
                'reason': f'Stop-loss hit: {current_premium} >= {sl_premium}',
                'exit_type': 'STOP_LOSS',
                'exit_premium': current_premium
            }
        
        # Exit Condition 2: Target hit (IMMEDIATE)
        if current_premium <= target_premium:
            return {
                'should_exit': True,
                'reason': f'Target hit: {current_premium} <= {target_premium}',
                'exit_type': 'TARGET',
                'exit_premium': current_premium
            }
        
        # Exit Condition 3: Analyst's Conditional 1:30 PM Logic
        if current_time >= dt_time(13, 30):
            # Sub-condition A: If profit target hit by 12:00 PM → Already exited above
            
            # Sub-condition B: If still in trade at 1:30 PM AND premium > entry → HOLD
            if current_premium > entry_premium:
                # Theta not working - exit now
                return {
                    'should_exit': True,
                    'reason': 'Analyst Rule: 1:30 PM reached but premium > entry (theta failed)',
                    'exit_type': 'TIME_THETA_FAILURE',
                    'exit_premium': current_premium
                }
            else:
                # Premium below entry (profitable) - hold till 3:00 PM with trailing SL
                logger.info("⏰ 1:30 PM reached - Holding with trailing SL till 3:00 PM")
                return {'should_exit': False, 'reason': 'Holding with trailing SL'}
        
        # Exit Condition 4: 3:00 PM Final Exit
        if current_time >= dt_time(15, 0):
            return {
                'should_exit': True,
                'reason': 'Final time-based exit (3:00 PM)',
                'exit_type': 'TIME_FINAL',
                'exit_premium': current_premium
            }
        
        # Exit Condition 5: Trailing Stop-Loss (Analyst's 20% decay rule)
        if current_premium <= entry_premium * 0.8:  # 20% decay
            # Move SL to breakeven
            if current_premium >= entry_premium * 0.95:  # Within 5% of breakeven
                return {
                    'should_exit': True,
                    'reason': 'Trailing SL hit at breakeven after 20% decay',
                    'exit_type': 'TRAILING_SL',
                    'exit_premium': current_premium
                }
        
        return {'should_exit': False, 'reason': 'All exit conditions not met'}
    
    def _generate_exit_signal(self, df: pd.DataFrame, current_idx: int, exit_check: Dict) -> Dict:
        """Generate exit signal"""
        logger.info(f"🛑 EXIT SIGNAL: {exit_check['reason']}")
        
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
        """Return no-signal response"""
        return {
            'signal_type': 'NO_TRADE',
            'confidence': 0,
            'setup_detected': False,
            'reason': reason
        }
    
    def _get_todays_index(self) -> str:
        """Get today's index (NIFTY or SENSEX)"""
        from config_vwap_strangle import get_todays_index
        return get_todays_index()
    
    def _get_todays_expiry(self) -> str:
        """Get today's expiry date"""
        symbol, expiry = self.strike_selector.get_todays_index_and_expiry()
        return expiry
    
    def cleanup(self):
        """Cleanup when strategy is done"""
        if self.vwap_chart:
            self.vwap_chart.stop()
        logger.info("Strategy cleanup completed")
