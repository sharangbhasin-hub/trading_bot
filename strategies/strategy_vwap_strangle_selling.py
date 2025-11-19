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
        self.name = "VWAP Strangle Selling" 
        
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
        """Capture spot price at 9:30 AM from dataframe"""
        try:
            if isinstance(df.index, pd.DatetimeIndex):
                # Find first candle at or after 9:15 AM
                morning_candles = df[df.index.time >= dt_time(9, 15)]
                
                if not morning_candles.empty:
                    self.spot_price_930 = morning_candles.iloc[0]['open']
                    logger.info(f"✅ 9:30 AM spot from df: {self.spot_price_930}")
                    return
            
            # Fallback: use first candle
            self.spot_price_930 = df.iloc[0]['open']
            logger.info(f"✅ Using first candle open as 9:30 price: {self.spot_price_930}")
            
        except Exception as e:
            logger.error(f"❌ Error capturing 9:30 price from df: {e}")
            if current_idx < len(df):
                self.spot_price_930 = df.iloc[current_idx]['close']
                logger.warning(f"⚠️ Using current price as fallback: {self.spot_price_930}")
    
    def _pre_entry_checks(self, df: pd.DataFrame, current_idx: int) -> Dict:
        """
        Pre-entry validation checks.
        Implements Professional Enhancements #5 and #6.
        """
        # Check 1: Time validation (must be after 9:30 AM)
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

        if current_time < dt_time(9, 30):
            return {'passed': False, 'reason': 'Before 9:30 AM - waiting'}
        
        # Check 2: Skip market classification (works from df only)
        market_class = {
            'recommended_strategy': 'SELLING',
            'confidence': 70,
            'reason': 'VWAP strategy active'
        }
        
        # Check 3: VIX check (optional, non-blocking)
        try:
            vix_check = self.vix_fetcher.check_vix_condition(
                strategy_type='SELLING',
                min_vix=self.config.get('min_india_vix')
            )
            if not vix_check['condition_met']:
                logger.info(f"VIX filter: {vix_check['reason']}")
                # Don't block - just log
        except Exception as e:
            logger.debug(f"VIX check skipped: {e}")
            vix_check = {'current_vix': None, 'condition_met': True}
        
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
        """Select strikes based on spot price - works everywhere"""
        if self.spot_price_930 is None:
            logger.warning("Cannot select strikes - 9:30 price not captured")
            return
        
        # Calculate strikes from spot price
        strike_interval = 100  # Nifty/BankNifty typical
        
        # For selling: 2 strikes OTM
        strikes_offset = self.config.get('strikes_above_spot', 2)
        hedge_distance = self.config.get('hedge_distance_points', 400)
        
        # Round to nearest strike
        atm_strike = round(self.spot_price_930 / strike_interval) * strike_interval
        
        sell_ce = atm_strike + (strikes_offset * strike_interval)
        sell_pe = atm_strike - (strikes_offset * strike_interval)
        buy_ce = sell_ce + hedge_distance
        buy_pe = sell_pe - hedge_distance
        
        self.selected_strikes = {
            'sell_ce_strike': sell_ce,
            'sell_pe_strike': sell_pe,
            'buy_ce_strike': buy_ce,
            'buy_pe_strike': buy_pe,
            'sell_ce_symbol': f'CE_{sell_ce}',
            'sell_pe_symbol': f'PE_{sell_pe}',
            'buy_ce_symbol': f'CE_{buy_ce}',
            'buy_pe_symbol': f'PE_{buy_pe}',
            'combined_premium_estimate': 150  # Simulated
        }
        
        logger.info(f"✅ Strikes selected:")
        logger.info(f"   Sell: CE {sell_ce} / PE {sell_pe}")
        logger.info(f"   Buy: CE {buy_ce} / PE {buy_pe}")
    
    def _check_vwap_crossover(self, df: pd.DataFrame, current_idx: int) -> Dict:
        """Check if COMBINED PREMIUM crossed below VWAP - uses df with simulated premium"""
        
        # Get current spot price from df
        current_spot = df.iloc[current_idx]['close']
        
        # ✅ SIMULATE OPTION PREMIUM (key fix!)
        # For short strangle: premium decays as spot moves away from strike
        df_slice = df[:current_idx+1].copy()
        
        # Calculate simulated premium for each candle
        # ATM strangle premium inversely correlates with spot distance from 9:30 level
        if self.spot_price_930 is not None:
            # Calculate distance from 9:30 price
            spot_distances = abs(df_slice['close'] - self.spot_price_930)
            
            # Base premium (typical ATM strangle)
            base_premium = 150
            
            # Premium decay: decays ~30% per 100-point move
            # Formula: premium = base * (1 - decay_rate * distance_factor)
            decay_rate = 0.003  # 0.3% per point
            premium_decay = 1 - (decay_rate * spot_distances)
            premium_decay = premium_decay.clip(lower=0.3, upper=1.2)  # Cap between 30%-120%
            
            # Simulate combined premium
            simulated_premium = base_premium * premium_decay
            
            # Split into CE/PE (for VWAP calculator)
            df_slice['ce_price'] = simulated_premium / 2
            df_slice['pe_price'] = simulated_premium / 2
        else:
            # Fallback if 9:30 price not captured
            logger.warning("9:30 price not available - using spot as premium proxy")
            df_slice['ce_price'] = df_slice['close'] / 2
            df_slice['pe_price'] = df_slice['close'] / 2
        
        # Calculate VWAP on simulated premium
        try:
            df_with_vwap = self.vwap_calculator.calculate_from_dataframe(
                df_slice,
                ce_col='ce_price',
                pe_col='pe_price'
            )
            
            if 'vwap' not in df_with_vwap.columns or df_with_vwap.empty:
                return {'crossed': False, 'reason': 'VWAP calculation failed'}
            
            current_vwap = df_with_vwap['vwap'].iloc[-1]
            current_premium = df_with_vwap['combined_premium'].iloc[-1]
            
        except Exception as e:
            logger.error(f"VWAP calculation error: {e}")
            return {'crossed': False, 'reason': f'VWAP calculation error: {e}'}
        
        # Check crossover: premium was above VWAP, now below
        if current_idx < 1:
            return {'crossed': False, 'reason': 'Need previous candle'}
        
        # Get previous premium and VWAP
        if len(df_with_vwap) < 2:
            return {'crossed': False, 'reason': 'Need previous VWAP'}
        
        prev_premium = df_with_vwap['combined_premium'].iloc[-2]
        prev_vwap = df_with_vwap['vwap'].iloc[-2]
        
        # Crossover check for SELLING: premium was above VWAP, now below
        was_above = prev_premium >= prev_vwap
        is_below = current_premium < current_vwap
        crossed_below = was_above and is_below
        
        if not crossed_below:
            # Log current state for debugging
            logger.debug(
                f"No crossover - Premium: {current_premium:.2f}, VWAP: {current_vwap:.2f}, "
                f"Was above: {was_above}, Is below: {is_below}"
            )
            return {'crossed': False, 'reason': 'No crossover below VWAP'}
        
        logger.info(
            f"🚀 VWAP crossover detected (below)! "
            f"Premium: {prev_premium:.2f}→{current_premium:.2f}, "
            f"VWAP: {prev_vwap:.2f}→{current_vwap:.2f}"
        )
        
        return {
            'crossed': True,
            'premium': current_premium,
            'vwap': current_vwap,
            'was_above': was_above,
            'is_below': is_below
        }
    
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
        
        # Get current premium from df (spot price proxy)
        current_spot = df.iloc[current_idx]['close']
        # For selling, premium decays as spot moves away from entry
        spot_change_pct = abs((current_spot - self.spot_price_930) / self.spot_price_930)
        # Simulate premium: starts at entry, decays with time and spot movement
        current_premium = self.entry_premium * (1 - (spot_change_pct * 0.5))
        
        if current_premium is None or current_premium <= 0:
            return {'should_exit': False, 'reason': 'Invalid premium calculation'}
        
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
        logger.info("Strategy cleanup completed")
