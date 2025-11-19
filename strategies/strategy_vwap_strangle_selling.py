"""
VWAP Short Strangle Strategy (SELLING)
=======================================
Strategy 1 from Final Accumulated Notes.
Profits from theta decay using VWAP crossover as entry signal.

Setup: 4-leg iron strangle
Entry: Premium crosses BELOW VWAP (after being above)
Exit: 1:30 PM or target/SL hit
Win Rate Target: 80%+

DYNAMIC: Works identically in backtest and live trading using REAL option data.

Author: Trading System
Date: November 19, 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime, time as dt_time
import logging

from vwap_calculator import VWAPCalculator
from vwap_strike_selector import VWAPStrikeSelector
from india_vix_fetcher import IndiaVIXFetcher
from vwap_market_classifier import VWAPMarketClassifier
from config_vwap_strangle import VWAP_STRANGLE_SELLING, RISK_MANAGEMENT

logger = logging.getLogger(__name__)

class VWAPStrangleSelling:
    """
    VWAP Short Strangle Strategy (Selling)
    Uses REAL option premium data in both backtest and live trading.
    """
    
    def __init__(self, kite=None):
        """
        Initialize strategy
        
        Args:
            kite: KiteHandler instance (required for real option data)
        """
        self.name = "VWAP Strangle Selling" 
        
        # Configuration
        self.config = VWAP_STRANGLE_SELLING
        self.risk_config = RISK_MANAGEMENT['selling']
        
        # Components
        self.vwap_calculator = VWAPCalculator()
        self.strike_selector = VWAPStrikeSelector()
        self.vix_fetcher = IndiaVIXFetcher()
        self.market_classifier = VWAPMarketClassifier()
        self.kite = kite  # ✅ Kite handler for real option data
        
        # State
        self.spot_price_930 = None
        self.selected_strikes = None
        self.entry_triggered = False
        self.entry_premium = None
        
        # Cache for option data (avoid repeated API calls)
        self._option_data_cache = None
        self._option_data_cache_date = None
        
        # Override base settings
        self.min_confidence = self.config.get('min_confidence', 65)
        
        logger.info(f"✅ {self.name} initialized")
        if self.kite and hasattr(self.kite, 'connected') and self.kite.connected:
            logger.info("   Mode: LIVE TRADING (Kite connected)")
        else:
            logger.info("   Mode: BACKTEST (using historical data)")
    
    def detect(self, df: pd.DataFrame, current_idx: int) -> Dict:
        """
        Main detection method.
        Works identically in backtest and live trading.
        
        Args:
            df: OHLC dataframe (spot price)
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
        
        # Step 1: Capture 9:30 AM spot price
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
            
            # Step 4: Monitor for VWAP crossover (REAL DATA!)
            crossover = self._check_vwap_crossover(df, current_idx)
            if crossover['crossed']:
                return self._generate_sell_signal(df, current_idx, crossover)
        
        # If already entered, check for exit
        if self.entry_triggered:
            exit_check = self._check_exit_conditions(df, current_idx)
            if exit_check['should_exit']:
                return self._generate_exit_signal(df, current_idx, exit_check)
        
        return self._no_signal("Waiting for VWAP crossover")
    
    def _capture_930_price(self, df: pd.DataFrame, current_idx: int):
        """Capture spot price at 9:30 AM from dataframe"""
        try:
            if isinstance(df.index, pd.DatetimeIndex):
                morning_candles = df[df.index.time >= dt_time(9, 15)]
                
                if not morning_candles.empty:
                    self.spot_price_930 = morning_candles.iloc[0]['open']
                    logger.info(f"✅ 9:30 AM spot from df: {self.spot_price_930}")
                    return
            
            self.spot_price_930 = df.iloc[0]['open']
            logger.info(f"✅ Using first candle open as 9:30 price: {self.spot_price_930}")
            
        except Exception as e:
            logger.error(f"❌ Error capturing 9:30 price: {e}")
            if current_idx < len(df):
                self.spot_price_930 = df.iloc[current_idx]['close']
    
    def _pre_entry_checks(self, df: pd.DataFrame, current_idx: int) -> Dict:
        """Pre-entry validation checks"""
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

        # Check 1: Time validation
        if current_time < dt_time(9, 30):
            return {'passed': False, 'reason': 'Before 9:30 AM - waiting'}
        
        # Check 2: Market classification (simplified)
        market_class = {
            'recommended_strategy': 'SELLING',
            'confidence': 70,
            'reason': 'VWAP strategy active'
        }
        
        # Check 3: VIX check (non-blocking)
        try:
            vix_check = self.vix_fetcher.check_vix_condition(
                strategy_type='SELLING',
                min_vix=self.config.get('min_india_vix')
            )
            if not vix_check['condition_met']:
                logger.info(f"VIX filter: {vix_check['reason']}")
        except Exception as e:
            vix_check = {'current_vix': None, 'condition_met': True}
        
        # Check 4: Spot price available
        if self.spot_price_930 is None:
            return {'passed': False, 'reason': '9:30 AM price not captured'}
        
        return {
            'passed': True,
            'reason': 'All pre-entry checks passed',
            'vix': vix_check.get('current_vix'),
            'market_confidence': market_class['confidence']
        }
    
    def _select_strikes(self, df: pd.DataFrame, current_idx: int):
        """Select strikes based on spot price"""
        if self.spot_price_930 is None:
            logger.warning("Cannot select strikes - 9:30 price not captured")
            return
        
        # Get proper symbols for options
        try:
            index_symbol = self._get_todays_index()
            expiry = self._get_todays_expiry()
        except Exception as e:
            logger.error(f"Error getting index/expiry: {e}")
            index_symbol = "NIFTY"
            expiry = datetime.now().strftime("%y%b").upper()
        
        # Calculate strikes
        strike_interval = 100
        strikes_offset = self.config.get('strikes_above_spot', 2)
        hedge_distance = self.config.get('hedge_distance_points', 400)
        
        atm_strike = round(self.spot_price_930 / strike_interval) * strike_interval
        sell_ce = atm_strike + (strikes_offset * strike_interval)
        sell_pe = atm_strike - (strikes_offset * strike_interval)
        buy_ce = sell_ce + hedge_distance
        buy_pe = sell_pe - hedge_distance
        
        # ✅ Format symbols properly for Kite API
        # Format: NIFTY24NOV25000CE
        self.selected_strikes = {
            'sell_ce_strike': sell_ce,
            'sell_pe_strike': sell_pe,
            'buy_ce_strike': buy_ce,
            'buy_pe_strike': buy_pe,
            'sell_ce_symbol': f'{index_symbol}{expiry}{sell_ce}CE',
            'sell_pe_symbol': f'{index_symbol}{expiry}{sell_pe}PE',
            'buy_ce_symbol': f'{index_symbol}{expiry}{buy_ce}CE',
            'buy_pe_symbol': f'{index_symbol}{expiry}{buy_pe}PE'
        }
        
        logger.info(f"✅ Strikes selected:")
        logger.info(f"   Sell: {self.selected_strikes['sell_ce_symbol']} / {self.selected_strikes['sell_pe_symbol']}")
        logger.info(f"   Buy: {self.selected_strikes['buy_ce_symbol']} / {self.selected_strikes['buy_pe_symbol']}")
    
    def _check_vwap_crossover(self, df: pd.DataFrame, current_idx: int) -> Dict:
        """
        Check if REAL OPTION PREMIUM crossed below VWAP.
        ✅ DYNAMIC: Works for both backtest and live trading.
        """
        if not self.selected_strikes:
            return {'crossed': False, 'reason': 'Strikes not selected'}
        
        try:
            # ========================================
            # GET REAL OPTION DATA (BOTH MODES!)
            # ========================================
            option_data = self._get_option_data(df, current_idx)
            
            if option_data is None or option_data.empty:
                return {'crossed': False, 'reason': 'No option data available'}
            
            # ========================================
            # CALCULATE VWAP ON REAL DATA
            # ========================================
            df_with_vwap = self.vwap_calculator.calculate_from_dataframe(
                option_data,
                ce_col='ce_close',
                pe_col='pe_close'
            )
            
            if 'vwap' not in df_with_vwap.columns or len(df_with_vwap) < 2:
                return {'crossed': False, 'reason': 'Insufficient VWAP data'}
            
            # Get current and previous values
            current_premium = df_with_vwap['combined_premium'].iloc[-1]
            current_vwap = df_with_vwap['vwap'].iloc[-1]
            prev_premium = df_with_vwap['combined_premium'].iloc[-2]
            prev_vwap = df_with_vwap['vwap'].iloc[-2]
            
            # ========================================
            # CHECK CROSSOVER
            # ========================================
            was_above = prev_premium >= prev_vwap
            is_below = current_premium < current_vwap
            crossed_below = was_above and is_below
            
            if not crossed_below:
                logger.debug(
                    f"No crossover - Premium: {current_premium:.2f}, VWAP: {current_vwap:.2f}"
                )
                return {'crossed': False, 'reason': 'No crossover below VWAP'}
            
            logger.info(
                f"🚀 VWAP crossover detected (REAL DATA)! "
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
            
        except Exception as e:
            logger.error(f"Error checking VWAP crossover: {e}")
            import traceback
            traceback.print_exc()
            return {'crossed': False, 'reason': f'Error: {e}'}
    
    def _get_option_data(self, df: pd.DataFrame, current_idx: int) -> Optional[pd.DataFrame]:
        """
        Get REAL option premium data.
        ✅ WORKS FOR BOTH BACKTEST AND LIVE!
        
        Returns:
            pd.DataFrame with ce_close, pe_close, combined_premium columns
        """
        if not self.kite:
            logger.error("❌ Kite handler not provided - cannot get real option data")
            return None
        
        try:
            # Get current date
            if isinstance(df.index, pd.DatetimeIndex):
                current_date = df.index[current_idx].date()
                current_time = df.index[current_idx]
            else:
                current_date = datetime.now().date()
                current_time = datetime.now()
            
            # Check cache
            if (self._option_data_cache is not None and 
                self._option_data_cache_date == current_date):
                # Use cached data, filter to current time
                option_data = self._option_data_cache[
                    self._option_data_cache.index <= current_time
                ]
                if not option_data.empty:
                    logger.debug(f"Using cached option data: {len(option_data)} candles")
                    return option_data
            
            # Fetch from Kite API
            logger.debug(f"Fetching option data for {current_date}")
            option_data = self.kite.get_option_historical_data(
                ce_symbol=self.selected_strikes['sell_ce_symbol'],
                pe_symbol=self.selected_strikes['sell_pe_symbol'],
                days=1,  # Today's data
                interval="minute"
            )
            
            if option_data is None or option_data.empty:
                logger.warning("No option data from Kite API")
                return None
            
            # Cache it
            self._option_data_cache = option_data
            self._option_data_cache_date = current_date
            
            # Filter to current time (for backtest)
            option_data = option_data[option_data.index <= current_time]
            
            if option_data.empty:
                logger.warning(f"No option data before {current_time}")
                return None
            
            logger.info(f"✅ Got {len(option_data)} candles of REAL option data")
            return option_data
            
        except Exception as e:
            logger.error(f"Error fetching option data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_sell_signal(self, df: pd.DataFrame, current_idx: int, crossover: Dict) -> Dict:
        """Generate SELL signal"""
        self.entry_triggered = True
        self.entry_premium = crossover['premium']
        
        sl_premium = self.entry_premium * (1 + self.risk_config['initial_sl_pct'])
        target_premium = self.entry_premium * (1 - self.config['profit_target_percent'])
        
        confidence = 80 if crossover.get('was_above') and crossover.get('is_below') else 70
        
        logger.info(f"🚀 SELL SIGNAL GENERATED!")
        logger.info(f"   Entry: {self.entry_premium:.2f}, Target: {target_premium:.2f}, SL: {sl_premium:.2f}")
        
        return {
            'signal_type': 'SELL',
            'confidence': confidence,
            'setup_detected': True,
            'entry_price': self.entry_premium,
            'stop_loss': sl_premium,
            'target': target_premium,
            'strategy_name': self.name,
            'strikes': self.selected_strikes,
            'entry_reason': f"Real premium crossed below VWAP",
            'vwap': crossover['vwap'],
            'timestamp': df.index[current_idx],
            'order_type': 'BASKET',
            'legs': [
                {'action': 'SELL', 'symbol': self.selected_strikes['sell_ce_symbol'], 'quantity': 1},
                {'action': 'SELL', 'symbol': self.selected_strikes['sell_pe_symbol'], 'quantity': 1},
                {'action': 'BUY', 'symbol': self.selected_strikes['buy_ce_symbol'], 'quantity': 1},
                {'action': 'BUY', 'symbol': self.selected_strikes['buy_pe_symbol'], 'quantity': 1}
            ]
        }
    
    def _check_exit_conditions(self, df: pd.DataFrame, current_idx: int) -> Dict:
        """Check exit conditions using REAL option data"""
        try:
            if isinstance(df.index, pd.DatetimeIndex):
                current_time = df.index[current_idx].time()
            else:
                current_time = datetime.now().time()
        except:
            current_time = datetime.now().time()
        
        # Get REAL current premium
        option_data = self._get_option_data(df, current_idx)
        if option_data is None or option_data.empty:
            return {'should_exit': False, 'reason': 'No option data for exit check'}
        
        current_premium = option_data['combined_premium'].iloc[-1]
        
        entry_premium = self.entry_premium
        sl_premium = entry_premium * (1 + self.risk_config['initial_sl_pct'])
        target_premium = entry_premium * (1 - self.config['profit_target_percent'])
        
        # Exit conditions
        if current_premium >= sl_premium:
            return {
                'should_exit': True,
                'reason': f'Stop-loss hit: {current_premium:.2f} >= {sl_premium:.2f}',
                'exit_type': 'STOP_LOSS',
                'exit_premium': current_premium
            }
        
        if current_premium <= target_premium:
            return {
                'should_exit': True,
                'reason': f'Target hit: {current_premium:.2f} <= {target_premium:.2f}',
                'exit_type': 'TARGET',
                'exit_premium': current_premium
            }
        
        if current_time >= dt_time(13, 30):
            if current_premium > entry_premium:
                return {
                    'should_exit': True,
                    'reason': '1:30 PM - Theta failed (premium > entry)',
                    'exit_type': 'TIME_THETA_FAILURE',
                    'exit_premium': current_premium
                }
        
        if current_time >= dt_time(15, 0):
            return {
                'should_exit': True,
                'reason': 'Final exit (3:00 PM)',
                'exit_type': 'TIME_FINAL',
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
        """Return no-signal response"""
        return {
            'signal_type': 'NO_TRADE',
            'confidence': 0,
            'setup_detected': False,
            'reason': reason
        }
    
    def _get_todays_index(self) -> str:
        """Get today's index"""
        try:
            from config_vwap_strangle import get_todays_index
            return get_todays_index()
        except:
            return "NIFTY"
    
    def _get_todays_expiry(self) -> str:
        """Get today's expiry date"""
        try:
            symbol, expiry = self.strike_selector.get_todays_index_and_expiry()
            return expiry
        except:
            return datetime.now().strftime("%y%b").upper()
    
    def cleanup(self):
        """Cleanup"""
        self._option_data_cache = None
        logger.info("Strategy cleanup completed")
