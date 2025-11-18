"""
Strike Selector for VWAP-Strangle Strategies
=============================================
Selects option strikes based on 9:30 AM spot price.
Follows exact rules from Final Accumulated Notes.

Rules:
- CE: 2 strikes ABOVE 9:30 AM spot price
- PE: 2 strikes BELOW 9:30 AM spot price
- Hedges: 400 points further OTM

Author: Trading System  
Date: October 25, 2025
"""

import pandas as pd
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import logging
from kite_handler import get_kite_handler
from config_vwap_strangle import VWAP_STRANGLE_SELLING, VWAP_STRANGLE_BUYING, WEEKLY_SCHEDULE

logger = logging.getLogger(__name__)

class VWAPStrikeSelector:
    """
    Selects option strikes for VWAP-Strangle strategies based on 9:30 AM spot price.
    """
    
    def __init__(self):
        """Initialize strike selector"""
        self.kite = None
        self.spot_price_930 = None  # Spot price at 9:30 AM
        self.selected_strikes = None
    
    def _initialize_kite(self) -> bool:
        """Initialize Kite connection"""
        if self.kite is None:
            self.kite = get_kite_handler()
            if self.kite is None or not self.kite.connected:
                logger.error("Kite handler not initialized")
                return False
        return True
    
    def capture_930_spot_price(self, symbol: str) -> Optional[float]:
        """
        Capture spot price at 9:30 AM (closing price of 9:15 AM candle).
        From Final Notes: "At 9:30 AM, note the closing price of the 9:15 AM candle"
        
        Args:
            symbol: Index symbol (e.g., 'NIFTY 50', 'SENSEX')
        
        Returns:
            float: Spot price at 9:30 AM
        """
        if not self._initialize_kite():
            return None
        
        try:
            # Get current quote
            quote = self.kite.kite.quote(f"NSE:{symbol}")
            
            if not quote or f"NSE:{symbol}" not in quote:
                logger.error(f"Quote not found for {symbol}")
                return None
            
            data = quote[f"NSE:{symbol}"]
            spot_price = data.get('last_price', None)
            
            if spot_price:
                self.spot_price_930 = spot_price
                logger.info(f"9:30 AM spot price captured: {symbol} = {spot_price}")
                return spot_price
            
            return None
            
        except Exception as e:
            logger.error(f"Error capturing 9:30 AM price: {e}")
            return None
    
    def select_strikes_for_selling(self, 
                                   symbol: str,
                                   expiry_date: str,
                                   spot_price: Optional[float] = None) -> Dict:
        """
        Select 4-leg strikes for SELLING strategy.
        
        From Final Notes Step 2:
        - Sell CE: 2 strikes above spot
        - Sell PE: 2 strikes below spot  
        - Buy CE hedge: 400 points above sold CE
        - Buy PE hedge: 400 points below sold PE
        
        Args:
            symbol: 'NIFTY' or 'SENSEX'
            expiry_date: Expiry date in 'YYYY-MM-DD' format
            spot_price: Override spot price (uses cached 9:30 price if None)
        
        Returns:
            dict: {
                'spot_price': float,
                'sell_ce_strike': int,
                'sell_pe_strike': int,
                'buy_ce_strike': int (hedge),
                'buy_pe_strike': int (hedge),
                'sell_ce_symbol': str,
                'sell_pe_symbol': str,
                'buy_ce_symbol': str,
                'buy_pe_symbol': str,
                'combined_premium_estimate': float
            }
        """
        if spot_price is None:
            spot_price = self.spot_price_930
        
        if spot_price is None:
            logger.error("Spot price not available. Call capture_930_spot_price() first")
            return {}
        
        if not self._initialize_kite():
            return {}
        
        try:
            # Get option chain
            option_chain = self.kite.get_option_chain(
                symbol=symbol,
                expiry=expiry_date
            )
            
            if option_chain.empty:
                logger.error("Option chain is empty")
                return {}
            
            # Find ATM strike
            atm_strike = self._find_atm_strike(spot_price, option_chain)
            strike_gap = self._get_strike_gap(symbol)
            
            # Calculate strikes (2 strikes = 2 * strike_gap)
            sell_ce_strike = atm_strike + (2 * strike_gap)
            sell_pe_strike = atm_strike - (2 * strike_gap)
            
            # Calculate hedge strikes (400 points further OTM)
            buy_ce_strike = sell_ce_strike + 400
            buy_pe_strike = sell_pe_strike - 400
            
            logger.info(f"Selected strikes - Sell CE: {sell_ce_strike}, Sell PE: {sell_pe_strike}")
            logger.info(f"Hedge strikes - Buy CE: {buy_ce_strike}, Buy PE: {buy_pe_strike}")
            
            # Get tradingsymbols
            sell_ce_symbol = self._get_option_symbol(option_chain, sell_ce_strike, 'CE')
            sell_pe_symbol = self._get_option_symbol(option_chain, sell_pe_strike, 'PE')
            buy_ce_symbol = self._get_option_symbol(option_chain, buy_ce_strike, 'CE')
            buy_pe_symbol = self._get_option_symbol(option_chain, buy_pe_strike, 'PE')
            
            # Estimate premiums
            sell_ce_premium = self._get_option_premium(option_chain, sell_ce_strike, 'CE')
            sell_pe_premium = self._get_option_premium(option_chain, sell_pe_strike, 'PE')
            buy_ce_premium = self._get_option_premium(option_chain, buy_ce_strike, 'CE')
            buy_pe_premium = self._get_option_premium(option_chain, buy_pe_strike, 'PE')
            
            # Combined premium = (Sell CE + Sell PE) - (Buy CE + Buy PE)
            combined_premium = (sell_ce_premium + sell_pe_premium) - (buy_ce_premium + buy_pe_premium)
            
            result = {
                'strategy_type': 'SELLING',
                'spot_price': spot_price,
                'atm_strike': atm_strike,
                'sell_ce_strike': sell_ce_strike,
                'sell_pe_strike': sell_pe_strike,
                'buy_ce_strike': buy_ce_strike,
                'buy_pe_strike': buy_pe_strike,
                'sell_ce_symbol': sell_ce_symbol,
                'sell_pe_symbol': sell_pe_symbol,
                'buy_ce_symbol': buy_ce_symbol,
                'buy_pe_symbol': buy_pe_symbol,
                'sell_ce_premium': sell_ce_premium,
                'sell_pe_premium': sell_pe_premium,
                'buy_ce_premium': buy_ce_premium,
                'buy_pe_premium': buy_pe_premium,
                'combined_premium_estimate': combined_premium,
                'expiry_date': expiry_date
            }
            
            self.selected_strikes = result
            return result
            
        except Exception as e:
            logger.error(f"Error selecting strikes: {e}")
            return {}
    
    def select_strikes_for_buying(self,
                                  symbol: str,
                                  expiry_date: str,
                                  spot_price: Optional[float] = None) -> Dict:
        """
        Select strikes for BUYING strategy.
        
        From Final Notes:
        - Add same strikes to watchlist (2 above for CE, 2 below for PE)
        - Don't execute yet - wait for VWAP signal to choose stronger leg
        
        Args:
            symbol: 'NIFTY' or 'SENSEX'
            expiry_date: Expiry date
            spot_price: Override spot price
        
        Returns:
            dict: {
                'spot_price': float,
                'ce_strike': int,
                'pe_strike': int,
                'ce_symbol': str,
                'pe_symbol': str,
                'ce_premium': float,
                'pe_premium': float
            }
        """
        if spot_price is None:
            spot_price = self.spot_price_930
        
        if spot_price is None:
            logger.error("Spot price not available")
            return {}
        
        if not self._initialize_kite():
            return {}
        
        try:
            option_chain = self.kite.get_option_chain(
                symbol=symbol,
                expiry=expiry_date
            )
            
            if option_chain.empty:
                return {}
            
            atm_strike = self._find_atm_strike(spot_price, option_chain)
            strike_gap = self._get_strike_gap(symbol)
            
            # Same strikes as selling (for chart consistency)
            ce_strike = atm_strike + (2 * strike_gap)
            pe_strike = atm_strike - (2 * strike_gap)
            
            ce_symbol = self._get_option_symbol(option_chain, ce_strike, 'CE')
            pe_symbol = self._get_option_symbol(option_chain, pe_strike, 'PE')
            
            ce_premium = self._get_option_premium(option_chain, ce_strike, 'CE')
            pe_premium = self._get_option_premium(option_chain, pe_strike, 'PE')
            
            result = {
                'strategy_type': 'BUYING',
                'spot_price': spot_price,
                'atm_strike': atm_strike,
                'ce_strike': ce_strike,
                'pe_strike': pe_strike,
                'ce_symbol': ce_symbol,
                'pe_symbol': pe_symbol,
                'ce_premium': ce_premium,
                'pe_premium': pe_premium,
                'combined_premium': ce_premium + pe_premium,
                'expiry_date': expiry_date
            }
            
            self.selected_strikes = result
            return result
            
        except Exception as e:
            logger.error(f"Error selecting buying strikes: {e}")
            return {}
    
    def _find_atm_strike(self, spot_price: float, option_chain: pd.DataFrame) -> int:
        """Find ATM (At-The-Money) strike"""
        strikes = option_chain['strike'].unique()
        strikes = sorted(strikes)
        
        # Find closest strike to spot price
        atm = min(strikes, key=lambda x: abs(x - spot_price))
        return int(atm)
    
    def _get_strike_gap(self, symbol: str) -> int:
        """Get strike gap for symbol"""
        # Nifty: 50 points gap
        # Bank Nifty: 100 points gap  
        # Sensex: 100 points gap
        
        if 'NIFTY' in symbol.upper() and 'BANK' not in symbol.upper():
            return 50
        elif 'BANK' in symbol.upper():
            return 100
        elif 'SENSEX' in symbol.upper():
            return 100
        else:
            return 50  # Default
    
    def _get_option_symbol(self, option_chain: pd.DataFrame, 
                          strike: int, option_type: str) -> Optional[str]:
        """Get tradingsymbol for given strike and type"""
        filtered = option_chain[
            (option_chain['strike'] == strike) &
            (option_chain['instrument_type'] == option_type)
        ]
        
        if not filtered.empty:
            return filtered.iloc[0]['tradingsymbol']
        return None
    
    def _get_option_premium(self, option_chain: pd.DataFrame,
                           strike: int, option_type: str) -> float:
        """Get last price (premium) for option"""
        filtered = option_chain[
            (option_chain['strike'] == strike) &
            (option_chain['instrument_type'] == option_type)
        ]
        
        if not filtered.empty:
            return filtered.iloc[0].get('last_price', 0.0)
        return 0.0
    
    def get_todays_index_and_expiry(self) -> Tuple[str, str]:
        """
        Get today's trading index and expiry based on weekly schedule.
        From Final Notes: Mon-Thu = Nifty, Fri = Sensex
        
        Returns:
            tuple: (index_symbol, expiry_date)
        """
        from datetime import datetime
        import pytz
        
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist)
        day_name = now.strftime('%A')
        
        schedule = WEEKLY_SCHEDULE['schedule']
        
        if day_name in schedule:
            index = schedule[day_name]['index']
            
            # Get nearest weekly expiry
            if not self._initialize_kite():
                return None, None
            
            expiries = self.kite.get_expiries(index)
            if expiries:
                nearest_expiry = expiries[0]  # Assuming sorted
                return index, nearest_expiry
        
        return 'NIFTY', None
