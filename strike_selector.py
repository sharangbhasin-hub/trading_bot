"""
Strike Selection Logic for Intraday Options
Implements Slightly ITM strategy with OTM/ATM display
"""

import pandas as pd
from typing import Dict, Tuple, Optional

class StrikeSelector:
    """
    Selects optimal option contract based on trend analysis
    Shows OTM, ATM, ITM but recommends ITM for final trade
    """
    
    def select_contract(self, 
                       trend_analysis: Dict,
                       calls_df: pd.DataFrame,
                       puts_df: pd.DataFrame,
                       spot_price: float) -> Dict:
        """
        Main function: Selects option contract based on trend
        Returns all three (OTM, ATM, ITM) with ITM as final recommendation
        """
        
        direction = trend_analysis['direction']
        action = trend_analysis['action']
        
        if action == "CALL":
            return self._select_call_options(calls_df, spot_price, trend_analysis)
        elif action == "PUT":
            return self._select_put_options(puts_df, spot_price, trend_analysis)
        else:
            return {'error': 'No clear trend - WAIT for better setup'}
    
    def _select_call_options(self, calls_df: pd.DataFrame, 
                            spot_price: float, 
                            trend_analysis: Dict) -> Dict:
        """Select Call options: OTM, ATM, ITM"""
        
        # Get available strikes
        strikes = sorted(calls_df['strike'].unique())
        
        # Find nearest expiry (for intraday - today or tomorrow)
        nearest_expiry = calls_df['expiry'].min()
        expiry_calls = calls_df[calls_df['expiry'] == nearest_expiry]
        
        # Select strikes
        atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
        
        # OTM: 1-2 strikes above spot
        otm_strikes = [s for s in strikes if s > spot_price]
        otm_strike = otm_strikes[0] if otm_strikes else atm_strike
        
        # ITM: 1-2 strikes below spot (SLIGHTLY ITM - your requirement)
        itm_strikes = [s for s in strikes if s < spot_price]
        itm_strike = itm_strikes[-1] if itm_strikes else atm_strike  # Closest below
        
        # Get contract details
        otm_contract = expiry_calls[expiry_calls['strike'] == otm_strike].iloc[0]
        atm_contract = expiry_calls[expiry_calls['strike'] == atm_strike].iloc[0]
        itm_contract = expiry_calls[expiry_calls['strike'] == itm_strike].iloc[0]
        
        return {
            'type': 'CALL',
            'direction': 'BULLISH',
            'spot_price': spot_price,
            'expiry': nearest_expiry.strftime('%Y-%m-%d'),
            'options': {
                'OTM': self._format_contract(otm_contract, spot_price, 'OTM'),
                'ATM': self._format_contract(atm_contract, spot_price, 'ATM'),
                'ITM': self._format_contract(itm_contract, spot_price, 'ITM')
            },
            'recommended': self._format_contract(itm_contract, spot_price, 'ITM'),
            'recommendation_reason': 'Slightly ITM provides intrinsic value protection with good profit potential',
            'trend_confidence': trend_analysis['confidence']
        }
    
    def _select_put_options(self, puts_df: pd.DataFrame,
                           spot_price: float,
                           trend_analysis: Dict) -> Dict:
        """Select Put options: OTM, ATM, ITM"""
        
        strikes = sorted(puts_df['strike'].unique())
        nearest_expiry = puts_df['expiry'].min()
        expiry_puts = puts_df[puts_df['expiry'] == nearest_expiry]
        
        atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
        
        # OTM: 1-2 strikes below spot
        otm_strikes = [s for s in strikes if s < spot_price]
        otm_strike = otm_strikes[-1] if otm_strikes else atm_strike
        
        # ITM: 1-2 strikes above spot (SLIGHTLY ITM)
        itm_strikes = [s for s in strikes if s > spot_price]
        itm_strike = itm_strikes[0] if itm_strikes else atm_strike  # Closest above
        
        otm_contract = expiry_puts[expiry_puts['strike'] == otm_strike].iloc[0]
        atm_contract = expiry_puts[expiry_puts['strike'] == atm_strike].iloc[0]
        itm_contract = expiry_puts[expiry_puts['strike'] == itm_strike].iloc[0]
        
        return {
            'type': 'PUT',
            'direction': 'BEARISH',
            'spot_price': spot_price,
            'expiry': nearest_expiry.strftime('%Y-%m-%d'),
            'options': {
                'OTM': self._format_contract(otm_contract, spot_price, 'OTM'),
                'ATM': self._format_contract(atm_contract, spot_price, 'ATM'),
                'ITM': self._format_contract(itm_contract, spot_price, 'ITM')
            },
            'recommended': self._format_contract(itm_contract, spot_price, 'ITM'),
            'recommendation_reason': 'Slightly ITM provides intrinsic value protection with good profit potential',
            'trend_confidence': trend_analysis['confidence']
        }
    
    def _format_contract(self, contract: pd.Series, spot_price: float, 
                        moneyness: str) -> Dict:
        """Format contract details for display"""
        return {
            'tradingsymbol': contract['tradingsymbol'],
            'strike': contract['strike'],
            'expiry': contract['expiry'].strftime('%Y-%m-%d'),
            'instrument_token': int(contract['instrument_token']),
            'lot_size': int(contract['lot_size']),
            'moneyness': moneyness,
            'distance_from_spot': abs(contract['strike'] - spot_price),
            'intrinsic_value': max(0, spot_price - contract['strike']) if moneyness != 'OTM' else 0
        }
