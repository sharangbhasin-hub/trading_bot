"""
Strike Selection Logic for Intraday Options
Implements Slightly ITM strategy with OTM/ATM display
"""

import pandas as pd
from typing import Dict, Optional
from datetime import datetime

class StrikeSelector:
    """
    Selects optimal option contract based on trend analysis
    Shows OTM, ATM, ITM but recommends Slightly ITM for final trade
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
        
        print(f"\n{'='*60}")
        print(f"STRIKE SELECTION")
        print(f"{'='*60}")
        print(f"Direction: {direction}")
        print(f"Action: {action}")
        print(f"Spot Price: ₹{spot_price:,.2f}")
        print(f"{'='*60}\n")
        
        if action == "CALL":
            return self._select_call_options(calls_df, spot_price, trend_analysis)
        elif action == "PUT":
            return self._select_put_options(puts_df, spot_price, trend_analysis)
        else:
            return {
                'error': 'No clear trend detected',
                'recommendation': 'WAIT for better market conditions',
                'reason': 'Trend analysis shows conflicting signals or low confidence'
            }
    
    def _select_call_options(self, calls_df: pd.DataFrame, 
                            spot_price: float, 
                            trend_analysis: Dict) -> Dict:
        """
        Select Call options: OTM, ATM, ITM
        For BULLISH trend - expecting price to rise
        """
        
        if calls_df.empty:
            return {'error': 'No call options available'}
        
        # Get available strikes sorted
        strikes = sorted(calls_df['strike'].unique())
        
        # Find nearest expiry (for intraday - current week)
        nearest_expiry = calls_df['expiry'].min()
        expiry_calls = calls_df[calls_df['expiry'] == nearest_expiry]
        
        # Calculate strike distances
        strike_distances = {s: abs(s - spot_price) for s in strikes}
        
        # ATM: Closest to spot price
        atm_strike = min(strike_distances, key=strike_distances.get)
        
        # OTM: 1-2 strikes ABOVE spot (cheaper, higher risk)
        otm_strikes = [s for s in strikes if s > spot_price]
        otm_strike = otm_strikes[0] if otm_strikes else atm_strike
        
        # ITM: 1 strike BELOW spot (SLIGHTLY ITM - your requirement)
        itm_strikes = [s for s in strikes if s < spot_price]
        itm_strike = itm_strikes[-1] if itm_strikes else atm_strike
        
        print(f"📊 CALL OPTION STRIKES SELECTED:")
        print(f"   OTM Strike: ₹{otm_strike:,.0f} (Above spot)")
        print(f"   ATM Strike: ₹{atm_strike:,.0f} (At spot)")
        print(f"   ITM Strike: ₹{itm_strike:,.0f} (Below spot - RECOMMENDED)")
        print()
        
        # Get contract details
        try:
            otm_contract = expiry_calls[expiry_calls['strike'] == otm_strike].iloc[0]
            atm_contract = expiry_calls[expiry_calls['strike'] == atm_strike].iloc[0]
            itm_contract = expiry_calls[expiry_calls['strike'] == itm_strike].iloc[0]
        except IndexError:
            return {'error': 'Could not find contracts for selected strikes'}
        
        return {
            'type': 'CALL',
            'direction': 'BULLISH',
            'spot_price': spot_price,
            'expiry': nearest_expiry.strftime('%Y-%m-%d'),
            'days_to_expiry': (nearest_expiry - pd.Timestamp.now()).days,
            'options': {
                'OTM': self._format_contract(otm_contract, spot_price, 'OTM'),
                'ATM': self._format_contract(atm_contract, spot_price, 'ATM'),
                'ITM': self._format_contract(itm_contract, spot_price, 'ITM')
            },
            'recommended': self._format_contract(itm_contract, spot_price, 'ITM'),
            'recommendation_reason': 'Slightly ITM Call provides built-in intrinsic value protection while maintaining good profit potential for bullish moves',
            'trend_confidence': trend_analysis['confidence'],
            'combined_score': trend_analysis['combined_score']
        }
    
    def _select_put_options(self, puts_df: pd.DataFrame,
                           spot_price: float,
                           trend_analysis: Dict) -> Dict:
        """
        Select Put options: OTM, ATM, ITM
        For BEARISH trend - expecting price to fall
        """
        
        if puts_df.empty:
            return {'error': 'No put options available'}
        
        strikes = sorted(puts_df['strike'].unique())
        nearest_expiry = puts_df['expiry'].min()
        expiry_puts = puts_df[puts_df['expiry'] == nearest_expiry]
        
        strike_distances = {s: abs(s - spot_price) for s in strikes}
        
        # ATM: Closest to spot price
        atm_strike = min(strike_distances, key=strike_distances.get)
        
        # OTM: 1-2 strikes BELOW spot (cheaper, higher risk)
        otm_strikes = [s for s in strikes if s < spot_price]
        otm_strike = otm_strikes[-1] if otm_strikes else atm_strike
        
        # ITM: 1 strike ABOVE spot (SLIGHTLY ITM)
        itm_strikes = [s for s in strikes if s > spot_price]
        itm_strike = itm_strikes[0] if itm_strikes else atm_strike
        
        print(f"📊 PUT OPTION STRIKES SELECTED:")
        print(f"   OTM Strike: ₹{otm_strike:,.0f} (Below spot)")
        print(f"   ATM Strike: ₹{atm_strike:,.0f} (At spot)")
        print(f"   ITM Strike: ₹{itm_strike:,.0f} (Above spot - RECOMMENDED)")
        print()
        
        try:
            otm_contract = expiry_puts[expiry_puts['strike'] == otm_strike].iloc[0]
            atm_contract = expiry_puts[expiry_puts['strike'] == atm_strike].iloc[0]
            itm_contract = expiry_puts[expiry_puts['strike'] == itm_strike].iloc[0]
        except IndexError:
            return {'error': 'Could not find contracts for selected strikes'}
        
        return {
            'type': 'PUT',
            'direction': 'BEARISH',
            'spot_price': spot_price,
            'expiry': nearest_expiry.strftime('%Y-%m-%d'),
            'days_to_expiry': (nearest_expiry - pd.Timestamp.now()).days,
            'options': {
                'OTM': self._format_contract(otm_contract, spot_price, 'OTM'),
                'ATM': self._format_contract(atm_contract, spot_price, 'ATM'),
                'ITM': self._format_contract(itm_contract, spot_price, 'ITM')
            },
            'recommended': self._format_contract(itm_contract, spot_price, 'ITM'),
            'recommendation_reason': 'Slightly ITM Put provides built-in intrinsic value protection while maintaining good profit potential for bearish moves',
            'trend_confidence': trend_analysis['confidence'],
            'combined_score': trend_analysis['combined_score']
        }
    
    def _format_contract(self, contract: pd.Series, spot_price: float, 
                        moneyness: str) -> Dict:
        """Format contract details for display"""
        
        strike = contract['strike']
        
        # Calculate intrinsic value
        if moneyness == 'ITM':
            if 'CE' in contract['tradingsymbol']:  # Call
                intrinsic_value = max(0, spot_price - strike)
            else:  # Put
                intrinsic_value = max(0, strike - spot_price)
        else:
            intrinsic_value = 0
        
        return {
            'tradingsymbol': contract['tradingsymbol'],
            'strike': float(strike),
            'expiry': contract['expiry'].strftime('%Y-%m-%d'),
            'instrument_token': int(contract['instrument_token']),
            'lot_size': int(contract['lot_size']),
            'moneyness': moneyness,
            'distance_from_spot': round(abs(strike - spot_price), 2),
            'intrinsic_value': round(intrinsic_value, 2),
            'percentage_from_spot': round((strike - spot_price) / spot_price * 100, 2)
        }
