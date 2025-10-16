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
        Main function: Selects option contract based on consensus trend
        Now uses 'overall_trend' from market consensus instead of old trend_analysis
        Returns all three (OTM, ATM, ITM) with ITM as final recommendation
        """
        
        # Extract trend from consensus (new format)
        overall_trend = trend_analysis.get('overall_trend', 'Neutral')
        consensus_bullish_pct = trend_analysis.get('consensus_bullish_pct', 50)
        consensus_bearish_pct = trend_analysis.get('consensus_bearish_pct', 50)
        
        # Determine direction and action from consensus
        if 'bullish' in overall_trend.lower():
            direction = "BULLISH"
            action = "CALL"
        elif 'bearish' in overall_trend.lower():
            direction = "BEARISH"
            action = "PUT"
        else:
            direction = "NEUTRAL"
            action = "HOLD"
        
        print(f"\n{'='*60}")
        print(f"STRIKE SELECTION (Based on Market Consensus)")
        print(f"{'='*60}")
        print(f"Overall Trend: {overall_trend}")
        print(f"Direction: {direction}")
        print(f"Action: {action}")
        print(f"Consensus: {consensus_bullish_pct:.1f}% Bullish, {consensus_bearish_pct:.1f}% Bearish")
        print(f"Spot Price: ₹{spot_price:,.2f}")
        print(f"{'='*60}\n")
        
        if action == "CALL":
            return self._select_call_options(calls_df, spot_price, trend_analysis)
        elif action == "PUT":
            return self._select_put_options(puts_df, spot_price, trend_analysis)
        else:
            return {
                'error': 'No clear trend detected from consensus',
                'recommendation': 'WAIT for better market conditions',
                'reason': f'Market consensus shows mixed signals: {overall_trend}',
                'consensus': {
                    'bullish_pct': consensus_bullish_pct,
                    'bearish_pct': consensus_bearish_pct,
                    'trend': overall_trend
                }
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

        # ✅ NEW: Check expiry day risk (Loophole #4)
        nearest_expiry_ts = pd.Timestamp(nearest_expiry)
        days_to_expiry = (nearest_expiry_ts - pd.Timestamp.now()).days

        if days_to_expiry <= 0:
            print(f"🚫 EXPIRY DAY DETECTED - Trading disabled")
            return {
                'error': 'Expiry day trading disabled',
                'recommendation': 'WAIT - Do not trade on expiry day',
                'reason': 'Expiry day (Thursday) has extreme theta decay and price manipulation risk',
                'days_to_expiry': days_to_expiry,
                'expiry_date': nearest_expiry.strftime('%Y-%m-%d')
            }
        
        if days_to_expiry == 1:
            print(f"⚠️  WARNING: Only 1 day to expiry - High risk")
            print(f"   Consider using next week expiry for safer trades")

        expiry_calls = calls_df[calls_df['expiry'] == nearest_expiry]

        # ✅ NEW: Check if expiry has contracts (Loophole #10)
        if expiry_calls.empty:
            print(f"⚠️  No contracts for nearest expiry: {nearest_expiry.strftime('%Y-%m-%d')}")
            
            # Try next available expiry
            all_expiries = sorted(calls_df['expiry'].unique())
            print(f"   Available expiries: {[exp.strftime('%Y-%m-%d') for exp in all_expiries]}")
            
            if len(all_expiries) > 1:
                nearest_expiry = all_expiries[1]
                expiry_calls = calls_df[calls_df['expiry'] == nearest_expiry]
                print(f"   ✅ Using next expiry: {nearest_expiry.strftime('%Y-%m-%d')}")
                
                # Re-check days to expiry for new expiry
                nearest_expiry_ts = pd.Timestamp(nearest_expiry)
                days_to_expiry = (nearest_expiry_ts - pd.Timestamp.now()).days

                if expiry_calls.empty:
                    return {
                        'error': 'No valid expiry with contracts found',
                        'available_expiries': [exp.strftime('%Y-%m-%d') for exp in all_expiries],
                        'recommendation': 'WAIT - Reload options chain data'
                    }
            else:
                return {
                    'error': 'No valid expiry found',
                    'recommendation': 'WAIT - No alternative expiries available'
                }
        
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

        # ✅ NEW: Check expiry day risk (Loophole #4)
        nearest_expiry_ts = pd.Timestamp(nearest_expiry)
        days_to_expiry = (nearest_expiry_ts - pd.Timestamp.now()).days
        
        if days_to_expiry <= 0:
            print(f"🚫 EXPIRY DAY DETECTED - Trading disabled")
            return {
                'error': 'Expiry day trading disabled',
                'recommendation': 'WAIT - Do not trade on expiry day',
                'reason': 'Expiry day (Thursday) has extreme theta decay and price manipulation risk',
                'days_to_expiry': days_to_expiry,
                'expiry_date': nearest_expiry.strftime('%Y-%m-%d')
            }
        
        if days_to_expiry == 1:
            print(f"⚠️  WARNING: Only 1 day to expiry - High risk")

        expiry_puts = puts_df[puts_df['expiry'] == nearest_expiry]

        # ✅ NEW: Check if expiry has contracts (Loophole #10)
        if expiry_puts.empty:
            print(f"⚠️  No contracts for nearest expiry: {nearest_expiry.strftime('%Y-%m-%d')}")
            
            all_expiries = sorted(puts_df['expiry'].unique())
            print(f"   Available expiries: {[exp.strftime('%Y-%m-%d') for exp in all_expiries]}")
            
            if len(all_expiries) > 1:
                nearest_expiry = all_expiries[1]
                expiry_puts = puts_df[puts_df['expiry'] == nearest_expiry]
                print(f"   ✅ Using next expiry: {nearest_expiry.strftime('%Y-%m-%d')}")
                
                nearest_expiry_ts = pd.Timestamp(nearest_expiry)
                days_to_expiry = (nearest_expiry_ts - pd.Timestamp.now()).days

                
                if expiry_puts.empty:
                    return {
                        'error': 'No valid expiry with contracts found',
                        'available_expiries': [exp.strftime('%Y-%m-%d') for exp in all_expiries],
                        'recommendation': 'WAIT - Reload options chain data'
                    }
            else:
                return {
                    'error': 'No valid expiry found',
                    'recommendation': 'WAIT - No alternative expiries available'
                }
        
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
