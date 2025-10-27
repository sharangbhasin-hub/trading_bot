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
        
        # ✅ PROFESSIONAL EXPIRY SELECTION: Skip risky expiries (0-1 days)
        all_expiries = sorted(calls_df['expiry'].unique())
        
        if not all_expiries:
            return {
                'error': 'No expiries available',
                'recommendation': 'RELOAD options chain data'
            }
        
        # Loop through expiries to find first safe one (2+ days away)
        nearest_expiry = None
        days_to_expiry = -1
        
        for expiry in all_expiries:
            expiry_ts = pd.Timestamp(expiry)
            days = (expiry_ts - pd.Timestamp.now()).days
            
            # Professional rule: Minimum 2 days to expiry for ITM strategies
            if days >= 2:
                nearest_expiry = expiry
                days_to_expiry = days
                print(f"✅ Selected expiry: {expiry.strftime('%Y-%m-%d')} ({days} days away)")
                break
            else:
                print(f"⚠️ Skipping expiry: {expiry.strftime('%Y-%m-%d')} ({days} days - too close)")
        
        # If no valid expiry found (all are 0-1 days away), block trading
        if nearest_expiry is None or days_to_expiry < 2:
            return {
                'error': 'All available expiries too close to expiry',
                'recommendation': 'WAIT - Do not trade on expiry week',
                'reason': 'Professional traders avoid contracts with <2 days to expiry (high theta decay + manipulation risk)',
                'available_expiries': [exp.strftime('%Y-%m-%d') for exp in all_expiries],
                'days_until_next_safe_expiry': days_to_expiry if days_to_expiry > 0 else 'N/A'
            }
        
        # Filter contracts for the selected safe expiry
        expiry_calls = calls_df[calls_df['expiry'] == nearest_expiry]
        
        if expiry_calls.empty:
            return {
                'error': f'No contracts found for expiry {nearest_expiry.strftime("%Y-%m-%d")}',
                'recommendation': 'RELOAD options chain or try different index'
            }
        
        print(f"\n📅 EXPIRY DETAILS:")
        print(f"   Selected Expiry: {nearest_expiry.strftime('%Y-%m-%d %A')}")
        print(f"   Days to Expiry: {days_to_expiry}")
        print(f"   Contracts Available: {len(expiry_calls)}")
        print(f"   ✅ Safe to trade (ITM strategy)\n")
        
        # ✅ FIXED: Calculate strike distances correctly
        strike_distances = {s: abs(s - spot_price) for s in strikes}
        
        # ATM: Closest to spot price
        atm_strike = min(strike_distances, key=strike_distances.get)
        
        # ✅ FIXED: Separate OTM selection logic
        # For CALLS: OTM = strikes ABOVE spot (expecting price to rise)
        otm_strikes = [s for s in strikes if s > spot_price]
        if len(otm_strikes) >= 2:
            otm_strike = otm_strikes[1]  # Take 2nd strike above (more OTM)
        elif len(otm_strikes) == 1:
            otm_strike = otm_strikes[0]  # Take 1st strike above
        else:
            otm_strike = atm_strike  # Fallback to ATM
        
        # ✅ FIXED: ITM selection for CALLS
        # For CALLS: ITM = strikes BELOW spot (already has intrinsic value)
        itm_strikes = [s for s in strikes if s < spot_price]
        if len(itm_strikes) >= 1:
            itm_strike = itm_strikes[-1]  # Take closest strike below spot
        else:
            itm_strike = atm_strike  # Fallback to ATM
        
        print(f"\n📊 CALL OPTION STRIKES SELECTED:")
        print(f"   Spot Price: ₹{spot_price:,.0f}")
        print(f"   ITM Strike: ₹{itm_strike:,.0f} (Below spot - RECOMMENDED)")
        print(f"   ATM Strike: ₹{atm_strike:,.0f} (At spot)")
        print(f"   OTM Strike: ₹{otm_strike:,.0f} (Above spot)")
        print()
        
        # ✅ VALIDATION: Ensure all three strikes are different
        if itm_strike == atm_strike == otm_strike:
            print("⚠️ WARNING: All strikes are same - insufficient strike range")
        elif atm_strike == otm_strike:
            print(f"⚠️ WARNING: ATM and OTM are same ({atm_strike}) - only 1 strike above spot available")
            # Force OTM to be different if possible
            higher_otm = [s for s in strikes if s > atm_strike]
            if higher_otm:
                otm_strike = higher_otm[0]
                print(f"   ✅ Corrected OTM to: ₹{otm_strike:,.0f}")
        
        # Get contract details
        try:
            # ✅ FIXED: Explicit filtering for each strike
            itm_contract = expiry_calls[expiry_calls['strike'] == itm_strike].iloc[0]
            atm_contract = expiry_calls[expiry_calls['strike'] == atm_strike].iloc[0]
            otm_contract = expiry_calls[expiry_calls['strike'] == otm_strike].iloc[0]
            
            # ✅ DEBUG: Print trading symbols to verify
            print(f"   ITM Contract: {itm_contract['tradingsymbol']}")
            print(f"   ATM Contract: {atm_contract['tradingsymbol']}")
            print(f"   OTM Contract: {otm_contract['tradingsymbol']}")
            
        except IndexError as e:
            return {'error': f'Could not find contracts for selected strikes: {str(e)}'}
        
        return {
            'type': 'CALL',
            'direction': 'BULLISH',
            'spot_price': spot_price,
            'expiry': nearest_expiry.strftime('%Y-%m-%d'),
            'days_to_expiry': days_to_expiry,
            'options': {
                'OTM': self._format_contract(otm_contract, spot_price, 'OTM'),
                'ATM': self._format_contract(atm_contract, spot_price, 'ATM'),
                'ITM': self._format_contract(itm_contract, spot_price, 'ITM')
            },
            'recommended': self._format_contract(itm_contract, spot_price, 'ITM'),
            'recommendation_reason': 'Slightly ITM Call provides built-in intrinsic value protection while maintaining good profit potential for bullish moves',
            'consensus_bullish_pct': trend_analysis.get('consensus_bullish_pct', 50),
            'consensus_bearish_pct': trend_analysis.get('consensus_bearish_pct', 50),
            'overall_trend': trend_analysis.get('overall_trend', 'Neutral')
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
        
        # ✅ PROFESSIONAL EXPIRY SELECTION: Skip risky expiries (0-1 days)
        all_expiries = sorted(puts_df['expiry'].unique())
        
        if not all_expiries:
            return {
                'error': 'No expiries available',
                'recommendation': 'RELOAD options chain data'
            }
        
        # Loop through expiries to find first safe one (2+ days away)
        nearest_expiry = None
        days_to_expiry = -1
        
        for expiry in all_expiries:
            expiry_ts = pd.Timestamp(expiry)
            days = (expiry_ts - pd.Timestamp.now()).days
            
            # Professional rule: Minimum 2 days to expiry for ITM strategies
            if days >= 2:
                nearest_expiry = expiry
                days_to_expiry = days
                print(f"✅ Selected expiry: {expiry.strftime('%Y-%m-%d')} ({days} days away)")
                break
            else:
                print(f"⚠️ Skipping expiry: {expiry.strftime('%Y-%m-%d')} ({days} days - too close)")
        
        # If no valid expiry found (all are 0-1 days away), block trading
        if nearest_expiry is None or days_to_expiry < 2:
            return {
                'error': 'All available expiries too close to expiry',
                'recommendation': 'WAIT - Do not trade on expiry week',
                'reason': 'Professional traders avoid contracts with <2 days to expiry (high theta decay + manipulation risk)',
                'available_expiries': [exp.strftime('%Y-%m-%d') for exp in all_expiries],
                'days_until_next_safe_expiry': days_to_expiry if days_to_expiry > 0 else 'N/A'
            }
        
        # Filter contracts for the selected safe expiry
        expiry_puts = puts_df[puts_df['expiry'] == nearest_expiry]
        
        if expiry_puts.empty:
            return {
                'error': f'No contracts found for expiry {nearest_expiry.strftime("%Y-%m-%d")}',
                'recommendation': 'RELOAD options chain or try different index'
            }
        
        print(f"\n📅 EXPIRY DETAILS:")
        print(f"   Selected Expiry: {nearest_expiry.strftime('%Y-%m-%d %A')}")
        print(f"   Days to Expiry: {days_to_expiry}")
        print(f"   Contracts Available: {len(expiry_puts)}")
        print(f"   ✅ Safe to trade (ITM strategy)\n")
        
        # ✅ FIXED: Calculate strike distances correctly
        strike_distances = {s: abs(s - spot_price) for s in strikes}
        
        # ATM: Closest to spot price
        atm_strike = min(strike_distances, key=strike_distances.get)
        
        # ✅ FIXED: OTM for PUTS = strikes BELOW spot (cheaper, expecting price to fall)
        otm_strikes = [s for s in strikes if s < spot_price]
        if len(otm_strikes) >= 2:
            otm_strike = otm_strikes[-2]  # Take 2nd strike below (more OTM)
        elif len(otm_strikes) == 1:
            otm_strike = otm_strikes[-1]  # Take 1st strike below
        else:
            otm_strike = atm_strike  # Fallback to ATM
        
        # ✅ FIXED: ITM for PUTS = strikes ABOVE spot (has intrinsic value)
        itm_strikes = [s for s in strikes if s > spot_price]
        if len(itm_strikes) >= 1:
            itm_strike = itm_strikes[0]  # Take closest strike above spot (slightly ITM)
        else:
            itm_strike = atm_strike  # Fallback to ATM
        
        print(f"\n📊 PUT OPTION STRIKES SELECTED:")
        print(f"   Spot Price: ₹{spot_price:,.0f}")
        print(f"   OTM Strike: ₹{otm_strike:,.0f} (Below spot)")
        print(f"   ATM Strike: ₹{atm_strike:,.0f} (At spot)")
        print(f"   ITM Strike: ₹{itm_strike:,.0f} (Above spot - RECOMMENDED)")
        print()
        
        # ✅ VALIDATION: Ensure all three strikes are different
        if itm_strike == atm_strike == otm_strike:
            print("⚠️ WARNING: All strikes are same - insufficient strike range")
        elif atm_strike == otm_strike:
            print(f"⚠️ WARNING: ATM and OTM are same ({atm_strike}) - only 1 strike below spot available")
            # Force OTM to be different if possible
            lower_otm = [s for s in strikes if s < atm_strike]
            if lower_otm:
                otm_strike = lower_otm[-1]
                print(f"   ✅ Corrected OTM to: ₹{otm_strike:,.0f}")
        
        # Get contract details
        try:
            # ✅ FIXED: Explicit filtering for each strike
            otm_contract = expiry_puts[expiry_puts['strike'] == otm_strike].iloc[0]
            atm_contract = expiry_puts[expiry_puts['strike'] == atm_strike].iloc[0]
            itm_contract = expiry_puts[expiry_puts['strike'] == itm_strike].iloc[0]
            
            # ✅ DEBUG: Print trading symbols to verify
            print(f"   OTM Contract: {otm_contract['tradingsymbol']}")
            print(f"   ATM Contract: {atm_contract['tradingsymbol']}")
            print(f"   ITM Contract: {itm_contract['tradingsymbol']}")
            
        except IndexError as e:
            return {'error': f'Could not find contracts for selected strikes: {str(e)}'}
        
        return {
            'type': 'PUT',
            'direction': 'BEARISH',
            'spot_price': spot_price,
            'expiry': nearest_expiry.strftime('%Y-%m-%d'),
            'days_to_expiry': days_to_expiry,
            'options': {
                'OTM': self._format_contract(otm_contract, spot_price, 'OTM'),
                'ATM': self._format_contract(atm_contract, spot_price, 'ATM'),
                'ITM': self._format_contract(itm_contract, spot_price, 'ITM')
            },
            'recommended': self._format_contract(itm_contract, spot_price, 'ITM'),
            'recommendation_reason': 'Slightly ITM Put provides built-in intrinsic value protection while maintaining good profit potential for bearish moves',
            'consensus_bullish_pct': trend_analysis.get('consensus_bullish_pct', 50),
            'consensus_bearish_pct': trend_analysis.get('consensus_bearish_pct', 50),
            'overall_trend': trend_analysis.get('overall_trend', 'Neutral')
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
