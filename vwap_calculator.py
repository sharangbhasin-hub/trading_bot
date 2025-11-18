"""
VWAP Calculator for Combined Options Premium
============================================
Calculates Volume-Weighted Average Price for option strangle/straddle premiums.
Used for VWAP-Strangle strategy entry signals.

VWAP Formula:
VWAP = Σ(Price × Volume) / Σ(Volume)

For options without volume data, we use tick-count as proxy for volume.

Author: Trading System
Date: October 25, 2025
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from datetime import datetime, time as dt_time
import logging

logger = logging.getLogger(__name__)

class VWAPCalculator:
    """
    Calculates VWAP for combined option premiums (strangle/straddle).
    Designed for 1-minute timeframe as per strategy requirements.
    """
    
    def __init__(self):
        """Initialize VWAP calculator"""
        self.reset()
    
    def reset(self):
        """Reset calculator for new trading day"""
        self.cumulative_tp_volume = 0  # Cumulative (Typical Price × Volume)
        self.cumulative_volume = 0      # Cumulative Volume
        self.vwap_history = []         # List of (timestamp, vwap) tuples
        self.last_reset_time = None
        logger.debug("VWAP Calculator reset")
    
    def calculate_combined_premium(self, ce_price: float, pe_price: float) -> float:
        """
        Calculate combined premium for strangle.
        
        Args:
            ce_price: Call option premium
            pe_price: Put option premium
        
        Returns:
            float: Combined premium (CE + PE)
        """
        return ce_price + pe_price
    
    def update(self, 
               timestamp: datetime,
               ce_price: float,
               pe_price: float,
               volume: Optional[float] = None) -> float:
        """
        Update VWAP with new combined premium data.
        
        Args:
            timestamp: Current timestamp
            ce_price: Call option price
            pe_price: Put option price
            volume: Trading volume (if available, else uses 1.0)
        
        Returns:
            float: Current VWAP value
        """
        # Calculate combined premium
        combined_premium = self.calculate_combined_premium(ce_price, pe_price)
        
        # Use volume = 1 if not provided (equal weight each tick)
        if volume is None or volume == 0:
            volume = 1.0
        
        # Calculate typical price (for options, use close price)
        typical_price = combined_premium
        
        # Update cumulative values
        self.cumulative_tp_volume += (typical_price * volume)
        self.cumulative_volume += volume
        
        # Calculate VWAP
        if self.cumulative_volume > 0:
            vwap = self.cumulative_tp_volume / self.cumulative_volume
        else:
            vwap = combined_premium  # Fallback to current price
        
        # Store history
        self.vwap_history.append({
            'timestamp': timestamp,
            'combined_premium': combined_premium,
            'vwap': vwap,
            'volume': volume
        })
        
        return vwap
    
    def get_current_vwap(self) -> Optional[float]:
        """
        Get most recent VWAP value.
        
        Returns:
            float: Current VWAP or None if no data
        """
        if not self.vwap_history:
            return None
        return self.vwap_history[-1]['vwap']
    
    def check_crossover(self, 
                        current_premium: float,
                        direction: str = 'below') -> Dict:
        """
        Check if combined premium crosses VWAP.
        Used for entry signal generation.
        
        Args:
            current_premium: Current combined premium
            direction: 'below' (for selling) or 'above' (for buying)
        
        Returns:
            dict: {
                'crossed': bool,
                'direction': str,
                'premium': float,
                'vwap': float,
                'signal_type': str
            }
        """
        current_vwap = self.get_current_vwap()
        
        if current_vwap is None or len(self.vwap_history) < 2:
            return {
                'crossed': False,
                'direction': None,
                'premium': current_premium,
                'vwap': current_vwap,
                'signal_type': None,
                'reason': 'Insufficient data'
            }
        
        # Get previous values
        prev_data = self.vwap_history[-2]
        prev_premium = prev_data['combined_premium']
        prev_vwap = prev_data['vwap']
        
        # Check for crossover based on direction
        if direction == 'below':
            # For SELLING: Check if premium crossed FROM above TO below
            # Step 1: Was premium above VWAP? (prev_premium > prev_vwap)
            # Step 2: Is premium now below VWAP? (current_premium < current_vwap)
            was_above = prev_premium > prev_vwap
            is_below = current_premium < current_vwap
            crossed = was_above and is_below
            
            return {
                'crossed': crossed,
                'direction': 'below',
                'premium': current_premium,
                'vwap': current_vwap,
                'signal_type': 'SELL_STRANGLE' if crossed else None,
                'was_above': was_above,
                'is_below': is_below
            }
        
        elif direction == 'above':
            # For BUYING: Check if premium crossed FROM below TO above
            was_below = prev_premium < prev_vwap
            is_above = current_premium > current_vwap
            crossed = was_below and is_above
            
            return {
                'crossed': crossed,
                'direction': 'above',
                'premium': current_premium,
                'vwap': current_vwap,
                'signal_type': 'BUY_DIRECTIONAL' if crossed else None,
                'was_below': was_below,
                'is_above': is_above
            }
        
        else:
            raise ValueError(f"Invalid direction: {direction}. Must be 'below' or 'above'")
    
    def get_vwap_series(self) -> pd.DataFrame:
        """
        Get VWAP history as DataFrame.
        Useful for charting and analysis.
        
        Returns:
            pd.DataFrame: Columns: timestamp, combined_premium, vwap, volume
        """
        if not self.vwap_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.vwap_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        return df
    
    def should_reset(self, current_time: datetime) -> bool:
        """
        Check if VWAP should reset for new trading day.
        VWAP resets at market open (9:15 AM IST).
        
        Args:
            current_time: Current datetime
        
        Returns:
            bool: True if should reset
        """
        market_open = dt_time(9, 15)
        
        # Reset if:
        # 1. First calculation of the day
        # 2. Crossed market open time
        if self.last_reset_time is None:
            return True
        
        # Check if we've crossed 9:15 AM
        if current_time.time() < market_open and self.last_reset_time.time() >= market_open:
            return True
        
        # Check if it's a new day
        if current_time.date() > self.last_reset_time.date():
            return True
        
        return False
    
    def calculate_from_dataframe(self, df: pd.DataFrame,
                                  ce_col: str = 'ce_price',
                                  pe_col: str = 'pe_price',
                                  volume_col: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate VWAP for entire DataFrame (batch mode).
        Used for backtesting.
        
        Args:
            df: DataFrame with price data
            ce_col: Column name for CE prices
            pe_col: Column name for PE prices
            volume_col: Column name for volume (optional)
        
        Returns:
            pd.DataFrame: Original df with added 'vwap' and 'combined_premium' columns
        """
        df = df.copy()
        
        # Calculate combined premium
        df['combined_premium'] = df[ce_col] + df[pe_col]
        
        # Get volume
        if volume_col and volume_col in df.columns:
            volume = df[volume_col]
        else:
            volume = 1.0  # Equal weight
        
        # Calculate typical price (use combined premium)
        typical_price = df['combined_premium']
        
        # Calculate cumulative values
        df['tp_volume'] = typical_price * volume
        df['cumulative_tp_volume'] = df['tp_volume'].cumsum()
        df['cumulative_volume'] = volume.cumsum() if isinstance(volume, pd.Series) else volume * len(df)
        
        # Calculate VWAP
        if isinstance(df['cumulative_volume'], pd.Series):
            df['vwap'] = df['cumulative_tp_volume'] / df['cumulative_volume']
        else:
            df['vwap'] = df['cumulative_tp_volume'] / df['cumulative_volume']
        
        # Clean up intermediate columns
        df = df.drop(['tp_volume', 'cumulative_tp_volume', 'cumulative_volume'], axis=1)
        
        return df

# ============================================================================
# STANDALONE FUNCTIONS (for backward compatibility)
# ============================================================================

def calculate_vwap_from_ticks(ticks_df: pd.DataFrame,
                              price_col: str = 'last_price',
                              volume_col: str = 'volume') -> float:
    """
    Calculate VWAP from tick data.
    
    Args:
        ticks_df: DataFrame with tick data
        price_col: Column name for price
        volume_col: Column name for volume
    
    Returns:
        float: VWAP value
    """
    if ticks_df.empty:
        return None
    
    if volume_col not in ticks_df.columns:
        # No volume data - use equal weight
        return ticks_df[price_col].mean()
    
    tp_volume = (ticks_df[price_col] * ticks_df[volume_col]).sum()
    total_volume = ticks_df[volume_col].sum()
    
    if total_volume == 0:
        return ticks_df[price_col].mean()
    
    return tp_volume / total_volume
