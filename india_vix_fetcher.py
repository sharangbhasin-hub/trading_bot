"""
India VIX Fetcher
=================
Fetches India VIX (Volatility Index) for IV filtering in VWAP-Strangle strategies.
Professional Enhancement #5 from analysis.

India VIX Symbol in Kite: "INDIA VIX" (NSE index)

Author: Trading System
Date: October 25, 2025
"""

import pandas as pd
from typing import Optional, Dict
from datetime import datetime, timedelta
import logging
from kite_handler import get_kite_handler

logger = logging.getLogger(__name__)

class IndiaVIXFetcher:
    """
    Fetches and caches India VIX data for volatility-based strategy filtering.
    """
    
    def __init__(self):
        """Initialize VIX fetcher"""
        self.kite = None
        self.current_vix = None
        self.last_fetch_time = None
        self.cache_duration_minutes = 5  # Cache for 5 minutes
        self.vix_history = []
    
    def _initialize_kite(self) -> bool:
        """Initialize Kite connection"""
        if self.kite is None:
            self.kite = get_kite_handler()
            if self.kite is None or not self.kite.connected:
                logger.error("Kite handler not initialized")
                return False
        return True
    
    def get_current_vix(self, use_cache: bool = True) -> Optional[float]:
        """
        Get current India VIX value.
        
        Args:
            use_cache: Use cached value if available
        
        Returns:
            float: Current VIX value or None if unavailable
        """
        # Check cache
        if use_cache and self._is_cache_valid():
            logger.debug(f"Using cached VIX: {self.current_vix}")
            return self.current_vix
        
        # Fetch fresh data
        if not self._initialize_kite():
            return None
        
        try:
            # India VIX is traded as an index on NSE
            # Symbol: "INDIA VIX"
            vix_quote = self.kite.kite.quote("NSE:INDIA VIX")
            
            if vix_quote and "NSE:INDIA VIX" in vix_quote:
                vix_data = vix_quote["NSE:INDIA VIX"]
                self.current_vix = vix_data.get('last_price', None)
                self.last_fetch_time = datetime.now()
                
                # Store in history
                self.vix_history.append({
                    'timestamp': self.last_fetch_time,
                    'vix': self.current_vix
                })
                
                logger.info(f"India VIX fetched: {self.current_vix}")
                return self.current_vix
            else:
                logger.warning("India VIX quote not found")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching India VIX: {e}")
            return None
    
    def _is_cache_valid(self) -> bool:
        """Check if cached VIX is still valid"""
        if self.current_vix is None or self.last_fetch_time is None:
            return False
        
        elapsed = (datetime.now() - self.last_fetch_time).total_seconds() / 60
        return elapsed < self.cache_duration_minutes
    
    def get_vix_percentile(self, lookback_days: int = 30) -> Optional[float]:
        """
        Calculate VIX percentile over lookback period.
        Used for Professional Enhancement #5.
        
        Args:
            lookback_days: Number of days to look back
        
        Returns:
            float: Percentile (0-100) of current VIX
        """
        if not self._initialize_kite():
            return None
        
        try:
            # Fetch historical VIX data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Note: Kite may not have historical data for India VIX
            # This is a placeholder - implement based on data availability
            historical_df = self.kite.kite.historical_data(
                instrument_token=self._get_vix_token(),
                from_date=start_date,
                to_date=end_date,
                interval='day'
            )
            
            if historical_df:
                df = pd.DataFrame(historical_df)
                current_vix = self.get_current_vix()
                
                if current_vix and not df.empty:
                    percentile = (df['close'] < current_vix).sum() / len(df) * 100
                    return percentile
            
            return None
            
        except Exception as e:
            logger.warning(f"Could not calculate VIX percentile: {e}")
            return None
    
    def _get_vix_token(self) -> Optional[int]:
        """Get instrument token for India VIX"""
        if not self._initialize_kite():
            return None
        
        try:
            instruments = self.kite.search_instruments("INDIA VIX", exchange="NSE")
            if instruments:
                return instruments[0]['instrument_token']
            return None
        except Exception as e:
            logger.error(f"Error getting VIX token: {e}")
            return None
    
    def check_vix_condition(self, strategy_type: str, 
                           min_vix: Optional[float] = None,
                           max_vix: Optional[float] = None) -> Dict:
        """
        Check if VIX meets strategy requirements.
        Used by both selling and buying strategies.
        
        Args:
            strategy_type: 'SELLING' or 'BUYING'
            min_vix: Minimum VIX required (for selling)
            max_vix: Maximum VIX allowed (for buying)
        
        Returns:
            dict: {
                'condition_met': bool,
                'current_vix': float,
                'reason': str
            }
        """
        current_vix = self.get_current_vix()
        
        if current_vix is None:
            return {
                'condition_met': False,
                'current_vix': None,
                'reason': 'VIX data unavailable'
            }
        
        if strategy_type == 'SELLING':
            # Selling strategy: want high VIX (expensive premiums)
            if min_vix and current_vix >= min_vix:
                return {
                    'condition_met': True,
                    'current_vix': current_vix,
                    'reason': f'VIX {current_vix} >= {min_vix} (good for selling)'
                }
            else:
                return {
                    'condition_met': False,
                    'current_vix': current_vix,
                    'reason': f'VIX {current_vix} < {min_vix} (premiums too cheap)'
                }
        
        elif strategy_type == 'BUYING':
            # Buying strategy: want low VIX (cheap premiums)
            if max_vix and current_vix <= max_vix:
                return {
                    'condition_met': True,
                    'current_vix': current_vix,
                    'reason': f'VIX {current_vix} <= {max_vix} (good for buying)'
                }
            else:
                return {
                    'condition_met': False,
                    'current_vix': current_vix,
                    'reason': f'VIX {current_vix} > {max_vix} (premiums too expensive)'
                }
        
        return {
            'condition_met': False,
            'current_vix': current_vix,
            'reason': 'Invalid strategy type'
        }
    
    def get_vix_history_df(self) -> pd.DataFrame:
        """Get VIX history as DataFrame"""
        if not self.vix_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.vix_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

# ============================================================================
# STANDALONE FUNCTIONS
# ============================================================================

def get_india_vix() -> Optional[float]:
    """
    Quick function to get current India VIX.
    
    Returns:
        float: Current VIX value
    """
    fetcher = IndiaVIXFetcher()
    return fetcher.get_current_vix()
