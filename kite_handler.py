"""
Kite Connect Handler - Complete integration for Indian markets
Handles: Authentication, Instruments, Quotes, Historical Data
"""

from kiteconnect import KiteConnect, KiteTicker
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
import time

from config import (
    KITE_API_KEY, 
    KITE_API_SECRET, 
    KITE_ACCESS_TOKEN,
    TRADING_CONFIG,
    update_instruments_cache,
    INDEX_OPTIONS_REFERENCE
)
from database import (
    insert_instruments,
    get_instrument_by_symbol,
    search_instruments as db_search_instruments
)

# ============================================================================
# KITE CONNECT HANDLER CLASS
# ============================================================================

class KiteHandler:
    """Main handler for Kite Connect API operations"""
    
    def __init__(self):
        """Initialize Kite Connect with credentials"""
        self.kite = None
        self.connected = False
        self.user_profile = None
        self.instruments_df = None
        self.last_instrument_fetch = None
        
    def initialize(self) -> Tuple[bool, str]:
        """
        Initialize Kite Connect session
        Returns: (success: bool, message: str)
        """
        if not KITE_API_KEY or not KITE_ACCESS_TOKEN:
            return False, "❌ Kite API credentials missing in .env file"
        
        try:
            # Initialize KiteConnect
            self.kite = KiteConnect(api_key=KITE_API_KEY)
            self.kite.set_access_token(KITE_ACCESS_TOKEN)
            
            # Verify connection by fetching profile
            self.user_profile = self.kite.profile()
            self.connected = True
            
            username = self.user_profile.get('user_name', 'User')
            return True, f"✅ Connected as {username}"
            
        except Exception as e:
            self.connected = False
            return False, f"❌ Kite Connect initialization failed: {str(e)}"
    
    # ========================================================================
    # INSTRUMENT MANAGEMENT
    # ========================================================================
    
    def fetch_and_cache_instruments(self, exchange: str = "NSE") -> bool:
        """
        Fetch all instruments from Kite and cache them
        This is called once during initialization
        """
        if not self.connected:
            print("❌ Not connected to Kite")
            return False
        
        try:
            print(f"📥 Fetching instruments from {exchange}...")
            
            # Fetch instruments from Kite API
            instruments = self.kite.instruments(exchange)
            
            # Convert to DataFrame for easier manipulation
            self.instruments_df = pd.DataFrame(instruments)
            self.last_instrument_fetch = datetime.now()
            
            # Store in database
            instruments_dict = self.instruments_df.to_dict('records')
            insert_instruments(instruments_dict)
            
            # Update config cache with index options data
            self._cache_index_options()
            
            print(f"✅ Cached {len(self.instruments_df)} instruments from {exchange}")
            return True
            
        except Exception as e:
            print(f"❌ Error fetching instruments: {e}")
            return False
    
    def _cache_index_options(self):
        """
        Extract and cache index options lot sizes and tick sizes
        This populates the config cache with dynamic data
        """
        if self.instruments_df is None or self.instruments_df.empty:
            return
        
        index_config = {}
        
        for index_name in INDEX_OPTIONS_REFERENCE:
            # Find instruments matching this index
            index_instruments = self.instruments_df[
                (self.instruments_df['name'] == index_name) & 
                (self.instruments_df['segment'] == 'NFO-OPT')
            ]
            
            if not index_instruments.empty:
                # Get lot size and tick size (should be same for all strikes/expiries)
                first_inst = index_instruments.iloc[0]
                
                index_config[index_name] = {
                    "symbol": index_name,
                    "lot_size": int(first_inst['lot_size']),
                    "tick_size": float(first_inst['tick_size']),
                    "exchange": "NFO"
                }
        
        # Update global config cache
        update_instruments_cache(index_config)
        print(f"✅ Cached {len(index_config)} index configurations dynamically")
    
    def search_instruments(
        self, 
        query: str, 
        exchange: str = "NSE", 
        segment: str = "EQ"
    ) -> pd.DataFrame:
        """
        Search instruments by symbol or name
        Args:
            query: Search keyword
            exchange: NSE, BSE, NFO, etc.
            segment: EQ (equity), FUT (futures), OPT (options)
        Returns:
            DataFrame with matching instruments
        """
        if self.instruments_df is None or self.instruments_df.empty:
            return pd.DataFrame()
        
        # Filter by exchange and segment
        filtered = self.instruments_df[
            (self.instruments_df['exchange'] == exchange)
        ]
        
        if segment:
            filtered = filtered[filtered['segment'].str.contains(segment, na=False)]
        
        # Search in tradingsymbol or name
        query_upper = query.upper()
        matches = filtered[
            filtered['tradingsymbol'].str.contains(query_upper, na=False) |
            filtered['name'].str.contains(query_upper, na=False, case=False)
        ]
        
        return matches[['tradingsymbol', 'name', 'instrument_token', 'lot_size', 'tick_size', 'exchange', 'segment']]
    
    def get_instrument_token(self, symbol: str, exchange: str = "NSE") -> Optional[int]:
        """
        Get instrument token for a symbol
        Required for WebSocket streaming
        """
        if self.instruments_df is None or self.instruments_df.empty:
            return None
        
        match = self.instruments_df[
            (self.instruments_df['tradingsymbol'] == symbol) &
            (self.instruments_df['exchange'] == exchange)
        ]
        
        if not match.empty:
            return int(match.iloc[0]['instrument_token'])
        
        return None
    
    # ========================================================================
    # MARKET DATA - QUOTES
    # ========================================================================
    
    def get_quote(self, symbols: List[str], exchange: str = "NSE") -> Dict:
        """
        Get real-time quotes for multiple symbols
        Args:
            symbols: List of trading symbols ['RELIANCE', 'TCS']
            exchange: Exchange code
        Returns:
            Dict with quotes data
        """
        if not self.connected:
            return {}
        
        try:
            # Format symbols with exchange prefix
            formatted_symbols = [f"{exchange}:{symbol}" for symbol in symbols]
            
            # Fetch quotes from Kite
            quotes = self.kite.quote(formatted_symbols)
            
            return quotes
            
        except Exception as e:
            print(f"❌ Error fetching quotes: {e}")
            return {}
    
    def get_ltp(self, symbols: List[str], exchange: str = "NSE") -> Dict:
        """
        Get Last Traded Price for symbols
        Faster than full quote if you only need LTP
        """
        if not self.connected:
            return {}
        
        try:
            formatted_symbols = [f"{exchange}:{symbol}" for symbol in symbols]
            ltp_data = self.kite.ltp(formatted_symbols)
            return ltp_data
            
        except Exception as e:
            print(f"❌ Error fetching LTP: {e}")
            return {}
    
    def get_ohlc(self, symbols: List[str], exchange: str = "NSE") -> Dict:
        """
        Get OHLC (Open, High, Low, Close) for symbols
        Includes previous close and volume
        """
        if not self.connected:
            return {}
        
        try:
            formatted_symbols = [f"{exchange}:{symbol}" for symbol in symbols]
            ohlc_data = self.kite.ohlc(formatted_symbols)
            return ohlc_data
            
        except Exception as e:
            print(f"❌ Error fetching OHLC: {e}")
            return {}
    
    # ========================================================================
    # MARKET DATA - HISTORICAL
    # ========================================================================
    
    def get_historical_data(
        self,
        instrument_token: int,
        from_date: datetime,
        to_date: datetime,
        interval: str = "minute"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data
        Args:
            instrument_token: Token from instruments list
            from_date: Start date
            to_date: End date
            interval: minute, 3minute, 5minute, 15minute, 30minute, 60minute, day
        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        if not self.connected:
            return None
        
        try:
            # Fetch historical data
            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            
            if not data:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching historical data: {e}")
            return None
    
    def get_historical_data_by_symbol(
        self,
        symbol: str,
        exchange: str = "NSE",
        days: int = 7,
        interval: str = "minute"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical data using symbol (convenience method)
        """
        # Get instrument token
        instrument_token = self.get_instrument_token(symbol, exchange)
        
        if not instrument_token:
            print(f"❌ Could not find instrument token for {symbol}")
            return None
        
        # Calculate date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        return self.get_historical_data(
            instrument_token=instrument_token,
            from_date=from_date,
            to_date=to_date,
            interval=interval
        )
    
    # ========================================================================
    # OPTIONS CHAIN
    # ========================================================================
    
    def get_option_chain(
        self,
        index_symbol: str,
        expiry_date: str = None
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Get options chain for an index
        Args:
            index_symbol: NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY
            expiry_date: Optional specific expiry (YYYY-MM-DD format)
        Returns:
            (calls_df, puts_df) - Two DataFrames for calls and puts
        """
        if not self.connected or self.instruments_df is None:
            return None, None
        
        try:
            # Filter options for this index
            options = self.instruments_df[
                (self.instruments_df['name'] == index_symbol) &
                (self.instruments_df['segment'] == 'NFO-OPT')
            ].copy()
            
            if options.empty:
                return None, None
            
            # Filter by expiry if specified
            if expiry_date:
                options = options[options['expiry'] == pd.to_datetime(expiry_date)]
            else:
                # Get nearest expiry
                nearest_expiry = options['expiry'].min()
                options = options[options['expiry'] == nearest_expiry]
            
            # Separate calls and puts
            calls = options[options['instrument_type'] == 'CE'].copy()
            puts = options[options['instrument_type'] == 'PE'].copy()
            
            # Sort by strike
            calls = calls.sort_values('strike')
            puts = puts.sort_values('strike')
            
            return calls, puts
            
        except Exception as e:
            print(f"❌ Error fetching option chain: {e}")
            return None, None
    
    def get_option_greeks(self, instrument_tokens: List[int]) -> Dict:
        """
        Get option greeks (delta, gamma, theta, vega)
        Note: Kite provides basic greeks in quote data
        """
        if not self.connected:
            return {}
        
        try:
            quotes = self.kite.quote([f"NFO:{token}" for token in instrument_tokens])
            return quotes
        except Exception as e:
            print(f"❌ Error fetching option greeks: {e}")
            return {}
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_market_status(self) -> Dict:
        """Get current market status from Kite"""
        if not self.connected:
            return {"status": "DISCONNECTED"}
        
        try:
            # Note: Kite doesn't have direct market status API
            # We can infer from quote availability
            return {"status": "CONNECTED", "session": "active"}
        except:
            return {"status": "ERROR"}
    
    def refresh_access_token(self) -> Tuple[bool, str]:
        """
        Refresh access token if expired
        Note: For production, implement proper token refresh logic
        """
        return self.initialize()


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

# Create a singleton instance that can be imported
_kite_handler_instance = None

def get_kite_handler() -> KiteHandler:
    """Get or create singleton KiteHandler instance"""
    global _kite_handler_instance
    
    if _kite_handler_instance is None:
        _kite_handler_instance = KiteHandler()
    
    return _kite_handler_instance


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def initialize_kite() -> Tuple[bool, str]:
    """Initialize Kite Connect (convenience function)"""
    handler = get_kite_handler()
    success, message = handler.initialize()
    
    if success:
        # Fetch and cache instruments on successful connection
        handler.fetch_and_cache_instruments("NSE")
        handler.fetch_and_cache_instruments("NFO")
    
    return success, message
