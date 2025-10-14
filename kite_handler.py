"""
Kite Connect Handler - Complete integration for Indian markets
Based on official Kite Connect API documentation
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
    TRADING_CONFIG
)

from cache_utils import (
    update_instruments_cache,
    update_indices_cache
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
        """Initialize Kite Connect session"""
        if not KITE_API_KEY or not KITE_ACCESS_TOKEN:
            return False, "❌ Kite API credentials missing"
        
        try:
            self.kite = KiteConnect(api_key=KITE_API_KEY)
            self.kite.set_access_token(KITE_ACCESS_TOKEN)
            
            self.user_profile = self.kite.profile()
            self.connected = True
            
            username = self.user_profile.get('user_name', 'User')
            return True, f"✅ Connected as {username}"
            
        except Exception as e:
            self.connected = False
            return False, f"❌ Connection failed: {str(e)}"
    
    # ========================================================================
    # INSTRUMENT MANAGEMENT - CORRECTED
    # ========================================================================
    
    def fetch_and_cache_instruments(self, exchange: str = "NSE") -> bool:
        """Fetch instruments from Kite API"""
        if not self.connected:
            print("❌ Not connected to Kite")
            return False
        
        try:
            print(f"📥 Fetching {exchange} instruments...")
            
            instruments = self.kite.instruments(exchange)
            
            if self.instruments_df is None:
                self.instruments_df = pd.DataFrame(instruments)
            else:
                new_df = pd.DataFrame(instruments)
                self.instruments_df = pd.concat([self.instruments_df, new_df], ignore_index=True)
            
            self.last_instrument_fetch = datetime.now()
            
            # Store in database
            instruments_dict = self.instruments_df.to_dict('records')
            insert_instruments(instruments_dict)
            
            print(f"✅ Loaded {len(instruments)} instruments from {exchange}")
            
            # Cache index options data after NFO is loaded
            if exchange == "NFO":
                self._cache_index_options()
            
            return True
            
        except Exception as e:
            print(f"❌ Error fetching instruments: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _cache_index_options(self):
        """Extract and cache index lot sizes from NFO-OPT"""
        if self.instruments_df is None or self.instruments_df.empty:
            return
        
        try:
            # Get unique index names from NFO options
            nfo_options = self.instruments_df[
                self.instruments_df['segment'] == 'NFO-OPT'
            ]
            
            if nfo_options.empty:
                print("⚠️ No NFO options found")
                return
            
            index_names = nfo_options['name'].dropna().unique()
            
            index_config = {}
            for index_name in index_names:
                idx_opts = nfo_options[nfo_options['name'] == index_name]
                if not idx_opts.empty:
                    first = idx_opts.iloc[0]
                    index_config[index_name] = {
                        "symbol": index_name,
                        "lot_size": int(first['lot_size']),
                        "tick_size": float(first['tick_size']),
                        "exchange": "NFO"
                    }
            
            update_instruments_cache(index_config)
            print(f"✅ Cached {len(index_config)} index configs")
            
        except Exception as e:
            print(f"❌ Error caching index options: {e}")
    
    # ========================================================================
    # INDEX MANAGEMENT - CORRECTED BASED ON DOCS
    # ========================================================================
    
    def get_indices_by_exchange(self, exchange: str = "NSE") -> List[str]:
        """
        Get available indices from NFO-OPT underlying names
        According to Kite docs, index names are in the 'name' field of options
        """
        if self.instruments_df is None or self.instruments_df.empty:
            print("⚠️ Instruments not loaded yet")
            return []
        
        try:
            # Get indices from NFO options underlying names
            nfo_options = self.instruments_df[
                self.instruments_df['segment'] == 'NFO-OPT'
            ]
            
            if nfo_options.empty:
                print("❌ No NFO options found")
                return []
            
            # Get unique index names
            index_names = sorted(nfo_options['name'].dropna().unique().tolist())
            
            # Update cache
            update_indices_cache(exchange, index_names)
            
            print(f"✅ Found {len(index_names)} indices: {index_names[:5]}")
            return index_names
            
        except Exception as e:
            print(f"❌ Error getting indices: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_index_ltp(self, index_name: str, exchange: str = "NSE") -> Optional[float]:
        """
        Get Last Traded Price for an index - Fully dynamic, no hardcoded mappings
        Searches through instruments to find the correct tradingsymbol
        """
        if not self.connected:
            print("❌ Not connected to Kite")
            return None
        
        if self.instruments_df is None or self.instruments_df.empty:
            print("❌ Instruments not loaded")
            return None
        
        try:
            # Search for this index in instruments DataFrame
            # The index might appear in different segments/formats
            matches = self.instruments_df[
                (self.instruments_df['name'] == index_name) |
                (self.instruments_df['tradingsymbol'].str.contains(index_name, case=False, na=False))
            ]
            
            if matches.empty:
                print(f"⚠️ No instrument found for: {index_name}")
                return None
            
            print(f"🔍 Found {len(matches)} matches for {index_name}")
            
            # Try each match until we get a successful quote
            for idx, row in matches.iterrows():
                try:
                    symbol = f"{row['exchange']}:{row['tradingsymbol']}"
                    print(f"  Trying: {symbol} (segment: {row['segment']})")
                    
                    quote_data = self.kite.quote([symbol])
                    
                    if symbol in quote_data and 'last_price' in quote_data[symbol]:
                        price = quote_data[symbol]['last_price']
                        print(f"  ✅ Success! Price: {price}")
                        return price
                        
                except Exception as e:
                    print(f"  ⚠️ Failed: {str(e)[:50]}")
                    continue
            
            # If no matches worked, return None
            print(f"❌ Could not fetch price for {index_name} from any match")
            return None
            
        except Exception as e:
            print(f"❌ Error in get_index_ltp: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # ========================================================================
    # OPTIONS CHAIN - CORRECTED
    # ========================================================================
    
    def get_option_chain(
        self,
        index_symbol: str,
        expiry_date: str = None
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[List[str]]]:
        """Get options chain for an index"""
        if not self.connected or self.instruments_df is None:
            return None, None, None
        
        try:
            # Filter for this index's options
            options = self.instruments_df[
                (self.instruments_df['name'] == index_symbol) &
                (self.instruments_df['segment'] == 'NFO-OPT')
            ].copy()
            
            if options.empty:
                print(f"❌ No options found for {index_symbol}")
                return None, None, None
            
            # Get all expiries
            all_expiries = sorted(options['expiry'].dropna().unique())
            all_expiries_str = [exp.strftime('%Y-%m-%d') if pd.notna(exp) else None for exp in all_expiries]
            all_expiries_str = [e for e in all_expiries_str if e is not None]
            
            # Filter by expiry if specified
            if expiry_date:
                options = options[options['expiry'] == pd.to_datetime(expiry_date)]
                if options.empty:
                    print(f"⚠️ No options for expiry {expiry_date}")
                    return None, None, all_expiries_str
            
            # Separate calls and puts
            calls = options[options['instrument_type'] == 'CE'].copy()
            puts = options[options['instrument_type'] == 'PE'].copy()
            
            # Sort by expiry and strike
            calls = calls.sort_values(['expiry', 'strike'])
            puts = puts.sort_values(['expiry', 'strike'])
            
            print(f"✅ {len(calls)} Calls, {len(puts)} Puts across {len(all_expiries)} expiries")
            
            return calls, puts, all_expiries_str
            
        except Exception as e:
            print(f"❌ Error fetching option chain: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    # ========================================================================
    # MARKET DATA - QUOTES
    # ========================================================================
    
    def get_quote(self, symbols: List[str], exchange: str = "NSE") -> Dict:
        """Get quotes for symbols"""
        if not self.connected:
            return {}
        
        try:
            formatted_symbols = [f"{exchange}:{symbol}" for symbol in symbols]
            quotes = self.kite.quote(formatted_symbols)
            return quotes
        except Exception as e:
            print(f"❌ Error fetching quotes: {e}")
            return {}
    
    def get_ltp(self, symbols: List[str], exchange: str = "NSE") -> Dict:
        """Get LTP for symbols"""
        if not self.connected:
            return {}
        
        try:
            formatted_symbols = [f"{exchange}:{symbol}" for symbol in symbols]
            ltp_data = self.kite.ltp(formatted_symbols)
            return ltp_data
        except Exception as e:
            print(f"❌ Error fetching LTP: {e}")
            return {}
    
    # ========================================================================
    # INSTRUMENT SEARCH
    # ========================================================================
    
    def search_instruments(
        self, 
        query: str, 
        exchange: str = "NSE", 
        segment: str = "EQ"
    ) -> pd.DataFrame:
        """Search instruments"""
        if self.instruments_df is None or self.instruments_df.empty:
            return pd.DataFrame()
        
        filtered = self.instruments_df[
            (self.instruments_df['exchange'] == exchange)
        ]
        
        if segment:
            filtered = filtered[filtered['segment'].str.contains(segment, na=False)]
        
        query_upper = query.upper()
        matches = filtered[
            filtered['tradingsymbol'].str.contains(query_upper, na=False) |
            filtered['name'].str.contains(query_upper, na=False, case=False)
        ]
        
        return matches[['tradingsymbol', 'name', 'instrument_token', 'lot_size', 'tick_size', 'exchange', 'segment']]
    
    def get_instrument_token(self, symbol: str, exchange: str = "NSE") -> Optional[int]:
        """Get instrument token for symbol"""
        if self.instruments_df is None or self.instruments_df.empty:
            return None
        
        match = self.instruments_df[
            (self.instruments_df['tradingsymbol'] == symbol) &
            (self.instruments_df['exchange'] == exchange)
        ]
        
        if not match.empty:
            return int(match.iloc[0]['instrument_token'])
        
        return None

# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_kite_handler_instance = None

def get_kite_handler() -> KiteHandler:
    """Get or create singleton instance"""
    global _kite_handler_instance
    
    if _kite_handler_instance is None:
        _kite_handler_instance = KiteHandler()
    
    return _kite_handler_instance

def initialize_kite() -> Tuple[bool, str]:
    """Initialize Kite and fetch instruments"""
    handler = get_kite_handler()
    success, message = handler.initialize()
    
    if success:
        print("📥 Fetching instruments...")
        # Fetch NSE first, then NFO (NFO must be second to cache indices)
        handler.fetch_and_cache_instruments("NSE")
        handler.fetch_and_cache_instruments("NFO")
        print("✅ Instruments loaded and cached")
    
    return success, message
