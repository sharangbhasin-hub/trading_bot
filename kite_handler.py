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
        self.index_symbol_cache = {}  # Cache for index -> working tradingsymbol
        self.index_token_map = {}  # ✅ NEW: Map for index name → token
        
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
    # INSTRUMENT MANAGEMENT
    # ========================================================================
    
    def fetch_and_cache_instruments(self, exchange: str = "NSE") -> bool:
        """Fetch instruments from Kite API and remove duplicates"""
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
                
                # ✅ NEW: Remove duplicates based on instrument_token
                # This ensures each instrument appears only once
                initial_count = len(self.instruments_df)
                self.instruments_df = self.instruments_df.drop_duplicates(
                    subset=['instrument_token'], 
                    keep='first'
                )
                removed_count = initial_count - len(self.instruments_df)
                if removed_count > 0:
                    print(f"🧹 Removed {removed_count} duplicate instruments")
            
            self.last_instrument_fetch = datetime.now()
            
            # Store in database
            instruments_dict = self.instruments_df.to_dict('records')
            insert_instruments(instruments_dict)
            
            print(f"✅ Loaded {len(instruments)} instruments from {exchange}")
            print(f"📊 Total unique instruments in memory: {len(self.instruments_df)}")
            
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

            # ✅ NEW: Build index token map
            self._build_index_token_map()
        
        except Exception as e:
            print(f"❌ Error caching index options: {e}")

    def _build_index_token_map(self):
        """
        Build a mapping of index names to their instrument tokens
        Stores metadata for fast lookup - fully dynamic
        """
        if self.instruments_df is None or self.instruments_df.empty:
            return
        
        try:
            # Get all indices from NFO-OPT underlying names
            nfo_options = self.instruments_df[
                self.instruments_df['segment'] == 'NFO-OPT'
            ]
            
            if nfo_options.empty:
                print("⚠️ No NFO options found for index mapping")
                return
            
            # Get unique index names
            index_names = nfo_options['name'].dropna().unique()
            
            print(f"\n{'='*60}")
            print(f"BUILDING INDEX TOKEN MAP")
            print(f"{'='*60}")
            print(f"Found {len(index_names)} index names from NFO-OPT")
            print(f"Index names: {list(index_names)[:10]}")  # Show first 10
            
            # For each index, find its spot instrument in NSE
            self.index_token_map = {}
            
            for index_name in index_names:
                # ✅ NEW: Try multiple search strategies
                
                # Strategy 1: Look for exact name match in NSE (any segment)
                index_match = self.instruments_df[
                    (self.instruments_df['name'] == index_name) &
                    (self.instruments_df['exchange'] == 'NSE')
                ]
                
                # Strategy 2: If not found, try tradingsymbol match
                if index_match.empty:
                    # Remove spaces and try
                    symbol_variant = index_name.replace(' ', '')
                    index_match = self.instruments_df[
                        (self.instruments_df['tradingsymbol'] == symbol_variant) &
                        (self.instruments_df['exchange'] == 'NSE')
                    ]
                
                # Strategy 3: Try with spaces in tradingsymbol
                if index_match.empty:
                    index_match = self.instruments_df[
                        (self.instruments_df['tradingsymbol'] == index_name) &
                        (self.instruments_df['exchange'] == 'NSE')
                    ]
                
                if not index_match.empty:
                    # Take the first match (usually the index itself, not futures/options)
                    first_match = index_match.iloc[0]
                    token = int(first_match['instrument_token'])
                    tradingsymbol = first_match['tradingsymbol']
                    segment = first_match['segment']
                    
                    self.index_token_map[index_name] = {
                        'token': token,
                        'tradingsymbol': tradingsymbol,
                        'name': index_name,
                        'exchange': 'NSE',
                        'segment': segment
                    }
                    
                    print(f"✅ Mapped: '{index_name}' → Token: {token}, Symbol: '{tradingsymbol}', Segment: '{segment}'")
                else:
                    print(f"⚠️ Could not find spot instrument for: '{index_name}'")
            
            print(f"\n✅ Built index token map for {len(self.index_token_map)} indices")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"❌ Error building index token map: {e}")
            import traceback
            traceback.print_exc()
    
    # ========================================================================
    # INDEX MANAGEMENT
    # ========================================================================
    
    def get_indices_by_exchange(self, exchange: str = "NSE") -> List[str]:
        """
        Get available indices from NFO-OPT underlying names
        Fully dynamic - extracts from loaded instruments
        """
        if self.instruments_df is None or self.instruments_df.empty:
            print("❌ instruments_df is None or empty")
            return []
        
        try:
            print(f"\n{'='*60}")
            print(f"SEARCHING FOR INDICES IN {exchange}")
            print(f"{'='*60}")
            print(f"Total instruments loaded: {len(self.instruments_df)}")
            
            # Show unique segments available
            unique_segments = self.instruments_df['segment'].unique()
            print(f"Available segments: {unique_segments}")
            
            # Count by segment
            segment_counts = self.instruments_df['segment'].value_counts()
            print(f"\nInstrument counts by segment:")
            for seg, count in segment_counts.items():
                print(f"  {seg}: {count}")
            
            # For NSE, look in NFO-OPT for underlying index names
            # This is where index options are listed
            nfo_options = self.instruments_df[
                self.instruments_df['segment'] == 'NFO-OPT'
            ]
            
            print(f"\nNFO-OPT instruments found: {len(nfo_options)}")
            
            if nfo_options.empty:
                print("❌ No NFO-OPT segment found - cannot extract indices")
                print("\nTrying alternative: Looking for 'OPT' in segment name...")
                
                # Fallback: look for any segment containing 'OPT'
                options_alt = self.instruments_df[
                    self.instruments_df['segment'].str.contains('OPT', case=False, na=False)
                ]
                
                if not options_alt.empty:
                    print(f"✅ Found {len(options_alt)} options using alternative method")
                    nfo_options = options_alt
                else:
                    return []
            
            # Get unique underlying names (these are the indices)
            index_names = nfo_options['name'].dropna().unique().tolist()
            
            print(f"\nRaw index names found: {len(index_names)}")
            if index_names:
                print(f"Sample names: {index_names[:10]}")
            
            # Sort and clean
            index_names = sorted(set(index_names))
            
            # Update cache
            from cache_utils import update_indices_cache
            update_indices_cache(exchange, index_names)
            
            print(f"\n✅ FINAL: Found {len(index_names)} indices for {exchange}")
            print(f"Indices: {index_names}")
            print(f"{'='*60}\n")
            
            return index_names
            
        except Exception as e:
            print(f"❌ Error getting indices: {e}")
            import traceback
            traceback.print_exc()
            return []

    
    def get_index_ltp(self, index_name: str, exchange: str = "NSE") -> Optional[float]:
        """
        Get Last Traded Price for an index - Fully dynamic with caching
        """
        if not self.connected:
            return None
        
        if self.instruments_df is None or self.instruments_df.empty:
            return None
        
        try:
            # Check cache first
            if index_name in self.index_symbol_cache:
                cached_symbol = self.index_symbol_cache[index_name]
                try:
                    quote_data = self.kite.quote([cached_symbol])
                    if cached_symbol in quote_data and 'last_price' in quote_data[cached_symbol]:
                        return quote_data[cached_symbol]['last_price']
                except:
                    # Cache invalid, remove it
                    del self.index_symbol_cache[index_name]
            
            # Search in instruments
            matches = self.instruments_df[
                (self.instruments_df['name'] == index_name) |
                (self.instruments_df['tradingsymbol'].str.contains(index_name, case=False, na=False))
            ]
            
            if matches.empty:
                print(f"⚠️ No instrument found for: {index_name}")
                return None
            
            print(f"🔍 Found {len(matches)} matches for {index_name}")
            
            # Try each match
            for idx, row in matches.iterrows():
                try:
                    symbol = f"{row['exchange']}:{row['tradingsymbol']}"
                    quote_data = self.kite.quote([symbol])
                    
                    if symbol in quote_data and 'last_price' in quote_data[symbol]:
                        price = quote_data[symbol]['last_price']
                        
                        # Cache the working symbol for future use
                        self.index_symbol_cache[index_name] = symbol
                        
                        print(f"✅ Found price for {symbol}: {price}")
                        return price
                        
                except Exception as e:
                    continue
            
            print(f"❌ Could not fetch price for {index_name}")
            return None
            
        except Exception as e:
            print(f"❌ Error fetching index LTP: {e}")
            return None
    
    # ========================================================================
    # OPTIONS CHAIN
    # ========================================================================
    
    def get_option_chain(
        self,
        index_symbol: str,
        expiry_date: str = None
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[List[str]]]:
        """Get options chain for an index - with duplicate removal"""
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
            
            # ✅ NEW: Remove duplicates by tradingsymbol
            # This ensures each option contract appears only once
            initial_count = len(options)
            options = options.drop_duplicates(subset=['tradingsymbol'], keep='first')
            removed_count = initial_count - len(options)
            if removed_count > 0:
                print(f"🧹 Removed {removed_count} duplicate option contracts")
            
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
            
            # ✅ NEW: Additional deduplication by strike+expiry
            # Safety measure to ensure unique strike-expiry combinations
            calls_initial = len(calls)
            calls = calls.drop_duplicates(subset=['strike', 'expiry'], keep='first')
            if len(calls) < calls_initial:
                print(f"🧹 Removed {calls_initial - len(calls)} duplicate call strikes")
            
            puts_initial = len(puts)
            puts = puts.drop_duplicates(subset=['strike', 'expiry'], keep='first')
            if len(puts) < puts_initial:
                print(f"🧹 Removed {puts_initial - len(puts)} duplicate put strikes")
            
            # Sort by expiry and strike
            calls = calls.sort_values(['expiry', 'strike'])
            puts = puts.sort_values(['expiry', 'strike'])
            
            print(f"✅ {len(calls)} unique Calls, {len(puts)} unique Puts across {len(all_expiries)} expiries")
            
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
    
    def get_ohlc(self, symbols: List[str], exchange: str = "NSE") -> Dict:
        """Get OHLC data"""
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
    # HISTORICAL DATA
    # ========================================================================
    
    def get_historical_data(
        self,
        instrument_token: int,
        from_date: datetime,
        to_date: datetime,
        interval: str = "minute"
    ) -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV data"""
        if not self.connected:
            return None
        
        try:
            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            
            if not data:
                return None
            
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
        """Fetch historical data using symbol"""
        instrument_token = self.get_instrument_token(symbol, exchange)
        
        if not instrument_token:
            print(f"❌ Could not find instrument token for {symbol}")
            return None
        
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        return self.get_historical_data(
            instrument_token=instrument_token,
            from_date=from_date,
            to_date=to_date,
            interval=interval
        )
    
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
        """
        Get instrument token for symbol
        Enhanced to handle symbol variations (spaces, underscores, name vs tradingsymbol)
        """
        if self.instruments_df is None or self.instruments_df.empty:
            return None
        
        # Try exact match on tradingsymbol first
        match = self.instruments_df[
            (self.instruments_df['tradingsymbol'] == symbol) &
            (self.instruments_df['exchange'] == exchange)
        ]
        
        if not match.empty:
            return int(match.iloc[0]['instrument_token'])
        
        # Try matching by name (important for indices like "NIFTY 50")
        match = self.instruments_df[
            (self.instruments_df['name'] == symbol) &
            (self.instruments_df['exchange'] == exchange)
        ]
        
        if not match.empty:
            return int(match.iloc[0]['instrument_token'])
        
        # Try symbol variations (handle different formats)
        symbol_variations = [
            symbol,
            symbol.replace(' ', ''),      # "NIFTY 50" → "NIFTY50"
            symbol.replace(' ', '_'),     # "NIFTY 50" → "NIFTY_50"
            symbol.replace('_', ''),      # "NIFTY_50" → "NIFTY50"
            symbol.replace('_', ' '),     # "NIFTY_50" → "NIFTY 50"
            symbol.split()[0] if ' ' in symbol else symbol  # "NIFTY 50" → "NIFTY"
        ]
        
        # Remove duplicates while preserving order
        symbol_variations = list(dict.fromkeys(symbol_variations))
        
        # Try each variation
        for var in symbol_variations:
            # Try tradingsymbol
            match = self.instruments_df[
                (self.instruments_df['tradingsymbol'] == var) &
                (self.instruments_df['exchange'] == exchange)
            ]
            
            if not match.empty:
                print(f"✅ Found token using variation: '{var}' for '{symbol}'")
                return int(match.iloc[0]['instrument_token'])
            
            # Try name
            match = self.instruments_df[
                (self.instruments_df['name'] == var) &
                (self.instruments_df['exchange'] == exchange)
            ]
            
            if not match.empty:
                print(f"✅ Found token using name variation: '{var}' for '{symbol}'")
                return int(match.iloc[0]['instrument_token'])
        
        print(f"⚠️ Could not find instrument token for: {symbol} on {exchange}")
        print(f"   Tried variations: {symbol_variations}")
        return None
    
    def get_index_instrument_token(self, index_name: str) -> Optional[int]:
        """
        Get instrument token specifically for indices - fully dynamic
        Uses pre-built index token map for fast lookups
        """
        # Check if we have the index token map
        if not hasattr(self, 'index_token_map'):
            print("⚠️ Index token map not built, building now...")
            self._build_index_token_map()
        
        # Direct lookup in map
        if index_name in self.index_token_map:
            token = self.index_token_map[index_name]['token']
            print(f"✅ Found index token from map: '{index_name}' → {token}")
            return token
        
        # Fallback: Try standard get_instrument_token
        print(f"⚠️ '{index_name}' not in index map, trying standard lookup...")
        return self.get_instrument_token(index_name, 'NSE')

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
    """
    Initialize Kite and fetch instruments - Fully Dynamic
    Fetches ALL available exchanges from Kite API
    """
    handler = get_kite_handler()
    success, message = handler.initialize()
    
    if not success:
        return success, message
    
    try:
        print("📥 Fetching available exchanges...")
        
        # Get available exchanges dynamically
        # Kite Connect provides instruments() without exchange parameter to get all
        # Or we can try common exchanges and see which ones work
        available_exchanges = ["NFO", "NSE", "BSE", "BFO", "MCX", "CDS"]
        
        loaded_exchanges = []
        
        for exchange in available_exchanges:
            try:
                print(f"📥 Attempting to fetch {exchange} instruments...")
                if handler.fetch_and_cache_instruments(exchange):
                    loaded_exchanges.append(exchange)
                    print(f"✅ Loaded {exchange}")
                else:
                    print(f"⚠️  Skipped {exchange} (no data or error)")
            except Exception as e:
                print(f"⚠️  Could not load {exchange}: {str(e)}")
                continue
        
        if loaded_exchanges:
            print(f"✅ Successfully loaded instruments from: {', '.join(loaded_exchanges)}")
            return True, f"✅ Connected and loaded {len(loaded_exchanges)} exchanges"
        else:
            return False, "❌ No instruments could be loaded from any exchange"
    
    except Exception as e:
        print(f"❌ Error during instrument loading: {e}")
        import traceback
        traceback.print_exc()
        return False, f"❌ Failed to load instruments: {str(e)}"
