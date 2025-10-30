"""
Unified Data Handler - Multi-Market Orchestration Layer
Routes data requests to appropriate market-specific handlers
Provides consistent interface for all asset classes
"""

import pandas as pd
from pathlib import Path
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# ✅ ADD THIS: Timeframe normalization mapping
TIMEFRAME_MAPPING = {
    # Normalize to Alpaca format (1min, 5min, 15min, 30min, 1h, 4h, 1d)
    '1m': '1min',
    '5m': '5min',      # ← YOUR BUG IS HERE!
    '15m': '15min',
    '30m': '30min',
    '1h': '1h',
    '4h': '4h',
    '1d': '1d',
    '1min': '1min',
    '5min': '5min',
    '15min': '15min',
    '30min': '30min',
    'D': '1d',
    'd': '1d',
    'day': '1d',
    'daily': '1d',
    'h': '1h',
    'hour': '1h',
    'w': '1w',
    'week': '1w',
    'm': '1mo',
    'month': '1mo',
}

def normalize_timeframe(tf: str) -> str:
    """
    ✅ Normalize timeframe to Alpaca format
    Converts: '5m' → '5min', etc.
    """
    normalized = TIMEFRAME_MAPPING.get(tf, tf)
    if normalized != tf:
        logger.debug(f"🔄 Normalized timeframe: {tf} → {normalized}")
    return normalized

class UnifiedDataHandler:
    """
    Unified interface for multi-asset trading data
    
    Supported Markets:
    - Indian Markets (NSE/BSE) via Kite Connect
    - US Stocks via Alpaca
    - Cryptocurrencies via Alpaca & CCXT
    - (Forex via OANDA)
    """
    
    # Market type constants
    MARKET_INDIAN = "Indian Markets"
    MARKET_US_STOCKS = "US Stocks"
    MARKET_CRYPTO_ALPACA = "Cryptocurrency (Alpaca)"
    MARKET_CRYPTO_BINANCE = "Cryptocurrency (Binance)"
    MARKET_FOREX = "Forex (OANDA)"
    
    def __init__(self, market_type: str):
        """
        Initialize unified handler for specified market
        
        Args:
            market_type: Market type constant (MARKET_INDIAN, MARKET_US_STOCKS, etc.)
        """
        self.market_type = market_type
        self.handler = None
        self.connected = False

        # ✅ ADD CACHE DIRECTORY
        self.cache_dir = Path('backtesting/results/data_cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Cache directory: {self.cache_dir}")
        
        # Initialize appropriate handler
        self._initialize_handler()
    
    def _initialize_handler(self):
        """Initialize the appropriate market handler based on market type"""
        try:
            if self.market_type == self.MARKET_INDIAN:
                from kite_handler import get_kite_handler
                self.handler = get_kite_handler()
                logger.info("✅ Initialized Kite handler for Indian Markets")
            
            elif self.market_type == self.MARKET_US_STOCKS:
                from alpaca_handler import get_alpaca_handler
                self.handler = get_alpaca_handler(mode='paper')
                logger.info("✅ Initialized Alpaca handler for US Stocks")
            
            elif self.market_type == self.MARKET_CRYPTO_ALPACA:
                from alpaca_handler import get_alpaca_handler
                self.handler = get_alpaca_handler(mode='paper')
                logger.info("✅ Initialized Alpaca handler for Cryptocurrency")
            
            elif self.market_type == self.MARKET_CRYPTO_BINANCE:
                from crypto_handler import get_crypto_handler
                self.handler = get_crypto_handler(exchange='binance')
                logger.info("✅ Initialized CCXT handler for Cryptocurrency (Binance)")

            elif self.market_type == self.MARKET_FOREX:  # ✅ NEW BLOCK
                from forex_handler import get_forex_handler
                self.handler = get_forex_handler(account_type='practice')
                logger.info("✅ Initialized OANDA handler for Forex")
            
            else:
                raise ValueError(f"Unknown market type: {self.market_type}")
            
            self.connected = True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize handler for {self.market_type}: {e}")
            self.connected = False
            raise
    
    def get_available_symbols_by_category(self, category: str = None) -> List[Dict]:
        """
        Get available symbols for the selected market
        
        Args:
            category: Optional category filter (market-specific)
        
        Returns:
            List of symbol dictionaries with metadata
        """
        if not self.connected:
            raise Exception(f"Handler not connected for {self.market_type}")
        
        try:
            return self.handler.get_available_symbols_by_category(category)
        except Exception as e:
            logger.error(f"Failed to get symbols: {e}")
            return []
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """
        Get detailed information about a symbol
        
        Args:
            symbol: Symbol name (format depends on market)
        
        Returns:
            Dictionary with symbol metadata or None
        """
        if not self.connected:
            raise Exception(f"Handler not connected for {self.market_type}")
        
        try:
            return self.handler.get_symbol_info(symbol)
        except Exception as e:
            logger.error(f"Failed to get symbol info for {symbol}: {e}")
            return None
    
    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '5min',
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical OHLC data for backtesting (with caching)
        
        Args:
            symbol: Trading symbol (format depends on market)
            start_date: Start date for data
            end_date: End date for data
            timeframe: Timeframe string ('5min', '1h', 'day', etc.)
            use_cache: Whether to use cache (False for live trading)
        
        Returns:
            Pandas DataFrame with OHLC data in standardized format
        """
        if not self.connected:
            raise Exception(f"Handler not connected for {self.market_type}")
        
        try:
            # ✅ STEP 0: Normalize timeframe (FIX FOR EUR/USD ERROR)
            timeframe = normalize_timeframe(timeframe)  # ← ADD THIS!
            logger.debug(f"Using normalized timeframe: {timeframe}")
            
            # ✅ STEP 1: Normalize symbol format for Forex (EUR/USD -> EUR_USD)
            api_symbol = symbol
            if self.market_type == self.MARKET_FOREX:
                api_symbol = symbol.replace('/', '_')
                logger.info(f"🔄 Normalized Forex symbol: {symbol} -> {api_symbol}")
            
            # ✅ STEP 2: BYPASS CACHE IF use_cache=False (for live trading)
            if not use_cache:
                logger.info(f"🌐 Fetching LIVE data for {symbol} (cache bypassed)")
                df = self.handler.get_historical_data(
                    symbol=api_symbol,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe=timeframe,
                    use_cache=False
                )
                
                if df.empty:
                    logger.warning(f"No data returned for {symbol}")
                else:
                    logger.info(f"✅ Retrieved {len(df)} bars for {symbol}")
                
                return df
            
            # ✅ STEP 3: Generate cache key (for backtesting)
            cache_key = self._generate_cache_key(symbol, start_date, end_date, timeframe)
            cache_file = self.cache_dir / f"{cache_key}.parquet"
            
            # ✅ STEP 4: Check cache first
            if cache_file.exists():
                try:
                    logger.info(f"📁 Loading from cache: {cache_file.name}")
                    df = pd.read_parquet(cache_file)
                    logger.info(f"✅ Retrieved {len(df)} bars from cache for {symbol}")
                    return df
                except Exception as e:
                    logger.warning(f"Cache read failed: {e}. Fetching from API...")
            
            # ✅ STEP 5: Fetch from API (cache miss)
            logger.info(f"🌐 Fetching {symbol} data from {start_date} to {end_date} ({timeframe})")
            
            df = self.handler.get_historical_data(
                symbol=api_symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe,
                use_cache=False
            )
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
            else:
                logger.info(f"✅ Retrieved {len(df)} bars for {symbol}")
                
                # ✅ STEP 6: Save to cache
                try:
                    df.to_parquet(cache_file)
                    logger.info(f"💾 Saved to cache: {cache_file.name}")
                except Exception as e:
                    logger.warning(f"Cache save failed: {e}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _generate_cache_key(self, symbol: str, start_date: datetime, end_date: datetime, timeframe: str) -> str:
        """
        Generate unique cache key for data request
        
        Returns:
            String like: BTCUSDT_20240101_20241231_5min_hash
        """
        # Clean symbol (replace / with _)
        clean_symbol = symbol.replace('/', '_')
        
        # Format dates
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        
        # Create base key
        base_key = f"{clean_symbol}_{start_str}_{end_str}_{timeframe}"
        
        # Add market type hash (to separate crypto/stocks with same symbols)
        market_hash = hashlib.md5(self.market_type.encode()).hexdigest()[:8]
        
        return f"{base_key}_{market_hash}"
    
    def clear_cache(self, symbol: str = None):
        """
        Clear cache for specific symbol or all symbols
        
        Args:
            symbol: Optional symbol to clear (clears all if None)
        """
        if symbol:
            # Clear specific symbol
            pattern = f"{symbol.replace('/', '_')}_*.parquet"
            for cache_file in self.cache_dir.glob(pattern):
                cache_file.unlink()
                logger.info(f"🗑️ Cleared cache: {cache_file.name}")
        else:
            # Clear all cache
            for cache_file in self.cache_dir.glob("*.parquet"):
                cache_file.unlink()
            logger.info(f"🗑️ Cleared all cache files")
    
    def get_market_categories(self) -> List[str]:
        """
        Get available symbol categories for current market
        Filter categories based on market type
        
        Returns:
            List of category names
        """
        if not self.connected:
            return []
        
        try:
            # Check if handler has available_symbols dictionary
            if hasattr(self.handler, 'available_symbols') and isinstance(self.handler.available_symbols, dict):
                all_categories = list(self.handler.available_symbols.keys())
                
                # Filter categories based on market type
                if self.market_type == self.MARKET_US_STOCKS:
                    # US Stocks: Show only stock-related categories
                    allowed = ['Popular Stocks', 'Tech Stocks', 'ETFs']
                    categories = [cat for cat in all_categories if cat in allowed]
                    
                elif self.market_type == self.MARKET_CRYPTO_ALPACA:
                    # Crypto: Show only crypto categories
                    allowed = ['Cryptocurrencies']
                    categories = [cat for cat in all_categories if cat in allowed]
                    
                else:
                    # Other markets: Show all categories
                    categories = all_categories
                
                # Filter out empty categories
                categories = [cat for cat in categories if self.handler.available_symbols.get(cat)]
                return categories if categories else ['All']
            
            # Fallback: Get all symbols and extract categories
            all_symbols = self.handler.get_available_symbols_by_category()
            if all_symbols and len(all_symbols) > 0:
                if isinstance(all_symbols[0], dict):
                    # Extract unique categories
                    categories = list(set([s.get('category', 'All') for s in all_symbols if s.get('category')]))
                    return sorted(categories) if categories else ['All']
            
            return ['All']
            
        except Exception as e:
            logger.error(f"Failed to get categories: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ['All']

def get_unified_handler(market_type: str) -> UnifiedDataHandler:
    """
    Get or create UnifiedDataHandler instance (singleton pattern per market)
    
    Args:
        market_type: Market type constant
    
    Returns:
        UnifiedDataHandler instance
    """
    if not hasattr(get_unified_handler, '_instances'):
        get_unified_handler._instances = {}
    
    if market_type not in get_unified_handler._instances:
        get_unified_handler._instances[market_type] = UnifiedDataHandler(market_type)
    
    return get_unified_handler._instances[market_type]


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_all_market_types() -> List[str]:
    """Get list of all supported market types"""
    return [
        UnifiedDataHandler.MARKET_INDIAN,
        UnifiedDataHandler.MARKET_US_STOCKS,
        UnifiedDataHandler.MARKET_CRYPTO_ALPACA,
        UnifiedDataHandler.MARKET_CRYPTO_BINANCE,
        UnifiedDataHandler.MARKET_FOREX
    ]


def get_market_display_info() -> Dict[str, Dict]:
    """
    Get display information for each market type
    Useful for UI rendering
    """
    return {
        UnifiedDataHandler.MARKET_INDIAN: {
            'name': 'Indian Markets',
            'description': 'NSE & BSE stocks and options',
            'icon': '🇮🇳',
            'provider': 'Kite Connect',
            'assets': ['Stocks', 'Options', 'Indices']
        },
        UnifiedDataHandler.MARKET_US_STOCKS: {
            'name': 'US Stocks',
            'description': 'NYSE & NASDAQ stocks',
            'icon': '🇺🇸',
            'provider': 'Alpaca',
            'assets': ['Stocks', 'ETFs']
        },
        UnifiedDataHandler.MARKET_CRYPTO_ALPACA: {
            'name': 'Crypto (Alpaca)',
            'description': 'Major cryptocurrencies',
            'icon': '₿',
            'provider': 'Alpaca',
            'assets': ['Bitcoin', 'Ethereum', 'Altcoins']
        },
        UnifiedDataHandler.MARKET_CRYPTO_BINANCE: {
            'name': 'Cryptocurrency',
            'description': '100+ cryptocurrencies (Auto-selects best exchange)',
            'icon': '🪙',
            'provider': 'Multi-Exchange via CCXT',
            'assets': ['Bitcoin', 'Ethereum', 'DeFi', 'Altcoins']
        },
        UnifiedDataHandler.MARKET_FOREX: {  # ✅ NEW BLOCK
            'name': 'Forex',
            'description': '28+ major currency pairs & commodities',
            'icon': '💱',
            'provider': 'OANDA',
            'assets': ['EUR/USD', 'GBP/USD', 'USD/JPY', 'Gold', 'Oil']
        }    
    }
