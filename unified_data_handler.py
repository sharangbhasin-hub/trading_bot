"""
Unified Data Handler - Multi-Market Orchestration Layer
Routes data requests to appropriate market-specific handlers
Provides consistent interface for all asset classes
"""

import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class UnifiedDataHandler:
    """
    Unified interface for multi-asset trading data
    
    Supported Markets:
    - Indian Markets (NSE/BSE) via Kite Connect
    - US Stocks via Alpaca
    - Cryptocurrencies via Alpaca & CCXT
    - (Future: Forex via OANDA)
    """
    
    # Market type constants
    MARKET_INDIAN = "Indian Markets"
    MARKET_US_STOCKS = "US Stocks"
    MARKET_CRYPTO_ALPACA = "Cryptocurrency (Alpaca)"
    MARKET_CRYPTO_BINANCE = "Cryptocurrency (Binance)"
    
    def __init__(self, market_type: str):
        """
        Initialize unified handler for specified market
        
        Args:
            market_type: Market type constant (MARKET_INDIAN, MARKET_US_STOCKS, etc.)
        """
        self.market_type = market_type
        self.handler = None
        self.connected = False
        
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
        timeframe: str = '5min'
    ) -> pd.DataFrame:
        """
        Fetch historical OHLC data for backtesting
        
        Args:
            symbol: Trading symbol (format depends on market)
            start_date: Start date for data
            end_date: End date for data
            timeframe: Timeframe string ('5min', '1h', 'day', etc.)
        
        Returns:
            Pandas DataFrame with OHLC data in standardized format
        """
        if not self.connected:
            raise Exception(f"Handler not connected for {self.market_type}")
        
        try:
            logger.info(f"Fetching {symbol} data from {start_date} to {end_date} ({timeframe})")
            
            df = self.handler.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe
            )
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
            else:
                logger.info(f"✅ Retrieved {len(df)} bars for {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_market_categories(self) -> List[str]:
        """
        Get available symbol categories for current market
        
        Returns:
            List of category names
        """
        if not self.connected:
            return []
        
        try:
            # Get all symbols by category
            all_symbols = self.handler.get_available_symbols_by_category()
            
            # Extract unique categories
            if all_symbols and len(all_symbols) > 0:
                # Assuming symbols have 'category' key
                categories = list(set([s.get('category', 'All') for s in all_symbols if s.get('category')]))
                return sorted(categories)
            
            # If handler has available_symbols attribute with categories
            if hasattr(self.handler, 'available_symbols'):
                return list(self.handler.available_symbols.keys())
            
            return ['All']
            
        except Exception as e:
            logger.error(f"Failed to get categories: {e}")
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
        UnifiedDataHandler.MARKET_CRYPTO_BINANCE
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
            'name': 'Crypto (Binance)',
            'description': '100+ cryptocurrencies',
            'icon': '🪙',
            'provider': 'Binance via CCXT',
            'assets': ['Bitcoin', 'Ethereum', 'DeFi', 'Altcoins']
        }
    }
