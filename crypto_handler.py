"""
Cryptocurrency Handler - CCXT Integration
Fetches crypto data from Binance, Coinbase, Bybit, etc.
Cloud-compatible, works on Streamlit Cloud
100% free - no account needed for historical data
"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# TIMEFRAME MAPPING (CCXT format)
# ============================================================================
TIMEFRAME_MAP = {
    '1min': '1m',
    '5min': '5m',
    '15min': '15m',
    '30min': '30m',
    '1h': '1h',
    'hour': '1h',
    '4h': '4h',
    'day': '1d',
    'daily': '1d',
    'week': '1w',
    'month': '1M'
}

class CryptoHandler:
    """
    CCXT Cryptocurrency Handler for backtesting
    
    Features:
    - Free historical data (no API keys needed)
    - Supports 100+ cryptocurrencies
    - Multiple exchange support
    - Cloud-compatible
    """
    
    def __init__(self, exchange_id='binance'):
        """
        Initialize CCXT exchange connection
        
        Args:
            exchange_id: Exchange name ('binance', 'coinbase', 'bybit', 'kraken')
        """
        self.exchange_id = exchange_id
        self.exchange = None
        self.connected = False
        self.available_symbols = {}
        
        # Initialize exchange
        self._initialize_exchange()
    
    def _initialize_exchange(self):
        """Initialize CCXT exchange connection"""
        try:
            # Create exchange instance
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class({
                'enableRateLimit': True,  # Respect API rate limits
                'timeout': 30000,  # 30 second timeout
            })
            
            logger.info(f"✅ {self.exchange_id.title()} exchange initialized")
            
            # Load markets (symbols)
            self.exchange.load_markets()
            self.connected = True
            
            logger.info(f"✅ Loaded {len(self.exchange.markets)} markets from {self.exchange_id}")
            
            # Categorize available symbols
            self._load_available_symbols()
            
        except Exception as e:
            logger.error(f"❌ Exchange initialization failed: {e}")
            self.connected = False
            raise
    
    def _load_available_symbols(self):
        """
        Dynamically load and categorize all tradeable symbols
        NO HARDCODED LISTS - fetches from API
        """
        try:
            markets = self.exchange.markets
            
            # Categorize symbols
            spot_usdt = []
            spot_btc = []
            defi_coins = []
            top_coins = []
            meme_coins = []
            
            # Reference lists for categorization
            top_by_mcap = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOT', 
                          'MATIC', 'LINK', 'UNI', 'ATOM', 'LTC', 'BCH', 'XLM',
                          'ALGO', 'VET', 'FIL', 'TRX', 'ETC', 'THETA', 'XMR']
            
            defi_tokens = ['AAVE', 'UNI', 'COMP', 'MKR', 'SNX', 'YFI', 'SUSHI', 
                          'CRV', 'BAL', '1INCH', 'CAKE', 'LDO']
            
            meme_tokens = ['DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BONK']
            
            for symbol, market in markets.items():
                # Skip if not active or not spot market
                if not market.get('active', False) or not market.get('spot', False):
                    continue
                
                symbol_parts = symbol.split('/')
                if len(symbol_parts) != 2:
                    continue
                
                base, quote = symbol_parts
                
                asset_dict = {
                    'symbol': symbol,
                    'base': base,
                    'quote': quote,
                    'description': f"{base} vs {quote}"
                }
                
                # USDT pairs (most liquid)
                if quote == 'USDT':
                    spot_usdt.append(asset_dict)
                    
                    # Check if top coin
                    if base in top_by_mcap:
                        top_coins.append(asset_dict)
                    
                    # Check if DeFi
                    if base in defi_tokens:
                        defi_coins.append(asset_dict)
                    
                    # Check if meme coin
                    if base in meme_tokens:
                        meme_coins.append(asset_dict)
                
                # BTC pairs
                elif quote == 'BTC':
                    spot_btc.append(asset_dict)
            
            # Sort and limit
            self.available_symbols = {
                'Top Cryptocurrencies': sorted(
                    [c for c in top_coins],
                    key=lambda x: top_by_mcap.index(x['base']) if x['base'] in top_by_mcap else 999
                )[:20],
                'DeFi Tokens': sorted(defi_coins, key=lambda x: x['symbol'])[:15],
                'Meme Coins': sorted(meme_coins, key=lambda x: x['symbol'])[:10],
                'Spot USDT Pairs': sorted(spot_usdt, key=lambda x: x['symbol'])[:50],
                'Spot BTC Pairs': sorted(spot_btc, key=lambda x: x['symbol'])[:30]
            }
            
            # Log summary
            total = sum(len(v) for v in self.available_symbols.values())
            logger.info(f"✅ Categorized {total} trading pairs:")
            for category, symbols in self.available_symbols.items():
                if symbols:
                    logger.info(f"  - {category}: {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Failed to load symbols: {e}")
            self.available_symbols = {}
    
    def get_available_symbols_by_category(self, category: str = None) -> List[Dict]:
        """
        Get available symbols, optionally filtered by category
        
        Args:
            category: 'Top Cryptocurrencies', 'DeFi Tokens', etc.
        
        Returns:
            List of symbol dictionaries with metadata
        """
        if not self.available_symbols:
            self._load_available_symbols()
        
        if category:
            return self.available_symbols.get(category, [])
        else:
            # Return all symbols flattened
            all_symbols = []
            for symbols_list in self.available_symbols.values():
                all_symbols.extend(symbols_list)
            return all_symbols
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """
        Get detailed information about a symbol
        
        Args:
            symbol: Symbol name (e.g., 'BTC/USDT')
        
        Returns:
            Dictionary with symbol metadata or None
        """
        try:
            market = self.exchange.market(symbol)
            
            return {
                'symbol': symbol,
                'base': market['base'],
                'quote': market['quote'],
                'active': market['active'],
                'spot': market.get('spot', False),
                'min_amount': market['limits']['amount']['min'],
                'max_amount': market['limits']['amount']['max'],
                'min_price': market['limits']['price']['min'],
                'max_price': market['limits']['price']['max'],
                'precision_amount': market['precision']['amount'],
                'precision_price': market['precision']['price'],
                'exchange': self.exchange_id
            }
        except Exception as e:
            logger.warning(f"Symbol {symbol} not found: {e}")
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
            symbol: Trading symbol (e.g., 'BTC/USDT')
            start_date: Start date for data
            end_date: End date for data
            timeframe: Timeframe string ('5min', '1h', 'day', etc.)
        
        Returns:
            Pandas DataFrame with OHLC data (same format as Kite/Alpaca)
        """
        if not self.connected:
            raise Exception(f"{self.exchange_id} not connected")
        
        # Map timeframe string to CCXT format
        ccxt_timeframe = TIMEFRAME_MAP.get(timeframe.lower())
        if ccxt_timeframe is None:
            raise ValueError(f"Invalid timeframe: {timeframe}. Valid options: {list(TIMEFRAME_MAP.keys())}")
        
        try:
            # Convert dates to milliseconds (CCXT format)
            since = int(start_date.timestamp() * 1000)
            end_ms = int(end_date.timestamp() * 1000)
            
            all_candles = []
            limit = 1000  # Max candles per request (exchange dependent)
            
            # Fetch data in chunks
            while since < end_ms:
                try:
                    candles = self.exchange.fetch_ohlcv(
                        symbol,
                        ccxt_timeframe,
                        since,
                        limit
                    )
                    
                    if not candles or len(candles) == 0:
                        break
                    
                    all_candles.extend(candles)
                    
                    # Update 'since' to last candle timestamp + 1ms
                    since = candles[-1][0] + 1
                    
                    # Stop if we've reached end_date
                    if candles[-1][0] >= end_ms:
                        break
                    
                    # Respect rate limits
                    if self.exchange.rateLimit:
                        import time
                        time.sleep(self.exchange.rateLimit / 1000)
                    
                except Exception as e:
                    logger.warning(f"Error fetching batch: {e}")
                    break
            
            if not all_candles:
                logger.warning(f"No data received for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Filter to exact date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            # Keep only OHLCV columns (standard format)
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            logger.info(f"✅ Fetched {len(df)} bars for {symbol} ({timeframe}) from {self.exchange_id}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_live_ticker(self, symbol: str) -> Optional[Dict]:
        """
        Get current live ticker data
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
        
        Returns:
            Dictionary with current prices or None
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            
            return {
                'symbol': symbol,
                'bid': ticker.get('bid'),
                'ask': ticker.get('ask'),
                'last': ticker['last'],
                'high': ticker.get('high'),
                'low': ticker.get('low'),
                'volume': ticker.get('quoteVolume', ticker.get('volume')),
                'timestamp': datetime.fromtimestamp(ticker['timestamp'] / 1000) if ticker.get('timestamp') else datetime.now(),
                'change_24h': ticker.get('percentage', 0),
                'exchange': self.exchange_id
            }
        except Exception as e:
            logger.warning(f"No ticker data for {symbol}: {e}")
            return None


# ============================================================================
# MODULE-LEVEL FUNCTIONS
# ============================================================================

def get_crypto_handler(exchange='binance') -> CryptoHandler:
    """
    Get or create CryptoHandler instance (singleton pattern)
    
    Args:
        exchange: Exchange name ('binance', 'coinbase', 'bybit', 'kraken')
    
    Returns:
        CryptoHandler instance
    """
    if not hasattr(get_crypto_handler, '_instance'):
        get_crypto_handler._instance = {}
    
    if exchange not in get_crypto_handler._instance:
        get_crypto_handler._instance[exchange] = CryptoHandler(exchange)
    
    return get_crypto_handler._instance[exchange]
