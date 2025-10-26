"""
Cryptocurrency Handler - CCXT Integration
Cloud-compatible, works on Streamlit Cloud
Handles Bitcoin, Ethereum, and all major cryptocurrencies
100% dynamic - fetches symbols from exchange APIs
"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import os
from dotenv import load_dotenv

load_dotenv()
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
    CCXT Cryptocurrency Handler for backtesting and paper trading
    Cloud-compatible - no local installation required
    
    Features:
    - Dynamic symbol loading from Binance API
    - Historical data for backtesting
    - Live tick data for paper trading
    - Supports 100+ cryptocurrencies
    """
    
    def __init__(self, exchange_id='binance', mode='backtest'):
        """
        Initialize CCXT exchange connection
        
        Args:
            exchange_id: Exchange name ('binance', 'coinbase', 'bybit')
            mode: 'backtest' for historical data, 'paper' for live trading
        """
        self.exchange_id = exchange_id
        self.mode = mode
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
            
            logger.info(f"✅ Loaded {len(self.exchange.markets)} markets")
            
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
            # Get all markets
            markets = self.exchange.markets
            
            # Categorize symbols
            spot_usdt = []
            spot_btc = []
            futures = []
            defi_coins = []
            top_coins = []
            
            # Top coins by market cap (known list for categorization)
            top_100 = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOT', 
                      'MATIC', 'LINK', 'UNI', 'ATOM', 'LTC', 'BCH', 'XLM',
                      'ALGO', 'VET', 'FIL', 'TRX', 'ETC', 'THETA', 'XMR',
                      'AAVE', 'COMP', 'MKR', 'SNX', 'YFI', 'SUSHI', 'CRV']
            
            for symbol, market in markets.items():
                # Skip if not active
                if not market.get('active', False):
                    continue
                
                symbol_parts = symbol.split('/')
                if len(symbol_parts) != 2:
                    continue
                
                base, quote = symbol_parts
                
                # USDT pairs (most liquid)
                if quote == 'USDT' and market['spot']:
                    spot_usdt.append({
                        'symbol': symbol,
                        'base': base,
                        'quote': quote,
                        'description': f"{base} vs Tether",
                        'category': 'Spot USDT'
                    })
                    
                    # Check if top coin
                    if base in top_100:
                        top_coins.append({
                            'symbol': symbol,
                            'base': base,
                            'quote': quote,
                            'description': f"{base} vs Tether (Top 100)",
                            'category': 'Top Cryptocurrencies'
                        })
                
                # BTC pairs
                elif quote == 'BTC' and market['spot']:
                    spot_btc.append({
                        'symbol': symbol,
                        'base': base,
                        'quote': quote,
                        'description': f"{base} vs Bitcoin",
                        'category': 'Spot BTC'
                    })
                
                # Futures/Perpetual
                elif market.get('future', False) or market.get('swap', False):
                    futures.append({
                        'symbol': symbol,
                        'base': base,
                        'quote': quote,
                        'description': f"{base} Perpetual Futures",
                        'category': 'Futures'
                    })
                
                # DeFi tokens
                if base in ['AAVE', 'UNI', 'COMP', 'MKR', 'SNX', 'YFI', 'SUSHI', 'CRV', 'LINK']:
                    if quote == 'USDT':
                        defi_coins.append({
                            'symbol': symbol,
                            'base': base,
                            'quote': quote,
                            'description': f"{base} DeFi Token",
                            'category': 'DeFi Tokens'
                        })
            
            # Store categorized symbols
            self.available_symbols = {
                'Top Cryptocurrencies': sorted(top_coins, key=lambda x: top_100.index(x['base']) if x['base'] in top_100 else 999)[:20],  # Top 20
                'Spot USDT': sorted(spot_usdt, key=lambda x: x['symbol'])[:50],  # Top 50 by name
                'DeFi Tokens': sorted(defi_coins, key=lambda x: x['symbol']),
                'Spot BTC': sorted(spot_btc, key=lambda x: x['symbol'])[:30],  # Top 30
                'Futures': sorted(futures, key=lambda x: x['symbol'])[:20]  # Top 20
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
            category: 'Top Cryptocurrencies', 'Spot USDT', 'DeFi Tokens', etc.
        
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
                'future': market.get('future', False),
                'min_amount': market['limits']['amount']['min'],
                'max_amount': market['limits']['amount']['max'],
                'min_price': market['limits']['price']['min'],
                'max_price': market['limits']['price']['max'],
                'precision_amount': market['precision']['amount'],
                'precision_price': market['precision']['price']
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
            Pandas DataFrame with OHLC data (same format as Kite)
        """
        if not self.connected:
            raise Exception(f"{self.exchange_id} not connected")
        
        # Map timeframe string to CCXT format
        ccxt_timeframe = TIMEFRAME_MAP.get(timeframe.lower())
        if ccxt_timeframe is None:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
        try:
            # Convert dates to milliseconds (CCXT format)
            since = int(start_date.timestamp() * 1000)
            end_ms = int(end_date.timestamp() * 1000)
            
            all_candles = []
            limit = 1000  # Max candles per request
            
            # Fetch data in chunks (exchanges limit to ~1000 candles per request)
            while since < end_ms:
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
                
                # Avoid rate limits
                self.exchange.sleep(self.exchange.rateLimit)
            
            if not all_candles:
                logger.warning(f"No data received for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Keep only OHLCV columns (standard format matching Kite)
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # Filter to exact date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            logger.info(f"✅ Fetched {len(df)} bars for {symbol} ({timeframe})")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")
            return pd.DataFrame()
    
    def get_live_tick(self, symbol: str) -> Optional[Dict]:
        """
        Get current live ticker data for paper trading
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
        
        Returns:
            Dictionary with current prices or None
        """
        if self.mode != 'paper':
            logger.warning("Live tick data only available in 'paper' mode")
            return None
        
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            
            return {
                'symbol': symbol,
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'last': ticker['last'],
                'high': ticker['high'],
                'low': ticker['low'],
                'volume': ticker['quoteVolume'],
                'timestamp': datetime.fromtimestamp(ticker['timestamp'] / 1000),
                'change': ticker.get('percentage', 0)
            }
        except Exception as e:
            logger.warning(f"No ticker data for {symbol}: {e}")
            return None


# ============================================================================
# MODULE-LEVEL FUNCTIONS
# ============================================================================

def get_crypto_handler(exchange='binance', mode='backtest') -> CryptoHandler:
    """
    Get or create CryptoHandler instance (singleton pattern)
    
    Args:
        exchange: Exchange name ('binance', 'coinbase', 'bybit')
        mode: 'backtest' or 'paper'
    
    Returns:
        CryptoHandler instance
    """
    if not hasattr(get_crypto_handler, '_instance'):
        get_crypto_handler._instance = {}
    
    key = f"{exchange}_{mode}"
    if key not in get_crypto_handler._instance:
        get_crypto_handler._instance[key] = CryptoHandler(exchange, mode)
    
    return get_crypto_handler._instance[key]
