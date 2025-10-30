"""
Alpaca Handler - Multi-Asset Trading Integration
Handles Stocks, Crypto, and Options via Alpaca API
Cloud-compatible, works on Streamlit Cloud
"""

import alpaca_trade_api as tradeapi
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ============================================================================
# TIMEFRAME MAPPING (Alpaca format)
# ============================================================================
# ============================================================================
# TIMEFRAME MAPPING (Alpaca format)
# ============================================================================
TIMEFRAME_MAP = {
    # Minute timeframes (both formats supported)
    '1m': '1Min',
    '5m': '5Min',
    '15m': '15Min',
    '30m': '30Min',
    '1min': '1Min',      # ← ADD
    '5min': '5Min',      # ← ADD
    '15min': '15Min',    # ← ADD
    '30min': '30Min',    # ← ADD
    
    # Hour timeframes
    '1h': '1Hour',
    '4h': '4Hour',
    'h': '1Hour',
    'hour': '1Hour',
    '1hour': '1Hour',
    
    # Day timeframes
    '1d': '1Day',
    'd': '1Day',
    'D': '1Day',
    'day': '1Day',
    'daily': '1Day',
    
    # Week timeframes
    '1w': '1Week',
    'w': '1Week',
    'week': '1Week',
    
    # Month timeframes
    '1mo': '1Month',
    'mo': '1Month',
    'month': '1Month',
}

class AlpacaHandler:
    """
    Alpaca API Handler for backtesting and paper trading
    
    Supports:
    - US Stocks (NYSE, NASDAQ)
    - Cryptocurrencies (BTC, ETH, etc.)
    - Stock Options
    """
    
    def __init__(self, mode='paper'):
        """
        Initialize Alpaca API connection
        
        Args:
            mode: 'paper' for paper trading (default), 'live' for live trading
        """
        self.mode = mode
        self.api = None
        self.connected = False
        self.available_symbols = {}
        
        # Get credentials from environment
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        # Set base URL based on mode
        if mode == 'paper':
            self.base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        else:
            self.base_url = 'https://api.alpaca.markets'
        
        # Initialize connection
        self._initialize_alpaca()
    
    def _initialize_alpaca(self):
        """Initialize Alpaca API connection"""
        try:
            if not self.api_key or not self.secret_key:
                raise ValueError("Alpaca API credentials not found in .env file")
            
            # Create API instance
            self.api = tradeapi.REST(
                key_id=self.api_key,
                secret_key=self.secret_key,
                base_url=self.base_url,
                api_version='v2'
            )
            
            # Verify connection by getting account info
            account = self.api.get_account()
            
            logger.info(f"✅ Alpaca API connected ({self.mode} mode)")
            logger.info(f"   Account Status: {account.status}")
            logger.info(f"   Buying Power: ${float(account.buying_power):,.2f}")
            logger.info(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
            
            self.connected = True
            
            # Load available symbols
            self._load_available_symbols()
            
        except Exception as e:
            logger.error(f"❌ Alpaca initialization failed: {e}")
            self.connected = False
            raise
    
    def _load_available_symbols(self):
        """
        Load tradeable symbols - Dynamic crypto loading + curated stocks
        """
        logger.info("Loading symbols...")
        
        # ============================================================
        # DYNAMIC CRYPTO LOADING (from Alpaca API)
        # ============================================================
        crypto_symbols = []
        try:
            logger.info("Fetching crypto assets from Alpaca API...")
            
            # Fetch all active crypto assets
            all_crypto_assets = self.api.list_assets(status='active', asset_class='crypto')
            
            logger.info(f"Found {len(all_crypto_assets)} active crypto assets from API")
            
            for asset in all_crypto_assets:
                if asset.tradable and asset.exchange == 'CRYPTO':
                    # Alpaca returns crypto symbols as 'BTC/USD' in asset.symbol
                    # Convert to BTCUSD format for consistency
                    raw_symbol = asset.symbol
                    
                    # Remove slash: BTC/USD -> BTCUSD
                    symbol_no_slash = raw_symbol.replace('/', '')
                    
                    crypto_symbols.append({
                        'symbol': symbol_no_slash,  # BTCUSD format
                        'name': asset.name if hasattr(asset, 'name') and asset.name else raw_symbol,
                        'exchange': 'CRYPTO',
                        'asset_class': 'crypto',
                        'tradable': True
                    })
            
            logger.info(f"✅ Dynamically loaded {len(crypto_symbols)} tradable crypto symbols")
        
        except Exception as e:
            logger.warning(f"⚠️ Could not fetch crypto symbols from API: {e}")
            logger.info("Falling back to hardcoded popular cryptos")
            
            # Fallback to hardcoded 4 popular cryptos
            crypto_symbols = [
                {'symbol': 'BTCUSD', 'name': 'Bitcoin', 'exchange': 'CRYPTO', 'asset_class': 'crypto', 'tradable': True},
                {'symbol': 'ETHUSD', 'name': 'Ethereum', 'exchange': 'CRYPTO', 'asset_class': 'crypto', 'tradable': True},
                {'symbol': 'BCHUSD', 'name': 'Bitcoin Cash', 'exchange': 'CRYPTO', 'asset_class': 'crypto', 'tradable': True},
                {'symbol': 'LTCUSD', 'name': 'Litecoin', 'exchange': 'CRYPTO', 'asset_class': 'crypto', 'tradable': True}
            ]
        
        # ============================================================
        # STOCKS & ETFs (Keep curated lists for reliability)
        # ============================================================
        self.available_symbols = {
            'Popular Stocks': [
                {'symbol': 'AAPL', 'name': 'Apple Inc', 'exchange': 'NASDAQ', 'asset_class': 'us_equity', 'tradable': True},
                {'symbol': 'TSLA', 'name': 'Tesla Inc', 'exchange': 'NASDAQ', 'asset_class': 'us_equity', 'tradable': True},
                {'symbol': 'MSFT', 'name': 'Microsoft Corp', 'exchange': 'NASDAQ', 'asset_class': 'us_equity', 'tradable': True},
                {'symbol': 'GOOGL', 'name': 'Alphabet Inc Class A', 'exchange': 'NASDAQ', 'asset_class': 'us_equity', 'tradable': True},
                {'symbol': 'AMZN', 'name': 'Amazon.com Inc', 'exchange': 'NASDAQ', 'asset_class': 'us_equity', 'tradable': True},
                {'symbol': 'META', 'name': 'Meta Platforms Inc', 'exchange': 'NASDAQ', 'asset_class': 'us_equity', 'tradable': True},
                {'symbol': 'NVDA', 'name': 'NVIDIA Corp', 'exchange': 'NASDAQ', 'asset_class': 'us_equity', 'tradable': True},
                {'symbol': 'JPM', 'name': 'JPMorgan Chase & Co', 'exchange': 'NYSE', 'asset_class': 'us_equity', 'tradable': True},
                {'symbol': 'V', 'name': 'Visa Inc', 'exchange': 'NYSE', 'asset_class': 'us_equity', 'tradable': True},
                {'symbol': 'WMT', 'name': 'Walmart Inc', 'exchange': 'NYSE', 'asset_class': 'us_equity', 'tradable': True},
                {'symbol': 'DIS', 'name': 'Walt Disney Co', 'exchange': 'NYSE', 'asset_class': 'us_equity', 'tradable': True},
                {'symbol': 'NFLX', 'name': 'Netflix Inc', 'exchange': 'NASDAQ', 'asset_class': 'us_equity', 'tradable': True},
                {'symbol': 'PYPL', 'name': 'PayPal Holdings Inc', 'exchange': 'NASDAQ', 'asset_class': 'us_equity', 'tradable': True},
                {'symbol': 'AMD', 'name': 'Advanced Micro Devices Inc', 'exchange': 'NASDAQ', 'asset_class': 'us_equity', 'tradable': True},
                {'symbol': 'INTC', 'name': 'Intel Corp', 'exchange': 'NASDAQ', 'asset_class': 'us_equity', 'tradable': True}
            ],
            'Tech Stocks': [
                {'symbol': 'AAPL', 'name': 'Apple Inc', 'exchange': 'NASDAQ', 'asset_class': 'us_equity', 'tradable': True},
                {'symbol': 'MSFT', 'name': 'Microsoft Corp', 'exchange': 'NASDAQ', 'asset_class': 'us_equity', 'tradable': True},
                {'symbol': 'GOOGL', 'name': 'Alphabet Inc', 'exchange': 'NASDAQ', 'asset_class': 'us_equity', 'tradable': True},
                {'symbol': 'META', 'name': 'Meta Platforms Inc', 'exchange': 'NASDAQ', 'asset_class': 'us_equity', 'tradable': True},
                {'symbol': 'NVDA', 'name': 'NVIDIA Corp', 'exchange': 'NASDAQ', 'asset_class': 'us_equity', 'tradable': True},
                {'symbol': 'AMD', 'name': 'Advanced Micro Devices', 'exchange': 'NASDAQ', 'asset_class': 'us_equity', 'tradable': True},
                {'symbol': 'INTC', 'name': 'Intel Corp', 'exchange': 'NASDAQ', 'asset_class': 'us_equity', 'tradable': True},
                {'symbol': 'CSCO', 'name': 'Cisco Systems Inc', 'exchange': 'NASDAQ', 'asset_class': 'us_equity', 'tradable': True}
            ],
            'ETFs': [
                {'symbol': 'SPY', 'name': 'SPDR S&P 500 ETF', 'exchange': 'NYSE', 'asset_class': 'etf', 'tradable': True},
                {'symbol': 'QQQ', 'name': 'Invesco QQQ Trust', 'exchange': 'NASDAQ', 'asset_class': 'etf', 'tradable': True},
                {'symbol': 'DIA', 'name': 'SPDR Dow Jones Industrial Average ETF', 'exchange': 'NYSE', 'asset_class': 'etf', 'tradable': True},
                {'symbol': 'IWM', 'name': 'iShares Russell 2000 ETF', 'exchange': 'NYSE', 'asset_class': 'etf', 'tradable': True},
                {'symbol': 'VOO', 'name': 'Vanguard S&P 500 ETF', 'exchange': 'NYSE', 'asset_class': 'etf', 'tradable': True}
            ],
            'Cryptocurrencies': crypto_symbols  # ← Use dynamically loaded list
        }
        
        # Log summary
        total = sum(len(v) for v in self.available_symbols.values())
        logger.info(f"✅ Loaded {total} total symbols:")
        for category, assets in self.available_symbols.items():
            logger.info(f"   - {category}: {len(assets)} assets")
    
    def get_available_symbols_by_category(self, category: str = None) -> List[Dict]:
        """
        Get available symbols, optionally filtered by category
        
        Args:
            category: 'Popular Stocks', 'ETFs', 'Cryptocurrencies', 'All Stocks'
        
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
            for assets_list in self.available_symbols.values():
                all_symbols.extend(assets_list)
            return all_symbols
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """
        Get detailed information about a symbol (DYNAMIC - works for live + backtest)
        
        Args:
            symbol: Symbol name (e.g., 'AAPL', 'BTCUSD')
        
        Returns:
            Dictionary with symbol metadata or None
        """
        # ✅ DYNAMIC METHOD 1: Detect crypto by symbol suffix
        # Crypto symbols on Alpaca end with USD (BTCUSD, ETHUSD, etc.)
        is_crypto_symbol = symbol.endswith('USD') and len(symbol) > 3
        
        if is_crypto_symbol:
            # For crypto, construct metadata dynamically
            logger.info(f"Detected crypto symbol: {symbol} (by suffix)")
            return {
                'symbol': symbol,
                'name': symbol.replace('USD', ' / USD'),
                'asset_class': 'crypto',
                'exchange': 'CRYPTO',
                'tradable': True
            }
        
        # ✅ DYNAMIC METHOD 2: For stocks, try API (this works for stocks)
        try:
            asset_obj = self.api.get_asset(symbol)
            
            logger.info(f"Fetched {symbol} from API: {getattr(asset_obj, 'asset_class', 'us_equity')}")
            
            return {
                'symbol': asset_obj.symbol,
                'name': getattr(asset_obj, 'name', symbol),
                'asset_class': getattr(asset_obj, 'asset_class', 'us_equity'),
                'exchange': getattr(asset_obj, 'exchange', 'UNKNOWN'),
                'tradable': getattr(asset_obj, 'tradable', True),
            }
        except Exception as e:
            logger.warning(f"Could not fetch {symbol} from API: {e}")
            
            # ✅ FALLBACK: Check hardcoded list (for backtesting only)
            for category, assets in self.available_symbols.items():
                for asset in assets:
                    if asset['symbol'] == symbol:
                        logger.info(f"Found {symbol} in hardcoded list: {category}")
                        return asset
            
            logger.error(f"Symbol {symbol} not found (API + hardcoded list failed)")
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
        Fetch historical OHLC data for backtesting
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'BTCUSD')
            start_date: Start date for data
            end_date: End date for data
            timeframe: Timeframe string ('5min', '1h', 'day', etc.)
        
        Returns:
            Pandas DataFrame with OHLC data (same format as Kite)
        """
        if not self.connected:
            raise Exception("Alpaca not connected")
        
        # Map timeframe string to Alpaca format
        # ✅ Map timeframe string to Alpaca format (with validation)
        normalized_tf = timeframe.lower().strip()
        alpaca_timeframe = TIMEFRAME_MAP.get(normalized_tf)
        
        if alpaca_timeframe is None:
            logger.error(f"❌ Invalid timeframe: {timeframe}. Valid options: {list(TIMEFRAME_MAP.keys())}")
            logger.warning(f"Falling back to 5Min")
            alpaca_timeframe = '5Min'
        else:
            logger.debug(f"✅ Mapped timeframe: {timeframe} → {alpaca_timeframe}")
        
        try:
            logger.info(f"Fetching {symbol} data from Alpaca...")
            logger.info(f"  Timeframe: {alpaca_timeframe}")
            logger.info(f"  Date Range: {start_date} to {end_date}")
            
            # Check if crypto (use crypto endpoint) or stock (use bars endpoint)
            asset_info = self.get_symbol_info(symbol)
            is_crypto = asset_info and asset_info.get('asset_class') == 'crypto'
            
            if is_crypto:
                logger.info(f"  Detected crypto asset")
                
                # ✅ FIX: Convert crypto symbol to Alpaca format (BTC/USD)
                if '/' not in symbol:
                    # Convert BTCUSD -> BTC/USD
                    if symbol.endswith('USDT'):
                        symbol = f"{symbol[:-4]}/{symbol[-4:]}"  # BTC/USDT
                    elif symbol.endswith('USD'):
                        symbol = f"{symbol[:-3]}/{symbol[-3:]}"  # BTC/USD -> BTC/USD
                    else:
                        # Default: assume last 3 chars are quote currency
                        symbol = f"{symbol[:-3]}/{symbol[-3:]}"
                    logger.info(f"  Converted symbol format to: {symbol}")
                
                # Crypto bars
                bars = self.api.get_crypto_bars(
                    symbol,  # ← Now BTC/USD (correct format!)
                    alpaca_timeframe,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d')
                ).df
            else:
                logger.info(f"  Detected stock asset")
                # Stock bars - use alpaca_trade_api TimeFrame
                from alpaca_trade_api.rest import TimeFrame as AlpacaTimeFrame
                
                # Map string timeframe to correct TimeFrame format
                timeframe_map = {
                    '1Min': AlpacaTimeFrame.Minute,              # ✅ FIXED
                    '5Min': AlpacaTimeFrame(5, AlpacaTimeFrame.Minute),   # ✅ FIXED
                    '15Min': AlpacaTimeFrame(15, AlpacaTimeFrame.Minute), # ✅ FIXED
                    '30Min': AlpacaTimeFrame(30, AlpacaTimeFrame.Minute), # ✅ FIXED
                    '1Hour': AlpacaTimeFrame.Hour,               # ✅ FIXED
                    '4Hour': AlpacaTimeFrame(4, AlpacaTimeFrame.Hour),    # ✅ FIXED
                    '1Day': AlpacaTimeFrame.Day,                 # ✅ FIXED
                    '1Week': AlpacaTimeFrame.Week,               # ✅ FIXED
                    '1Month': AlpacaTimeFrame.Month              # ✅ FIXED
                }
                
                tf_enum = timeframe_map.get(alpaca_timeframe, AlpacaTimeFrame(5, AlpacaTimeFrame.Minute))
                
                # Fetch bars
                bars = self.api.get_bars(
                    symbol,
                    tf_enum,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    adjustment='raw',
                    feed='iex'  # Use IEX feed for free tier
                ).df
            
            if bars is None or bars.empty:
                logger.warning(f"No data received for {symbol} {timeframe}")
                logger.warning(f"  This could mean:")
                logger.warning(f"  1. Symbol doesn't have data for this period")
                logger.warning(f"  2. Markets were closed (weekend/holiday)")
                logger.warning(f"  3. API rate limit reached")
                return pd.DataFrame()
            
            # Alpaca returns multi-index for multiple symbols, flatten if needed
            if isinstance(bars.index, pd.MultiIndex):
                bars = bars.xs(symbol, level='symbol')
            
            # Standardize column names to match Kite format (lowercase)
            bars = bars.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            
            # Ensure index is timezone-naive datetime
            if bars.index.tz is not None:
                bars.index = bars.index.tz_localize(None)
            
            # Ensure index is named 'timestamp'
            bars.index.name = 'timestamp'
            
            # Keep only OHLCV columns
            bars = bars[['open', 'high', 'low', 'close', 'volume']]
            
            logger.info(f"✅ Fetched {len(bars)} bars for {symbol} ({timeframe})")
            logger.info(f"   Date range: {bars.index.min()} to {bars.index.max()}")
            
            return bars
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def get_live_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get current live quote for paper trading
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Dictionary with current prices or None
        """
        try:
            # Check if crypto or stock
            asset_info = self.get_symbol_info(symbol)
            is_crypto = asset_info and asset_info['asset_class'] == 'crypto'
            
            if is_crypto:
                quote = self.api.get_latest_crypto_quote(symbol)
                return {
                    'symbol': symbol,
                    'bid': float(quote.bp),
                    'ask': float(quote.ap),
                    'timestamp': quote.t
                }
            else:
                quote = self.api.get_latest_quote(symbol)
                return {
                    'symbol': symbol,
                    'bid': float(quote.bp),
                    'ask': float(quote.ap),
                    'timestamp': quote.t
                }
        except Exception as e:
            logger.warning(f"No quote data for {symbol}: {e}")
            return None
    
    def place_order(self, order_params: Dict) -> Dict:
        """
        Place a paper trading order
        
        Args:
            order_params: Dictionary with order details
                - symbol: Trading symbol
                - qty: Quantity (fractional supported for stocks)
                - side: 'buy' or 'sell'
                - type: 'market' or 'limit'
                - time_in_force: 'day', 'gtc', 'ioc', etc.
                - limit_price: Required if type='limit'
        
        Returns:
            Dictionary with order result
        """
        if not self.connected:
            raise Exception("Alpaca not connected")
        
        try:
            order = self.api.submit_order(
                symbol=order_params['symbol'],
                qty=order_params.get('qty', 1),
                side=order_params['side'],
                type=order_params.get('type', 'market'),
                time_in_force=order_params.get('time_in_force', 'gtc'),
                limit_price=order_params.get('limit_price') if order_params.get('type') == 'limit' else None
            )
            
            return {
                'success': True,
                'order_id': order.id,
                'symbol': order.symbol,
                'qty': float(order.qty),
                'side': order.side,
                'type': order.type,
                'status': order.status
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_open_positions(self) -> List[Dict]:
        """
        Get all open positions
        
        Returns:
            List of position dictionaries
        """
        try:
            positions = self.api.list_positions()
            
            result = []
            for pos in positions:
                result.append({
                    'symbol': pos.symbol,
                    'qty': float(pos.qty),
                    'side': 'long' if float(pos.qty) > 0 else 'short',
                    'avg_entry_price': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price),
                    'market_value': float(pos.market_value),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc) * 100,  # Percentage
                    'cost_basis': float(pos.cost_basis)
                })
            
            return result
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    def close_position(self, symbol: str) -> Dict:
        """
        Close an open position
        
        Args:
            symbol: Symbol to close
        
        Returns:
            Dictionary with result
        """
        try:
            self.api.close_position(symbol)
            return {'success': True, 'symbol': symbol}
        except Exception as e:
            return {'success': False, 'error': str(e)}


# ============================================================================
# MODULE-LEVEL FUNCTIONS
# ============================================================================

def get_alpaca_handler(mode='paper') -> AlpacaHandler:
    """
    Get or create AlpacaHandler instance (singleton pattern)
    
    Args:
        mode: 'paper' or 'live'
    
    Returns:
        AlpacaHandler instance
    """
    if not hasattr(get_alpaca_handler, '_instance'):
        get_alpaca_handler._instance = {}
    
    if mode not in get_alpaca_handler._instance:
        get_alpaca_handler._instance[mode] = AlpacaHandler(mode=mode)
    
    return get_alpaca_handler._instance[mode]
