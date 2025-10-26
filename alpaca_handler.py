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
TIMEFRAME_MAP = {
    '1min': '1Min',
    '5min': '5Min',
    '15min': '15Min',
    '30min': '30Min',
    '1h': '1Hour',
    'hour': '1Hour',
    '4h': '4Hour',
    'day': '1Day',
    'daily': '1Day',
    'week': '1Week',
    'month': '1Month'
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
        Dynamically load tradeable symbols from Alpaca
        Categorizes into Stocks, Crypto, and Popular assets
        """
        try:
            # Get all tradeable assets
            all_assets = self.api.list_assets(status='active')
            
            stocks = []
            crypto = []
            popular_stocks = []
            etfs = []
            
            # Popular tickers for categorization
            popular_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 
                             'JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'HD', 'DIS',
                             'NFLX', 'PYPL', 'INTC', 'CSCO', 'PFE']
            
            etf_tickers = ['SPY', 'QQQ', 'IWM', 'DIA', 'VOO', 'VTI', 'GLD', 'SLV']
            
            for asset in all_assets:
                if not asset.tradable:
                    continue
                
                asset_dict = {
                    'symbol': asset.symbol,
                    'name': asset.name,
                    'exchange': asset.exchange,
                    'asset_class': asset.asset_class,
                    'tradable': asset.tradable
                }
                
                # Categorize
                if asset.asset_class == 'crypto':
                    crypto.append(asset_dict)
                elif asset.symbol in popular_tickers:
                    popular_stocks.append(asset_dict)
                elif asset.symbol in etf_tickers:
                    etfs.append(asset_dict)
                elif asset.asset_class == 'us_equity':
                    stocks.append(asset_dict)
            
            # Store categorized symbols
            self.available_symbols = {
                'Popular Stocks': sorted(popular_stocks, key=lambda x: popular_tickers.index(x['symbol']) if x['symbol'] in popular_tickers else 999)[:20],
                'ETFs': sorted(etfs, key=lambda x: x['symbol'])[:10],
                'Cryptocurrencies': sorted(crypto, key=lambda x: x['symbol'])[:20],
                'All Stocks': sorted(stocks, key=lambda x: x['symbol'])[:100]  # Top 100 by alphabet
            }
            
            # Log summary
            total = sum(len(v) for v in self.available_symbols.values())
            logger.info(f"✅ Loaded {total} tradeable assets:")
            for category, assets in self.available_symbols.items():
                if assets:
                    logger.info(f"  - {category}: {len(assets)} assets")
            
        except Exception as e:
            logger.error(f"Failed to load symbols: {e}")
            self.available_symbols = {}
    
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
        Get detailed information about a symbol
        
        Args:
            symbol: Symbol name (e.g., 'AAPL', 'BTCUSD')
        
        Returns:
            Dictionary with symbol metadata or None
        """
        try:
            asset = self.api.get_asset(symbol)
            
            return {
                'symbol': asset.symbol,
                'name': asset.name,
                'asset_class': asset.asset_class,
                'exchange': asset.exchange,
                'tradable': asset.tradable,
                'marginable': asset.marginable,
                'shortable': asset.shortable,
                'easy_to_borrow': asset.easy_to_borrow,
                'fractionable': asset.fractionable
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
        alpaca_timeframe = TIMEFRAME_MAP.get(timeframe.lower())
        if alpaca_timeframe is None:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
        try:
            # Check if crypto (use crypto endpoint) or stock (use bars endpoint)
            asset_info = self.get_symbol_info(symbol)
            is_crypto = asset_info and asset_info['asset_class'] == 'crypto'
            
            if is_crypto:
                # Crypto bars
                bars = self.api.get_crypto_bars(
                    symbol,
                    alpaca_timeframe,
                    start=start_date.isoformat(),
                    end=end_date.isoformat()
                ).df
            else:
                # Stock bars
                bars = self.api.get_bars(
                    symbol,
                    alpaca_timeframe,
                    start=start_date.isoformat(),
                    end=end_date.isoformat(),
                    adjustment='raw'
                ).df
            
            if bars.empty:
                logger.warning(f"No data received for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Standardize column names to match Kite format
            bars = bars.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            
            # Ensure index is named 'timestamp'
            bars.index.name = 'timestamp'
            
            # Keep only OHLCV columns
            bars = bars[['open', 'high', 'low', 'close', 'volume']]
            
            logger.info(f"✅ Fetched {len(bars)} bars for {symbol} ({timeframe})")
            
            return bars
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")
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
