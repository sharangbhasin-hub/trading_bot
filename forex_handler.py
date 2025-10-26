"""
Forex Handler - MT5 Integration for Multi-Asset Trading
Handles both backtesting (historical data) and paper trading (live data)
100% dynamic - fetches all symbols from MT5 API
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ============================================================================
# MT5 TIMEFRAME MAPPING
# ============================================================================
TIMEFRAME_MAP = {
    '1min': mt5.TIMEFRAME_M1,
    '5min': mt5.TIMEFRAME_M5,
    '15min': mt5.TIMEFRAME_M15,
    '30min': mt5.TIMEFRAME_M30,
    '1h': mt5.TIMEFRAME_H1,
    'hour': mt5.TIMEFRAME_H1,
    '4h': mt5.TIMEFRAME_H4,
    'day': mt5.TIMEFRAME_D1,
    'daily': mt5.TIMEFRAME_D1,
    'week': mt5.TIMEFRAME_W1,
    'month': mt5.TIMEFRAME_MN1
}

class ForexHandler:
    """
    MT5 Forex Handler for backtesting and paper trading
    
    Features:
    - Dynamic symbol loading from MT5 API (no hardcoded pairs)
    - Historical data for backtesting
    - Live tick data for paper trading
    - Order execution simulation via MT5 demo account
    """
    
    def __init__(self, mode='backtest'):
        """
        Initialize MT5 connection
        
        Args:
            mode: 'backtest' for historical data, 'paper' for live trading
        """
        self.mode = mode
        self.connected = False
        self.account_info = None
        self.available_symbols = []
        
        # Get credentials from environment
        self.login = int(os.getenv('MT5_LOGIN', 0))
        self.password = os.getenv('MT5_PASSWORD', '')
        self.server = os.getenv('MT5_SERVER', '')
        self.terminal_path = os.getenv('MT5_PATH', None)
        
        # Initialize connection
        self._initialize_mt5()
    
    def _initialize_mt5(self):
        """Initialize MT5 terminal connection"""
        try:
            # Initialize MT5 (auto-detects terminal if MT5_PATH not set)
            if self.terminal_path:
                if not mt5.initialize(path=self.terminal_path):
                    raise Exception(f"MT5 initialize failed with path: {mt5.last_error()}")
            else:
                if not mt5.initialize():
                    raise Exception(f"MT5 initialize failed: {mt5.last_error()}")
            
            logger.info("✅ MT5 terminal initialized successfully")
            
            # Login to account (required for some brokers)
            if self.login and self.password and self.server:
                authorized = mt5.login(
                    login=self.login,
                    password=self.password,
                    server=self.server
                )
                
                if not authorized:
                    logger.warning(f"MT5 login failed: {mt5.last_error()}")
                    logger.warning("Continuing with terminal-level authentication")
                else:
                    logger.info(f"✅ Logged into MT5 account: {self.login}")
            
            # Get account info
            self.account_info = mt5.account_info()
            if self.account_info:
                logger.info(f"Account balance: ${self.account_info.balance}")
                logger.info(f"Account leverage: 1:{self.account_info.leverage}")
            
            self.connected = True
            
            # Load available symbols dynamically
            self._load_available_symbols()
            
        except Exception as e:
            logger.error(f"❌ MT5 initialization failed: {e}")
            self.connected = False
            raise
    
    def _load_available_symbols(self):
        """
        Dynamically load all tradeable symbols from MT5
        NO HARDCODED LISTS - fetches from API
        """
        try:
            # Get all symbols from MT5
            symbols = mt5.symbols_get()
            
            if symbols is None or len(symbols) == 0:
                logger.warning("No symbols found. Make sure MT5 terminal is logged in.")
                return
            
            # Filter and categorize symbols
            forex_majors = []
            forex_minors = []
            forex_exotics = []
            metals = []
            indices = []
            commodities = []
            
            for symbol in symbols:
                # Must be visible and tradeable
                if not symbol.visible or not symbol.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
                    continue
                
                symbol_name = symbol.name
                
                # Categorize based on symbol characteristics
                # Forex majors (contain USD and major currencies)
                if any(pair in symbol_name for pair in ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 
                                                          'AUDUSD', 'USDCAD', 'NZDUSD']):
                    forex_majors.append({
                        'symbol': symbol_name,
                        'description': symbol.description,
                        'digits': symbol.digits,
                        'point': symbol.point,
                        'spread': symbol.spread,
                        'category': 'Forex Majors'
                    })
                
                # Metals (Gold, Silver)
                elif any(metal in symbol_name for metal in ['XAU', 'XAG', 'GOLD', 'SILVER']):
                    metals.append({
                        'symbol': symbol_name,
                        'description': symbol.description,
                        'digits': symbol.digits,
                        'category': 'Metals'
                    })
                
                # Indices (US30, NAS100, etc.)
                elif any(idx in symbol_name for idx in ['US30', 'NAS100', 'SPX500', 'DJ30', 
                                                          'GER40', 'UK100', 'JPN225']):
                    indices.append({
                        'symbol': symbol_name,
                        'description': symbol.description,
                        'digits': symbol.digits,
                        'category': 'Indices'
                    })
                
                # Commodities (Oil, etc.)
                elif any(comm in symbol_name for comm in ['OIL', 'BRENT', 'WTI', 'CRUDEOIL']):
                    commodities.append({
                        'symbol': symbol_name,
                        'description': symbol.description,
                        'digits': symbol.digits,
                        'category': 'Commodities'
                    })
                
                # Forex minors and exotics
                elif 'USD' in symbol_name or any(curr in symbol_name for curr in 
                                                   ['EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD']):
                    if symbol_name not in [s['symbol'] for s in forex_majors]:
                        forex_minors.append({
                            'symbol': symbol_name,
                            'description': symbol.description,
                            'digits': symbol.digits,
                            'category': 'Forex Minors'
                        })
            
            # Store categorized symbols
            self.available_symbols = {
                'Forex Majors': forex_majors,
                'Forex Minors': forex_minors,
                'Metals': metals,
                'Indices': indices,
                'Commodities': commodities
            }
            
            # Log summary
            total = sum(len(v) for v in self.available_symbols.values())
            logger.info(f"✅ Loaded {total} tradeable symbols from MT5:")
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
            category: 'Forex Majors', 'Metals', 'Indices', etc. None = all symbols
        
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
            symbol: Symbol name (e.g., 'EURUSD')
        
        Returns:
            Dictionary with symbol metadata or None
        """
        info = mt5.symbol_info(symbol)
        
        if info is None:
            logger.warning(f"Symbol {symbol} not found")
            return None
        
        return {
            'symbol': info.name,
            'description': info.description,
            'digits': info.digits,
            'point': info.point,
            'spread': info.spread,
            'contract_size': info.trade_contract_size,
            'min_volume': info.volume_min,
            'max_volume': info.volume_max,
            'volume_step': info.volume_step,
            'currency_base': info.currency_base,
            'currency_profit': info.currency_profit,
            'trade_mode': info.trade_mode
        }
    
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
            symbol: Trading symbol (e.g., 'EURUSD')
            start_date: Start date for data
            end_date: End date for data
            timeframe: Timeframe string ('5min', '1h', 'day', etc.)
        
        Returns:
            Pandas DataFrame with OHLC data (same format as Kite)
        """
        if not self.connected:
            raise Exception("MT5 not connected. Call initialize first.")
        
        # Map timeframe string to MT5 constant
        mt5_timeframe = TIMEFRAME_MAP.get(timeframe.lower())
        if mt5_timeframe is None:
            raise ValueError(f"Invalid timeframe: {timeframe}. Valid: {list(TIMEFRAME_MAP.keys())}")
        
        try:
            # Fetch rates from MT5
            rates = mt5.copy_rates_range(
                symbol,
                mt5_timeframe,
                start_date,
                end_date
            )
            
            if rates is None or len(rates) == 0:
                logger.warning(f"No data received for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            
            # Convert timestamp to datetime
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Rename columns to match your existing format
            df = df.rename(columns={
                'time': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume',
                'real_volume': 'volume_real'
            })
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Keep only OHLCV columns (standard format)
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            logger.info(f"✅ Fetched {len(df)} bars for {symbol} ({timeframe})")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")
            return pd.DataFrame()
    
    def get_live_tick(self, symbol: str) -> Optional[Dict]:
        """
        Get current live tick data for paper trading
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Dictionary with current prices or None
        """
        if self.mode != 'paper':
            logger.warning("Live tick data only available in 'paper' mode")
            return None
        
        tick = mt5.symbol_info_tick(symbol)
        
        if tick is None:
            logger.warning(f"No tick data for {symbol}")
            return None
        
        return {
            'symbol': symbol,
            'bid': tick.bid,
            'ask': tick.ask,
            'last': tick.last,
            'time': datetime.fromtimestamp(tick.time),
            'volume': tick.volume,
            'spread': tick.ask - tick.bid
        }
    
    def place_order(self, order_params: Dict) -> Dict:
        """
        Place a paper trading order via MT5 demo account
        
        Args:
            order_params: Dictionary with order details
                - symbol: Trading symbol
                - action: 'BUY' or 'SELL'
                - volume: Lot size (e.g., 0.01 for micro lot)
                - stop_loss: SL price (optional)
                - take_profit: TP price (optional)
                - comment: Order comment
        
        Returns:
            Dictionary with order result
        """
        if self.mode != 'paper':
            raise Exception("Order placement only allowed in 'paper' mode")
        
        if not self.connected:
            raise Exception("MT5 not connected")
        
        # Prepare order request
        symbol = order_params['symbol']
        action = order_params['action'].upper()
        volume = order_params.get('volume', 0.01)
        
        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {'success': False, 'error': f'Symbol {symbol} not found'}
        
        price = tick.ask if action == 'BUY' else tick.bid
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": mt5.ORDER_TYPE_BUY if action == 'BUY' else mt5.ORDER_TYPE_SELL,
            "price": price,
            "deviation": 10,
            "magic": 234000,
            "comment": order_params.get('comment', f"Paper trade via Python"),
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Add SL/TP if provided
        if 'stop_loss' in order_params and order_params['stop_loss']:
            request['sl'] = float(order_params['stop_loss'])
        
        if 'take_profit' in order_params and order_params['take_profit']:
            request['tp'] = float(order_params['take_profit'])
        
        # Send order
        result = mt5.order_send(request)
        
        if result is None:
            return {'success': False, 'error': 'Order send failed'}
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {
                'success': False,
                'error': f'Order failed: {result.comment}',
                'retcode': result.retcode
            }
        
        return {
            'success': True,
            'order_id': result.order,
            'volume': result.volume,
            'price': result.price,
            'comment': result.comment
        }
    
    def get_open_positions(self) -> List[Dict]:
        """
        Get all open positions (for paper trading)
        
        Returns:
            List of position dictionaries
        """
        if self.mode != 'paper':
            return []
        
        positions = mt5.positions_get()
        
        if positions is None or len(positions) == 0:
            return []
        
        result = []
        for pos in positions:
            result.append({
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'type': 'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL',
                'volume': pos.volume,
                'price_open': pos.price_open,
                'price_current': pos.price_current,
                'profit': pos.profit,
                'stop_loss': pos.sl,
                'take_profit': pos.tp,
                'time': datetime.fromtimestamp(pos.time),
                'comment': pos.comment
            })
        
        return result
    
    def close_position(self, ticket: int) -> Dict:
        """
        Close an open position
        
        Args:
            ticket: Position ticket number
        
        Returns:
            Dictionary with close result
        """
        if self.mode != 'paper':
            raise Exception("Position close only in 'paper' mode")
        
        # Get position details
        positions = mt5.positions_get(ticket=ticket)
        if not positions or len(positions) == 0:
            return {'success': False, 'error': f'Position {ticket} not found'}
        
        position = positions[0]
        
        # Prepare close request
        close_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,
            "position": ticket,
            "price": mt5.symbol_info_tick(position.symbol).bid if position.type == 0 else mt5.symbol_info_tick(position.symbol).ask,
            "deviation": 10,
            "magic": 234000,
            "comment": "Close by Python",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(close_request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {'success': False, 'error': result.comment}
        
        return {
            'success': True,
            'ticket': ticket,
            'close_price': result.price,
            'profit': position.profit
        }
    
    def shutdown(self):
        """Shutdown MT5 connection"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("MT5 connection closed")


# ============================================================================
# MODULE-LEVEL FUNCTIONS
# ============================================================================

def get_forex_handler(mode='backtest') -> ForexHandler:
    """
    Get or create ForexHandler instance (singleton pattern)
    
    Args:
        mode: 'backtest' or 'paper'
    
    Returns:
        ForexHandler instance
    """
    if not hasattr(get_forex_handler, '_instance'):
        get_forex_handler._instance = {}
    
    if mode not in get_forex_handler._instance:
        get_forex_handler._instance[mode] = ForexHandler(mode=mode)
    
    return get_forex_handler._instance[mode]
