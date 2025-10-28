"""
Forex Handler - OANDA API Integration
Fetches forex data for EUR/USD, GBP/USD, USD/JPY, etc.
Cloud-compatible, works on Streamlit Cloud
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import os

logger = logging.getLogger(__name__)

# Timeframe mapping (OANDA format)
TIMEFRAME_MAP = {
    '1min': 'M1',
    '5min': 'M5',
    '15min': 'M15',
    '30min': 'M30',
    '1h': 'H1',
    'hour': 'H1',
    '4h': 'H4',
    'day': 'D',
    'daily': 'D',
    'week': 'W',
    'month': 'M'
}

class ForexHandler:
    """
    OANDA Forex Handler for backtesting
    Features:
    - Free historical data (practice account)
    - Dynamic instrument loading from OANDA API
    - Cloud-compatible REST API
    """
    
    def __init__(self, account_type='practice'):
        """
        Initialize OANDA connection
        Args:
            account_type: 'practice' or 'live'
        """
        self.account_type = account_type
        self.api_key = os.getenv('OANDA_API_KEY', '')
        self.account_id = os.getenv('OANDA_ACCOUNT_ID', '')
        
        # Set base URL based on account type
        if account_type == 'practice':
            self.base_url = 'https://api-fxpractice.oanda.com'
        else:
            self.base_url = 'https://api-fxtrade.oanda.com'
        
        self.connected = False
        self.available_symbols = {}
        
        # Initialize connection
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Test OANDA API connection"""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            # Test connection
            response = requests.get(
                f'{self.base_url}/v3/accounts/{self.account_id}',
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                self.connected = True
                logger.info(f"✅ Connected to OANDA ({self.account_type})")
                self._load_available_symbols()
            else:
                logger.error(f"❌ OANDA connection failed: {response.text}")
                self.connected = False
                
        except Exception as e:
            logger.error(f"Failed to connect to OANDA: {e}")
            self.connected = False
    
    def _load_available_symbols(self):
        """
        ✅ DYNAMICALLY fetch all available instruments from OANDA API
        """
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            # Fetch all instruments from OANDA API
            url = f'{self.base_url}/v3/accounts/{self.account_id}/instruments'
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch instruments: {response.text}")
                return
            
            data = response.json()
            instruments = data.get('instruments', [])
            
            if not instruments:
                logger.warning("No instruments returned from OANDA")
                return
            
            # Categorize instruments dynamically
            major_forex = []
            minor_forex = []
            exotic_forex = []
            cfds = []
            metals = []
            commodities = []
            indices = []
            bonds = []
            
            for instrument in instruments:
                symbol = instrument.get('name', '')
                display_name = instrument.get('displayName', '')
                instrument_type = instrument.get('type', 'CURRENCY')
                pip_location = instrument.get('pipLocation', -4)
                
                # Build instrument info
                info = {
                    'symbol': symbol,
                    'displayName': display_name,
                    'type': instrument_type,
                    'pipLocation': pip_location,
                    'marginRate': instrument.get('marginRate', '0.02'),
                    'minimumTradeSize': instrument.get('minimumTradeSize', '1'),
                    'maximumOrderUnits': instrument.get('maximumOrderUnits', '100000000')
                }
                
                # Categorize based on instrument type and symbol pattern
                if instrument_type == 'CURRENCY':
                    # Major Forex Pairs (contain USD and major currencies)
                    major_currencies = ['EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD']
                    base_currency = symbol.split('_')[0] if '_' in symbol else ''
                    quote_currency = symbol.split('_')[1] if '_' in symbol else ''
                    
                    is_major = (
                        ('USD' in [base_currency, quote_currency]) and
                        (base_currency in major_currencies or quote_currency in major_currencies)
                    )
                    
                    # Exotic pairs (contain USD with non-major currencies)
                    is_exotic = (
                        'USD' in [base_currency, quote_currency] and
                        not is_major
                    )
                    
                    if is_major:
                        major_forex.append(info)
                    elif is_exotic:
                        exotic_forex.append(info)
                    else:
                        # Minor pairs (crosses without USD)
                        minor_forex.append(info)
                
                elif instrument_type == 'CFD':
                    cfds.append(info)
                
                elif instrument_type == 'METAL':
                    metals.append(info)
                
                elif 'XAU' in symbol or 'XAG' in symbol:
                    # Gold/Silver
                    metals.append(info)
                
                elif any(x in symbol for x in ['WTI', 'BCO', 'NATGAS']):
                    # Oil & Gas commodities
                    commodities.append(info)
                
                elif any(x in symbol for x in ['US30', 'SPX', 'NAS', 'UK100', 'DE30', 'JP225']):
                    # Stock indices
                    indices.append(info)
                
                elif any(x in symbol for x in ['BUND', 'USB']):
                    # Bonds
                    bonds.append(info)
            
            # Store categorized symbols
            self.available_symbols = {}
            
            if major_forex:
                self.available_symbols['Major Forex Pairs'] = major_forex
            if minor_forex:
                self.available_symbols['Minor Forex Pairs'] = minor_forex
            if exotic_forex:
                self.available_symbols['Exotic Forex Pairs'] = exotic_forex
            if metals:
                self.available_symbols['Metals (Gold/Silver)'] = metals
            if commodities:
                self.available_symbols['Commodities (Oil/Gas)'] = commodities
            if indices:
                self.available_symbols['Stock Indices'] = indices
            if cfds:
                self.available_symbols['CFDs'] = cfds
            if bonds:
                self.available_symbols['Bonds'] = bonds
            
            # Log summary
            total_instruments = sum(len(v) for v in self.available_symbols.values())
            logger.info(f"✅ Loaded {total_instruments} instruments from OANDA:")
            for category, symbols in self.available_symbols.items():
                logger.info(f"   - {category}: {len(symbols)} instruments")
                
        except Exception as e:
            logger.error(f"Failed to load OANDA instruments: {e}")
            self.available_symbols = {}
    
    def get_available_symbols_by_category(self, category: str = None) -> List[Dict]:
        """Get available instruments by category"""
        if category:
            return self.available_symbols.get(category, [])
        else:
            # Return all instruments flattened
            all_symbols = []
            for symbols_list in self.available_symbols.values():
                all_symbols.extend(symbols_list)
            return all_symbols
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get info about a specific instrument"""
        for category, symbols in self.available_symbols.items():
            for sym in symbols:
                if sym['symbol'] == symbol:
                    return sym
        return None
    
    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '5min'
    ) -> pd.DataFrame:
        """
        Fetch historical forex data from OANDA
        Args:
            symbol: Instrument name (e.g., 'EUR_USD')
            start_date: Start date
            end_date: End date
            timeframe: Timeframe ('5min', '1h', 'day', etc.)
        Returns:
            DataFrame with OHLC data
        """
        if not self.connected:
            raise Exception("OANDA not connected")
        
        # Map timeframe
        oanda_timeframe = TIMEFRAME_MAP.get(timeframe.lower())
        if not oanda_timeframe:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            # Convert dates to OANDA format (RFC3339)
            from_time = start_date.strftime('%Y-%m-%dT%H:%M:%S.000000Z')
            to_time = end_date.strftime('%Y-%m-%dT%H:%M:%S.000000Z')
            
            # Build request URL
            url = f'{self.base_url}/v3/instruments/{symbol}/candles'
            params = {
                'from': from_time,
                'to': to_time,
                'granularity': oanda_timeframe,
                'price': 'M'  # Mid prices
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"OANDA API error: {response.text}")
                return pd.DataFrame()
            
            data = response.json()
            candles = data.get('candles', [])
            
            if not candles:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            rows = []
            for candle in candles:
                if candle['complete']:  # Only complete candles
                    rows.append({
                        'timestamp': pd.to_datetime(candle['time']),
                        'open': float(candle['mid']['o']),
                        'high': float(candle['mid']['h']),
                        'low': float(candle['mid']['l']),
                        'close': float(candle['mid']['c']),
                        'volume': int(candle['volume'])
                    })
            
            df = pd.DataFrame(rows)
            
            if df.empty:
                return df
            
            df.set_index('timestamp', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            logger.info(f"✅ Fetched {len(df)} bars for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return pd.DataFrame()

def get_forex_handler(account_type='practice') -> ForexHandler:
    """Get or create ForexHandler instance (singleton pattern)"""
    if not hasattr(get_forex_handler, '_instance'):
        get_forex_handler._instance = {}
    
    if account_type not in get_forex_handler._instance:
        get_forex_handler._instance[account_type] = ForexHandler(account_type)
    
    return get_forex_handler._instance[account_type]
