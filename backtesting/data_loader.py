"""
Data Loader for Backtesting
Fetches historical data and stores it locally
"""
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import logging

from backtesting.config import BacktestConfig, get_trading_days  # ← Add get_trading_days


# Setup logging
logging.basicConfig(
    level=getattr(logging, BacktestConfig.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads historical data for backtesting
    Uses Kite API to fetch data and caches it locally
    """
    
    def __init__(self, kite_handler):
        """
        Initialize data loader
        
        Args:
            kite_handler: Instance of KiteHandler from your existing code
        """
        self.kite = kite_handler
        self.config = BacktestConfig()
        self.cache_dir = Path(self.config.RESULTS_DIR) / 'data_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_historical_data(self, index, start_date, end_date):
        """
        Fetch historical data for given date range
        
        Args:
            index: 'NIFTY' or 'BANKNIFTY'
            start_date: datetime object
            end_date: datetime object
        
        Returns:
            Dictionary with all timeframes
        """
        logger.info(f"Fetching data for {index} from {start_date} to {end_date}")
        
        # Check if cached
        cache_file = self.cache_dir / f"{index}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
        
        if cache_file.exists():
            logger.info(f"Loading from cache: {cache_file}")
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            return self._deserialize_data(cached_data)
        
        # Get instrument token
        instrument_token = self._get_instrument_token(index)
        logger.info(f"Using instrument token: {instrument_token}")
        
        # Fetch data for each timeframe
        all_data = {
            'dates': [],
            'data': {}
        }
        
        # Calculate date chunks (60 days per API call for intraday)
        date_chunks = self._split_date_range(start_date, end_date, days=60)
        logger.info(f"Split into {len(date_chunks)} chunks of ~60 days each")
        
        for timeframe_name, timeframe_code in self.config.TIMEFRAMES.items():
            logger.info(f"Fetching {timeframe_name} data using interval '{timeframe_code}'...")
            
            timeframe_data = pd.DataFrame()
            
            for idx, (chunk_start, chunk_end) in enumerate(date_chunks):
                try:
                    logger.info(f"  Chunk {idx+1}/{len(date_chunks)}: {chunk_start.date()} to {chunk_end.date()}")
                    
                    # Call your kite_handler's get_historical_data method
                    # Format: get_historical_data(instrument_token, from_date, to_date, interval)
                    chunk_data = self.kite.get_historical_data(
                        instrument_token,
                        chunk_start,  # datetime object (your method accepts datetime)
                        chunk_end,    # datetime object
                        timeframe_code  # '5minute', '15minute', '60minute', 'day'
                    )
                    
                    # Check if data was returned
                    if chunk_data is not None and isinstance(chunk_data, pd.DataFrame) and not chunk_data.empty:
                        logger.info(f"    Received {len(chunk_data)} candles")
                        timeframe_data = pd.concat([timeframe_data, chunk_data], ignore_index=True)
                    else:
                        logger.warning(f"    No data returned for this chunk")
                    
                    # Small delay to respect rate limits
                    import time
                    time.sleep(0.3)
                    
                except Exception as e:
                    logger.error(f"  Error fetching {timeframe_name} for chunk {idx+1}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue with next chunk
            
            # Process fetched data
            if not timeframe_data.empty:
                logger.info(f"Total {timeframe_name} candles fetched: {len(timeframe_data)}")
                
                # Ensure datetime index
                if 'date' in timeframe_data.columns:
                    timeframe_data['date'] = pd.to_datetime(timeframe_data['date'])
                    timeframe_data.set_index('date', inplace=True)
                elif not isinstance(timeframe_data.index, pd.DatetimeIndex):
                    timeframe_data.index = pd.to_datetime(timeframe_data.index)
                
                # Remove duplicates
                timeframe_data = timeframe_data[~timeframe_data.index.duplicated(keep='first')]
                
                # Group by date
                for date in timeframe_data.index.normalize().unique():
                    date_str = date.strftime('%Y-%m-%d')
                    
                    if date_str not in all_data['data']:
                        all_data['data'][date_str] = {}
                        if date_str not in all_data['dates']:
                            all_data['dates'].append(date_str)
                    
                    # Filter data for this date
                    day_data = timeframe_data[timeframe_data.index.normalize() == date]
                    all_data['data'][date_str][timeframe_name] = day_data
            else:
                logger.warning(f"No {timeframe_name} data fetched for entire period!")
        
        # Sort dates
        all_data['dates'].sort()
        
        # Log summary
        logger.info("=" * 60)
        logger.info("DATA FETCH SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total dates fetched: {len(all_data['dates'])}")
        
        if all_data['dates']:
            logger.info(f"Date range: {all_data['dates'][0]} to {all_data['dates'][-1]}")
            logger.info(f"Timeframes per date: {list(all_data['data'][all_data['dates'][0]].keys())}")
            
            # Cache the data
            self._cache_data(all_data, cache_file)
            logger.info(f"Successfully fetched and cached data for {len(all_data['dates'])} trading days")
        else:
            logger.error("NO DATA WAS FETCHED!")
            logger.error("Possible issues:")
            logger.error("1. Kite API connection problem")
            logger.error("2. Instrument token is incorrect")
            logger.error("3. Date range has no trading days (weekends/holidays)")
            logger.error("4. API rate limit exceeded")
        
        logger.info("=" * 60)
        
        return all_data
        
    def _get_instrument_token(self, index):
        """
        Get instrument token for index
        
        Args:
            index: 'NIFTY' or 'BANKNIFTY'
        
        Returns:
            int: Instrument token
        """
        # Hardcoded tokens for NSE indices (most reliable method)
        INDEX_TOKENS = {
            'NIFTY': 256265,      # NIFTY 50
            'NIFTY 50': 256265,
            'BANKNIFTY': 260105,  # BANK NIFTY
            'BANK NIFTY': 260105,
            'NIFTY BANK': 260105,
        }
        
        # Try exact match first
        if index in INDEX_TOKENS:
            logger.info(f"Found token for {index}: {INDEX_TOKENS[index]}")
            return INDEX_TOKENS[index]
        
        # Try case-insensitive match
        index_upper = index.upper()
        for key, token in INDEX_TOKENS.items():
            if key.upper() == index_upper:
                logger.info(f"Found token for {index} (matched {key}): {token}")
                return token
        
        # If your kite_handler has index_token_map, try that
        if hasattr(self.kite, 'index_token_map') and index in self.kite.index_token_map:
            token = self.kite.index_token_map[index]
            logger.info(f"Found token for {index} in kite.index_token_map: {token}")
            return token
        
        # Last resort: try searching (may not work on all systems)
        try:
            logger.info(f"Attempting to search for {index}...")
            instruments = self.kite.kite.instruments('NSE')  # Get all instruments
            
            # Filter for index
            index_instruments = instruments[instruments['tradingsymbol'].str.contains(index, case=False, na=False)]
            
            if not index_instruments.empty:
                token = index_instruments.iloc[0]['instrument_token']
                logger.info(f"Found token for {index} via search: {token}")
                return token
        except Exception as e:
            logger.warning(f"Could not search instruments: {e}")
        
        # If nothing works, show available indices
        raise ValueError(
            f"Could not find instrument token for '{index}'.\n"
            f"Available indices: {', '.join(INDEX_TOKENS.keys())}\n"
            f"Please use one of these exact names."
        )

    def _split_date_range(self, start_date, end_date, days=60):
        """Split date range into chunks"""
        chunks = []
        current_start = start_date
        
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=days), end_date)
            chunks.append((current_start, current_end))
            current_start = current_end + timedelta(days=1)
        
        return chunks
    
    def _cache_data(self, data, cache_file):
        """Cache data to JSON file"""
        # Convert DataFrames to dict for JSON serialization
        serialized_data = {
            'dates': data['dates'],
            'data': {}
        }
        
        for date, timeframes in data['data'].items():
            serialized_data['data'][date] = {}
            for tf, df in timeframes.items():
                serialized_data['data'][date][tf] = df.reset_index().to_dict('records')
        
        with open(cache_file, 'w') as f:
            json.dump(serialized_data, f)
        
        logger.info(f"Cached data to {cache_file}")
    
    def _deserialize_data(self, serialized_data):
        """Convert cached JSON back to DataFrames"""
        data = {
            'dates': serialized_data['dates'],
            'data': {}
        }
        
        for date, timeframes in serialized_data['data'].items():
            data['data'][date] = {}
            for tf, records in timeframes.items():
                df = pd.DataFrame(records)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                data['data'][date][tf] = df
        
        return data
    
    def validate_data(self, data):
        """
        Validate data quality
        
        Returns:
            Dict with validation results
        """
        validation = {
            'is_valid': True,
            'issues': [],
            'stats': {}
        }
        
        # Check if data exists
        if not data or not data.get('dates'):
            validation['is_valid'] = False
            validation['issues'].append("No data fetched. Please check API connection and date range.")
            validation['stats']['total_days'] = 0
            validation['stats']['date_range'] = 'None'
            return validation
        
        # Check for missing dates
        from backtesting.config import get_trading_days
        
        expected_days = len(get_trading_days(
            datetime.strptime(data['dates'][0], '%Y-%m-%d'),
            datetime.strptime(data['dates'][-1], '%Y-%m-%d')
        ))
        
        actual_days = len(data['dates'])
        
        if actual_days < expected_days * 0.9:  # Allow 10% missing
            validation['issues'].append(f"Missing dates: Expected {expected_days}, got {actual_days}")
            # Don't mark as invalid for now, just warn
        
        # Check each day
        for date_str in data['dates']:
            day_data = data['data'][date_str]
            
            # Check all timeframes present
            missing_tfs = set(self.config.TIMEFRAMES.keys()) - set(day_data.keys())
            if missing_tfs:
                validation['issues'].append(f"{date_str}: Missing timeframes {missing_tfs}")
            
            # Check candle counts
            for tf, df in day_data.items():
                if df.empty:
                    validation['issues'].append(f"{date_str} {tf}: Empty DataFrame")
        
        validation['stats']['total_days'] = actual_days
        validation['stats']['date_range'] = f"{data['dates'][0]} to {data['dates'][-1]}"
        
        return validation


# ===== STANDALONE SCRIPT FOR DATA DOWNLOAD =====

if __name__ == '__main__':
    """
    Standalone script to download all historical data
    Run this first before backtesting
    """
    import sys
    sys.path.append('..')
    
    from kite_handler import get_kite_handler
    
    # Initialize Kite
    kite = get_kite_handler()
    
    loader = DataLoader(kite)
    
    # Download 2024 data
    print("Downloading 2024 data...")
    for index in BacktestConfig.INDICES:
        data_2024 = loader.fetch_historical_data(
            index,
            BacktestConfig.BACKTEST_START_DATE,
            BacktestConfig.BACKTEST_END_DATE
        )
        
        # Validate
        validation = loader.validate_data(data_2024)
        print(f"{index} 2024: {validation}")
    
    # Download 2025 data
    print("\nDownloading 2025 data...")
    for index in BacktestConfig.INDICES:
        data_2025 = loader.fetch_historical_data(
            index,
            BacktestConfig.FORWARD_TEST_START_DATE,
            BacktestConfig.FORWARD_TEST_END_DATE
        )
        
        # Validate
        validation = loader.validate_data(data_2025)
        print(f"{index} 2025: {validation}")
    
    print("\n✅ Data download complete!")
