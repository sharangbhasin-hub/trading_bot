"""
Live Data Manager for Paper Trading
====================================

Manages real-time data fetching for paper trading with 60-second polling.
Maintains circular buffer of recent candles for strategy analysis.

Features:
- Thread-safe 60-second polling loop
- Circular buffer for recent candles
- Multi-symbol support
- Automatic retry with exponential backoff
- Exchange failover (crypto)
- Rate limit protection

Author: Trading System
Last Updated: October 29, 2025
"""

import threading
import time
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Optional, Any, Callable
import logging

from .config import get_config

logger = logging.getLogger(__name__)


class LiveDataManager:
    """
    Production-grade live data manager with polling strategy.
    
    Architecture:
    - One thread per symbol
    - 60-second polling interval
    - Circular buffer (last 500 candles)
    - Automatic error recovery
    - Thread-safe operations
    """
    
    def __init__(self, unified_handler):
        """
        Initialize live data manager.
        
        Args:
            unified_handler: UnifiedDataHandler instance (crypto or forex)
        """
        self.handler = unified_handler
        self.config = get_config('data')
        
        # Data storage (thread-safe with locks)
        self.buffers: Dict[str, deque] = {}  # symbol -> deque of candles
        self.latest_candles: Dict[str, Dict] = {}  # symbol -> latest candle
        self.buffer_locks: Dict[str, threading.Lock] = {}  # symbol -> lock
        
        # Thread management
        self.threads: Dict[str, threading.Thread] = {}
        self.running: Dict[str, bool] = {}
        self.stop_flags: Dict[str, threading.Event] = {}
        
        # Statistics
        self.fetch_counts: Dict[str, int] = {}
        self.error_counts: Dict[str, int] = {}
        self.last_fetch_times: Dict[str, datetime] = {}
        
        # Callbacks
        self.on_new_candle_callbacks: Dict[str, List[Callable]] = {}
        
        logger.info("LiveDataManager initialized")
    
    # ========================================================================
    # START/STOP FEED
    # ========================================================================
    
    def start_feed(
        self, 
        symbol: str, 
        timeframe: str = '1min',
        on_new_candle: Optional[Callable] = None
    ) -> bool:
        """
        Start live data feed for symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT', 'EUR/USD')
            timeframe: Candle timeframe ('1min', '5min', '1h', etc.)
            on_new_candle: Optional callback function(symbol, candle_dict)
        
        Returns:
            bool: True if started successfully
        
        Example:
            >>> manager.start_feed('BTC/USDT', '1min', lambda s, c: print(f"New candle: {c}"))
        """
        if symbol in self.running and self.running[symbol]:
            logger.warning(f"Feed already running for {symbol}")
            return False
        
        # Initialize data structures
        buffer_size = self.config['candles_buffer_size']
        self.buffers[symbol] = deque(maxlen=buffer_size)
        self.buffer_locks[symbol] = threading.Lock()
        self.running[symbol] = True
        self.stop_flags[symbol] = threading.Event()
        self.fetch_counts[symbol] = 0
        self.error_counts[symbol] = 0
        
        # Register callback
        if on_new_candle:
            if symbol not in self.on_new_candle_callbacks:
                self.on_new_candle_callbacks[symbol] = []
            self.on_new_candle_callbacks[symbol].append(on_new_candle)
        
        # Start polling thread
        thread = threading.Thread(
            target=self._polling_loop,
            args=(symbol, timeframe),
            daemon=True,
            name=f"LiveDataFeed-{symbol}"
        )
        thread.start()
        self.threads[symbol] = thread
        
        logger.info(f"✅ Started live feed: {symbol} ({timeframe})")
        return True
    
    def stop_feed(self, symbol: str) -> bool:
        """
        Stop live data feed for symbol.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            bool: True if stopped successfully
        """
        if symbol not in self.running or not self.running[symbol]:
            logger.warning(f"Feed not running for {symbol}")
            return False
        
        # Set stop flag
        self.running[symbol] = False
        self.stop_flags[symbol].set()
        
        # Wait for thread to finish (timeout 5 seconds)
        if symbol in self.threads:
            self.threads[symbol].join(timeout=5.0)
        
        # Cleanup
        self.threads.pop(symbol, None)
        self.stop_flags.pop(symbol, None)
        self.on_new_candle_callbacks.pop(symbol, None)
        
        logger.info(f"⏹ Stopped live feed: {symbol}")
        return True
    
    def stop_all_feeds(self):
        """Stop all running feeds."""
        symbols = list(self.running.keys())
        for symbol in symbols:
            self.stop_feed(symbol)
        
        logger.info("All feeds stopped")
    
    # ========================================================================
    # POLLING LOOP (PRIVATE)
    # ========================================================================
    
    def _polling_loop(self, symbol: str, timeframe: str):
        """
        Main polling loop that runs every 60 seconds.
        
        Args:
            symbol: Trading symbol
            timeframe: Candle timeframe
        """
        logger.info(f"Polling loop started for {symbol}")
        
        polling_interval = self.config['polling_interval_seconds']
        retry_attempts = self.config['retry_attempts']
        
        # Initial fetch (get recent history)
        self._initial_fetch(symbol, timeframe)
        
        while self.running.get(symbol, False):
            try:
                # Fetch recent candles
                success = self._fetch_recent_candles(symbol, timeframe, retry_attempts)
                
                if success:
                    self.fetch_counts[symbol] += 1
                    self.last_fetch_times[symbol] = datetime.now()
                else:
                    self.error_counts[symbol] += 1
                
                # Wait for next poll (or stop signal)
                if self.stop_flags[symbol].wait(timeout=polling_interval):
                    break  # Stop signal received
                
            except Exception as e:
                logger.error(f"Error in polling loop for {symbol}: {e}")
                self.error_counts[symbol] += 1
                time.sleep(5)  # Brief pause on error
        
        logger.info(f"Polling loop stopped for {symbol}")
    
    def _initial_fetch(self, symbol: str, timeframe: str):
        """
        Initial fetch of historical data to populate buffer.
        
        Args:
            symbol: Trading symbol
            timeframe: Candle timeframe
        """
        try:
            # Calculate date range (last 2 hours for 1min, last 2 days for 1h)
            if timeframe == '1min':
                start_date = datetime.now() - timedelta(hours=2)
            elif timeframe == '5min':
                start_date = datetime.now() - timedelta(hours=10)
            elif timeframe == '1h':
                start_date = datetime.now() - timedelta(days=2)
            else:
                start_date = datetime.now() - timedelta(hours=24)
            
            end_date = datetime.now()
            
            logger.info(f"Initial fetch for {symbol}: {start_date} to {end_date}")
            
            # Fetch data
            df = self.handler.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe
            )
            
            if not df.empty:
                # Convert DataFrame to list of candle dicts
                candles = df.reset_index().to_dict('records')
                
                with self.buffer_locks[symbol]:
                    for candle in candles:
                        self.buffers[symbol].append(candle)
                        self.latest_candles[symbol] = candle
                
                logger.info(f"✅ Initial fetch complete: {symbol}, {len(candles)} candles")
            else:
                logger.warning(f"No data in initial fetch for {symbol}")
                
        except Exception as e:
            logger.error(f"Initial fetch failed for {symbol}: {e}")
    
    def _fetch_recent_candles(self, symbol: str, timeframe: str, max_retries: int) -> bool:
        """
        Fetch recent candles with retry logic.
        
        Args:
            symbol: Trading symbol
            timeframe: Candle timeframe
            max_retries: Maximum retry attempts
        
        Returns:
            bool: True if successful
        """
        retry_delay = self.config['retry_delay_seconds']
        
        for attempt in range(max_retries):
            try:
                # Fetch last 100 candles (ensures we catch any gaps)
                if timeframe == '1min':
                    start_date = datetime.now() - timedelta(hours=2)
                elif timeframe == '5min':
                    start_date = datetime.now() - timedelta(hours=10)
                elif timeframe == '1h':
                    start_date = datetime.now() - timedelta(days=5)
                else:
                    start_date = datetime.now() - timedelta(hours=24)
                
                end_date = datetime.now()
                
                # ✅ LOG FETCH ATTEMPT
                logger.info(f"🔄 Polling fetch for {symbol}: {start_date.strftime('%H:%M')} to {end_date.strftime('%H:%M')}")
                
                df = self.handler.get_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe=timeframe
                )
                
                if not df.empty:
                    # ✅ LOG DATA RECEIVED
                    logger.info(f"📥 Received {len(df)} candles from API for {symbol}")
                    
                    # Get new candles only (not already in buffer)
                    new_candles = self._filter_new_candles(symbol, df)
                    
                    if new_candles:
                        with self.buffer_locks[symbol]:
                            for candle in new_candles:
                                self.buffers[symbol].append(candle)
                                self.latest_candles[symbol] = candle
                        
                        logger.info(f"✅ Fetched {len(new_candles)} new candles for {symbol}")
                        
                        # Trigger callbacks
                        self._trigger_callbacks(symbol, new_candles)
                    else:
                        logger.info(f"⏭ No new candles for {symbol} (data already up-to-date)")
                    
                    return True
                else:
                    logger.warning(f"⚠️ Empty data returned for {symbol}")
                    return False
                    
            except Exception as e:
                logger.warning(f"❌ Fetch attempt {attempt+1}/{max_retries} failed for {symbol}: {e}")
                
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
        
        logger.error(f"🚫 All fetch attempts failed for {symbol}")
        return False
    
    def _filter_new_candles(self, symbol: str, df: pd.DataFrame) -> List[Dict]:
        """
        Filter out candles that are already in buffer.
        
        Args:
            symbol: Trading symbol
            df: DataFrame with new candles
        
        Returns:
            List of new candle dicts
        """
        if symbol not in self.latest_candles:
            # First fetch - all candles are new
            logger.info(f"🆕 First fetch for {symbol}: {len(df)} candles")
            return df.reset_index().to_dict('records')
        
        latest = self.latest_candles[symbol]
        
        # Get timestamp of latest candle
        if 'timestamp' in latest:
            latest_time = latest['timestamp']
        elif 'date' in latest:
            latest_time = latest['date']
        else:
            # Fallback - return all candles
            logger.warning(f"⚠️ No timestamp in latest candle for {symbol}")
            return df.reset_index().to_dict('records')
        
        # Filter candles newer than latest
        df_reset = df.reset_index()
        
        if 'timestamp' in df_reset.columns:
            time_col = 'timestamp'
        elif 'date' in df_reset.columns:
            time_col = 'date'
        else:
            time_col = df_reset.index.name
        
        # ✅ ADD COMPREHENSIVE LOGGING
        logger.info(f"🔍 Filter check for {symbol}:")
        logger.info(f"   Latest time in buffer: {latest_time}")
        if not df_reset.empty:
            logger.info(f"   New data range: {df_reset[time_col].min()} to {df_reset[time_col].max()}")
            logger.info(f"   Total new rows fetched: {len(df_reset)}")
        else:
            logger.warning(f"   ⚠️ DataFrame is empty!")
        
        # Filter for candles strictly after latest_time
        new_df = df_reset[df_reset[time_col] > latest_time]
        
        # ✅ LOG FILTERING RESULT
        logger.info(f"   Filtered to: {len(new_df)} new candles")
        
        if not new_df.empty:
            logger.info(f"✅ Found {len(new_df)} new candles for {symbol}")
            # Log details of new candles
            for idx, candle in new_df.iterrows():
                logger.info(f"   📊 New candle: {candle[time_col]} | Close: {candle.get('close', 'N/A')}")
        else:
            logger.info(f"⏸ No new candles for {symbol} (all data already in buffer)")
        
        return new_df.to_dict('records')
    
    def _trigger_callbacks(self, symbol: str, new_candles: List[Dict]):
        """
        Trigger callbacks for new candles.
        
        Args:
            symbol: Trading symbol
            new_candles: List of new candle dicts
        """
        if symbol not in self.on_new_candle_callbacks:
            return
        
        for candle in new_candles:
            for callback in self.on_new_candle_callbacks[symbol]:
                try:
                    callback(symbol, candle)
                except Exception as e:
                    logger.error(f"Callback error for {symbol}: {e}")
    
    # ========================================================================
    # DATA RETRIEVAL
    # ========================================================================
    
    def get_recent_candles(self, symbol: str, count: int = 100) -> List[Dict]:
        """
        Get recent candles from buffer.
        
        Args:
            symbol: Trading symbol
            count: Number of candles to retrieve
        
        Returns:
            List of candle dicts (newest last)
        
        Example:
            >>> candles = manager.get_recent_candles('BTC/USDT', 50)
            >>> latest = candles[-1]
            >>> print(f"Latest close: {latest['close']}")
        """
        if symbol not in self.buffers:
            logger.warning(f"No buffer for {symbol}")
            return []
        
        with self.buffer_locks[symbol]:
            buffer_list = list(self.buffers[symbol])
        
        return buffer_list[-count:] if len(buffer_list) > count else buffer_list
    
    def get_latest_candle(self, symbol: str) -> Optional[Dict]:
        """
        Get the most recent candle.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Latest candle dict or None
        """
        return self.latest_candles.get(symbol)
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get latest close price.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Latest close price or None
        """
        candle = self.get_latest_candle(symbol)
        return candle['close'] if candle and 'close' in candle else None
    
    def get_candles_as_dataframe(self, symbol: str, count: int = 100) -> pd.DataFrame:
        """
        Get recent candles as pandas DataFrame.
        
        Args:
            symbol: Trading symbol
            count: Number of candles
        
        Returns:
            DataFrame with OHLCV data
        """
        candles = self.get_recent_candles(symbol, count)
        
        if not candles:
            return pd.DataFrame()
        
        df = pd.DataFrame(candles)
        
        # Set timestamp as index
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
        elif 'date' in df.columns:
            df.set_index('date', inplace=True)
        
        return df
    
    # ========================================================================
    # STATISTICS & MONITORING
    # ========================================================================
    
    def get_feed_status(self, symbol: str) -> Dict[str, Any]:
        """
        Get status information for a feed.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Dict with status information
        """
        return {
            'symbol': symbol,
            'running': self.running.get(symbol, False),
            'buffer_size': len(self.buffers.get(symbol, [])),
            'fetch_count': self.fetch_counts.get(symbol, 0),
            'error_count': self.error_counts.get(symbol, 0),
            'last_fetch': self.last_fetch_times.get(symbol),
            'latest_candle': self.latest_candles.get(symbol),
            'has_data': symbol in self.buffers and len(self.buffers[symbol]) > 0
        }
    
    def get_all_feed_statuses(self) -> Dict[str, Dict]:
        """Get status for all active feeds."""
        return {
            symbol: self.get_feed_status(symbol)
            for symbol in self.running.keys()
        }


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    # Import after conditional to avoid dependency issues
    import sys
    sys.path.append('..')
    
    from handlers.unified_data_handler import UnifiedDataHandler, get_unified_handler
    
    print("=" * 70)
    print("LIVE DATA MANAGER TEST")
    print("=" * 70)
    
    # Initialize handler (using crypto for test)
    print("\n1️⃣ Initializing unified handler...")
    handler = get_unified_handler(UnifiedDataHandler.MARKET_CRYPTO_BINANCE)
    
    # Create live data manager
    print("\n2️⃣ Creating live data manager...")
    manager = LiveDataManager(handler)
    
    # Define callback
    def on_new_candle(symbol, candle):
        print(f"   🔔 New candle for {symbol}: Close=${candle.get('close', 0):.2f}")
    
    # Start feed
    print("\n3️⃣ Starting live feed for BTC/USDT...")
    manager.start_feed('BTC/USDT', '1min', on_new_candle=on_new_candle)
    
    # Wait and monitor
    print("\n4️⃣ Monitoring for 180 seconds (3 minutes)...")
    print("   Press Ctrl+C to stop early")
    
    try:
        for i in range(6):  # 6 x 30 seconds = 3 minutes
            time.sleep(30)
            
            status = manager.get_feed_status('BTC/USDT')
            print(f"\n   📊 Status after {(i+1)*30}s:")
            print(f"      Fetches: {status['fetch_count']}, Errors: {status['error_count']}")
            print(f"      Buffer Size: {status['buffer_size']} candles")
            print(f"      Latest Price: ${manager.get_latest_price('BTC/USDT'):,.2f}")
            
    except KeyboardInterrupt:
        print("\n\n⏸ Interrupted by user")
    
    # Stop feed
    print("\n5️⃣ Stopping feed...")
    manager.stop_feed('BTC/USDT')
    
    # Get final candles
    print("\n6️⃣ Retrieving final candles...")
    candles = manager.get_recent_candles('BTC/USDT', 10)
    print(f"   Retrieved {len(candles)} candles")
    if candles:
        print(f"   Last candle: Close=${candles[-1].get('close', 0):.2f}")
    
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE")
    print("=" * 70)
