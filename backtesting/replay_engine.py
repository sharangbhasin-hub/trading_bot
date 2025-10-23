"""
Replay Engine - Walk-Forward Simulator
Simulates live market conditions from historical data
"""
import pandas as pd
from datetime import datetime, timedelta
import logging

from backtesting.config import BacktestConfig

logger = logging.getLogger(__name__)


class ReplayEngine:
    """
    Replays historical market data as if it were live
    Ensures zero lookahead bias
    """
    
    def __init__(self, historical_data):
        """
        Initialize replay engine
        
        Args:
            historical_data: Dict from DataLoader with all historical data
        """
        self.historical_data = historical_data
        self.data = historical_data
        self.config = BacktestConfig()
        self.current_date = None
        self.current_time = None
        self.current_data_cache = {}
        
    def get_trading_dates(self):
        """Get list of all trading dates"""
        return self.historical_data['dates']
    
    def set_current_timestamp(self, date, time):
        """
        Set the current replay timestamp
        
        Args:
            date: Date string 'YYYY-MM-DD'
            time: Time string 'HH:MM'
        """
        self.current_date = date
        self.current_time = time
        
        # Clear cache when date changes
        if date not in self.current_data_cache:
            self.current_data_cache = {}
    
    def get_data_upto_timestamp(self, timeframe, lookback_candles=None):
        """Get data up to current timestamp with timeframe mapping"""
        if not self.current_date or not self.current_time:
            return pd.DataFrame()
        
        # ✅ CORRECT FIX: Map from what backtest_runner requests to what data_loader stored
        # backtest_runner requests: '5min', '15min', '1h', 'daily'
        # These are ALREADY correct - no mapping needed!
        actual_timeframe = timeframe  # Use as-is
        
        # Check if date exists in data
        if self.current_date not in self.data['data']:
            logger.warning(f"Date {self.current_date} not in data")
            return pd.DataFrame()
        
        # Check if timeframe exists for this date
        available_keys = list(self.data['data'][self.current_date].keys())
        if actual_timeframe not in available_keys:
            logger.warning(f"Timeframe '{actual_timeframe}' not available for {self.current_date}")
            logger.warning(f"Available: {available_keys}")
            return pd.DataFrame()
        
        # Get DataFrame for this timeframe
        df = self.data['data'][self.current_date][actual_timeframe].copy()
        
        if df.empty:
            return df
        
        # Create timezone-aware datetime if needed
        current_datetime = datetime.strptime(f"{self.current_date} {self.current_time}", "%Y-%m-%d %H:%M")
        
        # Make datetime timezone-aware to match the data
        if df.index.tz is not None:
            import pytz
            ist = pytz.timezone('Asia/Kolkata')
            current_datetime = ist.localize(current_datetime)
        
        # Filter up to current time
        df = df[df.index <= current_datetime]
        
        # Apply lookback if specified
        if lookback_candles and len(df) > lookback_candles:
            df = df.tail(lookback_candles)
        
        return df

    def get_current_spot_price(self):
        """
        Get current spot price at current timestamp
        
        Returns:
            Float: Current spot price (close of most recent candle)
        """
        df_5min = self.get_data_upto_timestamp('5min', lookback_candles=1)
        
        if df_5min.empty:
            return None
        
        return df_5min['close'].iloc[-1]
    
    def get_current_candle(self, timeframe='5min'):
        """
        Get the most recent completed candle
        
        Args:
            timeframe: Timeframe to get candle from
        
        Returns:
            Series: OHLC data of current candle
        """
        df = self.get_data_upto_timestamp(timeframe, lookback_candles=1)
        
        if df.empty:
            return None
        
        return df.iloc[-1]
    
    def iterate_timestamps(self, date):
        """
        Generator that yields timestamps for a given date
        
        Args:
            date: Date string 'YYYY-MM-DD'
        
        Yields:
            Time string 'HH:MM' at configured intervals
        """
        start_time = datetime.strptime(self.config.MARKET_OPEN_TIME, "%H:%M")
        end_time = datetime.strptime(self.config.MARKET_CLOSE_TIME, "%H:%M")
        interval = timedelta(minutes=self.config.REPLAY_INTERVAL_MINUTES)
        
        current = start_time
        
        while current <= end_time:
            yield current.strftime("%H:%M")
            current += interval
    
    def get_support_resistance(self, df_15min=None):
        """
        Calculate support and resistance levels
        Uses same logic as main system
        
        Args:
            df_15min: 15-minute DataFrame (optional, will fetch if not provided)
        
        Returns:
            Tuple: (support, resistance)
        """
        if df_15min is None:
            df_15min = self.get_data_upto_timestamp('15min')
        
        if df_15min.empty or len(df_15min) < 20:
            # Not enough data, use simple min/max
            support = df_15min['low'].min() if not df_15min.empty else 0
            resistance = df_15min['high'].max() if not df_15min.empty else 0
            return support, resistance
        
        # Use recent data for S/R calculation
        lookback = min(50, len(df_15min))
        recent = df_15min.tail(lookback)
        current_price = recent['close'].iloc[-1]
        
        # Find pivot highs (resistance candidates)
        pivot_highs = []
        for i in range(5, len(recent) - 5):
            high = recent['high'].iloc[i]
            if (high == recent['high'].iloc[i-5:i].max() and 
                high == recent['high'].iloc[i+1:i+6].max()):
                pivot_highs.append(high)
        
        # Find pivot lows (support candidates)
        pivot_lows = []
        for i in range(5, len(recent) - 5):
            low = recent['low'].iloc[i]
            if (low == recent['low'].iloc[i-5:i].min() and 
                low == recent['low'].iloc[i+1:i+6].min()):
                pivot_lows.append(low)
        
        # Get nearest support below current price
        support_candidates = [s for s in pivot_lows if s < current_price]
        support = max(support_candidates) if support_candidates else recent['low'].min()
        
        # Get nearest resistance above current price
        resistance_candidates = [r for r in pivot_highs if r > current_price]
        resistance = min(resistance_candidates) if resistance_candidates else recent['high'].max()
        
        return support, resistance
    
    def is_market_hours(self, time_str):
        """Check if given time is within market hours"""
        time = datetime.strptime(time_str, "%H:%M").time()
        market_open = datetime.strptime(self.config.MARKET_OPEN_TIME, "%H:%M").time()
        market_close = datetime.strptime(self.config.MARKET_CLOSE_TIME, "%H:%M").time()
        
        return market_open <= time <= market_close
    
    def is_eod_close_time(self, time_str):
        """Check if it's time to close all positions (end of day)"""
        eod_time = datetime.strptime(self.config.EOD_CLOSE_TIME, "%H:%M").time()
        current_time = datetime.strptime(time_str, "%H:%M").time()
        
        return current_time >= eod_time

    def calculate_atr(self, timeframe='5min', period=14):
        """
        Calculate Average True Range (ATR)
        
        Args:
            timeframe: Timeframe to calculate ATR on ('5min', '15min', etc.)
            period: ATR period (default: 14)
        
        Returns:
            float: Current ATR value, or None if insufficient data
        """
        df = self.get_data_upto_timestamp(timeframe, lookback_candles=period + 10)
        
        if df is None or df.empty or len(df) < period:
            logger.warning(f"Insufficient data for ATR calculation: {len(df) if not df.empty else 0} candles")
            return None
        
        # Calculate True Range
        df = df.copy()
        df['h-l'] = df['high'] - df['low']
        df['h-pc'] = abs(df['high'] - df['close'].shift(1))
        df['l-pc'] = abs(df['low'] - df['close'].shift(1))
        
        df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        
        # Calculate ATR using EMA
        atr = df['tr'].ewm(span=period, adjust=False).mean().iloc[-1]
        
        logger.debug(f"ATR({period}) on {timeframe}: {atr:.2f}")
        
        return atr
    
    def get_atr_multiplier_for_signal(self, signal_type, confidence):
        """
        Get ATR multiplier based on signal strength
        
        Args:
            signal_type: 'CALL' or 'PUT'
            confidence: Signal confidence (0-100)
        
        Returns:
            float: ATR multiplier for stop loss
        """
        # Higher confidence = tighter stop (lower multiplier)
        # Lower confidence = wider stop (higher multiplier)
        
        if confidence >= 70:
            return 4.0  # Tight stop for high confidence
        elif confidence >= 50:
            return 5.0  # Medium stop for medium confidence
        else:
            return 6.0  # Wide stop for low confidence
    
    def get_volatility_adjusted_rr(self, atr, entry_price):
        """
        Calculate volatility-adjusted risk/reward ratio
        
        Args:
            atr: Current ATR value
            entry_price: Entry price
        
        Returns:
            float: Suggested R:R ratio based on volatility
        """
        # Calculate ATR as percentage of entry price
        atr_percent = (atr / entry_price) * 100
        
        # Higher volatility = wider stops = better R:R needed
        if atr_percent > 1.5:
            return 2.0  # High volatility, need 1:2 R:R
        elif atr_percent > 0.8:
            return 1.5  # Medium volatility, 1:1.5 R:R
        else:
            return 1.2  # Low volatility, 1:1.2 R:R acceptable
