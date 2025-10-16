"""
Technical Indicators for Intraday Options Trading
Optimized settings based on timeframe with proven strategies
"""

import pandas as pd
import numpy as np


def get_indicator_settings(timeframe: str) -> dict:
    """
    Get optimized indicator settings based on timeframe
    
    Args:
        timeframe: '5minute', '15minute', 'hour', 'day'
        
    Returns:
        dict: Indicator settings optimized for the timeframe
    """
    # Map various interval names to standard timeframes
    timeframe_mapping = {
        '5minute': '5minute',
        '15minute': '15minute',
        '60minute': 'hour',
        'hour': 'hour',
        'day': 'day',
        'DAILY': 'day',
        'HOURLY': 'hour',
        '15MIN': '15minute',
        '5MIN': '5minute'
    }
    
    # Normalize timeframe
    normalized_tf = timeframe_mapping.get(timeframe, '15minute')
    
    settings = {
        '5minute': {
            'ema_fast': 9,
            'ema_slow': 21,
            'rsi_period': 9,
            'macd_fast': 5,
            'macd_slow': 13,
            'macd_signal': 5,
            'bb_period': 20,
            'bb_std': 2,
            'atr_period': 14,
            'supertrend_period': 7,
            'supertrend_multiplier': 3,
            'stochastic_k': 8,
            'stochastic_d': 3,
            'sma_period': 50
        },
        '15minute': {
            'ema_fast': 20,
            'ema_slow': 50,
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2,
            'atr_period': 14,
            'supertrend_period': 10,
            'supertrend_multiplier': 3,
            'stochastic_k': 14,
            'stochastic_d': 3,
            'sma_period': 100
        },
        'hour': {
            'ema_fast': 20,
            'ema_slow': 50,
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2,
            'atr_period': 14,
            'supertrend_period': 10,
            'supertrend_multiplier': 3,
            'stochastic_k': 14,
            'stochastic_d': 3,
            'sma_period': 200
        },
        'day': {
            'ema_fast': 50,
            'ema_slow': 200,
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2,
            'atr_period': 14,
            'supertrend_period': 10,
            'supertrend_multiplier': 3,
            'stochastic_k': 14,
            'stochastic_d': 3,
            'sma_period': 200
        }
    }
    
    return settings.get(normalized_tf, settings['15minute'])


def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period, adjust=False).mean()


def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return data.rolling(window=period).mean()


def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index
    RSI > 70: Overbought
    RSI < 30: Oversold
    """
    delta = data.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """
    Calculate MACD with custom periods
    
    Args:
        data: Price series
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
        
    Returns:
        tuple: (macd_line, signal_line, histogram)
    """
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2) -> tuple:
    """
    Calculate Bollinger Bands
    Returns: (upper_band, middle_band, lower_band)
    """
    middle_band = calculate_sma(data, period)
    std = data.rolling(window=period).std()
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return upper_band, middle_band, lower_band


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range
    Essential for volatility measurement and stop-loss placement
    """
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr


def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3) -> tuple:
    """
    Calculate Supertrend Indicator
    Best for intraday trend detection
    
    Returns:
        tuple: (supertrend_values, direction_series)
               direction: 1 = uptrend, -1 = downtrend
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    atr = calculate_atr(df, period).values
    
    hl_avg = (high + low) / 2
    upper_band = hl_avg + (multiplier * atr)
    lower_band = hl_avg - (multiplier * atr)
    
    supertrend = np.zeros(len(close))
    direction = np.zeros(len(close))
    
    supertrend[0] = lower_band[0]
    direction[0] = 1
    
    for i in range(1, len(close)):
        if close[i] > upper_band[i-1]:
            direction[i] = 1
            supertrend[i] = lower_band[i]
        elif close[i] < lower_band[i-1]:
            direction[i] = -1
            supertrend[i] = upper_band[i]
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1:
                supertrend[i] = max(lower_band[i], supertrend[i-1])
            else:
                supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    return pd.Series(supertrend, index=df.index), pd.Series(direction, index=df.index)


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Volume Weighted Average Price
    THE #1 indicator for intraday trading
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    return vwap


def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> tuple:
    """
    Calculate Stochastic Oscillator
    
    Returns:
        tuple: (%K, %D)
    """
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    
    k_percent = 100 * ((df['close'] - low_min) / (high_max - low_min))
    d_percent = k_percent.rolling(window=d_period).mean()
    
    return k_percent, d_percent
