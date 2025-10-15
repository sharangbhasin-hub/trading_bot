"""
Technical Indicators for Intraday Options Trading
Optimized settings based on proven strategies
"""

import pandas as pd
import numpy as np

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
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Avoid division by zero
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50)  # Neutral RSI for NaN values

def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3) -> tuple:
    """
    Calculate Supertrend Indicator
    Best for intraday trend detection
    Returns: (supertrend_values, direction_series)
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    # Calculate ATR (Average True Range)
    tr = np.zeros(len(df))
    for i in range(1, len(df)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    atr = pd.Series(tr).rolling(window=period).mean().values
    
    # Calculate basic bands
    hl_avg = (high + low) / 2
    upper_band = hl_avg + (multiplier * atr)
    lower_band = hl_avg - (multiplier * atr)
    
    # Initialize supertrend and direction
    supertrend = np.zeros(len(df))
    direction = np.zeros(len(df), dtype=int)
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(df)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower_band[i]
            direction[i] = 1  # Uptrend
        else:
            supertrend[i] = upper_band[i]
            direction[i] = -1  # Downtrend
    
    return pd.Series(supertrend, index=df.index), pd.Series(direction, index=df.index)

def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """
    Calculate MACD (Moving Average Convergence Divergence)
    Returns: (macd_line, signal_line, histogram)
    """
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2) -> tuple:
    """
    Calculate Bollinger Bands
    Returns: (upper_band, middle_band, lower_band)
    """
    middle_band = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    upper_band = middle_band + (std_dev * std)
    lower_band = middle_band - (std_dev * std)
    
    return upper_band, middle_band, lower_band

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR)
    Used for volatility measurement and stop-loss placement
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    tr = np.zeros(len(df))
    for i in range(1, len(df)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    atr = pd.Series(tr).rolling(window=period).mean()
    return atr
