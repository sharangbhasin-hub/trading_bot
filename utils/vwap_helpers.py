"""
VWAP Strategy Helper Functions
===============================
Utility functions for VWAP strategies.

Author: Trading System
Date: November 18, 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time as dt_time, timedelta
import pytz
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# TIME & DATE UTILITIES
# ============================================================================

def is_market_open(current_time: datetime = None) -> bool:
    """
    Check if market is currently open.
    Indian market hours: 9:15 AM - 3:30 PM IST
    
    Args:
        current_time: Time to check (default: now)
    
    Returns:
        bool: True if market is open
    """
    if current_time is None:
        ist = pytz.timezone('Asia/Kolkata')
        current_time = datetime.now(ist)
    
    # Check if weekday (Mon-Fri)
    if current_time.weekday() >= 5:  # 5=Saturday, 6=Sunday
        return False
    
    # Check time
    market_open = dt_time(9, 15)
    market_close = dt_time(15, 30)
    
    current = current_time.time()
    
    return market_open <= current <= market_close

def get_930am_candle_time() -> dt_time:
    """Get 9:30 AM time object"""
    return dt_time(9, 30)

def is_within_entry_window(current_time: datetime,
                           strategy_type: str) -> bool:
    """
    Check if current time is within entry window for strategy.
    
    Args:
        current_time: Current time
        strategy_type: 'SELLING' or 'BUYING'
    
    Returns:
        bool: True if within window
    """
    time = current_time.time()
    
    if strategy_type == 'SELLING':
        # 9:30 AM - 10:30 AM
        return dt_time(9, 30) <= time <= dt_time(10, 30)
    else:  # BUYING
        # 9:30 AM - 11:00 AM
        return dt_time(9, 30) <= time <= dt_time(11, 0)

def calculate_hold_duration(entry_time: datetime,
                           exit_time: datetime) -> Dict:
    """
    Calculate trade hold duration.
    
    Args:
        entry_time: Entry timestamp
        exit_time: Exit timestamp
    
    Returns:
        dict: Duration breakdown
    """
    duration = exit_time - entry_time
    
    total_seconds = duration.total_seconds()
    minutes = int(total_seconds / 60)
    hours = minutes / 60
    
    return {
        'total_seconds': total_seconds,
        'total_minutes': minutes,
        'hours': round(hours, 2),
        'formatted': f"{int(hours)}h {minutes % 60}m"
    }

# ============================================================================
# PRICE & PREMIUM UTILITIES
# ============================================================================

def calculate_premium_change_pct(entry_premium: float,
                                 exit_premium: float,
                                 strategy_type: str = 'SELLING') -> float:
    """
    Calculate % change in premium.
    
    Args:
        entry_premium: Entry premium
        exit_premium: Exit premium
        strategy_type: 'SELLING' or 'BUYING'
    
    Returns:
        float: % change (positive = profit for selling, negative = loss)
    """
    if strategy_type == 'SELLING':
        # For selling: lower premium = profit
        change_pct = ((entry_premium - exit_premium) / entry_premium) * 100
    else:
        # For buying: higher premium = profit
        change_pct = ((exit_premium - entry_premium) / entry_premium) * 100
    
    return change_pct

def calculate_points_to_rupees(points: float, lot_size: int = 25) -> float:
    """
    Convert points to rupees.
    
    Args:
        points: Number of points
        lot_size: Lot size
    
    Returns:
        float: Amount in ₹
    """
    return points * lot_size

def calculate_rupees_to_points(amount: float, lot_size: int = 25) -> float:
    """
    Convert rupees to points.
    
    Args:
        amount: Amount in ₹
        lot_size: Lot size
    
    Returns:
        float: Number of points
    """
    return amount / lot_size

# ============================================================================
# STRIKE SELECTION UTILITIES
# ============================================================================

def round_to_nearest_strike(price: float, index: str = 'NIFTY') -> int:
    """
    Round price to nearest strike.
    
    Args:
        price: Spot price
        index: Index name
    
    Returns:
        int: Nearest strike
    """
    strike_gaps = {
        'NIFTY': 50,
        'BANKNIFTY': 100,
        'SENSEX': 100,
        'FINNIFTY': 50
    }
    
    gap = strike_gaps.get(index, 50)
    return int(round(price / gap) * gap)

def calculate_strike_from_spot(spot_price: float,
                               strikes_offset: int,
                               option_type: str,
                               index: str = 'NIFTY') -> int:
    """
    Calculate option strike from spot price.
    
    Args:
        spot_price: Current spot price
        strikes_offset: Number of strikes away (e.g., 2)
        option_type: 'CE' or 'PE'
        index: Index name
    
    Returns:
        int: Strike price
    """
    atm = round_to_nearest_strike(spot_price, index)
    
    strike_gaps = {
        'NIFTY': 50,
        'BANKNIFTY': 100,
        'SENSEX': 100,
        'FINNIFTY': 50
    }
    
    gap = strike_gaps.get(index, 50)
    
    if option_type == 'CE':
        return atm + (strikes_offset * gap)
    else:  # PE
        return atm - (strikes_offset * gap)

# ============================================================================
# MARKET CONDITION UTILITIES
# ============================================================================

def classify_gap_open(current_price: float,
                     previous_close: float) -> Dict:
    """
    Classify gap at market open.
    
    Args:
        current_price: Current price
        previous_close: Previous day's close
    
    Returns:
        dict: Gap classification
    """
    gap_pct = ((current_price - previous_close) / previous_close) * 100
    
    if abs(gap_pct) < 0.3:
        gap_type = 'FLAT'
    elif abs(gap_pct) < 0.5:
        gap_type = 'SMALL'
    elif abs(gap_pct) < 1.0:
        gap_type = 'MEDIUM'
    else:
        gap_type = 'LARGE'
    
    direction = 'UP' if gap_pct > 0 else 'DOWN' if gap_pct < 0 else 'NONE'
    
    return {
        'gap_pct': gap_pct,
        'gap_type': gap_type,
        'direction': direction,
        'is_significant': abs(gap_pct) >= 0.5
    }

def calculate_daily_range_pct(high: float, low: float, close: float) -> float:
    """
    Calculate daily range as % of close.
    
    Args:
        high: Day's high
        low: Day's low
        close: Day's close
    
    Returns:
        float: Range %
    """
    range_val = high - low
    return (range_val / close) * 100 if close > 0 else 0

# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_premium_data(premium: float, label: str = "Premium") -> bool:
    """
    Validate premium value.
    
    Args:
        premium: Premium value to validate
        label: Label for logging
    
    Returns:
        bool: True if valid
    """
    if premium is None:
        logger.error(f"{label} is None")
        return False
    
    if premium <= 0:
        logger.error(f"{label} is <= 0: {premium}")
        return False
    
    if premium > 10000:  # Sanity check
        logger.warning(f"{label} seems unusually high: {premium}")
    
    return True

def validate_strike_selection(strikes: Dict) -> bool:
    """
    Validate strike selection dictionary.
    
    Args:
        strikes: Strike selection dict
    
    Returns:
        bool: True if valid
    """
    required_fields_selling = [
        'sell_ce_strike', 'sell_pe_strike',
        'buy_ce_strike', 'buy_pe_strike'
    ]
    
    required_fields_buying = [
        'ce_strike', 'pe_strike'
    ]
    
    if 'strategy_type' not in strikes:
        logger.error("strategy_type missing from strikes")
        return False
    
    if strikes['strategy_type'] == 'SELLING':
        required = required_fields_selling
    else:
        required = required_fields_buying
    
    for field in required:
        if field not in strikes:
            logger.error(f"Required field '{field}' missing from strikes")
            return False
    
    return True

# ============================================================================
# FORMATTING UTILITIES
# ============================================================================

def format_currency(amount: float) -> str:
    """
    Format amount as Indian currency.
    
    Args:
        amount: Amount in ₹
    
    Returns:
        str: Formatted string
    """
    if amount >= 0:
        return f"₹{amount:,.2f}"
    else:
        return f"-₹{abs(amount):,.2f}"

def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format value as percentage.
    
    Args:
        value: Value
        decimals: Decimal places
    
    Returns:
        str: Formatted percentage
    """
    if value >= 0:
        return f"+{value:.{decimals}f}%"
    else:
        return f"{value:.{decimals}f}%"

def format_trade_summary(trade_result: Dict) -> str:
    """
    Format trade result as human-readable summary.
    
    Args:
        trade_result: Trade result dict
    
    Returns:
        str: Formatted summary
    """
    lines = [
        f"Trade: {trade_result.get('trade_id', 'N/A')}",
        f"Strategy: {trade_result.get('strategy_type', 'N/A')}",
        f"Entry: {trade_result.get('entry_premium', 0):.1f}",
        f"Exit: {trade_result.get('exit_premium', 0):.1f}",
        f"P&L: {format_currency(trade_result.get('net_pnl', 0))}",
        f"Result: {'✅ WIN' if trade_result.get('net_pnl', 0) > 0 else '❌ LOSS'}"
    ]
    
    return "\n".join(lines)

# ============================================================================
# DATA PROCESSING UTILITIES
# ============================================================================

def resample_to_timeframe(df: pd.DataFrame,
                          timeframe: str = '1min') -> pd.DataFrame:
    """
    Resample tick data to specified timeframe.
    
    Args:
        df: DataFrame with timestamp index
        timeframe: Timeframe ('1min', '5min', '15min', etc.)
    
    Returns:
        pd.DataFrame: Resampled data
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    # Only resample columns that exist
    resample_dict = {k: v for k, v in ohlc_dict.items() if k in df.columns}
    
    resampled = df.resample(timeframe).agg(resample_dict)
    
    return resampled.dropna()

def fill_missing_candles(df: pd.DataFrame,
                         freq: str = '1min') -> pd.DataFrame:
    """
    Fill missing candles in OHLC data.
    
    Args:
        df: OHLC dataframe
        freq: Frequency
    
    Returns:
        pd.DataFrame: Data with filled candles
    """
    df_filled = df.asfreq(freq, method='ffill')
    return df_filled

# ============================================================================
# LOGGING UTILITIES
# ============================================================================

def log_signal_details(signal: Dict, logger_instance: logging.Logger = None):
    """
    Log signal details in formatted way.
    
    Args:
        signal: Signal dictionary
        logger_instance: Logger to use (default: module logger)
    """
    log = logger_instance or logger
    
    log.info("=" * 60)
    log.info(f"SIGNAL: {signal.get('signal_type', 'UNKNOWN')}")
    log.info(f"Strategy: {signal.get('strategy_name', 'N/A')}")
    log.info(f"Confidence: {signal.get('confidence', 0)}%")
    log.info(f"Entry: {signal.get('entry_price', 0):.2f}")
    log.info(f"Stop-Loss: {signal.get('stop_loss', 0):.2f}")
    log.info(f"Target: {signal.get('target', 0):.2f}")
    log.info(f"Reason: {signal.get('entry_reason', 'N/A')}")
    log.info("=" * 60)
