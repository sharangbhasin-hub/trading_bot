import os
from dotenv import load_dotenv
from datetime import time as dt_time
import pytz

load_dotenv()

# ============================================================================
# KITE CONNECT CONFIGURATION
# ============================================================================
KITE_API_KEY = os.getenv("KITE_API_KEY")
KITE_API_SECRET = os.getenv("KITE_API_SECRET")
KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN")

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================
DB_NAME = os.getenv("DB_NAME", "trading_data.db")

# ============================================================================
# MARKET CONFIGURATION - INDIA ONLY
# ============================================================================
# Note: Market timings are standard NSE hours but kept in constants
# for easy modification if NSE changes them
MARKET_CONFIG = {
    "timezone": "Asia/Kolkata",
    "market_open": dt_time(9, 15),
    "market_close": dt_time(15, 30),
    "currency": "₹"
}

# ============================================================================
# TRADING CONFIGURATION (From Environment)
# ============================================================================
TRADING_CONFIG = {
    "mode": "INTRADAY",
    "product_type": os.getenv("PRODUCT_TYPE", "MIS"),
    "exchange": os.getenv("DEFAULT_EXCHANGE", "NSE"),
    "segment": ["EQ", "NFO"],  # NSE Equity and F&O
}

# ============================================================================
# STREAMING CONFIGURATION (From Environment)
# ============================================================================
STREAMING_CONFIG = {
    "max_instruments": int(os.getenv("MAX_STREAMING_INSTRUMENTS", "200")),
    "mode": os.getenv("STREAMING_MODE", "full"),
    "reconnect_attempts": int(os.getenv("RECONNECT_ATTEMPTS", "5")),
    "reconnect_delay": int(os.getenv("RECONNECT_DELAY_SECONDS", "3"))
}

# ============================================================================
# INDEX OPTIONS - REFERENCE NAMES ONLY
# ============================================================================
# These are just reference names for UI display
# All trading parameters (lot size, tick size) fetched from Kite API
INDEX_OPTIONS_REFERENCE = [
    "NIFTY",
    "BANKNIFTY", 
    "FINNIFTY",
    "MIDCPNIFTY"
]

# Cache for dynamic instrument data (populated by kite_handler)
_INSTRUMENTS_CACHE = {}

def get_instrument_config(symbol: str) -> dict:
    """
    Get instrument configuration dynamically from cache.
    Returns None if not yet fetched from Kite API.
    """
    return _INSTRUMENTS_CACHE.get(symbol)

def update_instruments_cache(instruments_data: dict):
    """
    Update instruments cache with live data from Kite API.
    Called by kite_handler after fetching instrument dump.
    """
    global _INSTRUMENTS_CACHE
    _INSTRUMENTS_CACHE.update(instruments_data)

def clear_instruments_cache():
    """Clear instruments cache (useful for refresh)"""
    global _INSTRUMENTS_CACHE
    _INSTRUMENTS_CACHE = {}

# ============================================================================
# VALIDATION
# ============================================================================
def validate_config():
    """Validate essential configuration"""
    errors = []
    
    if not KITE_API_KEY:
        errors.append("KITE_API_KEY is missing in .env file")
    if not KITE_API_SECRET:
        errors.append("KITE_API_SECRET is missing in .env file")
    if not KITE_ACCESS_TOKEN:
        errors.append("KITE_ACCESS_TOKEN is missing in .env file")
    
    # Validate numeric values
    try:
        if STREAMING_CONFIG["max_instruments"] > 3000:
            errors.append("MAX_STREAMING_INSTRUMENTS cannot exceed 3000 (Kite API limit)")
    except (ValueError, TypeError):
        errors.append("MAX_STREAMING_INSTRUMENTS must be a valid number")
    
    return errors

def get_market_status():
    """Check if market is currently open"""
    import datetime
    
    tz = pytz.timezone(MARKET_CONFIG["timezone"])
    now = datetime.datetime.now(tz)
    
    # Check if weekend
    if now.weekday() >= 5:
        return {
            "status": "CLOSED",
            "reason": "Weekend",
            "time": now.strftime("%H:%M:%S %Z")
        }
    
    current_time = now.time()
    market_open = MARKET_CONFIG["market_open"]
    market_close = MARKET_CONFIG["market_close"]
    
    if market_open <= current_time <= market_close:
        return {
            "status": "OPEN",
            "reason": "Market Hours",
            "time": now.strftime("%H:%M:%S %Z")
        }
    else:
        return {
            "status": "CLOSED",
            "reason": "Outside Trading Hours",
            "time": now.strftime("%H:%M:%S %Z")
        }

def get_config_summary():
    """Return configuration summary for debugging"""
    return {
        "database": DB_NAME,
        "streaming": STREAMING_CONFIG,
        "trading": TRADING_CONFIG,
        "market": MARKET_CONFIG,
        "instruments_cached": len(_INSTRUMENTS_CACHE)
    }
