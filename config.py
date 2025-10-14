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
MARKET_CONFIG = {
    "timezone": "Asia/Kolkata",
    "market_open": dt_time(9, 15),
    "market_close": dt_time(15, 30),
    "currency": "₹"
}

# ============================================================================
# TRADING CONFIGURATION
# ============================================================================
TRADING_CONFIG = {
    "mode": "INTRADAY",
    "product_type": os.getenv("PRODUCT_TYPE", "MIS"),
    "exchange": os.getenv("DEFAULT_EXCHANGE", "NSE"),
    "segment": ["EQ", "NFO"],
}

# ============================================================================
# STREAMING CONFIGURATION
# ============================================================================
STREAMING_CONFIG = {
    "max_instruments": int(os.getenv("MAX_STREAMING_INSTRUMENTS", "200")),
    "mode": os.getenv("STREAMING_MODE", "full"),
    "reconnect_attempts": int(os.getenv("RECONNECT_ATTEMPTS", "5")),
    "reconnect_delay": int(os.getenv("RECONNECT_DELAY_SECONDS", "3"))
}

# ============================================================================
# SUPPORTED EXCHANGES (Will be verified dynamically)
# ============================================================================
SUPPORTED_EXCHANGES = ["NSE", "NFO", "BSE", "BFO", "MCX", "CDS"]
# These are attempted during initialization
# Only the ones that successfully load will be available


# ============================================================================
# VALIDATION
# ============================================================================
def validate_config():
    """Validate essential configuration"""
    errors = []
    
    if not KITE_API_KEY:
        errors.append("KITE_API_KEY is missing in .env or Streamlit secrets")
    if not KITE_API_SECRET:
        errors.append("KITE_API_SECRET is missing in .env or Streamlit secrets")
    if not KITE_ACCESS_TOKEN:
        errors.append("KITE_ACCESS_TOKEN is missing in .env or Streamlit secrets")
    
    try:
        if STREAMING_CONFIG["max_instruments"] > 3000:
            errors.append("MAX_STREAMING_INSTRUMENTS cannot exceed 3000")
    except (ValueError, TypeError):
        errors.append("MAX_STREAMING_INSTRUMENTS must be a valid number")
    
    return errors

def get_market_status():
    """Check if market is currently open"""
    import datetime
    
    tz = pytz.timezone(MARKET_CONFIG["timezone"])
    now = datetime.datetime.now(tz)
    
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
    """Return configuration summary"""
    try:
        from cache_utils import get_cache_summary
        cache_info = get_cache_summary()
    except:
        cache_info = {}
    
    return {
        "database": DB_NAME,
        "streaming": STREAMING_CONFIG,
        "trading": TRADING_CONFIG,
        "market": MARKET_CONFIG,
        **cache_info
    }
