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
    "market_open": dt_time(9, 15),   # 9:15 AM IST
    "market_close": dt_time(15, 30),  # 3:30 PM IST
    "currency": "₹"
}

# ============================================================================
# TRADING CONFIGURATION
# ============================================================================
TRADING_CONFIG = {
    "mode": "INTRADAY",
    "product_type": "MIS",  # Margin Intraday Square-off
    "exchange": "NSE",
    "segment": ["EQ", "NFO"],  # Equity and Futures & Options
}

# ============================================================================
# STREAMING CONFIGURATION
# ============================================================================
# Kite Connect allows max 3000 instruments in WebSocket
# But for optimal performance, recommended is 200-300
STREAMING_CONFIG = {
    "max_instruments": 200,
    "mode": "full",  # full, quote, or ltp
    "reconnect_attempts": 5,
    "reconnect_delay": 3  # seconds
}

# ============================================================================
# INDEX OPTIONS CONFIGURATION
# ============================================================================
# Note: These are reference indices only
# Actual lot sizes, tick sizes will be fetched dynamically from Kite API
INDEX_OPTIONS_REFERENCE = {
    "NIFTY": "NIFTY 50",
    "BANKNIFTY": "NIFTY BANK",
    "FINNIFTY": "NIFTY FIN SERVICE",
    "MIDCPNIFTY": "NIFTY MID SELECT"
}

# Cache for dynamic lot sizes (will be populated from Kite API)
_INDEX_LOT_SIZES_CACHE = {}

def get_index_config(index_name: str) -> dict:
    """
    Get index configuration dynamically from cache.
    This will be populated by kite_handler after fetching instruments.
    """
    return _INDEX_LOT_SIZES_CACHE.get(index_name, {
        "symbol": index_name,
        "lot_size": None,  # To be fetched
        "exchange": "NFO",
        "tick_size": None  # To be fetched
    })

def update_index_config(index_data: dict):
    """
    Update index configuration with live data from Kite API.
    Called by kite_handler after fetching instruments.
    """
    global _INDEX_LOT_SIZES_CACHE
    _INDEX_LOT_SIZES_CACHE.update(index_data)

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
    
    return errors

def get_market_status():
    """Check if market is currently open"""
    import datetime
    
    tz = pytz.timezone(MARKET_CONFIG["timezone"])
    now = datetime.datetime.now(tz)
    
    # Check if weekend
    if now.weekday() >= 5:  # Saturday or Sunday
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
