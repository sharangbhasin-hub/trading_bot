"""
Cache utilities for instruments and indices
Prevents circular import issues
"""

# Cache for dynamic indices
_INDICES_CACHE = {}

# Cache for dynamic instrument data
_INSTRUMENTS_CACHE = {}

def get_indices_by_exchange(exchange: str) -> list:
    """Get available indices for an exchange from cache"""
    return _INDICES_CACHE.get(exchange, [])

def update_indices_cache(exchange: str, indices: list):
    """Update indices cache with data from Kite API"""
    global _INDICES_CACHE
    _INDICES_CACHE[exchange] = indices

def get_instrument_config(symbol: str) -> dict:
    """Get instrument configuration dynamically from cache"""
    return _INSTRUMENTS_CACHE.get(symbol)

def update_instruments_cache(instruments_data: dict):
    """Update instruments cache with live data from Kite API"""
    global _INSTRUMENTS_CACHE
    _INSTRUMENTS_CACHE.update(instruments_data)

def clear_instruments_cache():
    """Clear instruments cache"""
    global _INSTRUMENTS_CACHE
    _INSTRUMENTS_CACHE = {}

def get_cache_summary():
    """Get summary of cached data"""
    return {
        "instruments_cached": len(_INSTRUMENTS_CACHE),
        "indices_cached": {k: len(v) for k, v in _INDICES_CACHE.items()}
    }
