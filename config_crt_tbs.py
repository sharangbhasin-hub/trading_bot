"""
CRT-TBS Strategy Configuration
===============================

Configuration presets for different trading styles and markets:
- Scalping: 1H → 1min
- Intraday: 1D → 1H  
- Short-term: 4H → 5min
- Crypto: 4H → 15min (24/7 markets)
- Forex: 4H → 5min (24/5 markets)
- Commodities: 1D → 1H (session-based)

Based on CRT-TBS strategy documentation and finalized specifications.

Author: Trading System
Date: October 27, 2025
"""

# Scalping Configuration (1H → 1min)
CRT_TBS_SCALPING = {
    'name': 'CRT_TBS_SCALPING',
    'description': 'Scalping setup with 1H HTF and 1min LTF',
    
    # Timeframes (per documentation)
    'htf': '1H',
    'ltf': '1min',
    
    # CRT Detection
    'crt_method': 'body_vs_wicks',
    'crt_min_body_ratio': 0.50,
    
    # Key Level Detection
    'ohp_olp_lookback': 20,
    'swing_left': 2,
    'swing_right': 2,
    'fvg_min_gap_percent': 0.03,
    'ob_min_consecutive': 3,
    'rb_min_wick_ratio': 0.30,
    
    # TBS Detection
    'allow_multi_candle': True,
    'valid_candle_range': [1, 8],
    
    # Model #1 Detection
    'model1_method': 'body_vs_wicks',
    'model1_min_body_ratio': None,
    'max_wait_candles': 8,
    
    # Risk Management
    'min_rr_ratio': 1.5,
    'tp_multiplier': 1.5,
    'sl_distance_pips': 10,
    'stop_buffer': 0.0,
    'risk_per_trade': 1.0,
    
    # Position Management
    'position_scaling': [0.5, 0.5],
    'move_to_breakeven': 'after_tp1',
}

# Intraday Configuration (1D → 1H) - FOR STOCKS
CRT_TBS_INTRADAY = {
    'name': 'CRT_TBS_INTRADAY',
    'description': 'Day trading setup for stocks with 1D HTF and 1H LTF',
    
    # Timeframes (per documentation)
    'htf': '1D',
    'ltf': '1H',
    
    # CRT Detection
    'crt_method': 'body_vs_wicks',
    'crt_min_body_ratio': 0.50,
    
    # Key Level Detection
    'ohp_olp_lookback': 10,
    'swing_left': 2,
    'swing_right': 2,
    'fvg_min_gap_percent': 0.03,
    'ob_min_consecutive': 3,
    'rb_min_wick_ratio': 0.30,
    
    # TBS Detection
    'allow_multi_candle': True,
    'valid_candle_range': [1, 8],
    
    # Model #1 Detection
    'model1_method': 'body_vs_wicks',
    'model1_min_body_ratio': None,
    'max_wait_candles': 6,
    
    # Risk Management
    'min_rr_ratio': 2.0,
    'tp_multiplier': 2.0,
    'sl_distance_pips': 50,
    'stop_buffer': 0.0,
    'risk_per_trade': 1.0,
    
    # Position Management
    'position_scaling': [0.5, 0.5],
    'move_to_breakeven': 'after_tp1',
}

# Short-term Configuration (4H → 5min)
CRT_TBS_SHORTTERM = {
    'name': 'CRT_TBS_SHORTTERM',
    'description': 'Short-term trading setup with 4H HTF and 5min LTF',
    
    # Timeframes (per documentation)
    'htf': '4H',
    'ltf': '5min',
    
    # CRT Detection
    'crt_method': 'body_vs_wicks',
    'crt_min_body_ratio': 0.50,
    
    # Key Level Detection
    'ohp_olp_lookback': 15,
    'swing_left': 2,
    'swing_right': 2,
    'fvg_min_gap_percent': 0.03,
    'ob_min_consecutive': 3,
    'rb_min_wick_ratio': 0.30,
    
    # TBS Detection
    'allow_multi_candle': True,
    'valid_candle_range': [1, 8],
    
    # Model #1 Detection
    'model1_method': 'body_vs_wicks',
    'model1_min_body_ratio': None,
    'max_wait_candles': 8,
    
    # Risk Management
    'min_rr_ratio': 2.0, 
    'tp_multiplier': 2.0,
    'stop_buffer': 0.0,
    'risk_per_trade': 1.0,
    
    # Position Management
    'position_scaling': [0.5, 0.5],
    'move_to_breakeven': 'after_tp1',
}

# ============================================================================
# CRYPTO Configuration (4H → 15min) - FOR 24/7 CRYPTO MARKETS
# ============================================================================
CRT_TBS_CRYPTO = {
    'name': 'CRT_TBS_CRYPTO',
    'description': 'Cryptocurrency trading setup with 4H HTF and 15min LTF',
    
    # Timeframes optimized for 24/7 crypto markets
    'htf': '4H',      # 4-hour for CRT (better than 1D for crypto)
    'ltf': '15min',   # 15-minute for TBS
    
    # CRT Detection
    'crt_method': 'body_vs_wicks',
    'crt_min_body_ratio': 0.55,  # Slightly stricter for volatile crypto
    
    # Key Level Detection
    'ohp_olp_lookback': 20,  # 20 4H candles = ~3.3 days
    'swing_left': 2,
    'swing_right': 2,
    'fvg_min_gap_percent': 0.05,  # Wider for crypto volatility
    'ob_min_consecutive': 3,
    'rb_min_wick_ratio': 0.30,
    
    # TBS Detection
    'allow_multi_candle': True,
    'valid_candle_range': [1, 8],
    
    # Model #1 Detection
    'model1_method': 'body_vs_wicks',
    'model1_min_body_ratio': None,
    'max_wait_candles': 12,  # 3 hours max wait (crypto faster)
    
    # Risk Management (crypto is volatile)
    'min_rr_ratio': 2.0,  # Higher RR for crypto
    'tp_multiplier': 2.0,
    'stop_buffer': 0.0,
    'risk_per_trade': 1.0,
    
    # Position Management
    'position_scaling': [0.5, 0.5],
    'move_to_breakeven': 'after_tp1',
}

# ============================================================================
# FOREX Configuration (4H → 5min) - FOR FOREX MARKETS
# ============================================================================
CRT_TBS_FOREX = {
    'name': 'CRT_TBS_FOREX',
    'description': 'Forex trading setup with 4H HTF and 5min LTF',
    
    # Timeframes for forex
    'htf': '4H',
    'ltf': '5min',
    
    # CRT Detection
    'crt_method': 'body_vs_wicks',
    'crt_min_body_ratio': 0.50,
    
    # Key Level Detection
    'ohp_olp_lookback': 15,
    'swing_left': 2,
    'swing_right': 2,
    'fvg_min_gap_percent': 0.02,  # Tighter for forex
    'ob_min_consecutive': 3,
    'rb_min_wick_ratio': 0.30,
    
    # TBS Detection
    'allow_multi_candle': True,
    'valid_candle_range': [1, 8],
    
    # Model #1 Detection
    'model1_method': 'body_vs_wicks',
    'model1_min_body_ratio': None,
    'max_wait_candles': 10,  # 50 minutes
    
    # Risk Management
    'min_rr_ratio': 2.0,
    'tp_multiplier': 2.0,
    'sl_distance_pips': 30,
    'stop_buffer': 0.0,
    'risk_per_trade': 1.0,
    
    # Position Management
    'position_scaling': [0.5, 0.5],
    'move_to_breakeven': 'after_tp1',
}

# ============================================================================
# COMMODITIES Configuration (1D → 1H) - SAME AS STOCKS
# ============================================================================
CRT_TBS_COMMODITIES = CRT_TBS_INTRADAY.copy()
CRT_TBS_COMMODITIES['name'] = 'CRT_TBS_COMMODITIES'
CRT_TBS_COMMODITIES['description'] = 'Commodities trading (same as stocks: 1D → 1H)'

# Optimization Presets for Backtesting
CRT_TBS_OPTIMIZATION_PARAMS = {
    'description': 'Parameter ranges for optimization',
    
    # CRT body ratio variations (if testing ratio method)
    'crt_body_ratio_range': [0.50, 0.55, 0.60, 0.65, 0.70],
    
    # OHP/OLP lookback variations
    'lookback_1h_range': [15, 20, 25, 30],
    'lookback_4h_range': [10, 15, 20],
    'lookback_1d_range': [10, 15, 20, 25],
    
    # Model #1 body ratio variations
    'model1_body_ratio_range': [0.55, 0.60, 0.65, 0.70],
    
    # Model #1 timeout variations
    'max_wait_candles_range': [5, 6, 8, 10],
    
    # Risk-reward minimum variations
    'min_rr_ratio_range': [0.8, 1.0, 1.2, 1.5, 2.0],
    
    # FVG gap minimum variations
    'fvg_gap_range': [0.02, 0.03, 0.05, 0.08],
}

# Default configuration (Intraday)
DEFAULT_CONFIG = CRT_TBS_INTRADAY

def get_config(market_type: str = None, config_name: str = None):
    """
    Get configuration by market type or explicit config name.
    
    Priority (follows strategy constructor):
    1. config_name (explicit: 'scalping', 'intraday', 'short_term', 'crypto', 'forex', 'commodities')
    2. market_type (auto-detect: 'Cryptocurrency', 'Forex', 'Commodities', 'Stock')
    3. Default (fallback: intraday)
    
    Args:
        market_type: Market type string ('Cryptocurrency', 'Forex', 'Commodities', 'Stock')
        config_name: Explicit config name ('scalping', 'intraday', 'short_term', 'crypto', 'forex', 'commodities')
    
    Returns:
        Configuration dictionary (always returns a copy)
    
    Examples:
        # Explicit config name (highest priority)
        get_config(config_name='scalping')  # Returns CRT_TBS_SCALPING (1H → 1min)
        
        # Market type only (auto-select best config for that market)
        get_config(market_type='Forex')  # Returns CRT_TBS_FOREX (4H → 5min)
        
        # Both provided (config_name wins)
        get_config(market_type='Forex', config_name='scalping')  # Returns CRT_TBS_SCALPING
    """
    # ✅ PRIORITY 1: Explicit config_name (from UI trading mode selection)
    if config_name:
        # Direct config lookup
        config_map = {
            'scalping': CRT_TBS_SCALPING,        # 1H → 1min
            'intraday': CRT_TBS_INTRADAY,        # 1D → 1H
            'short_term': CRT_TBS_SHORTTERM,     # 4H → 5min
            'shortterm': CRT_TBS_SHORTTERM,      # Alias (no underscore)
            'crypto': CRT_TBS_CRYPTO,            # 4H → 15min
            'forex': CRT_TBS_FOREX,              # 4H → 5min
            'commodities': CRT_TBS_COMMODITIES,  # 1D → 1H
        }
        
        config = config_map.get(config_name.lower(), None)
        
        if config:
            return config.copy()
        else:
            # Invalid config_name provided - log warning and fallback
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"⚠️ Invalid config_name '{config_name}'. "
                f"Valid options: {list(config_map.keys())}. Falling back to intraday."
            )
            return DEFAULT_CONFIG.copy()
    
    # ✅ PRIORITY 2: Market type (auto-select best config for that market)
    if market_type:
        market_config_map = {
            'Cryptocurrency': CRT_TBS_CRYPTO,     # 4H → 15min
            'Forex': CRT_TBS_FOREX,               # 4H → 5min
            'Commodities': CRT_TBS_COMMODITIES,   # 1D → 1H
            'Stock': CRT_TBS_INTRADAY,            # 1D → 1H
            'Indian Markets': CRT_TBS_INTRADAY,   # 1D → 1H (alias)
        }
        
        # Case-insensitive lookup with partial matching
        market_lower = market_type.lower()
        
        # Try exact match first
        for key, config in market_config_map.items():
            if key.lower() == market_lower:
                return config.copy()
        
        # Try partial match (e.g., 'crypto' matches 'Cryptocurrency')
        for key, config in market_config_map.items():
            if market_lower in key.lower() or key.lower() in market_lower:
                return config.copy()
        
        # No match found - log warning and fallback
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"⚠️ Unknown market_type '{market_type}'. "
            f"Valid options: {list(market_config_map.keys())}. Falling back to intraday."
        )
        return DEFAULT_CONFIG.copy()
    
    # ✅ PRIORITY 3: Default fallback (no config_name or market_type provided)
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ No config_name or market_type provided. Using default intraday config.")
    return DEFAULT_CONFIG.copy()

# Export for easy import
__all__ = [
    'CRT_TBS_SCALPING',
    'CRT_TBS_INTRADAY',
    'CRT_TBS_SHORTTERM',
    'CRT_TBS_CRYPTO',          # ← NEW
    'CRT_TBS_FOREX',           # ← NEW
    'CRT_TBS_COMMODITIES',     # ← NEW
    'CRT_TBS_OPTIMIZATION_PARAMS',
    'DEFAULT_CONFIG',
    'get_config'
]
