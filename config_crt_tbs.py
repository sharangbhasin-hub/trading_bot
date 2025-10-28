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
    'min_rr_ratio': 1.0,
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
    'min_rr_ratio': 1.0,
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
    'min_rr_ratio': 1.0,
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
    'min_rr_ratio': 1.5,
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

# Configuration selector function
def get_config(trading_style: str = 'intraday', market_type: str = None):
    """
    Get configuration for specified trading style and market.
    
    Args:
        trading_style: 'scalping', 'intraday', 'shortterm', 'crypto', 'forex', 'commodities'
        market_type: Optional market type for auto-detection
            ('Indian Markets', 'US Stocks', 'Cryptocurrency', 'Forex', 'Commodities')
    
    Returns:
        Configuration dictionary
    
    Examples:
        # Auto-detect based on market type
        config = get_config(market_type='Cryptocurrency (Binance)')
        config = get_config(market_type='Forex (OANDA)')
        
        # Or use trading style
        config = get_config(trading_style='crypto')
    """
    # Market-based config selection (overrides trading_style if provided)
    if market_type:
        market_lower = market_type.lower()
        
        if 'crypto' in market_lower or 'binance' in market_lower or 'kucoin' in market_lower:
            return CRT_TBS_CRYPTO.copy()
        elif 'forex' in market_lower or 'oanda' in market_lower or 'fxcm' in market_lower:
            return CRT_TBS_FOREX.copy()
        elif 'commodit' in market_lower or 'gold' in market_lower or 'crude' in market_lower:
            return CRT_TBS_COMMODITIES.copy()
        elif 'indian' in market_lower or 'us stock' in market_lower or 'alpaca' in market_lower:
            return CRT_TBS_INTRADAY.copy()
    
    # Style-based config selection
    configs = {
        'scalping': CRT_TBS_SCALPING,
        'intraday': CRT_TBS_INTRADAY,
        'shortterm': CRT_TBS_SHORTTERM,
        'crypto': CRT_TBS_CRYPTO,
        'forex': CRT_TBS_FOREX,
        'commodities': CRT_TBS_COMMODITIES,
    }
    
    return configs.get(trading_style.lower(), DEFAULT_CONFIG).copy()


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
