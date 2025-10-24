"""
CRT-TBS Strategy Configuration
===============================

Configuration presets for different trading styles:
- Scalping: 1H → 1min
- Intraday: 1D → 1H  
- Short-term: 4H → 5min

Based on CRT-TBS strategy documentation and finalized specifications.

Author: Trading System
Date: October 23, 2025
"""

# Scalping Configuration (1H → 1min)
CRT_TBS_SCALPING = {
    'name': 'CRT_TBS_SCALPING',
    'description': 'Scalping setup with 1H HTF and 1min LTF',
    
    # Timeframes (per documentation)
    'htf': '1H',
    'ltf': '1m',
    
    # ✅ CHANGE: CRT Detection (now supports both methods)
    'crt_method': 'body_vs_wicks',  # Options: 'body_vs_wicks' or 'ratio'
    'crt_min_body_ratio': 0.50,  # Used if method='ratio' (50% = more candles, 70% = fewer)
    
    # Key Level Detection
    'ohp_olp_lookback': 20,  # 20 hours for 1H timeframe
    'swing_left': 2,
    'swing_right': 2,
    'fvg_min_gap_percent': 0.03,  # 5 basis points
    'ob_min_consecutive': 3,
    'rb_min_wick_ratio': 0.30,
    
    # TBS Detection
    'allow_multi_candle': True,  # Enable A+ TBS patterns
    'valid_candle_range': [1, 8],  # Document-specified range
    
    # Model #1 Detection
    'model1_method': 'body_vs_wicks',
    'model1_min_body_ratio': None,
    'max_wait_candles': 8,  # 10 minutes max wait
    
    # Risk Management
    'min_rr_ratio': 1.0,  # Higher for scalping due to costs
    'stop_buffer': 0.0,  # No buffer per documentation
    'risk_per_trade': 1.0,  # 1% risk per trade
    
    # Position Management (document-specified)
    'position_scaling': [0.5, 0.5],  # 50% at TP1, 50% at TP2
    'move_to_breakeven': 'after_tp1',  # Critical rule
}

# Intraday Configuration (1D → 1H)
CRT_TBS_INTRADAY = {
    'name': 'CRT_TBS_INTRADAY',
    'description': 'Day trading setup with 1D HTF and 1H LTF',
    
    # Timeframes (per documentation)
    'htf': '1D',
    'ltf': '1H',
    
    # ✅ CHANGE: CRT Detection (now supports both methods)
    'crt_method': 'body_vs_wicks',  # Options: 'body_vs_wicks' or 'ratio'
    'crt_min_body_ratio': 0.50,  # Used if method='ratio' (50% = more candles, 70% = fewer)
    
    # Key Level Detection
    'ohp_olp_lookback': 10,  # 10 days for daily timeframe
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
    'max_wait_candles': 6,  # 8 hours max wait
    
    # Risk Management
    'min_rr_ratio': 1.0,  # Conservative minimum
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
    'ltf': '5m',
    
    # ✅ CHANGE: CRT Detection (now supports both methods)
    'crt_method': 'body_vs_wicks',  # Options: 'body_vs_wicks' or 'ratio'
    'crt_min_body_ratio': 0.50,  # Used if method='ratio' (50% = more candles, 70% = fewer)
    
    # Key Level Detection
    'ohp_olp_lookback': 15,  # 15 4H candles = ~2.5 days
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
    'max_wait_candles': 8,  # 50 minutes max wait
    
    # Risk Management
    'min_rr_ratio': 1.0,  # Moderate
    'stop_buffer': 0.0,
    'risk_per_trade': 1.0,
    
    # Position Management
    'position_scaling': [0.5, 0.5],
    'move_to_breakeven': 'after_tp1',
}

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
def get_config(trading_style: str = 'intraday'):
    """
    Get configuration for specified trading style.
    
    Args:
        trading_style: 'scalping', 'intraday', or 'shortterm'
    
    Returns:
        Configuration dictionary
    """
    configs = {
        'scalping': CRT_TBS_SCALPING,
        'intraday': CRT_TBS_INTRADAY,
        'shortterm': CRT_TBS_SHORTTERM,
    }
    
    return configs.get(trading_style.lower(), DEFAULT_CONFIG).copy()


# Export for easy import
__all__ = [
    'CRT_TBS_SCALPING',
    'CRT_TBS_INTRADAY',
    'CRT_TBS_SHORTTERM',
    'CRT_TBS_OPTIMIZATION_PARAMS',
    'DEFAULT_CONFIG',
    'get_config'
]
