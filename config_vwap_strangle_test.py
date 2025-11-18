"""
Test Configuration for VWAP Strategies
=======================================
Scaled-down configuration for testing and paper trading.
Lower capital requirements, smaller position sizes.

Author: Trading System
Date: November 18, 2025
"""

from config_vwap_strangle import *

# ============================================================================
# OVERRIDE FOR TESTING
# ============================================================================

# Use paper trading mode
TRADING_MODE['mode'] = 'PAPER'

# Lower capital for testing
TEST_CAPITAL = 50000  # ₹50,000 for testing

# ============================================================================
# SELLING STRATEGY - TEST VERSION
# ============================================================================
VWAP_STRANGLE_SELLING_TEST = VWAP_STRANGLE_SELLING.copy()
VWAP_STRANGLE_SELLING_TEST.update({
    'name': 'VWAP_STRANGLE_SELLING_TEST',
    'min_lots': 1,
    'max_lots': 1,  # Only 1 lot for testing
    'min_capital_per_lot': 10000,  # Lower requirement
    
    # More conservative for testing
    'initial_sl_percent': 0.25,  # 25% SL (even tighter)
    'profit_target_percent': 0.50,  # 50% target (higher)
})

# ============================================================================
# BUYING STRATEGY - TEST VERSION
# ============================================================================
VWAP_STRANGLE_BUYING_TEST = VWAP_STRANGLE_BUYING.copy()
VWAP_STRANGLE_BUYING_TEST.update({
    'name': 'VWAP_STRANGLE_BUYING_TEST',
    'min_lots': 1,
    'max_lots': 1,  # Only 1 lot
    
    # Conservative targets
    'profit_targets': {
        'method': 'fixed_points',
        'primary_target_points': 15,  # Smaller target for testing
        'quick_scalp_targets': [5, 10, 15]
    }
})

# ============================================================================
# DAILY LIMITS - TEST VERSION
# ============================================================================
DAILY_RISK_LIMITS_TEST = DAILY_RISK_LIMITS.copy()
DAILY_RISK_LIMITS_TEST.update({
    'max_trades_per_day': 1,  # Only 1 trade for testing
    'max_daily_loss_pct': 2.0,  # 2% max loss
})

# ============================================================================
# TRANSACTION COSTS - TESTING (Same as production)
# ============================================================================
# Keep costs realistic even in testing
TRANSACTION_COSTS_TEST = TRANSACTION_COSTS.copy()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_test_config(strategy_type: str) -> Dict:
    """
    Get test configuration for strategy.
    
    Args:
        strategy_type: 'SELLING' or 'BUYING'
    
    Returns:
        dict: Test configuration
    """
    if strategy_type == 'SELLING':
        return VWAP_STRANGLE_SELLING_TEST
    else:
        return VWAP_STRANGLE_BUYING_TEST

def validate_test_mode() -> Dict:
    """
    Validate that system is in test mode.
    
    Returns:
        dict: Validation status
    """
    checks = {
        'trading_mode_is_paper': TRADING_MODE['mode'] == 'PAPER',
        'max_lots_is_one': (VWAP_STRANGLE_SELLING_TEST['max_lots'] == 1 and 
                           VWAP_STRANGLE_BUYING_TEST['max_lots'] == 1),
        'daily_trades_limited': DAILY_RISK_LIMITS_TEST['max_trades_per_day'] == 1
    }
    
    all_passed = all(checks.values())
    
    return {
        'test_mode_active': all_passed,
        'checks': checks,
        'warnings': [] if all_passed else ['Test mode not properly configured!']
    }
