"""
VWAP-Strangle Strategy Configuration
====================================
Two distinct options trading strategies based on VWAP crossover signals.
Based on: Final Accumulated Notes from video analysis

Strategy 1: SHORT STRANGLE (80% usage)
Strategy 2: DIRECTIONAL BUYING (20% usage)

Professional enhancements included:
- 1% position sizing rule
- Improved stop-loss levels (30% vs 70%)
- India VIX filters
- Market type classification

Author: Trading System
Date: October 25, 2025
"""

# ============================================================================
# STRATEGY 1: SHORT STRANGLE (SELLING)
# ============================================================================
VWAP_STRANGLE_SELLING = {
    'name': 'VWAP_STRANGLE_SELLING',
    'description': 'Short strangle with VWAP entry - profit from theta decay',
    'strategy_type': 'SELLING',
    'usage_frequency': 0.80,  # 80% of the time
    
    # Timing (from Final Notes - Step 1)
    'setup_time_ist': '09:30',  # Must note 9:30 AM spot price
    'entry_window_start': '09:30',  # Wait for VWAP signal after 9:30
    'exit_time_ist': '13:30',   # Exit by 1:30 PM (from notes)
    
    # Strike Selection (from Final Notes - Step 2)
    'strikes_above_spot': 2,  # CE: 2 strikes above 9:30 AM spot
    'strikes_below_spot': 2,  # PE: 2 strikes below 9:30 AM spot
    'hedge_distance_points': 400,  # 400 points OTM for margin reduction
    
    # Basket Order (4-leg strategy from notes)
    'legs': [
        {'type': 'SELL_CE', 'offset_strikes': 2, 'hedge': False},
        {'type': 'SELL_PE', 'offset_strikes': 2, 'hedge': False},
        {'type': 'BUY_CE', 'offset_strikes': 2, 'hedge': True, 'distance_points': 400},
        {'type': 'BUY_PE', 'offset_strikes': 2, 'hedge': True, 'distance_points': 400}
    ],
    
    # Entry Signal (from Final Notes - Step 3)
    'entry_signal': {
        'method': 'vwap_crossover',
        'condition': 'premium_crosses_below_vwap',  # First ABOVE, then BELOW
        'chart_timeframe': '1m',  # 1-minute Sensibull chart
        'yellow_line': 'VWAP',  # Volume-weighted average price
        'blue_line': 'combined_premium'  # Real-time combined premium
    },
    
    # Risk Management (Professional Enhancement #1 & #2)
    'initial_sl_percent': 0.30,  # 30% SL (vs 70% in original)
    'initial_sl_calculation': 'combined_premium',  # Track combined premium, not legs
    'trailing_sl_enabled': True,
    'trail_to_breakeven_when': 'both_legs_profit',  # From notes
    'trail_method': 'combined_premium_cost_to_cost',  # CTC from notes
    
    # Position Sizing (Professional Enhancement #1)
    'position_size_method': '1_percent_rule',
    'max_risk_per_trade_pct': 1.0,  # 1% of capital max
    
    # Capital Requirements (from Final Notes)
    'min_capital_per_lot': 30000,  # ₹30,000 minimum
    'max_capital_per_lot': 50000,  # ₹50,000 maximum
    
    # IV Filter (Professional Enhancement #5)
    'use_iv_filter': True,
    'min_india_vix': 15.0,  # Only sell when VIX >= 15 (high premium)
    'iv_percentile_min': 50,  # Sell when IV in upper 50th percentile
    
    # Profit Targets (Professional Enhancement #4)
    'profit_target_method': 'percentage_of_premium',
    'profit_target_percent': 0.45,  # 45% of collected premium (vs 40-50%)
    'partial_profit_enabled': False,  # Simple: all or nothing
    
    # Market Type Filter (Professional Enhancement #6)
    'market_conditions': {
        'range_bound': True,   # Favor selling in range-bound markets
        'avoid_gap_open_pct': 0.5,  # Avoid if gap > 0.5%
        'avoid_breakout': True,  # Don't sell on breakouts
        'check_global_cues': True  # Check Nifty futures, SGX
    },
    
    # Execution
    'order_type': 'BASKET',  # Execute all 4 legs simultaneously
    'order_placement': 'MARKET',  # Market orders for instant fill
    
    # Backtest Settings
    'enabled_for_backtest': True,
    'min_confidence': 65  # Minimum confidence to execute
}

# ============================================================================
# STRATEGY 2: DIRECTIONAL BUYING
# ============================================================================
VWAP_STRANGLE_BUYING = {
    'name': 'VWAP_STRANGLE_BUYING',
    'description': 'Directional option buying with VWAP momentum signal',
    'strategy_type': 'BUYING',
    'usage_frequency': 0.20,  # 20% of the time
    
    # Timing (from Final Notes - Step 1)
    'setup_time_ist': '09:30',
    'entry_window_start': '09:30',
    
    # Strike Selection (from Final Notes - Step 1)
    'strikes_for_watchlist': {
        'call_strikes_above': 2,  # Same as selling for chart consistency
        'put_strikes_below': 2
    },
    
    # Entry Signal (from Final Notes - Step 2)
    'entry_signal': {
        'method': 'vwap_crossover',
        'condition': 'premium_crosses_above_vwap',  # Opposite of selling
        'chart_timeframe': '1m',
        'choose_stronger_leg': True,  # Buy whichever (CE/PE) rising faster
        'momentum_check': 'watchlist_comparison'  # Compare CE vs PE speed
    },
    
    # Risk Management (Professional Enhancement #2)
    'initial_sl_percent': 0.50,  # 50% SL on premium (from notes)
    'risk_reward_ratio': 2.0,  # 1:2 minimum R:R (professional standard)
    'sl_calculation_method': 'fixed_rr_based',  # SL based on target/RR
    
    # Profit Targets (from Final Notes - Step 3 + Professional Enhancement #4)
    'profit_targets': {
        'quick_scalp_points': [10, 20, 30],  # From notes: "capture 10-20-30 quick points"
        'primary_target_points': 20,  # Default target
        'trailing_enabled': True,
        'book_partial': False  # All or nothing for simplicity
    },
    
    # Position Sizing (Professional Enhancement #1)
    'position_size_method': '1_percent_rule',
    'max_risk_per_trade_pct': 1.0,
    
    # IV Filter (Professional Enhancement #5)
    'use_iv_filter': True,
    'max_india_vix': 13.0,  # Buy when VIX < 13 (cheap premiums)
    'iv_percentile_max': 50,  # Buy when IV in lower 50th percentile
    
    # Market Type Filter (Professional Enhancement #6)
    'market_conditions': {
        'trending': True,  # Favor buying in trending markets
        'gap_open_min_pct': 0.5,  # Look for gaps > 0.5%
        'breakout_detection': True,  # Enter on breakouts
        'strong_global_cues': True  # Require supporting global markets
    },
    
    # Execution
    'order_type': 'SINGLE',  # Buy only one leg (CE or PE)
    'order_placement': 'MARKET',
    
    # Backtest Settings
    'enabled_for_backtest': True,
    'min_confidence': 65
}

# ============================================================================
# WEEKLY TRADING PLAN (from Final Notes)
# ============================================================================
WEEKLY_SCHEDULE = {
    'enabled': True,
    'post_nov_20_schedule': True,  # Activated post-November 20th
    'schedule': {
        'Monday': {'index': 'NIFTY', 'expiry': 'weekly'},
        'Tuesday': {'index': 'NIFTY', 'expiry': 'weekly'},
        'Wednesday': {'index': 'NIFTY', 'expiry': 'weekly'},
        'Thursday': {'index': 'NIFTY', 'expiry': 'weekly'},
        'Friday': {'index': 'SENSEX', 'expiry': 'weekly'}  # Sensex on Friday only
    }
}

# ============================================================================
# SENSIBULL INTEGRATION
# ============================================================================
SENSIBULL_CONFIG = {
    'enabled': True,
    'login_method': 'zerodha_credentials',  # Free with Zerodha account
    'chart_type': 'multi_straddle_strangle',  # From "Analyse" tab
    'chart_timeframe': '1m',  # 1-minute candles
    'update_frequency_seconds': 5,  # Real-time updates every 5 seconds
    'data_points': {
        'vwap_line': 'yellow',  # Yellow line on chart
        'premium_line': 'blue',  # Blue/purple line on chart
    },
    # Fallback if Sensibull unavailable
    'fallback_to_manual_vwap': True,  # Calculate VWAP from tick data
}

# ============================================================================
# RISK PARAMETERS (Combined from Professional Enhancements)
# ============================================================================
RISK_MANAGEMENT = {
    # Position Sizing
    'capital_allocation_pct': 100,  # Can use full capital (limited by 1% rule)
    'max_positions_per_day': 1,  # One strategy per day (selling OR buying)
    'max_loss_per_day_pct': 3.0,  # Stop trading if lose 3% in a day
    
    # Stop-Loss Logic
    'selling': {
        'initial_sl_pct': 0.30,  # 30% above entry premium
        'trail_sl_when': 'both_legs_profitable',
        'trail_method': 'cost_to_cost',
        'max_loss_points': None  # Calculated dynamically
    },
    'buying': {
        'initial_sl_pct': 0.50,  # 50% below entry premium
        'trail_sl_when': 'in_profit',
        'trail_method': 'breakeven_lock',
        'rr_ratio': 2.0
    },
    
    # Adjustment Rules (Professional Enhancement #3)
    'adjustment_on_challenge': 'CLOSE_ALL',  # Don't leg out - close everything
    'partial_exit_enabled': False,  # Keep it simple
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_todays_index():
    """
    Determine which index to trade based on day of week.
    Returns: 'NIFTY' or 'SENSEX'
    """
    from datetime import datetime
    import pytz
    
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    day_name = now.strftime('%A')
    
    schedule = WEEKLY_SCHEDULE['schedule']
    if day_name in schedule:
        return schedule[day_name]['index']
    return 'NIFTY'  # Default

def get_strategy_for_market_condition(india_vix, gap_open_pct, is_breakout):
    """
    Decide which strategy to use based on market conditions.
    Professional Enhancement #6
    
    Args:
        india_vix: Current India VIX value
        gap_open_pct: Gap at open as percentage
        is_breakout: Boolean - is market breaking S/R?
    
    Returns:
        'SELLING' or 'BUYING' or None
    """
    # Selling conditions (from Professional Enhancement #6)
    if india_vix >= VWAP_STRANGLE_SELLING['min_india_vix']:
        if abs(gap_open_pct) < 0.5 and not is_breakout:
            return 'SELLING'
    
    # Buying conditions
    if india_vix <= VWAP_STRANGLE_BUYING['max_india_vix']:
        if abs(gap_open_pct) >= 0.5 or is_breakout:
            return 'BUYING'
    
    return None  # No trade today

def calculate_position_size(capital, entry_premium, sl_premium, lot_size=25):
    """
    Calculate position size using 1% rule (Professional Enhancement #1)
    
    Args:
        capital: Total trading capital (₹)
        entry_premium: Entry premium price
        sl_premium: Stop-loss premium price
        lot_size: Lot size (default: 25 for Nifty)
    
    Returns:
        int: Number of lots to trade
    """
    max_risk = capital * 0.01  # 1% of capital
    risk_per_lot = abs(sl_premium - entry_premium) * lot_size
    
    if risk_per_lot == 0:
        return 0
    
    lots = int(max_risk / risk_per_lot)
    return max(1, lots)  # At least 1 lot if risk allows

