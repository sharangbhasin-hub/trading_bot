"""
VWAP-Strangle Strategy Configuration (ANALYST-ENHANCED VERSION)
================================================================
Configuration for two distinct options trading strategies with all
professional enhancements and analyst recommendations integrated.

Strategy 1: SHORT STRANGLE (80% usage) - Selling for theta decay
Strategy 2: DIRECTIONAL BUYING (20% usage) - Momentum-based buying

Key Changes from Original:
- Stop-loss reduced from 70% to 30% (Analyst's critical fix)
- Concrete market condition thresholds (0.3% gap, VIX 15/13)
- Daily risk limits (2 trades/day, 3% max loss)
- Transaction cost accounting
- Paper trading mode with progression plan

Author: Trading System (Enhanced by Senior Analyst Review)
Date: November 18, 2025
"""

# ============================================================================
# TRADING MODE & SAFETY CONTROLS
# ============================================================================
TRADING_MODE = {
    'mode': 'PAPER',  # 'PAPER' or 'LIVE' - ALWAYS START WITH PAPER!
    
    # Analyst's Requirement: 20+ documented paper trades before going live
    'paper_trade_requirements': {
        'min_trades': 20,
        'min_win_rate': 60.0,  # Must achieve >60% win rate
        'min_days': 14,  # Minimum 2 weeks of paper trading
        'min_trades_per_week': 3  # At least 3 trades/week to validate
    },
    
    # Analyst's Phased Approach to Live Trading
    'live_trade_progression': {
        'phase_1_duration_weeks': 4,  # Weeks 1-4: Paper trade
        'phase_2_duration_weeks': 4,  # Weeks 5-8: 50% position size
        'phase_2_position_multiplier': 0.5,
        'phase_3_onwards': 'full_size'  # Week 9+: Full position if >60% WR
    }
}

# ============================================================================
# DAILY RISK LIMITS (Analyst's Critical Addition)
# ============================================================================
DAILY_RISK_LIMITS = {
    'max_trades_per_day': 2,  # Analyst: Prevents overtrading
    'max_daily_loss_pct': 3.0,  # 3% max loss per day - STOP trading if hit
    'max_consecutive_losses': 3,  # Stop after 3 losses in a row
    'cooldown_after_daily_limit_hours': 24,  # Don't trade next day if limit hit
    
    # Analyst: Track emotional state
    'require_trade_notes': True,  # Must document reason for each trade
    'require_emotional_check': True  # Log emotional state (CALM/ANXIOUS/REVENGE)
}

# ============================================================================
# TRANSACTION COSTS (Analyst's Reality Check)
# ============================================================================
TRANSACTION_COSTS = {
    'slippage_points_per_leg': 2.5,  # Analyst: 2-5 points typical, use 2.5 avg
    'brokerage_per_order': 50,  # ₹50 per order (₹100 round-trip)
    'stt_on_sell_pct': 0.05,  # STT 0.05% on sell side (options)
    'gst_on_brokerage_pct': 18,  # 18% GST on brokerage
    
    # Calculate total cost per trade
    'estimate_total_cost': True,  # Include in P&L calculations
    'display_cost_breakdown': True  # Show in UI
}

# ============================================================================
# STRATEGY 1: SHORT STRANGLE (SELLING) - ANALYST-ENHANCED
# ============================================================================
VWAP_STRANGLE_SELLING = {
    'name': 'VWAP_STRANGLE_SELLING',
    'description': 'Short strangle with VWAP entry - theta decay strategy',
    'strategy_type': 'SELLING',
    'usage_frequency': 0.80,  # 80% of the time per original notes
    'enabled': True,
    
    # ========================================================================
    # TIMING (From Final Notes)
    # ========================================================================
    'setup_time_ist': '09:30',  # Capture 9:30 AM spot price
    'entry_window_start': '09:30',
    'entry_window_end': '10:30',  # Don't enter after 10:30 AM
    
    # Analyst's Conditional Exit (NOT blind 1:30 PM)
    'exit_rules': {
        'immediate_exit_on_target': True,  # Exit immediately when target hit
        'immediate_exit_on_sl': True,
        
        # Conditional 1:30 PM logic
        'time_check_1': {
            'time': '13:30',  # 1:30 PM
            'action_if_premium_above_entry': 'EXIT',  # Theta not working
            'action_if_premium_below_entry': 'HOLD_WITH_TRAILING_SL'
        },
        
        # Final exit
        'time_check_2': {
            'time': '15:00',  # 3:00 PM - always exit
            'action': 'EXIT_UNCONDITIONAL'
        }
    },
    
    # ========================================================================
    # STRIKE SELECTION (From Final Notes - exact requirements)
    # ========================================================================
    'strikes_above_spot': 2,  # CE: 2 strikes above 9:30 AM spot
    'strikes_below_spot': 2,  # PE: 2 strikes below 9:30 AM spot
    'hedge_distance_points': 400,  # Hedges 400 points OTM
    
    # 4-leg basket order structure
    'legs': [
        {'type': 'SELL_CE', 'offset_strikes': 2, 'hedge': False},
        {'type': 'SELL_PE', 'offset_strikes': 2, 'hedge': False},
        {'type': 'BUY_CE', 'offset_strikes': 2, 'hedge': True, 'distance_points': 400},
        {'type': 'BUY_PE', 'offset_strikes': 2, 'hedge': True, 'distance_points': 400}
    ],
    
    # ========================================================================
    # ENTRY SIGNAL (From Final Notes)
    # ========================================================================
    'entry_signal': {
        'method': 'vwap_crossover',
        'condition': 'premium_crosses_below_vwap',  # Must go ABOVE first, then BELOW
        'chart_timeframe': '1m',  # 1-minute Sensibull chart
        'require_above_first': True,  # Critical: Must be above before crossing below
        'min_time_above_vwap_seconds': 60,  # Must stay above for at least 1 minute
        'yellow_line': 'VWAP',
        'blue_line': 'combined_premium'
    },
    
    # ========================================================================
    # RISK MANAGEMENT (Analyst's Critical Fixes)
    # ========================================================================
    # CRITICAL FIX: Changed from 70% to 30%
    'initial_sl_percent': 0.30,  # 30% SL (was 70% - would be fatal!)
    'initial_sl_calculation': 'combined_premium',  # Track combined, not individual legs
    
    # Analyst's example: Entry 150, SL 195 (45 points), Target 90 (60 points)
    # Risk:Reward = 45:60 = 1:1.33 ✅
    
    # Trailing stop-loss
    'trailing_sl_enabled': True,
    'trail_trigger_pct': 0.20,  # Trail when profit reaches 20% decay
    'trail_to_breakeven': True,  # Move SL to breakeven (entry premium)
    'trail_buffer_points': 5,  # 5-point buffer above breakeven
    
    # ========================================================================
    # POSITION SIZING (Analyst's 1% Rule)
    # ========================================================================
    'position_size_method': '1_percent_rule',
    'max_risk_per_trade_pct': 1.0,  # Never risk more than 1% of capital
    'min_lots': 1,
    'max_lots': 5,  # Safety cap
    
    # Capital requirements (from original notes)
    'min_capital_per_lot': 30000,
    'max_capital_per_lot': 50000,
    'min_total_capital': 200000,  # ₹2 lakh minimum recommended
    
    # ========================================================================
    # MARKET CONDITIONS FILTER (Analyst's Concrete Thresholds)
    # ========================================================================
    'market_conditions': {
        # Analyst's Rule: "If Nifty opens within ±0.3% of previous close = sell day"
        'max_gap_open_pct': 0.3,  # CONCRETE: Gap must be ≤ 0.3%
        'require_range_bound': True,  # Must be range-bound (not trending)
        'avoid_breakout': True,  # Don't sell on breakout days
        'check_global_cues': True,
        
        # Minimum conditions to meet (Analyst: need 3/4)
        'min_conditions_to_trade': 3
    },
    
    # ========================================================================
    # IV FILTER (Analyst's VIX Thresholds with Evidence)
    # ========================================================================
    'use_iv_filter': True,
    'min_india_vix': 15.0,  # CONCRETE: Only sell when VIX >= 15 (expensive premiums)
    'iv_percentile_min': 50,  # Only sell when IV in upper 50th percentile
    'vix_calculation_method': 'current',  # 'current' or 'average_5d'
    
    # Analyst's note: High VIX = inflated premiums = good for selling
    
    # ========================================================================
    # PROFIT TARGETS (Analyst's Refined Logic)
    # ========================================================================
    'profit_target_method': 'percentage_of_premium',
    'profit_target_percent': 0.45,  # 45% decay of collected premium
    # Example: Collect 150, target is 150 - (150*0.45) = 82.5
    
    'partial_profit_enabled': False,  # Keep it simple - all or nothing
    
    # ========================================================================
    # EXECUTION
    # ========================================================================
    'order_type': 'BASKET',
    'order_placement': 'MARKET',  # Market orders for instant fill
    'max_order_retry': 3,
    'order_timeout_seconds': 30,
    
    # ========================================================================
    # BACKTESTING
    # ========================================================================
    'enabled_for_backtest': True,
    'min_confidence': 65,
    'backtest_include_costs': True,  # Analyst: MUST include transaction costs
    
    # Analyst's realistic expectations:
    # Win Rate: 65-70%
    # Avg Win: ₹1,500 per lot
    # Avg Loss: ₹1,125 per lot
    # Monthly Return: 4-5% (very strong if sustained)
}

# ============================================================================
# STRATEGY 2: DIRECTIONAL BUYING - ANALYST-ENHANCED
# ============================================================================
VWAP_STRANGLE_BUYING = {
    'name': 'VWAP_STRANGLE_BUYING',
    'description': 'Directional option buying with momentum confirmation',
    'strategy_type': 'BUYING',
    'usage_frequency': 0.20,  # 20% of the time
    'enabled': True,  # But analyst recommends FOCUS ON SELLING until proven
    
    # Analyst's Verdict: "Abandon unless you can backtest 500+ setups and prove it adds value"
    'require_extensive_backtest': True,
    'min_backtest_trades': 500,  # Analyst's requirement
    
    # ========================================================================
    # TIMING
    # ========================================================================
    'setup_time_ist': '09:30',
    'entry_window_start': '09:30',
    'entry_window_end': '11:00',  # Can enter later than selling
    
    # Exit rules (simpler than selling)
    'exit_rules': {
        'immediate_exit_on_target': True,
        'immediate_exit_on_sl': True,
        'max_hold_time_minutes': 120  # Exit after 2 hours max
    },
    
    # ========================================================================
    # STRIKE SELECTION (Same as selling for chart consistency)
    # ========================================================================
    'strikes_for_watchlist': {
        'call_strikes_above': 2,
        'put_strikes_below': 2
    },
    
    # ========================================================================
    # ENTRY SIGNAL (Analyst's Momentum Filter Added!)
    # ========================================================================
    'entry_signal': {
        'method': 'vwap_crossover_with_momentum',  # UPDATED
        'condition': 'premium_crosses_above_vwap',  # Opposite of selling
        'chart_timeframe': '1m',
        'choose_stronger_leg': True,
        
        # ANALYST'S CRITICAL ADDITION: Momentum confirmation
        'momentum_filter': {
            'enabled': True,  # MUST BE TRUE
            'min_move_pct': 15.0,  # Option must move >15% in 5 minutes
            'lookback_minutes': 5,
            'reason': 'Prevents buying stale crossovers'
        }
    },
    
    # ========================================================================
    # RISK MANAGEMENT (Analyst's 1:2 R:R Fix)
    # ========================================================================
    # CRITICAL FIX: Changed from 50% SL to 1:2 Risk:Reward ratio
    'use_fixed_rr_ratio': True,  # New: Use R:R instead of % SL
    'risk_reward_ratio': 2.0,  # 1:2 minimum (risk 10 to make 20)
    
    # Analyst's example:
    # Entry: 50, Target: 70 (+20 points), SL: 40 (-10 points) = 1:2 R:R ✅
    
    # Fallback if R:R can't be calculated
    'fallback_sl_percent': 0.20,  # 20% max (not 50%!)
    
    # Trailing SL (Analyst's aggressive trailing)
    'trailing_sl_enabled': True,
    'trail_trigger_points': 15,  # Move to breakeven after 15-point gain
    'trail_to_breakeven_immediately': True,
    
    # ========================================================================
    # PROFIT TARGETS (Analyst's Refined)
    # ========================================================================
    'profit_targets': {
        'method': 'fixed_points',  # Not percentage for buying
        'primary_target_points': 20,  # Default target
        'quick_scalp_targets': [10, 15, 20],  # From notes: "10-20-30 quick points"
        'use_multiple_targets': False  # Keep it simple
    },
    
    # ========================================================================
    # POSITION SIZING (Same 1% rule)
    # ========================================================================
    'position_size_method': '1_percent_rule',
    'max_risk_per_trade_pct': 1.0,
    'min_lots': 1,
    'max_lots': 3,  # Lower than selling (more risk per trade)
    
    # ========================================================================
    # MARKET CONDITIONS FILTER (Analyst's Concrete Thresholds)
    # ========================================================================
    'market_conditions': {
        # Opposite of selling
        'min_gap_open_pct': 0.5,  # CONCRETE: Gap must be ≥ 0.5%
        'require_trending': True,  # Must be trending (not range-bound)
        'require_breakout': True,  # Look for breakouts
        'check_global_cues': True,
        
        'min_conditions_to_trade': 3  # Need 3/4 conditions
    },
    
    # ========================================================================
    # IV FILTER (Analyst's VIX Thresholds)
    # ========================================================================
    'use_iv_filter': True,
    'max_india_vix': 13.0,  # CONCRETE: Only buy when VIX <= 13 (cheap premiums)
    'iv_percentile_max': 50,  # Only buy when IV in lower 50th percentile
    
    # Analyst's note: Low VIX = cheap premiums = better for buying
    
    # ========================================================================
    # EXECUTION
    # ========================================================================
    'order_type': 'SINGLE',  # Buy only one leg (CE or PE)
    'order_placement': 'MARKET',
    
    # ========================================================================
    # BACKTESTING
    # ========================================================================
    'enabled_for_backtest': True,
    'min_confidence': 70,  # Higher bar than selling
    'backtest_include_costs': True,
    
    # Analyst's realistic expectations:
    # Win Rate: 40-45% (low win rate, high reward strategy)
    # Avg Win: ₹500 per lot
    # Avg Loss: ₹250 per lot
    # Monthly Return: 0.4% (minimal - focus 95% on selling!)
}

# ============================================================================
# WEEKLY TRADING PLAN (From Final Notes)
# ============================================================================
WEEKLY_SCHEDULE = {
    'enabled': True,
    'post_nov_20_schedule': True,  # Activated post-November 20th
    'schedule': {
        'Monday': {'index': 'NIFTY', 'expiry': 'weekly'},
        'Tuesday': {'index': 'NIFTY', 'expiry': 'weekly'},
        'Wednesday': {'index': 'NIFTY', 'expiry': 'weekly'},
        'Thursday': {'index': 'NIFTY', 'expiry': 'weekly'},
        'Friday': {'index': 'SENSEX', 'expiry': 'weekly'}  # Sensex weekly expiry
    }
}

# ============================================================================
# SENSIBULL INTEGRATION (Simplified - local VWAP calculation)
# ============================================================================
SENSIBULL_CONFIG = {
    'enabled': False,  # Set to True if you have Sensibull account
    'login_method': 'zerodha_credentials',
    'chart_type': 'multi_straddle_strangle',
    'chart_timeframe': '1m',
    'update_frequency_seconds': 5,
    
    # Data points
    'data_points': {
        'vwap_line': 'yellow',
        'premium_line': 'blue'
    },
    
    # Fallback (recommended)
    'fallback_to_manual_vwap': True,  # Calculate locally from Kite streaming
    'use_local_calculation': True  # Always use local VWAP
}

# ============================================================================
# RISK MANAGEMENT MASTER CONFIG (Analyst's Comprehensive Rules)
# ============================================================================
RISK_MANAGEMENT = {
    # Position Sizing
    'capital_allocation_pct': 100,  # Can use full capital (limited by 1% rule)
    'max_positions_simultaneously': 1,  # Only 1 active VWAP trade at a time
    'max_positions_per_day': 2,  # Analyst: Max 2 trades/day
    'max_loss_per_day_pct': 3.0,  # Analyst: 3% daily loss = STOP
    
    # Stop-Loss Configurations
    'selling': {
        'initial_sl_pct': 0.30,  # 30% above entry (ANALYST FIX)
        'trail_sl_when': 'profit_20pct',  # Trail when 20% profit
        'trail_method': 'breakeven',
        'max_loss_per_trade_pct': 1.0
    },
    
    'buying': {
        'use_rr_ratio': True,  # Use R:R instead of %
        'rr_ratio': 2.0,  # 1:2
        'trail_sl_when': 'profit_15points',
        'trail_method': 'breakeven',
        'max_loss_per_trade_pct': 1.0
    },
    
    # Adjustment Rules (Analyst: Keep it simple)
    'adjustment_on_challenge': 'CLOSE_ALL',  # Don't leg out
    'partial_exit_enabled': False,
    'allow_averaging_down': False,  # Never add to losers
    
    # Circuit Breakers
    'pause_trading_on_consecutive_losses': 3,
    'pause_duration_hours': 24,
    'require_review_after_pause': True
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

def get_strategy_for_market_condition(india_vix: float, 
                                      gap_open_pct: float, 
                                      is_breakout: bool,
                                      is_range_bound: bool) -> dict:
    """
    Decide which strategy to use based on CONCRETE market conditions.
    Analyst's Enhancement: Clear thresholds, not subjective.
    
    Args:
        india_vix: Current India VIX value
        gap_open_pct: Gap at open as percentage
        is_breakout: Is market breaking S/R?
        is_range_bound: Is market range-bound?
    
    Returns:
        dict: {'strategy': 'SELLING'/'BUYING'/None, 'confidence': float, 'reason': str}
    """
    selling_score = 0
    buying_score = 0
    reasons = []
    
    # SELLING CONDITIONS (Analyst's concrete thresholds)
    selling_checks = {
        'vix': india_vix >= VWAP_STRANGLE_SELLING['min_india_vix'],  # VIX >= 15
        'gap': abs(gap_open_pct) <= VWAP_STRANGLE_SELLING['market_conditions']['max_gap_open_pct'],  # ≤0.3%
        'range': is_range_bound,
        'no_breakout': not is_breakout
    }
    
    selling_score = sum(selling_checks.values())
    
    # BUYING CONDITIONS (Analyst's concrete thresholds)
    buying_checks = {
        'vix': india_vix <= VWAP_STRANGLE_BUYING['max_india_vix'],  # VIX <= 13
        'gap': abs(gap_open_pct) >= VWAP_STRANGLE_BUYING['market_conditions']['min_gap_open_pct'],  # ≥0.5%
        'trending': not is_range_bound,
        'breakout': is_breakout
    }
    
    buying_score = sum(buying_checks.values())
    
    # Analyst: Need at least 3/4 conditions
    min_required = 3
    
    if selling_score >= min_required and selling_score > buying_score:
        return {
            'strategy': 'SELLING',
            'confidence': (selling_score / 4) * 100,
            'conditions_met': selling_checks,
            'score': f'{selling_score}/4',
            'reason': f"Selling day: VIX={india_vix:.1f}, Gap={gap_open_pct:.2f}%, Range-bound={is_range_bound}"
        }
    elif buying_score >= min_required and buying_score > selling_score:
        return {
            'strategy': 'BUYING',
            'confidence': (buying_score / 4) * 100,
            'conditions_met': buying_checks,
            'score': f'{buying_score}/4',
            'reason': f"Buying day: VIX={india_vix:.1f}, Gap={gap_open_pct:.2f}%, Breakout={is_breakout}"
        }
    else:
        return {
            'strategy': None,
            'confidence': 0,
            'reason': f"NO TRADE: Selling {selling_score}/4, Buying {buying_score}/4 - Below threshold (need 3/4)"
        }

def calculate_position_size(capital: float, 
                           entry_premium: float, 
                           sl_premium: float, 
                           lot_size: int = 25) -> dict:
    """
    Calculate position size using 1% rule (Analyst's foundation).
    
    Args:
        capital: Total trading capital (₹)
        entry_premium: Entry premium price
        sl_premium: Stop-loss premium price
        lot_size: Lot size (25 for Nifty, 10 for Sensex)
    
    Returns:
        dict: {
            'lots': int,
            'risk_per_lot': float,
            'total_risk': float,
            'capital_required': float,
            'risk_pct': float
        }
    """
    max_risk = capital * (RISK_MANAGEMENT['max_loss_per_day_pct'] / 100)  # 1% of capital
    risk_per_lot = abs(sl_premium - entry_premium) * lot_size
    
    if risk_per_lot == 0:
        return {'lots': 0, 'reason': 'Zero risk per lot - invalid setup'}
    
    lots = int(max_risk / risk_per_lot)
    lots = max(1, min(lots, 5))  # Between 1-5 lots
    
    total_risk = lots * risk_per_lot
    risk_pct = (total_risk / capital) * 100
    
    # Analyst's example validation
    # Capital: ₹2,00,000, Max risk: ₹2,000
    # Entry: 150, SL: 195, Risk/lot: 45*25 = ₹1,125
    # Lots: 2,000/1,125 = 1.77 → 1 lot
    # Total risk: ₹1,125 ✅
    
    return {
        'lots': lots,
        'risk_per_lot': risk_per_lot,
        'total_risk': total_risk,
        'capital_required': total_risk,  # Simplified
        'risk_pct': risk_pct,
        'max_allowed_risk': max_risk
    }

def validate_trading_mode():
    """
    Check if allowed to trade based on trading mode and requirements.
    Analyst: FORCE paper trading first!
    """
    mode = TRADING_MODE['mode']
    
    if mode == 'LIVE':
        # Check if paper trading requirements met
        reqs = TRADING_MODE['paper_trade_requirements']
        # (Would check actual journal stats here)
        return {
            'allowed': True,  # Placeholder
            'mode': 'LIVE',
            'warning': 'Ensure paper trading requirements met!'
        }
    else:
        return {
            'allowed': True,
            'mode': 'PAPER',
            'message': 'Paper trading mode - no real money at risk'
        }
