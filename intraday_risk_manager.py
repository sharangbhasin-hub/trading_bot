"""
Intraday Risk Manager for ITM Options Trading
Calculates stop-loss, targets, position sizing, and trailing strategies
Optimized for same-day options trades on indices
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime

class IntradayITMRiskManager:
    """
    Risk management specifically for Intraday ITM Options
    Combines premium-based, index-based, and swing-point stops
    """
    
    def __init__(self):
        # Risk parameters for ITM options
        self.max_risk_percent = 25  # Maximum 25% loss for strong signals
        self.tight_stop_percent = 20  # 20% for moderate signals
        self.very_tight_stop_percent = 15  # 15% for weak signals
        
        # Target parameters
        self.partial_profit_target = 35  # Book 50% at 35% profit
        self.main_target = 50  # Book 30% more at 50% profit
        self.stretch_multiplier = 3.0  # Trail remaining 20%
        
        # Position sizing
        self.max_capital_per_trade = 15000  # Maximum ₹15,000 per trade
        self.max_lots = 3  # Maximum 3 lots for safety
        
    def calculate_comprehensive_risk_plan(
        self,
        itm_premium: float,
        spot_price: float,
        strike_price: float,
        option_type: str,  # 'CE' or 'PE'
        confluence_score: int,
        support_level: float,
        resistance_level: float,
        df_5min: pd.DataFrame,
        lot_size: int = 50,
        candlestick_pattern: Optional[Dict] = None,
        atr: Optional[float] = None
    ) -> Dict:
        """
        Calculate comprehensive risk management plan for ITM options
        
        Returns complete plan with:
        - Multiple stop-loss levels
        - Tiered profit targets
        - Position sizing
        - Trailing strategy
        - Risk-reward metrics
        """
        
        try:
            results = {}
            
            # Determine trade direction
            direction = 'CALL' if option_type == 'CE' else 'PUT'
            
            # ========== STOP-LOSS CALCULATIONS ==========
            
            # 1. PREMIUM-BASED STOP LOSS (Primary for ITM)
            if confluence_score >= 9:
                stop_percent = 0.25  # Very strong signal - standard stop
                position_size_multiplier = 1.0  # Full position
            elif confluence_score >= 8:
                stop_percent = 0.25  # Strong signal - standard stop
                position_size_multiplier = 1.0
            elif confluence_score >= 7:
                stop_percent = 0.20  # Moderate signal - tighter stop
                position_size_multiplier = 0.5  # Half position
            else:
                stop_percent = 0.15  # Weak signal - very tight stop
                position_size_multiplier = 0.25  # Quarter position
            
            premium_stop_loss = itm_premium * (1 - stop_percent)
            risk_per_share = itm_premium - premium_stop_loss
            
            results['premium_stop_loss'] = round(premium_stop_loss, 2)
            results['stop_loss_percent'] = round(stop_percent * 100, 1)
            results['risk_per_share'] = round(risk_per_share, 2)
            
            # 2. INDEX-BASED STOP LOSS
            if direction == 'CALL':
                # For calls: Exit if index breaks below support
                index_stop = support_level if support_level > 0 else spot_price * 0.995
                index_stop_distance = ((spot_price - index_stop) / spot_price) * 100
            else:  # PUT
                # For puts: Exit if index breaks above resistance
                index_stop = resistance_level if resistance_level > 0 else spot_price * 1.005
                index_stop_distance = ((index_stop - spot_price) / spot_price) * 100
            
            results['index_stop_level'] = round(index_stop, 2)
            results['index_stop_distance_pct'] = round(abs(index_stop_distance), 2)
            
            # 3. ATR-BASED STOP (if ATR available)
            if atr and atr > 0:
                atr_multiplier = 1.5
                if direction == 'CALL':
                    atr_stop_index = spot_price - (atr * atr_multiplier)
                else:
                    atr_stop_index = spot_price + (atr * atr_multiplier)
                
                results['atr_stop_level'] = round(atr_stop_index, 2)
                results['atr_value'] = round(atr, 2)
            else:
                results['atr_stop_level'] = None
                results['atr_value'] = 0
            
            # 4. PATTERN-BASED ADJUSTMENT
            pattern_adjustment = 1.0
            if candlestick_pattern:
                strength = candlestick_pattern.get('strength', 0)
                if strength >= 90:
                    pattern_adjustment = 0.95  # Tighter stop for very strong patterns
                elif strength >= 80:
                    pattern_adjustment = 1.0  # Standard
                else:
                    pattern_adjustment = 1.05  # Wider stop for weaker patterns
            
            adjusted_premium_stop = premium_stop_loss * pattern_adjustment
            results['adjusted_premium_stop'] = round(adjusted_premium_stop, 2)
            results['pattern_adjustment_factor'] = pattern_adjustment
            
            # 5. SWING POINT STOP (5-min chart)
            swing_stop = self._calculate_swing_stop(df_5min, direction)
            results['swing_point_stop_index'] = round(swing_stop, 2) if swing_stop else None
            
            # RECOMMENDED STOP LOSS (use most conservative)
            results['recommended_stop_loss'] = results['adjusted_premium_stop']
            results['stop_loss_type'] = 'Premium-Based'
            results['backup_stop'] = results['index_stop_level']
            
            # ========== PROFIT TARGETS ==========
            
            # Target multiplier based on confluence score
            if confluence_score >= 9:
                target_mult = 1.2  # More aggressive for very strong signals
            elif confluence_score >= 8:
                target_mult = 1.0  # Standard
            else:
                target_mult = 0.8  # Conservative for moderate signals
            
            # Calculate targets based on risk-reward multiples
            
            # TARGET 1: 1.5R (Book 50% position)
            target1_premium = itm_premium + (risk_per_share * 1.5 * target_mult)
            target1_profit_pct = ((target1_premium - itm_premium) / itm_premium) * 100
            
            results['target1'] = {
                'premium': round(target1_premium, 2),
                'profit_amount': round(target1_premium - itm_premium, 2),
                'profit_percent': round(target1_profit_pct, 2),
                'action': 'Book 50% position',
                'risk_reward': f"1:{round(1.5 * target_mult, 1)}",
                'move_stop_to': 'Breakeven (entry price)'
            }
            
            # TARGET 2: 2R (Book 30% more)
            target2_premium = itm_premium + (risk_per_share * 2.0 * target_mult)
            target2_profit_pct = ((target2_premium - itm_premium) / itm_premium) * 100
            
            results['target2'] = {
                'premium': round(target2_premium, 2),
                'profit_amount': round(target2_premium - itm_premium, 2),
                'profit_percent': round(target2_profit_pct, 2),
                'action': 'Book 30% position',
                'risk_reward': f"1:{round(2.0 * target_mult, 1)}",
                'move_stop_to': f"Target 1 (₹{target1_premium:.2f})"
            }
            
            # TARGET 3: 3R (Trail remaining 20%)
            target3_premium = itm_premium + (risk_per_share * 3.0 * target_mult)
            target3_profit_pct = ((target3_premium - itm_premium) / itm_premium) * 100
            
            results['target3'] = {
                'premium': round(target3_premium, 2),
                'profit_amount': round(target3_premium - itm_premium, 2),
                'profit_percent': round(target3_profit_pct, 2),
                'action': 'Trail remaining 20%',
                'risk_reward': f"1:{round(3.0 * target_mult, 1)}",
                'move_stop_to': 'Trail with 5-min swing points'
            }
            
            # ========== POSITION SIZING ==========
            
            # Calculate affordable lots based on capital
            cost_per_lot = itm_premium * lot_size
            
            # Apply position size multiplier based on confluence
            adjusted_capital = self.max_capital_per_trade * position_size_multiplier
            
            affordable_lots = int(adjusted_capital / cost_per_lot)
            recommended_lots = max(1, min(affordable_lots, self.max_lots))
            
            # Reduce lots further for moderate signals
            if confluence_score == 7:
                recommended_lots = max(1, recommended_lots // 2)
            
            total_capital_required = cost_per_lot * recommended_lots
            max_loss_amount = risk_per_share * lot_size * recommended_lots
            
            results['position_sizing'] = {
                'premium_per_share': itm_premium,
                'lot_size': lot_size,
                'recommended_lots': recommended_lots,
                'cost_per_lot': round(cost_per_lot, 2),
                'total_capital_required': round(total_capital_required, 2),
                'max_loss_amount': round(max_loss_amount, 2),
                'max_loss_percent': round(stop_percent * 100, 1),
                'position_size_factor': position_size_multiplier,
                'reason': self._get_position_size_reason(confluence_score)
            }
            
            # ========== TRAILING STOP STRATEGY ==========
            
            results['trailing_strategy'] = {
                'initial_stop': results['recommended_stop_loss'],
                'move_to_breakeven_at': f"{target1_profit_pct:.1f}% profit (Target 1)",
                'trail_from': f"{target2_profit_pct:.1f}% profit (Target 2)",
                'trail_method': f"Use 5-min swing {'lows' if direction == 'CALL' else 'highs'}",
                'trail_distance': 'Lock in 70% of profit as price moves favorably',
                'final_exit_time': '3:15 PM IST (mandatory intraday exit)',
                'exit_conditions': [
                    f"Premium falls to ₹{results['recommended_stop_loss']:.2f}",
                    f"Index {'breaks below' if direction == 'CALL' else 'breaks above'} ₹{index_stop:.2f}",
                    "Opposite candlestick pattern forms",
                    "3:15 PM IST reached"
                ]
            }
            
            # ========== RISK-REWARD SUMMARY ==========
            
            # Calculate potential rewards at each target
            target1_total_reward = results['target1']['profit_amount'] * lot_size * recommended_lots * 0.5  # 50% position
            target2_total_reward = results['target2']['profit_amount'] * lot_size * recommended_lots * 0.3  # 30% position
            target3_total_reward = results['target3']['profit_amount'] * lot_size * recommended_lots * 0.2  # 20% position
            
            # Expected reward if all targets hit
            expected_total_reward = target1_total_reward + target2_total_reward + target3_total_reward
            
            # Overall R:R ratio
            overall_rr = expected_total_reward / max_loss_amount if max_loss_amount > 0 else 0
            
            results['risk_reward_summary'] = {
                'max_risk': round(max_loss_amount, 2),
                'target1_reward': round(target1_total_reward, 2),
                'target2_reward': round(target2_total_reward, 2),
                'target3_reward': round(target3_total_reward, 2),
                'expected_total_reward': round(expected_total_reward, 2),
                'overall_risk_reward': round(overall_rr, 2),
                'win_probability': self._estimate_win_probability(confluence_score)
            }
            
            # ========== EXECUTION CHECKLIST ==========
            
            results['execution_checklist'] = {
                'before_entry': [
                    "✅ Confluence score >= 7",
                    "✅ Wait for 5-min candle close",
                    "✅ Confirm pattern completion",
                    "✅ Check volume spike",
                    "✅ Verify ITM strike selected"
                ],
                'at_entry': [
                    f"✅ Buy {recommended_lots} lot(s) of {option_type}",
                    f"✅ Entry price: ₹{itm_premium:.2f}",
                    f"✅ Set stop-loss alert at ₹{results['recommended_stop_loss']:.2f}",
                    f"✅ Set index stop alert at ₹{index_stop:.2f}",
                    f"✅ Capital deployed: ₹{total_capital_required:.2f}"
                ],
                'during_trade': [
                    f"✅ Book 50% at ₹{results['target1']['premium']:.2f}",
                    "✅ Move stop to breakeven",
                    f"✅ Book 30% at ₹{results['target2']['premium']:.2f}",
                    f"✅ Trail remaining 20% with 5-min swings",
                    "✅ Monitor for opposite signals"
                ],
                'exit_conditions': [
                    f"❌ Stop-loss hit: Exit immediately at ₹{results['recommended_stop_loss']:.2f}",
                    f"❌ Index stop breached: Exit at ₹{index_stop:.2f}",
                    "❌ Opposite pattern forms: Exit current candle",
                    "⏰ 3:15 PM IST: Exit all positions (no exceptions)"
                ]
            }
            
            # ========== TRADE QUALITY SCORE ==========
            
            trade_quality = self._calculate_trade_quality(
                confluence_score,
                overall_rr,
                results['position_sizing']['position_size_factor']
            )
            
            results['trade_quality'] = trade_quality
            
            return results
            
        except Exception as e:
            # Fallback with basic calculations
            return {
                'error': str(e),
                'premium_stop_loss': itm_premium * 0.80,
                'recommended_stop_loss': itm_premium * 0.80,
                'target1': {'premium': itm_premium * 1.30},
                'position_sizing': {'recommended_lots': 1}
            }
    
    def _calculate_swing_stop(self, df_5min: pd.DataFrame, direction: str) -> Optional[float]:
        """
        Calculate swing point for trailing stop based on 5-min chart
        For CALL: Recent swing low (last 10 candles)
        For PUT: Recent swing high (last 10 candles)
        """
        try:
            if df_5min is None or df_5min.empty or len(df_5min) < 10:
                return None
            
            recent_data = df_5min.tail(10)
            
            if direction == 'CALL':
                # Find lowest low in last 10 candles
                swing_low = recent_data['low'].min()
                return swing_low
            else:  # PUT
                # Find highest high in last 10 candles
                swing_high = recent_data['high'].max()
                return swing_high
                
        except Exception:
            return None
    
    def _get_position_size_reason(self, confluence_score: int) -> str:
        """Get explanation for position sizing"""
        if confluence_score >= 9:
            return "Very strong signal - full position size"
        elif confluence_score >= 8:
            return "Strong signal - full position size"
        elif confluence_score == 7:
            return "Moderate signal - reduced to 50% position size"
        else:
            return "Weak signal - minimal position size (25%)"
    
    def _estimate_win_probability(self, confluence_score: int) -> str:
        """Estimate win probability based on confluence score"""
        if confluence_score >= 9:
            return "75-80%"
        elif confluence_score >= 8:
            return "70-75%"
        elif confluence_score == 7:
            return "60-65%"
        else:
            return "50-55%"
    
    def _calculate_trade_quality(
        self,
        confluence_score: int,
        risk_reward_ratio: float,
        position_size_factor: float
    ) -> Dict:
        """
        Calculate overall trade quality score
        Combines confluence, R:R, and position sizing
        """
        
        quality_score = 0
        max_quality = 100
        
        # Confluence contribution (50 points max)
        quality_score += (confluence_score / 11) * 50
        
        # R:R contribution (30 points max)
        if risk_reward_ratio >= 2.0:
            quality_score += 30
        elif risk_reward_ratio >= 1.5:
            quality_score += 20
        else:
            quality_score += 10
        
        # Position sizing contribution (20 points max)
        quality_score += position_size_factor * 20
        
        quality_percent = (quality_score / max_quality) * 100
        
        # Determine quality grade
        if quality_percent >= 80:
            grade = 'A+'
            description = 'Excellent trade setup'
        elif quality_percent >= 70:
            grade = 'A'
            description = 'Very good trade setup'
        elif quality_percent >= 60:
            grade = 'B'
            description = 'Good trade setup'
        elif quality_percent >= 50:
            grade = 'C'
            description = 'Acceptable trade setup'
        else:
            grade = 'D'
            description = 'Weak trade setup - avoid'
        
        return {
            'score': round(quality_score, 1),
            'max_score': max_quality,
            'percent': round(quality_percent, 1),
            'grade': grade,
            'description': description
        }
