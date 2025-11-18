"""
VWAP Position Sizer - 1% Risk Rule Implementation
==================================================
Calculates optimal position size using the 1% rule.
Analyst's Foundation: "Maximum potential loss on any trade must not exceed 1% of capital"

Author: Trading System
Date: November 18, 2025
"""

import logging
from typing import Dict, Optional
from datetime import datetime
from config_vwap_strangle import (
    RISK_MANAGEMENT,
    TRANSACTION_COSTS,
    VWAP_STRANGLE_SELLING,
    VWAP_STRANGLE_BUYING
)

logger = logging.getLogger(__name__)

class VWAPPositionSizer:
    """
    Position sizing calculator for VWAP strategies.
    Implements 1% risk rule with transaction cost adjustments.
    """
    
    def __init__(self, total_capital: float):
        """
        Initialize position sizer.
        
        Args:
            total_capital: Total trading capital in ₹
        """
        self.total_capital = total_capital
        self.max_risk_pct = RISK_MANAGEMENT['selling']['max_loss_per_trade_pct']
        
        # Lot sizes for different indices
        self.lot_sizes = {
            'NIFTY': 25,
            'BANKNIFTY': 15,
            'SENSEX': 10,
            'FINNIFTY': 40
        }
        
        # Track current positions
        self.active_positions = []
        self.capital_in_use = 0.0
        
        logger.info(f"Position Sizer initialized - Capital: ₹{total_capital:,.0f}")
    
    def calculate_lots_for_selling(self,
                                   entry_premium: float,
                                   sl_premium: float,
                                   index: str = 'NIFTY',
                                   include_costs: bool = True) -> Dict:
        """
        Calculate position size for SELLING strategy (4-leg iron condor).
        
        Args:
            entry_premium: Combined premium at entry (CE+PE-hedges)
            sl_premium: Combined premium at stop-loss
            index: 'NIFTY', 'SENSEX', etc.
            include_costs: Include transaction costs in risk calculation
        
        Returns:
            dict: Position sizing details
        """
        lot_size = self.lot_sizes.get(index, 25)
        max_risk_amount = self.total_capital * (self.max_risk_pct / 100)
        
        # Calculate risk per lot (price difference * lot size)
        price_risk_per_lot = abs(sl_premium - entry_premium) * lot_size
        
        # Add transaction costs if requested
        if include_costs:
            num_legs = 4
            cost_per_lot = self._calculate_transaction_cost(num_legs, lot_size)
            total_risk_per_lot = price_risk_per_lot + cost_per_lot
        else:
            total_risk_per_lot = price_risk_per_lot
            cost_per_lot = 0
        
        # Calculate number of lots
        if total_risk_per_lot == 0:
            return self._error_result("Invalid setup - zero risk per lot")
        
        calculated_lots = max_risk_amount / total_risk_per_lot
        
        # Apply limits from config
        min_lots = VWAP_STRANGLE_SELLING.get('min_lots', 1)
        max_lots = VWAP_STRANGLE_SELLING.get('max_lots', 5)
        
        lots = int(calculated_lots)
        lots = max(min_lots, min(lots, max_lots))
        
        # Calculate actual risk with rounded lots
        actual_risk = lots * total_risk_per_lot
        actual_risk_pct = (actual_risk / self.total_capital) * 100
        
        # Calculate margin required
        spread_width = 400  # From config: hedge 400 points away
        margin_per_lot = spread_width * lot_size
        total_margin = margin_per_lot * lots
        
        result = {
            'strategy_type': 'SELLING',
            'index': index,
            'lot_size': lot_size,
            'lots': lots,
            'calculated_lots': calculated_lots,
            
            # Risk breakdown
            'entry_premium': entry_premium,
            'sl_premium': sl_premium,
            'price_risk_per_lot': price_risk_per_lot,
            'transaction_cost_per_lot': cost_per_lot,
            'total_risk_per_lot': total_risk_per_lot,
            'total_risk': actual_risk,
            'risk_pct': actual_risk_pct,
            'max_allowed_risk': max_risk_amount,
            
            # Capital requirements
            'margin_required': total_margin,
            'capital_available': self.total_capital - self.capital_in_use,
            'can_execute': total_margin <= (self.total_capital - self.capital_in_use),
            
            # Validation
            'within_risk_limit': actual_risk_pct <= self.max_risk_pct,
            'recommendation': self._get_recommendation(actual_risk_pct, total_margin)
        }
        
        logger.info(f"Position Size Calculated - Selling: {lots} lots, Risk: ₹{actual_risk:.0f} ({actual_risk_pct:.2f}%)")
        
        return result
    
    def calculate_lots_for_buying(self,
                                  entry_premium: float,
                                  target_premium: float,
                                  index: str = 'NIFTY',
                                  include_costs: bool = True) -> Dict:
        """
        Calculate position size for BUYING strategy (1-leg directional).
        Uses 1:2 Risk:Reward ratio to determine SL.
        
        Args:
            entry_premium: Entry premium for single option
            target_premium: Target premium
            index: Index name
            include_costs: Include transaction costs
        
        Returns:
            dict: Position sizing details
        """
        lot_size = self.lot_sizes.get(index, 25)
        max_risk_amount = self.total_capital * (self.max_risk_pct / 100)
        
        # Calculate SL using 1:2 R:R
        profit_target = target_premium - entry_premium
        sl_distance = profit_target / 2  # Half of profit for 1:2 R:R
        sl_premium = entry_premium - sl_distance
        
        # Calculate risk per lot
        price_risk_per_lot = abs(entry_premium - sl_premium) * lot_size
        
        # Add transaction costs
        if include_costs:
            num_legs = 1
            cost_per_lot = self._calculate_transaction_cost(num_legs, lot_size)
            total_risk_per_lot = price_risk_per_lot + cost_per_lot
        else:
            total_risk_per_lot = price_risk_per_lot
            cost_per_lot = 0
        
        # Calculate lots
        if total_risk_per_lot == 0:
            return self._error_result("Invalid setup - zero risk")
        
        calculated_lots = max_risk_amount / total_risk_per_lot
        
        # Apply limits
        min_lots = VWAP_STRANGLE_BUYING.get('min_lots', 1)
        max_lots = VWAP_STRANGLE_BUYING.get('max_lots', 3)
        
        lots = int(calculated_lots)
        lots = max(min_lots, min(lots, max_lots))
        
        # Calculate actual values
        actual_risk = lots * total_risk_per_lot
        actual_risk_pct = (actual_risk / self.total_capital) * 100
        
        # For buying, capital required = premium * lot size * lots
        capital_required = entry_premium * lot_size * lots
        
        result = {
            'strategy_type': 'BUYING',
            'index': index,
            'lot_size': lot_size,
            'lots': lots,
            'calculated_lots': calculated_lots,
            
            # Risk breakdown
            'entry_premium': entry_premium,
            'sl_premium': sl_premium,
            'target_premium': target_premium,
            'risk_reward_ratio': 2.0,
            'price_risk_per_lot': price_risk_per_lot,
            'transaction_cost_per_lot': cost_per_lot,
            'total_risk_per_lot': total_risk_per_lot,
            'total_risk': actual_risk,
            'risk_pct': actual_risk_pct,
            'max_allowed_risk': max_risk_amount,
            
            # Capital requirements
            'capital_required': capital_required,
            'capital_available': self.total_capital - self.capital_in_use,
            'can_execute': capital_required <= (self.total_capital - self.capital_in_use),
            
            # Validation
            'within_risk_limit': actual_risk_pct <= self.max_risk_pct,
            'recommendation': self._get_recommendation(actual_risk_pct, capital_required)
        }
        
        logger.info(f"Position Size Calculated - Buying: {lots} lots, Risk: ₹{actual_risk:.0f} ({actual_risk_pct:.2f}%)")
        
        return result
    
    def _calculate_transaction_cost(self, num_legs: int, lot_size: int) -> float:
        """Calculate total transaction cost per lot (entry + exit)."""
        slippage_per_leg = TRANSACTION_COSTS['slippage_points_per_leg']
        total_slippage_points = slippage_per_leg * num_legs * 2  # *2 for entry+exit
        slippage_amount = total_slippage_points * lot_size
        
        brokerage_per_order = TRANSACTION_COSTS['brokerage_per_order']
        total_brokerage = brokerage_per_order * 2  # Entry + exit
        
        gst = total_brokerage * (TRANSACTION_COSTS['gst_on_brokerage_pct'] / 100)
        
        total_cost = slippage_amount + total_brokerage + gst
        
        return total_cost
    
    def _get_recommendation(self, risk_pct: float, capital_required: float) -> str:
        """Generate recommendation based on risk metrics"""
        if risk_pct > self.max_risk_pct:
            return f"⚠️ RISK TOO HIGH: {risk_pct:.2f}% exceeds {self.max_risk_pct}% limit"
        elif capital_required > (self.total_capital - self.capital_in_use):
            return f"⚠️ INSUFFICIENT CAPITAL"
        elif risk_pct < 0.5:
            return f"✅ CONSERVATIVE: Risk {risk_pct:.2f}%"
        elif risk_pct < 1.0:
            return f"✅ OPTIMAL: Risk {risk_pct:.2f}%"
        else:
            return f"⚠️ CAUTION: Risk {risk_pct:.2f}% at limit"
    
    def _error_result(self, reason: str) -> Dict:
        """Return error result"""
        return {
            'lots': 0,
            'error': True,
            'reason': reason,
            'can_execute': False
        }
    
    def allocate_position(self, position_id: str, capital: float):
        """Allocate capital to a position."""
        self.active_positions.append({
            'id': position_id,
            'capital': capital,
            'timestamp': datetime.now()
        })
        self.capital_in_use += capital
        logger.info(f"Capital allocated: ₹{capital:.0f} to position {position_id}")
    
    def release_position(self, position_id: str):
        """Release capital from closed position"""
        for pos in self.active_positions:
            if pos['id'] == position_id:
                self.capital_in_use -= pos['capital']
                self.active_positions.remove(pos)
                logger.info(f"Capital released: ₹{pos['capital']:.0f}")
                break
    
    def get_available_capital(self) -> float:
        """Get available capital for new positions"""
        return self.total_capital - self.capital_in_use
