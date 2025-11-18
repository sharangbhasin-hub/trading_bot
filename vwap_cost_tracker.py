"""
Transaction Cost Tracker for VWAP Strategies
=============================================
Tracks all transaction costs: slippage, brokerage, STT, GST.
Analyst's Reality Check: "Assume 2-5 points slippage + ₹40-60 brokerage"

These costs eat into profits and must be accounted in backtests!

Author: Trading System (Analyst-Enhanced)
Date: November 18, 2025
"""

import logging
from typing import Dict, List
from datetime import datetime
from config_vwap_strangle import TRANSACTION_COSTS

logger = logging.getLogger(__name__)

class VWAPCostTracker:
    """
    Comprehensive transaction cost tracking.
    Ensures realistic P&L calculations by accounting for all costs.
    """
    
    def __init__(self):
        """Initialize cost tracker with config values"""
        self.slippage_per_leg = TRANSACTION_COSTS['slippage_points_per_leg']
        self.brokerage_per_order = TRANSACTION_COSTS['brokerage_per_order']
        self.stt_pct = TRANSACTION_COSTS['stt_on_sell_pct']
        self.gst_pct = TRANSACTION_COSTS['gst_on_brokerage_pct']
        
        # Historical cost tracking
        self.cost_history = []
        
        logger.info("Cost Tracker initialized")
        logger.info(f"Slippage: {self.slippage_per_leg} points/leg, Brokerage: ₹{self.brokerage_per_order}/order")
    
    def calculate_selling_costs(self, 
                                entry_premium: float,
                                exit_premium: float,
                                lot_size: int = 25,
                                num_legs: int = 4) -> Dict:
        """
        Calculate all costs for SELLING strategy (4-leg iron condor).
        
        Analyst's breakdown:
        - Slippage: 2.5 points * 4 legs * 2 (entry+exit) = 20 points per lot
        - Brokerage: ₹50 * 2 (entry+exit) = ₹100
        - GST: ₹100 * 18% = ₹18
        - Total: 20 * 25 + 100 + 18 = ₹618 per lot
        
        Args:
            entry_premium: Combined premium at entry
            exit_premium: Combined premium at exit
            lot_size: Lot size (25 for Nifty)
            num_legs: Number of legs (4 for iron condor)
        
        Returns:
            dict: Detailed cost breakdown
        """
        # 1. SLIPPAGE COSTS
        # Entry slippage: Each leg gets slipped
        entry_slippage_points = self.slippage_per_leg * num_legs
        entry_slippage_amount = entry_slippage_points * lot_size
        
        # Exit slippage: Same
        exit_slippage_points = self.slippage_per_leg * num_legs
        exit_slippage_amount = exit_slippage_points * lot_size
        
        total_slippage_points = entry_slippage_points + exit_slippage_points
        total_slippage_amount = entry_slippage_amount + exit_slippage_amount
        
        # 2. BROKERAGE COSTS
        # Entry: 1 basket order
        entry_brokerage = self.brokerage_per_order
        
        # Exit: 1 basket order
        exit_brokerage = self.brokerage_per_order
        
        total_brokerage = entry_brokerage + exit_brokerage
        
        # 3. GST ON BROKERAGE
        gst_amount = total_brokerage * (self.gst_pct / 100)
        
        # 4. STT (Securities Transaction Tax)
        # For options: 0.05% on SELL side (premium * quantity)
        # Simplified: Negligible for most trades
        stt_amount = 0  # Can be calculated if needed
        
        # 5. EXCHANGE CHARGES (minimal, ignored for now)
        exchange_charges = 0
        
        # TOTAL COST
        total_cost = total_slippage_amount + total_brokerage + gst_amount + stt_amount + exchange_charges
        
        # Calculate impact on P&L
        gross_pnl = (entry_premium - exit_premium) * lot_size  # For selling, lower exit = profit
        net_pnl = gross_pnl - total_cost
        cost_impact_pct = (total_cost / abs(gross_pnl) * 100) if gross_pnl != 0 else 0
        
        result = {
            'strategy_type': 'SELLING',
            'num_legs': num_legs,
            'lot_size': lot_size,
            
            # Slippage breakdown
            'slippage': {
                'entry_points': entry_slippage_points,
                'exit_points': exit_slippage_points,
                'total_points': total_slippage_points,
                'entry_amount': entry_slippage_amount,
                'exit_amount': exit_slippage_amount,
                'total_amount': total_slippage_amount
            },
            
            # Brokerage breakdown
            'brokerage': {
                'entry': entry_brokerage,
                'exit': exit_brokerage,
                'total': total_brokerage
            },
            
            # Taxes and charges
            'gst': gst_amount,
            'stt': stt_amount,
            'exchange_charges': exchange_charges,
            
            # Total costs
            'total_cost': total_cost,
            'cost_per_lot': total_cost,
            
            # P&L impact
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'cost_impact_pct': cost_impact_pct,
            
            # Timestamp
            'calculated_at': datetime.now()
        }
        
        # Store in history
        self.cost_history.append(result)
        
        logger.info(f"Selling Costs: Total ₹{total_cost:.2f} (Slippage: ₹{total_slippage_amount:.0f}, Brokerage+GST: ₹{total_brokerage + gst_amount:.0f})")
        
        return result
    
    def calculate_buying_costs(self,
                               entry_premium: float,
                               exit_premium: float,
                               lot_size: int = 25,
                               num_legs: int = 1) -> Dict:
        """
        Calculate all costs for BUYING strategy (1-leg directional).
        
        Analyst's breakdown (simpler than selling):
        - Slippage: 2.5 points * 1 leg * 2 (entry+exit) = 5 points per lot
        - Brokerage: ₹50 * 2 = ₹100
        - GST: ₹100 * 18% = ₹18
        - Total: 5 * 25 + 100 + 18 = ₹243 per lot
        
        Args:
            entry_premium: Entry premium
            exit_premium: Exit premium
            lot_size: Lot size
            num_legs: Number of legs (1 for single option)
        
        Returns:
            dict: Detailed cost breakdown
        """
        # 1. SLIPPAGE
        entry_slippage_points = self.slippage_per_leg * num_legs
        entry_slippage_amount = entry_slippage_points * lot_size
        
        exit_slippage_points = self.slippage_per_leg * num_legs
        exit_slippage_amount = exit_slippage_points * lot_size
        
        total_slippage_points = entry_slippage_points + exit_slippage_points
        total_slippage_amount = entry_slippage_amount + exit_slippage_amount
        
        # 2. BROKERAGE
        entry_brokerage = self.brokerage_per_order
        exit_brokerage = self.brokerage_per_order
        total_brokerage = entry_brokerage + exit_brokerage
        
        # 3. GST
        gst_amount = total_brokerage * (self.gst_pct / 100)
        
        # 4. STT (minimal for buying)
        stt_amount = 0
        
        # TOTAL
        total_cost = total_slippage_amount + total_brokerage + gst_amount + stt_amount
        
        # P&L impact
        gross_pnl = (exit_premium - entry_premium) * lot_size  # For buying, higher exit = profit
        net_pnl = gross_pnl - total_cost
        cost_impact_pct = (total_cost / abs(gross_pnl) * 100) if gross_pnl != 0 else 0
        
        result = {
            'strategy_type': 'BUYING',
            'num_legs': num_legs,
            'lot_size': lot_size,
            
            # Slippage
            'slippage': {
                'entry_points': entry_slippage_points,
                'exit_points': exit_slippage_points,
                'total_points': total_slippage_points,
                'entry_amount': entry_slippage_amount,
                'exit_amount': exit_slippage_amount,
                'total_amount': total_slippage_amount
            },
            
            # Brokerage
            'brokerage': {
                'entry': entry_brokerage,
                'exit': exit_brokerage,
                'total': total_brokerage
            },
            
            # Taxes
            'gst': gst_amount,
            'stt': stt_amount,
            'exchange_charges': 0,
            
            # Total
            'total_cost': total_cost,
            'cost_per_lot': total_cost,
            
            # P&L impact
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'cost_impact_pct': cost_impact_pct,
            
            'calculated_at': datetime.now()
        }
        
        self.cost_history.append(result)
        
        logger.info(f"Buying Costs: Total ₹{total_cost:.2f} (Slippage: ₹{total_slippage_amount:.0f}, Brokerage+GST: ₹{total_brokerage + gst_amount:.0f})")
        
        return result
    
    def get_average_cost_per_trade(self, strategy_type: str = None) -> float:
        """
        Calculate average cost per trade from history.
        
        Args:
            strategy_type: 'SELLING', 'BUYING', or None for all
        
        Returns:
            float: Average cost in ₹
        """
        if not self.cost_history:
            return 0.0
        
        if strategy_type:
            costs = [c['total_cost'] for c in self.cost_history if c['strategy_type'] == strategy_type]
        else:
            costs = [c['total_cost'] for c in self.cost_history]
        
        return sum(costs) / len(costs) if costs else 0.0
    
    def get_cost_summary(self) -> Dict:
        """
        Get summary of all tracked costs.
        Analyst: Use this to validate your backtest includes realistic costs!
        
        Returns:
            dict: Cost statistics
        """
        if not self.cost_history:
            return {'total_trades': 0}
        
        total_cost = sum(c['total_cost'] for c in self.cost_history)
        total_slippage = sum(c['slippage']['total_amount'] for c in self.cost_history)
        total_brokerage = sum(c['brokerage']['total'] for c in self.cost_history)
        total_gst = sum(c['gst'] for c in self.cost_history)
        
        # Separate by strategy type
        selling_costs = [c for c in self.cost_history if c['strategy_type'] == 'SELLING']
        buying_costs = [c for c in self.cost_history if c['strategy_type'] == 'BUYING']
        
        return {
            'total_trades': len(self.cost_history),
            'total_cost': total_cost,
            'avg_cost_per_trade': total_cost / len(self.cost_history),
            
            # Breakdown
            'total_slippage': total_slippage,
            'total_brokerage': total_brokerage,
            'total_gst': total_gst,
            
            # By strategy
            'selling': {
                'count': len(selling_costs),
                'total_cost': sum(c['total_cost'] for c in selling_costs),
                'avg_cost': sum(c['total_cost'] for c in selling_costs) / len(selling_costs) if selling_costs else 0
            },
            'buying': {
                'count': len(buying_costs),
                'total_cost': sum(c['total_cost'] for c in buying_costs),
                'avg_cost': sum(c['total_cost'] for c in buying_costs) / len(buying_costs) if buying_costs else 0
            },
            
            # Analyst's key metric
            'cost_as_pct_of_avg_trade': None  # Calculate if we have P&L data
        }
    
    def estimate_monthly_costs(self, trades_per_month: int, strategy_type: str = 'SELLING') -> Dict:
        """
        Estimate monthly transaction costs based on trading frequency.
        Analyst: Factor this into your monthly return expectations!
        
        Args:
            trades_per_month: Expected number of trades per month
            strategy_type: 'SELLING' or 'BUYING'
        
        Returns:
            dict: Monthly cost estimates
        """
        # Use historical average or calculate fresh
        if strategy_type == 'SELLING':
            avg_cost = self.get_average_cost_per_trade('SELLING')
            if avg_cost == 0:
                # Calculate from scratch
                avg_cost = self.calculate_selling_costs(150, 90, 25, 4)['total_cost']
        else:
            avg_cost = self.get_average_cost_per_trade('BUYING')
            if avg_cost == 0:
                avg_cost = self.calculate_buying_costs(50, 70, 25, 1)['total_cost']
        
        monthly_cost = avg_cost * trades_per_month
        
        # Analyst's example: 15 trades/month
        # Selling: ₹618 * 15 = ₹9,270/month in costs
        # On ₹2L capital, that's 4.6% reduction in returns!
        
        return {
            'strategy_type': strategy_type,
            'trades_per_month': trades_per_month,
            'avg_cost_per_trade': avg_cost,
            'monthly_cost': monthly_cost,
            'annual_cost': monthly_cost * 12,
            
            # Impact on different capital levels
            'impact_on_200k_capital': (monthly_cost / 200000) * 100,  # % per month
            'impact_on_500k_capital': (monthly_cost / 500000) * 100
        }

# ============================================================================
# STANDALONE FUNCTIONS
# ============================================================================

def quick_cost_estimate(strategy_type: str, lot_size: int = 25) -> float:
    """
    Quick estimate of transaction costs.
    
    Args:
        strategy_type: 'SELLING' or 'BUYING'
        lot_size: Lot size
    
    Returns:
        float: Estimated cost in ₹
    """
    tracker = VWAPCostTracker()
    
    if strategy_type == 'SELLING':
        result = tracker.calculate_selling_costs(150, 90, lot_size, 4)
    else:
        result = tracker.calculate_buying_costs(50, 70, lot_size, 1)
    
    return result['total_cost']
