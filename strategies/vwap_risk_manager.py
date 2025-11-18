"""
VWAP Risk Manager - Comprehensive Risk Management
==================================================
Integrates all risk components:
- Position sizing (1% rule)
- Daily limits (2 trades, 3% loss)
- Transaction costs
- Trade journal logging

Author: Trading System (Analyst-Enhanced)
Date: November 18, 2025
"""

import logging
from typing import Dict, Optional
from datetime import datetime
from vwap_position_sizer import VWAPPositionSizer
from vwap_daily_risk_limiter import VWAPDailyRiskLimiter
from vwap_cost_tracker import VWAPCostTracker
from vwap_trade_journal import VWAPTradeJournal

logger = logging.getLogger(__name__)

class VWAPRiskManager:
    """
    Master risk manager coordinating all risk management components.
    Analyst: This is your command center for risk control.
    """
    
    def __init__(self, total_capital: float, journal_file: str = 'vwap_journal.csv'):
        """
        Initialize comprehensive risk manager.
        
        Args:
            total_capital: Total trading capital
            journal_file: Path to trade journal CSV
        """
        self.total_capital = total_capital
        
        # Initialize components
        self.position_sizer = VWAPPositionSizer(total_capital)
        self.daily_limiter = VWAPDailyRiskLimiter(total_capital)
        self.cost_tracker = VWAPCostTracker()
        self.journal = VWAPTradeJournal(journal_file)
        
        # Active trade tracking
        self.active_trade = None
        
        logger.info(f"VWAP Risk Manager initialized - Capital: ₹{total_capital:,.0f}")
    
    def can_take_trade(self) -> Dict:
        """
        Master check: Can we take a new trade?
        Checks all daily limits.
        
        Returns:
            dict: Permission status
        """
        return self.daily_limiter.can_take_trade()
    
    def calculate_position_size(self,
                                strategy_type: str,
                                entry_premium: float,
                                sl_premium: float,
                                index: str = 'NIFTY') -> Dict:
        """
        Calculate position size for trade.
        
        Args:
            strategy_type: 'SELLING' or 'BUYING'
            entry_premium: Entry premium
            sl_premium: Stop-loss premium
            index: Index name
        
        Returns:
            dict: Position sizing details
        """
        if strategy_type == 'SELLING':
            return self.position_sizer.calculate_lots_for_selling(
                entry_premium=entry_premium,
                sl_premium=sl_premium,
                index=index,
                include_costs=True
            )
        else:
            # For buying, calculate target from entry
            target_premium = entry_premium + 20  # Default 20-point target
            return self.position_sizer.calculate_lots_for_buying(
                entry_premium=entry_premium,
                target_premium=target_premium,
                index=index,
                include_costs=True
            )
    
    def start_trade(self,
                    trade_id: str,
                    strategy_type: str,
                    position_size_result: Dict,
                    entry_details: Dict) -> Dict:
        """
        Start a new trade with full risk management.
        
        Args:
            trade_id: Unique trade identifier
            strategy_type: 'SELLING' or 'BUYING'
            position_size_result: Output from calculate_position_size()
            entry_details: Additional entry details for journal
        
        Returns:
            dict: Trade initialization status
        """
        # Record entry with daily limiter
        self.daily_limiter.record_trade_entry(
            trade_id=trade_id,
            risk_amount=position_size_result['total_risk']
        )
        
        # Allocate capital
        self.position_sizer.allocate_position(
            position_id=trade_id,
            capital=position_size_result.get('capital_required', position_size_result['total_risk'])
        )
        
        # Store active trade
        self.active_trade = {
            'trade_id': trade_id,
            'strategy_type': strategy_type,
            'entry_time': datetime.now(),
            'entry_premium': entry_details.get('entry_premium'),
            'lots': position_size_result['lots'],
            'position_size_result': position_size_result,
            'entry_details': entry_details
        }
        
        logger.info(f"Trade {trade_id} started - {strategy_type} - {position_size_result['lots']} lots")
        
        return {
            'success': True,
            'trade_id': trade_id,
            'lots': position_size_result['lots'],
            'risk': position_size_result['total_risk']
        }
    
    def close_trade(self,
                    trade_id: str,
                    exit_premium: float,
                    exit_reason: str,
                    exit_type: str,
                    exit_details: Optional[Dict] = None) -> Dict:
        """
        Close trade and calculate final P&L including costs.
        
        Args:
            trade_id: Trade identifier
            exit_premium: Exit premium
            exit_reason: Reason for exit
            exit_type: Type of exit (TARGET/STOP_LOSS/TIME)
            exit_details: Additional exit details
        
        Returns:
            dict: Final trade P&L
        """
        if not self.active_trade or self.active_trade['trade_id'] != trade_id:
            logger.error(f"Trade {trade_id} not found in active trades")
            return {'success': False, 'reason': 'Trade not found'}
        
        trade = self.active_trade
        strategy_type = trade['strategy_type']
        entry_premium = trade['entry_premium']
        lots = trade['lots']
        
        # Get lot size
        index = trade['entry_details'].get('index', 'NIFTY')
        lot_size = self.position_sizer.lot_sizes.get(index, 25)
        
        # Calculate transaction costs
        num_legs = 4 if strategy_type == 'SELLING' else 1
        
        if strategy_type == 'SELLING':
            cost_result = self.cost_tracker.calculate_selling_costs(
                entry_premium=entry_premium,
                exit_premium=exit_premium,
                lot_size=lot_size,
                num_legs=num_legs
            )
            # For selling: lower exit = profit
            gross_pnl_points = entry_premium - exit_premium
        else:
            cost_result = self.cost_tracker.calculate_buying_costs(
                entry_premium=entry_premium,
                exit_premium=exit_premium,
                lot_size=lot_size,
                num_legs=num_legs
            )
            # For buying: higher exit = profit
            gross_pnl_points = exit_premium - entry_premium
        
        # Calculate P&L
        gross_pnl = gross_pnl_points * lot_size * lots
        total_costs = cost_result['total_cost'] * lots
        net_pnl = gross_pnl - total_costs
        
        # Record with daily limiter
        self.daily_limiter.record_trade_result(trade_id, net_pnl)
        
        # Release capital
        self.position_sizer.release_position(trade_id)
        
        # Log to journal
        journal_entry = {
            'date': datetime.now().date().isoformat(),
            'day_of_week': datetime.now().strftime('%A'),
            'strategy_type': strategy_type,
            'index': index,
            
            # Entry
            'entry_time': trade['entry_time'].strftime('%H:%M:%S'),
            'entry_premium': entry_premium,
            **trade['entry_details'],
            
            # Exit
            'exit_time': datetime.now().strftime('%H:%M:%S'),
            'exit_premium': exit_premium,
            'exit_reason': exit_reason,
            'exit_type': exit_type,
            'hold_duration_minutes': (datetime.now() - trade['entry_time']).total_seconds() / 60,
            
            # P&L
            'gross_pnl_points': gross_pnl_points,
            'gross_pnl_amount': gross_pnl,
            'transaction_costs': total_costs,
            'net_pnl_amount': net_pnl,
            'pnl_percent': (net_pnl / (entry_premium * lot_size * lots) * 100) if entry_premium > 0 else 0,
            
            # Position
            'lots_traded': lots,
            'capital_used': trade['position_size_result'].get('capital_required'),
            
            **(exit_details or {})
        }
        
        self.journal.log_trade(journal_entry)
        
        # Clear active trade
        self.active_trade = None
        
        result = {
            'success': True,
            'trade_id': trade_id,
            'gross_pnl': gross_pnl,
            'costs': total_costs,
            'net_pnl': net_pnl,
            'pnl_pct': journal_entry['pnl_percent'],
            'journal_entry': journal_entry
        }
        
        logger.info(f"Trade {trade_id} closed - Net P&L: ₹{net_pnl:.2f} (Gross: ₹{gross_pnl:.2f}, Costs: ₹{total_costs:.2f})")
        
        return result
    
    def get_daily_summary(self) -> Dict:
        """Get comprehensive daily summary"""
        limiter_summary = self.daily_limiter.get_daily_summary()
        journal_stats = self.journal.get_statistics()
        
        return {
            **limiter_summary,
            **journal_stats,
            'capital_utilization': self.position_sizer.get_utilization_pct()
        }
    
    def get_journal_stats(self) -> Dict:
        """Get journal statistics"""
        return self.journal.get_statistics()
    
    def export_journal(self, output_file: str = None) -> str:
        """Export journal to Excel"""
        return self.journal.export_for_review(output_file)
