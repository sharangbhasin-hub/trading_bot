"""
VWAP Trade Journal - Comprehensive Trade Logging
=================================================
Analyst's Requirement: "Obsessive journaling of every trade"

Tracks everything:
- Entry/exit premiums
- Market conditions (VIX, gap, structure)
- P&L breakdown
- Emotional state (CALM/ANXIOUS/REVENGE)
- Post-trade notes

Purpose: Build a database to analyze what works and what doesn't.

Author: Trading System (Analyst-Enhanced)
Date: November 18, 2025
"""

import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Optional, List
import json
import os

logger = logging.getLogger(__name__)

class VWAPTradeJournal:
    """
    Comprehensive trade journal for VWAP strategies.
    Analyst: "This is your most important tool for continuous improvement."
    """
    
    def __init__(self, journal_file: str = 'vwap_trade_journal.csv'):
        """
        Initialize trade journal.
        
        Args:
            journal_file: CSV file path for journal storage
        """
        self.journal_file = journal_file
        self.df = None
        self.load_or_create()
        
        logger.info(f"Trade Journal initialized: {journal_file}")
    
    def load_or_create(self):
        """Load existing journal or create new with proper schema"""
        try:
            self.df = pd.read_csv(self.journal_file)
            logger.info(f"Loaded existing journal: {len(self.df)} trades")
        except FileNotFoundError:
            # Create new journal with all required columns
            self.df = pd.DataFrame(columns=[
                # Trade Identification
                'trade_id',
                'date',
                'day_of_week',
                'strategy_type',  # SELLING or BUYING
                'index',  # NIFTY or SENSEX
                
                # Pre-Trade Analysis
                'market_classification',  # What did classifier say?
                'india_vix_entry',
                'market_gap_pct',
                'market_structure',  # RANGE, TRENDING, BREAKOUT
                'is_range_bound',
                'is_breakout',
                'conditions_score',  # How many conditions met (e.g., 3/4)
                
                # Entry Details
                'entry_time',
                'spot_price_930',  # 9:30 AM spot price
                'entry_premium',
                'entry_vwap',
                'vwap_crossover_direction',  # 'above' or 'below'
                'strikes_json',  # JSON of all strikes used
                
                # Position Sizing
                'lots_traded',
                'position_size_method',  # '1_percent_rule'
                'capital_used',
                'risk_amount',
                'risk_pct',
                
                # Exit Details
                'exit_time',
                'exit_premium',
                'exit_reason',  # TARGET, STOP_LOSS, TIME, etc.
                'exit_type',
                'hold_duration_minutes',
                
                # P&L Breakdown
                'gross_pnl_points',
                'gross_pnl_amount',
                'transaction_costs',
                'net_pnl_amount',
                'pnl_percent',
                'roi_pct',  # Return on capital used
                
                # Risk Metrics
                'initial_sl_premium',
                'actual_sl_hit',
                'target_premium',
                'actual_target_hit',
                'risk_reward_ratio',
                
                # Emotional & Discipline
                'confidence_at_entry',  # 0-100
                'emotional_state',  # CALM, ANXIOUS, REVENGE, FOMO
                'followed_rules',  # YES/NO
                'rule_violations',  # Text description if any
                
                # Market Context
                'india_vix_exit',
                'max_drawdown_during_trade',
                'max_profit_during_trade',
                
                # Notes
                'pre_trade_plan',
                'post_trade_review',
                'lessons_learned',
                'tags',  # Comma-separated tags
                
                # Metadata
                'created_at',
                'updated_at'
            ])
            logger.info("Created new trade journal")
    
    def log_trade(self, trade_data: dict):
        """
        Add trade to journal.
        Analyst: Fill this out IMMEDIATELY after closing trade while memory is fresh!
        
        Args:
            trade_data: Dictionary with all trade details
        """
        # Add metadata
        trade_data['trade_id'] = self._generate_trade_id()
        trade_data['created_at'] = datetime.now().isoformat()
        trade_data['updated_at'] = datetime.now().isoformat()
        
        # Validate required fields
        required_fields = ['date', 'strategy_type', 'entry_premium', 'exit_premium', 'net_pnl_amount']
        missing = [f for f in required_fields if f not in trade_data]
        if missing:
            logger.warning(f"Missing required fields: {missing}")
        
        # Add to dataframe
        self.df = pd.concat([self.df, pd.DataFrame([trade_data])], ignore_index=True)
        
        # Auto-save
        self.save()
        
        logger.info(f"Trade logged: {trade_data['trade_id']} - P&L: ₹{trade_data.get('net_pnl_amount', 0):.0f}")
    
    def update_trade(self, trade_id: str, updates: dict):
        """
        Update existing trade entry.
        Use this to add post-trade analysis.
        
        Args:
            trade_id: Trade identifier
            updates: Dictionary of fields to update
        """
        idx = self.df[self.df['trade_id'] == trade_id].index
        
        if len(idx) == 0:
            logger.warning(f"Trade {trade_id} not found")
            return
        
        updates['updated_at'] = datetime.now().isoformat()
        
        for key, value in updates.items():
            self.df.loc[idx, key] = value
        
        self.save()
        logger.info(f"Trade {trade_id} updated")
    
    def save(self):
        """Save journal to CSV"""
        self.df.to_csv(self.journal_file, index=False)
    
    def _generate_trade_id(self) -> str:
        """Generate unique trade ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        trade_num = len(self.df) + 1
        return f"VWAP_{timestamp}_{trade_num:04d}"
    
    def get_statistics(self) -> Dict:
        """
        Calculate comprehensive journal statistics.
        Analyst's Key Metrics: Win rate must be >60% to continue!
        
        Returns:
            dict: All statistics
        """
        if len(self.df) == 0:
            return {'total_trades': 0, 'message': 'No trades in journal'}
        
        # Filter closed trades only
        closed = self.df[self.df['exit_time'].notna()].copy()
        
        if len(closed) == 0:
            return {'total_trades': 0, 'message': 'No closed trades'}
        
        # Convert P&L to numeric
        closed['net_pnl_amount'] = pd.to_numeric(closed['net_pnl_amount'], errors='coerce')
        
        # Win/Loss split
        wins = closed[closed['net_pnl_amount'] > 0]
        losses = closed[closed['net_pnl_amount'] <= 0]
        
        # Calculate metrics
        total_trades = len(closed)
        num_wins = len(wins)
        num_losses = len(losses)
        win_rate = (num_wins / total_trades * 100) if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = closed['net_pnl_amount'].sum()
        avg_win = wins['net_pnl_amount'].mean() if len(wins) > 0 else 0
        avg_loss = losses['net_pnl_amount'].mean() if len(losses) > 0 else 0
        
        # Expectancy
        expectancy = closed['net_pnl_amount'].mean()
        
        # Largest win/loss
        largest_win = wins['net_pnl_amount'].max() if len(wins) > 0 else 0
        largest_loss = losses['net_pnl_amount'].min() if len(losses) > 0 else 0
        
        # Consecutive streaks
        current_streak = self._calculate_current_streak(closed)
        max_win_streak = self._calculate_max_streak(closed, 'win')
        max_loss_streak = self._calculate_max_streak(closed, 'loss')
        
        # By strategy type
        selling_stats = self._calculate_strategy_stats(closed, 'SELLING')
        buying_stats = self._calculate_strategy_stats(closed, 'BUYING')
        
        # Analyst's verdict
        meets_requirement = win_rate >= 60.0
        
        stats = {
            'total_trades': total_trades,
            'wins': num_wins,
            'losses': num_losses,
            'win_rate': round(win_rate, 2),
            
            # P&L
            'total_pnl': round(total_pnl, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'expectancy': round(expectancy, 2),
            'profit_factor': abs(wins['net_pnl_amount'].sum() / losses['net_pnl_amount'].sum()) if len(losses) > 0 and losses['net_pnl_amount'].sum() != 0 else None,
            
            # Extremes
            'largest_win': round(largest_win, 2),
            'largest_loss': round(largest_loss, 2),
            
            # Streaks
            'current_streak': current_streak,
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
            
            # By strategy
            'selling': selling_stats,
            'buying': buying_stats,
            
            # Analyst's verdict
            'meets_60pct_requirement': meets_requirement,
            'analyst_verdict': '✅ APPROVED for live trading' if meets_requirement and total_trades >= 20 else '❌ Continue paper trading'
        }
        
        return stats
    
    def _calculate_strategy_stats(self, df: pd.DataFrame, strategy_type: str) -> Dict:
        """Calculate stats for specific strategy type"""
        strategy_df = df[df['strategy_type'] == strategy_type]
        
        if len(strategy_df) == 0:
            return {'count': 0}
        
        strategy_df['net_pnl_amount'] = pd.to_numeric(strategy_df['net_pnl_amount'], errors='coerce')
        
        wins = strategy_df[strategy_df['net_pnl_amount'] > 0]
        
        return {
            'count': len(strategy_df),
            'wins': len(wins),
            'win_rate': (len(wins) / len(strategy_df) * 100) if len(strategy_df) > 0 else 0,
            'total_pnl': strategy_df['net_pnl_amount'].sum(),
            'avg_pnl': strategy_df['net_pnl_amount'].mean()
        }
    
    def _calculate_current_streak(self, df: pd.DataFrame) -> Dict:
        """Calculate current win/loss streak"""
        if len(df) == 0:
            return {'type': None, 'count': 0}
        
        # Sort by date
        df_sorted = df.sort_values('date')
        last_trades = df_sorted.tail(10)  # Look at last 10 trades
        
        streak_type = None
        streak_count = 0
        
        for _, trade in last_trades.iterrows():
            pnl = pd.to_numeric(trade['net_pnl_amount'], errors='coerce')
            if pd.isna(pnl):
                continue
            
            trade_type = 'win' if pnl > 0 else 'loss'
            
            if streak_type is None:
                streak_type = trade_type
                streak_count = 1
            elif streak_type == trade_type:
                streak_count += 1
            else:
                streak_type = trade_type
                streak_count = 1
        
        return {'type': streak_type, 'count': streak_count}
    
    def _calculate_max_streak(self, df: pd.DataFrame, streak_type: str) -> int:
        """Calculate maximum consecutive wins or losses"""
        if len(df) == 0:
            return 0
        
        df_sorted = df.sort_values('date')
        
        max_streak = 0
        current_streak = 0
        
        for _, trade in df_sorted.iterrows():
            pnl = pd.to_numeric(trade['net_pnl_amount'], errors='coerce')
            if pd.isna(pnl):
                continue
            
            is_win = pnl > 0
            
            if (streak_type == 'win' and is_win) or (streak_type == 'loss' and not is_win):
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def get_recent_trades(self, n: int = 10) -> pd.DataFrame:
        """Get n most recent trades"""
        if len(self.df) == 0:
            return pd.DataFrame()
        
        return self.df.sort_values('date', ascending=False).head(n)
    
    def analyze_by_market_condition(self) -> Dict:
        """
        Analyze performance by market condition.
        Analyst: Find out which market types you do best in!
        
        Returns:
            dict: Performance breakdown by market condition
        """
        if len(self.df) == 0:
            return {}
        
        closed = self.df[self.df['exit_time'].notna()].copy()
        closed['net_pnl_amount'] = pd.to_numeric(closed['net_pnl_amount'], errors='coerce')
        
        results = {}
        
        # By market structure
        for structure in ['RANGE', 'TRENDING', 'BREAKOUT']:
            structure_df = closed[closed['market_structure'] == structure]
            if len(structure_df) > 0:
                wins = structure_df[structure_df['net_pnl_amount'] > 0]
                results[structure] = {
                    'trades': len(structure_df),
                    'win_rate': (len(wins) / len(structure_df) * 100),
                    'avg_pnl': structure_df['net_pnl_amount'].mean()
                }
        
        # By VIX level
        closed['vix_category'] = pd.cut(closed['india_vix_entry'], 
                                        bins=[0, 13, 15, 20, 100],
                                        labels=['LOW (<13)', 'MEDIUM (13-15)', 'HIGH (15-20)', 'VERY HIGH (>20)'])
        
        for vix_cat in closed['vix_category'].dropna().unique():
            vix_df = closed[closed['vix_category'] == vix_cat]
            wins = vix_df[vix_df['net_pnl_amount'] > 0]
            results[f'VIX_{vix_cat}'] = {
                'trades': len(vix_df),
                'win_rate': (len(wins) / len(vix_df) * 100) if len(vix_df) > 0 else 0,
                'avg_pnl': vix_df['net_pnl_amount'].mean()
            }
        
        return results
    
    def export_for_review(self, output_file: str = None):
        """
        Export journal to Excel for detailed review.
        
        Args:
            output_file: Excel file path (default: auto-generated)
        """
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'vwap_journal_export_{timestamp}.xlsx'
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # All trades
            self.df.to_excel(writer, sheet_name='All Trades', index=False)
            
            # Statistics
            stats_df = pd.DataFrame([self.get_statistics()])
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)
            
            # By market condition
            market_analysis = self.analyze_by_market_condition()
            if market_analysis:
                market_df = pd.DataFrame(market_analysis).T
                market_df.to_excel(writer, sheet_name='By Market Condition')
        
        logger.info(f"Journal exported to: {output_file}")
        return output_file

# ============================================================================
# TEMPLATE FUNCTIONS
# ============================================================================

def create_trade_entry_template() -> Dict:
    """
    Create empty template for trade entry.
    Use this to ensure you don't miss any fields.
    
    Returns:
        dict: Template with all fields
    """
    return {
        'date': datetime.now().date().isoformat(),
        'day_of_week': datetime.now().strftime('%A'),
        'strategy_type': '',  # SELLING or BUYING
        'index': '',  # NIFTY or SENSEX
        
        # Pre-trade
        'market_classification': '',
        'india_vix_entry': None,
        'market_gap_pct': None,
        'market_structure': '',
        'is_range_bound': None,
        'is_breakout': None,
        'conditions_score': '',
        
        # Entry
        'entry_time': '',
        'spot_price_930': None,
        'entry_premium': None,
        'entry_vwap': None,
        'vwap_crossover_direction': '',
        'strikes_json': '',
        
        # Position
        'lots_traded': None,
        'position_size_method': '1_percent_rule',
        'capital_used': None,
        'risk_amount': None,
        'risk_pct': None,
        
        # Exit (fill after close)
        'exit_time': '',
        'exit_premium': None,
        'exit_reason': '',
        'exit_type': '',
        'hold_duration_minutes': None,
        
        # P&L (fill after close)
        'gross_pnl_points': None,
        'gross_pnl_amount': None,
        'transaction_costs': None,
        'net_pnl_amount': None,
        'pnl_percent': None,
        'roi_pct': None,
        
        # Risk metrics
        'initial_sl_premium': None,
        'actual_sl_hit': None,
        'target_premium': None,
        'actual_target_hit': None,
        'risk_reward_ratio': None,
        
        # Emotional (CRITICAL - analyst's requirement)
        'confidence_at_entry': None,  # 0-100
        'emotional_state': '',  # CALM, ANXIOUS, REVENGE, FOMO
        'followed_rules': 'YES',
        'rule_violations': '',
        
        # Notes (fill IMMEDIATELY after trade)
        'pre_trade_plan': '',
        'post_trade_review': '',
        'lessons_learned': '',
        'tags': ''
    }
