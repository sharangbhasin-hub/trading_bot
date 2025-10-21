"""
Exit Analyzer
Analyzes exit strategies and suggests optimizations
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ExitAnalyzer:
    """
    Analyzes trade exits and suggests improvements
    """
    
    def __init__(self, trades_df):
        """
        Initialize exit analyzer
        
        Args:
            trades_df: DataFrame of all trades
        """
        self.trades_df = trades_df
        
    def analyze_mfe_mae(self):
        """
        Analyze Max Favorable/Adverse Excursion
        
        Returns:
            Dict with MFE/MAE analysis
        """
        if self.trades_df.empty:
            return {}
        
        df = self.trades_df.copy()
        
        wins = df[df['pnl'] > 0]
        losses = df[df['pnl'] <= 0]
        
        analysis = {
            'total_trades': len(df),
            'avg_mfe': df['max_favorable_excursion'].mean(),
            'avg_mae': df['max_adverse_excursion'].mean(),
        }
        
        # Analyze winners
        if len(wins) > 0:
            # Calculate how much more they could have made
            wins['target_distance'] = abs(wins['target'] - wins['entry_price'])
            wins['mfe_vs_target'] = wins['max_favorable_excursion'] / wins['target_distance']
            
            analysis['wins_reached_2x_target_pct'] = len(wins[wins['mfe_vs_target'] >= 2.0]) / len(wins) * 100
            analysis['avg_mfe_vs_target_ratio'] = wins['mfe_vs_target'].mean()
            
            if analysis['avg_mfe_vs_target_ratio'] > 1.5:
                analysis['target_recommendation'] = f"Increase targets by {(analysis['avg_mfe_vs_target_ratio'] - 1) * 100:.0f}%"
            else:
                analysis['target_recommendation'] = "Targets are appropriate"
        
        # Analyze losers
        if len(losses) > 0:
            # Calculate if stops were too wide
            losses['stop_distance'] = abs(losses['stop_loss'] - losses['entry_price'])
            losses['mae_vs_stop'] = abs(losses['max_adverse_excursion']) / losses['stop_distance']
            
            analysis['losses_with_mae_less_than_stop_pct'] = len(losses[losses['mae_vs_stop'] < 0.7]) / len(losses) * 100
            analysis['avg_mae_vs_stop_ratio'] = losses['mae_vs_stop'].mean()
            
            if analysis['avg_mae_vs_stop_ratio'] < 0.7:
                analysis['stop_recommendation'] = f"Tighten stops by {(1 - analysis['avg_mae_vs_stop_ratio']) * 100:.0f}%"
            else:
                analysis['stop_recommendation'] = "Stops are appropriate"
        
        return analysis
    
    def analyze_holding_periods(self):
        """
        Analyze optimal holding periods
        
        Returns:
            Dict with holding period analysis
        """
        if self.trades_df.empty:
            return {}
        
        df = self.trades_df.copy()
        
        wins = df[df['pnl'] > 0]
        losses = df[df['pnl'] <= 0]
        
        analysis = {
            'avg_holding_period_all': df['holding_period_minutes'].mean(),
            'median_holding_period_all': df['holding_period_minutes'].median(),
        }
        
        if len(wins) > 0:
            analysis['avg_holding_period_wins'] = wins['holding_period_minutes'].mean()
            analysis['median_holding_period_wins'] = wins['holding_period_minutes'].median()
            
            # Calculate what % of wins happened quickly
            quick_wins = wins[wins['holding_period_minutes'] <= 60]
            analysis['quick_wins_pct'] = len(quick_wins) / len(wins) * 100
        
        if len(losses) > 0:
            analysis['avg_holding_period_losses'] = losses['holding_period_minutes'].mean()
            analysis['median_holding_period_losses'] = losses['holding_period_minutes'].median()
            
            # Check if losses that held too long
            long_losses = losses[losses['holding_period_minutes'] > 180]  # 3 hours
            analysis['long_losses_pct'] = len(long_losses) / len(losses) * 100
            
            if analysis['long_losses_pct'] > 30:
                analysis['time_stop_recommendation'] = "Add time-based stop: Close if no target in 2-3 hours"
            else:
                analysis['time_stop_recommendation'] = "No time-based stop needed"
        
        return analysis
    
    def suggest_exit_improvements(self):
        """
        Suggest specific exit rule improvements
        
        Returns:
            List of recommendations
        """
        mfe_mae = self.analyze_mfe_mae()
        holding = self.analyze_holding_periods()
        
        recommendations = []
        
        # Target recommendations
        if 'target_recommendation' in mfe_mae:
            recommendations.append({
                'category': 'TARGET',
                'recommendation': mfe_mae['target_recommendation'],
                'priority': 'HIGH' if 'Increase' in mfe_mae['target_recommendation'] else 'MEDIUM'
            })
        
        # Stop loss recommendations
        if 'stop_recommendation' in mfe_mae:
            recommendations.append({
                'category': 'STOP_LOSS',
                'recommendation': mfe_mae['stop_recommendation'],
                'priority': 'HIGH' if 'Tighten' in mfe_mae['stop_recommendation'] else 'MEDIUM'
            })
        
        # Time-based stop recommendations
        if 'time_stop_recommendation' in holding:
            recommendations.append({
                'category': 'TIME_BASED_EXIT',
                'recommendation': holding['time_stop_recommendation'],
                'priority': 'MEDIUM' if 'Add' in holding['time_stop_recommendation'] else 'LOW'
            })
        
        return recommendations
