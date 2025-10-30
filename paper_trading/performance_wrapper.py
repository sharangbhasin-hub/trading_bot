"""
Performance Analytics Wrapper for Paper Trading
================================================

Adapts the backtest performance_analyzer for real-time paper trading.
Provides comprehensive performance metrics and analysis.

Features:
- Real-time performance tracking
- Win rate, profit factor, Sharpe ratio
- Drawdown analysis
- Trade distribution analytics
- Per-strategy breakdowns

Author: Trading System
Last Updated: October 29, 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class PaperTradingPerformanceAnalyzer:
    """
    Performance analyzer adapted for paper trading.
    Uses existing PerformanceAnalyzer with paper trading data.
    """
    
    def __init__(self, trade_database):
        """
        Initialize performance analyzer.
        
        Args:
            trade_database: TradeDatabase instance
        """
        self.db = trade_database
        logger.info("PaperTradingPerformanceAnalyzer initialized")
    
    def get_performance_summary(self, days: int = 30) -> Dict:
        """
        Get comprehensive performance summary for period.
        
        Args:
            days: Number of days to analyze (default: 30)
        
        Returns:
            Dict with performance metrics
        """
        # Get closed trades
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        trades = self.db.get_closed_trades(start_date, end_date)
        
        if not trades:
            return self._empty_summary()
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(trades)
        
        # Calculate metrics
        metrics = {
            'period': {
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'days': days
            },
            'overall': self._calculate_overall_metrics(df),
            'returns': self._calculate_returns(df),
            'risk': self._calculate_risk_metrics(df),
            'efficiency': self._calculate_efficiency_metrics(df),
            'distribution': self._calculate_distribution(df),
            'strategy_breakdown': self._calculate_strategy_breakdown(df),
            'time_analysis': self._calculate_time_analysis(df)
        }
        
        return metrics
    
    def _calculate_overall_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate basic overall metrics."""
        total_trades = len(df)
        winning_trades = len(df[df['pnl_usd'] > 0])
        losing_trades = len(df[df['pnl_usd'] < 0])
        
        total_profit = df[df['pnl_usd'] > 0]['pnl_usd'].sum()
        total_loss = abs(df[df['pnl_usd'] < 0]['pnl_usd'].sum())
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            'total_pnl': df['pnl_usd'].sum(),
            'total_profit': total_profit,
            'total_loss': total_loss,
            'profit_factor': (total_profit / total_loss) if total_loss > 0 else 0,
            'avg_win': total_profit / winning_trades if winning_trades > 0 else 0,
            'avg_loss': total_loss / losing_trades if losing_trades > 0 else 0,
            'largest_win': df['pnl_usd'].max(),
            'largest_loss': df['pnl_usd'].min()
        }
    
    def _calculate_returns(self, df: pd.DataFrame) -> Dict:
        """Calculate return-based metrics."""
        df_sorted = df.sort_values('timestamp')
        
        # Cumulative returns
        df_sorted['cumulative_pnl'] = df_sorted['pnl_usd'].cumsum()
        
        return {
            'total_return_usd': df_sorted['cumulative_pnl'].iloc[-1],
            'avg_return_per_trade': df['pnl_usd'].mean(),
            'median_return_per_trade': df['pnl_usd'].median(),
            'std_return': df['pnl_usd'].std(),
            'best_trade': df['pnl_usd'].max(),
            'worst_trade': df['pnl_usd'].min(),
            'return_range': df['pnl_usd'].max() - df['pnl_usd'].min()
        }
    
    def _calculate_risk_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate risk metrics including Sharpe and drawdowns."""
        df_sorted = df.sort_values('timestamp')
        
        # Calculate equity curve
        df_sorted['cumulative_pnl'] = df_sorted['pnl_usd'].cumsum()
        
        # Sharpe Ratio (annualized)
        returns = df_sorted['pnl_usd']
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0
        
        # Drawdown analysis
        equity = df_sorted['cumulative_pnl']
        running_max = equity.cummax()
        drawdown = equity - running_max
        max_drawdown = drawdown.min()
        max_drawdown_pct = (max_drawdown / running_max.max() * 100) if running_max.max() > 0 else 0
        
        # Consecutive losses
        losses = (df_sorted['pnl_usd'] < 0).astype(int)
        consecutive_losses = (losses * (losses.groupby((losses != losses.shift()).cumsum()).cumcount() + 1)).max()
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_usd': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'consecutive_losses': int(consecutive_losses) if not pd.isna(consecutive_losses) else 0,
            'volatility': returns.std(),
            'downside_deviation': returns[returns < 0].std() if len(returns[returns < 0]) > 0 else 0
        }
    
    def _calculate_efficiency_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate efficiency metrics."""
        try:
            # ✅ FIX: Properly convert timestamps to datetime
            df_copy = df.copy()
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], errors='coerce')
            df_copy['exit_timestamp'] = pd.to_datetime(df_copy['exit_timestamp'], errors='coerce')
            
            # Remove rows where conversion failed
            df_copy = df_copy.dropna(subset=['timestamp', 'exit_timestamp'])
            
            if len(df_copy) == 0:
                return {
                    'avg_trade_duration': 'N/A',
                    'trades_per_day': 0,
                    'expectancy': 0,
                    'avg_win_duration': 'N/A',
                    'avg_loss_duration': 'N/A'
                }
            
            # Calculate average trade duration
            df_copy['duration'] = df_copy['exit_timestamp'] - df_copy['timestamp']
            avg_duration = df_copy['duration'].mean()
            
            # Trades per day
            date_range = (df_copy['timestamp'].max() - df_copy['timestamp'].min()).days
            trades_per_day = len(df_copy) / max(date_range, 1)
            
            # Expectancy
            winning_trades = df_copy[df_copy['pnl_usd'] > 0]
            losing_trades = df_copy[df_copy['pnl_usd'] < 0]
            
            win_rate = len(winning_trades) / len(df_copy) if len(df_copy) > 0 else 0
            avg_win = winning_trades['pnl_usd'].mean() if len(winning_trades) > 0 else 0
            avg_loss = abs(losing_trades['pnl_usd'].mean()) if len(losing_trades) > 0 else 0
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
            
            # Calculate durations for winners and losers
            avg_win_duration = str(df_copy[df_copy['pnl_usd'] > 0]['duration'].mean()) if len(winning_trades) > 0 else 'N/A'
            avg_loss_duration = str(df_copy[df_copy['pnl_usd'] < 0]['duration'].mean()) if len(losing_trades) > 0 else 'N/A'
            
            return {
                'avg_trade_duration': str(avg_duration),
                'trades_per_day': round(trades_per_day, 2),
                'expectancy': round(expectancy, 2),
                'avg_win_duration': avg_win_duration,
                'avg_loss_duration': avg_loss_duration
            }
        
        except Exception as e:
            logger.error(f"Error calculating efficiency metrics: {e}")
            return {
                'avg_trade_duration': 'Error',
                'trades_per_day': 0,
                'expectancy': 0,
                'avg_win_duration': 'Error',
                'avg_loss_duration': 'Error'
            }
    
    def _calculate_distribution(self, df: pd.DataFrame) -> Dict:
        """Calculate trade distribution by exit reason."""
        exit_reasons = df['exit_reason'].value_counts().to_dict()
        
        # Calculate P&L by exit reason
        exit_pnl = {}
        for reason in exit_reasons.keys():
            exit_pnl[reason] = df[df['exit_reason'] == reason]['pnl_usd'].sum()
        
        return {
            'by_exit_reason': exit_reasons,
            'pnl_by_exit_reason': exit_pnl,
            'sl_trades': df[df['exit_reason'] == 'SL']['pnl_usd'].count(),
            'tp_trades': df[df['exit_reason'] == 'TP']['pnl_usd'].count(),
            'manual_trades': df[df['exit_reason'] == 'MANUAL']['pnl_usd'].count()
        }
    
    def _calculate_strategy_breakdown(self, df: pd.DataFrame) -> Dict:
        """Calculate performance by strategy."""
        strategies = df['strategy_name'].unique()
        
        breakdown = {}
        for strategy in strategies:
            strategy_df = df[df['strategy_name'] == strategy]
            breakdown[strategy] = {
                'total_trades': len(strategy_df),
                'winning_trades': len(strategy_df[strategy_df['pnl_usd'] > 0]),
                'win_rate': (len(strategy_df[strategy_df['pnl_usd'] > 0]) / len(strategy_df) * 100) if len(strategy_df) > 0 else 0,
                'total_pnl': strategy_df['pnl_usd'].sum(),
                'avg_pnl': strategy_df['pnl_usd'].mean()
            }
        
        return breakdown
    
    def _calculate_time_analysis(self, df: pd.DataFrame) -> Dict:
        """Analyze performance by time periods."""
        try:
            df_copy = df.copy()
            
            # ✅ FIX: Convert timestamps with error handling..
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], errors='coerce')
            df_copy = df_copy.dropna(subset=['timestamp'])
            
            if len(df_copy) == 0:
                return {
                    'hourly_pnl': {},
                    'daily_pnl': {},
                    'best_hour': 0,
                    'worst_hour': 0,
                    'best_day': 0
                }
            
            # Extract hour and day of week
            df_copy['hour'] = df_copy['timestamp'].dt.hour
            df_copy['day_of_week'] = df_copy['timestamp'].dt.dayofweek
            
            # Group by hour and day
            hourly_pnl = df_copy.groupby('hour')['pnl_usd'].sum().to_dict()
            daily_pnl = df_copy.groupby('day_of_week')['pnl_usd'].sum().to_dict()
            
            # Best and worst hours
            hourly_avg = df_copy.groupby('hour')['pnl_usd'].mean()
            best_hour = int(hourly_avg.idxmax()) if not hourly_avg.empty else 0
            worst_hour = int(hourly_avg.idxmin()) if not hourly_avg.empty else 0
            
            # Best day
            daily_avg = df_copy.groupby('day_of_week')['pnl_usd'].mean()
            best_day = int(daily_avg.idxmax()) if not daily_avg.empty else 0
            
            return {
                'hourly_pnl': hourly_pnl,
                'daily_pnl': daily_pnl,
                'best_hour': best_hour,
                'worst_hour': worst_hour,
                'best_day': best_day
            }
        
        except Exception as e:
            logger.error(f"Error calculating time analysis: {e}")
            return {
                'hourly_pnl': {},
                'daily_pnl': {},
                'best_hour': 0,
                'worst_hour': 0,
                'best_day': 0
            }
    
    def _empty_summary(self) -> Dict:
        """Return empty summary when no trades."""
        return {
            'period': {},
            'overall': {'total_trades': 0},
            'returns': {},
            'risk': {},
            'efficiency': {},
            'distribution': {},
            'strategy_breakdown': {},
            'time_analysis': {}
        }
    
    def get_equity_curve(self, days: int = 30) -> pd.DataFrame:
        """
        Get equity curve data for charting.
        
        Args:
            days: Number of days
        
        Returns:
            DataFrame with timestamp and cumulative P&L
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        trades = self.db.get_closed_trades(start_date, end_date)
        
        if not trades:
            return pd.DataFrame(columns=['timestamp', 'equity'])
        
        df = pd.DataFrame(trades)
        df = df.sort_values('timestamp')
        df['cumulative_pnl'] = df['pnl_usd'].cumsum()
        
        return df[['timestamp', 'cumulative_pnl']].rename(columns={'cumulative_pnl': 'equity'})
    
    def get_monthly_summary(self) -> Dict:
        """Get month-by-month performance summary."""
        # Get all trades
        trades = self.db.get_closed_trades(
            datetime.now() - timedelta(days=365),
            datetime.now()
        )
        
        if not trades:
            return {}
        
        df = pd.DataFrame(trades)
        df['month'] = pd.to_datetime(df['timestamp']).dt.to_period('M')
        
        monthly = df.groupby('month').agg({
            'pnl_usd': ['sum', 'count'],
        }).reset_index()
        
        monthly.columns = ['month', 'pnl', 'trades']
        monthly['month'] = monthly['month'].astype(str)
        
        return monthly.to_dict('records')


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    from trade_database import TradeDatabase
    
    print("=" * 70)
    print("PERFORMANCE ANALYZER TEST")
    print("=" * 70)
    
    # Initialize
    db = TradeDatabase("paper_trading/data/test_trades.db")
    analyzer = PaperTradingPerformanceAnalyzer(db)
    
    # Get summary
    print("\n1️⃣ Getting performance summary...")
    summary = analyzer.get_performance_summary(days=30)
    
    print("\n📊 Overall Metrics:")
    if summary['overall']['total_trades'] > 0:
        print(f"   Total Trades: {summary['overall']['total_trades']}")
        print(f"   Win Rate: {summary['overall']['win_rate']:.1f}%")
        print(f"   Total P&L: ${summary['overall']['total_pnl']:,.2f}")
        print(f"   Profit Factor: {summary['overall']['profit_factor']:.2f}")
    else:
        print("   No trades found")
    
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE")
    print("=" * 70)
