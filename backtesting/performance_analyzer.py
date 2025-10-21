"""
Performance Analyzer
Calculates comprehensive performance metrics from backtest results
"""
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from backtesting.config import BacktestConfig

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Analyzes backtest performance and calculates metrics
    """
    
    def __init__(self, trades_df, signals_df):
        """
        Initialize performance analyzer
        
        Args:
            trades_df: DataFrame of all closed trades
            signals_df: DataFrame of all signals generated
        """
        self.trades_df = trades_df
        self.signals_df = signals_df
        self.config = BacktestConfig()
        
    def calculate_all_metrics(self):
        """
        Calculate all performance metrics
        
        Returns:
            Dict with comprehensive metrics
        """
        if self.trades_df.empty:
            return self._empty_metrics()
        
        metrics = {}
        
        # Basic metrics
        metrics.update(self._calculate_basic_metrics())
        
        # Win/Loss analysis
        metrics.update(self._calculate_win_loss_metrics())
        
        # Risk metrics
        metrics.update(self._calculate_risk_metrics())
        
        # Time-based metrics
        metrics.update(self._calculate_time_metrics())
        
        # Strategy-specific metrics
        metrics.update(self._calculate_strategy_metrics())
        
        # Confidence analysis
        metrics.update(self._calculate_confidence_metrics())
        
        # Signal vs execution analysis
        metrics.update(self._calculate_signal_metrics())
        
        return metrics
    
    def _empty_metrics(self):
        """Return empty metrics dict"""
        return {
            'total_trades': 0,
            'total_signals': len(self.signals_df),
            'win_rate': 0,
            'total_pnl': 0,
            'message': 'No trades executed'
        }
    
    def _calculate_basic_metrics(self):
        """Calculate basic performance metrics"""
        df = self.trades_df
        
        total_trades = len(df)
        wins = df[df['pnl'] > 0]
        losses = df[df['pnl'] <= 0]
        
        return {
            'total_trades': total_trades,
            'total_signals': len(self.signals_df),
            'signals_converted_to_trades': total_trades / len(self.signals_df) * 100 if len(self.signals_df) > 0 else 0,
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / total_trades * 100 if total_trades > 0 else 0,
            'loss_rate': len(losses) / total_trades * 100 if total_trades > 0 else 0,
        }
    
    def _calculate_win_loss_metrics(self):
        """Calculate win/loss specific metrics"""
        df = self.trades_df
        wins = df[df['pnl'] > 0]
        losses = df[df['pnl'] <= 0]
        
        total_wins_pnl = wins['pnl'].sum() if len(wins) > 0 else 0
        total_losses_pnl = abs(losses['pnl'].sum()) if len(losses) > 0 else 0
        
        return {
            'total_pnl': df['pnl'].sum(),
            'total_pnl_percent': df['pnl_percent'].sum(),
            'avg_pnl': df['pnl'].mean(),
            'avg_pnl_percent': df['pnl_percent'].mean(),
            'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
            'avg_win_percent': wins['pnl_percent'].mean() if len(wins) > 0 else 0,
            'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0,
            'avg_loss_percent': losses['pnl_percent'].mean() if len(losses) > 0 else 0,
            'largest_win': wins['pnl'].max() if len(wins) > 0 else 0,
            'largest_loss': losses['pnl'].min() if len(losses) > 0 else 0,
            'profit_factor': total_wins_pnl / total_losses_pnl if total_losses_pnl > 0 else 0,
            'expectancy': (len(wins) * wins['pnl'].mean() + len(losses) * losses['pnl'].mean()) / len(df) if len(df) > 0 else 0,
        }
    
    def _calculate_risk_metrics(self):
        """Calculate risk-related metrics"""
        df = self.trades_df.copy()
        
        # Sort by date/time
        df_sorted = df.sort_values(['entry_date', 'entry_time'])
        
        # Calculate cumulative P&L
        df_sorted['cumulative_pnl'] = df_sorted['pnl'].cumsum()
        
        # Calculate running maximum
        df_sorted['running_max'] = df_sorted['cumulative_pnl'].cummax()
        
        # Calculate drawdown
        df_sorted['drawdown'] = df_sorted['running_max'] - df_sorted['cumulative_pnl']
        
        max_drawdown = df_sorted['drawdown'].max()
        max_drawdown_pct = (max_drawdown / df_sorted['running_max'].max() * 100) if df_sorted['running_max'].max() > 0 else 0
        
        # Consecutive wins/losses
        df_sorted['streak'] = (df_sorted['pnl'] > 0).astype(int)
        df_sorted['streak_id'] = (df_sorted['streak'] != df_sorted['streak'].shift()).cumsum()
        
        win_streaks = df_sorted[df_sorted['streak'] == 1].groupby('streak_id').size()
        loss_streaks = df_sorted[df_sorted['streak'] == 0].groupby('streak_id').size()
        
        max_consecutive_wins = win_streaks.max() if len(win_streaks) > 0 else 0
        max_consecutive_losses = loss_streaks.max() if len(loss_streaks) > 0 else 0
        
        # Calculate Sharpe Ratio (simplified)
        returns = df_sorted['pnl_percent']
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_percent': max_drawdown_pct,
            'max_consecutive_wins': int(max_consecutive_wins),
            'max_consecutive_losses': int(max_consecutive_losses),
            'sharpe_ratio': sharpe_ratio,
            'avg_rr_ratio': df['max_favorable_excursion'].mean() / abs(df['max_adverse_excursion'].mean()) if df['max_adverse_excursion'].mean() != 0 else 0,
        }
    
    def _calculate_time_metrics(self):
        """Calculate time-based metrics"""
        df = self.trades_df
        
        return {
            'avg_holding_period_minutes': df['holding_period_minutes'].mean(),
            'min_holding_period_minutes': df['holding_period_minutes'].min(),
            'max_holding_period_minutes': df['holding_period_minutes'].max(),
            'trades_per_day': len(df) / df['entry_date'].nunique() if df['entry_date'].nunique() > 0 else 0,
        }
    
    def _calculate_strategy_metrics(self):
        """Calculate per-strategy metrics"""
        df = self.trades_df
        
        strategy_stats = {}
        
        for strategy in df['strategy_name'].unique():
            strategy_trades = df[df['strategy_name'] == strategy]
            wins = strategy_trades[strategy_trades['pnl'] > 0]
            
            strategy_stats[strategy] = {
                'trades': len(strategy_trades),
                'win_rate': len(wins) / len(strategy_trades) * 100 if len(strategy_trades) > 0 else 0,
                'total_pnl': strategy_trades['pnl'].sum(),
                'avg_pnl': strategy_trades['pnl'].mean(),
            }
        
        # Find best and worst strategies
        if strategy_stats:
            best_strategy = max(strategy_stats.items(), key=lambda x: x[1]['win_rate'])
            worst_strategy = min(strategy_stats.items(), key=lambda x: x[1]['win_rate'])
            
            return {
                'strategy_breakdown': strategy_stats,
                'best_strategy_name': best_strategy[0],
                'best_strategy_win_rate': best_strategy[1]['win_rate'],
                'worst_strategy_name': worst_strategy[0],
                'worst_strategy_win_rate': worst_strategy[1]['win_rate'],
            }
        
        return {'strategy_breakdown': {}}
    
    def _calculate_confidence_metrics(self):
        """Calculate confidence-level metrics"""
        df = self.trades_df
        
        # Bin confidence levels
        bins = [0, 70, 75, 80, 85, 100]
        labels = ['65-70%', '70-75%', '75-80%', '80-85%', '85-100%']
        
        df['confidence_bin'] = pd.cut(df['confidence'], bins=bins, labels=labels)
        
        confidence_stats = {}
        
        for conf_bin in df['confidence_bin'].unique():
            if pd.isna(conf_bin):
                continue
            
            conf_trades = df[df['confidence_bin'] == conf_bin]
            wins = conf_trades[conf_trades['pnl'] > 0]
            
            confidence_stats[str(conf_bin)] = {
                'trades': len(conf_trades),
                'win_rate': len(wins) / len(conf_trades) * 100 if len(conf_trades) > 0 else 0,
                'avg_pnl': conf_trades['pnl'].mean(),
            }
        
        return {
            'confidence_breakdown': confidence_stats,
            'avg_confidence': df['confidence'].mean(),
        }
    
    def _calculate_signal_metrics(self):
        """Calculate signal generation metrics"""
        signals = self.signals_df
        
        return {
            'total_signals_generated': len(signals),
            'call_signals': len(signals[signals['signal_type'] == 'CALL']),
            'put_signals': len(signals[signals['signal_type'] == 'PUT']),
            'signals_per_day': len(signals) / signals['date'].nunique() if signals['date'].nunique() > 0 else 0,
            'avg_signal_confidence': signals['confidence'].mean() if 'confidence' in signals.columns else 0,
        }
    
    def get_equity_curve(self):
        """
        Generate equity curve data
        
        Returns:
            DataFrame with date, cumulative_pnl
        """
        if self.trades_df.empty:
            return pd.DataFrame()
        
        df = self.trades_df.copy()
        df_sorted = df.sort_values(['entry_date', 'entry_time'])
        df_sorted['cumulative_pnl'] = df_sorted['pnl'].cumsum()
        
        # Create datetime column
        df_sorted['datetime'] = pd.to_datetime(df_sorted['entry_date'] + ' ' + df_sorted['entry_time'])
        
        equity_curve = df_sorted[['datetime', 'cumulative_pnl']].copy()
        
        return equity_curve
    
    def get_drawdown_series(self):
        """
        Generate drawdown series
        
        Returns:
            DataFrame with date, drawdown
        """
        if self.trades_df.empty:
            return pd.DataFrame()
        
        df = self.trades_df.copy()
        df_sorted = df.sort_values(['entry_date', 'entry_time'])
        df_sorted['cumulative_pnl'] = df_sorted['pnl'].cumsum()
        df_sorted['running_max'] = df_sorted['cumulative_pnl'].cummax()
        df_sorted['drawdown'] = df_sorted['running_max'] - df_sorted['cumulative_pnl']
        
        df_sorted['datetime'] = pd.to_datetime(df_sorted['entry_date'] + ' ' + df_sorted['entry_time'])
        
        drawdown_series = df_sorted[['datetime', 'drawdown']].copy()
        
        return drawdown_series
    
    def get_win_rate_by_time(self):
        """
        Calculate win rate by time of day
        
        Returns:
            DataFrame with time_period, win_rate
        """
        if self.trades_df.empty:
            return pd.DataFrame()
        
        df = self.trades_df.copy()
        
        # Extract hour from entry_time
        df['hour'] = pd.to_datetime(df['entry_time'], format='%H:%M').dt.hour
        
        # Define time periods
        def get_time_period(hour):
            if 9 <= hour < 11:
                return 'Morning (9-11)'
            elif 11 <= hour < 13:
                return 'Midday (11-13)'
            elif 13 <= hour < 15:
                return 'Afternoon (13-15)'
            else:
                return 'Closing (15-15:30)'
        
        df['time_period'] = df['hour'].apply(get_time_period)
        
        # Calculate win rate per period
        time_stats = []
        
        for period in df['time_period'].unique():
            period_trades = df[df['time_period'] == period]
            wins = period_trades[period_trades['pnl'] > 0]
            
            time_stats.append({
                'time_period': period,
                'trades': len(period_trades),
                'win_rate': len(wins) / len(period_trades) * 100 if len(period_trades) > 0 else 0,
                'avg_pnl': period_trades['pnl'].mean(),
            })
        
        return pd.DataFrame(time_stats)
    
    def get_validation_status(self):
        """
        Check if performance meets minimum thresholds
        
        Returns:
            Dict with validation status
        """
        metrics = self.calculate_all_metrics()
        
        validation = {
            'is_viable': True,
            'is_optimal': True,
            'issues': [],
            'recommendations': []
        }
        
        # Check minimum thresholds
        if metrics['win_rate'] < self.config.MIN_WIN_RATE * 100:
            validation['is_viable'] = False
            validation['issues'].append(f"Win rate {metrics['win_rate']:.1f}% below minimum {self.config.MIN_WIN_RATE*100}%")
        
        if metrics['profit_factor'] < self.config.MIN_PROFIT_FACTOR:
            validation['is_viable'] = False
            validation['issues'].append(f"Profit factor {metrics['profit_factor']:.2f} below minimum {self.config.MIN_PROFIT_FACTOR}")
        
        if metrics['max_drawdown_percent'] > self.config.MAX_DRAWDOWN_PCT:
            validation['is_viable'] = False
            validation['issues'].append(f"Max drawdown {metrics['max_drawdown_percent']:.1f}% exceeds maximum {self.config.MAX_DRAWDOWN_PCT}%")
        
        # Check optimal thresholds
        if metrics['win_rate'] < self.config.TARGET_WIN_RATE * 100:
            validation['is_optimal'] = False
            validation['recommendations'].append(f"Target win rate is {self.config.TARGET_WIN_RATE*100}%, currently {metrics['win_rate']:.1f}%")
        
        if metrics['profit_factor'] < self.config.TARGET_PROFIT_FACTOR:
            validation['is_optimal'] = False
            validation['recommendations'].append(f"Target profit factor is {self.config.TARGET_PROFIT_FACTOR}, currently {metrics['profit_factor']:.2f}")
        
        # Overall verdict
        if validation['is_viable'] and validation['is_optimal']:
            validation['verdict'] = '✅ EXCELLENT - System meets all targets'
        elif validation['is_viable']:
            validation['verdict'] = '⚠️ VIABLE - System profitable but can be improved'
        else:
            validation['verdict'] = '❌ NOT VIABLE - System does not meet minimum requirements'
        
        return validation
