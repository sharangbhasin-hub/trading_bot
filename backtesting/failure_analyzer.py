"""
Failure Analyzer
Analyzes losing trades to identify patterns
"""
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class FailureAnalyzer:
    """
    Analyzes failed trades to identify patterns
    """
    
    def __init__(self, trades_df, signals_df):
        """
        Initialize failure analyzer
        
        Args:
            trades_df: DataFrame of all trades
            signals_df: DataFrame of all signals
        """
        self.trades_df = trades_df
        self.signals_df = signals_df
        
    def categorize_failures(self):
        """
        Categorize losing trades by failure type
        
        Returns:
            Dict with failure categories
        """
        if self.trades_df.empty:
            return {}
        
        losses = self.trades_df[self.trades_df['pnl'] <= 0].copy()
        
        if losses.empty:
            return {'message': 'No losing trades'}
        
        categories = {
            'immediate_reversal': [],
            'stop_too_tight': [],
            'target_never_reached': [],
            'time_based_exit': [],
            'unknown': []
        }
        
        for _, trade in losses.iterrows():
            # Immediate reversal: stopped out within 15 minutes
            if trade['holding_period_minutes'] <= 15:
                categories['immediate_reversal'].append(trade['trade_id'])
            
            # Stop too tight: MAE < 70% of stop distance
            elif trade['exit_reason'] == 'STOP_LOSS':
                stop_distance = abs(trade['stop_loss'] - trade['entry_price'])
                if abs(trade['max_adverse_excursion']) < stop_distance * 0.7:
                    categories['stop_too_tight'].append(trade['trade_id'])
                else:
                    categories['unknown'].append(trade['trade_id'])
            
            # Target never reached: EOD exit
            elif trade['exit_reason'] == 'EOD':
                categories['target_never_reached'].append(trade['trade_id'])
            
            # Time-based exit
            elif 'TIMEOUT' in str(trade['exit_reason']):
                categories['time_based_exit'].append(trade['trade_id'])
            
            else:
                categories['unknown'].append(trade['trade_id'])
        
        # Calculate percentages
        total_losses = len(losses)
        category_stats = {}
        
        for cat, trade_ids in categories.items():
            category_stats[cat] = {
                'count': len(trade_ids),
                'percentage': len(trade_ids) / total_losses * 100 if total_losses > 0 else 0,
                'trade_ids': trade_ids
            }
        
        return category_stats
    
    def analyze_candlestick_patterns(self):
        """
        Analyze which candlestick patterns fail most
        
        Returns:
            DataFrame with pattern performance
        """
        if self.trades_df.empty:
            return pd.DataFrame()
        
        # Group by candlestick pattern
        pattern_stats = []
        
        for pattern in self.trades_df['candlestick_pattern'].unique():
            if pd.isna(pattern):
                continue
            
            pattern_trades = self.trades_df[self.trades_df['candlestick_pattern'] == pattern]
            wins = pattern_trades[pattern_trades['pnl'] > 0]
            
            pattern_stats.append({
                'pattern': pattern,
                'total_trades': len(pattern_trades),
                'winning_trades': len(wins),
                'win_rate': len(wins) / len(pattern_trades) * 100 if len(pattern_trades) > 0 else 0,
                'avg_pnl': pattern_trades['pnl'].mean()
            })
        
        df = pd.DataFrame(pattern_stats)
        df = df.sort_values('win_rate', ascending=False)
        
        return df
    
    def get_failure_recommendations(self):
        """
        Get specific recommendations based on failure analysis
        
        Returns:
            List of actionable recommendations
        """
        recommendations = []
        categories = self.categorize_failures()
        
        # ✅ CHECK IF WE HAVE ANY LOSING TRADES FIRST
        total_losses = sum(cat.get('count', 0) for cat in categories.values())
        
        if total_losses == 0:
            return ['No losing trades to analyze - All trades were winners or no trades executed']
        
        # Immediate reversals (use .get() with default)
        immediate_reversal = categories.get('immediate_reversal', {})
        if immediate_reversal.get('percentage', 0) > 25:
            recommendations.append("⚠️ High immediate reversals (>25%) - Consider:")
            recommendations.append("  • Adding confirmation candles")
            recommendations.append("  • Waiting for pullback entry")
            recommendations.append("  • Checking for divergence before entry")
        
        # Stop loss issues
        stop_hit_early = categories.get('stop_hit_early', {})
        if stop_hit_early.get('percentage', 0) > 20:
            recommendations.append("⚠️ Many stops hit early (>20%) - Consider:")
            recommendations.append("  • Widening stops to account for volatility")
            recommendations.append("  • Using ATR-based stops")
            recommendations.append("  • Better entry timing")
        
        # Target issues
        near_target = categories.get('near_target', {})
        if near_target.get('percentage', 0) > 15:
            recommendations.append("⚠️ Many near-miss targets (>15%) - Consider:")
            recommendations.append("  • Taking partial profits near target")
            recommendations.append("  • Using trailing stops")
            recommendations.append("  • Adjusting target levels")
        
        # Time-based failures
        timeout = categories.get('timeout', {})
        if timeout.get('percentage', 0) > 20:
            recommendations.append("⚠️ Many timeouts (>20%) - Consider:")
            recommendations.append("  • Shortening holding period")
            recommendations.append("  • Adding time-based stops")
            recommendations.append("  • Better trend confirmation")
        
        # Wrong direction
        wrong_direction = categories.get('wrong_direction', {})
        if wrong_direction.get('percentage', 0) > 30:
            recommendations.append("⚠️ High wrong direction rate (>30%) - Consider:")
            recommendations.append("  • Improving trend identification")
            recommendations.append("  • Using higher timeframe confirmation")
            recommendations.append("  • Avoiding choppy markets")
        
        if not recommendations:
            recommendations.append("✅ Failure distribution is acceptable - no major systematic issues detected")
        
        return recommendations
