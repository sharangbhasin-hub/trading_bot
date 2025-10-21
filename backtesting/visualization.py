"""
Visualization Module
Generates all charts and graphs for backtest results
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import logging

from backtesting.config import BacktestConfig

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100


class Visualizer:
    """
    Creates visualizations for backtest results
    """
    
    def __init__(self, output_dir):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save charts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = BacktestConfig()
        
    def plot_equity_curve(self, equity_curve_df, title="Equity Curve"):
        """
        Plot cumulative P&L over time
        
        Args:
            equity_curve_df: DataFrame with datetime, cumulative_pnl
            title: Chart title
        
        Returns:
            Path to saved image
        """
        fig, ax = plt.subplots(figsize=(self.config.CHART_WIDTH, self.config.CHART_HEIGHT))
        
        if equity_curve_df.empty:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            output_path = self.output_dir / 'equity_curve.png'
            plt.savefig(output_path, dpi=self.config.CHART_DPI, bbox_inches='tight')
            plt.close()
            return output_path
        
        # Plot equity curve
        ax.plot(equity_curve_df['datetime'], equity_curve_df['cumulative_pnl'], 
                linewidth=2, color='#2E86AB', label='Cumulative P&L')
        
        # Add zero line
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Breakeven')
        
        # Formatting
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative P&L (Points)', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        
        # Add stats box
        final_pnl = equity_curve_df['cumulative_pnl'].iloc[-1]
        stats_text = f'Final P&L: {final_pnl:,.0f} points'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Save
        output_path = self.output_dir / 'equity_curve.png'
        plt.savefig(output_path, dpi=self.config.CHART_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved equity curve to {output_path}")
        return output_path
    
    def plot_drawdown_chart(self, drawdown_df, title="Drawdown Analysis"):
        """
        Plot drawdown over time
        
        Args:
            drawdown_df: DataFrame with datetime, drawdown
            title: Chart title
        
        Returns:
            Path to saved image
        """
        fig, ax = plt.subplots(figsize=(self.config.CHART_WIDTH, self.config.CHART_HEIGHT))
        
        if drawdown_df.empty:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            output_path = self.output_dir / 'drawdown.png'
            plt.savefig(output_path, dpi=self.config.CHART_DPI, bbox_inches='tight')
            plt.close()
            return output_path
        
        # Plot drawdown as area
        ax.fill_between(drawdown_df['datetime'], drawdown_df['drawdown'], 
                        0, color='#A23B72', alpha=0.5, label='Drawdown')
        ax.plot(drawdown_df['datetime'], drawdown_df['drawdown'], 
                color='#A23B72', linewidth=1.5)
        
        # Formatting
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (Points)', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # Drawdown goes down
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        
        # Add max drawdown
        max_dd = drawdown_df['drawdown'].max()
        stats_text = f'Max Drawdown: {max_dd:,.0f} points'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Save
        output_path = self.output_dir / 'drawdown.png'
        plt.savefig(output_path, dpi=self.config.CHART_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved drawdown chart to {output_path}")
        return output_path
    
    def plot_strategy_comparison(self, strategy_stats, title="Strategy Performance Comparison"):
        """
        Bar chart comparing strategies
        
        Args:
            strategy_stats: Dict of {strategy_name: {trades, win_rate, total_pnl}}
            title: Chart title
        
        Returns:
            Path to saved image
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.config.CHART_WIDTH, self.config.CHART_HEIGHT))
        
        if not strategy_stats:
            ax1.text(0.5, 0.5, 'No data available', ha='center', va='center')
            ax2.text(0.5, 0.5, 'No data available', ha='center', va='center')
            output_path = self.output_dir / 'strategy_comparison.png'
            plt.savefig(output_path, dpi=self.config.CHART_DPI, bbox_inches='tight')
            plt.close()
            return output_path
        
        strategies = list(strategy_stats.keys())
        win_rates = [strategy_stats[s]['win_rate'] for s in strategies]
        pnls = [strategy_stats[s]['total_pnl'] for s in strategies]
        
        # Win rate comparison
        colors_wr = ['#06A77D' if wr >= 60 else '#F77F00' if wr >= 50 else '#D62828' for wr in win_rates]
        ax1.barh(strategies, win_rates, color=colors_wr)
        ax1.set_xlabel('Win Rate (%)', fontsize=11)
        ax1.set_title('Win Rate by Strategy', fontsize=13, fontweight='bold')
        ax1.axvline(x=60, color='green', linestyle='--', alpha=0.5, label='Target: 60%')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Total P&L comparison
        colors_pnl = ['#06A77D' if p > 0 else '#D62828' for p in pnls]
        ax2.barh(strategies, pnls, color=colors_pnl)
        ax2.set_xlabel('Total P&L (Points)', fontsize=11)
        ax2.set_title('Total P&L by Strategy', fontsize=13, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.7)
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        
        # Save
        output_path = self.output_dir / 'strategy_comparison.png'
        plt.savefig(output_path, dpi=self.config.CHART_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved strategy comparison to {output_path}")
        return output_path
    
    def plot_win_rate_by_time(self, time_stats_df, title="Win Rate by Time of Day"):
        """
        Plot win rate by time period
        
        Args:
            time_stats_df: DataFrame with time_period, win_rate
            title: Chart title
        
        Returns:
            Path to saved image
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if time_stats_df.empty:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            output_path = self.output_dir / 'win_rate_by_time.png'
            plt.savefig(output_path, dpi=self.config.CHART_DPI, bbox_inches='tight')
            plt.close()
            return output_path
        
        # Sort by time period
        time_order = ['Morning (9-11)', 'Midday (11-13)', 'Afternoon (13-15)', 'Closing (15-15:30)']
        time_stats_df['time_period'] = pd.Categorical(time_stats_df['time_period'], 
                                                        categories=time_order, ordered=True)
        time_stats_df = time_stats_df.sort_values('time_period')
        
        # Color based on win rate
        colors = ['#06A77D' if wr >= 60 else '#F77F00' if wr >= 50 else '#D62828' 
                  for wr in time_stats_df['win_rate']]
        
        ax.bar(time_stats_df['time_period'], time_stats_df['win_rate'], color=colors)
        ax.axhline(y=60, color='green', linestyle='--', alpha=0.5, label='Target: 60%')
        ax.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='Breakeven: 50%')
        
        # Formatting
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Time Period', fontsize=12)
        ax.set_ylabel('Win Rate (%)', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (period, row) in enumerate(time_stats_df.iterrows()):
            ax.text(i, row['win_rate'] + 2, f"{row['win_rate']:.1f}%", 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Save
        output_path = self.output_dir / 'win_rate_by_time.png'
        plt.savefig(output_path, dpi=self.config.CHART_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved win rate by time to {output_path}")
        return output_path
    
    def plot_pnl_distribution(self, trades_df, title="P&L Distribution"):
        """
        Histogram of trade P&L
        
        Args:
            trades_df: DataFrame of all trades
            title: Chart title
        
        Returns:
            Path to saved image
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if trades_df.empty:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            output_path = self.output_dir / 'pnl_distribution.png'
            plt.savefig(output_path, dpi=self.config.CHART_DPI, bbox_inches='tight')
            plt.close()
            return output_path
        
        # Separate wins and losses
        wins = trades_df[trades_df['pnl'] > 0]['pnl']
        losses = trades_df[trades_df['pnl'] <= 0]['pnl']
        
        # Plot histograms
        ax.hist(wins, bins=20, color='#06A77D', alpha=0.7, label=f'Wins (n={len(wins)})')
        ax.hist(losses, bins=20, color='#D62828', alpha=0.7, label=f'Losses (n={len(losses)})')
        
        # Add vertical line at zero
        ax.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Breakeven')
        
        # Formatting
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('P&L (Points)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add stats
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        stats_text = f'Avg Win: {avg_win:.1f}\nAvg Loss: {avg_loss:.1f}'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Save
        output_path = self.output_dir / 'pnl_distribution.png'
        plt.savefig(output_path, dpi=self.config.CHART_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved P&L distribution to {output_path}")
        return output_path
    
    def plot_confidence_vs_winrate(self, trades_df, title="Confidence vs Win Rate"):
        """
        Scatter plot of confidence vs win rate
        
        Args:
            trades_df: DataFrame of all trades
            title: Chart title
        
        Returns:
            Path to saved image
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if trades_df.empty:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            output_path = self.output_dir / 'confidence_vs_winrate.png'
            plt.savefig(output_path, dpi=self.config.CHART_DPI, bbox_inches='tight')
            plt.close()
            return output_path
        
        # Bin confidence levels
        bins = [65, 70, 75, 80, 85, 100]
        trades_df['confidence_bin'] = pd.cut(trades_df['confidence'], bins=bins)
        
        # Calculate win rate per bin
        confidence_stats = []
        for bin_val in trades_df['confidence_bin'].unique():
            if pd.isna(bin_val):
                continue
            bin_trades = trades_df[trades_df['confidence_bin'] == bin_val]
            wins = bin_trades[bin_trades['pnl'] > 0]
            
            confidence_stats.append({
                'confidence_mid': bin_val.mid,
                'win_rate': len(wins) / len(bin_trades) * 100 if len(bin_trades) > 0 else 0,
                'trades': len(bin_trades)
            })
        
        if not confidence_stats:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center')
            output_path = self.output_dir / 'confidence_vs_winrate.png'
            plt.savefig(output_path, dpi=self.config.CHART_DPI, bbox_inches='tight')
            plt.close()
            return output_path
        
        conf_df = pd.DataFrame(confidence_stats)
        
        # Plot
        scatter = ax.scatter(conf_df['confidence_mid'], conf_df['win_rate'], 
                            s=conf_df['trades']*10, alpha=0.6, c=conf_df['win_rate'],
                            cmap='RdYlGn', vmin=40, vmax=80)
        
        # Add trend line
        z = np.polyfit(conf_df['confidence_mid'], conf_df['win_rate'], 1)
        p = np.poly1d(z)
        ax.plot(conf_df['confidence_mid'], p(conf_df['confidence_mid']), 
                "r--", alpha=0.8, linewidth=2, label='Trend')
        
        # Formatting
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Confidence Level (%)', fontsize=12)
        ax.set_ylabel('Win Rate (%)', fontsize=12)
        ax.axhline(y=60, color='green', linestyle='--', alpha=0.5, label='Target: 60%')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Win Rate (%)', rotation=270, labelpad=20)
        
        # Save
        output_path = self.output_dir / 'confidence_vs_winrate.png'
        plt.savefig(output_path, dpi=self.config.CHART_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved confidence vs win rate to {output_path}")
        return output_path
    
    def plot_market_condition_breakdown(self, condition_stats, title="Performance by Market Condition"):
        """
        Plot performance by market condition
        
        Args:
            condition_stats: Dict of {condition: {trades, win_rate, total_pnl}}
            title: Chart title
        
        Returns:
            Path to saved image
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.config.CHART_WIDTH, self.config.CHART_HEIGHT))
        
        if not condition_stats:
            ax1.text(0.5, 0.5, 'No data available', ha='center', va='center')
            ax2.text(0.5, 0.5, 'No data available', ha='center', va='center')
            output_path = self.output_dir / 'market_conditions.png'
            plt.savefig(output_path, dpi=self.config.CHART_DPI, bbox_inches='tight')
            plt.close()
            return output_path
        
        conditions = list(condition_stats.keys())
        win_rates = [condition_stats[c]['win_rate'] for c in conditions]
        trades = [condition_stats[c]['trades'] for c in conditions]
        
        # Win rate by condition
        colors = ['#06A77D' if wr >= 60 else '#F77F00' if wr >= 50 else '#D62828' for wr in win_rates]
        ax1.bar(conditions, win_rates, color=colors)
        ax1.axhline(y=60, color='green', linestyle='--', alpha=0.5, label='Target: 60%')
        ax1.set_title('Win Rate by Market Condition', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Win Rate (%)', fontsize=11)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Trade count by condition
        ax2.bar(conditions, trades, color='#2E86AB', alpha=0.7)
        ax2.set_title('Trade Count by Market Condition', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Number of Trades', fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        
        # Save
        output_path = self.output_dir / 'market_conditions.png'
        plt.savefig(output_path, dpi=self.config.CHART_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved market condition breakdown to {output_path}")
        return output_path
    
    def generate_all_charts(self, analyzer, market_classifier, trades_df):
        """
        Generate all charts at once
        
        Args:
            analyzer: PerformanceAnalyzer instance
            market_classifier: MarketClassifier instance
            trades_df: DataFrame of trades
        
        Returns:
            Dict of {chart_name: path}
        """
        logger.info("Generating all visualizations...")
        
        charts = {}
        
        try:
            # Equity curve
            equity_curve = analyzer.get_equity_curve()
            charts['equity_curve'] = self.plot_equity_curve(equity_curve)
        except Exception as e:
            logger.error(f"Error generating equity curve: {e}")
        
        try:
            # Drawdown
            drawdown = analyzer.get_drawdown_series()
            charts['drawdown'] = self.plot_drawdown_chart(drawdown)
        except Exception as e:
            logger.error(f"Error generating drawdown chart: {e}")
        
        try:
            # Strategy comparison
            metrics = analyzer.calculate_all_metrics()
            if 'strategy_breakdown' in metrics:
                charts['strategy_comparison'] = self.plot_strategy_comparison(
                    metrics['strategy_breakdown']
                )
        except Exception as e:
            logger.error(f"Error generating strategy comparison: {e}")
        
        try:
            # Win rate by time
            time_stats = analyzer.get_win_rate_by_time()
            charts['win_rate_by_time'] = self.plot_win_rate_by_time(time_stats)
        except Exception as e:
            logger.error(f"Error generating win rate by time: {e}")
        
        try:
            # P&L distribution
            charts['pnl_distribution'] = self.plot_pnl_distribution(trades_df)
        except Exception as e:
            logger.error(f"Error generating P&L distribution: {e}")
        
        try:
            # Confidence vs win rate
            charts['confidence_vs_winrate'] = self.plot_confidence_vs_winrate(trades_df)
        except Exception as e:
            logger.error(f"Error generating confidence vs win rate: {e}")
        
        try:
            # Market conditions
            condition_stats = market_classifier.analyze_performance_by_condition(trades_df)
            charts['market_conditions'] = self.plot_market_condition_breakdown(condition_stats)
        except Exception as e:
            logger.error(f"Error generating market conditions: {e}")
        
        logger.info(f"Generated {len(charts)} charts")
        
        return charts
