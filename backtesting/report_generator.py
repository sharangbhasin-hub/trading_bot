"""
Report Generator
Creates comprehensive HTML and CSV reports from backtest results
"""
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import logging

from backtesting.config import BacktestConfig

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates comprehensive reports from backtest results
    """
    
    def __init__(self, output_dir):
        """
        Initialize report generator
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = BacktestConfig()
        
    def generate_html_report(self, metrics, validation, charts, recommendations):
        """
        Generate HTML dashboard report
        
        Args:
            metrics: Dict of performance metrics
            validation: Validation status dict
            charts: Dict of chart paths
            recommendations: List of recommendations
        
        Returns:
            Path to HTML file
        """
        html_content = self._build_html(metrics, validation, charts, recommendations)
        
        output_path = self.output_dir / 'backtest_report.html'
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Generated HTML report: {output_path}")
        return output_path
    
    def _build_html(self, metrics, validation, charts, recommendations):
        """Build HTML content"""
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Report - {datetime.now().strftime('%Y-%m-%d')}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: #f5f5f5;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{ 
            max-width: 1400px; 
            margin: 0 auto; 
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        h1 {{ 
            color: #2C3E50; 
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
            margin-bottom: 30px;
        }}
        h2 {{ 
            color: #34495E; 
            margin-top: 30px;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ECF0F1;
        }}
        .verdict {{
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            font-size: 18px;
            font-weight: bold;
            text-align: center;
        }}
        .verdict.excellent {{ background: #d4edda; color: #155724; border: 2px solid #c3e6cb; }}
        .verdict.viable {{ background: #fff3cd; color: #856404; border: 2px solid #ffeaa7; }}
        .verdict.not-viable {{ background: #f8d7da; color: #721c24; border: 2px solid #f5c6cb; }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .metric-card.positive {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
        .metric-card.warning {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }}
        .metric-card.negative {{ background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); }}
        
        .metric-label {{ 
            font-size: 14px; 
            opacity: 0.9;
            margin-bottom: 8px;
        }}
        .metric-value {{ 
            font-size: 32px; 
            font-weight: bold;
        }}
        .metric-unit {{ font-size: 18px; opacity: 0.8; }}
        
        .chart-container {{
            margin: 30px 0;
            text-align: center;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
        }}
        th {{
            background: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #ECF0F1;
        }}
        tr:hover {{ background: #F8F9FA; }}
        
        .recommendation {{
            background: #E8F4F8;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }}
        .recommendation.high {{ border-left-color: #e74c3c; background: #FADBD8; }}
        .recommendation.medium {{ border-left-color: #f39c12; background: #FEF5E7; }}
        .recommendation.low {{ border-left-color: #27ae60; background: #D5F4E6; }}
        
        .issue {{
            background: #f8d7da;
            border-left: 4px solid #dc3545;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
            color: #721c24;
        }}
        
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #ECF0F1;
            text-align: center;
            color: #7F8C8D;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Backtest Performance Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Test Period:</strong> {metrics.get('test_period', 'N/A')}</p>
        
        <!-- Verdict -->
        <div class="verdict {self._get_verdict_class(validation.get('verdict', ''))}">
            {validation.get('verdict', 'Unknown')}
        </div>
        
        <!-- Key Metrics -->
        <h2>📈 Key Performance Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card positive">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value">{metrics.get('total_trades', 0)}</div>
            </div>
            <div class="metric-card {self._get_metric_class(metrics.get('win_rate', 0), 60, 50)}">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{metrics.get('win_rate', 0):.1f}<span class="metric-unit">%</span></div>
            </div>
            <div class="metric-card {self._get_metric_class(metrics.get('profit_factor', 0), 2.0, 1.5)}">
                <div class="metric-label">Profit Factor</div>
                <div class="metric-value">{metrics.get('profit_factor', 0):.2f}</div>
            </div>
            <div class="metric-card {'positive' if metrics.get('total_pnl', 0) > 0 else 'negative'}">
                <div class="metric-label">Total P&L</div>
                <div class="metric-value">{metrics.get('total_pnl', 0):,.0f}<span class="metric-unit">pts</span></div>
            </div>
            <div class="metric-card positive">
                <div class="metric-label">Average Win</div>
                <div class="metric-value">{metrics.get('avg_win', 0):.1f}<span class="metric-unit">pts</span></div>
            </div>
            <div class="metric-card warning">
                <div class="metric-label">Average Loss</div>
                <div class="metric-value">{metrics.get('avg_loss', 0):.1f}<span class="metric-unit">pts</span></div>
            </div>
            <div class="metric-card warning">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value">{metrics.get('max_drawdown', 0):,.0f}<span class="metric-unit">pts</span></div>
            </div>
            <div class="metric-card positive">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value">{metrics.get('sharpe_ratio', 0):.2f}</div>
            </div>
        </div>
        
        <!-- Charts -->
        <h2>📉 Performance Charts</h2>
        {self._generate_charts_html(charts)}
        
        <!-- Strategy Breakdown -->
        <h2>🎯 Strategy Performance</h2>
        {self._generate_strategy_table(metrics.get('strategy_breakdown', {}))}
        
        <!-- Validation Issues -->
        {self._generate_issues_html(validation.get('issues', []))}
        
        <!-- Recommendations -->
        <h2>💡 Recommendations</h2>
        {self._generate_recommendations_html(recommendations)}
        
        <div class="footer">
            <p>Generated by Backtesting System v1.0</p>
            <p>© {datetime.now().year} Trading Bot - All Rights Reserved</p>
        </div>
    </div>
</body>
</html>
"""
        return html
    
    def _get_verdict_class(self, verdict):
        """Get CSS class for verdict"""
        if '✅' in verdict:
            return 'excellent'
        elif '⚠️' in verdict:
            return 'viable'
        else:
            return 'not-viable'
    
    def _get_metric_class(self, value, good_threshold, ok_threshold):
        """Get CSS class based on metric value"""
        if value >= good_threshold:
            return 'positive'
        elif value >= ok_threshold:
            return 'warning'
        else:
            return 'negative'
    
    def _generate_charts_html(self, charts):
        """Generate HTML for charts"""
        if not charts:
            return '<p>No charts available</p>'
        
        html = ''
        for chart_name, chart_path in charts.items():
            if chart_path and Path(chart_path).exists():
                html += f'''
                <div class="chart-container">
                    <h3>{chart_name.replace('_', ' ').title()}</h3>
                    <img src="{chart_path.name}" alt="{chart_name}">
                </div>
                '''
        return html
    
    def _generate_strategy_table(self, strategy_breakdown):
        """Generate HTML table for strategy performance"""
        if not strategy_breakdown:
            return '<p>No strategy data available</p>'
        
        html = '<table><thead><tr><th>Strategy</th><th>Trades</th><th>Win Rate</th><th>Total P&L</th><th>Avg P&L</th></tr></thead><tbody>'
        
        for strategy, stats in strategy_breakdown.items():
            html += f'''
            <tr>
                <td><strong>{strategy}</strong></td>
                <td>{stats['trades']}</td>
                <td>{stats['win_rate']:.1f}%</td>
                <td>{stats['total_pnl']:,.0f} pts</td>
                <td>{stats['avg_pnl']:.1f} pts</td>
            </tr>
            '''
        
        html += '</tbody></table>'
        return html
    
    def _generate_issues_html(self, issues):
        """Generate HTML for validation issues"""
        if not issues:
            return ''
        
        html = '<h2>⚠️ Validation Issues</h2>'
        for issue in issues:
            html += f'<div class="issue">❌ {issue}</div>'
        return html
    
    def _generate_recommendations_html(self, recommendations):
        """Generate HTML for recommendations"""
        if not recommendations:
            return '<p>No recommendations at this time.</p>'
        
        html = ''
        for rec in recommendations:
            priority = rec.get('priority', 'MEDIUM').lower()
            category = rec.get('category', 'General')
            recommendation = rec.get('recommendation', '')
            
            html += f'''
            <div class="recommendation {priority}">
                <strong>{category}:</strong> {recommendation}
                <span style="float: right; font-size: 12px; opacity: 0.8;">Priority: {priority.upper()}</span>
            </div>
            '''
        return html
    
    def export_trades_csv(self, trades_df):
        """
        Export trades to CSV
        
        Args:
            trades_df: DataFrame of all trades
        
        Returns:
            Path to CSV file
        """
        output_path = self.output_dir / 'trades.csv'
        trades_df.to_csv(output_path, index=False)
        logger.info(f"Exported trades to {output_path}")
        return output_path
    
    def export_signals_csv(self, signals_df):
        """
        Export signals to CSV
        
        Args:
            signals_df: DataFrame of all signals
        
        Returns:
            Path to CSV file
        """
        output_path = self.output_dir / 'signals.csv'
        signals_df.to_csv(output_path, index=False)
        logger.info(f"Exported signals to {output_path}")
        return output_path
    
    def export_metrics_json(self, metrics):
        """
        Export metrics to JSON
        
        Args:
            metrics: Dict of all metrics
        
        Returns:
            Path to JSON file
        """
        output_path = self.output_dir / 'metrics.json'
        
        # Convert numpy types to native Python types
        def convert_types(obj):
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        metrics_clean = convert_types(metrics)
        
        with open(output_path, 'w') as f:
            json.dump(metrics_clean, f, indent=2)
        
        logger.info(f"Exported metrics to {output_path}")
        return output_path
