"""
Chart Visualizer for Paper Trading
===================================

Real-time chart visualization with price action, indicators, and entry/exit markers.
Integrates with Plotly for interactive charting.

Features:
- OHLCV candlestick charts
- Technical indicators (EMA, SMA, RSI, Volume)
- Entry/Exit/SL/TP markers
- Real-time updates
- SMC pattern overlays (Order Blocks, FVG, etc.)

Author: Trading System
Last Updated: October 29, 2025
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ChartVisualizer:
    """
    Production-grade chart visualizer for paper trading.
    
    Features:
    - Interactive candlestick charts
    - Technical indicator overlays
    - Trade markers with annotations
    - Real-time updates
    - Custom SMC pattern visualization
    """
    
    def __init__(self, height: int = 600):
        """
        Initialize chart visualizer.
        
        Args:
            height: Chart height in pixels (default: 600)
        """
        self.height = height
        self.colors = {
            'up': '#26A69A',      # Green for bullish
            'down': '#EF5350',    # Red for bearish
            'buy': '#00C853',     # Buy signal
            'sell': '#FF1744',    # Sell signal
            'sl': '#FFC107',      # Stop Loss
            'tp': '#4CAF50',      # Take Profit
            'grid': '#E0E0E0',    # Grid lines
            'bg': '#FAFAFA'       # Background
        }
        
        logger.info("ChartVisualizer initialized")
    
    def create_candlestick_chart(
        self,
        df: pd.DataFrame,
        title: str = "Price Chart",
        show_volume: bool = True,
        show_indicators: bool = True
    ) -> go.Figure:
        """
        Create interactive candlestick chart.
        
        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
            title: Chart title
            show_volume: Whether to show volume subplot
            show_indicators: Whether to show technical indicators
        
        Returns:
            Plotly Figure object
        
        Example:
            >>> df = data_manager.get_candles_as_dataframe('BTC/USDT', 100)
            >>> fig = visualizer.create_candlestick_chart(df, "BTC/USDT")
        """
        # Validate DataFrame
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # Create subplots
        if show_volume:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3],
                subplot_titles=(title, 'Volume')
            )
        else:
            fig = make_subplots(rows=1, cols=1, subplot_titles=(title,))
        
        # Add candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                increasing_line_color=self.colors['up'],
                decreasing_line_color=self.colors['down']
            ),
            row=1, col=1
        )
        
        # Add indicators if requested
        if show_indicators:
            self._add_indicators(fig, df, row=1, col=1)
        
        # Add volume if requested
        if show_volume and 'volume' in df.columns:
            colors = [self.colors['up'] if df['close'].iloc[i] >= df['open'].iloc[i] 
                     else self.colors['down'] for i in range(len(df))]
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['volume'],
                    name='Volume',
                    marker_color=colors,
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=self.height,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            template='plotly_white',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes
        fig.update_xaxes(
            showgrid=True,
            gridcolor=self.colors['grid'],
            type='category' if len(df) < 50 else 'date'
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor=self.colors['grid']
        )
        
        return fig
    
    def _add_indicators(self, fig: go.Figure, df: pd.DataFrame, row: int = 1, col: int = 1):
        """
        Add technical indicators to chart.
        
        Args:
            fig: Plotly figure
            df: DataFrame with price data
            row: Subplot row
            col: Subplot column
        """
        # EMA 9
        if len(df) >= 9:
            ema9 = df['close'].ewm(span=9, adjust=False).mean()
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=ema9,
                    name='EMA 9',
                    line=dict(color='#2196F3', width=1),
                    opacity=0.7
                ),
                row=row, col=col
            )
        
        # EMA 21
        if len(df) >= 21:
            ema21 = df['close'].ewm(span=21, adjust=False).mean()
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=ema21,
                    name='EMA 21',
                    line=dict(color='#FF9800', width=1),
                    opacity=0.7
                ),
                row=row, col=col
            )
        
        # EMA 50
        if len(df) >= 50:
            ema50 = df['close'].ewm(span=50, adjust=False).mean()
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=ema50,
                    name='EMA 50',
                    line=dict(color='#9C27B0', width=1.5),
                    opacity=0.7
                ),
                row=row, col=col
            )
    
    def add_trade_markers(
        self,
        fig: go.Figure,
        trades: List[Dict],
        df: pd.DataFrame
    ) -> go.Figure:
        """
        Add trade entry/exit markers to chart.
        
        Args:
            fig: Plotly figure
            trades: List of trade dictionaries with entry/exit info
            df: DataFrame with price data
        
        Returns:
            Updated figure with markers
        
        Example:
            >>> trades = order_manager.get_closed_trades()
            >>> fig = visualizer.add_trade_markers(fig, trades, df)
        """
        for trade in trades:
            # Parse timestamps
            entry_time = pd.to_datetime(trade['timestamp'])
            exit_time = pd.to_datetime(trade['exit_timestamp']) if trade.get('exit_timestamp') else None
            
            # Find closest index in DataFrame
            entry_idx = self._find_closest_index(df.index, entry_time)
            
            # Add entry marker
            color = self.colors['buy'] if trade['direction'] == 'BUY' else self.colors['sell']
            symbol = 'triangle-up' if trade['direction'] == 'BUY' else 'triangle-down'
            
            fig.add_trace(
                go.Scatter(
                    x=[df.index[entry_idx]],
                    y=[trade['entry_price']],
                    mode='markers+text',
                    name=f"Entry #{trade['id']}",
                    marker=dict(
                        size=12,
                        color=color,
                        symbol=symbol,
                        line=dict(width=2, color='white')
                    ),
                    text=[f"#{trade['id']}"],
                    textposition='top center',
                    showlegend=False
                )
            )
            
            # Add SL/TP lines
            if entry_idx is not None:
                # Stop Loss line
                fig.add_hline(
                    y=trade['stop_loss'],
                    line=dict(color=self.colors['sl'], dash='dash', width=1),
                    annotation_text=f"SL: {trade['stop_loss']:.2f}",
                    annotation_position="right",
                    row=1, col=1
                )
                
                # Take Profit line
                fig.add_hline(
                    y=trade['take_profit'],
                    line=dict(color=self.colors['tp'], dash='dash', width=1),
                    annotation_text=f"TP: {trade['take_profit']:.2f}",
                    annotation_position="right",
                    row=1, col=1
                )
            
            # Add exit marker if available
            if exit_time and trade.get('exit_price'):
                exit_idx = self._find_closest_index(df.index, exit_time)
                if exit_idx is not None:
                    exit_color = self.colors['tp'] if trade['pnl_usd'] > 0 else self.colors['sl']
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[df.index[exit_idx]],
                            y=[trade['exit_price']],
                            mode='markers',
                            name=f"Exit #{trade['id']}",
                            marker=dict(
                                size=10,
                                color=exit_color,
                                symbol='x',
                                line=dict(width=2, color='white')
                            ),
                            showlegend=False
                        )
                    )
        
        return fig
    
    def add_smc_patterns(
        self,
        fig: go.Figure,
        patterns: Dict[str, List[Dict]],
        df: pd.DataFrame
    ) -> go.Figure:
        """
        Add SMC pattern overlays (Order Blocks, FVG, etc.).
        
        Args:
            fig: Plotly figure
            patterns: Dictionary of patterns {'order_blocks': [...], 'fvg': [...]}
            df: DataFrame with price data
        
        Returns:
            Updated figure with pattern overlays
        
        Example:
            >>> patterns = {
            ...     'order_blocks': [{'start': 100, 'end': 110, 'high': 67500, 'low': 67400, 'type': 'bullish'}],
            ...     'fvg': [{'index': 120, 'top': 67600, 'bottom': 67550, 'type': 'bullish'}]
            ... }
            >>> fig = visualizer.add_smc_patterns(fig, patterns, df)
        """
        # Add Order Blocks
        if 'order_blocks' in patterns:
            for ob in patterns['order_blocks']:
                color = 'rgba(0, 200, 83, 0.2)' if ob['type'] == 'bullish' else 'rgba(255, 23, 68, 0.2)'
                
                fig.add_shape(
                    type="rect",
                    x0=df.index[ob['start']],
                    x1=df.index[ob['end']],
                    y0=ob['low'],
                    y1=ob['high'],
                    fillcolor=color,
                    line=dict(color=color.replace('0.2', '0.5'), width=1),
                    layer='below'
                )
        
        # Add Fair Value Gaps (FVG)
        if 'fvg' in patterns:
            for fvg in patterns['fvg']:
                color = 'rgba(33, 150, 243, 0.2)' if fvg['type'] == 'bullish' else 'rgba(255, 152, 0, 0.2)'
                
                fig.add_shape(
                    type="rect",
                    x0=df.index[max(0, fvg['index'] - 1)],
                    x1=df.index[min(len(df) - 1, fvg['index'] + 5)],
                    y0=fvg['bottom'],
                    y1=fvg['top'],
                    fillcolor=color,
                    line=dict(color=color.replace('0.2', '0.5'), width=1, dash='dot'),
                    layer='below'
                )
        
        # Add Liquidity Levels
        if 'liquidity' in patterns:
            for liq in patterns['liquidity']:
                fig.add_hline(
                    y=liq['price'],
                    line=dict(color='purple', dash='dot', width=1),
                    annotation_text=f"Liquidity: {liq['price']:.2f}",
                    annotation_position="left"
                )
        
        return fig
    
    def _find_closest_index(self, index: pd.Index, timestamp: datetime) -> Optional[int]:
        """
        Find closest index to given timestamp.
        
        Args:
            index: DataFrame index (DatetimeIndex)
            timestamp: Target timestamp
        
        Returns:
            Closest index position or None
        """
        try:
            # Convert to datetime if not already
            if not isinstance(index, pd.DatetimeIndex):
                index = pd.to_datetime(index)
            
            # Find closest
            idx = index.get_indexer([timestamp], method='nearest')[0]
            
            if idx == -1 or idx >= len(index):
                return None
            
            return idx
            
        except Exception as e:
            logger.error(f"Error finding closest index: {e}")
            return None
    
    def create_performance_chart(
        self,
        equity_curve: pd.Series,
        title: str = "Equity Curve"
    ) -> go.Figure:
        """
        Create equity curve chart.
        
        Args:
            equity_curve: Series with equity values over time
            title: Chart title
        
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        # Add equity line
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
                mode='lines',
                name='Equity',
                line=dict(color='#2196F3', width=2),
                fill='tozeroy',
                fillcolor='rgba(33, 150, 243, 0.1)'
            )
        )
        
        # Add initial balance line
        if len(equity_curve) > 0:
            initial = equity_curve.iloc[0]
            fig.add_hline(
                y=initial,
                line=dict(color='gray', dash='dash', width=1),
                annotation_text=f"Initial: ${initial:,.2f}",
                annotation_position="right"
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Equity (USD)",
            height=400,
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    import numpy as np
    
    print("=" * 70)
    print("CHART VISUALIZER TEST")
    print("=" * 70)
    
    # Generate sample data
    dates = pd.date_range(start='2025-10-01', periods=100, freq='1H')
    np.random.seed(42)
    
    # Create OHLCV data
    close = 67000 + np.cumsum(np.random.randn(100) * 50)
    open_price = close + np.random.randn(100) * 20
    high = np.maximum(open_price, close) + np.abs(np.random.randn(100) * 30)
    low = np.minimum(open_price, close) - np.abs(np.random.randn(100) * 30)
    volume = np.random.randint(1000, 10000, 100)
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    # Initialize visualizer
    visualizer = ChartVisualizer(height=600)
    
    # Test 1: Create candlestick chart
    print("\n1️⃣ Creating candlestick chart...")
    fig = visualizer.create_candlestick_chart(
        df,
        title="BTC/USDT - Test Chart",
        show_volume=True,
        show_indicators=True
    )
    print("✅ Chart created successfully")
    
    # Test 2: Add trade markers
    print("\n2️⃣ Adding trade markers...")
    sample_trades = [
        {
            'id': 1,
            'timestamp': dates[20],
            'direction': 'BUY',
            'entry_price': df['close'].iloc[20],
            'stop_loss': df['close'].iloc[20] - 100,
            'take_profit': df['close'].iloc[20] + 200,
            'exit_timestamp': dates[30],
            'exit_price': df['close'].iloc[30],
            'pnl_usd': 150.0
        },
        {
            'id': 2,
            'timestamp': dates[50],
            'direction': 'SELL',
            'entry_price': df['close'].iloc[50],
            'stop_loss': df['close'].iloc[50] + 100,
            'take_profit': df['close'].iloc[50] - 200,
            'exit_timestamp': dates[60],
            'exit_price': df['close'].iloc[60],
            'pnl_usd': -80.0
        }
    ]
    
    fig = visualizer.add_trade_markers(fig, sample_trades, df)
    print("✅ Trade markers added")
    
    # Test 3: Add SMC patterns
    print("\n3️⃣ Adding SMC patterns...")
    patterns = {
        'order_blocks': [
            {'start': 40, 'end': 45, 'high': df['high'].iloc[40:45].max(), 
             'low': df['low'].iloc[40:45].min(), 'type': 'bullish'}
        ],
        'fvg': [
            {'index': 70, 'top': df['high'].iloc[70], 
             'bottom': df['low'].iloc[70] - 50, 'type': 'bullish'}
        ]
    }
    
    fig = visualizer.add_smc_patterns(fig, patterns, df)
    print("✅ SMC patterns added")
    
    # Save to HTML
    output_file = "paper_trading/test_chart.html"
    fig.write_html(output_file)
    print(f"\n✅ Chart saved to {output_file}")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED")
    print("=" * 70)
