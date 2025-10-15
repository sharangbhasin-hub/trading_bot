"""
Chart Builder for Technical Analysis
Creates interactive Plotly charts for multiple timeframes
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Optional
from indicators import (
    calculate_ema,
    calculate_sma,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands
)

class ChartBuilder:
    """Build interactive technical analysis charts"""
    
    def __init__(self):
        self.color_scheme = {
            'bullish': '#00ff00',
            'bearish': '#ff0000',
            'neutral': '#888888',
            'background': '#0e1117',
            'grid': '#262730'
        }
    
    def create_multi_timeframe_chart(self, data_dict: Dict) -> Optional[go.Figure]:
        """
        Create combined chart with multiple timeframes
        
        Args:
            data_dict: Dict with keys like '5mindata', '15mindata', '60mindata', 'daydata'
        
        Returns:
            Plotly Figure object
        """
        try:
            # Find available timeframes
            available_timeframes = []
            timeframe_labels = {
                '5mindata': '5-Minute',
                '15mindata': '15-Minute',
                '60mindata': '1-Hour',
                'daydata': 'Daily'
            }
            
            for tf_key in ['5mindata', '15mindata', '60mindata', 'daydata']:
                if tf_key in data_dict and data_dict[tf_key] is not None:
                    df = data_dict[tf_key]
                    if not df.empty and len(df) >= 20:
                        available_timeframes.append((tf_key, timeframe_labels[tf_key]))
            
            if not available_timeframes:
                return None
            
            # Create subplots (2 rows per timeframe: price + RSI)
            num_rows = len(available_timeframes) * 2
            fig = make_subplots(
                rows=num_rows,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=[f"{label} - Price & Moving Averages" if i % 2 == 0 
                               else f"{label} - RSI" 
                               for tf_key, label in available_timeframes 
                               for i in range(2)],
                row_heights=[0.7, 0.3] * len(available_timeframes)
            )
            
            # Plot each timeframe
            for idx, (tf_key, label) in enumerate(available_timeframes):
                df = data_dict[tf_key]
                row_price = idx * 2 + 1
                row_rsi = idx * 2 + 2
                
                # Candlestick chart
                fig.add_trace(
                    go.Candlestick(
                        x=df.index,
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name=f'{label} Price',
                        increasing_line_color='#00ff00',
                        decreasing_line_color='#ff0000'
                    ),
                    row=row_price, col=1
                )
                
                # Moving Averages
                if len(df) >= 20:
                    ema20 = calculate_ema(df['close'], 20)
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=ema20,
                            mode='lines',
                            name=f'{label} EMA20',
                            line=dict(color='orange', width=1.5)
                        ),
                        row=row_price, col=1
                    )
                
                if len(df) >= 50:
                    ema50 = calculate_ema(df['close'], 50)
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=ema50,
                            mode='lines',
                            name=f'{label} EMA50',
                            line=dict(color='blue', width=1.5)
                        ),
                        row=row_price, col=1
                    )
                
                # Bollinger Bands
                if len(df) >= 20:
                    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['close'], 20, 2)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=bb_upper,
                            mode='lines',
                            name=f'{label} BB Upper',
                            line=dict(color='rgba(173, 204, 255, 0.5)', width=1, dash='dash'),
                            showlegend=False
                        ),
                        row=row_price, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=bb_lower,
                            mode='lines',
                            name=f'{label} BB Lower',
                            line=dict(color='rgba(173, 204, 255, 0.5)', width=1, dash='dash'),
                            fill='tonexty',
                            fillcolor='rgba(173, 204, 255, 0.1)',
                            showlegend=False
                        ),
                        row=row_price, col=1
                    )
                
                # RSI
                if len(df) >= 14:
                    rsi = calculate_rsi(df['close'], 14)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=rsi,
                            mode='lines',
                            name=f'{label} RSI',
                            line=dict(color='purple', width=2)
                        ),
                        row=row_rsi, col=1
                    )
                    
                    # RSI overbought/oversold lines
                    fig.add_hline(y=70, line_dash="dash", line_color="red", 
                                 opacity=0.5, row=row_rsi, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", 
                                 opacity=0.5, row=row_rsi, col=1)
                    fig.add_hline(y=50, line_dash="dot", line_color="gray", 
                                 opacity=0.3, row=row_rsi, col=1)
            
            # Update layout
            fig.update_layout(
                height=300 * len(available_timeframes) * 2,
                showlegend=True,
                xaxis_rangeslider_visible=False,
                template='plotly_dark',
                hovermode='x unified',
                title_text="Multi-Timeframe Technical Analysis",
                title_font_size=20
            )
            
            # Update y-axes labels
            for idx in range(len(available_timeframes)):
                row_rsi = idx * 2 + 2
                fig.update_yaxes(title_text="RSI", row=row_rsi, col=1, range=[0, 100])
            
            return fig
        
        except Exception as e:
            print(f"❌ Chart creation failed: {e}")
            return None
    
    def create_macd_chart(self, df: pd.DataFrame, title: str = "MACD Analysis") -> Optional[go.Figure]:
        """
        Create standalone MACD chart
        
        Args:
            df: OHLC DataFrame
            title: Chart title
        
        Returns:
            Plotly Figure object
        """
        if df is None or df.empty or len(df) < 26:
            return None
        
        try:
            macd_line, signal_line, histogram = calculate_macd(df['close'])
            
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=['Price', 'MACD'],
                row_heights=[0.7, 0.3]
            )
            
            # Price candlestick
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='Price',
                    increasing_line_color='#00ff00',
                    decreasing_line_color='#ff0000'
                ),
                row=1, col=1
            )
            
            # MACD Line
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=macd_line,
                    mode='lines',
                    name='MACD',
                    line=dict(color='blue', width=2)
                ),
                row=2, col=1
            )
            
            # Signal Line
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=signal_line,
                    mode='lines',
                    name='Signal',
                    line=dict(color='orange', width=2)
                ),
                row=2, col=1
            )
            
            # Histogram
            colors = ['green' if val >= 0 else 'red' for val in histogram]
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=histogram,
                    name='Histogram',
                    marker_color=colors,
                    opacity=0.5
                ),
                row=2, col=1
            )
            
            # Zero line
            fig.add_hline(y=0, line_dash="dash", line_color="white", 
                         opacity=0.5, row=2, col=1)
            
            fig.update_layout(
                height=600,
                showlegend=True,
                xaxis_rangeslider_visible=False,
                template='plotly_dark',
                hovermode='x unified',
                title_text=title,
                title_font_size=18
            )
            
            return fig
        
        except Exception as e:
            print(f"❌ MACD chart creation failed: {e}")
            return None
