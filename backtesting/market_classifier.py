"""
Market Classifier
Classifies market conditions (trending, ranging, volatile, etc.)
"""
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from backtesting.config import BacktestConfig

logger = logging.getLogger(__name__)


class MarketClassifier:
    """
    Classifies market conditions for analysis
    """
    
    def __init__(self):
        """Initialize market classifier"""
        self.config = BacktestConfig()
        self.classifications = {}
        
    def classify_day(self, daily_data, date_str):
        """
        Classify a single trading day
        
        Args:
            daily_data: Dict with all timeframes for the day
            date_str: Date string 'YYYY-MM-DD'
        
        Returns:
            Dict with classification
        """
        # Get 15-min and daily data
        df_15min = daily_data.get('15min', pd.DataFrame())
        df_daily = daily_data.get('daily', pd.DataFrame())

        # Check for None or non-DataFrame types
        if df_15min is None or df_daily is None:
            return {'date': date_str, 'classification': 'UNKNOWN', 'reason': 'Insufficient data'}
        
        if not isinstance(df_15min, pd.DataFrame) or not isinstance(df_daily, pd.DataFrame):
            return {'date': date_str, 'classification': 'UNKNOWN', 'reason': 'Insufficient data'}
        
        if df_15min.empty or df_daily.empty:
            return {'date': date_str, 'classification': 'UNKNOWN', 'reason': 'Insufficient data'}
        
        classification = {
            'date': date_str,
            'classification': None,
            'subtype': None,
            'adx': None,
            'atr': None,
            'atr_ratio': None,
            'gap_percent': None,
            'reason': []
        }
        
        # Calculate ADX
        adx = self._calculate_adx(df_15min)
        classification['adx'] = adx
        
        # Calculate ATR
        atr_current = self._calculate_atr(df_15min)
        atr_avg = self._calculate_atr_average(df_15min, periods=20)
        atr_ratio = atr_current / atr_avg if atr_avg > 0 else 1.0
        
        classification['atr'] = atr_current
        classification['atr_ratio'] = atr_ratio
        
        # Check for gap
        if len(df_daily) > 1:
            prev_close = df_daily['close'].iloc[-2]
            today_open = df_daily['open'].iloc[-1]
            gap_percent = abs((today_open - prev_close) / prev_close * 100)
            classification['gap_percent'] = gap_percent
        else:
            gap_percent = 0
            classification['gap_percent'] = 0
        
        # Classify based on conditions
        
        # 1. BREAKOUT DAY (Gap up/down)
        if gap_percent > self.config.GAP_THRESHOLD_PERCENT:
            classification['classification'] = 'BREAKOUT'
            if df_15min['close'].iloc[-1] > df_15min['open'].iloc[0]:
                classification['subtype'] = 'GAP_UP'
            else:
                classification['subtype'] = 'GAP_DOWN'
            classification['reason'].append(f'Gap {gap_percent:.2f}%')
        
        # 2. VOLATILE DAY (High ATR)
        elif atr_ratio > self.config.ATR_VOLATILE_MULTIPLIER:
            classification['classification'] = 'VOLATILE'
            classification['subtype'] = 'HIGH_ATR'
            classification['reason'].append(f'ATR {atr_ratio:.2f}x average')
        
        # 3. TRENDING DAY (High ADX)
        elif adx > self.config.ADX_TRENDING_THRESHOLD:
            classification['classification'] = 'TRENDING'
            
            # Determine if bullish or bearish
            close_start = df_15min['close'].iloc[0]
            close_end = df_15min['close'].iloc[-1]
            
            if close_end > close_start:
                classification['subtype'] = 'BULLISH'
            else:
                classification['subtype'] = 'BEARISH'
            
            classification['reason'].append(f'ADX {adx:.1f}')
        
        # 4. RANGING DAY (Low ADX)
        elif adx < self.config.ADX_RANGING_THRESHOLD:
            classification['classification'] = 'RANGING'
            classification['subtype'] = 'SIDEWAYS'
            classification['reason'].append(f'ADX {adx:.1f} (low)')
        
        # 5. NORMAL DAY
        else:
            classification['classification'] = 'NORMAL'
            classification['subtype'] = 'MIXED'
            classification['reason'].append(f'ADX {adx:.1f}, ATR ratio {atr_ratio:.2f}')
        
        # Store classification
        self.classifications[date_str] = classification
        
        return classification
    
    def _calculate_adx(self, df, period=14):
        """Calculate Average Directional Index"""
        if len(df) < period + 1:
            return 20  # Default neutral value
        
        df = df.copy()
        
        # Calculate True Range
        df['tr'] = pd.DataFrame({
            'hl': df['high'] - df['low'],
            'hc': abs(df['high'] - df['close'].shift()),
            'lc': abs(df['low'] - df['close'].shift())
        }).max(axis=1)
        
        # Calculate Directional Movement
        df['up_move'] = df['high'] - df['high'].shift()
        df['down_move'] = df['low'].shift() - df['low']
        
        df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
        df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
        
        # Smooth with exponential moving average
        atr = df['tr'].ewm(span=period, adjust=False).mean()
        plus_di = 100 * (df['plus_dm'].ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (df['minus_dm'].ewm(span=period, adjust=False).mean() / atr)
        
        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return adx.iloc[-1] if not adx.empty else 20
    
    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        if len(df) < period + 1:
            return df['high'].iloc[-1] - df['low'].iloc[-1] if not df.empty else 0
        
        df = df.copy()
        
        df['tr'] = pd.DataFrame({
            'hl': df['high'] - df['low'],
            'hc': abs(df['high'] - df['close'].shift()),
            'lc': abs(df['low'] - df['close'].shift())
        }).max(axis=1)
        
        atr = df['tr'].ewm(span=period, adjust=False).mean()
        
        return atr.iloc[-1]
    
    def _calculate_atr_average(self, df, periods=20):
        """Calculate average ATR over multiple periods"""
        if len(df) < periods + 14:
            return self._calculate_atr(df)
        
        atr_values = []
        for i in range(periods):
            subset = df.iloc[-(periods-i):]
            atr_values.append(self._calculate_atr(subset))
        
        return np.mean(atr_values)
    
    def get_all_classifications(self):
        """Get all day classifications"""
        return self.classifications
    
    def get_classification_summary(self):
        """Get summary of classifications"""
        if not self.classifications:
            return {}
        
        df = pd.DataFrame(self.classifications.values())
        
        summary = {
            'total_days': len(df),
            'trending_days': len(df[df['classification'] == 'TRENDING']),
            'ranging_days': len(df[df['classification'] == 'RANGING']),
            'volatile_days': len(df[df['classification'] == 'VOLATILE']),
            'breakout_days': len(df[df['classification'] == 'BREAKOUT']),
            'normal_days': len(df[df['classification'] == 'NORMAL']),
            'trending_pct': len(df[df['classification'] == 'TRENDING']) / len(df) * 100 if len(df) > 0 else 0,
            'ranging_pct': len(df[df['classification'] == 'RANGING']) / len(df) * 100 if len(df) > 0 else 0,
            'volatile_pct': len(df[df['classification'] == 'VOLATILE']) / len(df) * 100 if len(df) > 0 else 0,
        }
        
        return summary
    
    def analyze_performance_by_condition(self, trades_df):
        """
        Analyze trading performance by market condition
        
        Args:
            trades_df: DataFrame of all trades
        
        Returns:
            Dict with performance breakdown by condition
        """
        if trades_df.empty or not self.classifications:
            return {}
        
        # Merge trades with classifications
        trades_df = trades_df.copy()
        trades_df['market_condition'] = trades_df['entry_date'].map(
            lambda x: self.classifications.get(x, {}).get('classification', 'UNKNOWN')
        )
        
        condition_stats = {}
        
        for condition in trades_df['market_condition'].unique():
            condition_trades = trades_df[trades_df['market_condition'] == condition]
            wins = condition_trades[condition_trades['pnl'] > 0]
            
            condition_stats[condition] = {
                'trades': len(condition_trades),
                'win_rate': len(wins) / len(condition_trades) * 100 if len(condition_trades) > 0 else 0,
                'total_pnl': condition_trades['pnl'].sum(),
                'avg_pnl': condition_trades['pnl'].mean(),
            }
        
        return condition_stats
