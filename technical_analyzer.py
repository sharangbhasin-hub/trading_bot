"""
Technical Analysis Engine
Combines indicators and generates comprehensive analysis
"""
import pandas as pd
from typing import Dict, Optional
from indicators import (
    calculate_ema,
    calculate_sma,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_atr
)

class TechnicalAnalyzer:
    """Perform comprehensive technical analysis"""
    
    def __init__(self):
        pass
    
    def analyze_all(self, df: pd.DataFrame, timeframe: str = 'day') -> Dict:
        """
        Run complete technical analysis on dataframe
        
        Args:
            df: OHLC DataFrame
            timeframe: Timeframe name (for display)
        
        Returns:
            Dict with all indicator values and interpretations
        """
        if df is None or df.empty or len(df) < 50:
            return {
                'error': 'Insufficient data',
                'timeframe': timeframe
            }
        
        try:
            close = df['close']
            
            # Moving Averages
            ema_20 = calculate_ema(close, 20).iloc[-1]
            ema_50 = calculate_ema(close, 50).iloc[-1]
            sma_200 = calculate_sma(close, 200).iloc[-1] if len(df) >= 200 else None
            
            # RSI
            rsi = calculate_rsi(close, 14).iloc[-1]
            
            # MACD
            macd_line, signal_line, histogram = calculate_macd(close)
            macd_values = {
                'macd': round(macd_line.iloc[-1], 2),
                'signal': round(signal_line.iloc[-1], 2),
                'histogram': round(histogram.iloc[-1], 2)
            }
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, 20, 2)
            
            # ATR
            atr = calculate_atr(df, 14).iloc[-1]
            
            # Current price
            current_price = close.iloc[-1]
            
            # Generate interpretations
            ma_signal = self._interpret_moving_averages(current_price, ema_20, ema_50, sma_200)
            rsi_signal = self._interpret_rsi(rsi)
            macd_signal = self._interpret_macd(macd_values['histogram'])
            bb_signal = self._interpret_bollinger_bands(current_price, bb_upper.iloc[-1], bb_lower.iloc[-1])
            
            return {
                'timeframe': timeframe,
                'current_price': round(current_price, 2),
                'rsi': round(rsi, 2),
                'rsi_signal': rsi_signal,
                'ema_20': round(ema_20, 2),
                'ema_50': round(ema_50, 2),
                'sma_200': round(sma_200, 2) if sma_200 else None,
                'ma_signal': ma_signal,
                'macd': macd_values,
                'macd_signal': macd_signal,
                'bb_upper': round(bb_upper.iloc[-1], 2),
                'bb_middle': round(bb_middle.iloc[-1], 2),
                'bb_lower': round(bb_lower.iloc[-1], 2),
                'bb_signal': bb_signal,
                'atr': round(atr, 2),
                'error': None
            }
        
        except Exception as e:
            return {
                'error': f'Analysis failed: {str(e)}',
                'timeframe': timeframe
            }
    
    def _interpret_moving_averages(self, price: float, ema20: float, ema50: float, sma200: Optional[float]) -> str:
        """Interpret MA signals"""
        if price > ema20 and ema20 > ema50:
            return '🟢 Strong Bullish (Price > EMA20 > EMA50)'
        elif price < ema20 and ema20 < ema50:
            return '🔴 Strong Bearish (Price < EMA20 < EMA50)'
        elif price > ema20:
            return '🟡 Bullish (Price > EMA20)'
        elif price < ema20:
            return '🟡 Bearish (Price < EMA20)'
        else:
            return '⚪ Neutral'
    
    def _interpret_rsi(self, rsi: float) -> str:
        """Interpret RSI value"""
        if rsi > 70:
            return '🔴 Overbought (Consider taking profits)'
        elif rsi < 30:
            return '🟢 Oversold (Potential buying opportunity)'
        elif rsi > 60:
            return '🟡 Bullish Momentum'
        elif rsi < 40:
            return '🟡 Bearish Momentum'
        else:
            return '⚪ Neutral Zone'
    
    def _interpret_macd(self, histogram: float) -> str:
        """Interpret MACD histogram"""
        if histogram > 0:
            return '🟢 Bullish (MACD above signal)'
        elif histogram < 0:
            return '🔴 Bearish (MACD below signal)'
        else:
            return '⚪ Neutral'
    
    def _interpret_bollinger_bands(self, price: float, upper: float, lower: float) -> str:
        """Interpret Bollinger Bands position"""
        if price >= upper:
            return '🔴 At Upper Band (Overbought zone)'
        elif price <= lower:
            return '🟢 At Lower Band (Oversold zone)'
        else:
            middle = (upper + lower) / 2
            if price > middle:
                return '🟡 Above Middle (Bullish)'
            else:
                return '🟡 Below Middle (Bearish)'
