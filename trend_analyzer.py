"""
Multi-Timeframe Trend Analysis for Intraday Options
Combines multiple indicators with weighted scoring
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple
from indicators import *

class TrendAnalyzer:
    """
    Analyzes market trend using multi-timeframe approach
    Combines indicators for final decision
    """
    
    def __init__(self, kite_handler):
        self.kite = kite_handler
        
        # Weightings for intraday (as per your requirement)
        self.weights = {
            'daily': 0.40,    # 40%
            'hourly': 0.30,   # 30%
            '15min': 0.30     # 30%
        }
    
    def analyze_trend(self, index_symbol: str, spot_price: float) -> Dict:
        """
        Main function: Combines all timeframes and indicators
        Returns comprehensive trend analysis
        """
        
        print(f"\n{'='*60}")
        print(f"ANALYZING TREND FOR {index_symbol}")
        print(f"Current Spot Price: ₹{spot_price:,.2f}")
        print(f"{'='*60}\n")
        
        # Fetch data for all timeframes
        daily_data = self._fetch_data(index_symbol, 'day', 60)
        hourly_data = self._fetch_data(index_symbol, '60minute', 10)
        min15_data = self._fetch_data(index_symbol, '15minute', 7)
        
        # Analyze each timeframe
        daily_score, daily_signals = self._analyze_timeframe(
            daily_data, spot_price, 'DAILY'
        )
        
        hourly_score, hourly_signals = self._analyze_timeframe(
            hourly_data, spot_price, 'HOURLY'
        )
        
        min15_score, min15_signals = self._analyze_timeframe(
            min15_data, spot_price, '15MIN'
        )
        
        # Calculate weighted combined score
        combined_score = (
            daily_score * self.weights['daily'] +
            hourly_score * self.weights['hourly'] +
            min15_score * self.weights['15min']
        )
        
        # Determine final direction
        if combined_score >= 0.6:
            direction = "BULLISH"
            confidence = "HIGH"
            action = "CALL"
        elif combined_score >= 0.3:
            direction = "BULLISH"
            confidence = "MODERATE"
            action = "CALL"
        elif combined_score <= -0.6:
            direction = "BEARISH"
            confidence = "HIGH"
            action = "PUT"
        elif combined_score <= -0.3:
            direction = "BEARISH"
            confidence = "MODERATE"
            action = "PUT"
        else:
            direction = "NEUTRAL"
            confidence = "LOW"
            action = "WAIT"
        
        return {
            'direction': direction,
            'confidence': confidence,
            'action': action,
            'combined_score': combined_score,
            'spot_price': spot_price,
            'timeframe_analysis': {
                'daily': {'score': daily_score, 'signals': daily_signals},
                'hourly': {'score': hourly_score, 'signals': hourly_signals},
                '15min': {'score': min15_score, 'signals': min15_signals}
            }
        }
    
    def _analyze_timeframe(self, df: pd.DataFrame, spot_price: float, 
                          timeframe: str) -> Tuple[float, Dict]:
        """
        Analyze single timeframe with multiple indicators
        Returns combined score (-1 to +1) and individual signals
        """
        signals = {}
        scores = []
        
        # 1. Moving Average Analysis
        ema20 = calculate_ema(df['close'], 20).iloc[-1]
        sma50 = calculate_sma(df['close'], 50).iloc[-1]
        
        if spot_price > ema20 > sma50:
            ma_score = 1.0  # Strong bullish
        elif spot_price > ema20:
            ma_score = 0.5  # Moderate bullish
        elif spot_price < ema20 < sma50:
            ma_score = -1.0  # Strong bearish
        elif spot_price < ema20:
            ma_score = -0.5  # Moderate bearish
        else:
            ma_score = 0
        
        signals['moving_averages'] = {
            'score': ma_score,
            'ema20': ema20,
            'sma50': sma50,
            'signal': 'BULLISH' if ma_score > 0 else 'BEARISH' if ma_score < 0 else 'NEUTRAL'
        }
        scores.append(ma_score)
        
        # 2. Supertrend (Best for intraday)
        supertrend, direction = calculate_supertrend(df)
        latest_direction = direction.iloc[-1]
        
        st_score = 1.0 if latest_direction == 1 else -1.0
        signals['supertrend'] = {
            'score': st_score,
            'value': supertrend.iloc[-1],
            'direction': 'BULLISH' if latest_direction == 1 else 'BEARISH'
        }
        scores.append(st_score)
        
        # 3. RSI (Momentum filter)
        rsi = calculate_rsi(df['close'], 14).iloc[-1]
        
        if rsi > 55 and rsi < 70:
            rsi_score = 0.5  # Bullish but not overbought
        elif rsi > 70:
            rsi_score = 0  # Overbought - neutral
        elif rsi < 45 and rsi > 30:
            rsi_score = -0.5  # Bearish but not oversold
        elif rsi < 30:
            rsi_score = 0  # Oversold - neutral
        else:
            rsi_score = 0
        
        signals['rsi'] = {
            'score': rsi_score,
            'value': rsi,
            'signal': 'BULLISH' if rsi > 50 else 'BEARISH' if rsi < 50 else 'NEUTRAL'
        }
        scores.append(rsi_score * 0.5)  # Half weight for RSI
        
        # 4. MACD (Trend confirmation)
        macd_line, signal_line, histogram = calculate_macd(df['close'])
        macd_current = macd_line.iloc[-1]
        signal_current = signal_line.iloc[-1]
        
        if macd_current > signal_current and histogram.iloc[-1] > 0:
            macd_score = 1.0
        elif macd_current > signal_current:
            macd_score = 0.5
        elif macd_current < signal_current and histogram.iloc[-1] < 0:
            macd_score = -1.0
        elif macd_current < signal_current:
            macd_score = -0.5
        else:
            macd_score = 0
        
        signals['macd'] = {
            'score': macd_score,
            'macd_line': macd_current,
            'signal_line': signal_current,
            'signal': 'BULLISH' if macd_score > 0 else 'BEARISH' if macd_score < 0 else 'NEUTRAL'
        }
        scores.append(macd_score)
        
        # Calculate average score for this timeframe
        timeframe_score = sum(scores) / len(scores)
        
        print(f"{timeframe} Analysis:")
        print(f"  Combined Score: {timeframe_score:+.2f}")
        print(f"  MA Signal: {signals['moving_averages']['signal']}")
        print(f"  Supertrend: {signals['supertrend']['direction']}")
        print(f"  RSI: {rsi:.2f} ({signals['rsi']['signal']})")
        print(f"  MACD: {signals['macd']['signal']}")
        print()
        
        return timeframe_score, signals
    
    def _fetch_data(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        """Fetch historical data from Kite"""
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        token = self.kite.get_instrument_token(symbol, 'NSE')
        if not token:
            raise ValueError(f"Could not find instrument token for {symbol}")
        
        data = self.kite.get_historical_data(token, from_date, to_date, interval)
        return data
