"""
Multi-Timeframe Trend Analysis for Intraday Options
Combines multiple indicators with weighted scoring
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple
from indicators import (
    calculate_ema,
    calculate_sma,
    calculate_rsi,
    calculate_supertrend,
    calculate_macd
)

class TrendAnalyzer:
    """
    Analyzes market trend using multi-timeframe approach
    Combines 4 indicators with weighted scoring
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
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        try:
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
            
            print(f"\n{'='*60}")
            print(f"COMBINED ANALYSIS RESULT")
            print(f"{'='*60}")
            print(f"Daily Score (40%): {daily_score:+.2f}")
            print(f"Hourly Score (30%): {hourly_score:+.2f}")
            print(f"15-Min Score (30%): {min15_score:+.2f}")
            print(f"\n🎯 FINAL COMBINED SCORE: {combined_score:+.2f}")
            print(f"{'='*60}\n")
            
            # Determine final direction with confidence levels
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
            
            print(f"📊 FINAL DECISION:")
            print(f"   Direction: {direction}")
            print(f"   Confidence: {confidence}")
            print(f"   Action: {action}")
            print(f"{'='*60}\n")
            
            return {
                'direction': direction,
                'confidence': confidence,
                'action': action,
                'combined_score': combined_score,
                'spot_price': spot_price,
                'timestamp': datetime.now().isoformat(),
                'timeframe_analysis': {
                    'daily': {'score': daily_score, 'signals': daily_signals},
                    'hourly': {'score': hourly_score, 'signals': hourly_signals},
                    '15min': {'score': min15_score, 'signals': min15_signals}
                }
            }
        
        except Exception as e:
            print(f"❌ Error in trend analysis: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'direction': 'ERROR',
                'confidence': 'NONE',
                'action': 'WAIT',
                'combined_score': 0,
                'spot_price': spot_price,
                'error': str(e)
            }
    
    def _analyze_timeframe(self, df: pd.DataFrame, spot_price: float, 
                          timeframe: str) -> Tuple[float, Dict]:
        """
        Analyze single timeframe with 4 indicators
        Returns combined score (-1 to +1) and individual signals
        """
        signals = {}
        scores = []
        
        # 1. MOVING AVERAGE ANALYSIS (Primary trend)
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
            'ema20': round(ema20, 2),
            'sma50': round(sma50, 2),
            'signal': 'BULLISH' if ma_score > 0 else 'BEARISH' if ma_score < 0 else 'NEUTRAL'
        }
        scores.append(ma_score)
        
        # 2. SUPERTREND (Best for intraday trend)
        supertrend, direction = calculate_supertrend(df)
        latest_direction = direction.iloc[-1]
        
        st_score = 1.0 if latest_direction == 1 else -1.0
        signals['supertrend'] = {
            'score': st_score,
            'value': round(supertrend.iloc[-1], 2),
            'direction': 'BULLISH' if latest_direction == 1 else 'BEARISH'
        }
        scores.append(st_score)
        
        # 3. RSI (Momentum filter - prevent overbought/oversold trades)
        rsi = calculate_rsi(df['close'], 14).iloc[-1]
        
        if 55 < rsi < 70:
            rsi_score = 0.5  # Bullish but not overbought
        elif rsi >= 70:
            rsi_score = 0  # Overbought - neutral signal
        elif 30 < rsi < 45:
            rsi_score = -0.5  # Bearish but not oversold
        elif rsi <= 30:
            rsi_score = 0  # Oversold - neutral signal
        else:
            rsi_score = 0.1 if rsi > 50 else -0.1  # Mild bias
        
        signals['rsi'] = {
            'score': rsi_score,
            'value': round(rsi, 2),
            'signal': 'BULLISH' if rsi > 50 else 'BEARISH' if rsi < 50 else 'NEUTRAL',
            'status': 'OVERBOUGHT' if rsi > 70 else 'OVERSOLD' if rsi < 30 else 'NORMAL'
        }
        scores.append(rsi_score * 0.7)  # Reduced weight for RSI
        
        # 4. MACD (Trend confirmation)
        macd_line, signal_line, histogram = calculate_macd(df['close'])
        macd_current = macd_line.iloc[-1]
        signal_current = signal_line.iloc[-1]
        hist_current = histogram.iloc[-1]
        
        if macd_current > signal_current and hist_current > 0:
            macd_score = 1.0  # Strong bullish
        elif macd_current > signal_current:
            macd_score = 0.5  # Moderate bullish
        elif macd_current < signal_current and hist_current < 0:
            macd_score = -1.0  # Strong bearish
        elif macd_current < signal_current:
            macd_score = -0.5  # Moderate bearish
        else:
            macd_score = 0
        
        signals['macd'] = {
            'score': macd_score,
            'macd_line': round(macd_current, 2),
            'signal_line': round(signal_current, 2),
            'histogram': round(hist_current, 2),
            'signal': 'BULLISH' if macd_score > 0 else 'BEARISH' if macd_score < 0 else 'NEUTRAL'
        }
        scores.append(macd_score)
        
        # Calculate average score for this timeframe
        timeframe_score = sum(scores) / len(scores)
        
        print(f"📈 {timeframe} TIMEFRAME ANALYSIS:")
        print(f"   Combined Score: {timeframe_score:+.2f}")
        print(f"   MA Signal: {signals['moving_averages']['signal']} (Score: {ma_score:+.2f})")
        print(f"   Supertrend: {signals['supertrend']['direction']} (Score: {st_score:+.2f})")
        print(f"   RSI: {rsi:.2f} - {signals['rsi']['signal']} ({signals['rsi']['status']})")
        print(f"   MACD: {signals['macd']['signal']} (Score: {macd_score:+.2f})")
        print()
        
        return timeframe_score, signals
    
    def _fetch_data(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        """Fetch historical data from Kite"""
        try:
            # For indices, we need to use the index symbol directly
            # Try to get instrument token
            token = self.kite.get_instrument_token(symbol, 'NSE')
            
            if not token:
                # Fallback: Try with NIFTY 50, NIFTY BANK variations
                print(f"⚠️ Could not find token for {symbol}, trying alternatives...")
                return None
            
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            
            data = self.kite.get_historical_data(
                instrument_token=token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            
            if data is None or data.empty:
                raise ValueError(f"No data returned for {symbol}")
            
            print(f"✅ Fetched {len(data)} candles for {interval} timeframe")
            return data
            
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
            raise
