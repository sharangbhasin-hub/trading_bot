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
import time
from config import get_market_status

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
        
        # ✅ NEW: Rate limiting to prevent API bans
        self.last_api_call = 0
        self.min_call_interval = 1.0  # 1 second between API calls
        self.api_call_count = 0
        self.max_calls_per_minute = 10  # Kite limit
    
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

        # ✅ NEW: Check if market is open
        market_status = get_market_status()
        
        if market_status['status'] != 'OPEN':
            print(f"⚠️ Market is {market_status['status']}: {market_status['reason']}")
            return {
                'direction': 'NEUTRAL',
                'confidence': 'NONE',
                'action': 'WAIT',
                'combined_score': 0,
                'spot_price': spot_price,
                'timestamp': datetime.now().isoformat(),
                'error': f"Market is {market_status['status']}: {market_status['reason']}",
                'market_status': market_status
            }
        
        print(f"✅ Market is OPEN - Proceeding with analysis\n")

        try:
            # Fetch data for all timeframes
            daily_data = self._fetch_data(index_symbol, 'day', 90)
            hourly_data = self._fetch_data(index_symbol, '60minute', 15)
            min15_data = self._fetch_data(index_symbol, '15minute', 10)
            
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

            # ✅ NEW: Boost confidence when indicators agree strongly
            all_timeframe_signals = []
            
            # Add daily signals
            all_timeframe_signals.extend([
                daily_signals['moving_averages']['signal'],
                daily_signals['supertrend']['direction'],
                daily_signals['rsi']['signal'],
                daily_signals['macd']['signal']
            ])
            
            # Add hourly signals
            all_timeframe_signals.extend([
                hourly_signals['moving_averages']['signal'],
                hourly_signals['supertrend']['direction'],
                hourly_signals['rsi']['signal'],
                hourly_signals['macd']['signal']
            ])
            
            # Add 15min signals
            all_timeframe_signals.extend([
                min15_signals['moving_averages']['signal'],
                min15_signals['supertrend']['direction'],
                min15_signals['rsi']['signal'],
                min15_signals['macd']['signal']
            ])
            
            bullish_count = all_timeframe_signals.count('BULLISH')
            bearish_count = all_timeframe_signals.count('BEARISH')
            total_signals = len(all_timeframe_signals)
            
            # If 70%+ indicators agree, boost the score
            agreement_ratio = max(bullish_count, bearish_count) / total_signals
            if agreement_ratio >= 0.7:
                boost_factor = 1.2  # 20% boost
                combined_score = combined_score * boost_factor
                print(f"✅ Strong agreement detected: {bullish_count} bullish, {bearish_count} bearish out of {total_signals}")
                print(f"   Boosting combined score by {(boost_factor - 1) * 100:.0f}%")
            
            print(f"\n{'='*60}")
            print(f"COMBINED ANALYSIS RESULT")
            print(f"{'='*60}")
            print(f"Daily Score (40%): {daily_score:+.2f}")
            print(f"Hourly Score (30%): {hourly_score:+.2f}")
            print(f"15-Min Score (30%): {min15_score:+.2f}")
            print(f"\n🎯 FINAL COMBINED SCORE: {combined_score:+.2f}")
            print(f"{'='*60}\n")
            
            # Determine final direction with confidence levels
            # ✅ IMPROVED: More realistic thresholds for real market conditions
            if combined_score >= 0.4:  # Changed from 0.6
                direction = "BULLISH"
                confidence = "HIGH"
                action = "CALL"
            elif combined_score >= 0.2:  # Changed from 0.3
                direction = "BULLISH"
                confidence = "MODERATE"
                action = "CALL"
            elif combined_score <= -0.4:  # Changed from -0.6
                direction = "BEARISH"
                confidence = "HIGH"
                action = "PUT"
            elif combined_score <= -0.2:  # Changed from -0.3
                direction = "BEARISH"
                confidence = "MODERATE"
                action = "PUT"
            elif combined_score > 0.1:  # NEW: Weak bullish zone
                direction = "BULLISH"
                confidence = "LOW"
                action = "CALL"
            elif combined_score < -0.1:  # NEW: Weak bearish zone
                direction = "BEARISH"
                confidence = "LOW"
                action = "PUT"
            else:  # Only between -0.1 and 0.1 = truly neutral
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
        Analyze single timeframe with optimized indicator settings
        Each timeframe uses different settings for accuracy
        """
        signals = {}
        scores = []
        
        # Timeframe-specific settings (OPTIMIZED FOR INTRADAY)
        if timeframe == 'DAILY':
            # Daily: Longer periods, filter noise
            ema_period = 20
            sma_period = 50
            supertrend_period = 10
            supertrend_multiplier = 3.0
            rsi_period = 14
            macd_fast, macd_slow, macd_signal = 12, 26, 9
            
        elif timeframe == 'HOURLY':
            # Hourly: Medium periods, catch swings
            ema_period = 13
            sma_period = 34  # Fibonacci
            supertrend_period = 7
            supertrend_multiplier = 2.5
            rsi_period = 9
            macd_fast, macd_slow, macd_signal = 8, 17, 9
            
        else:  # 15MIN
            # 15-min: Short periods, quick entries
            ema_period = 9
            sma_period = 21  # Fibonacci
            supertrend_period = 5
            supertrend_multiplier = 2.0
            rsi_period = 7
            macd_fast, macd_slow, macd_signal = 5, 13, 5
        
        print(f"   Settings: EMA{ema_period}, SMA{sma_period}, ST({supertrend_period},{supertrend_multiplier}), RSI{rsi_period}")
        
        # 1. MOVING AVERAGE ANALYSIS (with timeframe-specific periods)
        ema = calculate_ema(df['close'], ema_period).iloc[-1]
        sma = calculate_sma(df['close'], sma_period).iloc[-1]
        
        if spot_price > ema > sma:
            ma_score = 1.0  # Strong bullish
        elif spot_price > ema:
            ma_score = 0.5  # Moderate bullish
        elif spot_price < ema < sma:
            ma_score = -1.0  # Strong bearish
        elif spot_price < ema:
            ma_score = -0.5  # Moderate bearish
        else:
            ma_score = 0
        
        signals['moving_averages'] = {
            'score': ma_score,
            'ema': round(ema, 2),
            'sma': round(sma, 2),
            'ema_period': ema_period,
            'sma_period': sma_period,
            'signal': 'BULLISH' if ma_score > 0 else 'BEARISH' if ma_score < 0 else 'NEUTRAL'
        }
        scores.append(ma_score)
        
        # 2. SUPERTREND (with timeframe-specific settings)
        # Modify supertrend to accept period and multiplier
        supertrend, direction = calculate_supertrend(
            df, 
            period=supertrend_period, 
            multiplier=supertrend_multiplier
        )
        latest_direction = direction.iloc[-1]
        
        st_score = 1.0 if latest_direction == 1 else -1.0
        signals['supertrend'] = {
            'score': st_score,
            'value': round(supertrend.iloc[-1], 2),
            'period': supertrend_period,
            'multiplier': supertrend_multiplier,
            'direction': 'BULLISH' if latest_direction == 1 else 'BEARISH'
        }
        scores.append(st_score)
        
        # 3. RSI (with timeframe-specific period)
        rsi = calculate_rsi(df['close'], rsi_period).iloc[-1]
        
        # Timeframe-adjusted RSI thresholds
        if timeframe == 'DAILY':
            overbought, oversold = 70, 30
        elif timeframe == 'HOURLY':
            overbought, oversold = 65, 35
        else:  # 15MIN
            overbought, oversold = 60, 40  # Tighter for quick moves
        
        if 55 < rsi < overbought:
            rsi_score = 0.5
        elif rsi >= overbought:
            rsi_score = 0
        elif oversold < rsi < 45:
            rsi_score = -0.5
        elif rsi <= oversold:
            rsi_score = 0
        else:
            rsi_score = 0.1 if rsi > 50 else -0.1
        
        signals['rsi'] = {
            'score': rsi_score,
            'value': round(rsi, 2),
            'period': rsi_period,
            'overbought': overbought,
            'oversold': oversold,
            'signal': 'BULLISH' if rsi > 50 else 'BEARISH' if rsi < 50 else 'NEUTRAL',
            'status': f'OVERBOUGHT (>{overbought})' if rsi > overbought else f'OVERSOLD (<{oversold})' if rsi < oversold else 'NORMAL'
        }
        scores.append(rsi_score)
        
        # 4. MACD (with timeframe-specific settings)
        macd_line, signal_line, histogram = calculate_macd(
            df['close'], 
            fast=macd_fast, 
            slow=macd_slow, 
            signal=macd_signal
        )
        macd_current = macd_line.iloc[-1]
        signal_current = signal_line.iloc[-1]
        hist_current = histogram.iloc[-1]
        
        if macd_current > signal_current and hist_current > 0:
            macd_score = 1.0
        elif macd_current > signal_current:
            macd_score = 0.5
        elif macd_current < signal_current and hist_current < 0:
            macd_score = -1.0
        elif macd_current < signal_current:
            macd_score = -0.5
        else:
            macd_score = 0
        
        signals['macd'] = {
            'score': macd_score,
            'macd_line': round(macd_current, 2),
            'signal_line': round(signal_current, 2),
            'histogram': round(hist_current, 2),
            'fast': macd_fast,
            'slow': macd_slow,
            'signal_period': macd_signal,
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
        """
        Fetch historical data from Kite with validation
        Ensures sufficient data for indicator calculations
        """
        try:
            # Define minimum required candles per timeframe
            min_required = {
                'day': 60,       # Need 60 days for SMA 50
                '60minute': 100, # Need ~100 hours for calculations
                '15minute': 200  # Need ~200 15-min candles
            }
            
            required_count = min_required.get(interval, 50)
            
            # Get instrument token
            # ✅ NEW: Use index-specific lookup
            # ✅ Get instrument token from pre-built map
            print(f"🔍 Looking up instrument token for: '{symbol}'")
            
            # Check if index_token_map exists and has the symbol
            if hasattr(self.kite, 'index_token_map') and symbol in self.kite.index_token_map:
                token = self.kite.index_token_map[symbol]['token']
                print(f"✅ Found token from map: {token}")
            else:
                # Fallback: try direct lookup
                print(f"⚠️ Symbol '{symbol}' not in index_token_map, trying direct lookup...")
                token = self.kite.get_instrument_token(symbol, 'NSE')
                
                if not token:
                    print(f"❌ Could not find token for {symbol}")
                    print(f"   Available in map: {list(self.kite.index_token_map.keys())[:10] if hasattr(self.kite, 'index_token_map') else 'Map not built'}")
                    raise ValueError(f"Instrument token not found for {symbol}")
            
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            
            print(f"📥 Fetching {interval} data from {from_date.date()} to {to_date.date()}...")

            # ✅ NEW: Rate limiting
            elapsed = time.time() - self.last_api_call
            if elapsed < self.min_call_interval:
                sleep_time = self.min_call_interval - elapsed
                print(f"⏱️  Rate limiting: Waiting {sleep_time:.2f}s before next API call...")
                time.sleep(sleep_time)
            
            # Increment call counter
            self.api_call_count += 1
            
            if self.api_call_count >= self.max_calls_per_minute:
                print(f"⚠️  Approaching API rate limit ({self.api_call_count} calls)")
            
            data = self.kite.get_historical_data(
                instrument_token=token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            
            # ✅ NEW: Update last call timestamp
            self.last_api_call = time.time()
            
            # Validate data received
            if data is None or data.empty:
                raise ValueError(f"No data returned for {symbol} on {interval} timeframe")
            
            # Check if sufficient data
            if len(data) < required_count:
                print(f"⚠️ Warning: Got {len(data)} candles, recommended minimum is {required_count}")
                print(f"   Analysis may be less accurate with limited data")
                
                # If critically insufficient, raise error
                if len(data) < required_count * 0.5:  # If less than 50% of required
                    raise ValueError(
                        f"Insufficient data: Got {len(data)} candles, need at least {int(required_count * 0.5)} for {interval}"
                    )
                else:
                    print(f"✅ Proceeding with {len(data)} candles (minimum 50% threshold met)")
            
            print(f"✅ Fetched {len(data)} candles for {interval} timeframe (required: {required_count})")
            
            # Validate required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            return data
            
        except Exception as e:
            print(f"❌ Error fetching data for {symbol} ({interval}): {e}")
            raise
