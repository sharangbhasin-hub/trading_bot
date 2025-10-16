"""
Multi-Timeframe Trend Analysis for Intraday Options
Combines multiple indicators with weighted scoring
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple, List
from indicators import (
    calculate_ema,
    calculate_sma,
    calculate_rsi,
    calculate_supertrend,
    calculate_macd,
    get_indicator_settings
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

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate ADX (Average Directional Index) to detect trending vs sideways market
        ADX < 20 = Sideways/Weak trend
        ADX 20-25 = Emerging trend
        ADX > 25 = Strong trend
        """
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            # Calculate +DM and -DM
            plus_dm = high.diff()
            minus_dm = -low.diff()
            
            # Set negative values to 0
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            # Calculate True Range
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Smooth using Wilder's smoothing (exponential moving average)
            atr = tr.ewm(alpha=1/period, adjust=False).mean()
            plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
            minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
            
            # Calculate DX and ADX
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
            adx = dx.ewm(alpha=1/period, adjust=False).mean()
            
            return adx.iloc[-1]
        
        except Exception as e:
            print(f"⚠️ Error calculating ADX: {e}")
            return 20  # Return neutral value on error
    
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
            
            # ==========================================
            # ✅ FILTER 1: Market Regime Detection (ADX)
            # ==========================================
            adx_value = self._calculate_adx(daily_data, period=14)
            print(f"🔍 FILTER 1 - Market Regime Detection:")
            print(f"   ADX Value: {adx_value:.2f}")
            
            if adx_value < 20:
                market_regime = "SIDEWAYS"
                print(f"   ⚠️  SIDEWAYS Market (ADX < 20) - High risk of false signals")
            elif adx_value < 25:
                market_regime = "WEAK_TREND"
                print(f"   ⚡ WEAK Trend (ADX 20-25) - Emerging trend")
            else:
                market_regime = "TRENDING"
                print(f"   ✅ STRONG Trend (ADX > 25) - Good trading conditions")
            
            # ==========================================
            # ✅ FILTER 2: Volume Confirmation
            # ==========================================
            recent_volume = daily_data['volume'].iloc[-5:].mean()  # Last 5 days avg
            avg_volume = daily_data['volume'].iloc[-20:].mean()    # Last 20 days avg
            volume_ratio = recent_volume / avg_volume
            
            print(f"\n🔍 FILTER 2 - Volume Confirmation:")
            print(f"   Volume Ratio: {volume_ratio:.2f}x average")
            
            if volume_ratio >= 1.2:
                volume_strength = "HIGH"
                print(f"   ✅ HIGH Volume ({volume_ratio:.2f}x) - Strong conviction")
            elif volume_ratio >= 0.8:
                volume_strength = "NORMAL"
                print(f"   ✓  NORMAL Volume ({volume_ratio:.2f}x) - Average activity")
            else:
                volume_strength = "LOW"
                print(f"   ⚠️  LOW Volume ({volume_ratio:.2f}x) - Weak signal")
            
            # ==========================================
            # ✅ FILTER 3: Indicator Agreement Check
            # ==========================================
            bullish_pct = bullish_count / total_signals
            bearish_pct = bearish_count / total_signals
            max_agreement = max(bullish_pct, bearish_pct)
            
            print(f"\n🔍 FILTER 3 - Indicator Agreement:")
            print(f"   Bullish: {bullish_count}/{total_signals} ({bullish_pct*100:.1f}%)")
            print(f"   Bearish: {bearish_count}/{total_signals} ({bearish_pct*100:.1f}%)")
            
            if max_agreement >= 0.75:
                agreement_level = "VERY_HIGH"
                print(f"   ✅ VERY HIGH Agreement ({max_agreement*100:.0f}%) - Strong consensus")
            elif max_agreement >= 0.6:
                agreement_level = "HIGH"
                print(f"   ✅ HIGH Agreement ({max_agreement*100:.0f}%) - Good consensus")
            elif max_agreement >= 0.5:
                agreement_level = "MODERATE"
                print(f"   ⚡ MODERATE Agreement ({max_agreement*100:.0f}%) - Mixed signals")
            else:
                agreement_level = "LOW"
                print(f"   ⚠️  LOW Agreement ({max_agreement*100:.0f}%) - Conflicting signals")
            
            print(f"\n{'='*60}")
            print(f"FILTER RESULTS SUMMARY")
            print(f"{'='*60}")
            print(f"Market Regime: {market_regime} (ADX: {adx_value:.2f})")
            print(f"Volume Strength: {volume_strength} ({volume_ratio:.2f}x)")
            print(f"Agreement Level: {agreement_level} ({max_agreement*100:.0f}%)")
            print(f"{'='*60}\n")
            
            # ==========================================
            # ✅ APPLY FILTERS TO DECISION LOGIC
            # ==========================================
            
            # Store original score for reference
            original_score = combined_score
            
            # FILTER 1: Sideways Market - Increase threshold
            if market_regime == "SIDEWAYS":
                print(f"🚨 FILTER 1 ACTIVE: Sideways market detected")
                print(f"   Requiring stronger signals (threshold raised by 50%)")
                # Require 50% stronger signal in sideways markets
                if abs(combined_score) < 0.5:
                    combined_score = 0  # Force to neutral
                    print(f"   ⚠️  Signal too weak for sideways market - setting to NEUTRAL")
            
            # FILTER 2: Low Volume - Downgrade confidence
            if volume_strength == "LOW" and market_regime != "TRENDING":
                print(f"🚨 FILTER 2 ACTIVE: Low volume in non-trending market")
                print(f"   Reducing signal strength by 30%")
                combined_score = combined_score * 0.7
            
            # FILTER 3: Low Agreement - Require stronger threshold
            if agreement_level in ["LOW", "MODERATE"]:
                print(f"🚨 FILTER 3 ACTIVE: Insufficient indicator agreement")
                print(f"   Requiring 60%+ agreement for trade signals")
                if max_agreement < 0.6:
                    # Not enough agreement, force neutral unless very strong signal
                    if abs(combined_score) < 0.6:
                        combined_score = 0
                        print(f"   ⚠️  Agreement too low - setting to NEUTRAL")
            
            if combined_score != original_score:
                print(f"\n📊 SCORE ADJUSTED BY FILTERS:")
                print(f"   Original: {original_score:+.2f}")
                print(f"   After Filters: {combined_score:+.2f}")
                print(f"   Change: {combined_score - original_score:+.2f}")
            
            print(f"\n{'='*60}\n")

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
                },
                # ✅ NEW: Filter results
                'filters': {
                    'adx': {
                        'value': adx_value,
                        'regime': market_regime
                    },
                    'volume': {
                        'ratio': volume_ratio,
                        'strength': volume_strength
                    },
                    'agreement': {
                        'bullish_pct': bullish_pct,
                        'bearish_pct': bearish_pct,
                        'level': agreement_level
                    }
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
        # ✅ NEW: Get optimized settings from centralized function
        # Map your timeframe names to standard names
        timeframe_map = {
            'DAILY': 'day',
            'HOURLY': 'hour',
            '15MIN': '15minute'
        }
        
        tf = timeframe_map.get(timeframe, '15minute')
        settings = get_indicator_settings(tf)
        
        # Extract settings for easy access
        ema_period = settings['ema_fast']
        sma_period = settings['ema_slow']
        supertrend_period = settings['supertrend_period']
        supertrend_multiplier = settings['supertrend_multiplier']
        rsi_period = settings['rsi_period']
        macd_fast = settings['macd_fast']
        macd_slow = settings['macd_slow']
        macd_signal = settings['macd_signal']
        
        print(f"   ⚙️  Settings for {timeframe}: EMA({ema_period},{sma_period}), ST({supertrend_period},{supertrend_multiplier}), RSI({rsi_period}), MACD({macd_fast},{macd_slow},{macd_signal})")

        
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
            # ✅ NEW: Use index-specific lookup with defensive checks
            print(f"🔍 Looking up instrument token for: '{symbol}'")
            
            # ✅ NEW: DEFENSIVE CHECK - Build map if missing
            if not hasattr(self.kite, 'index_token_map'):
                print("⚠️ index_token_map attribute missing! Building now...")
                self.kite._build_index_token_map()
            
            # ✅ NEW: DEFENSIVE CHECK - Rebuild if empty
            if not self.kite.index_token_map:
                print("⚠️ index_token_map is empty! Rebuilding...")
                self.kite._build_index_token_map()
                
                # If still empty, something is seriously wrong
                if not self.kite.index_token_map:
                    raise ValueError(
                        "Failed to build index_token_map. "
                        "Instruments may not be loaded properly. "
                        "Please restart the application."
                    )
            
            # Check if index_token_map exists and has the symbol
            if symbol in self.kite.index_token_map:
                token = self.kite.index_token_map[symbol]['token']
                print(f"✅ Found token from map: {token}")
            else:
                # Fallback: try direct lookup
                print(f"⚠️ Symbol '{symbol}' not in index_token_map, trying direct lookup...")
                print(f"   Available in map: {list(self.kite.index_token_map.keys())}")
                
                token = self.kite.get_instrument_token(symbol, 'NSE')
                
                if not token:
                    raise ValueError(
                        f"Instrument token not found for '{symbol}'. "
                        f"Available indices: {list(self.kite.index_token_map.keys())}"
                    )
            
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

    def analyze_volume_confirmation_intraday(self, df_5min: pd.DataFrame) -> Dict:
        """
        Volume analysis specifically for intraday options entry confirmation
        Analyzes 5-min chart volume patterns
        
        Returns:
            Dict with volume confirmation status, ratio, and strength
        """
        try:
            if df_5min is None or df_5min.empty or len(df_5min) < 20:
                return {
                    'volume_confirmed': False,
                    'volume_ratio': 1.0,
                    'strength': 'INSUFFICIENT_DATA',
                    'error': 'Not enough candles for volume analysis'
                }
            
            # Get volume column (handle different naming)
            vol_col = None
            for col in ['volume', 'Volume', 'VOLUME']:
                if col in df_5min.columns:
                    vol_col = col
                    break
            
            if vol_col is None:
                return {
                    'volume_confirmed': False,
                    'volume_ratio': 1.0,
                    'strength': 'NO_VOLUME_DATA',
                    'error': 'Volume column not found'
                }
            
            # Calculate average volume (last 20 candles)
            avg_volume = df_5min[vol_col].tail(20).mean()
            
            # Current candle volume
            current_volume = df_5min[vol_col].iloc[-1]
            
            # Previous candle volume (for trend)
            prev_volume = df_5min[vol_col].iloc[-2]
            
            # Volume ratio
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Volume trend (increasing or decreasing)
            volume_increasing = current_volume > prev_volume
            
            # Last 3 candles average (short-term trend)
            recent_3_avg = df_5min[vol_col].tail(3).mean()
            volume_trending_up = recent_3_avg > avg_volume
            
            # Check if current candle is bullish or bearish
            last_candle_bullish = df_5min['close'].iloc[-1] > df_5min['open'].iloc[-1]
            
            # Determine strength and confirmation
            confirmed = False
            strength = 'WEAK'
            
            if volume_ratio >= 2.0:
                confirmed = True
                strength = 'VERY_STRONG'
            elif volume_ratio >= 1.5:
                confirmed = True
                strength = 'STRONG'
            elif volume_ratio >= 1.2:
                confirmed = True
                strength = 'MODERATE'
            else:
                confirmed = False
                strength = 'WEAK'
            
            # Additional context
            volume_with_trend = (last_candle_bullish and volume_increasing) or \
                               (not last_candle_bullish and volume_increasing)
            
            return {
                'volume_confirmed': confirmed,
                'volume_ratio': round(volume_ratio, 2),
                'strength': strength,
                'current_volume': int(current_volume),
                'avg_volume': int(avg_volume),
                'volume_increasing': volume_increasing,
                'volume_trending_up': volume_trending_up,
                'bullish_volume': last_candle_bullish and confirmed,
                'bearish_volume': not last_candle_bullish and confirmed,
                'volume_with_trend': volume_with_trend
            }
            
        except Exception as e:
            return {
                'volume_confirmed': False,
                'volume_ratio': 1.0,
                'strength': 'ERROR',
                'error': str(e)
            }

    def calculate_confluence_score_intraday(
        self,
        trend_analysis: Dict,
        candlestick_patterns: List[Dict],
        volume_confirmation: Dict,
        chart_patterns: List[Dict],
        spot_price: float,
        support_level: float,
        resistance_level: float
    ) -> Dict:
        """
        5-Factor Confluence Scoring Engine for Intraday ITM Options
        
        Scoring System (11 points maximum):
        1. Trend Structure: +3 points (Daily +2, alignment +1)
        2. Support/Resistance Position: +1 point
        3. Indicator Alignment: +2 points (3+ indicators)
        4. Candlestick Pattern: +2 points (high-strength)
        5. Volume Confirmation: +1 point (>1.5x avg)
        6. Chart Pattern: +2 points (valid breakout)
        
        Thresholds:
        - Score >= 8: STRONG SIGNAL → Execute ITM trade
        - Score 7: MODERATE SIGNAL → Trade with tight stop
        - Score < 7: NO TRADE → Wait for better setup
        """
        
        score = 0
        breakdown = []
        
        try:
            # ========== 1. TREND STRUCTURE (max 3 points) ==========
            
            overall_trend = trend_analysis.get('overallTrend', 'NEUTRAL')
            bullish_pct = trend_analysis.get('consensusBullishPct', 50)
            bearish_pct = trend_analysis.get('consensusBearishPct', 50)
            
            if 'bullish' in overall_trend.lower():
                score += 2
                breakdown.append(f"+2: Daily trend BULLISH ({bullish_pct:.0f}% consensus)")
                trend_direction = 'bullish'
            elif 'bearish' in overall_trend.lower():
                score += 2
                breakdown.append(f"+2: Daily trend BEARISH ({bearish_pct:.0f}% consensus)")
                trend_direction = 'bearish'
            else:
                breakdown.append("0: Daily trend NEUTRAL - avoid trading")
                trend_direction = 'neutral'
            
            # Check 15-min alignment (if daily is clear)
            if trend_direction != 'neutral':
                timeframe_analysis = trend_analysis.get('timeframeAnalysis', {})
                min_15_data = timeframe_analysis.get('15min', {})
                
                if min_15_data:
                    min_15_trend = min_15_data.get('trend', '')
                    if trend_direction in min_15_trend.lower():
                        score += 1
                        breakdown.append("+1: 15-min trend aligned with daily")
                    else:
                        breakdown.append("0: 15-min trend NOT aligned")
                else:
                    breakdown.append("0: 15-min data not available")
            
            # ========== 2. SUPPORT/RESISTANCE POSITION (max 1 point) ==========
            
            # Check if price is at key level (±0.3% tolerance for intraday)
            at_support = False
            at_resistance = False
            
            if support_level > 0:
                distance_from_support = abs(spot_price - support_level) / support_level
                at_support = distance_from_support < 0.003  # Within 0.3%
            
            if resistance_level > 0:
                distance_from_resistance = abs(spot_price - resistance_level) / resistance_level
                at_resistance = distance_from_resistance < 0.003
            
            if trend_direction == 'bullish' and at_support:
                score += 1
                breakdown.append(f"+1: Price at support ({support_level:.2f}) - bullish setup")
            elif trend_direction == 'bearish' and at_resistance:
                score += 1
                breakdown.append(f"+1: Price at resistance ({resistance_level:.2f}) - bearish setup")
            else:
                if at_support or at_resistance:
                    breakdown.append("0: At key level but not aligned with trend")
                else:
                    breakdown.append("0: Not at key support/resistance level")
            
            # ========== 3. INDICATOR ALIGNMENT (max 2 points) ==========
            
            # Count aligned indicators from timeframe analysis
            indicators_aligned = 0
            indicator_details = []
            
            timeframe_analysis = trend_analysis.get('timeframeAnalysis', {})
            
            for tf_name, tf_data in timeframe_analysis.items():
                if isinstance(tf_data, dict):
                    # RSI alignment
                    rsi_data = tf_data.get('rsi', {})
                    if rsi_data:
                        rsi_value = rsi_data.get('value', 50)
                        rsi_signal = rsi_data.get('signal', '')
                        
                        if trend_direction == 'bullish' and rsi_value > 50:
                            indicators_aligned += 1
                            indicator_details.append(f"✅ RSI ({tf_name}): {rsi_value:.1f}")
                        elif trend_direction == 'bearish' and rsi_value < 50:
                            indicators_aligned += 1
                            indicator_details.append(f"✅ RSI ({tf_name}): {rsi_value:.1f}")
                    
                    # MACD alignment
                    macd_data = tf_data.get('macd', {})
                    if macd_data:
                        macd_signal = macd_data.get('signal', '')
                        if trend_direction in macd_signal.lower():
                            indicators_aligned += 1
                            indicator_details.append(f"✅ MACD ({tf_name})")
                    
                    # EMA alignment
                    ema_data = tf_data.get('ema', {})
                    if ema_data:
                        ema_signal = ema_data.get('signal', '')
                        if trend_direction in ema_signal.lower():
                            indicators_aligned += 1
                            indicator_details.append(f"✅ EMA ({tf_name})")
            
            if indicators_aligned >= 3:
                score += 2
                breakdown.append(f"+2: {indicators_aligned} indicators aligned with trend")
            elif indicators_aligned >= 2:
                score += 1
                breakdown.append(f"+1: {indicators_aligned} indicators aligned")
            else:
                breakdown.append(f"0: Only {indicators_aligned} indicators aligned (need 3+)")
            
            # ========== 4. CANDLESTICK PATTERN (max 2 points) ==========
            
            if candlestick_patterns and len(candlestick_patterns) > 0:
                # Get strongest pattern
                strongest = max(candlestick_patterns, key=lambda p: p.get('strength', 0))
                
                pattern_name = strongest.get('pattern', '')
                pattern_type = strongest.get('type', '')
                pattern_strength = strongest.get('strength', 0)
                
                # Pattern must be high strength (>= 80) AND aligned with trend
                if pattern_strength >= 80:
                    if pattern_type == 'bullish' and trend_direction == 'bullish':
                        score += 2
                        breakdown.append(f"+2: {pattern_name} (bullish, {pattern_strength}%)")
                    elif pattern_type == 'bearish' and trend_direction == 'bearish':
                        score += 2
                        breakdown.append(f"+2: {pattern_name} (bearish, {pattern_strength}%)")
                    else:
                        breakdown.append(f"0: {pattern_name} not aligned with trend")
                else:
                    breakdown.append(f"0: {pattern_name} strength below 80% ({pattern_strength}%)")
            else:
                breakdown.append("0: No valid candlestick patterns detected")
            
            # ========== 5. VOLUME CONFIRMATION (max 1 point) ==========
            
            volume_confirmed = volume_confirmation.get('volume_confirmed', False)
            volume_ratio = volume_confirmation.get('volume_ratio', 1.0)
            volume_strength = volume_confirmation.get('strength', 'WEAK')
            
            if volume_confirmed:
                score += 1
                breakdown.append(f"+1: Volume confirmed ({volume_ratio:.2f}x, {volume_strength})")
            else:
                breakdown.append(f"0: Volume not confirmed ({volume_ratio:.2f}x, {volume_strength})")
            
            # ========== 6. CHART PATTERN (max 2 points) ==========
            
            if chart_patterns and len(chart_patterns) > 0:
                # Get highest confidence pattern (handle both dict and list structures)
                if isinstance(chart_patterns, dict):
                    # Enhanced structure: get tradeable patterns
                    tradeable_charts = chart_patterns.get('tradeable', [])
                    if tradeable_charts:
                        bestpattern = max(tradeable_charts, key=lambda p: p.get('confidence', 0))
                    else:
                        bestpattern = None
                else:
                    # Old structure: simple list
                    bestpattern = max(chart_patterns, key=lambda p: p.get('confidence', 0))
                
                if bestpattern:
                    chartname = bestpattern.get('pattern', '')
                    charttype = bestpattern.get('type', '')
                    chartconfidence = bestpattern.get('confidence', 0)
                    
                    # Pattern must be high confidence (>= 75) AND aligned with trend
                    if chartconfidence >= 75:
                        if charttype == 'bullish' and trend_direction == 'bullish':
                            score += 2
                            breakdown.append(f"+2: {chartname} breakout ({chartconfidence}% confidence)")
                        elif charttype == 'bearish' and trend_direction == 'bearish':
                            score += 2
                            breakdown.append(f"+2: {chartname} breakdown ({chartconfidence}% confidence)")
                        else:
                            breakdown.append(f"0: {chartname} not aligned with trend")
                    else:
                        breakdown.append(f"0: {chartname} confidence below 75%")
                else:
                    breakdown.append("0: No tradeable chart patterns detected")
            else:
                breakdown.append("0: No chart patterns detected")
            
            # ========== NEW: DEDUCT POINTS FOR WARNING CHART PATTERNS ==========
            
            # Check if chart_patterns has warning patterns
            chart_warnings = []
            
            if isinstance(chart_patterns, dict):
                # Enhanced structure: {'tradeable': [...], 'warnings': [...]}
                chart_warnings = chart_patterns.get('warnings', [])
            elif isinstance(chart_patterns, list):
                # Old structure: check for patterns marked as warnings
                chart_warnings = [p for p in chart_patterns if p.get('category') == 'warning']
            
            if chart_warnings and len(chart_warnings) > 0:
                # Count warnings by severity
                high_severity_chart = sum(1 for w in chart_warnings if w.get('severity') == 'HIGH')
                medium_severity_chart = sum(1 for w in chart_warnings if w.get('severity') == 'MEDIUM')
                
                chart_penalty = 0
                
                if high_severity_chart > 0:
                    chart_penalty = -2
                    breakdown.append(f"-2: {high_severity_chart} slow chart pattern(s) detected (H&S, Double Top/Bottom - too slow for intraday)")
                elif medium_severity_chart > 0:
                    chart_penalty = -1
                    breakdown.append(f"-1: {medium_severity_chart} unreliable chart pattern(s) (Symmetrical Triangle, Double patterns)")
                
                # Apply penalty to score
                score += chart_penalty
            
            # ========== FINAL SIGNAL DETERMINATION ==========
            
            abs_score = abs(score)
            
            # Determine signal and action
            if abs_score >= 8:
                signal = 'STRONG_BUY' if trend_direction == 'bullish' else 'STRONG_SELL' if trend_direction == 'bearish' else 'NO_TRADE'
                action = 'EXECUTE_ITM_TRADE'
                tradedirection = 'CALL' if trend_direction == 'bullish' else 'PUT' if trend_direction == 'bearish' else 'NONE'
            elif abs_score >= 7:
                signal = 'MODERATE_BUY' if trend_direction == 'bullish' else 'MODERATE_SELL' if trend_direction == 'bearish' else 'NO_TRADE'
                action = 'TRADE_WITH_TIGHT_STOP'
                tradedirection = 'CALL' if trend_direction == 'bullish' else 'PUT' if trend_direction == 'bearish' else 'NONE'
            else:
                signal = 'NO_TRADE'
                action = 'WAIT_FOR_BETTER_SETUP'
                tradedirection = 'NONE'
            
            return {
                'confluence_score': score,
                'max_score': 11,
                'signal': signal,
                'action': action,
                'trade_direction': tradedirection,
                'breakdown': breakdown,
                'trend_direction': trend_direction,
                'indicators_aligned_count': indicators_aligned,
                'indicator_details': indicator_details,
                'candlestick_detected': len(candlestickpatterns) > 0 if candlestickpatterns else False,
                'chart_pattern_detected': len(chart_patterns) > 0 if chart_patterns else False,
                'volume_confirmed': volumeconfirmed,
                'at_key_level': atsupport or atresistance
            }
            
            # ========== FINAL SIGNAL DETERMINATION ==========
            
            abs_score = abs(score)
            
            # Determine signal and action
            if abs_score >= 8:
                signal = 'STRONG_BUY' if trend_direction == 'bullish' else 'STRONG_SELL' if trend_direction == 'bearish' else 'NO_TRADE'
                action = 'EXECUTE_ITM_TRADE'
                trade_direction = 'CALL' if trend_direction == 'bullish' else 'PUT' if trend_direction == 'bearish' else 'NONE'
            elif abs_score >= 7:
                signal = 'MODERATE_BUY' if trend_direction == 'bullish' else 'MODERATE_SELL' if trend_direction == 'bearish' else 'NO_TRADE'
                action = 'TRADE_WITH_TIGHT_STOP'
                trade_direction = 'CALL' if trend_direction == 'bullish' else 'PUT' if trend_direction == 'bearish' else 'NONE'
            else:
                signal = 'NO_TRADE'
                action = 'WAIT_FOR_BETTER_SETUP'
                trade_direction = 'NONE'
            
            return {
                'confluence_score': score,
                'max_score': 11,
                'signal': signal,
                'action': action,
                'trade_direction': trade_direction,
                'breakdown': breakdown,
                'trend_direction': trend_direction,
                'indicators_aligned_count': indicators_aligned,
                'indicator_details': indicator_details,
                'candlestick_detected': len(candlestick_patterns) > 0 if candlestick_patterns else False,
                'chart_pattern_detected': len(chart_patterns) > 0 if chart_patterns else False,
                'volume_confirmed': volume_confirmed,
                'at_key_level': at_support or at_resistance
            }
            
        except Exception as e:
            return {
                'confluence_score': 0,
                'max_score': 11,
                'signal': 'ERROR',
                'action': 'NO_TRADE',
                'trade_direction': 'NONE',
                'breakdown': [f"Error: {str(e)}"],
                'error': str(e)
            }
