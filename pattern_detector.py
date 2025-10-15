"""
Pattern Detection & Technical Analysis
Migrated from old codebase - adapted for Kite Connect
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import streamlit as st

class PatternDetector:
    """Detects candlestick patterns and technical setups"""
    
    def __init__(self):
        self.patterns_detected = []
    
    def detect_all_patterns(self, df: pd.DataFrame, support: float = 0, resistance: float = 0) -> List[Dict]:
        """
        Detect all candlestick patterns in the dataframe
        Args:
            df: OHLCV DataFrame
            support: Support level
            resistance: Resistance level
        Returns:
            List of detected patterns with metadata
        """
        patterns = []
        
        if df is None or df.empty or len(df) < 3:
            return [{
                'pattern': 'Insufficient Data',
                'type': 'neutral',
                'strength': 0,
                'confidence': 0,
                'category': 'none',
                'description': 'Not enough data for pattern detection'
            }]
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return [{
                'pattern': 'Data Error',
                'type': 'neutral',
                'strength': 0,
                'confidence': 0,
                'category': 'none',
                'description': 'Missing required OHLC columns'
            }]
        
        # Detect various patterns
        patterns.extend(self._detect_engulfing(df))
        patterns.extend(self._detect_doji(df))
        patterns.extend(self._detect_hammer(df))
        patterns.extend(self._detect_shooting_star(df))
        patterns.extend(self._detect_morning_evening_star(df))
        
        # Sort by strength
        patterns.sort(key=lambda x: x.get('strength', 0), reverse=True)
        
        if not patterns:
            patterns.append({
                'pattern': 'No Significant Pattern',
                'type': 'neutral',
                'strength': 0,
                'confidence': 0,
                'category': 'none',
                'description': 'No candlestick patterns detected'
            })
        
        return patterns
    
    def _detect_engulfing(self, df: pd.DataFrame) -> List[Dict]:
        """Detect bullish/bearish engulfing patterns"""
        patterns = []
        if len(df) < 2:
            return patterns
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Bullish Engulfing
        if (prev['close'] < prev['open'] and  # Previous candle bearish
            last['close'] > last['open'] and   # Current candle bullish
            last['open'] <= prev['close'] and  # Opens at or below previous close
            last['close'] >= prev['open']):    # Closes at or above previous open
            
            body_size = abs(last['close'] - last['open'])
            prev_body = abs(prev['close'] - prev['open'])
            strength = min(100, (body_size / prev_body) * 50)
            
            patterns.append({
                'pattern': 'Bullish Engulfing',
                'type': 'bullish',
                'strength': round(strength, 2),
                'confidence': round(min(95, strength * 0.9), 2),
                'category': 'reversal',
                'description': 'Strong bullish reversal pattern - buyers overpowered sellers'
            })
        
        # Bearish Engulfing
        elif (prev['close'] > prev['open'] and  # Previous candle bullish
              last['close'] < last['open'] and   # Current candle bearish
              last['open'] >= prev['close'] and  # Opens at or above previous close
              last['close'] <= prev['open']):    # Closes at or below previous open
            
            body_size = abs(last['close'] - last['open'])
            prev_body = abs(prev['close'] - prev['open'])
            strength = min(100, (body_size / prev_body) * 50)
            
            patterns.append({
                'pattern': 'Bearish Engulfing',
                'type': 'bearish',
                'strength': round(strength, 2),
                'confidence': round(min(95, strength * 0.9), 2),
                'category': 'reversal',
                'description': 'Strong bearish reversal pattern - sellers overpowered buyers'
            })
        
        return patterns
    
    def _detect_doji(self, df: pd.DataFrame) -> List[Dict]:
        """Detect Doji patterns"""
        patterns = []
        if len(df) < 1:
            return patterns
        
        last = df.iloc[-1]
        body = abs(last['close'] - last['open'])
        range_size = last['high'] - last['low']
        
        if range_size == 0:
            return patterns
        
        # Doji: Very small body relative to range
        if body / range_size < 0.1:
            strength = 60
            patterns.append({
                'pattern': 'Doji',
                'type': 'neutral',
                'strength': strength,
                'confidence': 70,
                'category': 'indecision',
                'description': 'Market indecision - potential reversal or continuation'
            })
        
        return patterns
    
    def _detect_hammer(self, df: pd.DataFrame) -> List[Dict]:
        """Detect Hammer and Inverted Hammer"""
        patterns = []
        if len(df) < 1:
            return patterns
        
        last = df.iloc[-1]
        body = abs(last['close'] - last['open'])
        upper_shadow = last['high'] - max(last['close'], last['open'])
        lower_shadow = min(last['close'], last['open']) - last['low']
        range_size = last['high'] - last['low']
        
        if range_size == 0:
            return patterns
        
        # Hammer: Long lower shadow, small body, little upper shadow
        if lower_shadow > 2 * body and upper_shadow < body:
            strength = min(85, (lower_shadow / range_size) * 100)
            patterns.append({
                'pattern': 'Hammer',
                'type': 'bullish',
                'strength': round(strength, 2),
                'confidence': round(strength * 0.85, 2),
                'category': 'reversal',
                'description': 'Bullish reversal - buyers rejected lower prices'
            })
        
        # Inverted Hammer: Long upper shadow, small body, little lower shadow
        elif upper_shadow > 2 * body and lower_shadow < body:
            strength = min(75, (upper_shadow / range_size) * 90)
            patterns.append({
                'pattern': 'Inverted Hammer',
                'type': 'bullish',
                'strength': round(strength, 2),
                'confidence': round(strength * 0.80, 2),
                'category': 'reversal',
                'description': 'Potential bullish reversal - needs confirmation'
            })
        
        return patterns
    
    def _detect_shooting_star(self, df: pd.DataFrame) -> List[Dict]:
        """Detect Shooting Star pattern"""
        patterns = []
        if len(df) < 1:
            return patterns
        
        last = df.iloc[-1]
        body = abs(last['close'] - last['open'])
        upper_shadow = last['high'] - max(last['close'], last['open'])
        lower_shadow = min(last['close'], last['open']) - last['low']
        range_size = last['high'] - last['low']
        
        if range_size == 0:
            return patterns
        
        # Shooting Star: Long upper shadow, small body at bottom, little lower shadow
        if upper_shadow > 2 * body and lower_shadow < body and last['close'] < last['open']:
            strength = min(85, (upper_shadow / range_size) * 100)
            patterns.append({
                'pattern': 'Shooting Star',
                'type': 'bearish',
                'strength': round(strength, 2),
                'confidence': round(strength * 0.85, 2),
                'category': 'reversal',
                'description': 'Bearish reversal - sellers rejected higher prices'
            })
        
        return patterns
    
    def _detect_morning_evening_star(self, df: pd.DataFrame) -> List[Dict]:
        """Detect Morning Star and Evening Star patterns"""
        patterns = []
        if len(df) < 3:
            return patterns
        
        c1 = df.iloc[-3]  # First candle
        c2 = df.iloc[-2]  # Middle (star) candle
        c3 = df.iloc[-1]  # Third candle
        
        # Morning Star (Bullish)
        if (c1['close'] < c1['open'] and  # First candle bearish
            abs(c2['close'] - c2['open']) < abs(c1['close'] - c1['open']) * 0.3 and  # Small middle
            c3['close'] > c3['open'] and  # Third candle bullish
            c3['close'] > (c1['open'] + c1['close']) / 2):  # Closes above midpoint of first
            
            patterns.append({
                'pattern': 'Morning Star',
                'type': 'bullish',
                'strength': 88,
                'confidence': 85,
                'category': 'reversal',
                'description': 'Strong bullish reversal - three-candle pattern signaling trend change'
            })
        
        # Evening Star (Bearish)
        elif (c1['close'] > c1['open'] and  # First candle bullish
              abs(c2['close'] - c2['open']) < abs(c1['close'] - c1['open']) * 0.3 and  # Small middle
              c3['close'] < c3['open'] and  # Third candle bearish
              c3['close'] < (c1['open'] + c1['close']) / 2):  # Closes below midpoint of first
            
            patterns.append({
                'pattern': 'Evening Star',
                'type': 'bearish',
                'strength': 88,
                'confidence': 85,
                'category': 'reversal',
                'description': 'Strong bearish reversal - three-candle pattern signaling trend change'
            })
        
        return patterns
    
    def run_confirmation_checklist(self, analysis_results: Dict) -> Dict:
        """
        5-Point Trade Confirmation Checklist
        Returns confirmation status for each point
        """
        checklist = {
            'data_available': False,
            '1. At Key S/R Level': '❌ PENDING',
            '2. Price Rejection': '❌ PENDING',
            '3. Chart Pattern Confirmed': '❌ PENDING',
            '4. Candlestick Signal': '❌ PENDING',
            '5. Indicator Alignment': '❌ PENDING',
            'FINAL_SIGNAL': 'HOLD',
            'error': None
        }
        
        try:
            # Check if we have required data
            if '5mdata' not in analysis_results or analysis_results.get('5mdata') is None:
                checklist['error'] = '5-minute data not available'
                return checklist
            
            df = analysis_results['5mdata']
            if df.empty or len(df) < 3:
                checklist['error'] = 'Insufficient 5-minute candles'
                return checklist
            
            support = analysis_results.get('support', 0)
            resistance = analysis_results.get('resistance', 0)
            latest_price = analysis_results.get('latest_price', 0)
            
            if support == 0 or resistance == 0 or latest_price == 0:
                checklist['error'] = 'Support/Resistance levels not calculated'
                return checklist
            
            checklist['data_available'] = True
            confirmations = 0
            
            # Point 1: At Key S/R Level
            support_distance = abs(latest_price - support) / latest_price
            resistance_distance = abs(latest_price - resistance) / latest_price
            
            if support_distance < 0.01 or resistance_distance < 0.01:  # Within 1%
                checklist['1. At Key S/R Level'] = '✅ YES'
                confirmations += 1
            else:
                checklist['1. At Key S/R Level'] = f'⚠️ NO (Support: {support_distance*100:.2f}% away)'
            
            # Point 2: Price Rejection
            last_candle = df.iloc[-1]
            body = abs(last_candle['close'] - last_candle['open'])
            upper_wick = last_candle['high'] - max(last_candle['close'], last_candle['open'])
            lower_wick = min(last_candle['close'], last_candle['open']) - last_candle['low']
            
            if upper_wick > 2 * body or lower_wick > 2 * body:
                checklist['2. Price Rejection'] = '✅ YES (Long wick detected)'
                confirmations += 1
            else:
                checklist['2. Price Rejection'] = '⚠️ NO (No strong rejection)'
            
            # Point 3: Chart Pattern
            patterns = analysis_results.get('all_patterns', [])
            strong_patterns = [p for p in patterns if p.get('strength', 0) >= 70]
            
            if strong_patterns:
                checklist['3. Chart Pattern Confirmed'] = f"✅ YES ({strong_patterns[0]['pattern']})"
                confirmations += 1
            else:
                checklist['3. Chart Pattern Confirmed'] = '⚠️ NO (No strong pattern)'
            
            # Point 4: Candlestick Signal
            candlestick_pattern = analysis_results.get('candlestick_pattern', '')
            pattern_type = analysis_results.get('pattern_type', 'neutral')
            
            if pattern_type in ['bullish', 'bearish']:
                checklist['4. Candlestick Signal'] = f'✅ YES ({candlestick_pattern})'
                confirmations += 1
            else:
                checklist['4. Candlestick Signal'] = '⚠️ NO (Neutral/No pattern)'
            
            # Point 5: Indicator Alignment
            rsi = analysis_results.get('rsi', 50)
            macd = analysis_results.get('macd', {})
            macd_histogram = macd.get('histogram', 0)
            
            indicators_bullish = 0
            indicators_bearish = 0
            
            if rsi > 50:
                indicators_bullish += 1
            elif rsi < 50:
                indicators_bearish += 1
            
            if macd_histogram > 0:
                indicators_bullish += 1
            elif macd_histogram < 0:
                indicators_bearish += 1
            
            if indicators_bullish >= 2:
                checklist['5. Indicator Alignment'] = '✅ YES (Bullish alignment)'
                confirmations += 1
            elif indicators_bearish >= 2:
                checklist['5. Indicator Alignment'] = '✅ YES (Bearish alignment)'
                confirmations += 1
            else:
                checklist['5. Indicator Alignment'] = '⚠️ NO (Mixed signals)'
            
            # Final Signal
            if confirmations >= 3:
                if pattern_type == 'bullish':
                    checklist['FINAL_SIGNAL'] = '🟢 BUY SIGNAL'
                elif pattern_type == 'bearish':
                    checklist['FINAL_SIGNAL'] = '🔴 SELL SIGNAL'
                else:
                    checklist['FINAL_SIGNAL'] = 'HOLD (Neutral pattern)'
            else:
                checklist['FINAL_SIGNAL'] = f'HOLD ({confirmations}/5 confirmations)'
            
        except Exception as e:
            checklist['error'] = f'Checklist generation failed: {str(e)}'
        
        return checklist
