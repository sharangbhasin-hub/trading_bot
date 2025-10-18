"""
Pattern Detection & Technical Analysis
Complete implementation with classification system for intraday options trading
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import streamlit as st

class PatternDetector:
    """Detects candlestick patterns and classifies them for intraday trading"""
    
    def __init__(self):
        self.patterns_detected = []
        
        # TRADEABLE PATTERNS - High probability for intraday options
        self.INTRADAY_TRADEABLE_PATTERNS = [
            # Bullish - Single Candle
            'Hammer', 'Inverted Hammer',
            # Bullish - Two Candle
            'Bullish Engulfing', 'Piercing Pattern',
            # Bullish - Three Candle (strong momentum only)
            'Three White Soldiers',
            # Bearish - Single Candle
            'Shooting Star', 'Hanging Man',
            # Bearish - Two Candle
            'Bearish Engulfing', 'Dark Cloud Cover',
            # Bearish - Three Candle (strong momentum only)
            'Three Black Crows'
        ]
        
        # WARNING PATTERNS - Indecision/Slow patterns (NO TRADE)
        self.INTRADAY_WARNING_PATTERNS = {
            # HIGH SEVERITY - DO NOT TRADE
            'Doji': {
                'reason': 'Indecision - no clear direction',
                'action': 'Do NOT trade - wait for directional candle',
                'severity': 'HIGH'
            },
            'Spinning Top': {
                'reason': 'Indecision - weak momentum',
                'action': 'Do NOT trade - wait for confirmation',
                'severity': 'HIGH'
            },
            'Long Legged Doji': {
                'reason': 'High indecision - volatile but directionless',
                'action': 'Avoid trading',
                'severity': 'HIGH'
            },
            
            # MEDIUM SEVERITY - WAIT FOR CONFIRMATION
            'Dragonfly Doji': {
                'reason': 'Indecision at support - needs next candle confirmation',
                'action': 'Wait for next candle',
                'severity': 'MEDIUM'
            },
            'Gravestone Doji': {
                'reason': 'Indecision at resistance - needs next candle confirmation',
                'action': 'Wait for next candle',
                'severity': 'MEDIUM'
            },
            'Morning Star': {
                'reason': 'Too slow (3-candle pattern)',
                'action': 'Wait for faster confirmation',
                'severity': 'MEDIUM'
            },
            'Evening Star': {
                'reason': 'Too slow (3-candle pattern)',
                'action': 'Wait for faster confirmation',
                'severity': 'MEDIUM'
            },
            'Morning Doji Star': {
                'reason': 'Too slow + indecision',
                'action': 'Avoid trading',
                'severity': 'MEDIUM'
            },
            'Evening Doji Star': {
                'reason': 'Too slow + indecision',
                'action': 'Avoid trading',
                'severity': 'MEDIUM'
            },
            
            # LOW SEVERITY - WEAK PATTERNS
            'Bullish Harami': {
                'reason': 'Weak signal (only 70% strength)',
                'action': 'Prefer stronger patterns',
                'severity': 'LOW'
            },
            'Bearish Harami': {
                'reason': 'Weak signal (only 70% strength)',
                'action': 'Prefer stronger patterns',
                'severity': 'LOW'
            },
            'Harami Cross': {
                'reason': 'Weak indecision',
                'action': 'Avoid trading',
                'severity': 'MEDIUM'
            }
        }
    
    def detect_all_patterns(self, df: pd.DataFrame, support: float = 0, resistance: float = 0) -> List[Dict]:
        """Detect ALL candlestick patterns in the dataframe"""
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
        
        # Detect all patterns
        patterns.extend(self.detect_engulfing(df))
        patterns.extend(self.detect_piercing_dark_cloud(df))
        patterns.extend(self.detect_doji_variants(df))
        patterns.extend(self.detect_spinning_top(df))
        patterns.extend(self.detect_hammer(df))
        patterns.extend(self.detect_hanging_man(df))
        patterns.extend(self.detect_shooting_star(df))
        patterns.extend(self.detect_morning_evening_star(df))
        patterns.extend(self.detect_harami_patterns(df))
        patterns.extend(self.detect_three_soldiers_crows(df))
        
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
    
    def detect_engulfing(self, df: pd.DataFrame) -> List[Dict]:
        """Detect Bullish/Bearish Engulfing"""
        patterns = []
        if len(df) < 2:
            return patterns
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Bullish Engulfing
        if (prev['close'] < prev['open'] and last['close'] > last['open'] and
            last['open'] <= prev['close'] and last['close'] >= prev['open']):
            body_size = abs(last['close'] - last['open'])
            prev_body = abs(prev['close'] - prev['open'])
            strength = min(100, (body_size / prev_body) * 50)
            patterns.append({
                'pattern': 'Bullish Engulfing',
                'type': 'bullish',
                'strength': round(strength, 2),
                'confidence': round(min(95, strength * 0.9), 2),
                'category': 'reversal',
                'description': 'Strong bullish reversal - buyers overpowered sellers'
            })
        
        # Bearish Engulfing
        elif (prev['close'] > prev['open'] and last['close'] < last['open'] and
              last['open'] >= prev['close'] and last['close'] <= prev['open']):
            body_size = abs(last['close'] - last['open'])
            prev_body = abs(prev['close'] - prev['open'])
            strength = min(100, (body_size / prev_body) * 50)
            patterns.append({
                'pattern': 'Bearish Engulfing',
                'type': 'bearish',
                'strength': round(strength, 2),
                'confidence': round(min(95, strength * 0.9), 2),
                'category': 'reversal',
                'description': 'Strong bearish reversal - sellers overpowered buyers'
            })
        
        return patterns
    
    def detect_piercing_dark_cloud(self, df: pd.DataFrame) -> List[Dict]:
        """Detect Piercing Pattern & Dark Cloud Cover"""
        patterns = []
        if len(df) < 2:
            return patterns
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        prev_body = abs(prev['close'] - prev['open'])
        prev_mid = (prev['open'] + prev['close']) / 2
        
        # Piercing Pattern
        if (prev['close'] < prev['open'] and last['close'] > last['open'] and
            last['open'] < prev['low'] and last['close'] > prev_mid and last['close'] < prev['open']):
            strength = min(85, ((last['close'] - prev_mid) / prev_body) * 100)
            patterns.append({
                'pattern': 'Piercing Pattern',
                'type': 'bullish',
                'strength': round(strength, 2),
                'confidence': 80,
                'category': 'reversal',
                'description': 'Bullish reversal - buyers pushing above midpoint'
            })
        
        # Dark Cloud Cover
        elif (prev['close'] > prev['open'] and last['close'] < last['open'] and
              last['open'] > prev['high'] and last['close'] < prev_mid and last['close'] > prev['open']):
            strength = min(85, ((prev_mid - last['close']) / prev_body) * 100)
            patterns.append({
                'pattern': 'Dark Cloud Cover',
                'type': 'bearish',
                'strength': round(strength, 2),
                'confidence': 80,
                'category': 'reversal',
                'description': 'Bearish reversal - sellers pushing below midpoint'
            })
        
        return patterns
    
    def detect_doji_variants(self, df: pd.DataFrame) -> List[Dict]:
        """Detect ALL Doji variants"""
        patterns = []
        if len(df) < 1:
            return patterns
        
        last = df.iloc[-1]
        body = abs(last['close'] - last['open'])
        range_size = last['high'] - last['low']
        upper_shadow = last['high'] - max(last['close'], last['open'])
        lower_shadow = min(last['close'], last['open']) - last['low']
        
        if range_size == 0:
            return patterns
        
        body_ratio = body / range_size
        
        # Classic Doji
        if body_ratio < 0.1:
            patterns.append({
                'pattern': 'Doji',
                'type': 'neutral',
                'strength': 60,
                'confidence': 70,
                'category': 'indecision',
                'description': 'Market indecision - NO TRADE signal'
            })
        
        # Long Legged Doji
        if body_ratio < 0.05 and upper_shadow > body * 3 and lower_shadow > body * 3:
            patterns.append({
                'pattern': 'Long Legged Doji',
                'type': 'neutral',
                'strength': 65,
                'confidence': 75,
                'category': 'indecision',
                'description': 'High volatility indecision - NO TRADE'
            })
        
        # Dragonfly Doji
        if body_ratio < 0.1 and lower_shadow > body * 3 and upper_shadow < body:
            patterns.append({
                'pattern': 'Dragonfly Doji',
                'type': 'bullish',
                'strength': 70,
                'confidence': 75,
                'category': 'reversal',
                'description': 'Potential bullish reversal at support - WAIT for confirmation'
            })
        
        # Gravestone Doji
        if body_ratio < 0.1 and upper_shadow > body * 3 and lower_shadow < body:
            patterns.append({
                'pattern': 'Gravestone Doji',
                'type': 'bearish',
                'strength': 70,
                'confidence': 75,
                'category': 'reversal',
                'description': 'Potential bearish reversal at resistance - WAIT for confirmation'
            })
        
        return patterns
    
    def detect_spinning_top(self, df: pd.DataFrame) -> List[Dict]:
        """Detect Spinning Top pattern"""
        patterns = []
        if len(df) < 1:
            return patterns
        
        last = df.iloc[-1]
        body = abs(last['close'] - last['open'])
        range_size = last['high'] - last['low']
        upper_shadow = last['high'] - max(last['close'], last['open'])
        lower_shadow = min(last['close'], last['open']) - last['low']
        
        if range_size == 0:
            return patterns
        
        # Spinning Top: Small body, similar upper/lower shadows
        if (0.1 < body / range_size < 0.3 and 
            0.5 < upper_shadow / body < 2 and 
            0.5 < lower_shadow / body < 2):
            patterns.append({
                'pattern': 'Spinning Top',
                'type': 'neutral',
                'strength': 55,
                'confidence': 65,
                'category': 'indecision',
                'description': 'Weak momentum indecision - NO TRADE signal'
            })
        
        return patterns
    
    def detect_hammer(self, df: pd.DataFrame) -> List[Dict]:
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
        
        # Hammer
        if lower_shadow >= 2 * body and upper_shadow <= body * 0.3:
            strength = min(85, (lower_shadow / range_size) * 100)
            patterns.append({
                'pattern': 'Hammer',
                'type': 'bullish',
                'strength': round(strength, 2),
                'confidence': round(strength * 0.85, 2),
                'category': 'reversal',
                'description': 'Bullish reversal at support - TRADEABLE'
            })
        
        # Inverted Hammer
        elif upper_shadow >= 2 * body and lower_shadow <= body * 0.3:
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
    
    def detect_hanging_man(self, df: pd.DataFrame) -> List[Dict]:
        """Detect Hanging Man pattern"""
        patterns = []
        if len(df) < 2:
            return patterns
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        body = abs(last['close'] - last['open'])
        lower_shadow = min(last['close'], last['open']) - last['low']
        upper_shadow = last['high'] - max(last['close'], last['open'])
        range_size = last['high'] - last['low']
        
        # Hanging Man: After uptrend, small body at top, long lower shadow
        if (prev['close'] > prev['open'] and  # Uptrend
            lower_shadow >= 2 * body and upper_shadow <= body * 0.3 and
            last['close'] < last['open']):  # Bearish close
            strength = min(80, (lower_shadow / range_size) * 95)
            patterns.append({
                'pattern': 'Hanging Man',
                'type': 'bearish',
                'strength': round(strength, 2),
                'confidence': 75,
                'category': 'reversal',
                'description': 'Bearish reversal after uptrend - TRADEABLE'
            })
        
        return patterns
    
    def detect_shooting_star(self, df: pd.DataFrame) -> List[Dict]:
        """Detect Shooting Star"""
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
        
        if upper_shadow >= 2 * body and lower_shadow <= body * 0.3 and last['close'] < last['open']:
            strength = min(85, (upper_shadow / range_size) * 100)
            patterns.append({
                'pattern': 'Shooting Star',
                'type': 'bearish',
                'strength': round(strength, 2),
                'confidence': round(strength * 0.85, 2),
                'category': 'reversal',
                'description': 'Bearish reversal at resistance - TRADEABLE'
            })
        
        return patterns
    
    def detect_morning_evening_star(self, df: pd.DataFrame) -> List[Dict]:
        """Detect Morning Star, Evening Star, and Doji Star variants"""
        patterns = []
        if len(df) < 3:
            return patterns
        
        c1 = df.iloc[-3]
        c2 = df.iloc[-2]
        c3 = df.iloc[-1]
        
        c2_body = abs(c2['close'] - c2['open'])
        c2_is_doji = c2_body / (c2['high'] - c2['low']) < 0.1 if (c2['high'] - c2['low']) > 0 else False
        
        # Morning Star
        if (c1['close'] < c1['open'] and
            abs(c2['close'] - c2['open']) < (c1['high'] - c1['low']) * 0.3 and
            c3['close'] > c3['open'] and
            c3['close'] > (c1['open'] + c1['close']) / 2):
            
            if c2_is_doji:
                patterns.append({
                    'pattern': 'Morning Doji Star',
                    'type': 'bullish',
                    'strength': 75,
                    'confidence': 75,
                    'category': 'reversal',
                    'description': '3-candle bullish - TOO SLOW for intraday'
                })
            else:
                patterns.append({
                    'pattern': 'Morning Star',
                    'type': 'bullish',
                    'strength': 88,
                    'confidence': 85,
                    'category': 'reversal',
                    'description': '3-candle bullish - TOO SLOW for intraday'
                })
        
        # Evening Star
        elif (c1['close'] > c1['open'] and
              abs(c2['close'] - c2['open']) < (c1['high'] - c1['low']) * 0.3 and
              c3['close'] < c3['open'] and
              c3['close'] < (c1['open'] + c1['close']) / 2):
            
            if c2_is_doji:
                patterns.append({
                    'pattern': 'Evening Doji Star',
                    'type': 'bearish',
                    'strength': 75,
                    'confidence': 75,
                    'category': 'reversal',
                    'description': '3-candle bearish - TOO SLOW for intraday'
                })
            else:
                patterns.append({
                    'pattern': 'Evening Star',
                    'type': 'bearish',
                    'strength': 88,
                    'confidence': 85,
                    'category': 'reversal',
                    'description': '3-candle bearish - TOO SLOW for intraday'
                })
        
        return patterns
    
    def detect_harami_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect Harami and Harami Cross patterns"""
        patterns = []
        if len(df) < 2:
            return patterns
        
        prev = df.iloc[-2]
        last = df.iloc[-1]
        prev_body = abs(prev['close'] - prev['open'])
        last_body = abs(last['close'] - last['open'])
        
        # Check if last candle is inside previous candle's body
        if (last['open'] < max(prev['open'], prev['close']) and
            last['close'] > min(prev['open'], prev['close']) and
            last_body < prev_body * 0.5):
            
            # Harami Cross (Doji inside)
            if last_body / (last['high'] - last['low']) < 0.1 if (last['high'] - last['low']) > 0 else False:
                patterns.append({
                    'pattern': 'Harami Cross',
                    'type': 'neutral',
                    'strength': 60,
                    'confidence': 65,
                    'category': 'reversal',
                    'description': 'Weak indecision - NO TRADE'
                })
            # Bullish Harami
            elif prev['close'] < prev['open'] and last['close'] > last['open']:
                patterns.append({
                    'pattern': 'Bullish Harami',
                    'type': 'bullish',
                    'strength': 70,
                    'confidence': 70,
                    'category': 'reversal',
                    'description': 'Weak bullish signal - prefer stronger patterns'
                })
            # Bearish Harami
            elif prev['close'] > prev['open'] and last['close'] < last['open']:
                patterns.append({
                    'pattern': 'Bearish Harami',
                    'type': 'bearish',
                    'strength': 70,
                    'confidence': 70,
                    'category': 'reversal',
                    'description': 'Weak bearish signal - prefer stronger patterns'
                })
        
        return patterns
    
    def detect_three_soldiers_crows(self, df: pd.DataFrame) -> List[Dict]:
        """Detect Three White Soldiers & Three Black Crows"""
        patterns = []
        if len(df) < 3:
            return patterns
        
        c1 = df.iloc[-3]
        c2 = df.iloc[-2]
        c3 = df.iloc[-1]
        
        # Three White Soldiers
        if (c1['close'] > c1['open'] and c2['close'] > c2['open'] and c3['close'] > c3['open'] and
            c2['close'] > c1['close'] and c3['close'] > c2['close'] and
            c2['open'] > c1['open'] and c3['open'] > c2['open']):
            patterns.append({
                'pattern': 'Three White Soldiers',
                'type': 'bullish',
                'strength': 90,
                'confidence': 90,
                'category': 'continuation',
                'description': 'Strong bullish momentum - TRADEABLE'
            })
        
        # Three Black Crows
        elif (c1['close'] < c1['open'] and c2['close'] < c2['open'] and c3['close'] < c3['open'] and
              c2['close'] < c1['close'] and c3['close'] < c2['close'] and
              c2['open'] < c1['open'] and c3['open'] < c2['open']):
            patterns.append({
                'pattern': 'Three Black Crows',
                'type': 'bearish',
                'strength': 90,
                'confidence': 90,
                'category': 'continuation',
                'description': 'Strong bearish momentum - TRADEABLE'
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
    
    def filter_patterns_for_intraday_options(self, all_patterns: List[Dict]) -> Dict:
        """
        Enhanced filter for INTRADAY OPTIONS TRADING
        Separates patterns into TRADEABLE vs WARNING categories
        """
        tradeable_patterns = []
        warning_patterns = []
        
        for pattern in all_patterns:
            pattern_name = pattern.get('pattern', '')
            strength = pattern.get('strength', 0)
            
            # Check if tradeable
            if pattern_name in self.INTRADAY_TRADEABLE_PATTERNS:
                if strength >= 75:  # Must have high strength
                    tradeable_patterns.append(pattern)
                else:
                    warning_patterns.append({
                        **pattern,
                        'warning_reason': f'Pattern detected but strength too low ({strength}%)',
                        'warning_action': 'Wait for stronger setup (≥75%)',
                        'severity': 'LOW'
                    })
            
            # Check if warning pattern
            elif pattern_name in self.INTRADAY_WARNING_PATTERNS:
                warning_info = self.INTRADAY_WARNING_PATTERNS[pattern_name]
                warning_patterns.append({
                    **pattern,
                    'warning_reason': warning_info['reason'],
                    'warning_action': warning_info['action'],
                    'severity': warning_info['severity']
                })
        
        # Sort tradeable by strength (highest first)
        tradeable_patterns.sort(key=lambda x: x.get('strength', 0), reverse=True)
        
        # Sort warnings by severity (HIGH first)
        severity_order = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        warning_patterns.sort(
            key=lambda x: severity_order.get(x.get('severity', 'LOW'), 0),
            reverse=True
        )
        
        return {
            'tradeable': tradeable_patterns,
            'warnings': warning_patterns,
            'has_tradeable': len(tradeable_patterns) > 0,
            'has_warnings': len(warning_patterns) > 0,
            'total_tradeable': len(tradeable_patterns),
            'total_warnings': len(warning_patterns)
        }
