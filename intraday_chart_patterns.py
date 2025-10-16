"""
Enhanced Intraday Chart Pattern Detection for Options Trading
Analyzes 5-minute chart for fast-forming patterns (10-20 candles max)

NOW INCLUDES:
- Tradeable patterns (fast, actionable)
- Warning patterns (too slow, unreliable, or conflicting)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

class IntradayChartPatternDetector:
    """
    Enhanced chart pattern detector for intraday options
    Separates TRADEABLE patterns from WARNING patterns
    """
    
    def __init__(self):
        self.min_candles = 10
        self.max_candles = 25  # Max pattern length for intraday
        
        # Define warning patterns with reasons
        self.WARNING_PATTERNS_INFO = {
            'Head and Shoulders': {
                'reason': 'Too slow (50+ candles = 4+ hours)',
                'action': 'Better for swing trading, not intraday',
                'severity': 'HIGH',
                'formation_time': '4-6 hours'
            },
            'Inverse Head and Shoulders': {
                'reason': 'Too slow (50+ candles = 4+ hours)',
                'action': 'Better for swing trading, not intraday',
                'severity': 'HIGH',
                'formation_time': '4-6 hours'
            },
            'Double Top': {
                'reason': 'Too slow (30-40 candles = 2.5-3.5 hours)',
                'action': 'Wait for faster reversal patterns',
                'severity': 'MEDIUM',
                'formation_time': '2.5-3.5 hours'
            },
            'Double Bottom': {
                'reason': 'Too slow (30-40 candles = 2.5-3.5 hours)',
                'action': 'Wait for faster reversal patterns',
                'severity': 'MEDIUM',
                'formation_time': '2.5-3.5 hours'
            },
            'Cup and Handle': {
                'reason': 'Very slow pattern (hours to days)',
                'action': 'Not suitable for intraday trading',
                'severity': 'HIGH',
                'formation_time': '4+ hours'
            },
            'Symmetrical Triangle': {
                'reason': 'Unreliable direction (can break either way)',
                'action': 'Prefer directional patterns (Ascending/Descending)',
                'severity': 'MEDIUM',
                'formation_time': '1-2 hours'
            },
            'Wedge (Incomplete)': {
                'reason': 'Pattern forming but not yet actionable',
                'action': 'Wait for clear breakout',
                'severity': 'LOW',
                'formation_time': 'Varies'
            }
        }
    
    def detect_all_patterns(self, df_5min: pd.DataFrame) -> Dict:
        """
        Enhanced: Detect all patterns and categorize as TRADEABLE or WARNING
        
        Returns:
            {
                'tradeable': [...],  # Fast, actionable patterns
                'warnings': [...],   # Slow or unreliable patterns
                'has_tradeable': bool,
                'has_warnings': bool
            }
        """
        
        if df_5min is None or len(df_5min) < self.min_candles:
            return {
                'tradeable': [],
                'warnings': [],
                'has_tradeable': False,
                'has_warnings': False
            }
        
        tradeable = []
        warnings = []
        
        # ===== DETECT TRADEABLE PATTERNS =====
        
        bull_flag = self._detect_bull_flag(df_5min)
        if bull_flag['detected']:
            tradeable.append(bull_flag)
        
        bear_flag = self._detect_bear_flag(df_5min)
        if bear_flag['detected']:
            tradeable.append(bear_flag)
        
        asc_triangle = self._detect_ascending_triangle(df_5min)
        if asc_triangle['detected']:
            tradeable.append(asc_triangle)
        
        desc_triangle = self._detect_descending_triangle(df_5min)
        if desc_triangle['detected']:
            tradeable.append(desc_triangle)
        
        rectangle = self._detect_rectangle_breakout(df_5min)
        if rectangle['detected']:
            tradeable.append(rectangle)
        
        # ===== DETECT WARNING PATTERNS =====
        
        head_shoulders = self._detect_head_and_shoulders(df_5min)
        if head_shoulders['detected']:
            warnings.append(head_shoulders)
        
        double_top = self._detect_double_top(df_5min)
        if double_top['detected']:
            warnings.append(double_top)
        
        double_bottom = self._detect_double_bottom(df_5min)
        if double_bottom['detected']:
            warnings.append(double_bottom)
        
        symmetrical = self._detect_symmetrical_triangle(df_5min)
        if symmetrical['detected']:
            warnings.append(symmetrical)
        
        # Sort tradeable by confidence
        tradeable.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        # Sort warnings by severity
        severity_order = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        warnings.sort(
            key=lambda x: severity_order.get(x.get('severity', 'LOW'), 0),
            reverse=True
        )
        
        return {
            'tradeable': tradeable,
            'warnings': warnings,
            'has_tradeable': len(tradeable) > 0,
            'has_warnings': len(warnings) > 0,
            'total_tradeable': len(tradeable),
            'total_warnings': len(warnings)
        }
    
    # ========== TRADEABLE PATTERNS (from Phase 2) ==========
    
    def _detect_bull_flag(self, df: pd.DataFrame) -> Dict:
        """Bull Flag: Sharp upward move + downward consolidation (10-15 candles)"""
        try:
            if len(df) < 15:
                return {'detected': False}
            
            recent = df.tail(20).copy()
            closes = recent['close'].values
            highs = recent['high'].values
            lows = recent['low'].values
            volumes = recent['volume'].values if 'volume' in recent.columns else None
            
            pole_start = closes[0]
            pole_end = closes[10]
            pole_gain_pct = ((pole_end - pole_start) / pole_start) * 100
            
            flag_data = closes[10:]
            flag_slope = np.polyfit(range(len(flag_data)), flag_data, 1)[0]
            flag_volatility = np.std(flag_data) / np.mean(flag_data)
            
            pole_strong = pole_gain_pct > 1.5
            flag_descending = -0.5 < flag_slope < 0
            flag_tight = flag_volatility < 0.015
            
            volume_decreasing = True
            if volumes is not None:
                pole_volume = np.mean(volumes[:10])
                flag_volume = np.mean(volumes[10:])
                volume_decreasing = flag_volume < pole_volume * 1.2
            
            current_price = closes[-1]
            flag_high = np.max(highs[10:])
            near_breakout = current_price >= flag_high * 0.995
            
            if pole_strong and flag_descending and flag_tight and near_breakout:
                pole_height = pole_end - pole_start
                breakout_level = flag_high
                target = breakout_level + pole_height
                
                confidence = 75
                if volume_decreasing:
                    confidence += 10
                if pole_gain_pct > 2.5:
                    confidence += 5
                
                return {
                    'detected': True,
                    'pattern': 'Bull Flag',
                    'type': 'bullish',
                    'category': 'continuation',
                    'confidence': min(confidence, 95),
                    'strength': 85,
                    'breakout_level': round(breakout_level, 2),
                    'target': round(target, 2),
                    'pole_gain': round(pole_gain_pct, 2),
                    'candles_formed': len(recent),
                    'description': f'Bullish continuation - {pole_gain_pct:.1f}% pole, tight flag',
                    'entry_condition': f'Enter on 5-min close above {breakout_level:.2f} with volume'
                }
            
            return {'detected': False}
            
        except Exception as e:
            return {'detected': False, 'error': str(e)}
    
    def _detect_bear_flag(self, df: pd.DataFrame) -> Dict:
        """Bear Flag: Sharp downward move + upward consolidation"""
        try:
            if len(df) < 15:
                return {'detected': False}
            
            recent = df.tail(20).copy()
            closes = recent['close'].values
            highs = recent['high'].values
            lows = recent['low'].values
            
            pole_start = closes[0]
            pole_end = closes[10]
            pole_loss_pct = ((pole_start - pole_end) / pole_start) * 100
            
            flag_data = closes[10:]
            flag_slope = np.polyfit(range(len(flag_data)), flag_data, 1)[0]
            flag_volatility = np.std(flag_data) / np.mean(flag_data)
            
            pole_strong = pole_loss_pct > 1.5
            flag_ascending = 0 < flag_slope < 0.5
            flag_tight = flag_volatility < 0.015
            
            current_price = closes[-1]
            flag_low = np.min(lows[10:])
            near_breakdown = current_price <= flag_low * 1.005
            
            if pole_strong and flag_ascending and flag_tight and near_breakdown:
                pole_height = pole_start - pole_end
                breakdown_level = flag_low
                target = breakdown_level - pole_height
                
                return {
                    'detected': True,
                    'pattern': 'Bear Flag',
                    'type': 'bearish',
                    'category': 'continuation',
                    'confidence': 80,
                    'strength': 85,
                    'breakdown_level': round(breakdown_level, 2),
                    'target': round(target, 2),
                    'pole_loss': round(pole_loss_pct, 2),
                    'candles_formed': len(recent),
                    'description': f'Bearish continuation - {pole_loss_pct:.1f}% pole, tight flag',
                    'entry_condition': f'Enter on 5-min close below {breakdown_level:.2f} with volume'
                }
            
            return {'detected': False}
            
        except Exception as e:
            return {'detected': False, 'error': str(e)}
    
    def _detect_ascending_triangle(self, df: pd.DataFrame) -> Dict:
        """Ascending Triangle: Flat resistance + rising support (15-20 candles)"""
        try:
            if len(df) < 15:
                return {'detected': False}
            
            recent = df.tail(20).copy()
            highs = recent['high'].values
            lows = recent['low'].values
            closes = recent['close'].values
            
            resistance_zone = np.max(highs[-15:])
            resistance_touches = sum(1 for h in highs[-15:] if abs(h - resistance_zone) / resistance_zone < 0.003)
            
            support_lows = []
            for i in range(len(lows) - 5, len(lows)):
                if lows[i] == np.min(lows[max(0, i-3):i+3]):
                    support_lows.append((i, lows[i]))
            
            if len(support_lows) >= 2:
                support_rising = all(
                    support_lows[i][1] < support_lows[i+1][1] 
                    for i in range(len(support_lows)-1)
                )
            else:
                support_rising = False
            
            current_price = closes[-1]
            near_resistance = current_price >= resistance_zone * 0.997
            
            if resistance_touches >= 2 and support_rising and near_resistance:
                target = resistance_zone * 1.01
                
                return {
                    'detected': True,
                    'pattern': 'Ascending Triangle',
                    'type': 'bullish',
                    'category': 'breakout',
                    'confidence': 85,
                    'strength': 88,
                    'resistance_level': round(resistance_zone, 2),
                    'breakout_level': round(resistance_zone, 2),
                    'target': round(target, 2),
                    'candles_formed': len(recent),
                    'description': 'Bullish breakout setup - price coiling at resistance',
                    'entry_condition': f'Enter on 5-min close above {resistance_zone:.2f} with volume spike'
                }
            
            return {'detected': False}
            
        except Exception as e:
            return {'detected': False, 'error': str(e)}
    
    def _detect_descending_triangle(self, df: pd.DataFrame) -> Dict:
        """Descending Triangle: Flat support + declining resistance"""
        try:
            if len(df) < 15:
                return {'detected': False}
            
            recent = df.tail(20).copy()
            highs = recent['high'].values
            lows = recent['low'].values
            closes = recent['close'].values
            
            support_zone = np.min(lows[-15:])
            support_touches = sum(1 for l in lows[-15:] if abs(l - support_zone) / support_zone < 0.003)
            
            resistance_highs = []
            for i in range(len(highs) - 5, len(highs)):
                if highs[i] == np.max(highs[max(0, i-3):i+3]):
                    resistance_highs.append((i, highs[i]))
            
            if len(resistance_highs) >= 2:
                resistance_declining = all(
                    resistance_highs[i][1] > resistance_highs[i+1][1] 
                    for i in range(len(resistance_highs)-1)
                )
            else:
                resistance_declining = False
            
            current_price = closes[-1]
            near_support = current_price <= support_zone * 1.003
            
            if support_touches >= 2 and resistance_declining and near_support:
                target = support_zone * 0.99
                
                return {
                    'detected': True,
                    'pattern': 'Descending Triangle',
                    'type': 'bearish',
                    'category': 'breakout',
                    'confidence': 85,
                    'strength': 88,
                    'support_level': round(support_zone, 2),
                    'breakdown_level': round(support_zone, 2),
                    'target': round(target, 2),
                    'candles_formed': len(recent),
                    'description': 'Bearish breakdown setup - price testing support',
                    'entry_condition': f'Enter on 5-min close below {support_zone:.2f} with volume spike'
                }
            
            return {'detected': False}
            
        except Exception as e:
            return {'detected': False, 'error': str(e)}
    
    def _detect_rectangle_breakout(self, df: pd.DataFrame) -> Dict:
        """Rectangle: Price bouncing between parallel S/R"""
        try:
            if len(df) < 12:
                return {'detected': False}
            
            recent = df.tail(15).copy()
            highs = recent['high'].values
            lows = recent['low'].values
            closes = recent['close'].values
            
            resistance = np.max(highs)
            support = np.min(lows)
            
            range_pct = ((resistance - support) / support) * 100
            
            if not (0.5 < range_pct < 2.5):
                return {'detected': False}
            
            resistance_touches = sum(1 for h in highs if abs(h - resistance) / resistance < 0.003)
            support_touches = sum(1 for l in lows if abs(l - support) / support < 0.003)
            
            if resistance_touches < 2 or support_touches < 2:
                return {'detected': False}
            
            current_price = closes[-1]
            near_resistance_break = current_price >= resistance * 0.998
            near_support_break = current_price <= support * 1.002
            
            if near_resistance_break:
                return {
                    'detected': True,
                    'pattern': 'Rectangle Breakout',
                    'type': 'bullish',
                    'category': 'breakout',
                    'confidence': 75,
                    'strength': 80,
                    'resistance_level': round(resistance, 2),
                    'support_level': round(support, 2),
                    'breakout_level': round(resistance, 2),
                    'target': round(resistance + (resistance - support), 2),
                    'range_pct': round(range_pct, 2),
                    'description': f'Rectangle - {range_pct:.1f}% range, bullish breakout',
                    'entry_condition': f'Enter on close above {resistance:.2f}'
                }
            elif near_support_break:
                return {
                    'detected': True,
                    'pattern': 'Rectangle Breakdown',
                    'type': 'bearish',
                    'category': 'breakout',
                    'confidence': 75,
                    'strength': 80,
                    'resistance_level': round(resistance, 2),
                    'support_level': round(support, 2),
                    'breakdown_level': round(support, 2),
                    'target': round(support - (resistance - support), 2),
                    'range_pct': round(range_pct, 2),
                    'description': f'Rectangle - {range_pct:.1f}% range, bearish breakdown',
                    'entry_condition': f'Enter on close below {support:.2f}'
                }
            
            return {'detected': False}
            
        except Exception as e:
            return {'detected': False, 'error': str(e)}
    
    # ========== WARNING PATTERNS (Too Slow/Unreliable) ==========
    
    def _detect_head_and_shoulders(self, df: pd.DataFrame) -> Dict:
        """
        WARNING: Head & Shoulders (too slow - needs 50+ candles)
        Detection is basic - mainly to warn users
        """
        try:
            if len(df) < 40:
                return {'detected': False}
            
            recent = df.tail(60).copy()
            highs = recent['high'].values
            
            # Very basic H&S detection (simplified)
            if len(highs) >= 50:
                mid_point = len(highs) // 2
                head = np.max(highs[mid_point-10:mid_point+10])
                left_shoulder = np.max(highs[:mid_point-10])
                right_shoulder = np.max(highs[mid_point+10:])
                
                # If head is clearly higher than shoulders
                if head > left_shoulder * 1.02 and head > right_shoulder * 1.02:
                    pattern_info = self.WARNING_PATTERNS_INFO['Head and Shoulders']
                    
                    return {
                        'detected': True,
                        'pattern': 'Head and Shoulders',
                        'type': 'bearish',
                        'category': 'warning',
                        'warning_reason': pattern_info['reason'],
                        'warning_action': pattern_info['action'],
                        'severity': pattern_info['severity'],
                        'formation_time': pattern_info['formation_time'],
                        'candles_formed': len(recent),
                        'description': 'WARNING: Pattern too slow for intraday (4+ hours to form)'
                    }
            
            return {'detected': False}
            
        except Exception:
            return {'detected': False}
    
    def _detect_double_top(self, df: pd.DataFrame) -> Dict:
        """WARNING: Double Top (too slow - needs 30-40 candles)"""
        try:
            if len(df) < 25:
                return {'detected': False}
            
            recent = df.tail(40).copy()
            highs = recent['high'].values
            
            # Find two peaks that are roughly equal
            peaks = []
            for i in range(5, len(highs)-5):
                if highs[i] == np.max(highs[i-5:i+5]):
                    peaks.append((i, highs[i]))
            
            if len(peaks) >= 2:
                # Check if last two peaks are similar (within 0.5%)
                peak1 = peaks[-2][1]
                peak2 = peaks[-1][1]
                
                if abs(peak1 - peak2) / peak1 < 0.005:
                    pattern_info = self.WARNING_PATTERNS_INFO['Double Top']
                    
                    return {
                        'detected': True,
                        'pattern': 'Double Top',
                        'type': 'bearish',
                        'category': 'warning',
                        'warning_reason': pattern_info['reason'],
                        'warning_action': pattern_info['action'],
                        'severity': pattern_info['severity'],
                        'formation_time': pattern_info['formation_time'],
                        'candles_formed': len(recent),
                        'description': 'WARNING: Pattern too slow for intraday (2.5-3.5 hours)'
                    }
            
            return {'detected': False}
            
        except Exception:
            return {'detected': False}
    
    def _detect_double_bottom(self, df: pd.DataFrame) -> Dict:
        """WARNING: Double Bottom (too slow - needs 30-40 candles)"""
        try:
            if len(df) < 25:
                return {'detected': False}
            
            recent = df.tail(40).copy()
            lows = recent['low'].values
            
            # Find two troughs that are roughly equal
            troughs = []
            for i in range(5, len(lows)-5):
                if lows[i] == np.min(lows[i-5:i+5]):
                    troughs.append((i, lows[i]))
            
            if len(troughs) >= 2:
                trough1 = troughs[-2][1]
                trough2 = troughs[-1][1]
                
                if abs(trough1 - trough2) / trough1 < 0.005:
                    pattern_info = self.WARNING_PATTERNS_INFO['Double Bottom']
                    
                    return {
                        'detected': True,
                        'pattern': 'Double Bottom',
                        'type': 'bullish',
                        'category': 'warning',
                        'warning_reason': pattern_info['reason'],
                        'warning_action': pattern_info['action'],
                        'severity': pattern_info['severity'],
                        'formation_time': pattern_info['formation_time'],
                        'candles_formed': len(recent),
                        'description': 'WARNING: Pattern too slow for intraday (2.5-3.5 hours)'
                    }
            
            return {'detected': False}
            
        except Exception:
            return {'detected': False}
    
    def _detect_symmetrical_triangle(self, df: pd.DataFrame) -> Dict:
        """WARNING: Symmetrical Triangle (unreliable direction)"""
        try:
            if len(df) < 20:
                return {'detected': False}
            
            recent = df.tail(25).copy()
            highs = recent['high'].values
            lows = recent['low'].values
            
            # Check if highs are declining AND lows are rising (converging)
            high_slope = np.polyfit(range(len(highs)), highs, 1)[0]
            low_slope = np.polyfit(range(len(lows)), lows, 1)[0]
            
            # Symmetrical: highs declining, lows rising, converging
            if high_slope < -0.1 and low_slope > 0.1:
                # Check convergence
                range_start = highs[0] - lows[0]
                range_end = highs[-1] - lows[-1]
                
                if range_end < range_start * 0.7:  # Range tightening
                    pattern_info = self.WARNING_PATTERNS_INFO['Symmetrical Triangle']
                    
                    return {
                        'detected': True,
                        'pattern': 'Symmetrical Triangle',
                        'type': 'neutral',
                        'category': 'warning',
                        'warning_reason': pattern_info['reason'],
                        'warning_action': pattern_info['action'],
                        'severity': pattern_info['severity'],
                        'formation_time': pattern_info['formation_time'],
                        'candles_formed': len(recent),
                        'description': 'WARNING: Direction unreliable - can break either way'
                    }
            
            return {'detected': False}
            
        except Exception:
            return {'detected': False}
