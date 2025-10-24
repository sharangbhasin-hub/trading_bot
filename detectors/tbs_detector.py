"""
TBS (Turtle Body Soup) and Model #1 Detector for CRT-TBS Strategy
====================================================================

Detects TBS patterns and Model #1 confirmation on Lower Timeframes (LTF).
Based on CRT-TBS trading strategy documentation.

TBS Definition:
- First candle that manipulates old high/low with body close
- Used to identify false breakouts
- Validates 2-6 candle manipulation patterns (A+ TBS)

Model #1 Definition:
- Thick candle (orderblock) that forms after TBS
- Direction must be OPPOSITE to trade direction
- Entry triggered when price closes beyond Model #1

Author: Trading System
Date: October 23, 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


class TBSDetector:
    """
    Detects TBS (Turtle Body Soup) patterns and Model #1 confirmations on LTF.
    
    Core Strategy Components:
    1. TBS Pattern: Identifies manipulation candles
    2. Model #1: Confirms entry setup
    3. Entry Trigger: Validates breakout beyond Model #1
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize TBS Detector.
        
        Args:
            config: Configuration dictionary with:
                   - allow_multi_candle: Boolean (default True)
                   - valid_candle_range: [min, max] for A+ TBS (default [2, 6])
                   - model1_method: 'body_vs_wicks' or 'ratio' (default 'body_vs_wicks')
                   - model1_min_body_ratio: If using ratio method (default 0.60)
                   - max_wait_candles: Maximum candles to wait for Model #1 (default 10)
        """
        self.config = config or {}
        
        # TBS Configuration
        self.allow_multi_candle = self.config.get('allow_multi_candle', True)
        self.valid_candle_range = self.config.get('valid_candle_range', [2, 6])
        
        # Model #1 Configuration
        self.model1_method = self.config.get('model1_method', 'body_vs_wicks')
        self.model1_min_body_ratio = self.config.get('model1_min_body_ratio', 0.60)
        self.max_wait_candles = self.config.get('max_wait_candles', 10)
    
    def detect_tbs_pattern(
        self,
        df: pd.DataFrame,
        reference_level: float,
        direction: str,
        start_idx: Optional[int] = None
    ) -> Optional[Dict]:
        """
        Detect TBS (Turtle Body Soup) pattern on LTF.
        
        TBS Definition (per documentation):
        - First candle that manipulates the reference level with body close
        - For SELL: Candle high > reference_level AND close < reference_level
        - For BUY: Candle low < reference_level AND close > reference_level
        
        Args:
            df: LTF OHLC dataframe
            reference_level: Price level to be manipulated (from HTF)
            direction: 'sell' or 'buy'
            start_idx: Optional starting index to search from (default: 0)
        
        Returns:
            Dictionary with TBS info or None:
            {
                'tbs_index': int,
                'tbs_high': float,
                'tbs_low': float,
                'tbs_close': float,
                'manipulation_size': float,
                'is_a_plus_tbs': bool,
                'candles_in_pattern': int,
                'timestamp': datetime
            }
        """
        if start_idx is None:
            start_idx = 0
        
        df_search = df.iloc[start_idx:]
        
        if direction == 'sell':
            return self._detect_sell_tbs(df_search, reference_level, start_idx)
        elif direction == 'buy':
            return self._detect_buy_tbs(df_search, reference_level, start_idx)
        else:
            raise ValueError(f"Invalid direction: {direction}. Must be 'sell' or 'buy'.")
    
    def _detect_sell_tbs(
        self,
        df: pd.DataFrame,
        reference_high: float,
        offset: int
    ) -> Optional[Dict]:
        """
        Detect SELL TBS pattern.
        
        Pattern:
        - Candle sweeps above reference_high (manipulation)
        - Candle closes back below reference_high (body close)
        - Validates 2-6 candle patterns if multi-candle enabled
        """
        for i in range(len(df)):
            candle = df.iloc[i]
            
            # Check if candle manipulates high with body close
            if (candle['high'] > reference_high and 
                candle['close'] < reference_high):
                
                # Calculate manipulation size
                manipulation_size = candle['high'] - reference_high
                
                # ✅ FIX #4 OPTION B: Accept both A+ (2-6) and Single-Candle (1) TBS
                is_a_plus = False
                candles_in_pattern = 1
                
                if self.allow_multi_candle and i + 1 < len(df):
                    candles_in_pattern = self._count_manipulation_candles(
                        df.iloc[i:i+7],  # Max 6 candles after TBS
                        reference_high,
                        'sell'
                    )
                
                # Quality validation logic
                import logging
                logger = logging.getLogger(__name__)
                
                if candles_in_pattern == 1:
                    # ✅ Single-candle TBS is valid (Standard quality)
                    is_a_plus = False
                    logger.debug(f"SELL Single-Candle TBS at index {i}")
                    
                elif self.valid_candle_range[0] <= candles_in_pattern <= self.valid_candle_range[1]:
                    # ✅ A+ TBS pattern (2-6 candles)
                    is_a_plus = True
                    logger.debug(f"SELL A+ TBS at index {i}: {candles_in_pattern} candles")
                    
                else:
                    # ❌ REJECT: 7+ candle patterns (too messy/unreliable)
                    logger.info(
                        f"SELL TBS REJECTED at index {i}: {candles_in_pattern} candles "
                        f"(too many - max {self.valid_candle_range[1]} for A+)"
                    )
                    continue  # Skip and search for next TBS
                
                # ✅ If we reach here, pattern is valid (either A+ or single-candle)
                return {
                    'tbs_index': offset + i,
                    'tbs_high': candle['high'],
                    'tbs_low': candle['low'],
                    'tbs_close': candle['close'],
                    'tbs_open': candle['open'],
                    'manipulation_size': manipulation_size,
                    'reference_level': reference_high,
                    'direction': 'sell',
                    'is_a_plus_tbs': is_a_plus,
                    'candles_in_pattern': candles_in_pattern,
                    'timestamp': candle.name if hasattr(candle, 'name') else None
                }
        
        return None
    
    def _detect_buy_tbs(
        self,
        df: pd.DataFrame,
        reference_low: float,
        offset: int
    ) -> Optional[Dict]:
        """
        Detect BUY TBS pattern.
        
        Pattern:
        - Candle sweeps below reference_low (manipulation)
        - Candle closes back above reference_low (body close)
        - Validates 2-6 candle patterns if multi-candle enabled
        """
        for i in range(len(df)):
            candle = df.iloc[i]
            
            # Check if candle manipulates low with body close
            if (candle['low'] < reference_low and 
                candle['close'] > reference_low):
                
                # Calculate manipulation size
                manipulation_size = reference_low - candle['low']
                
                # ✅ FIX #4 OPTION B: Accept both A+ (2-6) and Single-Candle (1) TBS
                is_a_plus = False
                candles_in_pattern = 1
                
                if self.allow_multi_candle and i + 1 < len(df):
                    candles_in_pattern = self._count_manipulation_candles(
                        df.iloc[i:i+7],  # Max 6 candles after TBS
                        reference_low,
                        'buy'
                    )
                
                # Quality validation logic
                import logging
                logger = logging.getLogger(__name__)
                
                if candles_in_pattern == 1:
                    # ✅ Single-candle TBS is valid (Standard quality)
                    is_a_plus = False
                    logger.debug(f"BUY Single-Candle TBS at index {i}")
                    
                elif self.valid_candle_range[0] <= candles_in_pattern <= self.valid_candle_range[1]:
                    # ✅ A+ TBS pattern (2-6 candles)
                    is_a_plus = True
                    logger.debug(f"BUY A+ TBS at index {i}: {candles_in_pattern} candles")
                    
                else:
                    # ❌ REJECT: 7+ candle patterns (too messy/unreliable)
                    logger.info(
                        f"BUY TBS REJECTED at index {i}: {candles_in_pattern} candles "
                        f"(too many - max {self.valid_candle_range[1]} for A+)"
                    )
                    continue  # Skip and search for next TBS
                
                # ✅ If we reach here, pattern is valid (either A+ or single-candle)
                return {
                    'tbs_index': offset + i,
                    'tbs_high': candle['high'],
                    'tbs_low': candle['low'],
                    'tbs_close': candle['close'],
                    'tbs_open': candle['open'],
                    'manipulation_size': manipulation_size,
                    'reference_level': reference_low,
                    'direction': 'buy',
                    'is_a_plus_tbs': is_a_plus,
                    'candles_in_pattern': candles_in_pattern,
                    'timestamp': candle.name if hasattr(candle, 'name') else None
                }
        
        return None
    
    def _count_manipulation_candles(
        self,
        df: pd.DataFrame,
        reference_level: float,
        direction: str
    ) -> int:
        """
        Count consecutive candles that manipulate the reference level.
        Used for identifying A+ TBS patterns (2-6 candles).
        
        Args:
            df: Dataframe subset to check
            reference_level: Level being manipulated
            direction: 'sell' or 'buy'
        
        Returns:
            Number of candles in manipulation pattern
        """
        count = 0
        
        for i in range(len(df)):
            candle = df.iloc[i]
            
            if direction == 'sell':
                # Count candles that touch/exceed the reference high
                if candle['high'] >= reference_level:
                    count += 1
                else:
                    break
            else:  # buy
                # Count candles that touch/go below the reference low
                if candle['low'] <= reference_level:
                    count += 1
                else:
                    break
        
        return count
    
    def detect_model1(
        self,
        df: pd.DataFrame,
        tbs_info: Dict,
        direction: str
    ) -> Optional[Dict]:
        """
        Detect Model #1 after TBS pattern.
        
        Model #1 Definition (per documentation):
        - Thick candle (body > wicks)
        - Direction must be OPPOSITE to trade direction
        - Forms within max_wait_candles after TBS
        
        Critical Rule:
        - For SELL trade: Model #1 must be BULLISH candle
        - For BUY trade: Model #1 must be BEARISH candle
        
        Args:
            df: LTF dataframe starting from TBS candle
            tbs_info: TBS pattern dictionary from detect_tbs_pattern()
            direction: Trade direction ('sell' or 'buy')
        
        Returns:
            Dictionary with Model #1 info or None:
            {
                'model1_index': int,
                'model1_high': float,
                'model1_low': float,
                'model1_open': float,
                'model1_close': float,
                'body_size': float,
                'body_ratio': float,
                'candles_after_tbs': int,
                'timestamp': datetime
            }
        """
        tbs_idx = tbs_info['tbs_index']
        
        # Search window: TBS + 1 to TBS + max_wait_candles
        search_start = tbs_idx + 1
        search_end = min(tbs_idx + 1 + self.max_wait_candles, len(df))
        
        if search_start >= len(df):
            return None
        
        df_search = df.iloc[search_start:search_end]
        
        for i, (idx, candle) in enumerate(df_search.iterrows()):
            # Calculate candle components
            body_size = abs(candle['close'] - candle['open'])
            upper_wick = candle['high'] - max(candle['open'], candle['close'])
            lower_wick = min(candle['open'], candle['close']) - candle['low']
            total_range = candle['high'] - candle['low']
            
            if total_range == 0:
                continue
            
            body_ratio = body_size / total_range
            
            # Check if candle is "thick" (body > wicks)
            is_thick = False
            if self.model1_method == 'body_vs_wicks':
                is_thick = body_size > (upper_wick + lower_wick)
            else:  # ratio method
                is_thick = body_ratio >= self.model1_min_body_ratio
            
            if not is_thick:
                continue
            
            # Check direction requirement (OPPOSITE to trade)
            candle_direction = 'bullish' if candle['close'] > candle['open'] else 'bearish'
            
            if direction == 'sell' and candle_direction == 'bullish':
                # Valid Model #1 for sell trade
                return {
                    'model1_index': search_start + i,
                    'model1_high': candle['high'],
                    'model1_low': candle['low'],
                    'model1_open': candle['open'],
                    'model1_close': candle['close'],
                    'body_size': body_size,
                    'body_ratio': body_ratio,
                    'direction': candle_direction,
                    'candles_after_tbs': i + 1,
                    'timestamp': idx if isinstance(idx, pd.Timestamp) else None
                }
            
            elif direction == 'buy' and candle_direction == 'bearish':
                # Valid Model #1 for buy trade
                return {
                    'model1_index': search_start + i,
                    'model1_high': candle['high'],
                    'model1_low': candle['low'],
                    'model1_open': candle['open'],
                    'model1_close': candle['close'],
                    'body_size': body_size,
                    'body_ratio': body_ratio,
                    'direction': candle_direction,
                    'candles_after_tbs': i + 1,
                    'timestamp': idx if isinstance(idx, pd.Timestamp) else None
                }
        
        return None
    
    def check_entry_trigger(
        self,
        df: pd.DataFrame,
        model1_info: Dict,
        direction: str
    ) -> Optional[Dict]:
        """
        Check if entry trigger condition is met.
        
        Entry Trigger (per documentation):
        - For SELL: Current candle closes BELOW Model #1 low
        - For BUY: Current candle closes ABOVE Model #1 high
        
        Args:
            df: LTF dataframe
            model1_info: Model #1 dictionary from detect_model1()
            direction: Trade direction ('sell' or 'buy')
        
        Returns:
            Dictionary with entry info or None:
            {
                'entry_triggered': bool,
                'entry_candle_index': int,
                'entry_price': float (close price),
                'trigger_candle': Series,
                'timestamp': datetime
            }
        """
        model1_idx = model1_info['model1_index']
        
        # Check candles after Model #1
        if model1_idx + 1 >= len(df):
            return None
        
        df_after_model1 = df.iloc[model1_idx + 1:]
        
        for i, (idx, candle) in enumerate(df_after_model1.iterrows()):
            if direction == 'sell':
                # Entry: Close below Model #1 low
                if candle['close'] < model1_info['model1_low']:
                    return {
                        'entry_triggered': True,
                        'entry_candle_index': model1_idx + 1 + i,
                        'entry_price': candle['close'],
                        'entry_high': candle['high'],
                        'entry_low': candle['low'],
                        'entry_open': candle['open'],
                        'trigger_type': 'close_below_model1',
                        'timestamp': idx if isinstance(idx, pd.Timestamp) else None
                    }
            
            elif direction == 'buy':
                # Entry: Close above Model #1 high
                if candle['close'] > model1_info['model1_high']:
                    return {
                        'entry_triggered': True,
                        'entry_candle_index': model1_idx + 1 + i,
                        'entry_price': candle['close'],
                        'entry_high': candle['high'],
                        'entry_low': candle['low'],
                        'entry_open': candle['open'],
                        'trigger_type': 'close_above_model1',
                        'timestamp': idx if isinstance(idx, pd.Timestamp) else None
                    }
        
        return None
    
    def calculate_stop_loss(
        self,
        tbs_info: Dict,
        model1_info: Dict,
        direction: str,
        buffer: float = 0.0  # Keep parameter for compatibility but override
    ) -> float:
        """
        Calculate institutional stop loss.
        
        Institutional Method:
        - Use highest/lowest extreme (invalidation point)
        - Add MINIMAL buffer (0.05% only)
        - NO arbitrary large buffers
        
        Args:
            tbs_info: TBS pattern dictionary
            model1_info: Model #1 dictionary
            direction: Trade direction ('sell' or 'buy')
            buffer: IGNORED - Using institutional 0.05% buffer
        
        Returns:
            Stop loss price
        """
        if direction == 'sell':
            # SELL: Stop above highest point
            highest_point = max(
                tbs_info['tbs_high'],
                model1_info['model1_high']
            )
            
            # ✅ FIX #3: Institutional minimal buffer (0.05%)
            minimal_buffer = highest_point * 0.0005  # 0.05% only
            stop_loss = highest_point + minimal_buffer
            
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(
                f"SELL Stop Loss: Highest={highest_point:.2f} + "
                f"buffer={minimal_buffer:.2f} = {stop_loss:.2f}"
            )
            
        else:  # buy
            # BUY: Stop below lowest point
            lowest_point = min(
                tbs_info['tbs_low'],
                model1_info['model1_low']
            )
            
            # ✅ FIX #3: Institutional minimal buffer (0.05%)
            minimal_buffer = lowest_point * 0.0005  # 0.05% only
            stop_loss = lowest_point - minimal_buffer
            
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(
                f"BUY Stop Loss: Lowest={lowest_point:.2f} - "
                f"buffer={minimal_buffer:.2f} = {stop_loss:.2f}"
            )
        
        return stop_loss
    
    def validate_complete_setup(
        self,
        df: pd.DataFrame,
        reference_level: float,
        direction: str
    ) -> Optional[Dict]:
        """
        Complete validation: TBS → Model #1 → Entry Trigger.
        
        This is a convenience method that runs all detection steps
        and returns complete setup if all conditions are met.
        
        Args:
            df: LTF dataframe
            reference_level: HTF reference level
            direction: Trade direction
        
        Returns:
            Complete setup dictionary or None:
            {
                'tbs': Dict,
                'model1': Dict,
                'entry': Dict,
                'stop_loss': float,
                'is_complete': bool
            }
        """
        # Step 1: Detect TBS
        tbs = self.detect_tbs_pattern(df, reference_level, direction)
        if not tbs:
            return None
        
        # Step 2: Detect Model #1
        model1 = self.detect_model1(df, tbs, direction)
        if not model1:
            return None
        
        # Step 3: Check Entry Trigger
        entry = self.check_entry_trigger(df, model1, direction)
        if not entry:
            return None
        
        # Step 4: Calculate Stop Loss
        stop_loss = self.calculate_stop_loss(tbs, model1, direction)
        
        return {
            'tbs': tbs,
            'model1': model1,
            'entry': entry,
            'stop_loss': stop_loss,
            'direction': direction,
            'is_complete': True,
            'setup_quality': 'A+' if tbs['is_a_plus_tbs'] else 'Standard'
        }


# Example usage and testing
if __name__ == "__main__":
    import pandas as pd
    
    # Test with sample LTF data
    sample_ltf = pd.DataFrame({
        'open': [100.0, 101.5, 102.0, 101.0, 100.5, 99.5, 99.0, 98.5],
        'high': [101.0, 102.5, 103.0, 101.5, 101.0, 100.0, 99.5, 99.0],
        'low': [99.5, 101.0, 101.5, 100.0, 99.0, 98.5, 98.0, 97.5],
        'close': [101.0, 102.0, 101.5, 100.5, 99.5, 99.0, 98.5, 98.0]
    })
    
    # Test TBS detection for sell setup
    detector = TBSDetector()
    
    # Simulate HTF reference high at 102.0
    tbs_result = detector.detect_tbs_pattern(
        df=sample_ltf,
        reference_level=102.0,
        direction='sell'
    )
    
    print("TBS Detection Test:")
    print(f"TBS Found: {tbs_result is not None}")
    if tbs_result:
        print(f"TBS Index: {tbs_result['tbs_index']}")
        print(f"Is A+ TBS: {tbs_result['is_a_plus_tbs']}")
        print(f"Candles in Pattern: {tbs_result['candles_in_pattern']}")
        
        # Test Model #1 detection
        model1_result = detector.detect_model1(
            df=sample_ltf,
            tbs_info=tbs_result,
            direction='sell'
        )
        
        print(f"\nModel #1 Found: {model1_result is not None}")
        if model1_result:
            print(f"Model #1 Index: {model1_result['model1_index']}")
            print(f"Body Ratio: {model1_result['body_ratio']:.2%}")
            print(f"Candles After TBS: {model1_result['candles_after_tbs']}")
