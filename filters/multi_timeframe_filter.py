"""
Multi-Timeframe Filter
Ensures alignment across 4H, 1H, 15M timeframes
"""
import pandas as pd
from typing import Dict
from detectors.structure_detector import StructureDetector
from detectors.liquidity_detector import LiquidityDetector

class MultiTimeframeFilter:
    """Multi-Timeframe Alignment Filter"""
    
    def __init__(self):
        self.structure_detector = StructureDetector()
        self.liq_detector = LiquidityDetector()
        self.min_alignment_score = 60  # Minimum score to pass
    
    def check(self,
              df_4h: pd.DataFrame,
              df_1h: pd.DataFrame,
              df_15min: pd.DataFrame,
              overall_trend: str) -> Dict:
        """
        Check multi-timeframe alignment
        
        Returns:
        {
            'passed': bool,
            'alignment_score': 0-100,
            'direction': 'BULLISH' | 'BEARISH' | 'NEUTRAL',
            'reasoning': List[str]
        }
        """
        
        reasoning = []
        score = 0
        direction_votes = {'BULLISH': 0, 'BEARISH': 0, 'NEUTRAL': 0}
        
        # Step 1: Check 4H trend (highest weight = 40 points)
        trend_4h = self.structure_detector.detect_trend(df_4h)
        
        if trend_4h['trend'] == 'UPTREND':
            score += 40
            direction_votes['BULLISH'] += 40
            reasoning.append(f"4H: {trend_4h['structure_type']} Uptrend")
        elif trend_4h['trend'] == 'DOWNTREND':
            score += 40
            direction_votes['BEARISH'] += 40
            reasoning.append(f"4H: {trend_4h['structure_type']} Downtrend")
        else:
            direction_votes['NEUTRAL'] += 40
            reasoning.append("4H: Ranging/Choppy")
        
        # Step 2: Check 1H liquidity (weight = 30 points)
        liquidity_1h = self.liq_detector.find_liquidity_pools(df_1h)
        
        current_price_1h = df_1h['close'].iloc[-1]
        
        # Count liquidity pools above and below price
        high_pools_above = len([p for p in liquidity_1h['high_pools'] if p > current_price_1h])
        low_pools_below = len([p for p in liquidity_1h['low_pools'] if p < current_price_1h])
        
        if high_pools_above > low_pools_below:
            score += 30
            direction_votes['BULLISH'] += 30
            reasoning.append(f"1H: More liquidity above ({high_pools_above} pools)")
        elif low_pools_below > high_pools_above:
            score += 30
            direction_votes['BEARISH'] += 30
            reasoning.append(f"1H: More liquidity below ({low_pools_below} pools)")
        else:
            direction_votes['NEUTRAL'] += 30
            reasoning.append("1H: Balanced liquidity")
        
        # Step 3: Check 15M structure (weight = 30 points)
        bos_15m = self.structure_detector.detect_bos(df_15min)
        choch_15m = self.structure_detector.detect_choch(df_15min)
        
        if bos_15m:
            if bos_15m['type'] == 'BULLISH':
                score += 30
                direction_votes['BULLISH'] += 30
                reasoning.append("15M: Bullish BOS detected")
            else:
                score += 30
                direction_votes['BEARISH'] += 30
                reasoning.append("15M: Bearish BOS detected")
        elif choch_15m:
            if choch_15m['type'] == 'BULLISH':
                score += 30
                direction_votes['BULLISH'] += 30
                reasoning.append("15M: Bullish CHOCH detected")
            else:
                score += 30
                direction_votes['BEARISH'] += 30
                reasoning.append("15M: Bearish CHOCH detected")
        else:
            direction_votes['NEUTRAL'] += 30
            reasoning.append("15M: No clear structure break")
        
        # Determine final direction
        max_votes = max(direction_votes.values())
        final_direction = [k for k, v in direction_votes.items() if v == max_votes][0]
        
        # Check alignment with overall_trend
        if overall_trend in ['Bullish', 'Bearish']:
            trend_match = (
                (overall_trend == 'Bullish' and final_direction == 'BULLISH') or
                (overall_trend == 'Bearish' and final_direction == 'BEARISH')
            )
            if trend_match:
                reasoning.append("✅ Aligned with overall consensus")
            else:
                reasoning.append("⚠️ Conflicts with overall consensus")
        
        passed = score >= self.min_alignment_score
        
        return {
            'passed': passed,
            'alignment_score': score,
            'direction': final_direction,
            'reasoning': reasoning
        }
