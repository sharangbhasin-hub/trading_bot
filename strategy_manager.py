"""
Strategy Manager - Runs all strategies in parallel
"""
import pandas as pd
from typing import Dict, List
from strategies.strategy_ob_fvg import OrderBlockFVGStrategy
from strategies.strategy_liquidity_sweep import LiquiditySweepStrategy
from strategies.strategy_bos_retest import BOSRetestStrategy
from strategies.strategy_choch_ob import CHOCHOrderBlockStrategy

# Import other strategies as we implement them

class StrategyManager:
    """Manages all trading strategies"""
    
    def __init__(self):
        # Initialize all Tier 1 strategies
        self.tier1_strategies = [
            OrderBlockFVGStrategy(),
            LiquiditySweepStrategy(),
            BOSRetestStrategy(),
            CHOCHOrderBlockStrategy()
            # Add more as implemented
        ]
        
        self.tier2_strategies = []
        self.tier3_strategies = []
    
    def analyze_all(self,
                    df_5min: pd.DataFrame,
                    df_15min: pd.DataFrame,
                    df_1h: pd.DataFrame,
                    df_4h: pd.DataFrame,
                    spot_price: float,
                    support: float,
                    resistance: float,
                    overall_trend: str) -> Dict:
        """
        Run all strategies in parallel
        
        Returns:
        {
            'active_signals': [
                {
                    'strategy_name': str,
                    'signal': 'CALL' | 'PUT',
                    'confidence': 0-100,
                    'entry_price': float,
                    'stop_loss': float,
                    'target': float,
                    'reasoning': List[str]
                },
                ...
            ],
            'total_signals': int,
            'call_signals': int,
            'put_signals': int
        }
        """
        
        active_signals = []
        
        # Run all Tier 1 strategies
        for strategy in self.tier1_strategies:
            try:
                result = strategy.analyze(
                    df_5min=df_5min,
                    df_15min=df_15min,
                    df_1h=df_1h,
                    df_4h=df_4h,
                    spot_price=spot_price,
                    support=support,
                    resistance=resistance,
                    overall_trend=overall_trend
                )
                
                # Check if tradeable
                if strategy.is_tradeable(result):
                    active_signals.append({
                        'strategy_name': strategy.name,
                        'signal': result['signal'],
                        'confidence': result['confidence'],
                        'entry_price': result['entry_price'],
                        'stop_loss': result['stop_loss'],
                        'target': result['target'],
                        'reasoning': result['reasoning'],
                        'candlestick_pattern': result.get('candlestick_pattern')
                    })
            except Exception as e:
                print(f"Error in {strategy.name}: {str(e)}")
                continue
        
        # Count signals by direction
        call_signals = sum(1 for s in active_signals if s['signal'] == 'CALL')
        put_signals = sum(1 for s in active_signals if s['signal'] == 'PUT')
        
        return {
            'active_signals': active_signals,
            'total_signals': len(active_signals),
            'call_signals': call_signals,
            'put_signals': put_signals
        }
