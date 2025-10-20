"""
Strategy Manager - Runs all strategies in parallel with optional filtering
"""
import pandas as pd
from typing import Dict, List

# Tier 1 Strategies
from strategies.strategy_ob_fvg import OrderBlockFVGStrategy
from strategies.strategy_liquidity_sweep import LiquiditySweepStrategy
from strategies.strategy_bos_retest import BOSRetestStrategy
from strategies.strategy_choch_ob import CHOCHOrderBlockStrategy

# Tier 2 Strategies
from strategies.strategy_liq_grab_ob import LiquidityGrabOrderBlockStrategy
from strategies.strategy_bos_choch_liquidity import BOSCHOCHLiquidityStrategy
from strategies.strategy_ob_choch_combined import OBCHOCHCombinedStrategy
from strategies.strategy_fvg_double_bottom_top import FVGDoubleBottomTopStrategy

# Tier 3 (Optional)
from strategies.strategy_fake_breakout import FakeBreakoutStrategy

# Filter
from filters.multi_timeframe_filter import MultiTimeframeFilter

class StrategyManager:
    """Manages all trading strategies"""
    
    def __init__(self, use_mtf_filter: bool = False):
        """
        Initialize Strategy Manager
        
        Args:
            use_mtf_filter: If True, apply multi-timeframe filter before strategies
        """
        # Initialize all Tier 1 strategies (Core)
        self.tier1_strategies = [
            OrderBlockFVGStrategy(),
            LiquiditySweepStrategy(),
            BOSRetestStrategy(),
            CHOCHOrderBlockStrategy()
        ]
        
        # Initialize all Tier 2 strategies (Advanced)
        self.tier2_strategies = [
            LiquidityGrabOrderBlockStrategy(),
            BOSCHOCHLiquidityStrategy(),
            OBCHOCHCombinedStrategy(),
            FVGDoubleBottomTopStrategy()
        ]
        
        # Initialize Tier 3 strategies (Optional)
        self.tier3_strategies = [
            FakeBreakoutStrategy()
        ]
        
        # Multi-timeframe filter
        self.use_mtf_filter = use_mtf_filter
        self.mtf_filter = MultiTimeframeFilter()
    
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
                    'reasoning': List[str],
                    'candlestick_pattern': str | None,
                    'tier': 1 | 2 | 3
                },
                ...
            ],
            'total_signals': int,
            'call_signals': int,
            'put_signals': int,
            'tier1_signals': int,
            'tier2_signals': int,
            'tier3_signals': int,
            'filter_info': Dict (if filter enabled)
        }
        """
        
        result = {
            'active_signals': [],
            'total_signals': 0,
            'call_signals': 0,
            'put_signals': 0,
            'tier1_signals': 0,
            'tier2_signals': 0,
            'tier3_signals': 0,
            'filter_info': None
        }
        
        # Step 1: Apply multi-timeframe filter if enabled
        if self.use_mtf_filter:
            filter_result = self.mtf_filter.check(
                df_4h=df_4h,
                df_1h=df_1h,
                df_15min=df_15min,
                overall_trend=overall_trend
            )
            
            result['filter_info'] = filter_result
            
            if not filter_result['passed']:
                # Filter failed - return early with no signals
                return result
        
        active_signals = []
        
        # Step 2: Run all Tier 1 strategies
        for strategy in self.tier1_strategies:
            signal = self._run_strategy(
                strategy, 
                df_5min, df_15min, df_1h, df_4h,
                spot_price, support, resistance, overall_trend,
                tier=1
            )
            if signal:
                active_signals.append(signal)
        
        # Step 3: Run all Tier 2 strategies
        for strategy in self.tier2_strategies:
            signal = self._run_strategy(
                strategy,
                df_5min, df_15min, df_1h, df_4h,
                spot_price, support, resistance, overall_trend,
                tier=2
            )
            if signal:
                active_signals.append(signal)
        
        # Step 4: Run Tier 3 strategies (optional)
        for strategy in self.tier3_strategies:
            signal = self._run_strategy(
                strategy,
                df_5min, df_15min, df_1h, df_4h,
                spot_price, support, resistance, overall_trend,
                tier=3
            )
            if signal:
                active_signals.append(signal)
        
        # Sort by confidence (highest first)
        active_signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Count signals
        call_signals = sum(1 for s in active_signals if s['signal'] == 'CALL')
        put_signals = sum(1 for s in active_signals if s['signal'] == 'PUT')
        tier1_signals = sum(1 for s in active_signals if s['tier'] == 1)
        tier2_signals = sum(1 for s in active_signals if s['tier'] == 2)
        tier3_signals = sum(1 for s in active_signals if s['tier'] == 3)
        
        result['active_signals'] = active_signals
        result['total_signals'] = len(active_signals)
        result['call_signals'] = call_signals
        result['put_signals'] = put_signals
        result['tier1_signals'] = tier1_signals
        result['tier2_signals'] = tier2_signals
        result['tier3_signals'] = tier3_signals
        
        return result
    
    def _run_strategy(self, strategy, df_5min, df_15min, df_1h, df_4h,
                      spot_price, support, resistance, overall_trend, tier):
        """Helper to run a single strategy with error handling"""
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
                return {
                    'strategy_name': strategy.name,
                    'signal': result['signal'],
                    'confidence': result['confidence'],
                    'entry_price': result['entry_price'],
                    'stop_loss': result['stop_loss'],
                    'target': result['target'],
                    'reasoning': result['reasoning'],
                    'candlestick_pattern': result.get('candlestick_pattern'),
                    'tier': tier
                }
        except Exception as e:
            print(f"Error in {strategy.name}: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return None
