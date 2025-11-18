"""
Strategy Manager - Runs all strategies in parallel with optional filtering
"""
import pandas as pd
from typing import Dict, List

# Tier 1 Strategies
# from strategies.strategy_crt_tbs import StrategyCRTTBS

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
# from strategies.strategy_fake_breakout import FakeBreakoutStrategy

from strategies.strategy_vwap_strangle_selling import VWAPStrangleSelling
from strategies.strategy_vwap_strangle_buying import VWAPStrangleBuying

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

        # NEW: Initialize Tier 0 strategies (VWAP - Highest Priority)
        self.vwap_strategies = [
            VWAPStrangleSelling(),
            VWAPStrangleBuying()
        ]        
        
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

        self.tier3_strategies = []  # ✅ Initialize as empty list
        # Initialize Tier 3 strategies (Optional)
        # self.tier3_strategies = [
        #    FakeBreakoutStrategy()
        # ]
        
        # Multi-timeframe filter
        self.use_mtf_filter = use_mtf_filter
        self.mtf_filter = MultiTimeframeFilter()
    
    def validate_data(self, df: pd.DataFrame, name: str) -> Dict:
        """
        Validate dataframe before use
        
        Args:
            df: Dataframe to validate
            name: Name of the dataframe (for error messages)
            
        Returns:
            {
                'valid': bool,
                'error': str | None
            }
        """
        required_cols = ['open', 'high', 'low', 'close']
        
        # Check if dataframe exists and has data
        if df is None:
            return {
                'valid': False,
                'error': f"{name}: DataFrame is None"
            }
        
        if len(df) < 5:
            return {
                'valid': False,
                'error': f"{name}: Insufficient data (need 5+ candles, got {len(df)})"
            }
        
        # Check for required columns
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return {
                'valid': False,
                'error': f"{name}: Missing columns: {missing}"
            }
        
        # Check for null values
        if df[required_cols].isnull().any().any():
            return {
                'valid': False,
                'error': f"{name}: Contains null values in price columns"
            }
        
        # Check for zero or negative prices
        if (df[required_cols] <= 0).any().any():
            return {
                'valid': False,
                'error': f"{name}: Contains zero or negative prices"
            }
        
        # Check for invalid data types
        for col in required_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                return {
                    'valid': False,
                    'error': f"{name}: Column '{col}' is not numeric"
                }
        
        return {
            'valid': True,
            'error': None
        }
    
    def analyze_all(self,
                    df_5min: pd.DataFrame,
                    df_15min: pd.DataFrame,
                    df_1h: pd.DataFrame,
                    df_4h: pd.DataFrame,
                    spot_price: float,
                    support: float,
                    resistance: float,
                    overall_trend: str,
                    current_timestamp=None,
                    enable_vwap: bool = False,  # NEW PARAMETER
                    vwap_strategy_preference: str = 'AUTO') -> Dict: 
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
                    'risk_reward_ratio': float,  # NEW
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
            'filter_info': Dict (if filter enabled),
            'validation_errors': List[str],  # NEW
            'data_valid': bool  # NEW
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
            'filter_info': None,
            'validation_errors': [],
            'data_valid': True
        }
        
        # ====== NEW: STEP 0 - VALIDATE ALL DATA ======
        validation_checks = [
            ('5-min data', df_5min),
            ('15-min data', df_15min),
            ('1-hour data', df_1h)
            #  ('4-hour data', df_4h)
        ]
        
        for name, df in validation_checks:
            validation = self.validate_data(df, name)
            if not validation['valid']:
                result['validation_errors'].append(validation['error'])
                result['data_valid'] = False
        
        # If any validation failed, return early
        if not result['data_valid']:
            return result
        # ====== END VALIDATION ======

        # ====== NEW: VWAP STRATEGIES (Tier 0 - Highest Priority) ======
        if enable_vwap:
            import logging
            logger = logging.getLogger(__name__)
            logger.info("🎯 Running VWAP strategies...")
            
            # Determine which VWAP strategy to run
            if vwap_strategy_preference == 'AUTO':
                # Auto-select based on market conditions
                try:
                    from vwap_market_classifier import VWAPMarketClassifier
                    from config_vwap_strangle import get_todays_index
                    
                    classifier = VWAPMarketClassifier()
                    symbol = get_todays_index()
                    market_class = classifier.classify_market(symbol)
                    
                    recommended = market_class.get('recommended_strategy')
                    logger.info(f"📊 Market Classification: {recommended} (confidence: {market_class.get('confidence')}%)")
                    
                    if recommended == 'SELLING':
                        vwap_signal = self._run_vwap_strategy(self.vwap_strategies[0], df_5min, df_15min)
                        if vwap_signal:
                            result['active_signals'].append(vwap_signal)
                            result['total_signals'] = 1
                            logger.info("✅ VWAP SELLING signal detected - returning immediately")
                            return result  # Return immediately with VWAP signal
                    
                    elif recommended == 'BUYING':
                        vwap_signal = self._run_vwap_strategy(self.vwap_strategies[1], df_5min, df_15min)
                        if vwap_signal:
                            result['active_signals'].append(vwap_signal)
                            result['total_signals'] = 1
                            logger.info("✅ VWAP BUYING signal detected - returning immediately")
                            return result
                    
                    else:
                        logger.info("⚠️ No VWAP strategy recommended for current market conditions")
                
                except Exception as e:
                    logger.error(f"VWAP market classification error: {e}")
            
            elif vwap_strategy_preference == 'SELLING':
                vwap_signal = self._run_vwap_strategy(self.vwap_strategies[0], df_5min, df_15min)
                if vwap_signal:
                    result['active_signals'].append(vwap_signal)
                    result['total_signals'] = 1
                    return result
            
            elif vwap_strategy_preference == 'BUYING':
                vwap_signal = self._run_vwap_strategy(self.vwap_strategies[1], df_5min, df_15min)
                if vwap_signal:
                    result['active_signals'].append(vwap_signal)
                    result['total_signals'] = 1
                    return result
            
            elif vwap_strategy_preference == 'BOTH':
                for vwap_strat in self.vwap_strategies:
                    vwap_signal = self._run_vwap_strategy(vwap_strat, df_5min, df_15min)
                    if vwap_signal:
                        result['active_signals'].append(vwap_signal)
                
                if result['active_signals']:
                    result['total_signals'] = len(result['active_signals'])
                    return result
        # ====== END VWAP STRATEGIES ======
                        
        # Step 1: Apply multi-timeframe filter if enabled
        if self.use_mtf_filter:
            try:
                filter_result = self.mtf_filter.check(
                    df_4h=df_4h,
                    df_1h=df_1h,
                    df_15min=df_15min,
                    overall_trend=overall_trend
                )
                
                result['filter_info'] = filter_result
                
                if not filter_result['passed']:
                    # Filter failed - return early with no signals
                    result['validation_errors'].append(
                        f"Multi-timeframe filter failed (alignment score: {filter_result['alignment_score']}/100)"
                    )
                    return result
            except Exception as e:
                result['validation_errors'].append(f"MTF Filter error: {str(e)}")
                # Continue without filter if it fails

        self.current_timestamp = current_timestamp                        
                        
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

        # ✅ NEW: Determine consensus direction from strategies
        if call_signals > put_signals:
            result['consensus_direction'] = 'BULLISH'
        elif put_signals > call_signals:
            result['consensus_direction'] = 'BEARISH'
        else:
            result['consensus_direction'] = 'NEUTRAL'
        
        # ✅ NEW: Get highest confidence signal
        result['highest_confidence'] = max([s['confidence'] for s in active_signals]) if active_signals else 0
                        
        return result
    
    def _run_strategy(self, strategy, df_5min, df_15min, df_1h, df_4h,
                      spot_price, support, resistance, overall_trend, tier):
        """Helper to run a single strategy with error handling"""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.debug(f"\n{'='*60}")
        logger.debug(f"Running: {strategy.name} (Tier {tier})")
        logger.debug(f"{'='*60}")

        # ✅ FIX 5: Pass replay_engine to strategy for ATR calculation
        if hasattr(self, 'replay_engine'):
            strategy.replay_engine = self.replay_engine
        
        # ========== MARKET REGIME FILTER & DATA VALIDATION ==========
        # Market Regime Filter
        if hasattr(strategy, 'check_market_regime'):
            strategy_type = self._get_strategy_type(strategy.name)
            should_trade, regime_reason = strategy.check_market_regime(
                df_15min, 
                len(df_15min)-1, 
                strategy_type
            )
            
            if not should_trade:
                logger.debug(f"❌ FILTERED BY MARKET REGIME: {regime_reason}")
                return None
            else:
                logger.debug(f"✅ Market regime suitable: {regime_reason}")
        
        # DataFrame Validation
        if hasattr(strategy, 'df_validator'):
            is_valid, errors = strategy.df_validator.validate_ohlc(
                df_15min, 
                strict=False, 
                min_rows=20
            )
            
            if not is_valid:
                error_msg = errors[0] if errors else 'Unknown validation error'
                logger.debug(f"❌ DATA VALIDATION FAILED: {error_msg}")
                return None
            else:
                logger.debug(f"✅ Data validation passed")
        # ========== END VALIDATION BLOCK ==========
                          
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
            
            logger.debug(f"Strategy returned: signal={result.get('signal')}, confidence={result.get('confidence')}")
            logger.debug(f"Setup detected: {result.get('setup_detected')}")
            logger.debug(f"Retest confirmed: {result.get('retest_confirmed')}")
            logger.debug(f"Reasoning: {result.get('reasoning')}")
            
            # Check if tradeable (✅ NOW PASSES TIMESTAMP)
            is_valid = strategy.is_tradeable(result, timestamp=self.current_timestamp)
            logger.debug(f"is_tradeable() result: {is_valid}")
            
            if is_valid:
                logger.info(f"✅ {strategy.name} GENERATED VALID SIGNAL!")
                return {
                    'strategy_name': strategy.name,
                    'signal': result['signal'],
                    'confidence': result['confidence'],
                    'entry_price': result['entry_price'],
                    'stop_loss': result['stop_loss'],
                    'target': result['target'],
                    'risk_reward_ratio': result.get('risk_reward_ratio', 0.0),
                    'reasoning': result['reasoning'],
                    'candlestick_pattern': result.get('candlestick_pattern'),
                    'setup_detected': result.get('setup_detected', False),
                    'retest_confirmed': result.get('retest_confirmed', False),
                    'tier': tier
                }
            else:
                logger.debug(f"❌ {strategy.name} signal rejected by is_tradeable()")
                logger.debug(f"   Likely reason: confidence too low, R:R invalid, or retest required")
                
        except Exception as e:
            logger.error(f"❌ ERROR in {strategy.name}: {str(e)}")
            import traceback
            traceback.print_exc()
        
        logger.info(f"{'='*60}\n")
        return None

    def _get_strategy_type(self, strategy_name: str) -> str:
        """Map strategy names to their market regime types"""
        
        # Trend-following strategies
        if any(keyword in strategy_name.lower() for keyword in ['bos', 'choch', 'retest']):
            return 'TREND_FOLLOWING'
        
        # Breakout strategies
        if any(keyword in strategy_name.lower() for keyword in ['fvg', 'double', 'breakout']):
            return 'BREAKOUT'
        
        # Mean-reversion strategies  
        if any(keyword in strategy_name.lower() for keyword in ['liquidity', 'grab', 'sweep']):
            return 'MEAN_REVERSION'
        
        # Default to trend-following
        return 'TREND_FOLLOWING'

    def _run_vwap_strategy(self, strategy, df_5min: pd.DataFrame, df_15min: pd.DataFrame):
        """
        Helper to run VWAP strategy with error handling.
        VWAP strategies use different interface than regular strategies.
        
        Args:
            strategy: VWAP strategy instance
            df_5min: 5-minute dataframe
            df_15min: 15-minute dataframe (for current_idx)
        
        Returns:
            Signal dict or None
        """
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Running VWAP Strategy: {strategy.name}")
        logger.info(f"{'='*60}")
        
        try:
            # VWAP strategies use detect() method with current_idx
            current_idx = len(df_15min) - 1
            
            result = strategy.detect(df_15min, current_idx)
            
            logger.info(f"VWAP result: signal={result.get('signal_type')}, confidence={result.get('confidence')}")
            logger.info(f"Setup detected: {result.get('setup_detected')}")
            
            # Check if signal was generated
            if result.get('setup_detected') and result.get('signal_type') in ['SELL', 'BUY']:
                logger.info(f"✅ {strategy.name} GENERATED VALID SIGNAL!")
                
                # Convert VWAP signal format to standard format
                return {
                    'strategy_name': strategy.name,
                    'signal': 'CALL' if result['signal_type'] == 'BUY' else 'PUT',
                    'confidence': result.get('confidence', 70),
                    'entry_price': result.get('entry_price', 0),
                    'stop_loss': result.get('stop_loss', 0),
                    'target': result.get('target', 0),
                    'risk_reward_ratio': result.get('risk_reward_ratio', 0),
                    'reasoning': [result.get('entry_reason', 'VWAP crossover')],
                    'candlestick_pattern': None,
                    'setup_detected': True,
                    'retest_confirmed': True,
                    'tier': 0  # VWAP is Tier 0 (highest priority)
                }
            else:
                logger.info(f"❌ {strategy.name} - no signal (reason: {result.get('reason', 'unknown')})")
                
        except Exception as e:
            logger.error(f"❌ ERROR in {strategy.name}: {str(e)}")
            import traceback
            traceback.print_exc()
        
        logger.info(f"{'='*60}\n")
        return None

