"""
Strategy Manager - Runs all strategies in parallel with optional filtering
"""
import pandas as pd
from typing import Dict, List

import logging
logger = logging.getLogger(__name__)

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
    
    def __init__(self, use_mtf_filter: bool = False, kite=None):
        """
        Initialize Strategy Manager
        
        Args:
            use_mtf_filter: If True, apply multi-timeframe filter before strategies
        """
        self.kite = kite
        
        # NEW: Initialize Tier 0 strategies (VWAP - Highest Priority)
        self.vwap_strategies = [
            VWAPStrangleSelling(kite=self.kite),
            VWAPStrangleBuying(kite=self.kite)
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

        # Log initialization mode
        if self.kite and hasattr(self.kite, 'connected') and self.kite.connected:
            logger.info("✅ StrategyManager initialized with Kite (LIVE MODE)")
        elif self.kite:
            logger.info("✅ StrategyManager initialized with Kite (BACKTEST MODE)")
        else:
            logger.warning("⚠️ StrategyManager initialized WITHOUT Kite - VWAP strategies will NOT work!")
    
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

    def get_available_strategies(self) -> List[Dict]:
        """
        ✅ DYNAMIC STRATEGY DISCOVERY (No Hardcoding)
        
        Returns all available strategies by reading from the initialized tier lists.
        This ensures new strategies appear automatically when added to __init__().
        
        Returns:
            List of dicts with strategy metadata:
            [
                {
                    'name': 'VWAP Strangle Selling',           # User-friendly name
                    'class_name': 'VWAPStrangleSelling',       # Class name for routing
                    'tier': 0,                                  # Priority tier
                    'category': 'VWAP'                          # Category for UI grouping
                },
                ...
            ]
        """
        strategies = []
        
        # Tier 0: VWAP Strategies (Highest Priority)
        for strategy in self.vwap_strategies:
            strategies.append({
                'name': strategy.name,                    # e.g., "VWAP Strangle Selling"
                'class_name': strategy.__class__.__name__,  # e.g., "VWAPStrangleSelling"
                'tier': 0,
                'category': 'VWAP'
            })
        
        # Tier 1: Core SMC Strategies
        for strategy in self.tier1_strategies:
            strategies.append({
                'name': strategy.name,                    # e.g., "Order Block + FVG"
                'class_name': strategy.__class__.__name__,  # e.g., "OrderBlockFVGStrategy"
                'tier': 1,
                'category': 'Core SMC'
            })
        
        # Tier 2: Advanced SMC Strategies
        for strategy in self.tier2_strategies:
            strategies.append({
                'name': strategy.name,                    # e.g., "FVG + Double Bottom/Top"
                'class_name': strategy.__class__.__name__,  # e.g., "FVGDoubleBottomTopStrategy"
                'tier': 2,
                'category': 'Advanced SMC'
            })
        
        # Tier 3: Optional Strategies (if any)
        for strategy in self.tier3_strategies:
            strategies.append({
                'name': strategy.name,
                'class_name': strategy.__class__.__name__,
                'tier': 3,
                'category': 'Optional'
            })
        
        logger.info(f"✅ Discovered {len(strategies)} strategies dynamically")
        
        return strategies

    def analyze_single(self,
                       strategy_name: str,
                       df_5min: pd.DataFrame,
                       df_15min: pd.DataFrame,
                       df_1h: pd.DataFrame,
                       df_4h: pd.DataFrame,
                       spot_price: float,
                       support: float,
                       resistance: float,
                       overall_trend: str,
                       current_timestamp=None) -> Dict:
        """
        ✅ NEW METHOD: Run SINGLE strategy (for backtesting individual strategies)
        
        This method is ONLY used for backtesting to test strategies individually.
        Live trading continues using analyze_all() which runs all strategies in parallel.
        
        Args:
            strategy_name: User-friendly name (e.g., "Order Block + FVG") 
                          OR class name (e.g., "OrderBlockFVGStrategy")
            ... (all other parameters same as analyze_all)
        
        Returns:
            Same format as analyze_all(), but with only 1 strategy's signal (or empty)
        """
        
        result = {
            'active_signals': [],
            'total_signals': 0,
            'call_signals': 0,
            'put_signals': 0,
            'tier1_signals': 0,
            'tier2_signals': 0,
            'tier3_signals': 0,
            'vwap_signals': 0,
            'filter_info': None,
            'validation_errors': [],
            'data_valid': True,
            'consensus_direction': 'NEUTRAL',
            'highest_confidence': 0
        }
        
        # ====== DATA VALIDATION (same as analyze_all) ======
        validation_checks = [
            ('5-min data', df_5min),
            ('15-min data', df_15min),
            ('1-hour data', df_1h)
        ]
        
        for name, df in validation_checks:
            validation = self.validate_data(df, name)
            if not validation['valid']:
                result['validation_errors'].append(validation['error'])
                result['data_valid'] = False
        
        if not result['data_valid']:
            return result
        
        # ====== MULTI-TIMEFRAME FILTER (optional, same as analyze_all) ======
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
                    result['validation_errors'].append(
                        f"Multi-timeframe filter failed (alignment score: {filter_result['alignment_score']}/100)"
                    )
                    return result
            except Exception as e:
                result['validation_errors'].append(f"MTF Filter error: {str(e)}")
        
        self.current_timestamp = current_timestamp
        
        # ====== FIND AND RUN THE REQUESTED STRATEGY ======
        logger.info(f"🎯 Running SINGLE strategy: {strategy_name}")
        
        # Search in all tier lists for matching strategy
        all_strategies = (
            [(s, 0) for s in self.vwap_strategies] +       # Tier 0
            [(s, 1) for s in self.tier1_strategies] +       # Tier 1
            [(s, 2) for s in self.tier2_strategies] +       # Tier 2
            [(s, 3) for s in self.tier3_strategies]         # Tier 3
        )
        
        target_strategy = None
        target_tier = None
        
        for strategy, tier in all_strategies:
            # Match by either user-friendly name OR class name
            if (strategy.name == strategy_name or 
                strategy.__class__.__name__ == strategy_name):
                target_strategy = strategy
                target_tier = tier
                logger.info(f"✅ Found strategy: {strategy.name} (Tier {tier})")
                break
        
        if not target_strategy:
            error_msg = f"Strategy '{strategy_name}' not found"
            logger.error(f"❌ {error_msg}")
            result['validation_errors'].append(error_msg)
            return result
        
        # ====== RUN THE STRATEGY ======
        signal = None
        
        # Special handling for VWAP strategies (different interface)
        if target_tier == 0:  # VWAP
            logger.info("📊 Running VWAP strategy (special interface)")
            
            try:
                from vwap_market_classifier import VWAPMarketClassifier
                from config_vwap_strangle import get_todays_index
                
                classifier = VWAPMarketClassifier()
                symbol = get_todays_index()
                market_class = classifier.classify_market(symbol)
                
                signal = self._run_vwap_strategy(
                    target_strategy,
                    df_5min,
                    df_15min,
                    market_class
                )
            except Exception as e:
                logger.error(f"❌ VWAP strategy error: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        else:  # Regular SMC strategies
            logger.info("📊 Running regular SMC strategy")
            signal = self._run_strategy(
                target_strategy,
                df_5min, df_15min, df_1h, df_4h,
                spot_price, support, resistance, overall_trend,
                tier=target_tier
            )
        
        # ====== PROCESS RESULT ======
        if signal:
            result['active_signals'] = [signal]
            result['total_signals'] = 1
            
            if signal['signal'] == 'CALL':
                result['call_signals'] = 1
                result['consensus_direction'] = 'BULLISH'
            else:
                result['put_signals'] = 1
                result['consensus_direction'] = 'BEARISH'
            
            # Update tier counts
            if target_tier == 0:
                result['vwap_signals'] = 1
            elif target_tier == 1:
                result['tier1_signals'] = 1
            elif target_tier == 2:
                result['tier2_signals'] = 1
            elif target_tier == 3:
                result['tier3_signals'] = 1
            
            result['highest_confidence'] = signal['confidence']
            
            logger.info(f"✅ Strategy generated signal: {signal['signal']} (confidence: {signal['confidence']}%)")
        else:
            logger.info(f"❌ Strategy did not generate a tradeable signal")
        
        return result
    
    def analyze_all(self,
                    df_5min: pd.DataFrame,
                    df_15min: pd.DataFrame,
                    df_1h: pd.DataFrame,
                    df_4h: pd.DataFrame,
                    spot_price: float,
                    support: float,
                    resistance: float,
                    overall_trend: str,
                    current_timestamp=None) -> Dict: 
        """
        Run all strategies in parallel - NEVER stops if one fails
        
        Returns:
        {
            'active_signals': [...],
            'total_signals': int,
            'call_signals': int,
            'put_signals': int,
            'tier1_signals': int,
            'tier2_signals': int,
            'tier3_signals': int,
            'vwap_signals': int,
            'filter_info': Dict,
            'validation_errors': List[str],
            'strategy_errors': List[str],  # NEW: Track failed strategies
            'data_valid': bool,
            'consensus_direction': str,
            'highest_confidence': int
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
            'vwap_signals': 0,
            'filter_info': None,
            'validation_errors': [],
            'strategy_errors': [],  # NEW: Track which strategies failed
            'data_valid': True
        }
        
        # ====== STEP 0: VALIDATE ALL DATA ======
        validation_checks = [
            ('5-min data', df_5min),
            ('15-min data', df_15min),
            ('1-hour data', df_1h)
        ]
        
        for name, df in validation_checks:
            validation = self.validate_data(df, name)
            if not validation['valid']:
                result['validation_errors'].append(validation['error'])
                result['data_valid'] = False
        
        if not result['data_valid']:
            return result
        
        # ====== STEP 1: MULTI-TIMEFRAME FILTER ======
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
                    result['validation_errors'].append(
                        f"Multi-timeframe filter failed (alignment score: {filter_result['alignment_score']}/100)"
                    )
                    return result
            except Exception as e:
                result['validation_errors'].append(f"MTF Filter error: {str(e)}")
    
        self.current_timestamp = current_timestamp
        active_signals = []
        
        # ====== TIER 0: VWAP STRATEGIES ======
        logger.info("="*80)
        logger.info("🎯 TIER 0: VWAP STRATEGIES")
        logger.info("="*80)
        
        for idx, strategy in enumerate(self.vwap_strategies, 1):
            try:
                logger.info(f"\n[{idx}/{len(self.vwap_strategies)}] Testing: {strategy.name}")
                
                from vwap_market_classifier import VWAPMarketClassifier
                from config_vwap_strangle import get_todays_index
                
                classifier = VWAPMarketClassifier()
                symbol = get_todays_index()
                market_class = classifier.classify_market(symbol)
                
                signal = self._run_vwap_strategy(strategy, df_5min, df_15min, market_class)
                
                if signal:
                    active_signals.append(signal)
                    logger.info(f"✅ {strategy.name} - SIGNAL ADDED")
                else:
                    logger.info(f"⚠️ {strategy.name} - No signal")
                    
            except Exception as e:
                error_msg = f"VWAP '{strategy.name}' failed: {str(e)}"
                logger.error(f"❌ {error_msg}")
                result['strategy_errors'].append(error_msg)
                import traceback
                logger.error(traceback.format_exc())
                continue  # CONTINUE TO NEXT STRATEGY
        
        # ====== TIER 1: CORE SMC STRATEGIES ======
        logger.info("\n" + "="*80)
        logger.info("🎯 TIER 1: CORE SMC STRATEGIES")
        logger.info("="*80)
        
        for idx, strategy in enumerate(self.tier1_strategies, 1):
            try:
                logger.info(f"\n[{idx}/{len(self.tier1_strategies)}] Running: {strategy.name}")
                
                signal = self._run_strategy(
                    strategy, df_5min, df_15min, df_1h, df_4h,
                    spot_price, support, resistance, overall_trend, tier=1
                )
                
                if signal:
                    active_signals.append(signal)
                    logger.info(f"✅ {strategy.name} - SIGNAL ADDED")
                else:
                    logger.info(f"⚠️ {strategy.name} - No signal")
                    
            except Exception as e:
                error_msg = f"Tier 1 '{strategy.name}' failed: {str(e)}"
                logger.error(f"❌ {error_msg}")
                result['strategy_errors'].append(error_msg)
                import traceback
                logger.error(traceback.format_exc())
                continue  # CONTINUE TO NEXT STRATEGY
        
        # ====== TIER 2: ADVANCED SMC STRATEGIES ======
        logger.info("\n" + "="*80)
        logger.info("🎯 TIER 2: ADVANCED SMC STRATEGIES")
        logger.info("="*80)
        
        for idx, strategy in enumerate(self.tier2_strategies, 1):
            try:
                logger.info(f"\n[{idx}/{len(self.tier2_strategies)}] Running: {strategy.name}")
                
                signal = self._run_strategy(
                    strategy, df_5min, df_15min, df_1h, df_4h,
                    spot_price, support, resistance, overall_trend, tier=2
                )
                
                if signal:
                    active_signals.append(signal)
                    logger.info(f"✅ {strategy.name} - SIGNAL ADDED")
                else:
                    logger.info(f"⚠️ {strategy.name} - No signal")
                    
            except Exception as e:
                error_msg = f"Tier 2 '{strategy.name}' failed: {str(e)}"
                logger.error(f"❌ {error_msg}")
                result['strategy_errors'].append(error_msg)
                import traceback
                logger.error(traceback.format_exc())
                continue  # CONTINUE TO NEXT STRATEGY
        
        # ====== TIER 3: OPTIONAL STRATEGIES ======
        if self.tier3_strategies:
            logger.info("\n" + "="*80)
            logger.info("🎯 TIER 3: OPTIONAL STRATEGIES")
            logger.info("="*80)
            
            for idx, strategy in enumerate(self.tier3_strategies, 1):
                try:
                    logger.info(f"\n[{idx}/{len(self.tier3_strategies)}] Running: {strategy.name}")
                    
                    signal = self._run_strategy(
                        strategy, df_5min, df_15min, df_1h, df_4h,
                        spot_price, support, resistance, overall_trend, tier=3
                    )
                    
                    if signal:
                        active_signals.append(signal)
                        logger.info(f"✅ {strategy.name} - SIGNAL ADDED")
                    else:
                        logger.info(f"⚠️ {strategy.name} - No signal")
                        
                except Exception as e:
                    error_msg = f"Tier 3 '{strategy.name}' failed: {str(e)}"
                    logger.error(f"❌ {error_msg}")
                    result['strategy_errors'].append(error_msg)
                    import traceback
                    logger.error(traceback.format_exc())
                    continue  # CONTINUE TO NEXT STRATEGY
        
        # ====== FINAL SUMMARY ======
        logger.info("\n" + "="*80)
        logger.info("📊 STRATEGY EXECUTION SUMMARY")
        logger.info("="*80)
        logger.info(f"✅ Successful signals: {len(active_signals)}")
        logger.info(f"❌ Failed strategies: {len(result['strategy_errors'])}")
        
        if result['strategy_errors']:
            logger.warning("\n⚠️ Strategies that encountered errors:")
            for error in result['strategy_errors']:
                logger.warning(f"  - {error}")
        
        # Sort and count signals
        active_signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        result['active_signals'] = active_signals
        result['total_signals'] = len(active_signals)
        result['call_signals'] = sum(1 for s in active_signals if s['signal'] == 'CALL')
        result['put_signals'] = sum(1 for s in active_signals if s['signal'] == 'PUT')
        result['vwap_signals'] = sum(1 for s in active_signals if s.get('tier') == 0)
        result['tier1_signals'] = sum(1 for s in active_signals if s['tier'] == 1)
        result['tier2_signals'] = sum(1 for s in active_signals if s['tier'] == 2)
        result['tier3_signals'] = sum(1 for s in active_signals if s['tier'] == 3)
        
        # Consensus
        if result['call_signals'] > result['put_signals']:
            result['consensus_direction'] = 'BULLISH'
        elif result['put_signals'] > result['call_signals']:
            result['consensus_direction'] = 'BEARISH'
        else:
            result['consensus_direction'] = 'NEUTRAL'
        
        result['highest_confidence'] = max([s['confidence'] for s in active_signals]) if active_signals else 0
        
        logger.info(f"\n🎯 Final Consensus: {result['consensus_direction']}")
        logger.info(f"📊 Highest Confidence: {result['highest_confidence']}%")
        logger.info("="*80 + "\n")
        
        return result
    
    def _run_strategy(self, strategy, df_5min, df_15min, df_1h, df_4h,
                      spot_price, support, resistance, overall_trend, tier):
        """Helper to run a single strategy with BULLETPROOF error handling"""
        
        strategy_name = strategy.name if hasattr(strategy, 'name') else str(strategy)
        
        logger.info(f"\n{'─'*60}")
        logger.info(f"🔍 Strategy: {strategy_name}")
        logger.info(f"{'─'*60}")
        
        try:
            # Pass replay_engine if available (for backtesting)
            if hasattr(self, 'replay_engine'):
                strategy.replay_engine = self.replay_engine
            
            # Market regime filter (if strategy has it)
            if hasattr(strategy, 'check_market_regime'):
                strategy_type = self._get_strategy_type(strategy_name)
                should_trade, regime_reason = strategy.check_market_regime(
                    df_15min, len(df_15min)-1, strategy_type
                )
                
                if not should_trade:
                    logger.info(f"⚠️ Market regime unsuitable: {regime_reason}")
                    return None
                else:
                    logger.debug(f"✅ Market regime OK: {regime_reason}")
            
            # Data validation (if strategy has validator)
            if hasattr(strategy, 'df_validator'):
                is_valid, errors = strategy.df_validator.validate_ohlc(
                    df_15min, strict=False, min_rows=20
                )
                
                if not is_valid:
                    error_msg = errors[0] if errors else 'Unknown validation error'
                    logger.info(f"⚠️ Data validation failed: {error_msg}")
                    return None
            
            # Run strategy analysis
            logger.debug("📊 Calling strategy.analyze()...")
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
            
            # Log intermediate results
            logger.info(f"📊 Analysis returned:")
            logger.info(f"   Signal: {result.get('signal', 'NONE')}")
            logger.info(f"   Confidence: {result.get('confidence', 0)}%")
            logger.info(f"   Setup: {result.get('setup_detected', False)}")
            logger.info(f"   Retest: {result.get('retest_confirmed', False)}")
            
            # Check if tradeable
            is_valid = strategy.is_tradeable(result, timestamp=self.current_timestamp)
            
            if is_valid:
                logger.info(f"✅ TRADEABLE signal generated!")
                return {
                    'strategy_name': strategy_name,
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
                # Log why signal was rejected
                logger.info(f"⚠️ Signal NOT tradeable:")
                
                reasons = []
                if not result.get('setup_detected'):
                    reasons.append("Setup not detected")
                if not result.get('retest_confirmed'):
                    reasons.append("Retest not confirmed")
                if result.get('confidence', 0) < 70:
                    reasons.append(f"Low confidence ({result.get('confidence', 0)}%)")
                if result.get('risk_reward_ratio', 0) < 1.5:
                    reasons.append(f"Poor R:R ({result.get('risk_reward_ratio', 0):.2f})")
                
                for reason in reasons:
                    logger.info(f"   - {reason}")
                
                return None
        
        except Exception as e:
            logger.error(f"❌ EXCEPTION in {strategy_name}:")
            logger.error(f"   Error: {str(e)}")
            
            import traceback
            logger.error("   Traceback:")
            for line in traceback.format_exc().split('\n'):
                if line.strip():
                    logger.error(f"   {line}")
            
            # RETURN NONE - DON'T CRASH
            return None
        
        finally:
            logger.info(f"{'─'*60}\n")

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

    def _run_vwap_strategy(self, strategy, df_5min: pd.DataFrame, df_15min: pd.DataFrame, market_class: Dict = None):
        """
        Helper to run VWAP strategy with error handling.
        VWAP strategies use different interface than regular strategies.
        
        Args:
            strategy: VWAP strategy instance
            df_5min: 5-minute dataframe
            df_15min: 15-minute dataframe (for current_idx)
            market_class: Market classification result from VWAPMarketClassifier
        
        Returns:
            Signal dict or None
        """
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Running VWAP Strategy: {strategy.name}")
        logger.info(f"{'='*60}")
        
        # ✅ Log market classification info
        if market_class:
            logger.info(f"Market Classification:")
            logger.info(f"  VIX: {market_class.get('conditions', {}).get('india_vix', 'N/A')}")
            logger.info(f"  Gap: {market_class.get('conditions', {}).get('gap_open_pct', 0):.2f}%")
            logger.info(f"  Range-bound: {market_class.get('conditions', {}).get('is_range_bound', False)}")
            logger.info(f"  Breakout: {market_class.get('conditions', {}).get('is_breakout', False)}")
        
        try:
            # VWAP strategies use detect() method with current_idx
            current_idx = len(df_15min) - 1
            
            logger.info(f"Calling {strategy.name}.detect() with current_idx={current_idx}")
            
            result = strategy.detect(df_15min, current_idx)
            
            logger.info(f"VWAP result:")
            logger.info(f"  Signal type: {result.get('signal_type')}")
            logger.info(f"  Confidence: {result.get('confidence')}")
            logger.info(f"  Setup detected: {result.get('setup_detected')}")
            logger.info(f"  Reason: {result.get('reason', 'N/A')}")
            
            # Check if signal was generated
            if result.get('setup_detected') and result.get('signal_type') in ['SELL', 'BUY']:
                logger.info(f"✅ {strategy.name} GENERATED VALID SIGNAL!")
                
                # ✅ Calculate R:R ratio if not present
                entry = result.get('entry_price', 0)
                sl = result.get('stop_loss', 0)
                target = result.get('target', 0)
                
                if entry and sl and target:
                    risk = abs(entry - sl)
                    reward = abs(target - entry)
                    rr_ratio = reward / risk if risk > 0 else 0
                else:
                    rr_ratio = 0
                
                # Convert VWAP signal format to standard format
                signal = {
                    'strategy_name': strategy.name,
                    'signal': 'CALL' if result['signal_type'] == 'BUY' else 'PUT',
                    'confidence': result.get('confidence', 70),
                    'entry_price': entry,
                    'stop_loss': sl,
                    'target': target,
                    'risk_reward_ratio': result.get('risk_reward_ratio', rr_ratio),
                    'reasoning': [result.get('entry_reason', 'VWAP crossover')],
                    'candlestick_pattern': None,
                    'setup_detected': True,
                    'retest_confirmed': True,
                    'tier': 0  # VWAP is Tier 0 (highest priority)
                }
                
                logger.info(f"Signal details:")
                logger.info(f"  Entry: {entry}")
                logger.info(f"  SL: {sl}")
                logger.info(f"  Target: {target}")
                logger.info(f"  R:R: {rr_ratio:.2f}")
                
                return signal
            else:
                reason = result.get('reason', 'unknown')
                logger.info(f"❌ {strategy.name} - no signal")
                logger.info(f"   Reason: {reason}")
                return None
                
        except Exception as e:
            logger.error(f"❌ ERROR in {strategy.name}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        
        finally:
            logger.info(f"{'='*60}\n")
