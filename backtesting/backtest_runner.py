"""
Backtest Runner
Main orchestrator for running backtests
"""
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path

from backtesting.config import BacktestConfig
from backtesting.data_loader import DataLoader
from backtesting.replay_engine import ReplayEngine
from backtesting.signal_recorder import SignalRecorder
from backtesting.trade_simulator import TradeSimulator
from backtesting.performance_analyzer import PerformanceAnalyzer
from backtesting.market_classifier import MarketClassifier
from backtesting.exit_analyzer import ExitAnalyzer
from backtesting.failure_analyzer import FailureAnalyzer
from backtesting.visualization import Visualizer
from backtesting.report_generator import ReportGenerator

# Import your existing strategy manager
import sys
sys.path.append('..')
from strategy_manager import StrategyManager
from account_risk_manager import AccountRiskManager

logger = logging.getLogger(__name__)


class BacktestRunner:
    """
    Main backtest orchestrator
    Runs complete backtest and generates all reports
    """
    
    def __init__(self, kite_handler, index='NIFTY', start_date=None, end_date=None):
        """
        Initialize backtest runner
        
        Args:
            kite_handler: KiteHandler instance
            index: 'NIFTY' or 'BANKNIFTY'
            start_date: datetime object (default: 2024-01-01)
            end_date: datetime object (default: 2024-12-31)
        """
        self.kite = kite_handler
        self.index = index
        self.config = BacktestConfig()
        
        # Use defaults if not provided
        self.start_date = start_date or self.config.BACKTEST_START_DATE
        self.end_date = end_date or self.config.BACKTEST_END_DATE
        
        # Initialize components
        self.data_loader = DataLoader(kite_handler)
        self.signal_recorder = SignalRecorder()
        self.trade_simulator = TradeSimulator()
        self.market_classifier = MarketClassifier()
        
        self.account_risk = AccountRiskManager(
            initial_capital=100000,  # ₹1 lakh starting capital
            mode='backtest'
        )
        logger.info(f"✅ Account Risk Manager initialized with ₹100,000 capital")
        
        # Results
        self.historical_data = None
        self.replay_engine = None
        
        # Output directory
        self.output_dir = Path(self.config.RESULTS_DIR) / f"{index}_{self.start_date.strftime('%Y%m%d')}_to_{self.end_date.strftime('%Y%m%d')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Backtest Runner initialized for {index} from {self.start_date} to {self.end_date}")
    
    def run_backtest(self, progress_callback=None):
        """
        Run complete backtest
        
        Args:
            progress_callback: Function to call with progress updates (optional)
        
        Returns:
            Dict with all results OR error dict
        """
        logger.info("=" * 80)
        logger.info("STARTING BACKTEST")
        logger.info("=" * 80)


        try:
            # ✅ VERIFY KITE IS CONNECTED
            if not hasattr(self.kite, 'connected') or not self.kite.connected:
                logger.error("Kite is not connected!")
                return {
                    'error': 'Kite Connect not initialized',
                    'details': {
                        'message': 'Please ensure Kite API is connected before running backtest',
                        'has_kite': hasattr(self, 'kite'),
                        'has_connected': hasattr(self.kite, 'connected') if hasattr(self, 'kite') else False
                    }
                }
            
            logger.info(f"✅ Kite Connected: {self.kite.user_profile.get('user_name', 'Unknown')}")

            # Step 1: Load data
            if progress_callback:
                progress_callback(5, "Loading historical data...")
            
            logger.info("Step 1: Loading historical data")
            self.historical_data = self.data_loader.fetch_historical_data(
                self.index,
                self.start_date,
                self.end_date
            )
            
            # Validate data
            validation = self.data_loader.validate_data(self.historical_data)
            if not validation['is_valid']:
                logger.error(f"Data validation failed: {validation['issues']}")
                return {
                    'error': 'Data validation failed',
                    'details': validation
                }
            
            # Check if we have dates
            if not self.historical_data.get('dates'):
                return {
                    'error': 'No data was fetched',
                    'details': {
                        'message': 'Historical data fetch returned empty results',
                        'index': self.index,
                        'date_range': f"{self.start_date} to {self.end_date}",
                        'validation': validation
                    }
                }
            
            logger.info(f"Loaded {len(self.historical_data['dates'])} trading days")
            
            # Step 2: Initialize replay engine
            if progress_callback:
                progress_callback(10, "Initializing replay engine...")
            
            logger.info("Step 2: Initializing replay engine")
            self.replay_engine = ReplayEngine(self.historical_data)
            
            # Step 3: Run day-by-day simulation
            trading_dates = self.replay_engine.get_trading_dates()
            total_days = len(trading_dates)
            
            logger.info(f"Step 3: Running simulation for {total_days} days")
            
            for day_idx, date_str in enumerate(trading_dates):
                # Update progress
                progress_pct = 10 + int((day_idx / total_days) * 75)
                if progress_callback:
                    progress_callback(progress_pct, f"Simulating {date_str}...")
                
                logger.info(f"Processing day {day_idx + 1}/{total_days}: {date_str}")
                
                # Classify market condition for this day
                day_data = self.historical_data['data'][date_str]
                self.market_classifier.classify_day(day_data, date_str)

                from datetime import datetime as dt
                trading_date = dt.strptime(date_str, '%Y-%m-%d').date()
                self.account_risk.new_trading_day(trading_date)
                
                # Iterate through timestamps
                for time_str in self.replay_engine.iterate_timestamps(date_str):
                    self.replay_engine.set_current_timestamp(date_str, time_str)
                    
                    # Get current timestamp
                    current_timestamp = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
                    
                    # Update open trades with current candle
                    current_candle = self.replay_engine.get_current_candle('5min')
                    if current_candle is not None:
                        # Track which trades were open before update
                        trades_before = len(self.trade_simulator.closed_trades)
                        
                        # Update trades (this may close some)
                        self.trade_simulator.update_trades(current_candle, current_timestamp)
                        
                        # ✅ FIX 4: Check if any trades closed and notify risk manager
                        trades_after = len(self.trade_simulator.closed_trades)
                        if trades_after > trades_before:
                            # New trades closed - notify risk manager
                            for trade in self.trade_simulator.closed_trades[trades_before:]:
                                is_win = trade.net_pnl > 0
                                self.account_risk.on_trade_closed(trade.net_pnl, is_win)
                                logger.info(f"✅ Risk manager notified: Trade #{trade.trade_id} closed with P&L={trade.net_pnl:.2f}")

                    # Check for new signals
                    signals = self._generate_signals(current_timestamp)
                    
                    # Open trades for signals
                    for signal in signals:
                        # Check if account risk allows this trade
                        can_trade, risk_reason = self.account_risk.can_take_trade(signal)
                        
                        if not can_trade:
                            logger.warning(f"🛑 Trade REJECTED by risk manager: {risk_reason}")
                            logger.warning(f"   Strategy: {signal.get('strategy_name')}")
                            logger.warning(f"   Signal: {signal.get('signal_type')}")
                            logger.warning(f"   Confidence: {signal.get('confidence')}")
                            continue  # Skip this trade
                        
                        # Risk check passed - open trade
                        logger.info(f"✅ Trade APPROVED by risk manager")
                        trade = self.trade_simulator.open_trade(signal, current_timestamp)
                        
                        # Notify risk manager
                        self.account_risk.on_trade_entered(signal)
                    
                    # Check if end of day - close all open positions
                    if self.replay_engine.is_eod_close_time(time_str):
                        spot_price = self.replay_engine.get_current_spot_price()
                        if spot_price:
                            # Track trades before EOD close
                            trades_before = len(self.trade_simulator.closed_trades)
                            
                            # Close all open trades
                            self.trade_simulator.close_all_open_trades(
                                spot_price,
                                current_timestamp,
                                reason='EOD'
                            )
                            
                            # ✅ FIX 4: Notify risk manager of EOD closes
                            trades_after = len(self.trade_simulator.closed_trades)
                            if trades_after > trades_before:
                                for trade in self.trade_simulator.closed_trades[trades_before:]:
                                    is_win = trade.net_pnl > 0
                                    self.account_risk.on_trade_closed(trade.net_pnl, is_win)
                                    logger.info(f"✅ EOD close notified to risk manager: P&L={trade.net_pnl:.2f}")

            
            # Step 4: Analyze results
            if progress_callback:
                progress_callback(85, "Analyzing results...")
            
            logger.info("Step 4: Analyzing results")
            results = self._analyze_results()
            
            # Step 5: Generate reports
            if progress_callback:
                progress_callback(95, "Generating reports...")
            
            logger.info("Step 5: Generating reports")
            self._generate_reports(results)
            
            if progress_callback:
                progress_callback(100, "Backtest complete!")
            
            logger.info("=" * 80)
            logger.info("BACKTEST COMPLETE")
            logger.info("=" * 80)
            
            return results
            
        except Exception as e:
            logger.error(f"Backtest error: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'error': str(e),
                'details': {
                    'traceback': traceback.format_exc(),
                    'index': self.index,
                    'date_range': f"{self.start_date} to {self.end_date}"
                }
            }

    
    def _generate_signals(self, current_timestamp):
        """
        Generate signals at current timestamp
        
        Args:
            current_timestamp: Current datetime
        
        Returns:
            List of signal dicts ready for trade simulator
        """
        # ✅ HEADER: Log each signal generation attempt
        logger.info(f"\n{'='*70}")
        logger.info(f"🔍 SIGNAL GENERATION START: {current_timestamp}")
        logger.info(f"{'='*70}")
    
        logger.info(f"🔍 DEBUG: About to call replay_engine with these timeframes:")
        logger.info(f"   '5min', '15min', '1h', 'daily'")
        
        # Get data up to current time
        df_5min = self.replay_engine.get_data_upto_timestamp('5min')
        df_15min = self.replay_engine.get_data_upto_timestamp('15min')
        df_1h = self.replay_engine.get_data_upto_timestamp('1h')
        df_daily = self.replay_engine.get_data_upto_timestamp('daily')
        
        # ✅ DEBUG: Log data availability
        logger.info(f"📊 Data availability:")
        logger.info(f"   5min:  {len(df_5min):4d} candles | Empty: {df_5min.empty}")
        logger.info(f"   15min: {len(df_15min):4d} candles | Empty: {df_15min.empty}")
        logger.info(f"   1h:    {len(df_1h):4d} candles | Empty: {df_1h.empty}")
        logger.info(f"   daily: {len(df_daily):4d} candles | Empty: {df_daily.empty}")
        
        # Check if we have enough data
        if df_5min.empty or df_15min.empty or len(df_15min) < 5:
            logger.warning(f"❌ Insufficient data - Skipping signal generation")
            logger.warning(f"   Reason: 5min empty={df_5min.empty}, 15min empty={df_15min.empty}, 15min count={len(df_15min)}")
            return []
        
        # Get support/resistance
        support, resistance = self.replay_engine.get_support_resistance(df_15min)
        logger.info(f"📈 Support: {support:.2f} | Resistance: {resistance:.2f}")
        
        # Get current spot price
        spot_price = self.replay_engine.get_current_spot_price()
        if not spot_price:
            logger.warning("❌ No spot price available - Skipping")
            return []
        
        logger.info(f"💰 Current Price: {spot_price:.2f}")
        
        # Initialize strategy manager (your existing code)
        strategy_manager = StrategyManager()

        # ✅ FIX 5: Pass replay engine to strategy manager for ATR
        strategy_manager.replay_engine = self.replay_engine
        
        # Run analysis
        try:
            logger.info(f"\n🎯 Calling StrategyManager.analyze_all()...")
            
            # ✅ Call strategy manager
            analysis_results = strategy_manager.analyze_all(
                df_5min=df_5min,       
                df_15min=df_15min,    
                df_1h=df_1h,          
                df_4h=df_daily,       
                spot_price=spot_price,    
                support=support,
                resistance=resistance,
                overall_trend="NEUTRAL",
                current_timestamp=current_timestamp  # For time filter
            )
            
            # ✅ DEBUG: Log what strategy manager returned
            logger.info(f"\n📋 Strategy Manager Results:")
            logger.info(f"   Type: {type(analysis_results)}")
            logger.info(f"   Keys: {list(analysis_results.keys()) if isinstance(analysis_results, dict) else 'N/A'}")
            logger.info(f"   total_signals: {analysis_results.get('total_signals', 'KEY NOT FOUND')}")
            
            # ✅ Get active signals list
            strategies_list = analysis_results.get('active_signals', [])
            logger.info(f"   active_signals count: {len(strategies_list)}")
            
            # ✅ FIX #2: Build signals list for trade simulator
            signals = []
            
            if analysis_results.get('total_signals', 0) > 0:
                logger.info(f"\n✅ Processing {len(strategies_list)} strategy results...")
                
                for idx, strategy_result in enumerate(strategies_list):
                    logger.info(f"\n   Strategy #{idx+1}: {strategy_result.get('strategy_name')}")
                    logger.info(f"      signal: {strategy_result.get('signal')}")
                    logger.info(f"      confidence: {strategy_result.get('confidence')}")
                    
                    # Get signal value
                    signal_value = strategy_result.get('signal')
                    
                    # ✅ Check if valid signal (CALL or PUT)
                    if signal_value and str(signal_value).upper() in ['CALL', 'PUT']:
                        logger.info(f"      ✅ VALID SIGNAL DETECTED: {signal_value}")
    
                        # ✅ FIX #2: Build complete signal dict for trade simulator
                        signal_dict = {
                            'timestamp': current_timestamp,
                            'signal_type': signal_value.upper(),  # 'CALL' or 'PUT'
                            'strategy_name': strategy_result.get('strategy_name', 'Unknown'),
                            'confidence': strategy_result.get('confidence', 0),
                            'entry_price': strategy_result.get('entry_price', spot_price),
                            'stop_loss': strategy_result.get('stop_loss', 0),
                            'target': strategy_result.get('target', 0),
                            'risk_reward_ratio': strategy_result.get('risk_reward_ratio', 0.0),
                            'reasoning': strategy_result.get('reasoning', []),
                            'tier': strategy_result.get('tier', 1),
                            'support': support,
                            'resistance': resistance,
                            'spot_price': spot_price
                        }
    
                        # Record signal for analytics
                        signal_id = self.signal_recorder.record_signal(
                            current_timestamp,
                            strategy_result
                        )
                        signal_dict['signal_id'] = signal_id
                        
                        # Add to signals list
                        signals.append(signal_dict)
                        
                        logger.info(f"      📝 Signal recorded with ID: {signal_id}")
                        logger.info(f"      Entry: {signal_dict['entry_price']:.2f}")
                        logger.info(f"      SL: {signal_dict['stop_loss']:.2f}")
                        logger.info(f"      Target: {signal_dict['target']:.2f}")
                    else:
                        logger.info(f"      ❌ INVALID SIGNAL: {signal_value}")
            else:
                logger.info(f"\n❌ No signals from strategy manager")
                logger.info(f"   total_signals: {analysis_results.get('total_signals', 0)}")
            
            logger.info(f"\n{'='*70}")
            logger.info(f"✅ Total Valid Signals Generated: {len(signals)}")
            logger.info(f"{'='*70}\n")
            
            return signals
            
        except Exception as e:
            logger.error(f"\n{'='*70}")
            logger.error(f"💥 ERROR in signal generation at {current_timestamp}")
            logger.error(f"   Error: {str(e)}")
            logger.error(f"{'='*70}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _analyze_results(self):
        """Analyze backtest results"""
        # Get DataFrames
        trades_df = self.trade_simulator.get_closed_trades_df()
        signals_df = self.signal_recorder.to_dataframe()
        
        # Initialize analyzers
        performance_analyzer = PerformanceAnalyzer(trades_df, signals_df)
        exit_analyzer = ExitAnalyzer(trades_df)
        failure_analyzer = FailureAnalyzer(trades_df, signals_df)
        
        # Calculate metrics
        metrics = performance_analyzer.calculate_all_metrics()
        validation = performance_analyzer.get_validation_status()
        
        # Additional analysis
        mfe_mae_analysis = exit_analyzer.analyze_mfe_mae()
        holding_analysis = exit_analyzer.analyze_holding_periods()
        failure_categories = failure_analyzer.categorize_failures()
        
        # Market condition analysis
        market_summary = self.market_classifier.get_classification_summary()
        condition_performance = self.market_classifier.analyze_performance_by_condition(trades_df)
        
        # Collect recommendations
        recommendations = []
        recommendations.extend(exit_analyzer.suggest_exit_improvements())
        recommendations.extend(failure_analyzer.get_failure_recommendations())
        
        results = {
            'metrics': metrics,
            'validation': validation,
            'trades_df': trades_df,
            'signals_df': signals_df,
            'mfe_mae_analysis': mfe_mae_analysis,
            'holding_analysis': holding_analysis,
            'failure_categories': failure_categories,
            'market_summary': market_summary,
            'condition_performance': condition_performance,
            'recommendations': recommendations,
            'test_period': f"{self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}"
        }
        
        # Add test period to metrics
        results['metrics']['test_period'] = results['test_period']

        # ✅ FIX 4: Add risk manager stats
        risk_status = self.account_risk.get_status()
        daily_summary = self.account_risk.get_daily_summary()
        
        results['risk_manager'] = {
            'status': risk_status,
            'daily_summary': daily_summary,
            'final_capital': risk_status['current_capital'],
            'total_return_pct': risk_status['total_return_pct'],
            'max_drawdown': risk_status['drawdown'],
            'days_paused': sum(1 for stats in daily_summary.values() if stats.get('was_paused', False))
        }
        
        # Log risk manager summary
        logger.info("=" * 80)
        logger.info("ACCOUNT RISK MANAGER SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Initial Capital: ₹{risk_status['initial_capital']:,.0f}")
        logger.info(f"Final Capital: ₹{risk_status['current_capital']:,.2f}")
        logger.info(f"Total Return: {risk_status['total_return_pct']:.2f}%")
        logger.info(f"Max Drawdown: ₹{risk_status['drawdown']:,.2f} ({risk_status['drawdown_pct']:.2f}%)")
        logger.info(f"Days Paused: {results['risk_manager']['days_paused']}")
        logger.info(f"Final Open Positions: {risk_status['open_positions']}")
        logger.info("=" * 80)
        
        return results
    
    def _generate_reports(self, results):
        """Generate all reports"""
        # Initialize generators
        visualizer = Visualizer(self.output_dir)
        report_generator = ReportGenerator(self.output_dir)
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        performance_analyzer = PerformanceAnalyzer(results['trades_df'], results['signals_df'])
        
        charts = visualizer.generate_all_charts(
            performance_analyzer,
            self.market_classifier,
            results['trades_df']
        )
        
        # Generate HTML report
        logger.info("Generating HTML report...")
        html_path = report_generator.generate_html_report(
            results['metrics'],
            results['validation'],
            charts,
            results['recommendations']
        )
        
        # Export CSVs
        logger.info("Exporting data files...")
        report_generator.export_trades_csv(results['trades_df'])
        report_generator.export_signals_csv(results['signals_df'])
        report_generator.export_metrics_json(results['metrics'])
        
        logger.info(f"All reports saved to: {self.output_dir}")
        
        return {
            'html_report': html_path,
            'charts': charts
        }
