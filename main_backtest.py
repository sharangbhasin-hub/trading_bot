"""
Backtesting Streamlit Application
User interface for running and analyzing backtests
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append('.')

from backtesting.backtest_runner import BacktestRunner
from backtesting.config import BacktestConfig, get_trading_days
from kite_handler import get_kite_handler
from config_crt_tbs import get_config
from unified_data_handler import get_unified_handler, get_all_market_types, get_market_display_info, UnifiedDataHandler

# Page config
st.set_page_config(
    page_title="Trading Strategy Backtester",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2C3E50;
        text-align: center;
        padding: 20px 0;
        border-bottom: 3px solid #3498db;
        margin-bottom: 30px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-box {
        background: #d4edda;
        border: 2px solid #c3e6cb;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .warning-box {
        background: #fff3cd;
        border: 2px solid #ffeaa7;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .error-box {
        background: #f8d7da;
        border: 2px solid #f5c6cb;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header">📊 Trading Strategy Backtester</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'backtest_complete' not in st.session_state:
        st.session_state.backtest_complete = False
    if 'backtest_results' not in st.session_state:
        st.session_state.backtest_results = None
    if 'strategy_choice' not in st.session_state:
        st.session_state.strategy_choice = 'All SMC Strategies'
    if 'trading_style' not in st.session_state:
        st.session_state.trading_style = None
    if 'timeframe_map' not in st.session_state:
        st.session_state.timeframe_map = {}
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Backtest Configuration")
    
    # === NEW: Market Selection ===
    st.sidebar.subheader("🌍 Select Market")
    
    # Get available markets
    market_types = get_all_market_types()
    market_info = get_market_display_info()
    
    selected_market = st.sidebar.selectbox(
        "Market Type",
        options=market_types,
        format_func=lambda x: f"{market_info[x]['icon']} {market_info[x]['name']}",
        help="Choose which market to backtest on",
        key="market_type_selector"
    )
    
    # Show market info in expander
    with st.sidebar.expander(f"ℹ️ About {market_info[selected_market]['name']}"):
        st.write(f"**Provider:** {market_info[selected_market]['provider']}")
        st.write(f"**Assets:** {', '.join(market_info[selected_market]['assets'])}")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Select Symbol")
    
    # === Dynamic Symbol Selection Based on Market ===
    if selected_market == UnifiedDataHandler.MARKET_INDIAN:
        # For Indian markets, keep existing logic
        index = st.sidebar.selectbox(
            "Select Index",
            options=['NIFTY', 'BANKNIFTY'],
            help="Choose the index to backtest"
        )
    else:
        # For other markets, use unified handler
        try:
            unified_handler = get_unified_handler(selected_market)
            
            # Check if handler connected successfully
            if not unified_handler.connected:
                st.sidebar.error(f"❌ Failed to connect to {selected_market}")
                st.sidebar.info("💡 Try selecting Indian Markets or check your API credentials")
                index = None
            else:
                categories = unified_handler.get_market_categories()
                
                # Debug: Show what we got
                logger.info(f"Categories for {selected_market}: {categories}")
                
                if not categories or categories == ['All']:
                    st.sidebar.warning("⚠️ No symbol categories found. Handler may still be initializing...")
                    st.sidebar.info("Please wait a moment and refresh the page")
                    index = None
                elif len(categories) > 1:
                    selected_category = st.sidebar.selectbox(
                        "Symbol Category",
                        options=categories,
                        help="Filter symbols by category"
                    )
                else:
                    selected_category = categories[0] if categories else None
            
            if selected_category:
                available_symbols = unified_handler.get_available_symbols_by_category(selected_category)
                
                if available_symbols:
                    symbol_options = [s['symbol'] for s in available_symbols]
                    
                    index = st.sidebar.selectbox(
                        "Select Symbol",
                        options=symbol_options,
                        format_func=lambda x: f"{x}",
                        help="Choose the symbol to backtest"
                    )
                else:
                    st.sidebar.warning("No symbols available")
                    index = None
            else:
                st.sidebar.warning("No categories available")
                index = None
        except Exception as e:
            st.sidebar.error(f"Error loading symbols: {e}")
            index = None
    
    # ✅ ADD THIS ENTIRE BLOCK:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Strategy Selection")
    
    strategy_choice = st.sidebar.selectbox(
        "Select Strategy Mode",
        options=['All SMC Strategies', 'CRT-TBS'],
        help="Choose which strategies to test"
    )
    
    # CRT-TBS specific configuration
    trading_style = None
    timeframe_map = {}
    
    if strategy_choice == 'CRT-TBS':
        st.sidebar.subheader("⏱️ CRT-TBS Configuration")
        
        trading_style = st.sidebar.radio(
            "Trading Style",
            options=['scalping', 'intraday', 'shortterm'],
            index=1,  # Default to intraday
            help="Choose timeframe combination"
        )
        
        # Timeframe mapping
        timeframe_map = {
            'scalping': '1H → 1min',
            'intraday': '1D → 1H',
            'shortterm': '4H → 5min'
        }
        
        st.sidebar.info(f"**Timeframes:** {timeframe_map[trading_style]}")
        
        # Show configuration preview
        with st.sidebar.expander("📋 View Configuration"):
            config = get_config(trading_style)
            st.write(f"**HTF:** {config['htf']}")
            st.write(f"**LTF:** {config['ltf']}")
            st.write(f"**Min RR Ratio:** {config['min_rr_ratio']}")
            st.write(f"**Risk per Trade:** {config['risk_per_trade']}%")
    
    # Test period selection
    st.sidebar.subheader("📅 Test Period")
    
    test_type = st.sidebar.radio(
        "Select Test Type",
        options=['Historical Backtest (2024)', 'Out-of-Sample Test (2025)', 'Custom Period'],
        help="Choose which period to test"
    )
    
    config = BacktestConfig()
    
    if test_type == 'Historical Backtest (2024)':
        start_date = config.BACKTEST_START_DATE
        end_date = config.BACKTEST_END_DATE
        st.sidebar.info(f"Testing: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    elif test_type == 'Out-of-Sample Test (2025)':
        start_date = config.FORWARD_TEST_START_DATE
        end_date = config.FORWARD_TEST_END_DATE
        st.sidebar.info(f"Testing: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    else:  # Custom period
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime(2024, 1, 1),
                min_value=datetime(2023, 1, 1),
                max_value=datetime.now()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime(2024, 12, 31),
                min_value=datetime(2023, 1, 1),
                max_value=datetime.now()
            )
        
        # Convert to datetime
        start_date = datetime.combine(start_date, datetime.min.time())
        end_date = datetime.combine(end_date, datetime.min.time())
    
    # Trading days calculation
    trading_days = len(get_trading_days(start_date, end_date))
    st.sidebar.metric("Estimated Trading Days", trading_days)
    
    st.sidebar.markdown("---")
    
    # === ADD THIS CHECK ===
    # Disable run button if no symbol selected
    can_run = index is not None and index != ''
    
    if not can_run:
        st.sidebar.warning("⚠️ Please select a valid symbol to run backtest")
    
    # Run button
    run_button = st.sidebar.button(
        "🚀 Run Backtest",
        type="primary",
        use_container_width=True,
        disabled=not can_run  # ← ADD THIS
    )

    # ✅ ADD CLEAR CACHE BUTTON HERE
    st.sidebar.markdown("---")
    st.sidebar.subheader("🗑️ Cache Management")
    
    from pathlib import Path
    cache_dir = Path('backtest_results/data_cache')
    
    if cache_dir.exists():
        cache_files = list(cache_dir.glob('*.json'))
        
        if cache_files:
            st.sidebar.info(f"📦 {len(cache_files)} cached file(s)")
            
            # Show cache size
            total_size = sum(f.stat().st_size for f in cache_files)
            size_mb = total_size / (1024 * 1024)
            st.sidebar.write(f"Cache size: {size_mb:.2f} MB")
            
            # Clear cache button
            if st.sidebar.button("🗑️ Clear All Cache", use_container_width=True):
                try:
                    deleted_count = 0
                    for cache_file in cache_files:
                        cache_file.unlink()
                        deleted_count += 1
                    
                    st.sidebar.success(f"✅ Deleted {deleted_count} cache file(s)")
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Error clearing cache: {e}")
        else:
            st.sidebar.write("No cache files found")
    else:
        st.sidebar.write("Cache directory doesn't exist yet")
    
    # Main area
    if not st.session_state.backtest_complete and not run_button:
        # Welcome screen
        st.info("👈 Configure your backtest in the sidebar and click 'Run Backtest' to begin")

        # ✅ ADD THIS:
        st.markdown("### Available Strategies")
        st.write("**All SMC Strategies:** Test all Smart Money Concept strategies in parallel")
        st.write("**CRT-TBS:** Multi-timeframe institutional price action strategy (Scalping, Intraday, Short-term)")
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📊 What is Backtesting?")
            st.write("""
            Backtesting simulates your trading strategies on historical data to evaluate performance 
            without risking real capital. It helps you:
            - Validate strategy effectiveness
            - Identify optimal parameters
            - Understand risk vs reward
            - Discover market conditions where strategies work best
            """)
        
        with col2:
            st.subheader("🎯 What You'll Get")
            st.write("""
            - **Performance Metrics**: Win rate, profit factor, drawdown
            - **Strategy Analysis**: Which strategies work best
            - **Market Insights**: Performance by market conditions
            - **Exit Analysis**: Optimize stops and targets
            - **Failure Analysis**: Understand what goes wrong
            - **Visual Reports**: Charts and comprehensive HTML report
            """)
        
        with col3:
            st.subheader("⚡ Quick Tips")
            st.write("""
            - Start with 2024 historical data (12 months)
            - Then test on 2025 out-of-sample data
            - Compare results between periods
            - Backtests take 5-15 minutes depending on period
            - Results are saved for later review
            """)
        
        # Show sample metrics
        st.markdown("---")
        st.subheader("📈 Sample Metrics Preview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Trades", "142", delta="Sample")
        with col2:
            st.metric("Win Rate", "64.1%", delta="+4.1%", delta_color="normal")
        with col3:
            st.metric("Profit Factor", "2.07", delta="+0.57", delta_color="normal")
        with col4:
            st.metric("Total P&L", "+1,847 pts", delta="Sample")
    
    elif run_button:
        # Run backtest
        st.session_state.backtest_complete = False
        
        try:
            # === Initialize appropriate handler based on market ===
            kite = None
            unified_handler = None
            
            if selected_market == UnifiedDataHandler.MARKET_INDIAN:
                # Initialize Kite for Indian markets
                with st.spinner("Initializing connection to Kite..."):
                    kite = get_kite_handler()
                    
                    if not kite.connected:
                        success, message = kite.initialize()
                        if not success:
                            st.error(f"❌ Failed to connect to Kite: {message}")
                            st.stop()
                
                st.success(f"✅ Connected to Kite API as {kite.user_profile.get('user_name', 'User')}")
            else:
                # Initialize unified handler for other markets
                with st.spinner(f"Initializing {market_info[selected_market]['name']}..."):
                    unified_handler = get_unified_handler(selected_market)
                    
                    if not unified_handler.connected:
                        st.error(f"❌ Failed to initialize {market_info[selected_market]['name']}")
                        st.stop()
                
                st.success(f"✅ Connected to {market_info[selected_market]['provider']}")

            # === Initialize backtest runner with appropriate handler ===
            runner_kwargs = {
                'kite_handler': kite if selected_market == UnifiedDataHandler.MARKET_INDIAN else None,
                'unified_handler': unified_handler if selected_market != UnifiedDataHandler.MARKET_INDIAN else None,
                'index': index,
                'start_date': start_date,
                'end_date': end_date,
                'market_type': selected_market  # Pass market type for reference
            }
            
            # Add strategy-specific parameters
            if strategy_choice == 'CRT-TBS':
                runner_kwargs['strategy_name'] = 'CRT_TBS'
                runner_kwargs['trading_style'] = trading_style
                st.info(f"🎯 Running CRT-TBS backtest ({trading_style.title()} mode)")
            else:
                runner_kwargs['strategy_name'] = 'ALL_SMC'
                st.info(f"🎯 Running all SMC strategies in parallel")
            
            runner = BacktestRunner(**runner_kwargs)
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(percent, message):
                progress_bar.progress(percent / 100)
                status_text.text(message)
            
            # Run backtest
            with st.spinner("Running backtest..."):
                results = runner.run_backtest(progress_callback=update_progress)
            
            # Store results
            st.session_state.backtest_results = results
            st.session_state.backtest_complete = True

            st.session_state.strategy_choice = strategy_choice
            st.session_state.trading_style = trading_style
            st.session_state.timeframe_map = timeframe_map
            
            # Clear progress
            progress_bar.empty()
            status_text.empty()
            
            st.success("✅ Backtest completed successfully!")
            st.balloons()
            
            # Rerun to show results
            st.rerun()
        
        except Exception as e:
            st.error(f"❌ Error running backtest: {str(e)}")
            st.exception(e)
    
    elif st.session_state.backtest_complete and st.session_state.backtest_results:
        # Display results
        results = st.session_state.backtest_results
        
        # Check if backtest encountered errors
        if 'error' in results:
            st.error(f"❌ Backtest failed: {results['error']}")
            if 'details' in results:
                with st.expander("Error Details"):
                    st.json(results['details'])
            
            # Reset button
            if st.button("🔄 Try Again", use_container_width=True):
                st.session_state.backtest_complete = False
                st.session_state.backtest_results = None
                st.rerun()
            return
        
        # Check if we have metrics
        if 'metrics' not in results:
            st.error("❌ Backtest completed but no metrics were generated.")
            st.write("Results keys:", list(results.keys()))
            
            # Reset button
            if st.button("🔄 Try Again", use_container_width=True):
                st.session_state.backtest_complete = False
                st.session_state.backtest_results = None
                st.rerun()
            return
        
        # Now safe to access metrics
        metrics = results['metrics']
        validation = results.get('validation', {})
        
        # Verdict banner
        verdict = validation.get('verdict', 'Unknown')
        if '✅' in verdict:
            st.markdown(f'<div class="success-box"><h2 style="margin:0;">{verdict}</h2></div>', unsafe_allow_html=True)
        elif '⚠️' in verdict:
            st.markdown(f'<div class="warning-box"><h2 style="margin:0;">{verdict}</h2></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="error-box"><h2 style="margin:0;">{verdict}</h2></div>', unsafe_allow_html=True)

        # ✅ ADD THIS BEFORE KEY METRICS:
        # Strategy Summary Banner
        st.subheader("📊 Backtest Summary")
        col1, col2, col3 = st.columns(3)
        
        strategy_name = st.session_state.get('strategy_choice', 'Unknown')
        style = st.session_state.get('trading_style')
        tfmap = st.session_state.get('timeframe_map', {})
        
        with col1:
            st.metric("Strategy", strategy_name)
        with col2:
            if strategy_name == 'CRT-TBS' and style:
                st.metric("Trading Style", style.title())
                if style in tfmap:
                    st.caption(f"Timeframes: {tfmap[style]}")
            else:
                st.metric("Mode", "Multi-Strategy")
                st.caption("All SMC strategies combined")
        with col3:
            st.metric("Index", index)
            st.caption(f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        st.markdown("---")
        
        # Key metrics
        st.subheader("📊 Key Performance Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total Trades",
                f"{metrics.get('total_trades', 0):,}",
                help="Total number of trades executed"
            )
        
        with col2:
            win_rate = metrics.get('win_rate', 0)
            st.metric(
                "Win Rate",
                f"{win_rate:.1f}%",
                delta=f"{win_rate - 50:.1f}%",
                delta_color="normal",
                help="Percentage of winning trades"
            )
        
        with col3:
            pf = metrics.get('profit_factor', 0)
            st.metric(
                "Profit Factor",
                f"{pf:.2f}",
                delta=f"{pf - 1.5:.2f}",
                delta_color="normal",
                help="Ratio of gross profit to gross loss"
            )
        
        with col4:
            total_pnl = metrics.get('total_pnl', 0)
            st.metric(
                "Total P&L",
                f"{total_pnl:,.0f} pts",
                delta=f"{total_pnl:,.0f}",
                delta_color="normal",
                help="Total profit/loss in points"
            )
        
        with col5:
            max_dd = metrics.get('max_drawdown', 0)
            st.metric(
                "Max Drawdown",
                f"{max_dd:,.0f} pts",
                delta=f"-{max_dd:,.0f}",
                delta_color="inverse",
                help="Maximum peak-to-trough decline"
            )
        
        st.markdown("---")
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Performance",
            "🎯 Strategies",
            "🌡️ Market Conditions",
            "📋 Trade Log",
            "💡 Recommendations",
            "📥 Download"
        ])
        
        with tab1:
            st.subheader("Performance Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Win/Loss Breakdown**")
                st.write(f"- Winning Trades: {metrics.get('winning_trades', 0)}")
                st.write(f"- Losing Trades: {metrics.get('losing_trades', 0)}")
                st.write(f"- Average Win: {metrics.get('avg_win', 0):.1f} pts")
                st.write(f"- Average Loss: {metrics.get('avg_loss', 0):.1f} pts")
                st.write(f"- Largest Win: {metrics.get('largest_win', 0):.1f} pts")
                st.write(f"- Largest Loss: {metrics.get('largest_loss', 0):.1f} pts")
            
            with col2:
                st.markdown("**Risk Metrics**")
                st.write(f"- Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
                st.write(f"- Max Consecutive Wins: {metrics.get('max_consecutive_wins', 0)}")
                st.write(f"- Max Consecutive Losses: {metrics.get('max_consecutive_losses', 0)}")
                st.write(f"- Average Holding Period: {metrics.get('avg_holding_period_minutes', 0):.0f} minutes")
                st.write(f"- Trades per Day: {metrics.get('trades_per_day', 0):.1f}")
            
            # Charts would go here (if implemented)
            st.info("📊 Charts are generated and saved in the output directory")
        
        with tab2:
            st.subheader("Strategy Performance Breakdown")
            
            strategy_breakdown = metrics.get('strategy_breakdown', {})
            
            if strategy_breakdown:
                # Create DataFrame
                strategy_df = pd.DataFrame.from_dict(strategy_breakdown, orient='index')
                strategy_df = strategy_df.sort_values('win_rate', ascending=False)
                
                # Display table
                st.dataframe(
                    strategy_df.style.format({
                        'win_rate': '{:.1f}%',
                        'total_pnl': '{:,.0f}',
                        'avg_pnl': '{:.1f}'
                    }).background_gradient(subset=['win_rate'], cmap='RdYlGn', vmin=40, vmax=80),
                    use_container_width=True
                )
                
                # Highlight best/worst
                best = metrics.get('best_strategy_name', 'N/A')
                best_wr = metrics.get('best_strategy_win_rate', 0)
                worst = metrics.get('worst_strategy_name', 'N/A')
                worst_wr = metrics.get('worst_strategy_win_rate', 0)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"🏆 Best: **{best}** ({best_wr:.1f}%)")
                with col2:
                    st.warning(f"⚠️ Worst: **{worst}** ({worst_wr:.1f}%)")
            else:
                st.info("No strategy breakdown available")
        
        with tab3:
            st.subheader("Market Condition Analysis")
            
            market_summary = results.get('market_summary', {})
            condition_performance = results.get('condition_performance', {})
            
            if market_summary:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Trending Days", f"{market_summary.get('trending_days', 0)} ({market_summary.get('trending_pct', 0):.0f}%)")
                with col2:
                    st.metric("Ranging Days", f"{market_summary.get('ranging_days', 0)} ({market_summary.get('ranging_pct', 0):.0f}%)")
                with col3:
                    st.metric("Volatile Days", f"{market_summary.get('volatile_days', 0)} ({market_summary.get('volatile_pct', 0):.0f}%)")
            
            if condition_performance:
                st.markdown("**Performance by Condition**")
                
                condition_df = pd.DataFrame.from_dict(condition_performance, orient='index')
                st.dataframe(
                    condition_df.style.format({
                        'win_rate': '{:.1f}%',
                        'total_pnl': '{:,.0f}',
                        'avg_pnl': '{:.1f}'
                    }).background_gradient(subset=['win_rate'], cmap='RdYlGn', vmin=40, vmax=80),
                    use_container_width=True
                )
        
        with tab4:
            st.subheader("Trade Log")
            
            trades_df = results.get('trades_df', pd.DataFrame())
            
            if not trades_df.empty:
                # Display first 100 trades
                st.dataframe(
                    trades_df.head(100).style.applymap(
                        lambda x: 'background-color: #d4edda' if isinstance(x, (int, float)) and x > 0 else ('background-color: #f8d7da' if isinstance(x, (int, float)) and x < 0 else ''),
                        subset=['pnl']
                    ),
                    use_container_width=True,
                    height=400
                )
                
                st.info(f"Showing first 100 of {len(trades_df)} total trades. Download full log in the Download tab.")
            else:
                st.warning("No trades executed during backtest period")
        
        with tab5:
            st.subheader("💡 Recommendations")
            
            recommendations = results.get('recommendations', [])
            
            if recommendations:
                for rec in recommendations:
                    # ✅ Handle both string and dict formats
                    if isinstance(rec, str):
                        # Simple string recommendation
                        st.info(rec)
                    elif isinstance(rec, dict):
                        # Dictionary recommendation with priority
                        priority = rec.get('priority', 'MEDIUM')
                        category = rec.get('category', 'General')
                        recommendation = rec.get('recommendation', rec.get('text', str(rec)))
                        
                        if priority == 'HIGH':
                            st.error(f"**{category}:** {recommendation}")
                        elif priority == 'MEDIUM':
                            st.warning(f"**{category}:** {recommendation}")
                        else:
                            st.info(f"**{category}:** {recommendation}")
                    else:
                        # Unknown format, just display as string
                        st.info(str(rec))
            else:
                st.success("✅ No major issues identified. System performing well!")
            
            # Validation issues
            if validation.get('issues'):
                st.subheader("⚠️ Validation Issues")
                for issue in validation['issues']:
                    st.error(f"❌ {issue}")
        
        with tab6:
            st.subheader("📥 Download Complete Backtest Results")
            
            st.write("Click the button below to download all reports, charts, and data in a single ZIP file.")
            
            # Create ZIP file
            import zipfile
            import io
            import json
            import os
            
            zip_buffer = io.BytesIO()
            
            try:
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    
                    # ✅ 1. Add trades CSV
                    trades_df = results.get('trades_df', pd.DataFrame())
                    if not trades_df.empty:
                        zip_file.writestr(
                            f"trades_{index}_{start_date.strftime('%Y%m%d')}.csv",
                            trades_df.to_csv(index=False)
                        )
                        st.success(f"✅ Added: trades.csv ({len(trades_df)} trades)")
                    
                    # ✅ 2. Add signals CSV
                    signals_df = results.get('signals_df', pd.DataFrame())
                    if not signals_df.empty:
                        zip_file.writestr(
                            f"signals_{index}_{start_date.strftime('%Y%m%d')}.csv",
                            signals_df.to_csv(index=False)
                        )
                        st.success(f"✅ Added: signals.csv ({len(signals_df)} signals)")
                    
                    # ✅ 3. Add metrics JSON
                    zip_file.writestr(
                        f"metrics_{index}_{start_date.strftime('%Y%m%d')}.json",
                        json.dumps(metrics, indent=2, default=str)
                    )
                    st.success("✅ Added: metrics.json")
                    
                    # ✅ 4. Add complete results JSON (all tabs data)
                    complete_results = {
                        'metrics': metrics,
                        'validation': validation,
                        'market_summary': results.get('market_summary', {}),
                        'condition_performance': results.get('condition_performance', {}),
                        'strategy_breakdown': metrics.get('strategy_breakdown', {}),
                        'recommendations': results.get('recommendations', []),
                        'mfe_mae_analysis': results.get('mfe_mae_analysis', {}),
                        'holding_analysis': results.get('holding_analysis', {}),
                        'failure_categories': results.get('failure_categories', {}),
                        'test_period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                    }
                    
                    zip_file.writestr(
                        f"complete_results_{index}_{start_date.strftime('%Y%m%d')}.json",
                        json.dumps(complete_results, indent=2, default=str)
                    )
                    st.success("✅ Added: complete_results.json")
                    
                    # ✅ 5. Add performance summary TXT
                    summary_text = f"""
                    ==========================================
                    BACKTEST PERFORMANCE SUMMARY
                    ==========================================
                    Strategy: {st.session_state.get('strategy_choice', 'Unknown')}
                    """
                    
                    # Add strategy-specific details
                    if st.session_state.get('strategy_choice') == 'CRT-TBS':
                        style = st.session_state.get('trading_style', 'unknown')
                        tfmap = st.session_state.get('timeframe_map', {})
                        summary_text += f"Trading Style: {style.title()}\n"
                        if style in tfmap:
                            summary_text += f"Timeframes: {tfmap[style]}\n"
                    
                    summary_text += f"""Index: {index}
                    Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}
                    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    
                    ==========================================
                    KEY METRICS
                    ==========================================
                    Total Trades:              {metrics.get('total_trades', 0):,}
                    Win Rate:                  {metrics.get('win_rate', 0):.2f}%
                    Profit Factor:             {metrics.get('profit_factor', 0):.2f}
                    Total P&L:                 {metrics.get('total_pnl', 0):,.2f} points
                    Max Drawdown:              {metrics.get('max_drawdown', 0):,.2f} points
                    Sharpe Ratio:              {metrics.get('sharpe_ratio', 0):.2f}
                    
                    ==========================================
                    WIN/LOSS BREAKDOWN
                    ==========================================
                    Winning Trades:            {metrics.get('winning_trades', 0)}
                    Losing Trades:             {metrics.get('losing_trades', 0)}
                    Average Win:               {metrics.get('avg_win', 0):.2f} points
                    Average Loss:              {metrics.get('avg_loss', 0):.2f} points
                    Largest Win:               {metrics.get('largest_win', 0):.2f} points
                    Largest Loss:              {metrics.get('largest_loss', 0):.2f} points
                    
                    ==========================================
                    RISK METRICS
                    ==========================================
                    Max Consecutive Wins:      {metrics.get('max_consecutive_wins', 0)}
                    Max Consecutive Losses:    {metrics.get('max_consecutive_losses', 0)}
                    Average Holding Period:    {metrics.get('avg_holding_period_minutes', 0):.0f} minutes
                    Trades per Day:            {metrics.get('trades_per_day', 0):.2f}
                    
                    ==========================================
                    STRATEGY BREAKDOWN
                    ==========================================
                    """
                    
                    # Add strategy breakdown
                    strategy_breakdown = metrics.get('strategy_breakdown', {})
                    if strategy_breakdown:
                        for strategy, stats in strategy_breakdown.items():
                            summary_text += f"""
{strategy}:
  - Trades:     {stats.get('trades', 0)}
  - Win Rate:   {stats.get('win_rate', 0):.2f}%
  - Total P&L:  {stats.get('total_pnl', 0):,.2f}
  - Avg P&L:    {stats.get('avg_pnl', 0):.2f}
"""
                    
                    # Add market conditions
                    summary_text += f"""
==========================================
MARKET CONDITIONS
==========================================
"""
                    market_summary = results.get('market_summary', {})
                    if market_summary:
                        summary_text += f"""
Trending Days:    {market_summary.get('trending_days', 0)} ({market_summary.get('trending_pct', 0):.0f}%)
Ranging Days:     {market_summary.get('ranging_days', 0)} ({market_summary.get('ranging_pct', 0):.0f}%)
Volatile Days:    {market_summary.get('volatile_days', 0)} ({market_summary.get('volatile_pct', 0):.0f}%)
"""
                    
                    # Add recommendations
                    summary_text += f"""
==========================================
RECOMMENDATIONS
==========================================
"""
                    recommendations = results.get('recommendations', [])
                    if recommendations:
                        for i, rec in enumerate(recommendations, 1):
                            if isinstance(rec, dict):
                                summary_text += f"{i}. [{rec.get('priority', 'INFO')}] {rec.get('recommendation', rec.get('text', str(rec)))}\n"
                            else:
                                summary_text += f"{i}. {str(rec)}\n"
                    else:
                        summary_text += "No recommendations - system performing well!\n"
                    
                    # Add validation status
                    summary_text += f"""
==========================================
VALIDATION STATUS
==========================================
Verdict: {validation.get('verdict', 'Unknown')}
"""
                    if validation.get('issues'):
                        summary_text += "\nIssues:\n"
                        for issue in validation['issues']:
                            summary_text += f"  - {issue}\n"
                    
                    zip_file.writestr(
                        f"summary_report_{index}_{start_date.strftime('%Y%m%d')}.txt",
                        summary_text
                    )
                    st.success("✅ Added: summary_report.txt")
                    
                    # ✅ 6. Add HTML report if exists
                    output_dir = Path(config.RESULTS_DIR) / f"{index}_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}"
                    html_path = output_dir / "backtest_report.html"
                    
                    if html_path.exists():
                        with open(html_path, 'r', encoding='utf-8') as f:
                            zip_file.writestr(
                                f"backtest_report_{index}.html",
                                f.read()
                            )
                        st.success("✅ Added: backtest_report.html")
                    
                    # ✅ 7. Add all PNG charts
                    if output_dir.exists():
                        png_files = list(output_dir.glob('*.png'))
                        for png_file in png_files:
                            zip_file.write(
                                png_file,
                                arcname=f"charts/{png_file.name}"
                            )
                        if png_files:
                            st.success(f"✅ Added: {len(png_files)} chart(s)")
                
                # ✅ Download button
                zip_buffer.seek(0)
                
                st.markdown("---")
                st.download_button(
                    label="📦 Download Complete Results (ZIP)",
                    data=zip_buffer,
                    file_name=f"backtest_{index}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.zip",
                    mime="application/zip",
                    use_container_width=True,
                    type="primary"
                )
                
                st.info("💡 **Tip:** This ZIP contains all trades, signals, metrics, charts, and reports from all tabs above.")
                
            except Exception as e:
                st.error(f"❌ Error creating ZIP file: {str(e)}")
                st.exception(e)
            
            # Also show individual file location
            st.markdown("---")
            st.write("**Original files location:**")
            output_dir = Path(config.RESULTS_DIR) / f"{index}_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}"
            st.code(str(output_dir))
            
            # Reset button
            if st.button("🔄 Run New Backtest", use_container_width=True):
                st.session_state.backtest_complete = False
                st.session_state.backtest_results = None
                st.rerun()

if __name__ == '__main__':
    main()
