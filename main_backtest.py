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
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Backtest Configuration")
    
    # Index selection
    index = st.sidebar.selectbox(
        "Select Index",
        options=['NIFTY', 'BANKNIFTY'],
        help="Choose the index to backtest"
    )
    
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
    
    # Run button
    run_button = st.sidebar.button(
        "🚀 Run Backtest",
        type="primary",
        use_container_width=True
    )
    
    # Main area
    if not st.session_state.backtest_complete and not run_button:
        # Welcome screen
        st.info("👈 Configure your backtest in the sidebar and click 'Run Backtest' to begin")
        
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
            # Initialize Kite
            with st.spinner("Initializing connection to Kite..."):
                kite = get_kite_handler()
            
            st.success("✅ Connected to Kite API")
            
            # Initialize backtest runner
            runner = BacktestRunner(
                kite_handler=kite,
                index=index,
                start_date=start_date,
                end_date=end_date
            )
            
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
                    priority = rec.get('priority', 'MEDIUM')
                    category = rec.get('category', 'General')
                    recommendation = rec.get('recommendation', '')
                    
                    if priority == 'HIGH':
                        st.error(f"**{category}:** {recommendation}")
                    elif priority == 'MEDIUM':
                        st.warning(f"**{category}:** {recommendation}")
                    else:
                        st.info(f"**{category}:** {recommendation}")
            else:
                st.success("✅ No major issues identified. System performing well!")
            
            # Validation issues
            if validation.get('issues'):
                st.subheader("⚠️ Validation Issues")
                for issue in validation['issues']:
                    st.error(f"❌ {issue}")
        
        with tab6:
            st.subheader("📥 Download Reports")
            
            st.write("All reports have been generated and saved to:")
            output_dir = Path(config.RESULTS_DIR) / f"{index}_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}"
            st.code(str(output_dir))
            
            st.markdown("**Available Files:**")
            st.write("- `backtest_report.html` - Comprehensive HTML report")
            st.write("- `trades.csv` - Complete trade log")
            st.write("- `signals.csv` - All signals generated")
            st.write("- `metrics.json` - Performance metrics")
            st.write("- `*.png` - All performance charts")
            
            # Reset button
            if st.button("🔄 Run New Backtest", use_container_width=True):
                st.session_state.backtest_complete = False
                st.session_state.backtest_results = None
                st.rerun()


if __name__ == '__main__':
    main()
