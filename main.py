"""
Main Streamlit Application - Index Options Trading Platform
Focused on Index Options Contracts only
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import pytz
from typing import Optional, Dict, List

# Define IST timezone for accurate timestamps
IST = pytz.timezone('Asia/Kolkata')

# Fixed imports - separate cache functions
from config import (
    validate_config,
    get_market_status,
    MARKET_CONFIG,
    TRADING_CONFIG,
    get_config_summary
)

# Import from cache_utils
from cache_utils import (
    get_indices_by_exchange,
    get_instrument_config
)

from database import (
    init_database,
    get_latest_ticks,
    get_watchlist,
    get_trade_history,
    clear_old_tick_data
)

from kite_handler import (
    initialize_kite,
    get_kite_handler
)

from streaming import (
    start_streaming,
    stop_streaming,
    subscribe_instruments,
    unsubscribe_instruments,
    get_streaming_handler,
    get_streaming_status
)

from pattern_detector import PatternDetector
from fibonacci_calculator import FibonacciCalculator
from news_fetcher import NewsFetcher
from technical_analyzer import TechnicalAnalyzer
from chart_builder import ChartBuilder

from indicators import (
    calculate_rsi, 
    calculate_macd, 
    calculate_ema, 
    calculate_sma, 
    calculate_bollinger_bands,
    calculate_supertrend
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Index Options Trading Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.kite_connected = False
        st.session_state.streaming_active = False
        st.session_state.selected_options = []
        st.session_state.live_data = {}
        st.session_state.last_refresh = None
        st.session_state.available_indices = []

        st.session_state.analysis_in_progress = False
        st.session_state.last_analysis_time = None

        # ✅ NEW: Auto-refresh state
        st.session_state['auto_refresh_interval'] = None  # Default: disabled
        st.session_state['last_refresh_time'] = None

init_session_state()

# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_app():
    """Initialize application (database, Kite, streaming)"""
    
    # Step 1: Validate configuration
    errors = validate_config()
    if errors:
        st.error("❌ Configuration Errors:")
        for error in errors:
            st.error(f"  • {error}")
        st.stop()
    
    # Step 2: Initialize database
    with st.spinner("📊 Initializing database..."):
        init_database()
    
    # Step 3: Initialize Kite Connect AND fetch instruments
    if not st.session_state.kite_connected:
        with st.spinner("🔌 Connecting to Kite and loading instruments..."):
            success, message = initialize_kite()

        if success:
            st.session_state.kite_connected = True
            st.success(message)
            
            # Verify instruments loaded
            kite = get_kite_handler()
            if kite.instruments_df is not None and not kite.instruments_df.empty:
                st.success(f"✅ Loaded {len(kite.instruments_df)} instruments")
                
                # ✅ NEW: CACHE MAP STATUS IN SESSION STATE
                if hasattr(kite, 'index_token_map') and kite.index_token_map:
                    st.session_state['index_map_loaded'] = True
                    st.session_state['index_map_count'] = len(kite.index_token_map)
                    st.success(f"✅ Index mapping ready ({len(kite.index_token_map)} indices)")
                else:
                    st.warning("⚠️ Index mapping not built")
                    st.session_state['index_map_loaded'] = False
            else:
                st.error("❌ No instruments were loaded")
                st.stop()
        else:
            st.error(message)
            st.stop()
    
    # Step 4: Start streaming
    if not st.session_state.streaming_active:
        with st.spinner("📡 Starting streaming..."):
            if start_streaming():
                st.session_state.streaming_active = True
                st.success("✅ Real-time streaming started")
                
                # Subscribe to watchlist instruments
                watchlist = get_watchlist()
                if watchlist:
                    subscribe_instruments(watchlist)
            else:
                st.warning("⚠️ Streaming could not start")
    
    st.session_state.initialized = True
    time.sleep(0.5)  # Brief pause

def check_analysis_timeout():
    """Check if analysis is stuck and reset if needed"""
    if st.session_state.get('analysis_in_progress', False):
        start_time_str = st.session_state.get('last_analysis_time')
        
        if start_time_str:
            try:
                from datetime import datetime
                current_time = datetime.now()
                start_time = datetime.strptime(start_time_str, '%H:%M:%S')
                
                # Create datetime for comparison
                start_dt = current_time.replace(
                    hour=start_time.hour,
                    minute=start_time.minute,
                    second=start_time.second,
                    microsecond=0
                )
                
                elapsed = (current_time - start_dt).total_seconds()
                if elapsed < 0:
                    elapsed += 86400  # Handle midnight crossing
                
                # 60 second timeout
                if elapsed > 60:
                    st.session_state['analysis_in_progress'] = False
                    return True, elapsed
                
                return False, elapsed
            except:
                pass
    
    return False, 0

def get_index_futures_token(kite, index_symbol: str) -> Optional[int]:
    """
    Get the current month futures token for an index
    E.g., NIFTY -> NIFTY25DECFUT
    """
    try:
        # Remove exchange prefix
        clean_symbol = index_symbol.replace('NSE:', '').replace('BSE:', '')
        
        # Search for futures
        instruments = kite.kite.instruments('NFO')
        
        # Filter for index futures, current/near month
        futures = [i for i in instruments if 
                   i['tradingsymbol'].startswith(clean_symbol) and 
                   i['instrument_type'] == 'FUT' and
                   i['segment'] == 'NFO-FUT']
        
        if futures:
            # Sort by expiry, get nearest
            futures.sort(key=lambda x: x['expiry'])
            return futures[0]['instrument_token']
    except:
        pass
    return None

# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar():
    """Render sidebar with controls and status"""
    
    st.sidebar.title("📊 Index Options Platform")
    st.sidebar.markdown("---")

    # ✅ EMERGENCY RESET (if analysis stuck)
    if st.session_state.get('analysis_in_progress', False):
        is_stuck, elapsed = check_analysis_timeout()
        
        if is_stuck:
            st.sidebar.error(f"⚠️ Analysis stuck ({int(elapsed)}s)!")
            if st.sidebar.button("🔄 Reset Analysis", type="primary"):
                st.session_state['analysis_in_progress'] = False
                st.success("✅ Reset complete!")
                time.sleep(0.5)
                st.rerun()
        else:
            st.sidebar.warning(f"⏳ Analysis: {int(elapsed)}s / 60s")

    # Developer Tools
    
    # ✅ NEW: Auto-Refresh Configuration
    st.sidebar.subheader("⚡ Auto-Refresh Settings")
    
    # Enable/disable auto-refresh
    auto_refresh_enabled = st.sidebar.checkbox(
        "Enable Auto-Refresh",
        value=False,  # Default OFF for safety
        help="Automatically reload options chain and re-analyze market"
    )
    
    if auto_refresh_enabled:
        # Refresh interval selector
        refresh_interval = st.sidebar.select_slider(
            "Refresh Interval (seconds)",
            options=[10, 15, 30, 60, 120, 300],  # Safe intervals only
            value=30,  # Default 30 seconds
            help="⚠️ Minimum 10 seconds to avoid API rate limits"
        )
        
        # Show warning for fast intervals
        if refresh_interval < 30:
            st.sidebar.warning(f"⚠️ {refresh_interval}s refresh is aggressive. Monitor API usage!")
        
        # Store in session state
        st.session_state['auto_refresh_interval'] = refresh_interval
        
        # Show next refresh countdown
        if 'last_refresh_time' in st.session_state:
            last_refresh_time = st.session_state.get('last_refresh_time', None)
            
            if last_refresh_time is None or not isinstance(last_refresh_time, datetime):
                # Initialize it to current time safely
                st.session_state['last_refresh_time'] = datetime.now()
                elapsed = 0
            else:
                elapsed = (datetime.now() - last_refresh_time).total_seconds()

            next_refresh = max(0, refresh_interval - elapsed)
            st.sidebar.info(f"🔄 Next refresh in: {int(next_refresh)}s")
    else:
        st.session_state['auto_refresh_interval'] = None
    
    st.sidebar.markdown("---")
    
    # Developer Tools
    with st.sidebar.expander("🔧 Developer Tools"):   
        col1, col2 = st.columns(2)
        
        with col1:
            # Regular module reload
            if st.button("🔃 Reload Code", use_container_width=True):
                import sys
                import importlib
                
                modules = ['config', 'kite_handler', 'trend_analyzer', 
                           'strike_selector', 'indicators', 'database']
                
                for mod in modules:
                    if mod in sys.modules:
                        try:
                            importlib.reload(sys.modules[mod])
                        except Exception as e:
                            st.error(f"Failed to reload {mod}: {e}")
                
                st.success("✅ Modules reloaded!")
                st.caption("⚠️ Note: Doesn't reload API data or singletons")
                time.sleep(0.5)
                st.rerun()
        
        with col2:
            # Full restart (clears everything)
            if st.button("🔄 Full Reset", use_container_width=True, type="primary"):
                st.warning("⚠️ This will clear ALL data and restart the app")
                
                # Clear session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                
                # Clear singleton
                import kite_handler
                kite_handler._kite_handler_instance = None
                
                st.success("✅ Full reset complete! Reloading...")
                time.sleep(1)
                st.rerun()
    
    # Show module status
    st.caption("**Module Status:**")
    import sys
    modules_loaded = [m for m in ['kite_handler', 'trend_analyzer', 'strike_selector'] if m in sys.modules]
    st.caption(f"Loaded: {', '.join(modules_loaded)}")

    # System Status
    st.sidebar.subheader("🔌 System Status")
    
    # Market Status
    market_status = get_market_status()
    if market_status['status'] == 'OPEN':
        st.sidebar.success(f"🟢 Market: {market_status['status']}")
    else:
        st.sidebar.error(f"🔴 Market: {market_status['status']}")
    st.sidebar.caption(f"📅 {market_status['time']}")
    st.sidebar.caption(f"Reason: {market_status['reason']}")
    
    # Kite Connection Status
    if st.session_state.kite_connected:
        st.sidebar.success("✅ Kite: Connected")
    else:
        st.sidebar.error("❌ Kite: Disconnected")
    
    # Streaming Status
    streaming_status = get_streaming_status()
    if streaming_status['connected']:
        st.sidebar.success(f"✅ Streaming: Active")
        st.sidebar.caption(f"📊 {streaming_status['subscribed_count']}/{streaming_status['max_instruments']} instruments")
        st.sidebar.caption(f"🕐 Last tick: {streaming_status['last_tick']}")
    else:
        st.sidebar.error("❌ Streaming: Inactive")
    
    st.sidebar.markdown("---")
    
    # Controls
    st.sidebar.subheader("⚙️ Controls")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Old Data", use_container_width=True):
            clear_old_tick_data(days=7)
            st.success("Cleared!")
    
    # Streaming Toggle
    if st.session_state.streaming_active:
        if st.sidebar.button("⏸️ Stop Streaming", use_container_width=True, type="secondary"):
            stop_streaming()
            st.session_state.streaming_active = False
            st.rerun()
    else:
        if st.sidebar.button("▶️ Start Streaming", use_container_width=True, type="primary"):
            start_streaming()
            st.session_state.streaming_active = True
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # Trading Configuration
    st.sidebar.subheader("📋 Configuration")
    st.sidebar.caption(f"**Product:** {TRADING_CONFIG['product_type']}")
    st.sidebar.caption(f"**Mode:** {TRADING_CONFIG['mode']}")
    st.sidebar.caption(f"**Focus:** Index Options Only")
    
    st.sidebar.markdown("---")
    
    # Debug Info (collapsible)
    with st.sidebar.expander("🔍 Debug Info"):
        config_summary = get_config_summary()
        st.json(config_summary)

# ============================================================================
# INDEX OPTIONS TAB
# ============================================================================

def render_index_options_tab():
    """Render index options analysis interface"""
    st.header("📊 Index Options Analysis")
    
    # Simple check
    if not st.session_state.get('initialized', False):
        st.info("⏳ Initializing... Please wait.")
        return
    
    kite = get_kite_handler()
    
    # Exchange and Index Selection
    col1, col2, col3 = st.columns([2, 3, 2])
    
    with col1:
        # Exchange selection
        exchange = st.selectbox(
            "Select Exchange",
            options=["NSE", "BSE"],
            key="selected_exchange"
        )
    
    with col2:
        # Fetch available indices for selected exchange
        available_indices = kite.get_indices_by_exchange(exchange)
        
        if available_indices:
            st.session_state.available_indices = available_indices
            
            selected_index = st.selectbox(
                f"Select {exchange} Index",
                options=available_indices,
                key="selected_index"
            )
        else:
            st.warning(f"No indices found for {exchange}")
            st.info("💡 Try clicking 'Refresh Indices' button below")
            
            if st.button("🔄 Refresh Indices"):
                # Force re-fetch instruments
                kite.fetch_and_cache_instruments("NSE")
                kite.fetch_and_cache_instruments("NFO")
                st.rerun()
            
            selected_index = None
    
    with col3:
        # Display lot size for selected index
        if selected_index:
            index_config = get_instrument_config(selected_index)
            
            if index_config:
                st.metric("Lot Size", index_config.get('lot_size', 'N/A'))
                st.caption(f"Tick: {index_config.get('tick_size', 'N/A')}")
            else:
                st.info("Loading...")
    
    # Stop here if no index selected
    if not selected_index:
        return
    
    st.markdown("---")
    
    # Display Current Index Price
    col_price1, col_price2, col_price3 = st.columns([1, 1, 1])
    
    with col_price1:
        # Fetch current index price
        index_ltp = kite.get_index_ltp(selected_index, exchange)
        
        if index_ltp:
            st.metric(
                label=f"{selected_index} Current Price",
                value=f"₹{index_ltp:,.2f}",
                delta=None
            )
        else:
            st.warning("Unable to fetch current price")
    
    with col_price2:
        st.empty()  # Placeholder for future metrics
    
    with col_price3:
        st.empty()  # Placeholder for future metrics
    
    st.markdown("---")
    
    # Expiry Selection (Optional)
    col_exp1, col_exp2 = st.columns([3, 1])
    
    with col_exp1:
        filter_by_expiry = st.checkbox("Filter by specific expiry?", value=False)
        
        if filter_by_expiry:
            expiry_date = st.date_input(
                "Select Expiry Date",
                value=datetime.now().date(),
                key="expiry_filter"
            )
        else:
            expiry_date = None
    
    with col_exp2:
        st.empty()
    
    # Load Options Chain Button
    if st.button("🔍 Load Options Chain", type="primary", use_container_width=True):
        with st.spinner(f"Loading {selected_index} options chain..."):
            
            expiry_str = expiry_date.strftime('%Y-%m-%d') if expiry_date else None
            
            calls_df, puts_df, all_expiries = kite.get_option_chain(
                selected_index,
                expiry_date=expiry_str
            )
            
            if calls_df is not None and puts_df is not None:
                st.session_state.options_chain = {
                    'calls': calls_df,
                    'puts': puts_df,
                    'index': selected_index,
                    'all_expiries': all_expiries,
                    'index_price': index_ltp
                }
                
                if expiry_date:
                    st.success(f"✅ Loaded {len(calls_df)} Calls and {len(puts_df)} Puts for expiry: {expiry_str}")
                else:
                    st.success(f"✅ Loaded {len(calls_df)} Calls and {len(puts_df)} Puts across {len(all_expiries)} expiries")
            else:
                st.error("Failed to load options chain")
    
    # Display Options Chain (rest remains same)
    if 'options_chain' in st.session_state:
        st.markdown("---")
        st.subheader("📋 Options Chain Data")
        
        # Show available expiries
        if st.session_state.options_chain.get('all_expiries'):
            with st.expander("📅 Available Expiries", expanded=False):
                expiries_list = st.session_state.options_chain['all_expiries']
                st.write(f"**Total Expiries:** {len(expiries_list)}")
                
                # Display in columns
                exp_cols = st.columns(4)
                for idx, exp in enumerate(expiries_list):
                    with exp_cols[idx % 4]:
                        st.caption(exp)
        
        st.markdown("---")
        
        # Tabs for Calls, Puts, Combined
        tab1, tab2, tab3 = st.tabs(["📈 Call Options", "📉 Put Options", "🎯 Combined View"])
        
        with tab1:
            st.subheader("Call Options (CE)")
            calls_df = st.session_state.options_chain['calls']
            
            if not calls_df.empty:
                # Display important columns
                display_cols = ['expiry', 'strike', 'tradingsymbol', 'lot_size', 'instrument_token']
                available_cols = [col for col in display_cols if col in calls_df.columns]
                
                st.dataframe(
                    calls_df[available_cols],
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
                
                st.caption(f"**Total Call Options:** {len(calls_df)}")
            else:
                st.info("No call options data")
        
        with tab2:
            st.subheader("Put Options (PE)")
            puts_df = st.session_state.options_chain['puts']
            
            if not puts_df.empty:
                display_cols = ['expiry', 'strike', 'tradingsymbol', 'lot_size', 'instrument_token']
                available_cols = [col for col in display_cols if col in puts_df.columns]
                
                st.dataframe(
                    puts_df[available_cols],
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
                
                st.caption(f"**Total Put Options:** {len(puts_df)}")
            else:
                st.info("No put options data")
            
        with tab3:
            st.subheader("🎯 Automated Contract Recommendation")
            st.caption("Multi-timeframe trend analysis with ITM strike selection")
            
            # Get data from session
            calls_df = st.session_state.options_chain['calls']
            puts_df = st.session_state.options_chain['puts']
            index_symbol = st.session_state.options_chain['index']
            spot_price = st.session_state.options_chain.get('index_price')
            
            if spot_price is None:
                st.error("Spot price not available. Please reload options chain.")
                return
            
            # Display current info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Index", index_symbol)
            with col2:
                st.metric("Spot Price", f"₹{spot_price:,.2f}")
            with col3:
                current_time = datetime.now(IST).strftime("%H:%M:%S")  # ✅ FIXED
                st.metric("Time", current_time)
            
            st.markdown("---")
            
            # ✅ FIXED: Check timeout first
            is_timeout, elapsed = check_analysis_timeout()
            
            if is_timeout:
                st.error(f"❌ Analysis Timeout ({int(elapsed)}s)")
                st.warning("Click 'Reset Analysis' in sidebar or button below")
                if st.button("🔄 Reset Now"):
                    st.session_state['analysis_in_progress'] = False
                    st.rerun()
                return
            
            # Show progress if running
            if st.session_state.get('analysis_in_progress', False):
                st.info(f"⏳ Analysis running... ({int(elapsed)}s / 60s)")
                st.caption(f"Started at: {st.session_state.get('last_analysis_time')}")
                
                if st.button("⏹️ Cancel Analysis"):
                    st.session_state['analysis_in_progress'] = False
                    st.rerun()
                
                # Auto-refresh to update timer
                time.sleep(1)
                st.rerun()
                return
            
            # ✅ UPDATED: Analysis button with safe auto-refresh
            auto_refresh_interval = st.session_state.get('auto_refresh_interval')
            
            if auto_refresh_interval:
                # AUTO-REFRESH MODE
                st.info(f"🔄 Auto-refresh: Every {auto_refresh_interval}s")
                
                last_refresh = st.session_state.get('last_refresh_time')
                should_refresh = False
                
                if last_refresh is None:
                    should_refresh = True
                else:
                    elapsed_since_refresh = (datetime.now() - last_refresh).total_seconds()
                    if elapsed_since_refresh >= auto_refresh_interval:
                        should_refresh = True
                    else:
                        # Show countdown
                        next_in = int(auto_refresh_interval - elapsed_since_refresh)
                        st.caption(f"⏱️ Next: {next_in}s")
                
                # Control buttons
                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button("🎯 Analyze Now", use_container_width=True):
                        should_refresh = True
                with col2:
                    if st.button("⏸️ Pause", use_container_width=True):
                        st.session_state['auto_refresh_interval'] = None
                        st.rerun()
                
                # Trigger analysis
                if should_refresh:
                    st.session_state['analysis_in_progress'] = True
                    st.session_state['last_analysis_time'] = datetime.now().strftime("%H:%M:%S")
                    st.session_state['last_refresh_time'] = datetime.now()
                    
                    try:
                        # STEP 1: Refresh options chain
                        with st.spinner(f"🔄 Refreshing {index_symbol}..."):
                            expiry_str = expiry_date.strftime('%Y-%m-%d') if 'expiry_date' in locals() and expiry_date else None
                            calls_df, puts_df, all_expiries = kite.get_option_chain(index_symbol, expiry_date=expiry_str)
                            
                            if calls_df is not None and puts_df is not None:
                                st.session_state['options_chain'].update({
                                    'calls': calls_df,
                                    'puts': puts_df,
                                    'index_price': index_ltp
                                })
                                spot_price = index_ltp
                        
                        # STEP 2: Trend analysis
                        # from trend_analyzer import TrendAnalyzer
                        # from strike_selector import StrikeSelector
                        
                        # with st.spinner("📊 Analyzing..."):
                        #     analyzer = TrendAnalyzer(kite)
                        #    trend_analysis = analyzer.analyze_trend(index_symbol, spot_price)
                        
                        if 'error' in trend_analysis:
                            st.warning(f"⚠️ {trend_analysis['error']}")
                            st.session_state['analysis_in_progress'] = False
                            time.sleep(2)
                            st.rerun()
                        
                        # STEP 3: Select strikes
                        with st.spinner("🎯 Selecting..."):
                            selector = StrikeSelector()
                            recommendation = selector.select_contract(trend_analysis, calls_df, puts_df, spot_price)
                        
                        # Store results
                        st.session_state['recommendation'] = {
                            'trend': trend_analysis,
                            'contracts': recommendation,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        st.success("✅ Complete!")
                        
                    except Exception as e:
                        st.error(f"❌ Failed: {str(e)}")
                    
                    finally:
                        st.session_state['analysis_in_progress'] = False
                    
                    time.sleep(1)
                    st.rerun()
                
                # Auto-rerun for countdown
                else:
                    time.sleep(1)
                    st.rerun()

            else:
                # MANUAL MODE
                if st.button("🎯 Analyze Market & Get Recommendation", type="primary", use_container_width=True):
                    # ✅ NEW: VALIDATE INSTRUMENT MAP BEFORE ANALYSIS
                    kite = get_kite_handler()
                    
                    # Check if index_token_map exists and is populated
                    if not hasattr(kite, 'index_token_map') or not kite.index_token_map:
                        st.error("❌ Instrument mapping not ready!")
                        st.warning("Please wait... Rebuilding instrument map...")
                        
                        # Force rebuild
                        with st.spinner("🔧 Building instrument map..."):
                            kite._build_index_token_map()
                        
                        # Verify after rebuild
                        if not kite.index_token_map:
                            st.error("❌ Failed to build instrument map. Please restart the application.")
                            st.stop()
                        else:
                            st.success(f"✅ Built map with {len(kite.index_token_map)} indices")
                            time.sleep(1)
                    
                    # ✅ NEW: VERIFY SELECTED INDEX IS IN MAP
                    if index_symbol not in kite.index_token_map:
                        st.error(f"❌ '{index_symbol}' not found in instrument map!")
                        st.warning("Attempting to rebuild map...")
                        
                        with st.spinner("🔧 Rebuilding..."):
                            kite._build_index_token_map()
                        
                        if index_symbol not in kite.index_token_map:
                            st.error(f"❌ '{index_symbol}' still not found after rebuild.")
                            st.info(f"Available indices: {list(kite.index_token_map.keys())}")
                            st.stop()
                    
                    st.session_state['analysis_in_progress'] = True
                    st.session_state['last_analysis_time'] = datetime.now().strftime("%H:%M:%S")
                    
                    try:
                        # from trend_analyzer import TrendAnalyzer
                        # from strike_selector import StrikeSelector
                        
                        # with st.spinner("📊 Analyzing trend..."):
                        #    analyzer = TrendAnalyzer(kite)
                        #    trend_analysis = analyzer.analyze_trend(index_symbol, spot_price)
                        
                        if 'error' in trend_analysis:
                            st.warning(f"⚠️ {trend_analysis['error']}")
                            st.session_state['analysis_in_progress'] = False
                            return
                        
                        with st.spinner("🎯 Selecting contracts..."):
                            selector = StrikeSelector()
                            recommendation = selector.select_contract(trend_analysis, calls_df, puts_df, spot_price)
                        
                        st.session_state['recommendation'] = {
                            'trend': trend_analysis,
                            'contracts': recommendation,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        st.success("✅ Complete!")
                        
                    except Exception as e:
                        st.error(f"❌ Failed: {str(e)}")
                        with st.expander("Error"):
                            st.exception(e)
                    
                    finally:
                        st.session_state['analysis_in_progress'] = False
                        st.rerun()

            # Display results if available
            if 'recommendation' in st.session_state:
                rec = st.session_state.recommendation
                trend = rec['trend']
                contracts = rec['contracts']
                
                st.markdown("---")
                
                # ✅ NEW: Check if trend analysis was successful
                if 'error' in trend:
                    st.error(f"❌ Analysis Error: {trend.get('error', 'Unknown error')}")
                    
                    if 'market_status' in trend:
                        st.info("**Market Status:**")
                        st.json(trend['market_status'])
                    
                    st.warning("⚠️ Cannot display recommendation. Please try again when market is open.")
                    st.stop()
                
                # SECTION 1: Trend Analysis Summary
                st.subheader("📊 Market Trend Analysis")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    direction_emoji = "🟢" if trend['direction'] == "BULLISH" else "🔴" if trend['direction'] == "BEARISH" else "🟡"
                    st.metric("Direction", f"{direction_emoji} {trend['direction']}")
                
                with col2:
                    conf_color = "🟢" if trend['confidence'] == "HIGH" else "🟡" if trend['confidence'] == "MODERATE" else "🔴"
                    st.metric("Confidence", f"{conf_color} {trend['confidence']}")
                
                with col3:
                    st.metric("Action", trend['action'])
                
                with col4:
                    score = trend['combined_score']
                    score_display = f"{score:+.2f}"
                    st.metric("Combined Score", score_display)
                
                # Detailed Timeframe Analysis
                # ✅ NEW: Only show if timeframe_analysis exists
                if 'timeframe_analysis' in trend:
                    with st.expander("📈 Detailed Timeframe Breakdown", expanded=False):
                        tf_data = trend['timeframe_analysis']
                        
                        subtab1, subtab2, subtab3 = st.tabs(["Daily (40%)", "Hourly (30%)", "15-Min (30%)"])
                        
                        with subtab1:
                            st.write("**Daily Timeframe Analysis:**")
                            st.json(tf_data['daily'])
                        
                        with subtab2:
                            st.write("**Hourly Timeframe Analysis:**")
                            st.json(tf_data['hourly'])
                        
                        with subtab3:
                            st.write("**15-Minute Timeframe Analysis:**")
                            st.json(tf_data['15min'])
                
                st.markdown("---")
                
                # SECTION 2: Contract Recommendations
                st.subheader("🎯 Option Contract Recommendations")
                
                if 'error' in contracts:
                    st.warning(f"⚠️ {contracts['error']}")
                    if 'recommendation' in contracts:
                        st.info(f"💡 {contracts['recommendation']}")
                    if 'reason' in contracts:
                        st.caption(contracts['reason'])
                else:
                    # Create columns for all three options display
                    rec_col1, rec_col2, rec_col3 = st.columns(3)
                    
                    with rec_col1:
                        st.success("**✅ ITM (RECOMMENDED)**")
                        itm = contracts['recommended']
                        st.write(f"**{itm['tradingsymbol']}**")
                        st.write(f"Strike: ₹{itm['strike']:,.0f}")
                        st.write(f"Lot Size: {itm['lot_size']}")
                        st.write(f"Expiry: {itm['expiry']}")
                        st.write(f"Distance: ₹{itm['distance_from_spot']:.0f}")
                        st.write(f"Intrinsic: ₹{itm['intrinsic_value']:.0f}")
                    
                    with rec_col2:
                        st.info("**⚖️ ATM (Reference)**")
                        atm = contracts['options']['ATM']
                        st.write(f"**{atm['tradingsymbol']}**")
                        st.write(f"Strike: ₹{atm['strike']:,.0f}")
                        st.write(f"Lot Size: {atm['lot_size']}")
                        st.write(f"Expiry: {atm['expiry']}")
                        st.write(f"Distance: ₹{atm['distance_from_spot']:.0f}")
                    
                    with rec_col3:
                        st.info("**🎲 OTM (Reference)**")
                        otm = contracts['options']['OTM']
                        st.write(f"**{otm['tradingsymbol']}**")
                        st.write(f"Strike: ₹{otm['strike']:,.0f}")
                        st.write(f"Lot Size: {otm['lot_size']}")
                        st.write(f"Expiry: {otm['expiry']}")
                        st.write(f"Distance: ₹{otm['distance_from_spot']:.0f}")
                    
                    # Full details expander
                    with st.expander("📋 Complete ITM Contract Details", expanded=False):
                        st.write(f"**Type:** {contracts['type']}")
                        st.write(f"**Direction:** {contracts['direction']}")
                        st.write(f"**Instrument Token:** {itm['instrument_token']}")
                        st.write(f"**Days to Expiry:** {contracts['days_to_expiry']}")
                        st.write(f"**Percentage from Spot:** {itm['percentage_from_spot']:.2f}%")
                        st.info(f"**💡 Recommendation Reason:** {contracts['recommendation_reason']}")
                    
                    # Analysis timestamp
                    st.caption(f"Analysis completed at: {rec['timestamp']}")
            
            # Analysis Section - Expandable/Collapsible
            with st.expander("📊 Indicator & News Analysis", expanded=False):
                
                try:
                    # Initialize analyzers
                    pattern_detector = PatternDetector()
                    fib_calculator = FibonacciCalculator()
                    news_fetcher = NewsFetcher()
                    tech_analyzer = TechnicalAnalyzer()
                    chart_builder = ChartBuilder()
                    
                    # Get index details
                    index_symbol = st.session_state.options_chain['index']
                    spot_price = st.session_state.options_chain.get('index_price')
                    
                    if not spot_price:
                        st.warning("⚠️ Spot price not available. Please reload options chain.")
                    else:
                        # Fetch multi-timeframe data
                        with st.spinner("🔄 Fetching multi-timeframe data..."):
                            # Get index token
                            index_token = None
                            
                            # Method 1: Use index_token_map if available
                            if hasattr(kite, 'index_token_map') and index_symbol in kite.index_token_map:
                                index_token = kite.index_token_map[index_symbol]
                                print(f"✅ Found index token from map: {index_token}")
                            
                            # Method 2: Search in instruments
                            if not index_token:
                                instrument = kite.search_instruments(index_symbol, exchange='NSE')
                                if instrument and len(instrument) > 0:
                                    index_token = instrument[0]['instrument_token']
                                    print(f"✅ Found index token from search: {index_token}")

                            
                            if index_token:
                                # Get historical data for multiple timeframes
                                futures_token = get_index_futures_token(kite, index_symbol)
                                
                                if futures_token:
                                    st.info(f"📊 Using index futures for intraday data (Token: {futures_token})")
                                    mtf_data = kite.get_multi_timeframe_data(futures_token, days=90)
                                else:
                                    st.info("📊 Using index spot data (daily only)")
                                    mtf_data = kite.get_multi_timeframe_data(index_token, days=90)

                                
                                # Check if we have data (prefer 5min, fallback to 15min, then daily)
                                df_analysis = None
                                timeframe_used = ""
                                
                                if '5mindata' in mtf_data and mtf_data['5mindata'] is not None and not mtf_data['5mindata'].empty:
                                    df_analysis = mtf_data['5mindata']
                                    timeframe_used = "5-minute"
                                elif '15mindata' in mtf_data and mtf_data['15mindata'] is not None and not mtf_data['15mindata'].empty:
                                    df_analysis = mtf_data['15mindata']
                                    timeframe_used = "15-minute"
                                    st.info("ℹ️ Using 15-minute data (5-minute not available)")
                                elif '60mindata' in mtf_data and mtf_data['60mindata'] is not None and not mtf_data['60mindata'].empty:
                                    df_analysis = mtf_data['60mindata']
                                    timeframe_used = "1-hour"
                                    st.info("ℹ️ Using 1-hour data (shorter timeframes not available)")
                                elif 'daydata' in mtf_data and mtf_data['daydata'] is not None and not mtf_data['daydata'].empty:
                                    df_analysis = mtf_data['daydata']
                                    timeframe_used = "daily"
                                    st.info("ℹ️ Using daily data (intraday timeframes not available for indices)")
                                else:
                                    df_analysis = None
                                
                                if df_analysis is not None and not df_analysis.empty:
                                    df_5min = df_analysis  # Use this variable to keep rest of code working
                                    st.success(f"✅ Analysis using {timeframe_used} timeframe ({len(df_5min)} candles)")

                                    # ==============================================================
                                    # OVERALL MARKET CONSENSUS (NEW SECTION - Add at the top)
                                    # ==============================================================
                                    st.markdown("## 🎯 Overall Market Consensus")
                                    st.markdown("*Based on analysis of Technical Indicators, Moving Averages, MACD, Fibonacci, and News Sentiment*")
                                    st.markdown("")
                                    
                                    # Initialize vote counters
                                    bullish_votes = 0
                                    bearish_votes = 0
                                    neutral_votes = 0
                                    total_weight = 0
                                    
                                    # Dictionary to track individual signals
                                    signal_details = []
                                    
                                    # ============================================
                                    # 1. TECHNICAL INDICATORS SUMMARY (Weight: 25%)
                                    # ============================================
                                    tech_weight = 25
                                    if 'bullish_signals' in st.session_state and 'bearish_signals' in st.session_state:
                                        tech_bullish = st.session_state['bullish_signals']
                                        tech_bearish = st.session_state['bearish_signals']
                                        
                                        if tech_bullish > tech_bearish:
                                            bullish_votes += tech_weight
                                            signal_details.append({'indicator': 'Technical Indicators', 'signal': 'Bullish', 'weight': tech_weight})
                                        elif tech_bearish > tech_bullish:
                                            bearish_votes += tech_weight
                                            signal_details.append({'indicator': 'Technical Indicators', 'signal': 'Bearish', 'weight': tech_weight})
                                        else:
                                            neutral_votes += tech_weight
                                            signal_details.append({'indicator': 'Technical Indicators', 'signal': 'Neutral', 'weight': tech_weight})
                                        
                                        total_weight += tech_weight
                                    
                                    # ============================================
                                    # 2. MOVING AVERAGES (Weight: 15%)
                                    # ============================================
                                    ma_weight = 15
                                    if 'ma_above_count' in st.session_state and 'ma_total_count' in st.session_state:
                                        ma_above = st.session_state['ma_above_count']
                                        ma_total = st.session_state['ma_total_count']
                                        
                                        if ma_above >= ma_total * 0.7:
                                            bullish_votes += ma_weight
                                            signal_details.append({'indicator': 'Moving Averages', 'signal': 'Bullish', 'weight': ma_weight})
                                        elif ma_above <= ma_total * 0.3:
                                            bearish_votes += ma_weight
                                            signal_details.append({'indicator': 'Moving Averages', 'signal': 'Bearish', 'weight': ma_weight})
                                        else:
                                            neutral_votes += ma_weight
                                            signal_details.append({'indicator': 'Moving Averages', 'signal': 'Neutral', 'weight': ma_weight})
                                        
                                        total_weight += ma_weight
                                    
                                    # ============================================
                                    # 3. MACD (Weight: 10%)
                                    # ============================================
                                    macd_weight = 10
                                    if 'macd_signal' in st.session_state:
                                        macd_signal = st.session_state['macd_signal']
                                        
                                        if 'bullish' in macd_signal.lower():
                                            bullish_votes += macd_weight
                                            signal_details.append({'indicator': 'MACD', 'signal': 'Bullish', 'weight': macd_weight})
                                        elif 'bearish' in macd_signal.lower():
                                            bearish_votes += macd_weight
                                            signal_details.append({'indicator': 'MACD', 'signal': 'Bearish', 'weight': macd_weight})
                                        else:
                                            neutral_votes += macd_weight
                                            signal_details.append({'indicator': 'MACD', 'signal': 'Neutral', 'weight': macd_weight})
                                        
                                        total_weight += macd_weight
                                    
                                    # ============================================
                                    # 4. FIBONACCI (Weight: 10%)
                                    # ============================================
                                    fib_weight = 10
                                    if 'fib_trend' in st.session_state:
                                        fib_trend = st.session_state['fib_trend']
                                        
                                        if fib_trend == 'uptrend':
                                            bullish_votes += fib_weight
                                            signal_details.append({'indicator': 'Fibonacci', 'signal': 'Bullish', 'weight': fib_weight})
                                        elif fib_trend == 'downtrend':
                                            bearish_votes += fib_weight
                                            signal_details.append({'indicator': 'Fibonacci', 'signal': 'Bearish', 'weight': fib_weight})
                                        else:
                                            neutral_votes += fib_weight
                                            signal_details.append({'indicator': 'Fibonacci', 'signal': 'Neutral', 'weight': fib_weight})
                                        
                                        total_weight += fib_weight
                                    
                                    # ============================================
                                    # 5. NEWS SENTIMENT (Weight: 10%)
                                    # ============================================
                                    news_weight = 10
                                    if 'news_sentiment' in st.session_state:
                                        news_sent = st.session_state['news_sentiment']
                                        
                                        if 'bullish' in news_sent.lower():
                                            bullish_votes += news_weight
                                            signal_details.append({'indicator': 'News Sentiment', 'signal': 'Bullish', 'weight': news_weight})
                                        elif 'bearish' in news_sent.lower():
                                            bearish_votes += news_weight
                                            signal_details.append({'indicator': 'News Sentiment', 'signal': 'Bearish', 'weight': news_weight})
                                        else:
                                            neutral_votes += news_weight
                                            signal_details.append({'indicator': 'News Sentiment', 'signal': 'Neutral', 'weight': news_weight})
                                        
                                        total_weight += news_weight

                                    # 6. PURE PRICE ACTION (Weight: 30%)
                                    pa_weight = 30
                                    if 'price_action_verdict' in st.session_state:
                                        pa_verdict = st.session_state['price_action_verdict']
                                        
                                        if 'bullish' in pa_verdict.lower():
                                            bullish_votes += pa_weight
                                            signal_details.append({'indicator': 'Price Action (Daily)', 'signal': 'Bullish', 'weight': pa_weight})
                                        elif 'bearish' in pa_verdict.lower():
                                            bearish_votes += pa_weight
                                            signal_details.append({'indicator': 'Price Action (Daily)', 'signal': 'Bearish', 'weight': pa_weight})
                                        else:
                                            neutral_votes += pa_weight
                                            signal_details.append({'indicator': 'Price Action (Daily)', 'signal': 'Neutral', 'weight': pa_weight})
                                        
                                        total_weight += pa_weight

                                    # ============================================
                                    # CALCULATE OVERALL CONSENSUS
                                    # ============================================
                                    if total_weight > 0:
                                        bullish_pct = (bullish_votes / total_weight) * 100
                                        bearish_pct = (bearish_votes / total_weight) * 100
                                        neutral_pct = (neutral_votes / total_weight) * 100
                                        
                                        # Determine overall trend
                                        if bullish_pct >= 60:
                                            st.success("# 🟢 STRONG BULLISH CONSENSUS")
                                            overall_trend = "Strong Bullish"
                                            trend_desc = f"**{bullish_pct:.0f}% of indicators agree** - Market shows strong upward momentum"
                                            recommendation = "📈 **Primary Strategy:** CALL options recommended. Look for ATM or slightly OTM strikes."
                                        
                                        elif bullish_pct > bearish_pct and bullish_pct >= 40:
                                            st.success("# 🟢 BULLISH CONSENSUS")
                                            overall_trend = "Bullish"
                                            trend_desc = f"**{bullish_pct:.0f}% bullish vs {bearish_pct:.0f}% bearish** - Bullish bias prevails"
                                            recommendation = "📈 **Primary Strategy:** CALL options preferred. Wait for pullback entries."
                                        
                                        elif bearish_pct >= 60:
                                            st.error("# 🔴 STRONG BEARISH CONSENSUS")
                                            overall_trend = "Strong Bearish"
                                            trend_desc = f"**{bearish_pct:.0f}% of indicators agree** - Market shows strong downward pressure"
                                            recommendation = "📉 **Primary Strategy:** PUT options recommended. Look for ATM or slightly OTM strikes."
                                        
                                        elif bearish_pct > bullish_pct and bearish_pct >= 40:
                                            st.error("# 🔴 BEARISH CONSENSUS")
                                            overall_trend = "Bearish"
                                            trend_desc = f"**{bearish_pct:.0f}% bearish vs {bullish_pct:.0f}% bullish** - Bearish bias prevails"
                                            recommendation = "📉 **Primary Strategy:** PUT options preferred. Wait for rally entries."
                                        
                                        else:
                                            st.info("# ⚪ MIXED SIGNALS / NO CLEAR CONSENSUS")
                                            overall_trend = "Neutral"
                                            trend_desc = f"**Conflicting signals:** {bullish_pct:.0f}% bullish, {bearish_pct:.0f}% bearish, {neutral_pct:.0f}% neutral"
                                            recommendation = "⚠️ **Primary Strategy:** AVOID directional trades. Wait for clearer alignment or consider neutral strategies (Iron Condor, Straddle)."
                                        
                                        st.markdown(trend_desc)
                                        st.markdown("")
                                        st.info(recommendation)
                                        
                                        st.markdown("")
                                        
                                        # Detailed Breakdown Table
                                        st.markdown("### 📊 Signal Breakdown by Indicator")
                                        
                                        breakdown_df = pd.DataFrame(signal_details)
                                        if not breakdown_df.empty:
                                            st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
                                        
                                        # Confidence Meter
                                        st.markdown("")
                                        st.markdown("### 📈 Confidence Level")
                                        
                                        max_pct = max(bullish_pct, bearish_pct, neutral_pct)
                                        if max_pct >= 70:
                                            confidence = "Very High"
                                            conf_color = "🟢"
                                        elif max_pct >= 55:
                                            confidence = "High"
                                            conf_color = "🟡"
                                        elif max_pct >= 40:
                                            confidence = "Moderate"
                                            conf_color = "🟠"
                                        else:
                                            confidence = "Low"
                                            conf_color = "🔴"
                                        
                                        st.metric("Consensus Strength", f"{conf_color} {confidence}", f"{max_pct:.0f}% agreement")
                                    
                                    else:
                                        st.warning("Insufficient data to calculate overall consensus")
                                    
                                    st.markdown("---")
                                    st.markdown("*Note: This consensus combines Technical (75% weight) and News (25% weight) analysis*")
                                    st.markdown("---")

                
                                    # ==============================================================
                                    # SECTION 1: Pattern Detection & Trade Confirmation
                                    # ==============================================================
                                    st.markdown("### 🎯 Pattern Detection & Trade Confirmation")
                                    
                                    # Detect patterns
                                    all_patterns = pattern_detector.detect_all_patterns(
                                        df_analysis,
                                        support=df_analysis['low'].min(),
                                        resistance=df_analysis['high'].max()
                                    )
                                    
                                    # Display strongest pattern with better formatting
                                    if all_patterns and all_patterns[0]['pattern'] != 'No Significant Pattern':
                                        strongest = all_patterns[0]
                                        pattern_type = strongest['type']
                                        
                                        # Color-coded display based on pattern type
                                        if pattern_type == 'bullish':
                                            st.success(f"**🟢 {strongest['pattern']}** detected (Strength: {strongest['strength']}%)")
                                        elif pattern_type == 'bearish':
                                            st.error(f"**🔴 {strongest['pattern']}** detected (Strength: {strongest['strength']}%)")
                                        else:
                                            st.info(f"**⚪ {strongest['pattern']}** detected")
                                        
                                        # Show description in italic
                                        st.markdown(f"*{strongest['description']}*")
                                        
                                        st.markdown("")  # Add spacing
                                    else:
                                        st.info("ℹ️ No significant candlestick pattern detected")
                                    
                                    st.markdown("")  # Add spacing
                                    
                                    # 5-Point Trade Confirmation Checklist
                                    st.markdown("#### ✅ 5-Point Trade Confirmation Checklist")
                                    
                                    # Prepare analysis results for checklist
                                    analysis_results = {
                                        '5mdata': df_analysis,
                                        'support': df_analysis['low'].min(),
                                        'resistance': df_analysis['high'].max(),
                                        'latest_price': spot_price,
                                        'all_patterns': all_patterns,
                                        'candlestick_pattern': all_patterns[0]['pattern'] if all_patterns else 'None',
                                        'pattern_type': all_patterns[0]['type'] if all_patterns else 'neutral',
                                        'rsi': calculate_rsi(df_analysis['close'], 14).iloc[-1] if len(df_analysis) >= 14 else 50
                                    }
                                    
                                    # Add MACD values
                                    if len(df_analysis) >= 26:
                                        macd_line, signal_line, histogram = calculate_macd(df_analysis['close'])
                                        analysis_results['macd'] = {
                                            'histogram': histogram.iloc[-1]
                                        }
                                    else:
                                        analysis_results['macd'] = {'histogram': 0}
                                    
                                    # Run checklist
                                    checklist = pattern_detector.run_confirmation_checklist(analysis_results)
                                    
                                    if checklist.get('error'):
                                        st.warning(f"⚠️ {checklist['error']}")
                                    elif checklist.get('data_available'):
                                        # Display checklist items in 2 columns with better formatting
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            st.markdown(f"**1. At Key S/R Level:** {checklist['1. At Key S/R Level']}")
                                            st.markdown(f"**2. Price Rejection:** {checklist['2. Price Rejection']}")
                                            st.markdown(f"**3. Chart Pattern Confirmed:** {checklist['3. Chart Pattern Confirmed']}")
                                        
                                        with col2:
                                            st.markdown(f"**4. Candlestick Signal:** {checklist['4. Candlestick Signal']}")
                                            st.markdown(f"**5. Indicator Alignment:** {checklist['5. Indicator Alignment']}")
                                        
                                        st.markdown("")  # Add spacing
                                        
                                        # Final signal with prominent display
                                        signal = checklist['FINAL_SIGNAL']
                                        
                                        # Convert BUY/SELL to BULLISH/BEARISH for options trading
                                        if '🟢 BUY' in signal:
                                            signal_display = signal.replace('BUY SIGNAL', 'BULLISH SIGNAL')
                                            st.success(f"### {signal_display}")
                                            st.caption("✅ Strong bullish setup detected. Consider CALL options.")
                                        elif '🔴 SELL' in signal:
                                            signal_display = signal.replace('SELL SIGNAL', 'BEARISH SIGNAL')
                                            st.error(f"### {signal_display}")
                                            st.caption("✅ Strong bearish setup detected. Consider PUT options.")
                                        else:
                                            # HOLD signal with warning
                                            st.info(f"### {signal}")
                                            
                                            # Count confirmations
                                            confirmations_count = sum([
                                                '✅' in checklist['1. At Key S/R Level'],
                                                '✅' in checklist['2. Price Rejection'],
                                                '✅' in checklist['3. Chart Pattern Confirmed'],
                                                '✅' in checklist['4. Candlestick Signal'],
                                                '✅' in checklist['5. Indicator Alignment']
                                            ])
                                            
                                            if confirmations_count < 3:
                                                st.warning(f"⚠️ Insufficient confirmations. Wait for better setup.")
                                    
                                    st.markdown("---")

                                    # ==============================================================
                                    # Section 1.0 PURE DAILY PRICE ACTION ANALYSIS (NO INDICATORS)
                                    # ==============================================================
                                    st.markdown("## 📊 Daily Price Action Analysis")
                                    st.caption("*Analyzing candle structure without indicators - Pure price action*")
                                    st.markdown("")
                                    
                                    if 'daydata' in mtf_data and mtf_data['daydata'] is not None:
                                        df_daily = mtf_data['daydata']
                                        
                                        if len(df_daily) >= 20:
                                            # Get recent candles
                                            recent_candles = df_daily.tail(20)
                                            current_candle = df_daily.iloc[-1]
                                            prev_candle = df_daily.iloc[-2]
                                            
                                            price_action_signals = []
                                            bullish_pa = 0
                                            bearish_pa = 0
                                            
                                            # ====================================
                                            # 1. TREND STRUCTURE ANALYSIS
                                            # ====================================
                                            highs = df_daily['high'].tail(10)
                                            lows = df_daily['low'].tail(10)
                                            
                                            # Check for Higher Highs & Higher Lows (Uptrend)
                                            higher_highs = 0
                                            higher_lows = 0
                                            lower_highs = 0
                                            lower_lows = 0
                                            
                                            for i in range(1, len(highs)):
                                                if highs.iloc[i] > highs.iloc[i-1]:
                                                    higher_highs += 1
                                                else:
                                                    lower_highs += 1
                                                
                                                if lows.iloc[i] > lows.iloc[i-1]:
                                                    higher_lows += 1
                                                else:
                                                    lower_lows += 1
                                            
                                            if higher_highs > lower_highs and higher_lows > lower_lows:
                                                bullish_pa += 2  # Strong weight
                                                price_action_signals.append("✅ **Trend Structure:** Higher Highs + Higher Lows (Strong Uptrend)")
                                            elif lower_highs > higher_highs and lower_lows > higher_lows:
                                                bearish_pa += 2  # Strong weight
                                                price_action_signals.append("❌ **Trend Structure:** Lower Highs + Lower Lows (Strong Downtrend)")
                                            else:
                                                price_action_signals.append("⚪ **Trend Structure:** Sideways/Choppy (No clear trend)")
                                            
                                            # ====================================
                                            # 2. SUPPORT & RESISTANCE ANALYSIS
                                            # ====================================
                                            # Find recent swing high (resistance) and swing low (support)
                                            swing_high = df_daily['high'].tail(20).max()
                                            swing_low = df_daily['low'].tail(20).min()
                                            current_price = current_candle['close']
                                            
                                            # Check distance from swing points
                                            dist_from_high = ((swing_high - current_price) / current_price) * 100
                                            dist_from_low = ((current_price - swing_low) / current_price) * 100
                                            
                                            if dist_from_high < 2:  # Within 2% of resistance
                                                if current_price > swing_high:
                                                    bullish_pa += 1
                                                    price_action_signals.append(f"✅ **Breakout:** Price broke above resistance ({swing_high:.0f})")
                                                else:
                                                    price_action_signals.append(f"⚠️ **Near Resistance:** Price at {swing_high:.0f} (could reverse)")
                                            
                                            if dist_from_low < 2:  # Within 2% of support
                                                if current_price < swing_low:
                                                    bearish_pa += 1
                                                    price_action_signals.append(f"❌ **Breakdown:** Price broke below support ({swing_low:.0f})")
                                                else:
                                                    price_action_signals.append(f"⚠️ **Near Support:** Price at {swing_low:.0f} (could bounce)")
                                            
                                            # ====================================
                                            # 3. DAILY CANDLE PATTERN
                                            # ====================================
                                            body = abs(current_candle['close'] - current_candle['open'])
                                            upper_wick = current_candle['high'] - max(current_candle['open'], current_candle['close'])
                                            lower_wick = min(current_candle['open'], current_candle['close']) - current_candle['low']
                                            
                                            # Bullish engulfing
                                            if (current_candle['close'] > current_candle['open'] and 
                                                prev_candle['close'] < prev_candle['open'] and
                                                current_candle['close'] > prev_candle['open'] and
                                                current_candle['open'] < prev_candle['close']):
                                                bullish_pa += 1
                                                price_action_signals.append("✅ **Candle Pattern:** Bullish Engulfing (Strong reversal signal)")
                                            
                                            # Bearish engulfing
                                            elif (current_candle['close'] < current_candle['open'] and 
                                                  prev_candle['close'] > prev_candle['open'] and
                                                  current_candle['close'] < prev_candle['open'] and
                                                  current_candle['open'] > prev_candle['close']):
                                                bearish_pa += 1
                                                price_action_signals.append("❌ **Candle Pattern:** Bearish Engulfing (Strong reversal signal)")
                                            
                                            # Hammer (bullish reversal)
                                            elif lower_wick > (body * 2) and upper_wick < body:
                                                bullish_pa += 1
                                                price_action_signals.append("✅ **Candle Pattern:** Hammer (Bullish reversal)")
                                            
                                            # Shooting star (bearish reversal)
                                            elif upper_wick > (body * 2) and lower_wick < body:
                                                bearish_pa += 1
                                                price_action_signals.append("❌ **Candle Pattern:** Shooting Star (Bearish reversal)")
                                            
                                            # Strong bullish candle
                                            elif current_candle['close'] > current_candle['open'] and body > (current_candle['high'] - current_candle['low']) * 0.7:
                                                bullish_pa += 1
                                                price_action_signals.append("✅ **Candle Pattern:** Strong Bullish Candle (Body > 70%)")
                                            
                                            # Strong bearish candle
                                            elif current_candle['close'] < current_candle['open'] and body > (current_candle['high'] - current_candle['low']) * 0.7:
                                                bearish_pa += 1
                                                price_action_signals.append("❌ **Candle Pattern:** Strong Bearish Candle (Body > 70%)")
                                            
                                            else:
                                                price_action_signals.append("⚪ **Candle Pattern:** Neutral/Indecision (Doji or small body)")
                                            
                                            # ====================================
                                            # 4. MOMENTUM (Close vs Previous)
                                            # ====================================
                                            last_3_closes = df_daily['close'].tail(3)
                                            if all(last_3_closes.iloc[i] > last_3_closes.iloc[i-1] for i in range(1, len(last_3_closes))):
                                                bullish_pa += 1
                                                price_action_signals.append("✅ **Momentum:** 3 consecutive higher closes (Strong bullish momentum)")
                                            elif all(last_3_closes.iloc[i] < last_3_closes.iloc[i-1] for i in range(1, len(last_3_closes))):
                                                bearish_pa += 1
                                                price_action_signals.append("❌ **Momentum:** 3 consecutive lower closes (Strong bearish momentum)")
                                            
                                            # ====================================
                                            # FINAL PRICE ACTION VERDICT
                                            # ====================================
                                            total_pa = bullish_pa + bearish_pa
                                            
                                            if total_pa > 0:
                                                bullish_pa_pct = (bullish_pa / total_pa) * 100
                                                
                                                if bullish_pa_pct >= 70:
                                                    st.success("### 🟢 BULLISH PRICE ACTION (Daily Chart)")
                                                    st.markdown("**Pure candle analysis shows strong upward bias**")
                                                    pa_verdict = "Bullish"
                                                elif bullish_pa_pct >= 50:
                                                    st.success("### 🟢 MODERATELY BULLISH")
                                                    st.markdown("**Daily candles favor upside**")
                                                    pa_verdict = "Bullish"
                                                elif bullish_pa_pct <= 30:
                                                    st.error("### 🔴 BEARISH PRICE ACTION (Daily Chart)")
                                                    st.markdown("**Pure candle analysis shows strong downward bias**")
                                                    pa_verdict = "Bearish"
                                                elif bullish_pa_pct < 50:
                                                    st.error("### 🔴 MODERATELY BEARISH")
                                                    st.markdown("**Daily candles favor downside**")
                                                    pa_verdict = "Bearish"
                                                else:
                                                    st.info("### ⚪ NEUTRAL PRICE ACTION")
                                                    st.markdown("**Daily candles show no clear direction**")
                                                    pa_verdict = "Neutral"
                                                
                                                # Show all signals
                                                st.markdown("**Price Action Signals:**")
                                                for signal in price_action_signals:
                                                    st.markdown(f"- {signal}")
                                                
                                                # Store for consensus
                                                st.session_state['price_action_verdict'] = pa_verdict
                                                st.session_state['price_action_bullish_pct'] = bullish_pa_pct
                                            
                                            st.markdown("---")

                                    # ==============================================================
                                    # SECTION 2: Technical Indicators Summary
                                    # ==============================================================
                                    st.markdown("### 📊 Technical Indicators Summary")
                                    
                                    # Analyze multiple timeframes
                                    timeframes_to_analyze = [
                                        ('5mindata', '5-Minute'),
                                        ('15mindata', '15-Minute'),
                                        ('60mindata', '1-Hour'),
                                        ('daydata', 'Daily')
                                    ]
                                    
                                    summary_data = []
                                    bullish_signals = 0
                                    bearish_signals = 0
                                    total_signals = 0
                                    
                                    for tf_key, tf_label in timeframes_to_analyze:
                                        if tf_key in mtf_data and mtf_data[tf_key] is not None:
                                            df_tf = mtf_data[tf_key]
                                            if len(df_tf) >= 50:
                                                analysis = tech_analyzer.analyze_all(df_tf, tf_label)
                                                
                                                if not analysis.get('error'):
                                                    summary_data.append({
                                                        'Timeframe': tf_label,
                                                        'RSI': f"{analysis['rsi']} ({analysis['rsi_signal']})",
                                                        'MA Signal': analysis['ma_signal'],
                                                        'MACD': analysis['macd_signal'],
                                                        'BB Position': analysis['bb_signal']
                                                    })
                                                    
                                                    # Count signals for overall conclusion
                                                    if '🟢' in analysis['rsi_signal'] or 'Bullish' in analysis['ma_signal'] or '🟢' in analysis['macd_signal']:
                                                        bullish_signals += 1
                                                    if '🔴' in analysis['rsi_signal'] or 'Bearish' in analysis['ma_signal'] or '🔴' in analysis['macd_signal']:
                                                        bearish_signals += 1
                                                    total_signals += 1
                                    
                                    # Display multi-timeframe table
                                    if summary_data:
                                        summary_df = pd.DataFrame(summary_data)
                                        st.dataframe(summary_df, use_container_width=True, hide_index=True)
                                    else:
                                        st.warning("Insufficient data for technical indicators")
                                    
                                    st.markdown("")  # Spacing
                                    
                                    # ===========================================================
                                    # ADDITIONAL INDICATORS (from old UI)
                                    # ===========================================================
                                    if 'daydata' in mtf_data and mtf_data['daydata'] is not None:
                                        df_daily = mtf_data['daydata']
                                        
                                        if len(df_daily) >= 50:
                                            close = df_daily['close']
                                            current_price = close.iloc[-1]
                                            
                                            # Create 4-column layout for additional indicators
                                            col1, col2, col3, col4 = st.columns(4)
                                            
                                            # === COLUMN 1: Bollinger Bands ===
                                            with col1:
                                                st.markdown("**Bollinger Bands**")
                                                bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, 20, 2)
                                                
                                                st.write(f"Upper: ₹{bb_upper.iloc[-1]:.2f}")
                                                st.write(f"Middle: ₹{bb_middle.iloc[-1]:.2f}")
                                                st.write(f"Lower: ₹{bb_lower.iloc[-1]:.2f}")
                                                
                                                # BB Status
                                                if current_price >= bb_upper.iloc[-1]:
                                                    st.info("⚪ Within Bands")
                                                elif current_price <= bb_lower.iloc[-1]:
                                                    st.success("🟢 At Lower Band")
                                                else:
                                                    st.info("⚪ Within Bands")
                                            
                                            # === COLUMN 2: Stochastic Momentum ===
                                            with col2:
                                                st.markdown("**Stochastic Momentum**")
                                                
                                                # Calculate Stochastic (simple implementation)
                                                high_14 = df_daily['high'].rolling(window=14).max()
                                                low_14 = df_daily['low'].rolling(window=14).min()
                                                k_percent = 100 * ((close - low_14) / (high_14 - low_14))
                                                d_percent = k_percent.rolling(window=3).mean()
                                                
                                                k_val = k_percent.iloc[-1] if not pd.isna(k_percent.iloc[-1]) else 50
                                                d_val = d_percent.iloc[-1] if not pd.isna(d_percent.iloc[-1]) else 50
                                                
                                                st.write(f"%K: {k_val:.2f}")
                                                st.write(f"%D: {d_val:.2f}")
                                                
                                                # Crossover status
                                                if k_val > d_val:
                                                    st.error("🔴 No Crossover")
                                                else:
                                                    st.info("⚪ No Crossover")
                                            
                                            # === COLUMN 3: VWAP/VWMA ===
                                            with col3:
                                                st.markdown("**VWAP/VWMA**")
                                                
                                                # Calculate VWAP (Volume Weighted Average Price)
                                                if 'volume' in df_daily.columns:
                                                    vwap = (df_daily['close'] * df_daily['volume']).cumsum() / df_daily['volume'].cumsum()
                                                    vwap_val = vwap.iloc[-1]
                                                else:
                                                    vwap_val = current_price
                                                
                                                # VWMA (Volume Weighted Moving Average)
                                                if 'volume' in df_daily.columns:
                                                    vwma = (df_daily['close'] * df_daily['volume']).rolling(window=20).sum() / df_daily['volume'].rolling(window=20).sum()
                                                    vwma_val = vwma.iloc[-1]
                                                else:
                                                    vwma_val = current_price
                                                
                                                st.write(f"VWAP: ₹{vwap_val:.2f}")
                                                st.write(f"VWMA: ₹{vwma_val:.2f}")
                                                
                                                # VWAP status
                                                if current_price > vwap_val:
                                                    st.success("🟢 Above VWAP (Bullish)")
                                                else:
                                                    st.error("🔴 Below VWAP (Bearish)")
                                            
                                            # === COLUMN 4: SuperTrend ===
                                            with col4:
                                                st.markdown("**SuperTrend**")
                                                
                                                # Calculate SuperTrend
                                                from indicators import calculate_supertrend
                                                supertrend_values, supertrend_direction = calculate_supertrend(df_daily, period=10, multiplier=3)
                                                
                                                st.write(f"Value: ₹{supertrend_values.iloc[-1]:.2f}")
                                                
                                                # Direction
                                                if supertrend_direction.iloc[-1] == 1:
                                                    st.error("🔴 DOWNTREND")
                                                else:
                                                    st.success("🟢 UPTREND")
                                    
                                    st.markdown("")  # Spacing
                                    
                                    # ===========================================================
                                    # OVERALL MARKET SIGNAL
                                    # ===========================================================
                                    st.markdown("### 📊 Overall Technical Signal")
                                    
                                    if total_signals > 0:
                                        bullish_pct = (bullish_signals / (total_signals * 3)) * 100  # 3 indicators per timeframe
                                        bearish_pct = (bearish_signals / (total_signals * 3)) * 100
                                        
                                        if bullish_signals > bearish_signals:
                                            st.success(f"### 🟢 BULLISH TREND")
                                            st.markdown(f"**Strength:** {bullish_pct:.1f}% of indicators showing bullish signals")
                                            st.markdown("**Recommendation:** Consider CALL options or bullish strategies")
                                        elif bearish_signals > bullish_signals:
                                            st.error(f"### 🔴 BEARISH TREND")
                                            st.markdown(f"**Strength:** {bearish_pct:.1f}% of indicators showing bearish signals")
                                            st.markdown("**Recommendation:** Consider PUT options or bearish strategies")
                                        else:
                                            st.info(f"### ⚪ NEUTRAL / MIXED SIGNALS")
                                            st.markdown("**Recommendation:** Wait for clearer directional signals before trading")
                                    else:
                                        st.warning("Insufficient data to determine overall market signal")

                                    # Store for consensus
                                    st.session_state['bullish_signals'] = bullish_signals
                                    st.session_state['bearish_signals'] = bearish_signals

                                    st.markdown("---")

                                    # ==============================================================
                                    # SECTION 3: Moving Averages Analysis
                                    # ==============================================================
                                    st.markdown("### 📈 Moving Averages Analysis")
                                    
                                    if 'daydata' in mtf_data and mtf_data['daydata'] is not None:
                                        df_daily = mtf_data['daydata']
                                        
                                        if len(df_daily) >= 50:  # Reduced requirement from 200 to 50
                                            close = df_daily['close']
                                            current_price = spot_price if spot_price else close.iloc[-1]
                                            
                                            # Calculate available MAs based on data length
                                            ma_data = {}
                                            
                                            # EMA 20 (needs 20+ candles)
                                            if len(df_daily) >= 20:
                                                ema20 = calculate_ema(close, 20).iloc[-1]
                                                ma_data['EMA 20'] = ema20
                                            
                                            # EMA 50 (needs 50+ candles)
                                            if len(df_daily) >= 50:
                                                ema50 = calculate_ema(close, 50).iloc[-1]
                                                ma_data['EMA 50'] = ema50
                                            
                                            # SMA 100 (needs 100+ candles) - Alternative to SMA 200
                                            if len(df_daily) >= 100:
                                                sma100 = calculate_sma(close, 100).iloc[-1]
                                                ma_data['SMA 100'] = sma100
                                            
                                            # SMA 200 (needs 200+ candles) - Only if available
                                            if len(df_daily) >= 200:
                                                sma200 = calculate_sma(close, 200).iloc[-1]
                                                ma_data['SMA 200'] = sma200
                                            
                                            # Display available moving averages
                                            if ma_data:
                                                # Create dynamic columns based on available MAs
                                                num_cols = len(ma_data)
                                                cols = st.columns(num_cols)
                                                
                                                for idx, (ma_name, ma_value) in enumerate(ma_data.items()):
                                                    with cols[idx]:
                                                        st.metric(ma_name, f"₹{ma_value:,.2f}")
                                                        
                                                        # Calculate distance from current price
                                                        delta = current_price - ma_value
                                                        delta_pct = (delta / ma_value) * 100
                                                        
                                                        if delta > 0:
                                                            st.success(f"✅ +{delta_pct:.2f}% above")
                                                        else:
                                                            st.error(f"❌ {delta_pct:.2f}% below")
                                                
                                                st.markdown("")  # Spacing
                                                
                                                # Overall MA Interpretation
                                                st.markdown("**📊 Moving Average Interpretation:**")
                                                
                                                above_count = sum(1 for ma_value in ma_data.values() if current_price > ma_value)
                                                total_count = len(ma_data)
                                                
                                                if above_count == total_count:
                                                    st.success("🟢 **Strong Bullish**: Price is above ALL moving averages (strong uptrend)")
                                                elif above_count >= total_count * 0.7:
                                                    st.success("🟢 **Bullish**: Price is above most moving averages (uptrend)")
                                                elif above_count <= total_count * 0.3:
                                                    st.error("🔴 **Bearish**: Price is below most moving averages (downtrend)")
                                                elif above_count == 0:
                                                    st.error("🔴 **Strong Bearish**: Price is below ALL moving averages (strong downtrend)")
                                                else:
                                                    st.info("⚪ **Neutral**: Mixed signals - price near moving averages")
                                                
                                                # Add data availability note
                                                st.caption(f"ℹ️ Analysis based on {len(df_daily)} days of historical data. " + 
                                                          ("SMA 200 not available (requires 200+ days)." if len(df_daily) < 200 else ""))
                                            
                                            else:
                                                st.warning("⚠️ Insufficient data for moving average analysis (need at least 20 days)")
                                        
                                        else:
                                            st.warning(f"⚠️ Insufficient daily data for MA analysis (have {len(df_daily)} days, need at least 50)")
                                    else:
                                        st.warning("⚠️ Daily data not available for moving average analysis")


                                    # Store for consensus
                                    st.session_state['ma_above_count'] = above_count
                                    st.session_state['ma_total_count'] = total_count

                                    st.markdown("---")
                                    
                                    # ==============================================================
                                    # SECTION 4: MACD Analysis
                                    # ==============================================================
                                    st.markdown("### 📊 MACD (Moving Average Convergence Divergence)")
                                    
                                    if 'daydata' in mtf_data and mtf_data['daydata'] is not None:
                                        df_daily = mtf_data['daydata']
                                        
                                        if len(df_daily) >= 26:
                                            # Calculate MACD
                                            macd_line, signal_line, histogram = calculate_macd(df_daily['close'])
                                            
                                            # Get latest values
                                            current_macd = macd_line.iloc[-1]
                                            current_signal = signal_line.iloc[-1]
                                            current_histogram = histogram.iloc[-1]
                                            
                                            # Previous values for crossover detection
                                            prev_macd = macd_line.iloc[-2] if len(macd_line) > 1 else current_macd
                                            prev_signal = signal_line.iloc[-2] if len(signal_line) > 1 else current_signal
                                            
                                            # Determine momentum
                                            if current_histogram > 0:
                                                momentum = "Bullish Momentum"
                                                momentum_color = "success"
                                                momentum_emoji = "🟢"
                                            else:
                                                momentum = "Bearish Momentum"
                                                momentum_color = "error"
                                                momentum_emoji = "🔴"
                                            
                                            # Display MACD values in columns
                                            col1, col2, col3, col4 = st.columns(4)
                                            
                                            with col1:
                                                st.metric("MACD Line", f"{current_macd:.2f}")
                                            
                                            with col2:
                                                st.metric("Signal Line", f"{current_signal:.2f}")
                                            
                                            with col3:
                                                st.metric("Histogram", f"{current_histogram:.2f}")
                                            
                                            with col4:
                                                if momentum_color == "success":
                                                    st.success(f"{momentum_emoji} {momentum}")
                                                else:
                                                    st.error(f"{momentum_emoji} {momentum}")
                                                st.caption("MACD above signal line" if current_histogram > 0 else "MACD below signal line")
                                            
                                            st.markdown("")  # Spacing
                                            
                                            # Crossover Status
                                            crossover_detected = False
                                            crossover_type = "No Crossover"
                                            
                                            if prev_macd <= prev_signal and current_macd > current_signal:
                                                crossover_detected = True
                                                crossover_type = "✅ Bullish Crossover (MACD above Signal)"
                                                st.success(f"**Crossover Status:** {crossover_type}")
                                            elif prev_macd >= prev_signal and current_macd < current_signal:
                                                crossover_detected = True
                                                crossover_type = "❌ Bearish Crossover (MACD below Signal)"
                                                st.error(f"**Crossover Status:** {crossover_type}")
                                            else:
                                                if current_histogram > 0:
                                                    st.info(f"**Crossover Status:** ✅ Bullish Crossover (MACD above Signal)")
                                                else:
                                                    st.warning(f"**Crossover Status:** ❌ No Crossover / Bearish")
                                            
                                            st.markdown("")  # Spacing
                                            
                                            # MACD Interpretation Guide
                                            st.markdown("**💡 MACD Interpretation:**")
                                            st.markdown(f"- **Histogram > 0:** Bullish momentum (MACD above signal)")
                                            st.markdown(f"- **Histogram < 0:** Bearish momentum (MACD below signal)")
                                            st.markdown(f"- **Crossovers:** Strong buy/sell signals when MACD crosses signal line")
                                            
                                            st.markdown("")  # Spacing
                                            
                                            # Detailed MACD Interpretation
                                            st.markdown("### 🔍 Detailed MACD Interpretation")
                                            
                                            col1, col2 = st.columns(2)
                                            
                                            with col1:
                                                # Crossover Status
                                                if current_macd > current_signal:
                                                    st.success("**Crossover Status:** 🟢 Bullish Crossover")
                                                    st.caption("MACD line is above Signal line")
                                                else:
                                                    st.error("**Crossover Status:** 🔴 Bearish Crossover")
                                                    st.caption("MACD line is below Signal line")
                                                
                                                # Histogram State
                                                if current_histogram > 0:
                                                    st.info(f"**Histogram State:** Positive (above zero)")
                                                    st.caption("Bullish momentum present")
                                                else:
                                                    st.warning(f"**Histogram State:** Negative (below zero)")
                                                    st.caption("Bearish momentum present")
                                            
                                            with col2:
                                                # Centerline Status
                                                if current_macd > 0:
                                                    st.success("**Centerline Status:** ✅ Above zero line - Long-term bullish trend")
                                                else:
                                                    st.error("**Centerline Status:** ❌ Below zero line - Long-term bearish trend")
                                                
                                                # Overall Trend Direction
                                                if current_histogram > 0 and current_macd > 0:
                                                    st.success("**Overall Trend:** 🟢 STRONG BULLISH")
                                                    st.caption("MACD momentum: Strong uptrend (consider CALL options)")
                                                elif current_histogram > 0:
                                                    st.success("**Overall Trend:** 🟢 BULLISH")
                                                    st.caption("MACD momentum: Moderate uptrend (consider CALL options)")
                                                elif current_histogram < 0 and current_macd < 0:
                                                    st.error("**Overall Trend:** 🔴 STRONG BEARISH")
                                                    st.caption("MACD momentum: Strong downtrend (consider PUT options)")
                                                else:
                                                    st.error("**Overall Trend:** 🔴 BEARISH")
                                                    st.caption("MACD momentum: Moderate downtrend (consider PUT options)")

                                            
                                            st.markdown("")  # Spacing
                                            
                                            # Understanding MACD Signals - Expandable
                                            with st.expander("📊 Understanding MACD Signals"):
                                                st.markdown("""
                                                **MACD Components:**
                                                - **MACD Line**: Difference between 12-period and 26-period EMAs
                                                - **Signal Line**: 9-period EMA of MACD line
                                                - **Histogram**: Difference between MACD and Signal line
                                                
                                                **Trading Signals:**
                                                1. **Bullish Crossover**: MACD crosses above Signal line → BUY signal
                                                2. **Bearish Crossover**: MACD crosses below Signal line → SELL signal
                                                3. **Centerline Cross**: MACD crosses zero line → Trend change confirmation
                                                4. **Divergence**: Price vs MACD divergence → Potential reversal
                                                
                                                **Interpretation:**
                                                - **Histogram growing**: Momentum increasing
                                                - **Histogram shrinking**: Momentum weakening
                                                - **Above zero**: Bullish trend
                                                - **Below zero**: Bearish trend
                                                """)
                                            
                                            st.markdown("")  # Spacing
                                            
                                            # Create MACD Chart
                                            macd_chart = chart_builder.create_macd_chart(
                                                df_daily,
                                                title=f"{index_symbol} - MACD Analysis (Daily)"
                                            )
                                            
                                            if macd_chart:
                                                st.plotly_chart(macd_chart, use_container_width=True)
                                            
                                            # Simple interpretation below chart
                                            if current_histogram > 0:
                                                st.success("🟢 **Bullish**: MACD line is above signal line (positive momentum)")
                                            else:
                                                st.error("🔴 **Bearish**: MACD line is below signal line (negative momentum)")
                                        
                                        else:
                                            st.warning("⚠️ Insufficient data for MACD (need 26+ candles)")
                                    else:
                                        st.warning("⚠️ Daily data not available for MACD analysis")
                                  
                                    # Store for consensus
                                    if current_histogram > 0:
                                        st.session_state['macd_signal'] = 'Bullish'
                                    else:
                                        st.session_state['macd_signal'] = 'Bearish'

                
                                    st.markdown("---")
                                    
                                    # ==============================================================
                                    # SECTION 5: Multi-Timeframe Technical Charts
                                    # ==============================================================
                                    st.markdown("### 📈 Multi-Timeframe Technical Charts")
                                    
                                    # Create tabs for different timeframes
                                    tab1, tab2, tab3, tab4 = st.tabs(["📊 Daily", "⏰ 1-Hour", "⏰ 15-Min", "⚡ 5-Min"])
                                    
                                    # Tab 1: Daily Chart
                                    with tab1:
                                        st.markdown("#### Daily Timeframe Analysis")
                                        
                                        if 'daydata' in mtf_data and mtf_data['daydata'] is not None and not mtf_data['daydata'].empty:
                                            df_daily = mtf_data['daydata']
                                            
                                            # Create daily chart
                                            daily_chart = chart_builder.create_macd_chart(df_daily, title=f"{index_symbol} - Daily Analysis")
                                            if daily_chart:
                                                st.plotly_chart(daily_chart, use_container_width=True)
                                            else:
                                                st.warning("Unable to generate daily chart")
                                        else:
                                            st.warning("Daily data not available")
                                    
                                    # Tab 2: 1-Hour Chart
                                    with tab2:
                                        st.markdown("#### 1-Hour Timeframe Analysis")
                                        
                                        if '60mindata' in mtf_data and mtf_data['60mindata'] is not None and not mtf_data['60mindata'].empty:
                                            df_60min = mtf_data['60mindata']
                                            
                                            # Create 1-hour chart
                                            hourly_chart = chart_builder.create_macd_chart(df_60min, title=f"{index_symbol} - 1-Hour Analysis")
                                            if hourly_chart:
                                                st.plotly_chart(hourly_chart, use_container_width=True)
                                            else:
                                                st.warning("Unable to generate 1-hour chart")
                                        else:
                                            st.info("ℹ️ 1-hour data not available (intraday data may not be available for indices)")
                                    
                                    # Tab 3: 15-Minute Chart
                                    with tab3:
                                        st.markdown("#### 15-Minute Timeframe Analysis")
                                        
                                        if '15mindata' in mtf_data and mtf_data['15mindata'] is not None and not mtf_data['15mindata'].empty:
                                            df_15min = mtf_data['15mindata']
                                            
                                            # Create 15-min chart
                                            min15_chart = chart_builder.create_macd_chart(df_15min, title=f"{index_symbol} - 15-Minute Analysis")
                                            if min15_chart:
                                                st.plotly_chart(min15_chart, use_container_width=True)
                                            else:
                                                st.warning("Unable to generate 15-minute chart")
                                        else:
                                            st.info("ℹ️ 15-minute data not available (intraday data may not be available for indices)")
                                    
                                    # Tab 4: 5-Minute Chart
                                    with tab4:
                                        st.markdown("#### 5-Minute Timeframe Analysis")
                                        
                                        if '5mindata' in mtf_data and mtf_data['5mindata'] is not None and not mtf_data['5mindata'].empty:
                                            df_5min = mtf_data['5mindata']
                                            
                                            # Create 5-min chart
                                            min5_chart = chart_builder.create_macd_chart(df_5min, title=f"{index_symbol} - 5-Minute Analysis")
                                            if min5_chart:
                                                st.plotly_chart(min5_chart, use_container_width=True)
                                            else:
                                                st.warning("Unable to generate 5-minute chart")
                                        else:
                                            st.info("ℹ️ 5-minute data not available (intraday data may not be available for indices)")
                                    
                                    st.markdown("---")

                                    # ==============================================================
                                    # SECTION 6 & 7: Latest News & News Sentiment Breakdown
                                    # ==============================================================
                                    st.markdown("### 📰 Latest News")
                                    
                                    # Clean symbol for news search
                                    clean_symbol = index_symbol.replace('NSE:', '').replace('BSE:', '').replace('MCX:', '').strip()
                                    
                                    # Fetch news with better error handling
                                    with st.spinner("🔄 Fetching latest news..."):
                                        try:
                                            news_articles = news_fetcher.fetch_news(clean_symbol, max_articles=10)
                                        except Exception as e:
                                            st.error(f"Error fetching news: {str(e)}")
                                            news_articles = []
                                    
                                    # Display news articles
                                    if news_articles and len(news_articles) > 0:
                                        # Check if we have real news (not error placeholder)
                                        has_real_news = not any('Unable to fetch' in article.get('title', '') or 
                                                                'No recent news' in article.get('title', '') 
                                                                for article in news_articles)
                                        
                                        if has_real_news:
                                            # Show top 5 news headlines
                                            for idx, article in enumerate(news_articles[:5], 1):
                                                sentiment_emoji = {
                                                    'Positive': '🟢',
                                                    'Negative': '🔴',
                                                    'Neutral': '⚪'
                                                }.get(article.get('sentiment', 'Neutral'), '⚪')
                                                
                                                title = article.get('title', 'No title')
                                                source = article.get('source', 'Unknown')
                                                date = article.get('date', 'Unknown')
                                                link = article.get('link', '#')
                                                
                                                with st.expander(f"{sentiment_emoji} {title[:100]}..."):
                                                    st.markdown(f"**Source:** {source}")
                                                    st.markdown(f"**Date:** {date}")
                                                    st.markdown(f"**Sentiment:** {article.get('sentiment', 'Neutral')}")
                                                    if link != '#':
                                                        st.markdown(f"[Read Full Article]({link})")
                                            
                                            st.markdown("")  # Spacing
                                            
                                            # ==============================================================
                                            # News Sentiment Breakdown
                                            # ==============================================================
                                            st.markdown("### 🎯 News Sentiment Breakdown")
                                            
                                            sentiment_summary = news_fetcher.get_sentiment_summary(news_articles)
                                            
                                            # Calculate overall score
                                            total_articles = sentiment_summary['total']
                                            if total_articles > 0:
                                                positive_score = sentiment_summary['positive']
                                                negative_score = sentiment_summary['negative']
                                                positive_pct = sentiment_summary['positive_pct']
                                                negative_pct = sentiment_summary['negative_pct']
                                                
                                                # Weighted score (-1 to +1 scale)
                                                net_score = (positive_score - negative_score) / total_articles
                                                
                                                # Determine overall sentiment (IMPROVED THRESHOLDS)
                                                if net_score > 0.2:  # Changed from 0.3 to 0.2 (more sensitive)
                                                    overall_label = "Positive"
                                                    overall_color = "success"
                                                    overall_emoji = "🟢"
                                                elif net_score < -0.2:  # Changed from -0.3 to -0.2 (more sensitive)
                                                    overall_label = "Negative"
                                                    overall_color = "error"
                                                    overall_emoji = "🔴"
                                                else:
                                                    overall_label = "Neutral"
                                                    overall_color = "info"
                                                    overall_emoji = "⚪"
                                                
                                                # Display sentiment breakdown in columns
                                                col1, col2, col3, col4 = st.columns(4)
                                                
                                                with col1:
                                                    st.markdown("**Overall**")
                                                    if overall_color == "success":
                                                        st.success(overall_label)
                                                    elif overall_color == "error":
                                                        st.error(overall_label)
                                                    else:
                                                        st.info(overall_label)
                                                
                                                with col2:
                                                    st.markdown("**Score**")
                                                    st.metric("", f"{net_score:.3f}")
                                                
                                                with col3:
                                                    st.markdown("**✅ Positive**")
                                                    st.metric("", positive_score, f"{positive_pct}%")
                                                
                                                with col4:
                                                    st.markdown("**❌ Negative**")
                                                    st.metric("", negative_score, f"{negative_pct}%")
                                                
                                                st.markdown("")  # Spacing
                                                
                                                # =========================================================
                                                # NEWS-BASED MARKET TREND PREDICTION (ENHANCED!)
                                                # =========================================================
                                                st.markdown("### 📊 News-Based Market Trend Prediction")
                                                
                                                # Determine trend with better logic
                                                if positive_pct >= 60:
                                                    # Strong bullish (60%+ positive)
                                                    st.success("## 🟢 STRONG BULLISH TREND")
                                                    st.markdown(f"**{positive_pct}% of news articles are positive** - Market sentiment is overwhelmingly bullish")
                                                    st.markdown("**📈 Market Direction:** Expected to move **UPWARD**")
                                                    st.markdown("")
                                                    st.info("💡 **Trading Strategy:** Strong buy signal for CALL options. Consider ATM or slightly OTM strikes.")
                                                
                                                elif positive_pct > negative_pct and positive_pct >= 40:
                                                    # Moderate bullish (40-60% positive, more positive than negative)
                                                    st.success("## 🟢 BULLISH TREND")
                                                    st.markdown(f"**{positive_pct}% positive vs {negative_pct}% negative** - Bullish sentiment prevailing")
                                                    st.markdown("**📈 Market Direction:** Likely to trend **BULLISH**")
                                                    st.markdown("")
                                                    st.info("💡 **Trading Strategy:** Consider CALL options. Wait for price confirmation before entry.")
                                                
                                                elif negative_pct >= 60:
                                                    # Strong bearish (60%+ negative)
                                                    st.error("## 🔴 STRONG BEARISH TREND")
                                                    st.markdown(f"**{negative_pct}% of news articles are negative** - Market sentiment is overwhelmingly bearish")
                                                    st.markdown("**📉 Market Direction:** Expected to move **DOWNWARD**")
                                                    st.markdown("")
                                                    st.info("💡 **Trading Strategy:** Strong sell signal for PUT options. Consider ATM or slightly OTM strikes.")
                                                
                                                elif negative_pct > positive_pct and negative_pct >= 40:
                                                    # Moderate bearish (40-60% negative, more negative than positive)
                                                    st.error("## 🔴 BEARISH TREND")
                                                    st.markdown(f"**{negative_pct}% negative vs {positive_pct}% positive** - Bearish sentiment prevailing")
                                                    st.markdown("**📉 Market Direction:** Likely to trend **BEARISH**")
                                                    st.markdown("")
                                                    st.info("💡 **Trading Strategy:** Consider PUT options. Wait for price confirmation before entry.")
                                                
                                                else:
                                                    # Neutral/Mixed (neither dominates)
                                                    st.info("## ⚪ NEUTRAL / MIXED SIGNALS")
                                                    st.markdown(f"**Mixed sentiment:** {positive_pct}% positive, {negative_pct}% negative, {sentiment_summary['neutral_pct']}% neutral")
                                                    st.markdown("**➡️ Market Direction:** **UNCERTAIN** based on news alone")
                                                    st.markdown("")
                                                    st.warning("⚠️ **Trading Strategy:** Avoid directional trades. Wait for clearer news sentiment OR rely on technical indicators for guidance.")
                                                
                                                # Per-Article Details Section
                                                st.markdown("")  # Spacing
                                                st.markdown(f"### 📊 Per-Article Details ({total_articles} articles)")
                                                
                                                for idx, article in enumerate(news_articles, 1):
                                                    sentiment = article.get('sentiment', 'Neutral')
                                                    sentiment_emoji = {
                                                        'Positive': '✅',
                                                        'Negative': '❌',
                                                        'Neutral': '⚪'
                                                    }.get(sentiment, '⚪')
                                                    
                                                    with st.expander(f"{sentiment_emoji} Article {idx}: {article.get('title', '')[:80]}..."):
                                                        st.markdown(f"**Title:** {article.get('title', 'No title')}")
                                                        st.markdown(f"**Source:** {article.get('source', 'Unknown')}")
                                                        st.markdown(f"**Date:** {article.get('date', 'Unknown')}")
                                                        st.markdown(f"**Sentiment:** {sentiment}")
                                                        if article.get('link') and article['link'] != '#':
                                                            st.markdown(f"[Read Full Article]({article['link']})")
                                            
                                            else:
                                                st.info("No articles available for sentiment analysis")
                                        
                                        else:
                                            # Placeholder news - show helpful message
                                            st.info(f"ℹ️ Unable to fetch recent news for {clean_symbol}. This could be due to:")
                                            st.markdown("- API rate limiting")
                                            st.markdown("- Network connectivity issues")
                                            st.markdown("- Symbol not found in news sources")
                                            st.markdown("")
                                            st.caption("💡 You can still rely on technical indicators for trading decisions")
                                    
                                    else:
                                        st.warning("⚠️ No news articles available at this time")

                                    # Store for consensus
                                    if net_score > 0.2:
                                        st.session_state['news_sentiment'] = 'Bullish'
                                    elif net_score < -0.2:
                                        st.session_state['news_sentiment'] = 'Bearish'
                                    else:
                                        st.session_state['news_sentiment'] = 'Neutral'


                
                                    st.markdown("---")

                                    
                                    # ==============================================================
                                    # SECTION 8: Fibonacci Retracement & Extension Levels
                                    # ==============================================================
                                    st.markdown("### 📐 Fibonacci Retracement & Extension Levels")
                                    
                                    if 'daydata' in mtf_data and mtf_data['daydata'] is not None:
                                        df_daily = mtf_data['daydata']
                                        
                                        fib_levels = fib_calculator.calculate_levels(df_daily)
                                        
                                        if fib_levels:
                                            current_price = spot_price if spot_price else df_daily['close'].iloc[-1]
                                            
                                            # Overall Trend Signal (NEW - As requested)
                                            if fib_levels['trend'] == 'uptrend':
                                                st.success("**📊 Overall Fibonacci Trend:** 🟢 **BULLISH / UPTREND**")
                                                st.caption("Price is in upward momentum. Look for retracement entries for CALL options.")
                                            else:
                                                st.error("**📊 Overall Fibonacci Trend:** 🔴 **BEARISH / DOWNTREND**")
                                                st.caption("Price is in downward momentum. Look for retracement entries for PUT options.")
                                            
                                            st.markdown("")  # Spacing
                                            
                                            # Create 2-column layout
                                            col1, col2 = st.columns([2, 1])
                                            
                                            # === LEFT COLUMN: Trend & Levels ===
                                            with col1:
                                                st.markdown("### 📊 Trend & Levels")
                                                
                                                # Trend indicator
                                                if fib_levels['trend'] == 'uptrend':
                                                    st.success(f"**Trend:** 🟢 UPTREND")
                                                else:
                                                    st.error(f"**Trend:** 🔴 DOWNTREND")
                                                
                                                st.markdown(f"**Swing High:** ₹{fib_levels['swing_high']:,.2f}")
                                                st.markdown(f"**Swing Low:** ₹{fib_levels['swing_low']:,.2f}")
                                                
                                                st.markdown("")  # Spacing
                                                st.markdown("#### 🎯 Key Retracement Levels:")
                                                
                                                # Display Fibonacci levels with checkmarks for levels near current price
                                                for level, price in fib_levels['fib_levels'].items():
                                                    distance = abs(current_price - price)
                                                    distance_pct = (distance / current_price) * 100
                                                    
                                                    # Check if near current price (within 1%)
                                                    if distance_pct < 1.0:
                                                        st.success(f"✅ **Fib {level}:** ₹{price:,.2f} ← Near Current Price")
                                                    else:
                                                        st.markdown(f"• **Fib {level}:** ₹{price:,.2f}")
                                                
                                                st.markdown("")  # Spacing
                                                
                                                # How to use Fibonacci guide
                                                with st.expander("💡 How to use Fibonacci:"):
                                                    st.markdown("""
                                                    **Uptrend:** Price retraces to 0.382, 0.5, or 0.618 → Buy opportunity
                                                    
                                                    **Downtrend:** Price rallies to 0.382, 0.5, or 0.618 → Sell opportunity
                                                    
                                                    **Extension levels** (1.272, 1.618, 2.0) → Profit targets
                                                    
                                                    **Key Levels:**
                                                    - **0.382 (38.2%)**: Shallow retracement, strong trend
                                                    - **0.500 (50.0%)**: Moderate retracement, common entry
                                                    - **0.618 (61.8%)**: Golden ratio, major support/resistance
                                                    - **0.786 (78.6%)**: Deep retracement, trend weakening
                                                    """)
                                            
                                            # === RIGHT COLUMN: Nearest Fib Targets ===
                                            with col2:
                                                st.markdown("### 🎯 Nearest Fib Targets")
                                                
                                                # Calculate distances and sort by nearest
                                                all_fib_prices = []
                                                
                                                # Add retracement levels
                                                for level, price in fib_levels['fib_levels'].items():
                                                    distance = price - current_price
                                                    all_fib_prices.append({
                                                        'level': level,
                                                        'price': price,
                                                        'distance': distance,
                                                        'abs_distance': abs(distance)
                                                    })
                                                
                                                # Add extension targets
                                                for target in fib_levels['targets']:
                                                    distance = target['price'] - current_price
                                                    all_fib_prices.append({
                                                        'level': target['level'],
                                                        'price': target['price'],
                                                        'distance': distance,
                                                        'abs_distance': abs(distance)
                                                    })
                                                
                                                # Sort by absolute distance (nearest first)
                                                all_fib_prices.sort(key=lambda x: x['abs_distance'])
                                                
                                                # Display top 3 nearest targets
                                                for i, target in enumerate(all_fib_prices[:3]):
                                                    level_name = f"Fib {target['level']}"
                                                    
                                                    st.markdown(f"**{level_name}**")
                                                    st.metric(
                                                        label="",
                                                        value=f"₹{target['price']:,.2f}",
                                                        delta=f"{'+' if target['distance'] > 0 else ''}₹{target['distance']:.2f}"
                                                    )
                                                    st.markdown("")  # Spacing between targets
                                            
                                            st.markdown("---")
                                            
                                            # Extension Targets Table (Keep existing table format)
                                            st.markdown("### 🎯 Extension Targets (Profit Levels):")
                                            
                                            target_data = []
                                            for target in fib_levels['targets']:
                                                target_data.append({
                                                    'Level': target['level'],
                                                    'Target Price': f"₹{target['price']:,.2f}",
                                                    'Distance': f"₹{target['distance']:,.2f}"
                                                })
                                            
                                            if target_data:
                                                target_df = pd.DataFrame(target_data)
                                                st.dataframe(target_df, use_container_width=True, hide_index=True)
                                        
                                        else:
                                            st.warning("⚠️ Unable to calculate Fibonacci levels (insufficient data)")
                                    else:
                                        st.warning("⚠️ Daily data not available for Fibonacci calculation")

                                    # Store for consensus
                                    st.session_state['fib_trend'] = fib_levels['trend']
                                    
                                    st.markdown("---")

                                
                                else:
                                    st.warning("⚠️ No historical data available for analysis. Please try again or check API connection.")
                            else:
                                st.error(f"❌ Unable to find instrument token for {index_symbol}")
                
                except Exception as e:
                    st.error(f"❌ Advanced analysis error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

# ============================================================================
# LIVE MONITOR TAB
# ============================================================================

def render_live_monitor_tab():
    """Render live data monitoring for subscribed options"""
    st.header("📡 Live Options Monitor")
    
    # Check watchlist
    watchlist_symbols = get_streaming_handler().get_subscribed_symbols()
    
    if not watchlist_symbols:
        st.info("📝 No options subscribed. Subscribe to options from Index Options tab.")
        return
    
    st.success(f"Monitoring {len(watchlist_symbols)} instruments")
    
    # Create placeholder for live updates
    live_placeholder = st.empty()
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("🔄 Auto-refresh (every 5 seconds)", value=True)
    
    if auto_refresh:
        # Display live data
        while auto_refresh:
            with live_placeholder.container():
                cols = st.columns(min(3, len(watchlist_symbols)))
                
                for idx, symbol in enumerate(watchlist_symbols):
                    with cols[idx % 3]:
                        # Get latest tick data
                        ticks_df = get_latest_ticks(symbol, limit=1)
                        
                        if not ticks_df.empty:
                            latest = ticks_df.iloc[0]
                            
                            # Display as metric card
                            st.metric(
                                label=symbol,
                                value=f"₹{latest['last_price']:.2f}",
                                delta=f"{latest['change']:.2f}%" if pd.notna(latest['change']) else None
                            )
                            
                            st.caption(f"Vol: {latest['volume']:,}" if pd.notna(latest['volume']) else "Vol: N/A")
                            st.caption(f"Time: {latest['timestamp']}")
                        else:
                            st.metric(label=symbol, value="No data")
            
            time.sleep(5)
            st.rerun()
    else:
        # Static display
        cols = st.columns(min(3, len(watchlist_symbols)))
        
        for idx, symbol in enumerate(watchlist_symbols):
            with cols[idx % 3]:
                ticks_df = get_latest_ticks(symbol, limit=1)
                
                if not ticks_df.empty:
                    latest = ticks_df.iloc[0]
                    st.metric(
                        label=symbol,
                        value=f"₹{latest['last_price']:.2f}",
                        delta=f"{latest['change']:.2f}%" if pd.notna(latest['change']) else None
                    )
                    st.caption(f"Vol: {latest['volume']:,}" if pd.notna(latest['volume']) else "Vol: N/A")
                else:
                    st.metric(label=symbol, value="No data")

# ============================================================================
# TRADE HISTORY TAB
# ============================================================================

def render_trade_history_tab():
    """Render trade history and logs"""
    st.header("📜 Trade History")
    
    trades_df = get_trade_history(limit=100)
    
    if not trades_df.empty:
        st.success(f"Showing {len(trades_df)} recent trades")
        
        st.dataframe(
            trades_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No trades executed yet")

# ============================================================================
# COMBINED ANALYSIS TAB (NEW)
# ============================================================================

def render_combined_analysis_tab():
    """
    Combined Analysis Tab - Automated contract recommendation
    No user inputs required - fully automated
    """
    st.header("🎯 Automated Contract Recommendation")
    st.caption("Multi-timeframe trend analysis with ITM contract selection")
    
    # Check prerequisites
    if not st.session_state.get('options_chain'):
        st.info("⏳ Please load options chain first:")
        st.write("1. Go to 'Index Options' tab")
        st.write("2. Select an index")
        st.write("3. Click 'Load Options Chain'")
        st.write("4. Then return here for automated analysis")
        return
    
    # Get loaded data
    chain_data = st.session_state.options_chain
    index_symbol = chain_data['index']
    spot_price = chain_data.get('index_price')
    calls_df = chain_data['calls']
    puts_df = chain_data['puts']
    
    if spot_price is None:
        st.error("Spot price not available. Please reload options chain.")
        return
    
    # Display current info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Index", index_symbol)
    with col2:
        st.metric("Spot Price", f"₹{spot_price:,.2f}")
    with col3:
        current_time = datetime.now().strftime("%H:%M:%S")
        st.metric("Time", current_time)
    
    st.markdown("---")
    
    # Analysis button
    if st.button("🔍 Analyze Market & Get Recommendation", type="primary", use_container_width=True):
        
        # Import modules
        from trend_analyzer import TrendAnalyzer
        from strike_selector import StrikeSelector
        
        kite = get_kite_handler()
        
        # Step 1: Multi-timeframe Trend Analysis
        with st.spinner("📊 Analyzing trend across Daily, Hourly, and 15-min timeframes..."):
            analyzer = TrendAnalyzer(kite)
            
            try:
                trend_analysis = analyzer.analyze_trend(index_symbol, spot_price)
            except Exception as e:
                st.error(f"Error in trend analysis: {str(e)}")
                st.info("This might be due to insufficient historical data or API limits.")
                with st.expander("🔍 Error Details"):
                    st.exception(e)
                return
        
        # Step 2: Strike Selection
        with st.spinner("🎯 Selecting optimal option contracts..."):
            selector = StrikeSelector()
            recommendation = selector.select_contract(
                trend_analysis,
                calls_df,
                puts_df,
                spot_price
            )
        
        # Store in session
        st.session_state.recommendation = {
            'trend': trend_analysis,
            'contracts': recommendation,
            'timestamp': datetime.now().isoformat()
        }
        
        st.success("✅ Analysis Complete!")
    
    # Display results if available
    if 'recommendation' in st.session_state:
        rec = st.session_state.recommendation
        trend = rec['trend']
        contracts = rec['contracts']
        
        st.markdown("---")
        
        # SECTION 1: Trend Analysis Summary
        st.subheader("📊 Market Trend Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            direction_emoji = "🟢" if trend['direction'] == "BULLISH" else "🔴" if trend['direction'] == "BEARISH" else "🟡"
            st.metric("Direction", f"{direction_emoji} {trend['direction']}")
        
        with col2:
            conf_color = "🟢" if trend['confidence'] == "HIGH" else "🟡" if trend['confidence'] == "MODERATE" else "🔴"
            st.metric("Confidence", f"{conf_color} {trend['confidence']}")
        
        with col3:
            st.metric("Action", trend['action'])
        
        with col4:
            score = trend['combined_score']
            score_display = f"{score:+.2f}"
            st.metric("Combined Score", score_display)
        
        # Detailed Timeframe Analysis
        with st.expander("📈 Detailed Timeframe Breakdown", expanded=False):
            tf_data = trend['timeframe_analysis']
            
            tab1, tab2, tab3 = st.tabs(["Daily (40%)", "Hourly (30%)", "15-Min (30%)"])
            
            with tab1:
                st.write("**Daily Timeframe Analysis:**")
                st.json(tf_data['daily'])
            
            with tab2:
                st.write("**Hourly Timeframe Analysis:**")
                st.json(tf_data['hourly'])
            
            with tab3:
                st.write("**15-Minute Timeframe Analysis:**")
                st.json(tf_data['15min'])
        
        st.markdown("---")
        
        # SECTION 2: Contract Recommendations
        st.subheader("🎯 Option Contract Recommendations")
        
        if 'error' in contracts:
            st.warning(f"⚠️ {contracts['error']}")
            if 'recommendation' in contracts:
                st.info(f"💡 {contracts['recommendation']}")
            if 'reason' in contracts:
                st.caption(contracts['reason'])
        else:
            # Display tabs for all three options
            tab1, tab2, tab3 = st.tabs([
                "✅ ITM (RECOMMENDED)", 
                "⚖️ ATM (Reference)", 
                "🎲 OTM (Reference)"
            ])
            
            with tab1:
                st.success("**🎯 FINAL RECOMMENDATION - Trade This Contract**")
                
                itm = contracts['recommended']
                
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Trading Symbol", itm['tradingsymbol'])
                    st.metric("Type", contracts['type'])
                    st.metric("Direction", contracts['direction'])
                
                with col_b:
                    st.metric("Strike Price", f"₹{itm['strike']:,.0f}")
                    st.metric("Lot Size", itm['lot_size'])
                    st.metric("Expiry", itm['expiry'])
                
                with col_c:
                    st.metric("Moneyness", itm['moneyness'])
                    st.metric("Distance from Spot", f"₹{itm['distance_from_spot']:.0f}")
                    st.metric("Intrinsic Value", f"₹{itm['intrinsic_value']:.0f}")
                
                st.info(f"**💡 Why ITM?** {contracts['recommendation_reason']}")
                
                st.markdown("**Contract Details:**")
                st.write(f"- **Instrument Token:** {itm['instrument_token']}")
                st.write(f"- **Days to Expiry:** {contracts['days_to_expiry']}")
                st.write(f"- **Percentage from Spot:** {itm['percentage_from_spot']:.2f}%")
            
            with tab2:
                st.info("**ATM Option - For Reference Only**")
                atm = contracts['options']['ATM']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Symbol:** {atm['tradingsymbol']}")
                    st.write(f"**Strike:** ₹{atm['strike']:,.0f}")
                    st.write(f"**Lot Size:** {atm['lot_size']}")
                
                with col2:
                    st.write(f"**Moneyness:** {atm['moneyness']}")
                    st.write(f"**Distance:** ₹{atm['distance_from_spot']:.0f}")
                    st.write(f"**Expiry:** {atm['expiry']}")
            
            with tab3:
                st.info("**OTM Option - For Reference Only**")
                otm = contracts['options']['OTM']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Symbol:** {otm['tradingsymbol']}")
                    st.write(f"**Strike:** ₹{otm['strike']:,.0f}")
                    st.write(f"**Lot Size:** {otm['lot_size']}")
                
                with col2:
                    st.write(f"**Moneyness:** {otm['moneyness']}")
                    st.write(f"**Distance:** ₹{otm['distance_from_spot']:.0f}")
                    st.write(f"**Expiry:** {otm['expiry']}")
        
        # Analysis timestamp
        st.caption(f"Analysis completed at: {rec['timestamp']}")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    
    # Render sidebar
    render_sidebar()
    
    # Initialize app on first run
    if not st.session_state.initialized:
        initialize_app()
    
    # Main content area with tabs (Stock Screener removed)
    tab1, tab2, tab3 = st.tabs([
        "📊 Index Options",
        "📡 Live Monitor",
        "📜 Trade History"
    ])
    
    with tab1:
        render_index_options_tab()
    
    with tab2:
        render_live_monitor_tab()
    
    with tab3:
        render_trade_history_tab()
    
    # Footer
    st.markdown("---")
    st.caption("🚀 Index Options Trading Platform | Data: Kite Connect | Made with Streamlit")

if __name__ == "__main__":
    main()
