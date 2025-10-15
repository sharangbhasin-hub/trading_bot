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
from indicators import calculate_rsi, calculate_macd, calculate_ema, calculate_sma

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
            with st.expander("📊 Analysis", expanded=False):
                
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
                                    # SECTION 1: Pattern Detection & Trade Confirmation
                                    # ==============================================================
                                    st.markdown("### 🎯 Pattern Detection & Trade Confirmation")
                                    
                                    # Detect patterns
                                    all_patterns = pattern_detector.detect_all_patterns(
                                        df_5min,
                                        support=df_5min['low'].min(),
                                        resistance=df_5min['high'].max()
                                    )
                                    
                                    # Display strongest pattern
                                    if all_patterns and all_patterns[0]['pattern'] != 'No Significant Pattern':
                                        strongest = all_patterns[0]
                                        pattern_type = strongest['type']
                                        
                                        # Color-coded display
                                        if pattern_type == 'bullish':
                                            st.success(f"**🟢 {strongest['pattern']}** detected (Strength: {strongest['strength']}%)")
                                        elif pattern_type == 'bearish':
                                            st.error(f"**🔴 {strongest['pattern']}** detected (Strength: {strongest['strength']}%)")
                                        else:
                                            st.info(f"**⚪ {strongest['pattern']}** detected")
                                        
                                        st.markdown(f"*{strongest['description']}*")
                                    else:
                                        st.info("No significant candlestick pattern detected")
                                    
                                    # 5-Point Trade Confirmation Checklist
                                    st.markdown("#### ✅ 5-Point Trade Confirmation Checklist")
                                    
                                    # Prepare analysis results for checklist
                                    analysis_results = {
                                        '5mdata': df_5min,
                                        'support': df_5min['low'].min(),
                                        'resistance': df_5min['high'].max(),
                                        'latest_price': spot_price,
                                        'all_patterns': all_patterns,
                                        'candlestick_pattern': all_patterns[0]['pattern'] if all_patterns else 'None',
                                        'pattern_type': all_patterns[0]['type'] if all_patterns else 'neutral',
                                        'rsi': calculate_rsi(df_5min['close'], 14).iloc[-1] if len(df_5min) >= 14 else 50
                                    }
                                    
                                    # Add MACD values
                                    if len(df_5min) >= 26:
                                        macd_line, signal_line, histogram = calculate_macd(df_5min['close'])
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
                                        # Display checklist items
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            st.markdown(f"**{checklist['1. At Key S/R Level']}**")
                                            st.markdown(f"**{checklist['2. Price Rejection']}**")
                                            st.markdown(f"**{checklist['3. Chart Pattern Confirmed']}**")
                                        
                                        with col2:
                                            st.markdown(f"**{checklist['4. Candlestick Signal']}**")
                                            st.markdown(f"**{checklist['5. Indicator Alignment']}**")
                                        
                                        # Final signal
                                        signal = checklist['FINAL_SIGNAL']
                                        if '🟢 BUY' in signal:
                                            st.success(f"### {signal}")
                                        elif '🔴 SELL' in signal:
                                            st.error(f"### {signal}")
                                        else:
                                            st.info(f"### {signal}")
                                    
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
                                    
                                    if summary_data:
                                        summary_df = pd.DataFrame(summary_data)
                                        st.dataframe(summary_df, use_container_width=True, hide_index=True)
                                    else:
                                        st.warning("Insufficient data for technical indicators")
                                    
                                    st.markdown("---")
                                    
                                    # ==============================================================
                                    # SECTION 3: Moving Averages Analysis
                                    # ==============================================================
                                    st.markdown("### 📈 Moving Averages Analysis")
                                    
                                    if 'daydata' in mtf_data and mtf_data['daydata'] is not None:
                                        df_daily = mtf_data['daydata']
                                        
                                        if len(df_daily) >= 200:
                                            close = df_daily['close']
                                            ema20 = calculate_ema(close, 20).iloc[-1]
                                            ema50 = calculate_ema(close, 50).iloc[-1]
                                            sma200 = calculate_sma(close, 200).iloc[-1]
                                            
                                            col1, col2, col3 = st.columns(3)
                                            
                                            with col1:
                                                st.metric("EMA 20", f"₹{ema20:,.2f}")
                                                delta_ema20 = ((spot_price - ema20) / ema20) * 100
                                                if delta_ema20 > 0:
                                                    st.success(f"+{delta_ema20:.2f}% above")
                                                else:
                                                    st.error(f"{delta_ema20:.2f}% below")
                                            
                                            with col2:
                                                st.metric("EMA 50", f"₹{ema50:,.2f}")
                                                delta_ema50 = ((spot_price - ema50) / ema50) * 100
                                                if delta_ema50 > 0:
                                                    st.success(f"+{delta_ema50:.2f}% above")
                                                else:
                                                    st.error(f"{delta_ema50:.2f}% below")
                                            
                                            with col3:
                                                st.metric("SMA 200", f"₹{sma200:,.2f}")
                                                delta_sma200 = ((spot_price - sma200) / sma200) * 100
                                                if delta_sma200 > 0:
                                                    st.success(f"+{delta_sma200:.2f}% above")
                                                else:
                                                    st.error(f"{delta_sma200:.2f}% below")
                                        else:
                                            st.warning("Insufficient daily data for MA analysis (need 200+ days)")
                                    else:
                                        st.warning("Daily data not available")
                                    
                                    st.markdown("---")
                                    
                                    # ==============================================================
                                    # SECTION 4: MACD Analysis
                                    # ==============================================================
                                    st.markdown("### 📊 MACD (Moving Average Convergence Divergence)")
                                    
                                    if 'daydata' in mtf_data and mtf_data['daydata'] is not None:
                                        df_daily = mtf_data['daydata']
                                        
                                        if len(df_daily) >= 26:
                                            macd_chart = chart_builder.create_macd_chart(
                                                df_daily,
                                                title=f"{index_symbol} - MACD Analysis (Daily)"
                                            )
                                            
                                            if macd_chart:
                                                st.plotly_chart(macd_chart, use_container_width=True)
                                            
                                            # MACD interpretation
                                            macd_line, signal_line, histogram = calculate_macd(df_daily['close'])
                                            current_histogram = histogram.iloc[-1]
                                            
                                            if current_histogram > 0:
                                                st.success("🟢 **Bullish**: MACD line is above signal line (positive momentum)")
                                            else:
                                                st.error("🔴 **Bearish**: MACD line is below signal line (negative momentum)")
                                        else:
                                            st.warning("Insufficient data for MACD (need 26+ candles)")
                                    else:
                                        st.warning("Daily data not available")
                                    
                                    st.markdown("---")
                                    
                                    # ==============================================================
                                    # SECTION 5: Multi-Timeframe Technical Charts
                                    # ==============================================================
                                    st.markdown("### 📈 Multi-Timeframe Technical Charts")
                                    
                                    mtf_chart = chart_builder.create_multi_timeframe_chart(mtf_data)
                                    
                                    if mtf_chart:
                                        st.plotly_chart(mtf_chart, use_container_width=True)
                                    else:
                                        st.warning("Unable to generate multi-timeframe chart")
                                    
                                    st.markdown("---")
                                    
                                    # ==============================================================
                                    # SECTION 6: Latest News & Sentiment
                                    # ==============================================================
                                    st.markdown("### 📰 Latest News")
                                    
                                    with st.spinner("🔄 Fetching latest news..."):
                                        # Clean symbol for news search (remove exchange prefix)
                                        clean_symbol = index_symbol.replace('NSE:', '').replace('BSE:', '')
                                        news_articles = news_fetcher.fetch_news(clean_symbol, max_articles=10)
                                    
                                    if news_articles:
                                        # Display news in expandable sections
                                        for idx, article in enumerate(news_articles[:5], 1):
                                            sentiment_emoji = {
                                                'Positive': '🟢',
                                                'Negative': '🔴',
                                                'Neutral': '⚪'
                                            }.get(article['sentiment'], '⚪')
                                            
                                            with st.expander(f"{sentiment_emoji} {article['title'][:100]}..."):
                                                st.markdown(f"**Source:** {article['source']}")
                                                st.markdown(f"**Date:** {article['date']}")
                                                st.markdown(f"**Sentiment:** {article['sentiment']}")
                                                if article['link'] != '#':
                                                    st.markdown(f"[Read Full Article]({article['link']})")
                                    else:
                                        st.info("No news articles found")
                                    
                                    st.markdown("---")
                                    
                                    # ==============================================================
                                    # SECTION 7: News Sentiment Breakdown
                                    # ==============================================================
                                    st.markdown("### 🎯 News Sentiment Breakdown")
                                    
                                    sentiment_summary = news_fetcher.get_sentiment_summary(news_articles)
                                    
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric(
                                            "🟢 Positive",
                                            sentiment_summary['positive'],
                                            f"{sentiment_summary['positive_pct']}%"
                                        )
                                    
                                    with col2:
                                        st.metric(
                                            "🔴 Negative",
                                            sentiment_summary['negative'],
                                            f"{sentiment_summary['negative_pct']}%"
                                        )
                                    
                                    with col3:
                                        st.metric(
                                            "⚪ Neutral",
                                            sentiment_summary['neutral'],
                                            f"{sentiment_summary['neutral_pct']}%"
                                        )
                                    
                                    # Overall sentiment
                                    if sentiment_summary['positive'] > sentiment_summary['negative']:
                                        st.success("📈 **Overall Sentiment: POSITIVE** - News flow suggests bullish sentiment")
                                    elif sentiment_summary['negative'] > sentiment_summary['positive']:
                                        st.error("📉 **Overall Sentiment: NEGATIVE** - News flow suggests bearish sentiment")
                                    else:
                                        st.info("➡️ **Overall Sentiment: NEUTRAL** - Mixed news sentiment")
                                    
                                    st.markdown("---")
                                    
                                    # ==============================================================
                                    # SECTION 8: Fibonacci Retracement & Extension Levels
                                    # ==============================================================
                                    st.markdown("### 📐 Fibonacci Retracement & Extension Levels")
                                    
                                    if 'daydata' in mtf_data and mtf_data['daydata'] is not None:
                                        df_daily = mtf_data['daydata']
                                        
                                        fib_levels = fib_calculator.calculate_levels(df_daily)
                                        
                                        if fib_levels:
                                            st.markdown(f"**Trend:** {fib_levels['trend'].upper()}")
                                            st.markdown(f"**Swing High:** ₹{fib_levels['swing_high']:,.2f}")
                                            st.markdown(f"**Swing Low:** ₹{fib_levels['swing_low']:,.2f}")
                                            
                                            st.markdown("#### 🎯 Key Retracement Levels:")
                                            
                                            fib_data = []
                                            for level, price in fib_levels['fib_levels'].items():
                                                distance = abs(spot_price - price)
                                                distance_pct = (distance / spot_price) * 100
                                                
                                                fib_data.append({
                                                    'Level': level,
                                                    'Price': f"₹{price:,.2f}",
                                                    'Distance': f"₹{distance:,.2f} ({distance_pct:.2f}%)"
                                                })
                                            
                                            fib_df = pd.DataFrame(fib_data)
                                            st.dataframe(fib_df, use_container_width=True, hide_index=True)
                                            
                                            st.markdown("#### 🎯 Extension Targets:")
                                            
                                            target_data = []
                                            for target in fib_levels['targets']:
                                                target_data.append({
                                                    'Level': target['level'],
                                                    'Target Price': f"₹{target['price']:,.2f}",
                                                    'Distance': f"₹{target['distance']:,.2f}"
                                                })
                                            
                                            target_df = pd.DataFrame(target_data)
                                            st.dataframe(target_df, use_container_width=True, hide_index=True)
                                        else:
                                            st.warning("Unable to calculate Fibonacci levels (insufficient data)")
                                    else:
                                        st.warning("Daily data not available for Fibonacci calculation")
                                
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
