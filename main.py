"""
Main Streamlit Application
Complete trading platform UI with real-time data, analysis, and monitoring
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import pytz

# Import our modules
from config import (
    validate_config,
    get_market_status,
    MARKET_CONFIG,
    TRADING_CONFIG,
    INDEX_OPTIONS_REFERENCE,
    get_instrument_config,
    get_config_summary
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

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Intraday Trading Platform",
    page_icon="📈",
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
        st.session_state.selected_stocks = []
        st.session_state.selected_options = []
        st.session_state.live_data = {}
        st.session_state.last_refresh = None

init_session_state()

# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_app():
    """Initialize application (database, Kite, streaming)"""
    with st.spinner("🔧 Initializing application..."):
        
        # Step 1: Validate configuration
        errors = validate_config()
        if errors:
            st.error("❌ Configuration Errors:")
            for error in errors:
                st.error(f"  • {error}")
            st.stop()
        
        # Step 2: Initialize database
        init_database()
        
        # Step 3: Initialize Kite Connect
        if not st.session_state.kite_connected:
            success, message = initialize_kite()
            if success:
                st.session_state.kite_connected = True
                st.success(message)
            else:
                st.error(message)
                st.stop()
        
        # Step 4: Start streaming
        if not st.session_state.streaming_active:
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
        time.sleep(1)  # Brief pause to show messages

# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar():
    """Render sidebar with controls and status"""
    
    st.sidebar.title("📈 Trading Platform")
    st.sidebar.markdown("---")
    
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
    st.sidebar.caption(f"**Exchange:** {TRADING_CONFIG['exchange']}")
    
    st.sidebar.markdown("---")
    
    # Debug Info (collapsible)
    with st.sidebar.expander("🔍 Debug Info"):
        config_summary = get_config_summary()
        st.json(config_summary)

# ============================================================================
# STOCK SCREENER TAB
# ============================================================================

def render_stock_screener_tab():
    """Render stock screening and search interface"""
    st.header("🔍 Stock Screener & Search")
    
    # Search Interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Search Stocks",
            placeholder="Enter stock name or symbol (e.g., RELIANCE, TCS)",
            key="stock_search"
        )
    
    with col2:
        exchange = st.selectbox("Exchange", ["NSE", "BSE"], key="stock_exchange")
    
    if search_query and len(search_query) >= 2:
        kite = get_kite_handler()
        results = kite.search_instruments(search_query, exchange=exchange, segment="EQ")
        
        if not results.empty:
            st.success(f"Found {len(results)} results")
            
            # Display results
            st.dataframe(
                results[['tradingsymbol', 'name', 'instrument_token']],
                use_container_width=True,
                hide_index=True
            )
            
            # Add to watchlist
            st.subheader("Add to Watchlist")
            
            selected_stock = st.selectbox(
                "Select stock to add",
                options=results['tradingsymbol'].tolist(),
                key="stock_to_add"
            )
            
            if st.button("➕ Add to Watchlist", type="primary"):
                selected_row = results[results['tradingsymbol'] == selected_stock].iloc[0]
                instrument_token = int(selected_row['instrument_token'])
                
                # Subscribe to streaming
                instruments = [{
                    'instrument_token': instrument_token,
                    'symbol': selected_stock
                }]
                
                if subscribe_instruments(instruments, mode="full"):
                    st.success(f"✅ Added {selected_stock} to watchlist and subscribed to live data")
                    if selected_stock not in st.session_state.selected_stocks:
                        st.session_state.selected_stocks.append(selected_stock)
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Failed to subscribe")
        else:
            st.warning("No results found")

# ============================================================================
# INDEX OPTIONS TAB
# ============================================================================

def render_index_options_tab():
    """Render index options analysis interface"""
    st.header("📊 Index Options Analysis")
    
    # Index Selection
    col1, col2 = st.columns([2, 2])
    
    with col1:
        selected_index = st.selectbox(
            "Select Index",
            options=INDEX_OPTIONS_REFERENCE,
            key="selected_index"
        )
    
    with col2:
        # Get index config (dynamically fetched)
        index_config = get_instrument_config(selected_index)
        
        if index_config:
            st.metric("Lot Size", index_config.get('lot_size', 'N/A'))
        else:
            st.warning("Loading index configuration...")
    
    # Get Options Chain
    if st.button("🔍 Load Options Chain", type="primary"):
        with st.spinner(f"Loading {selected_index} options chain..."):
            kite = get_kite_handler()
            calls_df, puts_df = kite.get_option_chain(selected_index)
            
            if calls_df is not None and puts_df is not None:
                st.session_state.options_chain = {
                    'calls': calls_df,
                    'puts': puts_df,
                    'index': selected_index
                }
                st.success(f"✅ Loaded {len(calls_df)} Calls and {len(puts_df)} Puts")
            else:
                st.error("Failed to load options chain")
    
    # Display Options Chain
    if 'options_chain' in st.session_state:
        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs(["📈 Calls", "📉 Puts", "🎯 Combined"])
        
        with tab1:
            st.subheader("Call Options")
            calls_df = st.session_state.options_chain['calls']
            st.dataframe(
                calls_df[['strike', 'tradingsymbol', 'expiry', 'lot_size']].head(20),
                use_container_width=True,
                hide_index=True
            )
        
        with tab2:
            st.subheader("Put Options")
            puts_df = st.session_state.options_chain['puts']
            st.dataframe(
                puts_df[['strike', 'tradingsymbol', 'expiry', 'lot_size']].head(20),
                use_container_width=True,
                hide_index=True
            )
        
        with tab3:
            st.subheader("Combined View")
            st.info("Advanced options analysis coming soon")

# ============================================================================
# LIVE MONITOR TAB
# ============================================================================

def render_live_monitor_tab():
    """Render live data monitoring"""
    st.header("📡 Live Market Monitor")
    
    # Check watchlist
    watchlist_symbols = get_streaming_handler().get_subscribed_symbols()
    
    if not watchlist_symbols:
        st.info("📝 No instruments in watchlist. Add stocks from the Stock Screener tab.")
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
# CHART ANALYSIS TAB
# ============================================================================

def render_chart_analysis_tab():
    """Render chart and technical analysis"""
    st.header("📈 Chart Analysis")
    
    watchlist_symbols = get_streaming_handler().get_subscribed_symbols()
    
    if not watchlist_symbols:
        st.info("Add instruments to watchlist first")
        return
    
    # Symbol selection
    selected_symbol = st.selectbox("Select Symbol", options=watchlist_symbols)
    
    # Timeframe selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        timeframe = st.selectbox(
            "Timeframe",
            options=["1 minute", "5 minute", "15 minute", "30 minute", "60 minute"],
            key="timeframe"
        )
    
    with col2:
        days = st.number_input("Days", min_value=1, max_value=30, value=5)
    
    if st.button("📊 Load Chart", type="primary"):
        with st.spinner("Loading chart data..."):
            kite = get_kite_handler()
            
            # Map timeframe
            interval_map = {
                "1 minute": "minute",
                "5 minute": "5minute",
                "15 minute": "15minute",
                "30 minute": "30minute",
                "60 minute": "60minute"
            }
            
            df = kite.get_historical_data_by_symbol(
                symbol=selected_symbol,
                exchange="NSE",
                days=days,
                interval=interval_map[timeframe]
            )
            
            if df is not None and not df.empty:
                # Create candlestick chart
                fig = go.Figure()
                
                fig.add_trace(go.Candlestick(
                    x=df['date'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name=selected_symbol
                ))
                
                fig.update_layout(
                    title=f"{selected_symbol} - {timeframe}",
                    yaxis_title="Price (₹)",
                    xaxis_title="Time",
                    height=600,
                    template="plotly_dark"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display OHLCV data
                st.subheader("📊 OHLCV Data")
                st.dataframe(
                    df[['date', 'open', 'high', 'low', 'close', 'volume']].tail(50),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.error("No data available")

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
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    
    # Render sidebar
    render_sidebar()
    
    # Initialize app on first run
    if not st.session_state.initialized:
        initialize_app()
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Stock Screener",
        "📊 Index Options",
        "📡 Live Monitor",
        "📈 Charts",
        "📜 Trade History"
    ])
    
    with tab1:
        render_stock_screener_tab()
    
    with tab2:
        render_index_options_tab()
    
    with tab3:
        render_live_monitor_tab()
    
    with tab4:
        render_chart_analysis_tab()
    
    with tab5:
        render_trade_history_tab()
    
    # Footer
    st.markdown("---")
    st.caption("🚀 Intraday Trading Platform | Data: Kite Connect | Made with Streamlit")

if __name__ == "__main__":
    main()
