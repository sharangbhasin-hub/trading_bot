"""
Paper Trading Application - Main UI
====================================

Complete Streamlit interface for real-time paper trading.
Supports Cryptocurrency and Forex markets with live signal generation.

Features:
- Real-time live data feeds
- Strategy signal generation (CRT-TBS + 9 SMC strategies)
- Automated order execution
- Position tracking with auto SL/TP
- Performance analytics
- Trade history and CSV export

Author: Trading System
Last Updated: October 29, 2025

Usage:
    streamlit run main_paper_trading.py
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import existing handlers
from handlers.unified_data_handler import UnifiedDataHandler, get_unified_handler
from strategies.strategy_manager import StrategyManager

# Import paper trading components
from paper_trading.config import PAPER_TRADING_CONFIG, get_config
from paper_trading.trade_database import TradeDatabase
from paper_trading.pnl_calculator import PnLCalculator, format_pnl
from paper_trading.live_data_manager import LiveDataManager
from paper_trading.paper_order_manager import PaperOrderManager
from paper_trading.chart_visualizer import ChartVisualizer
from paper_trading.performance_wrapper import PaperTradingPerformanceAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Paper Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .big-metric {
        font-size: 2rem;
        font-weight: bold;
    }
    .positive {
        color: #00C853;
    }
    .negative {
        color: #FF1744;
    }
    .neutral {
        color: #2196F3;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .trade-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables."""

    # Add to session state initialization
    if 'chart_visualizer' not in st.session_state:
        st.session_state.chart_visualizer = None

    if 'performance_analyzer' not in st.session_state:
        st.session_state.performance_analyzer = None
    
    # Trading status
    if 'trading_active' not in st.session_state:
        st.session_state.trading_active = False
    
    # Selected settings
    if 'market_type' not in st.session_state:
        st.session_state.market_type = 'Cryptocurrency'
    if 'symbol' not in st.session_state:
        st.session_state.symbol = 'BTC/USDT'
    if 'strategy_name' not in st.session_state:
        st.session_state.strategy_name = 'CRT-TBS'
    if 'trading_mode' not in st.session_state:
        st.session_state.trading_mode = 'Scalping (1H→1min)'
    
    # Components (initialized once)
    if 'components_initialized' not in st.session_state:
        st.session_state.components_initialized = False
    if 'trade_db' not in st.session_state:
        st.session_state.trade_db = None
    if 'pnl_calculator' not in st.session_state:
        st.session_state.pnl_calculator = None
    if 'order_manager' not in st.session_state:
        st.session_state.order_manager = None
    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = None
    if 'strategy_manager' not in st.session_state:
        st.session_state.strategy_manager = None
    
    # Latest data
    if 'latest_signal' not in st.session_state:
        st.session_state.latest_signal = None
    if 'latest_price' not in st.session_state:
        st.session_state.latest_price = None
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = None
    
    # Statistics
    if 'signals_generated' not in st.session_state:
        st.session_state.signals_generated = 0
    if 'signals_executed' not in st.session_state:
        st.session_state.signals_executed = 0

initialize_session_state()

# ============================================================================
# COMPONENT INITIALIZATION
# ============================================================================

def initialize_components():
    """Initialize all paper trading components."""
    try:
        # Database
        st.session_state.trade_db = TradeDatabase()
        
        # P&L Calculator
        st.session_state.pnl_calculator = PnLCalculator()
        
        # Order Manager
        initial_balance = PAPER_TRADING_CONFIG['initial_balance']
        st.session_state.order_manager = PaperOrderManager(
            trade_database=st.session_state.trade_db,
            pnl_calculator=st.session_state.pnl_calculator,
            initial_balance=initial_balance
        )
        
        # Strategy Manager (loaded dynamically based on selection)
        st.session_state.strategy_manager = StrategyManager()

        st.session_state.chart_visualizer = ChartVisualizer(height=500)

        st.session_state.performance_analyzer = PaperTradingPerformanceAnalyzer(
            st.session_state.trade_db
        )
        
        st.session_state.components_initialized = True
        logger.info("✅ All components initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        st.error(f"❌ Initialization failed: {e}")
        return False

if not st.session_state.components_initialized:
    with st.spinner("Initializing paper trading system..."):
        if initialize_components():
            st.success("✅ System initialized successfully!")
            time.sleep(1)
            st.rerun()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_market_handler(market_type: str):
    """Get appropriate data handler for market type."""
    if market_type == 'Cryptocurrency':
        return get_unified_handler(UnifiedDataHandler.MARKET_CRYPTO_BINANCE)
    else:  # Forex
        return get_unified_handler(UnifiedDataHandler.MARKET_FOREX)

def get_timeframes(trading_mode: str) -> tuple:
    """Get HTF and LTF based on trading mode."""
    if 'Scalping' in trading_mode:
        return ('1h', '1min')
    else:  # Intraday
        return ('1d', '1h')

def get_symbol_list(market_type: str) -> List[str]:
    """Get list of tradable symbols for market."""
    if market_type == 'Cryptocurrency':
        return PAPER_TRADING_CONFIG['crypto']['supported_pairs']
    else:
        return PAPER_TRADING_CONFIG['forex']['supported_pairs']

def get_strategy_list() -> List[str]:
    """Get list of available strategies."""
    return [
        'CRT-TBS',
        'Order Block + FVG',
        'BOS + CHoCH + Liquidity',
        'Fake Breakout',
        'Liquidity Sweep',
        'Liq Grab OB',
        'FVG Double Bottom/Top',
        'OB CHoCH Combined',
        'CHoCH OB',
        'BOS Retest'
    ]

def map_strategy_name_to_file(strategy_name: str) -> str:
    """Map display name to strategy file name."""
    mapping = {
        'CRT-TBS': 'strategy_crt_tbs',
        'Order Block + FVG': 'strategy_ob_fvg',
        'BOS + CHoCH + Liquidity': 'strategy_bos_choch_liquidity',
        'Fake Breakout': 'strategy_fake_breakout',
        'Liquidity Sweep': 'strategy_liquidity_sweep',
        'Liq Grab OB': 'strategy_liq_grab_ob',
        'FVG Double Bottom/Top': 'strategy_fvg_double_bottom_top',
        'OB CHoCH Combined': 'strategy_ob_choch_combined',
        'CHoCH OB': 'strategy_choch_ob',
        'BOS Retest': 'strategy_bos_retest'
    }
    return mapping.get(strategy_name, 'strategy_crt_tbs')

# ============================================================================
# SIGNAL GENERATION CALLBACK
# ============================================================================

def on_new_candle(symbol: str, candle: Dict):
    """
    Callback function triggered when new candle is received.
    Runs strategy analysis and generates signals.
    """
    try:
        # Get recent candles for strategy
        data_mgr = st.session_state.data_manager
        recent_candles = data_mgr.get_recent_candles(symbol, count=100)
        
        if len(recent_candles) < 50:
            logger.debug(f"Not enough candles for analysis ({len(recent_candles)})")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(recent_candles)
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
        elif 'date' in df.columns:
            df.set_index('date', inplace=True)
        
        # Load strategy
        strategy_file = map_strategy_name_to_file(st.session_state.strategy_name)
        strategy_module = st.session_state.strategy_manager.get_strategy(strategy_file)
        
        if not strategy_module:
            logger.error(f"Strategy not found: {strategy_file}")
            return
        
        # Get HTF and LTF
        htf, ltf = get_timeframes(st.session_state.trading_mode)
        
        # Run strategy analysis
        result = strategy_module.analyze(
            df_htf=df,  # For simplicity, using same data
            df_ltf=df,
            symbol=symbol
        )
        
        if result and result.get('signal') != 'NO_TRADE':
            # Valid signal generated
            signal_data = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'direction': result['signal'],
                'entry_price': result['entry'],
                'stop_loss': result['sl'],
                'take_profit': result['tp'],
                'strategy_name': st.session_state.strategy_name,
                'confidence': result.get('confidence', 0),
                'risk_reward_ratio': result.get('rr_ratio', 0),
                'market_type': 'crypto' if st.session_state.market_type == 'Cryptocurrency' else 'forex',
                'reasoning': result.get('reasoning', [])
            }
            
            # Store signal
            st.session_state.latest_signal = signal_data
            st.session_state.signals_generated += 1
            st.session_state.last_update_time = datetime.now()
            
            # Log to database
            st.session_state.trade_db.insert_signal({
                'timestamp': signal_data['timestamp'],
                'symbol': symbol,
                'strategy_name': st.session_state.strategy_name,
                'signal_type': signal_data['direction'],
                'entry_price': signal_data['entry_price'],
                'stop_loss': signal_data['stop_loss'],
                'take_profit': signal_data['take_profit'],
                'confidence': signal_data['confidence'],
                'reasoning': str(signal_data['reasoning']),
                'executed': False
            })
            
            logger.info(f"🔔 Signal generated: {signal_data['direction']} {symbol} @ {signal_data['entry_price']}")
        
        # Update latest price
        st.session_state.latest_price = candle.get('close', 0)
        st.session_state.last_update_time = datetime.now()
        
        # Check open positions for SL/TP
        current_price = candle.get('close', 0)
        st.session_state.order_manager.check_open_positions(symbol, current_price, current_price)
        
    except Exception as e:
        logger.error(f"Error in signal generation: {e}")

# ============================================================================
# TRADING CONTROL FUNCTIONS
# ============================================================================

def start_paper_trading():
    """Start paper trading with current settings."""
    try:
        # Get data handler
        handler = get_market_handler(st.session_state.market_type)
        
        # Initialize data manager
        st.session_state.data_manager = LiveDataManager(handler)
        
        # Get timeframes
        htf, ltf = get_timeframes(st.session_state.trading_mode)
        
        # Start live feed (using LTF for real-time signals)
        success = st.session_state.data_manager.start_feed(
            symbol=st.session_state.symbol,
            timeframe=ltf,
            on_new_candle=on_new_candle
        )
        
        if success:
            st.session_state.trading_active = True
            logger.info(f"✅ Paper trading started: {st.session_state.symbol}, {st.session_state.strategy_name}")
            return True
        else:
            logger.error("Failed to start data feed")
            return False
            
    except Exception as e:
        logger.error(f"Failed to start paper trading: {e}")
        st.error(f"❌ Failed to start: {e}")
        return False

def stop_paper_trading():
    """Stop paper trading."""
    try:
        if st.session_state.data_manager:
            st.session_state.data_manager.stop_feed(st.session_state.symbol)
        
        st.session_state.trading_active = False
        logger.info("⏹ Paper trading stopped")
        return True
        
    except Exception as e:
        logger.error(f"Failed to stop paper trading: {e}")
        return False

def execute_signal(signal: Dict):
    """Execute a signal as a paper trade."""
    try:
        current_price = st.session_state.latest_price or signal['entry_price']
        
        result = st.session_state.order_manager.place_order(
            signal=signal,
            current_price=current_price,
            exchange_price=current_price
        )
        
        if result['success']:
            st.session_state.signals_executed += 1
            st.session_state.latest_signal = None  # Clear signal after execution
            logger.info(f"✅ Signal executed: Trade #{result['trade_id']}")
            return True
        else:
            logger.warning(f"❌ Signal execution failed: {result['reason']}")
            st.error(f"❌ Order rejected: {result['reason']}")
            return False
            
    except Exception as e:
        logger.error(f"Error executing signal: {e}")
        st.error(f"❌ Execution error: {e}")
        return False

# ============================================================================
# UI SECTIONS
# ============================================================================
# ============================================================================
# SIDEBAR - CONFIGURATION & CONTROLS
# ============================================================================

with st.sidebar:
    st.title("📈 Paper Trading")
    st.markdown("---")
    
    # Trading Status Indicator
    if st.session_state.trading_active:
        st.success("🟢 **TRADING ACTIVE**")
    else:
        st.info("⚪ **TRADING STOPPED**")
    
    st.markdown("---")
    
    # Market Selection
    st.subheader("⚙️ Configuration")
    
    market_type = st.selectbox(
        "Market Type",
        ['Cryptocurrency', 'Forex'],
        index=0 if st.session_state.market_type == 'Cryptocurrency' else 1,
        disabled=st.session_state.trading_active
    )
    st.session_state.market_type = market_type
    
    # Symbol Selection
    symbols = get_symbol_list(market_type)
    symbol = st.selectbox(
        "Symbol",
        symbols,
        index=symbols.index(st.session_state.symbol) if st.session_state.symbol in symbols else 0,
        disabled=st.session_state.trading_active
    )
    st.session_state.symbol = symbol
    
    # Strategy Selection
    strategies = get_strategy_list()
    strategy_name = st.selectbox(
        "Strategy",
        strategies,
        index=strategies.index(st.session_state.strategy_name) if st.session_state.strategy_name in strategies else 0,
        disabled=st.session_state.trading_active
    )
    st.session_state.strategy_name = strategy_name
    
    # Trading Mode
    trading_mode = st.selectbox(
        "Trading Mode",
        ['Scalping (1H→1min)', 'Intraday (1D→1H)'],
        index=0 if 'Scalping' in st.session_state.trading_mode else 1,
        disabled=st.session_state.trading_active
    )
    st.session_state.trading_mode = trading_mode
    
    st.markdown("---")
    
    # Start/Stop Controls
    st.subheader("🎮 Controls")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not st.session_state.trading_active:
            if st.button("▶️ START", type="primary", use_container_width=True):
                with st.spinner("Starting paper trading..."):
                    if start_paper_trading():
                        st.success("✅ Started!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Failed to start")
    
    with col2:
        if st.session_state.trading_active:
            if st.button("⏹ STOP", type="secondary", use_container_width=True):
                if stop_paper_trading():
                    st.success("⏹ Stopped!")
                    time.sleep(1)
                    st.rerun()
    
    # Quick Actions
    st.markdown("---")
    st.subheader("⚡ Quick Actions")
    
    if st.button("📥 Export Trades to CSV", use_container_width=True):
        try:
            filename = f"paper_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            st.session_state.trade_db.export_to_csv(filename)
            st.success(f"✅ Exported to {filename}")
        except Exception as e:
            st.error(f"❌ Export failed: {e}")
    
    if st.button("💾 Backup Database", use_container_width=True):
        try:
            backup_path = st.session_state.trade_db.backup_database()
            st.success(f"✅ Backup created!")
        except Exception as e:
            st.error(f"❌ Backup failed: {e}")
    
    if st.button("🔄 Reset Daily Stats", use_container_width=True):
        st.session_state.order_manager.reset_daily_stats()
        st.success("✅ Daily stats reset!")

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

st.title("📊 Paper Trading Dashboard")
st.markdown(f"**{st.session_state.symbol}** | **{st.session_state.strategy_name}** | **{st.session_state.trading_mode}**")

# Auto-refresh indicator
if st.session_state.trading_active:
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        last_update = st.session_state.last_update_time
        if last_update:
            time_diff = (datetime.now() - last_update).total_seconds()
            st.info(f"🔄 Last update: {int(time_diff)}s ago | Auto-refreshing every 5s")
        else:
            st.info("⏳ Waiting for first data update...")

st.markdown("---")

# ============================================================================
# ACCOUNT OVERVIEW (TOP METRICS)
# ============================================================================

if st.session_state.order_manager:
    account_status = st.session_state.order_manager.get_account_status()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        balance = account_status['balance']
        balance_change = balance - account_status['initial_balance']
        balance_change_pct = (balance_change / account_status['initial_balance']) * 100
        
        st.metric(
            label="💰 Balance",
            value=f"${balance:,.2f}",
            delta=f"${balance_change:,.2f} ({balance_change_pct:+.2f}%)"
        )
    
    with col2:
        daily_pnl = account_status['daily_pnl']
        daily_color = "🟢" if daily_pnl > 0 else "🔴" if daily_pnl < 0 else "⚪"
        
        st.metric(
            label=f"{daily_color} Today's P&L",
            value=f"${daily_pnl:,.2f}",
            delta=None
        )
    
    with col3:
        equity = account_status['equity']
        unrealized = account_status['unrealized_pnl']
        
        st.metric(
            label="📈 Equity",
            value=f"${equity:,.2f}",
            delta=f"Unrealized: ${unrealized:,.2f}"
        )
    
    with col4:
        open_pos = account_status['open_positions']
        max_pos = PAPER_TRADING_CONFIG['risk_management']['max_open_positions']
        
        st.metric(
            label="📋 Open Positions",
            value=f"{open_pos} / {max_pos}"
        )
    
    with col5:
        win_rate = account_status['win_rate']
        win_rate_color = "🟢" if win_rate > 50 else "🟡" if win_rate > 40 else "🔴"
        
        st.metric(
            label=f"{win_rate_color} Win Rate",
            value=f"{win_rate:.1f}%",
            delta=f"{account_status['winning_trades']}W / {account_status['losing_trades']}L"
        )

st.markdown("---")

# ============================================================================
# TABS: Dashboard | Signals | Positions | History | Settings
# ============================================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Dashboard", 
    "🔔 Signals", 
    "📋 Positions", 
    "📜 History", 
    "⚙️ Settings", 
    "📈 Chart",
    "📈 Analytics"  # ✅ NEW TAB
])

# ----------------------------------------------------------------------------
# TAB 1: DASHBOARD
# ----------------------------------------------------------------------------

with tab1:
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("📈 Live Market Data")
        
        if st.session_state.trading_active and st.session_state.data_manager:
            # Get feed status
            feed_status = st.session_state.data_manager.get_feed_status(st.session_state.symbol)
            
            # Display current price
            if st.session_state.latest_price:
                st.markdown(f"### 💵 ${st.session_state.latest_price:,.2f}")
            
            # Feed statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Buffer Size", f"{feed_status['buffer_size']} candles")
            with col2:
                st.metric("Fetches", feed_status['fetch_count'])
            with col3:
                st.metric("Errors", feed_status['error_count'])
            
            # Recent candles table
            if feed_status['has_data']:
                recent_candles = st.session_state.data_manager.get_recent_candles(st.session_state.symbol, 10)
                if recent_candles:
                    df_recent = pd.DataFrame(recent_candles[-10:])
                    
                    # Format display columns
                    display_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume'] if 'timestamp' in df_recent.columns else ['date', 'open', 'high', 'low', 'close', 'volume']
                    
                    st.dataframe(
                        df_recent[display_cols].tail(10),
                        use_container_width=True,
                        hide_index=True
                    )
        else:
            st.info("▶️ Start paper trading to see live market data")
    
    with col_right:
        st.subheader("📊 Performance Summary")
        
        if st.session_state.order_manager:
            status = st.session_state.order_manager.get_account_status()
            
            # Performance metrics
            metrics_data = {
                'Metric': [
                    'Total Trades',
                    'Winning Trades',
                    'Losing Trades',
                    'Win Rate',
                    'Total P&L',
                    'Total P&L %',
                    'Daily P&L',
                    'Max Drawdown',
                    'Open Positions'
                ],
                'Value': [
                    f"{status['total_trades']}",
                    f"{status['winning_trades']} 🟢",
                    f"{status['losing_trades']} 🔴",
                    f"{status['win_rate']:.1f}%",
                    f"${status['total_pnl']:,.2f}",
                    f"{status['total_pnl_pct']:+.2f}%",
                    f"${status['daily_pnl']:,.2f}",
                    f"{status['max_drawdown']:.2f}%",
                    f"{status['open_positions']}"
                ]
            }
            
            st.dataframe(
                pd.DataFrame(metrics_data),
                use_container_width=True,
                hide_index=True
            )
            
            # Signal statistics
            st.markdown("### 🎯 Signal Stats")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Signals Generated", st.session_state.signals_generated)
            with col2:
                st.metric("Signals Executed", st.session_state.signals_executed)

# ----------------------------------------------------------------------------
# TAB 2: SIGNALS
# ----------------------------------------------------------------------------

with tab2:
    st.subheader("🔔 Latest Signal")
    
    if st.session_state.latest_signal:
        signal = st.session_state.latest_signal
        
        # Signal card
        signal_color = "green" if signal['direction'] == 'BUY' else "red"
        
        st.markdown(f"""
        <div style="background-color: rgba({'0, 255, 0' if signal['direction'] == 'BUY' else '255, 0, 0'}, 0.1); 
                    padding: 20px; border-radius: 10px; border: 2px solid {signal_color};">
            <h2 style="color: {signal_color};">{signal['direction']} {signal['symbol']}</h2>
            <p style="font-size: 1.2em;">
                <strong>Entry:</strong> ${signal['entry_price']:,.2f} | 
                <strong>SL:</strong> ${signal['stop_loss']:,.2f} | 
                <strong>TP:</strong> ${signal['take_profit']:,.2f}
            </p>
            <p>
                <strong>Strategy:</strong> {signal['strategy_name']} | 
                <strong>Confidence:</strong> {signal['confidence']}% | 
                <strong>R:R:</strong> {signal['risk_reward_ratio']:.2f}
            </p>
            <p><small>Generated: {signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</small></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 📝 Signal Reasoning")
        if signal.get('reasoning'):
            for reason in signal['reasoning']:
                st.markdown(f"- {reason}")
        
        # Execute button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("✅ Execute Trade", type="primary", use_container_width=True):
                if execute_signal(signal):
                    st.success("✅ Trade executed successfully!")
                    time.sleep(1)
                    st.rerun()
        
        with col2:
            if st.button("❌ Dismiss Signal", use_container_width=True):
                st.session_state.latest_signal = None
                st.info("Signal dismissed")
                time.sleep(1)
                st.rerun()
    
    else:
        st.info("⏳ No active signals. Waiting for trading opportunities...")
    
    st.markdown("---")
    
    # Recent signals log
    st.subheader("📜 Recent Signals (Last 10)")
    
    # Query from database
    # Note: This would require a query method in TradeDatabase
    st.info("Signal history will appear here once implemented")

# ----------------------------------------------------------------------------
# TAB 3: OPEN POSITIONS
# ----------------------------------------------------------------------------

with tab3:
    st.subheader("📋 Open Positions")
    
    if st.session_state.order_manager:
        open_positions = st.session_state.order_manager.get_open_positions_list()
        
        if open_positions:
            # Create DataFrame
            positions_data = []
            for pos in open_positions:
                positions_data.append({
                    'ID': pos['trade_id'],
                    'Symbol': pos['symbol'],
                    'Direction': pos['direction'],
                    'Entry': f"${pos['entry_price']:,.2f}",
                    'Current': f"${st.session_state.latest_price:,.2f}" if st.session_state.latest_price else "-",
                    'SL': f"${pos['stop_loss']:,.2f}",
                    'TP': f"${pos['take_profit']:,.2f}",
                    'Unrealized P&L': f"${pos.get('unrealized_pnl', 0):,.2f}",
                    'Strategy': pos['strategy_name'],
                    'Time': pos['timestamp'].strftime('%H:%M:%S') if isinstance(pos['timestamp'], datetime) else pos['timestamp']
                })
            
            df_positions = pd.DataFrame(positions_data)
            st.dataframe(df_positions, use_container_width=True, hide_index=True)
            
            # Manual close options
            st.markdown("---")
            st.subheader("🛑 Manual Close")
            
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                trade_ids = [pos['trade_id'] for pos in open_positions]
                selected_trade = st.selectbox("Select Trade to Close", trade_ids)
            
            with col2:
                close_price = st.number_input(
                    "Close Price",
                    value=st.session_state.latest_price if st.session_state.latest_price else 0.0,
                    step=0.01
                )
            
            with col3:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Close Position", type="secondary"):
                    if st.session_state.order_manager.close_position(selected_trade, close_price, 'MANUAL'):
                        st.success(f"✅ Position #{selected_trade} closed!")
                        time.sleep(1)
                        st.rerun()
        
        else:
            st.info("📭 No open positions")
    else:
        st.error("Order manager not initialized")

# ----------------------------------------------------------------------------
# TAB 4: TRADE HISTORY
# ----------------------------------------------------------------------------

with tab4:
    st.subheader("📜 Trade History")
    
    # Date range filter
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now().date() - timedelta(days=7))
    
    with col2:
        end_date = st.date_input("End Date", value=datetime.now().date())
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        filter_btn = st.button("🔍 Filter", type="primary")
    
    # Get trades from database
    if st.session_state.trade_db:
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.max.time())
        
        closed_trades = st.session_state.trade_db.get_closed_trades(
            start_date=start_datetime,
            end_date=end_datetime
        )
        
        if closed_trades:
            # Create DataFrame
            trades_data = []
            for trade in closed_trades:
                trades_data.append({
                    'ID': trade['id'],
                    'Time': trade['timestamp'],
                    'Symbol': trade['symbol'],
                    'Direction': trade['direction'],
                    'Entry': f"${trade['entry_price']:,.2f}",
                    'Exit': f"${trade['exit_price']:,.2f}" if trade['exit_price'] else "-",
                    'P&L': f"${trade['pnl_usd']:,.2f}" if trade['pnl_usd'] else "-",
                    'P&L %': f"{trade['pnl_pct']:,.2f}%" if trade['pnl_pct'] else "-",
                    'Exit Reason': trade['exit_reason'],
                    'Strategy': trade['strategy_name']
                })
            
            df_trades = pd.DataFrame(trades_data)
            st.dataframe(df_trades, use_container_width=True, hide_index=True)
            
            # Summary statistics
            st.markdown("---")
            st.subheader("📊 Period Statistics")
            
            stats = st.session_state.trade_db.get_trade_statistics(days=(end_date - start_date).days)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Trades", stats['total_trades'])
            
            with col2:
                st.metric("Win Rate", f"{stats['win_rate']:.1f}%")
            
            with col3:
                st.metric("Total P&L", f"${stats['total_pnl']:,.2f}")
            
            with col4:
                st.metric("Profit Factor", f"{stats['profit_factor']:.2f}")
        
        else:
            st.info("📭 No trades in selected date range")

# ----------------------------------------------------------------------------
# TAB 5: SETTINGS
# ----------------------------------------------------------------------------

with tab5:
    st.subheader("⚙️ Configuration Settings")
    
    # Risk Management
    st.markdown("### 🛡️ Risk Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.number_input(
            "Max Daily Loss (USD)",
            value=PAPER_TRADING_CONFIG['risk_management']['max_daily_loss_usd'],
            step=10.0,
            key="max_daily_loss"
        )
        
        st.number_input(
            "Max Open Positions",
            value=PAPER_TRADING_CONFIG['risk_management']['max_open_positions'],
            step=1,
            key="max_positions"
        )
    
    with col2:
        st.number_input(
            "Max Trades Per Day",
            value=PAPER_TRADING_CONFIG['risk_management']['max_trades_per_day'],
            step=1,
            key="max_trades_day"
        )
        
        st.number_input(
            "Min Risk/Reward Ratio",
            value=PAPER_TRADING_CONFIG['risk_management']['min_risk_reward_ratio'],
            step=0.1,
            key="min_rr"
        )
    
    st.markdown("---")
    
    # Crypto Settings
    st.markdown("### 💰 Cryptocurrency Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.number_input(
            "Investment Per Trade (USD)",
            value=PAPER_TRADING_CONFIG['crypto']['investment_per_trade_usd'],
            step=100.0,
            key="crypto_investment"
        )
    
    with col2:
        st.number_input(
            "Slippage (%)",
            value=PAPER_TRADING_CONFIG['crypto']['slippage_pct'] * 100,
            step=0.01,
            format="%.3f",
            key="crypto_slippage"
        )
    
    st.markdown("---")
    
    # Forex Settings
    st.markdown("### 💱 Forex Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.number_input(
            "Lot Size",
            value=PAPER_TRADING_CONFIG['forex']['lot_size'],
            step=0.01,
            format="%.2f",
            key="forex_lot_size"
        )
    
    with col2:
        st.checkbox(
            "Use OANDA Practice API",
            value=PAPER_TRADING_CONFIG['forex']['use_oanda_practice'],
            key="use_oanda_practice",
            help="Enable to place orders on OANDA practice account"
        )
    
    st.markdown("---")
    
    # Save button
    if st.button("💾 Save Settings", type="primary"):
        st.success("✅ Settings saved! (Note: Restart required for some changes)")

# ----------------------------------------------------------------------------
# TAB 6: LIVE CHART
# ----------------------------------------------------------------------------

with tab6:
    st.subheader("📈 Live Price Chart")
    
    if st.session_state.trading_active and st.session_state.data_manager:
        # Get recent candles
        df = st.session_state.data_manager.get_candles_as_dataframe(
            st.session_state.symbol,
            count=100
        )
        
        if not df.empty:
            try:
                # Create chart
                fig = st.session_state.chart_visualizer.create_candlestick_chart(
                    df,
                    title=f"{st.session_state.symbol} - {st.session_state.trading_mode}",
                    show_volume=True,
                    show_indicators=True
                )
                
                # Add trade markers if any closed trades exist
                closed_trades = st.session_state.trade_db.get_today_trades()
                if closed_trades:
                    fig = st.session_state.chart_visualizer.add_trade_markers(fig, closed_trades, df)
                
                # Display chart
                st.plotly_chart(fig, use_container_width=True)
                
                # Chart controls
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🔄 Refresh Chart", use_container_width=True):
                        st.rerun()
                
                with col2:
                    if st.button("📥 Download Chart", use_container_width=True):
                        filename = f"chart_{st.session_state.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                        fig.write_html(filename)
                        st.success(f"✅ Chart saved to {filename}")
                
                with col3:
                    candle_count = st.selectbox(
                        "Candles",
                        [50, 100, 200, 500],
                        index=1,
                        key="chart_candles"
                    )
                
                # Chart info
                st.info(f"📊 Showing last {len(df)} candles | Latest: ${df['close'].iloc[-1]:,.2f}")
                
            except Exception as e:
                st.error(f"❌ Chart error: {e}")
                logger.error(f"Chart visualization error: {e}")
        else:
            st.info("⏳ Waiting for chart data...")
    else:
        st.info("▶️ Start paper trading to view live charts")
        st.markdown("---")
        st.markdown("""
        **Chart Features:**
        - 📈 Interactive candlestick chart with zoom/pan
        - 📊 Volume subplot
        - 📉 Technical indicators (EMA 9, 21, 50)
        - 🎯 Trade entry/exit markers
        - 💾 Download as HTML
        """)

# ----------------------------------------------------------------------------
# TAB 7: PERFORMANCE ANALYTICS
# ----------------------------------------------------------------------------

with tab7:
    st.subheader("📊 Performance Analytics")
    
    # Period selector
    col1, col2 = st.columns([3, 1])
    
    with col1:
        period = st.selectbox(
            "Analysis Period",
            ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"],
            index=1
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        refresh_analytics = st.button("🔄 Refresh", key="refresh_analytics")
    
    # Map period to days
    period_map = {
        "Last 7 Days": 7,
        "Last 30 Days": 30,
        "Last 90 Days": 90,
        "All Time": 365
    }
    days = period_map[period]
    
    # Get performance summary
    if st.session_state.performance_analyzer:
        try:
            summary = st.session_state.performance_analyzer.get_performance_summary(days=days)
            
            if summary['overall']['total_trades'] > 0:
                # ============================================================
                # SECTION 1: Key Metrics
                # ============================================================
                st.markdown("### 🎯 Key Metrics")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric(
                        "Total Trades",
                        summary['overall']['total_trades']
                    )
                
                with col2:
                    win_rate = summary['overall']['win_rate']
                    st.metric(
                        "Win Rate",
                        f"{win_rate:.1f}%",
                        delta="Good" if win_rate > 50 else "Low"
                    )
                
                with col3:
                    pf = summary['overall']['profit_factor']
                    st.metric(
                        "Profit Factor",
                        f"{pf:.2f}",
                        delta="Profitable" if pf > 1 else "Losing"
                    )
                
                with col4:
                    st.metric(
                        "Total P&L",
                        f"${summary['overall']['total_pnl']:,.2f}"
                    )
                
                with col5:
                    sharpe = summary['risk']['sharpe_ratio']
                    st.metric(
                        "Sharpe Ratio",
                        f"{sharpe:.2f}",
                        delta="Excellent" if sharpe > 2 else "Good" if sharpe > 1 else "Poor"
                    )
                
                st.markdown("---")
                
                # ============================================================
                # SECTION 2: Returns & Risk
                # ============================================================
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.markdown("### 💰 Returns Analysis")
                    
                    returns_data = {
                        'Metric': [
                            'Avg Return/Trade',
                            'Median Return',
                            'Best Trade',
                            'Worst Trade',
                            'Std Deviation',
                            'Expectancy'
                        ],
                        'Value': [
                            f"${summary['returns']['avg_return_per_trade']:.2f}",
                            f"${summary['returns']['median_return_per_trade']:.2f}",
                            f"${summary['returns']['best_trade']:.2f}",
                            f"${summary['returns']['worst_trade']:.2f}",
                            f"${summary['returns']['std_return']:.2f}",
                            f"${summary['efficiency']['expectancy']:.2f}"
                        ]
                    }
                    
                    st.dataframe(
                        pd.DataFrame(returns_data),
                        use_container_width=True,
                        hide_index=True
                    )
                
                with col_right:
                    st.markdown("### ⚠️ Risk Metrics")
                    
                    risk_data = {
                        'Metric': [
                            'Sharpe Ratio',
                            'Max Drawdown',
                            'Drawdown %',
                            'Consecutive Losses',
                            'Volatility',
                            'Downside Dev'
                        ],
                        'Value': [
                            f"{summary['risk']['sharpe_ratio']:.2f}",
                            f"${summary['risk']['max_drawdown_usd']:.2f}",
                            f"{summary['risk']['max_drawdown_pct']:.2f}%",
                            f"{summary['risk']['consecutive_losses']}",
                            f"${summary['risk']['volatility']:.2f}",
                            f"${summary['risk']['downside_deviation']:.2f}"
                        ]
                    }
                    
                    st.dataframe(
                        pd.DataFrame(risk_data),
                        use_container_width=True,
                        hide_index=True
                    )
                
                st.markdown("---")
                
                # ============================================================
                # SECTION 3: Win/Loss Analysis
                # ============================================================
                st.markdown("### 📊 Win/Loss Breakdown")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("#### 🟢 Winning Trades")
                    st.metric("Count", summary['overall']['winning_trades'])
                    st.metric("Avg Win", f"${summary['overall']['avg_win']:.2f}")
                    st.metric("Total Profit", f"${summary['overall']['total_profit']:,.2f}")
                
                with col2:
                    st.markdown("#### 🔴 Losing Trades")
                    st.metric("Count", summary['overall']['losing_trades'])
                    st.metric("Avg Loss", f"${summary['overall']['avg_loss']:.2f}")
                    st.metric("Total Loss", f"${summary['overall']['total_loss']:,.2f}")
                
                with col3:
                    st.markdown("#### 📈 Ratios")
                    st.metric("Profit Factor", f"{summary['overall']['profit_factor']:.2f}")
                    st.metric("Win Rate", f"{summary['overall']['win_rate']:.1f}%")
                    st.metric("Avg Win/Loss", f"{summary['overall']['avg_win']/summary['overall']['avg_loss']:.2f}x" if summary['overall']['avg_loss'] > 0 else "N/A")
                
                st.markdown("---")
                
                # ============================================================
                # SECTION 4: Exit Reason Distribution
                # ============================================================
                st.markdown("### 🎯 Exit Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Exit Reason Count")
                    exit_df = pd.DataFrame({
                        'Reason': list(summary['distribution']['by_exit_reason'].keys()),
                        'Count': list(summary['distribution']['by_exit_reason'].values())
                    })
                    st.dataframe(exit_df, use_container_width=True, hide_index=True)
                
                with col2:
                    st.markdown("#### P&L by Exit Reason")
                    pnl_df = pd.DataFrame({
                        'Reason': list(summary['distribution']['pnl_by_exit_reason'].keys()),
                        'P&L': [f"${v:.2f}" for v in summary['distribution']['pnl_by_exit_reason'].values()]
                    })
                    st.dataframe(pnl_df, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                
                # ============================================================
                # SECTION 5: Strategy Breakdown
                # ============================================================
                if summary['strategy_breakdown']:
                    st.markdown("### 🎲 Strategy Performance")
                    
                    strategy_data = []
                    for strategy, metrics in summary['strategy_breakdown'].items():
                        strategy_data.append({
                            'Strategy': strategy,
                            'Trades': metrics['total_trades'],
                            'Win Rate': f"{metrics['win_rate']:.1f}%",
                            'Total P&L': f"${metrics['total_pnl']:.2f}",
                            'Avg P&L': f"${metrics['avg_pnl']:.2f}"
                        })
                    
                    st.dataframe(
                        pd.DataFrame(strategy_data),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    st.markdown("---")
                
                # ============================================================
                # SECTION 6: Equity Curve
                # ============================================================
                st.markdown("### 📈 Equity Curve")
                
                equity_data = st.session_state.performance_analyzer.get_equity_curve(days=days)
                
                if not equity_data.empty:
                    # Create equity curve chart using existing chart_visualizer
                    fig = st.session_state.chart_visualizer.create_performance_chart(
                        equity_data.set_index('timestamp')['equity'],
                        title=f"Equity Curve - {period}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough data for equity curve")
                
                st.markdown("---")
                
                # ============================================================
                # SECTION 7: Efficiency Metrics
                # ============================================================
                st.markdown("### ⚡ Efficiency Metrics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Avg Trade Duration",
                        summary['efficiency']['avg_trade_duration']
                    )
                
                with col2:
                    st.metric(
                        "Trades/Day",
                        f"{summary['efficiency']['trades_per_day']:.2f}"
                    )
                
                with col3:
                    st.metric(
                        "Expectancy",
                        f"${summary['efficiency']['expectancy']:.2f}"
                    )
            
            else:
                st.info("📭 No trades in selected period")
                st.markdown("""
                **Analytics will show:**
                - Win rate and profit factor
                - Sharpe ratio and risk metrics
                - Drawdown analysis
                - Strategy-wise breakdown
                - Time-based performance
                - Equity curve visualization
                
                Start trading to see analytics! 🚀
                """)
        
        except Exception as e:
            st.error(f"❌ Analytics error: {e}")
            logger.error(f"Performance analytics error: {e}")
    
    else:
        st.error("Performance analyzer not initialized")

# ============================================================================
# AUTO-REFRESH (if trading active)
# ============================================================================

if st.session_state.trading_active:
    # Auto-refresh every 5 seconds
    time.sleep(5)
    st.rerun()

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p><strong>Paper Trading System v1.0</strong> | Crypto & Forex | SMC Strategies</p>
    <p><small>⚠️ This is paper trading only. No real money involved.</small></p>
</div>
""", unsafe_allow_html=True)

