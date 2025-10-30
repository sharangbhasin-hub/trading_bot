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
from unified_data_handler import UnifiedDataHandler, get_unified_handler
from strategy_manager import StrategyManager

# Import paper trading components
from paper_trading.config import PAPER_TRADING_CONFIG, get_config
from paper_trading.trade_database import TradeDatabase
from paper_trading.pnl_calculator import PnLCalculator, format_pnl
from paper_trading.live_data_manager import LiveDataManager
from paper_trading.paper_order_manager import PaperOrderManager
from paper_trading.chart_visualizer import ChartVisualizer
from paper_trading.performance_wrapper import PaperTradingPerformanceAnalyzer
from paper_trading.multi_symbol_manager import MultiSymbolManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# PARALLEL MULTI-SYMBOL PROCESSING
# ============================================================================
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

class ParallelSymbolProcessor:
    """
    ✅ Process multiple symbols in parallel using threads.
    Ensures concurrent signal generation and execution.
    No hallucination - uses Python's native threading.
    """
    
    def __init__(self, max_workers=5):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.signal_queue = queue.Queue()  # Thread-safe
        logger.info(f"🔄 Parallel processor: {max_workers} workers")
    
    def process_symbol_async(self, config: Dict) -> Dict:
        """Process single symbol in thread."""
        try:
            symbol = config['symbol']
            data_fetcher = config['data_fetcher']
            strategy_runner = config['strategy_runner']
            
            # Fetch data
            data = data_fetcher(symbol)
            if data is None or (hasattr(data, 'empty') and data.empty):
                return {'symbol': symbol, 'success': False, 'signal': None, 'timestamp': datetime.now()}
            
            # Run strategy
            signal = strategy_runner(symbol, data)
            
            if signal and signal.get('action') != 'NO_TRADE':
                self.signal_queue.put({
                    'symbol': symbol,
                    'signal': signal,
                    'timestamp': datetime.now(),
                    'thread_id': threading.current_thread().ident
                })
                
                return {'symbol': symbol, 'success': True, 'signal': signal, 'timestamp': datetime.now()}
            else:
                return {'symbol': symbol, 'success': False, 'signal': None, 'timestamp': datetime.now()}
        
        except Exception as e:
            logger.error(f"❌ Thread error for {config['symbol']}: {e}")
            return {'symbol': config['symbol'], 'success': False, 'error': str(e), 'timestamp': datetime.now()}
    
    def submit_batch(self, configs: List[Dict]):
        """Submit multiple symbols for parallel processing."""
        futures = {}
        for config in configs:
            future = self.executor.submit(self.process_symbol_async, config)
            futures[config['symbol']] = future
        return futures
    
    def get_signals(self, timeout: Optional[float] = None) -> List[Dict]:
        """Get all signals from queue (non-blocking)."""
        signals = []
        try:
            while True:
                signal_data = self.signal_queue.get_nowait()
                signals.append(signal_data)
        except queue.Empty:
            pass
        return signals
    
    def process_all_signals(self, signals: List[Dict]):
        """Process signals in timestamp order."""
        signals_sorted = sorted(signals, key=lambda x: x['timestamp'])
        for sig_data in signals_sorted:
            try:
                on_signal_generated({**sig_data['signal'], 'timestamp': sig_data['timestamp']})
            except Exception as e:
                logger.error(f"❌ Signal execution error: {e}")
    
    def shutdown(self):
        """Graceful shutdown."""
        self.executor.shutdown(wait=True)
        logger.info("🛑 Parallel processor shutdown")

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

    if 'multi_symbol_manager' not in st.session_state:
        st.session_state.multi_symbol_manager = None
    
    if 'active_symbols' not in st.session_state:
        st.session_state.active_symbols = []

    if 'symbol_category' not in st.session_state:  
        st.session_state.symbol_category = 'Cryptocurrencies'  

    # Statistics
    if 'signals_generated' not in st.session_state:
        st.session_state.signals_generated = 0
    if 'signals_executed' not in st.session_state:
        st.session_state.signals_executed = 0

    # ✅ Parallel processor for multi-symbol
    if 'parallel_processor' not in st.session_state:
        st.session_state.parallel_processor = None
    
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


def get_market_handler(market_type: str):
    """Get appropriate data handler for market type."""
    if market_type == 'Cryptocurrency (Binance)':
        return get_unified_handler(UnifiedDataHandler.MARKET_CRYPTO_BINANCE)
    elif market_type == 'Cryptocurrency (Alpaca)':
        return get_unified_handler(UnifiedDataHandler.MARKET_CRYPTO_ALPACA)
    elif 'Forex' in market_type:
        return get_unified_handler(UnifiedDataHandler.MARKET_FOREX)
    else:
        # Backwards compatibility: default old 'Cryptocurrency' to Binance
        return get_unified_handler(UnifiedDataHandler.MARKET_CRYPTO_BINANCE)

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

        # Multi-Symbol Manager (use current market's handler)
        current_handler = get_market_handler(st.session_state.market_type)
        st.session_state.multi_symbol_manager = MultiSymbolManager(
            order_manager=st.session_state.order_manager,
            data_handler=current_handler,
            strategy_manager=st.session_state.strategy_manager
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

def get_timeframes(trading_mode: str) -> tuple:
    """Get HTF and LTF based on trading mode."""
    if 'Scalping' in trading_mode:
        return ('1h', '1min')
    else:  # Intraday
        return ('D', '1h')

def get_symbol_list(market_type: str, category: str = None) -> List[str]:
    """
    Get list of tradable symbols for market (DYNAMIC).
    Fetches live symbols from exchange handlers.
    
    Args:
        market_type: Market type string
        category: Symbol category (for Alpaca only)
    """
    try:
        if market_type == 'Cryptocurrency (Binance)':
            # Static list for Binance (CCXT doesn't have get_available_symbols)
            return PAPER_TRADING_CONFIG['crypto']['supported_pairs']
        
        elif market_type == 'Cryptocurrency (Alpaca)':
            # ✅ DYNAMIC: Fetch from Alpaca handler with category filtering
            handler = get_unified_handler(UnifiedDataHandler.MARKET_CRYPTO_ALPACA)
            
            if handler and hasattr(handler, 'get_available_symbols_by_category'):
                # Default category if none provided
                if not category:
                    category = 'Cryptocurrencies'
                
                assets = handler.get_available_symbols_by_category(category)
                
                # Convert BTCUSD -> BTC/USD format
                symbols = []
                for asset in assets:
                    symbol = asset['symbol']
                    # Convert BTCUSD to BTC/USD
                    if symbol.endswith('USD') and len(symbol) > 3:
                        formatted = f"{symbol[:-3]}/{symbol[-3:]}"
                        symbols.append(formatted)
                    else:
                        symbols.append(symbol)
                
                return symbols if symbols else ['BTC/USD', 'ETH/USD', 'LTC/USD', 'BCH/USD']
            else:
                # Fallback to hardcoded list
                return ['BTC/USD', 'ETH/USD', 'LTC/USD', 'BCH/USD']
        
        elif 'Forex' in market_type:
            # ✅ DYNAMIC: Fetch from OANDA handler with category filtering
            handler = get_unified_handler(UnifiedDataHandler.MARKET_FOREX)
            
            if handler and hasattr(handler, 'get_available_symbols_by_category'):
                # Default category if none provided
                if not category:
                    category = 'Major Forex Pairs'
                
                instruments = handler.get_available_symbols_by_category(category)
                
                # Convert EUR_USD -> EUR/USD format
                symbols = []
                for instrument in instruments:
                    symbol = instrument['symbol']
                    # Convert EUR_USD to EUR/USD
                    formatted = symbol.replace('_', '/')
                    symbols.append(formatted)
                
                return symbols if symbols else ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD']
            else:
                # Fallback to config
                return PAPER_TRADING_CONFIG['forex']['supported_pairs']
        
        else:
            # Backwards compatibility
            return PAPER_TRADING_CONFIG['crypto']['supported_pairs']
    
    except Exception as e:
        logger.error(f"Error fetching symbols for {market_type}: {e}")
        # Fallback based on market type
        if 'Alpaca' in market_type:
            return ['BTC/USD', 'ETH/USD', 'LTC/USD', 'BCH/USD']
        elif 'Forex' in market_type:
            return PAPER_TRADING_CONFIG['forex']['supported_pairs']
        else:
            return PAPER_TRADING_CONFIG['crypto']['supported_pairs']

def get_alpaca_categories() -> List[str]:
    """
    Get list of available categories for Alpaca.
    
    Returns:
        List of category names
    """
    try:
        handler = get_unified_handler(UnifiedDataHandler.MARKET_CRYPTO_ALPACA)
        
        if handler and hasattr(handler, 'available_symbols'):
            # Get categories from handler
            return list(handler.available_symbols.keys())
        else:
            # Fallback to known categories
            return ['Popular Stocks', 'Tech Stocks', 'ETFs', 'Cryptocurrencies']
    
    except Exception as e:
        logger.error(f"Error fetching Alpaca categories: {e}")
        return ['Cryptocurrencies']  # Safe fallback

def get_forex_categories() -> List[str]:
    """
    Get list of available categories for Forex (OANDA).
    
    Returns:
        List of category names
    """
    try:
        handler = get_unified_handler(UnifiedDataHandler.MARKET_FOREX)
        
        if handler and hasattr(handler, 'available_symbols'):
            # Get categories from handler
            return list(handler.available_symbols.keys())
        else:
            # Fallback to common categories
            return ['Major Forex Pairs', 'Minor Forex Pairs', 'Exotic Forex Pairs']
    
    except Exception as e:
        logger.error(f"Error fetching Forex categories: {e}")
        return ['Major Forex Pairs']  # Safe fallback

def get_api_symbol() -> str:
    """
    Get the normalized API symbol for buffer access.
    
    Returns:
        str: Normalized symbol (EUR_USD for Forex, EUR/USD for Crypto)
    """
    return st.session_state.get('api_symbol', st.session_state.symbol)

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
    🔥 PRODUCTION CALLBACK - Native Dual-Fetch Architecture
    
    Fetches NATIVE HTF and LTF data separately from exchange.
    NO resampling - uses exchange-official OHLC values.
    Mirrors real money trading architecture.
    
    ⚠️ NOTE: This runs in a background thread, so we access stored references
    instead of st.session_state directly.
    """
    try:
        # ✅ Check required references
        if not hasattr(on_new_candle, 'data_handler'):
            logger.error("⚠️ data_handler not initialized in callback")
            return
        
        # ✅ Get timeframes from strategy config
        htf, ltf = get_timeframes(on_new_candle.trading_mode)
        
        # ============================================================
        # 🔥 PRODUCTION: Fetch HTF data NATIVELY from exchange
        # ============================================================
        try:
            # Calculate date range for HTF
            end_time = datetime.now()
            
            # Determine start time based on HTF period
            if htf == '1d':
                start_time = end_time - timedelta(days=30)  # 30 days
            elif htf == '1h':
                start_time = end_time - timedelta(hours=30)  # 30 hours
            elif htf == '4h':
                start_time = end_time - timedelta(hours=120)  # 5 days
            else:
                start_time = end_time - timedelta(days=30)  # Default
            
            df_htf = on_new_candle.data_handler.get_historical_data(
                symbol=symbol,
                start_date=start_time,
                end_date=end_time,
                timeframe=htf,  # '1d' for Intraday, '1h' for Scalping
                use_cache=False  # ✅ Always fetch fresh data for live trading
            )
            
            if df_htf is None or len(df_htf) < 10:
                logger.warning(f"Insufficient HTF data for {symbol}: {len(df_htf) if df_htf is not None else 0} candles")
                return
            
            # Set index if needed
            if 'timestamp' in df_htf.columns and df_htf.index.name != 'timestamp':
                df_htf.set_index('timestamp', inplace=True)
            elif 'date' in df_htf.columns and df_htf.index.name != 'date':
                df_htf.set_index('date', inplace=True)
            
            logger.debug(f"✅ Fetched {len(df_htf)} native {htf} candles for HTF analysis")
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch HTF data for {symbol}: {str(e)}")
            return
        
        # ============================================================
        # 🔥 PRODUCTION: Fetch LTF data NATIVELY from exchange
        # ============================================================
        try:
            # Calculate date range for LTF
            end_time = datetime.now()
            
            # Determine start time based on LTF period
            if ltf == '1min':
                start_time = end_time - timedelta(hours=3)  # 3 hours
            elif ltf == '5min':
                start_time = end_time - timedelta(hours=12)  # 12 hours
            elif ltf == '15min':
                start_time = end_time - timedelta(hours=36)  # 36 hours
            elif ltf == '1h':
                start_time = end_time - timedelta(days=7)  # 7 days
            else:
                start_time = end_time - timedelta(days=7)  # Default
            
            df_ltf = on_new_candle.data_handler.get_historical_data(
                symbol=symbol,
                start_date=start_time,
                end_date=end_time,
                timeframe=ltf,  # '1h' for Intraday, '1min' for Scalping
                use_cache=False  # ✅ Always fetch fresh data for live trading
            )
            
            if df_ltf is None or len(df_ltf) < 50:
                logger.warning(f"Insufficient LTF data for {symbol}: {len(df_ltf) if df_ltf is not None else 0} candles")
                return
            
            # Set index if needed
            if 'timestamp' in df_ltf.columns and df_ltf.index.name != 'timestamp':
                df_ltf.set_index('timestamp', inplace=True)
            elif 'date' in df_ltf.columns and df_ltf.index.name != 'date':
                df_ltf.set_index('date', inplace=True)
            
            logger.debug(f"✅ Fetched {len(df_ltf)} native {ltf} candles for LTF analysis")
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch LTF data for {symbol}: {str(e)}")
            return
        
        # ============================================================
        # Load Strategy
        # ============================================================
        if on_new_candle.strategy_name == 'CRT-TBS':
            from strategies.strategy_crt_tbs import StrategyCRTTBS
            
            # Create instance once and reuse
            if not hasattr(on_new_candle, 'strategy_instance'):
                mode_name = on_new_candle.trading_mode.split('(')[0].strip().lower()
                
                on_new_candle.strategy_instance = StrategyCRTTBS(
                    market_type=on_new_candle.market_type,
                    config_name=mode_name
                )
                logger.info(f"✅ Strategy initialized: {on_new_candle.market_type} | {mode_name}")
            
            strategy_module = on_new_candle.strategy_instance
        else:
            logger.error(f"❌ Strategy '{on_new_candle.strategy_name}' not yet supported")
            return
        
        # ============================================================
        # 🔥 PRODUCTION: Run strategy with NATIVE exchange data
        # ============================================================
        result = strategy_module.generate_signals(
            df_htf=df_htf,  # Native HTF candles (1D for Intraday, 1H for Scalping)
            df_ltf=df_ltf   # Native LTF candles (1H for Intraday, 1min for Scalping)
        )
        
        # ============================================================
        # Process Signal (if generated)
        # ============================================================
        if result and result.get('action') != 'NO_TRADE':
            # Valid signal generated
            signal_data = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'direction': result['action'],
                'entry_price': result['entry_price'],
                'stop_loss': result['stop_loss'],
                'take_profit': result['take_profit_1'],
                'strategy_name': on_new_candle.strategy_name,
                'confidence': result.get('confidence', 0),
                'risk_reward_ratio': result.get('rr_ratio', 0),
                'market_type': 'crypto' if 'Cryptocurrency' in on_new_candle.market_type else 'forex',
                'reasoning': result.get('reasoning', [])
            }
            
            # Store signal
            # st.session_state.latest_signal = signal_data
            # st.session_state.signals_generated += 1
            # st.session_state.last_update_time = datetime.now()
            
            # Log to database
            on_new_candle.trade_db.insert_signal({
                'timestamp': signal_data['timestamp'],
                'symbol': symbol,
                'strategy_name': on_new_candle.strategy_name,
                'signal_type': signal_data['direction'],
                'entry_price': signal_data['entry_price'],
                'stop_loss': signal_data['stop_loss'],
                'take_profit': signal_data['take_profit'],
                'confidence': signal_data['confidence'],
                'reasoning': str(signal_data['reasoning']),
                'executed': False
            })
            
            logger.info(f"🔔 Signal generated: {signal_data['direction']} {symbol} @ {signal_data['entry_price']}")
            # ✅ TRIGGER SIGNAL EXECUTION CALLBACK
            try:
                on_signal_generated({
                    'action': signal_data['direction'],
                    'entry_price': signal_data['entry_price'],
                    'stop_loss': signal_data.get('stop_loss'),
                    'take_profit_1': signal_data.get('take_profit'),
                    'take_profit_2': signal_data.get('tp2', signal_data.get('take_profit')),
                    'symbol': symbol,
                    'rr_ratio': signal_data.get('risk_reward_ratio', 1.0),
                    'confidence': signal_data.get('confidence', 50),
                    'reasoning': signal_data.get('reasoning', [])
                })
            except Exception as e:
                logger.error(f"❌ Signal callback error: {e}")

        # Update latest price
        # st.session_state.latest_price = candle.get('close', 0)
        # st.session_state.last_update_time = datetime.now()
        
        # Check open positions for SL/TP
        current_price = candle.get('close', 0)
        on_new_candle.order_manager.check_open_positions(symbol, current_price, current_price) 
        
    except Exception as e:
        logger.error(f"❌ Error in callback: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

# ============================================================================
# ON_SIGNAL_GENERATED CALLBACK - EXECUTE TRADES FROM SIGNALS
# ============================================================================

def on_signal_generated(signal: Dict):
    """
    ✅ COMPLETE FIXED VERSION: Proper deduplication with tolerance + all existing features
    Execute trades from signals in background thread
    """
    # ✅ ALL IMPORTS AT TOP OF FUNCTION
    import json
    from pathlib import Path
    from datetime import datetime
    
    try:
        logger.info(f"🔔 Signal generated callback triggered")
        
        # ✅ STEP 1: Extract signal details
        action = signal.get('action', 'NO_TRADE')
        if action == 'NO_TRADE':
            logger.info("⏭ No valid trade setup")
            return
        
        # ✅ Extract basic details ONCE
        symbol = signal.get('symbol', 'UNKNOWN')
        entry_price = float(signal.get('entry_price', 0))
        stop_loss = float(signal.get('stop_loss', 0))
        tp1 = signal.get('take_profit_1')
        tp2 = signal.get('take_profit_2')
        
        if tp1 is not None:
            tp1 = float(tp1)
        
        # ✅ STEP 2: DEDUPLICATION WITH TOLERANCE (CRITICAL FIX!)
        pending_file = Path("paper_trading/data/pending_signals.json")
        DUPLICATE_TOLERANCE = 0.00001  # Float comparison tolerance (5 decimals)
        DUPLICATE_TIME_WINDOW = 5  # Minutes
        OLD_SIGNAL_AGE = 30  # Minutes - clean up signals older than this
        
        duplicate_detected = False
        fresh_pending_signals = []
        
        if pending_file.exists():
            try:
                with open(pending_file, 'r') as f:
                    pending_signals = json.load(f)
                
                current_time = datetime.now()
                
                # ✅ CHECK ALL EXISTING SIGNALS FOR DUPLICATES
                for existing_signal in pending_signals:
                    try:
                        existing_timestamp = existing_signal.get('timestamp')
                        signal_time = datetime.fromisoformat(existing_timestamp)
                        time_diff_minutes = (current_time - signal_time).total_seconds() / 60
                        
                        # ✅ TOLERANCE-BASED FLOAT COMPARISON (NOT EXACT MATCH!)
                        existing_entry = float(existing_signal.get('entry_price', 0))
                        existing_stop = float(existing_signal.get('stop_loss', 0))
                        existing_tp1 = float(existing_signal.get('take_profit_1', 0)) if existing_signal.get('take_profit_1') else 0
                        
                        # Check for duplicate using tolerance
                        is_duplicate = (
                            existing_signal.get('symbol') == symbol and
                            existing_signal.get('action') == action and
                            abs(existing_entry - entry_price) < DUPLICATE_TOLERANCE and
                            abs(existing_stop - stop_loss) < DUPLICATE_TOLERANCE and
                            abs(existing_tp1 - tp1) < DUPLICATE_TOLERANCE if tp1 else False
                        )
                        
                        if is_duplicate and time_diff_minutes < DUPLICATE_TIME_WINDOW:
                            # ✅ DUPLICATE FOUND AND RECENT - REJECT IT!
                            logger.warning(f"⏭️ DUPLICATE SIGNAL BLOCKED!")
                            logger.warning(f"   Symbol: {symbol} | Action: {action}")
                            logger.warning(f"   Entry: ${entry_price:.5f} (existing: ${existing_entry:.5f})")
                            logger.warning(f"   SL: ${stop_loss:.5f} (existing: ${existing_stop:.5f})")
                            logger.warning(f"   TP: ${tp1:.5f} (existing: ${existing_tp1:.5f})" if tp1 else "")
                            logger.warning(f"   Generated {time_diff_minutes:.1f}m ago")
                            duplicate_detected = True
                            break  # ← EXIT - Don't process this signal
                        
                        # ✅ CLEAN UP OLD SIGNALS (> 30 minutes)
                        if time_diff_minutes <= OLD_SIGNAL_AGE:
                            # Keep this signal (it's fresh)
                            fresh_pending_signals.append(existing_signal)
                        else:
                            # Remove old signal
                            logger.info(f"🧹 Removed old signal: {existing_signal.get('symbol')} (age: {time_diff_minutes:.0f}m)")
                    
                    except Exception as e:
                        logger.debug(f"Error processing existing signal: {e}")
                        # Keep signal if error occurs
                        fresh_pending_signals.append(existing_signal)
            
            except Exception as e:
                logger.debug(f"Deduplication check error: {e}")
                fresh_pending_signals = []
        
        # ✅ IF DUPLICATE DETECTED - STOP HERE!
        if duplicate_detected:
            logger.warning(f"🛑 Signal rejected (duplicate within {DUPLICATE_TIME_WINDOW}m)")
            return
        
        # ✅ STEP 3: Extract remaining signal details (only if not duplicate)
        rr_ratio = signal.get('rr_ratio', signal.get('risk_reward_ratio', 1.0))
        confidence = signal.get('confidence', 50)
        
        logger.info(f"✅ Signal is UNIQUE - proceeding with execution")
        logger.info(f"   Action: {action} | Entry: {entry_price} | SL: {stop_loss} | TP1: {tp1} | TP2: {tp2}")
        
        # ✅ STEP 4: Log signal details
        logger.info(f"📊 Signal Details:")
        logger.info(f"   Symbol: {symbol}")
        logger.info(f"   Type: {action}")
        logger.info(f"   Entry: {entry_price}")
        logger.info(f"   SL: {stop_loss}")
        logger.info(f"   TP1: {tp1}")
        logger.info(f"   Confidence: {confidence}%")
        
        # ✅ STEP 5: Calculate position size based on risk
        try:
            current_balance = PAPER_TRADING_CONFIG.get('initial_balance', 10000.0)
            risk_pct = PAPER_TRADING_CONFIG['risk_management']['risk_per_trade_pct']
            risk_amount = current_balance * (risk_pct / 100)
            
            # Calculate pip distance for position sizing
            pip_distance = abs(entry_price - stop_loss) * 10000  # Assume 4 decimal places
            
            if pip_distance > 0:
                position_size = risk_amount / pip_distance
            else:
                logger.error(f"❌ Invalid SL distance: {pip_distance}")
                return
            
            # Cap position size
            position_size = min(position_size, 1.0)  # Max 1 lot
            position_size = max(position_size, 0.01)  # Min 0.01 lot
            
            logger.info(f"   Position Size Calculated: {position_size:.4f} lots")
            logger.info(f"   Risk Amount: ${risk_amount:.2f} | Pip Distance: {pip_distance:.1f} pips")
            
        except Exception as e:
            logger.error(f"❌ Position sizing error: {e}")
            position_size = 0.1  # Fallback
            current_balance = 10000.0
            risk_amount = 0
            pip_distance = 0
            risk_pct = 0
        
        # ✅ STEP 6: Save signal to file for main thread to process
        try:
            pending_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Use cleaned up signals list (with old signals removed)
            pending_signals = fresh_pending_signals if fresh_pending_signals else []
            
            # ✅ ADD NEW SIGNAL TO QUEUE
            pending_signals.append({
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'action': action,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit_1': tp1,
                'take_profit_2': tp2,
                'confidence': confidence,
                'rr_ratio': rr_ratio,
                'strategy_name': signal.get('strategy_name', 'CRT-TBS'),
                'reasoning': signal.get('reasoning', []),
                # ✅ INCLUDE CALCULATIONS:
                'position_size': position_size,
                'risk_amount': risk_amount,
                'pip_distance': pip_distance,
                'current_balance': current_balance,
                'risk_pct': risk_pct
            })
            
            # Save all signals back to file
            with open(pending_file, 'w') as f:
                json.dump(pending_signals, f, indent=2, default=str)
            
            logger.info(f"✅ Signal saved to pending queue: {pending_file}")
            logger.info(f"📊 Total pending signals: {len(pending_signals)}")
            logger.info("📢 Main thread will process this signal for execution")
        
        except Exception as e:
            logger.error(f"❌ Error saving signal to queue: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    except Exception as e:
        logger.error(f"❌ Fatal error in signal callback: {e}")
        import traceback
        logger.error(traceback.format_exc())

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
        
        # ✅ Normalize symbol for API (Forex and Alpaca Crypto need different formats)
        api_symbol = st.session_state.symbol
        
        if 'Forex' in st.session_state.market_type:
            # Forex: EUR/USD -> EUR_USD
            api_symbol = st.session_state.symbol.replace('/', '_')
            logger.info(f"🔄 Normalized Forex symbol for feed: {st.session_state.symbol} -> {api_symbol}")
        
        elif 'Alpaca' in st.session_state.market_type:
            # Alpaca Crypto: BTC/USD -> BTCUSD (no slash)
            api_symbol = st.session_state.symbol.replace('/', '')
            logger.info(f"🔄 Normalized Alpaca symbol for feed: {st.session_state.symbol} -> {api_symbol}")
        
        # ✅ FIX: Store references for callback thread
        on_new_candle.data_manager = st.session_state.data_manager
        on_new_candle.strategy_manager = st.session_state.strategy_manager
        on_new_candle.order_manager = st.session_state.order_manager
        on_new_candle.trade_db = st.session_state.trade_db
        on_new_candle.strategy_name = st.session_state.strategy_name
        on_new_candle.market_type = st.session_state.market_type
        on_new_candle.trading_mode = st.session_state.trading_mode
        on_new_candle.data_handler = handler
        
        logger.info(f"✅ Callback references stored for {api_symbol}")
        
        # Start live feed (using normalized symbol)
        success = st.session_state.data_manager.start_feed(
            symbol=api_symbol,  # ← FIXED: Now uses 'EUR_USD' for Forex
            timeframe=ltf,
            on_new_candle=on_new_candle
        )
        
        if success:
            st.session_state.trading_active = True
            # Store normalized symbol for internal use
            st.session_state.api_symbol = api_symbol
            logger.info(f"✅ Paper trading started: {api_symbol} ({st.session_state.symbol}), {st.session_state.strategy_name}")
            return True
        else:
            logger.error("Failed to start data feed")
            return False
            
    except Exception as e:
        logger.error(f"Failed to start paper trading: {e}")
        import traceback
        logger.error(traceback.format_exc())
        st.error(f"❌ Failed to start: {e}")
        return False

def stop_paper_trading():
    """Stop paper trading."""
    try:
        if st.session_state.data_manager:
            st.session_state.data_manager.stop_feed(get_api_symbol())
        
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
# PROCESS PENDING SIGNALS FROM BACKGROUND THREADS (Main Thread)
# ============================================================================

def process_pending_signals():
    """
    ✅ Process pending signals saved by background thread callbacks.
    This runs in MAIN THREAD - can safely access st.session_state.
    Executes trades from pending_signals.json
    
    ✅ IMPROVED: Tracks failure count, removes old signals, cleans up stuck signals
    """
    try:
        from pathlib import Path
        import json
        from datetime import datetime  # ← ADD THIS
        
        pending_file = Path("paper_trading/data/pending_signals.json")
        
        if not pending_file.exists():
            return  # No pending signals
        
        # Load pending signals
        with open(pending_file, 'r') as f:
            pending_signals = json.load(f)
        
        if not pending_signals:
            return  # Empty list
        
        logger.info(f"📊 Processing {len(pending_signals)} pending signals from thread...")
        
        executed_signals = []
        failed_signals = []  # ← NEW: Track failed signals
        current_time = datetime.now()  # ← NEW: For age check
        
        # Process each signal
        for signal_data in pending_signals:
            try:
                # ✅ NEW: Check if signal is too old (30+ minutes)
                signal_time = datetime.fromisoformat(signal_data['timestamp'])
                age_minutes = (current_time - signal_time).total_seconds() / 60
                
                if age_minutes > 30:
                    logger.warning(f"🧹 Removing old signal: {signal_data['symbol']} (age: {age_minutes:.0f}m, never executed)")
                    executed_signals.append(signal_data)  # Mark for removal
                    continue  # Skip to next signal
                
                logger.info(f"🔄 Executing signal: {signal_data['symbol']} {signal_data['action']}")
                
                # ✅ NOW we CAN execute from main thread
                if not st.session_state.order_manager:
                    logger.error("Order manager not available")
                    failed_signals.append(signal_data)  # Keep for retry
                    continue
                
                # Map signal data to order format
                order_params = {
                    'symbol': signal_data['symbol'],
                    'direction': signal_data['action'],  # 'BUY' or 'SELL'
                    'entry_price': signal_data['entry_price'],
                    'stop_loss': signal_data['stop_loss'],
                    'take_profit': signal_data.get('take_profit_1', signal_data.get('take_profit')),
                    'strategy_name': signal_data.get('strategy_name', 'CRT-TBS'),
                    'confidence': signal_data.get('confidence', 50),
                    'market_type': 'forex' if 'forex' in signal_data.get('symbol', '').lower() else 'crypto'
                }
                
                # Execute trade
                order_result = st.session_state.order_manager.place_order(
                    signal=order_params,
                    current_price=signal_data['entry_price'],
                    exchange_price=signal_data['entry_price']
                )
                
                if order_result.get('success'):
                    logger.info(f"✅ Signal EXECUTED: {signal_data['symbol']} | Trade ID: {order_result.get('trade_id')}")
                    executed_signals.append(signal_data)
                    st.session_state.signals_executed += 1
                else:
                    # ✅ NEW: Track failure count
                    if 'failure_count' not in signal_data:
                        signal_data['failure_count'] = 0
                    
                    signal_data['failure_count'] += 1
                    reason = order_result.get('reason', 'Unknown')
                    
                    logger.error(f"❌ Execution failed (attempt {signal_data['failure_count']}): {reason}")
                    
                    # ✅ NEW: If failed 10+ times, remove it
                    if signal_data['failure_count'] >= 10:
                        logger.error(f"🧹 Removing signal after 10 failed attempts: {signal_data['symbol']}")
                        executed_signals.append(signal_data)  # Mark for removal
                    else:
                        failed_signals.append(signal_data)  # Keep for retry
            
            except Exception as e:
                logger.error(f"❌ Error processing signal: {e}")
                import traceback
                logger.error(traceback.format_exc())
                failed_signals.append(signal_data)  # Keep for retry
        
        # Update database - mark signals as executed
        try:
            for sig in executed_signals:
                st.session_state.trade_db.update_signal_executed(
                    timestamp=sig['timestamp'],
                    symbol=sig['symbol']
                )
        except:
            pass  # OK if database update fails
        
        # ✅ NEW: Save only failed signals (not executed, not too old, <10 failures)
        remaining_signals = failed_signals
        
        if remaining_signals:
            with open(pending_file, 'w') as f:
                json.dump(remaining_signals, f, indent=2, default=str)
            logger.info(f"📊 {len(remaining_signals)} signals remain pending")
        else:
            try:
                pending_file.unlink()  # Delete file if empty
            except:
                pass
            logger.info("✅ All pending signals processed or removed")
    
    except Exception as e:
        logger.error(f"❌ Error in process_pending_signals: {e}")
        import traceback
        logger.error(traceback.format_exc())

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

    # ✅ ADD THIS SECTION:
    # OANDA Status
    if st.session_state.order_manager and st.session_state.order_manager.oanda_handler:
        st.success("✅ **OANDA PRACTICE: CONNECTED**")
        
        # Show OANDA account balance
        try:
            summary = st.session_state.order_manager.oanda_handler.get_account_summary()
            if summary:
                st.metric(
                    "OANDA Balance",
                    f"${summary['balance']:,.2f}",
                    delta=f"Unrealized: ${summary['unrealized_pl']:,.2f}"
                )
        except:
            pass
    elif PAPER_TRADING_CONFIG['forex']['oanda_enabled']:
        st.warning("⚠️ **OANDA: OFFLINE (Local Simulation)**")
    
    st.markdown("---")
    
    # Market Selection
    st.subheader("⚙️ Configuration")
    
    # ✅ NEW: Flat list like backtesting (no nested dropdowns)
    market_options = [
        'Cryptocurrency (Binance)',
        'Cryptocurrency (Alpaca)',
        'Forex (OANDA)'
    ]
    
    # Determine current index
    current_market = st.session_state.market_type
    if current_market == 'Cryptocurrency':
        current_index = 0  # Default to Binance for backwards compatibility
    elif current_market == 'Cryptocurrency (Binance)':
        current_index = 0
    elif current_market == 'Cryptocurrency (Alpaca)':
        current_index = 1
    elif 'Forex' in current_market:
        current_index = 2
    else:
        current_index = 0
    
    market_type = st.selectbox(
        "Market Type",
        market_options,
        index=current_index,
        disabled=st.session_state.trading_active
    )
    st.session_state.market_type = market_type
    
    # Symbol Selection  
    # Check if Alpaca or Forex is selected (both need category selection)
    if market_type == 'Cryptocurrency (Alpaca)':
        # Show category selector for Alpaca
        categories = get_alpaca_categories()
        
        # Determine current category index
        current_category = st.session_state.get('symbol_category', 'Cryptocurrencies')
        current_category_index = categories.index(current_category) if current_category in categories else 0
        
        selected_category = st.selectbox(
            "Category",
            categories,
            index=current_category_index,
            disabled=st.session_state.trading_active,
            key="category_select"
        )
        st.session_state.symbol_category = selected_category
        
        # Get symbols for selected category
        symbols = get_symbol_list(market_type, category=selected_category)
    
    elif 'Forex' in market_type:
        # Show category selector for Forex
        categories = get_forex_categories()
        
        # Determine current category index
        current_category = st.session_state.get('symbol_category', 'Major Forex Pairs')
        current_category_index = categories.index(current_category) if current_category in categories else 0
        
        selected_category = st.selectbox(
            "Category",
            categories,
            index=current_category_index,
            disabled=st.session_state.trading_active,
            key="category_select"
        )
        st.session_state.symbol_category = selected_category
        
        # Get symbols for selected category
        symbols = get_symbol_list(market_type, category=selected_category)
    
    else:
        # For Binance, no category selection needed
        symbols = get_symbol_list(market_type)
    
    # Symbol selector (shown for all market types)
    symbol = st.selectbox(
        "Symbol",
        symbols,
        index=symbols.index(st.session_state.symbol) if st.session_state.symbol in symbols else 0,
        disabled=st.session_state.trading_active,
        key="symbol_select"
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
            if st.button("▶️ START", type="primary", width='stretch'):
                with st.spinner("Starting paper trading..."):
                    if start_paper_trading():
                        st.success("✅ Started!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Failed to start")
    
    with col2:
        if st.session_state.trading_active:
            if st.button("⏹ STOP", type="secondary", width='stretch'):
                if stop_paper_trading():
                    st.success("⏹ Stopped!")
                    time.sleep(1)
                    st.rerun()
    
    # Quick Actions
    st.markdown("---")
    st.subheader("⚡ Quick Actions")
    
    if st.button("📥 Export Trades to CSV", width='stretch'):
        try:
            filename = f"paper_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            st.session_state.trade_db.export_to_csv(filename)
            st.success(f"✅ Exported to {filename}")
        except Exception as e:
            st.error(f"❌ Export failed: {e}")
    
    if st.button("💾 Backup Database", width='stretch'):
        try:
            backup_path = st.session_state.trade_db.backup_database()
            st.success(f"✅ Backup created!")
        except Exception as e:
            st.error(f"❌ Backup failed: {e}")
    
    if st.button("🔄 Reset Daily Stats", width='stretch'):
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
    
    # ✅✅✅ ADD THIS ENTIRE SECTION HERE ✅✅✅
    # Multi-Symbol Overview
    if st.session_state.multi_symbol_manager and st.session_state.multi_symbol_manager.active_symbols:
        st.markdown("### 🔀 Multi-Symbol Overview")
        
        portfolio = st.session_state.multi_symbol_manager.get_portfolio_summary()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Symbols", portfolio['total_symbols'])
        
        with col2:
            total_pnl = sum(p['pnl'] for p in portfolio['symbols_performance'].values())
            st.metric("Portfolio P&L", f"${total_pnl:.2f}")
        
        with col3:
            total_signals = sum(p['signals_generated'] for p in portfolio['symbols_performance'].values())
            st.metric("Total Signals", total_signals)
        
        with col4:
            st.metric("Portfolio Positions", portfolio['total_open_positions'])
        
        st.markdown("---")

# ============================================================================
# TABS: Dashboard | Signals | Positions | History | Settings
# ============================================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📊 Dashboard", 
    "🔔 Signals", 
    "📋 Positions", 
    "📜 History", 
    "⚙️ Settings", 
    "📈 Chart",
    "📈 Analytics",
    "🌐 OANDA",
    "🔀 Multi-Symbol"
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
            feed_status = st.session_state.data_manager.get_feed_status(get_api_symbol())
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
                recent_candles = st.session_state.data_manager.get_recent_candles(get_api_symbol(), 10)

                if recent_candles:
                    df_recent = pd.DataFrame(recent_candles[-10:])
                    
                    # Format display columns
                    display_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume'] if 'timestamp' in df_recent.columns else ['date', 'open', 'high', 'low', 'close', 'volume']
                    
                    st.dataframe(
                        df_recent[display_cols].tail(10),
                        width='stretch',
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
                width='stretch',
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
            if st.button("✅ Execute Trade", type="primary", width='stretch'):
                if execute_signal(signal):
                    st.success("✅ Trade executed successfully!")
                    time.sleep(1)
                    st.rerun()
        
        with col2:
            if st.button("❌ Dismiss Signal", width='stretch'):
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
    if st.session_state.trade_db:
        try:
            recent_signals = st.session_state.trade_db.get_all_signals(limit=10)
            
            if recent_signals:
                signals_display = []
                for sig in recent_signals:
                    signals_display.append({
                        'Time': sig['timestamp'],
                        'Symbol': sig['symbol'],
                        'Type': '🟢 BUY' if sig['signal_type'] == 'BUY' else '🔴 SELL',
                        'Entry': f"${sig['entry_price']:.5f}",
                        'SL': f"${sig['stop_loss']:.5f}",
                        'TP': f"${sig['take_profit']:.5f}",
                        'Confidence': f"{sig.get('confidence', 0)}%",
                        'Executed': '✅' if sig.get('executed') else '⏳',
                        'Strategy': sig.get('strategy_name', 'Unknown')
                    })
                
                st.dataframe(
                    pd.DataFrame(signals_display),
                    width='stretch',
                    hide_index=True
                )
            else:
                st.info("📭 No signals generated yet.")
        
        except Exception as e:
            st.error(f"❌ Error loading signals: {e}")
            logger.error(f"Signal history error: {e}")

# ----------------------------------------------------------------------------
# TAB 3: OPEN POSITIONS
# ----------------------------------------------------------------------------

with tab3:
    st.subheader("📋 Open Positions")
    
    if st.session_state.multi_symbol_manager and st.session_state.multi_symbol_manager.active_symbols:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            symbol_filter = st.multiselect(
                "Filter by Symbol",
                options=['All'] + list(st.session_state.multi_symbol_manager.active_symbols),
                default=['All']
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            refresh_positions = st.button("🔄 Refresh", key="refresh_positions_btn")
    
    if st.session_state.order_manager:
        open_positions = st.session_state.order_manager.get_open_positions_list()
        
        # Filter positions by selected symbols
        if 'symbol_filter' in locals() and 'All' not in symbol_filter:
            open_positions = [p for p in open_positions if p['symbol'] in symbol_filter]

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
            st.dataframe(df_positions, width='stretch', hide_index=True)
            
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
            # Create DataFrame with trade details
            trades_data = []
            for trade in closed_trades:
                # Calculate P&L if not already calculated
                pnl = trade.get('pnl_usd', 0) or 0
                pnl_pct = trade.get('pnl_pct', 0) or 0
                amount = float(trade.get('quantity', 0) or 0) * float(trade.get('entry_price', 0) or 1)
                
                trades_data.append({
                    'ID': f"#{trade['id']}",
                    'Time': trade['timestamp'],
                    'Symbol': trade['symbol'],
                    'Direction': '🟢 BUY' if trade['direction'] == 'BUY' else '🔴 SELL',
                    'Quantity': f"{trade.get('quantity', 0):.4f}",
                    'Amount ($)': f"${amount:,.2f}",
                    'Entry': f"${trade['entry_price']:,.5f}",
                    'Exit': f"${trade.get('exit_price', 0):,.5f}" if trade.get('exit_price') else "-",
                    'P&L ($)': f"${pnl:,.2f}",
                    'P&L (%)': f"{pnl_pct:+.2f}%",
                    'Exit Reason': trade.get('exit_reason', '-'),
                    'Strategy': trade['strategy_name'],
                    'Confidence': f"{trade.get('confidence', 0)}%"
                })
            
            df_trades = pd.DataFrame(trades_data)
            
            # Color code the P&L columns
            st.dataframe(
                df_trades,
                width='stretch',
                hide_index=True
            )
            
            # Summary statistics
            st.markdown("---")
            st.subheader("📊 Period Statistics")
            
            total_trades = len(closed_trades)
            winning_trades = [t for t in closed_trades if (t.get('pnl_usd') or 0) > 0]
            losing_trades = [t for t in closed_trades if (t.get('pnl_usd') or 0) < 0]
            
            total_pnl = sum(t.get('pnl_usd', 0) or 0 for t in closed_trades)
            total_amount = sum(
                float(t.get('quantity', 0) or 0) * float(t.get('entry_price', 0) or 1) 
                for t in closed_trades
            )
            
            win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
            
            # Calculate profit factor
            total_wins = sum(t.get('pnl_usd', 0) or 0 for t in winning_trades)
            total_losses = abs(sum(t.get('pnl_usd', 0) or 0 for t in losing_trades))
            profit_factor = total_wins / total_losses if total_losses > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Trades", total_trades)
            
            with col2:
                st.metric("Win Rate", f"{win_rate:.1f}%")
            
            with col3:
                st.metric("Total P&L", f"${total_pnl:,.2f}")
            
            with col4:
                st.metric("Profit Factor", f"{profit_factor:.2f}")
            
            st.markdown("---")
            
            # Trade breakdown by outcome
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🟢 Winning Trades")
                st.metric("Count", len(winning_trades))
                if winning_trades:
                    avg_win = total_wins / len(winning_trades)
                    st.metric("Avg Win", f"${avg_win:,.2f}")
                    st.metric("Total Profit", f"${total_wins:,.2f}")
            
            with col2:
                st.markdown("#### 🔴 Losing Trades")
                st.metric("Count", len(losing_trades))
                if losing_trades:
                    avg_loss = total_losses / len(losing_trades)
                    st.metric("Avg Loss", f"-${avg_loss:,.2f}")
                    st.metric("Total Loss", f"-${total_losses:,.2f}")
            
            st.markdown("---")
            
            # Export option
            if st.button("📥 Export to CSV", key="export_trades"):
                csv_filename = f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                try:
                    st.session_state.trade_db.export_to_csv(
                        csv_filename,
                        start_date=start_datetime,
                        end_date=end_datetime
                    )
                    st.success(f"✅ Exported to {csv_filename}")
                except Exception as e:
                    st.error(f"❌ Export failed: {e}")
        
        else:
            st.info("📭 No closed trades in selected date range")
    else:
        st.error("Trade database not initialized")

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
            get_api_symbol(), count=100
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
                st.plotly_chart(fig, width='stretch')
                
                # Chart controls
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🔄 Refresh Chart", width='stretch'):
                        st.rerun()
                
                with col2:
                    if st.button("📥 Download Chart", width='stretch'):
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
                        width='stretch',
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
                        width='stretch',
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
                    st.dataframe(exit_df, width='stretch', hide_index=True)
                
                with col2:
                    st.markdown("#### P&L by Exit Reason")
                    pnl_df = pd.DataFrame({
                        'Reason': list(summary['distribution']['pnl_by_exit_reason'].keys()),
                        'P&L': [f"${v:.2f}" for v in summary['distribution']['pnl_by_exit_reason'].values()]
                    })
                    st.dataframe(pnl_df, width='stretch', hide_index=True)
                
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
                        width='stretch',
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
                    st.plotly_chart(fig, width='stretch')
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

# ----------------------------------------------------------------------------
# TAB 8: OANDA INTEGRATION
# ----------------------------------------------------------------------------
with tab8:
    st.subheader("🌐 OANDA Practice Integration")
    
    if st.session_state.order_manager and st.session_state.order_manager.oanda_handler:
        handler = st.session_state.order_manager.oanda_handler
        
        # Connection status
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ✅ Connection Status")
            st.success("✅ Connected to OANDA Practice API")
            
            if st.button("🔄 Refresh Connection", key="refresh_oanda"):
                if handler.test_connection():
                    st.success("✅ Connection verified")
                else:
                    st.error("❌ Connection lost")
        
        with col2:
            st.markdown("### 📊 Account Summary")
            try:
                summary = handler.get_account_summary()
                
                if summary:
                    st.metric("💰 Balance", f"${summary.get('balance', 0):,.2f}")
                    st.metric("📈 Unrealized P&L", f"${summary.get('unrealized_pl', 0):,.2f}")
                    st.metric("💵 NAV", f"${summary.get('nav', 0):,.2f}")
                    st.metric("📋 Open Positions", summary.get('open_position_count', 0))
            except:
                st.warning("Could not retrieve account summary")
        
        st.markdown("---")
        
        # Open positions from OANDA
        st.markdown("### 📋 OANDA Open Positions")
        
        try:
            positions = handler.get_open_positions()
            if positions:
                pos_data = []
                for pos in positions:
                    pos_data.append({
                        'Instrument': pos['instrument'],
                        'Long Units': pos.get('long_units', 0),
                        'Short Units': pos.get('short_units', 0),
                        'Unrealized P&L': f"${pos.get('unrealized_pl', 0):.2f}"
                    })
                
                st.dataframe(pd.DataFrame(pos_data), width='stretch', hide_index=True)
            else:
                st.info("No open positions on OANDA")
        except Exception as e:
            st.warning(f"Could not retrieve positions: {e}")
        
        st.markdown("---")
        
        # Live prices
        st.markdown("### 💱 Live Forex Prices")
        
        col1, col2, col3 = st.columns(3)
        
        forex_pairs = ['EUR_USD', 'GBP_USD', 'USD_JPY']
        for idx, pair in enumerate(forex_pairs):
            with [col1, col2, col3][idx]:
                try:
                    price = handler.get_current_price(pair)
                    if price:
                        st.metric(
                            pair.replace('_', '/'),
                            f"{price.get('bid', 0):.5f}",
                            delta=f"Spread: {price.get('spread', 0):.5f}"
                        )
                except:
                    st.warning(f"Could not get {pair} price")
    
    else:
        st.warning("⚠️ OANDA Practice API not connected")

# ============================================================================
# TAB 9: MULTI-SYMBOL TRADING - REUSES LEFT SIDEBAR PATTERN
# ============================================================================

with tab9:
    st.subheader("🔀 Multi-Symbol Trading (Parallel Execution)")
    
    if not st.session_state.multi_symbol_manager:
        st.error("Multi-symbol manager not initialized")
    else:
        msm = st.session_state.multi_symbol_manager
        
        # ============================================================
        # SECTION 1: ADD NEW SYMBOL (REUSE LEFT SIDEBAR PATTERN)
        # ============================================================
        st.markdown("### ➕ Add Trading Symbol")
        st.markdown("*Same selector pattern as left sidebar - add multiple symbols for parallel execution*")
        
        # Create columns matching left sidebar layout
        col_market, col_category, col_symbol, col_strategy, col_mode, col_action = st.columns([1.5, 1.5, 2, 2, 1.5, 1])
        
        # ✅ 1. MARKET TYPE SELECTOR (exactly like left sidebar)
        with col_market:
            st.markdown("**📊 Market Type**")
            market_options = [
                'Cryptocurrency (Binance)',
                'Cryptocurrency (Alpaca)',
                'Forex (OANDA)'
            ]
            add_market_type = st.selectbox(
                "Market",
                market_options,
                index=0,
                key="multi_market_type_select",
                label_visibility="collapsed"
            )
        
        # ✅ 2. CATEGORY SELECTOR (exactly like left sidebar)
        with col_category:
            st.markdown("**🏷️ Category**")
            
            if add_market_type == 'Cryptocurrency (Alpaca)':
                categories = get_alpaca_categories()
                category = st.selectbox(
                    "Category",
                    categories,
                    index=0,
                    key="multi_alpaca_category_select",
                    label_visibility="collapsed"
                )
            elif 'Forex' in add_market_type:
                categories = get_forex_categories()
                category = st.selectbox(
                    "Category",
                    categories,
                    index=0,
                    key="multi_forex_category_select",
                    label_visibility="collapsed"
                )
            else:
                category = None
                st.empty()
        
        # ✅ 3. SYMBOL SELECTOR (exactly like left sidebar)
        with col_symbol:
            st.markdown("**💰 Symbol**")
            
            # Get symbols using same function as left sidebar
            symbols = get_symbol_list(add_market_type, category=category)
            
            # Remove already-active symbols from list
            available_symbols = [s for s in symbols if s not in msm.active_symbols]
            
            if available_symbols:
                add_symbol = st.selectbox(
                    "Symbol",
                    available_symbols,
                    index=0,
                    key="multi_symbol_select",
                    label_visibility="collapsed"
                )
            else:
                st.warning("All symbols in this category already active")
                add_symbol = None
        
        # ✅ 4. STRATEGY SELECTOR (exactly like left sidebar)
        with col_strategy:
            st.markdown("**🎯 Strategy**")
            
            strategies = get_strategy_list()
            add_strategy = st.selectbox(
                "Strategy",
                strategies,
                index=0,
                key="multi_strategy_select",
                label_visibility="collapsed"
            )
        
        # ✅ 5. TRADING MODE SELECTOR (exactly like left sidebar)
        with col_mode:
            st.markdown("**⏱️ Mode**")
            
            modes = ['Scalping (1H→1min)', 'Intraday (1D→1H)']
            add_mode = st.selectbox(
                "Mode",
                modes,
                index=0,
                key="multi_mode_select",
                label_visibility="collapsed"
            )
        
        # ✅ 6. ADD BUTTON (consistent with left sidebar)
        with col_action:
            st.markdown("**🎮 Action**")
            add_symbol_button = st.button(
                "➕ Add",
                type="primary",
                key="multi_add_button",
                use_container_width=True
            )
        
        # ============================================================
        # ADD SYMBOL LOGIC (same as left sidebar START button)
        # ============================================================
        if add_symbol_button and add_symbol:
            try:
                # ✅ Initialize parallel processor if needed
                if st.session_state.parallel_processor is None:
                    st.session_state.parallel_processor = ParallelSymbolProcessor(max_workers=5)
                    logger.info(f"🔄 Parallel processor: 5 workers ready")
                
                # ✅ Extract timeframe from mode (same as left sidebar)
                htf, ltf = get_timeframes(add_mode)
                
                # ✅ Show preview before adding
                with st.expander("📋 Preview before adding", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Symbol", add_symbol)
                    with col2:
                        st.metric("Strategy", add_strategy)
                    with col3:
                        st.metric("Mode", add_mode.split('(')[0].strip())
                    with col4:
                        st.metric("Timeframe", ltf)
                
                # ✅ Add symbol using same logic as left sidebar
                result = msm.add_symbol(
                    symbol=add_symbol,
                    strategy_name=map_strategy_name_to_file(add_strategy),
                    timeframe=ltf
                )
                
                if result['success']:
                    st.success(f"✅ **{add_symbol} added to portfolio!**")
                    st.info(f"""\
                    Now running in parallel with other symbols:
                    - Data feed: Active (will fetch when trading starts)
                    - Strategy: {add_strategy}
                    - Mode: {add_mode}
                    
                    📊 Click **START** in sidebar to begin trading ALL active symbols simultaneously!
                    """)
                    st.session_state.active_symbols = list(msm.active_symbols)
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error(f"❌ Failed: {result.get('reason', 'Unknown error')}")
            
            except Exception as e:
                st.error(f"❌ Error: {e}")
                logger.error(f"Multi-symbol add error: {e}")
        
        st.markdown("---")
        
        # ============================================================
        # SECTION 2: ACTIVE SYMBOLS (PORTFOLIO VIEW)
        # ============================================================
        st.markdown("### 📊 Active Symbols (Running in Parallel)")
        
        if not msm.active_symbols:
            st.info("📭 No active symbols. Add symbols using the selector above.")
            st.markdown("*When you click **START** in the sidebar, all active symbols will run simultaneously!*")
        else:
            # ✅ Show summary metrics (like dashboard)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🟢 Active Symbols", len(msm.active_symbols), delta=f"Max: {msm.max_symbols}")
            
            with col2:
                portfolio = msm.get_portfolio_summary()
                st.metric("📋 Total Positions", portfolio['total_open_positions'])
            
            with col3:
                total_pnl = sum(
                    perf['pnl'] 
                    for perf in portfolio['symbols_performance'].values()
                )
                st.metric("💰 Portfolio P&L", f"${total_pnl:.2f}")
            
            with col4:
                executing = len([s for s in msm.active_symbols if msm.get_symbol_status(s) == 'active'])
                st.metric("⚡ Executing", f"{executing} / {len(msm.active_symbols)}")
            
            st.markdown("---")
            
            # ✅ ACTIVE SYMBOLS TABLE (clean, like positions table)
            st.markdown("**Portfolio Symbols**")
            
            symbols_data = msm.get_active_symbols_list()
            
            table_data = []
            for sym_info in symbols_data:
                symbol = sym_info['symbol']
                pnl = sym_info['pnl']
                
                # Market type indicator
                if "/" in symbol and "USD" in symbol:
                    market = "🟠 Crypto"
                else:
                    market = "🟢 Forex"
                
                # P&L color
                pnl_emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
                
                table_data.append({
                    'Market': market,
                    'Symbol': symbol,
                    'Strategy': sym_info['strategy'],
                    'Positions': sym_info['open_positions'],
                    'Signals': f"{sym_info['signals_executed']}/{sym_info['signals_generated']}",
                    'P&L': f"{pnl_emoji} ${pnl:.2f}",
                    'Win %': f"{sym_info['win_rate']:.1f}%",
                    'Status': "🟢 Active" if st.session_state.trading_active else "⚪ Ready"
                })
            
            st.dataframe(
                pd.DataFrame(table_data),
                use_container_width=True,
                hide_index=True
            )
            
            st.markdown("---")
            
            # ============================================================
            # SECTION 3: PARALLEL EXECUTION INFO
            # ============================================================
            st.markdown("### ⚡ Parallel Execution Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**How Multi-Symbol Works:**")
                st.markdown(f"""\
                ✅ **{len(msm.active_symbols)} symbols running simultaneously:**
                
                Each symbol:
                - Fetches data independently
                - Runs strategy in parallel thread
                - Generates signals concurrently
                - Executes trades to same account
                
                **Thread Pool:** 5 workers (optimized for I/O)
                **Refresh Rate:** Real-time (no delays)
                **Risk Management:** Portfolio-wide limits
                """)
            
            with col2:
                st.markdown("**Control Options:**")
                
                col_remove, col_stop = st.columns(2)
                
                with col_remove:
                    symbol_to_remove = st.selectbox(
                        "Remove symbol",
                        options=list(msm.active_symbols),
                        key="multi_remove_select",
                        label_visibility="collapsed"
                    )
                    
                    if st.button("🗑️ Remove", key="multi_remove_btn", use_container_width=True):
                        if msm.remove_symbol(symbol_to_remove, close_positions=True):
                            st.success(f"✅ {symbol_to_remove} removed")
                            st.session_state.active_symbols = list(msm.active_symbols)
                            time.sleep(1)
                            st.rerun()
                
                with col_stop:
                    if st.button("⛔ Stop All", key="multi_stop_all_btn", use_container_width=True, type="secondary"):
                        msm.stop_all(close_positions=False)
                        st.success("✅ All symbols stopped (positions remain open)")
                        st.session_state.active_symbols = []
                        time.sleep(1)
                        st.rerun()
            
            st.markdown("---")
            
            # ============================================================
            # SECTION 4: PER-SYMBOL DETAILS (expandable)
            # ============================================================
            st.markdown("### 🔍 Per-Symbol Performance")
            
            for symbol in msm.active_symbols:
                perf = msm.get_symbol_performance(symbol)
                
                with st.expander(f"{symbol} - {perf['win_rate']:.1f}% Win Rate", expanded=False):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Open Trades", perf['trades_open'])
                    
                    with col2:
                        st.metric("Closed Trades", perf['trades_closed'])
                    
                    with col3:
                        st.metric("Win Rate", f"{perf['win_rate']:.1f}%")
                    
                    with col4:
                        st.metric("P&L", f"${perf['pnl']:.2f}")
                    
                    # Signal rate progress
                    if perf['signals_generated'] > 0:
                        exec_rate = (perf['signals_executed'] / perf['signals_generated']) * 100
                        st.progress(exec_rate / 100, text=f"Signal Exec: {exec_rate:.1f}% ({perf['signals_executed']}/{perf['signals_generated']})")
                    
                    # Recent activity
                    col_signal, col_trade = st.columns(2)
                    with col_signal:
                        if perf['last_signal_time']:
                            st.caption(f"📡 Last Signal:\n{perf['last_signal_time'].strftime('%H:%M:%S')}")
                        else:
                            st.caption("📡 Last Signal:\n-")
                    
                    with col_trade:
                        if perf['last_trade_time']:
                            st.caption(f"📊 Last Trade:\n{perf['last_trade_time'].strftime('%H:%M:%S')}")
                        else:
                            st.caption("📊 Last Trade:\n-")
            
            st.markdown("---")
            
            # ============================================================
            # SECTION 5: TRADING STATUS
            # ============================================================
            st.markdown("### 🎮 Start/Stop Trading")
            
            if st.session_state.trading_active:
                st.success(f"""
                ✅ **TRADING ACTIVE**
                
                🟢 {len(msm.active_symbols)} symbols running in parallel
                ⚡ Data feeds active for all symbols
                🎯 Strategies executing concurrently
                """)
                
                if st.button("⏹️ STOP ALL TRADING", type="secondary", key="multi_stop_trading", use_container_width=True):
                    stop_paper_trading()
                    st.session_state.trading_active = False
                    st.success("✅ All trading stopped")
                    time.sleep(1)
                    st.rerun()
            
            else:
                st.info(f"""
                ⚪ **TRADING STOPPED**
                
                🔵 {len(msm.active_symbols)} symbols ready
                ⏳ Waiting to start
                """)
                
                if st.button("▶️ START ALL TRADING", type="primary", key="multi_start_trading", use_container_width=True):
                    with st.spinner("Starting parallel trading for all symbols..."):
                        if start_paper_trading():
                            st.success("✅ All symbols trading in parallel!")
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error("❌ Failed to start")

# ============================================================================
# AUTO-REFRESH (if trading active)
# ============================================================================

if st.session_state.trading_active:
    # ✅ PROCESS PENDING SIGNALS FIRST (from background threads)
    try:
        process_pending_signals()  # ← THIS EXECUTES TRADES!
    except Exception as e:
        logger.error(f"Error processing pending signals: {e}")
    
    # Then auto-refresh every 5 seconds
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

