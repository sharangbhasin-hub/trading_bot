"""
Real-time Market Data Streaming using Kite Connect WebSocket
Handles: Live ticks, automatic reconnection, multi-instrument streaming
"""

from kiteconnect import KiteTicker
import threading
import time
from datetime import datetime
from typing import List, Dict, Callable, Optional
from collections import defaultdict
import queue

from config import (
    KITE_API_KEY,
    KITE_ACCESS_TOKEN,
    STREAMING_CONFIG,
    get_market_status
)
from database import insert_tick_data, add_to_watchlist, get_watchlist

# ============================================================================
# STREAMING HANDLER CLASS
# ============================================================================

class StreamingHandler:
    """
    Handles real-time WebSocket streaming from Kite Connect
    Automatically manages subscriptions, reconnections, and data distribution
    """
    
    def __init__(self):
        """Initialize streaming handler"""
        self.ticker = None
        self.is_connected = False
        self.is_running = False
        
        # Subscription management
        self.subscribed_tokens = set()  # Set of instrument tokens
        self.token_to_symbol = {}  # Map token -> symbol
        self.symbol_to_token = {}  # Map symbol -> token
        
        # Callback management
        self.tick_callbacks = defaultdict(list)  # token -> list of callbacks
        self.global_callbacks = []  # Callbacks for all ticks
        
        # Data buffer for batch processing
        self.tick_buffer = queue.Queue(maxsize=1000)
        
        # Connection tracking
        self.reconnect_count = 0
        self.last_tick_time = None
        
        # Thread for processing ticks
        self.processor_thread = None
        self.processing = False
        
    # ========================================================================
    # CONNECTION MANAGEMENT
    # ========================================================================
    
    def start(self) -> bool:
        """
        Start the Kite websocket streaming.
        Return True if started successfully, False otherwise.
        """
        if not KITE_API_KEY or not KITE_ACCESS_TOKEN:
            print("❌ Missing Kite API credentials.")
            return False
    
        if self.is_running:
            print("⚠️ Streaming already running")
            return True
    
        try:
            print("🚀 Starting KiteTicker connection...")
    
            # Initialize KiteTicker instance
            self.ticker = KiteTicker(KITE_API_KEY, KITE_ACCESS_TOKEN)
    
            # Register event handlers
            self.ticker.on_connect = self.on_connect
            self.ticker.on_ticks = self.on_ticks
            self.ticker.on_close = self.on_close
            self.ticker.on_error = self.on_error
            self.ticker.on_reconnect = self.on_reconnect
            self.ticker.on_noreconnect = self.on_noreconnect
    
            # Start the websocket connection in background thread
            self.thread = threading.Thread(target=self.ticker.connect, daemon=True)
            self.thread.start()
    
            # Wait for connection confirmation: block or sleep briefly
            max_wait = 10  # seconds max wait for connection
            waited = 0
            while not self.is_connected and waited < max_wait:
                time.sleep(0.5)
                waited += 0.5
    
            if not self.is_connected:
                print(f"❌ Failed to connect to Kite websocket after {max_wait} seconds.")
                return False
    
            # Now mark streaming as running only after connection established
            self.is_running = True
    
            # Start tick processing thread
            self.processing = True
            self.processor_thread = threading.Thread(target=self.process_ticks, daemon=True)
            self.processor_thread.start()
    
            print("✅ Kite websocket streaming started successfully.")
            return True
    
        except Exception as e:
            print(f"❌ Exception starting streaming: {e}")
            return False
    
    def stop(self):
        """Stop WebSocket connection"""
        if not self.is_running:
            return
        
        try:
            self.is_running = False
            self.processing = False
            
            if self.ticker:
                self.ticker.close()
            
            self.is_connected = False
            print("✅ Streaming stopped")
            
        except Exception as e:
            print(f"❌ Error stopping streaming: {e}")
    
    # ========================================================================
    # WEBSOCKET CALLBACKS
    # ========================================================================
    
    def on_connect(self, ws, response):
        """Called when WebSocket connects"""
        self.is_connected = True
        self.reconnect_count = 0
        print(f"✅ WebSocket connected")
        
        # Resubscribe to instruments if any
        if self.subscribed_tokens:
            self._resubscribe()
    
    def on_ticks(self, ws, ticks):
        """
        Called when tick data is received
        Args:
            ticks: List of tick dictionaries
        """
        self.last_tick_time = datetime.now()
        
        # Add ticks to buffer for processing
        for tick in ticks:
            try:
                # Add symbol information
                token = tick.get('instrument_token')
                if token in self.token_to_symbol:
                    tick['symbol'] = self.token_to_symbol[token]
                
                self.tick_buffer.put_nowait(tick)
            except queue.Full:
                print("⚠️ Tick buffer full, dropping tick")
                break
    
    def on_close(self, ws, code, reason):
        """Called when WebSocket connection closes"""
        self.is_connected = False
        print(f"⚠️ WebSocket closed: {code} - {reason}")
    
    def on_error(self, ws, code, reason):
        """Called when WebSocket error occurs"""
        print(f"❌ WebSocket error: {code} - {reason}")
    
    def on_reconnect(self, ws, attempts_count):
        """Called when attempting to reconnect"""
        self.reconnect_count = attempts_count
        print(f"🔄 Reconnecting... (attempt {attempts_count})")
    
    def on_noreconnect(self, ws):
        """Called when max reconnection attempts reached"""
        self.is_connected = False
        print("❌ Max reconnection attempts reached")
        
        # Try to restart connection after delay
        time.sleep(STREAMING_CONFIG['reconnect_delay'])
        if self.is_running:
            print("🔄 Attempting manual restart...")
            self.start()
    
    # ========================================================================
    # SUBSCRIPTION MANAGEMENT
    # ========================================================================
    
    def subscribe(
        self, 
        instruments: List[Dict],
        mode: str = None
    ) -> bool:
        """
        Subscribe to instruments for streaming
        Args:
            instruments: List of dicts with 'instrument_token' and 'symbol'
            mode: 'ltp', 'quote', or 'full' (default from config)
        Returns:
            True if subscription successful
        """
        if not self.is_connected:
            print("⚠️ Not connected, waiting for connection...")
            # Store for later subscription
            for inst in instruments:
                token = inst['instrument_token']
                symbol = inst['symbol']
                self.subscribed_tokens.add(token)
                self.token_to_symbol[token] = symbol
                self.symbol_to_token[symbol] = token
            return False
        
        try:
            # Extract tokens
            tokens = [inst['instrument_token'] for inst in instruments]
            
            # Check limit
            total_subscribed = len(self.subscribed_tokens) + len(tokens)
            if total_subscribed > STREAMING_CONFIG['max_instruments']:
                print(f"⚠️ Subscription limit reached ({STREAMING_CONFIG['max_instruments']})")
                return False
            
            # Subscribe
            mode = mode or STREAMING_CONFIG['mode']
            
            if mode == 'ltp':
                self.ticker.subscribe(tokens)
                self.ticker.set_mode(self.ticker.MODE_LTP, tokens)
            elif mode == 'quote':
                self.ticker.subscribe(tokens)
                self.ticker.set_mode(self.ticker.MODE_QUOTE, tokens)
            else:  # full
                self.ticker.subscribe(tokens)
                self.ticker.set_mode(self.ticker.MODE_FULL, tokens)
            
            # Store subscription info
            for inst in instruments:
                token = inst['instrument_token']
                symbol = inst['symbol']
                self.subscribed_tokens.add(token)
                self.token_to_symbol[token] = symbol
                self.symbol_to_token[symbol] = token
                
                # Add to database watchlist
                add_to_watchlist(token, symbol)
            
            print(f"✅ Subscribed to {len(tokens)} instruments (mode: {mode})")
            return True
            
        except Exception as e:
            print(f"❌ Subscription failed: {e}")
            return False
    
    def unsubscribe(self, instruments: List[Dict]) -> bool:
        """
        Unsubscribe from instruments
        Args:
            instruments: List of dicts with 'instrument_token'
        """
        if not self.is_connected:
            return False
        
        try:
            tokens = [inst['instrument_token'] for inst in instruments]
            self.ticker.unsubscribe(tokens)
            
            # Remove from tracking
            for token in tokens:
                self.subscribed_tokens.discard(token)
                if token in self.token_to_symbol:
                    symbol = self.token_to_symbol[token]
                    del self.token_to_symbol[token]
                    if symbol in self.symbol_to_token:
                        del self.symbol_to_token[symbol]
            
            print(f"✅ Unsubscribed from {len(tokens)} instruments")
            return True
            
        except Exception as e:
            print(f"❌ Unsubscription failed: {e}")
            return False
    
    def resubscribe(self):
        """Resubscribe to all instruments after reconnection"""
        if not self.subscribed_tokens:
            return
        
        print(f"🔄 Resubscribing to {len(self.subscribed_tokens)} instruments")
        
        instruments = [
            {'instrument_token': token, 'symbol': self.token_to_symbol[token]}
            for token in self.subscribed_tokens
        ]
        
        # Clear existing subscriptions
        temp_tokens = self.subscribed_tokens.copy()
        self.subscribed_tokens.clear()
        
        # Resubscribe
        success = self.subscribe(instruments)
        
        if not success:
            # Restore tokens on failure
            self.subscribed_tokens = temp_tokens
    
    # ========================================================================
    # CALLBACK MANAGEMENT
    # ========================================================================
    
    def add_tick_callback(
        self,
        callback: Callable[[Dict], None],
        instrument_token: Optional[int] = None
    ):
        """
        Add callback for tick updates
        Args:
            callback: Function to call with tick data
            instrument_token: Specific token, or None for all ticks
        """
        if instrument_token:
            self.tick_callbacks[instrument_token].append(callback)
        else:
            self.global_callbacks.append(callback)
    
    def remove_tick_callback(
        self,
        callback: Callable,
        instrument_token: Optional[int] = None
    ):
        """Remove callback"""
        if instrument_token:
            if instrument_token in self.tick_callbacks:
                try:
                    self.tick_callbacks[instrument_token].remove(callback)
                except ValueError:
                    pass
        else:
            try:
                self.global_callbacks.remove(callback)
            except ValueError:
                pass
    
    # ========================================================================
    # TICK PROCESSING
    # ========================================================================
    
    def process_ticks(self):
        """
        Process ticks from buffer
        Runs in separate thread
        """
        while self.processing:
            try:
                # Get tick from buffer (with timeout)
                tick = self.tick_buffer.get(timeout=1)
                
                # Store in database
                insert_tick_data(tick)
                
                # Call callbacks
                token = tick.get('instrument_token')
                
                # Token-specific callbacks
                if token in self.tick_callbacks:
                    for callback in self.tick_callbacks[token]:
                        try:
                            callback(tick)
                        except Exception as e:
                            print(f"❌ Callback error: {e}")
                
                # Global callbacks
                for callback in self.global_callbacks:
                    try:
                        callback(tick)
                    except Exception as e:
                        print(f"❌ Global callback error: {e}")
                
                self.tick_buffer.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Tick processing error: {e}")
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_status(self) -> Dict:
        """Get streaming status"""
        return {
            "connected": self.is_connected,
            "running": self.is_running,
            "subscribed_count": len(self.subscribed_tokens),
            "max_instruments": STREAMING_CONFIG['max_instruments'],
            "reconnect_count": self.reconnect_count,
            "last_tick": self.last_tick_time.strftime("%H:%M:%S") if self.last_tick_time else "Never",
            "buffer_size": self.tick_buffer.qsize()
        }
    
    def get_subscribed_symbols(self) -> List[str]:
        """Get list of subscribed symbols"""
        return list(self.symbol_to_token.keys())
    
    def is_subscribed(self, symbol: str) -> bool:
        """Check if symbol is subscribed"""
        return symbol in self.symbol_to_token
    
    def get_subscription_count(self) -> int:
        """Get count of subscribed instruments"""
        return len(self.subscribed_tokens)


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_streaming_handler_instance = None

def get_streaming_handler() -> StreamingHandler:
    """Get or create singleton StreamingHandler instance"""
    global _streaming_handler_instance
    
    if _streaming_handler_instance is None:
        _streaming_handler_instance = StreamingHandler()
    
    return _streaming_handler_instance


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def start_streaming() -> bool:
    """Start streaming (convenience function)"""
    handler = get_streaming_handler()
    return handler.start()

def stop_streaming():
    """Stop streaming (convenience function)"""
    handler = get_streaming_handler()
    handler.stop()

def subscribe_instruments(instruments: List[Dict], mode: str = None) -> bool:
    """Subscribe to instruments (convenience function)"""
    handler = get_streaming_handler()
    return handler.subscribe(instruments, mode)

def unsubscribe_instruments(instruments: List[Dict]) -> bool:
    """Unsubscribe from instruments (convenience function)"""
    handler = get_streaming_handler()
    return handler.unsubscribe(instruments)

def add_callback(callback: Callable, instrument_token: Optional[int] = None):
    """Add tick callback (convenience function)"""
    handler = get_streaming_handler()
    handler.add_tick_callback(callback, instrument_token)

def get_streaming_status() -> Dict:
    """Get streaming status (convenience function)"""
    handler = get_streaming_handler()
    return handler.get_status()
