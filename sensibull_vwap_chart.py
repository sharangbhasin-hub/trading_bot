"""
Sensibull VWAP Chart Simulator
================================
Simulates Sensibull Multi Straddle-Strangle Chart functionality.
Since Sensibull doesn't have public API, we replicate the VWAP chart logic locally.

Chart Components (from Final Notes):
- Yellow Line: VWAP of combined premium
- Blue/Purple Line: Real-time combined premium
- Timeframe: 1-minute candles

Fallback: Calculate VWAP from Kite streaming data.

Author: Trading System
Date: October 25, 2025
"""

import pandas as pd
from typing import Dict, Optional, Callable
from datetime import datetime
import logging
from vwap_calculator import VWAPCalculator
from streaming import get_streaming_handler
import threading
import time

logger = logging.getLogger(__name__)

class SensibullVWAPChart:
    """
    Replicates Sensibull Multi Straddle-Strangle Chart locally.
    Subscribes to CE and PE options, calculates combined premium and VWAP.
    """
    
    def __init__(self, ce_symbol: str, pe_symbol: str):
        """
        Initialize chart for given CE and PE symbols.
        
        Args:
            ce_symbol: Call option tradingsymbol
            pe_symbol: Put option tradingsymbol
        """
        self.ce_symbol = ce_symbol
        self.pe_symbol = pe_symbol
        
        self.vwap_calculator = VWAPCalculator()
        self.streaming_handler = get_streaming_handler()
        
        # Latest prices
        self.ce_price = None
        self.pe_price = None
        self.combined_premium = None
        self.current_vwap = None
        
        # Chart data
        self.chart_data = []
        
        # State
        self.is_running = False
        self.update_thread = None
        
        # Callbacks
        self.on_vwap_cross_callback = None
    
    def start(self, on_vwap_cross: Optional[Callable] = None):
        """
        Start monitoring VWAP chart.
        Subscribe to options streaming and calculate VWAP real-time.
        
        Args:
            on_vwap_cross: Callback function when premium crosses VWAP
                          Signature: callback(direction, premium, vwap)
        """
        if self.is_running:
            logger.warning("Chart already running")
            return
        
        self.on_vwap_cross_callback = on_vwap_cross
        
        # Subscribe to streaming
        instruments = [self.ce_symbol, self.pe_symbol]
        
        success = self.streaming_handler.subscribe(
            instruments=instruments,
            on_tick=self._on_tick_received
        )
        
        if not success:
            logger.error("Failed to subscribe to streaming")
            return
        
        # Reset VWAP calculator
        self.vwap_calculator.reset()
        
        self.is_running = True
        logger.info(f"VWAP Chart started - CE: {self.ce_symbol}, PE: {self.pe_symbol}")
    
    def stop(self):
        """Stop monitoring"""
        if not self.is_running:
            return
        
        # Unsubscribe
        instruments = [self.ce_symbol, self.pe_symbol]
        self.streaming_handler.unsubscribe(instruments)
        
        self.is_running = False
        logger.info("VWAP Chart stopped")
    
    def _on_tick_received(self, tick_data: Dict):
        """
        Callback when new tick arrives.
        Update prices and calculate VWAP.
        """
        if not self.is_running:
            return
        
        tradingsymbol = tick_data.get('tradingsymbol')
        last_price = tick_data.get('last_price')
        timestamp = tick_data.get('exchange_timestamp', datetime.now())
        
        # Update prices
        if tradingsymbol == self.ce_symbol:
            self.ce_price = last_price
        elif tradingsymbol == self.pe_symbol:
            self.pe_price = last_price
        
        # Calculate combined premium and VWAP only when both prices available
        if self.ce_price is not None and self.pe_price is not None:
            self._update_vwap(timestamp)
    
    def _update_vwap(self, timestamp: datetime):
        """Update VWAP calculation"""
        # Calculate combined premium
        self.combined_premium = self.ce_price + self.pe_price
        
        # Update VWAP
        self.current_vwap = self.vwap_calculator.update(
            timestamp=timestamp,
            ce_price=self.ce_price,
            pe_price=self.pe_price,
            volume=None  # Using tick count as volume
        )
        
        # Store in chart data
        self.chart_data.append({
            'timestamp': timestamp,
            'ce_price': self.ce_price,
            'pe_price': self.pe_price,
            'combined_premium': self.combined_premium,
            'vwap': self.current_vwap
        })
        
        # Check for crossover
        if self.on_vwap_cross_callback and len(self.chart_data) >= 2:
            self._check_crossover()
    
    def _check_crossover(self):
        """Check if premium crossed VWAP and trigger callback"""
        prev_data = self.chart_data[-2]
        curr_data = self.chart_data[-1]
        
        prev_premium = prev_data['combined_premium']
        prev_vwap = prev_data['vwap']
        curr_premium = curr_data['combined_premium']
        curr_vwap = curr_data['vwap']
        
        # Check for crossover below (for SELLING)
        if prev_premium > prev_vwap and curr_premium < curr_vwap:
            logger.info(f"🔽 VWAP Crossover BELOW detected - Premium: {curr_premium}, VWAP: {curr_vwap}")
            if self.on_vwap_cross_callback:
                self.on_vwap_cross_callback('below', curr_premium, curr_vwap)
        
        # Check for crossover above (for BUYING)
        elif prev_premium < prev_vwap and curr_premium > curr_vwap:
            logger.info(f"🔼 VWAP Crossover ABOVE detected - Premium: {curr_premium}, VWAP: {curr_vwap}")
            if self.on_vwap_cross_callback:
                self.on_vwap_cross_callback('above', curr_premium, curr_vwap)
    
    def get_current_state(self) -> Dict:
        """Get current chart state"""
        return {
            'ce_price': self.ce_price,
            'pe_price': self.pe_price,
            'combined_premium': self.combined_premium,
            'vwap': self.current_vwap,
            'is_running': self.is_running,
            'data_points': len(self.chart_data)
        }
    
    def get_chart_dataframe(self) -> pd.DataFrame:
        """Get chart data as DataFrame for plotting"""
        if not self.chart_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.chart_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        return df
    
    def wait_for_crossover(self, direction: str, timeout_seconds: int = 600) -> Optional[Dict]:
        """
        Block and wait for VWAP crossover signal.
        Used for synchronous strategy execution.
        
        Args:
            direction: 'below' or 'above'
            timeout_seconds: Max seconds to wait
        
        Returns:
            dict: {'direction': str, 'premium': float, 'vwap': float} or None
        """
        crossover_event = threading.Event()
        crossover_data = {}
        
        def on_cross(cross_direction, premium, vwap):
            if cross_direction == direction:
                crossover_data['direction'] = cross_direction
                crossover_data['premium'] = premium
                crossover_data['vwap'] = vwap
                crossover_event.set()
        
        # Start monitoring
        self.start(on_vwap_cross=on_cross)
        
        # Wait for event
        triggered = crossover_event.wait(timeout=timeout_seconds)
        
        # Stop monitoring
        self.stop()
        
        if triggered:
            return crossover_data
        else:
            logger.warning(f"VWAP crossover timeout after {timeout_seconds}s")
            return None

# ============================================================================
# STANDALONE FUNCTION
# ============================================================================

def create_vwap_chart(ce_symbol: str, pe_symbol: str) -> SensibullVWAPChart:
    """
    Create VWAP chart instance.
    
    Args:
        ce_symbol: Call option symbol
        pe_symbol: Put option symbol
    
    Returns:
        SensibullVWAPChart instance
    """
    return SensibullVWAPChart(ce_symbol, pe_symbol)
