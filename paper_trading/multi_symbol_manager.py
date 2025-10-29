"""
Multi-Symbol Trading Manager
=============================

Manages paper trading across multiple symbols simultaneously.
Provides proper isolation and tracking for each symbol.

Features:
- Trade multiple crypto/forex pairs simultaneously
- Independent data feeds per symbol
- Symbol-level position tracking
- Portfolio-wide risk management
- Performance breakdown by symbol
- Automatic symbol rotation based on performance

Author: Trading System
Last Updated: October 29, 2025
"""

import logging
from typing import Dict, List, Optional, Set
from datetime import datetime
from collections import defaultdict
import threading

from .live_data_manager import LiveDataManager
from .paper_order_manager import PaperOrderManager
from .config import get_config

logger = logging.getLogger(__name__)


class MultiSymbolManager:
    """
    Manages paper trading across multiple symbols with proper isolation.
    
    Each symbol has:
    - Independent data feed
    - Separate signal tracking
    - Position limits
    - Performance metrics
    """
    
    def __init__(
        self,
        order_manager: PaperOrderManager,
        data_handler,
        strategy_manager
    ):
        """
        Initialize multi-symbol manager.
        
        Args:
            order_manager: Shared order manager for all symbols
            data_handler: Data handler instance
            strategy_manager: Strategy manager instance
        """
        self.order_manager = order_manager
        self.data_handler = data_handler
        self.strategy_manager = strategy_manager
        self.config = get_config()
        
        # Active symbols tracking
        self.active_symbols: Set[str] = set()
        self.symbol_feeds: Dict[str, LiveDataManager] = {}
        self.symbol_strategies: Dict[str, str] = {}  # symbol -> strategy_name
        
        # Per-symbol statistics
        self.symbol_stats = defaultdict(lambda: {
            'signals_generated': 0,
            'signals_executed': 0,
            'trades_open': 0,
            'trades_closed': 0,
            'pnl': 0.0,
            'win_rate': 0.0,
            'last_signal_time': None,
            'last_trade_time': None
        })
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Portfolio limits
        self.max_symbols = self.config['risk_management'].get('max_symbols', 5)
        self.max_positions_per_symbol = self.config['risk_management'].get('max_positions_per_symbol', 2)
        
        logger.info(f"MultiSymbolManager initialized (max {self.max_symbols} symbols)")
    
    # ========================================================================
    # SYMBOL MANAGEMENT
    # ========================================================================
    
    def add_symbol(
        self,
        symbol: str,
        strategy_name: str,
        timeframe: str = '1h',
        callback = None
    ) -> Dict:
        """
        Add symbol to active trading.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT', 'EUR/USD')
            strategy_name: Strategy to use for this symbol
            timeframe: Data timeframe
            callback: Optional callback for new candles
        
        Returns:
            Dict with success status and info
        """
        with self.lock:
            # Check if already active
            if symbol in self.active_symbols:
                return {
                    'success': False,
                    'reason': f'{symbol} already active'
                }
            
            # Check portfolio limits
            if len(self.active_symbols) >= self.max_symbols:
                return {
                    'success': False,
                    'reason': f'Max symbols reached ({self.max_symbols})'
                }
            
            try:
                # Create data feed for symbol
                data_manager = LiveDataManager(self.data_handler)
                
                # Create callback wrapper
                def symbol_callback(sym, candle):
                    self._handle_new_candle(sym, candle, strategy_name)
                    if callback:
                        callback(sym, candle)
                
                # Start feed
                success = data_manager.start_feed(
                    symbol=symbol,
                    timeframe=timeframe,
                    on_new_candle=symbol_callback
                )
                
                if not success:
                    return {
                        'success': False,
                        'reason': 'Failed to start data feed'
                    }
                
                # Add to active tracking
                self.active_symbols.add(symbol)
                self.symbol_feeds[symbol] = data_manager
                self.symbol_strategies[symbol] = strategy_name
                
                logger.info(f"✅ Added {symbol} with {strategy_name} strategy")
                
                return {
                    'success': True,
                    'symbol': symbol,
                    'strategy': strategy_name,
                    'timeframe': timeframe
                }
                
            except Exception as e:
                logger.error(f"Failed to add {symbol}: {e}")
                return {
                    'success': False,
                    'reason': str(e)
                }
    
    def remove_symbol(self, symbol: str, close_positions: bool = True) -> bool:
        """
        Remove symbol from active trading.
        
        Args:
            symbol: Symbol to remove
            close_positions: Whether to close open positions
        
        Returns:
            bool: True if removed successfully
        """
        with self.lock:
            if symbol not in self.active_symbols:
                logger.warning(f"{symbol} not in active symbols")
                return False
            
            try:
                # Stop data feed
                if symbol in self.symbol_feeds:
                    self.symbol_feeds[symbol].stop_feed(symbol)
                    del self.symbol_feeds[symbol]
                
                # Close positions if requested
                if close_positions:
                    self._close_symbol_positions(symbol)
                
                # Remove from tracking
                self.active_symbols.discard(symbol)
                self.symbol_strategies.pop(symbol, None)
                
                logger.info(f"✅ Removed {symbol}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to remove {symbol}: {e}")
                return False
    
    def stop_all(self, close_positions: bool = True):
        """
        Stop all active symbols.
        
        Args:
            close_positions: Whether to close all positions
        """
        symbols = list(self.active_symbols)
        for symbol in symbols:
            self.remove_symbol(symbol, close_positions)
        
        logger.info("✅ All symbols stopped")
    
    # ========================================================================
    # SIGNAL PROCESSING
    # ========================================================================
    
    def _handle_new_candle(self, symbol: str, candle: Dict, strategy_name: str):
        """
        Handle new candle for symbol - generate and process signals.
        
        Args:
            symbol: Trading symbol
            candle: New candle data
            strategy_name: Strategy to use
        """
        try:
            # Get recent data
            data_manager = self.symbol_feeds.get(symbol)
            if not data_manager:
                return
            
            recent_candles = data_manager.get_recent_candles(symbol, count=100)
            if len(recent_candles) < 50:
                return
            
            # Convert to DataFrame
            import pandas as pd
            df = pd.DataFrame(recent_candles)
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
            
            # Load strategy
            strategy_module = self.strategy_manager.get_strategy(strategy_name)
            if not strategy_module:
                logger.error(f"Strategy {strategy_name} not found for {symbol}")
                return
            
            # Run strategy
            result = strategy_module.analyze(
                df_htf=df,
                df_ltf=df,
                symbol=symbol
            )
            
            # Process signal
            if result and result.get('signal') != 'NO_TRADE':
                self._process_signal(symbol, result, candle)
            
            # Check open positions for SL/TP
            current_price = candle.get('close', 0)
            self.order_manager.check_open_positions(symbol, current_price, current_price)
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    def _process_signal(self, symbol: str, result: Dict, candle: Dict):
        """
        Process strategy signal for symbol.
        
        Args:
            symbol: Trading symbol
            result: Strategy result
            candle: Current candle
        """
        try:
            # Update stats
            self.symbol_stats[symbol]['signals_generated'] += 1
            self.symbol_stats[symbol]['last_signal_time'] = datetime.now()
            
            # Check symbol-specific limits
            if not self._check_symbol_limits(symbol):
                logger.debug(f"Symbol limits reached for {symbol}")
                return
            
            # Create signal
            signal_data = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'direction': result['signal'],
                'entry_price': result['entry'],
                'stop_loss': result['sl'],
                'take_profit': result['tp'],
                'strategy_name': self.symbol_strategies[symbol],
                'confidence': result.get('confidence', 0),
                'risk_reward_ratio': result.get('rr_ratio', 0),
                'market_type': self._get_market_type(symbol),
                'reasoning': result.get('reasoning', [])
            }
            
            # Attempt to place order
            current_price = candle.get('close', result['entry'])
            order_result = self.order_manager.place_order(
                signal=signal_data,
                current_price=current_price,
                exchange_price=current_price
            )
            
            if order_result['success']:
                self.symbol_stats[symbol]['signals_executed'] += 1
                self.symbol_stats[symbol]['trades_open'] += 1
                self.symbol_stats[symbol]['last_trade_time'] = datetime.now()
                logger.info(f"✅ Signal executed for {symbol}: {result['signal']}")
            else:
                logger.debug(f"Signal not executed for {symbol}: {order_result.get('reason')}")
                
        except Exception as e:
            logger.error(f"Error processing signal for {symbol}: {e}")
    
    def _close_symbol_positions(self, symbol: str):
        """Close all open positions for symbol."""
        positions = self.order_manager.get_open_positions_list()
        
        for position in positions:
            if position['symbol'] == symbol:
                # Get current price from feed
                current_price = position['entry_price']
                if symbol in self.symbol_feeds:
                    recent = self.symbol_feeds[symbol].get_recent_candles(symbol, 1)
                    if recent:
                        current_price = recent[-1].get('close', current_price)
                
                self.order_manager.close_position(
                    position['trade_id'],
                    current_price,
                    'SYMBOL_REMOVED'
                )
    
    # ========================================================================
    # RISK CHECKS
    # ========================================================================
    
    def _check_symbol_limits(self, symbol: str) -> bool:
        """
        Check if symbol can accept new positions.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            bool: True if limits allow new position
        """
        # Count open positions for this symbol
        positions = self.order_manager.get_open_positions_list()
        symbol_positions = sum(1 for p in positions if p['symbol'] == symbol)
        
        if symbol_positions >= self.max_positions_per_symbol:
            return False
        
        return True
    
    def _get_market_type(self, symbol: str) -> str:
        """Determine market type from symbol."""
        crypto_symbols = self.config['crypto']['supported_pairs']
        forex_symbols = self.config['forex']['supported_pairs']
        
        if symbol in crypto_symbols:
            return 'crypto'
        elif symbol in forex_symbols:
            return 'forex'
        else:
            # Default based on format
            return 'crypto' if '/' in symbol and 'USD' in symbol.split('/')[1] else 'forex'
    
    # ========================================================================
    # STATISTICS & MONITORING
    # ========================================================================
    
    def get_symbol_performance(self, symbol: str) -> Dict:
        """
        Get performance metrics for specific symbol.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Dict with performance metrics
        """
        stats = self.symbol_stats[symbol].copy()
        
        # Get closed trades for this symbol
        if hasattr(self.order_manager, 'db'):
            closed_trades = self.order_manager.db.get_closed_trades(
                start_date=datetime.now().replace(hour=0, minute=0, second=0),
                end_date=datetime.now()
            )
            
            symbol_trades = [t for t in closed_trades if t['symbol'] == symbol]
            
            if symbol_trades:
                winning = sum(1 for t in symbol_trades if t['pnl_usd'] > 0)
                stats['trades_closed'] = len(symbol_trades)
                stats['win_rate'] = (winning / len(symbol_trades)) * 100
                stats['pnl'] = sum(t['pnl_usd'] for t in symbol_trades)
        
        # Count current open positions
        open_positions = [p for p in self.order_manager.get_open_positions_list() if p['symbol'] == symbol]
        stats['trades_open'] = len(open_positions)
        
        return stats
    
    def get_portfolio_summary(self) -> Dict:
        """
        Get summary of entire portfolio.
        
        Returns:
            Dict with portfolio metrics
        """
        summary = {
            'active_symbols': list(self.active_symbols),
            'total_symbols': len(self.active_symbols),
            'total_open_positions': len(self.order_manager.get_open_positions_list()),
            'symbols_performance': {}
        }
        
        # Add per-symbol performance
        for symbol in self.active_symbols:
            summary['symbols_performance'][symbol] = self.get_symbol_performance(symbol)
        
        return summary
    
    def get_active_symbols_list(self) -> List[Dict]:
        """
        Get list of active symbols with details.
        
        Returns:
            List of symbol info dicts
        """
        result = []
        
        for symbol in self.active_symbols:
            perf = self.get_symbol_performance(symbol)
            
            result.append({
                'symbol': symbol,
                'strategy': self.symbol_strategies.get(symbol, 'Unknown'),
                'open_positions': perf['trades_open'],
                'signals_generated': perf['signals_generated'],
                'signals_executed': perf['signals_executed'],
                'pnl': perf['pnl'],
                'win_rate': perf['win_rate'],
                'last_signal': perf['last_signal_time'],
                'last_trade': perf['last_trade_time']
            })
        
        return result


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MULTI-SYMBOL MANAGER TEST")
    print("=" * 70)
    
    # This would require full system setup
    # Use with actual paper trading system
    
    print("\n✅ Module loaded successfully")
    print("Integrate with main_paper_trading.py for full testing")
    print("=" * 70)
