"""
Paper Order Manager
===================

Manages simulated paper trading orders with realistic fills and position tracking.
Automatically closes positions when Stop Loss or Take Profit is hit.

Features:
- Realistic order simulation with slippage
- Automatic SL/TP monitoring
- Position size validation
- Risk checks before order placement
- Thread-safe operations
- Comprehensive trade logging

Author: Trading System
Last Updated: October 29, 2025
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import threading
import logging

from .config import get_config
from .trade_database import TradeDatabase
from .pnl_calculator import PnLCalculator
from paper_trading.oanda_practice_handler import OandaPracticeHandler

logger = logging.getLogger(__name__)


class PaperOrderManager:
    """
    Production-grade paper order manager.
    
    Manages the complete lifecycle of paper trades:
    1. Order validation and risk checks
    2. Simulated order fills with slippage
    3. Position tracking
    4. Automatic SL/TP monitoring
    5. Trade closure with P&L calculation
    6. Database persistence
    """
    
    def __init__(
        self, 
        trade_database: TradeDatabase,
        pnl_calculator: PnLCalculator,
        initial_balance: float = None
    ):
        """
        Initialize paper order manager.
        
        Args:
            trade_database: TradeDatabase instance for persistence
            pnl_calculator: PnLCalculator instance for P&L calculations
            initial_balance: Starting account balance (USD)
        """
        self.db = trade_database
        self.pnl_calc = pnl_calculator
        self.config = get_config()
        
        # Account management
        if initial_balance is None:
            initial_balance = self.config['initial_balance']
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.equity = initial_balance  # Balance + unrealized P&L
        
        # Position tracking (thread-safe)
        self.open_positions: Dict[int, Dict] = {}  # trade_id -> position dict
        self.position_lock = threading.Lock()
        
        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.daily_pnl = 0.0
        self.daily_trade_count = 0
        self.max_drawdown = 0.0
        self.peak_balance = initial_balance
        
        # Risk management
        self.risk_config = self.config['risk_management']
        self.crypto_config = self.config['crypto']
        self.forex_config = self.config['forex']

        # OANDA Practice API handler (optional for Forex)
        self.oanda_handler = None
        if self.forex_config.get('oanda_enabled', False):
            try:
                self.oanda_handler = OandaPracticeHandler()
                if self.oanda_handler.test_connection():
                    logger.info("✅ OANDA Practice API connected and ready")
                else:
                    logger.warning("⚠️ OANDA connection failed - using local simulation")
                    self.oanda_handler = None
            except Exception as e:
                logger.warning(f"⚠️ OANDA handler initialization failed: {e}")
                logger.info("Using local paper trading simulation for Forex")
                self.oanda_handler = None
        
        logger.info(f"PaperOrderManager initialized: Balance=${self.current_balance:,.2f}")

        # ✅ ALPACA CRYPTO HANDLER (NEW)
        self.alpaca_handler = None
        if self.crypto_config.get('alpaca_enabled', False):
            try:
                from paper_trading.alpaca_crypto_handler import AlpacaCryptoHandler
                self.alpaca_handler = AlpacaCryptoHandler()
                if self.alpaca_handler.test_connection():
                    logger.info("✅ Alpaca Crypto API connected and ready")
                else:
                    logger.warning("⚠️ Alpaca connection failed - using local simulation")
                    self.alpaca_handler = None
            except Exception as e:
                logger.warning(f"⚠️ Alpaca handler initialization failed: {e}")
                logger.info("Using local paper trading simulation for Crypto")
                self.alpaca_handler = None
        
        logger.info(f"PaperOrderManager initialized: Balance=${self.current_balance:,.2f}")
    
    # ========================================================================
    # ORDER PLACEMENT
    # ========================================================================
    
    def place_order(
        self,
        signal: Dict[str, Any],
        current_price: float,
        exchange_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Place a paper trade order from strategy signal.
        
        Args:
            signal: Signal dictionary from strategy with keys:
                    - symbol: str
                    - direction: 'BUY' or 'SELL'
                    - entry_price: float
                    - stop_loss: float
                    - take_profit: float
                    - strategy_name: str
                    - confidence: int (0-100)
                    - market_type: 'crypto' or 'forex'
            current_price: Current market price
            exchange_price: Actual exchange price for verification (optional)
        
        Returns:
            Dict with order result:
                - success: bool
                - trade_id: int (if successful)
                - fill_price: float (if successful)
                - reason: str (if unsuccessful)
                - slippage: float
        
        Example:
            >>> signal = {
            ...     'symbol': 'BTC/USDT',
            ...     'direction': 'BUY',
            ...     'entry_price': 67250.00,
            ...     'stop_loss': 67100.00,
            ...     'take_profit': 67550.00,
            ...     'strategy_name': 'CRT-TBS',
            ...     'confidence': 85,
            ...     'market_type': 'crypto'
            ... }
            >>> result = order_mgr.place_order(signal, 67249.80)
        """
        # Validate signal
        required_fields = ['symbol', 'direction', 'entry_price', 'stop_loss', 
                          'take_profit', 'strategy_name', 'market_type']
        for field in required_fields:
            if field not in signal:
                return {'success': False, 'reason': f'Missing required field: {field}'}
        
        # Risk checks
        risk_check = self._check_risk_limits(signal)
        if not risk_check['allowed']:
            return {'success': False, 'reason': risk_check['reason']}
        
        # Calculate fill price with slippage
        fill_price = self._simulate_fill(signal['entry_price'], signal['direction'], signal['market_type'])
        slippage = abs(fill_price - signal['entry_price'])
        
        # Calculate position size
        position_size = self._calculate_position_size(signal, fill_price)

        # ✅ ADD THIS ENTIRE SECTION HERE (OANDA Execution):
        # ------------------------------------------------------------------
        # OANDA PRACTICE API EXECUTION (for Forex only)
        # ------------------------------------------------------------------
        oanda_trade_id = None
        actual_fill_price = fill_price  # Default to simulated price
        
        if signal['market_type'] == 'forex' and self.oanda_handler:
            try:
                # Convert symbol format
                instrument = self.oanda_handler.convert_symbol_to_oanda(signal['symbol'])
                
                # Calculate units from lot size
                units = self.oanda_handler.calculate_units(position_size['lot_size'])
                
                # Negative units for SELL orders
                if signal['direction'] == 'SELL':
                    units = -units
                
                # ============================================================
                # PRODUCTION FIX: Smart TP Selection with Validation
                # ============================================================
                # Professional traders use TP2 for OANDA because:
                # 1. TP1 (partial exit) is too close → gets rejected
                # 2. TP2 (final target) provides proper risk-adjusted return
                # 3. Accounts for spread + slippage in profit calculation
                
                # Select appropriate TP for OANDA (prefer TP2 for full exits)
                oanda_tp = signal.get('take_profit_2') or signal.get('take_profit_1') or signal.get('take_profit')
                
                # Validate TP distance (OANDA requires minimum profit after costs)
                min_pip_distance = 15  # Minimum 15 pips for profitable trade after spread/slippage
                entry_price = signal['entry_price']
                pip_distance = abs(oanda_tp - entry_price) * 10000  # Convert to pips
                
                if pip_distance < min_pip_distance:
                    # TP too close - skip OANDA, use local simulation
                    logger.warning(
                        f"⚠️ TP too close ({pip_distance:.1f} pips < {min_pip_distance} minimum). "
                        f"Using local simulation instead of OANDA."
                    )
                    logger.info("📊 Executing trade locally (safer for tight TPs)")
                    # Skip OANDA execution - continue to local simulation below
                    
                else:
                    # TP distance is acceptable - proceed with OANDA
                    logger.info(
                        f"🌐 Placing OANDA Practice order: {instrument} {units} units"
                    )
                    logger.info(
                        f"   Entry: {entry_price:.5f} | SL: {signal['stop_loss']:.5f} | "
                        f"TP: {oanda_tp:.5f} ({pip_distance:.1f} pips)"
                    )
                    
                    oanda_result = self.oanda_handler.place_market_order(
                        instrument=instrument,
                        units=units,
                        stop_loss=signal['stop_loss'],
                        take_profit=oanda_tp  # ✅ Use validated TP (TP2 preferred)
                    )
                    
                    if oanda_result and oanda_result.get('success'):
                        oanda_trade_id = oanda_result['trade_id']
                        actual_fill_price = oanda_result['fill_price']
                        
                        logger.info(
                            f"✅ OANDA order filled! "
                            f"Trade ID: {oanda_trade_id}, "
                            f"Fill: {actual_fill_price:.5f}"
                        )
                        
                        # Use actual OANDA fill price instead of simulated
                        fill_price = actual_fill_price
                    else:
                        logger.warning("⚠️ OANDA order failed - using local simulation")
            
            except Exception as e:
                logger.error(f"❌ OANDA execution error: {e}")
                logger.info("Falling back to local simulation")
        # ------------------------------------------------------------------        

        # ✅ ALPACA CRYPTO EXECUTION (NEW - similar to OANDA pattern)
        # ------------------------------------------------------------------
        alpaca_order_id = None
        actual_fill_price = fill_price  # Default to simulated price
        
        if signal['market_type'] == 'crypto' and self.alpaca_handler:
            try:
                # Determine order side
                alpaca_side = 'buy' if signal['direction'] == 'BUY' else 'sell'
                
                # Place order on Alpaca with bracket (SL/TP)
                logger.info(f"🌐 Placing Alpaca order: {signal['symbol']} {position_size['quantity']:.8f} {alpaca_side}")
                
                alpaca_result = self.alpaca_handler.place_market_order_with_bracket(
                    symbol=signal['symbol'],
                    qty=position_size['quantity'],
                    side=alpaca_side,
                    stop_loss=signal['stop_loss'],
                    take_profit=signal['take_profit']
                )
                
                if alpaca_result and alpaca_result.get('success'):
                    alpaca_order_id = alpaca_result['order_id']
                    
                    # Optionally update fill price from Alpaca
                    if 'filled_avg_price' in alpaca_result and alpaca_result['filled_avg_price']:
                        actual_fill_price = alpaca_result['filled_avg_price']
                    
                    logger.info(
                        f"✅ Alpaca order submitted! "
                        f"Order ID: {alpaca_order_id}, "
                        f"Status: {alpaca_result['status']}"
                    )
                    
                    fill_price = actual_fill_price
                else:
                    logger.warning("⚠️ Alpaca order failed - using local simulation")
            
            except Exception as e:
                logger.error(f"❌ Alpaca execution error: {e}")
                logger.info("Falling back to local simulation")
        # ------------------------------------------------------------------
        
        # Create trade record
        trade_data = {
            'timestamp': datetime.now(),
            'symbol': signal['symbol'],
            'exchange': signal.get('exchange', 'simulated'),
            'market_type': signal['market_type'],
            'direction': signal['direction'],
            'entry_price': fill_price,
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit'],
            'strategy_name': signal['strategy_name'],
            'confidence': signal.get('confidence', 0),
            'risk_reward_ratio': self._calculate_rr_ratio(fill_price, signal['stop_loss'], signal['take_profit'], signal['direction']),
            'status': 'OPEN',
            'exchange_price_at_entry': exchange_price or current_price,
            'slippage': slippage
        }
        
        # Add market-specific fields
        if signal['market_type'] == 'crypto':
            trade_data['quantity'] = position_size['quantity']
            trade_data['alpaca_order_id'] = alpaca_order_id
        else:  # forex
            trade_data['lot_size'] = position_size['lot_size']
        
        # Insert into database
        try:
            trade_id = self.db.insert_trade(trade_data)
            
            # Add to open positions
            with self.position_lock:
                self.open_positions[trade_id] = {
                    **trade_data,
                    'trade_id': trade_id,
                    'unrealized_pnl': 0.0,
                    'oanda_trade_id': oanda_trade_id
                }
            
            # Update statistics
            self.daily_trade_count += 1
            
            logger.info(f"✅ Order placed: #{trade_id} {signal['symbol']} {signal['direction']} @ ${fill_price:.2f}")
            
            try:
                return {
                    'success': True,
                    'trade_id': trade_id,
                    'symbol': signal['symbol'],
                    'direction': signal['direction'],
                    'quantity': position_size.get('quantity', position_size.get('lot_size', 0)),
                    'amount': (position_size.get('quantity', 0) * fill_price) if signal['market_type'] == 'crypto' else (position_size.get('lot_size', 0.1) * 100000),
                    'entry_price': fill_price,
                    'stop_loss': signal['stop_loss'],
                    'take_profit': signal['take_profit'],
                    'side': signal['direction'],
                    'timestamp': datetime.now(),
                    'fill_price': fill_price,
                    'slippage': slippage,
                    'position_size': position_size,
                    'status': 'OPEN'
                }
            
            except Exception as e:
                logger.error(f"Error formatting return: {e}")
                return {
                    'success': True,
                    'trade_id': trade_id,
                    'entry_price': fill_price,
                    'slippage': slippage
                }

        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return {'success': False, 'reason': str(e)}
    
    # ========================================================================
    # POSITION MONITORING (Auto SL/TP)
    # ========================================================================
    
    def check_open_positions(self, symbol: str, current_price: float, exchange_price: Optional[float] = None):
        """
        Check all open positions for SL/TP triggers.
        Should be called every time new price data is received.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            exchange_price: Actual exchange price for verification (optional)
        """
        with self.position_lock:
            positions_to_close = []
            
            for trade_id, position in self.open_positions.items():
                if position['symbol'] != symbol:
                    continue
                
                # Check Stop Loss
                if self._check_stop_loss(position, current_price):
                    positions_to_close.append((trade_id, current_price, 'SL', exchange_price))
                
                # Check Take Profit
                elif self._check_take_profit(position, current_price):
                    positions_to_close.append((trade_id, current_price, 'TP', exchange_price))
                
                # Update unrealized P&L
                else:
                    position['unrealized_pnl'] = self._calculate_unrealized_pnl(position, current_price)
        
        # Close positions (outside lock to avoid deadlock)
        for trade_id, price, reason, exchange_price in positions_to_close:
            self.close_position(trade_id, price, reason, exchange_price)
    
    def _check_stop_loss(self, position: Dict, current_price: float) -> bool:
        """Check if stop loss is hit."""
        if position['direction'] == 'BUY':
            return current_price <= position['stop_loss']
        else:  # SELL
            return current_price >= position['stop_loss']
    
    def _check_take_profit(self, position: Dict, current_price: float) -> bool:
        """Check if take profit is hit."""
        if position['direction'] == 'BUY':
            return current_price >= position['take_profit']
        else:  # SELL
            return current_price <= position['take_profit']
    
    def _calculate_unrealized_pnl(self, position: Dict, current_price: float) -> float:
        """Calculate unrealized P&L for open position."""
        try:
            if position['market_type'] == 'crypto':
                pnl_result = self.pnl_calc.calculate_crypto_pnl(
                    entry_price=position['entry_price'],
                    exit_price=current_price,
                    investment_usd=position['quantity'] * position['entry_price'],
                    direction=position['direction'],
                    include_costs=False  # Don't include costs for unrealized
                )
            else:  # forex
                pnl_result = self.pnl_calc.calculate_forex_pnl(
                    entry_price=position['entry_price'],
                    exit_price=current_price,
                    lot_size=position['lot_size'],
                    pair=position['symbol'],
                    direction=position['direction'],
                    include_costs=False
                )
            
            return pnl_result['gross_pnl_usd']
            
        except Exception as e:
            logger.error(f"Error calculating unrealized P&L: {e}")
            return 0.0
    
    # ========================================================================
    # POSITION CLOSURE
    # ========================================================================
    
    def close_position(
        self,
        trade_id: int,
        exit_price: float,
        reason: str = 'MANUAL',
        exchange_price: Optional[float] = None
    ) -> bool:
        """
        Close an open position.
        
        Args:
            trade_id: Trade ID to close
            exit_price: Exit price
            reason: Closure reason ('SL', 'TP', 'MANUAL', 'EOD')
            exchange_price: Actual exchange price for verification
        
        Returns:
            bool: True if closed successfully
        """
        with self.position_lock:
            if trade_id not in self.open_positions:
                logger.warning(f"Trade #{trade_id} not found in open positions")
                return False
            
            position = self.open_positions[trade_id]

        # ------------------------------------------------------------------
        # ✅ CLOSE ON OANDA/ALPACA (EXISTING + NEW)
        # ------------------------------------------------------------------
        actual_exit_price = exit_price
        
        # Close on OANDA (Forex)
        if position.get('oanda_trade_id') and self.oanda_handler:
            try:
                oanda_trade_id = position['oanda_trade_id']
                logger.info(f"🌐 Closing OANDA trade: {oanda_trade_id}")
                
                if self.oanda_handler.close_trade(oanda_trade_id):
                    logger.info(f"✅ OANDA trade closed successfully")
                else:
                    logger.warning("⚠️ OANDA close failed - continuing with local close")
            
            except Exception as e:
                logger.error(f"❌ OANDA close error: {e}")
                logger.info("Continuing with local close")
        
        # ✅ NEW: Close on Alpaca (Crypto)
        if position.get('alpaca_order_id') and self.alpaca_handler:
            try:
                alpaca_order_id = position['alpaca_order_id']
                logger.info(f"🌐 Closing Alpaca position: {position['symbol']}")
                
                if self.alpaca_handler.close_position(position['symbol']):
                    logger.info(f"✅ Alpaca position closed successfully")
                else:
                    logger.warning("⚠️ Alpaca close failed - continuing with local close")
            
            except Exception as e:
                logger.error(f"❌ Alpaca close error: {e}")
                logger.info("Continuing with local close")
        # ------------------------------------------------------------------
                
        # Apply slippage to exit price
        exit_price_with_slippage = self._simulate_fill(exit_price, position['direction'], position['market_type'], is_exit=True)
        
        # Calculate P&L
        try:
            if position['market_type'] == 'crypto':
                investment = position['quantity'] * position['entry_price']
                pnl_result = self.pnl_calc.calculate_crypto_pnl(
                    entry_price=position['entry_price'],
                    exit_price=exit_price_with_slippage,
                    investment_usd=investment,
                    direction=position['direction'],
                    include_costs=True
                )
            else:  # forex
                pnl_result = self.pnl_calc.calculate_forex_pnl(
                    entry_price=position['entry_price'],
                    exit_price=exit_price_with_slippage,
                    lot_size=position['lot_size'],
                    pair=position['symbol'],
                    direction=position['direction'],
                    include_costs=True
                )
            
            # Update trade in database
            updates = {
                'status': 'CLOSED',
                'exit_price': exit_price_with_slippage,
                'exit_timestamp': datetime.now(),
                'exit_reason': reason,
                'pnl_usd': pnl_result['pnl_usd'],
                'pnl_pct': pnl_result.get('pnl_pct', 0),
                'gross_pnl': pnl_result['gross_pnl_usd'],
                'transaction_costs': pnl_result['total_costs'],
                'net_pnl': pnl_result['pnl_usd'],
                'exchange_price_at_exit': exchange_price or exit_price
            }
            
            if position['market_type'] == 'forex':
                updates['pips'] = pnl_result.get('pnl_pips', 0)
            
            self.db.update_trade(trade_id, updates)
            
            # Update account balance
            self.current_balance += pnl_result['pnl_usd']
            self.daily_pnl += pnl_result['pnl_usd']
            self.total_pnl += pnl_result['pnl_usd']
            
            # Update statistics
            self.total_trades += 1
            if pnl_result['pnl_usd'] > 0:
                self.winning_trades += 1
            elif pnl_result['pnl_usd'] < 0:
                self.losing_trades += 1
            
            # Update drawdown
            if self.current_balance > self.peak_balance:
                self.peak_balance = self.current_balance
            
            drawdown_pct = ((self.peak_balance - self.current_balance) / self.peak_balance) * 100
            if drawdown_pct > self.max_drawdown:
                self.max_drawdown = drawdown_pct
            
            # Remove from open positions
            with self.position_lock:
                self.open_positions.pop(trade_id, None)
            
            logger.info(f"✅ Position closed: #{trade_id} {reason}, P&L=${pnl_result['pnl_usd']:.2f}, Balance=${self.current_balance:,.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to close position #{trade_id}: {e}")
            return False
    
    def close_all_positions(self, exit_price_dict: Dict[str, float], reason: str = 'EOD'):
        """
        Close all open positions.
        
        Args:
            exit_price_dict: Dict mapping symbol -> exit price
            reason: Closure reason
        """
        with self.position_lock:
            trade_ids = list(self.open_positions.keys())
        
        for trade_id in trade_ids:
            position = self.open_positions.get(trade_id)
            if position:
                symbol = position['symbol']
                exit_price = exit_price_dict.get(symbol, position['entry_price'])
                self.close_position(trade_id, exit_price, reason)
        
        logger.info(f"Closed {len(trade_ids)} positions ({reason})")
    
    # ========================================================================
    # RISK CHECKS
    # ========================================================================
    
    def _check_risk_limits(self, signal: Dict) -> Dict[str, Any]:
        """
        Check if order passes risk management rules.
        
        Args:
            signal: Order signal
        
        Returns:
            Dict with 'allowed' bool and 'reason' str
        """
        # Check 1: Maximum open positions
        if len(self.open_positions) >= self.risk_config['max_open_positions']:
            return {
                'allowed': False,
                'reason': f"Max open positions reached ({self.risk_config['max_open_positions']})"
            }
        
        # Check 2: Daily loss limit
        if self.daily_pnl <= -self.risk_config['max_daily_loss_usd']:
            return {
                'allowed': False,
                'reason': f"Daily loss limit reached (${-self.daily_pnl:.2f})"
            }
        
        # Check 3: Daily trade limit
        if self.daily_trade_count >= self.risk_config['max_trades_per_day']:
            return {
                'allowed': False,
                'reason': f"Daily trade limit reached ({self.daily_trade_count})"
            }
        
        # Check 4: Sufficient balance
        required_capital = self._calculate_required_capital(signal)
        if self.current_balance < required_capital:
            return {
                'allowed': False,
                'reason': f"Insufficient balance (required: ${required_capital:.2f}, available: ${self.current_balance:.2f})"
            }
        
        # Check 5: Risk/Reward ratio
        rr_ratio = self._calculate_rr_ratio(
            signal['entry_price'],
            signal['stop_loss'],
            signal.get('take_profit_1', signal.get('take_profit')),
            signal['direction']
        )
        if rr_ratio < self.risk_config['min_risk_reward_ratio']:
            return {
                'allowed': False,
                'reason': f"Risk/Reward ratio too low ({rr_ratio:.2f} < {self.risk_config['min_risk_reward_ratio']})"
            }
        
        return {'allowed': True, 'reason': 'All checks passed'}
    
    def _calculate_required_capital(self, signal: Dict) -> float:
        """Calculate capital required for trade."""
        if signal['market_type'] == 'crypto':
            return self.crypto_config['investment_per_trade_usd']
        else:  # forex
            # Margin calculation (simplified)
            return self.forex_config['lot_size'] * 100  # $100 per 0.01 lot
    
    def _calculate_rr_ratio(self, entry: float, sl: float, tp: float, direction: str) -> float:
        """Calculate risk/reward ratio."""
        if direction == 'BUY':
            risk = abs(entry - sl)
            reward = abs(tp - entry)
        else:  # SELL
            risk = abs(sl - entry)
            reward = abs(entry - tp)
        
        return reward / risk if risk > 0 else 0
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _simulate_fill(self, price: float, direction: str, market_type: str, is_exit: bool = False) -> float:
        """
        Simulate order fill with realistic slippage.
        
        Args:
            price: Order price
            direction: 'BUY' or 'SELL'
            market_type: 'crypto' or 'forex'
            is_exit: Whether this is an exit order
        
        Returns:
            Fill price with slippage
        """
        if market_type == 'crypto':
            slippage_pct = self.crypto_config['slippage_pct']
        else:
            # Convert pip slippage to percentage
            slippage_pips = self.forex_config['slippage_pips']
            slippage_pct = (slippage_pips * 0.0001) / price
        
        # Apply slippage in unfavorable direction
        if (direction == 'BUY' and not is_exit) or (direction == 'SELL' and is_exit):
            # Buying or closing short: pay higher
            fill_price = price * (1 + slippage_pct)
        else:
            # Selling or closing long: receive lower
            fill_price = price * (1 - slippage_pct)
        
        return fill_price
    
    def _calculate_position_size(self, signal: Dict, fill_price: float) -> Dict:
            """Calculate position size based on market type."""
            if signal['market_type'] == 'crypto':
                investment = self.crypto_config['investment_per_trade_usd']
                quantity = investment / fill_price
                return {'quantity': quantity, 'investment_usd': investment}
            else:  # forex
                # ✅ FIX: Use signal's position_size if provided, otherwise use config
                if 'position_size' in signal and signal['position_size']:
                    lot_size = float(signal['position_size'])
                    
                    # Apply safety limits
                    lot_size = max(self.forex_config['min_lot_size'], lot_size)
                    lot_size = min(self.forex_config['max_lot_size'], lot_size)
                    
                    logger.info(f"📊 Using signal position size: {lot_size:.4f} lots (limited to {self.forex_config['min_lot_size']}-{self.forex_config['max_lot_size']})")
                else:
                    # Fallback to config default (safe mode)
                    lot_size = self.forex_config['lot_size']
                    logger.warning(f"⚠️ No position_size in signal, using config default: {lot_size} lots")
                
                return {'lot_size': lot_size}
    
    # ========================================================================
    # ACCOUNT STATUS
    # ========================================================================
    
    def get_account_status(self) -> Dict[str, Any]:
        """
        Get current account status.
        
        Returns:
            Dict with account information
        """
        # Calculate equity (balance + unrealized P&L)
        unrealized_pnl = sum(pos['unrealized_pnl'] for pos in self.open_positions.values())
        equity = self.current_balance + unrealized_pnl
        
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        return {
            'balance': self.current_balance,
            'initial_balance': self.initial_balance,
            'equity': equity,
            'unrealized_pnl': unrealized_pnl,
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'total_pnl_pct': (self.total_pnl / self.initial_balance) * 100,
            'open_positions': len(self.open_positions),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'daily_trade_count': self.daily_trade_count,
            'max_drawdown': self.max_drawdown
        }
    
    def get_open_positions_list(self) -> List[Dict]:
        """Get list of all open positions."""
        with self.position_lock:
            return list(self.open_positions.values())
    
    def reset_daily_stats(self):
        """Reset daily statistics (call at start of new trading day)."""
        self.daily_pnl = 0.0
        self.daily_trade_count = 0
        logger.info("Daily statistics reset")


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PAPER ORDER MANAGER TEST")
    print("=" * 70)
    
    # Initialize components
    db = TradeDatabase("paper_trading/data/test_orders.db")
    pnl_calc = PnLCalculator()
    order_mgr = PaperOrderManager(db, pnl_calc, initial_balance=10000.0)
    
    # Test 1: Place BUY order
    print("\n1️⃣ TEST: Place BUY Order")
    print("-" * 70)
    signal = {
        'symbol': 'BTC/USDT',
        'direction': 'BUY',
        'entry_price': 67250.00,
        'stop_loss': 67100.00,
        'take_profit': 67550.00,
        'strategy_name': 'CRT-TBS',
        'confidence': 85,
        'market_type': 'crypto'
    }
    result = order_mgr.place_order(signal, 67249.80)
    print(f"✅ Order result: {result}")
    trade_id = result['trade_id'] if result['success'] else None
    
    # Test 2: Check position (price moves toward TP)
    print("\n2️⃣ TEST: Check Position (Price → TP)")
    print("-" * 70)
    order_mgr.check_open_positions('BTC/USDT', 67400.00)
    print(f"Open positions: {len(order_mgr.get_open_positions_list())}")
    
    # Test 3: TP Hit
    print("\n3️⃣ TEST: Take Profit Hit")
    print("-" * 70)
    order_mgr.check_open_positions('BTC/USDT', 67560.00)  # Above TP
    print(f"Open positions: {len(order_mgr.get_open_positions_list())}")
    
    # Test 4: Account Status
    print("\n4️⃣ TEST: Account Status")
    print("-" * 70)
    status = order_mgr.get_account_status()
    print(f"Balance: ${status['balance']:,.2f}")
    print(f"Total P&L: ${status['total_pnl']:,.2f}")
    print(f"Win Rate: {status['win_rate']:.1f}%")
    
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE")
    print("=" * 70)
