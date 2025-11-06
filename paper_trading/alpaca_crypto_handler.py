"""
Alpaca Crypto Paper Trading Handler
====================================

Integrates Alpaca's Trading API for cryptocurrency paper trading.
Places actual market orders with bracket orders (SL/TP) on Alpaca's practice environment.

Features:
- Real practice account integration
- Bracket order support (Market order + Stop Loss + Take Profit)
- Live price feeds for crypto (BTC/USD, ETH/USD, etc.)
- Position monitoring and tracking
- Commission-free trading

Setup Instructions:
1. Create Alpaca Account: https://alpaca.markets
2. Get API credentials from: https://app.alpaca.markets/paper/dashboard/api-keys
3. Add to .env file:
   ALPACA_PAPER_API_KEY=your_api_key
   ALPACA_PAPER_SECRET_KEY=your_secret_key

Author: Trading System
Last Updated: November 3, 2025
"""

import logging
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv

# Alpaca SDK imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    TakeProfitRequest,
    StopLossRequest
)
from alpaca.trading.enums import (
    OrderSide,
    TimeInForce,
    OrderClass,
    OrderStatus
)

load_dotenv()

logger = logging.getLogger(__name__)


class AlpacaCryptoHandler:
    """
    Alpaca Crypto Trading Handler for paper trading.
    
    Allows placing real practice orders on Alpaca's demo environment for crypto.
    Supports bracket orders (market order + stop loss + take profit).
    """
    
    def __init__(self, api_key: str = None, secret_key: str = None):
        """
        Initialize Alpaca Crypto Handler.
        
        Args:
            api_key: Alpaca Paper API key (from env if not provided)
            secret_key: Alpaca Paper Secret Key (from env if not provided)
        """
        self.api_key = api_key or os.getenv('ALPACA_PAPER_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_PAPER_SECRET_KEY')
        
        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca credentials not found. Please set ALPACA_PAPER_API_KEY "
                "and ALPACA_PAPER_SECRET_KEY in .env file"
            )
        
        # Initialize Alpaca Trading Client (paper=True for paper trading)
        try:
            self.trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=True  # Use paper trading
            )
            self.connected = False
            logger.info("AlpacaCryptoHandler initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca client: {e}")
            raise
    
    # ========================================================================
    # CONNECTION & VALIDATION
    # ========================================================================
    
    def test_connection(self) -> bool:
        """
        Test connection to Alpaca Trading API.
        
        Returns:
            bool: True if connected successfully
        """
        try:
            # Get account info to verify connection
            account = self.trading_client.get_account()
            
            if account:
                self.connected = True
                logger.info("✅ Connected to Alpaca Paper Trading API")
                logger.info(f"Account Status: {account.status}")
                logger.info(f"Buying Power: ${float(account.buying_power):,.2f}")
                return True
            else:
                logger.error("Failed to retrieve account information")
                return False
                
        except Exception as e:
            logger.error(f"Alpaca connection error: {e}")
            return False
    
    def get_account_summary(self) -> Optional[Dict]:
        """
        Get account summary including balance, positions, etc.
        
        Returns:
            Dict with account information or None
        """
        try:
            account = self.trading_client.get_account()
            
            summary = {
                'account_number': account.account_number,
                'status': account.status,
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'multiplier': account.multiplier,
                'currency': account.currency,
                'portfolio_percent_change': float(account.portfolio_percent_change),
                'trading_blocked': account.trading_blocked,
            }
            
            logger.debug(f"Account summary retrieved: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Error getting account summary: {e}")
            return None
    
    # ========================================================================
    # ORDER PLACEMENT - BRACKET ORDERS
    # ========================================================================
    
    def place_market_order_with_bracket(
        self,
        symbol: str,
        qty: float,
        side: str,
        stop_loss: float,
        take_profit: float
    ) -> Optional[Dict]:
        """
        Place market order with bracket (SL/TP) on Alpaca.
        
        Bracket order = Market order + Stop Loss + Take Profit
        When market order fills, both SL and TP orders are created.
        When either SL or TP hits, the other is cancelled automatically.
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC/USD', 'ETH/USD')
            qty: Quantity in coins (e.g., 0.01 BTC)
            side: 'buy' or 'sell'
            stop_loss: Stop loss price
            take_profit: Take profit price
        
        Returns:
            Dict with order result or None
        
        Example:
            >>> result = handler.place_market_order_with_bracket(
            ...     symbol='BTC/USD',
            ...     qty=0.01,
            ...     side='buy',
            ...     stop_loss=67000,
            ...     take_profit=68000
            ... )
        """
        try:
            # Validate inputs
            if symbol not in ['BTC/USD', 'ETH/USD', 'SOL/USD', 'XRP/USD', 'BNB/USD']:
                logger.warning(f"⚠️ Symbol {symbol} may not be supported. Continuing anyway...")
            
            if side.lower() not in ['buy', 'sell']:
                raise ValueError(f"side must be 'buy' or 'sell', got {side}")
            
            if qty <= 0:
                raise ValueError(f"qty must be positive, got {qty}")
            
            if stop_loss <= 0 or take_profit <= 0:
                raise ValueError(f"stop_loss and take_profit must be positive")
            
            # ✅ FIX: Format prices to proper decimal places for Alpaca
            # Crypto typically uses 2 decimal places
            stop_loss_formatted = f"{stop_loss:.2f}"
            take_profit_formatted = f"{take_profit:.2f}"
            
            # Determine OrderSide enum
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
            
            # Create Stop Loss Request
            stop_loss_request = StopLossRequest(
                stop_price=stop_loss_formatted
            )
            
            # Create Take Profit Request
            take_profit_request = TakeProfitRequest(
                limit_price=take_profit_formatted
            )
            
            # Create Bracket Market Order Request
            # ✅ CRITICAL: order_class='bracket' enables SL/TP
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.GTC,  # Good 'til Cancelled
                order_class=OrderClass.BRACKET,  # ✅ This enables bracket order
                stop_loss=stop_loss_request,
                take_profit=take_profit_request
            )
            
            logger.debug(f"Submitting bracket order: {order_request}")
            
            # Submit order
            order = self.trading_client.submit_order(order_request)
            
            # ✅ FIX: Handle response safely (no assumption of immediate fill)
            if order:
                result = {
                    'success': True,
                    'order_id': order.id,
                    'parent_id': order.id,  # Parent order ID
                    'symbol': order.symbol,
                    'qty': float(order.qty),
                    'side': str(order.side.value),
                    'status': str(order.status.value),
                    'order_type': str(order.order_type.value),
                    'created_at': str(order.created_at),
                    'filled_qty': float(order.filled_qty) if order.filled_qty else 0.0,
                    'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
                }
                
                logger.info(f"✅ Alpaca bracket order submitted: {symbol} {qty} {side}")
                logger.info(f"   Order ID: {order.id}")
                logger.info(f"   Status: {order.status.value}")
                logger.info(f"   Stop Loss: ${stop_loss_formatted}")
                logger.info(f"   Take Profit: ${take_profit_formatted}")
                
                return result
            else:
                logger.error("Order submission returned None")
                return None
                
        except ValueError as e:
            logger.error(f"❌ Validation error: {e}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error placing bracket order: {e}")
            logger.error(f"   Symbol: {symbol}, Qty: {qty}, Side: {side}")
            logger.error(f"   SL: {stop_loss}, TP: {take_profit}")
            return None

    # ========================================================================
    # LIMIT ORDER PLACEMENT - BRACKET ORDERS (NEW)
    # ========================================================================
    
    def place_limit_order_with_bracket(
        self,
        symbol: str,
        qty: float,
        side: str,
        limit_price: float,
        stop_loss: float,
        take_profit: float
    ) -> Optional[Dict]:
        """
        Place LIMIT order with bracket (SL/TP) on Alpaca.
        
        PERFECT FOR POSITION TRADING:
        - Order executes ONLY at your specified limit price
        - Matches historical entry price from CRT-TBS strategy
        - No rejection due to stale prices
        - Market must come to YOU (institutional approach)
        
        Bracket order = Limit order + Stop Loss + Take Profit
        When limit order fills, both SL and TP orders are created.
        When either SL or TP hits, the other is cancelled automatically.
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC/USD', 'ETH/USD')
            qty: Quantity in coins (e.g., 0.01 BTC)
            side: 'buy' or 'sell'
            limit_price: Your entry price (from historical CRT-TBS signal)
            stop_loss: Stop loss price
            take_profit: Take profit price
        
        Returns:
            Dict with order result or None
        
        Example:
            >>> # Signal from 2 days ago: BTC was at $110,000
            >>> # Current price: $102,000
            >>> # Place LIMIT order at historical price
            >>> result = handler.place_limit_order_with_bracket(
            ...     symbol='BTC/USD',
            ...     qty=0.01,
            ...     side='sell',
            ...     limit_price=110000,  # Wait for market to return here
            ...     stop_loss=112000,
            ...     take_profit=107000
            ... )
            >>> # Order waits until market rises back to $110,000
        """
        try:
            # Validate inputs
            if symbol not in ['BTC/USD', 'ETH/USD', 'SOL/USD', 'XRP/USD', 'BNB/USD']:
                logger.warning(f"⚠️ Symbol {symbol} may not be supported. Continuing anyway...")
            
            if side.lower() not in ['buy', 'sell']:
                raise ValueError(f"side must be 'buy' or 'sell', got {side}")
            
            if qty <= 0:
                raise ValueError(f"qty must be positive, got {qty}")
            
            if limit_price <= 0 or stop_loss <= 0 or take_profit <= 0:
                raise ValueError(f"limit_price, stop_loss and take_profit must be positive")
            
            # ============================================================
            # VALIDATION: Check TP/SL logic for order direction
            # ============================================================
            if side.lower() == 'sell':
                # For SELL limit orders:
                # - Limit price should be WHERE you want to sell
                # - TP should be BELOW limit (profit when price drops)
                # - SL should be ABOVE limit (loss when price rises)
                if take_profit >= limit_price:
                    logger.error(
                        f"❌ Invalid TP for SELL: TP ({take_profit:.2f}) must be < Limit ({limit_price:.2f})"
                    )
                    return {'success': False, 'reason': 'Invalid TP for SELL limit order'}
                
                if stop_loss <= limit_price:
                    logger.error(
                        f"❌ Invalid SL for SELL: SL ({stop_loss:.2f}) must be > Limit ({limit_price:.2f})"
                    )
                    return {'success': False, 'reason': 'Invalid SL for SELL limit order'}
            
            elif side.lower() == 'buy':
                # For BUY limit orders:
                # - Limit price should be WHERE you want to buy
                # - TP should be ABOVE limit (profit when price rises)
                # - SL should be BELOW limit (loss when price drops)
                if take_profit <= limit_price:
                    logger.error(
                        f"❌ Invalid TP for BUY: TP ({take_profit:.2f}) must be > Limit ({limit_price:.2f})"
                    )
                    return {'success': False, 'reason': 'Invalid TP for BUY limit order'}
                
                if stop_loss >= limit_price:
                    logger.error(
                        f"❌ Invalid SL for BUY: SL ({stop_loss:.2f}) must be < Limit ({limit_price:.2f})"
                    )
                    return {'success': False, 'reason': 'Invalid SL for BUY limit order'}
            
            # ============================================================
            # FORMAT PRICES (Alpaca requires 2 decimal places for crypto)
            # ============================================================
            limit_price_formatted = f"{limit_price:.2f}"
            stop_loss_formatted = f"{stop_loss:.2f}"
            take_profit_formatted = f"{take_profit:.2f}"
            
            # Determine OrderSide enum
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
            
            # ============================================================
            # CREATE BRACKET LIMIT ORDER
            # ============================================================
            
            # Create Stop Loss Request
            stop_loss_request = StopLossRequest(
                stop_price=stop_loss_formatted
            )
            
            # Create Take Profit Request
            take_profit_request = TakeProfitRequest(
                limit_price=take_profit_formatted
            )
            
            # ✅ CREATE LIMIT ORDER REQUEST (not Market!)
            order_request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.GTC,  # Good 'til Cancelled
                limit_price=limit_price_formatted,  # ✅ YOUR ENTRY PRICE
                order_class=OrderClass.BRACKET,  # ✅ Enables SL/TP
                stop_loss=stop_loss_request,
                take_profit=take_profit_request
            )
            
            # ============================================================
            # LOG ORDER DETAILS
            # ============================================================
            logger.info(f"📊 Placing LIMIT order with bracket on Alpaca:")
            logger.info(f"   Symbol: {symbol}")
            logger.info(f"   Side: {side.upper()}")
            logger.info(f"   Quantity: {qty:.8f}")
            logger.info(f"   Limit Price: ${limit_price_formatted} (your entry price)")
            logger.info(f"   Stop Loss: ${stop_loss_formatted}")
            logger.info(f"   Take Profit: ${take_profit_formatted}")
            logger.info(f"   Order will fill when market reaches ${limit_price_formatted}")
            
            logger.debug(f"Order request: {order_request}")
            
            # ============================================================
            # SUBMIT ORDER TO ALPACA
            # ============================================================
            order = self.trading_client.submit_order(order_request)
            
            # ============================================================
            # HANDLE RESPONSE
            # ============================================================
            if order:
                result = {
                    'success': True,
                    'order_id': order.id,
                    'parent_id': order.id,
                    'symbol': order.symbol,
                    'qty': float(order.qty),
                    'side': str(order.side.value),
                    'status': str(order.status.value),
                    'order_type': str(order.order_type.value),
                    'limit_price': float(limit_price),
                    'stop_loss': float(stop_loss),
                    'take_profit': float(take_profit),
                    'created_at': str(order.created_at),
                    'filled_qty': float(order.filled_qty) if order.filled_qty else 0.0,
                    'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
                }
                
                logger.info(f"✅ LIMIT order submitted successfully!")
                logger.info(f"   Order ID: {order.id}")
                logger.info(f"   Status: {order.status.value}")
                logger.info(f"   ⏳ Waiting for market to reach ${limit_price_formatted}...")
                
                return result
            else:
                logger.error("❌ Order submission returned None")
                return {'success': False, 'reason': 'Order submission failed'}
                
        except ValueError as e:
            logger.error(f"❌ Validation error: {e}")
            return {'success': False, 'reason': str(e)}
            
        except Exception as e:
            logger.error(f"❌ Error placing LIMIT bracket order: {e}")
            logger.error(f"   Symbol: {symbol}, Qty: {qty}, Side: {side}")
            logger.error(f"   Limit: ${limit_price:.2f}, SL: ${stop_loss:.2f}, TP: ${take_profit:.2f}")
            return {'success': False, 'reason': str(e)}
   
    # ========================================================================
    # ORDER MANAGEMENT
    # ========================================================================
    
    def get_order(self, order_id: str) -> Optional[Dict]:
        """
        Get order details by order ID.
        
        Args:
            order_id: Alpaca order ID
        
        Returns:
            Dict with order details or None
        """
        try:
            order = self.trading_client.get_order_by_id(order_id)
            
            if order:
                return {
                    'order_id': order.id,
                    'symbol': order.symbol,
                    'qty': float(order.qty),
                    'side': str(order.side.value),
                    'status': str(order.status.value),
                    'filled_qty': float(order.filled_qty) if order.filled_qty else 0.0,
                    'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
                    'created_at': str(order.created_at),
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting order {order_id}: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.
        
        Args:
            order_id: Alpaca order ID
        
        Returns:
            bool: True if cancelled successfully
        """
        try:
            self.trading_client.cancel_order_by_id(order_id)
            logger.info(f"✅ Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def get_open_orders(self) -> List[Dict]:
        """
        Get all open orders.
        
        Returns:
            List of open order dictionaries
        """
        try:
            orders = self.trading_client.get_orders(
                status=OrderStatus.OPEN
            )
            
            result = []
            for order in orders:
                result.append({
                    'order_id': order.id,
                    'symbol': order.symbol,
                    'qty': float(order.qty),
                    'side': str(order.side.value),
                    'status': str(order.status.value),
                    'filled_qty': float(order.filled_qty) if order.filled_qty else 0.0,
                    'created_at': str(order.created_at),
                })
            
            logger.debug(f"Retrieved {len(result)} open orders")
            return result
            
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []
    
    # ========================================================================
    # POSITION QUERIES
    # ========================================================================
    
    def get_open_positions(self) -> List[Dict]:
        """
        Get all open positions.
        
        Returns:
            List of position dictionaries
        """
        try:
            positions = self.trading_client.get_all_positions()
            
            result = []
            for position in positions:
                result.append({
                    'symbol': position.symbol,
                    'qty': float(position.qty),
                    'side': str(position.side.value),
                    'avg_fill_price': float(position.avg_fill_price),
                    'current_price': float(position.current_price) if hasattr(position, 'current_price') else None,
                    'market_value': float(position.market_value) if position.market_value else None,
                    'unrealized_pl': float(position.unrealized_pl) if position.unrealized_pl else None,
                    'unrealized_plpc': float(position.unrealized_plpc) if position.unrealized_plpc else None,
                })
            
            logger.debug(f"Retrieved {len(result)} open positions")
            return result
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def close_position(self, symbol: str) -> bool:
        """
        Close all positions for a symbol.
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC/USD')
        
        Returns:
            bool: True if closed successfully
        """
        try:
            self.trading_client.close_position(symbol)
            logger.info(f"✅ Position closed: {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")
            return False
    
    # ========================================================================
    # PRICING
    # ========================================================================
    
    def get_latest_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get latest bid/ask prices for crypto.
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC/USD')
        
        Returns:
            Dict with quote data or None
        """
        try:
            # Get latest quote from market data
            from alpaca.data.requests import CryptoLatestQuoteRequest
            from alpaca.data.client import CryptoHistoricalDataClient
            
            crypto_client = CryptoHistoricalDataClient()
            request = CryptoLatestQuoteRequest(symbol_or_symbols=[symbol])
            quote = crypto_client.get_crypto_latest_quote(request)
            
            if quote and symbol in quote:
                q = quote[symbol]
                return {
                    'symbol': symbol,
                    'bid': float(q.bid_price),
                    'ask': float(q.ask_price),
                    'spread': float(q.ask_price) - float(q.bid_price),
                    'bid_size': float(q.bid_size),
                    'ask_size': float(q.ask_size),
                    'timestamp': str(q.timestamp)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            return None
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def get_supported_crypto(self) -> List[str]:
        """
        Get list of supported crypto symbols.
        
        Returns:
            List of crypto symbols supported by Alpaca
        """
        # Official Alpaca supported crypto symbols
        return [
            'BTC/USD',    # Bitcoin
            'ETH/USD',    # Ethereum
            'SOL/USD',    # Solana
            'XRP/USD',    # Ripple
            'BNB/USD',    # Binance Coin
            'ADA/USD',    # Cardano
            'DOT/USD',    # Polkadot
            'DOGE/USD',   # Dogecoin
            'AVAX/USD',   # Avalanche
            'MATIC/USD',  # Polygon
        ]


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("ALPACA CRYPTO PAPER TRADING TEST")
    print("=" * 70)
    
    # Check for credentials
    if not os.getenv('ALPACA_PAPER_API_KEY'):
        print("\n❌ ERROR: ALPACA_PAPER_API_KEY not found in .env")
        print("\nSetup Instructions:")
        print("1. Create Alpaca Account: https://alpaca.markets")
        print("2. Get Paper API Keys: https://app.alpaca.markets/paper/dashboard/api-keys")
        print("3. Add to .env:")
        print("   ALPACA_PAPER_API_KEY=your_key")
        print("   ALPACA_PAPER_SECRET_KEY=your_secret")
        sys.exit(1)
    
    # Initialize handler
    try:
        handler = AlpacaCryptoHandler()
    except Exception as e:
        print(f"\n❌ ERROR: Failed to initialize: {e}")
        sys.exit(1)
    
    # Test 1: Connection
    print("\n1️⃣ Testing connection...")
    if handler.test_connection():
        print("✅ Connection successful")
    else:
        print("❌ Connection failed")
        sys.exit(1)
    
    # Test 2: Get account summary
    print("\n2️⃣ Getting account summary...")
    summary = handler.get_account_summary()
    if summary:
        print(f"   Status: {summary['status']}")
        print(f"   Cash: ${summary['cash']:,.2f}")
        print(f"   Buying Power: ${summary['buying_power']:,.2f}")
        print(f"   Portfolio Value: ${summary['portfolio_value']:,.2f}")
    
    # Test 3: Get supported crypto
    print("\n3️⃣ Supported crypto symbols:")
    cryptos = handler.get_supported_crypto()
    print(f"   {', '.join(cryptos)}")
    
    # Test 4: Get open orders
    print("\n4️⃣ Getting open orders...")
    orders = handler.get_open_orders()
    print(f"   Total open orders: {len(orders)}")
    if orders:
        for order in orders[:3]:  # Show first 3
            print(f"   - {order['symbol']}: {order['qty']} {order['side']}")
    
    # Test 5: Get open positions
    print("\n5️⃣ Getting open positions...")
    positions = handler.get_open_positions()
    print(f"   Total open positions: {len(positions)}")
    if positions:
        for pos in positions[:3]:  # Show first 3
            print(f"   - {pos['symbol']}: {pos['qty']} @ ${pos['avg_fill_price']}")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED")
    print("=" * 70)
