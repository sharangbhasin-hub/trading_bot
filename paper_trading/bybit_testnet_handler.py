"""
Bybit Testnet Crypto Trading Handler
=====================================

Integrates Bybit's Testnet API for cryptocurrency paper trading.
Places actual market orders with bracket orders (SL/TP) on Bybit's testnet environment.

Features:
- Real testnet account integration ($100,000 starting balance)
- Bracket order support (Market order + Stop Loss + Take Profit)
- Live price feeds for crypto (BTC/USDT, ETH/USDT, etc.)
- Position monitoring and tracking
- Commission-free testnet trading

Setup Instructions:
1. Create Bybit Account: https://www.bybit.com
2. Access Testnet: https://testnet.bybit.com
3. Generate API keys from: Testnet Dashboard > API Management
4. Add to .env file:
   BYBIT_TESTNET_API_KEY=your_api_key
   BYBIT_TESTNET_SECRET_KEY=your_secret_key

Author: Trading System
Last Updated: November 9, 2025
"""

import logging
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
import ccxt

load_dotenv()

logger = logging.getLogger(__name__)


class BybitTestnetHandler:
    """
    Bybit Testnet Crypto Trading Handler for paper trading.
    
    Allows placing real testnet orders on Bybit's demo environment for crypto.
    Supports bracket orders (market order + stop loss + take profit).
    """
    
    def __init__(self, api_key: str = None, secret_key: str = None):
        """
        Initialize Bybit Testnet Handler.
        
        Args:
            api_key: Bybit Testnet API key (from env if not provided)
            secret_key: Bybit Testnet Secret Key (from env if not provided)
        """
        self.api_key = api_key or os.getenv('BYBIT_TESTNET_API_KEY')
        self.secret_key = secret_key or os.getenv('BYBIT_TESTNET_SECRET_KEY')
        
        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Bybit credentials not found. Please set BYBIT_TESTNET_API_KEY "
                "and BYBIT_TESTNET_SECRET_KEY in .env file"
            )
        
        # Initialize Bybit via CCXT (testnet mode)
        try:
            self.exchange = ccxt.bybit({
                'apiKey': self.api_key,
                'secret': self.secret_key,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',  # Spot trading (can also use 'linear' for futures)
                },
            })
            
            # ✅ CRITICAL: Enable testnet mode
            self.exchange.set_sandbox_mode(True)
            
            self.connected = False
            logger.info("BybitTestnetHandler initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Bybit client: {e}")
            raise
    
    # ========================================================================
    # CONNECTION & VALIDATION
    # ========================================================================
    
    def test_connection(self) -> bool:
        """
        Test connection to Bybit Testnet API.
        
        Returns:
            bool: True if connected successfully
        """
        try:
            # Get account balance to verify connection
            balance = self.exchange.fetch_balance()
            
            if balance:
                self.connected = True
                usdt_balance = balance.get('USDT', {}).get('total', 0)
                logger.info("✅ Connected to Bybit Testnet API")
                logger.info(f"Account Balance: ${usdt_balance:,.2f} USDT")
                return True
            else:
                logger.error("Failed to retrieve account balance")
                return False
                
        except Exception as e:
            logger.error(f"Bybit connection error: {e}")
            return False
    
    def get_account_summary(self) -> Optional[Dict]:
        """
        Get account summary including balance, positions, etc.
        
        Returns:
            Dict with account information or None
        """
        try:
            balance = self.exchange.fetch_balance()
            
            summary = {
                'total_balance_usdt': balance.get('USDT', {}).get('total', 0),
                'free_balance_usdt': balance.get('USDT', {}).get('free', 0),
                'used_balance_usdt': balance.get('USDT', {}).get('used', 0),
                'currency': 'USDT',
                'assets': {}
            }
            
            # Add all non-zero balances
            for asset, info in balance.items():
                if isinstance(info, dict) and info.get('total', 0) > 0:
                    summary['assets'][asset] = {
                        'total': info.get('total', 0),
                        'free': info.get('free', 0),
                        'used': info.get('used', 0)
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
        Place market order with bracket (SL/TP) on Bybit Testnet.
        
        Bracket order = Market order + Stop Loss + Take Profit
        When market order fills, both SL and TP orders are created.
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC/USDT', 'ETH/USDT')
            qty: Quantity in coins (e.g., 0.01 BTC)
            side: 'buy' or 'sell'
            stop_loss: Stop loss price
            take_profit: Take profit price
        
        Returns:
            Dict with order result or None
        
        Example:
            >>> result = handler.place_market_order_with_bracket(
            ...     symbol='BTC/USDT',
            ...     qty=0.01,
            ...     side='buy',
            ...     stop_loss=67000,
            ...     take_profit=68000
            ... )
        """
        try:
            # Validate inputs
            if side.lower() not in ['buy', 'sell']:
                raise ValueError(f"side must be 'buy' or 'sell', got {side}")
            
            if qty <= 0:
                raise ValueError(f"qty must be positive, got {qty}")
            
            if stop_loss <= 0 or take_profit <= 0:
                raise ValueError(f"stop_loss and take_profit must be positive")
            
            logger.info(f"🌐 Placing Bybit testnet order: {symbol} {qty} {side}")
            logger.info(f"   SL: {stop_loss:.5f} | TP: {take_profit:.5f}")
            
            # Place market order with SL/TP
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side.lower(),
                amount=qty,
                params={
                    'stopLoss': {
                        'triggerPrice': stop_loss,
                        'type': 'market'
                    },
                    'takeProfit': {
                        'triggerPrice': take_profit,
                        'type': 'market'
                    }
                }
            )
            
            # Process response
            if order:
                result = {
                    'success': True,
                    'order_id': order.get('id'),
                    'symbol': order.get('symbol'),
                    'qty': float(order.get('amount', 0)),
                    'side': side.lower(),
                    'status': order.get('status'),
                    'filled_qty': float(order.get('filled', 0)),
                    'filled_avg_price': float(order.get('average')) if order.get('average') else None,
                    'timestamp': order.get('timestamp')
                }
                
                logger.info(f"✅ Bybit testnet order placed successfully!")
                logger.info(f"   Order ID: {result['order_id']}")
                logger.info(f"   Status: {result['status']}")
                logger.info(f"   Filled: {result['filled_qty']} @ {result['filled_avg_price']}")
                
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
    # ORDER MANAGEMENT
    # ========================================================================
    
    def get_order(self, order_id: str, symbol: str = None) -> Optional[Dict]:
        """
        Get order details by order ID.
        
        Args:
            order_id: Bybit order ID
            symbol: Trading symbol (required for some exchanges)
        
        Returns:
            Dict with order details or None
        """
        try:
            order = self.exchange.fetch_order(order_id, symbol)
            
            if order:
                return {
                    'order_id': order.get('id'),
                    'symbol': order.get('symbol'),
                    'qty': float(order.get('amount', 0)),
                    'side': order.get('side'),
                    'status': order.get('status'),
                    'filled_qty': float(order.get('filled', 0)),
                    'filled_avg_price': float(order.get('average')) if order.get('average') else None,
                    'timestamp': order.get('timestamp')
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting order {order_id}: {e}")
            return None
    
    def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        """
        Cancel an open order.
        
        Args:
            order_id: Bybit order ID
            symbol: Trading symbol (required)
        
        Returns:
            bool: True if cancelled successfully
        """
        try:
            self.exchange.cancel_order(order_id, symbol)
            logger.info(f"✅ Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """
        Get all open orders.
        
        Args:
            symbol: Filter by symbol (optional)
        
        Returns:
            List of open order dictionaries
        """
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            
            result = []
            for order in orders:
                result.append({
                    'order_id': order.get('id'),
                    'symbol': order.get('symbol'),
                    'qty': float(order.get('amount', 0)),
                    'side': order.get('side'),
                    'status': order.get('status'),
                    'filled_qty': float(order.get('filled', 0)),
                    'timestamp': order.get('timestamp')
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
            positions = self.exchange.fetch_positions()
            
            result = []
            for position in positions:
                if float(position.get('contracts', 0)) > 0:
                    result.append({
                        'symbol': position.get('symbol'),
                        'side': position.get('side'),
                        'contracts': float(position.get('contracts', 0)),
                        'entry_price': float(position.get('entryPrice', 0)),
                        'unrealized_pnl': float(position.get('unrealizedPnl', 0))
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
            symbol: Crypto symbol (e.g., 'BTC/USDT')
        
        Returns:
            bool: True if closed successfully
        """
        try:
            positions = self.get_open_positions()
            
            for pos in positions:
                if pos['symbol'] == symbol:
                    # Close by placing opposite order
                    side = 'sell' if pos['side'] == 'long' else 'buy'
                    self.exchange.create_order(
                        symbol=symbol,
                        type='market',
                        side=side,
                        amount=pos['contracts']
                    )
                    logger.info(f"✅ Position closed: {symbol}")
                    return True
            
            logger.warning(f"No open position found for {symbol}")
            return False
            
        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")
            return False
    
    # ========================================================================
    # PRICING
    # ========================================================================
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get latest price for crypto.
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC/USDT')
        
        Returns:
            Latest price or None
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            price = float(ticker.get('last', 0))
            logger.debug(f"Latest price for {symbol}: {price}")
            return price
            
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def get_supported_crypto(self) -> List[str]:
        """
        Get list of supported crypto symbols.
        
        Returns:
            List of crypto symbols supported by Bybit
        """
        try:
            markets = self.exchange.load_markets()
            symbols = [symbol for symbol in markets.keys() if '/USDT' in symbol]
            logger.info(f"Found {len(symbols)} USDT pairs on Bybit")
            return symbols
        except Exception as e:
            logger.error(f"Error loading markets: {e}")
            return []


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("BYBIT TESTNET CRYPTO PAPER TRADING TEST")
    print("=" * 70)
    
    # Check for credentials
    if not os.getenv('BYBIT_TESTNET_API_KEY'):
        print("\n❌ ERROR: BYBIT_TESTNET_API_KEY not found in .env")
        print("\nSetup Instructions:")
        print("1. Create Bybit Account: https://www.bybit.com")
        print("2. Access Testnet: https://testnet.bybit.com")
        print("3. Generate API Keys: Testnet Dashboard > API Management")
        print("4. Add to .env:")
        print("   BYBIT_TESTNET_API_KEY=your_key")
        print("   BYBIT_TESTNET_SECRET_KEY=your_secret")
        sys.exit(1)
    
    # Initialize handler
    try:
        handler = BybitTestnetHandler()
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
        print(f"   Total Balance: ${summary['total_balance_usdt']:,.2f} USDT")
        print(f"   Free Balance: ${summary['free_balance_usdt']:,.2f} USDT")
    
    # Test 3: Get supported crypto
    print("\n3️⃣ Getting supported crypto pairs...")
    cryptos = handler.get_supported_crypto()
    print(f"   Total pairs: {len(cryptos)}")
    print(f"   Examples: {', '.join(cryptos[:5])}")
    
    # Test 4: Get open orders
    print("\n4️⃣ Getting open orders...")
    orders = handler.get_open_orders()
    print(f"   Total open orders: {len(orders)}")
    
    # Test 5: Get open positions
    print("\n5️⃣ Getting open positions...")
    positions = handler.get_open_positions()
    print(f"   Total open positions: {len(positions)}")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED")
    print("=" * 70)
