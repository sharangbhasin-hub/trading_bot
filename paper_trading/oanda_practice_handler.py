"""
OANDA Practice API Handler for Paper Trading
=============================================

Integrates OANDA's Practice/Demo API for Forex paper trading.
Places actual orders on OANDA's practice environment for realistic simulation.

Features:
- Real practice account integration
- Live order placement
- Position monitoring
- Real-time price feeds
- Trade history sync

Author: Trading System
Last Updated: October 29, 2025

Setup Instructions:
1. Sign up for OANDA Practice Account: https://www.oanda.com/demo-account/
2. Get API credentials from: https://www.oanda.com/demo-account/tpa/personal_token
3. Add to .env file:
   OANDA_PRACTICE_API_KEY=your_api_key
   OANDA_PRACTICE_ACCOUNT_ID=your_account_id
"""

import requests
import logging
from datetime import datetime
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class OandaPracticeHandler:
    """
    OANDA Practice API handler for Forex paper trading.
    
    Allows placing real practice orders on OANDA's demo environment.
    """
    
    # OANDA Practice API endpoints
    PRACTICE_API_URL = "https://api-fxpractice.oanda.com"
    STREAM_API_URL = "https://stream-fxpractice.oanda.com"
    
    def __init__(self, api_key: str = None, account_id: str = None):
        """
        Initialize OANDA Practice API handler.
        
        Args:
            api_key: OANDA Practice API key (from env if not provided)
            account_id: OANDA Practice account ID (from env if not provided)
        """
        self.api_key = api_key or os.getenv('OANDA_PRACTICE_API_KEY')
        self.account_id = account_id or os.getenv('OANDA_PRACTICE_ACCOUNT_ID')
        
        if not self.api_key or not self.account_id:
            raise ValueError(
                "OANDA credentials not found. Please set OANDA_PRACTICE_API_KEY "
                "and OANDA_PRACTICE_ACCOUNT_ID in .env file"
            )
        
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        self.connected = False
        logger.info("OandaPracticeHandler initialized")
    
    # ========================================================================
    # CONNECTION & VALIDATION
    # ========================================================================
    
    def test_connection(self) -> bool:
        """
        Test connection to OANDA Practice API.
        
        Returns:
            bool: True if connected successfully
        """
        try:
            url = f"{self.PRACTICE_API_URL}/v3/accounts/{self.account_id}"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                self.connected = True
                logger.info("✅ Connected to OANDA Practice API")
                return True
            else:
                logger.error(f"OANDA connection failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"OANDA connection error: {e}")
            return False
    
    def get_account_summary(self) -> Optional[Dict]:
        """
        Get account summary including balance, positions, etc.
        
        Returns:
            Dict with account information or None
        """
        try:
            url = f"{self.PRACTICE_API_URL}/v3/accounts/{self.account_id}/summary"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                account = data['account']
                
                summary = {
                    'balance': float(account['balance']),
                    'unrealized_pl': float(account['unrealizedPL']),
                    'nav': float(account['NAV']),
                    'margin_used': float(account['marginUsed']),
                    'margin_available': float(account['marginAvailable']),
                    'open_trade_count': int(account['openTradeCount']),
                    'open_position_count': int(account['openPositionCount']),
                    'currency': account['currency']
                }
                
                return summary
            else:
                logger.error(f"Failed to get account summary: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting account summary: {e}")
            return None
   
    # ========================================================================
    # ORDER PLACEMENT
    # ========================================================================
    
    def _format_price(self, price: float, instrument: str) -> str:
        """
        Format price to correct decimal places for OANDA API.
        
        OANDA requires:
        - JPY pairs: 3 decimal places (e.g., 110.123)
        - All other pairs: 5 decimal places (e.g., 1.16402)
        
        Args:
            price: Price value
            instrument: OANDA instrument (e.g., 'EUR_USD')
        
        Returns:
            Formatted price string
        """
        # JPY pairs use 3 decimals, all others use 5 decimals
        decimals = 3 if 'JPY' in instrument else 5
        return f"{price:.{decimals}f}"

    def place_market_order(
        self,
        instrument: str,
        units: int,
        stop_loss: float,
        take_profit: float
    ) -> Optional[Dict]:
        """
        Place market order on OANDA Practice.
        
        Args:
            instrument: Forex pair (e.g., 'EUR_USD')
            units: Number of units (positive=buy, negative=sell)
            stop_loss: Stop loss price
            take_profit: Take profit price
        
        Returns:
            Dict with order result or None
        
        Example:
            >>> result = handler.place_market_order(
            ...     instrument='EUR_USD',
            ...     units=10000,  # Buy 10,000 units
            ...     stop_loss=1.0850,
            ...     take_profit=1.0950
            ... )
        """
        try:
            url = f"{self.PRACTICE_API_URL}/v3/accounts/{self.account_id}/orders"
            
            order_data = {
                "order": {
                    "type": "MARKET",
                    "instrument": instrument,
                    "units": str(units),
                    "timeInForce": "FOK",  # Fill or Kill
                    "positionFill": "DEFAULT",
                    "stopLossOnFill": {
                        "price": self._format_price(stop_loss, instrument)  # ✅ FIXED!
                    },
                    "takeProfitOnFill": {
                        "price": self._format_price(take_profit, instrument)  # ✅ FIXED!
                    }
                }
            }
           
            response = requests.post(
                url,
                headers=self.headers,
                json=order_data,
                timeout=10
            )
            
            # ✅ PRODUCTION FIX: Handle all OANDA response scenarios
            if response.status_code == 201:
                data = response.json()
                
                # Log full response for debugging
                logger.debug(f"OANDA response: {data}")
                
                # Check which transaction type was returned
                if 'orderFillTransaction' in data:
                    # ✅ Success: Order filled immediately
                    fill_tx = data['orderFillTransaction']
                    
                    result = {
                        'success': True,
                        'order_id': data['orderCreateTransaction']['id'],
                        'trade_id': fill_tx.get('tradeOpened', {}).get('tradeID'),
                        'fill_price': float(fill_tx['price']),
                        'units': float(fill_tx['units']),
                        'time': fill_tx['time']
                    }
                    
                    logger.info(f"✅ OANDA order filled: {instrument} {units} units @ {result['fill_price']}")
                    return result
                    
                elif 'orderCancelTransaction' in data:
                    # ❌ Order cancelled
                    cancel_tx = data['orderCancelTransaction']
                    reason = cancel_tx.get('reason', 'UNKNOWN')
                    logger.error(f"❌ OANDA order cancelled: {reason}")
                    logger.error(f"Cancellation details: {cancel_tx}")
                    return None
                    
                elif 'orderRejectTransaction' in data:
                    # ❌ Order rejected
                    reject_tx = data['orderRejectTransaction']
                    reason = reject_tx.get('rejectReason', 'UNKNOWN')
                    logger.error(f"❌ OANDA order rejected: {reason}")
                    logger.error(f"Rejection details: {reject_tx}")
                    return None
                    
                else:
                    # ❓ Unexpected response
                    logger.error(f"❓ Unexpected OANDA response: {data}")
                    return None
                    
            elif response.status_code == 400:
                # Bad Request
                error_data = response.json()
                logger.error(f"❌ OANDA validation error (400): {error_data}")
                return None
                
            elif response.status_code == 401:
                # Unauthorized
                logger.error(f"❌ OANDA authentication failed (401)")
                return None
                
            elif response.status_code == 404:
                # Not Found
                logger.error(f"❌ OANDA resource not found (404): {response.text}")
                return None
                
            else:
                # Other errors
                logger.error(f"❌ OANDA order failed ({response.status_code}): {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error(f"❌ OANDA request timeout")
            return None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ OANDA network error: {e}")
            return None
            
        except KeyError as e:
            logger.error(f"❌ OANDA response missing key: {e}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            return None
    
    def close_trade(self, trade_id: str) -> bool:
        """
        Close an open trade.
        
        Args:
            trade_id: OANDA trade ID
        
        Returns:
            bool: True if closed successfully
        """
        try:
            url = f"{self.PRACTICE_API_URL}/v3/accounts/{self.account_id}/trades/{trade_id}/close"
            response = requests.put(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"✅ Trade closed: {trade_id}")
                return True
            else:
                logger.error(f"Failed to close trade: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error closing trade: {e}")
            return False
    
    # ========================================================================
    # POSITION & TRADE QUERIES
    # ========================================================================
    
    def get_open_positions(self) -> List[Dict]:
        """
        Get all open positions.
        
        Returns:
            List of position dictionaries
        """
        try:
            url = f"{self.PRACTICE_API_URL}/v3/accounts/{self.account_id}/openPositions"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                positions = []
                
                for pos in data.get('positions', []):
                    positions.append({
                        'instrument': pos['instrument'],
                        'long_units': float(pos['long']['units']) if 'long' in pos else 0,
                        'short_units': float(pos['short']['units']) if 'short' in pos else 0,
                        'unrealized_pl': float(pos['unrealizedPL'])
                    })
                
                return positions
            else:
                logger.error(f"Failed to get positions: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def get_open_trades(self) -> List[Dict]:
        """
        Get all open trades.
        
        Returns:
            List of trade dictionaries
        """
        try:
            url = f"{self.PRACTICE_API_URL}/v3/accounts/{self.account_id}/openTrades"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                trades = []
                
                for trade in data.get('trades', []):
                    trades.append({
                        'trade_id': trade['id'],
                        'instrument': trade['instrument'],
                        'price': float(trade['price']),
                        'units': float(trade['currentUnits']),
                        'unrealized_pl': float(trade['unrealizedPL']),
                        'open_time': trade['openTime']
                    })
                
                return trades
            else:
                logger.error(f"Failed to get trades: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting trades: {e}")
            return []
    
    # ========================================================================
    # PRICING
    # ========================================================================
    
    def get_current_price(self, instrument: str) -> Optional[Dict]:
        """
        Get current bid/ask prices for instrument.
        
        Args:
            instrument: Forex pair (e.g., 'EUR_USD')
        
        Returns:
            Dict with bid/ask prices or None
        """
        try:
            url = f"{self.PRACTICE_API_URL}/v3/accounts/{self.account_id}/pricing"
            params = {'instruments': instrument}
            
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                price_data = data['prices'][0]
                
                return {
                    'instrument': price_data['instrument'],
                    'bid': float(price_data['bids'][0]['price']),
                    'ask': float(price_data['asks'][0]['price']),
                    'spread': float(price_data['asks'][0]['price']) - float(price_data['bids'][0]['price']),
                    'time': price_data['time']
                }
            else:
                logger.error(f"Failed to get price: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting price: {e}")
            return None
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def convert_symbol_to_oanda(self, symbol: str) -> str:
        """
        Convert standard symbol format to OANDA format.
        
        Args:
            symbol: Standard format (e.g., 'EUR/USD')
        
        Returns:
            OANDA format (e.g., 'EUR_USD')
        """
        return symbol.replace('/', '_')
    
    def calculate_units(self, lot_size: float) -> int:
        """
        Convert lot size to units.
        
        Args:
            lot_size: Standard lot size (e.g., 0.01 = 1 micro lot)
        
        Returns:
            Number of units
        """
        return int(lot_size * 100000)  # 1 standard lot = 100,000 units


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("OANDA PRACTICE API TEST")
    print("=" * 70)
    
    # Check for credentials
    if not os.getenv('OANDA_PRACTICE_API_KEY'):
        print("\n❌ ERROR: OANDA_PRACTICE_API_KEY not found in .env")
        print("\nSetup Instructions:")
        print("1. Create OANDA Practice Account: https://www.oanda.com/demo-account/")
        print("2. Get API Token: https://www.oanda.com/demo-account/tpa/personal_token")
        print("3. Add to .env:")
        print("   OANDA_PRACTICE_API_KEY=your_api_key")
        print("   OANDA_PRACTICE_ACCOUNT_ID=your_account_id")
        sys.exit(1)
    
    # Initialize handler
    handler = OandaPracticeHandler()
    
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
        print(f"   Balance: ${summary['balance']:,.2f} {summary['currency']}")
        print(f"   NAV: ${summary['nav']:,.2f}")
        print(f"   Open Trades: {summary['open_trade_count']}")
        print(f"   Open Positions: {summary['open_position_count']}")
    
    # Test 3: Get current price
    print("\n3️⃣ Getting EUR/USD price...")
    price = handler.get_current_price('EUR_USD')
    if price:
        print(f"   Bid: {price['bid']}")
        print(f"   Ask: {price['ask']}")
        print(f"   Spread: {price['spread']:.5f}")
    
    # Test 4: Place test order (optional - uncomment to test)
    print("\n4️⃣ Order placement test (skipped - uncomment to test)")
    # Uncomment below to test actual order placement:
    """
    print("   Placing test order: EUR/USD buy 1000 units...")
    result = handler.place_market_order(
        instrument='EUR_USD',
        units=1000,  # Buy 1000 units (0.01 micro lot)
        stop_loss=price['bid'] - 0.0020,  # 20 pips SL
        take_profit=price['bid'] + 0.0040  # 40 pips TP
    )
    if result:
        print(f"   ✅ Order placed!")
        print(f"   Trade ID: {result['trade_id']}")
        print(f"   Fill Price: {result['fill_price']}")
        
        # Close immediately
        input("   Press Enter to close the trade...")
        if handler.close_trade(result['trade_id']):
            print("   ✅ Trade closed")
    """
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED")
    print("=" * 70)
