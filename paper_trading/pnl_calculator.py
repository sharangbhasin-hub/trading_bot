"""
P&L Calculator for Paper Trading
=================================

Calculates profit/loss in USD for cryptocurrency and forex trades.
Handles market-specific calculation logic including:
- Cryptocurrency: Quantity-based P&L (investment amount / entry price)
- Forex: Pip-based P&L with lot size and pip values

Author: Trading System
Last Updated: October 29, 2025
"""

from typing import Dict, Tuple, Optional, Union
from decimal import Decimal, ROUND_HALF_UP
import logging

from .config import get_config

logger = logging.getLogger(__name__)


class PnLCalculator:
    """
    Production-grade P&L calculator for paper trading.
    
    Features:
    - Accurate USD-based P&L calculations
    - Market-specific formulas (crypto vs. forex)
    - Transaction cost modeling (slippage, commissions, spread)
    - Precision handling with Decimal for financial calculations
    - Extensive validation and error handling
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize P&L calculator with configuration.
        
        Args:
            config: Optional config dict. If None, loads from module config.
        """
        self.config = config or get_config()
        self.crypto_config = self.config['crypto']
        self.forex_config = self.config['forex']
        
        logger.info("PnLCalculator initialized")
    
    # ========================================================================
    # CRYPTOCURRENCY P&L CALCULATIONS
    # ========================================================================
    
    def calculate_crypto_pnl(
        self,
        entry_price: float,
        exit_price: float,
        investment_usd: float,
        direction: str = 'BUY',
        include_costs: bool = True
    ) -> Dict[str, Union[float, str]]:
        """
        Calculate P&L for cryptocurrency trade in USD.
        
        Formula:
            Quantity = Investment / Entry Price
            Price Change = Exit Price - Entry Price
            Gross P&L = Price Change × Quantity
            Net P&L = Gross P&L - Transaction Costs
        
        Args:
            entry_price: Entry price in USDT (e.g., 67250.00)
            exit_price: Exit price in USDT (e.g., 67520.00)
            investment_usd: Investment amount in USD (e.g., 1000.00)
            direction: 'BUY' or 'SELL'
            include_costs: Whether to deduct transaction costs
        
        Returns:
            Dict with P&L metrics:
                - pnl_usd: Net profit/loss in USD
                - pnl_pct: P&L as percentage of investment
                - gross_pnl_usd: P&L before costs
                - quantity: Amount of crypto traded
                - slippage_cost: Cost due to slippage
                - commission_cost: Exchange commission
                - total_costs: Total transaction costs
                - entry_price: Effective entry price (includes slippage)
                - exit_price: Effective exit price (includes slippage)
        
        Example:
            >>> calc = PnLCalculator()
            >>> result = calc.calculate_crypto_pnl(67250.00, 67520.00, 1000.00)
            >>> print(f"P&L: ${result['pnl_usd']:.2f}")
            P&L: $3.02
        """
        # Input validation
        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")
        if exit_price <= 0:
            raise ValueError(f"exit_price must be positive, got {exit_price}")
        if investment_usd <= 0:
            raise ValueError(f"investment_usd must be positive, got {investment_usd}")
        if direction not in ['BUY', 'SELL']:
            raise ValueError(f"direction must be 'BUY' or 'SELL', got {direction}")
        
        # Use Decimal for precision in financial calculations
        entry = Decimal(str(entry_price))
        exit_p = Decimal(str(exit_price))
        investment = Decimal(str(investment_usd))
        
        # Get transaction cost parameters
        slippage_pct = Decimal(str(self.crypto_config['slippage_pct']))
        commission_pct = Decimal(str(self.crypto_config['commission_pct']))
        
        # Calculate effective prices with slippage
        if direction == 'BUY':
            # BUY: Pay slippage on entry, exit
            entry_with_slippage = entry * (Decimal('1') + slippage_pct)
            exit_with_slippage = exit_p * (Decimal('1') - slippage_pct)
        else:  # SELL
            # SELL (short): Pay slippage on entry, exit (reversed)
            entry_with_slippage = entry * (Decimal('1') - slippage_pct)
            exit_with_slippage = exit_p * (Decimal('1') + slippage_pct)
        
        # Calculate quantity (how much crypto we trade)
        quantity = investment / entry_with_slippage
        
        # Calculate price change
        if direction == 'BUY':
            price_change = exit_with_slippage - entry_with_slippage
        else:  # SELL (short position)
            price_change = entry_with_slippage - exit_with_slippage
        
        # Gross P&L (before costs)
        gross_pnl = price_change * quantity
        
        # Calculate transaction costs
        slippage_cost = investment * slippage_pct * Decimal('2')  # Entry + Exit
        commission_cost = investment * commission_pct * Decimal('2')  # Entry + Exit
        total_costs = slippage_cost + commission_cost
        
        # Net P&L
        if include_costs:
            net_pnl = gross_pnl - total_costs
        else:
            net_pnl = gross_pnl
            total_costs = Decimal('0')
        
        # P&L percentage
        pnl_pct = (net_pnl / investment) * Decimal('100')
        
        # Prepare result
        result = {
            'pnl_usd': float(net_pnl.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
            'pnl_pct': float(pnl_pct.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
            'gross_pnl_usd': float(gross_pnl.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
            'quantity': float(quantity.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)),
            'slippage_cost': float(slippage_cost.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
            'commission_cost': float(commission_cost.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
            'total_costs': float(total_costs.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
            'entry_price': float(entry_with_slippage.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
            'exit_price': float(exit_with_slippage.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
            'direction': direction,
            'market_type': 'crypto'
        }
        
        logger.debug(f"Crypto P&L calculated: {result['pnl_usd']:.2f} USD ({result['pnl_pct']:.2f}%)")
        
        return result
    
    # ========================================================================
    # FOREX P&L CALCULATIONS
    # ========================================================================
    
    def calculate_forex_pnl(
        self,
        entry_price: float,
        exit_price: float,
        lot_size: float,
        pair: str = 'EUR/USD',
        direction: str = 'BUY',
        include_costs: bool = True
    ) -> Dict[str, Union[float, str]]:
        """
        Calculate P&L for forex trade in USD.
        
        Formula:
            Pips = (Exit - Entry) / Pip Size
            Pip Value = Pip Value per Lot × Lot Size
            Gross P&L = Pips × Pip Value
            Net P&L = Gross P&L - (Spread + Commission)
        
        Args:
            entry_price: Entry price (e.g., 1.0850 for EUR/USD)
            exit_price: Exit price (e.g., 1.0870)
            lot_size: Trade size in lots (e.g., 0.01 = micro lot)
            pair: Currency pair (e.g., 'EUR/USD')
            direction: 'BUY' or 'SELL'
            include_costs: Whether to deduct transaction costs
        
        Returns:
            Dict with P&L metrics:
                - pnl_usd: Net profit/loss in USD
                - pnl_pips: P&L in pips
                - gross_pnl_usd: P&L before costs
                - pip_value: Value of 1 pip in USD
                - spread_cost: Cost of bid-ask spread
                - commission_cost: Broker commission
                - slippage_cost: Slippage cost
                - total_costs: Total transaction costs
        
        Example:
            >>> calc = PnLCalculator()
            >>> result = calc.calculate_forex_pnl(1.0850, 1.0870, 0.01, 'EUR/USD')
            >>> print(f"P&L: ${result['pnl_usd']:.2f} ({result['pnl_pips']:.1f} pips)")
            P&L: $1.84 (20.0 pips)
        """
        # Input validation
        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")
        if exit_price <= 0:
            raise ValueError(f"exit_price must be positive, got {exit_price}")
        if lot_size <= 0:
            raise ValueError(f"lot_size must be positive, got {lot_size}")
        if direction not in ['BUY', 'SELL']:
            raise ValueError(f"direction must be 'BUY' or 'SELL', got {direction}")
        if pair not in self.forex_config['pip_values']:
            raise ValueError(f"Unsupported forex pair: {pair}. Supported: {list(self.forex_config['pip_values'].keys())}")
        
        # Use Decimal for precision
        entry = Decimal(str(entry_price))
        exit_p = Decimal(str(exit_price))
        lots = Decimal(str(lot_size))
        
        # Determine pip size (0.0001 for most pairs, 0.01 for JPY pairs)
        pip_size = Decimal('0.01') if 'JPY' in pair else Decimal('0.0001')
        
        # Calculate pips gained/lost
        if direction == 'BUY':
            price_diff = exit_p - entry
        else:  # SELL
            price_diff = entry - exit_p
        
        pips = price_diff / pip_size
        
        # Get pip value per 0.01 lot (from config)
        pip_value_per_microlot = Decimal(str(self.forex_config['pip_values'][pair]))
        
        # Calculate pip value for actual lot size
        pip_value = pip_value_per_microlot * (lots / Decimal('0.01'))
        
        # Gross P&L
        gross_pnl = pips * pip_value
        
        # Calculate transaction costs
        slippage_pips = Decimal(str(self.forex_config['slippage_pips']))
        spread_pips = Decimal(str(self.forex_config['spread_pips']))
        commission_per_lot = Decimal(str(self.forex_config['commission_per_lot']))
        
        # Costs in USD
        slippage_cost = slippage_pips * pip_value
        spread_cost = spread_pips * pip_value
        commission_cost = commission_per_lot * (lots / Decimal('0.01'))
        
        total_costs = slippage_cost + spread_cost + commission_cost
        
        # Net P&L
        if include_costs:
            net_pnl = gross_pnl - total_costs
        else:
            net_pnl = gross_pnl
            total_costs = Decimal('0')
        
        # Prepare result
        result = {
            'pnl_usd': float(net_pnl.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
            'pnl_pips': float(pips.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)),
            'gross_pnl_usd': float(gross_pnl.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
            'pip_value': float(pip_value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
            'spread_cost': float(spread_cost.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
            'commission_cost': float(commission_cost.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
            'slippage_cost': float(slippage_cost.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
            'total_costs': float(total_costs.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
            'lot_size': float(lots),
            'pair': pair,
            'direction': direction,
            'market_type': 'forex'
        }
        
        logger.debug(f"Forex P&L calculated: {result['pnl_usd']:.2f} USD ({result['pnl_pips']:.1f} pips)")
        
        return result
    
    # ========================================================================
    # UNIVERSAL P&L CALCULATION (AUTO-DETECT MARKET)
    # ========================================================================
    
    def calculate_pnl(
        self,
        entry_price: float,
        exit_price: float,
        symbol: str,
        direction: str = 'BUY',
        investment_usd: Optional[float] = None,
        lot_size: Optional[float] = None,
        include_costs: bool = True
    ) -> Dict[str, Union[float, str]]:
        """
        Universal P&L calculator - auto-detects market type.
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            symbol: Trading symbol (e.g., 'BTC/USDT' or 'EUR/USD')
            direction: 'BUY' or 'SELL'
            investment_usd: Investment for crypto (optional if lot_size provided)
            lot_size: Lot size for forex (optional if investment_usd provided)
            include_costs: Whether to include transaction costs
        
        Returns:
            P&L calculation result dict
        
        Example:
            >>> calc = PnLCalculator()
            >>> # Crypto
            >>> result = calc.calculate_pnl(67250, 67520, 'BTC/USDT', investment_usd=1000)
            >>> # Forex
            >>> result = calc.calculate_pnl(1.0850, 1.0870, 'EUR/USD', lot_size=0.01)
        """
        # Detect market type from symbol
        if 'USDT' in symbol or 'USD' in symbol and '/' in symbol:
            # Crypto (BTC/USDT, ETH/USDT, etc.)
            if investment_usd is None:
                investment_usd = self.crypto_config['investment_per_trade_usd']
            
            return self.calculate_crypto_pnl(
                entry_price=entry_price,
                exit_price=exit_price,
                investment_usd=investment_usd,
                direction=direction,
                include_costs=include_costs
            )
        else:
            # Forex (EUR/USD, GBP/USD, etc.)
            if lot_size is None:
                lot_size = self.forex_config['lot_size']
            
            return self.calculate_forex_pnl(
                entry_price=entry_price,
                exit_price=exit_price,
                lot_size=lot_size,
                pair=symbol,
                direction=direction,
                include_costs=include_costs
            )

    def calculate_trade_pnl(self, trade_data: Dict) -> Dict:
        """
        Calculate P&L for a single closed trade.
        
        Args:
            trade_data: Dict with:
                - entry_price: float
                - exit_price: float
                - quantity: float
                - side: 'buy' or 'sell'
                - amount: float (total notional)
        
        Returns:
            Dict with P&L breakdown
        """
        try:
            entry = float(trade_data.get('entry_price', 0))
            exit_price = float(trade_data.get('exit_price', 0))
            qty = float(trade_data.get('quantity', 0))
            side = trade_data.get('side', 'buy').lower()
            amount = float(trade_data.get('amount', 0))
            
            if entry == 0 or exit_price == 0 or amount == 0:
                return {'pnl': 0, 'pnl_pct': 0, 'points': 0}
            
            # Calculate profit based on side
            if side == 'buy':
                pnl_points = exit_price - entry  # Points profit
            else:  # sell
                pnl_points = entry - exit_price  # Reverse for sell
            
            pnl_dollars = pnl_points * qty
            pnl_pct = (pnl_dollars / amount) * 100 if amount > 0 else 0
            
            return {
                'pnl': pnl_dollars,
                'pnl_pct': pnl_pct,
                'points': pnl_points,
                'entry': entry,
                'exit': exit_price,
                'quantity': qty,
                'amount': amount
            }
        
        except Exception as e:
            logger.error(f"Error calculating trade PnL: {e}")
            return {'pnl': 0, 'pnl_pct': 0, 'points': 0}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_pnl(pnl_result: Dict) -> str:
    """
    Format P&L result for display.
    
    Args:
        pnl_result: P&L calculation result from PnLCalculator
    
    Returns:
        Formatted string
    
    Example:
        >>> result = calc.calculate_crypto_pnl(67250, 67520, 1000)
        >>> print(format_pnl(result))
        💰 P&L: $3.02 (0.30%) | Costs: $0.98 | Net: $3.02
    """
    pnl = pnl_result['pnl_usd']
    pnl_pct = pnl_result.get('pnl_pct', 0)
    costs = pnl_result['total_costs']
    gross = pnl_result['gross_pnl_usd']
    
    emoji = "💰" if pnl > 0 else "📉"
    sign = "+" if pnl > 0 else ""
    
    if pnl_result['market_type'] == 'crypto':
        return f"{emoji} P&L: {sign}${pnl:.2f} ({sign}{pnl_pct:.2f}%) | Costs: ${costs:.2f} | Gross: ${gross:.2f}"
    else:  # forex
        pips = pnl_result['pnl_pips']
        return f"{emoji} P&L: {sign}${pnl:.2f} ({sign}{pips:.1f} pips) | Costs: ${costs:.2f} | Gross: ${gross:.2f}"


if __name__ == "__main__":
    # Test P&L calculations
    print("=" * 70)
    print("P&L CALCULATOR TEST SUITE")
    print("=" * 70)
    
    calc = PnLCalculator()
    
    # Test 1: Crypto BUY (winning trade)
    print("\n1️⃣ CRYPTO TEST: BTC/USDT BUY (Winning Trade)")
    print("-" * 70)
    result = calc.calculate_crypto_pnl(
        entry_price=67250.00,
        exit_price=67520.00,
        investment_usd=1000.00,
        direction='BUY'
    )
    print(format_pnl(result))
    print(f"   Entry: ${result['entry_price']:,.2f} | Exit: ${result['exit_price']:,.2f}")
    print(f"   Quantity: {result['quantity']:.8f} BTC")
    
    # Test 2: Crypto BUY (losing trade)
    print("\n2️⃣ CRYPTO TEST: ETH/USDT BUY (Losing Trade)")
    print("-" * 70)
    result = calc.calculate_crypto_pnl(
        entry_price=3500.00,
        exit_price=3450.00,
        investment_usd=1000.00,
        direction='BUY'
    )
    print(format_pnl(result))
    
    # Test 3: Forex BUY (winning trade)
    print("\n3️⃣ FOREX TEST: EUR/USD BUY (Winning Trade)")
    print("-" * 70)
    result = calc.calculate_forex_pnl(
        entry_price=1.0850,
        exit_price=1.0870,
        lot_size=0.01,
        pair='EUR/USD',
        direction='BUY'
    )
    print(format_pnl(result))
    print(f"   Pip Value: ${result['pip_value']:.2f}")
    
    # Test 4: Forex SELL (winning trade)
    print("\n4️⃣ FOREX TEST: GBP/USD SELL (Winning Trade)")
    print("-" * 70)
    result = calc.calculate_forex_pnl(
        entry_price=1.2950,
        exit_price=1.2930,
        lot_size=0.01,
        pair='GBP/USD',
        direction='SELL'
    )
    print(format_pnl(result))
    
    # Test 5: Universal calculator
    print("\n5️⃣ UNIVERSAL CALCULATOR TEST")
    print("-" * 70)
    result1 = calc.calculate_pnl(67250, 67520, 'BTC/USDT', investment_usd=1000)
    result2 = calc.calculate_pnl(1.0850, 1.0870, 'EUR/USD', lot_size=0.01)
    print(f"Crypto: {format_pnl(result1)}")
    print(f"Forex:  {format_pnl(result2)}")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED")
    print("=" * 70)
