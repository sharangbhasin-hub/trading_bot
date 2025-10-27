"""
Account Risk Manager
Manages account-level risk limits and capital protection
Works for both backtesting and live trading
"""
import logging
from datetime import datetime, date
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class AccountRiskManager:
    """
    Manages account-level risk limits
    This is SEPARATE from trade-level risk (stop/target calculation)
    """
    
    def __init__(self, initial_capital: float = 1000000, mode: str = 'backtest'):
        """
        Initialize account risk manager
        
        Args:
            initial_capital: Starting capital in rupees
            mode: 'backtest' or 'live'
        """
        self.mode = mode
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # ✅ ACCOUNT-LEVEL RISK LIMITS
        self.limits = {
            # Daily limits
            'max_daily_loss_points': 10000,      # Stop after -2000 points/day
            'max_daily_loss_percent': 10.0,      # Or -2% of capital
            'max_trades_per_day': 10,            # Max 5 trades/day
            
            # Position limits
            'max_open_positions': 3,            # Max 2 concurrent trades
            'max_capital_per_trade_percent': 15.0,  # Max 10% capital per trade
            
            # Drawdown protection
            'max_drawdown_percent': 20.0,       # Pause at 10% drawdown
            'max_drawdown_points': 20000,        # Or 5000 points drawdown
            
            # Recovery rules
            'pause_after_consecutive_losses': 5,  # Pause after 3 losses in a row
            'resume_after_minutes': 60,          # Resume after 60 min pause
        }
        
        # Tracking variables
        self.daily_stats = {}
        self.current_date = None
        self.daily_pnl = 0
        self.daily_trade_count = 0
        self.open_positions_count = 0
        self.peak_capital = initial_capital
        self.current_drawdown = 0
        self.consecutive_losses = 0
        
        # Pause mechanism
        self.is_paused = False
        self.pause_reason = None
        self.pause_timestamp = None
        
        logger.info(f"AccountRiskManager initialized ({mode} mode)")
        logger.info(f"Capital: ₹{initial_capital:,.0f}, Daily loss limit: {self.limits['max_daily_loss_points']} pts")
    
    def new_trading_day(self, trading_date: date):
        """
        Reset daily stats for new trading day
        
        Args:
            trading_date: Date object for the trading day
        """
        # Save yesterday's stats
        if self.current_date and self.current_date != trading_date:
            self.daily_stats[self.current_date] = {
                'pnl': self.daily_pnl,
                'trades': self.daily_trade_count,
                'capital_end': self.current_capital,
                'was_paused': self.is_paused
            }
        
        # Reset for new day
        self.current_date = trading_date
        self.daily_pnl = 0
        self.daily_trade_count = 0
        self.is_paused = False
        self.pause_reason = None
        self.consecutive_losses = 0
        
        logger.info(f"📅 New trading day: {trading_date}, Capital: ₹{self.current_capital:,.0f}")
    
    def can_take_trade(self, signal: Dict) -> Tuple[bool, str]:
        """
        Check if trade can be taken (GATE KEEPER)
        
        Args:
            signal: Signal dict with entry, stop, target
        
        Returns:
            Tuple of (approved: bool, reason: str)
        """
        # 1. Check if trading is paused
        if self.is_paused:
            return False, f"❌ Trading paused: {self.pause_reason}"
        
        # 2. Check daily loss limit (points)
        if self.daily_pnl <= -self.limits['max_daily_loss_points']:
            self._pause_trading(f"Daily loss limit hit: {self.daily_pnl:.0f} pts")
            return False, self.pause_reason
        
        # 3. Check daily loss limit (percentage)
        daily_loss_pct = (self.daily_pnl / self.initial_capital) * 100
        if daily_loss_pct <= -self.limits['max_daily_loss_percent']:
            self._pause_trading(f"Daily loss % limit hit: {daily_loss_pct:.2f}%")
            return False, self.pause_reason
        
        # 4. Check max trades per day
        if self.daily_trade_count >= self.limits['max_trades_per_day']:
            return False, f"❌ Max trades/day reached: {self.daily_trade_count}/{self.limits['max_trades_per_day']}"
        
        # 5. Check max open positions
        if self.open_positions_count >= self.limits['max_open_positions']:
            return False, f"❌ Max open positions: {self.open_positions_count}/{self.limits['max_open_positions']}"
        
        # 6. Check drawdown limit
        drawdown_pct = (self.current_drawdown / self.peak_capital) * 100
        if drawdown_pct >= self.limits['max_drawdown_percent']:
            self._pause_trading(f"Max drawdown hit: {drawdown_pct:.2f}%")
            return False, self.pause_reason
        
        if self.current_drawdown >= self.limits['max_drawdown_points']:
            self._pause_trading(f"Max drawdown hit: {self.current_drawdown:.0f} pts")
            return False, self.pause_reason
        
        # 7. Check consecutive losses
        if self.consecutive_losses >= self.limits['pause_after_consecutive_losses']:
            self._pause_trading(f"{self.consecutive_losses} consecutive losses")
            return False, self.pause_reason
        
        # All checks passed
        return True, "✅ Trade approved"
    
    def on_trade_entered(self, signal: Dict):
        """
        Call when trade is opened
        
        Args:
            signal: Signal dict
        """
        self.daily_trade_count += 1
        self.open_positions_count += 1
        
        logger.info(f"📈 Trade #{self.daily_trade_count} opened, Open: {self.open_positions_count}")
    
    def on_trade_closed(self, pnl: float, is_win: bool):
        """
        Call when trade is closed
        
        Args:
            pnl: Net P&L (after transaction costs)
            is_win: True if profitable trade
        """
        self.open_positions_count = max(0, self.open_positions_count - 1)
        self.daily_pnl += pnl
        self.current_capital += pnl
        
        # Update consecutive losses
        if is_win:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
        
        # Update peak and drawdown
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
            self.current_drawdown = 0
        else:
            self.current_drawdown = self.peak_capital - self.current_capital
        
        # Log status
        outcome = "WIN ✅" if is_win else "LOSS ❌"
        logger.info(
            f"📉 Trade closed: {outcome}, P&L: ₹{pnl:.0f}, "
            f"Daily: ₹{self.daily_pnl:.0f}, Capital: ₹{self.current_capital:,.0f}"
        )
        
        # Check if we need to pause
        if self.consecutive_losses >= self.limits['pause_after_consecutive_losses']:
            self._pause_trading(f"{self.consecutive_losses} consecutive losses")
    
    def _pause_trading(self, reason: str):
        """Pause trading with reason"""
        self.is_paused = True
        self.pause_reason = reason
        self.pause_timestamp = datetime.now()
        
        logger.warning(f"🛑 TRADING PAUSED: {reason}")
    
    def get_status(self) -> Dict:
        """Get current status"""
        return {
            'mode': self.mode,
            'current_capital': round(self.current_capital, 2),
            'initial_capital': self.initial_capital,
            'total_return_pct': round(((self.current_capital - self.initial_capital) / self.initial_capital) * 100, 2),
            'daily_pnl': round(self.daily_pnl, 2),
            'daily_trades': self.daily_trade_count,
            'open_positions': self.open_positions_count,
            'drawdown': round(self.current_drawdown, 2),
            'drawdown_pct': round((self.current_drawdown / self.peak_capital) * 100, 2) if self.peak_capital > 0 else 0,
            'consecutive_losses': self.consecutive_losses,
            'is_paused': self.is_paused,
            'pause_reason': self.pause_reason,
            'current_date': str(self.current_date) if self.current_date else None
        }
    
    def get_daily_summary(self) -> Dict:
        """Get summary of all trading days"""
        return self.daily_stats
    
    def reset(self):
        """Reset to initial state"""
        self.current_capital = self.initial_capital
        self.daily_stats = {}
        self.current_date = None
        self.daily_pnl = 0
        self.daily_trade_count = 0
        self.open_positions_count = 0
        self.peak_capital = self.initial_capital
        self.current_drawdown = 0
        self.consecutive_losses = 0
        self.is_paused = False
        self.pause_reason = None
        
        logger.info("AccountRiskManager reset")
