"""
Trade Simulator
Simulates trade execution, tracks open positions, calculates P&L
"""
import pandas as pd
from datetime import datetime
import logging

from backtesting.config import BacktestConfig

logger = logging.getLogger(__name__)


class Trade:
    """Represents a single trade"""
    
    def __init__(self, signal, entry_timestamp):
        """
        Initialize trade from signal
        
        Args:
            signal: Signal dict from SignalRecorder
            entry_timestamp: DateTime of entry
        """
        self.trade_id = signal['signal_id']
        self.signal = signal
        self.entry_timestamp = entry_timestamp
        self.entry_price = signal['entry_price']
        self.stop_loss = signal['stop_loss']
        self.target = signal['target']
        self.signal_type = signal['signal_type']
        self.strategy_name = signal['strategy_name']
        self.confidence = signal['confidence']
        
        # Trade state
        self.status = 'OPEN'
        self.exit_timestamp = None
        self.exit_price = None
        self.exit_reason = None
        self.pnl = 0
        self.pnl_percent = 0
        
        # Excursion tracking
        self.max_favorable_excursion = 0
        self.max_adverse_excursion = 0
        
        # Add slippage to entry
        config = BacktestConfig()
        slippage = self.entry_price * (config.SLIPPAGE_PERCENT / 100)
        
        if self.signal_type == 'CALL':
            self.entry_price += slippage  # Pay higher for CALL
        else:
            self.entry_price -= slippage  # Get lower for PUT
    
    def update_excursion(self, current_price):
        """
        Update max favorable and adverse excursion
        
        Args:
            current_price: Current market price
        """
        if self.signal_type == 'CALL':
            # For CALL: profit when price goes up
            excursion = current_price - self.entry_price
        else:
            # For PUT: profit when price goes down
            excursion = self.entry_price - current_price
        
        # Update MFE (best price reached)
        if excursion > self.max_favorable_excursion:
            self.max_favorable_excursion = excursion
        
        # Update MAE (worst price reached)
        if excursion < self.max_adverse_excursion:
            self.max_adverse_excursion = excursion
    
    def check_exit(self, candle):
        """
        Check if trade should exit based on current candle
        
        Args:
            candle: Current OHLC candle (Series)
        
        Returns:
            Bool: True if exit condition met
        """
        if self.status != 'OPEN':
            return False
        
        # Check stop loss
        if self.signal_type == 'CALL':
            if candle['low'] <= self.stop_loss:
                self.exit_price = self.stop_loss
                self.exit_reason = 'STOP_LOSS'
                return True
        else:  # PUT
            if candle['high'] >= self.stop_loss:
                self.exit_price = self.stop_loss
                self.exit_reason = 'STOP_LOSS'
                return True
        
        # Check target
        if self.signal_type == 'CALL':
            if candle['high'] >= self.target:
                self.exit_price = self.target
                self.exit_reason = 'TARGET'
                return True
        else:  # PUT
            if candle['low'] <= self.target:
                self.exit_price = self.target
                self.exit_reason = 'TARGET'
                return True
        
        # Update excursion
        self.update_excursion(candle['close'])
        
        return False
    
    def close_at_market(self, exit_price, exit_timestamp, reason='EOD'):
        """
        Close trade at market price
        
        Args:
            exit_price: Exit price
            exit_timestamp: DateTime of exit
            reason: Reason for exit (EOD, TIMEOUT, etc.)
        """
        config = BacktestConfig()
        
        # Add slippage
        slippage = exit_price * (config.SLIPPAGE_PERCENT / 100)
        
        if self.signal_type == 'CALL':
            self.exit_price = exit_price - slippage  # Get lower when selling
        else:
            self.exit_price = exit_price + slippage  # Pay higher when covering
        
        self.exit_timestamp = exit_timestamp
        self.exit_reason = reason
        self.status = 'CLOSED'
        
        self._calculate_pnl()
    
    def close(self, exit_timestamp):
        """
        Close trade (exit price already set by check_exit)
        
        Args:
            exit_timestamp: DateTime of exit
        """
        self.exit_timestamp = exit_timestamp
        self.status = 'CLOSED'
        
        # Add slippage to exit
        config = BacktestConfig()
        slippage = self.exit_price * (config.SLIPPAGE_PERCENT / 100)
        
        if self.exit_reason == 'TARGET':
            # Exiting at profit, pay slippage
            if self.signal_type == 'CALL':
                self.exit_price -= slippage
            else:
                self.exit_price += slippage
        
        self._calculate_pnl()
    
    def _calculate_pnl(self):
        """Calculate P&L for closed trade"""
        if self.signal_type == 'CALL':
            self.pnl = self.exit_price - self.entry_price
        else:  # PUT
            self.pnl = self.entry_price - self.exit_price
        
        self.pnl_percent = (self.pnl / self.entry_price) * 100
    
    def to_dict(self):
        """Convert trade to dictionary"""
        holding_period = None
        if self.exit_timestamp:
            holding_period = (self.exit_timestamp - self.entry_timestamp).total_seconds() / 60  # minutes
        
        return {
            'trade_id': self.trade_id,
            'entry_date': self.entry_timestamp.strftime('%Y-%m-%d'),
            'entry_time': self.entry_timestamp.strftime('%H:%M'),
            'exit_date': self.exit_timestamp.strftime('%Y-%m-%d') if self.exit_timestamp else None,
            'exit_time': self.exit_timestamp.strftime('%H:%M') if self.exit_timestamp else None,
            'strategy_name': self.strategy_name,
            'signal_type': self.signal_type,
            'confidence': self.confidence,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'stop_loss': self.stop_loss,
            'target': self.target,
            'pnl': self.pnl,
            'pnl_percent': self.pnl_percent,
            'exit_reason': self.exit_reason,
            'holding_period_minutes': holding_period,
            'max_favorable_excursion': self.max_favorable_excursion,
            'max_adverse_excursion': self.max_adverse_excursion,
            'status': self.status
        }


class TradeSimulator:
    """
    Manages all trades during backtesting
    """
    
    def __init__(self):
        """Initialize trade simulator"""
        self.open_trades = []
        self.closed_trades = []
        self.config = BacktestConfig()
        
    def open_trade(self, signal, entry_timestamp):
        """
        Open a new trade
        
        Args:
            signal: Signal dict
            entry_timestamp: DateTime of entry
        
        Returns:
            Trade object
        """
        trade = Trade(signal, entry_timestamp)
        self.open_trades.append(trade)
        
        logger.info(f"Opened trade #{trade.trade_id}: {trade.strategy_name} {trade.signal_type} @ {trade.entry_price}")
        
        return trade
    
    def update_trades(self, current_candle, current_timestamp):
        """
        Update all open trades with current candle
        Check for exit conditions
        
        Args:
            current_candle: Current 5-minute candle (Series)
            current_timestamp: DateTime of current candle
        """
        trades_to_close = []
        
        for trade in self.open_trades:
            if trade.check_exit(current_candle):
                trade.close(current_timestamp)
                trades_to_close.append(trade)
        
        # Move closed trades
        for trade in trades_to_close:
            self.open_trades.remove(trade)
            self.closed_trades.append(trade)
            
            outcome = 'WIN' if trade.pnl > 0 else 'LOSS'
            logger.info(f"Closed trade #{trade.trade_id}: {outcome} {trade.pnl:.2f} pts ({trade.exit_reason})")
    
    def close_all_open_trades(self, exit_price, exit_timestamp, reason='EOD'):
        """
        Close all open trades at market price
        Called at end of day
        
        Args:
            exit_price: Current market price
            exit_timestamp: DateTime
            reason: Reason for closure
        """
        for trade in self.open_trades[:]:  # Copy list to avoid modification during iteration
            trade.close_at_market(exit_price, exit_timestamp, reason)
            self.open_trades.remove(trade)
            self.closed_trades.append(trade)
            
            outcome = 'WIN' if trade.pnl > 0 else 'LOSS'
            logger.info(f"EOD closed trade #{trade.trade_id}: {outcome} {trade.pnl:.2f} pts")
    
    def get_closed_trades_df(self):
        """Get DataFrame of all closed trades"""
        if not self.closed_trades:
            return pd.DataFrame()
        
        trades_data = [trade.to_dict() for trade in self.closed_trades]
        df = pd.DataFrame(trades_data)
        
        # Add outcome column
        df['outcome'] = df['pnl'].apply(lambda x: 'WIN' if x > 0 else 'LOSS')
        
        return df
    
    def get_summary_stats(self):
        """Get summary statistics of all trades"""
        df = self.get_closed_trades_df()
        
        if df.empty:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0
            }
        
        wins = df[df['pnl'] > 0]
        losses = df[df['pnl'] <= 0]
        
        stats = {
            'total_trades': len(df),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / len(df) * 100 if len(df) > 0 else 0,
            'total_pnl': df['pnl'].sum(),
            'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
            'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0,
            'largest_win': wins['pnl'].max() if len(wins) > 0 else 0,
            'largest_loss': losses['pnl'].min() if len(losses) > 0 else 0,
            'profit_factor': abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else 0,
            'avg_holding_period': df['holding_period_minutes'].mean(),
            'avg_rr_ratio': df.apply(lambda x: abs((x['target'] - x['entry_price']) / (x['entry_price'] - x['stop_loss'])), axis=1).mean()
        }
        
        # Calculate max drawdown
        df_sorted = df.sort_values(['entry_date', 'entry_time'])
        df_sorted['cumulative_pnl'] = df_sorted['pnl'].cumsum()
        df_sorted['running_max'] = df_sorted['cumulative_pnl'].cummax()
        df_sorted['drawdown'] = df_sorted['running_max'] - df_sorted['cumulative_pnl']
        
        stats['max_drawdown'] = df_sorted['drawdown'].max()
        
        return stats
    
    def reset(self):
        """Reset simulator"""
        self.open_trades = []
        self.closed_trades = []
        logger.info("Trade simulator reset")
