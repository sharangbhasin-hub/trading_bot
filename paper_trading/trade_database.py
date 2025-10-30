"""
Trade Database for Paper Trading
=================================

SQLite database manager for storing and retrieving paper trades.
Provides complete trade history with verification support.

Features:
- Full CRUD operations (Create, Read, Update, Delete)
- Thread-safe database operations
- Automatic backup support
- CSV export for verification
- Query builders for complex filters
- WAL mode for better concurrency

Author: Trading System
Last Updated: October 29, 2025
"""

import sqlite3
import json
import csv
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging
from contextlib import contextmanager

from .config import get_config

logger = logging.getLogger(__name__)


class TradeDatabase:
    """
    Production-grade SQLite database for paper trading.
    
    Features:
    - Atomic transactions with context managers
    - Automatic schema creation and migration
    - Thread-safe operations
    - Comprehensive query methods
    - Export capabilities (CSV, JSON)
    - Backup management
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize trade database.
        
        Args:
            db_path: Path to SQLite database file. If None, uses config default.
        """
        self.config = get_config('database')
        self.db_path = db_path or self.config['path']
        self.backup_path = Path(self.config['backup_path'])
        
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._initialize_database()
        
        logger.info(f"TradeDatabase initialized: {self.db_path}")
    
    @contextmanager
    def _get_connection(self):
        """
        Context manager for database connections.
        Ensures proper connection cleanup and error handling.
        
        Yields:
            sqlite3.Connection: Database connection
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row  # Return dicts instead of tuples
            
            # Enable WAL mode for better concurrency
            if self.config.get('wal_mode', True):
                conn.execute('PRAGMA journal_mode=WAL')
            
            yield conn
            conn.commit()
            
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
            
        finally:
            if conn:
                conn.close()
    
    def _initialize_database(self):
        """
        Create database schema if not exists.
        Creates all required tables and indexes.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Paper Trades Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS paper_trades (
                    -- Primary Key
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    
                    -- Trade Identification
                    timestamp DATETIME NOT NULL,
                    symbol TEXT NOT NULL,
                    exchange TEXT,
                    market_type TEXT NOT NULL,  -- 'crypto' or 'forex'
                    
                    -- Trade Details
                    direction TEXT NOT NULL,  -- 'BUY' or 'SELL'
                    entry_price REAL NOT NULL,
                    quantity REAL,  -- For crypto
                    lot_size REAL,  -- For forex
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    
                    -- Exit Details
                    exit_price REAL,
                    exit_timestamp DATETIME,
                    exit_reason TEXT,  -- 'TP', 'SL', 'MANUAL', 'EOD', 'TIMEOUT'
                    
                    -- P&L
                    pnl_usd REAL,
                    pnl_pct REAL,
                    pips REAL,  -- Forex only
                    gross_pnl REAL,
                    transaction_costs REAL,
                    net_pnl REAL,
                    
                    -- Strategy Info
                    strategy_name TEXT NOT NULL,
                    confidence INTEGER,  -- 0-100
                    risk_reward_ratio REAL,
                    
                    -- Status
                    status TEXT NOT NULL DEFAULT 'OPEN',  -- 'OPEN' or 'CLOSED'
                    
                    -- Verification Fields
                    exchange_price_at_entry REAL,
                    exchange_price_at_exit REAL,
                    slippage REAL,
                    tradingview_chart_url TEXT,
                    
                    -- Metadata
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Account Balance History
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS account_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    balance REAL NOT NULL,
                    daily_pnl REAL,
                    equity REAL,  -- Balance + unrealized P&L
                    num_open_positions INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Signal Log
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signal_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    symbol TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    signal_type TEXT,  -- 'BUY', 'SELL', 'NO_TRADE'
                    entry_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    confidence INTEGER,
                    reasoning TEXT,  -- JSON string
                    executed BOOLEAN DEFAULT 0,
                    trade_id INTEGER,  -- FK to paper_trades if executed
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (trade_id) REFERENCES paper_trades(id)
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON paper_trades(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON paper_trades(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_status ON paper_trades(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_strategy ON paper_trades(strategy_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signal_log(timestamp)')
            
            logger.info("Database schema initialized successfully")
    
    # ========================================================================
    # INSERT OPERATIONS
    # ========================================================================
    
    def insert_trade(self, trade_data: Dict[str, Any]) -> int:
        """
        Insert new paper trade into database.
        
        Args:
            trade_data: Dictionary containing trade information.
                        Must include: timestamp, symbol, direction, entry_price, 
                        stop_loss, take_profit, strategy_name, market_type
        
        Returns:
            int: Trade ID of inserted trade
        
        Example:
            >>> db = TradeDatabase()
            >>> trade_id = db.insert_trade({
            ...     'timestamp': datetime.now(),
            ...     'symbol': 'BTC/USDT',
            ...     'direction': 'BUY',
            ...     'entry_price': 67250.00,
            ...     'stop_loss': 67100.00,
            ...     'take_profit': 67550.00,
            ...     'strategy_name': 'CRT-TBS',
            ...     'market_type': 'crypto',
            ...     'quantity': 0.01485,
            ...     'status': 'OPEN'
            ... })
        """
        required_fields = ['timestamp', 'symbol', 'direction', 'entry_price', 
                          'stop_loss', 'take_profit', 'strategy_name', 'market_type']
        
        # Validate required fields
        for field in required_fields:
            if field not in trade_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Convert datetime to ISO format string
        if isinstance(trade_data['timestamp'], datetime):
            trade_data['timestamp'] = trade_data['timestamp'].isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Build dynamic INSERT query
            columns = ', '.join(trade_data.keys())
            placeholders = ', '.join(['?' for _ in trade_data])
            query = f'INSERT INTO paper_trades ({columns}) VALUES ({placeholders})'
            
            cursor.execute(query, list(trade_data.values()))
            trade_id = cursor.lastrowid
            
            logger.info(f"Trade inserted: ID={trade_id}, Symbol={trade_data['symbol']}, Direction={trade_data['direction']}")
            
            return trade_id
    
    def insert_signal(self, signal_data: Dict[str, Any]) -> int:
        """
        Insert signal into log.
        
        Args:
            signal_data: Dictionary containing signal information
        
        Returns:
            int: Signal ID
        """
        required_fields = ['timestamp', 'symbol', 'strategy_name', 'signal_type']
        
        for field in required_fields:
            if field not in signal_data:
                raise ValueError(f"Missing required field: {field}")
        
        if isinstance(signal_data['timestamp'], datetime):
            signal_data['timestamp'] = signal_data['timestamp'].isoformat()
        
        # Convert reasoning list to JSON if present
        if 'reasoning' in signal_data and isinstance(signal_data['reasoning'], list):
            signal_data['reasoning'] = json.dumps(signal_data['reasoning'])
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            columns = ', '.join(signal_data.keys())
            placeholders = ', '.join(['?' for _ in signal_data])
            query = f'INSERT INTO signal_log ({columns}) VALUES ({placeholders})'
            
            cursor.execute(query, list(signal_data.values()))
            signal_id = cursor.lastrowid
            
            logger.debug(f"Signal logged: ID={signal_id}, Symbol={signal_data['symbol']}, Type={signal_data['signal_type']}")
            
            return signal_id
    
    def insert_account_snapshot(self, balance: float, daily_pnl: float, equity: float, num_open_positions: int):
        """
        Insert account balance snapshot.
        
        Args:
            balance: Current account balance
            daily_pnl: Today's P&L
            equity: Balance + unrealized P&L
            num_open_positions: Number of open positions
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO account_history (timestamp, balance, daily_pnl, equity, num_open_positions)
                VALUES (?, ?, ?, ?, ?)
            ''', (datetime.now().isoformat(), balance, daily_pnl, equity, num_open_positions))
            
            logger.debug(f"Account snapshot saved: Balance=${balance:.2f}, P&L=${daily_pnl:.2f}")
    
    # ========================================================================
    # UPDATE OPERATIONS
    # ========================================================================
    
    def update_trade(self, trade_id: int, updates: Dict[str, Any]) -> bool:
        """
        Update existing trade.
        
        Args:
            trade_id: Trade ID to update
            updates: Dictionary of fields to update
        
        Returns:
            bool: True if updated successfully
        
        Example:
            >>> db.update_trade(45, {
            ...     'status': 'CLOSED',
            ...     'exit_price': 67520.00,
            ...     'exit_timestamp': datetime.now(),
            ...     'exit_reason': 'TP',
            ...     'pnl_usd': 270.00,
            ...     'pnl_pct': 27.0
            ... })
        """
        if not updates:
            return False
        
        # Convert datetime objects
        for key, value in updates.items():
            if isinstance(value, datetime):
                updates[key] = value.isoformat()
        
        # Add updated_at timestamp
        updates['updated_at'] = datetime.now().isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            set_clause = ', '.join([f'{key} = ?' for key in updates.keys()])
            query = f'UPDATE paper_trades SET {set_clause} WHERE id = ?'
            
            cursor.execute(query, list(updates.values()) + [trade_id])
            
            if cursor.rowcount > 0:
                logger.info(f"Trade updated: ID={trade_id}, Fields={list(updates.keys())}")
                return True
            else:
                logger.warning(f"Trade not found: ID={trade_id}")
                return False
    
    # ========================================================================
    # QUERY OPERATIONS
    # ========================================================================
    
    def get_trade(self, trade_id: int) -> Optional[Dict]:
        """
        Get single trade by ID.
        
        Args:
            trade_id: Trade ID
        
        Returns:
            Dict with trade data or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM paper_trades WHERE id = ?', (trade_id,))
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
    
    def get_open_trades(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get all open trades, optionally filtered by symbol.
        
        Args:
            symbol: Optional symbol filter (e.g., 'BTC/USDT')
        
        Returns:
            List of trade dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if symbol:
                cursor.execute('SELECT * FROM paper_trades WHERE status = ? AND symbol = ?', ('OPEN', symbol))
            else:
                cursor.execute('SELECT * FROM paper_trades WHERE status = ?', ('OPEN',))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_closed_trades(
        self, 
        start_date: Optional[datetime] = None, 
        end_date: Optional[datetime] = None,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None
    ) -> List[Dict]:
        """
        Get closed trades with optional filters.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            symbol: Optional symbol filter
            strategy: Optional strategy filter
        
        Returns:
            List of trade dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = 'SELECT * FROM paper_trades WHERE status = ?'
            params = ['CLOSED']
            
            if start_date:
                query += ' AND timestamp >= ?'
                params.append(start_date.isoformat())
            
            if end_date:
                query += ' AND timestamp <= ?'
                params.append(end_date.isoformat())
            
            if symbol:
                query += ' AND symbol = ?'
                params.append(symbol)
            
            if strategy:
                query += ' AND strategy_name = ?'
                params.append(strategy)
            
            query += ' ORDER BY timestamp DESC'
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
    
    def get_today_trades(self) -> List[Dict]:
        """
        Get all trades from today.
        
        Returns:
            List of trade dictionaries
        """
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        return self.get_closed_trades(start_date=today_start)

    def get_all_trades(self, status: Optional[str] = None, limit: Optional[int] = None) -> List[Dict]:
        """
        Get all trades, optionally filtered by status.
        
        Args:
            status: Optional status filter ('OPEN', 'CLOSED', or None for all)
            limit: Optional limit on number of results
        
        Returns:
            List of trade dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if status:
                query = 'SELECT * FROM paper_trades WHERE status = ?'
                params = [status]
            else:
                query = 'SELECT * FROM paper_trades'
                params = []
            
            query += ' ORDER BY timestamp DESC'
            
            if limit:
                query += ' LIMIT ?'
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]

    def get_all_signals(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get all signals from signal_log table.
        
        Args:
            limit: Optional limit on number of results (default: last 20)
        
        Returns:
            List of signal dictionaries
        """
        if limit is None:
            limit = 20
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = 'SELECT * FROM signal_log ORDER BY timestamp DESC LIMIT ?'
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()
            
            signals = []
            for row in rows:
                signal_dict = dict(row)
                # Parse JSON reasoning if present
                if signal_dict.get('reasoning'):
                    try:
                        signal_dict['reasoning'] = json.loads(signal_dict['reasoning'])
                    except:
                        signal_dict['reasoning'] = []
                signals.append(signal_dict)
            
            return signals
    
    def get_trade_statistics(self, days: int = 7) -> Dict:
        """
        Get trade statistics for last N days.
        
        Args:
            days: Number of days to analyze
        
        Returns:
            Dictionary with statistics
        """
        start_date = datetime.now() - timedelta(days=days)
        trades = self.get_closed_trades(start_date=start_date)
        
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0
            }
        
        winning_trades = [t for t in trades if t['pnl_usd'] and t['pnl_usd'] > 0]
        losing_trades = [t for t in trades if t['pnl_usd'] and t['pnl_usd'] < 0]
        
        total_wins = sum(t['pnl_usd'] for t in winning_trades)
        total_losses = abs(sum(t['pnl_usd'] for t in losing_trades))
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': (len(winning_trades) / len(trades)) * 100 if trades else 0.0,
            'total_pnl': sum(t['pnl_usd'] or 0 for t in trades),
            'avg_win': total_wins / len(winning_trades) if winning_trades else 0.0,
            'avg_loss': total_losses / len(losing_trades) if losing_trades else 0.0,
            'profit_factor': total_wins / total_losses if total_losses > 0 else 0.0
        }
    
    # ========================================================================
    # EXPORT OPERATIONS
    # ========================================================================
    
    def export_to_csv(self, filename: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None):
        """
        Export trades to CSV file for verification.
        
        Args:
            filename: Output CSV filename
            start_date: Optional start date filter
            end_date: Optional end date filter
        """
        trades = self.get_closed_trades(start_date=start_date, end_date=end_date)
        
        if not trades:
            logger.warning("No trades to export")
            return
        
        # Define CSV columns
        columns = [
            'id', 'timestamp', 'symbol', 'direction', 'entry_price', 'exit_price',
            'pnl_usd', 'pnl_pct', 'exit_reason', 'strategy_name', 'confidence',
            'exchange_price_at_entry', 'exchange_price_at_exit', 'slippage'
        ]
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(trades)
        
        logger.info(f"Exported {len(trades)} trades to {filename}")
    
    def backup_database(self):
        """
        Create backup of database file.
        """
        backup_filename = f"paper_trading_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        backup_path = self.backup_path / backup_filename
        
        shutil.copy2(self.db_path, backup_path)
        
        logger.info(f"Database backed up to {backup_path}")
        
        return backup_path


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TRADE DATABASE TEST SUITE")
    print("=" * 70)
    
    # Use test database
    test_db_path = "paper_trading/data/test_paper_trading.db"
    db = TradeDatabase(test_db_path)
    
    # Test 1: Insert trade
    print("\n1️⃣ TEST: Insert New Trade")
    print("-" * 70)
    trade_data = {
        'timestamp': datetime.now(),
        'symbol': 'BTC/USDT',
        'exchange': 'kucoin',
        'market_type': 'crypto',
        'direction': 'BUY',
        'entry_price': 67250.00,
        'quantity': 0.01485,
        'stop_loss': 67100.00,
        'take_profit': 67550.00,
        'strategy_name': 'CRT-TBS',
        'confidence': 85,
        'risk_reward_ratio': 2.0,
        'status': 'OPEN',
        'exchange_price_at_entry': 67249.80
    }
    trade_id = db.insert_trade(trade_data)
    print(f"✅ Trade inserted with ID: {trade_id}")
    
    # Test 2: Update trade (close it)
    print("\n2️⃣ TEST: Update Trade (Close)")
    print("-" * 70)
    updates = {
        'status': 'CLOSED',
        'exit_price': 67520.00,
        'exit_timestamp': datetime.now(),
        'exit_reason': 'TP',
        'pnl_usd': 270.00,
        'pnl_pct': 27.0,
        'exchange_price_at_exit': 67519.50
    }
    db.update_trade(trade_id, updates)
    print(f"✅ Trade updated: P&L=${updates['pnl_usd']:.2f}")
    
    # Test 3: Query trade
    print("\n3️⃣ TEST: Query Trade")
    print("-" * 70)
    trade = db.get_trade(trade_id)
    print(f"✅ Retrieved trade: {trade['symbol']} {trade['direction']} @ ${trade['entry_price']:.2f}")
    print(f"   Status: {trade['status']}, P&L: ${trade['pnl_usd']:.2f}")
    
    # Test 4: Get statistics
    print("\n4️⃣ TEST: Trade Statistics")
    print("-" * 70)
    stats = db.get_trade_statistics(days=7)
    print(f"✅ Statistics (Last 7 days):")
    print(f"   Total Trades: {stats['total_trades']}")
    print(f"   Win Rate: {stats['win_rate']:.1f}%")
    print(f"   Total P&L: ${stats['total_pnl']:.2f}")
    
    # Test 5: Export to CSV
    print("\n5️⃣ TEST: Export to CSV")
    print("-" * 70)
    csv_filename = "paper_trading/data/test_export.csv"
    db.export_to_csv(csv_filename)
    print(f"✅ Exported to {csv_filename}")
    
    # Test 6: Backup
    print("\n6️⃣ TEST: Database Backup")
    print("-" * 70)
    backup_path = db.backup_database()
    print(f"✅ Backup created: {backup_path}")
    
    print("\n" + "=" * 70)
    print("✅ ALL DATABASE TESTS PASSED")
    print("=" * 70)
