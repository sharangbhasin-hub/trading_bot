import sqlite3
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict
import os
import tempfile

# ============================================================================
# DATABASE CONFIGURATION WITH FALLBACK
# ============================================================================

# Import DB_NAME from config, with fallback
try:
    from config import DB_NAME
except ImportError:
    # Fallback if config import fails
    if os.environ.get('STREAMLIT_RUNTIME_ENV') == 'cloud':
        DB_NAME = os.path.join(tempfile.gettempdir(), 'trading_data.db')
    else:
        DB_NAME = 'trading_data.db'

# Global connection for singleton pattern
_db_connection = None

def get_db_connection():
    """Get or create database connection (singleton pattern)"""
    global _db_connection
    try:
        if _db_connection is None:
            _db_connection = sqlite3.connect(DB_NAME, check_same_thread=False)
        return _db_connection
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return None

# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================

def init_database():
    """Initialize SQLite database with all required tables"""
    try:
        # Ensure directory exists (if not in-memory)
        if DB_NAME != ':memory:':
            db_dir = os.path.dirname(DB_NAME)
            if db_dir and not os.path.exists(db_dir):
                try:
                    os.makedirs(db_dir, exist_ok=True)
                except Exception as e:
                    print(f"⚠️ Could not create directory: {e}")
        
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Table 1: Real-time tick data storage
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tick_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                instrument_token INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                last_price REAL,
                volume INTEGER,
                buy_quantity INTEGER,
                sell_quantity INTEGER,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                change REAL
            )
        """)
        
        # Table 2: OHLC data aggregated from ticks
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ohlc_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                UNIQUE(symbol, timestamp, timeframe)
            )
        """)
        
        # Table 3: Trade logs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                order_id TEXT,
                symbol TEXT NOT NULL,
                transaction_type TEXT NOT NULL,
                order_type TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL,
                status TEXT,
                product_type TEXT,
                exchange TEXT,
                notes TEXT
            )
        """)
        
        # Table 4: Instrument master (stocks & options)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS instruments (
                instrument_token INTEGER PRIMARY KEY,
                exchange_token INTEGER,
                tradingsymbol TEXT UNIQUE NOT NULL,
                name TEXT,
                last_price REAL,
                expiry DATE,
                strike REAL,
                tick_size REAL,
                lot_size INTEGER,
                instrument_type TEXT,
                segment TEXT,
                exchange TEXT,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Table 5: Watchlist/Subscribed instruments
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                instrument_token INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                added_on DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                UNIQUE(instrument_token)
            )
        """)
        
        # Create indices for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tick_symbol ON tick_data(symbol, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ohlc_symbol ON ohlc_data(symbol, timeframe, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol, timestamp)")
        
        conn.commit()
        conn.close()
        print(f"✅ Database initialized successfully: {DB_NAME}")
        return True
        
    except sqlite3.OperationalError as e:
        print(f"❌ SQLite OperationalError: {e}")
        print(f"⚠️ Database path: {DB_NAME}")
        print(f"⚠️ This usually means the directory is not writable")
        print(f"⚠️ App will continue WITHOUT persistent database storage")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected database initialization error: {e}")
        print(f"⚠️ App will continue WITHOUT persistent database storage")
        return False

# ============================================================================
# TICK DATA OPERATIONS
# ============================================================================

def insert_tick_data(tick: Dict):
    """Insert real-time tick data into database"""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO tick_data (
                instrument_token, symbol, last_price, volume,
                buy_quantity, sell_quantity, open, high, low, close, change
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            tick.get('instrument_token'),
            tick.get('symbol'),
            tick.get('last_price'),
            tick.get('volume'),
            tick.get('buy_quantity'),
            tick.get('sell_quantity'),
            tick.get('ohlc', {}).get('open'),
            tick.get('ohlc', {}).get('high'),
            tick.get('ohlc', {}).get('low'),
            tick.get('ohlc', {}).get('close'),
            tick.get('change')
        ))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        # Silent fail - don't break app if database unavailable
        return False

def get_latest_ticks(symbol: str, limit: int = 100) -> pd.DataFrame:
    """Get latest tick data for a symbol"""
    try:
        conn = sqlite3.connect(DB_NAME)
        query = f"""
            SELECT * FROM tick_data 
            WHERE symbol = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(symbol, limit))
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame()

# ============================================================================
# INSTRUMENT OPERATIONS
# ============================================================================

def insert_instruments(instruments: List[Dict]):
    """Bulk insert instruments into database"""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        for inst in instruments:
            cursor.execute("""
                INSERT OR REPLACE INTO instruments (
                    instrument_token, exchange_token, tradingsymbol, name,
                    last_price, expiry, strike, tick_size, lot_size,
                    instrument_type, segment, exchange
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                inst.get('instrument_token'),
                inst.get('exchange_token'),
                inst.get('tradingsymbol'),
                inst.get('name'),
                inst.get('last_price'),
                inst.get('expiry'),
                inst.get('strike'),
                inst.get('tick_size'),
                inst.get('lot_size'),
                inst.get('instrument_type'),
                inst.get('segment'),
                inst.get('exchange')
            ))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        return False

def get_instrument_by_symbol(symbol: str) -> Optional[Dict]:
    """Get instrument details by symbol"""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM instruments WHERE tradingsymbol = ?
        """, (symbol,))
        
        row = cursor.fetchone()
        
        if row:
            columns = [desc[0] for desc in cursor.description]
            result = dict(zip(columns, row))
            conn.close()
            return result
        
        conn.close()
        return None
    except Exception as e:
        return None

def search_instruments(query: str, segment: str = "NSE") -> pd.DataFrame:
    """Search instruments by name or symbol"""
    try:
        conn = sqlite3.connect(DB_NAME)
        sql = f"""
            SELECT tradingsymbol, name, instrument_type, exchange, last_price
            FROM instruments 
            WHERE (tradingsymbol LIKE ? OR name LIKE ?)
            AND exchange = ?
            LIMIT 50
        """
        df = pd.read_sql_query(sql, conn, params=(f"%{query}%", f"%{query}%", segment))
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame()

# ============================================================================
# WATCHLIST OPERATIONS
# ============================================================================

def add_to_watchlist(instrument_token: int, symbol: str) -> bool:
    """Add instrument to watchlist"""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR IGNORE INTO watchlist (instrument_token, symbol)
            VALUES (?, ?)
        """, (instrument_token, symbol))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        return False

def get_watchlist() -> List[Dict]:
    """Get all active instruments in watchlist"""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT instrument_token, symbol FROM watchlist 
            WHERE is_active = 1
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [{"instrument_token": row[0], "symbol": row[1]} for row in rows]
    except Exception as e:
        return []

def remove_from_watchlist(instrument_token: int) -> bool:
    """Remove instrument from watchlist"""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE watchlist SET is_active = 0 
            WHERE instrument_token = ?
        """, (instrument_token,))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        return False

# ============================================================================
# TRADE LOGGING
# ============================================================================

def log_trade(trade_data: Dict) -> bool:
    """Log trade execution to database"""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO trades (
                order_id, symbol, transaction_type, order_type,
                quantity, price, status, product_type, exchange, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_data.get('order_id'),
            trade_data.get('symbol'),
            trade_data.get('transaction_type'),
            trade_data.get('order_type'),
            trade_data.get('quantity'),
            trade_data.get('price'),
            trade_data.get('status'),
            trade_data.get('product_type'),
            trade_data.get('exchange'),
            trade_data.get('notes')
        ))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        return False

def get_trade_history(limit: int = 100) -> pd.DataFrame:
    """Get recent trade history"""
    try:
        conn = sqlite3.connect(DB_NAME)
        query = f"""
            SELECT * FROM trades 
            ORDER BY timestamp DESC 
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clear_old_tick_data(days: int = 7):
    """Clear tick data older than specified days"""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM tick_data 
            WHERE timestamp < datetime('now', '-' || ? || ' days')
        """, (days,))
        
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        print(f"✅ Deleted {deleted} old tick records")
        return True
    except Exception as e:
        return False
