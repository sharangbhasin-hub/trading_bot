"""
Paper Trading Configuration
===========================

Centralized configuration for paper trading system.
All settings for risk management, market-specific parameters, and system behavior.

Last Updated: October 29, 2025
"""

import os
from pathlib import Path
from typing import Dict, Any

# Base directory for paper trading
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
LOGS_DIR = BASE_DIR / 'logs'
BACKUPS_DIR = BASE_DIR / 'backups'

# Create directories if they don't exist
for directory in [DATA_DIR, LOGS_DIR, BACKUPS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


PAPER_TRADING_CONFIG: Dict[str, Any] = {
    # ============================================================================
    # ACCOUNT SETTINGS
    # ============================================================================
    'initial_balance': 10000.0,  # Starting capital in USD
    'currency': 'USD',
    
    # ============================================================================
    # RISK MANAGEMENT (Conservative settings for paper trading)
    # ============================================================================
    'risk_management': {
        'max_daily_loss_usd': 200.0,        # Max $200 loss per day
        'max_daily_loss_pct': 2.0,          # Max 2% of balance per day
        'max_open_positions': 3,            # Max 3 concurrent positions
        'max_position_size_pct': 30.0,      # Max 30% of balance per position
        'risk_per_trade_pct': 1.0,          # Risk 1% of balance per trade
        'max_trades_per_day': 10,           # Max 10 trades per day (prevent over-trading)
        'min_risk_reward_ratio': 1.0,       # Minimum RR ratio (scalping mode)
        'max_symbols': 5,                   # ✅ ADDED: Max symbols for multi-symbol trading
        'max_positions_per_symbol': 2,      # ✅ ADDED: Max positions per individual symbol
    },
    
    # ============================================================================
    # CRYPTOCURRENCY SETTINGS
    # ============================================================================
    'crypto': {
        'investment_per_trade_usd': 1000.0,  # Fixed $1,000 per trade
        'min_investment': 100.0,             # Minimum $100
        'max_investment': 5000.0,            # Maximum $5,000
        
        # Transaction costs (realistic estimates)
        'slippage_pct': 0.001,               # 0.1% slippage (market orders)
        'commission_pct': 0.001,             # 0.1% commission (maker/taker average)
        
        # Supported pairs
        'supported_pairs': [
            'BTC/USDT',
            'ETH/USDT',
            'BNB/USDT',
            'SOL/USDT',
            'XRP/USDT',
        ],
        
        # Exchange preferences (in order of priority)
        'exchange_priority': ['kucoin', 'bybit', 'okx'],
    },
    
    # ============================================================================
    # FOREX SETTINGS
    # ============================================================================
    'forex': {
        'lot_size': 0.01,                    # 0.01 lots (micro lot = 1,000 units)
        'min_lot_size': 0.01,
        'max_lot_size': 1.0,
        
        # Transaction costs
        'slippage_pips': 0.5,                # 0.5 pip slippage
        'commission_per_lot': 0.50,          # $0.50 per 0.01 lot (round-turn)
        'spread_pips': 1.0,                  # Typical spread (EUR/USD)
        
        # OANDA practice account integration
        'use_oanda_practice': True,         # Set True to use real OANDA practice API
        'oanda_enabled': True,
        
        # Supported pairs
        'supported_pairs': [
            'EUR/USD',
            'GBP/USD',
            'USD/JPY',
            'AUD/USD',
            'USD/CAD',
        ],
        
        # Pip values per 0.01 lot (for P&L calculation)
        'pip_values': {
            'EUR/USD': 0.10,  # $0.10 per pip
            'GBP/USD': 0.10,
            'USD/JPY': 0.09,
            'AUD/USD': 0.10,
            'USD/CAD': 0.09,
        },
    },
    
    # ============================================================================
    # DATA MANAGEMENT
    # ============================================================================
    'data': {
        'polling_interval_seconds': 60,      # Fetch data every 60 seconds
        'candles_buffer_size': 500,          # Keep last 500 candles in memory
        'retry_attempts': 3,                 # Retry failed requests 3 times
        'retry_delay_seconds': 5,            # Wait 5 seconds between retries
        'request_timeout_seconds': 10,       # Timeout after 10 seconds
    },
    
    # ============================================================================
    # UI SETTINGS
    # ============================================================================
    'ui': {
        'auto_refresh_seconds': 5,           # Auto-refresh UI every 5 seconds
        'max_signals_display': 10,           # Show last 10 signals
        'max_trades_display': 50,            # Show last 50 trades in history
        'chart_height': 400,                 # Chart height in pixels
    },
    
    # ============================================================================
    # DATABASE SETTINGS
    # ============================================================================
    'database': {
        'path': str(DATA_DIR / 'paper_trading.db'),
        'backup_frequency_hours': 24,        # Backup every 24 hours
        'backup_path': str(BACKUPS_DIR),
        'wal_mode': True,                    # Use WAL mode for better concurrency
    },
    
    # ============================================================================
    # LOGGING SETTINGS
    # ============================================================================
    'logging': {
        'level': 'INFO',                     # INFO, DEBUG, WARNING, ERROR
        'file': str(LOGS_DIR / 'paper_trading.log'),
        'max_size_mb': 10,                   # Max log file size
        'backup_count': 5,                   # Keep 5 backup log files
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    },
    
    # ============================================================================
    # TRADING MODES
    # ============================================================================
    'trading_modes': {
        'scalping': {
            'htf': '1h',                     # Higher timeframe
            'ltf': '1min',                   # Lower timeframe
            'min_rr_ratio': 1.0,             # Minimum RR for scalping
            'max_hold_time_minutes': 60,     # Max 1 hour
        },
        'intraday': {
            'htf': '1d',                     # Higher timeframe
            'ltf': '1h',                     # Lower timeframe
            'min_rr_ratio': 2.0,             # Minimum RR for intraday
            'max_hold_time_hours': 24,       # Max 1 day
        },
    },
}


def get_config(section: str = None) -> Dict[str, Any]:
    """
    Get configuration settings.
    
    Args:
        section: Optional section name to retrieve specific config section.
                 If None, returns full config.
    
    Returns:
        Configuration dictionary
    
    Example:
        >>> crypto_config = get_config('crypto')
        >>> investment = crypto_config['investment_per_trade_usd']
    """
    if section is None:
        return PAPER_TRADING_CONFIG
    
    if section not in PAPER_TRADING_CONFIG:
        raise KeyError(f"Configuration section '{section}' not found")
    
    return PAPER_TRADING_CONFIG[section]


def update_config(section: str, key: str, value: Any) -> None:
    """
    Update configuration value.
    
    Args:
        section: Configuration section name
        key: Configuration key within section
        value: New value to set
    
    Example:
        >>> update_config('crypto', 'investment_per_trade_usd', 2000.0)
    """
    if section not in PAPER_TRADING_CONFIG:
        raise KeyError(f"Configuration section '{section}' not found")
    
    if key not in PAPER_TRADING_CONFIG[section]:
        raise KeyError(f"Configuration key '{key}' not found in section '{section}'")
    
    PAPER_TRADING_CONFIG[section][key] = value


def validate_config() -> bool:
    """
    Validate configuration settings.
    
    Returns:
        True if configuration is valid, raises ValueError otherwise
    """
    # Validate risk management
    risk = PAPER_TRADING_CONFIG['risk_management']
    if risk['max_daily_loss_pct'] <= 0 or risk['max_daily_loss_pct'] > 100:
        raise ValueError("max_daily_loss_pct must be between 0 and 100")
    
    if risk['max_open_positions'] < 1:
        raise ValueError("max_open_positions must be at least 1")
    
    if risk['min_risk_reward_ratio'] < 0.5:
        raise ValueError("min_risk_reward_ratio should be at least 0.5")
    
    # Validate crypto settings
    crypto = PAPER_TRADING_CONFIG['crypto']
    if crypto['investment_per_trade_usd'] < crypto['min_investment']:
        raise ValueError("investment_per_trade_usd cannot be less than min_investment")
    
    if crypto['investment_per_trade_usd'] > crypto['max_investment']:
        raise ValueError("investment_per_trade_usd cannot exceed max_investment")
    
    # Validate forex settings
    forex = PAPER_TRADING_CONFIG['forex']
    if forex['lot_size'] < forex['min_lot_size']:
        raise ValueError("lot_size cannot be less than min_lot_size")
    
    if forex['lot_size'] > forex['max_lot_size']:
        raise ValueError("lot_size cannot exceed max_lot_size")
    
    # Validate data settings
    data = PAPER_TRADING_CONFIG['data']
    if data['polling_interval_seconds'] < 10:
        raise ValueError("polling_interval_seconds should be at least 10 to avoid rate limits")
    
    return True


# Validate configuration on module import
try:
    validate_config()
except ValueError as e:
    print(f"⚠️ Configuration validation warning: {e}")


if __name__ == "__main__":
    # Test configuration
    print("📋 Paper Trading Configuration")
    print("=" * 50)
    print(f"Initial Balance: ${PAPER_TRADING_CONFIG['initial_balance']:,.2f}")
    print(f"Max Daily Loss: ${PAPER_TRADING_CONFIG['risk_management']['max_daily_loss_usd']:,.2f}")
    print(f"Crypto Investment: ${PAPER_TRADING_CONFIG['crypto']['investment_per_trade_usd']:,.2f}")
    print(f"Forex Lot Size: {PAPER_TRADING_CONFIG['forex']['lot_size']} lots")
    print(f"Database: {PAPER_TRADING_CONFIG['database']['path']}")
    print("=" * 50)
    print("✅ Configuration validated successfully!")
