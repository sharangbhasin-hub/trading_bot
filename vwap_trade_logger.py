"""
VWAP Trade Logger - Simplified Trade Logging
=============================================
Lightweight trade logger for VWAP strategies.
Separate from journal for real-time logging needs.

Author: Trading System
Date: November 18, 2025
"""

import logging
from datetime import datetime
from typing import Dict, Optional
import json

logger = logging.getLogger(__name__)

class VWAPTradeLogger:
    """
    Simplified trade logger for real-time logging.
    Complements the journal system with lightweight logging.
    """
    
    def __init__(self, log_file: str = 'vwap_trades.log'):
        """
        Initialize trade logger.
        
        Args:
            log_file: Path to log file
        """
        self.log_file = log_file
        
        # Setup file handler
        self.file_handler = logging.FileHandler(log_file)
        self.file_handler.setLevel(logging.INFO)
        
        # Format: timestamp | level | message
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.file_handler.setFormatter(formatter)
        
        # Create logger
        self.trade_logger = logging.getLogger('vwap_trades')
        self.trade_logger.setLevel(logging.INFO)
        self.trade_logger.addHandler(self.file_handler)
        
        logger.info(f"Trade Logger initialized: {log_file}")
    
    def log_entry(self, trade_data: Dict):
        """
        Log trade entry.
        
        Args:
            trade_data: Trade entry details
        """
        log_msg = (
            f"ENTRY | {trade_data.get('strategy_type', 'UNKNOWN')} | "
            f"Premium: {trade_data.get('entry_premium', 0):.2f} | "
            f"SL: {trade_data.get('sl_premium', 0):.2f} | "
            f"Target: {trade_data.get('target_premium', 0):.2f} | "
            f"Lots: {trade_data.get('lots', 0)} | "
            f"Reason: {trade_data.get('entry_reason', 'N/A')}"
        )
        
        self.trade_logger.info(log_msg)
        
        # Also log as JSON for parsing
        self.trade_logger.debug(json.dumps({
            'event': 'ENTRY',
            'timestamp': datetime.now().isoformat(),
            **trade_data
        }))
    
    def log_exit(self, trade_data: Dict):
        """
        Log trade exit.
        
        Args:
            trade_data: Trade exit details
        """
        pnl = trade_data.get('net_pnl', 0)
        pnl_str = f"+₹{pnl:.2f}" if pnl > 0 else f"₹{pnl:.2f}"
        
        log_msg = (
            f"EXIT | {trade_data.get('strategy_type', 'UNKNOWN')} | "
            f"Exit Premium: {trade_data.get('exit_premium', 0):.2f} | "
            f"Exit Type: {trade_data.get('exit_type', 'UNKNOWN')} | "
            f"P&L: {pnl_str} | "
            f"Reason: {trade_data.get('exit_reason', 'N/A')}"
        )
        
        self.trade_logger.info(log_msg)
        
        # JSON log
        self.trade_logger.debug(json.dumps({
            'event': 'EXIT',
            'timestamp': datetime.now().isoformat(),
            **trade_data
        }))
    
    def log_signal(self, signal_data: Dict):
        """
        Log VWAP signal detection.
        
        Args:
            signal_data: Signal details
        """
        log_msg = (
            f"SIGNAL | {signal_data.get('signal_type', 'UNKNOWN')} | "
            f"Premium: {signal_data.get('premium', 0):.2f} | "
            f"VWAP: {signal_data.get('vwap', 0):.2f} | "
            f"Crossover: {signal_data.get('crossover_direction', 'N/A')}"
        )
        
        self.trade_logger.info(log_msg)
    
    def log_rejection(self, reason: str, details: Optional[Dict] = None):
        """
        Log trade rejection.
        
        Args:
            reason: Rejection reason
            details: Additional details
        """
        log_msg = f"REJECTED | Reason: {reason}"
        
        if details:
            log_msg += f" | Details: {json.dumps(details)}"
        
        self.trade_logger.warning(log_msg)
    
    def log_error(self, error_msg: str, exception: Optional[Exception] = None):
        """
        Log error.
        
        Args:
            error_msg: Error message
            exception: Exception object if available
        """
        if exception:
            self.trade_logger.error(f"ERROR | {error_msg} | Exception: {str(exception)}")
        else:
            self.trade_logger.error(f"ERROR | {error_msg}")
    
    def log_vwap_update(self, vwap_data: Dict):
        """
        Log VWAP calculation update (debug level).
        
        Args:
            vwap_data: VWAP data
        """
        self.trade_logger.debug(
            f"VWAP_UPDATE | "
            f"Premium: {vwap_data.get('combined_premium', 0):.2f} | "
            f"VWAP: {vwap_data.get('vwap', 0):.2f}"
        )
    
    def close(self):
        """Close logger and file handlers"""
        self.file_handler.close()
        self.trade_logger.removeHandler(self.file_handler)

# ============================================================================
# STANDALONE FUNCTIONS
# ============================================================================

def create_trade_logger(log_file: str = 'vwap_trades.log') -> VWAPTradeLogger:
    """
    Create trade logger instance.
    
    Args:
        log_file: Path to log file
    
    Returns:
        VWAPTradeLogger instance
    """
    return VWAPTradeLogger(log_file)
