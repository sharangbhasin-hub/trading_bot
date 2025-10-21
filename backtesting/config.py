"""
Backtesting Configuration
Contains all settings for historical testing
"""
from datetime import datetime, timedelta

class BacktestConfig:
    """Configuration for backtesting system"""
    
    # ===== TEST PERIODS =====
    
    # Phase 1: Historical Backtest (2024)
    BACKTEST_START_DATE = datetime(2024, 1, 1)
    BACKTEST_END_DATE = datetime(2024, 12, 31)
    
    # Phase 2: Out-of-Sample Test (2025)
    FORWARD_TEST_START_DATE = datetime(2025, 1, 1)
    FORWARD_TEST_END_DATE = datetime(2025, 10, 20)  # Up to yesterday
    
    # ===== MARKET PARAMETERS =====
    
    # Trading hours (IST)
    MARKET_OPEN_TIME = "09:15"
    MARKET_CLOSE_TIME = "15:30"
    
    # Indices to test
    INDICES = ['NIFTY', 'BANKNIFTY']
    
    # ===== DATA PARAMETERS =====
    
    # Timeframes to fetch
    TIMEFRAMES = {
        '5min': '5minute',
        '15min': '15minute',
        '1h': '60minute',
        'daily': 'day'
    }
    
    # Number of candles to keep in memory
    LOOKBACK_CANDLES = {
        '5min': 100,
        '15min': 50,
        '1h': 24,
        'daily': 10
    }
    
    # ===== BACKTESTING PARAMETERS =====
    
    # Replay interval (how often to check for signals)
    REPLAY_INTERVAL_MINUTES = 15  # Check every 15 minutes
    
    # Slippage simulation (in percentage)
    SLIPPAGE_PERCENT = 0.05  # 0.05% slippage on entry/exit
    
    # Commission (if any)
    COMMISSION_PERCENT = 0.0  # No commission for index testing
    
    # End-of-day close time (close all positions)
    EOD_CLOSE_TIME = "15:20"
    
    # ===== PARAMETER OPTIMIZATION =====
    
    # Parameters to test (grid search)
    PARAM_GRID = {
        'ob_threshold': [1.0, 1.2, 1.5, 1.8],
        'confidence_min': [65, 70, 75, 80],
        'stop_loss_multiplier': [0.3, 0.5, 0.7],
        'fvg_size_threshold': [0.0005, 0.001, 0.002]
    }
    
    # ===== OUTPUT SETTINGS =====
    
    # Where to store results
    RESULTS_DIR = 'backtest_results'
    
    # Report formats
    GENERATE_HTML = True
    GENERATE_PDF = False  # Requires additional libraries
    GENERATE_CSV = True
    
    # Chart settings
    CHART_DPI = 100
    CHART_WIDTH = 12
    CHART_HEIGHT = 6
    
    # ===== VALIDATION THRESHOLDS =====
    
    # Minimum viable performance
    MIN_WIN_RATE = 0.55  # 55%
    MIN_PROFIT_FACTOR = 1.5
    MIN_RR_RATIO = 1.5
    MAX_DRAWDOWN_PCT = 20.0
    
    # Optimal performance targets
    TARGET_WIN_RATE = 0.65  # 65%
    TARGET_PROFIT_FACTOR = 2.0
    TARGET_RR_RATIO = 2.0
    
    # ===== MARKET CONDITION CLASSIFICATION =====
    
    # ADX thresholds
    ADX_TRENDING_THRESHOLD = 25
    ADX_RANGING_THRESHOLD = 20
    
    # ATR volatility threshold (multiplier of average)
    ATR_VOLATILE_MULTIPLIER = 1.5
    
    # Gap threshold (percentage)
    GAP_THRESHOLD_PERCENT = 1.0
    
    # ===== LOGGING =====
    
    LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR
    LOG_FILE = 'backtest.log'


# ===== HELPER FUNCTIONS =====

def get_trading_days(start_date, end_date):
    """
    Get list of trading days between two dates
    Excludes weekends and major holidays
    """
    from pandas.tseries.holiday import USFederalHolidayCalendar
    from pandas.tseries.offsets import CustomBusinessDay
    
    # Indian market holidays (simplified - you may need to add more)
    indian_holidays = [
        datetime(2024, 1, 26),  # Republic Day
        datetime(2024, 3, 8),   # Holi
        datetime(2024, 3, 29),  # Good Friday
        datetime(2024, 8, 15),  # Independence Day
        datetime(2024, 10, 2),  # Gandhi Jayanti
        datetime(2024, 10, 12), # Dussehra
        datetime(2024, 10, 31), # Diwali
        datetime(2024, 11, 1),  # Diwali
        # Add 2025 holidays as needed
    ]
    
    import pandas as pd
    business_days = pd.date_range(
        start=start_date,
        end=end_date,
        freq='B'  # Business days (Mon-Fri)
    )
    
    # Filter out holidays
    trading_days = [day for day in business_days if day not in indian_holidays]
    
    return trading_days


def format_currency(amount):
    """Format number as Indian currency"""
    return f"₹{amount:,.2f}"


def format_percent(value):
    """Format number as percentage"""
    return f"{value:.2f}%"


def calculate_position_size(capital, risk_percent, stop_loss_distance):
    """
    Calculate position size based on capital and risk
    
    Args:
        capital: Total capital
        risk_percent: Percentage of capital to risk per trade
        stop_loss_distance: Distance to stop loss in points
    
    Returns:
        Position size (number of lots/contracts)
    """
    risk_amount = capital * (risk_percent / 100)
    position_size = risk_amount / stop_loss_distance
    return max(1, int(position_size))  # At least 1 lot
