"""
Trading Constants - Standardized Values
"""

class Timeframes:
    """Standardized timeframe names"""
    FIVE_MIN = "5min"
    FIFTEEN_MIN = "15min"
    ONE_HOUR = "1h"
    FOUR_HOUR = "4h"
    DAILY = "1d"
    
    @classmethod
    def display_name(cls, timeframe: str) -> str:
        """Convert to readable format"""
        mapping = {
            cls.FIVE_MIN: "5-Minute",
            cls.FIFTEEN_MIN: "15-Minute",
            cls.ONE_HOUR: "1-Hour",
            cls.FOUR_HOUR: "4-Hour",
            cls.DAILY: "Daily"
        }
        return mapping.get(timeframe, timeframe)


class DataRequirements:
    """Minimum candle requirements"""
    MIN_CANDLES_STRATEGY = 20      # For strategy execution
    MIN_CANDLES_REGIME = 16         # For market regime detection  
    MIN_CANDLES_STRUCTURE = 10      # For structure detection
    MIN_CANDLES_ORDERBLOCK = 20     # For order block detection
    MIN_CANDLES_FVG = 3             # For FVG detection
    MIN_CANDLES_LIQUIDITY = 20      # For liquidity detection
    MIN_CANDLES_RETEST = 5          # For retest detection
    
    # Technical indicator periods
    PERIOD_ADX = 14
    PERIOD_ATR = 14
    
    # Lookback periods
    LOOKBACK_CANDLES = 50           # Default lookback for patterns
