"""
VWAP Strategy Backtest Adapter
================================
Adapts VWAP strategies to work with existing backtest framework.
Handles option chain data simulation and VWAP calculation from historical data.
"""

from strategies.strategy_vwap_strangle_selling import VWAPStrangleSelling
from strategies.strategy_vwap_strangle_buying import VWAPStrangleBuying
from backtesting.backtest_runner import BacktestRunner
import pandas as pd

class VWAPBacktestAdapter:
    """Adapter for running VWAP strategies in backtest mode"""
    
    def __init__(self, strategy_type: str = 'SELLING'):
        """
        Args:
            strategy_type: 'SELLING' or 'BUYING'
        """
        if strategy_type == 'SELLING':
            self.strategy = VWAPStrangleSelling()
        else:
            self.strategy = VWAPStrangleBuying()
    
    def prepare_option_chain_data(self, historical_df: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate option chain from historical spot data.
        Used for backtesting when option chain history not available.
        """
        # Calculate synthetic option premiums using Black-Scholes or historical relationships
        # (Simplified for plan - actual implementation needed)
        pass
    
    def run_backtest(self, start_date: str, end_date: str) -> Dict:
        """Run backtest for VWAP strategy"""
        runner = BacktestRunner()
        # Use existing backtest infrastructure
        return runner.run(
            strategy=self.strategy,
            start_date=start_date,
            end_date=end_date
        )
