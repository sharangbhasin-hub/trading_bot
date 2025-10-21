"""
Parameter Optimizer
Tests different parameter combinations to find optimal settings
"""
import pandas as pd
import numpy as np
from itertools import product
import logging

from backtesting.config import BacktestConfig

logger = logging.getLogger(__name__)


class ParameterOptimizer:
    """
    Optimizes strategy parameters through grid search
    """
    
    def __init__(self, backtest_runner):
        """
        Initialize parameter optimizer
        
        Args:
            backtest_runner: Instance of BacktestRunner
        """
        self.runner = backtest_runner
        self.config = BacktestConfig()
        self.optimization_results = []
        
    def run_grid_search(self, param_grid=None):
        """
        Run grid search over parameter combinations
        
        Args:
            param_grid: Dict of parameters to test (default from config)
        
        Returns:
            DataFrame with all results
        """
        if param_grid is None:
            param_grid = self.config.PARAM_GRID
        
        logger.info(f"Starting grid search with {len(param_grid)} parameters")
        
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        total_combinations = len(combinations)
        logger.info(f"Testing {total_combinations} parameter combinations")
        
        # Test each combination
        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            
            logger.info(f"Testing combination {i+1}/{total_combinations}: {params}")
            
            # Run backtest with these parameters
            result = self._test_parameters(params)
            
            result['params'] = params
            result['combo_id'] = i
            
            self.optimization_results.append(result)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(self.optimization_results)
        
        return results_df
    
    def _test_parameters(self, params):
        """
        Test a single parameter combination
        
        Args:
            params: Dict of parameters
        
        Returns:
            Dict with performance metrics
        """
        # Apply parameters to strategy_manager or detectors
        # This is a simplified version - you'll need to actually
        # modify the parameters in your existing code
        
        # For now, we'll just record what would be tested
        result = {
            'win_rate': np.random.uniform(50, 75),  # Placeholder
            'profit_factor': np.random.uniform(1.2, 2.5),  # Placeholder
            'total_trades': np.random.randint(80, 200),  # Placeholder
            'total_pnl': np.random.uniform(-500, 2000),  # Placeholder
        }
        
        # Add parameter values
        for key, value in params.items():
            result[key] = value
        
        return result
    
    def get_best_parameters(self, metric='win_rate'):
        """
        Get best parameter combination based on metric
        
        Args:
            metric: Metric to optimize ('win_rate', 'profit_factor', etc.)
        
        Returns:
            Dict with best parameters
        """
        if not self.optimization_results:
            return {}
        
        results_df = pd.DataFrame(self.optimization_results)
        
        best_idx = results_df[metric].idxmax()
        best_result = results_df.loc[best_idx]
        
        return {
            'parameters': best_result['params'],
            'performance': {
                'win_rate': best_result['win_rate'],
                'profit_factor': best_result['profit_factor'],
                'total_trades': best_result['total_trades'],
                'total_pnl': best_result['total_pnl'],
            }
        }
    
    def get_parameter_sensitivity(self, parameter_name):
        """
        Analyze sensitivity to a specific parameter
        
        Args:
            parameter_name: Name of parameter to analyze
        
        Returns:
            DataFrame with parameter value vs metrics
        """
        if not self.optimization_results:
            return pd.DataFrame()
        
        results_df = pd.DataFrame(self.optimization_results)
        
        # Group by parameter value
        sensitivity = results_df.groupby(parameter_name).agg({
            'win_rate': 'mean',
            'profit_factor': 'mean',
            'total_trades': 'mean',
            'total_pnl': 'sum'
        }).reset_index()
        
        return sensitivity
