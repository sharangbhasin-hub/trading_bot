"""
Unit Tests for VWAP Strategies
===============================
Comprehensive test suite for all VWAP strategy components.

Run with: pytest test_vwap_strategies.py -v

Author: Trading System
Date: November 18, 2025
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vwap_calculator import VWAPCalculator
from vwap_position_sizer import VWAPPositionSizer
from vwap_daily_risk_limiter import VWAPDailyRiskLimiter
from vwap_cost_tracker import VWAPCostTracker
from config_vwap_strangle import (
    calculate_position_size,
    get_strategy_for_market_condition
)

class TestVWAPCalculator(unittest.TestCase):
    """Test VWAP calculation logic"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calculator = VWAPCalculator()
    
    def test_combined_premium_calculation(self):
        """Test combined premium calculation"""
        ce_price = 100.0
        pe_price = 50.0
        
        combined = self.calculator.calculate_combined_premium(ce_price, pe_price)
        
        self.assertEqual(combined, 150.0)
    
    def test_vwap_update(self):
        """Test VWAP update with new data"""
        timestamp = datetime.now()
        ce_price = 100.0
        pe_price = 50.0
        
        vwap = self.calculator.update(timestamp, ce_price, pe_price)
        
        # First update should equal combined premium
        self.assertEqual(vwap, 150.0)
    
    def test_vwap_crossover_below(self):
        """Test crossover detection (below)"""
        # Add data points
        timestamps = [datetime.now() + timedelta(seconds=i) for i in range(5)]
        
        # Premium starts above VWAP, then crosses below
        prices = [(110, 50), (105, 50), (100, 50), (95, 50), (90, 50)]
        
        for ts, (ce, pe) in zip(timestamps, prices):
            self.calculator.update(ts, ce, pe)
        
        # Check crossover
        crossover = self.calculator.check_crossover(140, direction='below')
        
        self.assertIsInstance(crossover, dict)
        self.assertIn('crossed', crossover)
    
    def test_vwap_reset(self):
        """Test VWAP reset"""
        # Add some data
        self.calculator.update(datetime.now(), 100, 50)
        self.calculator.update(datetime.now(), 100, 50)
        
        # Reset
        self.calculator.reset()
        
        # Should have no data
        self.assertEqual(len(self.calculator.vwap_history), 0)

class TestPositionSizer(unittest.TestCase):
    """Test position sizing logic"""
    
    def setUp(self):
        """Set up with ₹2,00,000 capital"""
        self.sizer = VWAPPositionSizer(total_capital=200000)
    
    def test_selling_position_size(self):
        """Test selling position size calculation"""
        # Analyst's example
        entry_premium = 150
        sl_premium = 195  # 30% above
        
        result = self.sizer.calculate_lots_for_selling(
            entry_premium=entry_premium,
            sl_premium=sl_premium,
            index='NIFTY',
            include_costs=True
        )
        
        # Should return valid result
        self.assertIn('lots', result)
        self.assertGreater(result['lots'], 0)
        self.assertLessEqual(result['risk_pct'], 1.0)  # Within 1%
    
    def test_buying_position_size(self):
        """Test buying position size with R:R"""
        entry_premium = 50
        target_premium = 70  # +20 points
        
        result = self.sizer.calculate_lots_for_buying(
            entry_premium=entry_premium,
            target_premium=target_premium,
            index='NIFTY',
            include_costs=True
        )
        
        self.assertIn('lots', result)
        self.assertEqual(result['risk_reward_ratio'], 2.0)
        self.assertLessEqual(result['risk_pct'], 1.0)
    
    def test_capital_allocation(self):
        """Test capital allocation tracking"""
        initial_available = self.sizer.get_available_capital()
        
        # Allocate position
        self.sizer.allocate_position('TEST_001', 50000)
        
        # Should reduce available capital
        after_allocation = self.sizer.get_available_capital()
        self.assertEqual(after_allocation, initial_available - 50000)
        
        # Release position
        self.sizer.release_position('TEST_001')
        
        # Should restore available capital
        after_release = self.sizer.get_available_capital()
        self.assertEqual(after_release, initial_available)

class TestDailyRiskLimiter(unittest.TestCase):
    """Test daily risk limiting"""
    
    def setUp(self):
        """Set up limiter"""
        self.limiter = VWAPDailyRiskLimiter(total_capital=200000)
    
    def test_trade_limit(self):
        """Test max trades per day limit"""
        # Should allow first trade
        check1 = self.limiter.can_take_trade()
        self.assertTrue(check1['allowed'])
        
        # Record 2 trades
        self.limiter.record_trade_entry('TRADE_001', 1000)
        self.limiter.record_trade_entry('TRADE_002', 1000)
        
        # Should NOT allow 3rd trade
        check2 = self.limiter.can_take_trade()
        self.assertFalse(check2['allowed'])
    
    def test_daily_loss_limit(self):
        """Test daily loss limit"""
        # Record large loss (3% of capital = ₹6,000)
        self.limiter.record_trade_entry('TRADE_001', 2000)
        self.limiter.record_trade_result('TRADE_001', -6000)
        
        # Should NOT allow new trade
        check = self.limiter.can_take_trade()
        self.assertFalse(check['allowed'])
        self.assertIn('loss limit', check['reason'].lower())
    
    def test_consecutive_losses(self):
        """Test consecutive loss tracking"""
        # Record 3 consecutive losses
        for i in range(3):
            trade_id = f'TRADE_{i:03d}'
            self.limiter.record_trade_entry(trade_id, 500)
            self.limiter.record_trade_result(trade_id, -500)
        
        # Should pause trading
        check = self.limiter.can_take_trade()
        self.assertFalse(check['allowed'])
        self.assertIn('consecutive', check['reason'].lower())

class TestCostTracker(unittest.TestCase):
    """Test transaction cost tracking"""
    
    def setUp(self):
        """Set up cost tracker"""
        self.tracker = VWAPCostTracker()
    
    def test_selling_costs(self):
        """Test selling cost calculation"""
        result = self.tracker.calculate_selling_costs(
            entry_premium=150,
            exit_premium=90,
            lot_size=25,
            num_legs=4
        )
        
        # Should have all cost components
        self.assertIn('slippage', result)
        self.assertIn('brokerage', result)
        self.assertIn('gst', result)
        self.assertIn('total_cost', result)
        
        # Total cost should be positive
        self.assertGreater(result['total_cost'], 0)
        
        # Should calculate P&L impact
        self.assertIn('gross_pnl', result)
        self.assertIn('net_pnl', result)
    
    def test_buying_costs(self):
        """Test buying cost calculation"""
        result = self.tracker.calculate_buying_costs(
            entry_premium=50,
            exit_premium=70,
            lot_size=25,
            num_legs=1
        )
        
        # Should be cheaper than selling (fewer legs)
        selling_cost = self.tracker.calculate_selling_costs(150, 90, 25, 4)
        
        self.assertLess(result['total_cost'], selling_cost['total_cost'])

class TestConfigFunctions(unittest.TestCase):
    """Test configuration helper functions"""
    
    def test_position_size_calculation(self):
        """Test position size helper function"""
        result = calculate_position_size(
            capital=200000,
            entry=150,
            sl=195,
            lot_size=25
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('lots', result)
        self.assertGreater(result['lots'], 0)
    
    def test_market_condition_strategy_selection(self):
        """Test strategy selection based on market conditions"""
        # Selling conditions
        result_selling = get_strategy_for_market_condition(
            india_vix=16.0,  # High VIX
            gap_open_pct=0.2,  # Small gap
            is_breakout=False,
            is_range_bound=True
        )
        
        self.assertEqual(result_selling['strategy'], 'SELLING')
        
        # Buying conditions
        result_buying = get_strategy_for_market_condition(
            india_vix=12.0,  # Low VIX
            gap_open_pct=0.6,  # Large gap
            is_breakout=True,
            is_range_bound=False
        )
        
        self.assertEqual(result_buying['strategy'], 'BUYING')
        
        # No clear signal
        result_none = get_strategy_for_market_condition(
            india_vix=14.0,
            gap_open_pct=0.4,
            is_breakout=False,
            is_range_bound=False
        )
        
        self.assertIsNone(result_none['strategy'])

# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
