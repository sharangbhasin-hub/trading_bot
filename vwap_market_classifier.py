"""
Market Condition Classifier for VWAP-Strangle Strategies
=========================================================
Determines whether market conditions favor SELLING or BUYING strategy.
Implements Professional Enhancement #6 from analysis.

Decision Logic:
- SELLING: Range-bound, low gap, no breakout, high VIX
- BUYING: Trending, large gap, breakout, low VIX

Author: Trading System
Date: October 25, 2025
"""

import pandas as pd
from typing import Optional, Dict, Tuple
from datetime import datetime, time as dt_time
import logging
from kite_handler import get_kite_handler
from india_vix_fetcher import IndiaVIXFetcher

logger = logging.getLogger(__name__)

class VWAPMarketClassifier:
    """
    Classifies market conditions to determine optimal VWAP-Strangle strategy.
    """
    
    def __init__(self):
        """Initialize classifier"""
        self.kite = None
        self.vix_fetcher = IndiaVIXFetcher()
    
    def _initialize_kite(self) -> bool:
        """Initialize Kite connection"""
        if self.kite is None:
            self.kite = get_kite_handler()
            if self.kite is None or not self.kite.connected:
                logger.error("Kite handler not initialized")
                return False
        return True
    
    def classify_market(self, symbol: str) -> Dict:
        """
        Classify market conditions for strategy selection.
        Should be run before market open or at 9:30 AM.
        
        Args:
            symbol: Index symbol (e.g., 'NIFTY 50', 'SENSEX')
        
        Returns:
            dict: {
                'recommended_strategy': 'SELLING' or 'BUYING' or None,
                'confidence': float (0-100),
                'conditions': dict,
                'reason': str
            }
        """
        if not self._initialize_kite():
            return self._error_result("Kite not initialized")
        
        # Gather all conditions
        conditions = {}
        
        # 1. Gap analysis
        gap_pct = self._calculate_gap_open(symbol)
        conditions['gap_open_pct'] = gap_pct
        conditions['is_gap_open'] = abs(gap_pct) >= 0.5
        
        # 2. Range-bound vs trending
        is_range_bound = self._check_range_bound(symbol)
        conditions['is_range_bound'] = is_range_bound
        conditions['is_trending'] = not is_range_bound
        
        # 3. Breakout detection
        is_breakout = self._detect_breakout(symbol)
        conditions['is_breakout'] = is_breakout
        
        # 4. India VIX
        vix = self.vix_fetcher.get_current_vix()
        conditions['india_vix'] = vix
        conditions['vix_high'] = vix >= 15 if vix else False
        conditions['vix_low'] = vix <= 13 if vix else False
        
        # 5. Global cues
        global_sentiment = self._check_global_cues()
        conditions['global_sentiment'] = global_sentiment
        
        # Decision logic
        return self._make_decision(conditions)
    
    def _calculate_gap_open(self, symbol: str) -> float:
        """
        Calculate gap % between previous close and current open.
        
        Returns:
            float: Gap percentage (positive = gap up, negative = gap down)
        """
        try:
            # Get quote
            quote = self.kite.kite.quote(f"NSE:{symbol}")
            
            if not quote or f"NSE:{symbol}" not in quote:
                return 0.0
            
            data = quote[f"NSE:{symbol}"]
            ohlc = data.get('ohlc', {})
            
            prev_close = ohlc.get('close', None)
            current_price = data.get('last_price', None)
            
            if prev_close and current_price:
                gap_pct = ((current_price - prev_close) / prev_close) * 100
                logger.info(f"{symbol} gap: {gap_pct:.2f}%")
                return gap_pct
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating gap: {e}")
            return 0.0
    
    def _check_range_bound(self, symbol: str, lookback_days: int = 5) -> bool:
        """
        Check if market is range-bound on higher timeframe (1H chart).
        
        Returns:
            bool: True if range-bound, False if trending
        """
        try:
            from datetime import timedelta
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Get hourly data
            historical_data = self.kite.get_historical_data(
                symbol=symbol,
                interval='60minute',
                from_date=start_date.strftime('%Y-%m-%d'),
                to_date=end_date.strftime('%Y-%m-%d')
            )
            
            if historical_data.empty:
                return True  # Default to range-bound if no data
            
            # Calculate range
            high_max = historical_data['high'].max()
            low_min = historical_data['low'].min()
            range_pct = ((high_max - low_min) / low_min) * 100
            
            # If range < 3%, consider it range-bound
            is_range = range_pct < 3.0
            logger.info(f"{symbol} range: {range_pct:.2f}% - Range-bound: {is_range}")
            
            return is_range
            
        except Exception as e:
            logger.error(f"Error checking range: {e}")
            return True  # Default to range-bound
    
    def _detect_breakout(self, symbol: str) -> bool:
        """
        Detect if market is breaking major support/resistance.
        
        Returns:
            bool: True if breakout detected
        """
        try:
            from datetime import timedelta
            
            # Get recent data (last 10 days)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=10)
            
            historical_data = self.kite.get_historical_data(
                symbol=symbol,
                interval='day',
                from_date=start_date.strftime('%Y-%m-%d'),
                to_date=end_date.strftime('%Y-%m-%d')
            )
            
            if historical_data.empty or len(historical_data) < 5:
                return False
            
            # Simple breakout: Current price > recent high or < recent low
            recent_high = historical_data['high'].iloc[:-1].max()  # Exclude today
            recent_low = historical_data['low'].iloc[:-1].min()
            current_price = historical_data['close'].iloc[-1]
            
            breakout_up = current_price > recent_high
            breakout_down = current_price < recent_low
            
            is_breakout = breakout_up or breakout_down
            
            if is_breakout:
                direction = "UP" if breakout_up else "DOWN"
                logger.info(f"{symbol} breakout detected: {direction}")
            
            return is_breakout
            
        except Exception as e:
            logger.error(f"Error detecting breakout: {e}")
            return False
    
    def _check_global_cues(self) -> str:
        """
        Check global market sentiment.
        
        Returns:
            str: 'BULLISH', 'BEARISH', or 'NEUTRAL'
        """
        # Simplified: Check SGX Nifty or US futures
        # In production, integrate with news_fetcher or external API
        
        try:
            # Placeholder: Check NIFTY FUTURES
            # In real implementation, check overnight moves
            return 'NEUTRAL'
            
        except Exception as e:
            logger.error(f"Error checking global cues: {e}")
            return 'NEUTRAL'
    
    def _make_decision(self, conditions: Dict) -> Dict:
        """
        UPDATED: Analyst's concrete threshold logic (not scoring).
        
        SELLING Day Criteria (ALL concrete):
        1. VIX >= 15
        2. Gap <= 0.3%
        3. Range-bound = True
        4. No breakout = True
        
        BUYING Day Criteria (ALL concrete):
        1. VIX <= 13
        2. Gap >= 0.5%
        3. Trending = True (not range-bound)
        4. Breakout = True
        
        Need: 3/4 conditions met minimum
        """
        vix = conditions.get('india_vix')
        gap_pct = conditions.get('gap_open_pct', 0)
        is_range_bound = conditions.get('is_range_bound', False)
        is_breakout = conditions.get('is_breakout', False)
        
        # SELLING CONDITIONS (Analyst's concrete thresholds)
        selling_conditions = {
            'vix_check': vix >= 15 if vix else False,  # CONCRETE: VIX >= 15
            'gap_check': abs(gap_pct) <= 0.3,  # CONCRETE: Gap <= 0.3%
            'range_check': is_range_bound,  # Must be range-bound
            'no_breakout': not is_breakout  # No breakout
        }
        
        # BUYING CONDITIONS (Analyst's concrete thresholds)
        buying_conditions = {
            'vix_check': vix <= 13 if vix else False,  # CONCRETE: VIX <= 13
            'gap_check': abs(gap_pct) >= 0.5,  # CONCRETE: Gap >= 0.5%
            'trending_check': not is_range_bound,  # Must be trending
            'breakout_check': is_breakout  # Must have breakout
        }
        
        # Count conditions met
        selling_score = sum(selling_conditions.values())
        buying_score = sum(buying_conditions.values())
        
        # Analyst's rule: Need at least 3/4 conditions
        min_required = 3
        
        # Build reason strings
        selling_reasons = []
        if selling_conditions['vix_check']:
            selling_reasons.append(f"VIX {vix:.1f} >= 15 ✓")
        if selling_conditions['gap_check']:
            selling_reasons.append(f"Gap {gap_pct:.2f}% <= 0.3% ✓")
        if selling_conditions['range_check']:
            selling_reasons.append("Range-bound ✓")
        if selling_conditions['no_breakout']:
            selling_reasons.append("No breakout ✓")
        
        buying_reasons = []
        if buying_conditions['vix_check']:
            buying_reasons.append(f"VIX {vix:.1f} <= 13 ✓")
        if buying_conditions['gap_check']:
            buying_reasons.append(f"Gap {gap_pct:.2f}% >= 0.5% ✓")
        if buying_conditions['trending_check']:
            buying_reasons.append("Trending ✓")
        if buying_conditions['breakout_check']:
            buying_reasons.append("Breakout ✓")
        
        # Decision logic
        if selling_score >= min_required and selling_score > buying_score:
            return {
                'recommended_strategy': 'SELLING',
                'confidence': (selling_score / 4) * 100,
                'conditions': conditions,
                'conditions_met': selling_conditions,
                'score': f'{selling_score}/4',
                'selling_score': selling_score,
                'buying_score': buying_score,
                'reason': f"SELLING day: {', '.join(selling_reasons)} (Met {selling_score}/4 conditions)"
            }
        
        elif buying_score >= min_required and buying_score > selling_score:
            return {
                'recommended_strategy': 'BUYING',
                'confidence': (buying_score / 4) * 100,
                'conditions': conditions,
                'conditions_met': buying_conditions,
                'score': f'{buying_score}/4',
                'selling_score': selling_score,
                'buying_score': buying_score,
                'reason': f"BUYING day: {', '.join(buying_reasons)} (Met {buying_score}/4 conditions)"
            }
        
        else:
            return {
                'recommended_strategy': None,
                'confidence': 0,
                'conditions': conditions,
                'score': f'Selling {selling_score}/4, Buying {buying_score}/4',
                'selling_score': selling_score,
                'buying_score': buying_score,
                'reason': f"NO TRADE: Conditions not met (need 3/4). Selling: {selling_score}/4, Buying: {buying_score}/4"
            }
    
    def _error_result(self, error_msg: str) -> Dict:
        """Return error result"""
        return {
            'recommended_strategy': None,
            'confidence': 0,
            'conditions': {},
            'reason': error_msg
        }

# ============================================================================
# STANDALONE FUNCTION
# ============================================================================

def get_strategy_recommendation(symbol: str = 'NIFTY 50') -> str:
    """
    Quick function to get strategy recommendation.
    
    Returns:
        str: 'SELLING', 'BUYING', or None
    """
    classifier = VWAPMarketClassifier()
    result = classifier.classify_market(symbol)
    return result['recommended_strategy']
