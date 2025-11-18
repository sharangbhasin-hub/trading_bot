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
        Make final strategy recommendation based on all conditions.
        
        Logic from Professional Enhancement #6:
        SELLING: Range-bound + Small gap + No breakout + High VIX
        BUYING: Trending + Large gap + Breakout + Low VIX
        """
        selling_score = 0
        buying_score = 0
        reasons = []
        
        # Scoring system
        if conditions.get('is_range_bound'):
            selling_score += 2
            reasons.append("Market is range-bound (favor selling)")
        else:
            buying_score += 2
            reasons.append("Market is trending (favor buying)")
        
        if conditions.get('is_gap_open'):
            buying_score += 2
            reasons.append(f"Gap open detected: {conditions['gap_open_pct']:.2f}% (favor buying)")
        else:
            selling_score += 1
            reasons.append("No significant gap (favor selling)")
        
        if conditions.get('is_breakout'):
            buying_score += 2
            reasons.append("Breakout detected (favor buying)")
        else:
            selling_score += 1
            reasons.append("No breakout (favor selling)")
        
        if conditions.get('vix_high'):
            selling_score += 2
            reasons.append(f"VIX high: {conditions['india_vix']} (favor selling)")
        elif conditions.get('vix_low'):
            buying_score += 2
            reasons.append(f"VIX low: {conditions['india_vix']} (favor buying)")
        
        # Global sentiment
        if conditions.get('global_sentiment') == 'BULLISH':
            buying_score += 1
        elif conditions.get('global_sentiment') == 'BEARISH':
            buying_score += 1  # Can buy puts
        
        # Final decision
        max_score = max(selling_score, buying_score)
        
        if selling_score > buying_score and selling_score >= 4:
            strategy = 'SELLING'
            confidence = min(100, (selling_score / 7) * 100)
        elif buying_score > selling_score and buying_score >= 4:
            strategy = 'BUYING'
            confidence = min(100, (buying_score / 7) * 100)
        else:
            strategy = None
            confidence = 0
            reasons.append("Conditions not clear - NO TRADE recommended")
        
        return {
            'recommended_strategy': strategy,
            'confidence': round(confidence, 2),
            'conditions': conditions,
            'selling_score': selling_score,
            'buying_score': buying_score,
            'reason': '; '.join(reasons)
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
