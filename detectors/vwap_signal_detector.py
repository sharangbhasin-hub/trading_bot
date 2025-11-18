"""
VWAP Signal Detector for Backtesting
=====================================
Detects VWAP crossover signals from historical data.
Used by backtest engine to generate signals on past data.

Author: Trading System (Analyst-Enhanced)
Date: November 18, 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from vwap_calculator import VWAPCalculator

logger = logging.getLogger(__name__)

class VWAPSignalDetector:
    """
    Detects VWAP crossover signals from historical option data.
    Designed for backtesting with complete historical data.
    """
    
    def __init__(self):
        """Initialize detector"""
        self.vwap_calculator = VWAPCalculator()
        logger.info("VWAP Signal Detector initialized")
    
    def prepare_data(self,
                     ce_df: pd.DataFrame,
                     pe_df: pd.DataFrame,
                     reset_daily: bool = True) -> pd.DataFrame:
        """
        Prepare combined option data with VWAP calculations.
        
        Args:
            ce_df: DataFrame with CE prices (columns: timestamp, close/last_price)
            pe_df: DataFrame with PE prices
            reset_daily: Reset VWAP calculation each trading day
        
        Returns:
            pd.DataFrame: Combined data with VWAP
        """
        # Merge CE and PE data
        combined = pd.merge(
            ce_df[['timestamp', 'close']].rename(columns={'close': 'ce_price'}),
            pe_df[['timestamp', 'close']].rename(columns={'close': 'pe_price'}),
            on='timestamp',
            how='inner'
        )
        
        combined['timestamp'] = pd.to_datetime(combined['timestamp'])
        combined = combined.set_index('timestamp').sort_index()
        
        # Calculate combined premium
        combined['combined_premium'] = combined['ce_price'] + combined['pe_price']
        
        # Calculate VWAP
        if reset_daily:
            # Reset VWAP at start of each day
            combined['date'] = combined.index.date
            combined['vwap'] = combined.groupby('date').apply(
                lambda x: self._calculate_vwap_series(x)
            ).reset_index(level=0, drop=True)
        else:
            # Continuous VWAP
            combined['vwap'] = self._calculate_vwap_series(combined)
        
        return combined
    
    def _calculate_vwap_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate VWAP for a dataframe segment.
        
        Args:
            df: DataFrame with combined_premium column
        
        Returns:
            pd.Series: VWAP values
        """
        # Use volume if available, else equal weight
        if 'volume' in df.columns:
            volume = df['volume']
        else:
            volume = 1.0
        
        typical_price = df['combined_premium']
        tp_volume = typical_price * volume
        
        cumulative_tp_volume = tp_volume.cumsum()
        cumulative_volume = volume.cumsum() if isinstance(volume, pd.Series) else len(df)
        
        vwap = cumulative_tp_volume / cumulative_volume
        
        return vwap
    
    def detect_crossovers(self,
                         prepared_data: pd.DataFrame,
                         direction: str = 'below') -> pd.DataFrame:
        """
        Detect all VWAP crossover points in historical data.
        
        Args:
            prepared_data: Output from prepare_data()
            direction: 'below' for selling, 'above' for buying
        
        Returns:
            pd.DataFrame: Rows where crossover occurred
        """
        df = prepared_data.copy()
        
        # Detect crossovers
        df['prev_premium'] = df['combined_premium'].shift(1)
        df['prev_vwap'] = df['vwap'].shift(1)
        
        if direction == 'below':
            # Premium crosses from above to below VWAP
            df['crossed'] = (
                (df['prev_premium'] > df['prev_vwap']) &  # Was above
                (df['combined_premium'] < df['vwap'])      # Now below
            )
        else:  # 'above'
            # Premium crosses from below to above VWAP
            df['crossed'] = (
                (df['prev_premium'] < df['prev_vwap']) &  # Was below
                (df['combined_premium'] > df['vwap'])      # Now above
            )
        
        # Extract crossover points
        crossovers = df[df['crossed'] == True].copy()
        
        # Add signal metadata
        crossovers['signal_type'] = 'SELL' if direction == 'below' else 'BUY'
        crossovers['crossover_direction'] = direction
        crossovers['entry_premium'] = crossovers['combined_premium']
        crossovers['entry_vwap'] = crossovers['vwap']
        
        logger.info(f"Detected {len(crossovers)} crossovers ({direction})")
        
        return crossovers
    
    def detect_with_momentum_filter(self,
                                   prepared_data: pd.DataFrame,
                                   ce_df: pd.DataFrame,
                                   pe_df: pd.DataFrame,
                                   min_move_pct: float = 15.0,
                                   lookback_minutes: int = 5) -> pd.DataFrame:
        """
        Detect crossovers with momentum filter (for buying strategy).
        Analyst's requirement: Option must move >15% in 5 minutes.
        
        Args:
            prepared_data: Combined data with VWAP
            ce_df: Original CE dataframe
            pe_df: Original PE dataframe
            min_move_pct: Minimum % move required
            lookback_minutes: Lookback period for momentum
        
        Returns:
            pd.DataFrame: Filtered crossovers with momentum
        """
        # First detect all 'above' crossovers
        crossovers = self.detect_crossovers(prepared_data, direction='above')
        
        if crossovers.empty:
            return crossovers
        
        # For each crossover, check momentum
        filtered_crossovers = []
        
        for idx, row in crossovers.iterrows():
            # Get 5-minute window before crossover
            window_start = idx - pd.Timedelta(minutes=lookback_minutes)
            
            # Get CE and PE prices in window
            ce_window = ce_df[(ce_df.index >= window_start) & (ce_df.index <= idx)]
            pe_window = pe_df[(pe_df.index >= window_start) & (pe_df.index <= idx)]
            
            if len(ce_window) < 2 or len(pe_window) < 2:
                continue
            
            # Calculate % change
            ce_change = ((ce_window['close'].iloc[-1] - ce_window['close'].iloc[0]) / 
                        ce_window['close'].iloc[0] * 100)
            pe_change = ((pe_window['close'].iloc[-1] - pe_window['close'].iloc[0]) / 
                        pe_window['close'].iloc[0] * 100)
            
            # Check if either meets threshold
            if abs(ce_change) >= min_move_pct or abs(pe_change) >= min_move_pct:
                row_data = row.to_dict()
                row_data['ce_momentum_pct'] = ce_change
                row_data['pe_momentum_pct'] = pe_change
                row_data['chosen_leg'] = 'CE' if abs(ce_change) > abs(pe_change) else 'PE'
                row_data['momentum_pct'] = max(abs(ce_change), abs(pe_change))
                filtered_crossovers.append(row_data)
        
        result = pd.DataFrame(filtered_crossovers)
        
        logger.info(f"Momentum filter: {len(crossovers)} → {len(result)} signals (>{min_move_pct}% move)")
        
        return result
    
    def generate_signals_for_backtest(self,
                                     ce_df: pd.DataFrame,
                                     pe_df: pd.DataFrame,
                                     strategy_type: str = 'SELLING',
                                     apply_momentum_filter: bool = True) -> List[Dict]:
        """
        Generate complete signal list for backtesting.
        
        Args:
            ce_df: Call option historical data
            pe_df: Put option historical data
            strategy_type: 'SELLING' or 'BUYING'
            apply_momentum_filter: Apply momentum filter for buying
        
        Returns:
            list: List of signal dictionaries
        """
        # Prepare data
        prepared = self.prepare_data(ce_df, pe_df, reset_daily=True)
        
        # Detect crossovers
        if strategy_type == 'SELLING':
            crossovers = self.detect_crossovers(prepared, direction='below')
        else:
            if apply_momentum_filter:
                crossovers = self.detect_with_momentum_filter(
                    prepared, ce_df, pe_df,
                    min_move_pct=15.0,
                    lookback_minutes=5
                )
            else:
                crossovers = self.detect_crossovers(prepared, direction='above')
        
        # Convert to signal list
        signals = []
        for idx, row in crossovers.iterrows():
            signal = {
                'timestamp': idx,
                'signal_type': row['signal_type'],
                'entry_premium': row['entry_premium'],
                'entry_vwap': row['entry_vwap'],
                'ce_price': row['ce_price'],
                'pe_price': row['pe_price'],
                'strategy_type': strategy_type
            }
            
            # Add momentum data if available
            if 'momentum_pct' in row:
                signal['momentum_pct'] = row['momentum_pct']
                signal['chosen_leg'] = row['chosen_leg']
            
            signals.append(signal)
        
        logger.info(f"Generated {len(signals)} signals for {strategy_type} strategy")
        
        return signals
    
    def calculate_signal_quality_metrics(self, signals: List[Dict]) -> Dict:
        """
        Calculate quality metrics for detected signals.
        Helps validate backtest data quality.
        
        Args:
            signals: List of signals from generate_signals_for_backtest()
        
        Returns:
            dict: Quality metrics
        """
        if not signals:
            return {'signal_count': 0}
        
        df = pd.DataFrame(signals)
        
        # Time distribution
        df['hour'] = df['timestamp'].dt.hour
        hourly_dist = df['hour'].value_counts().to_dict()
        
        # Signal frequency
        df['date'] = df['timestamp'].dt.date
        daily_signal_count = df.groupby('date').size()
        
        metrics = {
            'total_signals': len(signals),
            'avg_signals_per_day': daily_signal_count.mean(),
            'max_signals_per_day': daily_signal_count.max(),
            'min_signals_per_day': daily_signal_count.min(),
            'hourly_distribution': hourly_dist,
            'avg_entry_premium': df['entry_premium'].mean(),
            'avg_vwap': df['entry_vwap'].mean()
        }
        
        # Momentum metrics (for buying)
        if 'momentum_pct' in df.columns:
            metrics['avg_momentum_pct'] = df['momentum_pct'].mean()
            metrics['max_momentum_pct'] = df['momentum_pct'].max()
        
        return metrics

# ============================================================================
# STANDALONE FUNCTIONS
# ============================================================================

def quick_signal_detection(ce_df: pd.DataFrame,
                          pe_df: pd.DataFrame,
                          strategy_type: str = 'SELLING') -> int:
    """
    Quick function to count signals in historical data.
    
    Args:
        ce_df: CE dataframe
        pe_df: PE dataframe
        strategy_type: 'SELLING' or 'BUYING'
    
    Returns:
        int: Number of signals detected
    """
    detector = VWAPSignalDetector()
    signals = detector.generate_signals_for_backtest(ce_df, pe_df, strategy_type)
    return len(signals)
