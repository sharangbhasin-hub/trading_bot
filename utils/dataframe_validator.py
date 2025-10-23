"""
DataFrame Validation Utility
Ensures data quality and prevents crashes from invalid data
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class DataFrameValidator:
    """
    Validates DataFrame structure and data quality
    """
    
    # Required columns for OHLC data
    REQUIRED_OHLC_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
    
    # Optional columns that may be present
    OPTIONAL_COLUMNS = ['timestamp', 'datetime', 'date']
    
    @staticmethod
    def validate_ohlc(df: pd.DataFrame, 
                      strict: bool = True,
                      min_rows: int = 20) -> Tuple[bool, List[str]]:
        """
        Validate OHLC DataFrame
        
        Args:
            df: DataFrame to validate
            strict: If True, raise errors on validation failure
            min_rows: Minimum required rows
            
        Returns:
            Tuple of (is_valid: bool, error_messages: List[str])
        """
        errors = []
        
        # Check if DataFrame is None or empty
        if df is None:
            errors.append("DataFrame is None")
            if strict:
                raise ValueError("DataFrame is None")
            return (False, errors)
        
        if df.empty:
            errors.append("DataFrame is empty")
            if strict:
                raise ValueError("DataFrame is empty")
            return (False, errors)
        
        # Check minimum rows
        if len(df) < min_rows:
            errors.append(f"Insufficient data: {len(df)} rows (minimum {min_rows} required)")
            if strict:
                raise ValueError(f"Insufficient data: {len(df)} rows")
            return (False, errors)
        
        # Check required columns
        missing_cols = set(DataFrameValidator.REQUIRED_OHLC_COLUMNS) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            if strict:
                raise ValueError(f"Missing required columns: {missing_cols}")
            return (False, errors)
        
        # Validate data types
        for col in DataFrameValidator.REQUIRED_OHLC_COLUMNS:
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Column '{col}' must be numeric, got {df[col].dtype}")
                if strict:
                    raise TypeError(f"Column '{col}' must be numeric")
        
        # Check for NaN values
        for col in DataFrameValidator.REQUIRED_OHLC_COLUMNS:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                errors.append(f"Column '{col}' contains {nan_count} NaN values")
                if strict:
                    raise ValueError(f"Column '{col}' contains NaN values")
        
        # Check for infinity values
        for col in DataFrameValidator.REQUIRED_OHLC_COLUMNS:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                errors.append(f"Column '{col}' contains {inf_count} infinity values")
                if strict:
                    raise ValueError(f"Column '{col}' contains infinity values")
        
        # Validate OHLC relationships
        invalid_ohlc = DataFrameValidator._validate_ohlc_relationships(df)
        if invalid_ohlc:
            errors.extend(invalid_ohlc)
            if strict:
                raise ValueError(f"Invalid OHLC relationships: {invalid_ohlc}")
        
        # Check for negative prices
        for col in ['open', 'high', 'low', 'close']:
            negative_count = (df[col] <= 0).sum()
            if negative_count > 0:
                errors.append(f"Column '{col}' contains {negative_count} non-positive values")
                if strict:
                    raise ValueError(f"Column '{col}' contains non-positive values")
        
        # Check volume
        if (df['volume'] < 0).any():
            errors.append("Volume contains negative values")
            if strict:
                raise ValueError("Volume cannot be negative")
        
        return (len(errors) == 0, errors)
    
    @staticmethod
    def _validate_ohlc_relationships(df: pd.DataFrame) -> List[str]:
        """
        Validate that OHLC relationships are correct
        (high >= open, close; low <= open, close)
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        # High should be >= open and close
        high_invalid = ((df['high'] < df['open']) | (df['high'] < df['close'])).sum()
        if high_invalid > 0:
            errors.append(f"{high_invalid} bars have high < open or close")
        
        # Low should be <= open and close
        low_invalid = ((df['low'] > df['open']) | (df['low'] > df['close'])).sum()
        if low_invalid > 0:
            errors.append(f"{low_invalid} bars have low > open or close")
        
        # High should be >= low
        hl_invalid = (df['high'] < df['low']).sum()
        if hl_invalid > 0:
            errors.append(f"{hl_invalid} bars have high < low")
        
        return errors
    
    @staticmethod
    def sanitize_dataframe(df: pd.DataFrame, 
                          fill_method: str = 'ffill') -> pd.DataFrame:
        """
        Clean and sanitize DataFrame
        
        Args:
            df: DataFrame to sanitize
            fill_method: Method to fill NaN values ('ffill', 'bfill', 'interpolate')
            
        Returns:
            Sanitized DataFrame
        """
        if df is None or df.empty:
            return df
        
        df_clean = df.copy()
        
        # Replace infinity with NaN
        df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Fill NaN values
        if fill_method == 'ffill':
            df_clean.fillna(method='ffill', inplace=True)
            df_clean.fillna(method='bfill', inplace=True)  # Backfill any remaining
        elif fill_method == 'bfill':
            df_clean.fillna(method='bfill', inplace=True)
            df_clean.fillna(method='ffill', inplace=True)  # Forward fill any remaining
        elif fill_method == 'interpolate':
            df_clean.interpolate(method='linear', inplace=True)
            df_clean.fillna(method='ffill', inplace=True)  # Fill any remaining
        
        # Ensure OHLC relationships are correct
        df_clean = DataFrameValidator._fix_ohlc_relationships(df_clean)
        
        return df_clean
    
    @staticmethod
    def _fix_ohlc_relationships(df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix invalid OHLC relationships
        
        Args:
            df: DataFrame with potentially invalid OHLC
            
        Returns:
            DataFrame with corrected OHLC
        """
        df_fixed = df.copy()
        
        # Ensure high is max of (open, close, high)
        df_fixed['high'] = df_fixed[['open', 'high', 'close']].max(axis=1)
        
        # Ensure low is min of (open, close, low)
        df_fixed['low'] = df_fixed[['open', 'low', 'close']].min(axis=1)
        
        return df_fixed
    
    @staticmethod
    def validate_index(df: pd.DataFrame, 
                      index: int,
                      min_lookback: int = 0) -> Tuple[bool, str]:
        """
        Validate if index is valid for data access
        
        Args:
            df: DataFrame
            index: Index to validate
            min_lookback: Minimum lookback required
            
        Returns:
            Tuple of (is_valid: bool, error_message: str)
        """
        if df is None or df.empty:
            return (False, "DataFrame is None or empty")
        
        if index < 0:
            return (False, f"Index {index} is negative")
        
        if index >= len(df):
            return (False, f"Index {index} exceeds DataFrame length {len(df)}")
        
        if index < min_lookback:
            return (False, f"Index {index} is less than required lookback {min_lookback}")
        
        return (True, "")
    
    @staticmethod
    def validate_slice(df: pd.DataFrame,
                      start: int,
                      end: int) -> Tuple[bool, str]:
        """
        Validate DataFrame slice
        
        Args:
            df: DataFrame
            start: Start index
            end: End index
            
        Returns:
            Tuple of (is_valid: bool, error_message: str)
        """
        if df is None or df.empty:
            return (False, "DataFrame is None or empty")
        
        if start < 0 or end < 0:
            return (False, f"Negative indices: start={start}, end={end}")
        
        if start > end:
            return (False, f"Start index {start} > end index {end}")
        
        if end > len(df):
            return (False, f"End index {end} exceeds DataFrame length {len(df)}")
        
        if end - start < 1:
            return (False, f"Slice too small: {end - start} rows")
        
        return (True, "")

