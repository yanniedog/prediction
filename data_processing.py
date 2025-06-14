import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import logging

# Import data_manager module
import data_manager

logger = logging.getLogger(__name__)

def validate_data(data: pd.DataFrame, is_normalized: bool = False, min_data_points: int = 100) -> Tuple[bool, str]:
    """Validate input data for required format and quality.
    
    Args:
        data (pd.DataFrame): Input data to validate
        is_normalized (bool): Whether the data is normalized. If True, validates relative relationships instead of absolute values.
        min_data_points (int): Minimum number of data points required (default 100, but can be overridden for tests)
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
        - is_valid: True if data is valid, False otherwise
        - error_message: Description of validation error if any
        
    Validation Rules:
    1. Must be a pandas DataFrame
    2. Must have required columns: open, high, low, close, volume
    3. No NaN values allowed
    4. No duplicate timestamps
    5. Timestamps must be monotonically increasing
    6. Price relationships must be valid:
       - For non-normalized data: low <= open/close <= high
       - For normalized data: relative relationships maintained
    7. No negative values in volume (for non-normalized data)
    8. Minimum data points required (configurable)
    9. No gaps larger than 7 days in data
    """
    try:
        # Check if input is a DataFrame
        if not isinstance(data, pd.DataFrame):
            return False, "Input must be a pandas DataFrame"
            
        if data.empty:
            return False, "Empty DataFrame provided"
            
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
            
        # Check for NaN values
        nan_cols = data[required_cols].columns[data[required_cols].isna().any()].tolist()
        if nan_cols:
            return False, f"NaN values found in columns: {nan_cols}"
            
        # Check for duplicate timestamps
        if data.index.duplicated().any():
            return False, "Duplicate timestamps found in data"
            
        # Check for non-monotonic timestamps
        if not data.index.is_monotonic_increasing:
            return False, "Timestamps must be in ascending order"
            
        # Check price relationships
        price_cols = ['open', 'high', 'low', 'close']
        if is_normalized:
            # For normalized data, check relative relationships
            for idx, row in data.iterrows():
                # Get the order of values
                values = [row['low'], row['open'], row['close'], row['high']]
                sorted_values = sorted(values)
                # Check if the order matches expected relationships
                if not (sorted_values[0] == row['low'] and sorted_values[-1] == row['high']):
                    return False, f"Invalid normalized price relationship at {idx}: relative order of prices not maintained"
        else:
            # For non-normalized data, check absolute relationships
            for idx, row in data.iterrows():
                if not (row['low'] <= row['open'] <= row['high'] and 
                       row['low'] <= row['close'] <= row['high']):
                    return False, f"Invalid price relationship at {idx}: low <= open/close <= high must be true"
                
        # Check for negative values in volume (only for non-normalized data)
        if not is_normalized and (data['volume'] < 0).any():
            return False, "Negative values found in columns: ['volume']"
            
        # Check minimum data points (configurable)
        if len(data) < min_data_points:
            return False, f"Insufficient data points (minimum {min_data_points} required)"
            
        # Check for large gaps
        if len(data) > 1 and isinstance(data.index, pd.DatetimeIndex):
            time_diffs = data.index.to_series().diff()
            max_gap = time_diffs.max()
            if max_gap > pd.Timedelta(days=7):
                return False, "Large gap detected in data"
                
        return True, "Data validation successful"
        
    except Exception as e:
        return False, f"Error during data validation: {str(e)}"

def process_data(data: pd.DataFrame, min_data_points: int = 10) -> pd.DataFrame:
    """Process and clean the input data.
    
    Args:
        data (pd.DataFrame): Input data to process
        min_data_points (int): Minimum data points required (default 10 for tests, can be overridden)
        
    Returns:
        pd.DataFrame: Processed data
        
    Processing Steps:
    1. Sort by timestamp
    2. Convert columns to numeric
    3. Handle NaN values
    4. Remove invalid price relationships
    5. Remove negative values
    """
    try:
        # Make a copy to avoid modifying original
        df = data.copy()
        
        # Sort by timestamp
        df = df.sort_index()
        
        # Convert columns to numeric
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Handle NaN values
        df = df.ffill().bfill()
        
        # Remove any remaining NaN values
        df = df.dropna()
        
        # Remove invalid price relationships
        df = df[
            (df['low'] <= df['open']) & 
            (df['open'] <= df['high']) & 
            (df['low'] <= df['close']) & 
            (df['close'] <= df['high'])
        ]
        
        # Remove negative values
        for col in numeric_cols:
            df = df[df[col] >= 0]
            
        # Validate final result with configurable minimum data points
        is_valid, message = validate_data(df, min_data_points=min_data_points)
        if not is_valid:
            raise ValueError(f"Data processing resulted in invalid data: {message}")
            
        return df
        
    except Exception as e:
        raise ValueError(f"Error processing data: {str(e)}")

def _select_data_source_and_lag(choice: int = None, max_lag: int = None) -> Tuple[Path, str, str, pd.DataFrame, int, int, int, str]:
    """Select data source and calculate maximum lag."""
    try:
        # Get data source
        db_path, symbol, timeframe = data_manager.manage_data_source(choice=choice)
        
        # Load and validate data
        data = data_manager.load_data(db_path, symbol, timeframe)
        if data is None or data.empty:
            raise ValueError("No data available")
            
        # Get symbol and timeframe IDs
        symbol_id = data_manager.get_symbol_id(db_path, symbol)
        timeframe_id = data_manager.get_timeframe_id(db_path, timeframe)
        
        # Validate data
        is_valid, message = validate_data(data)
        if not is_valid:
            raise ValueError(message)
            
        # Process data
        data = process_data(data)
        
        # Calculate maximum lag if not provided
        if max_lag is None:
            max_lag = calculate_max_lag(data)
        
        # Format date range in expected format
        start_date = data.index.min().strftime('%Y-%m-%d')
        end_date = data.index.max().strftime('%Y-%m-%d')
        data_daterange = f"{start_date}-{end_date}"
        
        return db_path, symbol, timeframe, data, max_lag, symbol_id, timeframe_id, data_daterange
        
    except Exception as e:
        logger.error(f"Error in data source selection: {str(e)}")
        raise

def validate_required_columns_and_nans(data: pd.DataFrame) -> Tuple[bool, str]:
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"
    nan_cols = data[required_cols].columns[data[required_cols].isna().any()].tolist()
    if nan_cols:
        return False, f"NaN values found in columns: {nan_cols}"
    if data.empty:
        return False, "Empty DataFrame after normalization"
    return True, "OK"

def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate DataFrame for data integrity and price relationships.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, message) where is_valid is a boolean indicating if the data is valid,
        and message is a string describing any validation errors found.
    """
    if df.empty:
        return False, "DataFrame is empty"
        
    # Check for required columns
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"
        
    # Check for NaN values
    if df[required_cols].isnull().any().any():
        return False, "Found NaN values in price columns"
        
    # Check for duplicate timestamps
    if isinstance(df.index, pd.DatetimeIndex):
        if not df.index.is_monotonic_increasing:
            return False, "Timestamps must be in ascending order"
        if df.index.duplicated().any():
            return False, "Duplicate timestamps found in data"
            
    # Check for sufficient data points
    if len(df) < 30:  # Minimum required for meaningful analysis
        return False, "Insufficient data points (minimum 30 required)"
        
    # Check for large gaps in data
    if isinstance(df.index, pd.DatetimeIndex):
        time_diff = df.index.to_series().diff()
        max_gap = time_diff.max()
        if max_gap > pd.Timedelta(days=7):  # Adjust threshold as needed
            return False, f"Large gaps detected in data (max gap: {max_gap})"
            
    # Validate price relationships
    for idx, row in df.iterrows():
        if not (row['low'] <= row['open'] <= row['high'] and 
                row['low'] <= row['close'] <= row['high']):
            return False, f"Invalid price relationship at {idx}: low <= open/close <= high must be true"
            
    return True, "Data validation passed"

def calculate_max_lag(data: pd.DataFrame) -> int:
    """Calculate maximum lag based on data characteristics."""
    if data is None or data.empty:
        return 10  # Default value
    
    # Simple heuristic: use 10% of data length as max lag, but at least 5 and at most 50
    max_lag = max(5, min(50, int(len(data) * 0.1)))
    return max_lag 