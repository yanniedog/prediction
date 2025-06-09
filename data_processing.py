import pandas as pd
from pathlib import Path
from typing import Tuple

def validate_data(data: pd.DataFrame) -> Tuple[bool, str]:
    """Validate input data for required format and quality.
    
    Args:
        data (pd.DataFrame): Input data to validate
        
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
    6. Price relationships must be valid (low <= open/close <= high)
    7. No negative values in any column
    8. Minimum 100 data points required
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
        for idx, row in data.iterrows():
            if not (row['low'] <= row['open'] <= row['high'] and 
                   row['low'] <= row['close'] <= row['high']):
                return False, f"Invalid price relationship at {idx}: low <= open/close <= high must be true"
                
        # Check for negative values
        neg_cols = data[required_cols].columns[(data[required_cols] < 0).any()].tolist()
        if neg_cols:
            return False, f"Negative values found in columns: {neg_cols}"
            
        # Check minimum data points
        if len(data) < 100:
            return False, "Insufficient data points (minimum 100 required)"
            
        # Check for large gaps
        if len(data) > 1 and isinstance(data.index, pd.DatetimeIndex):
            time_diffs = data.index.to_series().diff()
            max_gap = time_diffs.max()
            if max_gap > pd.Timedelta(days=7):
                return False, f"Large gap detected in data: {max_gap}"
                
        return True, "Data validation successful"
        
    except Exception as e:
        return False, f"Error during data validation: {str(e)}"

def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """Process and clean the input data.
    
    Args:
        data (pd.DataFrame): Input data to process
        
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
            
        # Validate final result
        is_valid, message = validate_data(df)
        if not is_valid:
            raise ValueError(f"Data processing resulted in invalid data: {message}")
            
        return df
        
    except Exception as e:
        raise ValueError(f"Error processing data: {str(e)}")

def _select_data_source_and_lag() -> Tuple[Path, str, str, int]:
    """Select data source and calculate maximum lag."""
    try:
        # Get data source
        db_path, symbol, timeframe = data_manager.manage_data_source()
        
        # Load and validate data
        data = data_manager.load_data(db_path, symbol, timeframe)
        if data is None or data.empty:
            raise ValueError("No data available")
            
        # Validate data
        is_valid, message = validate_data(data)
        if not is_valid:
            raise ValueError(message)
            
        # Process data
        data = process_data(data)
        
        # Calculate maximum lag
        max_lag = calculate_max_lag(data)
        
        return db_path, symbol, timeframe, max_lag
        
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