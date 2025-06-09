import pandas as pd
from typing import Tuple

def validate_data(data: pd.DataFrame) -> None:
    """Validate data for processing."""
    if data is None or not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame")
    if data.empty:
        raise ValueError("Input data cannot be empty")
        
    # Check for required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
        
    # Check for invalid values
    for col in required_cols:
        if data[col].isnull().any():
            raise ValueError(f"Invalid values detected in data: {col} contains NaN values")
        if (data[col] <= 0).any() and col != 'volume':
            raise ValueError(f"Invalid values detected in data: {col} contains non-positive values")
        if (data[col] < 0).any() and col == 'volume':
            raise ValueError(f"Invalid values detected in data: {col} contains negative values")
            
    # Check for duplicate dates
    if data.index.duplicated().any():
        raise ValueError("Duplicate dates found in data")
        
    # Check for large gaps
    if isinstance(data.index, pd.DatetimeIndex):
        time_diff = data.index.to_series().diff()
        max_gap = time_diff.max()
        if max_gap > pd.Timedelta(days=7):  # Adjust threshold as needed
            raise ValueError(f"Large gaps detected in data: maximum gap is {max_gap}")
            
    # Check for non-monotonic dates
    if not data.index.is_monotonic_increasing:
        raise ValueError("Data index is not monotonically increasing")
        
    # Check for price consistency
    if not ((data['high'] >= data['low']).all() and 
            (data['high'] >= data['open']).all() and 
            (data['high'] >= data['close']).all() and
            (data['low'] <= data['open']).all() and 
            (data['low'] <= data['close']).all()):
        raise ValueError("Invalid price relationships detected in data")

def process_data(data: pd.DataFrame, validate: bool = True) -> pd.DataFrame:
    """Process and validate data."""
    if validate:
        validate_data(data)
        
    # Make a copy to avoid modifying original
    processed = data.copy()
    
    # Sort by date if not already sorted
    if not processed.index.is_monotonic_increasing:
        processed = processed.sort_index()
        
    # Handle any remaining NaN values
    processed = processed.fillna(method='ffill').fillna(method='bfill')
    
    # Ensure all numeric columns are float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in processed.columns:
            processed[col] = processed[col].astype(float)
            
    return processed

def _select_data_source_and_lag() -> Tuple[pd.DataFrame, str, str, int]:
    """Select data source and calculate lag."""
    # Get data source
    db_path, symbol, timeframe = data_manager.manage_data_source()
    
    # Load and validate data
    data = data_manager.load_data(db_path)
    if data is None or data.empty:
        raise ValueError("No data available")
        
    # Process and validate data
    try:
        data = process_data(data, validate=True)
    except ValueError as e:
        logger.error(f"Data validation failed: {e}")
        raise
        
    # Calculate max lag
    max_lag = calculate_max_lag(data)
    
    return data, symbol, timeframe, max_lag 