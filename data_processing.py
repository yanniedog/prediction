import pandas as pd
from typing import Tuple, Path

def validate_data(data: pd.DataFrame) -> Tuple[bool, str]:
    """Validate input data for required format and quality."""
    try:
        # Check if input is a DataFrame
        if not isinstance(data, pd.DataFrame):
            return False, "Input must be a pandas DataFrame"
            
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
            
        # Check for NaN values
        if data[required_cols].isna().any().any():
            return False, "NaN values detected in data"
            
        # Check for duplicate dates
        if data.index.duplicated().any():
            return False, "Duplicate dates found in data"
            
        # Check for non-monotonic dates
        if not data.index.is_monotonic_increasing:
            return False, "Dates must be in ascending order"
            
        # Check price consistency
        price_cols = ['open', 'high', 'low', 'close']
        for idx, row in data.iterrows():
            if not (row['low'] <= row['open'] <= row['high'] and 
                   row['low'] <= row['close'] <= row['high']):
                return False, f"Invalid price relationship at {idx}"
                
        # Check for large gaps
        if len(data) > 1:
            time_diffs = data.index.to_series().diff()
            max_gap = time_diffs.max()
            if max_gap > pd.Timedelta(days=7):  # Adjust threshold as needed
                return False, f"Large gap detected in data: {max_gap}"
                
        return True, "Data validation successful"
        
    except Exception as e:
        return False, f"Error during data validation: {str(e)}"

def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """Process and clean the input data."""
    try:
        # Make a copy to avoid modifying original
        df = data.copy()
        
        # Sort by date
        df = df.sort_index()
        
        # Handle NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Ensure numeric columns are float
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Remove any remaining NaN values
        df = df.dropna()
        
        # Validate price relationships
        df = df[
            (df['low'] <= df['open']) & 
            (df['open'] <= df['high']) & 
            (df['low'] <= df['close']) & 
            (df['close'] <= df['high'])
        ]
        
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