import pandas as pd

def _select_data_source_and_lag(data=None):
    """Select data source and calculate lag.
    
    Args:
        data (pd.DataFrame, optional): Input data to validate. If None, prompts user for data source.
        
    Returns:
        tuple: (data, lag) where data is the selected DataFrame and lag is the calculated lag
        
    Raises:
        ValueError: If data is invalid or insufficient
    """
    if data is not None:
        # Validate input data
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a DataFrame")
            
        if data.empty:
            raise ValueError("Empty DataFrame provided")
            
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Check for invalid values
        if (data['high'] < data['low']).any():
            raise ValueError("High price cannot be less than low price")
        if (data['volume'] < 0).any():
            raise ValueError("Volume cannot be negative")
            
        # Check for duplicate dates
        if data.index.duplicated().any():
            raise ValueError("Duplicate dates found in data")
            
        # Check for non-monotonic dates
        if not data.index.is_monotonic_increasing:
            raise ValueError("Dates must be in ascending order")
            
        # Check for sufficient data
        if len(data) < 30:  # Minimum required data points
            raise ValueError("Insufficient data points (minimum 30 required)")
            
        # Calculate lag
        lag = _calculate_max_lag(data)
        return data, lag
        
    # If no data provided, prompt user for data source
    print("\n--- Data Source ---\n")
    action = input("Select data source action (d=download/update, l=load from DB, f=load from file): ").lower()
    
    if action == 'd':
        symbol = input("Enter symbol (e.g. BTCUSDT): ").upper()
        timeframe = input("Enter timeframe (e.g. 1h, 1d): ").lower()
        data = _download_data(symbol, timeframe)
    elif action == 'l':
        symbol = input("Enter symbol (e.g. BTCUSDT): ").upper()
        timeframe = input("Enter timeframe (e.g. 1h, 1d): ").lower()
        data = _load_from_db(symbol, timeframe)
    elif action == 'f':
        path = input("Enter file path: ")
        data = _load_from_file(path)
    else:
        raise ValueError("Invalid action selected")
        
    # Calculate lag
    lag = _calculate_max_lag(data)
    return data, lag 