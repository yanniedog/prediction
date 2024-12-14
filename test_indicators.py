import logging
import sys
import pandas as pd
import numpy as np
import talib as ta
import pandas_ta as pta
import json
from pathlib import Path

# Import the logging configuration from logging_setup.py
try:
    from logging_setup import configure_logging
except ImportError as e:
    print(f"Error importing logging_setup.py: {e}")
    sys.exit(1)

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

# Mapping of TA-Lib indicators to their required input columns
# Extend this mapping as needed based on your JSON configuration
INDICATOR_INPUT_MAPPING = {
    'adosc': ['high', 'low', 'close', 'volume'],
    'macd': ['close'],
    'ppo': ['close'],
    # Add other indicators as necessary
}

def load_indicator_params(params_path: str) -> dict:
    """
    Load indicator configurations from a JSON file.

    Args:
        params_path (str): Path to the indicator_params.json file.

    Returns:
        dict: Dictionary containing indicator configurations.
    """
    logger.info(f"Loading indicator parameters from '{params_path}'.")
    try:
        with open(params_path, 'r') as f:
            indicator_params = json.load(f)
        logger.debug(f"Indicator Parameters Loaded: {indicator_params}")
        return indicator_params
    except FileNotFoundError:
        logger.error(f"Indicator parameters file '{params_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from '{params_path}': {e}")
        sys.exit(1)

def generate_simulated_data(num_rows: int = 200) -> pd.DataFrame:
    """
    Generate simulated market data.

    Args:
        num_rows (int): Number of rows of data to generate.

    Returns:
        pd.DataFrame: DataFrame containing simulated market data.
    """
    logger.info("Generating simulated market data.")
    np.random.seed(42)  # For reproducibility
    data = pd.DataFrame({
        'open': np.random.uniform(100, 200, num_rows),
        'high': np.random.uniform(200, 300, num_rows),
        'low': np.random.uniform(50, 100, num_rows),
        'close': np.random.uniform(100, 200, num_rows),
        'volume': np.random.uniform(1000, 5000, num_rows)
    })
    logger.debug(f"Simulated Data (first 5 rows):\n{data.head()}")
    return data

def compute_ta_lib_indicator(data: pd.DataFrame, indicator_name: str, params: dict) -> bool:
    """
    Compute a TA-Lib indicator.

    Args:
        data (pd.DataFrame): DataFrame containing market data.
        indicator_name (str): Name of the TA-Lib indicator.
        params (dict): Parameters for the indicator.

    Returns:
        bool: True if computation was successful, False otherwise.
    """
    logger.info(f"Computing TA-Lib Indicator: {indicator_name} with parameters {params}")
    try:
        # Retrieve the TA-Lib function
        ta_func = getattr(ta, indicator_name.upper())
    except AttributeError:
        logger.error(f"TA-Lib does not have a function named '{indicator_name.upper()}'.")
        return False

    try:
        # Extract required input columns from the JSON config
        required_inputs = params.pop('required_inputs', params.pop('input_columns', ['close']))
        inputs = [data[col].values for col in required_inputs]

        # Pass only the numerical parameters to the TA-Lib function
        indicator_values = ta_func(*inputs, **params)

        # Handle functions that return multiple arrays (e.g., MACD)
        if isinstance(indicator_values, tuple):
            for idx, val in enumerate(indicator_values):
                col_name = f"{indicator_name.upper()}_{'_'.join([str(v) for v in params.values()])}_{idx}"
                data[col_name] = val
                logger.debug(f"Computed {col_name}: {val[:5]}...")
        else:
            col_name = f"{indicator_name.upper()}_{'_'.join([str(v) for v in params.values()])}"
            data[col_name] = indicator_values
            logger.debug(f"Computed {col_name}: {indicator_values[:5]}...")

        logger.info(f"Indicator '{indicator_name}' computed successfully.")
        return True

    except Exception as e:
        logger.error(f"Error computing TA-Lib indicator '{indicator_name}': {e}")
        return False

def compute_pandas_ta_indicator(data: pd.DataFrame, indicator_name: str, params: dict) -> bool:
    """
    Compute a pandas-ta indicator.

    Args:
        data (pd.DataFrame): DataFrame containing market data.
        indicator_name (str): Name of the pandas-ta indicator.
        params (dict): Parameters for the indicator.

    Returns:
        bool: True if computation was successful, False otherwise.
    """
    logger.info(f"Computing pandas-ta Indicator: {indicator_name} with parameters {params}")
    try:
        # Retrieve the pandas-ta function
        pta_func = getattr(pta, indicator_name.lower())
    except AttributeError:
        logger.error(f"pandas-ta does not have a function named '{indicator_name.lower()}'.")
        return False

    try:
        # Extract required input columns from the JSON config
        required_inputs = params.pop('required_inputs', params.pop('input_columns', ['close']))
        # Prepare the parameters excluding 'type' and 'required_inputs'
        indicator_params = {k: v['default'] for k, v in params.items() if k not in ['type', 'input_columns', 'required_inputs']}
        # Pass parameters to the pandas-ta function
        indicator_df = pta_func(**indicator_params, **{col: data[col] for col in required_inputs})

        # pandas-ta functions typically return a DataFrame
        if isinstance(indicator_df, pd.DataFrame):
            for col in indicator_df.columns:
                data[col] = indicator_df[col]
                logger.debug(f"Computed {col}: {indicator_df[col].head().tolist()}")
        else:
            # If a Series is returned
            col_name = f"{indicator_name.upper()}_{'_'.join([str(v) for v in indicator_params.values()])}"
            data[col_name] = indicator_df
            logger.debug(f"Computed {col_name}: {indicator_df.head().tolist()}")

        logger.info(f"Indicator '{indicator_name}' computed successfully.")
        return True

    except Exception as e:
        logger.error(f"Error computing pandas-ta indicator '{indicator_name}': {e}")
        return False

def compute_custom_indicator(data: pd.DataFrame, indicator_name: str, params: dict) -> bool:
    """
    Compute a custom indicator based on the indicator name.

    Args:
        data (pd.DataFrame): DataFrame containing market data.
        indicator_name (str): Name of the custom indicator.
        params (dict): Parameters for the indicator.

    Returns:
        bool: True if computation was successful, False otherwise.
    """
    logger.info(f"Computing Custom Indicator: {indicator_name} with parameters {params}")
    try:
        if indicator_name == 'obv_price_divergence':
            # Example implementation for On-Balance Volume (OBV) Price Divergence
            obv = ta.OBV(data['close'].values, data['volume'].values)
            data['OBV'] = obv
            # Compute divergence based on the 'method' parameter
            method = params.get('method', 'Difference')
            if method == 'Difference':
                data['OBV_Price_Diff'] = data['OBV'] - data['close']
            elif method == 'Ratio':
                data['OBV_Price_Ratio'] = data['OBV'] / data['close']
            elif method == 'Log':
                # Avoid division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    data['OBV_Price_Log'] = np.log(data['OBV'] / data['close'])
            else:
                logger.warning(f"Unknown method '{method}' for indicator '{indicator_name}'.")
                return False
            logger.info(f"Custom Indicator '{indicator_name}' computed successfully.")
            return True

        elif indicator_name == 'EyeX MFV Volume':
            # Example implementation for 'EyeX MFV Volume'
            ranges = params.get('ranges', [50, 75, 100, 200])
            # Compute Moving Finite Volume (MFV) as rolling sum over 'volume' for each range
            for r in ranges:
                col_name = f"EyeX_MFV_Volume_{r}"
                data[col_name] = data['volume'].rolling(window=r).sum()
                logger.debug(f"Computed {col_name}: {data[col_name].head().tolist()}")
            logger.info(f"Custom Indicator '{indicator_name}' computed successfully.")
            return True

        elif indicator_name == 'EyeX MFV S/R Bull':
            # Example implementation for 'EyeX MFV S/R Bull'
            ranges = params.get('ranges', [50, 75, 100, 200])
            pivot_lookback = params.get('pivot_lookback', 5)
            price_proximity = params.get('price_proximity', 0.00001)
            # Identify support/resistance levels based on MFV and pivot lookback
            for r in ranges:
                pivot_high = data['high'].rolling(window=pivot_lookback).max()
                pivot_low = data['low'].rolling(window=pivot_lookback).min()
                # Calculate proximity to pivot high
                data[f"EyeX_MFV_S_R_Bull_High_{r}"] = np.where(
                    (data['close'] >= (pivot_high - price_proximity)) &
                    (data['close'] <= (pivot_high + price_proximity)),
                    1, 0
                )
                # Calculate proximity to pivot low
                data[f"EyeX_MFV_S_R_Bull_Low_{r}"] = np.where(
                    (data['close'] >= (pivot_low - price_proximity)) &
                    (data['close'] <= (pivot_low + price_proximity)),
                    1, 0
                )
                logger.debug(f"Computed EyeX_MFV_S_R_Bull_High_{r}: {data[f'EyeX_MFV_S_R_Bull_High_{r}'].head().tolist()}")
                logger.debug(f"Computed EyeX_MFV_S_R_Bull_Low_{r}: {data[f'EyeX_MFV_S_R_Bull_Low_{r}'].head().tolist()}")
            logger.info(f"Custom Indicator '{indicator_name}' computed successfully.")
            return True

        else:
            logger.error(f"Unknown custom indicator '{indicator_name}'.")
            return False

    except Exception as e:
        logger.error(f"Error computing Custom Indicator '{indicator_name}': {e}")
        return False

def compute_indicators(data: pd.DataFrame, indicator_params: dict) -> pd.DataFrame:
    """
    Compute all indicators based on the provided configurations.

    Args:
        data (pd.DataFrame): DataFrame containing market data.
        indicator_params (dict): Dictionary of indicator configurations.

    Returns:
        pd.DataFrame: DataFrame with computed indicators.
    """
    logger.info("Starting computation of indicators.")

    for indicator_name, config in indicator_params.items():
        logger.info(f"Processing Indicator: {indicator_name}")
        indicator_type = config.get('type')
        params = config.get('parameters', {})
        input_columns = config.get('input_columns', ['close'])

        # Append required_inputs to params for later use
        # Use 'required_inputs' if defined, else default to 'input_columns'
        required_inputs = config.get('required_inputs', input_columns)
        params['required_inputs'] = required_inputs

        if indicator_type.lower() == 'ta-lib':
            success = compute_ta_lib_indicator(data, indicator_name, params)
            if not success and hasattr(pta, indicator_name.lower()):
                logger.info(f"Attempting to compute '{indicator_name}' using pandas-ta.")
                config['type'] = 'pandas-ta'
                success = compute_pandas_ta_indicator(data, indicator_name, params)
        elif indicator_type.lower() == 'pandas-ta':
            success = compute_pandas_ta_indicator(data, indicator_name, params)
        elif indicator_type.lower() == 'custom':
            success = compute_custom_indicator(data, indicator_name, params)
        else:
            logger.error(f"Unknown indicator type '{indicator_type}' for indicator '{indicator_name}'. Skipping.")
            success = False

        if not success:
            logger.warning(f"Indicator '{indicator_name}' failed to compute.")

    logger.info("Completed computation of all indicators.")
    return data

def validate_indicators(data: pd.DataFrame, indicator_params: dict) -> None:
    """
    Validate that all indicators have been computed correctly.

    Args:
        data (pd.DataFrame): DataFrame containing market data with indicators.
        indicator_params (dict): Dictionary of indicator configurations.
    """
    logger.info("Starting validation of computed indicators.")

    all_valid = True
    for indicator_name in indicator_params.keys():
        relevant_cols = [col for col in data.columns if col.startswith(indicator_name.upper())]
        if not relevant_cols:
            logger.error(f"No columns found for indicator '{indicator_name}'.")
            all_valid = False
            continue

        for col in relevant_cols:
            if data[col].isnull().all():
                logger.error(f"Indicator '{col}' contains only NaN values.")
                all_valid = False
            else:
                non_nan = data[col].notna().sum()
                logger.debug(f"Indicator '{col}' has {non_nan} non-NaN values out of {len(data)}.")
                if non_nan < len(data) * 0.5:
                    logger.warning(f"Indicator '{col}' has less than 50% non-NaN values.")

    if all_valid:
        logger.info("All indicators validated successfully.")
    else:
        logger.warning("Some indicators failed validation. Check logs for details.")

def main():
    # Define paths
    current_dir = Path(__file__).parent
    params_path = current_dir / 'indicator_params.json'

    # Load indicator configurations
    indicator_params = load_indicator_params(str(params_path))

    # Generate simulated data
    data = generate_simulated_data(num_rows=200)  # Increased rows for better indicator computation

    # Compute indicators
    data = compute_indicators(data, indicator_params)

    # Validate indicators
    validate_indicators(data, indicator_params)

    # Final Summary
    logger.info("Indicator testing completed.")
    print("Indicator testing completed. Check the log file for detailed information.")

if __name__ == "__main__":
    main()
