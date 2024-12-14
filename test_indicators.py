# test_indicators.py

import json
import logging
import pandas as pd
import numpy as np
import talib
import pandas_ta as ta
import os
from logging_setup import configure_logging  # Ensure this module exists and is correctly configured

def load_indicator_params(json_path):
    try:
        with open(json_path, 'r') as f:
            params = json.load(f)
        logging.info(f"Indicator Parameters Loaded: {list(params['indicators'].keys())}")
        return params['indicators']
    except Exception as e:
        logging.error(f"Failed to load indicator parameters from {json_path}: {e}")
        raise

def generate_simulated_data(num_rows=200):
    np.random.seed(0)
    data = pd.DataFrame({
        'open': np.random.uniform(100, 200, num_rows),
        'high': np.random.uniform(100, 300, num_rows),
        'low': np.random.uniform(50, 100, num_rows),
        'close': np.random.uniform(100, 200, num_rows),
        'volume': np.random.uniform(1000, 5000, num_rows)
    })
    logging.info("Generating simulated market data.")
    logging.debug(f"Simulated Data (first 5 rows):\n{data.head()}")
    return data

def compute_custom_indicator(indicator_name, params, data):
    try:
        if indicator_name == 'EyeX MFV Volume':
            # Volume-based moving averages
            ranges = params['ranges']['default']
            ranges = sorted(ranges)
            for r in ranges:
                column_name = f"MFV_Volume_{r}"
                data[column_name] = data['volume'].rolling(window=r).mean()
            logging.info(f"Indicator '{indicator_name}' computed successfully.")
        
        elif indicator_name == 'EyeX MFV S/R Bull':
            # Support/Resistance based on volume moving averages
            ranges = params['ranges']['default']
            pivot_lookback = params['pivot_lookback']['default']
            price_proximity = params['price_proximity']['default']
            
            # Calculate moving averages for different ranges
            for r in ranges:
                data[f"MFV_SRB_{r}"] = data['volume'].rolling(window=r).mean()
            
            # Identify pivots
            data['Pivot_High'] = data['high'].rolling(window=pivot_lookback, center=True).max()
            data['Pivot_Low'] = data['low'].rolling(window=pivot_lookback, center=True).min()
            
            # Calculate support/resistance based on price proximity
            data['Support'] = np.where(abs(data['close'] - data['Pivot_Low']) <= price_proximity, data['Pivot_Low'], np.nan)
            data['Resistance'] = np.where(abs(data['close'] - data['Pivot_High']) <= price_proximity, data['Pivot_High'], np.nan)
            
            logging.info(f"Indicator '{indicator_name}' computed successfully.")
        
        else:
            logging.warning(f"Custom indicator '{indicator_name}' is not implemented.")
            return
    except Exception as e:
        logging.error(f"Error computing Custom Indicator '{indicator_name}': {e}")
        raise

def compute_ta_lib_indicator(indicator_name, params, data):
    try:
        # Extract only the parameter values
        extracted_params = {}
        for key, value in params.items():
            if key in ['required_inputs', 'input_columns', 'conditions']:
                continue  # Skip non-parameter entries
            if 'default' in value:
                extracted_params[key] = value['default']
            else:
                extracted_params[key] = None  # or handle appropriately
        
        # Ensure all parameters are of correct type
        for key, val in extracted_params.items():
            if val is None:
                logging.warning(f"Parameter '{key}' for indicator '{indicator_name}' is missing. Using default value.")
        
        # Get the TA-Lib function
        func = getattr(talib, indicator_name.upper())
        
        # Prepare positional arguments
        required_inputs = params.get('required_inputs', [])
        positional_args = []
        for inp in required_inputs:
            if inp in data.columns:
                positional_args.append(data[inp].values)
            else:
                logging.warning(f"Required input '{inp}' for indicator '{indicator_name}' is missing in data.")
                positional_args.append(np.nan)  # or handle appropriately
        
        # Add the parameters as keyword arguments
        keyword_args = extracted_params
        
        # Log the function call details for debugging
        logging.debug(f"Calling TA-Lib {indicator_name.upper()} with positional args: {positional_args} and keyword args: {keyword_args}")
        
        # Compute the indicator
        result = func(*positional_args, **keyword_args)
        
        # TA-Lib functions typically return a numpy array
        # Assign to DataFrame
        data[f"{indicator_name.upper()}"] = result
        logging.info(f"Indicator '{indicator_name}' computed successfully.")
    
    except AttributeError:
        logging.error(f"TA-Lib does not have a function named '{indicator_name.upper()}'.")
        # Optionally, attempt pandas-ta computation
        raise
    except TypeError as te:
        logging.error(f"Error computing TA-Lib indicator '{indicator_name}': {te}")
        # Optionally, attempt pandas-ta computation
        raise
    except Exception as e:
        logging.error(f"Error computing TA-Lib indicator '{indicator_name}': {e}")
        # Optionally, attempt pandas-ta computation
        raise

def compute_pandas_ta_indicator(indicator_name, params, data):
    try:
        # Extract parameter values
        extracted_params = {}
        for key, value in params.items():
            if key in ['required_inputs', 'input_columns']:
                continue  # Skip non-parameter entries
            if 'default' in value:
                extracted_params[key] = value['default']
            else:
                extracted_params[key] = None  # or handle appropriately
        
        # Get the pandas-ta function
        func = getattr(ta, indicator_name.lower())
        
        # Prepare keyword arguments
        required_inputs = params.get('required_inputs', [])
        keyword_args = {}
        for inp in required_inputs:
            if inp in data.columns:
                keyword_args[inp] = data[inp]
            else:
                logging.warning(f"Required input '{inp}' for indicator '{indicator_name}' is missing in data.")
                keyword_args[inp] = np.nan  # or handle appropriately
        
        # Add the parameters
        keyword_args.update(extracted_params)
        
        # Log the function call details for debugging
        logging.debug(f"Calling pandas-ta {indicator_name.lower()} with keyword args: {keyword_args}")
        
        # Compute the indicator
        result = func(**keyword_args)
        
        # Assign to DataFrame
        # Some pandas-ta functions return a DataFrame, others return Series
        if isinstance(result, pd.DataFrame):
            for col in result.columns:
                data[col] = result[col]
        else:
            # Handle cases where parameters affect the column name
            # Generate a unique column name based on parameters
            param_values = '_'.join([str(v) for v in extracted_params.values() if v is not None])
            column_name = f"{indicator_name.upper()}_{param_values}" if param_values else indicator_name.upper()
            data[column_name] = result
        logging.info(f"Indicator '{indicator_name}' computed successfully.")
    
    except AttributeError:
        logging.error(f"pandas-ta does not have a function named '{indicator_name.lower()}'.")
        raise
    except TypeError as te:
        logging.error(f"Error computing pandas-ta indicator '{indicator_name}': {te}")
        raise
    except Exception as e:
        logging.error(f"Error computing pandas-ta indicator '{indicator_name}': {e}")
        raise

def compute_indicators(indicators, data):
    for indicator_name, config in indicators.items():
        logging.info(f"Processing Indicator: {indicator_name}")
        indicator_type = config.get('type', '').lower()
        params = config.get('parameters', {})
        required_inputs = config.get('required_inputs', [])
        input_columns = config.get('input_columns', [])
        
        # Check conditions if any
        conditions = config.get('conditions', [])
        if conditions:
            condition_met = True
            for condition in conditions:
                for param, rule in condition.items():
                    for rule_type, rule_value in rule.items():
                        # Handle 'greater_than', 'less_than', 'less_than_or_equal'
                        if rule_type == 'greater_than':
                            param_value = params[param]['default']
                            # Check if rule_value is a parameter name or a fixed value
                            if isinstance(rule_value, str) and rule_value in params:
                                compare_to = params[rule_value]['default']
                            else:
                                compare_to = float(rule_value)
                            if param_value <= compare_to:
                                condition_met = False
                                break
                        elif rule_type == 'less_than':
                            param_value = params[param]['default']
                            if isinstance(rule_value, str) and rule_value in params:
                                compare_to = params[rule_value]['default']
                            else:
                                compare_to = float(rule_value)
                            if param_value >= compare_to:
                                condition_met = False
                                break
                        elif rule_type == 'less_than_or_equal':
                            param_value = params[param]['default']
                            if isinstance(rule_value, str) and rule_value in params:
                                compare_to = params[rule_value]['default']
                            else:
                                compare_to = float(rule_value)
                            if param_value > compare_to:
                                condition_met = False
                                break
                        # Add more condition types as needed
                if not condition_met:
                    logging.info(f"Conditions not met for indicator '{indicator_name}'. Skipping computation.")
                    break
            if not condition_met:
                continue  # Skip this indicator
        
        # Check if required inputs are present
        missing_inputs = [inp for inp in required_inputs if inp not in data.columns]
        if missing_inputs:
            logging.warning(f"Missing required inputs {missing_inputs} for indicator '{indicator_name}'. Skipping.")
            continue
        
        try:
            if indicator_type == 'ta-lib':
                try:
                    compute_ta_lib_indicator(indicator_name, params, data)
                except Exception as e:
                    logging.warning(f"TA-Lib computation failed for indicator '{indicator_name}': {e}. Attempting pandas-ta.")
                    try:
                        compute_pandas_ta_indicator(indicator_name, params, data)
                    except Exception as e_pandas:
                        logging.error(f"Pandas-ta computation also failed for indicator '{indicator_name}': {e_pandas}.")
            elif indicator_type == 'pandas-ta':
                compute_pandas_ta_indicator(indicator_name, params, data)
            elif indicator_type == 'custom':
                compute_custom_indicator(indicator_name, params, data)
            else:
                logging.warning(f"Unknown indicator type '{indicator_type}' for indicator '{indicator_name}'. Skipping.")
        except Exception as e:
            logging.warning(f"Indicator '{indicator_name}' failed to compute: {e}")

def validate_indicators(indicators, data):
    for indicator_name, config in indicators.items():
        logging.info(f"Validating Indicator: {indicator_name}")
        # Check if the indicator's columns exist in data
        if config['type'] == 'custom':
            # For custom indicators, might have multiple columns
            if indicator_name == 'EyeX MFV Volume':
                # Check for MFV_Volume_* columns
                col_prefix = 'MFV_Volume_'
                mfv_cols = [col for col in data.columns if col.startswith(col_prefix)]
                if not mfv_cols:
                    logging.error(f"No columns found for indicator '{indicator_name}'.")
                    continue
                for col in mfv_cols:
                    non_nan = data[col].notna().sum()
                    total = len(data)
                    logging.debug(f"Indicator '{col}' has {non_nan} non-NaN values out of {total}.")
            elif indicator_name == 'EyeX MFV S/R Bull':
                # Check for MFV_SRB_*, Pivot_High, Pivot_Low, Support, Resistance
                required_cols = [col for col in data.columns if col.startswith('MFV_SRB_')] + ['Pivot_High', 'Pivot_Low', 'Support', 'Resistance']
                missing_cols = [col for col in required_cols if col not in data.columns]
                if missing_cols:
                    logging.error(f"No columns found for indicator '{indicator_name}'. Missing: {missing_cols}")
                    continue
                for col in required_cols:
                    non_nan = data[col].notna().sum()
                    total = len(data)
                    logging.debug(f"Indicator '{col}' has {non_nan} non-NaN values out of {total}.")
        else:
            # For ta-lib and pandas-ta indicators, expect a single column or parameterized columns
            # Handle cases where pandas-ta might append parameters to the column name
            possible_cols = [indicator_name.upper()]
            # Also consider parameterized column names for pandas-ta
            for col in data.columns:
                if col.startswith(indicator_name.upper()):
                    possible_cols.append(col)
            found = False
            for col in possible_cols:
                if col in data.columns:
                    non_nan = data[col].notna().sum()
                    total = len(data)
                    logging.debug(f"Indicator '{col}' has {non_nan} non-NaN values out of {total}.")
                    found = True
            if not found:
                logging.error(f"No columns found for indicator '{indicator_name}'.")
                continue
        
        # Additional validation criteria can be added here
        # For example, minimum number of non-NaN values
        # For simplicity, we'll skip this in this script
    
    logging.info("Completed validation of computed indicators.")

def main():
    try:
        # Configure logging using logging_setup.py
        configure_logging(log_file_prefix='predictions')
        
        # Load indicator parameters
        json_path = 'indicator_params.json'
        if not os.path.exists(json_path):
            logging.error(f"Indicator parameters file '{json_path}' does not exist.")
            return
        
        indicators = load_indicator_params(json_path)
        
        # Generate or load market data
        data = generate_simulated_data(num_rows=200)
        
        # Compute indicators
        logging.info("Starting computation of indicators.")
        compute_indicators(indicators, data)
        
        # Validate indicators
        logging.info("Starting validation of computed indicators.")
        validate_indicators(indicators, data)
        
        logging.info("Indicator testing completed.")
    
    except Exception as e:
        logging.error(f"An error occurred in the main execution: {e}")

if __name__ == "__main__":
    main()
