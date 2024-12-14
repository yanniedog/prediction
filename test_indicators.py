import json
import logging
import pandas as pd
import numpy as np
import talib
import pandas_ta as ta
import os

def setup_logging():
    log_filename = "predictions_20241215-030314.log"
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d(<module>)]: %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

logger = setup_logging()

def load_indicator_params(json_path):
    try:
        with open(json_path, 'r') as f:
            params = json.load(f)
        logger.info(f"Indicator Parameters Loaded: {list(params['indicators'].keys())}")
        return params['indicators']
    except Exception as e:
        logger.error(f"Failed to load indicator parameters from {json_path}: {e}")
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
    logger.info("Generating simulated market data.")
    logger.debug(f"Simulated Data (first 5 rows):\n{data.head()}")
    return data

def compute_custom_indicator(indicator_name, params, data):
    try:
        if indicator_name == 'obv_price_divergence':
            # Example implementation: On-Balance Volume divergence calculation
            method = params['method']['default']
            obv_method = params['obv_method']['default']
            obv_period = params['obv_period']['default']
            price_input_type = params['price_input_type']['default']
            price_method = params['price_method']['default']
            price_period = params['price_period']['default']
            smoothing = params['smoothing']['default']
            
            # Select price input
            if price_input_type == 'close':
                price = data['close']
            elif price_input_type == 'open':
                price = data['open']
            elif price_input_type == 'high':
                price = data['high']
            elif price_input_type == 'low':
                price = data['low']
            elif price_input_type == 'hl/2':
                price = (data['high'] + data['low']) / 2
            elif price_input_type == 'ohlc/4':
                price = (data['open'] + data['high'] + data['low'] + data['close']) / 4
            else:
                logger.warning(f"Unknown price_input_type '{price_input_type}' for indicator '{indicator_name}'. Using 'close'.")
                price = data['close']
            
            # Compute OBV
            obv = talib.OBV(data['close'], data['volume'])
            if obv_method.upper() == 'SMA':
                obv = talib.SMA(obv, timeperiod=obv_period)
            elif obv_method.upper() == 'EMA':
                obv = talib.EMA(obv, timeperiod=obv_period)
            else:
                logger.warning(f"Unknown obv_method '{obv_method}' for indicator '{indicator_name}'. Using 'SMA'.")
                obv = talib.SMA(obv, timeperiod=obv_period)
            
            # Compute price moving average
            if price_method.upper() == 'SMA':
                price_ma = talib.SMA(price, timeperiod=price_period)
            elif price_method.upper() == 'EMA':
                price_ma = talib.EMA(price, timeperiod=price_period)
            else:
                logger.warning(f"Unknown price_method '{price_method}' for indicator '{indicator_name}'. Using 'SMA'.")
                price_ma = talib.SMA(price, timeperiod=price_period)
            
            # Calculate divergence based on method
            if method == 'Difference':
                divergence = price_ma - obv
            elif method == 'Ratio':
                divergence = price_ma / (obv + 1e-10)  # Avoid division by zero
            elif method == 'Log':
                divergence = np.log(price_ma + 1) - np.log(obv + 1)
            else:
                logger.warning(f"Unknown method '{method}' for indicator '{indicator_name}'. Using 'Difference'.")
                divergence = price_ma - obv
            
            # Apply smoothing if necessary
            divergence_smoothed = divergence * (1 - smoothing)
            
            # Assign to data
            data[f"{indicator_name.upper()}"] = divergence_smoothed
            logger.info(f"Indicator '{indicator_name}' computed successfully.")
        
        elif indicator_name == 'EyeX MFV Volume':
            # Example implementation: Volume-based moving averages
            ranges = params['ranges']['default']
            ranges = sorted(ranges)
            for r in ranges:
                column_name = f"MFV_Volume_{r}"
                data[column_name] = data['volume'].rolling(window=r).mean()
            logger.info(f"Indicator '{indicator_name}' computed successfully.")
        
        elif indicator_name == 'EyeX MFV S/R Bull':
            # Example implementation: Support/Resistance based on volume moving averages
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
            
            logger.info(f"Indicator '{indicator_name}' computed successfully.")
        
        else:
            logger.warning(f"Custom indicator '{indicator_name}' is not implemented.")
            return
    except Exception as e:
        logger.error(f"Error computing Custom Indicator '{indicator_name}': {e}")
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
                logger.warning(f"Parameter '{key}' for indicator '{indicator_name}' is missing. Using default value.")
        
        # Get the TA-Lib function
        func = getattr(talib, indicator_name.upper())
        
        # Prepare arguments
        required_inputs = params.get('required_inputs', [])
        args = {}
        for inp in required_inputs:
            if inp in data.columns:
                args[inp] = data[inp].values
            else:
                logger.warning(f"Required input '{inp}' for indicator '{indicator_name}' is missing in data.")
                args[inp] = np.nan  # or handle appropriately
        
        # Add the parameters
        args.update(extracted_params)
        
        # Compute the indicator
        result = func(**args)
        
        # TA-Lib functions typically return a numpy array
        # Assign to DataFrame
        data[f"{indicator_name.upper()}"] = result
        logger.info(f"Indicator '{indicator_name}' computed successfully.")
    
    except AttributeError:
        logger.error(f"TA-Lib does not have a function named '{indicator_name.upper()}'.")
        # Optionally, attempt pandas-ta computation
        raise
    except TypeError as te:
        logger.error(f"Error computing TA-Lib indicator '{indicator_name}': {te}")
        # Optionally, attempt pandas-ta computation
        raise
    except Exception as e:
        logger.error(f"Error computing TA-Lib indicator '{indicator_name}': {e}")
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
        
        # Determine required inputs
        required_inputs = params.get('required_inputs', [])
        args = {}
        for inp in required_inputs:
            if inp in data.columns:
                args[inp] = data[inp]
            else:
                logger.warning(f"Required input '{inp}' for indicator '{indicator_name}' is missing in data.")
                args[inp] = np.nan  # or handle appropriately
        
        # Add the parameters
        args.update(extracted_params)
        
        # Compute the indicator
        result = func(**args)
        
        # Assign to DataFrame
        # Some pandas-ta functions return a DataFrame, others return Series
        if isinstance(result, pd.DataFrame):
            for col in result.columns:
                data[col] = result[col]
        else:
            param_values = '_'.join(map(str, extracted_params.values()))
            data[f"{indicator_name.upper()}_{param_values}"] = result
        logger.info(f"Indicator '{indicator_name}' computed successfully.")
    
    except AttributeError:
        logger.error(f"pandas-ta does not have a function named '{indicator_name.lower()}'.")
        raise
    except TypeError as te:
        logger.error(f"Error computing pandas-ta indicator '{indicator_name}': {te}")
        raise
    except Exception as e:
        logger.error(f"Error computing pandas-ta indicator '{indicator_name}': {e}")
        raise

def compute_indicators(indicators, data):
    for indicator_name, config in indicators.items():
        logger.info(f"Processing Indicator: {indicator_name}")
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
                    logger.info(f"Conditions not met for indicator '{indicator_name}'. Skipping computation.")
                    break
            if not condition_met:
                continue  # Skip this indicator
        
        # Check if required inputs are present
        missing_inputs = [inp for inp in required_inputs if inp not in data.columns]
        if missing_inputs:
            logger.warning(f"Missing required inputs {missing_inputs} for indicator '{indicator_name}'. Skipping.")
            continue
        
        try:
            if indicator_type == 'ta-lib':
                try:
                    compute_ta_lib_indicator(indicator_name, params, data)
                except Exception as e:
                    logger.warning(f"TA-Lib computation failed for indicator '{indicator_name}': {e}. Attempting pandas-ta.")
                    try:
                        compute_pandas_ta_indicator(indicator_name, params, data)
                    except Exception as e_pandas:
                        logger.error(f"Pandas-ta computation also failed for indicator '{indicator_name}': {e_pandas}.")
            elif indicator_type == 'pandas-ta':
                compute_pandas_ta_indicator(indicator_name, params, data)
            elif indicator_type == 'custom':
                compute_custom_indicator(indicator_name, params, data)
            else:
                logger.warning(f"Unknown indicator type '{indicator_type}' for indicator '{indicator_name}'. Skipping.")
        except Exception as e:
            logger.warning(f"Indicator '{indicator_name}' failed to compute: {e}")

def validate_indicators(indicators, data):
    for indicator_name, config in indicators.items():
        logger.info(f"Validating Indicator: {indicator_name}")
        # Check if the indicator's columns exist in data
        if config['type'] == 'custom':
            # For custom indicators, might have multiple columns
            if indicator_name == 'obv_price_divergence':
                col = indicator_name.upper()
                if col not in data.columns:
                    logger.error(f"No columns found for indicator '{indicator_name}'.")
                    continue
                non_nan = data[col].notna().sum()
                total = len(data)
                logger.debug(f"Indicator '{col}' has {non_nan} non-NaN values out of {total}.")
            elif indicator_name == 'EyeX MFV Volume':
                # Check for MFV_Volume and MFV_Volume_r
                col_prefix = 'MFV_Volume'
                mfv_cols = [col for col in data.columns if col.startswith(col_prefix)]
                if not mfv_cols:
                    logger.error(f"No columns found for indicator '{indicator_name}'.")
                    continue
                for col in mfv_cols:
                    non_nan = data[col].notna().sum()
                    total = len(data)
                    logger.debug(f"Indicator '{col}' has {non_nan} non-NaN values out of {total}.")
            elif indicator_name == 'EyeX MFV S/R Bull':
                # Check for MFV_SRB, Pivot_High, Pivot_Low, Support, Resistance
                required_cols = [col for col in data.columns if col.startswith('MFV_SRB')] + ['Pivot_High', 'Pivot_Low', 'Support', 'Resistance']
                missing_cols = [col for col in required_cols if col not in data.columns]
                if missing_cols:
                    logger.error(f"No columns found for indicator '{indicator_name}'. Missing: {missing_cols}")
                    continue
                for col in required_cols:
                    non_nan = data[col].notna().sum()
                    total = len(data)
                    logger.debug(f"Indicator '{col}' has {non_nan} non-NaN values out of {total}.")
        else:
            # For ta-lib and pandas-ta indicators, expect a single column
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
                    logger.debug(f"Indicator '{col}' has {non_nan} non-NaN values out of {total}.")
                    found = True
            if not found:
                logger.error(f"No columns found for indicator '{indicator_name}'.")
                continue
        
        # Additional validation criteria can be added here
        # For example, minimum number of non-NaN values
        # For simplicity, we'll skip this in this script
        
    logger.info("Completed validation of computed indicators.")

def main():
    try:
        # Load indicator parameters
        json_path = 'indicator_params.json'
        if not os.path.exists(json_path):
            logger.error(f"Indicator parameters file '{json_path}' does not exist.")
            return
        
        indicators = load_indicator_params(json_path)
        
        # Generate or load market data
        data = generate_simulated_data(num_rows=200)
        
        # Compute indicators
        logger.info("Starting computation of indicators.")
        compute_indicators(indicators, data)
        
        # Validate indicators
        logger.info("Starting validation of computed indicators.")
        validate_indicators(indicators, data)
        
        logger.info("Indicator testing completed.")
    
    except Exception as e:
        logger.error(f"An error occurred in the main execution: {e}")

if __name__ == "__main__":
    main()
