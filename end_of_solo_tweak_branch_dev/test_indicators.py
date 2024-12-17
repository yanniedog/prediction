# test_indicators.py
import json
import logging
import pandas as pd
import numpy as np
import talib
import pandas_ta as ta
import os
from logging_setup import configure_logging

def load_indicator_params(json_path):
    try:
        with open(json_path, 'r') as f:
            params = json.load(f)
        indicators = params.get('indicators', {})
        logging.info(f"Indicator Parameters Loaded: {list(indicators.keys())}")
        return indicators
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
    logging.debug(f"Simulated Data Summary:\n{data.describe()}")
    return data

def compute_custom_indicator(indicator_name, params, data):
    try:
        if indicator_name == 'EyeX MFV Volume':
            ranges = params.get('ranges', {}).get('default', [])
            ranges = sorted(ranges)
            for r in ranges:
                column_name = f"MFV_Volume_{r}"
                data[column_name] = data['volume'].rolling(window=r).mean()
            logging.info(f"Indicator '{indicator_name}' computed successfully.")
        
        elif indicator_name == 'EyeX MFV S/R Bull':
            ranges = params.get('ranges', {}).get('default', [])
            pivot_lookback = params.get('pivot_lookback', {}).get('default', 5)
            price_proximity = params.get('price_proximity', {}).get('default', 0.00001)
            for r in ranges:
                data[f"MFV_SRB_{r}"] = data['volume'].rolling(window=r).mean()
            data['Pivot_High'] = data['high'].rolling(window=pivot_lookback, center=True).max()
            data['Pivot_Low'] = data['low'].rolling(window=pivot_lookback, center=True).min()
            data['Support'] = np.where(abs(data['close'] - data['Pivot_Low']) <= price_proximity, data['Pivot_Low'], np.nan)
            data['Resistance'] = np.where(abs(data['close'] - data['Pivot_High']) <= price_proximity, data['Pivot_High'], np.nan)
            logging.info(f"Indicator '{indicator_name}' computed successfully.")
        
        else:
            logging.warning(f"Custom indicator '{indicator_name}' is not implemented.")
    except Exception as e:
        logging.error(f"Error computing Custom Indicator '{indicator_name}': {e}")
        raise

def compute_ta_lib_indicator(indicator_name, params, data):
    try:
        extracted_params = {k: v['default'] for k, v in params.items() if 'default' in v and k not in ['required_inputs', 'input_columns', 'conditions']}
        func = getattr(talib, indicator_name.upper())
        required_inputs = params.get('required_inputs', [])
        
        inputs = []
        for inp in required_inputs:
            if inp in data.columns:
                inputs.append(data[inp].values)
            else:
                logging.error(f"Required input '{inp}' for indicator '{indicator_name}' is missing in data.")
                raise ValueError(f"Missing required input: {inp}")
        
        logging.debug(f"TA-Lib Indicator '{indicator_name}': Inputs = {required_inputs}, Parameters = {extracted_params}")
        
        result = func(*inputs, **extracted_params)
        
        if isinstance(result, tuple):
            for idx, res in enumerate(result):
                data[f"{indicator_name.upper()}_{idx}"] = res
                logging.debug(f"Added column '{indicator_name.upper()}_{idx}' with {res.size} elements.")
        else:
            data[f"{indicator_name.upper()}"] = result
            logging.debug(f"Added column '{indicator_name.upper()}' with {result.size} elements.")
        logging.info(f"Indicator '{indicator_name}' computed successfully using TA-Lib.")
    except AttributeError:
        logging.error(f"TA-Lib does not have a function named '{indicator_name.upper()}'.")
        raise
    except TypeError as te:
        logging.error(f"Error computing TA-Lib indicator '{indicator_name}': {te}")
        raise
    except Exception as e:
        logging.error(f"Error computing TA-Lib indicator '{indicator_name}': {e}")
        raise

def compute_pandas_ta_indicator(indicator_name, params, data):
    try:
        extracted_params = {k: v['default'] for k, v in params.items() if 'default' in v and k not in ['required_inputs', 'input_columns', 'conditions']}
        func = getattr(ta, indicator_name.lower())
        required_inputs = params.get('required_inputs', [])
        
        inputs = []
        for inp in required_inputs:
            if inp in data.columns:
                inputs.append(data[inp])
            else:
                logging.error(f"Required input '{inp}' for indicator '{indicator_name}' is missing in data.")
                raise ValueError(f"Missing required input: {inp}")
        
        logging.debug(f"pandas-ta Indicator '{indicator_name}': Inputs = {required_inputs}, Parameters = {extracted_params}")
        
        result = func(*inputs, **extracted_params)
        
        if isinstance(result, pd.DataFrame):
            for col in result.columns:
                data[col] = result[col]
                logging.debug(f"Added column '{col}' with {result[col].notna().sum()} non-NaN values.")
        else:
            param_values = '_'.join(map(str, extracted_params.values()))
            data[f"{indicator_name.upper()}_{param_values}"] = result
            logging.debug(f"Added column '{indicator_name.upper()}_{param_values}' with {result.notna().sum()} non-NaN values.")
        logging.info(f"Indicator '{indicator_name}' computed successfully using pandas-ta.")
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
        conditions = config.get('conditions', [])
        
        if conditions:
            condition_met = True
            for condition in conditions:
                for param, rule in condition.items():
                    for rule_type, rule_value in rule.items():
                        param_value = params[param]['default']
                        if isinstance(rule_value, str) and rule_value in params:
                            compare_to = params[rule_value]['default']
                        else:
                            try:
                                compare_to = float(rule_value)
                            except ValueError:
                                logging.error(f"Invalid comparison value '{rule_value}' in conditions for indicator '{indicator_name}'.")
                                condition_met = False
                                break
                        if rule_type == 'greater_than' and not (param_value > compare_to):
                            condition_met = False
                            break
                        elif rule_type == 'less_than' and not (param_value < compare_to):
                            condition_met = False
                            break
                        elif rule_type == 'less_than_or_equal' and not (param_value <= compare_to):
                            condition_met = False
                            break
                if not condition_met:
                    logging.info(f"Conditions not met for indicator '{indicator_name}'. Skipping computation.")
                    break
            if not condition_met:
                continue
        
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
        if config['type'] == 'custom':
            if indicator_name == 'EyeX MFV Volume':
                col_prefix = 'MFV_Volume'
                mfv_cols = [col for col in data.columns if col.startswith(col_prefix)]
                if not mfv_cols:
                    logging.error(f"No columns found for indicator '{indicator_name}'.")
                    continue
                for col in mfv_cols:
                    non_nan = data[col].notna().sum()
                    total = len(data)
                    logging.debug(f"Indicator '{col}' has {non_nan} non-NaN values out of {total}.")
            elif indicator_name == 'EyeX MFV S/R Bull':
                required_cols = [col for col in data.columns if col.startswith('MFV_SRB')] + ['Pivot_High', 'Pivot_Low', 'Support', 'Resistance']
                missing_cols = [col for col in required_cols if col not in data.columns]
                if missing_cols:
                    logging.error(f"No columns found for indicator '{indicator_name}'. Missing: {missing_cols}")
                    continue
                for col in required_cols:
                    non_nan = data[col].notna().sum()
                    total = len(data)
                    logging.debug(f"Indicator '{col}' has {non_nan} non-NaN values out of {total}.")
        else:
            possible_cols = [indicator_name.upper()]
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
        logging.info("Completed validation of computed indicators.")

def main():
    try:
        configure_logging(log_file_prefix='predictions')
        json_path = 'indicator_params.json'
        if not os.path.exists(json_path):
            logging.error(f"Indicator parameters file '{json_path}' does not exist.")
            return
        indicators = load_indicator_params(json_path)
        data = generate_simulated_data(num_rows=200)
        logging.info("Starting computation of indicators.")
        compute_indicators(indicators, data)
        logging.info("Starting validation of computed indicators.")
        validate_indicators(indicators, data)
        logging.info("Indicator testing completed.")
    except Exception as e:
        logging.error(f"An error occurred in the main execution: {e}")

if __name__ == "__main__":
    main()
