# indicator_factory.py
import pandas as pd
import numpy as np
import talib as ta
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Set
import json
import os
from custom_indicators import (
    compute_obv_price_divergence, compute_volume_oscillator,
    compute_vwap, compute_pvi, compute_nvi, compute_returns, compute_volatility,
    custom_rsi, register_custom_indicator, _custom_indicator_registry,
    custom_indicators
)
import pandas_ta as pta

logger = logging.getLogger(__name__)

class IndicatorFactory:
    """Factory class for computing technical indicators."""

    def __init__(self, params_file: str = "indicator_params.json"):
        """Initialize with indicator parameters from JSON file."""
        self.params_file = params_file
        self.indicator_params = self._load_params()
        self._validate_params()
        # Register built-in custom indicators
        register_custom_indicator('custom_rsi', custom_rsi)

    def _load_params(self) -> Dict[str, Any]:
        """Load indicator parameters from JSON file."""
        try:
            if not os.path.exists(self.params_file):
                logger.error(f"Indicator parameters file not found: {self.params_file}")
                raise FileNotFoundError(f"Indicator parameters file not found: {self.params_file}")
            
            with open(self.params_file, 'r') as f:
                params = json.load(f)
            
            if not isinstance(params, dict):
                logger.error("Invalid indicator parameters format - must be a dictionary")
                raise ValueError("Invalid indicator parameters format")
            
            if not params:
                logger.error("No indicator parameters found in file")
                raise ValueError("Empty indicator parameters")
            
            # Validate each indicator definition
            for name, config in params.items():
                if not isinstance(config, dict):
                    logger.error(f"Invalid config format for indicator {name}")
                    raise ValueError(f"Invalid config format for indicator {name}")
                
                required_keys = ['name', 'type', 'required_inputs', 'params']
                missing_keys = [k for k in required_keys if k not in config]
                if missing_keys:
                    logger.error(f"Missing required keys {missing_keys} for indicator {name}")
                    raise ValueError(f"Missing required keys for indicator {name}")
                
                if config['type'] not in ['talib', 'custom', 'ta-lib', 'pandas-ta']:
                    logger.error(f"Invalid indicator type {config['type']} for {name}")
                    raise ValueError(f"Invalid indicator type for {name}")
                
                if not isinstance(config['required_inputs'], list):
                    logger.error(f"Invalid required_inputs format for indicator {name}")
                    raise ValueError(f"Invalid required_inputs format for {name}")
                
                if not isinstance(config['params'], dict):
                    logger.error(f"Invalid params format for indicator {name}")
                    raise ValueError(f"Invalid params format for {name}")
                
            logger.info(f"Successfully loaded and validated {len(params)} indicator configurations")
            return params
        
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in indicator parameters file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading indicator parameters: {e}", exc_info=True)
            raise

    def _validate_params(self) -> None:
        """Validate indicator parameters and their conditions."""
        for name, config in self.indicator_params.items():
            # Validate parameter ranges and conditions
            params = config.get('params', {})
            conditions = config.get('conditions', [])
            
            # Check parameter ranges
            for param_name, param_def in params.items():
                if isinstance(param_def, dict):
                    if 'min' in param_def and 'max' in param_def:
                        if param_def['min'] >= param_def['max']:
                            logger.error(f"Invalid parameter range for {name}.{param_name}: min >= max")
                            raise ValueError(f"Invalid parameter range for {name}.{param_name}")
                        
                    if 'default' in param_def:
                        default = param_def['default']
                        if 'min' in param_def and default < param_def['min']:
                            logger.error(f"Default value {default} below minimum for {name}.{param_name}")
                            raise ValueError(f"Invalid default value for {name}.{param_name}")
                        if 'max' in param_def and default > param_def['max']:
                            logger.error(f"Default value {default} above maximum for {name}.{param_name}")
                            raise ValueError(f"Invalid default value for {name}.{param_name}")
                        
            # Validate conditions
            for condition in conditions:
                if not isinstance(condition, dict):
                    logger.error(f"Invalid condition format for {name}")
                    raise ValueError(f"Invalid condition format for {name}")
                
                for param, rules in condition.items():
                    if param not in params:
                        logger.error(f"Condition references unknown parameter {param} for {name}")
                        raise ValueError(f"Invalid condition parameter for {name}")
                    
                    for rule, value in rules.items():
                        if rule not in ['gte', 'lte', 'gt', 'lt', 'eq']:
                            logger.error(f"Invalid condition operator {rule} for {name}.{param}")
                            raise ValueError(f"Invalid condition operator for {name}.{param}")
                        
                        if isinstance(value, str) and value not in params:
                            logger.error(f"Condition references unknown parameter {value} for {name}.{param}")
                            raise ValueError(f"Invalid condition reference for {name}.{param}")

    def _get_ta_lib_output_suffixes(self, func_name: str) -> List[str]:
        """Get output names for TA-Lib function. Default to function name."""
        return [func_name]

    def _compute_single_indicator(self, data: pd.DataFrame, indicator_name: str, config: Dict) -> pd.DataFrame:
        """Compute a single indicator using the provided configuration."""
        try:
            # Get indicator type from config or default to talib
            indicator_type = config.get('type', 'talib')
            
            # Normalize indicator name case
            indicator_name = indicator_name.upper() if indicator_type == 'talib' else indicator_name.lower()

            # Validate input data
            if data.empty:
                raise ValueError("Input data is empty")
            
            # Check for NaN values
            if data.isna().any().any():
                logger.warning("NaN values detected in input data")
                data = data.ffill().bfill()
            
            # Get required columns from config
            required_cols = config.get('required_inputs', ['close'])
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Get the indicator function
            if indicator_type == 'talib':
                # Handle BB/BBANDS special case
                if indicator_name in ['BB', 'BBANDS']:
                    ta_func = getattr(ta, 'BBANDS')
                else:
                    try:
                        ta_func = getattr(ta, indicator_name)
                    except AttributeError:
                        # Try lowercase if uppercase fails
                        ta_func = getattr(ta, indicator_name.lower())
            elif indicator_type == 'custom':
                # Try to get from custom registry first
                if indicator_name.lower() in _custom_indicator_registry:
                    ta_func = _custom_indicator_registry[indicator_name.lower()]
                else:
                    # Try to get from custom_indicators module
                    try:
                        ta_func = getattr(custom_indicators, f"compute_{indicator_name.lower()}")
                    except AttributeError:
                        raise ValueError(f"Custom indicator {indicator_name} not found")
            else:
                raise ValueError(f"Unsupported indicator type: {indicator_type}")
            
            # Prepare input arrays for TA-Lib functions
            inputs = {}
            for param_name, col_name in config.get('input_mapping', {}).items():
                if col_name in data.columns:
                    inputs[param_name] = data[col_name].values
            if not inputs and 'close' in data.columns:  # Default to close price if no mapping
                inputs['real'] = data['close'].values
            
            # Add any additional parameters from config
            params = config.get('params', {})
            
            # Convert parameter values from dict format if needed
            processed_params = {}
            for key, value in params.items():
                if isinstance(value, dict):
                    processed_params[key] = value.get('default', value.get('value', value))
                else:
                    processed_params[key] = value
            params = processed_params
            
            # Handle special cases for different indicators
            if indicator_name in ['BB', 'BBANDS']:
                # Ensure proper parameter names for BBANDS
                if 'timeperiod' in params:
                    timeperiod = params.pop('timeperiod')
                    params['timeperiod'] = int(timeperiod)
                if 'nbdevup' in params:
                    nbdevup = params.pop('nbdevup')
                    params['nbdevup'] = float(nbdevup)
                if 'nbdevdn' in params:
                    nbdevdn = params.pop('nbdevdn')
                    params['nbdevdn'] = float(nbdevdn)
                results = ta_func(inputs['real'], **params)
                # Rename columns to match indicator name
                if isinstance(results, tuple):
                    output_names = [f"{indicator_name}_upper", f"{indicator_name}_middle", f"{indicator_name}_lower"]
                    result_df = pd.DataFrame({name: values for name, values in zip(output_names, results)})
                else:
                    result_df = pd.DataFrame({f"{indicator_name}": results})
            else:
                # Compute indicator with all parameters
                try:
                    if indicator_type == 'custom':
                        results = ta_func(data, **params)
                    else:
                        results = ta_func(**inputs, **params)
                except TypeError as e:
                    # Handle positional arguments for TA-Lib functions
                    if "takes at least 1 positional argument" in str(e):
                        # Convert inputs dict to positional args in correct order
                        ordered_inputs = [inputs[param] for param in config.get('input_order', ['real'])]
                        results = ta_func(*ordered_inputs, **params)
                    else:
                        raise

                # Convert results to DataFrame if not already done
                if isinstance(results, tuple):
                    # Handle multiple outputs
                    output_names = config.get('output_names', [f"{indicator_name}_{i}" for i in range(len(results))])
                    result_df = pd.DataFrame({name: values for name, values in zip(output_names, results)})
                elif isinstance(results, pd.Series):
                    result_df = pd.DataFrame({indicator_name: results})
                elif isinstance(results, pd.DataFrame):
                    result_df = results
                else:
                    # Single output
                    result_df = pd.DataFrame({indicator_name: results})

            # Set index to match input data
            result_df.index = data.index
            return result_df

        except Exception as e:
            logger.error(f"Error computing indicator {indicator_name}: {str(e)}")
            raise ValueError(f"Error computing indicator {indicator_name}: {str(e)}")

    def compute_indicators(self, data: pd.DataFrame, indicators: Optional[List[str]] = None) -> pd.DataFrame:
        """Compute indicators for the given data."""
        if data is None:
            raise ValueError("Input data cannot be None")
        if data.empty:
            raise ValueError("Input data is empty")
        
        # Validate required columns
        required_cols = set()
        for config in self.indicator_params.values():
            required_cols.update(config.get('required_columns', ['close']))
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Filter indicators if specified
        if indicators:
            invalid_indicators = [ind for ind in indicators if ind not in self.indicator_params]
            if invalid_indicators:
                raise ValueError(f"Invalid indicators: {invalid_indicators}")
            configs = {name: self.indicator_params[name] for name in indicators}
        else:
            configs = self.indicator_params
        
        # Initialize result DataFrame with input data
        result = data.copy()
        
        # Compute each indicator
        for indicator_name, config in configs.items():
            try:
                indicator_result = self._compute_single_indicator(data, indicator_name, config)
                result = pd.concat([result, indicator_result], axis=1)
            except Exception as e:
                logger.error(f"Error computing {indicator_name}: {str(e)}")
                continue
        
        return result

    def get_available_indicators(self) -> List[str]:
        """Get list of available indicators."""
        return list(self.indicator_params.keys())

    def get_all_indicator_names(self) -> List[str]:
        """Return all indicator names (alias for get_available_indicators)."""
        return self.get_available_indicators()

    def validate_params(self, indicator_name: str, params: Dict[str, Any]) -> None:
        """Validate indicator parameters against their definitions.
        
        Args:
            indicator_name: Name of the indicator to validate
            params: Dictionary of parameter values to validate
            
        Raises:
            ValueError: If indicator name is invalid or parameters don't match definition
        """
        if indicator_name not in self.indicator_params:
            raise ValueError(f"Unknown indicator: {indicator_name}")
            
        definition = self.indicator_params[indicator_name]
        param_defs = definition.get('params', {})
        # Parameters that are actually required data columns, not user-supplied
        data_columns = {'real', 'close', 'open', 'high', 'low', 'volume', 'input', 'price'}
        
        # Check for required parameters (skip data columns)
        for param_name, param_def in param_defs.items():
            if param_name in data_columns:
                continue  # skip validation for data columns
            if param_name not in params:
                raise ValueError(f"Missing required parameter '{param_name}' for indicator '{indicator_name}'")
            # Validate parameter value is within range if min/max defined
            if isinstance(param_def, dict):
                value = params[param_name]
                if 'min' in param_def and value < param_def['min']:
                    raise ValueError(f"Parameter '{param_name}' value {value} below minimum {param_def['min']} for indicator '{indicator_name}'")
                if 'max' in param_def and value > param_def['max']:
                    raise ValueError(f"Parameter '{param_name}' value {value} above maximum {param_def['max']} for indicator '{indicator_name}'")
                # Validate type if specified
                if 'type' in param_def:
                    expected_type = param_def['type']
                    if expected_type == 'int' and not isinstance(value, int):
                        raise ValueError(f"Parameter '{param_name}' must be integer for indicator '{indicator_name}'")
                    elif expected_type == 'float' and not isinstance(value, (int, float)):
                        raise ValueError(f"Parameter '{param_name}' must be numeric for indicator '{indicator_name}'")
                    elif expected_type == 'str' and not isinstance(value, str):
                        raise ValueError(f"Parameter '{param_name}' must be string for indicator '{indicator_name}'")

    def get_indicator_params(self, name: str) -> Optional[Dict[str, Any]]:
        """Get parameters for a specific indicator."""
        indicator_def = self.indicator_params.get(name)
        if not indicator_def:
            raise ValueError("Unknown indicator")
        return indicator_def.get('params') if indicator_def else None

    def create_custom_indicator(self, name: str, func, data: pd.DataFrame, **params):
        """Register and compute a custom indicator on the fly."""
        result = func(data, **params)
        if isinstance(result, pd.Series):
            return pd.DataFrame({name: result})
        elif isinstance(result, pd.DataFrame):
            return result
        else:
            return pd.DataFrame({name: pd.Series(result, index=data.index)})

    def create_indicator(self, name: str, data: pd.DataFrame, **params):
        """Compute a standard indicator by name, using provided data and parameters."""
        # Validate input data first
        if data is None or not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")
        if data.empty:
            raise ValueError("Input data cannot be empty")
        
        # Find the indicator config
        config = None
        if name.lower() in self.indicator_params:
            config = self.indicator_params[name.lower()]
        elif name.upper() in self.indicator_params:
            config = self.indicator_params[name.upper()]
        elif name in self.indicator_params:
            config = self.indicator_params[name]
        else:
            raise ValueError(f"Unknown indicator: {name}")
        
        # Merge provided params with defaults
        merged_params = config['params'].copy()
        merged_params.update(params)
        
        # Handle BBANDS special case - convert period to timeperiod
        if name.upper() in ['BB', 'BBANDS']:
            if 'period' in merged_params:
                period_val = merged_params.pop('period')
                if isinstance(period_val, dict):
                    period_val = period_val.get('default', 20)
                merged_params['timeperiod'] = int(period_val)
            if 'std_dev' in merged_params:
                std_val = merged_params.pop('std_dev')
                if isinstance(std_val, dict):
                    std_val = std_val.get('default', 2.0)
                merged_params['nbdevup'] = float(std_val)
                merged_params['nbdevdn'] = float(std_val)
        
        # Check for minimum data length if period/length/timeperiod is specified
        period_param = None
        period_key = None
        for key in ["period", "length", "timeperiod"]:
            if key in merged_params:
                period_param = merged_params[key]
                period_key = key
                break
        if period_param is not None:
            try:
                # If the param is a dict, use its 'default' value
                if isinstance(period_param, dict):
                    period_val = int(period_param.get('default', 1))
                else:
                    period_val = int(period_param)
                if len(data) < period_val:
                    raise ValueError(f"Insufficient data: input data length ({len(data)}) is less than required {period_key} ({period_val}) for indicator '{name}'")
            except (ValueError, TypeError):
                raise ValueError(f"Invalid {period_key} value '{period_param}' for indicator '{name}'")
        
        # Build a config dict for _compute_single_indicator
        config_for_compute = config.copy()
        config_for_compute['params'] = merged_params
        
        # Validate required columns before computation
        required_cols = []
        if config['type'] in ['talib', 'ta-lib']:
            # Map parameter names to required columns for TA-Lib
            if 'high' in merged_params: required_cols.append('high')
            if 'low' in merged_params: required_cols.append('low')
            if 'close' in merged_params: required_cols.append('close')
            if 'volume' in merged_params: required_cols.append('volume')
            if 'open' in merged_params: required_cols.append('open')
        elif config['type'] == 'pandas-ta':
            required_cols = config.get('required_inputs', [])
        elif config['type'] == 'custom':
            required_cols = config.get('required_inputs', [])
        
        if required_cols:
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns for indicator '{name}': {missing_cols}")
        
        result = self._compute_single_indicator(data, name, config_for_compute)
        return result

    def plot_indicator(self, indicator_name: str, data: pd.DataFrame, params: Dict[str, Any], 
                      output_path: Optional[str] = None) -> None:
        """Plot an indicator's values against the price data.
        
        Args:
            indicator_name: Name of the indicator to plot
            data: DataFrame containing price data
            params: Parameters for the indicator
            output_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            # Compute the indicator
            indicator_values = self.create_indicator(indicator_name, data, **params)
            if indicator_values is None:
                raise ValueError(f"Failed to compute indicator {indicator_name}")
            
            # Create the plot
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Plot price on primary y-axis
            ax1.plot(data.index, data['close'], 'b-', label='Price')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            
            # Plot indicator on secondary y-axis
            ax2 = ax1.twinx()
            ax2.plot(data.index, indicator_values, 'r-', label=indicator_name)
            ax2.set_ylabel(indicator_name, color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            # Add legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            plt.title(f'{indicator_name} vs Price')
            plt.grid(True)
            
            if output_path:
                plt.savefig(output_path)
                plt.close()
            else:
                plt.show()
            
        except ImportError:
            logger.error("Matplotlib not installed. Cannot plot indicator.")
            raise
        except Exception as e:
            logger.error(f"Error plotting indicator {indicator_name}: {e}", exc_info=True)
            raise

def compute_configured_indicators(data: pd.DataFrame, indicator_configs: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, Set[int]]:
    """
    Compute indicators for multiple configurations and return the combined DataFrame with failed config IDs.
    
    Args:
        data: Input DataFrame with required columns
        indicator_configs: List of indicator configurations to compute
        
    Returns:
        Tuple of (DataFrame with computed indicators, Set of failed config IDs)
    """
    logger = logging.getLogger(__name__)
    factory = IndicatorFactory()
    result_df = data.copy()  # Initialize result_df at the start
    failed_config_ids: Set[int] = set()
    
    for config in indicator_configs:
        try:
            # Use dict.get() with default values to avoid None
            indicator_name = config.get('indicator_name', '')
            if not indicator_name:
                logger.error("Config missing indicator_name")
                continue
                
            # Use dict.get() with type checking
            config_id = config.get('config_id')
            if not isinstance(config_id, int):
                logger.error(f"Invalid config_id type for {indicator_name}")
                continue
                
            # Use dict.get() with default empty dict
            params = config.get('params', {})
            if not isinstance(params, dict):
                logger.error(f"Invalid params type for {indicator_name}")
                if isinstance(config_id, int):  # Only add if config_id is valid
                    failed_config_ids.add(config_id)
                continue
                
            # Get indicator definition using dict.get()
            indicator_def = factory.indicator_params.get(indicator_name)
            if not indicator_def:
                logger.error(f"Unknown indicator: {indicator_name}")
                if isinstance(config_id, int):  # Only add if config_id is valid
                    failed_config_ids.add(config_id)
                continue
                
            # Compute single indicator
            indicator_df = factory._compute_single_indicator(data, indicator_name, indicator_def)
            if indicator_df is None or indicator_df.empty:
                logger.error(f"Failed to compute indicator {indicator_name} (ID: {config_id})")
                if isinstance(config_id, int):  # Only add if config_id is valid
                    failed_config_ids.add(config_id)
                continue
                
            # Add computed columns to result, handling duplicates by appending _new
            for col in indicator_df.columns:
                if col in result_df.columns:
                    new_col = f"{col}_new"
                    result_df[new_col] = indicator_df[col]
                else:
                    result_df[col] = indicator_df[col]
                
        except Exception as e:
            logger.error(f"Error computing indicator config {config.get('config_id', 'unknown')}: {e}", exc_info=True)
            # Only add config_id if it exists and is an int
            config_id = config.get('config_id')
            if isinstance(config_id, int):
                failed_config_ids.add(config_id)
                
    return result_df, failed_config_ids