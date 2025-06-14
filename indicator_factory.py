# indicator_factory.py
"""Indicator factory for creating and managing technical indicators."""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Callable, Set, Tuple
import json
import os
from pathlib import Path
import inspect
import warnings

# Set matplotlib backend to non-interactive for testing
import matplotlib
matplotlib.use('Agg')

# Import talib conditionally
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    warnings.warn("TA-Lib not available. Some indicators may not work.")

# Import project modules
import config
import utils
from custom_indicators import (
    compute_obv_price_divergence, compute_volume_oscillator,
    compute_vwap, compute_pvi, compute_nvi, compute_returns, compute_volatility,
    custom_rsi, register_custom_indicator, _custom_indicator_registry,
    custom_indicators
)

# Try to import pandas_ta, but make it optional
try:
    import pandas_ta as pta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    pta = None
    warnings.warn("pandas_ta not available, some indicators may not work")

import itertools

logger = logging.getLogger(__name__)

# Global registry for custom indicators
_custom_indicator_registry = {}

class IndicatorFactory:
    """Factory class for creating and managing technical indicators."""
    
    def __init__(self, params_file: str = "indicator_params.json"):
        """Initialize the factory with indicator parameters."""
        self.params_file = params_file
        self.indicator_params = self._load_params()
        self._validate_params()
        
        # Log successful initialization
        logger.info(f"Successfully loaded and validated {len(self.indicator_params)} indicator configurations")
        # Register built-in custom indicators
        register_custom_indicator('custom_rsi', custom_rsi)
        register_custom_indicator('returns', compute_returns)

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
            
            # Create case-insensitive lookups
            expanded_params = {}
            for name, config in params.items():
                # Add original name
                expanded_params[name] = config
                # Add lowercase version
                expanded_params[name.lower()] = config
                # Add uppercase version
                expanded_params[name.upper()] = config
            
            logger.info(f"Successfully loaded and validated {len(params)} indicator configurations")
            return expanded_params
        
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

    def _compute_single_indicator(self, data: pd.DataFrame, indicator_name: str, config: Dict[str, Any], params: Dict[str, Any]) -> pd.DataFrame:
        """Compute a single indicator with the given configuration."""
        logger = logging.getLogger(__name__)
        
        try:
            # Validate inputs
            if data is None or data.empty:
                raise ValueError("Input data is empty")
            if not indicator_name:
                raise ValueError("Indicator name is required")
            if not isinstance(config, dict):
                raise ValueError("Config must be a dictionary")
            if not isinstance(params, dict):
                raise ValueError("Params must be a dictionary")

            # Get indicator type and function
            indicator_type = config.get('type', 'talib')
            required_inputs = config.get('required_inputs', ['close'])
            
            # Validate required inputs
            missing_inputs = [col for col in required_inputs if col not in data.columns]
            if missing_inputs:
                raise ValueError(f"Missing required inputs: {missing_inputs}")

            # Filter parameters to only include those valid for this indicator
            valid_params = {}
            param_defs = config.get('params', {})
            
            for param_name, param_value in params.items():
                # Only include parameters that are defined in the indicator config
                if param_name in param_defs:
                    valid_params[param_name] = param_value
                # Also include standard talib parameters that might be mapped
                elif param_name in ['timeperiod', 'nbdevup', 'nbdevdn', 'fastperiod', 'slowperiod', 'signalperiod']:
                    valid_params[param_name] = param_value

            # Handle different indicator types
            if indicator_type == 'talib':
                return self._compute_talib_indicator(data, indicator_name, valid_params)
            elif indicator_type == 'custom':
                return self._compute_custom_indicator(data, indicator_name, valid_params)
            else:
                raise ValueError(f"Unsupported indicator type: {indicator_type}")

        except Exception as e:
            logger.error(f"Error computing indicator {indicator_name}: {str(e)}")
            raise ValueError(f"Error computing indicator {indicator_name}: {str(e)}")

    def _compute_talib_indicator(self, data: pd.DataFrame, indicator_name: str, params: Dict[str, Any]) -> pd.DataFrame:
        """Compute a TA-Lib indicator."""
        try:
            import talib
            
            # Get the indicator config to find the correct TA-Lib function name
            config = None
            if indicator_name.lower() in self.indicator_params:
                config = self.indicator_params[indicator_name.lower()]
            elif indicator_name.upper() in self.indicator_params:
                config = self.indicator_params[indicator_name.upper()]
            elif indicator_name in self.indicator_params:
                config = self.indicator_params[indicator_name]
            
            # Use the name from config if available, otherwise use the input name
            if config and 'name' in config:
                func_name = config['name'].upper()
            else:
                func_name = indicator_name.upper()
            
            if not hasattr(talib, func_name):
                raise ValueError(f"TA-Lib function {func_name} not found")
            
            func = getattr(talib, func_name)
            
            # Filter parameters to only include those valid for this specific TA-Lib function
            # Get the function signature to know which parameters are valid
            sig = inspect.signature(func)
            valid_params = list(sig.parameters.keys())
            
            # Filter params to only include valid ones
            filtered_params = {}
            for key, value in params.items():
                if key in valid_params:
                    filtered_params[key] = value
            
            # Prepare input data based on the function signature
            # Check if the function expects OHLC data
            if len(sig.parameters) > 1:  # More than just the first parameter
                param_names = list(sig.parameters.keys())
                if 'high' in param_names and 'low' in param_names and 'close' in param_names:
                    if 'open' in param_names:
                        input_data = (data['open'], data['high'], data['low'], data['close'])
                    else:
                        input_data = (data['high'], data['low'], data['close'])
                elif 'close' in param_names:
                    input_data = data['close']
                elif 'volume' in param_names:
                    input_data = data['volume']
                else:
                    # Default to close price
                    input_data = data['close']
            else:
                # Single parameter function, default to close price
                input_data = data['close']
            
            # Call the function
            if isinstance(input_data, tuple):
                results = func(*input_data, **filtered_params)
            else:
                results = func(input_data, **filtered_params)
            
            # Handle results
            if isinstance(results, tuple):
                # Multiple outputs
                result_df = pd.DataFrame()
                # Use output names from config if available
                if config and 'output_names' in config:
                    output_names = config['output_names']
                    for i, result in enumerate(results):
                        if i < len(output_names):
                            col_name = output_names[i]
                        else:
                            col_name = f"{indicator_name}_{i}"
                        result_df[col_name] = result
                else:
                    suffixes = self._get_ta_lib_output_suffixes(func_name)
                    for i, result in enumerate(results):
                        if i < len(suffixes):
                            col_name = f"{indicator_name}_{suffixes[i]}"
                        else:
                            col_name = f"{indicator_name}_{i}"
                        result_df[col_name] = result
            elif isinstance(results, np.ndarray):
                # Single output
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
            logger.error(f"Error computing TA-Lib indicator {indicator_name}: {str(e)}")
            raise ValueError(f"Error computing TA-Lib indicator {indicator_name}: {str(e)}")

    def _compute_custom_indicator(self, data: pd.DataFrame, indicator_name: str, params: Dict[str, Any]) -> pd.DataFrame:
        """Compute a custom indicator."""
        try:
            if indicator_name not in _custom_indicator_registry:
                raise ValueError(f"Custom indicator {indicator_name} not found")
            
            func = _custom_indicator_registry[indicator_name]
            results = func(data, **params)
            
            if isinstance(results, pd.Series):
                return pd.DataFrame({indicator_name: results})
            elif isinstance(results, pd.DataFrame):
                return results
            else:
                return pd.DataFrame({indicator_name: pd.Series(results, index=data.index)})

        except Exception as e:
            logger.error(f"Error computing custom indicator {indicator_name}: {str(e)}")
            raise ValueError(f"Error computing custom indicator {indicator_name}: {str(e)}")

    def compute_indicators(self, data: pd.DataFrame, indicators: Optional[Union[List[str], Dict[str, dict]]] = None) -> pd.DataFrame:
        """Compute indicators for the given data."""
        if data is None:
            raise ValueError("Input data cannot be None")
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")
        if data.empty:
            raise ValueError("Input data is empty")

        # Validate required columns
        required_cols = set()
        for config in self.indicator_params.values():
            required_cols.update(config.get('required_inputs', ['close']))
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Support both list and dict for indicators argument
        indicator_configs = {}
        if indicators is None:
            indicator_names = list(self.indicator_params.keys())
        elif isinstance(indicators, dict):
            indicator_names = list(indicators.keys())
            indicator_configs = indicators
        elif isinstance(indicators, list):
            indicator_names = indicators
        else:
            raise ValueError("Indicators argument must be a list or dict")

        # Allow custom indicators registered at runtime
        all_indicators = set(self.indicator_params.keys()) | set(_custom_indicator_registry.keys())
        invalid_indicators = [ind for ind in indicator_names if ind not in all_indicators]
        if invalid_indicators:
            raise ValueError(f"Invalid indicators: {invalid_indicators}")

        result = data.copy()
        for name in indicator_names:
            # Prefer custom indicator registry if present
            if name in _custom_indicator_registry:
                func = _custom_indicator_registry[name]
                params = indicator_configs.get(name, {})
                ind_result = func(data, **params) if callable(func) else None
                if isinstance(ind_result, pd.Series):
                    result[name] = ind_result
                elif isinstance(ind_result, pd.DataFrame):
                    for col in ind_result.columns:
                        result[col] = ind_result[col]
                continue
            # Otherwise use standard indicator params
            config = self.indicator_params.get(name)
            if not config:
                continue
            params = indicator_configs.get(name, {})
            try:
                ind_result = self._compute_single_indicator(data, name, config, params)
                if isinstance(ind_result, pd.Series):
                    result[name] = ind_result
                elif isinstance(ind_result, pd.DataFrame):
                    for col in ind_result.columns:
                        result[col] = ind_result[col]
            except Exception as e:
                logger.error(f"Error computing indicator {name}: {e}")
        return result

    def get_available_indicators(self) -> List[str]:
        """Return a list of all available indicators, including custom ones."""
        names = set(self.indicator_params.keys())
        # Add custom indicators from the registry
        names.update(_custom_indicator_registry.keys())
        return sorted(names)

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
                # Check if parameter has a default value
                if isinstance(param_def, dict) and 'default' in param_def:
                    params[param_name] = param_def['default']
                else:
                    raise ValueError(f"Missing required parameter '{param_name}' for indicator '{indicator_name}'")
            
            # Validate parameter value is within range if min/max defined
            if isinstance(param_def, dict):
                value = params[param_name]
                
                # Validate type
                if 'type' in param_def:
                    expected_type = param_def['type']
                    if expected_type == 'int':
                        try:
                            if value is not None:  # Skip validation for None values
                                value = int(value)
                                params[param_name] = value  # Update with converted value
                        except (ValueError, TypeError):
                            raise ValueError(f"Parameter '{param_name}' must be integer for indicator '{indicator_name}'")
                    elif expected_type == 'float':
                        try:
                            if value is not None:  # Skip validation for None values
                                value = float(value)
                                params[param_name] = value  # Update with converted value
                        except (ValueError, TypeError):
                            raise ValueError(f"Parameter '{param_name}' must be numeric for indicator '{indicator_name}'")
                    elif expected_type == 'str' and not isinstance(value, str):
                        if value is not None:  # Skip validation for None values
                            raise ValueError(f"Parameter '{param_name}' must be string for indicator '{indicator_name}'")
                
                # Validate range
                if 'min' in param_def and value < param_def['min']:
                    raise ValueError(f"Parameter '{param_name}' value {value} below minimum {param_def['min']} for indicator '{indicator_name}'")
                if 'max' in param_def and value > param_def['max']:
                    raise ValueError(f"Parameter '{param_name}' value {value} above maximum {param_def['max']} for indicator '{indicator_name}'")
                
                # Validate conditions
                if 'conditions' in definition:
                    for condition in definition['conditions']:
                        for cond_param, rules in condition.items():
                            if cond_param not in params:
                                continue  # Skip if condition parameter not provided
                            for rule, rule_value in rules.items():
                                if rule == 'gte' and params[cond_param] < rule_value:
                                    raise ValueError(f"Parameter '{cond_param}' must be >= {rule_value} for indicator '{indicator_name}'")
                                elif rule == 'lte' and params[cond_param] > rule_value:
                                    raise ValueError(f"Parameter '{cond_param}' must be <= {rule_value} for indicator '{indicator_name}'")
                                elif rule == 'gt' and params[cond_param] <= rule_value:
                                    raise ValueError(f"Parameter '{cond_param}' must be > {rule_value} for indicator '{indicator_name}'")
                                elif rule == 'lt' and params[cond_param] >= rule_value:
                                    raise ValueError(f"Parameter '{cond_param}' must be < {rule_value} for indicator '{indicator_name}'")
                                elif rule == 'eq' and params[cond_param] != rule_value:
                                    raise ValueError(f"Parameter '{cond_param}' must be = {rule_value} for indicator '{indicator_name}'")

    def get_indicator_params(self, name: str) -> Optional[Dict[str, Any]]:
        """Get parameters for a specific indicator."""
        # Try different case variations
        indicator_def = self.indicator_params.get(name)
        if not indicator_def:
            indicator_def = self.indicator_params.get(name.lower())
        if not indicator_def:
            indicator_def = self.indicator_params.get(name.upper())
        
        if not indicator_def:
            raise ValueError("Unknown indicator")
        return indicator_def.get('params') if indicator_def else None

    def create_custom_indicator(self, name: str, func, data: pd.DataFrame, **params):
        """Register and compute a custom indicator on the fly."""
        if func is None:
            raise ValueError(f"Custom indicator function for '{name}' is None")
        # Register if not already present
        if name.lower() not in _custom_indicator_registry:
            register_custom_indicator(name, func)
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
        
        # Extract default values from config and merge with provided params
        merged_params = {}
        config_params = config.get('params', {})
        
        for param_name, param_def in config_params.items():
            if isinstance(param_def, dict) and 'default' in param_def:
                merged_params[param_name] = param_def['default']
            else:
                merged_params[param_name] = param_def
        
        # Override with provided params
        merged_params.update(params)
        
        # Unify period/timeperiod/length
        period_val = None
        for key in ["period", "timeperiod", "length"]:
            if key in merged_params:
                period_val = merged_params[key]
                break
        if period_val is not None:
            merged_params["period"] = period_val
            merged_params["timeperiod"] = period_val
            merged_params["length"] = period_val
        
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
        required_cols = config.get('required_inputs', [])
        if required_cols:
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns for indicator '{name}': {missing_cols}")
        
        result = self._compute_single_indicator(data, name, config_for_compute, merged_params)
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
                # Check if we're in a test environment or non-interactive backend
                if os.environ.get('PYTEST_CURRENT_TEST') or matplotlib.get_backend() == 'Agg':
                    # In test environment, just save to a temporary file and close
                    plt.savefig('/tmp/test_plot.png')
                    plt.close()
                else:
                    plt.show()
            
        except ImportError:
            logger.error("Matplotlib not installed. Cannot plot indicator.")
            raise
        except Exception as e:
            logger.error(f"Error plotting indicator {indicator_name}: {e}", exc_info=True)
            raise

    def compute_configured_indicators(self, data: pd.DataFrame, indicator_configs: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, Set[int]]:
        """
        Compute indicators for multiple configurations and return the combined DataFrame with failed config IDs.
        
        Args:
            data: Input DataFrame with required columns
            indicator_configs: List of indicator configurations to compute
            
        Returns:
            Tuple of (DataFrame with computed indicators, Set of failed config IDs)
        """
        logger = logging.getLogger(__name__)
        result_df = data.copy()  # Initialize result_df at the start
        failed_config_ids: Set[int] = set()
        
        # Handle empty indicator_configs
        if not indicator_configs:
            logger.warning("No indicator configurations provided")
            return result_df, failed_config_ids
        
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
                indicator_def = self.indicator_params.get(indicator_name)
                if not indicator_def:
                    logger.error(f"Unknown indicator: {indicator_name}")
                    if isinstance(config_id, int):  # Only add if config_id is valid
                        failed_config_ids.add(config_id)
                    continue
                    
                # Compute single indicator
                indicator_df = self._compute_single_indicator(data, indicator_name, indicator_def, params)
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

    def plot_indicators(self, data: pd.DataFrame, indicators: Optional[Union[List[str], Dict[str, dict]]] = None, 
                       output_path: Optional[str] = None) -> None:
        """Plot multiple indicators against the price data.
        
        Args:
            data: DataFrame containing price data
            indicators: Optional list of indicator names or dict of indicator configs
            output_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            # Compute all indicators
            indicator_values = self.compute_indicators(data, indicators)
            if indicator_values is None or indicator_values.empty:
                raise ValueError("Failed to compute indicators")
            
            # Create the plot
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Plot price on primary y-axis
            ax1.plot(data.index, data['close'], 'b-', label='Price')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            
            # Plot each indicator on secondary y-axis
            ax2 = ax1.twinx()
            colors = ['r', 'g', 'm', 'c', 'y']  # Colors for different indicators
            for i, (name, values) in enumerate(indicator_values.items()):
                if name == 'close':  # Skip price data
                    continue
                color = colors[i % len(colors)]
                ax2.plot(data.index, values, f'{color}-', label=name)
            
            ax2.set_ylabel('Indicator Values', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            # Add legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            plt.title('Indicators vs Price')
            plt.grid(True)
            
            if output_path:
                plt.savefig(output_path)
                plt.close()
            else:
                # Check if we're in a test environment or non-interactive backend
                if os.environ.get('PYTEST_CURRENT_TEST') or matplotlib.get_backend() == 'Agg':
                    # In test environment, just save to a temporary file and close
                    plt.savefig('/tmp/test_plot.png')
                    plt.close()
                else:
                    plt.show()
            
        except ImportError:
            logger.error("Matplotlib not installed. Cannot plot indicators.")
            raise
        except Exception as e:
            logger.error(f"Error plotting indicators: {e}", exc_info=True)
            raise

    def register_custom_indicator(self, name: str, func: Callable, params: Optional[Dict[str, Any]] = None) -> bool:
        """Register a custom indicator function.
        
        Args:
            name: Name of the indicator
            func: Function that computes the indicator
            params: Optional parameter definitions
            
        Returns:
            bool: True if registration was successful
        """
        try:
            if not callable(func):
                raise ValueError(f"Function for indicator {name} must be callable")
            
            # Normalize name to lowercase
            name = name.lower()
            
            # Get function signature to identify required parameters
            sig = inspect.signature(func)
            required_params = {
                name: param.default == inspect.Parameter.empty
                for name, param in sig.parameters.items()
                if name != 'data'  # Skip the data parameter
            }
            
            # Create config dictionary
            config = {
                'name': name,
                'type': 'custom',
                'required_inputs': ['close'],  # Default to close price
                'params': params or {},
                'function': func
            }
            
            # Add required parameters if not already defined
            for param_name, is_required in required_params.items():
                if param_name not in config['params']:
                    config['params'][param_name] = {
                        'type': 'float',
                        'default': None,
                        'required': is_required
                    }
            
            # Register in global registry
            _custom_indicator_registry[name] = func
            self.indicator_params[name] = config
            
            logger.info(f"Successfully registered custom indicator: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering custom indicator {name}: {e}")
            return False

    def generate_parameter_configurations(self, indicator_name: str, method: str = 'grid') -> List[Dict[str, Any]]:
        """Generate parameter configurations for an indicator.
        
        Args:
            indicator_name: Name of the indicator
            method: Method to use for generating configurations ('grid' or 'random')
            
        Returns:
            List of parameter configurations
        """
        if indicator_name not in self.indicator_params:
            raise ValueError(f"Unknown indicator: {indicator_name}")
            
        indicator_config = self.indicator_params[indicator_name]
        params = indicator_config.get('params', {})
        
        if not params:
            return [{}]
            
        if method == 'grid':
            # Generate grid of parameter values
            param_values = {}
            for param_name, param_config in params.items():
                if isinstance(param_config, dict):
                    min_val = param_config.get('min', 1)
                    max_val = param_config.get('max', 100)
                    step = param_config.get('step', (max_val - min_val) / 5)
                    param_values[param_name] = np.arange(min_val, max_val + step, step)
                else:
                    param_values[param_name] = [param_config]
            
            # Generate all combinations
            param_names = list(param_values.keys())
            param_combinations = list(itertools.product(*[param_values[name] for name in param_names]))
            
            # Convert to list of dictionaries
            configs = []
            for combo in param_combinations:
                param_config = {}
                for name, value in zip(param_names, combo):
                    # Map parameter names based on indicator type
                    if indicator_config['type'] == 'talib':
                        if name == 'period':
                            param_config['timeperiod'] = int(value)
                        elif name == 'std_dev':
                            param_config['nbdevup'] = float(value)
                            param_config['nbdevdn'] = float(value)
                        else:
                            param_config[name] = value
                    else:
                        param_config[name] = value
                configs.append(param_config)
            
            return configs
            
        elif method == 'random':
            # Generate random parameter values
            num_configs = 10  # Default number of random configurations
            configs = []
            
            for _ in range(num_configs):
                param_config = {}
                for param_name, param_def in params.items():
                    if isinstance(param_def, dict):
                        min_val = param_def.get('min', 1)
                        max_val = param_def.get('max', 100)
                        if param_def.get('type') == 'int':
                            value = np.random.randint(min_val, max_val + 1)
                        else:
                            value = np.random.uniform(min_val, max_val)
                        
                        # Map parameter names based on indicator type
                        if indicator_config['type'] == 'talib':
                            if param_name == 'period':
                                param_config['timeperiod'] = int(value)
                            elif param_name == 'std_dev':
                                param_config['nbdevup'] = float(value)
                                param_config['nbdevdn'] = float(value)
                            else:
                                param_config[param_name] = value
                        else:
                            param_config[param_name] = value
                
                configs.append(param_config)
            
            return configs
            
        else:
            raise ValueError(f"Unsupported method: {method}")

def compute_configured_indicators(data: pd.DataFrame, indicator_configs: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, Set[int]]:
    """Module-level function to compute configured indicators.
    
    Args:
        data: Input DataFrame with price data
        indicator_configs: List of indicator configurations
        
    Returns:
        Tuple of (DataFrame with indicators, Set of failed config IDs)
    """
    try:
        factory = IndicatorFactory()
        return factory.compute_configured_indicators(data, indicator_configs)
    except Exception as e:
        logger.error(f"Error in compute_configured_indicators: {e}")
        # Return original data and all config IDs as failed
        failed_config_ids = {config.get('config_id', 0) for config in indicator_configs if isinstance(config.get('config_id'), int)}
        return data.copy(), failed_config_ids

# Export the module-level function
__all__ = ['IndicatorFactory', 'compute_configured_indicators']