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
    compute_vwap, compute_pvi, compute_nvi, compute_returns, compute_volatility
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

    def _load_params(self) -> Dict[str, Any]:
        """Load indicator parameters from JSON file."""
        try:
            with open(self.params_file, 'r') as f:
                params = json.load(f)
            logger.info(f"Loaded {len(params)} indicator configurations from {self.params_file}")
            return params
        except Exception as e:
            logger.error(f"Error loading indicator parameters: {e}", exc_info=True)
            raise

    def _validate_params(self) -> None:
        """Validate indicator parameters."""
        required_keys = ['name', 'type', 'params']
        for name, config in self.indicator_params.items():
            missing = [k for k in required_keys if k not in config]
            if missing:
                raise ValueError(f"Missing required keys {missing} for indicator {name}")
            if config['type'] not in ['talib', 'custom', 'ta-lib', 'pandas-ta']:
                raise ValueError(f"Invalid indicator type {config['type']} for {name}")

    def _get_ta_lib_output_suffixes(self, func_name: str) -> List[str]:
        """Get output names for TA-Lib function. Default to function name."""
        return [func_name]

    def _compute_single_indicator(self, data: pd.DataFrame, name: str, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Compute a single indicator based on its configuration."""
        try:
            result_df = pd.DataFrame(index=data.index)
            
            if config['type'] in ['talib', 'ta-lib']:  # Handle both types the same way
                func_name = config['name']
                params = config['params']
                required_cols = []
                # Map parameter names to required columns
                if 'high' in params: required_cols.append('high')
                if 'low' in params: required_cols.append('low')
                if 'close' in params: required_cols.append('close')
                if 'volume' in params: required_cols.append('volume')
                if 'open' in params: required_cols.append('open')
                # Validate required columns
                missing_cols = [col for col in required_cols if col not in data.columns]
                if missing_cols:
                    logger.error(f"Missing required columns for {name}: {missing_cols}")
                    return result_df
                # Get TA-Lib function
                ta_func = getattr(ta, func_name.upper(), None)
                if not ta_func:
                    raise ValueError(f"Unknown TA-Lib function: {func_name}")
                # Prepare input arrays
                inputs = {}
                for param_name, col_value in params.items():
                    # If the value is a dict (e.g., {'default': 14}), use the default
                    if isinstance(col_value, dict):
                        value = col_value.get('default', None)
                    else:
                        value = col_value
                    # If value is a string and a column, use the column data
                    if isinstance(value, str) and value in data.columns:
                        inputs[param_name] = data[value].values
                    else:
                        # Use as-is (for timeperiod, float, int, etc.)
                        inputs[param_name] = value
                logger.debug(f"Calling TA-Lib function {func_name.upper()} with inputs: {inputs}")
                # Get output names
                output_names = self._get_ta_lib_output_suffixes(func_name.upper())
                # Compute indicator
                results = ta_func(**inputs)
                logger.debug(f"TA-Lib function {func_name.upper()} returned: {results}")
                if results is None:
                    logger.error(f"TA-Lib function {func_name.upper()} returned None for {name}")
                    return result_df
                # Handle different return types
                if isinstance(results, (np.ndarray, list)):
                    # Single array result
                    if len(output_names) > 0:
                        result_df[f"{name}_{output_names[0]}"] = results
                    else:
                        result_df[name] = results
                elif isinstance(results, tuple):
                    # Multiple array results
                    for i, output_name in enumerate(output_names):
                        if i < len(results):
                            result_df[f"{name}_{output_name}"] = results[i]
                else:
                    # Single value result
                    result_df[name] = results
                if result_df.empty:
                    logger.error(f"Result DataFrame is empty for {name}")
                return result_df

            elif config['type'] == 'pandas-ta':
                func_name = config['name']
                params = config['params']
                required_cols = config.get('required_inputs', [])
                missing_cols = [col for col in required_cols if col not in data.columns]
                if missing_cols:
                    logger.error(f"Missing required columns for {name}: {missing_cols}")
                    return None
                # Get pandas_ta function
                ta_func = getattr(pta, func_name, None)
                if not ta_func:
                    raise ValueError(f"Unknown pandas-ta function: {func_name}")
                # Prepare input arguments
                func_args = {k: v['default'] if isinstance(v, dict) and 'default' in v else v for k, v in params.items()}
                # Call the function
                result = ta_func(data, **func_args) if 'self' not in ta_func.__code__.co_varnames else ta_func(**{k: data[k] for k in required_cols}, **func_args)
                # result can be a Series or DataFrame
                if isinstance(result, pd.Series):
                    result_df = pd.DataFrame({f"{name}": result})
                else:
                    result_df = result.add_prefix(f"{name}_")
                return result_df

            elif config['type'] == 'custom':
                # Map indicator name to function
                custom_funcs = {
                    'OBV_PRICE_DIVERGENCE': compute_obv_price_divergence,
                    'VOLUME_OSCILLATOR': compute_volume_oscillator,
                    'VWAP': compute_vwap,
                    'PVI': compute_pvi,
                    'NVI': compute_nvi,
                    'RETURNS': compute_returns,
                    'VOLATILITY': compute_volatility
                }
                
                func = custom_funcs.get(name.upper())
                if not func:
                    raise ValueError(f"Unknown custom indicator: {name}")
                
                # Compute custom indicator
                return func(data, **config['params'])

            else:
                raise ValueError(f"Unsupported indicator type: {config['type']}")

        except Exception as e:
            logger.error(f"Error computing indicator {name}: {e}", exc_info=True)
            return pd.DataFrame(index=data.index)

    def compute_indicators(self, data: pd.DataFrame, indicators: Optional[List[str]] = None) -> pd.DataFrame:
        """Compute all or specified indicators for the given data."""
        # Validate input data
        if data is None or not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")
        if data.empty:
            raise ValueError("Input data cannot be empty")
        
        if indicators is None:
            indicators = list(self.indicator_params.keys())
        
        result_df = data.copy()
        for name in indicators:
            if name not in self.indicator_params:
                logger.warning(f"Unknown indicator: {name}")
                raise ValueError(f"Unknown indicator: {name}")
                
            config = self.indicator_params[name]
            indicator_df = self._compute_single_indicator(data, name, config)
            
            if indicator_df is not None:
                # Add new columns to result
                for col in indicator_df.columns:
                    if col in result_df.columns:
                        logger.warning(f"Column {col} already exists, overwriting")
                    result_df[col] = indicator_df[col]
            else:
                logger.warning(f"Failed to compute indicator: {name}")
        
        return result_df

    def get_available_indicators(self) -> List[str]:
        """Get list of available indicators."""
        return list(self.indicator_params.keys())

    def get_indicator_params(self, name: str) -> Optional[Dict[str, Any]]:
        """Get parameters for a specific indicator."""
        indicator_def = self.indicator_params.get(name)
        return indicator_def.get('params') if indicator_def else None

    def create_custom_indicator(self, name: str, func, data: pd.DataFrame, **params):
        """Register and compute a custom indicator on the fly."""
        result = func(data, **params)
        if isinstance(result, pd.Series):
            return result
        elif isinstance(result, pd.DataFrame) and result.shape[1] == 1:
            return result.iloc[:, 0]
        else:
            return pd.Series(result, name=name)

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
        
        # Check for minimum data length if period/length/timeperiod is specified (in merged_params)
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
                    raise ValueError(f"Input data length ({len(data)}) is less than required {period_key} ({period_val}) for indicator '{name}'")
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
            # Custom indicators have their own validation
            pass
            
        if required_cols:
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns for indicator '{name}': {missing_cols}")
        
        result = self._compute_single_indicator(data, name, config_for_compute)
        # Return as Series if only one column, else DataFrame
        if result is not None and isinstance(result, pd.DataFrame) and result.shape[1] == 1:
            return result.iloc[:, 0]
        return result

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
                
            # Add computed columns to result
            for col in indicator_df.columns:
                if col in result_df.columns:
                    logger.warning(f"Column {col} already exists, overwriting")
                result_df[col] = indicator_df[col]
                
        except Exception as e:
            logger.error(f"Error computing indicator config {config.get('config_id', 'unknown')}: {e}", exc_info=True)
            # Only add config_id if it exists and is an int
            config_id = config.get('config_id')
            if isinstance(config_id, int):
                failed_config_ids.add(config_id)
                
    return result_df, failed_config_ids