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
    compute_vwap, compute_pvi, compute_nvi
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
        """Get output names for TA-Lib function."""
        try:
            func_info = ta.get_function_groups().get(func_name)
            if not func_info:
                raise ValueError(f"Unknown TA-Lib function: {func_name}")
            return func_info.get('output_names', [func_name])
        except Exception as e:
            logger.error(f"Error getting TA-Lib output names for {func_name}: {e}", exc_info=True)
            raise

    def _compute_single_indicator(self, data: pd.DataFrame, name: str, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Compute a single indicator based on its configuration."""
        try:
            if config['type'] == 'talib':
                func_name = config['name']
                params = config['params']
                
                # Get required input columns
                required_cols = []
                if 'high' in params: required_cols.append('high')
                if 'low' in params: required_cols.append('low')
                if 'close' in params: required_cols.append('close')
                if 'volume' in params: required_cols.append('volume')
                if 'open' in params: required_cols.append('open')
                
                # Validate required columns
                missing_cols = [col for col in required_cols if col not in data.columns]
                if missing_cols:
                    logger.error(f"Missing required columns for {name}: {missing_cols}")
                    return None

                # Get TA-Lib function
                ta_func = getattr(ta, func_name)
                if not ta_func:
                    raise ValueError(f"Unknown TA-Lib function: {func_name}")

                # Prepare input arrays
                inputs = {}
                for param_name, col_name in params.items():
                    if col_name in data.columns:
                        inputs[param_name] = data[col_name].values
                    else:
                        raise ValueError(f"Column {col_name} not found in data for {name}")

                # Get output names
                output_names = self._get_ta_lib_output_suffixes(func_name)
                
                # Compute indicator
                results = ta_func(**inputs)
                
                # Handle single output
                if not isinstance(results, tuple):
                    results = (results,)
                
                # Create result DataFrame
                result_df = pd.DataFrame(index=data.index)
                for i, output_name in enumerate(output_names):
                    if i < len(results):
                        result_df[f"{name}_{output_name}"] = results[i]
                
                return result_df

            elif config['type'] == 'ta-lib':
                # ... existing code ...
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
                    'NVI': compute_nvi
                }
                
                func = custom_funcs.get(name)
                if not func:
                    raise ValueError(f"Unknown custom indicator: {name}")
                
                # Compute custom indicator
                return func(data, **config['params'])

            else:
                raise ValueError(f"Unsupported indicator type: {config['type']}")

        except Exception as e:
            logger.error(f"Error computing indicator {name}: {e}", exc_info=True)
            return None

    def compute_indicators(self, data: pd.DataFrame, indicators: Optional[List[str]] = None) -> pd.DataFrame:
        """Compute all or specified indicators for the given data."""
        if indicators is None:
            indicators = list(self.indicator_params.keys())
        
        result_df = data.copy()
        for name in indicators:
            if name not in self.indicator_params:
                logger.warning(f"Unknown indicator: {name}")
                continue
                
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