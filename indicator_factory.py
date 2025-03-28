# indicator_factory.py
import logging
import pandas as pd
import numpy as np
import talib as ta
try:
    import pandas_ta as pta
except ImportError:
    pta = None
from typing import List, Dict, Any, Optional
import json
import config
import utils
import custom_indicators # Import custom indicators module
import re
import inspect

logger = logging.getLogger(__name__)

_INDICATOR_DEFS: Optional[Dict[str, Dict]] = None

# --- _load_indicator_definitions remains the same ---
def _load_indicator_definitions():
    """Loads indicator definitions from JSON if not already loaded."""
    global _INDICATOR_DEFS
    if _INDICATOR_DEFS is None:
        try:
            with open(config.INDICATOR_PARAMS_PATH, 'r', encoding='utf-8') as f:
                _INDICATOR_DEFS = json.load(f).get("indicators", {})
            logger.info(f"Loaded {len(_INDICATOR_DEFS)} indicator definitions from {config.INDICATOR_PARAMS_PATH}")
        except FileNotFoundError: logger.error(f"Indicator definitions file not found: {config.INDICATOR_PARAMS_PATH}"); _INDICATOR_DEFS = {}
        except json.JSONDecodeError as json_err: logger.error(f"Error decoding JSON from {config.INDICATOR_PARAMS_PATH}: {json_err}"); _INDICATOR_DEFS = {}
        except Exception as e: logger.error(f"Failed to load indicator definitions: {e}", exc_info=True); _INDICATOR_DEFS = {}

# --- _get_indicator_definition remains the same ---
def _get_indicator_definition(indicator_name: str) -> Optional[Dict]:
    """Retrieves the definition for a single indicator."""
    if _INDICATOR_DEFS is None: _load_indicator_definitions()
    return _INDICATOR_DEFS.get(indicator_name)

# --- UPDATED _compute_single_indicator (with Param Logging and Tuple Fix) ---
def _compute_single_indicator(data: pd.DataFrame, config_details: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Computes a single indicator based on its configuration.
    Returns a DataFrame containing ONLY the new indicator column(s), or None on failure.
    Does NOT modify the input DataFrame.
    """
    indicator_name = config_details['indicator_name']
    # ***** Make a defensive copy of params from config_details *****
    params = config_details.get('params', {}).copy() # Use .get() for safety and copy()
    config_id = config_details['config_id']
    output_df = pd.DataFrame(index=data.index)

    indicator_def = _get_indicator_definition(indicator_name)
    if not indicator_def:
        logger.error(f"Computation skipped: Definition not found for '{indicator_name}'.")
        return None

    indicator_type = indicator_def.get('type')
    required_inputs = indicator_def.get('required_inputs', [])
    missing_cols = [col for col in required_inputs if col not in data.columns]
    if missing_cols:
        logger.error(f"Missing required columns {missing_cols} for '{indicator_name}'. Skipping config ID {config_id}.")
        return None

    computed_any = False
    try:
        # --- TA-Lib ---
        if indicator_type == 'ta-lib':
            ta_func = getattr(ta, indicator_name.upper(), None)
            if not ta_func: logger.error(f"TA-Lib func '{indicator_name.upper()}' not found. Skip Cfg {config_id}."); return None
            func_args = [];
            for col in required_inputs:
                 if not pd.api.types.is_numeric_dtype(data[col]): logger.error(f"Input '{col}' not numeric. Skip Cfg {config_id}."); return None
                 func_args.append(data[col].astype(float)) # Pass Series

            # Prepare final parameters to be passed
            final_params = params.copy() # Use the params passed in config_details
            # ***** LOG PARAMETERS BEING USED *****
            logger.debug(f"Calling TA-Lib {indicator_name.upper()} with {len(func_args)} args and final params: {final_params}")

            try: result = ta_func(*func_args, **final_params)
            except TypeError as te:
                 # Fallback only if initial call fails due to wrong args
                 logger.warning(f"TypeError TA-Lib {indicator_name.upper()}: {te}. Retrying filtered.")
                 try:
                     known_talib_params = ['timeperiod', 'fastperiod', 'slowperiod', 'signalperiod', 'fastk_period', 'slowk_period', 'slowd_period', 'fastd_period', 'timeperiod1', 'timeperiod2', 'timeperiod3', 'nbdevup', 'nbdevdn', 'nbdev', 'matype', 'fastlimit', 'slowlimit', 'acceleration', 'maximum', 'vfactor']
                     filtered_params = {k: v for k, v in params.items() if k in known_talib_params}
                     # ***** LOG FILTERED PARAMETERS *****
                     logger.debug(f"Retrying TA-Lib {indicator_name.upper()} with filtered params: {filtered_params}")
                     result = ta_func(*func_args, **filtered_params)
                 except Exception as retry_e: logger.error(f"Error retrying TA-Lib {indicator_name.upper()}: {retry_e}", exc_info=True); return None

            # Process result
            if isinstance(result, tuple):
                for idx, res_output in enumerate(result):
                    # >>>>>>>>>>>>>>>> FIX APPLIED HERE <<<<<<<<<<<<<<<<
                    # Check for both numpy array and pandas series
                    if isinstance(res_output, (np.ndarray, pd.Series)) and len(res_output) == len(data.index):
                        col_name = utils.get_config_identifier(indicator_name, config_id, idx)
                        output_df[col_name] = res_output # Assign array or series
                        computed_any = True
                    elif res_output is not None:
                         logger.warning(f"TA-Lib {indicator_name} tuple output element {idx} type/len mismatch. Type: {type(res_output)}")
                    else:
                         logger.debug(f"TA-Lib {indicator_name} tuple output element {idx} is None.")
                    # >>>>>>>>>>>>>>>> END FIX <<<<<<<<<<<<<<<<<<<<<<<<<<<
            elif isinstance(result, (np.ndarray, pd.Series)) and len(result) == len(data.index):
                 col_name = utils.get_config_identifier(indicator_name, config_id, None)
                 output_df[col_name] = result # Assign array or series
                 computed_any = True
            elif result is not None: logger.warning(f"TA-Lib {indicator_name} output type/len mismatch.")
            else: logger.debug(f"TA-Lib {indicator_name} returned None.")

        # --- Pandas-TA ---
        elif indicator_type == 'pandas-ta':
            if pta is None: logger.error(f"pandas_ta not installed. Skip Cfg {config_id}."); return None
            pta_func = None; is_accessor = False
            if hasattr(pta, indicator_name.lower()): pta_func = getattr(pta, indicator_name.lower())
            elif hasattr(data.ta, indicator_name.lower()): is_accessor = True
            else: logger.error(f"Pandas TA '{indicator_name.lower()}' not found. Skip Cfg {config_id}."); return None
            final_params = params.copy(); input_data = {col: data[col] for col in required_inputs}
            result = None
            try:
                # ***** LOG PARAMETERS BEING USED *****
                logger.debug(f"Calling Pandas TA '{indicator_name.lower()}' (Accessor: {is_accessor}) with final params: {final_params}")
                if pta_func: result = pta_func(**input_data, **final_params)
                elif is_accessor: result = getattr(data.ta, indicator_name.lower())(**final_params)
                else: raise AttributeError("Function not found")
            except Exception as pta_e: logger.error(f"Error calling Pandas TA '{indicator_name.lower()}' (Cfg {config_id}): {pta_e}", exc_info=True); return None
            # Process pta result
            if isinstance(result, pd.DataFrame):
                 logger.debug(f"Processing pta DF output. Cols: {list(result.columns)}")
                 for idx, col in enumerate(result.columns):
                      if col in result and result[col] is not None and pd.api.types.is_numeric_dtype(result[col]):
                           # Improved naming for pta columns
                           col_suffix = col.replace(f"{indicator_name.lower()}_", "", 1).upper() # Get suffix like '13_25_13' or 'UPPER'
                           if col_suffix != col.upper(): # Check if replacement happened
                                col_name_final = f"{indicator_name}_{config_id}_{col_suffix}"
                           else: # Fallback if no clear suffix
                                col_name_final = utils.get_config_identifier(indicator_name, config_id, idx)
                           if col_name_final in output_df.columns: col_name_final += f"_dup{idx}"
                           output_df[col_name_final] = result[col].reindex(data.index); computed_any = True
            elif isinstance(result, pd.Series):
                 if result is not None and pd.api.types.is_numeric_dtype(result):
                      col_name = utils.get_config_identifier(indicator_name, config_id, None)
                      output_df[col_name] = result.reindex(data.index); computed_any = True
            elif result is not None: logger.warning(f"Pandas TA '{indicator_name}' ret type {type(result)}. Skip.")

        # --- Custom ---
        elif indicator_type == 'custom':
            custom_func_name = indicator_def.get('function_name', f"compute_{indicator_name.lower()}")
            custom_func = getattr(custom_indicators, custom_func_name, None)
            if custom_func:
                # ***** LOG PARAMETERS BEING USED *****
                sig = inspect.signature(custom_func); valid_params = {k:v for k,v in params.items() if k in sig.parameters}
                logger.debug(f"Calling custom function {custom_func_name} with final params: {valid_params}")
                try:
                    temp_data = data.copy(); cols_before = set(temp_data.columns)
                    custom_func(temp_data, **valid_params) # Assumes modifies inplace
                    cols_after = set(temp_data.columns); new_cols = list(cols_after - cols_before)
                    if new_cols:
                        for new_col in new_cols:
                            final_col_name = utils.get_config_identifier(new_col, config_id, None)
                            if final_col_name in output_df.columns: final_col_name += f"_dup"
                            output_df[final_col_name] = temp_data[new_col]; computed_any = True
                        logger.debug(f"Custom func {custom_func_name} added cols: {new_cols} -> {list(output_df.columns)}")
                    else: logger.warning(f"Custom func {custom_func_name} added no new columns.")
                except Exception as custom_e: logger.error(f"Error executing custom func {custom_func_name} (Cfg {config_id}): {custom_e}", exc_info=True)
            else: logger.error(f"Custom func '{custom_func_name}' not found. Skip {indicator_name} (Cfg {config_id}).")
        else: logger.error(f"Unknown type '{indicator_type}' for '{indicator_name}'. Skip Cfg {config_id}."); return None

    except Exception as e:
        logger.error(f"General error computing indicator '{indicator_name}' (Config ID: {config_id}): {e}", exc_info=True)
        return None

    if computed_any and not output_df.empty:
        output_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in output_df.columns:
            if pd.api.types.is_numeric_dtype(output_df[col]): output_df[col] = output_df[col].astype(float)
        logger.debug(f"Successfully computed columns for {indicator_name} (Config ID: {config_id}) -> Columns: {list(output_df.columns)}") # Log computed columns
        return output_df
    else:
        logger.warning(f"No output generated or output empty for indicator {indicator_name} (Config ID: {config_id})")
        return None

# --- compute_configured_indicators (NO dropna) remains the same ---
def compute_configured_indicators(data: pd.DataFrame, configs_to_process: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Computes all indicators specified in configs_to_process.
    Returns a NEW DataFrame containing original data + computed indicator columns.
    Does NOT call dropna().
    """
    _load_indicator_definitions()
    if not _INDICATOR_DEFS: logger.error("Indicator definitions failed load."); return data
    if not configs_to_process: logger.warning("No configs provided for computation."); return data
    all_indicator_outputs = []
    logger.info(f"Processing {len(configs_to_process)} indicator configurations...")
    for i, config_details in enumerate(configs_to_process):
        logger.debug(f"Computing indicator {i+1}/{len(configs_to_process)}: {config_details['indicator_name']} (Config ID: {config_details['config_id']})")
        indicator_df = _compute_single_indicator(data, config_details)
        if indicator_df is not None and not indicator_df.empty:
            nan_check = indicator_df.isnull().all()
            all_nan_cols = nan_check[nan_check].index.tolist()
            if all_nan_cols:
                 logger.warning(f"Indicator {config_details['indicator_name']} (Cfg {config_details['config_id']}) produced all-NaN columns: {all_nan_cols}. Excluding them.")
                 indicator_df = indicator_df.drop(columns=all_nan_cols)
                 if indicator_df.empty: continue
            all_indicator_outputs.append(indicator_df)

    if all_indicator_outputs:
        logger.info(f"Concatenating results from {len(all_indicator_outputs)} successful indicator computations...")
        try:
            aligned_outputs = [df.reindex(data.index) for df in all_indicator_outputs]
            indicators_combined_df = pd.concat(aligned_outputs, axis=1)
            new_cols = indicators_combined_df.columns; existing_cols = data.columns
            overlap_cols = new_cols.intersection(existing_cols)
            if not overlap_cols.empty: logger.error(f"FATAL: Indicator columns overlap existing data: {list(overlap_cols)}"); raise ValueError("Column name conflict")
            dup_new_cols = new_cols[new_cols.duplicated()].unique()
            if not dup_new_cols.empty: logger.warning(f"Duplicate indicator output column names detected: {list(dup_new_cols)}.")
            data_with_indicators = pd.concat([data, indicators_combined_df], axis=1)
            computed_col_count = indicators_combined_df.shape[1]
            logger.info(f"Successfully added {computed_col_count} indicator columns.")
            logger.info(f"Indicator computation finished. DataFrame shape (with NaNs): {data_with_indicators.shape}")
            return data_with_indicators
        except Exception as concat_err: logger.error(f"Error during concatenation: {concat_err}", exc_info=True); return data
    else: logger.warning("No indicators were successfully computed or added."); return data