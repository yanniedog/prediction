# indicator_factory.py
import logging
import pandas as pd
import numpy as np
import talib as ta
try:
    import pandas_ta as pta
    PTA_AVAILABLE = True
except ImportError:
    pta = None
    PTA_AVAILABLE = False
    logging.getLogger(__name__).warning("pandas_ta library not found. Pandas TA indicators will not be available.")

from typing import List, Dict, Any, Optional, Tuple, Set # Added Set
import json
import re
import inspect

import config
import utils
import custom_indicators

logger = logging.getLogger(__name__)

_INDICATOR_DEFS: Optional[Dict[str, Dict]] = None
_TA_LIB_FUNCTION_GROUPS_CACHE: Optional[Dict] = None # Cache for TA-Lib function groups

# --- Constants for Ichimoku Filtering --- (Keep as before)
ICHIMOKU_COLUMNS_TO_KEEP = ['ISA_9', 'ISB_26', 'ITS_9', 'IKS_26', 'ICS_26'] # Default pandas-ta names
ICHIMOKU_OUTPUT_MAP = { # Map pandas-ta default names to clearer suffixes if desired
    'ISA_9': 'SPAN_A', 'ISB_26': 'SPAN_B', 'ITS_9': 'TENKAN', 'IKS_26': 'KIJUN', 'ICS_26': 'CHIKOU'
}
ICHIMOKU_COLUMNS_TO_EXCLUDE = ['TENKAN', 'KIJUN', 'CHIKOU'] # Exclude these from correlation

def _load_indicator_definitions() -> None:
    """Loads indicator definitions from JSON if not already loaded."""
    global _INDICATOR_DEFS
    if _INDICATOR_DEFS is None:
        try:
            with open(config.INDICATOR_PARAMS_PATH, 'r', encoding='utf-8') as f:
                full_json = json.load(f)
                _INDICATOR_DEFS = full_json.get("indicators", {})
            if not _INDICATOR_DEFS:
                 logger.warning(f"No 'indicators' key found or empty in {config.INDICATOR_PARAMS_PATH}")
            else:
                 logger.info(f"Loaded {len(_INDICATOR_DEFS)} indicator definitions from {config.INDICATOR_PARAMS_PATH}")
        except FileNotFoundError:
            logger.error(f"Indicator definitions file not found: {config.INDICATOR_PARAMS_PATH}")
            _INDICATOR_DEFS = {}
        except json.JSONDecodeError as json_err:
            logger.error(f"Error decoding JSON from {config.INDICATOR_PARAMS_PATH}: {json_err}")
            _INDICATOR_DEFS = {}
        except Exception as e:
            logger.error(f"Failed to load indicator definitions: {e}", exc_info=True)
            _INDICATOR_DEFS = {}

# --- Function to safely get TA-Lib output names ---
def _get_ta_lib_output_suffixes(ta_func_name: str) -> List[str]:
    """Safely retrieves the default output names for a TA-Lib function."""
    global _TA_LIB_FUNCTION_GROUPS_CACHE
    output_suffixes = []
    try:
        if _TA_LIB_FUNCTION_GROUPS_CACHE is None:
            _TA_LIB_FUNCTION_GROUPS_CACHE = ta.get_function_groups()

        if not isinstance(_TA_LIB_FUNCTION_GROUPS_CACHE, dict):
             logger.warning("ta.get_function_groups() did not return a dictionary. Cannot get output names.")
             return []

        # Iterate through the groups to find the one containing the function
        func_info = None
        for group_name, functions in _TA_LIB_FUNCTION_GROUPS_CACHE.items():
            if isinstance(functions, dict) and ta_func_name in functions:
                func_info = functions[ta_func_name]
                break # Found the function, no need to check other groups

        if func_info and isinstance(func_info, dict):
            output_suffixes = func_info.get('Output Names', [])
            if not isinstance(output_suffixes, list):
                 logger.warning(f"Expected list for 'Output Names' for {ta_func_name}, got {type(output_suffixes)}. Using default.")
                 output_suffixes = []
        # else: Function info not found or not a dict, output_suffixes remains []

    except Exception as e:
        logger.error(f"Error retrieving TA-Lib output suffixes for '{ta_func_name}': {e}", exc_info=False)
        output_suffixes = [] # Default to empty list on any error

    return output_suffixes
# --- End function ---


def _get_indicator_definition(indicator_name: str) -> Optional[Dict]:
    """Retrieves the definition for a single indicator by name."""
    if _INDICATOR_DEFS is None: _load_indicator_definitions()
    return _INDICATOR_DEFS.get(indicator_name)

def _compute_single_indicator(data: pd.DataFrame, config_details: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Computes a single indicator based on its configuration.
    Returns a DataFrame containing ONLY the new indicator column(s), or None on failure.
    """
    indicator_name = config_details['indicator_name']
    params = config_details.get('params', {}).copy()
    config_id = config_details.get('config_id')
    # Ensure config_id exists and is valid before proceeding
    if config_id is None or not isinstance(config_id, int):
         logger.error(f"Invalid or missing config_id in config_details for '{indicator_name}'. Skipping computation.")
         return None
    output_df = pd.DataFrame(index=data.index) # Initialize empty DF with correct index

    indicator_def = _get_indicator_definition(indicator_name)
    if not indicator_def:
        logger.error(f"Computation skipped: Definition not found for '{indicator_name}'.")
        return None # Return None on failure

    indicator_type = indicator_def.get('type')
    required_inputs = indicator_def.get('required_inputs', [])

    missing_cols = [col for col in required_inputs if col not in data.columns]
    if missing_cols:
        logger.error(f"Missing required columns {missing_cols} for '{indicator_name}'. Skipping Cfg {config_id}.")
        return None # Return None on failure

    computed_any = False
    try:
        # --- TA-Lib Indicators ---
        if indicator_type == 'ta-lib':
            ta_func_name = indicator_name.upper()
            ta_func = getattr(ta, ta_func_name, None)
            if not ta_func:
                logger.error(f"TA-Lib function '{ta_func_name}' not found. Skipping Cfg {config_id}.")
                return None

            final_params = params.copy()
            invalid_param_msg = None

            # MAType Handling (unchanged)
            matype_accepting_funcs = ['BBANDS', 'MA', 'APO', 'PPO', 'STOCH', 'STOCHF', 'STOCHRSI',
                                     'DEMA', 'EMA', 'KAMA', 'MIDPOINT', 'MIDPRICE', 'SMA', 'T3',
                                     'TEMA', 'TRIMA', 'WMA']
            if ta_func_name in matype_accepting_funcs:
                 if 'matype' in final_params:
                     matype_val = final_params['matype']
                     if not isinstance(matype_val, int) or not (0 <= matype_val <= 8):
                         logger.error(f"TA-Lib '{ta_func_name}' invalid matype: {matype_val}. Skip Cfg {config_id}.")
                         return None
            elif 'matype' in final_params:
                 logger.debug(f"Func {ta_func_name} ignores 'matype'. Removing for Cfg {config_id}.")
                 del final_params['matype']

            # Specific Parameter Validation (unchanged)
            # ... (TRIX, APO, PPO, MACD, ADOSC, ULTOSC, MAMA, SAR checks) ...
            min_period_funcs = ['SMA', 'EMA', 'DEMA', 'TEMA', 'WMA', 'TRIMA', 'KAMA', 'MIDPOINT', 'MIDPRICE', 'ADX', 'ADXR', 'AROON', 'AROONOSC', 'CCI', 'CMO', 'DX', 'MINUS_DI', 'MINUS_DM', 'PLUS_DI', 'PLUS_DM', 'RSI', 'WILLR', 'T3']
            if ta_func_name in min_period_funcs and final_params.get('timeperiod', -1) < 2:
                 invalid_param_msg = f"timeperiod({final_params.get('timeperiod')}) < 2"

            if invalid_param_msg:
                logger.error(f"TA-Lib '{ta_func_name}' invalid params ({invalid_param_msg}). Skip Cfg {config_id}.")
                return None

            # Prepare input series (ensure float64)
            func_args: List[pd.Series] = []
            for col in required_inputs:
                 if col not in data.columns: logger.error(f"Required col '{col}' not in data. Skip Cfg {config_id}."); return None
                 input_series = data[col]
                 if not pd.api.types.is_numeric_dtype(input_series):
                     logger.error(f"Input col '{col}' not numeric for TA-Lib '{ta_func_name}'. Skip Cfg {config_id}.")
                     return None
                 func_args.append(input_series.astype(np.float64))

            logger.debug(f"Calling TA-Lib '{ta_func_name}' (Cfg ID: {config_id}) with {len(func_args)} inputs, params: {final_params}")
            try:
                result = ta_func(*func_args, **final_params)
            except Exception as e: # Catch any exception during TA-Lib call
                 err_msg = str(e)
                 if "Bad Parameter" in err_msg or "TA_BAD_PARAM" in err_msg or "error code=2" in err_msg :
                      logger.error(f"TA-Lib '{ta_func_name}' (Cfg {config_id}) Bad Param error: {err_msg}. Params: {final_params}", exc_info=False)
                 else: logger.error(f"TA-Lib Exception calling '{ta_func_name}' (Cfg {config_id}, Params: {final_params}): {e}", exc_info=True)
                 return None # Return None on calculation error

            # Process TA-Lib result (use helper for output names)
            if isinstance(result, tuple):
                # --- Use helper function ---
                output_suffixes = _get_ta_lib_output_suffixes(ta_func_name)
                # --- End Use helper function ---
                for idx, res_output in enumerate(result):
                     if isinstance(res_output, (np.ndarray, pd.Series)) and len(res_output) == len(data.index):
                          # Use default numeric index if suffixes are missing or insufficient
                          suffix = output_suffixes[idx] if output_suffixes and idx < len(output_suffixes) else str(idx)
                          col_name = utils.get_config_identifier(indicator_name, config_id, suffix)
                          output_df[col_name] = res_output; computed_any = True
                     elif res_output is not None: logger.warning(f"TA-Lib '{ta_func_name}' output {idx} bad type/len. Skipping.")
            elif isinstance(result, (np.ndarray, pd.Series)) and len(result) == len(data.index):
                 col_name = utils.get_config_identifier(indicator_name, config_id, None)
                 output_df[col_name] = result; computed_any = True
            elif result is not None: logger.warning(f"TA-Lib '{ta_func_name}' bad output type/length. Skipping.")

        # --- Pandas-TA Indicators ---
        elif indicator_type == 'pandas-ta':
            if not PTA_AVAILABLE: logger.error(f"pandas_ta not installed. Skip '{indicator_name}' Cfg {config_id}."); return None
            pta_func_name = indicator_name.lower(); pta_func = None; is_accessor = False
            if hasattr(data.ta, pta_func_name): is_accessor = True
            elif hasattr(pta, pta_func_name): pta_func = getattr(pta, pta_func_name)
            else: logger.error(f"PTA func/accessor '{pta_func_name}' not found. Skip Cfg {config_id}."); return None

            final_params = params.copy(); result = None
            try:
                logger.debug(f"Calling PTA '{pta_func_name}' (Accessor: {is_accessor}, Cfg {config_id}) params: {final_params}")
                if is_accessor: # Call via accessor (e.g., data.ta.rsi())
                    func_ref = getattr(data.ta, pta_func_name); sig = inspect.signature(func_ref)
                    required_accessor_cols = set(sig.parameters.keys()) & {'open', 'high', 'low', 'close', 'volume'}
                    if not required_accessor_cols.issubset(data.columns):
                         missing = required_accessor_cols - set(data.columns)
                         logger.error(f"PTA accessor '{pta_func_name}' missing required cols {missing}. Skip Cfg {config_id}.")
                         return None
                    result = func_ref(**final_params)
                elif pta_func: # Call standalone function (e.g., pta.rsi(data['close']))
                    sig = inspect.signature(pta_func); expected_args = list(sig.parameters.keys())
                    call_args = {}; input_cols_provided = set()
                    for req_col in required_inputs: # Map required inputs to function args
                         if req_col not in data.columns: logger.error(f"PTA '{pta_func_name}' missing input '{req_col}'. Skip Cfg {config_id}."); return None
                         target_arg_name = req_col # Default
                         if req_col == 'close' and 'close' in expected_args: target_arg_name = 'close'
                         # ... add mappings for high, low, open, volume ...
                         if target_arg_name in expected_args: call_args[target_arg_name] = data[req_col].astype(float); input_cols_provided.add(target_arg_name)
                         elif req_col in expected_args: call_args[req_col] = data[req_col].astype(float); input_cols_provided.add(req_col)
                         else: logger.warning(f"PTA '{pta_func_name}': Cannot map input '{req_col}'. Skipping arg.")
                    result = pta_func(**call_args, **final_params)
                else: raise AttributeError(f"PTA state invalid for '{pta_func_name}'.")
            except Exception as pta_e: # Catch any exception during PTA call
                err_msg = str(pta_e)
                if "TA_BAD_PARAM" in err_msg or "error code 2" in err_msg: logger.error(f"PTA '{pta_func_name}' (Cfg {config_id}) Bad Param: {err_msg}. Params: {final_params}", exc_info=False)
                else: logger.error(f"PTA Error calling '{pta_func_name}' (Cfg {config_id}): {pta_e}", exc_info=True)
                return None # Return None on calculation error

            # Process PTA result (including Ichimoku filtering) (unchanged logic)
            if isinstance(result, tuple): # Handle tuple output
                 processed_outputs = []
                 for item in result:
                      if isinstance(item, pd.DataFrame): processed_outputs.append(item)
                      elif isinstance(item, pd.Series): processed_outputs.append(item.to_frame(name=item.name or f"{pta_func_name}_out"))
                 if not processed_outputs: result = None
                 else:
                      try: result = pd.concat(processed_outputs, axis=1)
                      except Exception as concat_err: logger.error(f"Error concat PTA tuple {pta_func_name} (Cfg {config_id}): {concat_err}"); result = None

            if isinstance(result, pd.DataFrame): # Handle DataFrame output
                 logger.debug(f"Processing pta DF output. Columns: {list(result.columns)}")
                 for col in result.columns:
                      if col not in result or result[col] is None: continue
                      series_to_add = result[col]
                      if pd.api.types.is_numeric_dtype(series_to_add):
                           is_ichimoku = indicator_name.lower() == 'ichimoku'; output_suffix = None
                           for pta_name, clean_suffix in ICHIMOKU_OUTPUT_MAP.items():
                                if col.startswith(pta_name): output_suffix = clean_suffix; break
                           if output_suffix is None: output_suffix = str(col)
                           if is_ichimoku and output_suffix in ICHIMOKU_COLUMNS_TO_EXCLUDE:
                                logger.debug(f"Excluding Ichimoku col '{col}' (Mapped: {output_suffix}).")
                                continue
                           col_name_final = utils.get_config_identifier(indicator_name, config_id, output_suffix)
                           # Ensure unique final column name
                           original_col_name = col_name_final; dup_counter = 1
                           while col_name_final in output_df.columns: col_name_final = f"{original_col_name}_dup{dup_counter}"; dup_counter += 1
                           output_df[col_name_final] = series_to_add.reindex(data.index); computed_any = True
                      # else: logger.debug(f"Skipping non-numeric pta col '{col}'.")
            elif isinstance(result, pd.Series): # Handle Series output
                 if result is not None and pd.api.types.is_numeric_dtype(result):
                      col_name = utils.get_config_identifier(indicator_name, config_id, None)
                      output_df[col_name] = result.reindex(data.index); computed_any = True
                 # else: logger.warning(f"PTA '{pta_func_name}' returned non-numeric/None Series.")
            elif result is not None: logger.warning(f"PTA '{pta_func_name}' returned unexpected type {type(result)}.")

        # --- Custom Indicators ---
        elif indicator_type == 'custom':
             # Custom indicator logic remains the same
            custom_func_name = indicator_def.get('function_name', f"compute_{indicator_name.lower()}")
            custom_func = getattr(custom_indicators, custom_func_name, None)
            if custom_func:
                sig = inspect.signature(custom_func); valid_params = {k:v for k,v in params.items() if k in sig.parameters}
                logger.debug(f"Calling custom func '{custom_func_name}' (Cfg {config_id}) params: {valid_params}")
                try:
                    result_df_custom = custom_func(data, **valid_params) # Pass data directly
                    if result_df_custom is None: logger.warning(f"Custom func '{custom_func_name}' (Cfg {config_id}) returned None."); return None
                    if not isinstance(result_df_custom, pd.DataFrame): logger.error(f"Custom func '{custom_func_name}' returned {type(result_df_custom)}, expected DF."); return None
                    if result_df_custom.empty: logger.info(f"Custom func '{custom_func_name}' returned empty DF.") # Not an error, just no output
                    else:
                        for new_col in result_df_custom.columns:
                             if new_col in result_df_custom and pd.api.types.is_numeric_dtype(result_df_custom[new_col]):
                                  suffix = new_col if len(result_df_custom.columns) > 1 else None
                                  final_col_name = utils.get_config_identifier(indicator_name, config_id, suffix)
                                  original_col_name = final_col_name; dup_counter = 1
                                  while final_col_name in output_df.columns: final_col_name = f"{original_col_name}_dup{dup_counter}"; dup_counter += 1
                                  output_df[final_col_name] = result_df_custom[new_col].reindex(data.index); computed_any = True
                             else: logger.warning(f"Custom func '{custom_func_name}' non-numeric col '{new_col}'. Skipping.")
                except Exception as custom_e: logger.error(f"Error exec custom func '{custom_func_name}' (Cfg {config_id}): {custom_e}", exc_info=True); return None
            else: logger.error(f"Custom func '{custom_func_name}' not found. Skip Cfg {config_id}."); return None

        # --- Unknown Indicator Type ---
        else:
            logger.error(f"Unknown indicator type '{indicator_type}'. Skip Cfg {config_id}."); return None

    except Exception as e:
        logger.error(f"General error computing indicator '{indicator_name}' (Cfg {config_id}): {e}", exc_info=True)
        return None # Return None on general failure

    # --- Final Processing ---
    if computed_any and not output_df.empty:
        output_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Convert numeric columns to float64
        for col in output_df.columns:
            if pd.api.types.is_numeric_dtype(output_df[col]):
                 output_df[col] = output_df[col].astype(np.float64)
            else: # Attempt conversion if not numeric, coerce errors
                 output_df[col] = pd.to_numeric(output_df[col], errors='coerce')
                 if output_df[col].isnull().all(): logger.warning(f"Col '{col}' became all NaN after coercion (Cfg {config_id}).")
                 else: output_df[col] = output_df[col].astype(np.float64)

        # Check if *all* computed columns are entirely NaN AFTER processing
        if output_df.isnull().all().all():
            logger.warning(f"All output columns are NaN for {indicator_name} (Cfg {config_id}). Treated as failure.")
            return None # Treat all-NaN output as a failure

        logger.debug(f"Success for {indicator_name} (Cfg {config_id}) -> Cols Added: {list(output_df.columns)}")
        return output_df
    else:
        # Only log if no columns were computed at all
        if not computed_any:
            logger.info(f"No output generated for '{indicator_name}' (Cfg {config_id}).")
        return None # Return None if no columns computed or initial DF was empty


def compute_configured_indicators(
        data: pd.DataFrame, configs_to_process: List[Dict[str, Any]]
) -> Tuple[pd.DataFrame, Set[int]]: # <-- Return DataFrame and Set of failed config IDs
    """
    Computes all indicators specified in the list of configurations.
    Returns a NEW DataFrame containing original data plus all successfully computed indicator columns,
    AND a set containing the config_ids that failed calculation.
    """
    _load_indicator_definitions()
    if not _INDICATOR_DEFS:
        logger.error("Indicator definitions missing. Cannot compute indicators.")
        return data.copy(), set(cfg.get('config_id') for cfg in configs_to_process if cfg.get('config_id') is not None) # Return all as failed
    if not configs_to_process:
        logger.warning("No indicator configurations provided.")
        return data.copy(), set()

    all_indicator_outputs: List[pd.DataFrame] = []
    failed_config_ids: Set[int] = set() # <-- Track failed IDs
    total_configs = len(configs_to_process)
    logger.info(f"Processing {total_configs} indicator configurations...")
    success_count = 0; fail_count = 0

    for i, config_details in enumerate(configs_to_process):
        indicator_name = config_details.get('indicator_name', 'Unknown')
        config_id = config_details.get('config_id', None) # Use None as default if missing

        # Simple progress print
        print(f"\rCalculating Indicators: {i+1}/{total_configs} ({indicator_name[:15]}...)", end="")

        # Skip if config_id is missing or invalid early
        if config_id is None or not isinstance(config_id, int):
             logger.error(f"Skipping config at index {i}: Invalid/missing config_id ({config_id}) for indicator '{indicator_name}'.")
             fail_count += 1
             # Add None or a placeholder if you need to track which *input* configs failed vs calculation fails?
             # For now, just don't add to failed_config_ids if the ID itself was bad.
             continue

        logger.debug(f"Computing indicator {i+1}/{total_configs}: {indicator_name} (Config ID: {config_id})")
        indicator_df = _compute_single_indicator(data, config_details) # Pass original data

        # Check failure conditions: None result, empty DataFrame, or all-NaN DataFrame
        if indicator_df is None or indicator_df.empty or indicator_df.isnull().all().all():
             logger.info(f"Failed/empty/all-NaN result for {indicator_name} (Config ID: {config_id}).")
             failed_config_ids.add(config_id) # <-- Add ID to failed set
             fail_count += 1
        else:
             all_indicator_outputs.append(indicator_df)
             success_count += 1

    print() # Newline after loop

    logger.info(f"Indicator computations done. Success: {success_count}, Failed/Skipped: {fail_count}.")

    if all_indicator_outputs:
        logger.info(f"Concatenating results from {len(all_indicator_outputs)} successful computations...")
        try:
            # Ensure alignment before concat
            aligned_outputs = [df.reindex(data.index) for df in all_indicator_outputs]
            indicators_combined_df = pd.concat(aligned_outputs, axis=1)

            # --- Column Name Conflict Checks --- (Remain the same)
            new_cols = indicators_combined_df.columns; existing_cols = data.columns
            overlap = new_cols.intersection(existing_cols)
            if not overlap.empty: logger.error(f"FATAL: Overlap new/existing cols: {list(overlap)}"); raise ValueError("Indicator column name conflict.")
            new_col_counts = new_cols.value_counts(); dup_new_cols = new_col_counts[new_col_counts > 1].index.tolist()
            if dup_new_cols: logger.error(f"FATAL: Duplicate indicator cols generated: {dup_new_cols}"); raise ValueError("Duplicate indicator column names.")
            # --- End Conflict Checks ---

            data_with_indicators = pd.concat([data.copy(), indicators_combined_df], axis=1)
            logger.info(f"Added {indicators_combined_df.shape[1]} indicator columns.")
            logger.info(f"Final DataFrame shape: {data_with_indicators.shape}")
            return data_with_indicators, failed_config_ids # <-- Return DF and failed IDs

        except Exception as concat_err:
            logger.error(f"Error concatenating indicator results: {concat_err}", exc_info=True)
            # Return original data, but mark ALL processed configs as failed in this case?
            all_processed_ids = {cfg.get('config_id') for cfg in configs_to_process if cfg.get('config_id') is not None}
            return data.copy(), failed_config_ids.union(all_processed_ids) # Add all potentially processed IDs to failed set
    else:
        logger.warning("No indicators successfully computed/added. Returning original data.")
        all_processed_ids = {cfg.get('config_id') for cfg in configs_to_process if cfg.get('config_id') is not None}
        return data.copy(), failed_config_ids.union(all_processed_ids) # Return all as failed if none succeeded# indicator_factory.py
import logging
import pandas as pd
import numpy as np
import talib as ta
try:
    import pandas_ta as pta
    PTA_AVAILABLE = True
except ImportError:
    pta = None
    PTA_AVAILABLE = False
    logging.getLogger(__name__).warning("pandas_ta library not found. Pandas TA indicators will not be available.")

from typing import List, Dict, Any, Optional, Tuple, Set # Added Set
import json
import re
import inspect

import config
import utils
import custom_indicators

logger = logging.getLogger(__name__)

_INDICATOR_DEFS: Optional[Dict[str, Dict]] = None
_TA_LIB_FUNCTION_GROUPS_CACHE: Optional[Dict] = None # Cache for TA-Lib function groups

# --- Constants for Ichimoku Filtering --- (Keep as before)
ICHIMOKU_COLUMNS_TO_KEEP = ['ISA_9', 'ISB_26', 'ITS_9', 'IKS_26', 'ICS_26'] # Default pandas-ta names
ICHIMOKU_OUTPUT_MAP = { # Map pandas-ta default names to clearer suffixes if desired
    'ISA_9': 'SPAN_A', 'ISB_26': 'SPAN_B', 'ITS_9': 'TENKAN', 'IKS_26': 'KIJUN', 'ICS_26': 'CHIKOU'
}
ICHIMOKU_COLUMNS_TO_EXCLUDE = ['TENKAN', 'KIJUN', 'CHIKOU'] # Exclude these from correlation

def _load_indicator_definitions() -> None:
    """Loads indicator definitions from JSON if not already loaded."""
    global _INDICATOR_DEFS
    if _INDICATOR_DEFS is None:
        try:
            with open(config.INDICATOR_PARAMS_PATH, 'r', encoding='utf-8') as f:
                full_json = json.load(f)
                _INDICATOR_DEFS = full_json.get("indicators", {})
            if not _INDICATOR_DEFS:
                 logger.warning(f"No 'indicators' key found or empty in {config.INDICATOR_PARAMS_PATH}")
            else:
                 logger.info(f"Loaded {len(_INDICATOR_DEFS)} indicator definitions from {config.INDICATOR_PARAMS_PATH}")
        except FileNotFoundError:
            logger.error(f"Indicator definitions file not found: {config.INDICATOR_PARAMS_PATH}")
            _INDICATOR_DEFS = {}
        except json.JSONDecodeError as json_err:
            logger.error(f"Error decoding JSON from {config.INDICATOR_PARAMS_PATH}: {json_err}")
            _INDICATOR_DEFS = {}
        except Exception as e:
            logger.error(f"Failed to load indicator definitions: {e}", exc_info=True)
            _INDICATOR_DEFS = {}

# --- Function to safely get TA-Lib output names ---
def _get_ta_lib_output_suffixes(ta_func_name: str) -> List[str]:
    """Safely retrieves the default output names for a TA-Lib function."""
    global _TA_LIB_FUNCTION_GROUPS_CACHE
    output_suffixes = []
    try:
        if _TA_LIB_FUNCTION_GROUPS_CACHE is None:
            _TA_LIB_FUNCTION_GROUPS_CACHE = ta.get_function_groups()

        if not isinstance(_TA_LIB_FUNCTION_GROUPS_CACHE, dict):
             logger.warning("ta.get_function_groups() did not return a dictionary. Cannot get output names.")
             return []

        # Iterate through the groups to find the one containing the function
        func_info = None
        for group_name, functions in _TA_LIB_FUNCTION_GROUPS_CACHE.items():
            if isinstance(functions, dict) and ta_func_name in functions:
                func_info = functions[ta_func_name]
                break # Found the function, no need to check other groups

        if func_info and isinstance(func_info, dict):
            output_suffixes = func_info.get('Output Names', [])
            if not isinstance(output_suffixes, list):
                 logger.warning(f"Expected list for 'Output Names' for {ta_func_name}, got {type(output_suffixes)}. Using default.")
                 output_suffixes = []
        # else: Function info not found or not a dict, output_suffixes remains []

    except Exception as e:
        logger.error(f"Error retrieving TA-Lib output suffixes for '{ta_func_name}': {e}", exc_info=False)
        output_suffixes = [] # Default to empty list on any error

    return output_suffixes
# --- End function ---


def _get_indicator_definition(indicator_name: str) -> Optional[Dict]:
    """Retrieves the definition for a single indicator by name."""
    if _INDICATOR_DEFS is None: _load_indicator_definitions()
    return _INDICATOR_DEFS.get(indicator_name)

def _compute_single_indicator(data: pd.DataFrame, config_details: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Computes a single indicator based on its configuration.
    Returns a DataFrame containing ONLY the new indicator column(s), or None on failure.
    """
    indicator_name = config_details['indicator_name']
    params = config_details.get('params', {}).copy()
    config_id = config_details.get('config_id')
    # Ensure config_id exists and is valid before proceeding
    if config_id is None or not isinstance(config_id, int):
         logger.error(f"Invalid or missing config_id in config_details for '{indicator_name}'. Skipping computation.")
         return None
    output_df = pd.DataFrame(index=data.index) # Initialize empty DF with correct index

    indicator_def = _get_indicator_definition(indicator_name)
    if not indicator_def:
        logger.error(f"Computation skipped: Definition not found for '{indicator_name}'.")
        return None # Return None on failure

    indicator_type = indicator_def.get('type')
    required_inputs = indicator_def.get('required_inputs', [])

    missing_cols = [col for col in required_inputs if col not in data.columns]
    if missing_cols:
        logger.error(f"Missing required columns {missing_cols} for '{indicator_name}'. Skipping Cfg {config_id}.")
        return None # Return None on failure

    computed_any = False
    try:
        # --- TA-Lib Indicators ---
        if indicator_type == 'ta-lib':
            ta_func_name = indicator_name.upper()
            ta_func = getattr(ta, ta_func_name, None)
            if not ta_func:
                logger.error(f"TA-Lib function '{ta_func_name}' not found. Skipping Cfg {config_id}.")
                return None

            final_params = params.copy()
            invalid_param_msg = None

            # MAType Handling (unchanged)
            matype_accepting_funcs = ['BBANDS', 'MA', 'APO', 'PPO', 'STOCH', 'STOCHF', 'STOCHRSI',
                                     'DEMA', 'EMA', 'KAMA', 'MIDPOINT', 'MIDPRICE', 'SMA', 'T3',
                                     'TEMA', 'TRIMA', 'WMA']
            if ta_func_name in matype_accepting_funcs:
                 if 'matype' in final_params:
                     matype_val = final_params['matype']
                     if not isinstance(matype_val, int) or not (0 <= matype_val <= 8):
                         logger.error(f"TA-Lib '{ta_func_name}' invalid matype: {matype_val}. Skip Cfg {config_id}.")
                         return None
            elif 'matype' in final_params:
                 logger.debug(f"Func {ta_func_name} ignores 'matype'. Removing for Cfg {config_id}.")
                 del final_params['matype']

            # Specific Parameter Validation (unchanged)
            # ... (TRIX, APO, PPO, MACD, ADOSC, ULTOSC, MAMA, SAR checks) ...
            min_period_funcs = ['SMA', 'EMA', 'DEMA', 'TEMA', 'WMA', 'TRIMA', 'KAMA', 'MIDPOINT', 'MIDPRICE', 'ADX', 'ADXR', 'AROON', 'AROONOSC', 'CCI', 'CMO', 'DX', 'MINUS_DI', 'MINUS_DM', 'PLUS_DI', 'PLUS_DM', 'RSI', 'WILLR', 'T3']
            if ta_func_name in min_period_funcs and final_params.get('timeperiod', -1) < 2:
                 invalid_param_msg = f"timeperiod({final_params.get('timeperiod')}) < 2"

            if invalid_param_msg:
                logger.error(f"TA-Lib '{ta_func_name}' invalid params ({invalid_param_msg}). Skip Cfg {config_id}.")
                return None

            # Prepare input series (ensure float64)
            func_args: List[pd.Series] = []
            for col in required_inputs:
                 if col not in data.columns: logger.error(f"Required col '{col}' not in data. Skip Cfg {config_id}."); return None
                 input_series = data[col]
                 if not pd.api.types.is_numeric_dtype(input_series):
                     logger.error(f"Input col '{col}' not numeric for TA-Lib '{ta_func_name}'. Skip Cfg {config_id}.")
                     return None
                 func_args.append(input_series.astype(np.float64))

            logger.debug(f"Calling TA-Lib '{ta_func_name}' (Cfg ID: {config_id}) with {len(func_args)} inputs, params: {final_params}")
            try:
                result = ta_func(*func_args, **final_params)
            except Exception as e: # Catch any exception during TA-Lib call
                 err_msg = str(e)
                 if "Bad Parameter" in err_msg or "TA_BAD_PARAM" in err_msg or "error code=2" in err_msg :
                      logger.error(f"TA-Lib '{ta_func_name}' (Cfg {config_id}) Bad Param error: {err_msg}. Params: {final_params}", exc_info=False)
                 else: logger.error(f"TA-Lib Exception calling '{ta_func_name}' (Cfg {config_id}, Params: {final_params}): {e}", exc_info=True)
                 return None # Return None on calculation error

            # Process TA-Lib result (use helper for output names)
            if isinstance(result, tuple):
                # --- Use helper function ---
                output_suffixes = _get_ta_lib_output_suffixes(ta_func_name)
                # --- End Use helper function ---
                for idx, res_output in enumerate(result):
                     if isinstance(res_output, (np.ndarray, pd.Series)) and len(res_output) == len(data.index):
                          # Use default numeric index if suffixes are missing or insufficient
                          suffix = output_suffixes[idx] if output_suffixes and idx < len(output_suffixes) else str(idx)
                          col_name = utils.get_config_identifier(indicator_name, config_id, suffix)
                          output_df[col_name] = res_output; computed_any = True
                     elif res_output is not None: logger.warning(f"TA-Lib '{ta_func_name}' output {idx} bad type/len. Skipping.")
            elif isinstance(result, (np.ndarray, pd.Series)) and len(result) == len(data.index):
                 col_name = utils.get_config_identifier(indicator_name, config_id, None)
                 output_df[col_name] = result; computed_any = True
            elif result is not None: logger.warning(f"TA-Lib '{ta_func_name}' bad output type/length. Skipping.")

        # --- Pandas-TA Indicators ---
        elif indicator_type == 'pandas-ta':
            if not PTA_AVAILABLE: logger.error(f"pandas_ta not installed. Skip '{indicator_name}' Cfg {config_id}."); return None
            pta_func_name = indicator_name.lower(); pta_func = None; is_accessor = False
            if hasattr(data.ta, pta_func_name): is_accessor = True
            elif hasattr(pta, pta_func_name): pta_func = getattr(pta, pta_func_name)
            else: logger.error(f"PTA func/accessor '{pta_func_name}' not found. Skip Cfg {config_id}."); return None

            final_params = params.copy(); result = None
            try:
                logger.debug(f"Calling PTA '{pta_func_name}' (Accessor: {is_accessor}, Cfg {config_id}) params: {final_params}")
                if is_accessor: # Call via accessor (e.g., data.ta.rsi())
                    func_ref = getattr(data.ta, pta_func_name); sig = inspect.signature(func_ref)
                    required_accessor_cols = set(sig.parameters.keys()) & {'open', 'high', 'low', 'close', 'volume'}
                    if not required_accessor_cols.issubset(data.columns):
                         missing = required_accessor_cols - set(data.columns)
                         logger.error(f"PTA accessor '{pta_func_name}' missing required cols {missing}. Skip Cfg {config_id}.")
                         return None
                    result = func_ref(**final_params)
                elif pta_func: # Call standalone function (e.g., pta.rsi(data['close']))
                    sig = inspect.signature(pta_func); expected_args = list(sig.parameters.keys())
                    call_args = {}; input_cols_provided = set()
                    for req_col in required_inputs: # Map required inputs to function args
                         if req_col not in data.columns: logger.error(f"PTA '{pta_func_name}' missing input '{req_col}'. Skip Cfg {config_id}."); return None
                         target_arg_name = req_col # Default
                         if req_col == 'close' and 'close' in expected_args: target_arg_name = 'close'
                         # ... add mappings for high, low, open, volume ...
                         if target_arg_name in expected_args: call_args[target_arg_name] = data[req_col].astype(float); input_cols_provided.add(target_arg_name)
                         elif req_col in expected_args: call_args[req_col] = data[req_col].astype(float); input_cols_provided.add(req_col)
                         else: logger.warning(f"PTA '{pta_func_name}': Cannot map input '{req_col}'. Skipping arg.")
                    result = pta_func(**call_args, **final_params)
                else: raise AttributeError(f"PTA state invalid for '{pta_func_name}'.")
            except Exception as pta_e: # Catch any exception during PTA call
                err_msg = str(pta_e)
                if "TA_BAD_PARAM" in err_msg or "error code 2" in err_msg: logger.error(f"PTA '{pta_func_name}' (Cfg {config_id}) Bad Param: {err_msg}. Params: {final_params}", exc_info=False)
                else: logger.error(f"PTA Error calling '{pta_func_name}' (Cfg {config_id}): {pta_e}", exc_info=True)
                return None # Return None on calculation error

            # Process PTA result (including Ichimoku filtering) (unchanged logic)
            if isinstance(result, tuple): # Handle tuple output
                 processed_outputs = []
                 for item in result:
                      if isinstance(item, pd.DataFrame): processed_outputs.append(item)
                      elif isinstance(item, pd.Series): processed_outputs.append(item.to_frame(name=item.name or f"{pta_func_name}_out"))
                 if not processed_outputs: result = None
                 else:
                      try: result = pd.concat(processed_outputs, axis=1)
                      except Exception as concat_err: logger.error(f"Error concat PTA tuple {pta_func_name} (Cfg {config_id}): {concat_err}"); result = None

            if isinstance(result, pd.DataFrame): # Handle DataFrame output
                 logger.debug(f"Processing pta DF output. Columns: {list(result.columns)}")
                 for col in result.columns:
                      if col not in result or result[col] is None: continue
                      series_to_add = result[col]
                      if pd.api.types.is_numeric_dtype(series_to_add):
                           is_ichimoku = indicator_name.lower() == 'ichimoku'; output_suffix = None
                           for pta_name, clean_suffix in ICHIMOKU_OUTPUT_MAP.items():
                                if col.startswith(pta_name): output_suffix = clean_suffix; break
                           if output_suffix is None: output_suffix = str(col)
                           if is_ichimoku and output_suffix in ICHIMOKU_COLUMNS_TO_EXCLUDE:
                                logger.debug(f"Excluding Ichimoku col '{col}' (Mapped: {output_suffix}).")
                                continue
                           col_name_final = utils.get_config_identifier(indicator_name, config_id, output_suffix)
                           # Ensure unique final column name
                           original_col_name = col_name_final; dup_counter = 1
                           while col_name_final in output_df.columns: col_name_final = f"{original_col_name}_dup{dup_counter}"; dup_counter += 1
                           output_df[col_name_final] = series_to_add.reindex(data.index); computed_any = True
                      # else: logger.debug(f"Skipping non-numeric pta col '{col}'.")
            elif isinstance(result, pd.Series): # Handle Series output
                 if result is not None and pd.api.types.is_numeric_dtype(result):
                      col_name = utils.get_config_identifier(indicator_name, config_id, None)
                      output_df[col_name] = result.reindex(data.index); computed_any = True
                 # else: logger.warning(f"PTA '{pta_func_name}' returned non-numeric/None Series.")
            elif result is not None: logger.warning(f"PTA '{pta_func_name}' returned unexpected type {type(result)}.")

        # --- Custom Indicators ---
        elif indicator_type == 'custom':
             # Custom indicator logic remains the same
            custom_func_name = indicator_def.get('function_name', f"compute_{indicator_name.lower()}")
            custom_func = getattr(custom_indicators, custom_func_name, None)
            if custom_func:
                sig = inspect.signature(custom_func); valid_params = {k:v for k,v in params.items() if k in sig.parameters}
                logger.debug(f"Calling custom func '{custom_func_name}' (Cfg {config_id}) params: {valid_params}")
                try:
                    result_df_custom = custom_func(data, **valid_params) # Pass data directly
                    if result_df_custom is None: logger.warning(f"Custom func '{custom_func_name}' (Cfg {config_id}) returned None."); return None
                    if not isinstance(result_df_custom, pd.DataFrame): logger.error(f"Custom func '{custom_func_name}' returned {type(result_df_custom)}, expected DF."); return None
                    if result_df_custom.empty: logger.info(f"Custom func '{custom_func_name}' returned empty DF.") # Not an error, just no output
                    else:
                        for new_col in result_df_custom.columns:
                             if new_col in result_df_custom and pd.api.types.is_numeric_dtype(result_df_custom[new_col]):
                                  suffix = new_col if len(result_df_custom.columns) > 1 else None
                                  final_col_name = utils.get_config_identifier(indicator_name, config_id, suffix)
                                  original_col_name = final_col_name; dup_counter = 1
                                  while final_col_name in output_df.columns: final_col_name = f"{original_col_name}_dup{dup_counter}"; dup_counter += 1
                                  output_df[final_col_name] = result_df_custom[new_col].reindex(data.index); computed_any = True
                             else: logger.warning(f"Custom func '{custom_func_name}' non-numeric col '{new_col}'. Skipping.")
                except Exception as custom_e: logger.error(f"Error exec custom func '{custom_func_name}' (Cfg {config_id}): {custom_e}", exc_info=True); return None
            else: logger.error(f"Custom func '{custom_func_name}' not found. Skip Cfg {config_id}."); return None

        # --- Unknown Indicator Type ---
        else:
            logger.error(f"Unknown indicator type '{indicator_type}'. Skip Cfg {config_id}."); return None

    except Exception as e:
        logger.error(f"General error computing indicator '{indicator_name}' (Cfg {config_id}): {e}", exc_info=True)
        return None # Return None on general failure

    # --- Final Processing ---
    if computed_any and not output_df.empty:
        output_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Convert numeric columns to float64
        for col in output_df.columns:
            if pd.api.types.is_numeric_dtype(output_df[col]):
                 output_df[col] = output_df[col].astype(np.float64)
            else: # Attempt conversion if not numeric, coerce errors
                 output_df[col] = pd.to_numeric(output_df[col], errors='coerce')
                 if output_df[col].isnull().all(): logger.warning(f"Col '{col}' became all NaN after coercion (Cfg {config_id}).")
                 else: output_df[col] = output_df[col].astype(np.float64)

        # Check if *all* computed columns are entirely NaN AFTER processing
        if output_df.isnull().all().all():
            logger.warning(f"All output columns are NaN for {indicator_name} (Cfg {config_id}). Treated as failure.")
            return None # Treat all-NaN output as a failure

        logger.debug(f"Success for {indicator_name} (Cfg {config_id}) -> Cols Added: {list(output_df.columns)}")
        return output_df
    else:
        # Only log if no columns were computed at all
        if not computed_any:
            logger.info(f"No output generated for '{indicator_name}' (Cfg {config_id}).")
        return None # Return None if no columns computed or initial DF was empty


def compute_configured_indicators(
        data: pd.DataFrame, configs_to_process: List[Dict[str, Any]]
) -> Tuple[pd.DataFrame, Set[int]]: # <-- Return DataFrame and Set of failed config IDs
    """
    Computes all indicators specified in the list of configurations.
    Returns a NEW DataFrame containing original data plus all successfully computed indicator columns,
    AND a set containing the config_ids that failed calculation.
    """
    _load_indicator_definitions()
    if not _INDICATOR_DEFS:
        logger.error("Indicator definitions missing. Cannot compute indicators.")
        return data.copy(), set(cfg.get('config_id') for cfg in configs_to_process if cfg.get('config_id') is not None) # Return all as failed
    if not configs_to_process:
        logger.warning("No indicator configurations provided.")
        return data.copy(), set()

    all_indicator_outputs: List[pd.DataFrame] = []
    failed_config_ids: Set[int] = set() # <-- Track failed IDs
    total_configs = len(configs_to_process)
    logger.info(f"Processing {total_configs} indicator configurations...")
    success_count = 0; fail_count = 0

    for i, config_details in enumerate(configs_to_process):
        indicator_name = config_details.get('indicator_name', 'Unknown')
        config_id = config_details.get('config_id', None) # Use None as default if missing

        # Simple progress print
        print(f"\rCalculating Indicators: {i+1}/{total_configs} ({indicator_name[:15]}...)", end="")

        # Skip if config_id is missing or invalid early
        if config_id is None or not isinstance(config_id, int):
             logger.error(f"Skipping config at index {i}: Invalid/missing config_id ({config_id}) for indicator '{indicator_name}'.")
             fail_count += 1
             # Add None or a placeholder if you need to track which *input* configs failed vs calculation fails?
             # For now, just don't add to failed_config_ids if the ID itself was bad.
             continue

        logger.debug(f"Computing indicator {i+1}/{total_configs}: {indicator_name} (Config ID: {config_id})")
        indicator_df = _compute_single_indicator(data, config_details) # Pass original data

        # Check failure conditions: None result, empty DataFrame, or all-NaN DataFrame
        if indicator_df is None or indicator_df.empty or indicator_df.isnull().all().all():
             logger.info(f"Failed/empty/all-NaN result for {indicator_name} (Config ID: {config_id}).")
             failed_config_ids.add(config_id) # <-- Add ID to failed set
             fail_count += 1
        else:
             all_indicator_outputs.append(indicator_df)
             success_count += 1

    print() # Newline after loop

    logger.info(f"Indicator computations done. Success: {success_count}, Failed/Skipped: {fail_count}.")

    if all_indicator_outputs:
        logger.info(f"Concatenating results from {len(all_indicator_outputs)} successful computations...")
        try:
            # Ensure alignment before concat
            aligned_outputs = [df.reindex(data.index) for df in all_indicator_outputs]
            indicators_combined_df = pd.concat(aligned_outputs, axis=1)

            # --- Column Name Conflict Checks --- (Remain the same)
            new_cols = indicators_combined_df.columns; existing_cols = data.columns
            overlap = new_cols.intersection(existing_cols)
            if not overlap.empty: logger.error(f"FATAL: Overlap new/existing cols: {list(overlap)}"); raise ValueError("Indicator column name conflict.")
            new_col_counts = new_cols.value_counts(); dup_new_cols = new_col_counts[new_col_counts > 1].index.tolist()
            if dup_new_cols: logger.error(f"FATAL: Duplicate indicator cols generated: {dup_new_cols}"); raise ValueError("Duplicate indicator column names.")
            # --- End Conflict Checks ---

            data_with_indicators = pd.concat([data.copy(), indicators_combined_df], axis=1)
            logger.info(f"Added {indicators_combined_df.shape[1]} indicator columns.")
            logger.info(f"Final DataFrame shape: {data_with_indicators.shape}")
            return data_with_indicators, failed_config_ids # <-- Return DF and failed IDs

        except Exception as concat_err:
            logger.error(f"Error concatenating indicator results: {concat_err}", exc_info=True)
            # Return original data, but mark ALL processed configs as failed in this case?
            all_processed_ids = {cfg.get('config_id') for cfg in configs_to_process if cfg.get('config_id') is not None}
            return data.copy(), failed_config_ids.union(all_processed_ids) # Add all potentially processed IDs to failed set
    else:
        logger.warning("No indicators successfully computed/added. Returning original data.")
        all_processed_ids = {cfg.get('config_id') for cfg in configs_to_process if cfg.get('config_id') is not None}
        return data.copy(), failed_config_ids.union(all_processed_ids) # Return all as failed if none succeeded