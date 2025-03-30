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

from typing import List, Dict, Any, Optional, Tuple
import json
import re
import inspect

import config
import utils
import custom_indicators

logger = logging.getLogger(__name__)

_INDICATOR_DEFS: Optional[Dict[str, Dict]] = None

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
        logger.error(f"Missing required columns {missing_cols} for '{indicator_name}'. Skipping Cfg {config_id}.")
        return None

    computed_any = False
    try:
        # --- TA-Lib Indicators ---
        if indicator_type == 'ta-lib':
            ta_func_name = indicator_name.upper()
            ta_func = getattr(ta, ta_func_name, None)
            if not ta_func:
                logger.error(f"TA-Lib function '{ta_func_name}' not found for '{indicator_name}'. Skipping Cfg {config_id}.")
                return None

            func_args: List[pd.Series] = []
            for col in required_inputs:
                 input_series = data[col]
                 if not pd.api.types.is_numeric_dtype(input_series):
                     logger.error(f"Input column '{col}' for TA-Lib '{ta_func_name}' not numeric. Skipping Cfg {config_id}.")
                     return None
                 func_args.append(input_series.astype(float))

            final_params = params.copy()
            logger.debug(f"Calling TA-Lib '{ta_func_name}' (Cfg ID: {config_id}) with {len(func_args)} inputs, params: {final_params}")

            # Pre-validation for known TA-Lib issues
            if ta_func_name == 'TRIX' and final_params.get('timeperiod', -1) == 1:
                 logger.error(f"TA-Lib 'TRIX' cannot accept timeperiod=1. Skipping Cfg {config_id}.")
                 return None
            # Add more specific checks if needed...

            try:
                result = ta_func(*func_args, **final_params)
            except (TypeError, ValueError) as e: # Catch common TA-Lib call errors
                 logger.error(f"Error calling TA-Lib '{ta_func_name}' (Cfg ID: {config_id}): {e}. Skipping.", exc_info=False)
                 return None

            # Process result
            if isinstance(result, tuple): # Multiple outputs
                for idx, res_output in enumerate(result):
                    if isinstance(res_output, (np.ndarray, pd.Series)) and len(res_output) == len(data.index):
                        col_name = utils.get_config_identifier(indicator_name, config_id, idx)
                        output_df[col_name] = res_output
                        computed_any = True
                        logger.debug(f"  -> Added TA-Lib output column {idx}: {col_name}")
                    elif res_output is not None:
                         logger.warning(f"TA-Lib '{ta_func_name}' output {idx} bad type/length. Skipping.")
            elif isinstance(result, (np.ndarray, pd.Series)) and len(result) == len(data.index): # Single output
                 col_name = utils.get_config_identifier(indicator_name, config_id, None)
                 output_df[col_name] = result
                 computed_any = True
                 logger.debug(f"  -> Added TA-Lib single output column: {col_name}")
            elif result is not None:
                 logger.warning(f"TA-Lib '{ta_func_name}' bad output type/length. Skipping.")

        # --- Pandas-TA Indicators ---
        elif indicator_type == 'pandas-ta':
            if not PTA_AVAILABLE:
                logger.error(f"pandas_ta not installed, cannot compute '{indicator_name}'. Skipping Cfg {config_id}.")
                return None

            pta_func_name = indicator_name.lower()
            pta_func = None; is_accessor = False
            if hasattr(pta, pta_func_name): pta_func = getattr(pta, pta_func_name)
            elif hasattr(data.ta, pta_func_name): is_accessor = True
            else:
                logger.error(f"Pandas TA function/accessor '{pta_func_name}' not found for '{indicator_name}'. Skipping Cfg {config_id}.")
                return None

            final_params = params.copy(); result = None
            input_data_dict = {col: data[col] for col in required_inputs}

            try:
                logger.debug(f"Calling Pandas TA '{pta_func_name}' (Accessor: {is_accessor}, Cfg ID: {config_id}) with params: {final_params}")
                if pta_func: result = pta_func(**input_data_dict, **final_params)
                elif is_accessor:
                    if all(c in ['open', 'high', 'low', 'close', 'volume'] for c in required_inputs):
                        result = getattr(data.ta, pta_func_name)(**final_params)
                    else:
                         logger.error(f"Pandas TA accessor '{pta_func_name}' needs OHLCV, but indicator requires {required_inputs}. Skipping Cfg {config_id}.")
                         return None
                else: raise AttributeError(f"PTA function/accessor check failed.")
            except Exception as pta_e:
                logger.error(f"Error calling Pandas TA '{pta_func_name}' (Cfg ID: {config_id}): {pta_e}", exc_info=True)
                return None

            # Process result
            if isinstance(result, pd.DataFrame):
                 logger.debug(f"Processing pta DataFrame output. Columns: {list(result.columns)}")
                 for idx, col in enumerate(result.columns):
                      if col in result and result[col] is not None and pd.api.types.is_numeric_dtype(result[col]):
                           col_suffix = col.upper()
                           base_prefix = f"{indicator_name.lower()}_"
                           if col.lower().startswith(base_prefix):
                               suffix_cand = col[len(base_prefix):]
                               if suffix_cand: col_suffix = suffix_cand.upper()
                           col_name_final = utils.get_config_identifier(indicator_name, config_id, col_suffix)
                           # Handle duplicates within this config's output
                           original_col_name = col_name_final; dup_counter = 1
                           while col_name_final in output_df.columns:
                               col_name_final = f"{original_col_name}_dup{dup_counter}"; dup_counter += 1
                           output_df[col_name_final] = result[col].reindex(data.index)
                           computed_any = True
                           logger.debug(f"  -> Added pta DF output column: {col_name_final}")
                      else: logger.debug(f"Skipping non-numeric/None column '{col}' from pta DF output.")
            elif isinstance(result, pd.Series):
                 if result is not None and pd.api.types.is_numeric_dtype(result):
                      col_name = utils.get_config_identifier(indicator_name, config_id, None)
                      output_df[col_name] = result.reindex(data.index)
                      computed_any = True
                      logger.debug(f"  -> Added pta Series output column: {col_name}")
                 else: logger.warning(f"PTA '{pta_func_name}' returned non-numeric/None Series. Skipping.")
            elif result is not None: logger.warning(f"PTA '{pta_func_name}' returned unexpected type: {type(result)}. Skipping.")

        # --- Custom Indicators ---
        elif indicator_type == 'custom':
            custom_func_name = indicator_def.get('function_name', f"compute_{indicator_name.lower()}")
            custom_func = getattr(custom_indicators, custom_func_name, None)
            if custom_func:
                sig = inspect.signature(custom_func)
                valid_params = {k:v for k,v in params.items() if k in sig.parameters}
                logger.debug(f"Calling custom func '{custom_func_name}' (Cfg ID: {config_id}) with params: {valid_params}")
                try:
                    # Custom functions are expected to return a modified COPY of the dataframe
                    # or a dataframe with just the new columns.
                    # Let's assume they return a dataframe with the new column(s).
                    result_df_custom = custom_func(data, **valid_params) # Pass original data

                    if not isinstance(result_df_custom, pd.DataFrame):
                         logger.error(f"Custom func '{custom_func_name}' (Cfg {config_id}) returned {type(result_df_custom)}, expected DataFrame. Skipping."); return None

                    # Identify columns added by the custom function
                    original_cols = set(data.columns)
                    new_cols = [col for col in result_df_custom.columns if col not in original_cols]

                    if new_cols:
                        logger.debug(f"Custom func '{custom_func_name}' added cols: {new_cols}")
                        for new_col in new_cols:
                             if new_col in result_df_custom and pd.api.types.is_numeric_dtype(result_df_custom[new_col]):
                                 # Assume the new column name IS the indicator name for identification
                                 final_col_name = utils.get_config_identifier(new_col, config_id, None)
                                 # Handle duplicates
                                 original_col_name = final_col_name; dup_counter = 1
                                 while final_col_name in output_df.columns:
                                     final_col_name = f"{original_col_name}_dup{dup_counter}"; dup_counter += 1
                                 output_df[final_col_name] = result_df_custom[new_col].reindex(data.index) # Align index
                                 computed_any = True
                                 logger.debug(f"  -> Added custom indicator output column: {final_col_name}")
                             else: logger.warning(f"Custom func '{custom_func_name}' added non-numeric/missing col '{new_col}'. Skipping.")
                    else: logger.warning(f"Custom func '{custom_func_name}' (Cfg {config_id}) added no new columns.")
                except Exception as custom_e:
                    logger.error(f"Error executing custom func '{custom_func_name}' (Cfg {config_id}): {custom_e}", exc_info=True)
            else: logger.error(f"Custom func '{custom_func_name}' not found for '{indicator_name}'. Skipping Cfg {config_id}.")

        # --- Unknown Indicator Type ---
        else:
            logger.error(f"Unknown indicator type '{indicator_type}' for '{indicator_name}'. Skipping Cfg {config_id}.")
            return None

    except Exception as e:
        logger.error(f"General error computing indicator '{indicator_name}' (Cfg {config_id}): {e}", exc_info=True)
        return None

    # --- Final Processing ---
    if computed_any and not output_df.empty:
        output_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in output_df.columns:
            if pd.api.types.is_numeric_dtype(output_df[col]):
                 output_df[col] = output_df[col].astype(np.float64) # Ensure float64
        logger.debug(f"Success for {indicator_name} (Cfg {config_id}) -> Columns: {list(output_df.columns)}")
        return output_df
    else:
        logger.warning(f"No output generated or empty for '{indicator_name}' (Cfg {config_id}).")
        return None

def compute_configured_indicators(data: pd.DataFrame, configs_to_process: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Computes all indicators specified in the list of configurations.
    Returns a NEW DataFrame containing original data plus all successfully computed indicator columns.
    """
    _load_indicator_definitions()
    if not _INDICATOR_DEFS:
        logger.error("Indicator definitions missing. Cannot compute indicators.")
        return data.copy()
    if not configs_to_process:
        logger.warning("No indicator configurations provided.")
        return data.copy()

    all_indicator_outputs: List[pd.DataFrame] = []
    logger.info(f"Processing {len(configs_to_process)} indicator configurations...")
    success_count = 0; fail_count = 0

    for i, config_details in enumerate(configs_to_process):
        indicator_name = config_details.get('indicator_name', 'Unknown')
        config_id = config_details.get('config_id', 'N/A')
        logger.info(f"Computing indicator {i+1}/{len(configs_to_process)}: {indicator_name} (Config ID: {config_id})")

        indicator_df = _compute_single_indicator(data, config_details) # Pass original data, func doesn't modify

        if indicator_df is not None and not indicator_df.empty:
            nan_check = indicator_df.isnull().all()
            all_nan_cols = nan_check[nan_check].index.tolist()
            if all_nan_cols:
                 logger.warning(f"Indicator {indicator_name} (Cfg {config_id}) produced only NaN columns: {all_nan_cols}. Excluding.")
                 fail_count += 1
            else:
                 all_indicator_outputs.append(indicator_df)
                 success_count += 1
        else:
             logger.warning(f"Failed/empty result for {indicator_name} (Config ID: {config_id}).")
             fail_count += 1

    logger.info(f"Individual computations done. Success: {success_count}, Failed/Skipped: {fail_count}.")

    if all_indicator_outputs:
        logger.info(f"Concatenating results from {len(all_indicator_outputs)} successful computations...")
        try:
            aligned_outputs = [df.reindex(data.index) for df in all_indicator_outputs]
            indicators_combined_df = pd.concat(aligned_outputs, axis=1)

            # Check for Column Name Conflicts
            new_cols = indicators_combined_df.columns; existing_cols = data.columns
            overlap = new_cols.intersection(existing_cols)
            if not overlap.empty:
                logger.error(f"FATAL: Overlap between new/existing columns: {list(overlap)}")
                raise ValueError("Indicator column name conflict detected with original data.")
            dup_new_cols = new_cols[new_cols.duplicated(keep=False)].unique()
            if not dup_new_cols.empty:
                logger.error(f"FATAL: Duplicate indicator column names detected: {list(dup_new_cols)}")
                raise ValueError("Duplicate indicator column names detected.")

            data_with_indicators = pd.concat([data.copy(), indicators_combined_df], axis=1)
            logger.info(f"Added {indicators_combined_df.shape[1]} indicator columns.")
            logger.info(f"Final DataFrame shape (with NaNs preserved): {data_with_indicators.shape}")
            return data_with_indicators

        except Exception as concat_err:
            logger.error(f"Error concatenating indicator results: {concat_err}", exc_info=True)
            return data.copy()
    else:
        logger.warning("No indicators successfully computed/added. Returning original data.")
        return data.copy()
