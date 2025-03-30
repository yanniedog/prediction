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

            final_params = params.copy() # Start with a copy of incoming params
            invalid_param_msg = None # Reset for each indicator

            # --- Refined MAType Handling ---
            # List of TA-Lib functions that *can* accept a 'matype' parameter
            matype_accepting_funcs = ['BBANDS', 'MA', 'APO', 'PPO', 'STOCH', 'STOCHF', 'STOCHRSI',
                                     'DEMA', 'EMA', 'KAMA', 'MIDPOINT', 'MIDPRICE', 'SMA', 'T3',
                                     'TEMA', 'TRIMA', 'WMA']
            if ta_func_name in matype_accepting_funcs:
                if 'matype' in final_params:
                    matype_val = final_params['matype']
                    # TA-Lib uses integer codes 0-8 for MA types
                    if not isinstance(matype_val, int) or not (0 <= matype_val <= 8):
                         logger.error(f"TA-Lib '{ta_func_name}' received invalid matype: {matype_val}. Must be integer 0-8. Skipping Cfg {config_id}.")
                         return None
                    # Keep valid matype in final_params if provided
                # else: # matype not in params, let TA-Lib handle its internal default (usually 0/SMA)
                #    pass # Don't add it explicitly
            elif 'matype' in final_params:
                 # Function doesn't accept matype, but it was provided (e.g., from optimization)
                 logger.debug(f"Function {ta_func_name} does not use 'matype'. Removing it from params for Cfg {config_id}.")
                 del final_params['matype'] # Remove extraneous parameter
            # --- END Refined MAType Handling ---

            # --- Existing Specific TA-Lib Parameter VALIDATION Checks ---
            # These checks look for invalid relationships between parameters (e.g., fast > slow)
            if ta_func_name == 'TRIX' and final_params.get('timeperiod', -1) == 1:
                invalid_param_msg = "timeperiod=1"
            elif ta_func_name == 'APO' and final_params.get('fastperiod', 1) >= final_params.get('slowperiod', 0):
                invalid_param_msg = f"fastperiod({final_params.get('fastperiod')}) >= slowperiod({final_params.get('slowperiod')})"
            elif ta_func_name == 'PPO' and final_params.get('fastperiod', 1) >= final_params.get('slowperiod', 0):
                 invalid_param_msg = f"fastperiod({final_params.get('fastperiod')}) >= slowperiod({final_params.get('slowperiod')})"
            elif ta_func_name == 'MACD' and final_params.get('fastperiod', 1) >= final_params.get('slowperiod', 0):
                 invalid_param_msg = f"fastperiod({final_params.get('fastperiod')}) >= slowperiod({final_params.get('slowperiod')})"
            elif ta_func_name == 'ADOSC' and final_params.get('fastperiod', 1) >= final_params.get('slowperiod', 0):
                 invalid_param_msg = f"fastperiod({final_params.get('fastperiod')}) >= slowperiod({final_params.get('slowperiod')})"
            elif ta_func_name == 'ULTOSC':
                if not (final_params.get('timeperiod1', 0) < final_params.get('timeperiod2', -1) < final_params.get('timeperiod3', -2)):
                    invalid_param_msg = f"timeperiods ({final_params.get('timeperiod1')},{final_params.get('timeperiod2')},{final_params.get('timeperiod3')}) not strictly increasing"
            elif ta_func_name == 'MAMA':
                if not (0.01 <= final_params.get('slowlimit', -1) < final_params.get('fastlimit', 0) <= 0.99):
                     invalid_param_msg = f"fastlimit({final_params.get('fastlimit')}) / slowlimit({final_params.get('slowlimit')}) invalid range or order"
            elif ta_func_name == 'SAR':
                if not (0 < final_params.get('acceleration', -1) <= final_params.get('maximum', -2)):
                     invalid_param_msg = f"acceleration({final_params.get('acceleration')}) / maximum({final_params.get('maximum')}) invalid range or order"

            min_period_funcs = ['SMA', 'EMA', 'DEMA', 'TEMA', 'WMA', 'TRIMA', 'KAMA', 'MIDPOINT', 'MIDPRICE', 'ADX', 'ADXR', 'AROON', 'AROONOSC', 'CCI', 'CMO', 'DX', 'MINUS_DI', 'MINUS_DM', 'PLUS_DI', 'PLUS_DM', 'RSI', 'WILLR', 'T3'] # Added MIDPRICE
            if ta_func_name in min_period_funcs and final_params.get('timeperiod', -1) < 2:
                 invalid_param_msg = f"timeperiod({final_params.get('timeperiod')}) < 2"
            # --- END Specific Parameter VALIDATION ---

            if invalid_param_msg:
                logger.error(f"TA-Lib '{ta_func_name}' invalid params relationship ({invalid_param_msg}). Skipping Cfg {config_id}.")
                return None

            # Prepare input series
            func_args: List[pd.Series] = []
            for col in required_inputs:
                 input_series = data[col]
                 if not pd.api.types.is_numeric_dtype(input_series):
                     logger.error(f"Input column '{col}' for TA-Lib '{ta_func_name}' not numeric. Skipping Cfg {config_id}.")
                     return None
                 func_args.append(input_series.astype(float))

            logger.debug(f"Calling TA-Lib '{ta_func_name}' (Cfg ID: {config_id}) with {len(func_args)} inputs, FINAL params: {final_params}")

            try:
                result = ta_func(*func_args, **final_params) # Use final_params dict
            except (TypeError, ValueError) as e: # Catch common TA-Lib call errors
                 logger.error(f"Error calling TA-Lib '{ta_func_name}' (Cfg ID: {config_id}, Params: {final_params}): {e}. Skipping.", exc_info=False) # Include params in log
                 return None
            except Exception as e: # Catch TA-Lib's Exception type for specific errors
                # Check if it's the 'Bad Parameter' error we expect might happen
                if "Bad Parameter" in str(e) or "TA_BAD_PARAM" in str(e):
                     logger.error(f"TA-Lib function '{ta_func_name}' (Cfg ID: {config_id}) failed with Bad Parameter (TA_BAD_PARAM): {e}. Params: {final_params}", exc_info=False)
                else: # Log other TA-Lib exceptions more generally
                     logger.error(f"Unhandled TA-Lib Exception calling '{ta_func_name}' (Cfg ID: {config_id}, Params: {final_params}): {e}", exc_info=True)
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
            if hasattr(data.ta, pta_func_name):
                is_accessor = True
            elif hasattr(pta, pta_func_name):
                pta_func = getattr(pta, pta_func_name)
            else:
                logger.error(f"Pandas TA function/accessor '{pta_func_name}' not found. Skipping Cfg {config_id}.")
                return None

            final_params = params.copy(); result = None
            try:
                logger.debug(f"Calling Pandas TA '{pta_func_name}' (Accessor: {is_accessor}, Cfg ID: {config_id}) with params: {final_params}")
                if is_accessor:
                    if all(c in data.columns for c in ['open', 'high', 'low', 'close']):
                        result = getattr(data.ta, pta_func_name)(**final_params)
                    else:
                         logger.error(f"Pandas TA accessor '{pta_func_name}' called, but base data missing OHLC. Skipping Cfg {config_id}.")
                         return None
                elif pta_func:
                    sig = inspect.signature(pta_func)
                    expected_args = list(sig.parameters.keys())
                    call_args = {}
                    for req_col in required_inputs:
                        if req_col not in data.columns:
                            logger.error(f"PTA '{pta_func_name}': Required input '{req_col}' missing from data. Skipping Cfg {config_id}.")
                            return None
                        mapped = False
                        if req_col in expected_args: call_args[req_col] = data[req_col]; mapped=True
                        else: # Attempt common mappings
                            if req_col == 'close' and 'close' in expected_args: call_args['close'] = data[req_col]; mapped=True
                            elif req_col == 'high' and 'high' in expected_args: call_args['high'] = data[req_col]; mapped=True
                            elif req_col == 'low' and 'low' in expected_args: call_args['low'] = data[req_col]; mapped=True
                            elif req_col == 'open' and 'open' in expected_args: call_args['open'] = data[req_col]; mapped=True
                            elif req_col == 'volume' and 'volume' in expected_args: call_args['volume'] = data[req_col]; mapped=True
                        if not mapped:
                            logger.warning(f"PTA '{pta_func_name}': Input '{req_col}' not directly in signature {expected_args} and no obvious map. Passing anyway.")
                            call_args[req_col] = data[req_col] # Pass anyway? Might fail.
                    result = pta_func(**call_args, **final_params)
                else: raise AttributeError(f"PTA function/accessor state invalid for '{pta_func_name}'.")

            except Exception as pta_e:
                # Check specifically for TA-Lib's Bad Parameter exception if it originates there
                # Search within the exception message and its arguments for the code
                if "TA_BAD_PARAM" in str(pta_e) or "error code 2" in str(pta_e):
                     logger.error(f"Error calling Pandas TA '{pta_func_name}' (Cfg ID: {config_id}): Underlying TA-Lib error TA_BAD_PARAM occurred. Params: {final_params}. Orig Err: {pta_e}", exc_info=False)
                else:
                    logger.error(f"Error calling Pandas TA '{pta_func_name}' (Cfg ID: {config_id}): {pta_e}", exc_info=True)
                return None

            # Handle Pandas-TA returning Tuple
            if isinstance(result, tuple):
                logger.debug(f"Processing pta Tuple output. Length: {len(result)}")
                processed_outputs = []
                for item_idx, item in enumerate(result):
                    if isinstance(item, pd.DataFrame): processed_outputs.append(item)
                    elif isinstance(item, pd.Series): processed_outputs.append(item.to_frame(name=item.name or f"{pta_func_name}_output_{item_idx}"))
                if not processed_outputs: logger.warning(f"PTA '{pta_func_name}' returned tuple, but no DF/Series found. Skipping."); result = None
                else:
                    logger.debug(f"Concatenating {len(processed_outputs)} DFs/Series from pta tuple.")
                    try: result = pd.concat(processed_outputs, axis=1)
                    except Exception as concat_err: logger.error(f"Error concatenating pta tuple results for {pta_func_name} (Cfg {config_id}): {concat_err}"); result = None

            # Process result (DF, Series, or concatenated DF)
            if isinstance(result, pd.DataFrame):
                 logger.debug(f"Processing pta DataFrame output. Columns: {list(result.columns)}")
                 for idx, col in enumerate(result.columns):
                      if col in result and result[col] is not None and pd.api.types.is_numeric_dtype(result[col]):
                           col_suffix = col; base_prefix = f"{indicator_name.lower()}_"
                           if isinstance(col, str) and col.lower().startswith(base_prefix): suffix_cand = col[len(base_prefix):]; col_suffix = suffix_cand.upper() if suffix_cand else str(idx)
                           elif isinstance(col, int): col_suffix = str(idx)
                           elif not isinstance(col, str): col_suffix = str(col)
                           col_name_final = utils.get_config_identifier(indicator_name, config_id, col_suffix)
                           original_col_name = col_name_final; dup_counter = 1
                           while col_name_final in output_df.columns: col_name_final = f"{original_col_name}_dup{dup_counter}"; dup_counter += 1
                           output_df[col_name_final] = result[col].reindex(data.index)
                           computed_any = True; logger.debug(f"  -> Added pta DF output column: {col_name_final}")
                      else: logger.debug(f"Skipping non-numeric/None column '{col}' from pta DF output.")
            elif isinstance(result, pd.Series):
                 if result is not None and pd.api.types.is_numeric_dtype(result):
                      col_name = utils.get_config_identifier(indicator_name, config_id, None)
                      output_df[col_name] = result.reindex(data.index); computed_any = True
                      logger.debug(f"  -> Added pta Series output column: {col_name}")
                 else: logger.warning(f"PTA '{pta_func_name}' returned non-numeric/None Series. Skipping.")
            elif result is not None: logger.warning(f"PTA '{pta_func_name}' returned unexpected type after tuple processing: {type(result)}. Skipping.")

        # --- Custom Indicators ---
        elif indicator_type == 'custom':
            custom_func_name = indicator_def.get('function_name', f"compute_{indicator_name.lower()}")
            custom_func = getattr(custom_indicators, custom_func_name, None)
            if custom_func:
                sig = inspect.signature(custom_func)
                valid_params = {k:v for k,v in params.items() if k in sig.parameters}
                logger.debug(f"Calling custom func '{custom_func_name}' (Cfg ID: {config_id}) with params: {valid_params}")
                try:
                    result_df_custom = custom_func(data, **valid_params)
                    if result_df_custom is None: logger.warning(f"Custom func '{custom_func_name}' (Cfg {config_id}) returned None. Skipping."); return None
                    if not isinstance(result_df_custom, pd.DataFrame): logger.error(f"Custom func '{custom_func_name}' (Cfg {config_id}) returned {type(result_df_custom)}, expected DataFrame/None. Skipping."); return None
                    if result_df_custom.empty: logger.warning(f"Custom func '{custom_func_name}' (Cfg {config_id}) returned empty DataFrame."); return None

                    new_cols = list(result_df_custom.columns)
                    if new_cols:
                        logger.debug(f"Custom func '{custom_func_name}' returned cols: {new_cols}")
                        for new_col in new_cols:
                             if new_col in result_df_custom and pd.api.types.is_numeric_dtype(result_df_custom[new_col]):
                                 base_name_for_id = new_col if len(new_cols) > 1 else indicator_name
                                 final_col_name = utils.get_config_identifier(base_name_for_id, config_id, None if len(new_cols) == 1 else new_col)
                                 original_col_name = final_col_name; dup_counter = 1
                                 while final_col_name in output_df.columns: final_col_name = f"{original_col_name}_dup{dup_counter}"; dup_counter += 1
                                 output_df[final_col_name] = result_df_custom[new_col].reindex(data.index)
                                 computed_any = True; logger.debug(f"  -> Added custom indicator output column: {final_col_name}")
                             else: logger.warning(f"Custom func '{custom_func_name}' returned non-numeric/missing col '{new_col}'. Skipping.")
                except Exception as custom_e:
                    logger.error(f"Error executing custom func '{custom_func_name}' (Cfg {config_id}): {custom_e}", exc_info=True)
                    return None
            else:
                logger.error(f"Custom func '{custom_func_name}' not found for '{indicator_name}'. Skipping Cfg {config_id}.")
                return None

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
                 output_df[col] = output_df[col].astype(np.float64)
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
        # Log progress at INFO level, computation details at DEBUG level
        if (i + 1) % 100 == 0 or (i + 1) == len(configs_to_process):
            logger.info(f"Computing indicator {i+1}/{len(configs_to_process)}: {indicator_name} (Config ID: {config_id})")
        else:
             logger.debug(f"Computing indicator {i+1}/{len(configs_to_process)}: {indicator_name} (Config ID: {config_id})")

        indicator_df = _compute_single_indicator(data, config_details)

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
             # ----- CHANGE WARNING TO INFO -----
             logger.info(f"Failed/empty result for {indicator_name} (Config ID: {config_id}).")
             # ----- END CHANGE -----
             fail_count += 1

    logger.info(f"Individual computations done. Success: {success_count}, Failed/Skipped: {fail_count}.")

    if all_indicator_outputs:
        logger.info(f"Concatenating results from {len(all_indicator_outputs)} successful computations...")
        try:
            aligned_outputs = [df.reindex(data.index) for df in all_indicator_outputs]
            indicators_combined_df = pd.concat(aligned_outputs, axis=1)

            new_cols = indicators_combined_df.columns; existing_cols = data.columns
            overlap = new_cols.intersection(existing_cols)
            if not overlap.empty: logger.error(f"FATAL: Overlap between new/existing columns: {list(overlap)}"); raise ValueError("Indicator column name conflict.")
            dup_new_cols = new_cols[new_cols.duplicated(keep=False)].unique()
            if not dup_new_cols.empty: logger.error(f"FATAL: Duplicate indicator column names detected: {list(dup_new_cols)}"); raise ValueError("Duplicate indicator column names.")

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