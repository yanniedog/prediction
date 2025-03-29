# parameter_optimizer.py
import logging
import random
import json
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import sqlite3

import config as app_config
import indicator_factory
import correlation_calculator
import sqlite_manager
import utils
import parameter_generator

logger = logging.getLogger(__name__)

def _get_config_hash(params: Dict[str, Any]) -> str:
    """Generates a stable SHA256 hash for a parameter dictionary."""
    config_str = json.dumps(params, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(config_str.encode('utf-8')).hexdigest()

# --- UPDATED FUNCTION ---
def _calculate_config_performance_and_correlations(
    params: Dict[str, Any],
    config_id: int,
    indicator_name: str,
    base_data_with_required: pd.DataFrame,
    max_lag: int,
    scoring_method: str = 'max_abs'
) -> Tuple[Optional[float], Optional[int], List[Optional[float]]]:
    """
    Calculates indicator and correlations, returning performance score, best lag, AND all correlations.

    Returns:
        Tuple[Optional[float], Optional[int], List[Optional[float]]]: (score, best_score_lag, correlation_values)
        Score calculation depends on scoring_method. Returns (None, None, []) on failure.
        The returned correlation_values list corresponds to the FIRST output column.
        best_score_lag is the lag associated with the highest score found across all outputs
                        (only meaningful for max-based scoring methods).
    """
    logger.debug(f"Calculating performance for Config ID {config_id} ({indicator_name}, {params}) using '{scoring_method}' score")

    # 1. Compute Indicator
    config_details = {'indicator_name': indicator_name, 'params': params, 'config_id': config_id}
    indicator_output_df = indicator_factory._compute_single_indicator(base_data_with_required.copy(), config_details)

    if indicator_output_df is None or indicator_output_df.empty:
        logger.warning(f"Indicator computation failed or returned empty for Config ID {config_id}. Score: None.")
        return None, None, []
    nan_check = indicator_output_df.isnull().all()
    all_nan_cols = nan_check[nan_check].index.tolist()
    if all_nan_cols:
        logger.warning(f"Indicator {indicator_name} (Cfg {config_id}) produced all-NaN columns: {all_nan_cols}. Score: None.")
        return None, None, []

    # 2. Combine with 'close' and drop NaNs for correlation input
    if 'close' not in base_data_with_required.columns:
        logger.error("Base data missing 'close' column for correlation calculation.")
        return None, None, []
    data_for_corr = pd.concat([base_data_with_required[['close']], indicator_output_df.reindex(base_data_with_required.index)], axis=1)
    initial_len = len(data_for_corr)
    data_for_corr.dropna(inplace=True)
    dropped_rows = initial_len - len(data_for_corr)
    if dropped_rows > 0: logger.debug(f"Dropped {dropped_rows} rows with NaNs before correlation for Config ID {config_id}.")

    min_required_len = max_lag + 3
    if len(data_for_corr) < min_required_len:
        logger.warning(f"Insufficient data ({len(data_for_corr)} rows) for correlation (max_lag={max_lag}) for Config ID {config_id}. Need {min_required_len}. Score: None.")
        return None, None, []

    # 3. Calculate Correlations for each output column
    all_output_correlations = {}
    output_columns = list(indicator_output_df.columns)
    logger.debug(f"Config ID {config_id}: Calculating correlations for columns: {output_columns}")

    for indicator_col in output_columns:
        col_correlations = [np.nan] * max_lag
        if indicator_col not in data_for_corr.columns:
            logger.error(f"Internal error: Column '{indicator_col}' not in data_for_corr. Skipping.")
            continue
        if data_for_corr[indicator_col].nunique(dropna=True) <= 1:
            logger.warning(f"Indicator column '{indicator_col}' (Config ID {config_id}) has no variance. Setting correlations to NaN.")
            all_output_correlations[indicator_col] = [np.nan] * max_lag
            continue
        for lag in range(1, max_lag + 1):
            correlation_value = correlation_calculator.calculate_correlation_indicator_vs_future_price( data_for_corr, indicator_col, lag )
            db_value = np.nan if correlation_value is None else float(correlation_value)
            col_correlations[lag-1] = db_value
        all_output_correlations[indicator_col] = col_correlations

    if not all_output_correlations:
        logger.warning(f"Config ID {config_id}: No correlation results generated.")
        return None, None, []

    # 4. Calculate Score and Best Lag based on chosen method across ALL output columns
    final_score = None
    best_score_lag = None # Lag associated with the overall best score for this config
    current_max_score_for_config = -float('inf') # Track the highest score value found

    for col, corrs in all_output_correlations.items():
        corrs_array = np.array(corrs, dtype=float)
        if np.isnan(corrs_array).all(): continue

        col_score = None
        col_best_lag = None # Lag associated with the best score *for this column*
        valid_corrs_mask = ~np.isnan(corrs_array)
        valid_corrs_array = corrs_array[valid_corrs_mask]
        valid_lags = np.array(range(1, max_lag + 1))[valid_corrs_mask]

        if len(valid_corrs_array) == 0: continue

        # Calculate score based on method
        if scoring_method == 'max_abs':
             abs_corrs_array = np.abs(valid_corrs_array)
             col_score = np.max(abs_corrs_array)
             col_best_lag_index = np.argmax(abs_corrs_array)
             col_best_lag = valid_lags[col_best_lag_index]
        elif scoring_method == 'mean_abs':
             abs_corrs_array = np.abs(valid_corrs_array)
             col_score = np.mean(abs_corrs_array)
             col_best_lag = None # Lag is not meaningful for mean score
        elif scoring_method == 'max_positive':
             if np.any(valid_corrs_array > 0): # Check if any positive values exist
                col_score = np.max(valid_corrs_array)
                col_best_lag_index = np.argmax(valid_corrs_array)
                col_best_lag = valid_lags[col_best_lag_index]
             else:
                col_score = 0.0 # Or perhaps -inf if we strictly want positive
                col_best_lag = None
        elif scoring_method == 'mean_positive':
             positive_corrs = valid_corrs_array[valid_corrs_array > 0]
             col_score = np.mean(positive_corrs) if len(positive_corrs) > 0 else 0.0
             col_best_lag = None # Lag is not meaningful for mean score
        elif scoring_method == 'max_negative':
             if np.any(valid_corrs_array < 0): # Check if any negative values exist
                 min_val = np.min(valid_corrs_array)
                 col_score = -min_val # Score is positive
                 col_best_lag_index = np.argmin(valid_corrs_array)
                 col_best_lag = valid_lags[col_best_lag_index]
             else:
                 col_score = 0.0 # No negative correlations found
                 col_best_lag = None
        elif scoring_method == 'mean_negative':
             negative_corrs = valid_corrs_array[valid_corrs_array < 0]
             if len(negative_corrs) > 0:
                 mean_neg = np.mean(negative_corrs)
                 col_score = -mean_neg # Score is positive
             else:
                 col_score = 0.0 # No negative correlations found
             col_best_lag = None # Lag is not meaningful for mean score
        else: # Default to max_abs
             logger.warning(f"Unknown scoring_method '{scoring_method}'. Defaulting to 'max_abs'.")
             abs_corrs_array = np.abs(valid_corrs_array)
             if len(abs_corrs_array) > 0:
                col_score = np.max(abs_corrs_array)
                col_best_lag_index = np.argmax(abs_corrs_array)
                col_best_lag = valid_lags[col_best_lag_index]

        # Update the overall best score and associated lag for this configuration
        # (comparing scores across different output columns of the same config)
        if col_score is not None and col_score > current_max_score_for_config:
            current_max_score_for_config = col_score
            final_score = col_score
            best_score_lag = col_best_lag # Store the lag associated with this column's best score


    if final_score is None:
        logger.warning(f"Config ID {config_id}: No valid scores calculated across output columns.")
        best_score_lag = None # Ensure lag is None if score is None

    # Log the final determined score and lag for the configuration
    logger.debug(f"Config ID {config_id}: Final Score ({scoring_method}) = {final_score if final_score is not None else 'N/A'} at Best Lag = {best_score_lag if best_score_lag is not None else 'N/A'}")

    first_col_name = output_columns[0] if output_columns else None
    first_col_correlations = all_output_correlations.get(first_col_name, [np.nan] * max_lag)
    if len(first_col_correlations) < max_lag: first_col_correlations.extend([np.nan] * (max_lag - len(first_col_correlations)))
    elif len(first_col_correlations) > max_lag: first_col_correlations = first_col_correlations[:max_lag]

    return final_score, best_score_lag, first_col_correlations


def _generate_candidate(
    base_config: Optional[Dict[str, Any]],
    param_defs: Dict[str, Dict],
    conditions: List[Dict],
    indicator_name: str,
    evaluated_hashes: set
) -> Optional[Dict[str, Any]]:
    """
    Generates a new candidate configuration.
    If base_config is provided, it perturbs it.
    Otherwise, it calls _generate_random_valid_config for exploration.
    Ensures the generated config hash is not in evaluated_hashes.
    Returns None if it can't generate a new valid config after several tries.
    """
    max_tries = 30
    attempt = 0
    period_params = [
        'fast', 'slow', 'fastperiod', 'slowperiod', 'signalperiod',
        'timeperiod', 'timeperiod1', 'timeperiod2', 'timeperiod3',
        'length', 'window', 'obv_period', 'price_period',
        'fastk_period', 'slowk_period', 'slowd_period', 'fastd_period',
        'tenkan', 'kijun', 'senkou'
    ]
    factor_params = ['fastlimit','slowlimit','acceleration','maximum','vfactor','smoothing']
    dev_scalar_params = ['scalar','nbdev','nbdevup','nbdevdn']

    while attempt < max_tries:
        attempt += 1
        candidate_params = None
        log_prefix = f"Candidate Gen (Try {attempt}/{max_tries}) for {indicator_name}:"

        if base_config:
            # --- Perturbation Logic ---
            candidate_params = base_config.copy()
            tweakable_params = [p for p, d in param_defs.items() if isinstance(d.get('default'), (int, float))]

            if not tweakable_params:
                 logger.warning(f"{log_prefix} No numeric tweakable parameters found for perturbation.")
                 return None

            param_to_tweak = random.choice(tweakable_params)
            details = param_defs[param_to_tweak]
            current_value = candidate_params[param_to_tweak]

            if isinstance(current_value, int):
                min_val = 2 if param_to_tweak in period_params else 1
                step_type = random.choice(['step', 'gauss'])
                if step_type == 'step':
                    step_size = max(1, int(current_value * 0.1))
                    change = random.choice([-step_size, step_size])
                else: # gauss
                    sigma = max(1, int(current_value * 0.15))
                    change = int(round(random.gauss(0, sigma)))

                new_value = max(min_val, current_value + change)
                if new_value == current_value and change != 0: new_value = max(min_val, current_value + (1 if change > 0 else -1))
                elif new_value == current_value: new_value = max(min_val, current_value + random.choice([-1,1]))
                candidate_params[param_to_tweak] = new_value

            elif isinstance(current_value, float):
                min_val = 0.01 if param_to_tweak in factor_params else (0.1 if param_to_tweak in dev_scalar_params else 0.0)
                sigma = max(0.01, abs(current_value * 0.15))
                change = round(random.gauss(0, sigma), 4)
                new_value = round(max(min_val, current_value + change), 4)
                if np.isclose(new_value, current_value) and not np.isclose(change, 0): new_value = round(max(min_val, current_value + (0.01 if change > 0 else -0.01)), 4)
                elif np.isclose(new_value, current_value): new_value = round(max(min_val, current_value + random.choice([-0.01, 0.01])), 4)
                candidate_params[param_to_tweak] = new_value
            else:
                 logger.warning(f"{log_prefix} Cannot perturb non-numeric param '{param_to_tweak}'. Trying again.")
                 base_config = None
                 continue

            logger.debug(f"{log_prefix} Perturbed '{param_to_tweak}' from {base_config.get(param_to_tweak)} to {candidate_params[param_to_tweak]}")

        else:
            # --- Random Generation Logic ---
            logger.debug(f"{log_prefix} Generating random candidate using parameter_generator.")
            candidate_params = parameter_generator._generate_random_valid_config(param_defs, conditions)
            if candidate_params is None:
                logger.warning(f"{log_prefix} Failed to generate a random valid candidate via helper.")
                continue


        # --- Validation and Uniqueness Check ---
        if candidate_params is None: continue

        candidate_hash = _get_config_hash(candidate_params)
        is_valid = parameter_generator.evaluate_conditions(candidate_params, conditions) # Always re-validate
        is_new = candidate_hash not in evaluated_hashes

        if is_valid and is_new:
            logger.debug(f"{log_prefix} Found new valid candidate: {candidate_params}")
            return candidate_params
        elif not is_valid:
             logger.debug(f"{log_prefix} Generated candidate {candidate_params} is invalid. Retrying.")
             base_config = None # Force random if invalid
        elif not is_new:
             logger.debug(f"{log_prefix} Candidate {candidate_params} hash {candidate_hash[:8]}... already evaluated. Retrying.")
             base_config = None # Force random if hash collision

    logger.warning(f"Could not generate a new, valid, unique configuration for {indicator_name} after {max_tries} tries.")
    return None


def optimize_parameters(
    indicator_name: str,
    indicator_def: Dict,
    base_data_with_required: pd.DataFrame,
    max_lag: int,
    num_iterations: int,
    target_configs: int,
    db_path: str,
    symbol_id: int,
    timeframe_id: int,
    scoring_method: str = 'max_abs'
) -> List[Dict[str, Any]]:
    """
    Optimizes parameters for a single indicator using an iterative feedback loop.
    Aims to find configurations maximizing the chosen scoring metric.

    Returns:
        List of dictionaries for the top 'target_configs', sorted by score (desc).
        Includes default config result if evaluation was successful.
        Format: [{'indicator_name': ..., 'params': ..., 'config_id': ..., 'score': ..., 'correlations': [...]}, ...]
    """
    logger.info(f"--- Starting Parameter Optimization for: {indicator_name} ---")
    logger.info(f"Iterations: {num_iterations}, Target Configs: {target_configs}, Max Lag: {max_lag}, Scoring: {scoring_method}")

    param_defs = indicator_def.get('parameters', {})
    conditions = indicator_def.get('conditions', [])

    evaluated_configs_results: List[Dict[str, Any]] = []
    evaluated_hashes: set = set()
    correlations_to_batch_insert: List[Tuple[int, int, int, int, Optional[float]]] = []
    default_config_result: Optional[Dict[str, Any]] = None

    # --- 1. Evaluate Default Configuration ---
    default_params = {k: v.get('default') for k, v in param_defs.items() if 'default' in v}
    if not default_params: logger.error(f"Indicator {indicator_name} has no default parameters. Cannot optimize."); return []
    if not parameter_generator.evaluate_conditions(default_params, conditions): logger.error(f"Default parameters for {indicator_name} are invalid. Cannot optimize."); return []

    logger.info(f"Evaluating default config: {default_params}")
    conn = sqlite_manager.create_connection(db_path)
    if not conn: return []
    try: default_config_id = sqlite_manager.get_or_create_indicator_config_id(conn, indicator_name, default_params)
    except Exception as e: logger.error(f"Failed to get/create config ID for default params: {e}", exc_info=True); conn.close(); return []
    finally: conn.close()

    default_hash = _get_config_hash(default_params)
    evaluated_hashes.add(default_hash)

    default_score, default_best_lag, default_corrs = _calculate_config_performance_and_correlations(
        default_params, default_config_id, indicator_name, base_data_with_required, max_lag, scoring_method
    )

    best_config_params = default_params
    best_score = -float('inf')
    best_score_overall_lag = None

    if default_score is not None:
        default_config_result = {
            'params': default_params, 'hash': default_hash, 'config_id': default_config_id,
            'score': default_score, 'correlations': default_corrs, 'best_lag': default_best_lag
        }
        evaluated_configs_results.append(default_config_result)
        best_score = default_score
        best_score_overall_lag = default_best_lag
        logger.info(f"Default config score: {best_score:.4f} (Best Lag: {best_score_overall_lag})")
        for lag_idx, corr_val in enumerate(default_corrs):
             correlations_to_batch_insert.append((symbol_id, timeframe_id, default_config_id, lag_idx + 1, corr_val))
    else:
        logger.warning("Default configuration failed evaluation. Starting optimization from scratch/random.")
        best_config_params = None


    # --- 2. Iterative Optimization Loop ---
    successful_evaluations = 1 if default_score is not None else 0
    iteration = 0

    while successful_evaluations < num_iterations:
        iteration += 1
        logger.debug(f"\n--- Optimizer Iteration {iteration} (Evaluations: {successful_evaluations}/{num_iterations}) ---")

        explore = random.random() < app_config.OPTIMIZER_RANDOM_EXPLORE_PROB
        base_for_candidate = None
        generation_method = "random"
        if not explore and best_config_params is not None:
            base_for_candidate = best_config_params
            generation_method = "perturbation"
        logger.debug(f"Generation method: {generation_method}")

        candidates_generated_this_iter = 0
        while candidates_generated_this_iter < app_config.OPTIMIZER_CANDIDATES_PER_ITERATION:
             if successful_evaluations >= num_iterations: break

             candidate_params = _generate_candidate(
                 base_for_candidate, param_defs, conditions, indicator_name, evaluated_hashes
             )

             if candidate_params is None:
                 logger.warning("Failed to generate a new valid candidate in this attempt.")
                 if generation_method == "perturbation":
                      base_for_candidate = None; generation_method = "random (fallback)"
                      logger.debug("Switching to random generation for next attempt in iteration.")
                      continue
                 else: logger.error("Failed to generate even a 'random' valid candidate. Breaking inner loop."); break

             candidate_hash = _get_config_hash(candidate_params)
             evaluated_hashes.add(candidate_hash)
             candidates_generated_this_iter += 1

             conn = sqlite_manager.create_connection(db_path)
             if not conn: continue
             try: config_id = sqlite_manager.get_or_create_indicator_config_id(conn, indicator_name, candidate_params)
             except Exception as e: logger.error(f"Failed to get/create config ID for candidate {candidate_params}: {e}"); conn.close(); continue
             finally: conn.close()

             score, best_lag, correlations = _calculate_config_performance_and_correlations(
                 candidate_params, config_id, indicator_name, base_data_with_required, max_lag, scoring_method
             )

             if score is not None:
                 logger.info(f"Iter {iteration}: Evaluated Config ID {config_id} ({candidate_params}). Score: {score:.4f} (Best Lag: {best_lag})")
                 evaluated_configs_results.append({
                     'params': candidate_params, 'hash': candidate_hash, 'config_id': config_id,
                     'score': score, 'correlations': correlations, 'best_lag': best_lag
                 })
                 successful_evaluations += 1

                 for lag_idx, corr_val in enumerate(correlations):
                     correlations_to_batch_insert.append((symbol_id, timeframe_id, config_id, lag_idx + 1, corr_val))

                 if score > best_score:
                     logger.info(f"***** New best score found: {score:.4f} at Lag {best_lag} (Config ID {config_id}) *****")
                     best_score = score
                     best_score_overall_lag = best_lag
                     best_config_params = candidate_params
             else:
                 logger.warning(f"Iter {iteration}: Evaluation failed for Config ID {config_id} ({candidate_params}).")

             if successful_evaluations >= num_iterations: break

        if successful_evaluations >= num_iterations:
            logger.info(f"Reached target number of {num_iterations} successful evaluations during iteration {iteration}.")
            break
        if candidates_generated_this_iter < app_config.OPTIMIZER_CANDIDATES_PER_ITERATION and generation_method == "random (fallback)":
             logger.warning(f"Could not generate sufficient candidates in iteration {iteration}.")

        if iteration >= num_iterations * 3:
             logger.warning(f"Optimizer potentially stuck after {iteration} iterations with only {successful_evaluations} successful evals. Stopping early.")
             break


    # --- 3. Batch Insert Correlations ---
    if correlations_to_batch_insert:
        logger.info(f"Attempting to batch insert correlations for {len(correlations_to_batch_insert)} records...")
        conn = sqlite_manager.create_connection(db_path)
        if conn:
             try:
                 success = sqlite_manager.batch_insert_correlations(conn, correlations_to_batch_insert)
                 if not success: logger.error("Batch insertion of correlations failed.")
             except Exception as db_err: logger.error(f"Error during batch correlation insert call: {db_err}", exc_info=True)
             finally: conn.close()
        else: logger.error("Could not connect to DB for batch correlation insert.")
    else: logger.info("No new correlations generated to batch insert.")


    # --- 4. Select Top Configurations ---
    if not evaluated_configs_results:
        logger.error(f"No configurations were successfully evaluated for {indicator_name}.")
        return []

    evaluated_configs_results.sort(key=lambda x: x.get('score', -float('inf')), reverse=True)

    top_configs_unique = []
    seen_hashes_top = set()
    for cfg in evaluated_configs_results:
        if len(top_configs_unique) >= target_configs: break
        if cfg['hash'] not in seen_hashes_top:
             top_configs_unique.append(cfg)
             seen_hashes_top.add(cfg['hash'])

    final_top_configs = top_configs_unique
    if default_config_result:
        default_in_top = any(cfg['config_id'] == default_config_result['config_id'] for cfg in final_top_configs)
        if not default_in_top:
            if len(final_top_configs) < target_configs:
                final_top_configs.append(default_config_result)
                logger.info(f"Added default config (ID: {default_config_id}, Score: {default_score:.4f}) to results as it wasn't in top {target_configs} and space was available.")
            elif final_top_configs:
                worst_score_in_top = final_top_configs[-1]['score'] if final_top_configs[-1].get('score') is not None else -float('inf')
                # Check score before replacing
                if default_config_result.get('score', -float('inf')) >= worst_score_in_top :
                    logger.warning(f"Replacing worst config (ID: {final_top_configs[-1]['config_id']}, Score: {final_top_configs[-1].get('score', 'N/A'):.4f}) with default (ID: {default_config_id}, Score: {default_score:.4f}) to ensure it's included in top {target_configs}.")
                    final_top_configs[-1] = default_config_result
                else: logger.info(f"Default config (Score: {default_score:.4f}) was not among top {target_configs} and its score was worse than the lowest ({worst_score_in_top:.4f}). Not included in final list.")
            # Re-sort only if default was added or replaced
            final_top_configs.sort(key=lambda x: x.get('score', -float('inf')), reverse=True)


    logger.info(f"--- Optimization Finished for {indicator_name} ---")
    logger.info(f"Evaluated {len(evaluated_hashes)} unique configurations ({successful_evaluations} successful evaluations).")
    # Log lag associated with the actual best score found
    best_overall_result = final_top_configs[0] if final_top_configs else None
    best_score_overall_lag_final = best_overall_result.get('best_lag') if best_overall_result else None
    logger.info(f"Best score found ({scoring_method}): {best_score:.4f} (Associated Lag: {best_score_overall_lag_final if best_score_overall_lag_final is not None else 'N/A'})")
    logger.info(f"Returning final top {len(final_top_configs)} configurations (incl. default if evaluated).")

    results_for_main = []
    for result_config in final_top_configs:
        params = result_config.get('params', {})
        config_id = result_config.get('config_id', -1)
        score = result_config.get('score', None)
        corrs = result_config.get('correlations', [])
        results_for_main.append({
            'indicator_name': indicator_name, 'params': params, 'config_id': config_id,
            'score': score, 'correlations': corrs
        })
        if config_id == -1: logger.warning(f"Result config missing 'config_id': {result_config}")

    return results_for_main