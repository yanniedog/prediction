# parameter_generator.py
import logging
import itertools
from typing import Dict, List, Any, Optional
import numpy as np
import random
import json # For default comparison hashing if needed

# Import config for fallback value
import config as app_config
import utils # For compare_param_dicts

logger = logging.getLogger(__name__)

# --- Helper functions _evaluate_single_condition and evaluate_conditions ---
# (No changes needed in these helper functions)
def _evaluate_single_condition(param_value: Any, op: str, condition_value: Any) -> bool:
    """Evaluates a single condition (e.g., param_value >= condition_value)."""
    operator_map = { 'gt': '>', 'gte': '>=', 'lt': '<', 'lte': '<=', 'eq': '==', 'neq': '!=' }
    mapped_op = operator_map.get(op)
    if not mapped_op:
        logger.error(f"Unsupported operator '{op}' in conditions.")
        return False
    try:
        # Allow comparison if types are compatible or one is None
        can_compare = False
        if isinstance(param_value, type(condition_value)) or param_value is None or condition_value is None:
            can_compare = True
        # Allow comparing int and float
        elif isinstance(param_value, (int, float)) and isinstance(condition_value, (int, float)):
            can_compare = True

        if not can_compare:
             # Log type mismatch specifically if not a None comparison
             if param_value is not None and condition_value is not None:
                 logger.warning(f"Type mismatch comparing {param_value} ({type(param_value)}) {mapped_op} {condition_value} ({type(condition_value)}). Condition fails.")
             # Standard comparison logic handles None checks correctly below
             pass # Allow comparison logic to proceed for None cases

        # Perform comparison
        if mapped_op == '>': return param_value > condition_value
        if mapped_op == '>=': return param_value >= condition_value
        if mapped_op == '<': return param_value < condition_value
        if mapped_op == '<=': return param_value <= condition_value
        if mapped_op == '==': return param_value == condition_value
        if mapped_op == '!=': return param_value != condition_value
        return False # Should not be reached if operator is valid
    except TypeError:
        # This might catch comparisons like None > 5, which should be False
        # Let's log it for debugging but return False as it's usually an invalid comparison
        logger.debug(f"TypeError comparing {param_value} {mapped_op} {condition_value}. Condition fails.")
        return False
    except Exception as e:
        logger.error(f"Error evaluating condition ({param_value} {mapped_op} {condition_value}): {e}")
        return False

def evaluate_conditions(params: Dict[str, Any], conditions: List[Dict[str, Dict[str, Any]]]) -> bool:
    """Checks if a parameter combination satisfies all conditions."""
    if not conditions: return True
    for condition_group in conditions:
        if not isinstance(condition_group, dict) or not condition_group: continue # Skip invalid condition groups
        param_name, ops_dict = list(condition_group.items())[0]
        current_param_value = params.get(param_name) # Handle missing params gracefully

        if not isinstance(ops_dict, dict): continue # Skip invalid ops format

        for op, value_or_ref in ops_dict.items():
            compare_value = None; is_param_ref = False
            # Check if value_or_ref is a parameter name
            if isinstance(value_or_ref, str) and value_or_ref in params:
                compare_value = params[value_or_ref]; is_param_ref = True
            # Check if value_or_ref is a literal value (None, number, bool, string)
            elif value_or_ref is None or isinstance(value_or_ref, (int, float, bool, str)):
                 compare_value = value_or_ref
            else: # Invalid condition value
                logger.error(f"Condition value '{value_or_ref}' for '{param_name}' is invalid type {type(value_or_ref)}. Fails."); return False

            # If it was a reference, ensure the referenced param exists
            if is_param_ref and value_or_ref not in params:
                # Allow condition to pass if the referenced parameter *also* doesn't exist (e.g., comparing two optional params)
                # But fail if the current parameter exists and the reference doesn't.
                if current_param_value is not None:
                    logger.error(f"Condition references missing param '{value_or_ref}'. Fails."); return False
                else:
                    # If both are missing, the comparison depends on the operator (e.g., == might be true, > false)
                    # Let _evaluate_single_condition handle None comparisons
                    pass

            if not _evaluate_single_condition(current_param_value, op, compare_value):
                logger.debug(f"Condition failed: {params} -> {param_name} ({current_param_value}) {op} {value_or_ref} ({compare_value})")
                return False
    return True
# --- End of helper functions ---


def generate_configurations(
    indicator_definition: Dict[str, Any], # Accept the specific definition
) -> List[Dict[str, Any]]:
    """Generates valid parameter combinations using small ranges around defaults,
       respecting per-indicator range steps if specified."""

    parameter_definitions = indicator_definition.get('parameters', {})
    conditions = indicator_definition.get('conditions', [])

    # Determine range_steps: Use indicator-specific first, then global fallback
    range_steps_indicator = indicator_definition.get('range_steps_default')
    if isinstance(range_steps_indicator, int) and range_steps_indicator >= 1:
        range_steps = range_steps_indicator
        logger.debug(f"Using indicator-specific range_steps: {range_steps}")
    else:
        range_steps = app_config.DEFAULTS.get("default_param_range_steps", 3) # Use global fallback
        logger.debug(f"Using global default range_steps: {range_steps}")
        if range_steps < 1:
             logger.warning(f"Global default range_steps={range_steps} invalid, using 1.")
             range_steps = 1

    num_values = (2 * range_steps) + 1
    logger.info(f"Generating parameter ranges with {num_values} values per parameter (Range Steps: {range_steps}).")

    param_ranges = {}
    # Define standard step sizes (can be tuned)
    int_step = 1
    float_step_pct = 0.10 # 10% of default value as step, bounded
    min_float_step = 0.01

    # Define parameter categories for potentially different step logic
    period_params = ['fast', 'slow', 'fastperiod', 'slowperiod', 'signalperiod', 'timeperiod', 'timeperiod1', 'timeperiod2', 'timeperiod3', 'length', 'window', 'obv_period', 'price_period', 'fastk_period', 'slowk_period', 'slowd_period', 'fastd_period', 'tenkan', 'kijun', 'senkou']
    factor_params = ['fastlimit','slowlimit','acceleration','maximum','vfactor','smoothing']
    dev_scalar_params = ['scalar','nbdev','nbdevup','nbdevdn']

    for param, details in parameter_definitions.items():
        default = details.get('default')
        if default is None:
            logger.debug(f"Param '{param}' has no default, skipping range gen.")
            continue

        values = []
        p_min = details.get('min')
        p_max = details.get('max')

        if isinstance(default, int):
            # Determine min boundary, using 2 for most periods, 1 otherwise
            # Use JSON min if available and valid
            min_bound = 1
            if param in period_params and name.lower() not in ['mom', 'roc', 'rocp', 'rocr', 'rocr100', 'atr', 'natr', 'beta', 'correl', 'signalperiod', 'fastk_period', 'slowk_period', 'slowd_period', 'fastd_period', 'tenkan', 'kijun', 'senkou', 'timeperiod1', 'timeperiod2', 'timeperiod3']:
                 min_bound = 2
            if p_min is not None and isinstance(p_min, int):
                min_bound = max(min_bound, p_min)

            # Determine max boundary if specified in JSON
            max_bound = p_max if (p_max is not None and isinstance(p_max, int)) else float('inf')

            start = max(min_bound, default - range_steps * int_step)
            # Generate values and ensure they are within bounds
            generated = [min(max_bound, max(min_bound, start + i * int_step)) for i in range(num_values)]
            values = sorted(list(set(generated + [default]))) # Add default and unique sort
            # Ensure bounds didn't create invalid single value if range was tight
            values = [v for v in values if min_bound <= v <= max_bound]


        elif isinstance(default, float):
            # Use JSON min/max if available, otherwise sensible defaults
            min_bound = 0.0
            if param in factor_params: min_bound = 0.01
            elif param in dev_scalar_params: min_bound = 0.1
            if p_min is not None and isinstance(p_min, (int, float)):
                 min_bound = max(min_bound, float(p_min))

            max_bound = float('inf')
            if param in factor_params: max_bound = 1.0
            elif param in dev_scalar_params: max_bound = 5.0
            if p_max is not None and isinstance(p_max, (int, float)):
                 max_bound = min(max_bound, float(p_max))

            # Calculate step size relative to default, but bounded
            step = max(min_float_step, abs(default * float_step_pct))

            start = default - range_steps * step
            # Generate values, round, clip to bounds
            generated = [round(min(max_bound, max(min_bound, start + i * step)), 4) for i in range(num_values)]
            values = sorted(list(set(generated + [default]))) # Add default and unique sort
             # Ensure bounds didn't create invalid single value if range was tight
            values = [v for v in values if min_bound <= v <= max_bound]

        else: # Handle non-numeric (e.g., string, bool) - just use the default
            values = [default]

        if values:
            param_ranges[param] = values
            logger.debug(f"Range '{param}': {values}")
        elif default is not None:
             # If range generation failed but default exists, use default only
             param_ranges[param] = [default]
             logger.warning(f"Could not generate range for '{param}', using default only: {default}")


    if not param_ranges:
        logger.warning("No parameters found with defaults to generate configurations.")
        # Try to generate just the default combo if possible
        default_combo = {p: d.get('default') for p, d in parameter_definitions.items() if d.get('default') is not None}
        return [default_combo] if default_combo and evaluate_conditions(default_combo, conditions) else []

    # Generate combinations and check conditions
    keys = list(param_ranges.keys())
    all_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*param_ranges.values())]
    valid_combinations = [combo for combo in all_combinations if evaluate_conditions(combo, conditions)]

    logger.info(f"Generated {len(all_combinations)} raw combinations, {len(valid_combinations)} valid after conditions.")

    # Ensure default configuration is included if it's valid
    default_combo = {p: d.get('default') for p, d in parameter_definitions.items() if d.get('default') is not None}
    if default_combo and evaluate_conditions(default_combo, conditions):
        is_present = any(utils.compare_param_dicts(valid_combo, default_combo) for valid_combo in valid_combinations)
        if not is_present:
            valid_combinations.append(default_combo)
            logger.info("Default config was valid but missing, added back.")
    elif default_combo:
        logger.warning("Default parameter combination failed condition checks.")


    # Deduplicate using helper function (more robust than simple hashing)
    unique_valid_combinations = []
    seen_hashes = set() # Use hash for quick check
    for combo in valid_combinations:
        is_duplicate = False
        # Quick hash check first
        combo_tuple_sorted = tuple(sorted(combo.items())); combo_hash = hash(combo_tuple_sorted)
        if combo_hash in seen_hashes:
            # If hash matches, perform full comparison
            for existing in unique_valid_combinations:
                 if utils.compare_param_dicts(combo, existing):
                     is_duplicate = True
                     break
        if not is_duplicate:
            unique_valid_combinations.append(combo)
            seen_hashes.add(combo_hash) # Add hash only if combo is truly unique

    if len(unique_valid_combinations) != len(valid_combinations):
         logger.info(f"Removed {len(valid_combinations) - len(unique_valid_combinations)} duplicates.")

    logger.info(f"Returning {len(unique_valid_combinations)} unique valid configurations.")
    return unique_valid_combinations


# --- Random Config Generation (No changes needed here) ---
def _generate_random_valid_config(
    parameter_definitions: Dict[str, Dict[str, Any]],
    conditions: List[Dict[str, Dict[str, Any]]],
    max_tries: int = 50
) -> Optional[Dict[str, Any]]:
    """Attempts to generate a *random* valid parameter configuration."""
    attempt = 0
    period_params = ['fast', 'slow', 'fastperiod', 'slowperiod', 'signalperiod', 'timeperiod', 'timeperiod1', 'timeperiod2', 'timeperiod3', 'length', 'window', 'obv_period', 'price_period', 'fastk_period', 'slowk_period', 'slowd_period', 'fastd_period', 'tenkan', 'kijun', 'senkou']
    factor_params = ['fastlimit','slowlimit','acceleration','maximum','vfactor','smoothing']
    dev_scalar_params = ['scalar','nbdev','nbdevup','nbdevdn']

    while attempt < max_tries:
        attempt += 1
        candidate_params = {}; has_numeric = False
        for param, details in parameter_definitions.items():
            default = details.get('default'); min_val = details.get('min'); max_val = details.get('max')
            # Use default only if min/max are not specified for random generation range
            use_default_for_range = default is not None and (min_val is None or max_val is None)

            if isinstance(default, int) or (isinstance(min_val, int) and isinstance(max_val, int)):
                has_numeric = True
                # Define bounds: prefer min/max, fallback to range around default
                lower_b = min_val if min_val is not None else (max(1, int(default * 0.25)) if use_default_for_range else 1)
                upper_b = max_val if max_val is not None else (max(lower_b + 1, int(default * 3.0), 200) if use_default_for_range else 200)
                # Ensure min bound >= 1 (or 2 for periods)
                strict_min_2 = param in period_params and name.lower() not in ['mom', 'roc', 'rocp', 'rocr', 'rocr100', 'atr', 'natr', 'beta', 'correl', 'signalperiod', 'fastk_period', 'slowk_period', 'slowd_period', 'fastd_period', 'tenkan', 'kijun', 'senkou', 'timeperiod1', 'timeperiod2', 'timeperiod3']
                min_check = 2 if strict_min_2 else 1
                lower_b = max(min_check, lower_b)
                if lower_b >= upper_b: upper_b = lower_b + 1 # Ensure range > 0
                candidate_params[param] = random.randint(lower_b, upper_b)

            elif isinstance(default, float) or (isinstance(min_val, float) and isinstance(max_val, float)):
                has_numeric = True
                # Define bounds: prefer min/max, fallback range around default
                lower_b = min_val if min_val is not None else (max(0.0, round(default * 0.25, 4)) if use_default_for_range else 0.0)
                upper_b = max_val if max_val is not None else (max(lower_b + 0.01, round(default * 3.0, 4)) if use_default_for_range else 1.0)
                 # Apply specific bounds based on param type if min/max were not defined
                if min_val is None:
                     if param in factor_params: lower_b = max(lower_b, 0.01)
                     elif param in dev_scalar_params: lower_b = max(lower_b, 0.1)
                     elif param in period_params: lower_b = max(lower_b, 2.0)
                if max_val is None:
                     if param in factor_params: upper_b = min(upper_b, 1.0)
                     elif param in dev_scalar_params: upper_b = min(upper_b, 5.0)
                if lower_b >= upper_b: upper_b = lower_b + 0.01 # Ensure range > 0
                candidate_params[param] = round(random.uniform(lower_b, upper_b), 4)

            elif default is not None: # Handle non-numeric (bool, string)
                candidate_params[param] = default
            # else: Skip if no default and no min/max provided

        if not candidate_params: # Skip if no params could be determined
             attempt = max_tries; continue

        # Check if only default is possible (no numeric tunable params)
        if not has_numeric and attempt == 1:
            default_params_only = {k: v.get('default') for k, v in parameter_definitions.items() if 'default' in v}
            return default_params_only if evaluate_conditions(default_params_only, conditions) else None

        if evaluate_conditions(candidate_params, conditions):
            logger.debug(f"Generated random valid config (try {attempt}): {candidate_params}")
            return candidate_params

    logger.warning(f"Could not generate random valid config after {max_tries} tries.")
    # Fallback: Return default if it's valid
    default_params_fallback = {k: v.get('default') for k, v in parameter_definitions.items() if 'default' in v}
    if default_params_fallback and evaluate_conditions(default_params_fallback, conditions):
        logger.warning("Falling back to default parameters as random generation failed.")
        return default_params_fallback
    return None