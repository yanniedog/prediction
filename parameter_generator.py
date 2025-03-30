# parameter_generator.py
import logging
import itertools
from typing import Dict, List, Any, Optional
import numpy as np
import random
import json # For default comparison hashing if needed
import utils # For compare_param_dicts

logger = logging.getLogger(__name__)

# --- Helper functions _evaluate_single_condition and evaluate_conditions ---
def _evaluate_single_condition(param_value: Any, op: str, condition_value: Any) -> bool:
    """Evaluates a single condition (e.g., param_value >= condition_value)."""
    operator_map = { 'gt': '>', 'gte': '>=', 'lt': '<', 'lte': '<=', 'eq': '==', 'neq': '!=' }
    mapped_op = operator_map.get(op)
    if not mapped_op:
        logger.error(f"Unsupported operator '{op}' in conditions.")
        return False
    try:
        if mapped_op == '>': return param_value > condition_value
        if mapped_op == '>=': return param_value >= condition_value
        if mapped_op == '<': return param_value < condition_value
        if mapped_op == '<=': return param_value <= condition_value
        if mapped_op == '==': return param_value == condition_value
        if mapped_op == '!=': return param_value != condition_value
        return False
    except TypeError:
        if condition_value is None: # Allow comparison with None
            if mapped_op == '==': return param_value is None
            if mapped_op == '!=': return param_value is not None
        logger.warning(f"Type error comparing {param_value} {mapped_op} {condition_value}. Condition fails.")
        return False
    except Exception as e:
        logger.error(f"Error evaluating condition ({param_value} {mapped_op} {condition_value}): {e}")
        return False

def evaluate_conditions(params: Dict[str, Any], conditions: List[Dict[str, Dict[str, Any]]]) -> bool:
    """Checks if a parameter combination satisfies all conditions."""
    if not conditions: return True
    for condition_group in conditions:
        param_name, ops_dict = list(condition_group.items())[0]
        current_param_value = params.get(param_name) # Handle missing params gracefully

        for op, value_or_ref in ops_dict.items():
            compare_value = None; is_param_ref = False
            if isinstance(value_or_ref, str) and value_or_ref in params:
                compare_value = params[value_or_ref]; is_param_ref = True
            elif value_or_ref is None or isinstance(value_or_ref, (int, float, bool, str)):
                 compare_value = value_or_ref
            else: logger.error(f"Condition value '{value_or_ref}' for '{param_name}' invalid. Fails."); return False
            if is_param_ref and value_or_ref not in params:
                logger.error(f"Condition references missing param '{value_or_ref}'. Fails."); return False
            if not _evaluate_single_condition(current_param_value, op, compare_value):
                logger.debug(f"Condition failed: {params} -> {param_name} ({current_param_value}) {op} {value_or_ref} ({compare_value})")
                return False
    return True
# --- End of helper functions ---

def generate_configurations(
    parameter_definitions: Dict[str, Dict[str, Any]],
    conditions: List[Dict[str, Dict[str, Any]]],
    range_steps: int = 2
) -> List[Dict[str, Any]]:
    """Generates valid parameter combinations using small ranges around defaults."""
    if range_steps < 1: logger.warning(f"range_steps={range_steps} invalid, using 1."); range_steps = 1
    num_values = (2 * range_steps) + 1
    logger.info(f"Generating parameter ranges with {num_values} values per parameter.")

    param_ranges = {}
    int_step = 1
    period_params = ['fast', 'slow', 'fastperiod', 'slowperiod', 'signalperiod', 'timeperiod', 'timeperiod1', 'timeperiod2', 'timeperiod3', 'length', 'window', 'obv_period', 'price_period', 'fastk_period', 'slowk_period', 'slowd_period', 'fastd_period', 'tenkan', 'kijun', 'senkou']
    factor_params = ['fastlimit','slowlimit','acceleration','maximum','vfactor','smoothing']
    dev_scalar_params = ['scalar','nbdev','nbdevup','nbdevdn']

    for param, details in parameter_definitions.items():
        default = details.get('default')
        if default is None: logger.debug(f"Param '{param}' has no default, skipping range gen."); continue

        values = []
        if isinstance(default, int):
            min_val = 2 if param in period_params else 1
            start = max(min_val, default - range_steps * int_step)
            generated = [max(min_val, start + i * int_step) for i in range(num_values)]
            values = sorted(list(set(generated + [default])))
        elif isinstance(default, float):
            min_val = 0.0; step = 0.1; max_val = float('inf') # Generic defaults
            if param in factor_params: min_val = 0.01; step = 0.02; max_val = 1.0
            elif param in dev_scalar_params: min_val = 0.1; step = 0.2 if default >= 1 else 0.1; max_val = 5.0
            elif param in period_params: min_val = 2.0; step = 1.0; max_val = 200.0 # Treat like ints
            else: step = max(0.01, abs(default * 0.1)); max_val = abs(default * 3.0) # Relative step
            start = round(default - range_steps * step, 4)
            generated = [round(min(max_val, max(min_val, start + i * step)), 4) for i in range(num_values)]
            values = sorted(list(set(generated + [default])))
        else: values = [default] # Non-numeric

        if values: param_ranges[param] = values; logger.debug(f"Range '{param}': {values}")

    if not param_ranges:
        logger.warning("No parameters found with defaults to generate configurations.")
        default_combo = {p: d.get('default') for p, d in parameter_definitions.items() if d.get('default') is not None}
        return [default_combo] if default_combo and evaluate_conditions(default_combo, conditions) else []

    keys = list(param_ranges.keys())
    all_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*param_ranges.values())]
    valid_combinations = [combo for combo in all_combinations if evaluate_conditions(combo, conditions)]
    logger.info(f"Generated {len(all_combinations)} raw combinations, {len(valid_combinations)} valid after conditions.")

    # Ensure default included if valid
    default_combo = {p: d.get('default') for p, d in parameter_definitions.items() if d.get('default') is not None}
    if default_combo and evaluate_conditions(default_combo, conditions):
        is_present = any(utils.compare_param_dicts(valid_combo, default_combo) for valid_combo in valid_combinations)
        if not is_present: valid_combinations.append(default_combo); logger.info("Default config added back.")

    # Deduplicate using helper
    unique_valid_combinations = []
    seen_hashes = set()
    for combo in valid_combinations:
        is_duplicate = False
        # Simplified hash check first
        combo_tuple_sorted = tuple(sorted(combo.items())); combo_hash = hash(combo_tuple_sorted)
        if combo_hash in seen_hashes:
            # Full comparison if hash collision or actual duplicate
            for existing in unique_valid_combinations:
                 if utils.compare_param_dicts(combo, existing): is_duplicate = True; break
        if not is_duplicate:
            unique_valid_combinations.append(combo); seen_hashes.add(combo_hash)

    if len(unique_valid_combinations) != len(valid_combinations):
         logger.info(f"Removed {len(valid_combinations) - len(unique_valid_combinations)} duplicates.")

    logger.info(f"Returning {len(unique_valid_combinations)} unique valid configurations.")
    return unique_valid_combinations

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
            if default is None: continue
            if isinstance(default, int):
                has_numeric = True; min_b = min_val if min_val is not None else (2 if param in period_params else 1)
                upper_b = max_val if max_val is not None else max(min_b + 1, int(default * 3.0), 200)
                lower_b = max(min_b, int(default * 0.25))
                if lower_b >= upper_b: lower_b = min_b
                candidate_params[param] = random.randint(lower_b, upper_b)
            elif isinstance(default, float):
                has_numeric = True; min_b = 0.0; upper_b = default * 3.0
                if param in factor_params: min_b = min_val if min_val is not None else 0.01; upper_b = max_val if max_val is not None else 1.0
                elif param in dev_scalar_params: min_b = min_val if min_val is not None else 0.1; upper_b = max_val if max_val is not None else 5.0
                else: min_b = min_val if min_val is not None else 0.0; upper_b = max_val if max_val is not None else max(min_b + 0.1, default * 3.0)
                lower_b = max(min_b, round(default * 0.25, 4)); upper_b = min(upper_b, round(default * 3.0, 4))
                if lower_b >= upper_b: upper_b = max(lower_b + 0.01, upper_b) # Ensure range has width
                candidate_params[param] = round(random.uniform(lower_b, upper_b), 4)
            else: candidate_params[param] = default # Handle non-numeric

        if not has_numeric and attempt == 1: # Only default possible
            default_params = {k: v.get('default') for k, v in parameter_definitions.items() if 'default' in v}
            return default_params if evaluate_conditions(default_params, conditions) else None

        if evaluate_conditions(candidate_params, conditions):
            logger.debug(f"Generated random valid config (try {attempt}): {candidate_params}")
            return candidate_params

    logger.warning(f"Could not generate random valid config after {max_tries} tries.")
    return None
