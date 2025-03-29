# parameter_generator.py
import logging
import itertools
from typing import Dict, List, Any, Optional
import numpy as np
import random # Import random

logger = logging.getLogger(__name__)

# --- Helper functions _evaluate_single_condition and evaluate_conditions remain unchanged ---
def _evaluate_single_condition(param_value: Any, op: str, condition_value: Any) -> bool:
    """Evaluates a single condition (e.g., param_value >= condition_value)."""
    operator_map = { 'gt': '>', 'gte': '>=', 'lt': '<', 'lte': '<=', 'eq': '==', 'neq': '!=' }
    mapped_op = operator_map.get(op)
    if not mapped_op:
        logger.error(f"Unsupported operator '{op}' in conditions.")
        return False
    try:
        # Use explicit checks instead of eval
        if mapped_op == '>': return param_value > condition_value
        if mapped_op == '>=': return param_value >= condition_value
        if mapped_op == '<': return param_value < condition_value
        if mapped_op == '<=': return param_value <= condition_value
        if mapped_op == '==': return param_value == condition_value
        if mapped_op == '!=': return param_value != condition_value
        return False
    except TypeError:
        # Allow comparison with None if explicitly specified
        if condition_value is None:
            if mapped_op == '==': return param_value is None
            if mapped_op == '!=': return param_value is not None
        logger.warning(f"Type error comparing {param_value} {mapped_op} {condition_value}. Condition fails.")
        return False
    except Exception as e:
        logger.error(f"Error evaluating condition ({param_value} {mapped_op} {condition_value}): {e}")
        return False

def evaluate_conditions(params: Dict[str, Any], conditions: List[Dict[str, Dict[str, Any]]]) -> bool:
    """Checks if a parameter combination satisfies all conditions."""
    if not conditions:
        return True

    for condition_group in conditions:
        param_name, ops_dict = list(condition_group.items())[0]

        if param_name not in params:
            # Allow conditions on params that might not be present (e.g., optional params)
            # If the condition requires the param to exist (e.g., > 0), it should fail naturally.
            # If the condition is == None, it should pass if the param is missing.
            # Let _evaluate_single_condition handle it based on the operator and value.
            pass # Don't immediately fail if param is missing

        current_param_value = params.get(param_name) # Use .get() to handle missing params gracefully

        for op, value_or_ref in ops_dict.items():
            compare_value = None
            is_param_ref = False
            if isinstance(value_or_ref, str) and value_or_ref in params:
                compare_value = params[value_or_ref]
                is_param_ref = True
            # Check it's a literal type we can compare OR None
            elif value_or_ref is None or isinstance(value_or_ref, (int, float, bool, str)):
                 compare_value = value_or_ref
            else:
                 logger.error(f"Condition value '{value_or_ref}' for param '{param_name}' is not a valid literal or parameter reference. Condition fails.")
                 return False

            # If it's a parameter reference, ensure the referenced parameter exists
            if is_param_ref and value_or_ref not in params:
                logger.error(f"Condition references parameter '{value_or_ref}' which is not present in {params}. Condition fails.")
                return False

            if not _evaluate_single_condition(current_param_value, op, compare_value):
                # Log the failing condition details
                logger.debug(f"Condition failed: {params} -> {param_name} ({current_param_value}) {op} {value_or_ref} ({compare_value})")
                return False

    return True
# --- End of unchanged helper functions ---


def generate_configurations(
    parameter_definitions: Dict[str, Dict[str, Any]],
    conditions: List[Dict[str, Dict[str, Any]]],
    range_steps: int = 2 # Default: 2 steps each side -> 5 values total (e.g., D-2, D-1, D, D+1, D+2)
) -> List[Dict[str, Any]]:
    """
    Generates valid parameter combinations based on definitions and conditions.
    The width of the generated ranges is controlled by `range_steps`.
    `range_steps` = 1 -> 3 values (e.g., D-1, D, D+1)
    `range_steps` = 2 -> 5 values (e.g., D-2, D-1, D, D+1, D+2)
    `range_steps` = 4 -> 9 values
    """
    if range_steps < 1:
        logger.warning(f"range_steps was {range_steps}, must be >= 1. Setting to 1.")
        range_steps = 1

    num_values_to_generate = (2 * range_steps) + 1 # Total number of values including default
    logger.info(f"Generating parameter ranges with range_steps={range_steps} ({num_values_to_generate} values per parameter).")

    param_ranges = {}
    int_step = 1
    # Updated list of known period-like parameters
    period_params = [
        'fast', 'slow', 'fastperiod', 'slowperiod', 'signalperiod',
        'timeperiod', 'timeperiod1', 'timeperiod2', 'timeperiod3',
        'length', 'window', 'obv_period', 'price_period',
        'fastk_period', 'slowk_period', 'slowd_period', 'fastd_period',
        'tenkan', 'kijun', 'senkou' # Added from Ichimoku
    ]
    # Parameters typically bounded between 0 and 1 (or small values)
    factor_params = ['fastlimit','slowlimit','acceleration','maximum','vfactor','smoothing']
    # Parameters typically > 0 but not strictly periods
    dev_scalar_params = ['scalar','nbdev','nbdevup','nbdevdn']

    for param, details in parameter_definitions.items():
        default = details.get('default')
        if default is None:
            logger.debug(f"Parameter '{param}' has no default. Skipping range generation.")
            # If no default, should it still be included with a single value?
            # For now, skip, assuming parameters need defaults to be varied this way.
            continue

        values = []
        if isinstance(default, int):
            min_val = 2 if param in period_params else 1
            start = max(min_val, default - range_steps * int_step)
            generated_values = [max(min_val, start + i * int_step) for i in range(num_values_to_generate)]
            values = sorted(list(set(generated_values + [default])))

        elif isinstance(default, float):
            if param in factor_params: # Small fractional floats (e.g., 0 to 1)
                step = 0.02
                min_val = 0.01
                max_val = 1.0 # Common upper bound for factors like vfactor
                start = max(min_val, round(default - range_steps * step, 4))
                generated_values = [round(min(max_val, max(min_val, start + i * step)), 4) for i in range(num_values_to_generate)]
                values = sorted(list(set(generated_values + [default])))
            elif param in dev_scalar_params: # Deviations, scalars (typically >= 0.1 or 1.0)
                 step = 0.2 if default >= 1 else 0.1
                 min_val = 0.1
                 start = round(default - range_steps * step, 2)
                 generated_values = [round(max(min_val, start + i * step), 2) for i in range(num_values_to_generate)]
                 values = sorted(list(set(generated_values + [default])))
            elif param in period_params: # Treat float periods like ints
                 min_val = 2.0
                 step = 1.0
                 start = max(min_val, default - range_steps * step)
                 generated_values = [round(max(min_val, start + i * step), 1) for i in range(num_values_to_generate)]
                 values = sorted(list(set(generated_values + [default])))
            else: # Other floats - generic approach
                 step = max(0.01, abs(default * 0.1)) # Relative step
                 min_val = 0.0
                 start = round(default - range_steps * step, 4)
                 generated_values = [round(max(min_val, start + i * step), 4) for i in range(num_values_to_generate)]
                 values = sorted(list(set(generated_values + [default])))

        else: # Non-numeric or complex defaults (e.g., strings like 'ema')
            # Consider allowing lists of options in json definition later
            values = [default]

        if values:
             param_ranges[param] = values
             logger.debug(f"Generated range for '{param}': {values}")

    if not param_ranges:
        logger.warning("No parameters found with defaults to generate configurations.")
        # Return default config if it exists and is valid
        default_combo = {p: d.get('default') for p, d in parameter_definitions.items() if d.get('default') is not None}
        if default_combo and evaluate_conditions(default_combo, conditions):
            return [default_combo]
        return [] # Return empty list if no parameters or default is invalid


    keys = list(param_ranges.keys())
    all_combinations_tuples = list(itertools.product(*param_ranges.values()))
    all_combinations = [dict(zip(keys, combo)) for combo in all_combinations_tuples]

    valid_combinations = [combo for combo in all_combinations if evaluate_conditions(combo, conditions)]

    logger.info(f"Generated {len(all_combinations)} total raw combinations, {len(valid_combinations)} valid combinations after applying conditions.")

    # Ensure default is included if valid
    default_combo = {p: d.get('default') for p, d in parameter_definitions.items() if d.get('default') is not None}
    if default_combo and evaluate_conditions(default_combo, conditions):
        is_present = any(utils.compare_param_dicts(valid_combo, default_combo) for valid_combo in valid_combinations)
        if not is_present:
             valid_combinations.append(default_combo)
             logger.info("Default configuration was valid but missing, added it back.")

    # Ensure uniqueness using helper function for float comparison
    unique_valid_combinations = []
    seen_hashes = set() # Use hashes for faster lookup of exact matches initially
    for combo in valid_combinations:
        combo_tuple_sorted = tuple(sorted(combo.items())) # Sort for consistent hashing
        combo_hash = hash(combo_tuple_sorted) # Basic hash for quick check

        is_duplicate = False
        if combo_hash in seen_hashes:
            # Potential hash collision or actual duplicate, do full comparison
            for existing_combo in unique_valid_combinations:
                 if utils.compare_param_dicts(combo, existing_combo):
                      is_duplicate = True
                      break
        if not is_duplicate:
            unique_valid_combinations.append(combo)
            seen_hashes.add(combo_hash)


    if len(unique_valid_combinations) != len(valid_combinations):
         logger.info(f"Removed {len(valid_combinations) - len(unique_valid_combinations)} duplicate configurations.")

    logger.info(f"Returning {len(unique_valid_combinations)} unique valid configurations.")
    return unique_valid_combinations

# >>> NEW FUNCTION <<<
def _generate_random_valid_config(
    parameter_definitions: Dict[str, Dict[str, Any]],
    conditions: List[Dict[str, Dict[str, Any]]],
    max_tries: int = 50
) -> Optional[Dict[str, Any]]:
    """
    Attempts to generate a *random* valid parameter configuration.

    Tries up to max_tries times to find a combination that satisfies conditions.
    Returns None if no valid combination is found.
    """
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
        candidate_params = {}
        has_numeric = False # Track if indicator has numeric params

        for param, details in parameter_definitions.items():
            default = details.get('default')
            # Determine reasonable min/max bounds for random generation
            min_val = details.get('min', None) # Allow defining min/max in json later
            max_val = details.get('max', None)

            if default is None: continue # Skip params without defaults for now

            if isinstance(default, int):
                has_numeric = True
                min_val = min_val if min_val is not None else (2 if param in period_params else 1)
                # Define a wider range for random exploration, e.g., up to 3x default or a fixed upper limit
                upper_bound = max_val if max_val is not None else max(min_val + 1, int(default * 3.0), 200) # Cap upper bound reasonably
                lower_bound = max(min_val, int(default * 0.25)) # Allow smaller values too
                if lower_bound >= upper_bound: lower_bound = min_val # Ensure lower < upper
                candidate_params[param] = random.randint(lower_bound, upper_bound)
            elif isinstance(default, float):
                has_numeric = True
                # Define bounds based on param type
                if param in factor_params:
                    min_val = min_val if min_val is not None else 0.01
                    max_val = max_val if max_val is not None else 1.0
                elif param in dev_scalar_params:
                    min_val = min_val if min_val is not None else 0.1
                    max_val = max_val if max_val is not None else 5.0
                else: # Generic float
                    min_val = min_val if min_val is not None else 0.0
                    max_val = max_val if max_val is not None else max(min_val + 0.1, default * 3.0) # Generic upper bound

                # Generate random float within bounds
                lower_bound = max(min_val, round(default * 0.25, 4))
                upper_bound = min(max_val, round(default * 3.0, 4))
                if lower_bound >= upper_bound: upper_bound = max(lower_bound + 0.01, max_val) # Ensure range

                candidate_params[param] = round(random.uniform(lower_bound, upper_bound), 4)

            elif isinstance(default, str):
                 # Handle potential choices if defined (future enhancement)
                 # For now, just use the default string
                 candidate_params[param] = default
            # Add bool handling if needed
            # elif isinstance(default, bool):
            #      candidate_params[param] = random.choice([True, False])
            else: # Other types, just use default
                 candidate_params[param] = default

        if not has_numeric and attempt == 1:
            logger.warning(f"Indicator has no numeric parameters with defaults. Random generation will only produce the default config.")
            default_params = {k: v.get('default') for k, v in parameter_definitions.items() if 'default' in v}
            return default_params if evaluate_conditions(default_params, conditions) else None

        # Check if the randomly generated combination is valid
        if evaluate_conditions(candidate_params, conditions):
            logger.debug(f"Generated random valid config (try {attempt}): {candidate_params}")
            return candidate_params

    logger.warning(f"Could not generate a random valid configuration after {max_tries} tries.")
    return None
# >>> END NEW FUNCTION <<<