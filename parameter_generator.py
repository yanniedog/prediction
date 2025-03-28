# parameter_generator.py
import logging
import itertools
from typing import Dict, List, Any, Optional
import numpy as np # Import numpy for rounding

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
            logger.warning(f"Condition parameter '{param_name}' not found in current combo {params}. Condition fails.")
            return False

        current_param_value = params[param_name]

        for op, value_or_ref in ops_dict.items():
            compare_value = None
            if isinstance(value_or_ref, str) and value_or_ref in params:
                compare_value = params[value_or_ref]
            elif not isinstance(value_or_ref, (str, list, dict)): # Check it's a literal type we can compare
                 compare_value = value_or_ref
            else:
                 logger.error(f"Condition value '{value_or_ref}' for param '{param_name}' is not a valid literal or parameter reference. Condition fails.")
                 return False

            if compare_value is None and not isinstance(value_or_ref, (int, float, type(None))): # Allow comparison with None literal if specified
                logger.error(f"Could not resolve comparison value '{value_or_ref}' for condition {param_name} {op}. Condition fails.")
                return False

            if not _evaluate_single_condition(current_param_value, op, compare_value):
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
    period_params = ['fast', 'slow', 'fastperiod', 'slowperiod', 'signalperiod', 'timeperiod', 'timeperiod1', 'timeperiod2', 'timeperiod3', 'length']

    for param, details in parameter_definitions.items():
        default = details.get('default')
        if default is None:
            logger.debug(f"Parameter '{param}' has no default. Skipping tweaking for this parameter.")
            continue # Skip parameters without defaults for range generation

        values = []
        if isinstance(default, int):
            min_val = 2 if param in period_params else 1
            # Generate range around the default
            start = max(min_val, default - range_steps * int_step)
            # Create the list
            generated_values = [max(min_val, start + i * int_step) for i in range(num_values_to_generate)]
            # Add default, remove duplicates, sort
            values = sorted(list(set(generated_values + [default])))

        elif isinstance(default, float):
            # Handle different types of floats differently
            if 0 < abs(default) < 1: # Small fractional floats (e.g., limits)
                step = 0.02 # Smaller step
                min_val = 0.01
                start = max(min_val, round(default - range_steps * step, 4))
                generated_values = [round(max(min_val, start + i * step), 4) for i in range(num_values_to_generate)]
                values = sorted(list(set(generated_values + [default])))
            elif param in period_params: # Treat float periods like ints
                 min_val = 2.0
                 step = 1.0
                 start = max(min_val, default - range_steps * step)
                 generated_values = [round(max(min_val, start + i * step), 1) for i in range(num_values_to_generate)]
                 values = sorted(list(set(generated_values + [default])))
            else: # Other floats (like nbdev, scalar, vfactor)
                 step = 0.2 # Larger step
                 min_val = 0.1 if param == 'scalar' or param == 'nbdev' else 0.0
                 start = round(default - range_steps * step, 2)
                 generated_values = [round(max(min_val, start + i * step), 2) for i in range(num_values_to_generate)]
                 values = sorted(list(set(generated_values + [default])))
        else: # Non-numeric or complex defaults (e.g., strings like 'ema') - don't generate range
            values = [default]

        if values:
             param_ranges[param] = values
             logger.debug(f"Generated range for '{param}': {values}")

    if not param_ranges:
        logger.warning("No parameters found with defaults to generate configurations.")
        return [{}] if not parameter_definitions else []

    keys = list(param_ranges.keys())
    all_combinations_tuples = list(itertools.product(*param_ranges.values()))
    all_combinations = [dict(zip(keys, combo)) for combo in all_combinations_tuples]

    valid_combinations = [combo for combo in all_combinations if evaluate_conditions(combo, conditions)]

    logger.info(f"Generated {len(all_combinations)} total raw combinations, {len(valid_combinations)} valid combinations after applying conditions.")

    # Ensure default is included if valid
    default_combo = {p: d.get('default') for p, d in parameter_definitions.items() if d.get('default') is not None}
    if default_combo and evaluate_conditions(default_combo, conditions):
        is_present = any(valid_combo == default_combo for valid_combo in valid_combinations)
        if not is_present:
             valid_combinations.append(default_combo)
             logger.info("Default configuration was valid but missing, added it back.")

    # Ensure uniqueness
    seen = set()
    unique_valid_combinations = []
    for combo in valid_combinations:
         try:
             combo_tuple = tuple(sorted(combo.items()))
             if combo_tuple not in seen:
                 unique_valid_combinations.append(combo)
                 seen.add(combo_tuple)
         except TypeError:
              logger.warning(f"Could not hash configuration for uniqueness check: {combo}")
              unique_valid_combinations.append(combo) # Add anyway

    if len(unique_valid_combinations) != len(valid_combinations):
         logger.info(f"Removed {len(valid_combinations) - len(unique_valid_combinations)} duplicate configurations.")

    logger.info(f"Returning {len(unique_valid_combinations)} unique valid configurations.")
    return unique_valid_combinations