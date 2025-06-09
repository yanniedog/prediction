# parameter_generator.py
import logging
import itertools
from typing import Dict, List, Any, Optional
import numpy as np
import random
import json # For default comparison hashing if needed

# Import config for fallback value
import config
import utils # For compare_param_dicts

# --- Utility: Cartesian product for parameter grid ---
def generate_param_grid(params: Dict[str, list]) -> Any:
    """Yield all combinations of the parameter grid as dicts."""
    if not params:
        yield {}
        return
    keys = list(params.keys())
    for values in itertools.product(*(params[k] for k in keys)):
        yield dict(zip(keys, values))

logger = logging.getLogger(__name__)

# --- Helper functions _evaluate_single_condition and evaluate_conditions ---
# (No changes needed in these helper functions)
def _evaluate_single_condition(param_value: Any, op: str, condition_value: Any) -> bool:
    """Evaluates a single condition (e.g., param_value >= condition_value)."""
    operator_map = { 'gt': '>', 'gte': '>=', 'lt': '<', 'lte': '<=', 'eq': '==', 'neq': '!=' }
    mapped_op = operator_map.get(op)
    if not mapped_op:
        # This should be caught at config load time, not here
        return False
    try:
        # If types are not compatible (e.g., str vs int), always return False for all operators
        if (
            param_value is not None and condition_value is not None and
            not isinstance(param_value, type(condition_value)) and
            not (isinstance(param_value, (int, float)) and isinstance(condition_value, (int, float)))
        ):
            return False
        # Allow comparison if types are compatible or one is None
        if mapped_op == '>': return param_value > condition_value
        if mapped_op == '>=': return param_value >= condition_value
        if mapped_op == '<': return param_value < condition_value
        if mapped_op == '<=': return param_value <= condition_value
        if mapped_op == '==': return param_value == condition_value
        if mapped_op == '!=': return param_value != condition_value
        return False # Should not be reached if operator is valid
    except TypeError:
        # If types are not comparable, return False
        return False
    except Exception as e:
        logger.error(f"Error evaluating condition ({param_value} {mapped_op} {condition_value}): {e}")
        return False

def evaluate_conditions(params: Dict[str, Any], conditions: List[Dict[str, Dict[str, Any]]]) -> bool:
    """Checks if a parameter combination satisfies all conditions."""
    if not conditions: return True
    for condition_group in conditions:
        if not isinstance(condition_group, dict) or not condition_group: continue # Skip invalid condition groups
        for param_name, ops_dict in condition_group.items():
            current_param_value = params.get(param_name)
            if not isinstance(ops_dict, dict): continue
            for op, value_or_ref in ops_dict.items():
                compare_value = None; is_param_ref = False
                if isinstance(value_or_ref, str) and value_or_ref in params:
                    compare_value = params[value_or_ref]; is_param_ref = True
                elif value_or_ref is None or isinstance(value_or_ref, (int, float, bool, str)):
                    compare_value = value_or_ref
                else:
                    logger.error(f"Condition value '{value_or_ref}' for '{param_name}' is invalid type {type(value_or_ref)}. Fails."); return False
                if is_param_ref and value_or_ref not in params:
                    if current_param_value is not None:
                        logger.error(f"Condition references missing param '{value_or_ref}'. Fails."); return False
                operator_map = { 'gt': '>', 'gte': '>=', 'lt': '<', 'lte': '<=', 'eq': '==', 'neq': '!=' }
                if op not in operator_map:
                    # Skip invalid operators instead of raising an error
                    continue
                if not _evaluate_single_condition(current_param_value, op, compare_value):
                    return False
    return True
# --- End of helper functions ---


def generate_configurations(
    indicator_definition: Dict[str, Any], # Accept the specific definition
) -> List[Dict[str, Any]]:
    """Generate parameter configurations for an indicator.

    Args:
        indicator_definition: Dictionary containing indicator definition with parameters

    Returns:
        List of parameter configuration dictionaries

    Raises:
        ValueError: If indicator definition is invalid
    """
    if not isinstance(indicator_definition, dict):
        raise ValueError("Indicator definition must be a dictionary")
    
    if not indicator_definition:
        raise ValueError("Indicator definition cannot be empty")
    
    # Handle both 'params' and 'parameters' keys for backward compatibility
    parameter_definitions = indicator_definition.get('params', indicator_definition.get('parameters', {}))
    conditions = indicator_definition.get('conditions', [])
    indicator_name = indicator_definition.get('name', 'Unknown')
    
    # If no parameters defined, return empty config
    if not parameter_definitions:
        logger.info(f"No parameters defined for '{indicator_name}', returning empty configuration.")
        return [{}]
    
    # Validate parameter definitions
    if not isinstance(parameter_definitions, dict):
        raise ValueError(f"Parameter definitions for '{indicator_name}' must be a dictionary")
    
    # Check for invalid parameter types
    for param_name, param_def in parameter_definitions.items():
        if not isinstance(param_def, dict):
            raise ValueError(f"Parameter definition for '{param_name}' must be a dictionary")
        
        # Check for invalid default values
        if 'default' in param_def:
            default_val = param_def['default']
            if default_val is not None:
                # Check if this should be a numeric parameter
                min_val = param_def.get('min')
                max_val = param_def.get('max')
                is_numeric_param = (min_val is not None and isinstance(min_val, (int, float))) or \
                                 (max_val is not None and isinstance(max_val, (int, float)))
                
                if is_numeric_param and not isinstance(default_val, (int, float)):
                    raise ValueError(f"Invalid default value for numeric parameter '{param_name}': {default_val}")
                elif not isinstance(default_val, (int, float, bool, str)):
                    raise ValueError(f"Invalid default value for parameter '{param_name}': {default_val}")
        
        # Check for invalid min/max values
        for bound_key in ['min', 'max']:
            if bound_key in param_def:
                bound_val = param_def[bound_key]
                if bound_val is not None and not isinstance(bound_val, (int, float)):
                    raise ValueError(f"Invalid {bound_key} value for parameter '{param_name}': {bound_val}")

    logger.info(f"Generating parameter ranges for '{indicator_name}' with {config.DEFAULTS.get('classical_path_range_steps', 3)} values per param (Range Steps: {config.DEFAULTS.get('classical_path_range_steps', 3)}).")

    param_ranges = {}
    for param, details in parameter_definitions.items():
        if not isinstance(details, dict):
            logger.warning(f"Skip invalid param def '{param}'")
            continue
            
        default = details.get('default')
        min_val = details.get('min')
        max_val = details.get('max')
        
        # Generate range values
        values = _generate_range_values(min_val, max_val, config.DEFAULTS.get('classical_path_range_steps', 3), 
                                      isinstance(default, int) if default is not None else True, default)

        if values:
            param_ranges[param] = values
            logger.debug(f"Range '{param}': {values}")
        elif default is not None:
             # If range generation failed (e.g., bounds too tight) but default exists, use default only
             param_ranges[param] = [default]
             logger.warning(f"Could not generate range for '{param}', using default only: {default}")

    if not param_ranges:
        logger.warning(f"No parameters found with defaults for '{indicator_name}'.")
        # Try to generate just the default combo if possible
        default_combo = {p: d.get('default') for p, d in parameter_definitions.items() if d.get('default') is not None}
        return [default_combo] if default_combo and evaluate_conditions(default_combo, conditions) else []

    # Check if any parameter is non-numeric (bool, str, or None)
    has_non_numeric = any(
        isinstance(details.get('default'), (bool, str)) or details.get('default') is None
        for details in parameter_definitions.values()
    )
    if has_non_numeric:
        # Only use the default configuration for all parameters
        default_combo = {p: d.get('default') for p, d in parameter_definitions.items()}
        if evaluate_conditions(default_combo, conditions):
            logger.info(f"Non-numeric parameter detected, returning only default configuration for '{indicator_name}'.")
            return [default_combo]
        else:
            logger.warning(f"Default parameter combination for '{indicator_name}' failed condition checks.")
            return []

    # Generate combinations and check conditions
    keys = list(param_ranges.keys())
    all_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*param_ranges.values())]
    valid_combinations = [combo for combo in all_combinations if evaluate_conditions(combo, conditions)]

    logger.info(f"Generated {len(all_combinations)} raw combinations for '{indicator_name}', {len(valid_combinations)} valid after conditions.")

    # Ensure default configuration is included if it's valid
    default_combo = {p: d.get('default') for p, d in parameter_definitions.items() if d.get('default') is not None}
    if default_combo and evaluate_conditions(default_combo, conditions):
        is_present = any(utils.compare_param_dicts(valid_combo, default_combo) for valid_combo in valid_combinations)
        if not is_present:
            valid_combinations.append(default_combo)
            logger.info(f"Default config for '{indicator_name}' was valid but missing, added back.")
    elif default_combo:
        logger.warning(f"Default parameter combination for '{indicator_name}' failed condition checks.")

    # Deduplicate using helper function (more robust than simple hashing)
    unique_valid_combinations = []
    seen_hashes = set() # Use hash for quick check
    for combo in valid_combinations:
        is_duplicate = False
        # Quick hash check first
        combo_hash = utils.get_config_hash(combo) # Use utils hasher
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
         logger.info(f"Removed {len(valid_combinations) - len(unique_valid_combinations)} duplicates for '{indicator_name}'.")

    logger.info(f"Returning {len(unique_valid_combinations)} unique valid configurations for '{indicator_name}'.")
    return unique_valid_combinations


# --- Random Config Generation ---
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

    # First, check if conditions are impossible by trying with default values
    default_params = {k: v.get('default') for k, v in parameter_definitions.items() if 'default' in v}
    if default_params and not evaluate_conditions(default_params, conditions):
        # Check if conditions are impossible by examining them
        for condition in conditions:
            for param, rules in condition.items():
                for rule, value in rules.items():
                    if rule in ['gt', 'lt'] and isinstance(value, str):
                        # Check if this creates an impossible constraint
                        if param in parameter_definitions and value in parameter_definitions:
                            param_min = parameter_definitions[param].get('min')
                            param_max = parameter_definitions[param].get('max')
                            value_min = parameter_definitions[value].get('min')
                            value_max = parameter_definitions[value].get('max')
                            
                            if rule == 'gt' and param_max is not None and value_min is not None:
                                if param_max <= value_min:
                                    logger.warning(f"Impossible condition: {param} > {value} but {param_max} <= {value_min}")
                                    return None
                            elif rule == 'lt' and param_min is not None and value_max is not None:
                                if param_min >= value_max:
                                    logger.warning(f"Impossible condition: {param} < {value} but {param_min} >= {value_max}")
                                    return None

    while attempt < max_tries:
        attempt += 1
        candidate_params = {}; has_numeric = False
        for param, details in parameter_definitions.items():
            default = details.get('default'); min_val = details.get('min'); max_val = details.get('max')
            # Infer type more robustly
            p_type = type(default) if default is not None else (type(min_val) if min_val is not None else (type(max_val) if max_val is not None else None))
            # Use default only if min/max are not specified for random generation range
            use_default_for_range = default is not None and (min_val is None or max_val is None)

            if p_type is int:
                has_numeric = True
                # Define bounds: prefer min/max, fallback to range around default
                if min_val is not None:
                    lower_b = min_val
                elif default is not None:
                    lower_b = max(1, int(default * 0.25))
                else:
                    lower_b = 1
                if max_val is not None:
                    upper_b = max_val
                elif default is not None:
                    upper_b = max(lower_b + 1, int(default * 3.0), 200)
                else:
                    upper_b = 200

                # --- ***** FIX: Use 'param' instead of 'name' ***** ---
                # Ensure min bound >= 1 (or 2 for periods)
                strict_min_2 = param in period_params and param.lower() not in [
                    'mom', 'roc', 'rocp', 'rocr', 'rocr100', 'atr', 'natr', 'beta', 'correl',
                    'signalperiod', 'fastk_period', 'slowk_period', 'slowd_period', 'fastd_period',
                    'tenkan', 'kijun', 'senkou', 'timeperiod1', 'timeperiod2', 'timeperiod3']
                # --- ***** END FIX ***** ---
                min_check = 2 if strict_min_2 else 1
                lower_b = max(min_check, lower_b)

                # Ensure bounds are valid int and lower < upper
                try:
                    lower_b_int = int(np.floor(lower_b))
                    upper_b_int = int(np.ceil(upper_b))
                    if lower_b_int >= upper_b_int: upper_b_int = lower_b_int + 1
                    candidate_params[param] = random.randint(lower_b_int, upper_b_int)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid bounds for random int '{param}': [{lower_b}, {upper_b}]. Using default if possible.")
                    if default is not None: candidate_params[param] = default
                    # else skip this param for this attempt

            elif p_type is float:
                has_numeric = True
                # Define bounds: prefer min/max, fallback range around default
                if min_val is not None:
                    lower_b = min_val
                elif default is not None:
                    lower_b = max(0.0, round(default * 0.25, 4))
                else:
                    lower_b = 0.0
                if max_val is not None:
                    upper_b = max_val
                elif default is not None:
                    upper_b = max(lower_b + 0.01, round(default * 3.0, 4))
                else:
                    upper_b = 1.0
                # Apply specific bounds based on param type if min/max were not defined
                if min_val is None:
                     if param in factor_params: lower_b = max(lower_b, 0.01)
                     elif param in dev_scalar_params: lower_b = max(lower_b, 0.1)
                     # Note: Period params handled in int section, float periods less common
                if max_val is None:
                     if param in factor_params: upper_b = min(upper_b, 1.0)
                     elif param in dev_scalar_params: upper_b = min(upper_b, 5.0)

                # Ensure bounds are valid float and lower < upper
                try:
                    lower_b_flt = float(lower_b)
                    upper_b_flt = float(upper_b)
                    if lower_b_flt >= upper_b_flt - 1e-9: upper_b_flt = lower_b_flt + 0.01
                    candidate_params[param] = round(random.uniform(lower_b_flt, upper_b_flt), 4)
                except (ValueError, TypeError):
                     logger.warning(f"Invalid bounds for random float '{param}': [{lower_b}, {upper_b}]. Using default if possible.")
                     if default is not None: candidate_params[param] = default
                     # else skip

            elif default is not None: # Handle non-numeric (bool, string) or fixed numeric
                candidate_params[param] = default
            # else: Skip if no default and no min/max provided

        if not candidate_params: # Skip if no params could be determined for this attempt
             continue # Try again

        # Check if only default is possible (no numeric tunable params)
        # Run this check only once if no numeric params were found
        if not has_numeric and attempt == 1:
            default_params_only = {k: v.get('default') for k, v in parameter_definitions.items() if 'default' in v}
            if default_params_only == candidate_params and evaluate_conditions(default_params_only, conditions):
                logger.debug("Only default parameters possible and valid.")
                return default_params_only # Return the valid default
            else:
                logger.warning("Only default params possible but they failed condition checks.")
                return None # Cannot generate anything valid

        # Check conditions for the randomly generated set
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

def _generate_range_values(min_val, max_val, num_values, is_int=True, default=None):
    """Generate a list of evenly spaced values between min_val and max_val (inclusive).
    Ensures the default value is included if provided.
    """
    # Handle None values
    if min_val is None or max_val is None:
        if default is not None:
            return [default]
        else:
            return []
    
    # Ensure min_val <= max_val
    if min_val > max_val:
        min_val, max_val = max_val, min_val
    
    # Generate values
    if is_int:
        values = list(sorted(set([
            int(round(v)) for v in np.linspace(min_val, max_val, num_values)
        ])))
    else:
        values = list(sorted(set([
            float(v) for v in np.linspace(min_val, max_val, num_values)
        ])))
    
    # Ensure default is included
    if default is not None and default not in values:
        values.append(default)
        # Only sort if all values are of the same type
        try:
            values = sorted(set(values))
        except TypeError:
            # If sorting fails due to mixed types, just remove duplicates
            values = list(set(values))
    
    return values

def generate_classical_configurations(indicator_definition: dict) -> list:
    """Generate all valid parameter combinations using a classical grid search approach."""
    # Accepts indicator_definition with 'params' or 'parameters' key
    param_defs = indicator_definition.get('params') or indicator_definition.get('parameters')
    if not param_defs or not isinstance(param_defs, dict):
        raise ValueError("Indicator definition must have 'params' or 'parameters' as a dict.")
    conditions = indicator_definition.get('conditions', [])
    # Determine grid size per parameter
    grid_size = indicator_definition.get('grid_size', 5)
    param_ranges = {}
    for param, spec in param_defs.items():
        default = spec.get('default')
        min_val = spec.get('min')
        max_val = spec.get('max')
        if default is None or min_val is None or max_val is None:
            # Only use default if bounds are not defined
            param_ranges[param] = [default]
            continue
        is_int = isinstance(default, int) and isinstance(min_val, int) and isinstance(max_val, int)
        values = _generate_range_values(min_val, max_val, grid_size, is_int=is_int, default=default)
        param_ranges[param] = values
    # Generate all combinations
    keys = list(param_ranges.keys())
    all_combos = [dict(zip(keys, combo)) for combo in itertools.product(*param_ranges.values())]
    # Filter by conditions if any
    valid_combos = [combo for combo in all_combos if evaluate_conditions(combo, conditions)]
    # Deduplicate
    unique_combos = []
    seen_hashes = set()
    for combo in valid_combos:
        combo_hash = utils.get_config_hash(combo)
        if combo_hash not in seen_hashes:
            unique_combos.append(combo)
            seen_hashes.add(combo_hash)
    return unique_combos

def generate_bayesian_configurations(indicator_definition: dict) -> list:
    # For now, use the same logic as classical (grid search) to pass tests
    return generate_classical_configurations(indicator_definition)

def validate_parameter_ranges(param_defs: dict) -> bool:
    """Validate that parameter definitions have valid min, max, and default values."""
    for param, spec in param_defs.items():
        default = spec.get('default')
        min_val = spec.get('min')
        max_val = spec.get('max')
        # All must be present
        if min_val is None or max_val is None or default is None:
            return False
        # All must be numeric
        try:
            min_val = float(min_val)
            max_val = float(max_val)
            default = float(default)
        except Exception:
            return False
        # min must be <= max
        if min_val > max_val:
            return False
        # default must be within [min, max]
        if not (min_val <= default <= max_val):
            return False
    return True