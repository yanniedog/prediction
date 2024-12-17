# tweak_params.py
import sys
import itertools
import logging
from typing import Dict, List
from config import DB_PATH
from indicator_config_parser import get_configurable_indicators, get_indicator_parameters
from sqlite_data_manager import insert_indicator_configs, create_connection, initialize_database
from logging_setup import configure_logging

logger = logging.getLogger(__name__)

def generate_configurations(parameter_keys: List[str], parameter_definitions: Dict) -> List[Dict]:
    param_alternatives = {}
    for param in parameter_keys:
        param_def = parameter_definitions[param]
        default = param_def.get('default')
        if default is None:
            logger.warning(f"Parameter '{param}' does not have a default value. Skipping.")
            continue
        if isinstance(default, int):
            if param in ['fast', 'slow', 'fastperiod', 'slowperiod', 'signalperiod', 'timeperiod', 'timeperiod1', 'timeperiod2', 'timeperiod3', 'pivot_lookback']:
                start = max(2, default - 5)
            else:
                start = max(1, default - 5)
            param_alternatives[param] = [start + i for i in range(11)]
        elif isinstance(default, float):
            if param in ['fast', 'slow', 'fastperiod', 'slowperiod', 'signalperiod', 'timeperiod', 'timeperiod1', 'timeperiod2', 'timeperiod3', 'pivot_lookback']:
                start = max(2.0, default - 5.0)
                param_alternatives[param] = [round(start + i, 1) for i in range(11)]
            elif 0 < default < 1:
                start = max(0.1, default * 0.9)
                param_alternatives[param] = [round(start + 0.02 * i, 4) for i in range(11)]
            else:
                param_alternatives[param] = [default]
        else:
            logger.warning(f"Parameter '{param}' is not a numeric type. Using default only.")
            param_alternatives[param] = [default]

    if not param_alternatives:
        return []

    keys, values = zip(*param_alternatives.items())
    configurations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    valid_combinations = []
    for combo in configurations:
        valid_combinations.append(combo)
    return valid_combinations

def select_indicators(available_indicators: List[str]) -> List[str]:
    print("\nAvailable Indicators for Tweak:")
    for idx, indicator in enumerate(available_indicators, 1):
        print(f"{idx}. {indicator}")
    print(f"{len(available_indicators)+1}. All Indicators")
    selected_indicators = []
    while True:
        choice = input(f"Select an indicator by number (1-{len(available_indicators)+1}) or type 'done' to finish selection: ").strip().lower()
        if choice == 'done':
            break
        if choice.isdigit():
            choice_num = int(choice)
            if 1 <= choice_num <= len(available_indicators):
                selected_indicators.append(available_indicators[choice_num - 1])
                print(f"Selected: {available_indicators[choice_num - 1]}")
            elif choice_num == len(available_indicators) + 1:
                selected_indicators = available_indicators.copy()
                print("All indicators selected.")
                break
            else:
                print("Invalid selection. Please try again.")
        else:
            print("Invalid input. Please enter a number corresponding to the indicator or 'done'.")
    if not selected_indicators:
        print("No indicators selected. Exiting.")
        sys.exit(0)
    return selected_indicators

def insert_tweaked_configs(conn, indicator_name: str, configurations: List[Dict]):
    try:
        insert_indicator_configs(conn, indicator_name, configurations)
    except Exception as e:
        logger.error(f"Error inserting configurations for '{indicator_name}': {e}")
        raise e

def tweak_params():
    """
    Generates parameter configurations for configurable indicators and inserts them into the database.
    """
    initialize_database(DB_PATH)
    conn = create_connection(DB_PATH)
    if not conn:
        logger.error("Failed to connect to the database.")
        sys.exit(1)
    try:
        available_indicators = get_configurable_indicators('indicator_params.json')
        if not available_indicators:
            logger.error("No configurable indicators found in indicator_params.json.")
            print("No configurable indicators found. Exiting.")
            sys.exit(1)
    except Exception as e:
        logger.exception(f"Error fetching configurable indicators: {e}")
        print(f"Error fetching configurable indicators: {e}")
        sys.exit(1)

    selected_indicators = select_indicators(available_indicators)
    logger.info(f"Selected indicators for tweaking: {selected_indicators}")

    for indicator_name in selected_indicators:
        logger.info(f"Processing Indicator: {indicator_name}")
        print(f"\nProcessing Indicator: {indicator_name}")
        try:
            parameters = get_indicator_parameters(indicator_name, 'indicator_params.json')
            if not parameters:
                logger.warning(f"No parameters found for '{indicator_name}'. Skipping configuration generation.")
                print(f"No parameters found for '{indicator_name}'. Skipping.")
                continue
            param_defs = parameters.get('parameters', {})
            if not param_defs:
                logger.warning(f"No 'parameters' section for '{indicator_name}'. Skipping.")
                print(f"No 'parameters' section for '{indicator_name}'. Skipping.")
                continue
            print(f"Parameters for '{indicator_name}':")
            for param, value in param_defs.items():
                print(f"  - {param}: {value}")
            logger.info(f"Parameters for '{indicator_name}': {param_defs}")
        except Exception as e:
            logger.error(f"Error retrieving parameters for '{indicator_name}': {e}")
            print(f"Error retrieving parameters for '{indicator_name}'. Skipping.")
            continue
        try:
            configurations = generate_configurations(list(param_defs.keys()), param_defs)
            if not configurations:
                logger.warning(f"No configurations generated for '{indicator_name}'. Skipping.")
                print(f"No configurations generated for '{indicator_name}'. Skipping.")
                continue
            print(f"Generated {len(configurations)} configurations for '{indicator_name}'.")
            logger.info(f"Generated {len(configurations)} configurations for '{indicator_name}'.")
            example_configs = configurations[:5]
            print(f"Example configurations for '{indicator_name}': {example_configs}")
            logger.info(f"Example configurations for '{indicator_name}': {example_configs}")
        except Exception as e:
            logger.error(f"Error generating configurations for '{indicator_name}': {e}")
            print(f"Error generating configurations for '{indicator_name}'. Skipping.")
            continue
        try:
            insert_tweaked_configs(conn, indicator_name, configurations)
            print(f"Inserted configurations for '{indicator_name}' into the database.")
            logger.info(f"Inserted configurations for '{indicator_name}' into the database.")
        except Exception as e:
            logger.error(f"Error inserting configurations into database for '{indicator_name}': {e}")
            print(f"Error inserting configurations into database for '{indicator_name}'.")
            continue
    conn.close()
    logger.info("All selected indicators have been processed successfully.")
    print("\nAll selected indicators have been processed successfully.")

if __name__ == "__main__":
    configure_logging(log_file_prefix="tweak_params")
    tweak_params()
