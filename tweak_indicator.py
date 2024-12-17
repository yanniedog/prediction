# tweak_indicator.py
import sys
import itertools
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import json
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

def fetch_available_indicators() -> List[str]:
    return get_configurable_indicators('indicator_params.json')

def parse_indicator_parameters(indicator_name: str) -> Optional[Dict]:
    return get_indicator_parameters(indicator_name, 'indicator_params.json')

def insert_tweaked_configs(conn, indicator_name: str, configurations: List[Dict]):
    try:
        insert_indicator_configs(conn, indicator_name, configurations)
    except Exception as e:
        logger.error(f"Error inserting configurations for '{indicator_name}': {e}")
        raise e

def run_tweak_indicator():
    """
    Generates parameter configurations for configurable indicators and inserts them into the database.
    """
    initialize_database(DB_PATH)
    conn = create_connection(DB_PATH)
    if not conn:
        logger.error("Failed to connect to the database.")
        sys.exit(1)
    try:
        available_indicators = fetch_available_indicators()
        if not available_indicators:
            logger.error("No configurable indicators found in indicator_params.json.")
            print("No configurable indicators found. Exiting.")
            sys.exit(1)
    except Exception as e:
        logger.exception(f"Error fetching configurable indicators: {e}")
        print(f"Error fetching configurable indicators: {e}")
        sys.exit(1)

    selected_indicators = []
    print("\nAvailable Indicators for Tweak:")
    for idx, indicator in enumerate(available_indicators, 1):
        print(f"{idx}. {indicator}")
    choice = input(f"Select an indicator by number (1-{len(available_indicators)}): ").strip()
    if not choice.isdigit() or int(choice) < 1 or int(choice) > len(available_indicators):
        logger.error("Invalid choice. Exiting.")
        print("Invalid choice. Exiting.")
        sys.exit(1)

    selected_indicator = available_indicators[int(choice) - 1]
    logger.info(f"Selected indicator: {selected_indicator}")
    print(f"Selected indicator: {selected_indicator}")

    try:
        parameters = parse_indicator_parameters(selected_indicator)
        if not parameters:
            logger.error(f"No parameters found for '{selected_indicator}'. Using base indicator.")
            print(f"No parameters found for '{selected_indicator}'. Using base indicator.")
            return selected_indicator
        else:
            logger.info(f"Parameters for '{selected_indicator}': {parameters}")
            print(f"Parameters for '{selected_indicator}': {parameters}")
    except Exception as e:
        logger.error(f"Error retrieving parameters for '{selected_indicator}': {e}")
        print(f"Error retrieving parameters for '{selected_indicator}'. Exiting.")
        sys.exit(1)

    if 'parameters' in parameters:
        configurations = generate_configurations(list(parameters['parameters'].keys()), parameters['parameters'])
    else:
        configurations = []

    if not configurations:
        logger.error(f"No configurations generated for '{selected_indicator}'. Using base indicator.")
        print(f"No configurations generated for '{selected_indicator}'. Using base indicator.")
    else:
        logger.info(f"Generated {len(configurations)} configurations for '{selected_indicator}'.")
        print(f"Generated {len(configurations)} configurations for '{selected_indicator}'.")
        example_configs = configurations[:15]
        logger.info(f"Example configurations for '{selected_indicator}': {example_configs}")
        print(f"Example configurations for '{selected_indicator}': {example_configs}")

    try:
        insert_tweaked_configs(conn, selected_indicator, configurations)
        if configurations:
            logger.info("Configurations stored in database:")
            for config in configurations[:15]:
                logger.info(config)
                print(config)
        print(f"Configurations for '{selected_indicator}' have been added to the database.")
    except Exception as e:
        logger.error(f"Error inserting configurations into database for '{selected_indicator}': {e}")
        print(f"Error inserting configurations into database for '{selected_indicator}'. Exiting.")
        sys.exit(1)

    conn.close()
    logger.info(f"Configurations for '{selected_indicator}' have been added to the database.")
    print(f"Configurations for '{selected_indicator}' have been added to the database.")

if __name__ == "__main__":
    configure_logging(log_file_prefix="tweak_indicator")
    run_tweak_indicator()
