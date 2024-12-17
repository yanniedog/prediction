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
            if default >= 1:
                start = max(1, default - 5)
                param_alternatives[param] = [start + i for i in range(11)]
            elif 0 < default < 1:
                param_alternatives[param] = [round(default * (0.9 + 0.02 * i), 4) for i in range(11)]
            else:
                logger.warning(f"Parameter '{param}' has a default value that does not fit the criteria. Using default only.")
                param_alternatives[param] = [default]
        elif isinstance(default, float):
            if default >= 1:
                param_alternatives[param] = [round(default - 5.0 + i, 1) for i in range(11)]
            elif 0 < default < 1:
                param_alternatives[param] = [round(default * (0.9 + 0.02 * i), 4) for i in range(11)]
            else:
                logger.warning(f"Parameter '{param}' has a default value that does not fit the criteria. Using default only.")
                param_alternatives[param] = [default]
        else:
            logger.warning(f"Parameter '{param}' is not a numeric type. Using default only.")
            param_alternatives[param] = [default]
    
    if not param_alternatives:
        return []
    
    keys, values = zip(*param_alternatives.items())
    configurations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return configurations

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
            logger.error(f"No parameters found for '{selected_indicator}'. Exiting.")
            print(f"No parameters found for '{selected_indicator}'. Exiting.")
            sys.exit(1)
        param_defs = parameters.get('parameters', {})
        if not param_defs:
            logger.error(f"No 'parameters' section for '{selected_indicator}'. Exiting.")
            print(f"No 'parameters' section for '{selected_indicator}'. Exiting.")
            sys.exit(1)
        print(f"Parameters for '{selected_indicator}': {param_defs}")
        logger.info(f"Parameters for '{selected_indicator}': {param_defs}")
    except Exception as e:
        logger.error(f"Error retrieving parameters for '{selected_indicator}': {e}")
        print(f"Error retrieving parameters for '{selected_indicator}'. Exiting.")
        sys.exit(1)
    
    try:
        configurations = generate_configurations(list(param_defs.keys()), param_defs)
        if not configurations:
            logger.error(f"No configurations generated for '{selected_indicator}'. Exiting.")
            print(f"No configurations generated for '{selected_indicator}'. Exiting.")
            sys.exit(1)
        print(f"Generated {len(configurations)} configurations for '{selected_indicator}'.")
        logger.info(f"Generated {len(configurations)} configurations for '{selected_indicator}'.")
        example_configs = configurations[:5]
        print(f"Example configurations for '{selected_indicator}': {example_configs}")
        logger.info(f"Example configurations for '{selected_indicator}': {example_configs}")
    except Exception as e:
        logger.error(f"Error generating configurations for '{selected_indicator}': {e}")
        print(f"Error generating configurations for '{selected_indicator}'. Exiting.")
        sys.exit(1)
    
    try:
        insert_tweaked_configs(conn, selected_indicator, configurations)
        print(f"Inserted configurations for '{selected_indicator}' into the database.")
        logger.info(f"Inserted configurations for '{selected_indicator}' into the database.")
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
