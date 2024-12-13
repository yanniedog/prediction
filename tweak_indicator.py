# tweak_indicator.py

import sys
import itertools
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import json

from config import DB_PATH
from indicator_config_parser import get_configurable_indicators, get_indicator_parameters
from sqlite_data_manager import insert_indicator_configs as sqlite_insert_indicator_configs, create_connection
from logging_setup import configure_logging

logger = logging.getLogger(__name__)

def generate_configurations(parameter_keys: List[str], default_params: Dict) -> List[Dict]:
    """
    Generate all possible configurations based on parameter ranges.
    
    Args:
        parameter_keys (List[str]): List of parameter names.
        default_params (Dict): Dictionary of parameter names and their default values.
    
    Returns:
        List[Dict]: List of configuration dictionaries.
    """
    param_ranges = {}
    for param in parameter_keys:
        default = default_params[param]
        if isinstance(default, int):
            start = max(1, default - 5)
            end = default + 5
            param_ranges[param] = list(range(start, end + 1))
        elif isinstance(default, float):
            # Generate a range around the default value
            param_ranges[param] = [round(default * factor, 4) for factor in np.arange(0.8, 1.21, 0.05)]
        elif isinstance(default, list):
            # For list-type parameters like 'ranges'
            param_ranges[param] = default
        else:
            param_ranges[param] = [default]
    if param_ranges:
        keys, values = zip(*param_ranges.items())
        configurations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    else:
        configurations = []
    return configurations

def insert_tweaked_configs(conn, indicator_name: str, configurations: List[Dict]):
    """
    Insert generated configurations into the database.
    
    Args:
        conn: Database connection object.
        indicator_name (str): Name of the indicator.
        configurations (List[Dict]): List of configuration dictionaries.
    """
    try:
        sqlite_insert_indicator_configs(conn, indicator_name, configurations)
        logger.info(f"Inserted {len(configurations)} configurations for indicator '{indicator_name}' into the database.")
    except Exception as e:
        logger.error(f"Error inserting configurations for '{indicator_name}': {e}")

def fetch_available_indicators(indicator_params_path='indicator_params.json') -> List[str]:
    """
    Fetch the list of configurable indicators.
    
    Args:
        indicator_params_path (str): Path to indicator_params.json
    
    Returns:
        List[str]: List of configurable indicator names.
    """
    return get_configurable_indicators(indicator_params_path)

def insert_indicator_configs(conn, indicator_name: str, configurations: List[Dict]):
    """
    Wrapper function to insert configurations.
    
    Args:
        conn: Database connection object.
        indicator_name (str): Name of the indicator.
        configurations (List[Dict]): List of configuration dictionaries.
    """
    insert_tweaked_configs(conn, indicator_name, configurations)

def parse_indicator_parameters(indicator_name: str, indicator_params_path='indicator_params.json') -> Optional[Dict]:
    """
    Parse parameters for a specific indicator.
    
    Args:
        indicator_name (str): Name of the indicator.
        indicator_params_path (str): Path to indicator_params.json
    
    Returns:
        Optional[Dict]: Dictionary of parameters or None.
    """
    return get_indicator_parameters(indicator_name, indicator_params_path)

def select_indicators(available_indicators: List[str]) -> List[str]:
    """
    Allow user to select indicators for tweaking.
    
    Args:
        available_indicators (List[str]): List of available indicator names.
    
    Returns:
        List[str]: List of selected indicator names.
    """
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

def main():
    """
    Main function to handle the tweaking of indicator configurations.
    """
    parser = argparse.ArgumentParser(description="Tweak Indicator Configurations Dynamically")
    parser.add_argument('--all', action='store_true', help='Configure all indicators automatically')
    args = parser.parse_args()

    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    configure_logging(log_file_prefix="tweak_indicators")
    logger.info("Starting tweak_indicator.py")

    conn = create_connection(DB_PATH)
    if not conn:
        logger.error("Failed to connect to the database.")
        sys.exit(1)

    try:
        available_indicators = fetch_available_indicators('indicator_params.json')
        if not available_indicators:
            logger.error("No configurable indicators found in indicator_params.json.")
            print("No configurable indicators found. Exiting.")
            sys.exit(1)
    except Exception as e:
        logger.exception(f"Error fetching configurable indicators: {e}")
        print(f"Error fetching configurable indicators: {e}")
        sys.exit(1)

    if args.all:
        selected_indicators = available_indicators
        logger.info("All indicators selected for tweaking.")
    else:
        selected_indicators = select_indicators(available_indicators)
        logger.info(f"Selected indicators for tweaking: {selected_indicators}")

    for indicator_name in selected_indicators:
        logger.info(f"Processing Indicator: {indicator_name}")
        print(f"\nProcessing Indicator: {indicator_name}")
        try:
            parameters = parse_indicator_parameters(indicator_name, 'indicator_params.json')
            if not parameters:
                logger.warning(f"No parameters found for '{indicator_name}'. Skipping configuration generation.")
                print(f"No parameters found for '{indicator_name}'. Skipping.")
                continue
            print(f"Parameters for '{indicator_name}':")
            for param, value in parameters.items():
                print(f"  - {param}: {value}")
            logger.info(f"Parameters for '{indicator_name}': {parameters}")
        except Exception as e:
            logger.error(f"Error retrieving parameters for '{indicator_name}': {e}")
            print(f"Error retrieving parameters for '{indicator_name}'. Skipping.")
            continue
        try:
            configurations = generate_configurations(list(parameters.keys()), parameters)
            if not configurations:
                logger.warning(f"No configurations generated for '{indicator_name}'.")
                print(f"No configurations generated for '{indicator_name}'.")
                continue
            print(f"Generated {len(configurations)} configurations for '{indicator_name}'.")
            logger.info(f"Generated {len(configurations)} configurations for '{indicator_name}'.")
        except Exception as e:
            logger.error(f"Error generating configurations for '{indicator_name}': {e}")
            print(f"Error generating configurations for '{indicator_name}'. Skipping.")
            continue
        try:
            insert_indicator_configs(conn, indicator_name, configurations)
        except Exception as e:
            logger.error(f"Error inserting configurations into database for '{indicator_name}': {e}")
            print(f"Error inserting configurations into database for '{indicator_name}'.")
            continue

    conn.close()
    logger.info("All selected indicators have been processed successfully.")
    print("\nAll selected indicators have been processed successfully.")

if __name__ == "__main__":
    main()
