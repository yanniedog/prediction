# tweak_indicator.py
import sys
import itertools
from pathlib import Path
import pandas as pd
import numpy as np
import talib as ta
import pandas_ta as pta
from sqlite_data_manager import initialize_database, insert_indicator_configs, create_connection
import inspect
import logging
import argparse

def fetch_available_indicators(indicators_module):
    """
    Fetch all available indicators by analyzing the compute_all_indicators function.
    Returns a list of indicator names.
    """
    # Create dummy data
    dummy_data = pd.DataFrame({
        'open': np.random.random(200) * 100,
        'high': np.random.random(200) * 100,
        'low': np.random.random(200) * 100,
        'close': np.random.random(200) * 100,
        'volume': np.random.randint(1, 1000, 200)
    })

    # Compute indicators
    try:
        data_with_indicators = indicators_module.compute_all_indicators(dummy_data.copy())
        indicators = list(data_with_indicators.columns)
        # Remove original columns
        original_cols = ['open', 'high', 'low', 'close', 'volume']
        dynamic_indicators = [ind for ind in indicators if ind not in original_cols]
        return dynamic_indicators
    except Exception as e:
        print(f"Error fetching available indicators: {e}")
        return []

def parse_indicator_parameters(indicators_module, indicator_name):
    """
    Dynamically parse the parameters required for an indicator based on its computation in indicators.py.
    Returns a dictionary of parameter names and their default values.
    """
    # Retrieve the compute_configured_indicators function
    compute_configured = indicators_module.compute_configured_indicators

    # Get the source code of compute_configured_indicators
    try:
        source = inspect.getsource(compute_configured)
    except Exception as e:
        print(f"Error retrieving source code for compute_configured_indicators: {e}")
        return {}

    # Look for the specific indicator's computation block
    # This simplistic approach assumes that each indicator has its own if/elif block
    indicator_block = None
    for line in source.split('\n'):
        if f"if base_indicator == '{indicator_name}':" in line or f"elif base_indicator == '{indicator_name}':" in line:
            indicator_block = line
            break

    if not indicator_block:
        # Indicator might be handled differently or not have parameters
        return {}

    # Attempt to extract parameter names and default values using a simplistic approach
    # This is fragile and assumes a certain coding style
    parameters = {}
    lines = source.split('\n')
    start = False
    for line in lines:
        if f"if base_indicator == '{indicator_name}':" in line or f"elif base_indicator == '{indicator_name}':" in line:
            start = True
            continue
        if start:
            if line.strip().startswith("else:") or line.strip().startswith("elif base_indicator"):
                break  # End of this indicator's block
            # Attempt to extract parameter assignments
            if '=' in line:
                parts = line.strip().split('=')
                if len(parts) >= 2:
                    var_name = parts[0].strip()
                    value = parts[1].strip().rstrip(',')
                    # Attempt to convert value to int or float
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                        parameters[var_name] = value
                    except:
                        pass  # Ignore if conversion fails
    return parameters

def generate_configurations(parameters):
    """
    Generate configurations based on parameter ranges.
    For each parameter, define a range based on the default value.
    Returns a list of dictionaries representing different configurations.
    """
    param_ranges = {}
    for param, default in parameters.items():
        if isinstance(default, int):
            # Define a range: default -5 to default +5, minimum 1
            start = max(1, default - 5)
            end = default + 5
            param_ranges[param] = list(range(start, end + 1))
        elif isinstance(default, float):
            # Define a range: default * 0.8 to default * 1.2 with step 0.05
            param_ranges[param] = [round(default * factor, 2) for factor in np.arange(0.8, 1.21, 0.05)]
        else:
            # For other types, use the default value only
            param_ranges[param] = [default]

    # Generate all possible combinations
    if param_ranges:
        keys, values = zip(*param_ranges.items())
        configurations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    else:
        configurations = []

    return configurations

def insert_tweaked_configs(indicator_name, configurations):
    """
    Insert the generated configurations into the SQLite database.
    """
    conn = create_connection()
    if not conn:
        print("Failed to connect to the database.")
        return

    try:
        insert_indicator_configs(conn, indicator_name, configurations)
    except Exception as e:
        print(f"Error inserting configurations for '{indicator_name}': {e}")
    finally:
        conn.close()

def main():
    """
    Main function to dynamically process indicators and generate configurations.
    Usage: python tweak_indicator.py [--all]
    """
    parser = argparse.ArgumentParser(description="Tweak Indicator Configurations")
    parser.add_argument('--all', action='store_true', help='Configure all indicators automatically')
    args = parser.parse_args()

    # Initialize the database
    initialize_database()

    # Import indicators.py as a module
    try:
        import indicators
    except ImportError as e:
        print(f"Error importing indicators.py: {e}")
        sys.exit(1)

    # Fetch available indicators
    available_indicators = fetch_available_indicators(indicators)
    if not available_indicators:
        print("No indicators found in indicators.py.")
        sys.exit(1)

    if args.all:
        selected_indicators = available_indicators
    else:
        print("Available Indicators:")
        for idx, indicator in enumerate(available_indicators, 1):
            print(f"{idx}. {indicator}")
        choice = input("Select an indicator by number (or type 'all' to configure all indicators): ").strip().lower()
        if choice == 'all':
            selected_indicators = available_indicators
        elif choice.isdigit() and 1 <= int(choice) <= len(available_indicators):
            selected_indicators = [available_indicators[int(choice)-1]]
        else:
            print("Invalid choice. Exiting.")
            sys.exit(1)

    print(f"Selected Indicators: {selected_indicators}")

    # Process each selected indicator
    for indicator_name in selected_indicators:
        print(f"\nProcessing Indicator: {indicator_name}")
        parameters = parse_indicator_parameters(indicators, indicator_name)
        if not parameters:
            print(f"No parameters found for '{indicator_name}'. Skipping configuration generation.")
            continue
        print(f"Parameters for '{indicator_name}': {parameters}")

        configurations = generate_configurations(parameters)
        if not configurations:
            print(f"No configurations generated for '{indicator_name}'.")
            continue
        print(f"Generated {len(configurations)} configurations for '{indicator_name}'.")

        # Insert configurations into the database
        insert_tweaked_configs(indicator_name, configurations)
        print(f"Configurations for '{indicator_name}' have been added to the database.")

    print("\nAll selected indicators have been processed.")

if __name__ == "__main__":
    main()
