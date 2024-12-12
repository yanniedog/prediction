# tweak-indicator.py
import sys, itertools, inspect, pandas as pd
from sqlite_data_manager import create_tables, insert_indicator_configs, list_indicators
from indicators import compute_all_indicators
from config import DB_PATH
import sqlite3

def extract_indicators():
    indicator_functions = [name for name, obj in inspect.getmembers(sys.modules['indicators'], inspect.isfunction)
                           if name.startswith('compute_')]
    return sorted(indicator_functions)

def get_configurations(params):
    return [dict(zip(params.keys(), values)) for values in itertools.product(*[
        range(max(1, params[p] - 5), min(100, params[p] + 5) + 1) for p in params
    ])]

def main():
    if len(sys.argv) != 3:
        print("Usage: python tweak-indicator.py SYMBOL TIMEFRAME")
        sys.exit(1)

    symbol, timeframe = sys.argv[1:]
    conn = sqlite3.connect(DB_PATH)
    create_tables(conn)

    print("Select an indicator to tweak:")
    indicators = extract_indicators()
    for i, ind in enumerate(indicators, 1):
        print(f"{i}. {ind}")
    selected_idx = int(input(f"Enter number (Default: 1): ") or 1) - 1
    selected_indicator = indicators[selected_idx]

    print(f"Tweaking {selected_indicator}...")
    default_parameters = {
        # Placeholder for real parameter mappings
        'compute_obv_price_divergence': {'obv_period': 14, 'price_period': 14},
        'compute_eyeX_MFV_volume': {'range1': 50, 'range2': 75, 'range3': 100, 'range4': 200},
        'compute_eyeX_MFV_support_resistance': {'range1': 50, 'range2': 75, 'range3': 100, 'range4': 200, 'pivot_lookback': 5}
    }
    if selected_indicator not in default_parameters:
        print("No parameter mapping available for this indicator.")
        sys.exit(1)

    params = default_parameters[selected_indicator]
    configs = get_configurations(params)

    insert_indicator_configs(conn, selected_indicator, configs)
    print(f"Inserted configurations for {selected_indicator} into the database.")

    print("Ready for correlation computations.")
    conn.close()

if __name__ == "__main__":
    main()
