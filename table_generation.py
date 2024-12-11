# filename: table_generation.py

import os
import pandas as pd
import numpy as np
import logging

def generate_statistical_summary(correlations: dict, max_lag: int) -> pd.DataFrame:
    """
    Generates a statistical summary (mean, std, min, max) for each indicator's correlations.

    Args:
        correlations (dict): Dictionary with indicator names as keys and list of correlation values as values.
        max_lag (int): Maximum lag period.

    Returns:
        pd.DataFrame: DataFrame containing statistical summaries for each indicator.
    """
    summary = {}
    for indicator, values in correlations.items():
        # Remove None or NaN values
        clean_values = [v for v in values if v is not None and not pd.isna(v)]
        if not clean_values:
            logging.warning(f"No valid correlation values for indicator '{indicator}'. Skipping.")
            continue
        summary[indicator] = {
            'mean': np.mean(clean_values),
            'std': np.std(clean_values),
            'min': np.min(clean_values),
            'max': np.max(clean_values)
        }
    summary_df = pd.DataFrame(summary).T
    return summary_df

def generate_best_indicator_table(correlations: dict, max_lag: int) -> pd.DataFrame:
    """
    Generates a table of best indicators based on the highest mean correlation.

    Args:
        correlations (dict): Dictionary with indicator names as keys and list of correlation values as values.
        max_lag (int): Maximum lag period.

    Returns:
        pd.DataFrame: DataFrame containing the best indicators.
    """
    summary_df = generate_statistical_summary(correlations, max_lag)
    if summary_df.empty:
        logging.warning("No data available to generate best indicators table.")
        return pd.DataFrame()
    
    # Sort indicators by mean correlation in descending order
    summary_df_sorted = summary_df.sort_values(by='mean', ascending=False)
    return summary_df_sorted.head(10)  # Assuming top 10 indicators are desired

def generate_correlation_csv(correlations: dict, max_lag: int, base_filename: str, csv_dir: str) -> None:
    """
    Generates a CSV file containing all correlation values for each indicator and lag.

    Args:
        correlations (dict): Dictionary with indicator names as keys and list of correlation values as values.
        max_lag (int): Maximum lag period.
        base_filename (str): Base name for the CSV file.
        csv_dir (str): Directory where the CSV should be saved.

    Returns:
        None
    """
    # Create DataFrame
    correlation_df = pd.DataFrame(correlations)
    # Assign lag periods as the index
    correlation_df.index = range(1, max_lag + 1)
    # Ensure the CSV directory exists
    os.makedirs(csv_dir, exist_ok=True)
    # Define file path
    file_path = os.path.join(csv_dir, f"{base_filename}_correlations.csv")
    # Save to CSV
    correlation_df.to_csv(file_path, index_label='Lag')