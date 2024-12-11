# filename: generate_heatmaps.py

import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from typing import Callable, Dict, List

def generate_heatmaps(
    data: pd.DataFrame,
    timestamp: str,
    time_interval: str,
    generate_heatmaps_flag: bool,
    cache: Dict[str, List[float]],
    calculate_correlation: Callable[[pd.DataFrame, str, int, bool], float],
    base_csv_filename: str
) -> None:
    """
    Generates heatmaps based on correlation data retrieved from the database.

    Args:
        data (pd.DataFrame): DataFrame containing all data with indicators.
        timestamp (str): Current timestamp in YYYYMMDD-HHMMSS format for filename prefixing.
        time_interval (str): Time interval between rows (e.g., 'minute', 'hour', 'day', 'week').
        generate_heatmaps_flag (bool): Boolean indicating if heatmaps should be generated.
        cache (Dict[str, List[float]]): Cache dictionary to store computed correlations.
        calculate_correlation (Callable): Function to calculate correlation for a given indicator and lag.
        base_csv_filename (str): Base filename of the original CSV file.
    """
    if not generate_heatmaps_flag:
        logging.info("Heatmap generation flag is set to False. Skipping heatmap generation.")
        return

    heatmaps_dir = 'heatmaps'
    os.makedirs(heatmaps_dir, exist_ok=True)
    logging.info(f"Ensured directory '{heatmaps_dir}' exists.")

    # Optional: Handle existing heatmap files
    existing_files = os.listdir(heatmaps_dir)
    if existing_files:
        logging.info(f"Existing files found in '{heatmaps_dir}' directory.")
        delete_choice = input(f"Do you want to delete existing heatmaps in '{heatmaps_dir}'? (y/n): ").strip().lower()
        if delete_choice == 'y':
            for file in existing_files:
                file_path = os.path.join(heatmaps_dir, file)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.remove(file_path)
                        logging.info(f"Deleted file '{file_path}'.")
                    elif os.path.isdir(file_path):
                        os.rmdir(file_path)
                        logging.info(f"Deleted directory '{file_path}'.")
                except Exception as e:
                    logging.error(f"Failed to delete '{file_path}'. Reason: {e}")
            logging.info(f"Cleared existing heatmaps in '{heatmaps_dir}'.")
        else:
            logging.info(f"Retaining existing heatmaps in '{heatmaps_dir}'.")

    # Identify original indicators excluding 'Close'
    original_indicators = [
        col for col in data.columns 
        if pd.api.types.is_numeric_dtype(data[col]) 
           and col != 'Close'  # Exclude 'Close'
           and data[col].notna().any() 
           and data[col].var() > 1e-6
    ]

    logging.info(f"Original indicators identified for heatmap generation: {original_indicators}")
    print(f"Original indicators identified for heatmap generation: {original_indicators}")

    # Calculate max_lag based on data length and desired lag
    max_lag = len(data) - 51  # Adjust the subtraction based on your specific requirements
    if max_lag < 1:
        logging.error("Insufficient data length to compute correlations with the specified max_lag.")
        print("Insufficient data length to compute correlations with the specified max_lag.")
        return

    logging.info(f"Calculated max_lag: {max_lag}")

    # Calculate correlations for each indicator across lags using caching
    correlations = {}
    for col in original_indicators:
        if col not in cache:
            # Start lags from 1 to avoid lag=0
            corr_list = Parallel(n_jobs=-1)(
                delayed(calculate_correlation)(data, col, lag, False) 
                for lag in range(1, max_lag + 1)
            )
            cache[col] = corr_list
            logging.info(f"Calculated and cached correlations for indicator '{col}'.")
        else:
            corr_list = cache[col]
            logging.info(f"Loaded cached correlations for indicator '{col}'.")
        correlations[col] = corr_list

    # Create a DataFrame for the correlations
    corr_df = pd.DataFrame(correlations, index=range(1, max_lag + 1))  # Start from lag=1

    # Drop rows and columns with all NaN values to clean the data
    corr_df.dropna(axis=1, how='all', inplace=True)
    corr_df.dropna(axis=0, how='all', inplace=True)

    logging.info("Created correlation DataFrame and removed NaN-only rows and columns.")

    # Standardize each row (indicator) to have values between -1 and 1
    def standardize_row(row: pd.Series) -> pd.Series:
        if row.max() - row.min() == 0:
            return row * 0  # Avoid division by zero if all values are the same
        return (row - row.min()) / (row.max() - row.min()) * 2 - 1

    standardized_corr_df = corr_df.apply(standardize_row, axis=1)

    logging.info("Standardized correlation DataFrame.")

    # Filter indicators by max correlation exceeding 0.25
    filtered_indicators = [
        col for col in standardized_corr_df.columns 
        if standardized_corr_df[col].max() > 0.25
    ]

    standardized_corr_df = standardized_corr_df[filtered_indicators]
    logging.info(f"Filtered indicators based on max correlation > 0.25: {filtered_indicators}")

    # Sort indicators based on the earliest lag time where correlation is 1.0
    def earliest_one_cor(col: str) -> int:
        return next((i for i, x in enumerate(standardized_corr_df[col], start=1) if x == 1.0), max_lag + 1)

    sorted_indicators_1 = sorted(
        filtered_indicators, 
        key=earliest_one_cor
    )
    sorted_standardized_corr_df_1 = standardized_corr_df[sorted_indicators_1]

    logging.info("Sorted indicators based on earliest lag with correlation 1.0.")

    # Plot the first heatmap
    plt.figure(figsize=(20, 15), dpi=300)
    sns.heatmap(
        sorted_standardized_corr_df_1.T, 
        annot=False, 
        cmap='coolwarm', 
        cbar=True, 
        xticklabels=True, 
        yticklabels=True
    )
    plt.title('Standardized Correlation of Indicators with Close Price at Various Time Lags\n(Sorted by Earliest 1.0 Correlation)', fontsize=14)
    plt.xlabel(f'Time Lag ({time_interval})', fontsize=12)
    plt.ylabel('Indicators', fontsize=12)

    # Dynamic tick labeling to prevent mismatch
    plt.xticks(
        ticks=np.arange(max_lag) + 0.5, 
        labels=range(1, max_lag + 1), 
        rotation=90, 
        fontsize=6
    )
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()

    heatmap_filename_1 = f"{timestamp}_{base_csv_filename}_combined_correlation_heatmap_1.png"
    heatmap_filepath_1 = os.path.join(heatmaps_dir, heatmap_filename_1)
    plt.savefig(heatmap_filepath_1, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved heatmap 1 as '{heatmap_filepath_1}'.")

    # Sort indicators based on the highest correlation at lag time 1
    sorted_indicators_2 = sorted(
        filtered_indicators, 
        key=lambda col: standardized_corr_df[col].iloc[0], 
        reverse=True
    )
    sorted_standardized_corr_df_2 = standardized_corr_df[sorted_indicators_2]

    logging.info("Sorted indicators based on highest correlation at lag 1.")

    # Plot the second heatmap
    plt.figure(figsize=(20, 15), dpi=300)
    sns.heatmap(
        sorted_standardized_corr_df_2.T, 
        annot=False, 
        cmap='coolwarm', 
        cbar=True, 
        xticklabels=True, 
        yticklabels=True
    )
    plt.title('Standardized Correlation of Indicators with Close Price at Various Time Lags\n(Sorted by Highest Correlation at Lag 1)', fontsize=14)
    plt.xlabel(f'Time Lag ({time_interval})', fontsize=12)
    plt.ylabel('Indicators', fontsize=12)

    plt.xticks(
        ticks=np.arange(max_lag) + 0.5, 
        labels=range(1, max_lag + 1), 
        rotation=90, 
        fontsize=6
    )
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()

    heatmap_filename_2 = f"{timestamp}_{base_csv_filename}_combined_correlation_heatmap_2.png"
    heatmap_filepath_2 = os.path.join(heatmaps_dir, heatmap_filename_2)
    plt.savefig(heatmap_filepath_2, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved heatmap 2 as '{heatmap_filepath_2}'.")

    # Generate heatmap with raw correlation values
    raw_corr_df = corr_df[filtered_indicators]
    sorted_indicators_3 = sorted(
        filtered_indicators, 
        key=lambda col: raw_corr_df[col].iloc[0], 
        reverse=True
    )
    sorted_raw_corr_df = raw_corr_df[sorted_indicators_3]

    logging.info("Sorted indicators based on highest raw correlation at lag 1.")

    # Plot the third heatmap with raw correlation values
    plt.figure(figsize=(20, 15), dpi=300)
    sns.heatmap(
        sorted_raw_corr_df.T, 
        annot=False, 
        cmap='coolwarm', 
        cbar=True, 
        xticklabels=True, 
        yticklabels=True
    )
    plt.title('Raw Correlation of Indicators with Close Price at Various Time Lags\n(Sorted by Highest Correlation at Lag 1)', fontsize=14)
    plt.xlabel(f'Time Lag ({time_interval})', fontsize=12)
    plt.ylabel('Indicators', fontsize=12)

    plt.xticks(
        ticks=np.arange(max_lag) + 0.5, 
        labels=range(1, max_lag + 1), 
        rotation=90, 
        fontsize=6
    )
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()

    heatmap_filename_3 = f"{timestamp}_{base_csv_filename}_combined_correlation_heatmap_3.png"
    heatmap_filepath_3 = os.path.join(heatmaps_dir, heatmap_filename_3)
    plt.savefig(heatmap_filepath_3, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved heatmap 3 as '{heatmap_filepath_3}'.")

    logging.info(f"Generated all combined correlation heatmaps in '{heatmaps_dir}' directory.")
    print(f"Heatmaps have been successfully generated and saved in '{heatmaps_dir}' directory.")