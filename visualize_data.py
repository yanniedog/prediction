# Filename: visualize_data.py

import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import numpy as np
from scipy.stats import t
from joblib import Parallel, delayed

def visualize_data(data, features, feature_columns, timestamp, is_reverse_chronological, time_interval, generate_charts, cache, calculate_correlation, base_csv_filename):
    """
    Generates and saves visualizations including average correlation plots for each indicator.

    Args:
        data (pd.DataFrame): Original DataFrame containing 'Date' and other non-feature columns.
        features (pd.DataFrame): DataFrame containing feature columns (original or PCA components).
        feature_columns (list): List of feature column names.
        timestamp (str): Current timestamp in YYYYMMDD-HHMMSS format for filename prefixing.
        is_reverse_chronological (bool): Boolean indicating if the data is in reverse chronological order.
        time_interval (str): Time interval between rows (e.g., 'minute', 'hour', 'day', 'week').
        generate_charts (bool): Boolean indicating if individual indicator charts should be generated.
        cache (dict): Cache dictionary to store computed correlations.
        calculate_correlation (function): Function to calculate correlation for a given indicator and lag.
        base_csv_filename (str): Base filename of the original CSV file.
    """
    if not generate_charts:
        return

    # Create directory for indicator charts
    charts_dir = 'indicator_charts'
    if not os.path.exists(charts_dir):
        os.makedirs(charts_dir)
    else:
        existing_files = os.listdir(charts_dir)
        if existing_files:
            logging.info(f"Existing files found in '{charts_dir}' directory.")
            delete_choice = input(f"Do you want to delete them? (y/n): ").lower()
            if delete_choice == 'y':
                for file in existing_files:
                    file_path = os.path.join(charts_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                logging.info(f"Deleted existing charts in '{charts_dir}'.")
            else:
                logging.info(f"Retaining existing charts in '{charts_dir}'.")

    logging.info("Generating individual indicator charts...")
    # Calculate average correlations for each indicator
    max_lag = len(data) - 51  # Maximum lag to consider (50 less than total rows)
    correlations = {}

    # Filter out original indicators, explicitly excluding 'Close'
    original_indicators = [
        col for col in feature_columns 
        if not any(future in col for future in ['future_1d', 'future_5d', 'future_10d', 'future_20d'])
           and col != 'Close'  # Exclude 'Close'
    ]

    # Remove indicators with all NaN values or very low variance
    original_indicators = [
        col for col in original_indicators 
        if data[col].notna().any() and data[col].var() > 1e-6
    ]

    # Debug statement to confirm exclusion
    print(f"Original indicators in visualize_data.py: {original_indicators}")

    # Parallelize correlation calculations
    for col in original_indicators:
        logging.info(f"Plotting {col} vs price...")
        if col not in cache:
            # Start lags from 1 to avoid lag=0
            corr_list = Parallel(n_jobs=-1)(
                delayed(calculate_correlation)(data, col, lag, is_reverse_chronological) 
                for lag in range(1, max_lag + 1)
            )
            cache[col] = corr_list
        else:
            corr_list = cache[col]
        correlations[col] = corr_list

        # Plot each indicator's average correlation
        plt.figure(figsize=(10, 4))
        plt.axhline(0, color='black', linewidth=0.5)  # Draw a line through the 0 point
        plt.axvline(0, color='black', linewidth=0.5)  # Draw a line through the 0 timepoint

        # Plot positive correlations in blue and negative correlations in red
        plt.fill_between(range(1, max_lag + 1), corr_list, where=np.array(corr_list) > 0, color='blue', alpha=0.3)
        plt.fill_between(range(1, max_lag + 1), corr_list, where=np.array(corr_list) < 0, color='red', alpha=0.3)

        # Calculate 95% confidence interval
        n = len(corr_list)
        if n > 1:
            std_err = np.std(corr_list, ddof=1) / np.sqrt(n)
            margin_of_error = t.ppf(0.975, n - 1) * std_err
            lower_bound = np.array(corr_list) - margin_of_error
            upper_bound = np.array(corr_list) + margin_of_error

            # Plot the 95% confidence interval as a ribbon
            plt.fill_between(range(1, max_lag + 1), lower_bound, upper_bound, color='gray', alpha=0.4, label='95% CI')

        plt.title(f'Average Correlation of {col} with Close Price', fontsize=10)
        plt.xlabel(f'Time Lag ({time_interval})', fontsize=8)
        plt.ylabel('Average Correlation', fontsize=8)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        plt.ylim(-1.0, 1.0)  # Set y-axis limits from -1.0 to 1.0
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add gridlines
        plt.tight_layout()
        filename = f"{timestamp}_{base_csv_filename}_{col}_correlation.png"
        filepath = os.path.join(charts_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
    logging.info(f"Generated individual indicator charts in '{charts_dir}'.")

    # Generate combined plot with all indicators overlaid
    combined_charts_dir = 'combined_charts'
    if not os.path.exists(combined_charts_dir):
        os.makedirs(combined_charts_dir)

    # Sort indicators based on their correlation at the highest lag point
    sorted_indicators = sorted(
        original_indicators, 
        key=lambda col: correlations[col][-1] if len(correlations[col]) > 0 else 0, 
        reverse=True
    )

    plt.figure(figsize=(15, 10))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(sorted_indicators)))
    for col, color in zip(sorted_indicators, colors):
        plt.plot(range(1, max_lag + 1), correlations[col], color=color, label=col)

    plt.axhline(0, color='black', linewidth=0.5)  # Draw a line through the 0 point
    plt.axvline(0, color='black', linewidth=0.5)  # Draw a line through the 0 timepoint
    plt.title('Average Correlation of All Indicators with Close Price', fontsize=14)
    plt.xlabel(f'Time Lag ({time_interval})', fontsize=12)
    plt.ylabel('Average Correlation', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(-1.0, 1.0)  # Set y-axis limits from -1.0 to 1.0
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add gridlines
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    plt.tight_layout()
    combined_filename = f"{timestamp}_{base_csv_filename}_combined_correlation.png"
    combined_filepath = os.path.join(combined_charts_dir, combined_filename)
    plt.savefig(combined_filepath, bbox_inches='tight')
    plt.close()
    logging.info(f"Generated combined correlation chart in '{combined_charts_dir}'.")

    # Generate combined plot showing maximum positive and negative correlations at each lag point
    max_positive_correlations = []
    max_negative_correlations = []

    for lag in range(1, max_lag + 1):
        lag_correlations = [correlations[col][lag - 1] for col in original_indicators]
        max_positive_correlations.append(max(lag_correlations, key=lambda x: x if x > 0 else -np.inf))
        max_negative_correlations.append(min(lag_correlations, key=lambda x: x if x < 0 else np.inf))

    plt.figure(figsize=(15, 10))
    plt.plot(range(1, max_lag + 1), max_positive_correlations, color='green', label='Max Positive Correlation')
    plt.plot(range(1, max_lag + 1), max_negative_correlations, color='red', label='Max Negative Correlation')

    plt.axhline(0, color='black', linewidth=0.5)  # Draw a line through the 0 point
    plt.axvline(0, color='black', linewidth=0.5)  # Draw a line through the 0 timepoint
    plt.title('Maximum Positive and Negative Correlations at Each Lag Point', fontsize=14)
    plt.xlabel(f'Time Lag ({time_interval})', fontsize=12)
    plt.ylabel('Correlation', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(-1.0, 1.0)  # Set y-axis limits from -1.0 to 1.0
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add gridlines
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    plt.tight_layout()
    combined_filename = f"{timestamp}_{base_csv_filename}_max_correlation.png"
    combined_filepath = os.path.join(combined_charts_dir, combined_filename)
    plt.savefig(combined_filepath, bbox_inches='tight')
    plt.close()
    logging.info(f"Generated maximum correlation chart in '{combined_charts_dir}'.")