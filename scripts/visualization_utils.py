# visualization_utils.py

import os
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Callable
from datetime import datetime
from joblib import Parallel, delayed
from scipy.stats import t, zscore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)


def generate_combined_correlation_chart(
    correlations: Dict[str, List[float]],
    max_lag: int,
    time_interval: str,
    timestamp: str,
    base_csv_filename: str,
    output_dir: str = 'combined_charts'
) -> None:
    """
    Generates a combined chart showing maximum positive, negative, and absolute correlations.

    Args:
        correlations (Dict[str, List[float]]): Correlation data.
        max_lag (int): Maximum lag.
        time_interval (str): Time interval string.
        timestamp (str): Timestamp string for filenames.
        base_csv_filename (str): Base CSV filename.
        output_dir (str): Directory to save the chart.
    """
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Ensured directory '{output_dir}' exists.")

    max_positive_correlations = []
    max_negative_correlations = []
    max_absolute_correlations = []

    for lag in range(1, max_lag + 1):
        lag_correlations = [
            correlations[col][lag - 1]
            for col in correlations
            if lag - 1 < len(correlations[col])
        ]

        pos_correlations = [x for x in lag_correlations if x > 0]
        max_pos = max(pos_correlations) if pos_correlations else 0

        neg_correlations = [x for x in lag_correlations if x < 0]
        max_neg = min(neg_correlations) if neg_correlations else 0

        max_abs = max(max_pos, abs(max_neg))

        logging.debug(f"Lag {lag}: Max Pos={max_pos}, Max Neg={max_neg}, Max Abs={max_abs}")

        max_positive_correlations.append(max_pos)
        max_negative_correlations.append(max_neg)
        max_absolute_correlations.append(max_abs)

    logging.info(f"First 5 Max Absolute Correlations: {max_absolute_correlations[:5]}")
    logging.info(f"Last 5 Max Absolute Correlations: {max_absolute_correlations[-5:]}")

    plt.figure(figsize=(15, 10))
    plt.plot(range(1, max_lag + 1), max_positive_correlations, color='green', label='Max Positive Correlation')
    plt.plot(range(1, max_lag + 1), max_negative_correlations, color='red', label='Max Negative Correlation')
    plt.plot(range(1, max_lag + 1), max_absolute_correlations, color='blue', label='Max Absolute Correlation')

    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.title('Maximum Positive, Negative, and Absolute Correlations at Each Lag Point', fontsize=14)
    plt.xlabel(f'Time Lag ({time_interval})', fontsize=12)
    plt.ylabel('Correlation', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(-1.0, 1.0)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    plt.tight_layout()

    combined_filename = f"{timestamp}_{base_csv_filename}_max_correlation.png"
    combined_filepath = os.path.join(output_dir, combined_filename)
    plt.savefig(combined_filepath, bbox_inches='tight')
    plt.close()
    logging.info(f"Generated maximum correlation chart in '{output_dir}' as '{combined_filename}'.")


def visualize_data(
    data: pd.DataFrame,
    features: pd.DataFrame,
    feature_columns: List[str],
    timestamp: str,
    is_reverse_chronological: bool,
    time_interval: str,
    generate_charts: bool,
    cache: Dict[str, Any],
    calculate_correlation_func: Callable[..., float],
    base_csv_filename: str
) -> None:
    """
    Generates individual indicator charts.

    Args:
        data (pd.DataFrame): The dataset.
        features (pd.DataFrame): Scaled features.
        feature_columns (List[str]): List of feature names.
        timestamp (str): Timestamp string.
        is_reverse_chronological (bool): Data ordering flag.
        time_interval (str): Time interval string.
        generate_charts (bool): Flag to generate charts.
        cache (Dict[str, Any]): Correlation cache.
        calculate_correlation_func (Callable): Function to calculate correlation.
        base_csv_filename (str): Base CSV filename.
    """
    if not generate_charts:
        logging.info("Chart generation is disabled. Skipping individual indicator charts.")
        return

    charts_dir = 'indicator_charts'
    os.makedirs(charts_dir, exist_ok=True)
    logging.info(f"Ensured directory '{charts_dir}' exists.")

    logging.info("Generating individual indicator charts...")
    max_lag = len(data) - 51
    if max_lag <= 0:
        logging.warning("Not enough data to calculate correlations with the specified max_lag.")
        return

    correlations = {}

    original_indicators = [
        col for col in feature_columns
        if not any(future in col for future in ['future_1d', 'future_5d', 'future_10d', 'future_20d'])
           and col != 'Close'
    ]

    original_indicators = [
        col for col in original_indicators
        if data[col].notna().any() and data[col].var() > 1e-6
    ]

    logging.info(f"Original indicators after filtering: {original_indicators}")

    for col in original_indicators:
        logging.info(f"Plotting {col} vs price...")
        if col not in cache:
            corr_list = Parallel(n_jobs=-1)(
                delayed(calculate_correlation_func)(data, col, lag, is_reverse_chronological)
                for lag in range(1, max_lag + 1)
            )
            cache[col] = corr_list
        else:
            corr_list = cache[col]
        correlations[col] = corr_list

        plt.figure(figsize=(10, 4))
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)

        plt.fill_between(range(1, max_lag + 1), corr_list, where=np.array(corr_list) > 0, color='blue', alpha=0.3)
        plt.fill_between(range(1, max_lag + 1), corr_list, where=np.array(corr_list) < 0, color='red', alpha=0.3)

        n = len(corr_list)
        if n > 1:
            std_err = np.std(corr_list, ddof=1) / np.sqrt(n)
            margin_of_error = t.ppf(0.975, n - 1) * std_err
            lower_bound = np.array(corr_list) - margin_of_error
            upper_bound = np.array(corr_list) + margin_of_error

            plt.fill_between(range(1, max_lag + 1), lower_bound, upper_bound, color='gray', alpha=0.4, label='95% CI')

        plt.title(f'Average Correlation of {col} with Close Price', fontsize=10)
        plt.xlabel(f'Time Lag ({time_interval})', fontsize=8)
        plt.ylabel('Average Correlation', fontsize=8)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        plt.ylim(-1.0, 1.0)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        if n > 1:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
        plt.tight_layout()
        filename = f"{timestamp}_{base_csv_filename}_{col}_correlation.png"
        filepath = os.path.join(charts_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
    logging.info(f"Generated individual indicator charts in '{charts_dir}'.")

    combined_charts_dir = 'combined_charts'
    os.makedirs(combined_charts_dir, exist_ok=True)
    logging.info(f"Ensured directory '{combined_charts_dir}' exists.")

    sorted_indicators = sorted(
        original_indicators,
        key=lambda col: correlations[col][-1] if len(correlations[col]) > 0 else 0,
        reverse=True
    )

    plt.figure(figsize=(15, 10))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(sorted_indicators)))
    for col, color in zip(sorted_indicators, colors):
        plt.plot(range(1, max_lag + 1), correlations[col], color=color, label=col)

    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.title('Average Correlation of All Indicators with Close Price', fontsize=14)
    plt.xlabel(f'Time Lag ({time_interval})', fontsize=12)
    plt.ylabel('Average Correlation', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(-1.0, 1.0)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    plt.tight_layout()
    combined_filename = f"{timestamp}_{base_csv_filename}_combined_correlation.png"
    combined_filepath = os.path.join(combined_charts_dir, combined_filename)
    plt.savefig(combined_filepath, bbox_inches='tight')
    plt.close()
    logging.info(f"Generated combined correlation chart in '{combined_charts_dir}'.")

    max_positive_correlations = []
    max_negative_correlations = []
    max_absolute_correlations = []

    for lag in range(1, max_lag + 1):
        lag_correlations = [correlations[col][lag - 1] for col in original_indicators]
        pos_values = [x for x in lag_correlations if x > 0]
        neg_values = [x for x in lag_correlations if x < 0]
        max_pos = max(pos_values) if pos_values else 0
        max_neg = min(neg_values) if neg_values else 0
        max_abs = max(max_pos, abs(max_neg))

        max_positive_correlations.append(max_pos)
        max_negative_correlations.append(max_neg)
        max_absolute_correlations.append(max_abs)

    logging.debug(f"Max Positive Correlations: {max_positive_correlations[:5]}")
    logging.debug(f"Max Negative Correlations: {max_negative_correlations[:5]}")
    logging.debug(f"Max Absolute Correlations: {max_absolute_correlations[:5]}")

    plt.figure(figsize=(15, 10))
    plt.plot(range(1, max_lag + 1), max_positive_correlations, color='green', label='Max Positive Correlation')
    plt.plot(range(1, max_lag + 1), max_negative_correlations, color='red', label='Max Negative Correlation')
    plt.plot(range(1, max_lag + 1), max_absolute_correlations, color='blue', label='Max Absolute Correlation')

    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.title('Maximum Positive, Negative, and Absolute Correlations at Each Lag Point', fontsize=14)
    plt.xlabel(f'Time Lag ({time_interval})', fontsize=12)
    plt.ylabel('Correlation', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(-1.0, 1.0)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    plt.tight_layout()
    combined_filename = f"{timestamp}_{base_csv_filename}_max_correlation.png"
    combined_filepath = os.path.join(combined_charts_dir, combined_filename)
    plt.savefig(combined_filepath, bbox_inches='tight')
    plt.close()
    logging.info(f"Generated maximum correlation chart in '{combined_charts_dir}'.")


def generate_heatmaps(
    data: pd.DataFrame, 
    timestamp: str, 
    time_interval: str, 
    generate_heatmaps_flag: bool, 
    cache: Dict[str, Any], 
    calculate_correlation: Callable[..., float], 
    base_csv_filename: str,
    delete_existing: bool = False,
    annotation: bool = False,
    max_indicators: int = None,
    output_format: str = 'png'
) -> None:
    """
    Generates heatmaps based on the provided data and parameters.

    Args:
        data (pd.DataFrame): DataFrame containing all data with indicators.
        timestamp (str): Current timestamp in YYYYMMDD-HHMMSS format for filename prefixing.
        time_interval (str): Time interval between rows (e.g., 'minute', 'hour', 'day', 'week').
        generate_heatmaps_flag (bool): Boolean indicating if heatmaps should be generated.
        cache (dict): Cache dictionary to store computed correlations.
        calculate_correlation (function): Function to calculate correlation for a given indicator and lag.
        base_csv_filename (str): Base filename of the original CSV file.
        delete_existing (bool, optional): Whether to delete existing heatmaps in the 'heatmaps' directory. Defaults to False.
        annotation (bool, optional): Whether to annotate heatmap cells with correlation values. Defaults to False.
        max_indicators (int, optional): Maximum number of indicators to include in the heatmaps. Defaults to None (include all).
        output_format (str, optional): File format for saved heatmaps (e.g., 'png', 'pdf'). Defaults to 'png'.
    """
    if not generate_heatmaps_flag:
        logging.info("Heatmap generation is disabled. Skipping heatmap creation.")
        return

    heatmaps_dir = 'heatmaps'
    os.makedirs(heatmaps_dir, exist_ok=True)
    existing_files = os.listdir(heatmaps_dir)

    if existing_files:
        if delete_existing:
            for file in existing_files:
                file_path = os.path.join(heatmaps_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            logging.info(f"Deleted existing heatmaps in '{heatmaps_dir}'.")
        else:
            logging.info(f"Existing files found in '{heatmaps_dir}' directory. Skipping deletion as 'delete_existing' is set to False.")

    original_indicators = [
        col for col in data.columns 
        if pd.api.types.is_numeric_dtype(data[col]) 
           and col != 'Close'
           and data[col].notna().any() 
           and data[col].var() > 1e-6
    ]

    logging.info(f"Original indicators in generate_heatmaps: {original_indicators}")

    max_lag = len(data) - 51
    if max_lag <= 0:
        logging.warning("Not enough data to calculate correlations with the specified max_lag.")
        return

    correlations = {}
    logging.info("Calculating correlations for each indicator...")

    def compute_correlation(col: str):
        if col not in cache:
            corr_list = [calculate_correlation(data, col, lag, False) for lag in range(1, max_lag + 1)]
            cache[col] = corr_list
            logging.debug(f"Calculated and cached correlations for {col}.")
        else:
            logging.debug(f"Retrieved cached correlations for {col}.")
        return cache[col]

    correlation_results = Parallel(n_jobs=-1)(
        delayed(compute_correlation)(col) for col in original_indicators
    )

    for col, corr_list in zip(original_indicators, correlation_results):
        correlations[col] = corr_list

    corr_df = pd.DataFrame(correlations, index=range(1, max_lag + 1))

    corr_df = corr_df.dropna(axis=1, how='all').dropna(axis=0, how='all')

    if corr_df.empty:
        logging.warning("Correlation DataFrame is empty after dropping NaN rows/columns. Exiting heatmap generation.")
        return

    standardized_corr_df = corr_df.apply(lambda x: zscore(x, nan_policy='omit'), axis=0)
    standardized_corr_df = standardized_corr_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    filtered_indicators = [
        col for col in standardized_corr_df.columns 
        if np.nanmax(np.abs(standardized_corr_df[col])) > 0.25
    ]
    standardized_corr_df = standardized_corr_df[filtered_indicators]

    if standardized_corr_df.empty:
        logging.warning("No indicators passed the correlation threshold. Exiting heatmap generation.")
        return

    if max_indicators is not None:
        filtered_indicators = filtered_indicators[:max_indicators]
        standardized_corr_df = standardized_corr_df[filtered_indicators]

    def earliest_max_lag(col: str) -> int:
        abs_corr = np.abs(standardized_corr_df[col])
        max_corr = np.nanmax(abs_corr)
        if max_corr == 0:
            return max_lag
        return abs_corr.idxmax()

    sorted_indicators = sorted(
        filtered_indicators, 
        key=lambda col: earliest_max_lag(col)
    )
    sorted_standardized_corr_df = standardized_corr_df[sorted_indicators]

    desired_label_count = 20
    step = max(1, max_lag // desired_label_count)

    x_ticks = range(0, max_lag, step)
    x_labels = [str(lag) for lag in range(1, max_lag + 1, step)]

    plt.figure(figsize=(20, max(10, len(sorted_indicators) * 0.3)), dpi=300)
    sns.heatmap(
        sorted_standardized_corr_df.T, 
        annot=annotation, 
        fmt=".2f" if annotation else None,
        cmap='RdBu_r', 
        cbar=True, 
        xticklabels=False,
        yticklabels=True,
        center=0,
        vmin=-3, vmax=3
    )
    plt.title('Standardized Correlation of Indicators with Close Price at Various Time Lags', fontsize=16)
    plt.xlabel(f'Time Lag ({time_interval})', fontsize=14)
    plt.ylabel('Indicators', fontsize=14)
    plt.xticks(ticks=np.arange(step/2, max_lag, step), labels=x_labels, rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    heatmap_filename_1 = f"{timestamp}_{base_csv_filename}_standardized_correlation_heatmap_sorted_earliest_max.{output_format}"
    heatmap_filepath_1 = os.path.join(heatmaps_dir, heatmap_filename_1)
    plt.savefig(heatmap_filepath_1, bbox_inches='tight')
    plt.close()
    logging.info(f"Generated standardized correlation heatmap sorted by earliest max correlation: '{heatmap_filename_1}'.")

    sorted_indicators_lag1 = sorted(
        filtered_indicators, 
        key=lambda col: standardized_corr_df[col].iloc[0], 
        reverse=True
    )
    sorted_standardized_corr_df_lag1 = standardized_corr_df[sorted_indicators_lag1]

    plt.figure(figsize=(20, max(10, len(sorted_indicators_lag1) * 0.3)), dpi=300)
    sns.heatmap(
        sorted_standardized_corr_df_lag1.T, 
        annot=annotation, 
        fmt=".2f" if annotation else None,
        cmap='RdBu_r', 
        cbar=True, 
        xticklabels=False, 
        yticklabels=True,
        center=0,
        vmin=-3, vmax=3
    )
    plt.title('Standardized Correlation of Indicators with Close Price at Various Time Lags (Sorted by Lag 1)', fontsize=16)
    plt.xlabel(f'Time Lag ({time_interval})', fontsize=14)
    plt.ylabel('Indicators', fontsize=14)
    plt.xticks(ticks=np.arange(step/2, max_lag, step), labels=x_labels, rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    heatmap_filename_2 = f"{timestamp}_{base_csv_filename}_standardized_correlation_heatmap_sorted_lag1.{output_format}"
    heatmap_filepath_2 = os.path.join(heatmaps_dir, heatmap_filename_2)
    plt.savefig(heatmap_filepath_2, bbox_inches='tight')
    plt.close()
    logging.info(f"Generated standardized correlation heatmap sorted by highest correlation at lag 1: '{heatmap_filename_2}'.")

    raw_corr_df = corr_df[filtered_indicators]
    sorted_indicators_raw = sorted(
        filtered_indicators, 
        key=lambda col: raw_corr_df[col].iloc[0], 
        reverse=True
    )
    sorted_raw_corr_df = raw_corr_df[sorted_indicators_raw]

    plt.figure(figsize=(20, max(10, len(sorted_indicators_raw) * 0.3)), dpi=300)
    sns.heatmap(
        sorted_raw_corr_df.T, 
        annot=annotation, 
        fmt=".2f" if annotation else None,
        cmap='RdBu_r', 
        cbar=True, 
        xticklabels=False, 
        yticklabels=True
    )
    plt.title('Raw Correlation of Indicators with Close Price at Various Time Lags (Sorted by Lag 1)', fontsize=16)
    plt.xlabel(f'Time Lag ({time_interval})', fontsize=14)
    plt.ylabel('Indicators', fontsize=14)
    plt.xticks(ticks=np.arange(step/2, max_lag, step), labels=x_labels, rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    heatmap_filename_3 = f"{timestamp}_{base_csv_filename}_raw_correlation_heatmap_sorted_lag1.{output_format}"
    heatmap_filepath_3 = os.path.join(heatmaps_dir, heatmap_filename_3)
    plt.savefig(heatmap_filepath_3, bbox_inches='tight')
    plt.close()
    logging.info(f"Generated raw correlation heatmap sorted by highest correlation at lag 1: '{heatmap_filename_3}'.")

    logging.info(f"Generated all requested heatmaps in '{heatmaps_dir}'.")


def calculate_correlation_example(data: pd.DataFrame, column: str, lag: int, reverse: bool = False) -> float:
    """
    Example correlation calculation function.

    Args:
        data (pd.DataFrame): The dataset.
        column (str): The column to calculate correlation with.
        lag (int): The lag value.
        reverse (bool, optional): Whether to reverse the lag direction. Defaults to False.

    Returns:
        float: Correlation coefficient.
    """
    if reverse:
        series1 = data[column].shift(lag)
        series2 = data['Close']
    else:
        series1 = data['Close'].shift(-lag)
        series2 = data[column]
    return series1.corr(series2)