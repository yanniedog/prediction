# scripts/visualization_utils.py

import os
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict
import logging
import numpy as np
from scipy.stats import t

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_individual_indicator_chart(
    indicator_name: str, 
    correlations: List[float], 
    max_lag: int, 
    timestamp: str, 
    base_csv_filename: str
) -> None:
    """
    Generates and saves an individual indicator chart that visualizes correlation across all lags,
    including area fills for positive/negative correlations and confidence intervals.

    Parameters:
    - indicator_name: Name of the indicator.
    - correlations: List of correlation values across lags.
    - max_lag: Maximum lag considered.
    - timestamp: Current timestamp for filename uniqueness.
    - base_csv_filename: Base name derived from symbol and timeframe.
    """
    charts_dir = 'indicator_charts'
    os.makedirs(charts_dir, exist_ok=True)
    
    lags = list(range(1, max_lag + 1))
    corr_array = np.array(correlations)
    
    plt.figure(figsize=(10, 6))
    
    # Plot correlation line
    plt.plot(lags, corr_array, marker='o', linestyle='-', color='blue', label='Correlation')
    
    # Area fill based on positive or negative correlation
    plt.fill_between(lags, corr_array, where=corr_array > 0, color='blue', alpha=0.3, interpolate=False, label='Positive Correlation')
    plt.fill_between(lags, corr_array, where=corr_array < 0, color='red', alpha=0.3, interpolate=False, label='Negative Correlation')
    
    # Calculate Confidence Intervals (95% CI)
    n = len(corr_array)
    if n > 1:
        std_err = np.std(corr_array, ddof=1) / np.sqrt(n)
        margin_of_error = t.ppf(0.975, n - 1) * std_err
        lower_bound = corr_array - margin_of_error
        upper_bound = corr_array + margin_of_error
        plt.fill_between(lags, lower_bound, upper_bound, color='gray', alpha=0.2, label='95% Confidence Interval')
    
    plt.title(f'Correlation of {indicator_name} with Close Price Across Lags')
    plt.xlabel('Lag')
    plt.ylabel('Correlation')
    plt.ylim(-1, 1)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.xticks(lags)
    plt.legend(loc='upper right', fontsize=8)
    
    filename = f"{timestamp}_{base_csv_filename}_{indicator_name}_correlation_across_lags.png"
    filepath = os.path.join(charts_dir, filename)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    logging.info(f"Generated individual indicator chart for '{indicator_name}' at '{filepath}'.")

def generate_combined_correlation_chart(
    correlations: Dict[str, List[float]], 
    max_lag: int, 
    time_interval: str, 
    timestamp: str, 
    base_csv_filename: str, 
    output_dir: str = 'combined_charts'
) -> None:
    """
    Generates and saves a combined correlation chart showing max positive, negative, and absolute correlations across all indicators.

    Parameters:
    - correlations: Dictionary with indicator names as keys and list of correlations as values.
    - max_lag: Maximum lag considered.
    - time_interval: Time interval string (e.g., '1w').
    - timestamp: Current timestamp for filename uniqueness.
    - base_csv_filename: Base name derived from symbol and timeframe.
    - output_dir: Directory to save the combined chart.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    max_positive_correlations = []
    max_negative_correlations = []
    max_absolute_correlations = []
    
    for lag in range(1, max_lag + 1):
        lag_correlations = [correlations[col][lag - 1] for col in correlations if lag - 1 < len(correlations[col])]
        pos_cor = [x for x in lag_correlations if x > 0]
        neg_cor = [x for x in lag_correlations if x < 0]
        max_pos = max(pos_cor) if pos_cor else 0
        max_neg = min(neg_cor) if neg_cor else 0
        max_abs = max(max_pos, abs(max_neg))
        max_positive_correlations.append(max_pos)
        max_negative_correlations.append(max_neg)
        max_absolute_correlations.append(max_abs)
    
    plt.figure(figsize=(15, 10))
    plt.plot(range(1, max_lag + 1), max_positive_correlations, color='green', label='Max Positive Correlation')
    plt.plot(range(1, max_lag + 1), max_negative_correlations, color='red', label='Max Negative Correlation')
    plt.plot(range(1, max_lag + 1), max_absolute_correlations, color='blue', label='Max Absolute Correlation')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.title('Maximum Positive, Negative, and Absolute Correlations at Each Lag', fontsize=14)
    plt.xlabel(f'Time Lag ({time_interval})', fontsize=12)
    plt.ylabel('Correlation', fontsize=12)
    plt.ylim(-1.0, 1.0)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    combined_filename = f"{timestamp}_{base_csv_filename}_max_correlation.png"
    combined_filepath = os.path.join(output_dir, combined_filename)
    plt.savefig(combined_filepath, bbox_inches='tight')
    plt.close()
    logging.info(f"Generated combined correlation chart at '{combined_filepath}'.")

def visualize_data(
    data: pd.DataFrame, 
    features: pd.DataFrame, 
    feature_columns: List[str], 
    timestamp: str, 
    is_reverse_chronological: bool, 
    time_interval: str, 
    generate_charts: bool, 
    correlations: Dict[str, List[float]], 
    calculate_correlation_func: Any, 
    base_csv_filename: str
) -> None:
    """
    Visualizes data by generating individual and combined correlation charts.

    Parameters:
    - data: Original DataFrame containing the data.
    - features: Scaled features DataFrame.
    - feature_columns: List of feature column names.
    - timestamp: Current timestamp for filename uniqueness.
    - is_reverse_chronological: Boolean indicating if the data is in reverse chronological order.
    - time_interval: Time interval string (e.g., '1w').
    - generate_charts: Boolean indicating whether to generate charts.
    - correlations: Dictionary with indicator names as keys and list of correlations as values.
    - calculate_correlation_func: Function to calculate correlation.
    - base_csv_filename: Base name derived from symbol and timeframe.
    """
    if not generate_charts:
        logging.info("Chart generation is disabled. Skipping visualization.")
        return
    
    try:
        # Generate individual indicator charts
        for indicator, corr_values in correlations.items():
            generate_individual_indicator_chart(
                indicator_name=indicator,
                correlations=corr_values,
                max_lag=len(corr_values),
                timestamp=timestamp,
                base_csv_filename=base_csv_filename
            )
        logging.info("All individual indicator charts generated successfully.")

        # Generate combined correlation chart
        generate_combined_correlation_chart(
            correlations=correlations,
            max_lag=len(next(iter(correlations.values()))) if correlations else 0,
            time_interval=time_interval,
            timestamp=timestamp,
            base_csv_filename=base_csv_filename
        )
        logging.info("Combined correlation chart generated successfully.")
    except Exception as e:
        logging.error(f"Error during visualization: {e}")
