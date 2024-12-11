# Filename: table_generation.py

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List

def generate_best_indicator_table(correlations: Dict[str, List[float]], max_lag: int) -> pd.DataFrame:
    """
    Generates a table indicating which indicator is the best for each individual lag period.

    Args:
        correlations (Dict[str, List[float]]): Dictionary containing correlations for each indicator.
        max_lag (int): Maximum lag to consider.

    Returns:
        pd.DataFrame: DataFrame containing the best indicator for each lag period.
    """
    best_indicators = {}
    for lag in range(1, max_lag + 1):
        # Find the indicator with the highest correlation at this lag
        best_indicator = max(
            correlations,
            key=lambda col: correlations[col][lag - 1] if lag - 1 < len(correlations[col]) else -np.inf
        )
        best_indicators[lag] = best_indicator

    best_indicators_df = pd.DataFrame(list(best_indicators.items()), columns=['Lag', 'Best Indicator'])
    logging.info("Best indicator table generated.")
    return best_indicators_df

def generate_statistical_summary(correlations: Dict[str, List[float]], max_lag: int) -> pd.DataFrame:
    """
    Generates a statistical summary page outlining the performance of each individual indicator.

    Args:
        correlations (Dict[str, List[float]]): Dictionary containing correlations for each indicator.
        max_lag (int): Maximum lag to consider.

    Returns:
        pd.DataFrame: DataFrame containing the statistical summary.
    """
    summary = {}
    for col in correlations:
        best_count = sum(
            1 for lag in range(1, max_lag + 1)
            if lag - 1 < len(correlations[col]) and correlations[col][lag - 1] == max(
                correlations[c][lag - 1] if lag - 1 < len(correlations[c]) else -np.inf
                for c in correlations
            )
        )
        valid_correlations = [corr for corr in correlations[col] if not pd.isna(corr)]
        average_corr = np.mean(valid_correlations) if valid_correlations else np.nan
        max_corr = np.max(valid_correlations) if valid_correlations else np.nan
        min_corr = np.min(valid_correlations) if valid_correlations else np.nan

        summary[col] = {
            'Best Count': best_count,
            'Average Correlation': average_corr,
            'Max Correlation': max_corr,
            'Min Correlation': min_corr
        }

    summary_df = pd.DataFrame(summary).T
    summary_df['Overall Performance Score'] = summary_df['Average Correlation'] * summary_df['Best Count']
    summary_df.sort_values(by='Overall Performance Score', ascending=False, inplace=True)
    logging.info("Statistical summary generated.")
    return summary_df

def generate_correlation_csv(
    correlations: Dict[str, List[float]],
    max_lag: int,
    base_csv_filename: str,
    reports_dir: str
) -> None:
    """
    Generates a CSV file containing each indicator's correlation values for each lag point.

    Args:
        correlations (Dict[str, List[float]]): Correlation data.
        max_lag (int): Maximum lag.
        base_csv_filename (str): Base CSV filename.
        reports_dir (str): Directory to save the CSV.
    """
    correlation_data = {}
    for col, corr_list in correlations.items():
        for lag, corr in enumerate(corr_list, start=1):
            correlation_data.setdefault(f"Lag_{lag}", {})[col] = corr
    correlation_df = pd.DataFrame(correlation_data).T
    correlation_csv = os.path.join(reports_dir, 'csv', f"{base_csv_filename}_correlation_table.csv")
    correlation_df.to_csv(correlation_csv, index=True)
    print(f"Generated correlation table: {correlation_csv}")
    logging.info(f"Correlation table saved as '{correlation_csv}'.")