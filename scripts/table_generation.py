# table_generation.py

import os
import pandas as pd
import numpy as np
import logging

def generate_best_indicator_table(correlations: dict, max_lag: int) -> pd.DataFrame:
    """
    Generates a table of the best indicators based on maximum absolute correlation across all lags.

    Parameters:
    - correlations: Dictionary with indicator names as keys and list of correlations as values.
    - max_lag: Maximum lag considered.

    Returns:
    - DataFrame containing the best indicators and their correlation details.
    """
    try:
        best_indicators = []
        for indicator, corr_values in correlations.items():
            abs_corr = [abs(c) if not pd.isna(c) else 0 for c in corr_values]
            if not abs_corr:
                continue
            max_abs = max(abs_corr)
            if max_abs == 0:
                continue
            lag = abs_corr.index(max_abs) + 1
            best_indicators.append({
                'Indicator': indicator,
                'Max_Absolute_Correlation': max_abs,
                'Lag': lag
            })
        
        best_indicators_df = pd.DataFrame(best_indicators)
        
        if best_indicators_df.empty:
            logging.warning("No valid indicators found for the best indicator table.")
            return best_indicators_df
        
        best_indicators_df.sort_values(by='Max_Absolute_Correlation', ascending=False, inplace=True)
        logging.info("Generated best indicator table.")
        return best_indicators_df
    except Exception as e:
        logging.error(f"Error generating best indicator table: {e}")
        return pd.DataFrame()

def generate_statistical_summary(correlations: dict, max_lag: int) -> pd.DataFrame:
    """
    Generates a statistical summary of correlations, including mean, median, and standard deviation.

    Parameters:
    - correlations: Dictionary with indicator names as keys and list of correlations as values.
    - max_lag: Maximum lag considered.

    Returns:
    - DataFrame containing the statistical summary.
    """
    try:
        summary = []
        for indicator, corr_values in correlations.items():
            valid_corr = [c for c in corr_values if not pd.isna(c)]
            if not valid_corr:
                continue
            summary.append({
                'Indicator': indicator,
                'Mean_Correlation': np.mean(valid_corr),
                'Median_Correlation': np.median(valid_corr),
                'Std_Deviation': np.std(valid_corr)
            })
        summary_df = pd.DataFrame(summary)
        
        if summary_df.empty:
            logging.warning("No valid indicators found for the statistical summary.")
            return summary_df
        
        logging.info("Generated statistical summary of correlations.")
        return summary_df
    except Exception as e:
        logging.error(f"Error generating statistical summary: {e}")
        return pd.DataFrame()

def generate_correlation_csv(correlations: dict, max_lag: int, base_csv_filename: str, csv_dir: str) -> None:
    """
    Saves the correlation data to a CSV file.

    Parameters:
    - correlations: Dictionary with indicator names as keys and list of correlations as values.
    - max_lag: Maximum lag considered.
    - base_csv_filename: Base name derived from symbol and timeframe.
    - csv_dir: Directory to save the CSV file.
    """
    try:
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir, exist_ok=True)
            logging.info(f"Created CSV directory at {csv_dir}.")
        
        corr_df = pd.DataFrame(correlations)
        
        if corr_df.empty:
            logging.warning("Correlation DataFrame is empty. Skipping CSV generation.")
            return
        
        corr_df.index = range(1, max_lag + 1)
        corr_df.index.name = 'Lag'
        
        csv_filepath = os.path.join(csv_dir, f"{base_csv_filename}_correlations.csv")
        corr_df.to_csv(csv_filepath)
        logging.info(f"Saved correlation CSV at {csv_filepath}.")
    except Exception as e:
        logging.error(f"Error generating correlation CSV: {e}")