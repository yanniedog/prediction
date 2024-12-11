# generate_heatmaps.py

import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import logging

def generate_heatmaps(
    data: pd.DataFrame, 
    timestamp: str, 
    time_interval: str, 
    generate_heatmaps_flag: bool, 
    correlations: dict, 
    calculate_correlation, 
    base_csv_filename: str
) -> None:
    """
    Generates and saves heatmaps of correlations.

    Parameters:
    - data: Original DataFrame containing the data.
    - timestamp: Current timestamp for filename uniqueness.
    - time_interval: Time interval string (e.g., '1w').
    - generate_heatmaps_flag: Boolean indicating whether to generate heatmaps.
    - correlations: Dictionary with indicator names as keys and list of correlations as values.
    - calculate_correlation: Function to calculate correlation.
    - base_csv_filename: Base name derived from symbol and timeframe.
    """
    if not generate_heatmaps_flag:
        logging.info("Heatmap generation flag is False. Skipping heatmap generation.")
        return

    try:
        corr_df = pd.DataFrame(correlations)
        
        if corr_df.empty:
            logging.warning("Correlation DataFrame is empty. Skipping heatmap generation.")
            return
        
        corr_matrix = corr_df.corr()

        if corr_matrix.empty or corr_matrix.shape[0] < 2:
            logging.warning("Correlation matrix is empty or has insufficient data. Skipping heatmap generation.")
            return
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=.5)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        
        heatmap_dir = 'heatmaps'
        os.makedirs(heatmap_dir, exist_ok=True)
        heatmap_filename = f"{timestamp}_{base_csv_filename}_correlation_heatmap.png"
        plt.savefig(os.path.join(heatmap_dir, heatmap_filename))
        plt.close()
        logging.info(f"Generated heatmap at {os.path.join(heatmap_dir, heatmap_filename)}.")
    except Exception as e:
        logging.error(f"Error generating heatmap: {e}")
