import os
import pandas as pd
import numpy as np
import logging

def generate_statistical_summary(correlations: dict, max_lag: int) -> pd.DataFrame:
    summary = {}
    for indicator, values in correlations.items():
        clean_values = [v for v in values if v is not None and not pd.isna(v)]
        if not clean_values:
            continue
        summary[indicator] = {'mean': np.mean(clean_values),'std': np.std(clean_values),'min': np.min(clean_values),'max': np.max(clean_values)}
    return pd.DataFrame(summary).T

def generate_best_indicator_table(correlations: dict, max_lag: int) -> pd.DataFrame:
    summary_df = generate_statistical_summary(correlations, max_lag)
    if summary_df.empty:
        return pd.DataFrame()
    summary_df_sorted = summary_df.sort_values(by='mean', ascending=False)
    return summary_df_sorted.head(10)

def generate_correlation_csv(correlations: dict, max_lag: int, base_filename: str, csv_dir: str) -> None:
    correlation_df = pd.DataFrame(correlations)
    correlation_df.index = range(1, max_lag + 1)
    os.makedirs(csv_dir, exist_ok=True)
    file_path = os.path.join(csv_dir, f"{base_filename}_correlations.csv")
    correlation_df.to_csv(file_path, index_label='Lag')
