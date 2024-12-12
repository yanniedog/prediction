import os, pandas as pd, numpy as np, logging

def generate_statistical_summary(correlations: dict, max_lag: int) -> pd.DataFrame:
    summary = {ind: {'mean': np.mean(vals), 'std': np.std(vals), 'min': np.min(vals), 'max': np.max(vals)} 
               for ind, vals in correlations.items() if vals and any(pd.notna(v) for v in vals)}
    return pd.DataFrame(summary).T

def generate_best_indicator_table(correlations: dict, max_lag: int) -> pd.DataFrame:
    summary_df = generate_statistical_summary(correlations, max_lag)
    return summary_df.sort_values(by='mean', ascending=False).head(10) if not summary_df.empty else pd.DataFrame()

def generate_correlation_csv(correlations: dict, max_lag: int, base_filename: str, csv_dir: str) -> None:
    df = pd.DataFrame(correlations, index=range(1, max_lag+1))
    os.makedirs(csv_dir, exist_ok=True)
    df.to_csv(os.path.join(csv_dir, f"{base_filename}_correlations.csv"), index_label='Lag')
