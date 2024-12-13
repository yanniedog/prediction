# generate_heatmaps.py
import logging
import os, matplotlib.pyplot as plt, seaborn as sns, pandas as pd, numpy as np
from joblib import Parallel, delayed
from typing import Callable, Dict, List

logger = logging.getLogger()

def generate_heatmaps(data: pd.DataFrame, timestamp: str, time_interval: str, flag: bool, cache: Dict[str, List[float]], calc_corr: Callable[[pd.DataFrame, str, int, bool], float], base_csv: str) -> None:
    if not flag: return
    os.makedirs('heatmaps', exist_ok=True)
    if os.listdir('heatmaps') and input("Delete existing heatmaps? (y/n): ").strip().lower() == 'y':
        for f in os.listdir('heatmaps'): os.remove(os.path.join('heatmaps', f))
    indicators = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col]) and col != 'Close' and data[col].notna().any() and data[col].var() >1e-6]
    max_lag = len(data) -51
    if max_lag <1: print("Insufficient data for correlations."); return
    correlations = {col: cache.get(col, Parallel(n_jobs=-1)(delayed(calc_corr)(data, col, lag, False) for lag in range(1, max_lag+1))) for col in indicators}
    cache.update(correlations)
    corr_df = pd.DataFrame(correlations, index=range(1, max_lag+1)).dropna(axis=1).dropna(axis=0)
    std_corr = corr_df.apply(lambda row: 0 if row.max()-row.min()==0 else (row-row.min())/(row.max()-row.min())*2-1, axis=1)
    filtered = [col for col in std_corr.columns if std_corr[col].max() >0.25]
    std_corr = std_corr[filtered]
    sorted_inds = sorted(filtered, key=lambda c: next((i for i, x in enumerate(std_corr[c],1) if x==1.0), max_lag+1))
    sns.heatmap(std_corr[sorted_inds].T, cmap='coolwarm', cbar=True, xticklabels=True, yticklabels=True)
    plt.title('Standardized Correlation of Indicators with Close Price at Various Lags\n(Sorted by Earliest 1.0 Correlation)', fontsize=14)
    plt.xlabel(f'Time Lag ({time_interval})', fontsize=12)
    plt.ylabel('Indicators', fontsize=12)
    plt.xticks(ticks=np.arange(max_lag)+0.5, labels=range(1,max_lag+1), rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join('heatmaps', f"{timestamp}_{base_csv}_heatmap1.png"), bbox_inches='tight')
    plt.close()
    sorted_inds = sorted(filtered, key=lambda c: std_corr[c].iloc[0], reverse=True)
    sns.heatmap(std_corr[sorted_inds].T, cmap='coolwarm', cbar=True, xticklabels=True, yticklabels=True)
    plt.title('Standardized Correlation Sorted by Highest Correlation at Lag 1', fontsize=14)
    plt.xlabel(f'Time Lag ({time_interval})', fontsize=12)
    plt.ylabel('Indicators', fontsize=12)
    plt.xticks(ticks=np.arange(max_lag)+0.5, labels=range(1,max_lag+1), rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join('heatmaps', f"{timestamp}_{base_csv}_heatmap2.png"), bbox_inches='tight')
    plt.close()
    sns.heatmap(corr_df[filtered].T, cmap='coolwarm', cbar=True, xticklabels=True, yticklabels=True)
    plt.title('Raw Correlation Sorted by Highest Correlation at Lag 1', fontsize=14)
    plt.xlabel(f'Time Lag ({time_interval})', fontsize=12)
    plt.ylabel('Indicators', fontsize=12)
    plt.xticks(ticks=np.arange(max_lag)+0.5, labels=range(1,max_lag+1), rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join('heatmaps', f"{timestamp}_{base_csv}_heatmap3.png"), bbox_inches='tight')
    plt.close()
    print("Heatmaps generated.")
