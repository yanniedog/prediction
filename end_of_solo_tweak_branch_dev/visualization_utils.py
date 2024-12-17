# visualization_utils.py

import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from typing import Any, Callable, Dict, List
from scipy.stats import t

logger = logging.getLogger()

def generate_combined_correlation_chart(
    correlations: Dict[str, List[float]],
    max_lag: int,
    time_interval: str,
    timestamp: str,
    base_csv: str,
    output_dir: str = 'combined_charts'
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    max_pos = [
        max(
            [
                correlations[col][lag - 1]
                for col in correlations
                if lag - 1 < len(correlations[col]) and correlations[col][lag - 1] > 0
            ],
            default=0
        )
        for lag in range(1, max_lag + 1)
    ]
    max_neg = [
        min(
            [
                correlations[col][lag - 1]
                for col in correlations
                if lag - 1 < len(correlations[col]) and correlations[col][lag - 1] < 0
            ],
            default=0
        )
        for lag in range(1, max_lag + 1)
    ]
    max_abs = [max(p, abs(n)) for p, n in zip(max_pos, max_neg)]
    plt.figure(figsize=(15, 10))
    plt.plot(range(1, max_lag + 1), max_pos, color='green', label='Max Positive Correlation')
    plt.plot(range(1, max_lag + 1), max_neg, color='red', label='Max Negative Correlation')
    plt.plot(range(1, max_lag + 1), max_abs, color='blue', label='Max Absolute Correlation')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.title('Max Positive, Negative, & Absolute Correlations per Lag', fontsize=14)
    plt.xlabel(f'Time Lag ({time_interval})', fontsize=12)
    plt.ylabel('Correlation', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(-1.0, 1.0)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{timestamp}_{base_csv}_max_correlation.png"), bbox_inches='tight')
    plt.close()

def visualize_data(
    data: pd.DataFrame,
    features: pd.DataFrame,
    feature_cols: List[str],
    timestamp: str,
    reverse: bool,
    time_interval: str,
    gen_charts: bool,
    correlations: Dict[str, Any],
    calc_corr: Callable[..., float],
    base_csv: str
) -> None:
    if not gen_charts:
        return
    charts_dir = 'indicator_charts'
    os.makedirs(charts_dir, exist_ok=True)
    max_lag = len(data) - 51
    if max_lag <= 0:
        return
    correlations = {
        col: correlations[col]
        for col in feature_cols
        if col != 'Close' and data[col].notna().any() and data[col].var() > 1e-6
    }
    for col, corr in correlations.items():
        plt.figure(figsize=(10, 4))
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.fill_between(range(1, max_lag + 1), corr, where=np.array(corr) > 0, color='blue', alpha=0.3)
        plt.fill_between(range(1, max_lag + 1), corr, where=np.array(corr) < 0, color='red', alpha=0.3)
        if len(corr) > 1:
            se = np.std(corr, ddof=1) / np.sqrt(len(corr))
            mo = t.ppf(0.975, len(corr) - 1) * se
            plt.fill_between(
                range(1, max_lag + 1),
                np.array(corr) - mo,
                np.array(corr) + mo,
                color='gray',
                alpha=0.4,
                label='95% CI'
            )
        plt.title(f'Correlation of {col} with Close', fontsize=10)
        plt.xlabel(f'Time Lag ({time_interval})', fontsize=8)
        plt.ylabel('Correlation', fontsize=8)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        plt.ylim(-1.0, 1.0)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, f"{timestamp}_{base_csv}_{col}_correlation.png"), bbox_inches='tight')
        plt.close()
    sorted_indicators = sorted(
        correlations,
        key=lambda c: correlations[c][-1] if correlations[c] else 0,
        reverse=True
    )
    plt.figure(figsize=(15, 10))
    for col, color in zip(sorted_indicators, plt.cm.rainbow(np.linspace(0, 1, len(sorted_indicators)))):
        plt.plot(range(1, max_lag + 1), correlations[col], color=color, label=col)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.title('All Indicators Correlation with Close', fontsize=14)
    plt.xlabel(f'Time Lag ({time_interval})', fontsize=12)
    plt.ylabel('Correlation', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(-1.0, 1.0)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join('combined_charts', f"{timestamp}_{base_csv}_combined_correlation.png"), bbox_inches='tight')
    plt.close()
    print("Combined correlation chart saved.")

def generate_heatmaps(
    data: pd.DataFrame,
    timestamp: str,
    time_interval: str,
    flag: bool,
    cache: Dict[str, List[float]],
    calc_corr: Callable[[pd.DataFrame, str, int, bool], float],
    base_csv: str
) -> None:
    if not flag:
        return
    os.makedirs('heatmaps', exist_ok=True)
    if os.listdir('heatmaps') and input("Delete existing heatmaps? (y/n): ").strip().lower() == 'y':
        for f in os.listdir('heatmaps'):
            file_path = os.path.join('heatmaps', f)
            if os.path.isfile(file_path):
                os.remove(file_path)
    indicators = [
        col for col in data.columns
        if pd.api.types.is_numeric_dtype(data[col]) and col != 'Close' and data[col].notna().any() and data[col].var() > 1e-6
    ]
    max_lag = len(data) - 51
    if max_lag < 1:
        print("Insufficient data for correlations.")
        return
    correlations = {
        col: cache.get(col, Parallel(n_jobs=-1)(
            delayed(calc_corr)(data, col, lag, False) for lag in range(1, max_lag + 1)
        ))
        for col in indicators
    }
    cache.update(correlations)
    corr_df = pd.DataFrame(correlations, index=range(1, max_lag + 1)).dropna(axis=1).dropna(axis=0)
    std_corr = corr_df.apply(
        lambda row: 0 if row.max() - row.min() == 0 else (row - row.min()) / (row.max() - row.min()) * 2 - 1,
        axis=1
    )
    filtered = [col for col in std_corr.columns if std_corr[col].max() > 0.25]
    std_corr = std_corr[filtered]
    sorted_indicators = sorted(
        filtered,
        key=lambda c: next(
            (i for i, x in enumerate(std_corr[c], 1) if x == 1.0),
            max_lag + 1
        )
    )
    sns.heatmap(std_corr[sorted_indicators].T, cmap='coolwarm', cbar=True, xticklabels=True, yticklabels=True)
    plt.title(
        'Standardized Correlation of Indicators with Close Price at Various Lags\n(Sorted by Earliest 1.0 Correlation)',
        fontsize=14
    )
    plt.xlabel(f'Time Lag ({time_interval})', fontsize=12)
    plt.ylabel('Indicators', fontsize=12)
    plt.xticks(ticks=np.arange(max_lag) + 0.5, labels=range(1, max_lag + 1), rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join('heatmaps', f"{timestamp}_{base_csv}_heatmap1.png"), bbox_inches='tight')
    plt.close()

    sorted_indicators = sorted(filtered, key=lambda c: std_corr[c].iloc[0], reverse=True)
    sns.heatmap(std_corr[sorted_indicators].T, cmap='coolwarm', cbar=True, xticklabels=True, yticklabels=True)
    plt.title('Standardized Correlation Sorted by Highest Correlation at Lag 1', fontsize=14)
    plt.xlabel(f'Time Lag ({time_interval})', fontsize=12)
    plt.ylabel('Indicators', fontsize=12)
    plt.xticks(ticks=np.arange(max_lag) + 0.5, labels=range(1, max_lag + 1), rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join('heatmaps', f"{timestamp}_{base_csv}_heatmap2.png"), bbox_inches='tight')
    plt.close()

    sns.heatmap(corr_df[filtered].T, cmap='coolwarm', cbar=True, xticklabels=True, yticklabels=True)
    plt.title('Raw Correlation Sorted by Highest Correlation at Lag 1', fontsize=14)
    plt.xlabel(f'Time Lag ({time_interval})', fontsize=12)
    plt.ylabel('Indicators', fontsize=12)
    plt.xticks(ticks=np.arange(max_lag) + 0.5, labels=range(1, max_lag + 1), rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join('heatmaps', f"{timestamp}_{base_csv}_heatmap3.png"), bbox_inches='tight')
    plt.close()
    print("Heatmaps generated.")
