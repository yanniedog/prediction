import os, matplotlib.pyplot as plt, seaborn as sns, pandas as pd, numpy as np
from typing import Dict, List, Any, Callable
from joblib import Parallel, delayed
from scipy.stats import t

def generate_combined_correlation_chart(correlations: Dict[str, List[float]], max_lag: int, time_interval: str, timestamp: str, base_csv: str, output_dir: str='combined_charts') -> None:
    os.makedirs(output_dir, exist_ok=True)
    max_pos = [max([correlations[col][lag-1] for col in correlations if lag-1 < len(correlations[col]) and correlations[col][lag-1] >0], default=0) for lag in range(1,max_lag+1)]
    max_neg = [min([correlations[col][lag-1] for col in correlations if lag-1 < len(correlations[col]) and correlations[col][lag-1] <0], default=0) for lag in range(1,max_lag+1)]
    max_abs = [max(p, abs(n)) for p, n in zip(max_pos, max_neg)]
    plt.figure(figsize=(15,10))
    plt.plot(range(1,max_lag+1), max_pos, color='green', label='Max Positive Correlation')
    plt.plot(range(1,max_lag+1), max_neg, color='red', label='Max Negative Correlation')
    plt.plot(range(1,max_lag+1), max_abs, color='blue', label='Max Absolute Correlation')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.title('Max Positive, Negative, & Absolute Correlations per Lag', fontsize=14)
    plt.xlabel(f'Time Lag ({time_interval})', fontsize=12)
    plt.ylabel('Correlation', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(-1.0,1.0)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{timestamp}_{base_csv}_max_correlation.png"), bbox_inches='tight')
    plt.close()

def visualize_data(data: pd.DataFrame, features: pd.DataFrame, feature_cols: List[str], timestamp: str, reverse: bool, time_interval: str, gen_charts: bool, correlations: Dict[str, Any], calc_corr: Callable[..., float], base_csv: str) -> None:
    if not gen_charts: return
    charts_dir = 'indicator_charts'
    os.makedirs(charts_dir, exist_ok=True)
    max_lag = len(data) -51
    if max_lag <=0: return
    correlations = {col: correlations[col] for col in feature_cols if col != 'Close' and data[col].notna().any() and data[col].var() >1e-6}
    for col, corr in correlations.items():
        plt.figure(figsize=(10,4))
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.fill_between(range(1,max_lag+1), corr, where=np.array(corr)>0, color='blue', alpha=0.3)
        plt.fill_between(range(1,max_lag+1), corr, where=np.array(corr)<0, color='red', alpha=0.3)
        if len(corr) >1:
            se = np.std(corr, ddof=1)/np.sqrt(len(corr))
            mo = t.ppf(0.975, len(corr)-1)*se
            plt.fill_between(range(1,max_lag+1), np.array(corr)-mo, np.array(corr)+mo, color='gray', alpha=0.4, label='95% CI')
        plt.title(f'Correlation of {col} with Close', fontsize=10)
        plt.xlabel(f'Time Lag ({time_interval})', fontsize=8)
        plt.ylabel('Correlation', fontsize=8)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        plt.ylim(-1.0,1.0)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, f"{timestamp}_{base_csv}_{col}_correlation.png"), bbox_inches='tight')
        plt.close()
    # Combined chart
    sorted_inds = sorted(correlations, key=lambda c: correlations[c][-1] if correlations[c] else 0, reverse=True)
    plt.figure(figsize=(15,10))
    for col, color in zip(sorted_inds, plt.cm.rainbow(np.linspace(0,1,len(sorted_inds)))):
        plt.plot(range(1,max_lag+1), correlations[col], color=color, label=col)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.title('All Indicators Correlation with Close', fontsize=14)
    plt.xlabel(f'Time Lag ({time_interval})', fontsize=12)
    plt.ylabel('Correlation', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(-1.0,1.0)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join('combined_charts', f"{timestamp}_{base_csv}_combined_correlation.png"), bbox_inches='tight')
    plt.close()
    print("Combined correlation chart saved.")
