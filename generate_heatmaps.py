import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from joblib import Parallel,delayed
from typing import Callable,Dict,List

def generate_heatmaps(data:pd.DataFrame,timestamp:str,time_interval:str,generate_heatmaps_flag:bool,cache:Dict[str,List[float]],calculate_correlation:Callable[[pd.DataFrame,str,int,bool],float],base_csv_filename:str)->None:
    if not generate_heatmaps_flag:
        return
    heatmaps_dir='heatmaps'
    os.makedirs(heatmaps_dir,exist_ok=True)
    existing_files=os.listdir(heatmaps_dir)
    if existing_files:
        delete_choice=input(f"Delete existing heatmaps in '{heatmaps_dir}'? (y/n): ").strip().lower()
        if delete_choice=='y':
            for file in existing_files:
                file_path=os.path.join(heatmaps_dir,file)
                if os.path.isfile(file_path)or os.path.islink(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
    original_indicators=[col for col in data.columns if pd.api.types.is_numeric_dtype(data[col]) and col!='Close' and data[col].notna().any() and data[col].var()>1e-6]
    max_lag = len(data)-51
    if max_lag<1:
        return
    correlations={}
    for col in original_indicators:
        corr_list = cache[col] if col in cache else Parallel(n_jobs=-1)(delayed(calculate_correlation)(data,col,lag,False)for lag in range(1,max_lag+1))
        cache[col] = corr_list
        correlations[col]=corr_list
    corr_df=pd.DataFrame(correlations,index=range(1,max_lag+1)).dropna(how='all',axis=0).dropna(how='all',axis=1)

    def standardize_row(row):
        rng=row.max()-row.min()
        return (row-row.min())/rng*2-1 if rng!=0 else row*0

    standardized_corr_df=corr_df.apply(standardize_row,axis=1)
    filtered_indicators=[col for col in standardized_corr_df.columns if standardized_corr_df[col].max()>0.25]
    standardized_corr_df=standardized_corr_df[filtered_indicators]

    def plot_heatmap(df, title, filename):
        plt.figure(figsize=(20,15),dpi=300)
        sns.heatmap(df.T, annot=False, cmap='coolwarm', cbar=True, xticklabels=True, yticklabels=True)
        plt.title(title, fontsize=14)
        plt.xlabel(f'Time Lag ({time_interval})', fontsize=12)
        plt.ylabel('Indicators', fontsize=12)
        plt.xticks(rotation=90, fontsize=6)
        plt.yticks(rotation=0, fontsize=6)
        plt.tight_layout()
        plt.savefig(os.path.join(heatmaps_dir,filename), bbox_inches='tight')
        plt.close()

    sorted_indicators_1=sorted(filtered_indicators,key=lambda col:next((i for i,x in enumerate(standardized_corr_df[col],start=1)if x==1.0),max_lag+1))
    sorted_standardized_corr_df_1=standardized_corr_df[sorted_indicators_1]
    plot_heatmap(sorted_standardized_corr_df_1,'Standardized Correlation (Earliest 1.0)','{}_{}_combined_correlation_heatmap_1.png'.format(timestamp,base_csv_filename))

    sorted_indicators_2=sorted(filtered_indicators,key=lambda col:standardized_corr_df[col].iloc[0],reverse=True)
    plot_heatmap(standardized_corr_df[sorted_indicators_2],'Standardized Correlation (Highest at Lag 1)','{}_{}_combined_correlation_heatmap_2.png'.format(timestamp,base_csv_filename))

    raw_corr_df=corr_df[filtered_indicators]
    sorted_indicators_3=sorted(filtered_indicators,key=lambda col:raw_corr_df[col].iloc[0],reverse=True)
    plot_heatmap(raw_corr_df[sorted_indicators_3],'Raw Correlation (Highest at Lag 1)','{}_{}_combined_correlation_heatmap_3.png'.format(timestamp,base_csv_filename))