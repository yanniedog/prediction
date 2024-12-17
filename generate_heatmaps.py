# generate_heatmaps.py
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
        delete_choice=input(f"Do you want to delete existing heatmaps in '{heatmaps_dir}'? (y/n): ").strip().lower()
        if delete_choice=='y':
            for file in existing_files:
                file_path=os.path.join(heatmaps_dir,file)
                try:
                    if os.path.isfile(file_path)or os.path.islink(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        os.rmdir(file_path)
                except:
                    pass
    original_indicators=[col for col in data.columns if pd.api.types.is_numeric_dtype(data[col]) and col!='Close' and data[col].notna().any() and data[col].var()>1e-6]
    print(f"Original indicators identified for heatmap generation: {original_indicators}")
    max_lag=len(data)-51
    if max_lag<1:
        print("Insufficient data length to compute correlations with the specified max_lag.")
        return
    correlations={}
    for col in original_indicators:
        if col not in cache:
            corr_list=Parallel(n_jobs=-1)(delayed(calculate_correlation)(data,col,lag,False)for lag in range(1,max_lag+1))
            cache[col]=corr_list
        else:
            corr_list=cache[col]
        correlations[col]=corr_list
    corr_df=pd.DataFrame(correlations,index=range(1,max_lag+1))
    corr_df.dropna(axis=1,how='all',inplace=True)
    corr_df.dropna(axis=0,how='all',inplace=True)
    def standardize_row(row:pd.Series)->pd.Series:
        if row.max()-row.min()==0:
            return row*0
        return (row-row.min())/(row.max()-row.min())*2-1
    standardized_corr_df=corr_df.apply(standardize_row,axis=1)
    filtered_indicators=[col for col in standardized_corr_df.columns if standardized_corr_df[col].max()>0.25]
    standardized_corr_df=standardized_corr_df[filtered_indicators]
    def earliest_one_cor(col:str)->int:
        return next((i for i,x in enumerate(standardized_corr_df[col],start=1)if x==1.0),max_lag+1)
    sorted_indicators_1=sorted(filtered_indicators,key=earliest_one_cor)
    sorted_standardized_corr_df_1=standardized_corr_df[sorted_indicators_1]
    plt.figure(figsize=(20,15),dpi=300)
    sns.heatmap(sorted_standardized_corr_df_1.T,annot=False,cmap='coolwarm',cbar=True,xticklabels=True,yticklabels=True)
    plt.title('Standardized Correlation of Indicators with Close Price at Various Time Lags\n(Sorted by Earliest 1.0 Correlation)',fontsize=14)
    plt.xlabel(f'Time Lag ({time_interval})',fontsize=12)
    plt.ylabel('Indicators',fontsize=12)
    plt.xticks(ticks=np.arange(max_lag)+0.5,labels=range(1,max_lag+1),rotation=90,fontsize=6)
    plt.yticks(rotation=0,fontsize=6)
    plt.tight_layout()
    heatmap_filename_1=f"{timestamp}_{base_csv_filename}_combined_correlation_heatmap_1.png"
    heatmap_filepath_1=os.path.join(heatmaps_dir,heatmap_filename_1)
    plt.savefig(heatmap_filepath_1,bbox_inches='tight')
    plt.close()
    def sort_by_lag1(col:str):
        return standardized_corr_df[col].iloc[0]
    sorted_indicators_2=sorted(filtered_indicators,key=sort_by_lag1,reverse=True)
    sorted_standardized_corr_df_2=standardized_corr_df[sorted_indicators_2]
    plt.figure(figsize=(20,15),dpi=300)
    sns.heatmap(sorted_standardized_corr_df_2.T,annot=False,cmap='coolwarm',cbar=True,xticklabels=True,yticklabels=True)
    plt.title('Standardized Correlation of Indicators with Close Price at Various Time Lags\n(Sorted by Highest Correlation at Lag 1)',fontsize=14)
    plt.xlabel(f'Time Lag ({time_interval})',fontsize=12)
    plt.ylabel('Indicators',fontsize=12)
    plt.xticks(ticks=np.arange(max_lag)+0.5,labels=range(1,max_lag+1),rotation=90,fontsize=6)
    plt.yticks(rotation=0,fontsize=6)
    plt.tight_layout()
    heatmap_filename_2=f"{timestamp}_{base_csv_filename}_combined_correlation_heatmap_2.png"
    heatmap_filepath_2=os.path.join(heatmaps_dir,heatmap_filename_2)
    plt.savefig(heatmap_filepath_2,bbox_inches='tight')
    plt.close()
    raw_corr_df=corr_df[filtered_indicators]
    def sort_by_lag1_raw(col:str):
        return raw_corr_df[col].iloc[0]
    sorted_indicators_3=sorted(filtered_indicators,key=sort_by_lag1_raw,reverse=True)
    sorted_raw_corr_df=raw_corr_df[sorted_indicators_3]
    plt.figure(figsize=(20,15),dpi=300)
    sns.heatmap(sorted_raw_corr_df.T,annot=False,cmap='coolwarm',cbar=True,xticklabels=True,yticklabels=True)
    plt.title('Raw Correlation of Indicators with Close Price at Various Time Lags\n(Sorted by Highest Correlation at Lag 1)',fontsize=14)
    plt.xlabel(f'Time Lag ({time_interval})',fontsize=12)
    plt.ylabel('Indicators',fontsize=12)
    plt.xticks(ticks=np.arange(max_lag)+0.5,labels=range(1,max_lag+1),rotation=90,fontsize=6)
    plt.yticks(rotation=0,fontsize=6)
    plt.tight_layout()
    heatmap_filename_3=f"{timestamp}_{base_csv_filename}_combined_correlation_heatmap_3.png"
    heatmap_filepath_3=os.path.join(heatmaps_dir,heatmap_filename_3)
    plt.savefig(heatmap_filepath_3,bbox_inches='tight')
    plt.close()
    print(f"Heatmaps have been successfully generated and saved in '{heatmaps_dir}' directory.")