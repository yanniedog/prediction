import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict,List,Any,Callable
from scipy.stats import t
import pandas as pd

def generate_combined_correlation_chart(correlations:Dict[str,List[float]],max_lag:int,time_interval:str,timestamp:str,base_csv_filename:str,output_dir:str='combined_charts')->None:
    os.makedirs(output_dir,exist_ok=True)
    max_positive_correlations=[]
    max_negative_correlations=[]
    max_absolute_correlations=[]
    for lag in range(1,max_lag+1):
        lag_correlations=[correlations[col][lag-1]for col in correlations if lag-1<len(correlations[col])]
        pos_cor=[x for x in lag_correlations if x>0]
        neg_cor=[x for x in lag_correlations if x<0]
        max_pos=max(pos_cor) if pos_cor else 0
        max_neg=min(neg_cor) if neg_cor else 0
        max_abs=max(max_pos,abs(max_neg))
        max_positive_correlations.append(max_pos)
        max_negative_correlations.append(max_neg)
        max_absolute_correlations.append(max_abs)
    plt.figure(figsize=(15,10))
    plt.plot(range(1,max_lag+1),max_positive_correlations,color='green',label='Max Positive Correlation')
    plt.plot(range(1,max_lag+1),max_negative_correlations,color='red',label='Max Negative Correlation')
    plt.plot(range(1,max_lag+1),max_absolute_correlations,color='blue',label='Max Absolute Correlation')
    plt.axhline(0,color='black',linewidth=0.5)
    plt.title('Maximum Positive, Negative, and Absolute Correlations at Each Lag',fontsize=14)
    plt.xlabel(f'Time Lag ({time_interval})',fontsize=12)
    plt.ylabel('Correlation',fontsize=12)
    plt.ylim(-1.0,1.0)
    plt.grid(True,linestyle='--',linewidth=0.5)
    plt.legend(loc='center left',bbox_to_anchor=(1,0.5),fontsize=10)
    plt.tight_layout()
    combined_filename=f"{timestamp}_{base_csv_filename}_max_correlation.png"
    plt.savefig(os.path.join(output_dir,combined_filename),bbox_inches='tight')
    plt.close()

def visualize_data(data:pd.DataFrame,features:pd.DataFrame,feature_columns:List[str],timestamp:str,is_reverse_chronological:bool,time_interval:str,generate_charts:bool,cache:Dict[str,Any],calculate_correlation_func:Callable[...,float],base_csv_filename:str)->None:
    if not generate_charts:
        return
    charts_dir='indicator_charts'
    os.makedirs(charts_dir,exist_ok=True)
    max_lag=len(data)-51
    if max_lag<=0:
        return
    correlations={}
    original_indicators=[col for col in feature_columns if data[col].notna().any() and data[col].var()>1e-6 and col.lower() not in ['open','high','low','close','volume']]
    for col in original_indicators:
        corr_list = cache[col] if col in cache else [calculate_correlation_func(data,col,lag,is_reverse_chronological) for lag in range(1,max_lag+1)]
        cache[col]=corr_list
        correlations[col]=corr_list
        plt.figure(figsize=(10,4))
        plt.axhline(0,color='black',linewidth=0.5)
        arr=np.array(corr_list)
        plt.fill_between(range(1,max_lag+1),arr,where=arr>0,color='blue',alpha=0.3)
        plt.fill_between(range(1,max_lag+1),arr,where=arr<0,color='red',alpha=0.3)
        n=len(corr_list)
        if n>1:
            std_err=np.std(corr_list,ddof=1)/np.sqrt(n)
            margin_of_error=t.ppf(0.975,n-1)*std_err
            lower_bound=arr-margin_of_error
            upper_bound=arr+margin_of_error
            plt.fill_between(range(1,max_lag+1),lower_bound,upper_bound,color='gray',alpha=0.4)
        plt.title(f'Average Correlation of {col} with Close Price',fontsize=10)
        plt.xlabel(f'Time Lag ({time_interval})',fontsize=8)
        plt.ylabel('Average Correlation',fontsize=8)
        plt.ylim(-1.0,1.0)
        plt.grid(True,linestyle='--',linewidth=0.5)
        plt.tight_layout()
        filepath=os.path.join(charts_dir,f"{timestamp}_{base_csv_filename}_{col}_correlation.png")
        plt.savefig(filepath,bbox_inches='tight')
        plt.close()
    combined_charts_dir='combined_charts'
    os.makedirs(combined_charts_dir,exist_ok=True)
    sorted_indicators=sorted(original_indicators,key=lambda c:correlations[c][-1] if correlations[c] else 0,reverse=True)
    plt.figure(figsize=(15,10))
    colors=plt.cm.rainbow(np.linspace(0,1,len(sorted_indicators)))
    for c,color in zip(sorted_indicators,colors):
        plt.plot(range(1,max_lag+1),correlations[c],color=color,label=c)
    plt.axhline(0,color='black',linewidth=0.5)
    plt.title('Average Correlation of All Indicators',fontsize=14)
    plt.xlabel(f'Time Lag ({time_interval})',fontsize=12)
    plt.ylabel('Average Correlation',fontsize=12)
    plt.ylim(-1.0,1.0)
    plt.grid(True,linestyle='--',linewidth=0.5)
    plt.legend(loc='center left',bbox_to_anchor=(1,0.5),fontsize=10)
    plt.tight_layout()
    combined_filepath=os.path.join(combined_charts_dir,f"{timestamp}_{base_csv_filename}_combined_correlation.png")
    plt.savefig(combined_filepath,bbox_inches='tight')
    plt.close()

    max_positive_correlations=[]
    max_negative_correlations=[]
    max_absolute_correlations=[]
    for lag in range(1,max_lag+1):
        lag_correlations=[correlations[col][lag-1]for col in original_indicators]
        pos=[x for x in lag_correlations if x>0]
        neg=[x for x in lag_correlations if x<0]
        max_pos=max(pos)if pos else 0
        max_neg=min(neg)if neg else 0
        max_abs=max(max_pos,abs(max_neg))
        max_positive_correlations.append(max_pos)
        max_negative_correlations.append(max_neg)
        max_absolute_correlations.append(max_abs)
    plt.figure(figsize=(15,10))
    plt.plot(range(1,max_lag+1),max_positive_correlations,color='green',label='Max Positive Correlation')
    plt.plot(range(1,max_lag+1),max_negative_correlations,color='red',label='Max Negative Correlation')
    plt.plot(range(1,max_lag+1),max_absolute_correlations,color='blue',label='Max Absolute Correlation')
    plt.axhline(0,color='black',linewidth=0.5)
    plt.title('Max Positive, Negative, and Absolute Correlations',fontsize=14)
    plt.xlabel(f'Time Lag ({time_interval})',fontsize=12)
    plt.ylabel('Correlation',fontsize=12)
    plt.ylim(-1.0,1.0)
    plt.grid(True,linestyle='--',linewidth=0.5)
    plt.legend(loc='center left',bbox_to_anchor=(1,0.5),fontsize=10)
    plt.tight_layout()
    max_corr_path=os.path.join(combined_charts_dir,f"{timestamp}_{base_csv_filename}_max_correlation.png")
    plt.savefig(max_corr_path,bbox_inches='tight')
    plt.close()
