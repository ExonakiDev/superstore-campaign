from typing import List

import pandas as pd
import matplotlib.pyplot as plt

from data_helpers import bin_column

def generate_bivariate_graph(df_input: pd.DataFrame, column_name: str, y_col: str, bins: List[float]):
    df = df_input[[column_name, y_col]].copy()
    df.dropna(inplace=True)
    
    binned_df = bin_column(df, column_name, bins)
    
    bar_width = 0.25
    fig = plt.subplots(figsize =(12, 8)) 
    
    sorted_bins = binned_df[column_name + "_bin"].unique().sort_values()
    
    x = []
    
    for s_bin in sorted_bins:
        response_value_counts = binned_df.loc[binned_df[column_name + "_bin"] == s_bin]["Response"].value_counts()
        total_num = response_value_counts.sum()

        try:
            responses = response_value_counts[1]
        except:
            responses = 0
        response_rate = responses / total_num
        
        x.append(response_rate)
    
    bin_names = [str(s_bin) for s_bin in sorted_bins]
    
    plt.bar(bin_names, x, width=bar_width)
    plt.xlabel(f"Bins for column {column_name}", fontweight ='bold', fontsize = 15)
    plt.ylabel(f"Response Rate: (Num Positive Responses / Total Responses)", fontweight ='bold', fontsize = 12)
    plt.show()
    
