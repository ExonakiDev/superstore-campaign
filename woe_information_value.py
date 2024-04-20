from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from data_helpers import bin_column

def calculate_woe_iv(df: pd.DataFrame, x_col: str, y_col: str) -> Tuple[float, float]:
    """
    Function to calculate IV (Information Value) & WOE

    Parameters
    ----------
    df: pd.DataFrame

    x_col: str
        One feature column name that we want to calculate IV and WOE for

    y_col: str
        Target column name
    """
    df_subset = df[[x_col, y_col]].copy()

    df_subset.dropna(inplace=True)

    unique_vals = df_subset[x_col].unique()

    good = len(df_subset[df_subset[y_col] == 1])
    bad = len(df_subset[df_subset[y_col] == 0])

    iv = 0
    
    for uv in unique_vals:
        num_goods = len(df_subset[(df_subset[y_col] == 1) & (df_subset[x_col] == uv)])
        num_bads = len(df_subset[(df_subset[y_col] == 0) & (df_subset[x_col] == uv)])

        good_dist = (num_goods/good)
        bad_dist = (num_bads/bad)

        if good_dist == 0 or bad_dist == 0:
            continue
        woe = np.log(good_dist/bad_dist)
        iv += ((good_dist - bad_dist) * woe)

    return woe, iv

def generate_woe_graph(df_input: pd.DataFrame, column_name: str, y_col: str, num_bins: str) -> pd.DataFrame:
    df = df_input[[column_name, y_col]].copy()
    df.dropna(inplace=True)
    
    plot_list = []

    total_good = len(df[df[y_col] == 1])
    total_bad = len(df[df[y_col] == 0])
    
    binned_df = bin_column(df, column_name, num_bins)
    sorted_bins = binned_df[column_name + "_bin"].unique().sort_values()

    # calculate WOE for each of the bins
    for s_bin in sorted_bins:
        subset_data = binned_df.loc[binned_df[column_name + "_bin"] == s_bin]
        
        good_dist = (len(subset_data[subset_data[y_col] == 1]) + 0.5) / total_good
        bad_dist = (len(subset_data[subset_data[y_col] == 0]) + 0.5) / total_bad

        if good_dist == 0 or bad_dist == 0:
            continue
            
        woe = np.log(good_dist/bad_dist)
        plot_list.append({"bin_name": s_bin, "woe": woe})

    plot_df = pd.DataFrame(plot_list)

    # generate plot
    plot_df["bin_name"] = plot_df["bin_name"].apply(lambda x: x.right)
    plt.plot(plot_df["bin_name"], plot_df["woe"])
    plt.xlabel(column_name + "_bin")
    plt.ylabel("WOE (Weight of Evidence)")
    plt.show()

    return plot_df