"""
This module serves as a place to declare helper functions and classes that assist in cleaning or manipulating the data.
"""
from typing import List

import pandas as pd

# function to filter outliers
def filter_outliers(df: pd.DataFrame, column_name: str, min_value, max_value) -> pd.DataFrame:
    """
    Function to filter out outliers for numeric value columns in a df.
    """
    initial_rows = len(df)
    
    df = df.loc[df[column_name] >= min_value]
    df = df.loc[df[column_name] <= max_value]

    final_rows = len(df)

    filtered_out_rows = initial_rows - final_rows
    print(f"Filtered out {filtered_out_rows} rows for column {column_name}")
    
    return df



def auto_bin_column(df_input: pd.DataFrame, column_name: str, num_bins: int, min_val = None, max_val = None):
    """
    Function for easy binning of columns based on number of bins, min val and max val
    """
    df = df_input.copy()
    
    if min_val is None:
        min_val = float(df[column_name].min())
    if max_val is None:
        max_val = float(df[column_name].max())

    print(f"min value in col: {min_val}")
    print(f"max value in col: {max_val}")
    
    col_range = max_val - min_val
    len_of_bin = (col_range / num_bins) + 1
    
    bins = [(min_val + (len_of_bin*num_bin)) for num_bin in range(num_bins + 1)]

    df[column_name + "_bin"] = pd.cut(df[column_name], bins, include_lowest=True)

    print(df[column_name + "_bin"].value_counts())
    
    return df


def bin_column(df_input: pd.DataFrame, column_name: str, bins: List[float]):
    """
    Function for easy binning of columns based on manual bins provided by user
    (Probably can be combined with the other function, but separate right now)
    """
    df = df_input.copy()

    df[column_name + "_bin"] = pd.cut(df[column_name], bins, include_lowest=True)

    print(df[column_name + "_bin"].value_counts())
    
    return df