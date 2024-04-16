from typing import Tuple

import pandas as pd
import numpy as np


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