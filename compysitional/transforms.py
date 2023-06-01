import numpy as np
import pandas as pd


def clr_df(comp_df: pd.DataFrame) -> pd.DataFrame:
    """Performs clr transform on a dataframe of compositions"""
    geo_means = np.exp(np.log(comp_df).mean(axis=1))
    return np.log(comp_df.div(geo_means, axis=0))
