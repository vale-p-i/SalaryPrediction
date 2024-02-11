import pandas as pd
import numpy as np


def remove_missing_values(df: pd.DataFrame, missing_value: str) -> pd.DataFrame:
    df = df.replace(missing_value, np.NaN)
    df = df.dropna()
    return df
