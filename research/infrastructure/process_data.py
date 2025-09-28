from sklearn.preprocessing import StandardScaler
import pandas as pd
from functools import reduce


def normalize_data(df):
    scaler = StandardScaler()
    normalized = scaler.fit_transform(df)
    normalized_df = pd.DataFrame(normalized, index=df.index, columns=df.columns)
    return normalized_df


def align_start_dates(df_dict):
    start_dates = [df.index.min() for df in df_dict.values()]
    global_start = max(start_dates)
    for name in df_dict:
        df_dict[name] = df_dict[name].loc[global_start:]
    return df_dict

def merge_data(df_dict):
    dfs = list(df_dict.values())
    aligned_df = reduce(lambda left, right: left.join(right, how="inner"), dfs)
    return aligned_df

def resample_data(df,cols,sampling,mean=False):
    for col in cols :
        df[col] = df[col].resample(sampling).mean() if mean else df[col].resample(sampling).last()
    return df