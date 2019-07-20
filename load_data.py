import pandas as pd
import numpy as np


df_cache = {}


def _preprocess(df):
    df['time_unix'] = pd.to_datetime(df.EventTime).astype(np.int64) // 10 ** 6
    # First Text Change basically only happens in one item, and is redundant with the action before
    # print(df[df.Observable == 'First Text Change'].AccessionNumber.value_counts())
    df = df[df.Observable != 'First Text Change']  # Hence, remove it
    df = df[df.Observable != 'Exit Item']  # Basically 100% redundant with "Enter Item"
    # "Calculator Buffer" is a better indicator of closing the calculator, because sometimes it gets
    # automatically closed if the student opens a new scratch area or switches problems
    df = df[df.Observable != 'Close Calculator']
    df = df[df.EventTime.notnull()]  # Two null rows causing some outliers in delta time
    df = df.loc[(df.shift(1) != df).any(axis=1)]  # Remove consecutive duplicate rows (keep first)
    df['delta_time_ms'] = 0
    for pid, pid_df in df.groupby('STUDENTID'):
        df.loc[pid_df.index, 'delta_time_ms'] = \
            (pid_df.time_unix.shift(-1) - pid_df.time_unix).fillna(0)
    df['EventTime'] = pd.to_datetime(df.EventTime)
    return df


def train_full():
    if 'train_full' not in df_cache:
        df = pd.read_csv('public_data/data_a_train.csv')
        label_df = pd.read_csv('public_data/data_train_label.csv')
        assert len(df.STUDENTID.unique()) == len(label_df.STUDENTID.unique())
        gt = {p: int(l) for p, l in label_df.values}
        df['label'] = [gt[p] for p in df.STUDENTID.values]
        df_cache['train_full'] = _preprocess(df)
    return df_cache['train_full'].copy()


def train_10m():
    # Return only first 10 minutes of data per participant
    if 'train_10m' not in df_cache:
        df = train_full()
        start_unix_map = {p: v.time_unix.min() for p, v in df.groupby('STUDENTID')}
        df['start_unix'] = [start_unix_map[p] for p in df.STUDENTID]
        df_cache['train_10m'] = df[df.time_unix < df.start_unix + 10 * 60 * 1000] \
            .drop(columns='start_unix')
    return df_cache['train_10m'].copy()


def train_20m():
    # Return only first 20 minutes of data per participant
    if 'train_20m' not in df_cache:
        df = train_full()
        start_unix_map = {p: v.time_unix.min() for p, v in df.groupby('STUDENTID')}
        df['start_unix'] = [start_unix_map[p] for p in df.STUDENTID]
        df_cache['train_20m'] = df[df.time_unix < df.start_unix + 20 * 60 * 1000] \
            .drop(columns='start_unix')
    return df_cache['train_20m'].copy()


def holdout_10m():
    if 'holdout_10m' not in df_cache:
        df_cache['holdout_10m'] = _preprocess(pd.read_csv('public_data/data_a_hidden_10.csv'))
    return df_cache['holdout_10m'].copy()


def holdout_20m():
    if 'holdout_20m' not in df_cache:
        df_cache['holdout_20m'] = _preprocess(pd.read_csv('public_data/data_a_hidden_20.csv'))
    return df_cache['holdout_20m'].copy()


def holdout_30m():
    if 'holdout_30m' not in df_cache:
        df_cache['holdout_30m'] = _preprocess(pd.read_csv('public_data/data_a_hidden_30.csv'))
    return df_cache['holdout_30m'].copy()


def all_unique_rows():
    # Return all data, including training and holdout data, but not including overlapping subsets of
    # the training data (10m, 20m)
    return pd.concat([train_full(), holdout_10m(), holdout_20m(), holdout_30m()],
                     ignore_index=True, sort=False)
