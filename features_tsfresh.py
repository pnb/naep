# TSFresh automatic timeseries feature extraction

# Possible timeseries:
#   Time delta between actions (i.e., action duration)  TODO: Try collapse actions within 100ms of each other
#   Item duration
#   Item duration only for actual problems (excluding directions/help and such)
#   Action duration for last five minutes
#   TODO: Item duration for last five minutes
import logging

from tsfresh.transformers import RelevantFeatureAugmenter
from tqdm import tqdm
import pandas as pd

import load_data


logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)  # Hide tsfresh calculate_relevance_table() warnings


def format_timeseries(pandas_df):
    # Map of stacked timeseries dataframes of different lengths, as expected by TSFresh
    # Returns X, y, ts_dfs
    print('Formatting timeseries data')
    actual_items = set([row.AccessionNumber for _, row in pandas_df.iterrows() if row.ItemType in
                       ['FillInBlank', 'MCSS', 'MatchMS', 'MultipleFillInBlank']])
    ts_dfs = {'delta_sec': [], 'per_item_sec': [], 'per_problem_sec': [], 'delta_sec_last5': []}
    for pid, pid_df in tqdm(pandas_df.groupby('STUDENTID')):  # groupby sorts by STUDENTID
        ts_dfs['delta_sec'].append(pd.DataFrame({'ts': pid_df.delta_time_ms.values / 1000}))
        ts_dfs['per_item_sec'].append(pd.DataFrame(
            {'ts': [(v.time_unix.max() - v.time_unix.min()) / 1000
             for _, v in pid_df.groupby('AccessionNumber')]}))
        ts_dfs['per_problem_sec'].append(pd.DataFrame(
            {'ts': [v.delta_time_ms.sum() / 1000 for i, v in pid_df.groupby('AccessionNumber')
             if i in actual_items]}))
        last5df = pid_df[pid_df.time_unix > pid_df.time_unix.max() - 5 * 60 * 1000]
        ts_dfs['delta_sec_last5'].append(pd.DataFrame({'ts': last5df.delta_time_ms.values / 1000}))
        for key in ts_dfs:
            ts_dfs[key][-1]['instance_index'] = pid
    for key in ts_dfs:
        ts_dfs[key] = pd.concat(ts_dfs[key])

    # Skeleton DF onto which features will be added
    X_df = pd.DataFrame(index=sorted(pandas_df.STUDENTID.unique()))
    if 'label' in pandas_df.columns:
        y_df = pd.DataFrame(index=X_df.index)
        y_df['label'] = [pandas_df[pandas_df.STUDENTID == i].label.iloc[0] for i in y_df.index]
    else:
        y_df = None
    return X_df, y_df, ts_dfs


# Extract features
for datalen, train_df, holdout_df in [('10m', load_data.train_10m(), load_data.holdout_10m()),
                                      ('20m', load_data.train_20m(), load_data.holdout_20m()),
                                      ('30m', load_data.train_full(), load_data.holdout_30m())]:
    print('\nProcessing data length', datalen)
    train_X, train_y, train_ts = format_timeseries(train_df)
    holdout_X, _, holdout_ts = format_timeseries(holdout_df)

    # Fit and apply a TSFRESH augmenter to extract features in training data
    augmenter = RelevantFeatureAugmenter(column_id='instance_index', column_value='ts',
                                         timeseries_container=train_ts)
    tsfresh_train_X = augmenter.fit_transform(train_X, train_y.label)
    tsfresh_train_X.insert(0, 'STUDENTID', train_X.index)
    tsfresh_train_X.to_csv('features_tsfresh/train_' + datalen + '.csv', index=False)
    print(len(tsfresh_train_X.columns) - 1, 'relevant features extracted')

    # Apply to holdout data
    augmenter.set_timeseries_container(holdout_ts)
    tsfresh_holdout_X = augmenter.transform(holdout_X)
    tsfresh_holdout_X.insert(0, 'STUDENTID', holdout_X.index)
    tsfresh_holdout_X.to_csv('features_tsfresh/holdout_' + datalen + '.csv', index=False)
