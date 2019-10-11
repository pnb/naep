# TSFresh automatic timeseries feature extraction

# Possible timeseries:
#   Time delta between actions (i.e., action duration)  TODO: Try collapse actions within 100ms of each other
#   Item duration
import logging

from sklearn import ensemble, model_selection, pipeline, metrics
from tsfresh.transformers import RelevantFeatureAugmenter
from tqdm import tqdm
import pandas as pd
import numpy as np

import load_data


RANDOM_SEED = 11798
CACHE_DIR = '/Users/pnb/sklearn_cache'
logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)  # Hide tsfresh calculate_relevance_table() warnings


def format_timeseries(pandas_df):
    # Map of stacked timeseries dataframes of different lengths, as expected by TSFresh
    # Returns X, y, ts_dfs
    print('Formatting timeseries data')
    ts_dfs = {'delta_sec': [], 'per_item_sec': []}
    for pid, pid_df in tqdm(pandas_df.groupby('STUDENTID')):  # groupby sorts by STUDENTID
        ts_dfs['delta_sec'].append(pd.DataFrame({'ts': pid_df.delta_time_ms.values / 1000}))
        ts_dfs['per_item_sec'].append(pd.DataFrame(
            {'ts': [(v.time_unix.max() - v.time_unix.min()) / 1000
             for _, v in pid_df.groupby('AccessionNumber')]}))
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


print('Loading data')
df = load_data.train_full()
X_df, y_df, ts_dfs = format_timeseries(df)

augmenter = RelevantFeatureAugmenter(column_id='instance_index', column_value='ts',
                                     timeseries_container=ts_dfs)
m = ensemble.ExtraTreesClassifier(200, random_state=RANDOM_SEED)
grid = {
    'model__min_samples_leaf': [1, 2, 4, 8, 16, 32],
    'model__max_features': [.1, .25, .5, .75, 1.0, 'auto'],
}
pipe = pipeline.Pipeline([
    ('augmenter', augmenter),
    ('model', m),
], memory=CACHE_DIR)
xval = model_selection.StratifiedKFold(4, shuffle=True, random_state=RANDOM_SEED)
gs = model_selection.GridSearchCV(pipe, grid, cv=xval, verbose=1,
                                  scoring=metrics.make_scorer(metrics.cohen_kappa_score))
scoring = {'AUC': metrics.make_scorer(metrics.roc_auc_score, needs_proba=True),
           'MCC': metrics.make_scorer(metrics.cohen_kappa_score),
           'Kappa': metrics.make_scorer(metrics.matthews_corrcoef)}

# Train model on all data and make predictions for competition hold-out set
hidden_result = pd.read_csv('public_data/hidden_label.csv')
hidden_result['holdout'] = 1
train_results = []
for datalen, train_df, holdout_df in [(10, load_data.train_10m(), load_data.holdout_10m()),
                                      (20, load_data.train_20m(), load_data.holdout_20m()),
                                      (30, load_data.train_full(), load_data.holdout_30m())]:
    print('Processing data length', datalen)
    train_X, train_y, train_ts = format_timeseries(train_df)
    holdout_X, _, holdout_ts = format_timeseries(holdout_df)
    augmenter.set_timeseries_container(train_ts)
    # Train and apply a TSFresh augmenter to extract features in training data
    aug_m = gs.fit(train_X, train_y.label).best_estimator_
    importances = pd.Series(data=aug_m.named_steps['model'].feature_importances_,
                            index=aug_m.named_steps['augmenter'].feature_selector.relevant_features)
    extracted_X = aug_m.named_steps['augmenter'].transform(train_X)
    # Pick only important features to save (probably all of them in most cases)
    extracted_X = extracted_X[importances[importances > .001].index]
    extracted_X.insert(0, 'STUDENTID', extracted_X.index)
    extracted_X.to_csv('features_tsfresh/train_' + str(datalen) + 'm.csv', index=False)
    # Now extract for corresponding holdout data
    aug_m.named_steps['augmenter'].set_timeseries_container(holdout_ts)
    extracted_X = aug_m.named_steps['augmenter'].transform(holdout_X)
    extracted_X = extracted_X[importances[importances > .001].index]
    extracted_X.insert(0, 'STUDENTID', extracted_X.index)
    extracted_X.to_csv('features_tsfresh/holdout_' + str(datalen) + 'm.csv', index=False)
