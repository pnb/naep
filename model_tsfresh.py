# TSFresh automatic timeseries feature extraction

# Possible timeseries:
#   Time delta between actions (i.e., action duration)
#   Item duration
import logging

from sklearn import ensemble, model_selection, pipeline, metrics
from tsfresh.transformers import RelevantFeatureAugmenter
from tqdm import tqdm
import pandas as pd

import load_data


RANDOM_SEED = 11798
CACHE_DIR = '/Users/pnb/sklearn_cache'
logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)  # Hide tsfresh calculate_relevance_table() warnings


print('Loading data')
df = load_data.train_full()

# Map of stacked timeseries dataframes of different lengths, as expected by TSFresh
print('Extracting timeseries data')
ts_dfs = {'delta_sec': [], 'per_item_sec': []}
for pid, pid_df in tqdm(df.groupby('STUDENTID')):  # groupby sorts by STUDENTID
    ts_dfs['delta_sec'].append(pd.DataFrame({'ts': pid_df.delta_time_ms.values / 1000}))
    ts_dfs['per_item_sec'].append(pd.DataFrame(
        {'ts': [(v.time_unix.max() - v.time_unix.min()) / 1000
         for _, v in pid_df.groupby('AccessionNumber')]}))
    for key in ts_dfs:
        ts_dfs[key][-1]['instance_index'] = pid
for key in ts_dfs:
    ts_dfs[key] = pd.concat(ts_dfs[key])

y_df = pd.DataFrame(index=sorted(df.STUDENTID.unique()))
y_df['label'] = [df[df.STUDENTID == i].label.iloc[0] for i in y_df.index]
X_df = pd.DataFrame(index=y_df.index)  # Skeleton DF onto which features will be added

# for ts_name, ts_df in ts_dfs.items():
#     print('Fitting model for timeseries:', ts_name)
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
gs = model_selection.GridSearchCV(pipe, grid, cv=xval, verbose=1)
scoring = {'AUC': metrics.make_scorer(metrics.roc_auc_score, needs_proba=True),
           'MCC': metrics.make_scorer(metrics.cohen_kappa_score),
           'Kappa': metrics.make_scorer(metrics.matthews_corrcoef)}
result = model_selection.cross_validate(gs, X_df, y_df.label, cv=xval, verbose=2, scoring=scoring)
print(result)
