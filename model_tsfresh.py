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
gs = model_selection.GridSearchCV(pipe, grid, cv=xval, verbose=1,
                                  scoring=metrics.make_scorer(metrics.cohen_kappa_score))
scoring = {'AUC': metrics.make_scorer(metrics.roc_auc_score, needs_proba=True),
           'MCC': metrics.make_scorer(metrics.cohen_kappa_score),
           'Kappa': metrics.make_scorer(metrics.matthews_corrcoef)}

# Just out of curiosity to see approximately how many features there are as well
# print('Fitting feature importance model')
# imp_m = gs.fit(X_df, y_df.label).best_estimator_
# imp = imp_m.named_steps['model'].feature_importances_
# feat_names = imp_m.named_steps['augmenter'].feature_selector.relevant_features
# print('\n'.join([f + ':\t' + str(i) for i, f in sorted(zip(imp, feat_names), reverse=True)]))

# Train model on all data and make predictions for competition hold-out set
print('Loading holdout data')
'''
# TODO: Just like model_fe, Add c-v predictions for training data above, plus perf metrics, to enable better fusion later
hidden_result = pd.read_csv('public_data/hidden_label.csv', index_col='STUDENTID')
hidden_result['pred'] = ''
hidden_result['data_length'] = ''
for datalen, train_df, holdout_df in [(10, load_data.train_10m(), load_data.holdout_10m()),
                                      (20, load_data.train_20m(), load_data.holdout_20m()),
                                      (30, load_data.train_full(), load_data.holdout_30m())]:
    print('Training/applying holdout model for', datalen, 'minutes')
    train_X, train_y, train_ts = format_timeseries(train_df)
    holdout_X, _, holdout_ts = format_timeseries(holdout_df)
    augmenter.set_timeseries_container(train_ts)
    gs.fit(train_X, train_y.label)
    gs.best_estimator_.named_steps['augmenter'].set_timeseries_container(holdout_ts)
    probs = gs.predict_proba(holdout_X).T[1]
    hidden_result.loc[holdout_X.index, 'pred'] = probs  # holdout_X.index is STUDENTID sorted
    hidden_result.loc[holdout_X.index, 'data_length'] = datalen
    print('Grid search best estimator:', gs.best_estimator_)
    print('Grid search scorer:', gs.scorer_)
    print('Grid search best score:', gs.best_score_)
hidden_result.to_csv('model_tsfresh.csv')
'''

# Train model on all data and make predictions for competition hold-out set
hidden_result = pd.read_csv('public_data/hidden_label.csv')
hidden_result['holdout'] = 1
train_results = []
for datalen, train_df, holdout_df in [(10, load_data.train_10m(), load_data.holdout_10m()),
                                      (20, load_data.train_20m(), load_data.holdout_20m()),
                                      (30, load_data.train_full(), load_data.holdout_30m())]:
    train_X, train_y, train_ts = format_timeseries(train_df)
    holdout_X, _, holdout_ts = format_timeseries(holdout_df)
    augmenter.set_timeseries_container(train_ts)
    # First cross-validate on training data to test accuracy on local (non-LB) data
    print('\nFitting cross-val model with', datalen, 'minutes data')
    result = model_selection.cross_validate(gs, train_X, train_y.label, cv=xval,
                                            verbose=2, scoring=scoring, return_estimator=True)
    print('\n'.join([k + ': ' + str(v) for k, v in result.items() if k.startswith('test_')]))
    train_results.append(pd.DataFrame({'STUDENTID': train_X.index}))
    train_results[-1]['label'] = train_y.label
    train_results[-1]['holdout'] = 0
    train_results[-1]['data_length'] = datalen
    train_results[-1]['kappa_mean'] = np.mean(result['test_Kappa'])
    train_results[-1]['kappa_min'] = np.min(result['test_Kappa'])
    train_results[-1]['auc_mean'] = np.mean(result['test_AUC'])
    train_results[-1]['auc_min'] = np.min(result['test_AUC'])
    # Save cross-validated predictions for training set, for later fusion tests
    print('\nSaving cross-validated predictions')
    for i, (_, test_i) in enumerate(xval.split(train_X, train_y)):
        test_pids = train_X.iloc[test_i].index
        test_preds = result['estimator'][i].predict_proba(train_X.iloc[test_i]).T[1]
        for pid, pred in zip(test_pids, test_preds):
            train_results[-1].loc[train_results[-1].STUDENTID == pid, 'pred'] = pred
    # Fit on all training data and apply to holdout data
    print('\nHoldout model for feature set with', datalen, 'minutes data')
    gs.fit(train_X, train_y.label)
    gs.best_estimator_.named_steps['augmenter'].set_timeseries_container(holdout_ts)
    probs = gs.predict_proba(holdout_X).T[1]
    print('Grid search best estimator:', gs.best_estimator_)
    print('Grid search scorer:', gs.scorer_)
    print('Grid search best score:', gs.best_score_)
    print('Train data positive class base rate:', np.mean(train_y.label))
    print('Predicted base rate (> .5 threshold):', np.mean(probs > .5))
    for pid, pred in zip(holdout_X.index.values, probs):
        hidden_result.loc[hidden_result.STUDENTID == pid, 'pred'] = pred
        hidden_result.loc[hidden_result.STUDENTID == pid, 'data_length'] = datalen
train_results.append(hidden_result)
pd.concat(train_results, ignore_index=True, sort=False) \
    .to_csv('model_tsfresh.csv', index=False)
