# Feature engineering

# Feature ideas (doubly-indented are implemented):
#       Time spent in each item (AccessionNumber)
#       Time spent in each item >= 5th percentile as calculated from full data
#       Count of VH items where time spent >= 5th percentile
#       Num actions for each item
#   Num of each type of action (Observable)
#       SD of time spent across items
#       Num times entered each item
#   Ordering behaviors (straight through vs. skipping problems) -- related to num navigations
#   Time spent and num actions for each item type (might be too redundant)
#   WTF behavior, especially at the end of the session (num Next event in last X mins vs. mean)
#       Coefficients of polynomials fit to time spent per problem
#       Coefficients of polynomials fit to overall timeseries
from collections import OrderedDict

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn import ensemble, model_selection, pipeline, metrics

import load_data
import misc_util


RANDOM_SEED = 11798
CACHE_DIR = '/Users/pnb/sklearn_cache'


def extract_features(pandas_df, freq_actions, item_5percentile_map):
    # freq_actions is a list of common actions that should be dummy-coded
    # item_5percentile_map is a mapping of item_name -> 5th percentile of time spent (ms) for item
    # Returns (X, y, list of feature names); y may be None if `label` is not in `pandas_df`
    print('Extracting features')
    rows = []
    for pid, pid_df in tqdm(pandas_df.groupby('STUDENTID')):
        rows.append(OrderedDict({
            'STUDENTID': pid,
            'label': pid_df.label.iloc[0] if 'label' in pid_df.columns else '',
            **{'sec_spent_' + e: v.delta_time_ms.sum() / 1000
               for e, v in pid_df.groupby('AccessionNumber')},
            **{'times_entered_' + e: sum(v.Observable == 'Enter Item')
               for e, v in pid_df.groupby('AccessionNumber')},
            **{'num_actions_' + e: len(v) for e, v in pid_df.groupby('AccessionNumber')},
            **{'count_' + e: len(v) for e, v in pid_df.groupby('Observable') if e in freq_actions},
            **{'percentile5_' + e: v.delta_time_ms.sum() >= item_5percentile_map[e]
               for e, v in pid_df.groupby('AccessionNumber')},
        }))
        rows[-1]['percentile5_vh_count'] = \
            sum(v for k, v in rows[-1].items() if k.startswith('percentile5_'))
        rows[-1]['sec_spent_std'] = \
            np.std([v for k, v in rows[-1].items() if k.startswith('sec_spent_')])
        # Coefficients of polynomials fitted to series of continuous values
        for ts_name, ts in [('delta_sec', pid_df.delta_time_ms.values / 1000),
                            ('per_item_sec', [(v.time_unix.max() - v.time_unix.min()) / 1000
                                              for _, v in pid_df.groupby('AccessionNumber')])]:
            for poly_degree in range(4):
                for i, c in enumerate(np.polyfit(np.arange(len(ts)), ts, poly_degree)):
                    rows[-1]['poly_' + ts_name + '_deg' + str(poly_degree) + '_coeff' + str(i)] = c
    X = pd.DataFrame.from_records(rows)
    for col in X:
        X.loc[X[col].isnull(), col] = 0
    y = X.label if 'label' in pandas_df.columns else None
    features = [f for f in X.columns if f not in ['STUDENTID', 'label']]
    return X, y, features


print('Loading data')
df = load_data.train_full()
freq_actions = df.Observable.value_counts()
freq_actions = freq_actions[freq_actions > 2000].index
item_5percentile_map = {i: v.groupby('STUDENTID').delta_time_ms.sum().quantile(.05)
                        for i, v in df.groupby('AccessionNumber')}
X, y, features = extract_features(df, freq_actions, item_5percentile_map)
features = [f for f in features if not f.startswith('poly_')]  # Exclude for now TODO: See if including increases prediction correlations with TSFresh model
print(len(features), 'features:', features)

fsets = misc_util.uncorrelated_feature_sets(X[features], max_rho=.5, verbose=2)
features = fsets[0]  # TODO: Just for now try the first one, then maybe do others/group together

# Set up model training parameters
m = ensemble.ExtraTreesClassifier(200, random_state=RANDOM_SEED)
grid = {
    'model__min_samples_leaf': [1, 2, 4, 8, 16, 32],
    'model__max_features': [.1, .25, .5, .75, 1.0, 'auto'],
}
pipe = pipeline.Pipeline([
    ('model', m),
], memory=CACHE_DIR)
xval = model_selection.StratifiedKFold(4, shuffle=True, random_state=RANDOM_SEED)
gs = model_selection.GridSearchCV(pipe, grid, cv=xval, verbose=1,
                                  scoring=metrics.make_scorer(metrics.cohen_kappa_score))

print('Fitting feature importance model')
imp = gs.fit(X[features], y).best_estimator_.named_steps['model'].feature_importances_
print('\n'.join([f + ':\t' + str(i) for i, f in sorted(zip(imp, features), reverse=True)]))

print('Fitting cross-validated model')
scoring = {'AUC': metrics.make_scorer(metrics.roc_auc_score, needs_proba=True),
           'MCC': metrics.make_scorer(metrics.cohen_kappa_score),
           'Kappa': metrics.make_scorer(metrics.matthews_corrcoef)}
result = model_selection.cross_validate(gs, X[features], y, cv=xval, verbose=2, scoring=scoring)
print(result)

# Train model on all data and make predictions for competition hold-out set
print('Loading holdout data')
hidden_result = pd.read_csv('public_data/hidden_label.csv', index_col='STUDENTID')
hidden_result['pred'] = ''
hidden_result['data_length'] = ''
for datalen, train_df, holdout_df in [(10, load_data.train_10m(), load_data.holdout_10m()),
                                      (20, load_data.train_20m(), load_data.holdout_20m()),
                                      (30, load_data.train_full(), load_data.holdout_30m())]:
    print('Training/applying holdout model for', datalen, 'minutes')
    train_X, train_y, train_feats = extract_features(train_df, freq_actions, item_5percentile_map)
    holdout_X, _, holdout_feats = extract_features(holdout_df, freq_actions, item_5percentile_map)
    assert set(train_feats) == set(holdout_feats), 'Feature mismatch between train/test data'
    probs = gs.fit(train_X[train_feats], train_y).predict_proba(holdout_X[train_feats]).T[1]
    hidden_result.loc[holdout_X.STUDENTID, 'pred'] = probs
    hidden_result.loc[holdout_X.STUDENTID, 'data_length'] = datalen
    print(gs.best_estimator_)
hidden_result.to_csv('model_fe.csv')
