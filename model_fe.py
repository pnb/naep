# Feature engineering

# Feature ideas (doubly-indented are implemented):
#       Time spent in each item (AccessionNumber)
#       Num actions for each item
#   SD of time spent across items
#   Num navigation events
#   Ordering behaviors (straight through vs. skipping problems)
#   Time spend and num actions for each item type (might be too redundant)
#   WTF behavior, especially at the end of the session
from collections import OrderedDict

import pandas as pd
from tqdm import tqdm
from sklearn import ensemble, model_selection, pipeline, metrics

import load_data


RANDOM_SEED = 11798


def extract_features(pandas_df):
    # Returns (X, y, list of feature names); y may be None if `label` is not in `pandas_df`
    print('Extracting features')
    rows = []
    for pid, pid_df in tqdm(pandas_df.groupby('STUDENTID')):
        rows.append(OrderedDict({
            'STUDENTID': pid,
            'label': pid_df.label.iloc[0] if 'label' in pid_df.columns else '',
            **{e + '_sec_spent': (v.time_unix.max() - v.time_unix.min()) / 1000
               for e, v in pid_df.groupby('AccessionNumber')},
            **{e + '_num_actions': len(v) for e, v in pid_df.groupby('AccessionNumber')},
        }))
    X = pd.DataFrame.from_records(rows)
    for col in X:
        X.loc[X[col].isnull(), col] = 0
    y = X.label if 'label' in pandas_df.columns else None
    features = [f for f in X.columns if f not in ['pid', 'label']]
    return X, y, features


print('Loading data')
df = load_data.train_full()
X, y, features = extract_features(df)

print('Fitting model')
m = ensemble.ExtraTreesClassifier(200, random_state=RANDOM_SEED)
grid = {
    'model__min_samples_leaf': [1, 2, 4, 8, 16, 32],
    'model__max_features': [.1, .25, .5, .75, 1.0, 'auto'],
}
pipe = pipeline.Pipeline([
    ('model', m),
])
xval = model_selection.StratifiedKFold(4, shuffle=True, random_state=RANDOM_SEED)
gs = model_selection.GridSearchCV(pipe, grid, cv=xval, verbose=1)
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
    train_X, train_y, train_feats = extract_features(train_df)
    holdout_X, _, holdout_feats = extract_features(holdout_df)
    assert set(train_feats) == set(holdout_feats), 'Feature mismatch between train/test data'
    probs = gs.fit(train_X[train_feats], train_y).predict_proba(holdout_X[train_feats]).T[1]
    hidden_result.loc[holdout_X.STUDENTID, 'pred'] = probs
    hidden_result.loc[holdout_X.STUDENTID, 'data_length'] = datalen
    # print(gs.best_estimator_)
hidden_result.to_csv('model_fe.csv')
