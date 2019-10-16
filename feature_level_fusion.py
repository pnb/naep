# Load/combine extracted feature sets, remove highly correlated features, and build models
from collections import OrderedDict

import pandas as pd
import numpy as np
from sklearn import ensemble, pipeline, model_selection, metrics
import xgboost

import load_data
import misc_util


RANDOM_SEED = 11798
CACHE_DIR = '/Users/pnb/sklearn_cache'


print('Loading labels from original data')
label_map = {row.STUDENTID: row.label for _, row in load_data.train_full().iterrows()}

# Set up model training parameters
m = ensemble.ExtraTreesClassifier(400, random_state=RANDOM_SEED)
m = xgboost.XGBClassifier(max_depth=3, learning_rate=.1, n_estimators=100, random_state=RANDOM_SEED, gamma=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1, reg_alpha=0, reg_lambda=1)
grid = {
    # 'model__min_samples_leaf': [1, 2, 4, 8, 16, 32],
    # 'model__max_features': [.1, .25, .5, .75, 1.0, 'auto'],

    'model__max_depth': [1, 2, 3, 4],  # XGBoost
    'model__learning_rate': [.05, .1, .2, .3],
    'model__n_estimators': [25, 50, 100, 200],
}
pipe = pipeline.Pipeline([
    ('model', m),
], memory=CACHE_DIR)
xval = model_selection.StratifiedKFold(4, shuffle=True, random_state=RANDOM_SEED)
gs = model_selection.GridSearchCV(pipe, grid, cv=xval, verbose=1,
                                  scoring=metrics.make_scorer(metrics.cohen_kappa_score))
scoring = {'AUC': metrics.make_scorer(metrics.roc_auc_score, needs_proba=True),
           'MCC': metrics.make_scorer(metrics.cohen_kappa_score),
           'Kappa': metrics.make_scorer(metrics.matthews_corrcoef)}

# Build models
hidden_result = pd.read_csv('public_data/hidden_label.csv')
train_result = []
for datalen in ['10m', '20m', '30m']:
    print('Processing data length', datalen)
    train_df = pd.read_csv('features_fe/train_' + datalen + '.csv')
    holdout_df = pd.read_csv('features_fe/holdout_' + datalen + '.csv')
    for fset in ['tsfresh', 'featuretools']:
        tdf = pd.read_csv('features_' + fset + '/train_' + datalen + '.csv')
        hdf = pd.read_csv('features_' + fset + '/holdout_' + datalen + '.csv')
        feat_names = [f for f in tdf if f not in train_df.columns]
        train_df[feat_names] = tdf[feat_names]
        holdout_df[feat_names] = hdf[feat_names]
    features = [f for f in train_df if f not in ['STUDENTID', 'label']]
    print(len(features), 'features combined')
    fsets = misc_util.uncorrelated_feature_sets(train_df[features], max_rho=.8,
                                                remove_perfect_corr=True, verbose=1)
    features = fsets[0]
    print(len(features), 'features after removing highly correlated features')
    train_y = [label_map[p] for p in train_df.STUDENTID]

    # First cross-validate on training data to test accuracy on local (non-LB) data
    print('\nFitting cross-val model for', datalen, 'data')
    preds = model_selection.cross_val_predict(gs, train_df[features], train_y, cv=xval, verbose=2,
                                              method='predict_proba').T[1]
    print('AUC =', metrics.roc_auc_score(train_y, preds))
    print('Kappa =', metrics.cohen_kappa_score(train_y, preds > .5))
    print('MCC =', metrics.matthews_corrcoef(train_y, preds > .5))
    for pid, truth, pred in zip(train_df.STUDENTID.values, train_y, preds):
        train_result.append(OrderedDict({'STUDENTID': pid, 'label': truth, 'pred': pred,
                                         'data_length': datalen}))

    # Fit on all training data and apply to holdout data
    print('\nFitting holdout model for', datalen, 'data')
    probs = gs.fit(train_df[features], train_y).predict_proba(holdout_df[features]).T[1]
    print('Grid search best estimator:', gs.best_estimator_)
    print('Grid search scorer:', gs.scorer_)
    print('Grid search best score:', gs.best_score_)
    print('Train data positive class base rate:', np.mean(train_y))
    print('Predicted base rate (> .5 threshold):', np.mean(probs > .5))
    for pid, pred in zip(holdout_df.STUDENTID.values, probs):
        hidden_result.loc[hidden_result.STUDENTID == pid, 'pred'] = pred
        hidden_result.loc[hidden_result.STUDENTID == pid, 'data_length'] = datalen

hidden_result.to_csv('feature_level_fusion.csv', index=False)
pd.DataFrame.from_records(train_result).to_csv('feature_level_fusion-train.csv', index=False)
