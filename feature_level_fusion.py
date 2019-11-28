# Load/combine extracted feature sets, remove highly correlated features, and build models
from collections import OrderedDict
import warnings

import pandas as pd
import numpy as np
from sklearn import ensemble, pipeline, model_selection, metrics
import xgboost
from skopt import BayesSearchCV, space

import load_data
import misc_util


RANDOM_SEED = 11798
CACHE_DIR = '/Users/pnb/sklearn_cache'
# A very repetitive BayesSearchCV warning I'd like to ignore
warnings.filterwarnings('ignore', message='The objective has been evaluated at this point before.')


print('Loading labels from original data')
label_map = {row.STUDENTID: row.label for _, row in load_data.train_full().iterrows()}

# Set up model training parameters
m = ensemble.ExtraTreesClassifier(400, random_state=RANDOM_SEED)
# m = xgboost.XGBClassifier(objective='binary:logistic', random_state=RANDOM_SEED)
bayes_grid = {
    'model__min_samples_leaf': space.Integer(1, 50),  # Extra-Trees
    'model__max_features': space.Real(.001, 1),
    'model__n_estimators': space.Integer(100, 500),  # Higher should be better, but let's see
    'model__criterion': ['gini', 'entropy'],
    'model__bootstrap': [True, False],

    # 'model__max_depth': space.Integer(1, 12),  # XGBoost
    # 'model__learning_rate': space.Real(.0001, .5),
    # 'model__n_estimators': space.Integer(5, 200),
    # 'model__gamma': space.Real(0, 8),
    # 'model__subsample': space.Real(.1, 1),
    # 'model__colsample_bynode': space.Real(.1, 1),
    # 'model__reg_alpha': space.Real(0, 8),
    # 'model__reg_lambda': space.Real(0, 8),
}
grid = {
    # 'uncorrelated_fs__max_rho': [.4, .5, .55, .6, .65, .7, .75, .8, .85, .9],

    'model__min_samples_leaf': [1, 2, 4, 8, 16, 32],
    'model__max_features': [.1, .25, .5, .75, 1.0, 'auto'],
}
xval = model_selection.StratifiedKFold(4, shuffle=True, random_state=RANDOM_SEED)
pipe = pipeline.Pipeline([
    # ('uncorrelated_fs', misc_util.UncorrelatedFeatureSelector(verbose=2)),
    ('model', m),
], memory=CACHE_DIR)
scoring = metrics.make_scorer(misc_util.thresh_restricted_auk, needs_proba=True)
# scoring = metrics.make_scorer(metrics.cohen_kappa_score)
# scoring = metrics.make_scorer(metrics.roc_auc_score, needs_proba=True)
# scoring = metrics.make_scorer(misc_util.adjusted_thresh_kappa, needs_proba=True)
# gs = model_selection.GridSearchCV(pipe, grid, cv=xval, verbose=1, n_jobs=3, scoring=scoring)
# Getting BayesSearchCV to work requires modifying site-packages/skopt/searchcv.py per:
#   https://github.com/scikit-optimize/scikit-optimize/issues/762
gs = BayesSearchCV(pipe, bayes_grid, n_iter=100, n_jobs=3, cv=xval, verbose=0, scoring=scoring,
                   random_state=RANDOM_SEED, optimizer_kwargs={'n_initial_points': 20})

# Build models
hidden_result = pd.read_csv('public_data/hidden_label.csv')
train_result = []
for datalen in ['10m', '20m', '30m']:
    print('\nProcessing data length', datalen)
    train_df = pd.read_csv('features_fe/train_' + datalen + '.csv')
    holdout_df = pd.read_csv('features_fe/holdout_' + datalen + '.csv')
    for fset in ['tsfresh', 'featuretools']:  # , 'similarity']:
        tdf = pd.read_csv('features_' + fset + '/train_' + datalen + '.csv')
        hdf = pd.read_csv('features_' + fset + '/holdout_' + datalen + '.csv')
        feat_names = [f for f in tdf if f not in train_df.columns]
        train_df[feat_names] = tdf[feat_names]
        holdout_df[feat_names] = hdf[feat_names]
    features = [f for f in train_df if f not in ['STUDENTID', 'label']]
    print(len(features), 'features combined')
    # Remove features that predict holdout vs. train very well
    acc_holdout = pd.read_csv('features_fe/is_holdout_accuracy-' + datalen + '.csv')
    high_diff_feats = acc_holdout[acc_holdout.mean_test_kappa > .1].feature.values
    features = [f for f in features if f not in high_diff_feats]
    print(len(features), 'features after removing those with differing train/holdout distributions')
    # TODO: Might be able to tune max_rho to get a higher AUC vs. higher kappa for later fusion
    # TODO: try prioritize features very likely related to outcome# priority = [f for f in features if 'percentile5_' in f or 'answer_' in f]
    fsets = misc_util.uncorrelated_feature_sets(train_df[features], max_rho=.8,
                                                remove_perfect_corr=True, verbose=2)
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
    pd.DataFrame(gs.cv_results_).to_csv('fusion_cv_results_' + datalen + '.csv', index=False)
    print('Hyperparameter search best estimator:', gs.best_estimator_)
    print('Hyperparameter search scorer:', gs.scorer_)
    print('Hyperparameter search best score:', gs.best_score_)
    print('Train data positive class base rate:', np.mean(train_y))
    print('Predicted base rate (> .5 threshold):', np.mean(probs > .5))
    for pid, pred in zip(holdout_df.STUDENTID.values, probs):
        hidden_result.loc[hidden_result.STUDENTID == pid, 'pred'] = pred
        hidden_result.loc[hidden_result.STUDENTID == pid, 'data_length'] = datalen

hidden_result.to_csv('feature_level_fusion.csv', index=False)
pd.DataFrame.from_records(train_result).to_csv('feature_level_fusion-train.csv', index=False)
