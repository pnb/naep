# Load/combine extracted feature sets, remove highly correlated features, and build models
from collections import OrderedDict
import warnings
import argparse

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


argparser = argparse.ArgumentParser(description='Train feature-level fusion models')
argparser.add_argument('model_type', type=str, choices=['extratrees', 'randomforest', 'xgboost'],
                       help='Type of model to train (classifier)')
argparser.add_argument('--entropy', action='store_true',
                       help='Split trees by information gain (default gini impurity)')
argparser.add_argument('--bootstrap', action='store_true',
                       help='Bootstrap samples in Extra-Trees or random forest')
args = argparser.parse_args()

print('Loading labels from original data')
label_map = {p: pdf.label.iloc[0] for p, pdf in load_data.train_full().groupby('STUDENTID')}

# Set up model training parameters
if args.model_type in ['extratrees', 'randomforest']:
    if args.model_type == 'extratrees':
        m = ensemble.ExtraTreesClassifier(400, random_state=RANDOM_SEED)
    else:
        m = ensemble.RandomForestClassifier(400, random_state=RANDOM_SEED)
    bayes_grid = {
        # 'min_samples_leaf': space.Integer(1, 50),
        'max_features': space.Real(.001, 1),
        'n_estimators': space.Integer(100, 500),  # Higher should be better, but let's see
        'bootstrap': [args.bootstrap],
        'max_samples': space.Real(.001, .999) if args.bootstrap else [None],
        'criterion': ['entropy' if args.entropy else 'gini'],
        'ccp_alpha': space.Real(0, .004),  # Range determined via ccp_alpha_explore.py
    }
elif args.model_type == 'xgboost':
    m = xgboost.XGBClassifier(objective='binary:logistic', random_state=RANDOM_SEED)
    bayes_grid = {
        'max_depth': space.Integer(1, 12),
        'learning_rate': space.Real(.0001, .5),
        'n_estimators': space.Integer(5, 200),
        'gamma': space.Real(0, 8),
        'subsample': space.Real(.1, 1),
        'colsample_bynode': space.Real(.1, 1),
        'reg_alpha': space.Real(0, 8),
        'reg_lambda': space.Real(0, 8),
        'num_parallel_tree': space.Integer(1, 10),
    }
model_prefix = 'predictions/' + args.model_type + ('-bootstrap' if args.bootstrap else '') + \
    ('-entropy' if args.entropy else '')

xval = model_selection.StratifiedKFold(4, shuffle=True, random_state=RANDOM_SEED)
scoring = metrics.make_scorer(misc_util.thresh_restricted_auk, needs_proba=True)
# scoring = metrics.make_scorer(metrics.cohen_kappa_score)
# scoring = metrics.make_scorer(metrics.roc_auc_score, needs_proba=True)
# scoring = metrics.make_scorer(misc_util.adjusted_thresh_kappa, needs_proba=True)
# gs = model_selection.GridSearchCV(pipe, grid, cv=xval, verbose=1, n_jobs=3, scoring=scoring)
# Getting BayesSearchCV to work requires modifying site-packages/skopt/searchcv.py per:
#   https://github.com/scikit-optimize/scikit-optimize/issues/762
gs = BayesSearchCV(m, bayes_grid, n_iter=100, n_jobs=3, cv=xval, verbose=0, scoring=scoring,
                   random_state=RANDOM_SEED, optimizer_kwargs={'n_initial_points': 20})

# Build models
hidden_result = pd.read_csv('public_data/hidden_label.csv')
train_result = []
for datalen in ['10m', '20m', '30m']:
    print('\nProcessing data length', datalen)
    feat_names = list(pd.read_csv('features_fe/filtered_features_' + datalen + '.csv').feature)
    train_df = pd.read_csv('features_fe/train_' + datalen + '.csv')[['STUDENTID'] + feat_names]
    holdout_df = pd.read_csv('features_fe/holdout_' + datalen + '.csv')[['STUDENTID'] + feat_names]
    for fset in ['features_tsfresh', 'features_featuretools', 'features_similarity']:
        feat_names = list(pd.read_csv(fset + '/filtered_features_' + datalen + '.csv').feature)
        tdf = pd.read_csv(fset + '/train_' + datalen + '.csv')[['STUDENTID'] + feat_names]
        hdf = pd.read_csv(fset + '/holdout_' + datalen + '.csv')[['STUDENTID'] + feat_names]
        assert all(tdf.STUDENTID == train_df.STUDENTID), fset + ' train STUDENTID mismatch'
        assert all(hdf.STUDENTID == holdout_df.STUDENTID), fset + ' holdout STUDENTID mismatch'
        train_df[feat_names] = tdf[feat_names]
        holdout_df[feat_names] = hdf[feat_names]
    train_df = train_df.fillna(0)  # TODO: What null values could remain?
    holdout_df = holdout_df.fillna(0)
    features = [f for f in train_df if f not in ['STUDENTID', 'label']]
    print(len(features), 'features combined')
    # TODO: Might be able to tune max_rho to get a higher AUC vs. higher kappa for later fusion
    fsets = misc_util.uncorrelated_feature_sets(train_df[features], max_rho=9,
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
    pd.DataFrame(gs.cv_results_).to_csv(model_prefix + '-cv_' + datalen + '.csv', index=False)
    print('Hyperparameter search best estimator:', gs.best_estimator_)
    print('Hyperparameter search scorer:', gs.scorer_)
    print('Hyperparameter search best score:', gs.best_score_)
    print('Train data positive class base rate:', np.mean(train_y))
    print('Predicted base rate (> .5 threshold):', np.mean(probs > .5))
    for pid, pred in zip(holdout_df.STUDENTID.values, probs):
        hidden_result.loc[hidden_result.STUDENTID == pid, 'pred'] = pred
        hidden_result.loc[hidden_result.STUDENTID == pid, 'data_length'] = datalen

hidden_result.to_csv(model_prefix + '.csv', index=False)
pd.DataFrame.from_records(train_result).to_csv(model_prefix + '-train.csv', index=False)
