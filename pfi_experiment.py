# Build default model and do permutation feature importance (PFI)
import warnings

import pandas as pd
import numpy as np
from sklearn import ensemble, model_selection, metrics, inspection
from skopt import BayesSearchCV, space
import shap

import load_data
import misc_util


RANDOM_SEED = 11798
# A very repetitive BayesSearchCV warning I'd like to ignore
warnings.filterwarnings('ignore', message='The objective has been evaluated at this point before.')


print('Loading labels from original data')
label_map = {p: pdf.label.iloc[0] for p, pdf in load_data.train_full().groupby('STUDENTID')}

# Set up model training parameters
m = ensemble.ExtraTreesClassifier(500, bootstrap=True, random_state=RANDOM_SEED)
bayes_grid = {
    'max_features': space.Real(.001, 1),
    'max_samples': space.Real(.001, .999),  # For bootstrapping
    'ccp_alpha': space.Real(0, .004),  # Range determined via ccp_alpha_explore.py
}

xval = model_selection.StratifiedKFold(4, shuffle=True, random_state=RANDOM_SEED)
scoring = metrics.make_scorer(misc_util.adjusted_thresh_kappa, needs_proba=True)
# Getting BayesSearchCV to work requires modifying site-packages/skopt/searchcv.py per:
#   https://github.com/scikit-optimize/scikit-optimize/issues/762
gs = BayesSearchCV(m, bayes_grid, n_iter=100, n_jobs=3, cv=xval, verbose=0, scoring=scoring,
                   random_state=RANDOM_SEED, optimizer_kwargs={'n_initial_points': 20})

# Build model for 30m data as an example
train_result = []
print('Loading data')
feat_names = list(pd.read_csv('features_fe/filtered_features_30m.csv').feature)
train_df = pd.read_csv('features_fe/train_30m.csv')[['STUDENTID'] + feat_names]
holdout_df = pd.read_csv('features_fe/holdout_30m.csv')[['STUDENTID'] + feat_names]
for fset in ['features_tsfresh', 'features_featuretools']:
    feat_names = list(pd.read_csv(fset + '/filtered_features_30m.csv').feature)
    tdf = pd.read_csv(fset + '/train_30m.csv')[['STUDENTID'] + feat_names]
    hdf = pd.read_csv(fset + '/holdout_30m.csv')[['STUDENTID'] + feat_names]
    assert all(tdf.STUDENTID == train_df.STUDENTID), fset + ' train STUDENTID mismatch'
    assert all(hdf.STUDENTID == holdout_df.STUDENTID), fset + ' holdout STUDENTID mismatch'
    train_df[feat_names] = tdf[feat_names]
    holdout_df[feat_names] = hdf[feat_names]
train_df = train_df.fillna(0)
holdout_df = holdout_df.fillna(0)
features = [f for f in train_df if f not in ['STUDENTID', 'label']]
print(len(features), 'features combined')
fsets = misc_util.uncorrelated_feature_sets(train_df[features], max_rho=1,
                                            remove_perfect_corr=True, verbose=2)
features = fsets[0]
print(len(features), 'features after removing perfectly correlated features')
y = np.array([label_map[p] for p in train_df.STUDENTID])

pfi_df = pd.DataFrame({'feature': features, 'pfi_mean': 0})
shap_df = pd.DataFrame(index=train_df.index, columns=features)
for fold_i, (train_i, test_i) in enumerate(xval.split(y, y)):
    print('\nFitting cross-val model fold', fold_i)
    train_X = train_df[features].iloc[train_i]
    test_X = train_df[features].iloc[test_i]
    gs.fit(train_X, y[train_i])

    print('SHAP feature importance')
    explainer = shap.TreeExplainer(gs.best_estimator_)
    shap_df.iloc[test_i] = explainer.shap_values(test_X)[0]

    print('Permutation feature importance')
    pfi = inspection.permutation_importance(gs, test_X, y[test_i], scoring=scoring, n_jobs=3,
                                            n_repeats=20, random_state=RANDOM_SEED)
    pfi_df.pfi_mean += pfi.importances_mean

mabs_shap = pd.DataFrame(shap_df.abs().mean(), columns=['mabs_shap'])
mabs_shap.index.name = 'feature'
mabs_shap.to_csv('features_fe/shap-30m.csv')
pfi_df.pfi_mean /= xval.get_n_splits()
pfi_df.to_csv('features_fe/pfi-30m.csv', index=False)

# PFI results are very odd. It returns surprising features as the most important (it's especially
# fond of backspace features), and it correlates negatively with both SHAP and one-feature AUC
# importance. Hence, I don't trust it.
