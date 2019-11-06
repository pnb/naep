# Sequence similarity features:
#   Mean/functionals of edit distance to all training instances in positive/negative class
# For different sequences:
#   AccessionNumber (exercise ID); length ~= 1/minute of data (10, 20 30)
#   Time spent on AccessionNumber in percentile bins (related to the 5% cutoff); length same
#   TODO: Time chunk, navigation
#   TODO: Time chunk * AccessionNumber -- e.g., divide each AccessionNumber into 30s chunks with a leftover chunk and make sequences
#   TODO: Observable (action); length probably too long
#   TODO: mean of [distance for Observable within each exercise] -- difficult to implement
#
# Levenshtein distance runs in O(N^2) so we must be careful with sequence length
from collections import OrderedDict

import editdistance
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn import ensemble, model_selection, pipeline, metrics

import misc_util
import load_data


RANDOM_SEED = 11798
CACHE_DIR = '/Users/pnb/sklearn_cache'


# Set up feature importance model training parameters
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

print('Loading data')
all_rows = load_data.all_unique_rows()
# Map item ID -> PID -> amount of time spent on item (for calculating percentiles/quantiles)
item_time_distribution = all_rows.groupby(['AccessionNumber', 'STUDENTID']).delta_time_ms.sum()
# Map item ID -> quantiles
item_time_quantiles = {i: item_time_distribution[i].quantile([.025, .05, .1, .3, .5, .75, 1])
                       for i in all_rows.AccessionNumber.unique()}

for datalen, train, holdout in [('10m', load_data.train_10m(), load_data.holdout_10m()),
                                ('20m', load_data.train_20m(), load_data.holdout_20m()),
                                ('30m', load_data.train_full(), load_data.holdout_30m())]:
    print('\nCalculating Levenshtein distance features for', datalen)
    combined_df = train.append(holdout, ignore_index=True, sort=False)

    # Precalculate item sequences with hash(), since comparing strings is slow otherwise
    seq_accession = {}
    for pid, df in tqdm(combined_df.groupby('STUDENTID'), desc='Making item sequences'):
        seq_accession[pid] = [hash(i) for i in df.AccessionNumber.drop_duplicates()]

    # Sequence of time spent per problem in terms of percentiles; somewhat addresses the target
    # label definition of spending at least 5th percentile time on each item
    # Precalculate these sequences since they can be time-consuming
    seq_percentile = {}
    for pid, df in tqdm(combined_df.groupby('STUDENTID'), desc='Making time spent sequences'):
        starts = df[df.AccessionNumber.shift(1) != df.AccessionNumber].time_unix
        stops = df[df.AccessionNumber.shift(-1) != df.AccessionNumber].time_unix
        assert pd.isna(starts).sum() == 0 and pd.isna(stops).sum() == 0, 'Unexpected NaNs'
        # Largest quantile that each time spent is less than
        seq_percentile[pid] = \
            [item_time_quantiles[i][stop - start < item_time_quantiles[i]].index[0]
             for i, start, stop in zip(df.loc[starts.index, 'AccessionNumber'], starts, stops)]

    # Calculate distance matrices to avoid re-calculation during feature extraction
    accession_dist = {}  # A <-> B distance
    percentile_dist = {}
    for dist_name, mat, seqs in [('accessiondist', accession_dist, seq_accession),
                                 ('percentiledist', percentile_dist, seq_percentile)]:
        for pida in tqdm(combined_df.STUDENTID.unique(), desc='Distances for ' + dist_name):
            mat[pida] = {}
            # We only care about distance to training, so will not use combined_df here
            for pidb in train.STUDENTID.unique():
                if pidb not in mat:
                    mat[pidb] = {}
                if pidb not in mat[pida]:  # Symmetric, only calculate once
                    mat[pida][pidb] = mat[pidb][pida] = editdistance.eval(seqs[pida], seqs[pidb])

    dist_quantiles = [.01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99]
    features = {}  # STUDENTID -> OrderedDict of features
    for dist_name, mat in [('accessiondist', accession_dist), ('percentiledist', percentile_dist)]:
        positive_pids = train[train.label == 1].STUDENTID.values
        negative_pids = train[train.label == 0].STUDENTID.values
        assert len(positive_pids) > 0 and len(negative_pids) > 0
        for pid in tqdm(combined_df.STUDENTID.unique(), desc='Features for ' + dist_name):
            # Include distance to self; otherwise tiny differences leak label info
            dist_pos = np.array([mat[pid][p] for p in positive_pids])
            dist_neg = np.array([mat[pid][p] for p in negative_pids])
            if pid not in features:
                features[pid] = OrderedDict({
                    'STUDENTID': pid,
                    'label': combined_df[combined_df.STUDENTID == pid].label.iloc[0],
                })
            features[pid].update({
                dist_name + '_mean_pos': np.mean(dist_pos),
                dist_name + '_mean_neg': np.mean(dist_neg),
                dist_name + '_mean_diff': np.mean(dist_pos) - np.mean(dist_neg),
                dist_name + '_std_pos': np.std(dist_pos),
                dist_name + '_std_neg': np.std(dist_neg),
                **{dist_name + '_quantile_pos_' + str(q): val
                   for q, val in zip(dist_quantiles, np.quantile(dist_pos, dist_quantiles))},
                **{dist_name + '_quantile_neg_' + str(q): val
                   for q, val in zip(dist_quantiles, np.quantile(dist_neg, dist_quantiles))},
            })
    features = pd.DataFrame.from_records(list(features.values()))
    feat_names = [f for f in features if f != 'STUDENTID' and f != 'label']

    print(len(feat_names), 'features extracted')
    fsets = misc_util.uncorrelated_feature_sets(features[feat_names], max_rho=.8,
                                                remove_perfect_corr=True, verbose=1)
    feat_names = fsets[0]
    print(len(feat_names), 'features after removing highly correlated features')

    print('Building feature importance model for', datalen)
    train_X = features[features.STUDENTID.isin(train.STUDENTID)]
    train_y = train_X.label
    holdout_X = features[features.STUDENTID.isin(holdout.STUDENTID)]
    imp = gs.fit(train_X[feat_names], train_y)
    imp_m = imp.best_estimator_
    importances = pd.Series(imp_m.named_steps['model'].feature_importances_, index=feat_names)
    print(importances)
    print('Grid search best kappa:', imp.best_score_)
    feat_names = [f for f in feat_names if importances[f] > .001]  # TODO: This cutoff is maybe too low here (only 20 features) since the feature importances add up to 1; probably worth revisiting in other files too
    print(len(feat_names), 'features after keeping only important features')
    # misc_util.tree_error_analysis(train_X[feat_names], train_y, xval, ['negative', 'positive'],
    #                               'graphs/seq_similarity_dt_' + datalen + '-')

    print('Saving features')
    train_X[['STUDENTID'] + feat_names] \
        .to_csv('features_similarity/train_' + datalen + '.csv', index=False)
    holdout_X[['STUDENTID'] + feat_names] \
        .to_csv('features_similarity/holdout_' + datalen + '.csv', index=False)
