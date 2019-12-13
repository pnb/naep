# Sequence similarity features for different sequences:
#   AccessionNumber (exercise ID); length ~= 1/minute of data (10, 20 30)
#   Time chunk * AccessionNumber; length = datalen / chunk size
#
# Levenshtein distance runs in O(N^2) so we must be careful with sequence length
from collections import OrderedDict

import editdistance
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn import model_selection, manifold

import load_data


RANDOM_SEED = 11798


print('Loading data')
for datalen, train, holdout in [('10m', load_data.train_10m(), load_data.holdout_10m()),
                                ('20m', load_data.train_20m(), load_data.holdout_20m()),
                                ('30m', load_data.train_full(), load_data.holdout_30m())]:
    print('\nCalculating Levenshtein distance features for', datalen)
    combined_df = train.append(holdout, ignore_index=True, sort=False)

    # Precalculate item sequences with hash(), since comparing strings is slow otherwise
    seq_accession = {}
    for pid, df in tqdm(combined_df.groupby('STUDENTID'), desc='Making item sequences'):
        seq_accession[pid] = [hash(i) for i in df.AccessionNumber.drop_duplicates()]

    # Sequences of chunks of time per item (AccessionNumber)
    seq_chunkitem = {}
    for pid, df in tqdm(combined_df.groupby('STUDENTID'), desc='Making time chunk sequences'):
        seq_chunkitem[pid] = []
        for chunk_start in range(df.time_unix.min(), df.time_unix.max(), 5000):
            chunk_end = chunk_start + 5000
            chunk = df[(df.time_unix >= chunk_start) & (df.time_unix < chunk_end)]
            if len(chunk) > 0:  # Chunk*AccessionNumber
                seq_chunkitem[pid].append(hash(chunk.AccessionNumber.iloc[-1]))
            else:  # No activity in chunk, so propagate the previous value
                seq_chunkitem[pid].append(seq_chunkitem[pid][-1])

    # Calculate distance matrices to avoid re-calculation during feature extraction
    accession_dist = {}  # A <-> B distance
    chunkitem_dist = {}
    for dist_name, mat, seqs in [('accessiondist', accession_dist, seq_accession),
                                 ('chunkitemdist', chunkitem_dist, seq_chunkitem)]:
        for pida in tqdm(combined_df.STUDENTID.unique(), desc='Distances for ' + dist_name):
            mat[pida] = {}
            for pidb in combined_df.STUDENTID.unique():
                if pidb not in mat:
                    mat[pidb] = {}
                if pidb not in mat[pida]:  # Symmetric, only calculate once
                    mat[pida][pidb] = mat[pidb][pida] = editdistance.eval(seqs[pida], seqs[pidb])

    # Set up PIDs to use in "training" the features
    print('Splitting data')
    xval_X = train.STUDENTID.unique()
    xval_y = [train[train.STUDENTID == p].label.iloc[0] for p in xval_X]
    train_folds = [xval_X]  # Initially train on all training data, apply to all holdout
    test_folds = [holdout.STUDENTID.unique()]
    xval = model_selection.StratifiedKFold(4, shuffle=True, random_state=RANDOM_SEED)
    for train_i, test_i in xval.split(xval_X, xval_y):
        train_folds.append(xval_X[train_i])
        test_folds.append(xval_X[test_i])

    dist_quantiles = [.01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99]
    features = {}  # STUDENTID -> OrderedDict of features
    for fold_i, (train_pids, test_pids) in enumerate(zip(train_folds, test_folds)):
        print('Features for fold', fold_i + 1, 'of', len(train_folds))
        for dist_name, mat in [('accessiondist', accession_dist),
                               ('chunkitemdist', chunkitem_dist)]:
            train_subset = train[train.STUDENTID.isin(train_pids)]
            positive_pids = train_subset[train_subset.label == 1].STUDENTID.unique()
            negative_pids = train_subset[train_subset.label == 0].STUDENTID.unique()
            assert len(positive_pids) > 0 and len(negative_pids) > 0
            for pid in tqdm(test_pids, desc='Features for ' + dist_name):
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

    # Coordinates on a lower-dimensional manifold derived from distances
    print('--- Multidimensional scaling')
    for dist_name, mat in [('accessiondist', accession_dist),
                           ('chunkitemdist', chunkitem_dist)]:
        for n_components in range(1, 4):
            print(dist_name, 'MDS features with', n_components, 'component(s)')
            mds = manifold.MDS(n_components=n_components, metric=True, n_init=5, max_iter=300,
                               verbose=1, random_state=RANDOM_SEED, dissimilarity='precomputed')
            mds_X = pd.DataFrame(mat)
            coords = mds.fit_transform(mds_X, None)
            for pid, coord in tqdm(zip(mds_X.index, coords), desc='Adding coordinate features'):
                for i, v in enumerate(coord):
                    features[pid][dist_name + '_mds' + str(n_components) + '_' + str(i)] = v

    print('Saving features')
    features = pd.DataFrame.from_records(list(features.values()))
    feat_names = [f for f in features if f != 'label']
    train_X = features[features.STUDENTID.isin(train.STUDENTID)].sort_values('STUDENTID')
    holdout_X = features[features.STUDENTID.isin(holdout.STUDENTID)].sort_values('STUDENTID')
    train_X[feat_names].to_csv('features_similarity/train_' + datalen + '.csv', index=False)
    holdout_X[feat_names].to_csv('features_similarity/holdout_' + datalen + '.csv', index=False)
