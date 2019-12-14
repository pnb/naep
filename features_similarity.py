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
    print('\n--- Calculating Levenshtein distances for', datalen)
    if datalen == '30m':  # Use 25m data instead since it seems better
        start_unix_map = {p: v.time_unix.min() for p, v in train.groupby('STUDENTID')}
        train['start_unix'] = [start_unix_map[p] for p in train.STUDENTID]
        train = train[train.time_unix < train.start_unix + 25 * 60000].drop(columns='start_unix')
        start_unix_map = {p: v.time_unix.min() for p, v in holdout.groupby('STUDENTID')}
        holdout['start_unix'] = [start_unix_map[p] for p in holdout.STUDENTID]
        holdout = holdout[holdout.time_unix < holdout.start_unix + 25 * 60000] \
            .drop(columns='start_unix')
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

    features = {}  # STUDENTID -> OrderedDict of features
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
                if pid not in features:
                    features[pid] = OrderedDict({
                        'STUDENTID': pid,
                        'label': combined_df[combined_df.STUDENTID == pid].label.iloc[0],
                    })
                for i, v in enumerate(coord):
                    features[pid][dist_name + '_mds' + str(n_components) + '_' + str(i)] = v

    print('Saving features')
    features = pd.DataFrame.from_records(list(features.values()))
    feat_names = [f for f in features if f != 'label']
    train_X = features[features.STUDENTID.isin(train.STUDENTID)].sort_values('STUDENTID')
    holdout_X = features[features.STUDENTID.isin(holdout.STUDENTID)].sort_values('STUDENTID')
    train_X[feat_names].to_csv('features_similarity/train_' + datalen + '.csv', index=False)
    holdout_X[feat_names].to_csv('features_similarity/holdout_' + datalen + '.csv', index=False)
