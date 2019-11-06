from sklearn import model_selection
import pandas as pd

import misc_util
import load_data


RANDOM_SEED = 11798


print('Loading labels from original data')
label_map = {row.STUDENTID: row.label for _, row in load_data.train_full().iterrows()}

xval = model_selection.StratifiedKFold(4, shuffle=True, random_state=RANDOM_SEED)

for datalen in ['10m', '20m', '30m']:
    print('\nProcessing data length', datalen)
    train_df = pd.read_csv('features_fe/train_' + datalen + '.csv')
    holdout_df = pd.read_csv('features_fe/holdout_' + datalen + '.csv')
    for fset in ['tsfresh', 'featuretools', 'similarity']:
        tdf = pd.read_csv('features_' + fset + '/train_' + datalen + '.csv')
        hdf = pd.read_csv('features_' + fset + '/holdout_' + datalen + '.csv')
        feat_names = [f for f in tdf if f not in train_df.columns]
        train_df[feat_names] = tdf[feat_names]
        holdout_df[feat_names] = hdf[feat_names]
    features = [f for f in train_df if f not in ['STUDENTID', 'label']]
    print(len(features), 'features combined')
    train_y = [label_map[p] for p in train_df.STUDENTID]

    acc_df = misc_util.per_feature_analysis(train_df[features], train_y, xval)
    # Maybe the output doesn't really belong in this folder, but it is vaguely related to FE efforts
    acc_df.to_csv('features_fe/one_feat_accuracy-' + datalen + '.csv', index=False)
