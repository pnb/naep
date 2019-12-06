import warnings

from sklearn import model_selection
import pandas as pd

import misc_util
import load_data


RANDOM_SEED = 11798
# Very repetitive warning message for features with no predictive power
warnings.filterwarnings('ignore', message='invalid value encountered in double_scalars')


print('Loading labels from original data')
label_map = {p: pdf.label.iloc[0] for p, pdf in load_data.train_full().groupby('STUDENTID')}

xval = model_selection.StratifiedKFold(4, shuffle=True, random_state=RANDOM_SEED)

for datalen in ['10m', '20m', '30m']:
    print('\nProcessing data length', datalen)
    train_df = pd.read_csv('features_fe/train_' + datalen + '.csv')
    for fset in ['tsfresh', 'featuretools', 'similarity']:
        tdf = pd.read_csv('features_' + fset + '/train_' + datalen + '.csv')
        assert all(tdf.STUDENTID == train_df.STUDENTID), fset + ' train STUDENTID mismatch'
        feat_names = [f for f in tdf if f not in train_df.columns]
        train_df[feat_names] = tdf[feat_names]
    train_df = train_df.fillna(0)
    features = [f for f in train_df if f not in ['STUDENTID', 'label']]
    print(len(features), 'features combined')
    train_y = [label_map[p] for p in train_df.STUDENTID]

    acc_df = misc_util.per_feature_analysis(train_df[features], train_y, xval)
    # Maybe the output doesn't really belong in this folder, but it is vaguely related to FE efforts
    acc_df.to_csv('features_fe/one_feat_accuracy-' + datalen + '.csv', index=False)
