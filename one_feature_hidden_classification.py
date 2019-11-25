# Classify hidden vs. not hidden status to determine if there are obvious differences between the
# given training data and the holdout data. This might also highlight mistakes made during the data
# dividing process (e.g., if the 10m training data is not the same length as 10m holdout).
import warnings

from sklearn import model_selection
import pandas as pd

import misc_util
import load_data


RANDOM_SEED = 11798
# Very repetitive warning message for features with no predictive power
warnings.filterwarnings('ignore', message='invalid value encountered in double_scalars')


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
    train_df = pd.concat([train_df, holdout_df])

    train_y = [1 if pid in holdout_df.STUDENTID.values else 0 for pid in train_df.STUDENTID]
    print('Proportion of data that is holdout:', sum(train_y) / len(train_y))

    acc_df = misc_util.per_feature_analysis(train_df[features], train_y, xval)
    # Maybe the output doesn't really belong in this folder, but it is vaguely related to FE efforts
    acc_df.to_csv('features_fe/is_holdout_accuracy-' + datalen + '.csv', index=False)
