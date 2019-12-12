import argparse
import os
import warnings

import pandas as pd
import numpy as np
from sklearn import model_selection

import misc_util


RANDOM_SEED = 11798
# Very repetitive warning message for features with no predictive power
warnings.filterwarnings('ignore', message='invalid value encountered in double_scalars')


argparser = argparse.ArgumentParser(
    description='Run supervised feature selection and then unsupervised feature selection on a set '
    'of features. This script assumes the existance of six feature files in the specified folder, '
    'with names like "train_30m.csv" and "holdout_10m.csv". Output will be written to files in the '
    'specified folder with names like "filtered_features_20m.csv"')
argparser.add_argument('feature_folder', type=str, help='Path to folder with feature files')
args = argparser.parse_args()


print('Loading data')
dfs = {
    'train_10m': pd.read_csv(os.path.join(args.feature_folder, 'train_10m.csv')),
    'train_20m': pd.read_csv(os.path.join(args.feature_folder, 'train_20m.csv')),
    'train_30m': pd.read_csv(os.path.join(args.feature_folder, 'train_30m.csv')),
    'holdout_10m': pd.read_csv(os.path.join(args.feature_folder, 'holdout_10m.csv')),
    'holdout_20m': pd.read_csv(os.path.join(args.feature_folder, 'holdout_20m.csv')),
    'holdout_30m': pd.read_csv(os.path.join(args.feature_folder, 'holdout_30m.csv')),
}

# Add in labels, make STUDENTID a column again
train_labels = pd.read_csv('public_data/data_train_label.csv', index_col='STUDENTID')
for k, df in dfs.items():
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # TODO: Maybe NaN replacement is unnecessary in sklearn 0.22
    df.fillna(0, inplace=True)
    if 'train_' in k:
        df['label'] = [train_labels.loc[p].EfficientlyCompletedBlockB for p in df.STUDENTID]

for datalen in ['10m', '20m', '30m']:
    print('\nProcessing data length', datalen)
    train_df, holdout_df = dfs['train_' + datalen], dfs['holdout_' + datalen]
    feat_names = [f for f in train_df if f not in ['STUDENTID', 'label']]
    print(len(feat_names), 'features')
    feat_names = [f for f in feat_names if len(train_df[f].unique()) > 1]
    print(len(feat_names), 'features after removing those with no variance')

    print('--- Supervised selection')
    xval = model_selection.StratifiedKFold(4, shuffle=True, random_state=RANDOM_SEED)
    acc_df = misc_util.per_feature_analysis(train_df[feat_names], train_df.label, xval)
    feat_names = list(acc_df[acc_df.min_test_auc > .5].feature)
    print(len(feat_names), 'features after keeping only min AUC > .5 features')

    print('--- Train vs. holdout distribution similarity selection')
    combo_df = pd.concat([train_df, holdout_df], sort=False).fillna(0)
    condition_y = [0] * len(train_df) + [1] * len(holdout_df)
    condition_df = misc_util.per_feature_analysis(combo_df[feat_names], condition_y, xval)
    feat_names = list(condition_df[condition_df.mean_test_auc < .55].feature)
    print(len(feat_names), 'features after discarding holdout vs. train classification AUC >= .55')

    print('--- Unsupervised selection')
    # Set priority order according to one-feature accuracy (TCFS)
    priority = []
    prev_acc = 999
    for _, row in acc_df.sort_values('mean_test_auc', ascending=False).iterrows():
        if len(priority) == 0 or row.mean_test_auc < prev_acc:
            priority.append([row.feature])
        else:
            priority[-1].append(row.feature)
        prev_acc = row.mean_test_auc
    fsets = misc_util.uncorrelated_feature_sets(train_df[feat_names], max_rho=.8, verbose=1,
                                                remove_perfect_corr=True, priority_order=priority)
    feat_names = fsets[0]
    print(len(feat_names), 'features after removing highly-correlated features')

    feat_names = [f for f in feat_names if f in holdout_df.columns]
    print(len(feat_names), 'features after including only those in common between train/holdout')

    print('Saving feature list')
    pd.DataFrame({'feature': feat_names}).to_csv(
        os.path.join(args.feature_folder, 'filtered_features_' + datalen + '.csv'), index=False)
