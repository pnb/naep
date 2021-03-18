import os

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, pearsonr
from tqdm import tqdm


print('Loading data')
train_dfs = []
holdout_dfs = []
for folder in ['features_fe', 'features_featuretools', 'features_tsfresh']:
    train_dfs.append(pd.read_csv(os.path.join(folder, 'train_30m.csv')))
    holdout_dfs.append(pd.read_csv(os.path.join(folder, 'holdout_30m.csv')))
train_df = pd.concat(train_dfs)
holdout_df = pd.concat(holdout_dfs)
for df in [train_df, holdout_df]:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

print('Loading previous one-feature results')
onefeat_df = pd.read_csv('features_fe/one_feat_accuracy-30m.csv')
print(len(onefeat_df), 'features')

# K-S test for comparison purposes to the method actually used
print('Kolmogorov-Smirnov test for distribution similarity')
for row_i, row in tqdm(onefeat_df.iterrows(), total=len(onefeat_df)):
    if row.feature in train_df and row.feature in holdout_df:
        ks = ks_2samp(train_df[row.feature], holdout_df[row.feature])  # .pvalue and .statistic
        onefeat_df.at[row_i, 'ks_statistic'] = ks.statistic
        onefeat_df.at[row_i, 'ks_p'] = ks.pvalue

# Compare to holdout_vs_train_mean_test_auc (and the _train_ version)
# A large K-S statistic (and small p-value) indicates that the distributions are different
print((onefeat_df.ks_p < .05).sum(), 'K-S p-values < .05')
valid_ks = onefeat_df[['ks_statistic', 'holdout_vs_train_mean_test_auc',
                       'holdout_vs_train_mean_train_auc']].dropna()
print('Correlation versus AUC method (4-fold test AUC):  r=%.3f, p=%.3f' %
      pearsonr(valid_ks.ks_statistic, valid_ks.holdout_vs_train_mean_test_auc))
print('Correlation versus AUC method (4-fold train AUC): r=%.3f, p=%.3f' %
      pearsonr(valid_ks.ks_statistic, valid_ks.holdout_vs_train_mean_train_auc))
# K-S correlates a lot higher with train AUC than test AUC. This is maybe a good argument for using
# the AUC method (more robust evidence of distribution similarity).
