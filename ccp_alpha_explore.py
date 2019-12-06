# Find a decent range of values for the cost-complexity pruning (ccp_alpha) parameter in decision
# trees, which is problem-specific
from sklearn import ensemble
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

import load_data
import misc_util


RANDOM_SEED = 11798
plt.style.use('nigel.mplstyle')


print('Loading labels from original data')
label_map = {p: pdf.label.iloc[0] for p, pdf in load_data.train_full().groupby('STUDENTID')}

for datalen in ['10m', '20m', '30m']:
    print('Building model for', datalen)
    feat_names = list(pd.read_csv('features_fe/filtered_features_' + datalen + '.csv').feature)
    train_df = pd.read_csv('features_fe/train_' + datalen + '.csv')[['STUDENTID'] + feat_names]
    for fset in ['features_tsfresh', 'features_featuretools']:
        feat_names = list(pd.read_csv(fset + '/filtered_features_' + datalen + '.csv').feature)
        tdf = pd.read_csv(fset + '/train_' + datalen + '.csv')[['STUDENTID'] + feat_names]
        assert all(tdf.STUDENTID == train_df.STUDENTID), fset + ' train STUDENTID mismatch'
        train_df[feat_names] = tdf[feat_names]
    train_df = train_df.fillna(0)
    features = [f for f in train_df if f not in ['STUDENTID', 'label']]
    print(len(features), 'features combined')
    fsets = misc_util.uncorrelated_feature_sets(train_df[features], max_rho=.8,
                                                remove_perfect_corr=True, verbose=2)
    features = fsets[0]
    print(len(features), 'features after removing highly correlated features')
    train_y = [label_map[p] for p in train_df.STUDENTID]

    m = ensemble.ExtraTreesClassifier(500, random_state=RANDOM_SEED)
    m.fit(train_df[features], train_y)
    alphas = []
    for tree in tqdm(m.estimators_):
        path = tree.cost_complexity_pruning_path(train_df[features], train_y)
        alphas.extend(path.ccp_alphas[:-1])
    plt.figure()
    plt.hist(alphas, bins=100)
    plt.show()
    plt.close()
