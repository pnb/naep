# Automatic feature engineering with featuretools ("deep feature synthesis")
import featuretools as ft
import numpy as np
import pandas as pd
from sklearn import model_selection, pipeline, metrics, ensemble

import load_data
import misc_util


RANDOM_SEED = 11798
CACHE_DIR = '/Users/pnb/sklearn_cache'  # TODO: convert to optional command-line parm; also add preprocessing as CLI


def extract_ft_features():
    # Combine all dataframes so that features will be consistent across them
    dfs = []
    dfs.append(load_data.train_full())
    dfs[-1]['STUDENTID'] = [str(p) + '_train_30m' for p in dfs[-1].STUDENTID]
    dfs.append(load_data.train_10m())
    dfs[-1]['STUDENTID'] = [str(p) + '_train_10m' for p in dfs[-1].STUDENTID]
    dfs.append(load_data.train_20m())
    dfs[-1]['STUDENTID'] = [str(p) + '_train_20m' for p in dfs[-1].STUDENTID]
    dfs.append(load_data.holdout_10m())
    dfs[-1]['STUDENTID'] = [str(p) + '_holdout_10m' for p in dfs[-1].STUDENTID]
    dfs[-1].insert(7, 'label', '')
    dfs.append(load_data.holdout_20m())
    dfs[-1]['STUDENTID'] = [str(p) + '_holdout_20m' for p in dfs[-1].STUDENTID]
    dfs[-1].insert(7, 'label', '')
    dfs.append(load_data.holdout_30m())
    dfs[-1]['STUDENTID'] = [str(p) + '_holdout_30m' for p in dfs[-1].STUDENTID]
    dfs[-1].insert(7, 'label', '')
    df = pd.concat(dfs).reset_index(drop=True)

    df = df[['STUDENTID', 'AccessionNumber', 'ItemType', 'Observable', 'EventTime']]
    df['row_index'] = df.index
    var_types = {
        'STUDENTID': ft.variable_types.Index,
        'AccessionNumber': ft.variable_types.Categorical,
        'ItemType': ft.variable_types.Categorical,
        'Observable': ft.variable_types.Categorical,
        'EventTime': ft.variable_types.TimeIndex,
    }
    es = ft.EntitySet().entity_from_dataframe('rows', dataframe=df, index='row_index',
                                              time_index='EventTime', variable_types=var_types)
    es = es.normalize_entity('rows', 'students', 'STUDENTID')
    es = es.normalize_entity('rows', 'items', 'AccessionNumber', additional_variables=['ItemType'])
    print('\n', es)
    print('\n', es['rows'].variables)
    es.plot('features_featuretools/entity_structure.png')
    es.add_interesting_values(max_values=10, verbose=True)
    es['rows']['AccessionNumber'].interesting_values = \
        [v for v in es['rows'].df.AccessionNumber.unique() if v.startswith('VH')]

    # Basically all the primitives that seemed to make any sense -- there may be more!
    ft.list_primitives().to_csv('features_featuretools/ft_primitives.csv', index=False)
    aggregation_primitives = [
        'max',
        'median',
        'mode',
        'time_since_first',
        'sum',
        'avg_time_between',
        'num_unique',
        'skew',
        'min',
        'trend',
        'mean',
        'count',
        'time_since_last',
        'std',
    ]
    transform_primitives = [
        'time_since_previous',
        'divide_by_feature',
        'greater_than_equal_to',
        'time_since',
        'cum_min',
        'cum_count',
        'month',
        'cum_max',
        'cum_mean',
        'weekday',
        'cum_sum',
        'percentile',
    ]
    feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity='students', verbose=True,
                                          agg_primitives=aggregation_primitives,
                                          trans_primitives=transform_primitives,
                                          where_primitives=aggregation_primitives)
    # One-hot encode categorical features where needed
    feature_matrix_enc, features_defs_enc = ft.encode_features(feature_matrix, feature_defs)
    ft.save_features(features_defs_enc, 'features_featuretools/feature_defs.json')
    print(len(features_defs_enc), 'features after one-hot encoding')
    # Re-split features into appropriate train/holdout sets
    print('Saving features files')
    feature_matrix_enc['source_file'] = ''
    for pid in feature_matrix_enc.index.unique():
        feature_matrix_enc.at[pid, 'source_file'] = pid[pid.index('_') + 1:]
    for source_file, feat_df in feature_matrix_enc.groupby('source_file'):
        # STUDENTID back to normal
        feat_df.insert(0, 'STUDENTID', [p[:p.index('_')] for p in feat_df.index])
        feat_df.to_csv('features_featuretools/ft_' + source_file + '.csv', index=False)


def unsupervised_feature_selection():
    # This is based on the non-overlapping datasets, including holdout data, so that the process
    # will hopefully generalize as well as possible to the holdout data
    df = pd.concat([pd.read_csv('features_featuretools/' + f, index_col=0) for f in
                    ['ft_train_30m.csv', 'ft_holdout_10m.csv', 'ft_holdout_20m.csv',
                     'ft_holdout_30m.csv']]) \
        .reset_index(drop=True).replace([np.inf, -np.inf], np.nan).fillna(0)
    features = [f for f in df if f != 'source_file' and f != 'STUDENTID']
    num_features_before = len(features)
    for col in list(features):
        if df[col].nunique() < len(df) * .05:
            features.remove(col)
    print('Removed', num_features_before - len(features), 'features with little variance')
    print(num_features_before, 'features initially')
    print(len(features), 'left after removing those with little variance')
    fsets = misc_util.uncorrelated_feature_sets(df[features], max_rho=.8, verbose=1,
                                                remove_perfect_corr=True)
    with open('features_featuretools/uncorrelated_feature_sets.txt', 'w') as outfile:
        for fset in fsets:
            outfile.write(';'.join(fset) + '\n')
            assert ';' not in ''.join(fset), 'Feature name has a semicolon in it'
    print(len(fsets), 'feature sets saved to file')


def load_final_data():
    # Return a tuple with dictionary of dataframes for train/holdout data, and a list of
    # uncorrelated feature sets (a list of lists)
    dfs = {
        'train_10m': pd.read_csv('features_featuretools/ft_train_10m.csv'),
        'train_20m': pd.read_csv('features_featuretools/ft_train_20m.csv'),
        'train_30m': pd.read_csv('features_featuretools/ft_train_30m.csv'),
        'holdout_10m': pd.read_csv('features_featuretools/ft_holdout_10m.csv'),
        'holdout_20m': pd.read_csv('features_featuretools/ft_holdout_20m.csv'),
        'holdout_30m': pd.read_csv('features_featuretools/ft_holdout_30m.csv'),
    }
    # Add in labels, make STUDENTID a column again
    train_labels = pd.read_csv('public_data/data_train_label.csv', index_col='STUDENTID')
    for k, df in dfs.items():
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        if 'train_' in k:
            df['label'] = [train_labels.loc[p].EfficientlyCompletedBlockB for p in df.STUDENTID]
    with open('features_featuretools/uncorrelated_feature_sets.txt') as infile:
        fsets = [l.strip().split(';') for l in infile]
    return dfs, fsets


# print('Loading raw data/extracting features (only needs to be done once)')
# extract_ft_features()

# print('Unsupervised feature selection (only needs to be done once)')
# unsupervised_feature_selection()

print('Loading feature sets')
dfs, feature_sets = load_final_data()

# Set up feature importance model
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

# Build feature importance models and save important features
for datalen in ['10m', '20m', '30m']:
    print('Processing data length', datalen)
    train_df, holdout_df = dfs['train_' + datalen], dfs['holdout_' + datalen]
    feat_names = feature_sets[0]
    print(len(feat_names), 'features')
    imp_m = gs.fit(train_df[feat_names], train_df.label).best_estimator_
    importances = pd.Series(imp_m.named_steps['model'].feature_importances_, index=feat_names)
    # Pick only important features and save
    feat_names = [f for f in feat_names if importances[f] > .001]
    print(len(feat_names), 'features after keeping only important features')
    train_df[['STUDENTID'] + feat_names].to_csv(
        'features_featuretools/train_' + datalen + '.csv', index=False)
    holdout_df[['STUDENTID'] + feat_names].to_csv(
        'features_featuretools/holdout_' + datalen + '.csv', index=False)
