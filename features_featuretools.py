# Automatic feature engineering with featuretools ("deep feature synthesis")
import featuretools as ft
import pandas as pd

import load_data


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
    'entropy',
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
    feat_df.sort_values('STUDENTID') \
        .to_csv('features_featuretools/' + source_file + '.csv', index=False)
