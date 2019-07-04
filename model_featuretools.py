# Automatic feature engineering with featuretools ("deep feature synthesis")
import featuretools as ft
import numpy as np

import load_data


df = load_data.data_full()
y_map = {p: p_df.label.iloc[0] for p, p_df in df.groupby('STUDENTID')}
df = df[['STUDENTID', 'AccessionNumber', 'ItemType', 'Observable', 'EventTime']]
df['row_index'] = df.index
es = ft.EntitySet().entity_from_dataframe('rows', dataframe=df, index='row_index',
                                          time_index='EventTime')
es = es.normalize_entity('rows', 'students', 'STUDENTID')
es = es.normalize_entity('rows', 'items', 'AccessionNumber', additional_variables=['ItemType'])
print('\n', es)
print('\n', es['rows'].variables)
es.plot('tmp.png')
# es.add_interesting_values(max_values=5, verbose=True)
es['rows']['AccessionNumber'].interesting_values = \
    [v for v in es['rows'].df.AccessionNumber.unique() if v.startswith('VH')]

# ft.list_primitives().to_csv('ft_primitives.csv', index=False)
aggregation_primitives = [
    'mean',
    'min',
    'max',
    'count',
    'avg_time_between',
    'time_since_first',
]
transform_primitives = [
    'time_since_previous',
    # 'cum_count',
]

feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity='students', verbose=True,
                                      agg_primitives=aggregation_primitives,
                                      trans_primitives=transform_primitives,
                                      where_primitives=aggregation_primitives)
print('\n', feature_matrix)
print('\n', feature_defs)
feature_matrix.to_csv('ft_data.csv', index=False)
