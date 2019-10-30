# Ad-hoc feature engineering approach to extracting features

# TODO: Finish implementing feature ideas
# Feature ideas (doubly-indented are implemented):
#       Time spent in each item (AccessionNumber)
#       Time spent in each item >= 5th percentile as calculated from full data
#       Count of VH items where time spent >= 5th percentile
#       Num actions for each item
#   Num of each type of action (Observable)
#       SD of time spent across items
#       Num times entered each item
#   Similarity of ordering to positive-class students, via dOSS or similar
#   Similarity of actions within each item
#   Similarity of sequences with new action types engineered, like pauses, calculator use
#       Rank of popularity of answers to questions
#       Count of top-k ranked answers for different k
#       Count of unanswered questions
#   Time spent and num actions for each item type (might be too redundant)
#   WTF behavior, especially at the end of the session (num Next event in last X mins vs. mean)
#       Coefficients of polynomials fit to time spent per problem
#       Coefficients of polynomials fit to overall timeseries
from collections import OrderedDict

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn import ensemble, model_selection, pipeline, metrics

import load_data
import misc_util


RANDOM_SEED = 11798
CACHE_DIR = '/Users/pnb/sklearn_cache'


def extract_features(pandas_df, freq_actions, item_5percentile_map, question_answer_counts):
    # freq_actions is a list of common actions that should be dummy-coded
    # item_5percentile_map is a mapping of item_name -> 5th percentile of time spent (ms) for item
    # question_answer_counts is a mapping of question ID -> Counter of answers
    # Returns (X, y, list of feature names); y may be None if `label` is not in `pandas_df`
    rows = []
    answer_ranks = {q: misc_util.answer_ranks(c) for q, c in question_answer_counts.items()}
    for pid, pid_df in tqdm(pandas_df.groupby('STUDENTID'), desc='Extracting features'):
        rows.append(OrderedDict({
            'STUDENTID': pid,
            'label': pid_df.label.iloc[0] if 'label' in pid_df.columns else None,
            **{'sec_spent_' + e: v.delta_time_ms.sum() / 1000
               for e, v in pid_df.groupby('AccessionNumber')},
            **{'times_entered_' + e: sum(v.Observable == 'Enter Item')
               for e, v in pid_df.groupby('AccessionNumber')},
            **{'num_actions_' + e: len(v) for e, v in pid_df.groupby('AccessionNumber')},
            **{'count_' + e: len(v) for e, v in pid_df.groupby('Observable') if e in freq_actions},
            **{'percentile5_' + e: int(v.delta_time_ms.sum() >= item_5percentile_map[e])
               for e, v in pid_df.groupby('AccessionNumber')},
        }))
        rows[-1]['percentile5_vh_count'] = \
            sum(v for k, v in rows[-1].items() if k.startswith('percentile5_'))
        rows[-1]['sec_spent_std'] = \
            np.std([v for k, v in rows[-1].items() if k.startswith('sec_spent_')])
        # Coefficients of polynomials fitted to series of continuous values
        for ts_name, ts in [('delta_sec', pid_df.delta_time_ms.values / 1000),
                            ('per_item_sec', [(v.time_unix.max() - v.time_unix.min()) / 1000
                                              for _, v in pid_df.groupby('AccessionNumber')])]:
            for poly_degree in range(4):
                for i, c in enumerate(np.polyfit(np.arange(len(ts)), ts, poly_degree)):
                    rows[-1]['poly_' + ts_name + '_deg' + str(poly_degree) + '_coeff' + str(i)] = c
        # Popularity of answers to questions
        answers = misc_util.final_answers_from_df(pid_df)[pid]
        for q_id, counts in question_answer_counts.items():
            if q_id not in answers or answers[q_id] == '':
                rows[-1]['answer_rank_' + q_id] = 1000
            elif answers[q_id] in answer_ranks[q_id]:  # Known answer
                rows[-1]['answer_rank_' + q_id] = answer_ranks[q_id][answers[q_id]]
            else:  # Unknown answer; must be from partial training data with a changed answer later
                rows[-1]['answer_rank_' + q_id] = max(answer_ranks[q_id].values()) + 1
        ranks = np.array([v for col, v in rows[-1].items() if col.startswith('answer_rank_')])
        for k in range(1, 6):  # Ranks start at 1
            rows[-1]['answer_count_top' + str(k)] = (ranks <= k).sum()
        rows[-1]['answer_count_unanswered'] = (ranks == 1000).sum()
    X = pd.DataFrame.from_records(rows)
    features = [f for f in X.columns if f not in ['STUDENTID', 'label']]
    for col in features:
        X.loc[X[col].isnull(), col] = 0
    y = X.label if 'label' in pandas_df.columns else None
    return X, y, features


print('Loading data')
df = load_data.all_unique_rows()
freq_actions = df.Observable.value_counts()
freq_actions = freq_actions[freq_actions >= 2464].index  # Average at least once per student
item_5percentile_map = {i: v.groupby('STUDENTID').delta_time_ms.sum().quantile(.05)
                        for i, v in df.groupby('AccessionNumber')}
student_answers = misc_util.final_answers_from_df(df, verbose=1)
question_answer_counts = misc_util.answer_counts(student_answers)

# Set up feature importance model training parameters
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

print('Loading holdout data')
dfs = {
    'train_10m': load_data.train_10m(),
    'train_20m': load_data.train_20m(),
    'train_30m': load_data.train_full(),
    'holdout_10m': load_data.holdout_10m(),
    'holdout_20m': load_data.holdout_20m(),
    'holdout_30m': load_data.holdout_30m(),
}
feat_dfs = {}
feat_ys = {}
for dsname in dfs:
    print('\nFeatures for', dsname)
    tx, ty, feat_names = extract_features(dfs[dsname], freq_actions, item_5percentile_map,
                                          question_answer_counts)
    feat_ys[dsname] = ty
    if dsname.endswith('30m'):  # Extract additional features from last 5 minutes
        print('Features from the last five minutes of data')
        last5 = dfs[dsname]
        ms_end = last5.groupby('STUDENTID').time_unix.max()
        last5_start = pd.Series([ms_end[pid] - 5 * 60 * 1000 for pid in last5.STUDENTID],
                                index=last5.index)
        last5 = last5[last5.time_unix > last5_start]
        last5x, _, _ = extract_features(last5, freq_actions, item_5percentile_map,
                                        question_answer_counts)
        for f in feat_names:
            if f in last5x:
                tx[f + '_last5'] = last5x[f]
        feat_names.extend([f for f in tx if f.endswith('_last5')])
    # feat_names = [f for f in feat_names if not f.startswith('answer_rank_')]  # TODO: Some answer features are garbage for now, esp. w/30minutes data
    # Remove 0-variance features since they will cause problems for calculating correlations
    feat_names = [f for f in feat_names if len(tx[f].unique()) > 1]
    print(len(feat_names), 'features')
    fsets = misc_util.uncorrelated_feature_sets(tx[feat_names], max_rho=.8,
                                                remove_perfect_corr=True, verbose=1)
    print(len(fsets[0]), 'features after removing highly-correlated features')
    feat_dfs[dsname] = tx[['STUDENTID'] + fsets[0]]

for datalen in ['10m', '20m', '30m']:
    print('Building feature importance model for', datalen)
    train_X, train_y = feat_dfs['train_' + datalen], feat_ys['train_' + datalen]
    holdout_X = feat_dfs['holdout_' + datalen]
    feat_names = [f for f in train_X if f in holdout_X.columns and f != 'STUDENTID']
    print(len(feat_names), 'features after keeping only matching train/holdout features')

    imp_m = gs.fit(train_X[feat_names], train_y).best_estimator_
    importances = pd.Series(imp_m.named_steps['model'].feature_importances_, index=feat_names)
    # Pick only important features and save
    feat_names = [f for f in feat_names if importances[f] > .001]
    print(len(feat_names), 'features after keeping only important features')
    train_X[['STUDENTID'] + feat_names].to_csv('features_fe/train_' + datalen + '.csv', index=False)
    holdout_X[['STUDENTID'] + feat_names].to_csv(
        'features_fe/holdout_' + datalen + '.csv', index=False)
