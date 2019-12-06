# Ad-hoc feature engineering approach to extracting features

# Feature ideas:
#   Time spent in each item (AccessionNumber)
#   Time spent in each item >= 5th percentile as calculated from full data
#   Count of VH items where time spent >= 5th percentile
#   Num actions for each item
#   Num of each type of action (Observable)
#   SD of time spent across items
#   Num times entered each item
#   Rank of popularity of answers to questions
#   Count of k-ranked answers for different k
#   Count of unanswered questions
#   Count of repeated actions
#   Coefficients of polynomials fit to time spent per problem
#   Coefficients of polynomials fit to overall timeseries
#   Correlation between problem correctness and problem order
from collections import OrderedDict
import warnings

import pandas as pd
import numpy as np
from tqdm import tqdm

import load_data
import misc_util


RANDOM_SEED = 11798
warnings.filterwarnings('ignore', 'Polyfit may be poorly conditioned')  # Polynomial features
warnings.filterwarnings('ignore', 'invalid value encountered in true_divide')  # Corr feature
warnings.filterwarnings('ignore', 'invalid value encountered in multiply')  # Corr feature
warnings.filterwarnings('ignore', 'Degrees of freedom <= 0 for slice')  # Corr feature
warnings.filterwarnings('ignore', 'invalid value encountered in double_scalars')
warnings.filterwarnings('ignore', 'divide by zero encountered in true_divide')
warnings.filterwarnings('ignore', 'Mean of empty slice.')


def extract_features(pandas_df, item_5percentile_map, question_answer_counts):
    # item_5percentile_map is a mapping of item_name -> 5th percentile of time spent (ms) for item
    # question_answer_counts is a mapping of question ID -> Counter of answers
    # Returns (X, y, list of feature names); y may be None if `label` is not in `pandas_df`
    rows = []
    answer_ranks = {q: misc_util.answer_ranks(c) for q, c in question_answer_counts.items()}
    actual_items = set([row.AccessionNumber for _, row in pandas_df.iterrows() if row.ItemType in
                       ['FillInBlank', 'MCSS', 'MatchMS', 'MultipleFillInBlank']])
    for pid, pid_df in tqdm(pandas_df.groupby('STUDENTID'), desc='Extracting features'):
        rows.append(OrderedDict({
            'STUDENTID': pid,
            'label': pid_df.label.iloc[0] if 'label' in pid_df.columns else None,
            **{'sec_spent_' + e: v.delta_time_ms.sum() / 1000
               for e, v in pid_df.groupby('AccessionNumber')},
            **{'times_entered_' + e: sum(v.Observable == 'Enter Item')
               for e, v in pid_df.groupby('AccessionNumber')},
            **{'num_actions_' + e: len(v) for e, v in pid_df.groupby('AccessionNumber')},
            **{'count_' + o: len(v) for o, v in pid_df.groupby('Observable')},
            **{'percentile5_' + e: int(v.delta_time_ms.sum() >= item_5percentile_map[e])
               for e, v in pid_df.groupby('AccessionNumber')},
            **{'backspaces_' + e: v.ExtendedInfo.str.contains('Backspace').sum()
               for e, v in pid_df.groupby('AccessionNumber')},
            **{'readtime_' + e: v[v.Observable == 'Enter Item'].iloc[0].delta_time_ms
               for e, v in pid_df.groupby('AccessionNumber')
               if 'Enter Item' in v.Observable.values},
            **{'repeat_extended_' + o: len(v) - len(v.ExtendedInfo.unique())
               for o, v in pid_df.groupby('Observable')},
            **{'repeat_observable_' + o:
               ((pid_df.Observable == o) & (pid_df.Observable.shift(1) == o)).sum()
               for o in pid_df.Observable.unique()}
        }))
        percentile5_map = {i[12:]: v for i, v in rows[-1].items() if i.startswith('percentile5_')}
        rows[-1]['percentile5_count'] = sum(percentile5_map.values())
        rows[-1]['percentile5_count_actual'] = \
            sum(v for k, v in percentile5_map.items() if k in actual_items)
        rows[-1]['sec_spent_std'] = \
            np.std([v for k, v in rows[-1].items() if k.startswith('sec_spent_')])
        actual_sec = [v for k, v in rows[-1].items() if k.startswith('sec_spent_') and
                      k[10:] in actual_items]
        if len(actual_sec) > 0:
            rows[-1]['sec_spent_actual_min'] = min(actual_sec)
        if len(actual_sec) > 1:
            rows[-1]['sec_spent_actual_2nd_smallest'] = \
                min(v for v in actual_sec if v != rows[-1]['sec_spent_actual_min'])
        rows[-1]['backspaces_total'] = pid_df.ExtendedInfo.str.contains('Backspace').sum()
        rows[-1]['readtime_total'] = \
            sum(v for k, v in rows[-1].items() if k.startswith('readtime_'))
        rows[-1]['sec_spent_total'] = (pid_df.time_unix.max() - pid_df.time_unix.min()) / 1000
        for col in set(rows[-1]) - set(['label']):
            assert not np.isnan(rows[-1][col])
        # Coefficients of polynomials fitted to series of continuous values
        for ts_name, ts in [('delta_sec', pid_df.delta_time_ms.values / 1000),
                            ('per_item_sec', [(v.time_unix.max() - v.time_unix.min()) / 1000
                                              for _, v in pid_df.groupby('AccessionNumber')])]:
            if len(ts) < 2:
                continue  # No variance in timeseries, so polyfit will fail
            for poly_degree in range(4):
                for i, c in enumerate(np.polyfit(np.arange(len(ts)), ts, poly_degree)):
                    rows[-1]['poly_' + ts_name + '_deg' + str(poly_degree) + '_coeff' + str(i)] = c
        # Popularity of answers to questions
        answers = misc_util.final_answers_from_df(pid_df)[pid]
        for q_id, _ in question_answer_counts.items():
            if q_id not in answers or answers[q_id] == '':
                rows[-1]['answer_rank_' + q_id] = 1000
            elif answers[q_id] in answer_ranks[q_id]:  # Known answer
                rows[-1]['answer_rank_' + q_id] = answer_ranks[q_id][answers[q_id]]
            else:  # Unknown answer; must be from partial training data with a changed answer later
                rows[-1]['answer_rank_' + q_id] = max(answer_ranks[q_id].values()) + 1
        ranks = {col[12:]: v for col, v in rows[-1].items() if col.startswith('answer_rank_')}
        for k in range(1, 6):  # Ranks start at 1
            rows[-1]['answer_count_rank' + str(k)] = (np.array(list(ranks.values())) == k).sum()
        rows[-1]['answer_count_rankother'] = sum(r > 5 and r < 1000 for r in ranks.values())
        rows[-1]['answer_rank_sum'] = sum(v for v in ranks.values() if v != 1000)
        rows[-1]['answer_rank_std'] = np.std([v for v in ranks.values() if v != 1000])
        rows[-1]['answer_count_unanswered'] = (np.array(list(ranks.values())) == 1000).sum()
        most_diff = ['VH139196_A,1', 'VH134366_5', 'VH134366_4', 'VH134366_3', 'VH134366_2',
                     'VH134366_1', 'VH098779', 'VH098597', 'VH098556', 'VH098522', 'VH098519']
        diff_ranks = np.array([v for k, v in ranks.items() if k in most_diff and v != 1000])
        rows[-1]['answer_rank_discriminative_sum'] = diff_ranks.sum()
        rows[-1]['answer_rank_discriminative_std'] = diff_ranks.std()
        for k in range(1, 6):
            rows[-1]['answer_count_discriminative_rank' + str(k)] = (diff_ranks == k).sum()
        rows[-1]['answer_count_discriminative_rankother'] = (diff_ranks > 5).sum()
        # Correlation between answer rank = 1 (correct-ish) and exercise order
        ordered_rank1 = [int(rows[-1]['answer_rank_' + q] == 1)
                         for q in pid_df.AccessionNumber.unique() if 'answer_rank_' + q in rows[-1]]
        rows[-1]['answer_rank1_order_corr'] = np.corrcoef(ordered_rank1,
                                                          np.arange(len(ordered_rank1)))[0, 1]
    X = pd.DataFrame.from_records(rows)
    features = [f for f in X.columns if f not in ['STUDENTID', 'label']]
    # TODO: Maybe no longer necessary to fill NaNs with tree-based classifiers in sklearn 0.22?
    for col in features:
        X.loc[X[col].isnull(), col] = 0
    y = X.label if 'label' in pandas_df.columns else None
    return X, y, features


print('Loading data')
item_5percentile_map = {i: v.groupby('STUDENTID').delta_time_ms.sum().quantile(.05)
                        for i, v in load_data.all_full_rows().groupby('AccessionNumber')}
student_answers = misc_util.final_answers_from_df(load_data.all_unique_rows(), verbose=1)
question_answer_counts = misc_util.answer_counts(student_answers)

print('Loading holdout data')
dfs = {
    'train_10m': load_data.train_10m(),
    'train_20m': load_data.train_20m(),
    'train_30m': load_data.train_full(),
    'holdout_10m': load_data.holdout_10m(),
    'holdout_20m': load_data.holdout_20m(),
    'holdout_30m': load_data.holdout_30m(),
}

for dsname in dfs:
    print('\nFeatures for', dsname)
    tx, ty, feat_names = extract_features(dfs[dsname], item_5percentile_map, question_answer_counts)
    # Extract additional features from the first 5 minutes
    print('Features from the first five minutes of data')
    first5 = dfs[dsname]
    ms_start = first5.groupby('STUDENTID').time_unix.min()
    first5_end = pd.Series([ms_start[pid] + 5 * 60 * 1000 for pid in first5.STUDENTID],
                           index=first5.index)
    first5 = first5[first5.time_unix < first5_end]
    first5x, _, _ = extract_features(first5, item_5percentile_map, question_answer_counts)
    for f in feat_names:
        if f in first5x:
            tx[f + '_first5'] = first5x[f]
    if dsname.endswith('30m'):  # Extract additional features from last 5 minutes
        print('Features from the last five minutes of data')
        last5 = dfs[dsname]
        ms_end = last5.groupby('STUDENTID').time_unix.max()
        last5_start = pd.Series([ms_end[pid] - 5 * 60 * 1000 for pid in last5.STUDENTID],
                                index=last5.index)
        last5 = last5[last5.time_unix > last5_start]
        last5x, _, _ = extract_features(last5, item_5percentile_map, question_answer_counts)
        for f in feat_names:
            if f in last5x:
                tx[f + '_last5'] = last5x[f]
    feat_names.extend([f for f in tx if f.endswith('_last5') or f.endswith('_first5')])
    print('Saving', len(feat_names), 'features')
    tx[['STUDENTID'] + feat_names].to_csv('features_fe/' + dsname + '.csv', index=False)
