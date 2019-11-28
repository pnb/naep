from pprint import pprint
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import load_data
import misc_util


plt.style.use('nigel.mplstyle')


df = load_data.train_full()

print(len(df), 'rows')
print(len(df.STUDENTID.unique()), 'unique students')
print(df.groupby('STUDENTID').label.mean().mean(), 'positive class base rate')
print(df.columns)
print(df.groupby(['ItemType', 'AccessionNumber']).size())
print('\nNumber of rows per student ID:')
print(df.groupby('STUDENTID').size().describe())
print('\nNumber of actions of each type:')
print(df.Observable.value_counts())
print('\nStudent*item pairs:', len(df.groupby(['STUDENTID', 'AccessionNumber'])))
print('Considering only actual problems:',
      len(df[df.AccessionNumber.str.startswith('VH')].groupby(['STUDENTID', 'AccessionNumber'])))
print(df[['AccessionNumber', 'delta_time_ms']].head())

# print(df[df.Observable == 'Click Choice'].ExtendedInfo.value_counts())
# print(df[df.Observable == 'DropChoice'].ExtendedInfo.value_counts())  # Drag and drop answer
# print(df[df.Observable == 'Math Keypress'].ExtendedInfo.value_counts())

# # No correlation between mean delta time and label
# xy = np.array([(gt[p], pid_df.delta_time_ms.mean()) for p, pid_df in df.groupby('STUDENTID')]).T
# print(np.corrcoef(xy))

# # No correlation for item counts
# for item in df.AccessionNumber.unique():
#     xy = np.array([(gt[p], len(pid_df[pid_df.AccessionNumber == item]))
#                    for p, pid_df in df.groupby('STUDENTID')]).T
#     print(item, np.corrcoef(xy)[0, 1])

# # No correlation for overall action length
# xy = np.array([(gt[p], len(pid_df)) for p, pid_df in df.groupby('STUDENTID')]).T
# print(np.corrcoef(xy))

# # Some weak correlations for action type counts
# for action in df.Observable.unique():
#     xy = np.array([(gt[p], len(pid_df[pid_df.Observable == action]))
#                    for p, pid_df in df.groupby('STUDENTID')]).T
#     print(action, np.corrcoef(xy)[0, 1])

# # Graph distribution of time spent per student per item
# for item, item_df in df.groupby('AccessionNumber'):
#     print('Plotting', item)
#     student_level = []
#     for _, pid_df in item_df.groupby('STUDENTID'):
#         student_level.append((pid_df.time_unix.max() - pid_df.time_unix.min()) / 1000)
#     plt.figure()
#     plt.hist(student_level, bins=50)
#     plt.title('Item: %s    # students: %d    5%%: %.1f' %
#               (item, len(student_level), np.percentile(student_level, 5)))
#     plt.savefig('graphs/hist-AccessionNumber-' + item + '.png')
#     plt.close()

# Correlations for time spent per student per item exceeding the 5th percentile
# There is actually something here!
# for item, item_df in df.groupby('AccessionNumber'):
#     student_level = []
#     for pid, pid_df in item_df.groupby('STUDENTID'):
#         student_level.append([pid_df.delta_time_ms.sum() / 1000, pid_df.label.iloc[0]])
#     student_level = np.array(student_level).T
#     student_level[0] = student_level[0] > np.percentile(student_level[0], 5)
#     print(item, np.corrcoef(student_level)[0, 1])

# Are items presented in the same order for all students?
# No, there were navigation tabs; most students went in order, but some did not
# orderings = [(p, d.AccessionNumber.unique()) for p, d in
#              df[(~df.AccessionNumber.isin(['EOSTimeLft', 'SecTimeOut', 'BlockRev', 'HELPMAT8'])) &
#                 (df.Observable == 'Enter Item')].groupby('STUDENTID')]
# orderings = sorted(orderings, key=lambda x: len(x[1]))
# match, mismatch = 0, 0
# for (pa, a), (pb, b) in zip(orderings[:-1], orderings[1:]):
#     if len(a) != len(b):
#         print('Length 1:', len(a), '    Length 2:', len(b))
#         mismatch += 1
#     elif any(a != b):
#         print(pa, pb, a[a != b], b[a != b])
#         mismatch += 1
#     else:
#         match += 1
# print(match, 'sequences matched;', mismatch, 'did not')

# What are the most popular answer(s) for each problem? What prop. students agree?
# Answer values for ItemType values:
# For MultipleFillInBlank:
#     Receive Focus (Observable) happens, with question num in ExtendedInfo
#     Math Keypress (Observable) happens, with ""contentLaTeX"":""$63$"" and other in ExtendedInfo
#     Equation Editor Button (Observable) may happen instead of Math Keypress
#     Lose Focus (Observable) happens, with question num in ExtendedInfo
# For FillInBlank:
#     Same as MultipleFillInBlank, except question num is always 1
# For CompositeCR:
#     Receive Focus (Observable) happens with question ID, just like MultipleFillInBlank
#     Equation Editor Button (Observable) happens, with contentLaTeX like MultipleFillInBlank
#     Lose Focus (Observable) happens, with question num in ExtendedInfo
# For MCSS:
#     Click Choice (Observable) happens with option selected in ExtendedInfo before colon
# For MatchMS : -- note the space
#     DropChoice (Observable) happens; the last time it has the student's answer in ExtendedInfo
# all_df = load_data.all_unique_rows()
# with pd.option_context('display.max_rows', None):
#     print(all_df.groupby(['ItemType', 'Observable']).size())
# answers = misc_util.final_answers_from_df(all_df, verbose=1)
# questions = misc_util.answer_counts(answers)
# pprint(questions)
# print('true1 pred1:')
# pprint(answers[2333267544])
# for q, a in answers[2333267544].items():
#     print('Q:', q, 'A:', a, 'rank:', misc_util.answer_ranks(questions[q])[a])
# print('\ntrue0 pred1:')
# pprint(answers[2333170834])
# for q, a in answers[2333170834].items():
#     print('Q:', q, 'A:', a, 'rank:', misc_util.answer_ranks(questions[q])[a])
# print('\nPositive class modal ranks:')
# answer_ranks = {q: misc_util.answer_ranks(c) for q, c in questions.items()}
# pos_ranks = {q: np.array([answer_ranks[q][answers[p][q]]
#                           for p in df[df.label == 1].STUDENTID.unique() if q in answers[p]])
#              for q in answer_ranks}
# neg_ranks = {q: np.array([answer_ranks[q][answers[p][q]]
#                           for p in df[df.label == 0].STUDENTID.unique() if q in answers[p]])
#              for q in answer_ranks}
# pos_count = len(df[df.label == 1].STUDENTID.unique())
# neg_count = len(df[df.label == 0].STUDENTID.unique())
# for q in sorted(pos_ranks):
#     print('\nQ:', q, 'Positive class most common:', Counter(pos_ranks[q]).most_common(4))
#     print('Q:', q, 'Negative class most common:', Counter(neg_ranks[q]).most_common(4))
#     print('Q:', q, 'Positive class rank 1 prop:', np.sum(pos_ranks[q] == 1) / pos_count)
#     print('Q:', q, 'Negative class rank 1 prop:', np.sum(neg_ranks[q] == 1) / neg_count)

# How long are periods of activity/inactivity?
# print('\nAction duration descriptives (in seconds):')
# print((df.delta_time_ms / 1000).describe())  # Most are very very short, but there are so many
# chunk_action_counts = []
# for pid, pid_df in tqdm(df.groupby('STUDENTID')):
#     for chunk_start in range(pid_df.time_unix.min(), pid_df.time_unix.max(), 30000):
#         chunk_end = chunk_start + 30000
#         chunk_size = ((pid_df.time_unix >= chunk_start) & (pid_df.time_unix < chunk_end)).sum()
#         chunk_action_counts.append(chunk_size)
# chunk_action_counts = pd.Series(chunk_action_counts)
# print(chunk_action_counts.describe())
# print('Prop. with 0 actions:', (chunk_action_counts == 0).sum() / len(chunk_action_counts))  # 22%

# Does erasing relate to outcome? Not that much.
# df['backspace'] = df.ExtendedInfo.str.contains('Backspace')
# # df.Observable == 'Math Keypress'
# print(df.backspace.value_counts())
# print(df.groupby(['STUDENTID', 'label']).backspace.sum())

# Do total times differ much across students or do they all use the full 30 minutes? They differ.
# total_times = df.groupby('STUDENTID').time_unix.max() - df.groupby('STUDENTID').time_unix.min()
# plt.figure()
# plt.hist(total_times / 1000 / 60, bins=50)
# plt.savefig('graphs/hist-totaltime.png')
# print(total_times.sort_values() / 1000 / 60)

# Does running into a timeout message indicate incompleteness? Not really.
# timeout = ['SecTimeOut' in pid_df.AccessionNumber.values for _, pid_df in df.groupby('STUDENTID')]
# print('\nProp. students encountering SecTimeOut:', np.mean(timeout))
# labels = [pid_df.label.iloc[0] for _, pid_df in df.groupby('STUDENTID')]
# print('Timeout <=> label correlation:', np.corrcoef(timeout, labels)[0, 1])
# timeleft = ['EOSTimeLft' in pid_df.AccessionNumber.values for _, pid_df in df.groupby('STUDENTID')]
# Seems like there is almost no overlap between time left and timeout (nearly complementary)
# print('Prop. students encountering EOSTimeLft:', np.mean(timeleft))
# print('Prop. students encountering EOSTimeLft and SecTimeOut:',
#       np.mean(np.array(timeout) & np.array(timeleft)))

# Does repeating the same ExtendedInfo indicate button mashing? It does seem to correlate w/label.
# Possibly different for different items (TTS vs answer selection, for example)
# Horizontal item scroll is the strongest correlation despite very low n
# repeats = {}
# consec = {}
# labels = []
# for _, pid_df in tqdm(df.groupby('STUDENTID'), desc='Finding repeat ExtendedInfo'):
#     labels.append(pid_df.label.iloc[0])
#     for obs, obs_df in pid_df.groupby('Observable'):
#         if obs not in repeats:
#             repeats[obs] = []
#             consec[obs] = []
#         repeats[obs].append(len(obs_df) - len(obs_df.ExtendedInfo.unique()))
#         consec[obs].append(((pid_df.Observable == obs) & (pid_df.Observable.shift(1) == obs)).sum())
# for obs in repeats:
#     print(obs, 'repeats rho:', pd.Series(repeats[obs]).corr(pd.Series(labels), method='spearman'))
#     print(obs, 'consecutive rho:',
#           pd.Series(consec[obs]).corr(pd.Series(labels), method='spearman'))

# Is usage of calculator (number of actions in ExtendedInfo) good? Barely.
labels = []
buffer_lens = []
for _, pid_df in tqdm(df.groupby('STUDENTID'), desc='Calculator buffer length'):
    labels.append(pid_df.label.iloc[0])
    buffer_lens.append(np.array([len(r.ExtendedInfo.split(',')) for _, r in
                                 pid_df[pid_df.Observable == 'Calculator Buffer'].iterrows()]))
    if len(buffer_lens[-1]) == 0:
        buffer_lens[-1] = np.array([0])
print('Buffer len mean rho:',
      pd.Series(labels).corr(pd.Series([l.mean() for l in buffer_lens]), method='spearman'))
print('Buffer len std rho:',
      pd.Series(labels).corr(pd.Series([l.std() for l in buffer_lens]), method='spearman'))
print('Buffer len max rho:',
      pd.Series(labels).corr(pd.Series([l.max() for l in buffer_lens]), method='spearman'))
print('Buffer len sum rho:',
      pd.Series(labels).corr(pd.Series([l.sum() for l in buffer_lens]), method='spearman'))
