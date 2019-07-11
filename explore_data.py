import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import load_data


plt.style.use('nigel.mplstyle')


df = load_data.train_full()

print(len(df), 'rows')
print(len(df.STUDENTID.unique()), 'unique students')
print(df.label.mean(), 'positive class base rate')
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
