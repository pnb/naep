from collections import Counter
import json
import re

from scipy import stats
import pandas as pd
import numpy as np
from tqdm import tqdm


def uncorrelated_feature_sets(pandas_df, max_rho=.8, remove_perfect_corr=False, verbose=0):
    """Given a dataset with some features, return a list of lists, where each sublist is a set of
    feature names such that no pair of features are correlated more than `max_rho`.

    Args:
        pandas_df (pd.DataFrame): Dataset, where every column is assumed to be a numeric feature
        max_rho (float): Maximum allowable pairwise absolute correlation between two features
        remove_perfect_corr (bool): If True, when a pair of features correlate perfectly (rho = 1),
                                    remove one of them completely from consideration
        verbose (int): Verbosity level

    Returns:
        list of lists: One or more sets of uncorrelated features
    """
    # Pairwise Spearman's rho correlation matrix with self-correlations set to 0
    rho = pd.DataFrame(index=pandas_df.columns, columns=pandas_df.columns, dtype=float)
    if verbose > 0:
        print('Computing pairwise correlations')
    for i, a in enumerate(tqdm(pandas_df.columns, disable=not verbose)):
        for b in pandas_df.columns:
            if np.isnan(rho.at[a, b]):
                if a == b:
                    rho.at[a, b] = 0
                else:
                    rho.at[a, b] = rho.at[b, a] = stats.spearmanr(pandas_df[a], pandas_df[b])[0]
    if rho.isnull().sum().sum() > 0:
        raise ValueError('Correlation matrix had NaN values; check that there are no missing values'
                         ' in inputs, and that each input feature has some variance')

    result = []
    current_set = list(pandas_df.columns)
    next_set = []
    while True:
        # Find maximum pairwise correlation to see if further splitting of feature set is needed
        highest_corr = rho.loc[current_set, current_set].max().max()
        if highest_corr > max_rho:
            a = rho.loc[current_set, current_set].max().idxmax()
            b = rho.loc[a, current_set].idxmax()
            if verbose > 1:
                print(a, 'correlated with', b, 'rho =', rho.at[a, b])
            # Break ties based on which of the pair has higher mean correlation with other features
            to_remove = a if rho.loc[a, current_set].mean() > rho.loc[b, current_set].mean() else b
            if highest_corr < 1 or not remove_perfect_corr:
                next_set.append(to_remove)
            current_set.remove(to_remove)
        elif len(next_set) > 0:
            if verbose > 0:
                print('Creating feature set of size', len(current_set))
            result.append(current_set)
            current_set = next_set
            next_set = []
        else:
            if len(current_set) > 0:
                if verbose > 0:
                    print('Creating feature set of size', len(current_set))
                result.append(current_set)
            break  # No correlations larger than max allowed, no remaining features to check
    return result


def final_answers_from_df(df):
    """Extract the final answer given to each question/sub-question by a student, given a Pandas
    DataFrame with sequences of actions from one or more students. This could be used to get the
    answer to a specific attempt for a specific student as well, by inputting a DataFrame with only
    data from that attempt.

    Args:
        df (pd.DataFrame): Sequence or subsequence of actions, e.g., from load_data.train_full()

    Returns:
        dict: Mapping of STUDENTID -> question/sub-question ID -> answer
    """
    answers = {}
    for i, row in df.iterrows():
        if row.STUDENTID not in answers:
            answers[row.STUDENTID] = {}
        if row.ItemType in ['MultipleFillInBlank', 'FillInBlank', 'CompositeCR']:
            if row.Observable == 'Receive Focus':
                subq = row.ExtendedInfo.replace('Part ', '').replace(', 1', ',1')  # Fix VH139196
            elif row.Observable == 'Math Keypress' or row.Observable == 'Equation Editor Button':
                answer = json.loads(row.ExtendedInfo)['contentLaTeX']
                answer = re.sub(r'(\.0*\$|\$|\\| |mathrm\{\w*\}|overline\{\})', '', answer)
                try:  # Parse fractions (ambitious...)
                    answer = re.sub(r'frac\{(-?\d+)\}\{(-?\d+)\}',
                                    lambda m: str(float(m.group(1)) /
                                                  float(m.group(2))).lstrip('0'), answer)
                except:
                    pass  # Cannot even begin to imagine the parsing errors
                answer = re.sub(r'^0\.', '.', answer)  # Leading 0 for decimals < 1
                answer = re.sub(r'(?<=\.)0$', '', answer)  # Unnecessary trailing decimal zeros
                answer = answer.replace('^circ', '')
            elif row.Observable == 'Lose Focus':
                answers[row.STUDENTID][row.AccessionNumber + '_' + subq] = answer
        elif row.ItemType == 'MCSS':
            if row.Observable == 'Click Choice':
                answers[row.STUDENTID][row.AccessionNumber] = \
                    row.ExtendedInfo[:row.ExtendedInfo.index(':')]
        elif row.ItemType == 'MatchMS ':
            if row.Observable == 'DropChoice':  # e.g., [{'source': '3', 'target': 1}, ...]
                for answer_pair in json.loads(row.ExtendedInfo):
                    subq_id, answer = answer_pair['source'], answer_pair['target']
                    answers[row.STUDENTID][row.AccessionNumber + '_' + subq_id] = answer
    return answers


def answer_counts(answers):
    """Rank the most popular answers to a question, given a per-student mapping of answers to
    questions from final_answers_from_df()

    Args:
        answers (dict): Mapping of student answers to questions, from final_answers_from_df()

    Returns:
        dict: Mapping of question/sub-question ID -> collections.Counter of answers
    """
    # Reformat as answers at the question level instead of student level
    questions = {q: Counter() for q in set(qid for pid_map in answers.values() for qid in pid_map)}
    for question in questions:
        for pid_answers in answers.values():
            if question in pid_answers:
                questions[question][pid_answers[question]] += 1
    return questions


if __name__ == '__main__':
    df = pd.DataFrame({'w': [2, 2, 3, 4, 5], 'x': [1, -2, 1, 3, 3], 'y': [5, 1, 3, 0, 1],
                       'z': [1.1, -1, 1, 5, 5], 'w2': [2, 2, 3, 4, 5]})
    print(uncorrelated_feature_sets(df, max_rho=.5, verbose=2, remove_perfect_corr=True))
