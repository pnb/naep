from collections import Counter, OrderedDict
import json
import re
import tempfile
import os
import subprocess

from scipy import stats
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn import model_selection, metrics, tree
from sklearn.base import BaseEstimator, TransformerMixin


RANDOM_SEED = 11798


def uncorrelated_feature_sets(pandas_df, max_rho=.8, remove_perfect_corr=False, verbose=0,
                              priority_order=[]):
    """Given a dataset with some features, return a list of lists, where each sublist is a set of
    feature names such that no pair of features are correlated more than `max_rho`. Ties will be
    broken among pairs of highly-correlated features by removing the feature with higher mean
    correlation with other features, unless tie-break preferences for either/both features are given
    in `priority_order`.

    Args:
        pandas_df (pd.DataFrame): Dataset, where every column is assumed to be a numeric feature
        max_rho (float): Maximum allowable pairwise absolute correlation between two features
        remove_perfect_corr (bool): If True, when a pair of features correlate perfectly (rho = 1),
                                    remove one of them completely from consideration
        verbose (int): Verbosity level
        priority_order (list of lists): Feature names to prefer keeping; each sublist will be given
                                        preference over subsequent sublists, and over any features
                                        not represented in any sublist

    Returns:
        list of lists: One or more sets of uncorrelated features
    """
    assert max_rho <= 1 and max_rho > 0, 'Maximum allowable correlation should be in (0, 1]'
    # Pairwise Spearman's rho correlation matrix with self-correlations set to 0
    rho = pd.DataFrame(index=pandas_df.columns, columns=pandas_df.columns, dtype=float)
    for i, a in enumerate(tqdm(pandas_df.columns, desc='Pairwise corr', disable=verbose < 1)):
        for b in pandas_df.columns[i:]:
            if a == b:
                rho.at[a, b] = 0
            else:
                rho.at[a, b] = rho.at[b, a] = \
                    abs(pandas_df[a].corr(pandas_df[b], method='spearman'))
    if verbose > 3:
        print(rho)
    if rho.isnull().sum().sum() > 0:
        raise ValueError('Correlation matrix had NaN values; check that each feature has variance')

    # Convert priority_order to a dict for faster/easier lookups
    priority = {f: i for i, sublist in enumerate(priority_order) for f in sublist}

    result = []
    current_set = list(pandas_df.columns)
    next_set = []
    while True:
        # Find maximum pairwise correlation to see if further splitting of feature set is needed
        highest_corr = rho.loc[current_set, current_set].max().max()
        if highest_corr > max_rho or highest_corr == 1:
            a = rho.loc[current_set, current_set].max().idxmax()
            b = rho.loc[a, current_set].idxmax()
            if verbose > 2:
                print(a, 'correlated with', b, 'rho =', rho.at[a, b])
            # Break ties based on which has higher mean correlation unless priority order is given
            to_remove = None
            if a in priority:
                if b not in priority or priority[a] < priority[b]:
                    to_remove = b
                elif b in priority and priority[a] > priority[b]:
                    to_remove = a
            elif b in priority:
                to_remove = a
            if not to_remove:  # Priority order not specified or a tie; use higher mean correlation
                if rho.loc[a, current_set].mean() < rho.loc[b, current_set].mean():
                    to_remove = b
                else:
                    to_remove = a
            if highest_corr < 1 or not remove_perfect_corr:
                next_set.append(to_remove)
            current_set.remove(to_remove)
        elif len(next_set) > 0:
            if verbose > 1:
                print('Creating feature set of size', len(current_set))
            result.append(current_set)
            current_set = next_set
            next_set = []
        else:
            if len(current_set) > 0:
                if verbose > 1:
                    print('Creating feature set of size', len(current_set))
                result.append(current_set)
            break  # No correlations larger than max allowed, no remaining features to check
    return result


class UncorrelatedFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, max_rho=.8, verbose=0):
        self.max_rho = max_rho
        self.verbose = verbose
        self.uncorrelated_features = []

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame), 'Only pd.DataFrame inputs for X are supported'
        if self.verbose > 0:
            print('Finding features correlated at rho <=', self.max_rho)
        fsets = uncorrelated_feature_sets(X, self.max_rho, remove_perfect_corr=True,
                                          verbose=self.verbose - 1)
        self.uncorrelated_features = fsets[0]
        if self.verbose > 0:
            print('Found', len(self.uncorrelated_features), 'uncorrelated features')
        return self

    def transform(self, X, y=None):
        return X[self.uncorrelated_features]


def final_answers_from_df(df, verbose=0):
    """Extract the final answer given to each question/sub-question by a student, given a Pandas
    DataFrame with sequences of actions from one or more students. This could be used to get the
    answer to a specific attempt for a specific student as well, by inputting a DataFrame with only
    data from that attempt.

    Args:
        df (pd.DataFrame): Sequence or subsequence of actions, e.g., from load_data.train_full()
        verbose (int): Verbosity level

    Returns:
        dict: Mapping of STUDENTID -> question/sub-question ID -> answer
    """
    answers = {}
    for _, row in tqdm(df.iterrows(), desc='Getting answers', disable=not verbose, total=len(df)):
        if row.STUDENTID not in answers:
            answers[row.STUDENTID] = {}
        if row.ItemType in ['MultipleFillInBlank', 'FillInBlank', 'CompositeCR']:
            if row.Observable == 'Receive Focus':
                subq = row.ExtendedInfo.replace('Part ', '').replace(', 1', ',1')  # Fix VH139196
                answer = ''
            elif row.Observable == 'Math Keypress' or row.Observable == 'Equation Editor Button':
                jdata = json.loads(row.ExtendedInfo)
                answer = re.sub(r'(\$|\\| |mathrm\{\w*\}|overline\{\})', '', jdata['contentLaTeX'])
                # Some actions don't show up immediately in contentLaTeX and must be appended
                if '{' not in answer:  # If there is a frac{} or anything going on, just give up
                    code = jdata['code'] if 'code' in jdata else ''
                    if code == 'Period':
                        answer += '.'
                    elif code.startswith('Digit'):
                        answer += code[5]
                try:  # Parse fractions (ambitious...)
                    answer = re.sub(r'frac\{(-?\d+)\}\{(-?\d+)\}',
                                    lambda m: str(float(m.group(1)) /
                                                  float(m.group(2))).lstrip('0'), answer)
                except:
                    pass  # Cannot even begin to imagine the parsing errors
                answer = answer.replace('^circ', '')
                answer = re.sub(r'^0\.', '.', answer)  # Leading 0 for decimals < 1
                if '.' in answer:
                    answer = re.sub(r'0+$', '', answer)  # Unnecessary trailing decimal zeros
                    if answer[-1] == '.':
                        answer = answer[:-1]  # Remove .0's
            elif row.Observable == 'Lose Focus':
                try:
                    answers[row.STUDENTID][row.AccessionNumber + '_' + subq] = answer
                except UnboundLocalError:  # subq not defined
                    pass  # Can only happen with incomplete data (e.g., for last 5 minutes of data)
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
    TODO: Make this into an expectation maximization problem instead to find source reliability, and
          then also return source (student) reliability as a possible feature

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


def answer_ranks(question_answer_counts):
    """Rank the popularity of answers to a single question, given a collections.Counter of answers
    to the question (e.g., one of the items obtained from answer_counts()). Ranking begins at 1
    (the most popular answer), and ties are counted as the same rank.

    Args:
        question_answer_counts (collections.Counter): Counter of answers to a question

    Returns:
        dict: Mapping of answer -> rank
    """
    assert type(question_answer_counts) == Counter
    ranks = {}
    unique_counts = 0
    last_count = None
    for ans, count in question_answer_counts.most_common():
        if last_count is None or count < last_count:
            unique_counts += 1
            last_count = count
        ranks[ans] = unique_counts
    return ranks


def tree_error_analysis(X, y, cv, class_names, output_filename_prefix):
    """Train a simple decision tree model and graph it to help find cases where new/better features
    are needed, or where current features may be behaving unexpectedly.

    Args:
        X (pd.DataFrame): Training data (column names are required)
        y (array): Labels for training data
        cv (int or sklearn cross-validator): Cross-validation method to apply
        class_names (array): List of labels for class names in ascending order (y=0, y=1, etc.)
        output_filename_prefix (str): Path + prefix for output graphs

    Returns:
        (dict, pd.DataFrame): Cross-validation results and predictions
    """
    assert len(class_names) == len(np.unique(y)), 'There must be one class name per class'
    scoring = {'AUC': metrics.make_scorer(metrics.roc_auc_score, needs_proba=True),
               'MCC': metrics.make_scorer(metrics.cohen_kappa_score),
               'Kappa': metrics.make_scorer(metrics.matthews_corrcoef)}
    m = tree.DecisionTreeClassifier(min_samples_leaf=8, random_state=RANDOM_SEED)
    res = model_selection.cross_validate(m, X, y, scoring=scoring, verbose=1, cv=cv,
                                         return_estimator=True)
    err_df = pd.DataFrame(index=X.index, data={'pred': '', 'truth': y, 'fold': '', 'leaf_size': ''})
    for fold_i, (_, test_i) in enumerate(tqdm(cv.split(X, y), desc='Graphing trees')):
        err_df.pred.iloc[test_i] = res['estimator'][fold_i].predict_proba(X.iloc[test_i]).T[1]
        err_df.fold.iloc[test_i] = fold_i
        # Graph trees, look for the most impure large leaves -- in the tree graphs or in a pivot
        # table filtered by truth value looking for common wrong predicted probabilities
        leaf_i = res['estimator'][fold_i].apply(X.iloc[test_i])
        leaf_sizes = np.bincount(leaf_i)  # Array with leaf index -> number of occurrences
        err_df.leaf_size.iloc[test_i] = [leaf_sizes[i] for i in leaf_i]
        with tempfile.TemporaryDirectory() as tmpdir:
            dotfile = os.path.join(tmpdir, 'tree.dot')
            tree.export_graphviz(res['estimator'][fold_i], out_file=dotfile,
                                 class_names=class_names, feature_names=X.columns, filled=True)
            subprocess.call(['dot', '-Tpng', dotfile, '-o',
                            output_filename_prefix + 'fold' + str(fold_i) + '.png', '-Gdpi=300'])
    return res, err_df


def per_feature_analysis(X, y, cv):
    """Explore individual feature predictive accuracy for every feature in a dataset, via a simple
    and fast CART model. This allows finding features that are especially effective and possible
    inspirations for future features, as well as features that may be severely over-fit to the
    training data and could need improvement

    Args:
        X (pd.DataFrame): Training data (columns are required)
        y (array): Labels for training data
        cv (int or sklearn cross-validator): Cross-validation method to apply

    Returns:
        pd.DataFrame: Results for each feature, probably for saving to a CSV file
    """
    scoring = {'AUC': metrics.make_scorer(metrics.roc_auc_score, needs_proba=True),
               'MCC': metrics.make_scorer(metrics.cohen_kappa_score),
               'Kappa': metrics.make_scorer(metrics.matthews_corrcoef)}
    m = tree.DecisionTreeClassifier(min_samples_leaf=8, random_state=RANDOM_SEED)
    result = []
    for feat in tqdm(X.columns, desc='Building 1-feature models'):
        scores = model_selection.cross_validate(m, X[[feat]], y, scoring=scoring, cv=cv,
                                                return_train_score=True)
        result.append(OrderedDict({
            'feature': feat,
            'mean_test_auc': np.mean(scores['test_AUC']),
            'min_test_auc': min(scores['test_AUC']),
            'mean_test_kappa': np.mean(scores['test_Kappa']),
            'min_test_kappa': min(scores['test_Kappa']),
            'mean_train_auc': np.mean(scores['train_AUC']),
            'mean_train_kappa': np.mean(scores['train_Kappa']),
        }))
    return pd.DataFrame.from_records(result)


def adjusted_thresh_kappa(y_true, y_pred, thresholds=100):
    """Cohen's kappa with the decision threshold adjusted to maximiize kappa. `thresholds` evenly-
    spaced cutoffs in [0, 1] will be evaluated.

    Args:
        y_true (Numpy array): Ground truth labels (0 or 1)
        y_pred (Numpy array): Predicted probabilities (must be continuous, probability-like)
        thresholds (int, optional): Number of thresholds to explore. Defaults to 100.

    Returns:
        float: Adjusted-threshold kappa
    """
    y_pred = np.array(y_pred)
    return max(metrics.cohen_kappa_score(y_true, y_pred > t)
               for t in np.linspace(0, 1, thresholds + 1))


def thresh_restricted_auk(y_true, y_pred, thresholds=100, auk_width=.1):
    """Area under the Cohen's kappa curve (AUK) restricted to a range centered around the ideal
    (maximum kappa) threshold. For example, if the ideal threshold for maximizing kappa is 0.64 and
    `auk_width` is set to 0.1, then the returned value will be AUK measured from 0.59 to 0.69 (i.e.,
    0.65 +/- 0.1).

    Args:
        y_true (Numpy array): Ground truth labels (0 or 1)
        y_pred (Numpy array): Predicted probabilities (must be continuous, probability-like)
        thresholds (int, optional): Number of thresholds to explore. Defaults to 100.
        auk_width (float, optional): Width of interval around ideal threshold for restricted-range
            AUK calculation. Defaults to 0.1.

    Returns:
        float: Restricted-range AUK (value normalized to [-1, 1] based on `auk_width`)
    """
    y_pred = np.array(y_pred)
    cuts = np.linspace(0, 1, thresholds + 1)
    kappas = np.array([metrics.cohen_kappa_score(y_true, y_pred > t) for t in cuts])
    ideal = cuts[np.argmax(kappas)]
    restricted_kappas = kappas[(cuts >= ideal - auk_width / 2) & (cuts <= ideal + auk_width / 2)]
    return sum(restricted_kappas) / len(restricted_kappas)


def kappa_plus_auc(y_true, y_pred, threshold=.5):
    """Sum of Cohen's kappa and the area under the receiver operating characteristic curve (AUC)

    Args:
        y_true (Numpy array): Ground truth labels (0 or 1)
        y_pred (Numpy array): Predicted probabilities (must be continuous, probability-like)
        threshold (float, optional): Decision threshold for calculating kappa (>=). Defaults to .5.

    Returns:
        float: Sum of kappa and AUC (in the range [-1, 2])
    """
    y_pred = np.array(y_pred)
    return metrics.cohen_kappa_score(y_true, y_pred >= threshold) + \
        metrics.roc_auc_score(y_true, y_pred)


if __name__ == '__main__':
    df = pd.DataFrame({'w': [2, 2, 3, 4, 5], 'x': [1, -2, 1, 3, 3], 'y': [5, 1, 3, 0, 1],
                       'z': [1.1, -1, 1, 5, 5], 'w2': [2, 2, 3, 4, 5]})
    print(uncorrelated_feature_sets(df, max_rho=.5, verbose=4, remove_perfect_corr=True))
    print('\nWith prioritizing x over z and z over w2:')
    print(uncorrelated_feature_sets(df, max_rho=.5, verbose=4, remove_perfect_corr=True,
                                    priority_order=[['x'], ['z']]))

    truth = [0, 1, 1, 1, 0, 0, 1, 1]
    preds = [.1, .5, .4, .6, .2, .3, .2, .9]
    print('Kappa:', metrics.cohen_kappa_score(truth, np.array(preds) >= .5))
    print('Threshold-adjusted kappa:', adjusted_thresh_kappa(truth, preds))
    print('Threshold-restricted AUK:', thresh_restricted_auk(truth, preds))
    print('Kappa + AUC:', kappa_plus_auc(truth, preds))
