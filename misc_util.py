from scipy import stats
import pandas as pd
import numpy as np
from tqdm import tqdm


def uncorrelated_feature_sets(pandas_df, max_rho=.8, verbose=0):
    """Given a dataset with some features, return a list of lists, where each sublist is a set of
    feature names such that no pair of features are correlated more than `max_rho`.

    Args:
        pandas_df (pd.DataFrame): Dataset, where every column is assumed to be a numeric feature
        max_rho (float): Maximum allowable pairwise absolute correlation between two features

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

    result = []
    current_set = list(pandas_df.columns)
    next_set = []
    while True:
        # Find maximum pairwise correlation to see if further splitting of feature set is needed
        if rho.loc[current_set, current_set].max().max() > max_rho:
            a = rho.loc[current_set, current_set].max().idxmax()
            b = rho.loc[a, current_set].idxmax()
            if verbose > 1:
                print(a, 'correlated with', b, 'rho =', rho.at[a, b])
            # Break ties based on which of the pair has higher mean correlation with other features
            to_remove = a if rho.loc[a, current_set].mean() > rho.loc[b, current_set].mean() else b
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


if __name__ == '__main__':
    df = pd.DataFrame({'w': [2, 2, 3, 4, 5], 'x': [1, -2, 1, 3, 3], 'y': [5, 1, 3, 0, 1],
                       'z': [1.1, -1, 1, 5, 5]})
    print(uncorrelated_feature_sets(df, max_rho=.5, verbose=2))
