# Test to see if STUDENTID is a good predictor even after cross-validation. It seemed like it might
# be, but this indicates probably not.
from sklearn import tree, model_selection, metrics
import numpy as np

import load_data


RANDOM_SEED = 11798


print('Loading data')
df = load_data.train_full()
X = df.STUDENTID.unique().reshape(-1, 1)
y = np.array([df[df.STUDENTID == p].label.iloc[0] for p in X[:, 0]])

print('Training model')
xval = model_selection.StratifiedKFold(4, shuffle=True, random_state=RANDOM_SEED)
m = tree.DecisionTreeClassifier()
hp_grid = {
    'min_samples_leaf': [1, 2, 4, 8, 16, 32],
}
gs = model_selection.GridSearchCV(m, hp_grid, cv=xval, verbose=1)
scoring = {'AUC': metrics.make_scorer(metrics.roc_auc_score, needs_proba=True),
           'MCC': metrics.make_scorer(metrics.cohen_kappa_score),
           'Kappa': metrics.make_scorer(metrics.matthews_corrcoef)}
result = model_selection.cross_validate(gs, X, y, cv=xval, verbose=2, scoring=scoring)
print(result)
