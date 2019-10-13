# Combine predictions from one or more models and put them in the correct submission format
# Expected base rate, if it matters: 0.6038961038961039
import re
import os

import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from sklearn import metrics
import numpy as np


VALIDITY_REGEX = r'^(\s*0*((0?\.[0-9]+)|(0\.?)|(1\.?)|(1\.0*))\s*,){1231}\s*0*((0?\.[0-9]+)|(0\.?)|(1\.?)|(1\.0*))\s*$'
plt.style.use('nigel.mplstyle')


df = pd.read_csv('feature_level_fusion.csv')

print('Jensen-Shannon distances between prediction sets (square root of divergence):')
print('JSD 10 <=> 20:', jensenshannon(df[df.data_length == '10m'].pred,
                                      df[df.data_length == '20m'].pred))
print('JSD 10 <=> 30:', jensenshannon(df[df.data_length == '10m'].pred.iloc[:-1],  # 30m is 1 short
                                      df[df.data_length == '30m'].pred))
print('JSD 20 <=> 30:', jensenshannon(df[df.data_length == '20m'].pred.iloc[:-1],
                                      df[df.data_length == '30m'].pred))

# Plot to show if distributions across the three combined datasets are notably different
plt.hist(df[df.data_length == '10m'].pred, bins=50, alpha=.5, label='10 minutes')
plt.hist(df[df.data_length == '20m'].pred, bins=50, alpha=.5, label='20 minutes')
plt.hist(df[df.data_length == '30m'].pred, bins=50, alpha=.5, label='30 minutes')
plt.xlim(0, 1)
plt.axvline(.5, linestyle='--', linewidth=1, color='black')
plt.legend(loc='upper left')
plt.title('10m pred. rate: %.3f, 20m pred. rate: %.3f, 30m pred. rate: %.3f' %
          ((df[df.data_length == '10m'].pred > .5).mean(),
           (df[df.data_length == '20m'].pred > .5).mean(),
           (df[df.data_length == '30m'].pred > .5).mean()))
plt.xlabel('Predicted probability')
plt.ylabel('Count')
plt.show()

# Plot kappa over decision thresholds
# TODO: See how ideal decision threshold varies across training folds/data lengths
train_preds = pd.read_csv('feature_level_fusion-train.csv')
kappas = [metrics.cohen_kappa_score(train_preds.label, train_preds.pred > t)
          for t in np.linspace(0, 1, 101)]
plt.figure()
plt.plot(np.linspace(0, 1, len(kappas)), kappas)
plt.show()
print('Maximum kappa =', max(kappas))
thresh = np.argmax(kappas) / (len(kappas) - 1)
print('At threshold =', thresh)
print('Holdout predicted rate at that threshold =', (df.pred > thresh).mean())

# Adjust and save predictions
preds = ','.join((df.pred / thresh / 2).astype(str))
print(((df.pred - thresh).abs() < .0001).sum(), 'predictions at exactly threshold')
assert re.match(VALIDITY_REGEX, preds)

with open('combine_predictions.txt', 'w') as outfile:
    outfile.write(preds)
