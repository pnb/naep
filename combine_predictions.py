# Combine predictions from one or more models and put them in the correct submission format
import re
import os

import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from sklearn import metrics
import numpy as np


VALIDITY_REGEX = r'^(\s*0*((0?\.[0-9]+)|(0\.?)|(1\.?)|(1\.0*))\s*,){1231}\s*0*((0?\.[0-9]+)|(0\.?)|(1\.?)|(1\.0*))\s*$'
plt.style.use('nigel.mplstyle')


# TODO: normalize predictions across folds, and change decision threshold to maybe improve kappa -- the kappa step can be done first, by changing predictions to fit expected base rate (0.6038961038961039)

model_fe = pd.read_csv('model_fe-0.csv')
model_fe = model_fe[model_fe.holdout == 1]
model_fe1 = pd.read_csv('model_fe-1.csv')
model_fe1 = model_fe1[model_fe1.holdout == 1]
model_tsfresh = pd.read_csv('model_tsfresh.csv')
model_tsfresh = model_tsfresh[model_tsfresh.holdout == 1]
model_featuretools = pd.read_csv('model_featuretools-0.csv')
model_featuretools = model_featuretools[model_featuretools.holdout == 1]
print('FE <=> TSFresh r =', model_fe.pred.corr(model_tsfresh.pred))
print('FE <=> TSFresh rho =', model_fe.pred.corr(model_tsfresh.pred, method='spearman'))
print('FE <=> Featuretools r =', model_fe.pred.corr(model_featuretools.pred))
print('FE <=> Featuretools rho =', model_fe.pred.corr(model_featuretools.pred, method='spearman'))

# TODO: This fusion approach does not work well for all models together, possibly because predictions are bad across folds?
'''
# Select the best models and fuse only those
models = []
for fname in sorted(os.listdir('.')):
    if fname.startswith('model_fe') and fname.endswith('.csv'):
        models.append(pd.read_csv(fname))
for data_len in sorted(models[0].data_length.unique()):
    votes = []
    models_for_len = sorted([m[m.data_length == data_len] for m in models],
                            key=lambda x: x.kappa_min.iloc[0], reverse=True)
    print(['%.3f' % m.kappa_min.iloc[0] for m in models_for_len])
    votes = [m for m in models_for_len  # Only keep models that are over 95% as good as the best
             if m.kappa_min.iloc[0] > models_for_len[0].kappa_min.iloc[0] * .95]
    print(len(votes))
    fused = votes[0].pred * votes[0].kappa_min
    for vote in votes[1:]:
        fused += vote.pred * vote.kappa_min
    fused /= sum(v.kappa_min.iloc[0] for v in votes)
    gt = votes[0][votes[0].holdout == 0].label.astype(int)
    print('Fused kappa:', data_len, metrics.cohen_kappa_score(gt, fused.loc[gt.index] > .5))
exit()
'''

# Adjust predictions so predicted rate matches base rate for each data length
# for data_len in [10, 20, 30]:
#     len_df = model_fe[model_fe.data_length == data_len]
#     thresh = len_df.pred.quantile(1 - 0.6038961038961039)
#     model_fe.loc[len_df.index, 'pred'] = [v / 3 if v < thresh else v / 3 + .5 for v in len_df.pred]

# model_fe.pred = .9 * model_fe.pred + .1 * model_fe1.pred  # Excel guesswork ("human in the loop")
# model_fe.pred = .5 * model_fe.pred + .5 * model_featuretools.pred
df = pd.read_csv('feature_level_fusion.csv')
preds = ','.join(df.pred.astype(str))
assert re.match(VALIDITY_REGEX, preds)

with open('combine_predictions.txt', 'w') as outfile:
    outfile.write(preds)

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

'''
# Plot kappa over decision thresholds
# TODO: Try modify predictions to use the best threshold
model_fe = pd.read_csv('model_fe-0.csv')
model_fe = model_fe[model_fe.holdout == 0]
kappas = [metrics.cohen_kappa_score(model_fe.label, model_fe.pred > t)
          for t in np.linspace(0, 1, 101)]
plt.figure()
plt.plot(np.linspace(0, 1, len(kappas)), kappas)
plt.show()
print('Maximum kappa =', max(kappas))
thresh = np.argmax(kappas) / (len(kappas) - 1)
print('At threshold =', thresh)
print('Holdout predicted rate at that threshold =', (df.pred > thresh).mean())
'''
