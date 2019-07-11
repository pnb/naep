# Combine predictions from one or more models and put them in the correct submission format
import re

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon


VALIDITY_REGEX = r'^(\s*0*((0?\.[0-9]+)|(0\.?)|(1\.?)|(1\.0*))\s*,){1231}\s*0*((0?\.[0-9]+)|(0\.?)|(1\.?)|(1\.0*))\s*$'
plt.style.use('nigel.mplstyle')


model_fe = pd.read_csv('model_fe-0.csv')
model_fe = model_fe[model_fe.holdout == 1]
model_tsfresh = pd.read_csv('model_tsfresh.csv')
model_featuretools = pd.read_csv('model_featuretools-0.csv')
print('FE <=> TSFresh r =', np.corrcoef(model_fe.pred, model_tsfresh.pred)[0, 1])
print('FE <=> TSFresh rho =', stats.spearmanr(model_fe.pred, model_tsfresh.pred)[0])
print('FE <=> Featuretools r =', np.corrcoef(model_fe.pred, model_featuretools.pred)[0, 1])
print('FE <=> Featuretools rho =', stats.spearmanr(model_fe.pred, model_featuretools.pred)[0])

df = model_fe
preds = ','.join(df.pred.astype(str))
assert re.match(VALIDITY_REGEX, preds)

with open('combine_predictions.txt', 'w') as outfile:
    outfile.write(preds)

print('Jensen-Shannon distances between prediction sets (square root of divergence):')
print('JSD 10 <=> 20:', jensenshannon(df[df.data_length == 10].pred, df[df.data_length == 20].pred))
print('JSD 10 <=> 30:', jensenshannon(df[df.data_length == 10].pred.iloc[:-1],  # 30m is 1 short
                                      df[df.data_length == 30].pred))
print('JSD 20 <=> 30:', jensenshannon(df[df.data_length == 20].pred.iloc[:-1],
                                      df[df.data_length == 30].pred))

# Plot to show if distributions across the three combined datasets are notably different
plt.figure(figsize=(8, 6))
plt.hist(df[df.data_length == 10].pred, bins=50, alpha=.5, label='10 minutes')
plt.hist(df[df.data_length == 20].pred, bins=50, alpha=.5, label='20 minutes')
plt.hist(df[df.data_length == 30].pred, bins=50, alpha=.5, label='30 minutes')
plt.xlim(0, 1)
plt.axvline(.5, linestyle='--', linewidth=1, color='black')
plt.legend(loc='upper left')
plt.title('10m pred. rate: %.3f, 20m pred. rate: %.3f, 30m pred. rate: %.3f' %
          ((df[df.data_length == 10].pred > .5).mean(), (df[df.data_length == 20].pred > .5).mean(),
           (df[df.data_length == 30].pred > .5).mean()))
plt.xlabel('Predicted probability')
plt.ylabel('Count')
plt.show()
