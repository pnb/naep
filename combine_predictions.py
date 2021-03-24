# Combine predictions from one or more models and put them in the correct submission format
# Expected base rate, if it matters: 0.6038961038961039
import re
import argparse

import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from sklearn import metrics
import numpy as np
from tqdm import tqdm


VALIDITY_REGEX = r'^(\s*0*((0?\.[0-9]+)|(0\.?)|(1\.?)|(1\.0*))\s*,){1231}\s*0*((0?\.[0-9]+)|(0\.?)|(1\.?)|(1\.0*))\s*$'
plt.style.use('nigel.mplstyle')


argparser = argparse.ArgumentParser(
    description='Combine one or more predictions files (decision-level fusion) and adjust '
    'thresholds to maximize kappa')
argparser.add_argument('prediction_files', nargs='+', type=str,
                       help='One or more filenames of holdout set predictions files; training '
                       'prediction filename will be inferred by adding "-train"')
args = argparser.parse_args()

df = pd.read_csv(args.prediction_files[0]).rename(columns={'pred': 'pred_0'})
train_df = pd.read_csv(
    args.prediction_files[0].replace('.csv', '-train.csv')).rename(columns={'pred': 'pred_0'})
for i in range(1, len(args.prediction_files)):
    df['pred_' + str(i)] = pd.read_csv(args.prediction_files[i]).pred
    train_df['pred_' + str(i)] = pd.read_csv(
        args.prediction_files[i].replace('.csv', '-train.csv')).pred

# Equal voting
df['pred'] = df[[c for c in df if c.startswith('pred_')]].sum(axis=1) / len(args.prediction_files)
train_df['pred'] = train_df[[c for c in train_df if c.startswith('pred_')]].sum(axis=1)
train_df['pred'] /= len(args.prediction_files)

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

# Rescale predictions to try improve kappa
print('Overall kappa at .5 threshold =',
      metrics.cohen_kappa_score(train_df.label, train_df.pred > .5))
print('Overall AUC =', metrics.roc_auc_score(train_df.label, train_df.pred))
plt.figure()
for datalen, dldf in train_df.groupby('data_length'):
    # Plot kappa over decision thresholds
    kappas = [metrics.cohen_kappa_score(dldf.label, dldf.pred > t)
              for t in tqdm(np.linspace(0, 1, 101), desc='Calculating kappas')]
    plt.plot(np.linspace(0, 1, len(kappas)), kappas, label=datalen + ' max = %.3f' % max(kappas))
    print(datalen, 'AUC =', metrics.roc_auc_score(dldf.label, dldf.pred))
    print(datalen, 'Truncated AUC =',
          metrics.roc_auc_score(dldf.label, dldf.pred.astype(str).str.slice(stop=6).astype(float)))
    # Adjust predictions in holdout data to match ideal training data threshold
    thresh = np.argmax(kappas) / (len(kappas) - 1)
    print(datalen, 'ideal threshold =', thresh)
    dfpreds = df[df.data_length == datalen].pred
    print('Holdout predicted rate at that threshold =', (dfpreds > thresh).mean())
    print(((dfpreds - thresh).abs() < .0001).sum(), datalen, 'predictions at exactly threshold')
    if thresh >= .5:
        df.loc[dfpreds.index, 'pred'] = dfpreds / thresh / 2
    else:
        df.loc[dfpreds.index, 'pred'] = 1 - (1 - dfpreds) / (1 - thresh) / 2
plt.legend(loc='upper left')
plt.show()

# Save predictions
preds = ','.join(df.pred.astype(str))
assert re.match(VALIDITY_REGEX, preds)
with open('combine_predictions.txt', 'w') as outfile:
    outfile.write(preds)
