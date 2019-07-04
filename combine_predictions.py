# Combine predictions from one or more models and put them in the correct submission format
import re

import pandas as pd
import matplotlib.pyplot as plt


VALIDITY_REGEX = r'^(\s*0*((0?\.[0-9]+)|(0\.?)|(1\.?)|(1\.0*))\s*,){1231}\s*0*((0?\.[0-9]+)|(0\.?)|(1\.?)|(1\.0*))\s*$'


df = pd.read_csv('model_fe.csv')
preds = ','.join(df.pred.astype(str))
assert re.match(VALIDITY_REGEX, preds)

with open('combine_predictions.txt', 'w') as outfile:
    outfile.write(preds)

# Plot to show if distributions across the three combined datasets are notably different
plt.figure(figsize=(8, 6))
plt.hist(df[df.data_length == 10].pred, bins=50, alpha=.5, label='10 minutes')
plt.hist(df[df.data_length == 20].pred, bins=50, alpha=.5, label='20 minutes')
plt.hist(df[df.data_length == 30].pred, bins=50, alpha=.5, label='30 minutes')
plt.axvline(.5, linestyle='--', linewidth=1, color='black')
plt.legend(loc='upper left')
plt.show()
