# Anomaly detection model; use self-supervised learning to predict future student actions, then
# calculate a timeseries of errors for these calculations and derive features from that
from keras import models, layers, utils
import numpy as np

import load_data
import nn_util


def make_sequences(df):
    # Preprocess data by one-hot encoding some things and rescaling others
    features = []
    for one_hot_col in ['AccessionNumber', 'ItemType', 'Observable']:
        uniq_vals = df[one_hot_col].value_counts()
        to_encode = uniq_vals[uniq_vals > 2000]
        for val in to_encode.index:
            features.append(one_hot_col + '_' + val.replace(' ', ''))
            df[features[-1]] = df[one_hot_col] == val
    df['delta_time'] = np.log(1 + df.delta_time_ms / 1000) / 3
    features.append('delta_time')

    X, y_i = seq = nn_util.make_sequences(df, features, participant_id_col='STUDENTID',
                                          sequence_len=8, verbose=True)
    y = df.iloc[y_i]
    y = y[features].values
    print('Saving sequences')
    np.save('model_anomaly-X_8_30.npy', X)
    np.save('model_anomaly-y_8_30.npy', y)
    np.save('model_anomaly-y_i_8_30.npy', y_i)
    np.save('model_anomaly-X_features.npy', features)
    # print(len(y_i))  # 377089


print('Loading data')
df = load_data.train_full()
# make_sequences(df)
X = np.load('model_anomaly-X_8_30.npy')
y = np.load('model_anomaly-y_8_30.npy')
y_i = np.load('model_anomaly-y_i_8_30.npy')
features = np.load('model_anomaly-X_features.npy')

