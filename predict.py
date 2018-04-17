from __future__ import absolute_import
from __future__ import print_function
from testFeatureExtract import *
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Lambda, merge, BatchNormalization, Activation, Input, Merge
from keras import backend as K
from siamese import *
from keras.optimizers import RMSprop, SGD, Adam
import keras.losses
from keras.preprocessing import text

keras.losses.contrastive_loss = contrastive_loss;
import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

df = pd.read_pickle('data/Features.pkl')
df = df.reindex(np.random.permutation(df.index))
# print(df)


# set number of train and test instances
num_train = int(df.shape[0] * 1)
num_test = df.shape[0] - num_train
print("Number of training pairs: %i" % (num_train))
print("Number of testing pairs: %i" % (num_test))

# init data data arrays
X_train = np.zeros([num_train, 2, 300])
X_test = np.zeros([num_test, 2, 300])
Y_train = np.zeros([num_train])
Y_test = np.zeros([num_test])

# format data
b = [a[None, :] for a in list(df['q1_feats'].values)]
q1_feats = np.concatenate(b, axis=0)

b = [a[None, :] for a in list(df['q2_feats'].values)]
q2_feats = np.concatenate(b, axis=0)

# fill data arrays with features
X_train[:, 0, :] = q1_feats[:num_train]
X_train[:, 1, :] = q2_feats[:num_train]
# Y_train = df[:num_train]['is_duplicate'].values

X_test[:, 0, :] = q1_feats[num_train:]
X_test[:, 1, :] = q2_feats[num_train:]
# Y_test = df[num_train:]['is_duplicate'].values

del b
del q1_feats
del q2_feats

net = keras.models.load_model("data/kerasmodel.h5", custom_objects={'Unknown loss function': contrastive_loss})
optimizer = Adam(lr=0.001)
net.compile(loss=contrastive_loss, optimizer=optimizer)

pred = net.predict([X_train[:, 0, :], X_train[:, 1, :]])
print(pred)

pd.to_pickle(pred, 'predictions.pkl')
