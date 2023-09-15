'''
    comments
'''


import os
import numpy as np
import datetime as dt
import torch
import _pickle as pkl
import tensorflow as tf
import autokeras as ak

from glob import glob
from torch import nn
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from autokeras import StructuredDataRegressor
from prepare_ISR import read_ISR_bin

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

# load the training and test data

cv_no = 5

fromFD = 'Inputs_wMLT' # 'Inputs_wMLT' Inputs_wMLT_F107p
train_ = np.load(f'/data/yang/projs/ISR_ANN/data/after_purge/{fromFD}/Train_Xandy_cv_{cv_no}.npz')['data']
test_ = np.load(f'/data/yang/projs/ISR_ANN/data/after_purge/{fromFD}/Test_Xandy_cv_{cv_no}.npz')['data']
# import pdb; pdb.set_trace()
# the last two columns, ..., mlt, nel

# do the train validation split
_train, _val = train_test_split(train_, test_size=0.1, random_state=49)

# trim the test set if test set is larger into the val set size
if test_.shape[0] > _val.shape[0]:
    np.random.seed(13)
    _test = test_[np.random.choice(test_.shape[0], size=_val.shape[0], replace=False)]

# assign the values, the norm should be among _train_total and _test_
# _train_total means the data before the train val split
X_train_total = train_[:, :-1]
y_train_total = train_[:, -1]

X_test_ = _test[:, :-1]
y_test_ = _test[:, -1]

# for the training set
X_train_ = _train[:, :-1]
y_train_ = _train[:, -1]
# for the val set
X_val_ = _val[:, :-1]
y_val_ = _val[:, -1]

# Convert the data into torch Tensors, "tch" as abrev of "torch"
X_train_total_tch = torch.tensor(X_train_total, dtype=torch.float).to(device)
y_train_total_tch = torch.tensor(y_train_total, dtype=torch.float).view(-1, 1).to(device, non_blocking=True)

X_test_tch = torch.tensor(X_test_, dtype=torch.float).to(device)
y_test_tch = torch.tensor(y_test_, dtype=torch.float).view(-1, 1).to(device, non_blocking=True)

X_train_tch = torch.tensor(X_train_, dtype=torch.float).to(device)
y_train_tch = torch.tensor(y_train_, dtype=torch.float).view(-1, 1).to(device, non_blocking=True)

X_val_tch = torch.tensor(X_val_, dtype=torch.float).to(device)
y_val_tch = torch.tensor(y_val_, dtype=torch.float).view(-1, 1).to(device, non_blocking=True)


# normalization
X_max_train_total, _ = torch.max(X_train_total_tch, 0)
X_max_test, _ = torch.max(X_test_tch, 0)
X_max = torch.max(X_max_train_total, X_max_test)

# norm on train val test sets, X only
X_train_norm = torch.div(X_train_tch, X_max)
X_val_norm = torch.div(X_val_tch, X_max)
X_test_norm = torch.div(X_test_tch, X_max)

X_train_total_norm = torch.div(X_train_total_tch, X_max)


gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[4], 'GPU')
logical_gpus = tf.config.list_logical_devices('GPU')

# use the StructuredDataRegressor

search = StructuredDataRegressor(max_trials=10,
                                 loss='mean_absolute_error',
                                 metrics=['mean_absolute_error', 'mean_squared_error'],
                                 objective='val_loss', # default: val_loss
                                 directory='./autoML/randomShit_checkFinalFitWeights',
                                 seed=49,
                                 tuner='greedy'
                                )

search.fit(X_train_total_norm.detach().cpu().clone().numpy(),
           y_train_total_tch.detach().cpu().clone().numpy(),
           validation_split=0.2,
           epochs=100)

# print the searched optimal
model_opt = search.export_model()
print(model_opt.summary())
# print(model_opt.optimizer.get_config())

