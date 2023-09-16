'''
    four models:
    without altitude
    
    2003 to 2018
    lst_yr_val = [2003, 2007]
    lst_yr_test = [2009, 2014]
    
    input params: ['dayno', 'slt', 'f10.7', 'ap3']
        cyclic on dayno and slt
    
    manual val set
    
'''

import pdb
import torch
import random
import os.path
import calendar
import numpy as np
import pandas as pd
import datetime as dt
import autokeras as ak
import tensorflow as tf

from torch import nn
from glob import glob
from collections import OrderedDict
from autokeras import StructuredDataRegressor


#%% to use a specific GPU, [please adjust accordingly]
gpu_no = 4
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[gpu_no], 'GPU')
logical_gpus = tf.config.list_logical_devices('GPU')


#%% load the data and split into train/val/test [please adjust accordingly]
path_ = './data'
lst_yr_val = [2003, 2007]
lst_yr_test = [2009, 2014]

# the input # ['dayno', 'slt', 'f10.7', 'ap3']
parms_input = ['year', 'month', 'dayno', 'slt', 'f10.7', 'ap3']
parms_output = ['nel']
# ['doy_cyc_sin', 'doy_cyc_cos', 'slt_cyc_sin', 'slt_cyc_cos', 'f10.7', 'ap3']
parms_input_cyc = ['year', 'dayno', 'slt', 'f10.7', 'ap3']

# load the data
data_total = pd.read_feather(os.path.join(path_, 'data2003to2018_fixed_mb_intv1_purge.lz4'))

## get the month from dayno
size_ = data_total.shape[0]
lst_mon = []

for idx in range(size_):  
    year, dayno = data_total['year'][idx], data_total['dayno'][idx]
    mon, day = (dt.datetime(year, 1, 1)+dt.timedelta(days=int(dayno-1))).month, (dt.datetime(year, 1, 1)+dt.timedelta(days=int(dayno-1))).day
    
    lst_mon.append(mon)

data_total['month'] = lst_mon    
##  

data_train = data_total[~data_total['year'].isin(lst_yr_val+lst_yr_test)]
data_val = data_total[data_total['year'].isin(lst_yr_val)]
data_test = data_total[data_total['year'].isin(lst_yr_test)]

# assign X and y
# the total: train + val + test
X_total = data_total[parms_input]
# the maximum values of each column, df series
X_total_max = X_total.max()

# the train
X_train, y_train = data_train[parms_input], data_train[parms_output]
# the val
X_val, y_val = data_val[parms_input], data_val[parms_output]
# the test
X_test, y_test = data_test[parms_input], data_test[parms_output]

# get the normalized
X_train_norm = X_train / X_total_max
X_val_norm = X_val / X_total_max
X_test_norm = X_test / X_total_max


#%% apply cyclic on doy and slt 
## get the normalized dayno
doy_cyc_sin_norm_train = (np.sin(X_train['dayno'].values * 2*np.pi / 365) + 1) / 2
doy_cyc_cos_norm_train = (np.cos(X_train['dayno'].values * 2*np.pi / 365) + 1) / 2

doy_cyc_sin_norm_val = (np.sin(X_val['dayno'].values * 2*np.pi / 365) + 1) / 2
doy_cyc_cos_norm_val = (np.cos(X_val['dayno'].values * 2*np.pi / 365) + 1) / 2

doy_cyc_sin_norm_test = (np.sin(X_test['dayno'].values * 2*np.pi / 365) + 1) / 2
doy_cyc_cos_norm_test = (np.cos(X_test['dayno'].values * 2*np.pi / 365) + 1) / 2

## get the normalized SLT
slt_cyc_sin_norm_train = (np.sin(X_train['slt'].values * 2*np.pi / 24) + 1) / 2
slt_cyc_cos_norm_train = (np.cos(X_train['slt'].values * 2*np.pi / 24) + 1) / 2

slt_cyc_sin_norm_val = (np.sin(X_val['slt'].values * 2*np.pi / 24) + 1) / 2
slt_cyc_cos_norm_val = (np.cos(X_val['slt'].values * 2*np.pi / 24) + 1) / 2

slt_cyc_sin_norm_test = (np.sin(X_test['slt'].values * 2*np.pi / 24) + 1) / 2
slt_cyc_cos_norm_test = (np.cos(X_test['slt'].values * 2*np.pi / 24) + 1) / 2

## manually add the normalized column 
# for dayno
X_train_norm.loc[:, 'doy_cyc_sin'] = doy_cyc_sin_norm_train
X_train_norm.loc[:, 'doy_cyc_cos'] = doy_cyc_cos_norm_train

X_val_norm.loc[:, 'doy_cyc_sin'] = doy_cyc_sin_norm_val
X_val_norm.loc[:, 'doy_cyc_cos'] = doy_cyc_cos_norm_val

X_test_norm.loc[:, 'doy_cyc_sin'] = doy_cyc_sin_norm_test
X_test_norm.loc[:, 'doy_cyc_cos'] = doy_cyc_cos_norm_test

# for SLT
X_train_norm.loc[:, 'slt_cyc_sin'] = slt_cyc_sin_norm_train
X_train_norm.loc[:, 'slt_cyc_cos'] = slt_cyc_cos_norm_train

X_val_norm.loc[:, 'slt_cyc_sin'] = slt_cyc_sin_norm_val
X_val_norm.loc[:, 'slt_cyc_cos'] = slt_cyc_cos_norm_val

X_test_norm.loc[:, 'slt_cyc_sin'] = slt_cyc_sin_norm_test
X_test_norm.loc[:, 'slt_cyc_cos'] = slt_cyc_cos_norm_test


## keep the needed columns
X_train_norm = X_train_norm[parms_input_cyc] # parms_input_cyc
X_val_norm = X_val_norm[parms_input_cyc] # 
X_test_norm = X_test_norm[parms_input_cyc] # 


#%% get into torch
X_train_norm_tch = torch.tensor(X_train_norm.values)
X_val_norm_tch = torch.tensor(X_val_norm.values)
X_test_norm_tch = torch.tensor(X_test_norm.values)


# use the StructuredDataRegressor, 
# refer to the manual of AutoKeras here: https://github.com/keras-team/autokeras
search = StructuredDataRegressor(max_trials=200,
                                 loss='mean_absolute_error',
                                 metrics=['mean_absolute_error', 'mean_squared_error'],
                                 objective='val_loss',
                                 directory='./autoML_fixed_yr/DNN_run9',
                                 seed=49,
                                 tuner='greedy'
                                 # distribution_strategy=tf.distribute.MirroredStrategy()
                                )

#%% shuffle training and validation sets
# shuffle the training and val sets
size_train, size_val = X_train_norm.shape[0], X_val_norm.shape[0]

idx_train = np.arange(size_train)
random.Random(49).shuffle(idx_train)

idx_val = np.arange(size_val)
random.Random(49).shuffle(idx_val)


# set the validation set
search.fit(X_train_norm.values[idx_train, :],
           y_train.values[idx_train, :],
           batch_size=512,
           validation_data=(X_val_norm.values[idx_val, :], y_val.values[idx_val, :]),
           validation_batch_size=512,
           epochs=2000)


# print the searched optimal
model_opt = search.export_model()

print(model_opt.summary())
print(model_opt.optimizer.get_config())
