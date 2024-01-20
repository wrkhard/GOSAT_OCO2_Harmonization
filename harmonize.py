# William Keely <william at belumenusc dot com>

import models
import utils


import numpy as np
import pickle
import pandas as pd
import h5py as h5
import glob
import os


# import plotting tools
import matplotlib.pyplot as plt

# import NGBoost
from ngboost import NGBRegressor

# import sklearn RF, and GPR
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# import sklearn tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2


feats_all = [
    'xco2_uncertainty',
    'psurf',
    'windspeed',
    't700',
    'fs',
    'tcwv',
    'tcwv_uncertainty',
    'dp',
    'dpfrac',
    's31',
    's32',
    'co2_grad_del',
    'dws',
    'offset_o2a_rel',
    'aod_dust',
    'aod_bc',
    'aod_oc',
    'aod_seasalt',
    'aod_sulfate',
    'aod_strataer',
    'aod_water',
    'aod_ice',
    'aod_fine',
    'aod_total',
    'ice_height',
    'water_height',
    'dust_height',
    'h2o_scale',
    'deltaT',
    'albedo_o2a',
    'albedo_wco2',
    'albedo_sco2',
    'albedo_slope_o2a',
    'albedo_slope_wco2',
    'albedo_slope_sco2',
    'rms_rel_wco2',
    'rms_rel_sco2',
]

# load data
file_path = '/Users/williamlumenus/Desktop/Projects/harmonization/Data/matched_gosat_v9_litevars_oco2_v10_xco2_20140906_20200630.pkl
df = pd.read_pickle(file_path)

# create training and testing data
train_df, test_df = split_evaluation_data_by_year(df, 2019)
gain = 72 
surface_type = 0
X_train, y_train, X_test, y_test = create_train_test_split(gain, surface_type,train_data, test_data, feats_all, 'xco2_oco2_aggregated_median')

# define a model
model = HarmonizationModel(n_estimators=100, max_depth=12, random_state=0)

# fit the model
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

# evaluate on test data
mse_score = mse(y_test, y_pred)
mae_score = mae(y_test, y_pred)
r2_score = r2(y_test, y_pred)

# save model
model.save_model('model_test.pkl')

# plot results
plt.scatter(y_test, y_pred)
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.title('Random Forest Model')
plt.show()


