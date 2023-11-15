import numpy as np
import pandas as pd
import h5py 

# import plotting tools
import matplotlib.pyplot as plt
import seaborn as sns

# MODELING .............................................................
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


def load_data():

    return data


def get_year(data):

    return data


def train_test_splt(data, train_years, test_years):
    '''
    Return temporal training/test split
    '''
    return X_train, X_test, y_train, y_test


def make_predictions(model, X_test, return_std=False):
    '''
    Return predictions and UQ
    '''
    if return_std == False:
        y_pred = model.predict(X_test)
        return y_pred
    else:
        y_pred, y_std = model.predict(X_test, return_std=True)
        return y_pred, y_std

# plotting utils

def viz_uncertainty(y_pred, y_std, y_test):
    '''
    Plot predictions and uncertainty
    '''
    return fig

def viz_feature_importance(model):
    '''
    Plot feature importance
    '''
    return fig

def viz_residuals(y_pred, y_test):
    '''
    Plot residuals
    '''
    return fig





