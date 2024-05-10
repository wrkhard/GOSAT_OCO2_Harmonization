# William Keely <william at belumenus dot com>

from models import HarmonizationModel, HarmonizationUQModel
from utils import *


import numpy as np
import pickle
import pandas as pd
import h5py as h5
import glob
import os


# import plotting tools
import matplotlib.pyplot as plt
import seaborn as sns
import uncertainty_toolbox as uct

# import NGBoost
from ngboost import NGBRegressor
from ngboost.distns import Normal, LogNormal
from ngboost.scores import LogScore, CRPScore

# import sklearn RF, and GPR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel

# import sklearn tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2

# import PCA
from sklearn.decomposition import PCA



def arg_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Harmonization')
    parser.add_argument('--verbose', type=bool, default=True, help='verbose')
    parser.add_argument('--n_estimators', type=int, default=50, help='n_estimators')
    parser.add_argument('--feats', type=str, default='all', help='features')
    parser.add_argument('--PCA', type=bool, default=False, help='Use PCA')
    return parser.parse_args()

def main(args):
    assert args.feats in ['land_H', 'land_M', 'ocean_H',], 'Invalid feature set'

    if args.feats == 'land_H':
        gain = 72
        surface_type = 1
        feats = [
            'aod_fine',
            'aod_strataer',
            'aod_total',
            'aod_oc',
            'aod_sulfate',
            'fs',

        ]




    if args.feats == 'land_M':
        gain = 77
        surface_type = 1
        feats = [
            'aod_sulfate',
            'albedo_wco2',
            'albedo_sco2',
            'dust_height',
            'albedo_o2a',
        ]
    
    if args.feats == 'ocean_H':
        gain = 72
        surface_type = 0

        feats = [
            'aod_fine',
            'aod_dust',
            'albedo_slope_wco2',
            'albedo_slope_sco2',
            'aod_sulfate',
        ]
   

    # load data
    file_path = '/Users/williamlumenus/Desktop/Projects/harmonization/Data/matched_gosat_v9_litevars_oco2_v11.1_xco2_20140906_20200630.pkl'
    df = pd.read_pickle(file_path)

    # remove soundings where oco_aggregated_xco2_stddev_ppm <= oco_max_allowed_xco2_variability
    df = df[df['xco2_oco2_aggregated_std'] <= 1.5]

    train_df, test_df = split_evaluation_data_by_year(df, 2019)
    # subsample the data
    if args.feats == 'land_H':
        train_df = train_df[train_df['gain'] == gain]
        test_df = test_df[test_df['gain'] == gain]
        train_df = train_df[train_df['gosat_surface_type'] == surface_type]
        test_df = test_df[test_df['gosat_surface_type'] == surface_type]
        # sub sample the data to 5000 soundings
        train_df = train_df.sample(10000)
        # test_df = test_df.sample(10000)
    if args.feats == 'land_M':
        train_df = train_df[train_df['gain'] == gain]
        test_df = test_df[test_df['gain'] == gain]
        train_df = train_df[train_df['gosat_surface_type'] == surface_type]
        test_df = test_df[test_df['gosat_surface_type'] == surface_type]
        # sub sample the data to 5000 soundings
        train_df = train_df.sample(10000)
        # test_df = test_df.sample(10000)
    if args.feats == 'ocean_H':
        train_df = train_df[train_df['gain'] == gain]
        test_df = test_df[test_df['gain'] == gain]
        train_df = train_df[train_df['gosat_surface_type'] == surface_type]
        test_df = test_df[test_df['gosat_surface_type'] == surface_type]
        # sub sample the data to 5000 soundings
        train_df = train_df.sample(5000)
        # test_df = test_df.sample(10000)
    
    if args.PCA:
         if args.PCA:
            feats = [
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
            X_train = train_df[feats]
            X_test = test_df[feats]
            # pca the feats
            pca = PCA(n_components=10)
            pca.fit(X_train)
            X_train = pca.transform(X_train)
            X_test = pca.transform(X_test)
            _, y_train, _, y_test = create_train_test_split(gain, surface_type,train_df, test_df, feats, 'xco2_oco2_aggregated_mean')
            if args.verbose:
                print('PCA explained variance: ', pca.explained_variance_ratio_)
                # plot the skree plot
                plt.plot(np.cumsum(pca.explained_variance_ratio_))
                plt.xlabel('number of components')
                plt.ylabel('cumulative explained variance')
                plt.show()
    else:
        X_train, y_train, X_test, y_test = create_train_test_split(gain, surface_type,train_df, test_df, feats, 'xco2_oco2_aggregated_mean')
        # Normalize the data
        X_train = (X_train - X_train.mean()) / X_train.std()
        X_test = (X_test - X_train.mean()) / X_train.std()
    # print('X_train head: ', X_train.head())
    # print('X_test head: ', X_test.head())
    print('X_train shape: ', X_train.shape)
    print('X_test shape: ', X_test.shape)



    kernel = ConstantKernel() + Matern(length_scale=1, nu=5/2) + WhiteKernel(noise_level=1)

    model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True, alpha=0.1, random_state=42)

    model.fit(X_train, y_train)

    y_pred, y_std = model.predict(X_test,return_std=True)
    y_pred_train, y_std_train = model.predict(X_train,return_std=True)

    # save model
    pickle.dump(model, open('model_'+str(surface_type)+'_'+str(gain)+'.pkl', 'wb'))
    # add predictions and ground truth XCO2 to the test & train dataframes
    test_df['xco2_gosat_ML'] = test_df['xco2_gosat_AK_Corr'] - y_pred
    test_df['xco2_gosat_true'] = test_df['xco2_gosat_AK_Corr'] - y_test
    train_df['xco2_gosat_ML'] = train_df['xco2_gosat_AK_Corr'] - y_pred_train
    train_df['xco2_gosat_true'] = train_df['xco2_gosat_AK_Corr'] - y_train
    test_df['xco2_gosat_ML_uncert'] = y_std
    train_df['xco2_gosat_ML_uncert'] = y_std_train
    # plotting
    save_path = '/Users/williamlumenus/Desktop/Projects/harmonization/GP_Plots/'

    viz_residuals(test_df,save_path,surface_type=surface_type,gain=gain,save_fig=True, training_dist = False)
    viz_residuals(train_df,save_path,surface_type=surface_type,gain=gain,save_fig=True, training_dist = True)
    viz_uncertainty(test_df,save_path,surface_type=surface_type,gain=gain,save_fig=True,OOD=True)
    viz_uncertainty(train_df,save_path,surface_type=surface_type,gain=gain,save_fig=True,OOD=False)

    # viz_feature_importance(model,feats,save_path,surface_type=surface_type,gain=gain,save_fig=True)
 
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    args = arg_parser()
    main(args)