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
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# import sklearn tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2

def arg_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Harmonization')
    parser.add_argument('--UQ', type=bool, default=True, help='Use UQ model')
    parser.add_argument('--verbose', type=bool, default=True, help='verbose')
    parser.add_argument('--n_estimators', type=int, default=50, help='n_estimators')
    parser.add_argument('--feats', type=str, default='all', help='features')
    return parser.parse_args()

def main(args):
    assert args.feats in ['all', 'land_H', 'land_M', 'ocean_H',], 'Invalid feature set'

    if args.feats == 'all':
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
            'xco2_gosat',
        ]
    if args.feats == 'land_H':
        # checkt that the surface type and gain are correct, print an error if they are not
        surface_type = 1 # 'Surface type should be 1 for land'
        gain = 72 # 'Gain should be 72 for H'
        feats_all = [
            'xco2_gosat',
            'xco2_uncertainty',
            'fs',
            'aod_strataer',
            't700',
            'dust_height',
            'albedo_slope_sco2',
            'aod_total',
            'deltaT',
            'rms_rel_wco2',
            'psurf',
            'aod_ice',
            'water_height',
            'aod_oc',
            'albedo_wco2',
            'aod_dust',
        ]
    if args.feats == 'land_M':
        # checkt that the surface type and gain are correct, print an error if they are not
        surface_type = 1
        gain = 77
        feats_all = [
            'aod_sulfate',
            'albedo_wco2',
            'albedo_sco2',
            'dust_height',
            'albedo_o2a',
        ]
    if args.feats == 'ocean_H':
        # checkt that the surface type and gain are correct, print an error if they are not
        surface_type = 0 #'Surface type should be 0 for ocean'
        gain = 72 # 'Gain should be 72 for H'
        feats_all = [
            'aod_fine',
            'aod_strataer',
            'aod_total',
            'aod_oc',
            'aod_sulfate'
            'fs',
        ]
    

    # load data
    file_path = '/Users/williamlumenus/Desktop/Projects/harmonization/Data/matched_gosat_v9_litevars_oco2_v11.1_xco2_20140906_20200630.pkl'
    df = pd.read_pickle(file_path)

    # remove soundings where oco_aggregated_xco2_stddev_ppm <= oco_max_allowed_xco2_variability
    df = df[df['xco2_oco2_aggregated_std'] <= 1.5]


    # create training and testing data
    train_df, test_df = split_evaluation_data_by_year(df, 2019)
    X_train, y_train, X_test, y_test = create_train_test_split(gain, surface_type,train_df, test_df, feats_all, 'xco2_oco2_aggregated_mean')
    print('X_train head: ', X_train.head())
    train_df = train_df[train_df['gain'] == gain]
    test_df = test_df[test_df['gain'] == gain]
    train_df = train_df[train_df['gosat_surface_type'] == surface_type]
    test_df = test_df[test_df['gosat_surface_type'] == surface_type]
    print('X_train shape: ', X_train.shape)
    print('X_test shape: ', X_test.shape)

    # define a model
    # model = HarmonizationModel(n_estimators=200, max_depth=25, random_state=0)
    if args.UQ:
        base_learner = DecisionTreeRegressor(max_depth=7,)
        # use 500 trees for Land-M
        model = NGBRegressor(n_estimators=args.n_estimators, learning_rate=0.01, Base=base_learner, Dist=Normal, Score=CRPScore, verbose=args.verbose)

    else:
        model = RandomForestRegressor(n_estimators=300, max_depth=25, random_state=0,)

    # if UQ:
    #     model = HarmonizationUQModel(n_estimators=700, learning_rate=0.01, minibatch_frac=0.5, max_depth = 10)
    # else:
    #     model = HarmonizationModel(n_estimators=200, max_depth=25, random_state=0)

    # fit the model
    model.fit(X_train, y_train, X_val=X_test, Y_val=y_test)


    # predict
    y_pred = model.predict(X_test, ) # max_iter = model.best_val_loss_itr)
    y_pred_train = model.predict(X_train, ) #max_iter = model.best_val_loss_itr)
    if args.UQ:
        y_pred_UQ = model.pred_dist(X_test, ) # max_iter = model.best_val_loss_itr)
        y_pred_train_UQ = model.pred_dist(X_train, ) # max_iter = model.best_val_loss_itr)
        y_pred_UQ = np.array(y_pred_UQ.params['scale'])
        y_pred_train_UQ = np.array(y_pred_train_UQ.params['scale'])
 

        


    # save model
    # model.save_model('model_test.pkl')
    if args.UQ:
        pickle.dump(model, open('model_UQ_'+str(surface_type)+'_'+str(gain)+'.pkl', 'wb'))
    else:
        pickle.dump(model, open('model_'+str(surface_type)+'_'+str(gain)+'.pkl', 'wb'))

    # add predictions and ground truth XCO2 to the test & train dataframes
    test_df['xco2_gosat_ML'] = test_df['xco2_gosat_AK_Corr'] - y_pred
    test_df['xco2_gosat_true'] = test_df['xco2_gosat_AK_Corr'] - y_test
    train_df['xco2_gosat_ML'] = train_df['xco2_gosat_AK_Corr'] - y_pred_train
    train_df['xco2_gosat_true'] = train_df['xco2_gosat_AK_Corr'] - y_train

    if args.UQ:
        test_df['xco2_gosat_ML_uncert'] = y_pred_UQ
        train_df['xco2_gosat_ML_uncert'] = y_pred_train_UQ
    
    # plotting
    save_path = '/Users/williamlumenus/Desktop/Projects/harmonization/NG_Plots/'

    viz_residuals(test_df,save_path,surface_type=surface_type,gain=gain,save_fig=True, training_dist = False)
    viz_residuals(train_df,save_path,surface_type=surface_type,gain=gain,save_fig=True, training_dist = True)
    viz_uncertainty(test_df,save_path,surface_type=surface_type,gain=gain,save_fig=True,OOD=True)
    viz_uncertainty(train_df,save_path,surface_type=surface_type,gain=gain,save_fig=True,OOD=False)

    viz_feature_importance(model,feats_all,save_path,surface_type=surface_type,gain=gain,save_fig=True)
 



if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    args = arg_parser()
    main(args)

