import numpy as np
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


def create_training_data(colocation_path,state_var_path,save_pickle=False, debug=False):
    '''
    Create training pickle from colocation and state variable data
    '''
    print('creating training data...')
    # variables to extract
    var_names = [
        'sounding_id',
        'xco2_uncertainty',
        'Retrieval/psurf',
        'Retrieval/windspeed',
        'Retrieval/t700',
        'Retrieval/fs',
        'Retrieval/tcwv',
        'Retrieval/tcwv_uncertainty',
        'Retrieval/dp',
        'Retrieval/dpfrac',
        'Retrieval/s31',
        'Retrieval/s32',
        'Retrieval/co2_grad_del',
        'Retrieval/dws',
        'Retrieval/offset_o2a_rel',
        'Retrieval/aod_dust',
        'Retrieval/aod_bc',
        'Retrieval/aod_oc',
        'Retrieval/aod_seasalt',
        'Retrieval/aod_sulfate',
        'Retrieval/aod_strataer',
        'Retrieval/aod_water',
        'Retrieval/aod_ice',
        'Retrieval/aod_fine',
        'Retrieval/aod_total',
        'Retrieval/ice_height',
        'Retrieval/water_height',
        'Retrieval/dust_height',
        'Retrieval/h2o_scale',
        'Retrieval/deltaT',
        'Retrieval/albedo_o2a',
        'Retrieval/albedo_wco2',
        'Retrieval/albedo_sco2',
        'Retrieval/albedo_slope_o2a',
        'Retrieval/albedo_slope_wco2',
        'Retrieval/albedo_slope_sco2',
        'Retrieval/rms_rel_wco2',
        'Retrieval/rms_rel_sco2',

    ]
    
    # load colocation data
    coloc_data = h5.File(colocation_path,'r')
    coloc_names = ['gosat_sid','latitude','longitude','gosat_QF','gain','gosat_surface_type','xco2_gosat', 'xco2_oco2_aggregated_mean', 'xco2_oco2_aggregated_median', 'xco2_oco2_aggregated_std']
    gosat_sid = np.array(coloc_data['Gosat/sounding_id'][:])
    latitude = np.array(coloc_data['Gosat/latitude'][:])
    longitude = np.array(coloc_data['Gosat/longitude'][:])
    gosat_QF = np.array(coloc_data['Gosat/xco2_quality_flag'][:])
    gain = np.array(coloc_data['Gosat/Sounding/gain'][:])
    gosat_surface_type = np.array(coloc_data['Gosat/Retrieval/surface_type'][:])
    xco2_gosat = np.array(coloc_data['Gosat/xco2'][:])
    xco2_oco2_aggregated_mean = np.array(coloc_data['oco_aggregated_xco2_mean_ppm'][:])
    xco2_oco2_aggregated_median = np.array(coloc_data['oco_aggregated_xco2_median_ppm'][:])
    xco2_oco2_aggregated_std = np.array(coloc_data['oco_aggregated_xco2_stddev_ppm'][:])

    # make dataframe
    coloc_df = pd.DataFrame(np.array([gosat_sid,latitude,longitude,gosat_QF,gain,gosat_surface_type,xco2_gosat,xco2_oco2_aggregated_mean,xco2_oco2_aggregated_median,xco2_oco2_aggregated_std]).T,columns=coloc_names)
    coloc_df['gosat_sid'] = coloc_df['gosat_sid'].astype(int)
    coloc_df['latitude'] = coloc_df['latitude'].astype(float)
    coloc_df['longitude'] = coloc_df['longitude'].astype(float)
    coloc_df['gosat_QF'] = coloc_df['gosat_QF'].astype(int)
    coloc_df['gain'] = coloc_df['gain'].astype(int)
    coloc_df['gosat_surface_type'] = coloc_df['gosat_surface_type'].astype(int)
    coloc_df['xco2_gosat'] = coloc_df['xco2_gosat'].astype(float)
    coloc_df['xco2_oco2_aggregated_mean'] = coloc_df['xco2_oco2_aggregated_mean'].astype(float)
    coloc_df['xco2_oco2_aggregated_median'] = coloc_df['xco2_oco2_aggregated_median'].astype(float)
    coloc_df['xco2_oco2_aggregated_std'] = coloc_df['xco2_oco2_aggregated_std'].astype(float)

    # remove 'Gosat/' from column names
    coloc_df.columns = [col.replace('Gosat/','') for col in coloc_df.columns]

    #  add the additional columns from var_names and fill with NaNs
    for var in var_names:
        coloc_df[var] = np.nan



    # glob .nc4 files in state_var_path
    state_var_files = glob.glob(state_var_path+'*.nc4')
    if debug:
        print('debug mode: only using first 10 files')
        state_var_files = state_var_files[:10]

    # loop through files and extract variables of interest
    for i, file in enumerate(state_var_files):
        print('extracting variables from file '+str(i)+' of '+str(len(state_var_files)))
        
        try:
            state_var_data = h5.File(file,'r')
            state_df = pd.DataFrame()
        except:
            continue
        for var in var_names:
            state_df[var] = np.array(state_var_data[var][:])

        # print number of rows
        print('number of rows in state_df: '+str(len(state_df)))
        
        # if sounding_id matches gosat_sid, then set that row in coloc_df[var_names] to the state_df[var_names] values
        n_matches = 0
        for sid in coloc_df['gosat_sid']:
            if sid in state_df['sounding_id'].values:
                n_matches += 1
                for var in var_names:
                    coloc_df.loc[coloc_df['gosat_sid']==sid,var] = state_df.loc[state_df['sounding_id']==sid,var].values
        print('number of matches: '+str(n_matches))

    # drop rows with NaNs
    print('dropping rows with NaNs...')
    coloc_df = coloc_df.dropna()
    print('number of observations available for training: '+str(len(coloc_df)))

    # remove 'Retrieval/' from column names
    coloc_df.columns = [col.replace('Retrieval/','') for col in coloc_df.columns]


 

    if not save_pickle:
        return coloc_df
    else:
        # save as pickle
        coloc_df.to_pickle('/Users/williamlumenus/Desktop/Projects/harmonization/Data/matched_gosat_v9_litevars_oco2_v10_xco2_20140906_20200630.pkl')
        print('done saving pickle')
        # return coloc_df


def _get_date_from_sid(data):
    '''
    Get date from sounding_id
    '''
    year = data['sounding_id'].astype(str).str[:4]
    month = data['sounding_id'].astype(str).str[4:6]
    # remove leading zeros
    month = month.str.lstrip('0')
    data['month'] = month
    data['year'] = year
    return data

def split_evaluation_data_by_year(data, test_year=2019):
    '''
    Split training data by year
    '''
    # assert year column exists in data otherwise create it
    if 'year' not in data.columns:
        data = _get_date_from_sid(data)
    # split data into train and test
    train = data.loc[data['year']!=str(test_year)]
    test = data.loc[data['year']==str(test_year)]

    return train, test

def create_train_test_split(gain, surface_type,train_data, test_data, features, label):
    '''
    Create train and test splits
    '''
    # filter by gain and surface type
    train_data = train_data.loc[(train_data['gain']==gain) & (train_data['surface_type']==surface_type)]
    test_data = test_data.loc[(test_data['gain']==gain) & (test_data['surface_type']==surface_type)]
    # split into X and y
    X_train = train_data[features]
    y_train = train_data['gosat_xco2'] - train_data[label]
    X_test = test_data[features]
    y_test = test_data['gosat_xco2'] - test_data[label]

    return X_train, y_train, X_test, y_test

def normalize_data(train_data, test_data, features):
    '''
    Normalize data
    '''
    # normalize data
    train_mean = train_data[features].mean()
    train_std = train_data[features].std()
    train_data[features] = (train_data[features] - train_mean) / train_std
    test_data[features] = (test_data[features] - train_mean) / train_std

    return train_data, test_data, train_mean, train_std

def k_fold_by_year(data):
    '''
    Split data into hold out sets by year
    '''
    # assert year column exists in data otherwise create it
    if 'year' not in data.columns:
        data = _get_date_from_sid(data)
    # get unique years
    years = data['year'].unique()
    # loop through years and create hold out sets
    for year in years:
        train = data.loc[data['year']!=str(year)]
        test = data.loc[data['year']==str(year)]
        yield train, test




 

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





