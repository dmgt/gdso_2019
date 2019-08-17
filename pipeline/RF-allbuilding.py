#!/usr/bin/env python
# coding: utf-8



# Based partly on http://mariofilho.com/how-to-predict-multiple-time-series-with-scikit-learn-with-sales-forecasting-example/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
# from rfpimp import *  - not easy to install on cluster
from timeit import default_timer as timer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from tempfile import TemporaryFile

# - This notebook takes as input two files for train and test data

# ### Define functions to fit + evaluate model

# Functions to use for prediction evaluation

start = timer()

def rmsle(ytrue, ypred):
    return np.sqrt(mean_squared_log_error(ytrue, ypred))


# Train the model, measure accuracy, and error metrics

def make_model(X, y, X_valid, y_valid, n_estimators,
              max_features, min_samples_leaf, random_state):

    #global rf
    
    rf = RandomForestRegressor(n_estimators=n_estimators,
                           n_jobs=-1,
                           oob_score=True,
                           max_features=max_features, 
                           min_samples_leaf=min_samples_leaf,
                           random_state = random_state)
                           #verbose = 1

    #Inputting train dataset into the model
    rf.fit(X, y)

    return rf


# Using the model to make a prediction & Error Analysis for Random Forest 

def run_model(rf, X_valid):
    #global n, h, oob_valid
    #n = rfnnodes(rf)
    #h = np.median(rfmaxdepths(rf))
    #global y_pred
    y_pred = rf.predict(X_valid)
    #global mae_valid, rmsle_valid, r2_score_valid
    mae_valid = mean_absolute_error(y_valid, y_pred)
    rmsle_valid = np.sqrt( mean_squared_error(y_valid, y_pred) )
    r2_score_valid = rf.score(X_valid, y_valid)
    
    oob_valid = rf.oob_score_

    print(f"RF OOB score {rf.oob_score_:.5f}")
    print(f"Validation R^2 {r2_score_valid:.5f}, RMSLE {rmsle_valid:.5f}, MAE {mae_valid:.2f}")
    
    return y_pred,rmsle_valid,mae_valid,r2_score_valid



# 
# - **Features for this model**: area, building_type, temperature, + one-hot encoded hour, day of week, month of year


# Setting 'parse_dates' in this case parses both dates and times
# These files are too large to commit so they're uploaded locally under `/exploring_models` but not pushed
train_data = pd.read_csv('/global/cscratch1/sd/lininger/work/ML/train_light.csv', parse_dates = ['timestamp'])
val_data = pd.read_csv('/global/cscratch1/sd/lininger/work/ML/test_light.csv', parse_dates = ['timestamp'])

#print(train_data.dtypes.head())
#print(val_data.dtypes.head())


train_data['weather_file'] = train_data.weather_file.astype('category')
val_data['weather_file'] = val_data.weather_file.astype('category')

# Count the number of unique occurances of buildinf name in train
array_buildings_train, obs_per_building_train = np.unique(train_data['building_name'], return_counts = True)
#array_buildings_train
print('obs_bld_train contains this information:')
print(obs_per_building_train)
#print(obs_per_building_train.sum())
#Print counts to file
obs_bld_train = TemporaryFile()
np.save('obs_bld_train', obs_per_building_train)

# Count the number of unique occurances of buildinf name in validate
array_buildings_val, obs_per_building_val = np.unique(val_data['building_name'], return_counts = True)
array_buildings_val
#Print counts to file
print(obs_per_building_val)
#print(obs_per_building_val.sum())
#number of counts for each building in train, val
obs_bld_val = TemporaryFile()
np.save('obs_bld_val', obs_per_building_val)


X = train_data.drop(['electricity', 'building_name', 'timestamp','weather_file','industry'],axis=1)
X_valid = val_data.drop(['electricity', 'building_name', 'timestamp','weather_file','industry'], axis=1)
y, y_valid = train_data['electricity'], val_data['electricity']

# print y_val data to file
y_actual = TemporaryFile()
np.save('y_actual_save', y_valid)


#Run the rf model as a first pass to get baseline without tuning 
rf = make_model(X, y, X_valid, y_valid, 
                n_estimators = 200 ,max_features = 'auto', min_samples_leaf = 5, random_state = 42)
y_pred,rmsle_valid,mae_valid,r2_score_valid = run_model(rf, X_valid)



print(y_pred.shape)

end = timer()
print (end - start)

# Characterize feature importance

def calc_feature_importance(rf):
    feature_importances = pd.DataFrame(rf.feature_importances_,
                            index = X.columns,
                            columns=['importance']).sort_values('importance',ascending=False)

    return feature_importances



calc_feature_importance(rf)


# ### Tune hyperparameters


def tune_maxf(X, y, X_valid, y_valid, ntrees, maxf_min, maxf_max, maxf_step = 0.1):

# Fix minleaf while tuning this
    minleaf = 1
    
    list_of_r2_valid = []
    list_of_rmsle_valid = []
    list_of_mae_valid = []
    
    list_of_maxf = np.arange(maxf_min, maxf_max, maxf_step ).tolist()

    for maxf in np.arange(maxf_min, maxf_max, maxf_step ):
       print(f"n_estimators={ntrees}, max_features={maxf}, min_samples_leaf={minleaf}")
       rf = make_model(X, y, X_valid, y_valid, n_estimators = ntrees, 
                       max_features = maxf, min_samples_leaf = minleaf, 
                       random_state = 42)
    
       y_pred,rmsle_valid,mae_valid,r2_score_valid = run_model(rf, X_valid)
        
       list_of_r2_valid.append(r2_score_valid)
       list_of_rmsle_valid.append(rmsle_valid)
       list_of_mae_valid.append(mae_valid)
    
    
    min1 = list_of_r2_valid.index(min(list_of_r2_valid))
    min2 = list_of_rmsle_valid.index(min(list_of_rmsle_valid))
    min3 = list_of_mae_valid.index(min(list_of_mae_valid ))
                                    
    tuned_maxf = list_of_maxf[min2]
    
    print(f"tuned_maxf ={tuned_maxf}")
    
    return tuned_maxf


tuned_maxf = tune_maxf(X, y, X_valid, y_valid, ntrees = 200, maxf_min = 0.1, maxf_max = 0.7, maxf_step = 0.1)


def tune_minleaf(X, y, X_valid, y_valid, ntrees, maxf, minleaf_min, minleaf_max, minleaf_step = 1):

    list_of_r2_valid = []
    list_of_rmsle_valid = []
    list_of_mae_valid = []
    
    list_of_minleaf = list(range(minleaf_min, minleaf_max, minleaf_step ))

    for minleaf in range(minleaf_min, minleaf_max, minleaf_step ):
       print(f"n_estimators={ntrees}, max_features={maxf}, min_samples_leaf={minleaf}")
       rf = make_model(X, y, X_valid, y_valid, n_estimators = ntrees, max_features = maxf, min_samples_leaf = minleaf, 
                  random_state = 42)
    
       y_pred,rmsle_valid,mae_valid,r2_score_valid = run_model(rf, X_valid)
        
       list_of_r2_valid.append(r2_score_valid)
       list_of_rmsle_valid.append(rmsle_valid)
       list_of_mae_valid.append(mae_valid)
    
    
    min1 = list_of_r2_valid.index(min(list_of_r2_valid))
    min2 = list_of_rmsle_valid.index(min(list_of_rmsle_valid))
    min3 = list_of_mae_valid.index(min(list_of_mae_valid ))
                                    
    tuned_minleaf = list_of_minleaf[min2]
    
    print(f"tuned_minleaf ={tuned_minleaf}")
    
    return tuned_minleaf



tuned_minleaf = tune_minleaf(X, y, X_valid, y_valid, ntrees = 200, maxf = 0.1, minleaf_min = 1, minleaf_max = 7, minleaf_step = 1)


# ### Fit model with selected hyperparameters


def fit_tuned_rf(X, y, X_valid, y_valid, ntrees, random_state = 42):
    
    rf = make_model(X, y, X_valid, y_valid, n_estimators = ntrees, 
                   max_features = tuned_maxf, min_samples_leaf = tuned_minleaf, 
                   random_state = 42)
    
    y_pred,rmsle_valid,mae_valid,r2_score_valid = run_model(rf, X_valid)

    total_len = y_pred.shape[0]
    num_buildings = len(val_data['building_name'].unique())

    #print(total_len)
    #print(num_buildings)
    y_pred_rf = y_pred
#    y_pred_rf = y_pred.reshape((num_buildings, total_len//num_buildings))
    
    return y_pred_rf

#np.save(r_pred_rf,y_pred_rfDATA)
#number of counts for each building in train, val


y_pred_rf = fit_tuned_rf(X, y, X_valid, y_valid, ntrees = 200)
y_pred_rf.shape

y_pred_DATA = TemporaryFile()
np.save('y_pred_DATA', y_pred_rf)
