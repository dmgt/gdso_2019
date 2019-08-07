import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import dateutil.parser
from numpy import random

def make_dataset(elec_path, meta_path, weather_path, weather_file, industry):
    '''
    INPUT
    elec_path: path to timeseries data of electricity meter (temp_open_utc_complete.csv)
    meta: path to meta data table (meta_open.csv)
    weather: path to weather data table (weatherX.csv)
    weather_file: name of the weather file
    industry: name of undustry to focus on
    OUTPUT
    A dataframe in which each record represents a building at a certain time
    columns =[
    building_name: name of the building from meta, str
    month: one-hot encoded
    day: from elec, int
    day_of_the_week: one-hot encoded
    hour: from elec, hour from weather is converted to the nearest :00, int
    area: from meta, float
    primary_space_usage: from meta (primaryspaceuse_abbrev), one-hot encoded
    electricity: from elec
    temperature: from weather
    ]
    ------------------------------------------------------------------------------------------------------------------------------------------
    comment:
    -humidity is sometimes missing in weather table
    '''
    #read tables
    elec = pd.read_csv(elec_path)
    meta = pd.read_csv(meta_path)
    weather = pd.read_csv(weather_path)

    #set 'uid' as index in meta
    meta = meta.set_index('uid')

    #parse date
    weather['timestamp'] = weather['timestamp'].apply(dateutil.parser.parse) #changing timestamp from string to date time
    elec['timestamp'] = elec['timestamp'].apply(dateutil.parser.parse) # ""

    #construct the dataframe to return
    # Use the inputted Industry to pull all buildings from that industry
    buildings = list(meta[(meta['newweatherfilename'] == weather_file) & (meta['industry']==industry)].index) 
    np.random.shuffle(buildings)
    train_buildings = buildings[:50]
    test_buildings = buildings[50:]
    #Creat an empty dataframe with 5 columns
    df_train = pd.DataFrame(columns=['building_name', 'timestamp', 'electricity', 'area', 'primary_space_usage']) 
    df_test = pd.DataFrame(columns=['building_name', 'timestamp', 'electricity', 'area', 'primary_space_usage']) 
    #for now, this loop is fine, but maybe rewrite to make code faster/ look for function in pandas
    for building in buildings:
    #filename followed by [['column name']] selects specific columns in datafram
        subdf = elec[['timestamp', building]] #extracting the timestamp and electricity data from all buildings in electricity table
        subdf.columns = ['timestamp', 'electricity'] #naming columns
        subdf['building_name'] = building #non temporal column with building name, all same entry
        subdf['area'] = meta.loc[building, 'sqm'] #"" making new column called area
        subdf['primary_space_usage'] = meta.loc[building, 'primaryspaceuse_abbrev']
        #combining the two tables. Where axis =0 puts the table under the first one, axis = 1 puts the table to the right 
        if building in train_buildings:
            df_train = pd.concat([df_train, subdf], axis=0, ignore_index=True) 
        else:
            df_test = pd.concat([df_test, subdf], axis=0, ignore_index=True)
        #df has 'building_name', timestamp, electricity meter, area, primary space usage
        # The timestamps are not matched up in the minutes between external and building data, cutoff the minutes and just match to hour 
        weather['rounded_timestamp'] = weather['timestamp'].apply(cutoff_minute) 
        #cutoff_minute is a function that is implemented below
    weather = weather.groupby('rounded_timestamp').first() #only the first observation in each hour is taken
    weather = weather['TemperatureC'] #only need temperature column

    df_train['timestamp'] = df_train['timestamp'].apply(cutoff_minute) #to drop timezone information
    df_test['timestamp'] = df_test['timestamp'].apply(cutoff_minute) #to drop timezone information
    df_train = df_train.join(weather, on='timestamp', how='inner', lsuffix='elec', rsuffix='weather') #join temperature data from weather table 
    df_test = df_test.join(weather, on='timestamp', how='inner', lsuffix='elec', rsuffix='weather')
    #join aligns the tables by the timestamp index, instead of merely merging them, so that each instinance of a timestamp for each building has weather data

    #Adding columns for the month, year, date, hour, and weekday
    df_train['month'] = df_train['timestamp'].apply(lambda x: x.month)
    df_train['year'] = df_train['timestamp'].apply(lambda x: x.year)
    df_train['date'] = df_train['timestamp'].apply(lambda x: x.day)
    df_train['hour'] = df_train['timestamp'].apply(lambda x: x.hour)
    df_train['weekday'] = df_train['timestamp'].apply(lambda x: x.dayofweek)
    df_test['month'] = df_test['timestamp'].apply(lambda x: x.month)
    df_test['year'] = df_test['timestamp'].apply(lambda x: x.year)
    df_test['date'] = df_test['timestamp'].apply(lambda x: x.day)
    df_test['hour'] = df_test['timestamp'].apply(lambda x: x.hour)
    df_test['weekday'] = df_test['timestamp'].apply(lambda x: x.dayofweek)

    #One-hot encode for month, year, date, hour, and weekday is pd.get_dummies
    #combining df with one-hot encodes the tables.  axis =1 puts the table under the first one, axis = 0 puts the table to the right 
    df_train = df_train.reset_index(drop=True)
    df_train = pd.concat([df_train, pd.get_dummies(df_train['month'], prefix='month')],axis=1) 
    df_train = pd.concat([df_train, pd.get_dummies(df_train['date'], prefix='date')], axis=1) 
    df_train = pd.concat([df_train, pd.get_dummies(df_train['hour'], prefix='hour')], axis=1) 
    df_train = pd.concat([df_train, pd.get_dummies(df_train['weekday'], prefix='wkday')], axis=1) 
    df_train = pd.concat([df_train, pd.get_dummies(df_train['primary_space_usage'], prefix='PSU')], axis=1) 
    df_test = df_test.reset_index(drop=True)
    df_test = pd.concat([df_test, pd.get_dummies(df_test['month'], prefix='month')],axis=1) 
    df_test = pd.concat([df_test, pd.get_dummies(df_test['date'], prefix='date')], axis=1) 
    df_test = pd.concat([df_test, pd.get_dummies(df_test['hour'], prefix='hour')], axis=1) 
    df_test = pd.concat([df_test, pd.get_dummies(df_test['weekday'], prefix='wkday')], axis=1) 
    df_test = pd.concat([df_test, pd.get_dummies(df_test['primary_space_usage'], prefix='PSU')], axis=1) 

    return df_train, df_test
    

# Function to get rid of minute in timestamp, aligning time by hour only    
def cutoff_minute(dt):
    '''
    INPUT
    a datetime object has year, month, day, hour, and minute
    OUTPUT
    a datetime object has year, month, day, and hour
    '''
    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour
    return datetime.datetime(year, month, day, hour)
