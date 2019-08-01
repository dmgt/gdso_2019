#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import dateutil.parser


# In[ ]:


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
    weather['timestamp'] = weather['timestamp'].apply(dateutil.parser.parse)
    elec['timestamp'] = elec['timestamp'].apply(dateutil.parser.parse)
    #construct the dataframe to return
    buildings = list(meta[(meta['newweatherfilename'] == 'weather1.csv') & (meta['industry']=='Education')].index) #name of the buildings
    df = pd.DataFrame(columns={'building_name', 'timestamp', 'electricity', 'area', 'primary_space_usage'}) #empty dataframe with 3 columns
    for building in buildings:
        subdf = elec[['timestamp', building]]
        subdf.columns = ['timestamp', 'electricity']
        subdf['building_name'] = building
        subdf['area'] = meta.loc[building, 'sqm']
        subdf['primary_space_usage'] = meta.loc[building, 'primaryspaceuse_abbrev']
        df = pd.concat([df, subdf], axis=0, ignore_index=True)
    #df has 'building_name', timestamp, electricity meter, area, primary space usage
    print('OK1')
    weather['rounded_timestamp'] = weather['timestamp'].apply(cutoff_minute) #cutoff_minute is implemented separately
    weather = weather.groupby('rounded_timestamp').first() #only the first observation in each hour is taken
    weather = weather['TemperatureC'] #only need temperature column
    print('OK2')
    df['timestamp'] = df['timestamp'].apply(cutoff_minute)
    df = df.join(weather, on='timestamp', how='inner', lsuffix='elec', rsuffix='weather') #join temperature data from weather table
    df = ... #TODO convert primary_space_usage into one-hot encode
    #df = add_month_day_hour(df) #TODO
    #df = add_day_of_the_week(df) #TODO
    return df.reset_index()


# In[ ]:


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

