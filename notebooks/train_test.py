#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import dateutil.parser
from numpy import random
random.seed(123)


# In[2]:


elec_path = '../data/processed/temp_open_utc_complete.csv'


# In[3]:


meta_path = '../data/raw/meta_open.csv'


# In[4]:


weather_path = '../data/external/weather/weather1.csv'


# In[5]:


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
    month: one-hot coded
    day: from elec, int
    day_of_the_week: one-hot coded
    hour: from elec, hour from weather is converted to the nearest :00, int
    area: from meta, float
    primary_space_usage: from meta (primaryspaceuse_abbrev), one-hot coded
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
    buildings = list(meta[(meta['newweatherfilename'] == weather_file) & (meta['industry']==industry)].index) #name of the buildings
    df = pd.DataFrame(columns=['building_name', 'timestamp', 'electricity', 'area', 'primary_space_usage']) #empty dataframe with 3 columns
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
    df['timestamp'] = df['timestamp'].apply(cutoff_minute)#drop time zone info
    df = df.join(weather, on='timestamp', how='inner', lsuffix='elec', rsuffix='weather') #join temperature data from weather table
    #df = add_month_day_hour(df) #to do
    #df = add_day_of_the_week()
    return df.reset_index()


# In[38]:


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

    #Creat an empty dataframe with 5 columns
    df = pd.DataFrame(columns=['building_name', 'timestamp', 'electricity', 'area', 'primary_space_usage']) 

    #for now, this loop is fine, but maybe rewrite to make code faster/ look for function in pandas
    for building in buildings:
    #filename followed by [['column name']] selects specific columns in datafram
        subdf = elec[['timestamp', building]] #extracting the timestamp and electricity data from all buildings in electricity table
        subdf.columns = ['timestamp', 'electricity'] #naming columns
        subdf['building_name'] = building #non temporal column with building name, all same entry
        subdf['area'] = meta.loc[building, 'sqm'] #"" making new column called area
        subdf['primary_space_usage'] = meta.loc[building, 'primaryspaceuse_abbrev']
        df = pd.concat([df, subdf], axis=0, ignore_index=True) #combining the two tables. Where axis =0 puts the table under the first one, axis = 1 puts the table to the right 
        #df has 'building_name', timestamp, electricity meter, area, primary space usage
        # The timestamps are not matched up in the minutes between external and building data, cutoff the minutes and just match to hour 
        weather['rounded_timestamp'] = weather['timestamp'].apply(cutoff_minute) 
        #cutoff_minute is a function that is implemented below
    weather = weather.groupby('rounded_timestamp').first() #only the first observation in each hour is taken
    weather = weather['TemperatureC'] #only need temperature column

    df['timestamp'] = df['timestamp'].apply(cutoff_minute) #to drop timezone information
    df = df.join(weather, on='timestamp', how='inner', lsuffix='elec', rsuffix='weather') #join temperature data from weather table 
    #join aligns the tables by the timestamp index, instead of merely merging them, so that each instinance of a timestamp for each building has weather data

    #Adding columns for the month, year, date, hour, and weekday
    df['month']=df['timestamp'].apply(lambda x: x.month)
    df['year']=df['timestamp'].apply(lambda x: x.year)
    df['date']=df['timestamp'].apply(lambda x: x.day)
    df['hour']=df['timestamp'].apply(lambda x: x.hour)
    df['weekday']=df['timestamp'].apply(lambda x: x.dayofweek)

    #One-hot encode for month, year, date, hour, and weekday is pd.get_dummies
    #combining df with one-hot encodes the tables.  axis =1 puts the table under the first one, axis = 0 puts the table to the right 
    df.reset_index()
    df = pd.concat([df, pd.get_dummies(df['month'], prefix='month')],axis=1) 
    df = pd.concat([df, pd.get_dummies(df['date'], prefix='date')], axis=1) 
    df = pd.concat([df, pd.get_dummies(df['hour'], prefix='hour')], axis=1) 
    df = pd.concat([df, pd.get_dummies(df['weekday'], prefix='wkday')], axis=1) 
    df = pd.concat([df, pd.get_dummies(df['primary_space_usage'], prefix='PSU')], axis=1) 
    return df.reset_index(drop=True)


# In[39]:


df = make_dataset(elec_path, meta_path, weather_path, 'weather1.csv', 'Education')
display(df)


# In[40]:


def train_test(df, meta_path, weather_file, industry):
    meta = pd.read_csv(meta_path)
    #set 'uid' as index in meta
    meta = meta.set_index('uid')
    buildings = list(meta[(meta['newweatherfilename'] == weather_file) & (meta['industry']==industry)].index) 
    np.random.shuffle(buildings)
    train_buildings = buildings[:50]#take 50 buildings
    test_buildings = buildings[50:]#take the others
    df_train = df[df['building_name'].isin(train_buildings)]
    df_test = df[df['building_name'].isin(test_buildings)]
    df_train = df_train.sort_values(['building_name', 'timestamp']).reset_index(drop=True)
    df_test = df_test.sort_values(['building_name', 'timestamp']).reset_index(drop=True)
    df_train.to_csv('../data/weather1_education_train.csv')
    df_test.to_csv('../data/weather1_education_test.csv')


# In[41]:


train_test(df, meta_path, 'weather1.csv', 'Education')


# In[44]:


train = pd.read_csv('../data/weather1_education_train.csv')
train.head()


# In[45]:


train.shape


# In[46]:


test = pd.read_csv('../data/weather1_education_test.csv')
test.head()


# In[47]:


test.shape


# In[48]:


train['building_name'].unique().shape


# In[49]:


test['building_name'].unique().shape

