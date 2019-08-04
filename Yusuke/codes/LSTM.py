#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
tf.enable_eager_execution()
from tensorflow import keras 
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[2]:


train = pd.read_csv('../data/weather1_education_train.csv')
train.head()


# In[3]:


test = pd.read_csv('../data/weather1_education_test.csv')


# In[4]:


#three_month = 2159 #from 12/01 0:00 to 1/31 23:00
#predict_len = 6598 #length of prediction, 8757 is the length of 1 year-long observation


# In[5]:


three_month = 6574 #from 12/01 0:00 to 8/31 23:00
predict_len = 8757 - three_month #length of prediction, 8757 is the length of 1 year-long observation


# In[6]:


inputs = layers.Input(shape=(1, three_month)) #temporal data: electricity
x = layers.LSTM(units=128, input_shape=(three_month, 1), activation='relu')(inputs)
x = layers.Dense(predict_len, activation='relu')(x)
model = keras.models.Model(inputs=inputs, outputs=x)


# In[7]:


opt = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae', 'mse']) 
model.summary()


# In[8]:


from dateutil import parser
train['timestamp'] = train['timestamp'].apply(parser.parse)


# In[9]:


from datetime import datetime
#end = datetime(2015, 2, 28, 23)
end = datetime(2015, 8, 31, 23)
X_train = train[train['timestamp'] <= end]
Y_train = train[train['timestamp'] > end]


# In[10]:


X_train = np.array(X_train['electricity'])
X_train = np.reshape(X_train, (50, 1, three_month))


# In[11]:


Y_train = np.array(Y_train['electricity'])
Y_train = np.reshape(Y_train, (50, predict_len))


# In[21]:


model.fit(X_train, Y_train, epochs=1000, batch_size=10)


# In[22]:


Y_fitted = model.predict(X_train)


# In[23]:


plt.figure(figsize=(20,10))
plt.plot(Y_fitted[0], label='fitted')
plt.plot(Y_train[0], label='observed')
plt.legend();

