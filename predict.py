import numpy as np
import pandas as pd

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from keras import regularizers, optimizers, callbacks, activations
from keras import backend as K

import tensorflow as tf

import csv
import time
from datetime import datetime
import re
import matplotlib.pyplot as plt

import train 

Xprepn, mXprepn, sXprepn, Xfull, full_size, nbcolumns, test_periods,lookback, model, model2, scaling_vector = train.execute_script('input.csv')

#remove warnings tensorflow
tf.logging.set_verbosity(tf.logging.ERROR)

##################################################   DATA PREP FOR NETWORK 1

Ypredict = np.zeros((1,full_size))
Xtest_full = np.array(Xprepn)

for _ in range(test_periods):
    # preparing the test data
    Xtest = np.array(Xtest_full[-1])
    Xtest = Xtest.reshape(1,1,nbcolumns)

    # adding the data to look back at
    for j in range(lookback-1):
        Xlookback = np.array(Xprepn.iloc[-j-2])
        Xlookback = Xlookback.reshape(1,1,nbcolumns)
        Xtest = np.append( Xtest, Xlookback , axis=1 )

    ###################################################   PREDICT / TIME_SCALE / EXPLODE 

    Yhatn = model.predict(Xtest)
    Yhats = train.scaling(Yhatn, scaling_vector)
    Yhat = train.denormalize(Yhats, mXprepn, sXprepn)
    Yhat2 = model2.predict(Yhat)
    
    ################################################### STORE OUTPUT AND MOVE 1 STEP
    Xtest_full = np.append( Xtest_full, Yhatn , axis=0 )
    Ypredict= np.append( Ypredict, Yhat2, axis=0)


############### PREPARE DATA AND WRITE FILE

row_names = [(max(Xfull.index)+ pd.Timedelta(15*i, unit = 'm')) for i in range(1,test_periods+1)]

Ypd = pd.DataFrame(Ypredict[1:],index =row_names, columns=Xfull.columns)
Ypd = Ypd.unstack().reset_index(name='demand').rename(columns={'level_1': 'datetime'}, inplace=False)
Ypd['day'] = ((Ypd.datetime - pd.to_datetime('1/1/1900'))/np.timedelta64(1,'D')).astype('int')
Ypd['timestamp'] = Ypd.datetime.dt.strftime('%H:%M')

del Ypd['datetime']

Ypd = Ypd[['geohash6', 'day', 'timestamp', 'demand']]

Ypd.to_csv('prediction.csv',sep=',', encoding='utf-8', index=False)


