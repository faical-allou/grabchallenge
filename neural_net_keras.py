from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
from keras import callbacks
from keras.models import load_model

import tensorflow as tf
import numpy as np
import csv
import time
from datetime import datetime
import pandas as pd
import re

#remove warnings tensorflow
tf.logging.set_verbosity(tf.logging.ERROR)

start_time = time.time()

##### some helper functions


def prep_input(data, start, end, start_time, precision_geo):
    
    data['geo_lim'] = data.geohash6.str[0:precision_geo]
    
    a = data.day.astype(int)  >= start
    b = data.day.astype(int)  <= end
    indices = a & b
    Xprep = data[indices]

    weekday = np.remainder(Xprep.day.astype(int) ,7)
    timestamps = ("0:0","0:15","0:30","0:45","10:0","10:15","10:30","10:45","11:0",
    "11:15","11:30","11:45","12:0","12:15","12:30","12:45","13:0","13:15","13:30",
    "13:45","14:0","14:15","14:30","14:45","15:0","15:15","15:30","15:45","16:0","16:15",
    "16:30","16:45","17:0","17:15","17:30","17:45","18:0","18:15","18:30","18:45","19:0",
    "19:15","19:30","19:45","1:0","1:15","1:30","1:45","20:0","20:15","20:30","20:45","21:0",
    "21:15","21:30","21:45","22:0","22:15","22:30","22:45","23:0","23:15","23:30","23:45","2:0",
    "2:15","2:30","2:45","3:0","3:15","3:30","3:45","4:0","4:15","4:30","4:45","5:0","5:15",
    "5:30","5:45","6:0","6:15","6:30","6:45","7:0","7:15","7:30","7:45",
    "8:0","8:15","8:30","8:45","9:0","9:15","9:30","9:45")

    onehot_geo = pd.get_dummies(pd.Categorical(Xprep.geo_lim, categories = np.unique(data.geo_lim)))
    onehot_day = pd.get_dummies(pd.Categorical(Xprep.day, categories = np.unique(data.day)))   
    onehot_weekday = pd.get_dummies(pd.Categorical(weekday, categories = np.arange(0,7)))
    onehot_timestamp = pd.get_dummies(pd.Categorical(Xprep.timestamp, categories = timestamps)) 
    
    X = np.hstack((onehot_geo, onehot_day, onehot_weekday, onehot_timestamp))
    Xtime = Xprep.timestamp
    return X, Xtime

def prep_label(data, start, end):
    a = data.day.astype(int)  >= start
    b = data.day.astype(int)  <= end
    indices = a & b
    Yprep = data[indices]

    Y = Yprep.demand.astype(float)
    Y = Y.to_numpy().reshape(-1,1)
    return Y

############## INPUT
data= pd.read_csv('training.csv')
print("reading data done : "+str(int(time.time()-start_time))+" s")

start_training = 1
end_training = 7
test_day = 21
test_times =["20:0", "20:05", "20:10", "20:15", "20:20"]
precision = 5 # number of digit in the geo param (max is 6)  this parameter increases size O(36^n)
train = True # otherwise use the latest

############### START
dayrange = list(range(start_training, end_training))
dayrange.append(test_day)

data = data[data.day.isin(dayrange)]

X,Xtime = prep_input(data , start_training , end_training , start_time , precision)
Y = prep_label(data , start_training , end_training)

Xtest,Xtesttime = prep_input( data, test_day , test_day , start_time , precision)
Ytest = prep_label( data, test_day , test_day)

filter_time = Xtesttime.isin(test_times)
Xtest = Xtest[filter_time,:]
Ytest = Ytest[filter_time,:]

print("data prep done : " +str(int(time.time()-start_time))+" s")
 
input_layer_size = X[0,:].size
print("size of each input vector : "+str(input_layer_size))
np.random.seed(1)

if train:
    # create model
    model = Sequential()
    model.add(Dense(1000, input_dim=input_layer_size, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['mean_absolute_percentage_error'])

    # Fit the model or reuse
    callback = [callbacks.EarlyStopping(monitor='mean_absolute_percentage_error', 
        min_delta=0.001, patience=3, verbose=0, mode='auto', baseline=None, restore_best_weights=False)]
    model.fit(X, Y, epochs=20, batch_size=60000, validation_split=0.1, callbacks = callback)

    # evaluate the model
    scores = model.evaluate(X, Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]))
    print("training in : "+str(int(time.time()-start_time))+" s")

    model.save("model.h5")
    
else:
    model = load_model('model.h5')

# calculate predictions
predictions = model.predict(Xtest)
mape = np.sum(abs(Ytest-predictions))/np.sum(Ytest)
mse = ((Ytest-predictions)**2).mean(axis=None)

myFile = open('output.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(np.hstack((Ytest,predictions)))
    writer.writerow(" ##### END ##### ")
    
print("test MAPE: %.2f%%" % (mape*100))
print("test MSE: %.2f%%" % (mse*100))
print("prediction in : "+str(int(time.time()-start_time))+" s")

model_name = "model "+"UTC "+str(datetime.now())+" slice= "+str(start_training)+" "+str(end_training)+" "+str(test_day)+" MAPE="+str(mape)+" MSE= "+str(mse)+".h5"
model_name= re.sub('[:]+', '', model_name)
model.save(model_name)
