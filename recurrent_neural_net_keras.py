from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from keras import regularizers, optimizers, callbacks, activations

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

###################################################   HELPER FUNCTIONS

def prep_input(data, precision_geo):
    
    data['geo_lim'] = data.geohash6.str[0:precision_geo]
    data['time'] = pd.to_datetime(data.timestamp, format='%H:%M')
    data['date_time_index'] = data.time + pd.TimedeltaIndex(data.day, unit = 'D')
    
    datagg = data.groupby(['geo_lim','date_time_index'])['demand'].mean().reset_index(name='mean')

    X = datagg.pivot(index='date_time_index', columns='geo_lim', values='mean').fillna(0, inplace=False)
    return X

def prep_label(data, start, end):
    a = data.day.astype(int)  >= start
    b = data.day.astype(int)  <= end
    indices = a & b
    Yprep = data[indices]

    Y = Yprep.demand.astype(float)
    Y = Y.to_numpy().reshape(-1,1)
    return Y

###################################################   INPUT
data= pd.read_csv('training.csv')
print("reading data done : "+str(int(time.time()-start_time))+" s")

training_periods = 96*7 # 96 is the number of intervals per day
test_periods = 5 # following the test period
precision = 4 # number of digit in the geo param (max is 6)  this parameter increases size O(36^n)
train = True # otherwise use the latest
lookback = 4*4 # number of periods to lookback; 4 per hour 

###################################################   DATA PREP

data.demand = data.demand
Xprep = prep_input(data , precision)

#remove first time step from Y and the last from X 
Yprep = Xprep[1:]
Xprep.drop(Xprep.tail(1).index,inplace=True)

# filtering for the dates we need
X = np.array(Xprep[lookback:training_periods])
Y = np.array(Yprep[lookback:training_periods])

#shaping for the LTSM layer requirement
X = X.reshape(X.shape[0],1,X.shape[1])

# adding the data to look back at
for j in range(lookback-1):
    Xlookback = np.array(Xprep[lookback-j-1:training_periods-j-1])
    Xlookback = Xlookback.reshape(Xlookback.shape[0],1,Xlookback.shape[1])
    X = np.append( X, Xlookback , axis=1 )

# preparing the test data
Xtest = Xprep[training_periods:training_periods+test_periods]
Ytest = Yprep[training_periods:training_periods+test_periods]

Xtest = np.array(Xtest)
Xtest = Xtest.reshape(Xtest.shape[0],1,Xtest.shape[1])

# adding the data to look back at
for j in range(lookback-1):
    Xlookback = np.array(Xprep[training_periods-j-1:training_periods+test_periods-j-1])
    Xlookback = Xlookback.reshape(Xlookback.shape[0],1,Xlookback.shape[1])
    Xtest = np.append( Xtest, Xlookback , axis=1 )

Ytest = np.array(Ytest)

print("data prep done : " +str(int(time.time()-start_time))+" s")
 
nbcolumns = len(X[0][0])

##################################################  HYPERPARAMETERS
h_layer1_nodes = int(nbcolumns*lookback*np.log(training_periods))
h_layeri_nodes = int(nbcolumns*lookback*np.log(training_periods))
h_layerf_nodes = int(nbcolumns*lookback*np.log(training_periods))

nb_h_layers = 0 # this doesn't account for the first and last hidden layers

print("size of each input vector : "+ str(nbcolumns))
print("size of 1st hidden layers : "+ str(h_layer1_nodes))
print("nb of i hidden layers : "+ str(nb_h_layers))
print("size of i hidden layers : "+ str(h_layeri_nodes))
print("size of f hidden layers : "+ str(h_layerf_nodes)+"\n")

###################################################   MODEL


if train:
    # create model
    model = Sequential()
    
    model.add(LSTM(h_layer1_nodes, input_shape=(lookback, nbcolumns),
                 activation='relu', return_sequences=True))  

    for _ in range(nb_h_layers):
        model.add(LSTM(h_layeri_nodes, activation='relu', return_sequences=True))
    
    model.add(LSTM(h_layerf_nodes, activation='relu'))
    
    model.add(Dense(nbcolumns, activation='relu'))

    model.compile(loss='mean_absolute_error', 
        optimizer='sgd', metrics=['mean_absolute_error'])

    model.fit(X, Y, epochs=200, batch_size=int(training_periods/10), 
        validation_split=0.2,  verbose=2)

    # evaluate the model
    scores = model.evaluate(X, Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]))
    print("training in : "+str(int(time.time()-start_time))+" s")

    model.save("model.h5")
    
else:
    # use the latest model saved
    model = load_model('model.h5')


###################################################   RESULTS
# print weights
list_weights = []
for layer in model.layers:

    weights=layer.get_weights()
    for i in range(len(weights)):
        if np.isscalar(weights[i]):
            list_weights.append(weights[i])
        else:
            for j in range(len(weights[i])):
                if np.isscalar(weights[i][j]):
                    list_weights.append(weights[i][j])
                else:
                    for k in range(len(weights[i][j])):
                        list_weights.append(weights[i][j][k])

print("min weights = "+str(np.min(list_weights)))
print("max weights = "+str(np.max(list_weights)))
print("mean weights = "+str(np.mean(list_weights)))
print("std weights = "+str(np.std(list_weights))+"\n")

# calculate predictions and metrics
predictions = model.predict(Xtest)

mape = np.sum(np.abs(Ytest-predictions))/np.sum(Ytest)
mse = ((Ytest-predictions)**2).mean(axis=None)
mae = (np.abs(Ytest-predictions)).mean(axis=None)

print("prediction in : "+str(int(time.time()-start_time))+" s")
#print and save results and weights
print("test MAE: " + str(mae*100))
print("test MAPE: %.2f%%" % (mape*100))
print("test MSE: %.2f%%" % (mse*100))

#print first rows
print("input :")
print(Xtest[0][0])
print("prediction :")
print(predictions[0])
print("actual :")
print(Ytest[0])

myFile = open('output.csv', 'w', newline='')
with myFile:
    writer = csv.writer(myFile)
    writer.writerow(Xprep.columns)
    for i in range(test_periods):
        writer.writerow((100*Ytest[i]).astype(int))
        writer.writerow((100*predictions[i]).astype(int))
        writer.writerow(" ")
    writer.writerow(" ##### END ##### ")
    
model_name = "./models/model "+"UTC "+str(datetime.now())+" "+str(training_periods)+" MAPE="+str(mape)+" MSE= "+str(mse)+".h5"
model_name= re.sub('[:]+', '', model_name)
model.save(model_name)
