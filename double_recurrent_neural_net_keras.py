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

def normalize(X):
    meanX = np.mean(X,axis=0)
    stdX = np.std(X,axis=0)
    Xp = (X-meanX)/stdX
    return Xp, meanX, stdX  

def denormalize(Xp,m,s):
    out = []
    for x in Xp:
        out.append(x*s+m)
    return np.array(out)

def scaling(Y,Yhat,X):
    lastX = np.array(X[:][-1][:])
    lastX = lastX.reshape(1,lookback,nbcolumns)
    init_scale = Y[-1]/ (model.predict(lastX)+0.001)
    init_scale[init_scale > 3] = 1
    out = Yhat*init_scale
    for i in range(1,len(Yhat)):       
        scaling_factors = np.array(Y[i-1]/(Yhat[i-1]+0.0001))
        scaling_factors[scaling_factors > 3] = 1
        out[i] = np.positive(Yhat[i]*scaling_factors)
    return np.array(out)
        
def print_weights(model):
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

###################################################   INPUT
data= pd.read_csv('training.csv')
print("reading data done : "+str(int(time.time()-start_time))+" s")

training_periods = 500 # 96 is the number of intervals per day
test_periods = 5 # following the test period
precision = 5 # number of digit in the geo param (max is 6)  this parameter increases size O(36^n)
lookback = 4*4 # number of periods to lookback; 4 per hour 

train = False   # otherwise use the latest
train2 = False   # for the second neural net
start_from_previous = True 

###################################################   DATA PREP

data.demand = data.demand
Xprep = prep_input(data , precision)
Xprepn, mXprepn, sXprepn = normalize(Xprep)

#remove first time step from Y and the last from X 
Yprepn = Xprepn[1:]
Xprepn.drop(Xprepn.tail(1).index,inplace=True)

# filtering for the dates we need
X = np.array(Xprepn[lookback:training_periods])
Y = np.array(Yprepn[lookback:training_periods])

#shaping for the LTSM layer requirement
X = X.reshape(X.shape[0],1,X.shape[1])

# adding the data to look back at
for j in range(lookback-1):
    Xlookback = np.array(Xprepn[lookback-j-1:training_periods-j-1])
    Xlookback = Xlookback.reshape(Xlookback.shape[0],1,Xlookback.shape[1])
    X = np.append( X, Xlookback , axis=1 )

# preparing the test data
Xtest = Xprepn[training_periods:training_periods+test_periods]
Ytest = Yprepn[training_periods:training_periods+test_periods]

Xtest = np.array(Xtest)

Xtest = Xtest.reshape(Xtest.shape[0],1,Xtest.shape[1])

# adding the data to look back at
for j in range(lookback-1):
    Xlookback = np.array(Xprepn[training_periods-j-1:training_periods+test_periods-j-1])
    Xlookback = Xlookback.reshape(Xlookback.shape[0],1,Xlookback.shape[1])
    Xtest = np.append( Xtest, Xlookback , axis=1 )

Ytest = np.array(Ytest)

print("data prep done : " +str(int(time.time()-start_time))+" s")
 
nbcolumns = len(X[0][0])

##################################################  HYPERPARAMETERS
h_layer1_nodes = int(nbcolumns*lookback)
h_layeri_nodes = int(nbcolumns*lookback)
h_layerf_nodes = int(nbcolumns*lookback)

nb_h_layers = 5     # this doesn't account for the first and last hidden layers (+2)

e = 1                           # epoch
b = int(training_periods/2)     # batch size

print("size of each input vector : "+ str(nbcolumns))
print("size of 1st hidden layers : "+ str(h_layer1_nodes))
print("nb of hidden recurent layers : "+ str(nb_h_layers+2))
print("size of i hidden layers : "+ str(h_layeri_nodes))
print("size of f hidden layers : "+ str(h_layerf_nodes)+"\n")

###################################################   MODEL
print("modelling : " +str(int(time.time()-start_time))+" s")
# create model
model = Sequential()
model.add(LSTM(h_layer1_nodes, input_shape=(lookback, nbcolumns), activation='tanh', return_sequences=True))  

for _ in range(nb_h_layers):
    model.add(LSTM(h_layeri_nodes, activation='tanh', return_sequences=True))
model.add(LSTM(h_layerf_nodes, activation='tanh'))
model.add(Dense(nbcolumns, activation='linear')) #output layer

if start_from_previous: model.load_weights("model.h5")
print("training : " +str(int(time.time()-start_time))+" s")

if train:
    model.compile(loss='mean_absolute_error', optimizer='sgd', metrics=['mean_absolute_error'])
    model.fit(X, Y, epochs=e, batch_size=b, validation_split=0.2,  verbose=2)

    scores = model.evaluate(X, Y)
    model.save_weights("model.h5")
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]))
    print("training in : "+str(int(time.time()-start_time))+" s")

else:
    model.load_weights("model.h5")

###################################################   SECONDARY MODEL FOR GEODATA
#### PREP DATA
print("second network in : "+str(int(time.time()-start_time))+" s")
full_size = len(data.geohash6.unique())
fitted_m1 = model.predict(X)

X2p = prep_input(data, len(data.geohash6[0]))
Y2p = X2p[1:]
X2p.drop(X2p.tail(1).index,inplace=True)
X2 = np.array(X2p[lookback:training_periods])
Y2 = np.array(Y2p[lookback:training_periods])

Ytest2 = np.array(Y2p[training_periods:training_periods+test_periods])

#### MODEL
print("Prep 2 done in : "+str(int(time.time()-start_time))+" s")

model2 = Sequential()
model2.add(Dense(full_size, activation='sigmoid', input_dim=nbcolumns))
model2.compile(loss='mse', optimizer='sgd', metrics=['mse'])

if train2:
    model2.fit(fitted_m1, Y2, epochs=1000, batch_size=100, validation_split=0.2,  verbose=2)
    model2.save_weights("model2.h5")
    scores = model2.evaluate(fitted_m1, Y2)
    print("\n%s: %.2f%%" % (model2.metrics_names[1], scores[1]))
    print("training in : "+str(int(time.time()-start_time))+" s")
else: 
    model2.load_weights("model2.h5")


###################################################   PREDICT / TIME_SCALE / EXPLODE 

# calculate predictions and metrics
predictions = model.predict(Xtest)
predictions = denormalize(predictions, mXprepn, sXprepn)
Ytest = denormalize(Ytest,mXprepn, sXprepn)

Yhat = scaling(Ytest,predictions, X)

###################################################   RESULTS

#print_weights(model)

mape = np.sum(np.abs(Ytest-Yhat))/np.sum(Ytest)
mse = ((Ytest-Yhat)**2).mean(axis=None)
mae = (np.abs(Ytest-Yhat)).mean(axis=None)

print("prediction in : "+str(int(time.time()-start_time))+" s")
print("test MAE: " + str(mae*100))
print("test MAPE: %.2f%%" % (mape*100))
print("test MSE: %.2f%%" % (mse*100))
print(" ")

myFile = open('output.csv', 'w', newline='')
with myFile:
    writer = csv.writer(myFile)
    writer.writerow(Xprepn.columns)
    for i in range(test_periods):
        writer.writerow((100*Ytest[i]).astype(int))
        writer.writerow((100*Yhat[i]).astype(int))
        writer.writerow(" ")
    writer.writerow(" ##### END ##### ")
    
model_name = "./models/model "+"UTC "+str(datetime.now())+" "+str(training_periods)+" MAPE="+str(mape)+" MSE= "+str(mse)+".h5"
model_name= re.sub('[:]+', '', model_name)
model.save(model_name)


####output of second model
print_weights(model2)

Yhat2 = model2.predict(Yhat)
mape = np.sum(np.abs(Ytest2-Yhat2))/np.sum(Ytest)
mse = ((Ytest2-Yhat2)**2).mean(axis=None)
mae = (np.abs(Ytest2-Yhat2)).mean(axis=None)

print("prediction in : "+str(int(time.time()-start_time))+" s")
print("test MAE: " + str(mae*100))
print("test MAPE: %.2f%%" % (mape*100))
print("test MSE: %.2f%%" % (mse*100))

myFile = open('output2.csv', 'w', newline='')
with myFile:
    writer = csv.writer(myFile)
    writer.writerow(X2p.columns)
    for i in range(test_periods):
        writer.writerow((100*Ytest2[i]).astype(int))
        writer.writerow((100*Yhat2[i]).astype(int))
        writer.writerow(" ")
    writer.writerow(" ##### END ##### ")