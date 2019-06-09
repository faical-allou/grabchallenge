from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from keras import regularizers, optimizers, callbacks, activations
from keras import backend as K

import tensorflow as tf
import numpy as np
import csv
import time
from datetime import datetime
import pandas as pd
import re
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

#remove warnings tensorflow
tf.logging.set_verbosity(tf.logging.ERROR)

start_time = time.time()

###################################################   HELPER FUNCTIONS

def prep_input(data, precision_geo):
    
    data['geo_lim'] = data.geohash6.str[0:precision_geo]
    data['time'] = pd.to_datetime(data.timestamp, format='%H:%M')
    data['date_time_index'] = data.time + pd.TimedeltaIndex(data.day, unit = 'D')
    
    datagg = data.groupby(['geo_lim','date_time_index'])['demand'].mean().reset_index(name='mean')

    Xagg = datagg.pivot(index='date_time_index', columns='geo_lim', values='mean').fillna(0, inplace=False)
    Xfull = data.pivot(index='date_time_index', columns='geohash6', values='demand').fillna(0, inplace=False)
    return Xagg, Xfull

def normalize(X):
    meanX = X.mean(axis=0)
    stdX = X.std(axis=0)
    Xp = (X-meanX)/stdX
    Xp[np.isnan(Xp)] = 0
    stdX[np.isnan(stdX)] = 0
    return Xp, meanX, stdX

def denormalize(Xp,m,s):
    out = []
    for x in Xp:
        out.append(x*s+m)
    return np.array(out).clip(min=0)

def scaling(Y,scaling_vector):
    scaling_vector[scaling_vector>3] = 1
    out = Y*scaling_vector  
    return np.array(out).clip(min=0)

def sum_pred_error(y_true, y_pred):
    return 10*K.abs(K.mean(y_pred)-K.mean(y_true))+K.mean(K.abs(y_pred-y_true))
       
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
data = pd.read_csv('training.csv')
print("reading data done : "+str(int(time.time()-start_time))+" s")

training_periods = 96*20+210       # 96 is the number of intervals per day
test_periods = 5 # following the test period
precision = 5 # number of digit in the geo param (max is 6)  this parameter increases size O(36^n)
lookback = 4*4 # number of periods to lookback; 4 per hour 

train = False   # otherwise use the latest
train2 = False   # for the second neural net
start_from_previous = True  # if you train do you start from the previous
start_from_previous2 = True # if you train2 do you start from the previous
scaling_vector = np.genfromtxt('scaling.v', delimiter=',')

##################################################  PREP INPUT
Xprep, Xfull = prep_input(data, precision)

##################################################  HYPERPARAMETERS
nbcolumns = len(Xprep.columns)
h_layer1_nodes = int(nbcolumns*lookback)
h_layeri_nodes = int(nbcolumns*lookback)
h_layerf_nodes = int(nbcolumns*lookback)

e = 10000                           # epoch
e2 = 10000
b = int(training_periods/2)     # batch size

print("size of each input vector : "+ str(nbcolumns))
print("size of 1st hidden layers : "+ str(h_layer1_nodes))
print("size of i hidden layers : "+ str(h_layeri_nodes))
print("size of f hidden layers : "+ str(h_layerf_nodes)+"\n")

###################################################   DATA PREP FOR NETWORK

Xprepn, mXprepn, sXprepn = normalize(Xprep)

#remove first time step from Y and the last from X 
Yprepn = Xprepn[1:]
Xprepn.drop(Xprepn.tail(1).index,inplace=True)

# filtering for the dates we need
Xn = np.array(Xprepn[lookback:training_periods])
Yn = np.array(Yprepn[lookback:training_periods])

#shaping for the LTSM layer requirement
Xn = Xn.reshape(Xn.shape[0],1,Xn.shape[1])

# adding the data to look back at
for j in range(lookback-1):
    Xlookback = np.array(Xprepn[lookback-j-1:training_periods-j-1])
    Xlookback = Xlookback.reshape(Xlookback.shape[0],1,Xlookback.shape[1])
    Xn = np.append( Xn, Xlookback , axis=1 )

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

###################################################   MODEL
print("modelling start: " +str(int(time.time()-start_time))+" s")
# create model
model = Sequential()
model.add(LSTM(h_layer1_nodes, input_shape=(lookback, nbcolumns), activation='tanh', return_sequences=True))  
model.add(LSTM(h_layerf_nodes, activation='tanh'))
model.add(Dense(nbcolumns, activation='linear')) #output layer

if start_from_previous: model.load_weights("model.h5")
print("training start: " +str(int(time.time()-start_time))+" s")

if train:
    model.compile(loss='mean_absolute_error', optimizer='sgd', metrics=['mean_absolute_error'])
    model.fit(Xn, Yn, epochs=e, batch_size=b, validation_split=0.2,  verbose=2)

    scores = model.evaluate(Xn, Yn)
    model.save_weights("model.h5")
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]))
    print("training1 in : "+str(int(time.time()-start_time))+" s")

else:
    model.load_weights("model.h5")

###################################################   SECONDARY MODEL FOR GEODATA
#### PREP DATA
print("second network reached: "+str(int(time.time()-start_time))+" s")
full_size = len(data.geohash6.unique())

Y2 = np.array(Xfull[lookback+1:training_periods+1])

Ytest2 = np.array(Xfull[training_periods+2:training_periods+test_periods+2])

#### MODEL
print("Prep 2 done : "+str(int(time.time()-start_time))+" s")

model2 = Sequential()
model2.add(Dense(full_size, activation='sigmoid', input_dim=nbcolumns,bias_regularizer=regularizers.l1(0.1)))

if start_from_previous2: model2.load_weights("model2.h5")

if train2:
    # start by scaling the output of model 1 and save it
    Yscalen = model.predict(Xn)
    scaling_vector = Yn[-1]/Yscalen[-1]

    myFile = open('scaling.v', 'w', newline='')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerow(scaling_vector)

    fitted_m1s = scaling(Yscalen, scaling_vector)
    fitted_m1 = denormalize(fitted_m1s, mXprepn, sXprepn)

    model2.compile(loss=sum_pred_error, optimizer='adam', metrics=['mean_absolute_error',sum_pred_error])

    model2.fit(fitted_m1, Y2, epochs=e2, batch_size=b, validation_split=0.2,  verbose=2)
    model2.save_weights("model2.h5")

    scores = model2.evaluate(fitted_m1, Y2)
    print(str(model2.metrics_names[1])+" :"+str( scores[1]))
    print("training2 in : "+str(int(time.time()-start_time))+" s")
    print(" ")
else: 
    model2.load_weights("model2.h5")

###################################################   PREDICT / TIME_SCALE / EXPLODE 

# calculate predictions and metrics

Yhatn = model.predict(Xtest)
Yhats = scaling(Yhatn, scaling_vector)
Yhat = denormalize(Yhats, mXprepn, sXprepn)

Ytest = denormalize(Ytest, mXprepn, sXprepn)


###################################################   RESULTS

if train : print_weights(model)

mape = np.sum(np.abs(Ytest-Yhat))/np.sum(Ytest)
mse = ((Ytest-Yhat)**2).mean(axis=None)
mae = (np.abs(Ytest-Yhat)).mean(axis=None)

print("all predictions in : "+str(int(time.time()-start_time))+" s\n")

print("test MAE2: %.2f" % mae)
print("test MAPE1: %.2f%%" % (mape*100))
print("test MSE1: %.2f%%" % (mse*100))
print(" ")

plt.figure(0)
plt.plot(Ytest[0], Yhat[0], 'o', color='black')
plt.plot([0,1],[0,1])
plt.xlabel("actual")
plt.ylabel("prediction")
plt.suptitle('On Aggregate Level')
plt.show(block=False)
input('press <ENTER> to continue')

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

###################################################   PREDICT / TIME_SCALE / EXPLODE 

Yhat2 = model2.predict(Yhat)

####output of second model
if train2: print_weights(model2)

mape = np.sum(np.abs(Ytest2-Yhat2))/np.sum(Ytest)
mse = ((Ytest2-Yhat2)**2).mean(axis=None)
mae = (np.abs(Ytest2-Yhat2)).mean(axis=None)

print("test MAE2: %.2f" % mae)
print("test MAPE2: %.2f%%" % (mape*100))
print("test MSE2: %.2f%%" % (mse*100))
print(" ")

plt.figure(1)
plt.plot(Ytest2[0], Yhat2[0], 'o', color='black')
plt.plot([0,1],[0,1])
plt.xlabel("actual")
plt.ylabel("prediction")
plt.suptitle('On Granular Level')
plt.show(block=False)
input('press <ENTER> to continue')

myFile = open('output2.csv', 'w', newline='')
with myFile:
    writer = csv.writer(myFile)
    writer.writerow(Xfull.columns)
    for i in range(test_periods):
        writer.writerow((100*Ytest2[i]).astype(int))
        writer.writerow((100*Yhat2[i]).astype(int))
        writer.writerow(" ")
    writer.writerow(" ##### END ##### ")
    
print("end of script in : "+str(int(time.time()-start_time))+" s")
